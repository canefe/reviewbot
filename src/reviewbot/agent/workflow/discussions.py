from typing import Any
from urllib.parse import quote

from rich.console import Console  # type: ignore

from reviewbot.agent.workflow.config import GitProviderConfig
from reviewbot.agent.workflow.diff_extract import create_file_position
from reviewbot.core.issues import IssueModel, IssueSeverity
from reviewbot.infra.gitlab.diff import FileDiff
from reviewbot.infra.gitlab.note import post_discussion, post_discussion_reply

console = Console()


def handle_file_issues(
    file_path: str,
    issues: list[IssueModel],
    gitlab_config: GitProviderConfig,
    file_diffs: list[FileDiff],
    diff_refs: dict[str, str],
) -> None:
    """
    Create positioned discussions for a capped set of high-priority issues, and
    group the rest into a single per-file discussion with replies.

    Args:
        file_path: Path to the file being reviewed
        issues: List of issues found in this file
        gitlab_config: GitLab API configuration
        file_diffs: List of file diffs from the MR
        diff_refs: Dict with base_sha, head_sha, start_sha
    """
    if not issues:
        return

    console.print(f"[cyan]Creating discussion for {file_path} with {len(issues)} issue(s)[/cyan]")

    # Get the file diff once
    file_diff = next((fd for fd in file_diffs if fd.new_path == file_path), None)
    base_sha = diff_refs.get("base_sha")
    head_sha = diff_refs.get("head_sha")
    start_sha = diff_refs.get("start_sha")
    project_web_url = diff_refs.get("project_web_url")

    # Severity, Color pairs
    severity_color_pairs = {
        IssueSeverity.HIGH: "red",  # red
        IssueSeverity.MEDIUM: "orange",  # yellow/orange
        IssueSeverity.LOW: "green",  # green
    }

    def build_location_line(issue: IssueModel) -> str:
        if project_web_url and head_sha:
            escaped_path = quote(issue.file_path, safe="/")
            if issue.start_line == issue.end_line:
                anchor = f"#L{issue.start_line}"
            else:
                anchor = f"#L{issue.start_line}-L{issue.end_line}"
            file_url = f"{project_web_url}/-/blob/{head_sha}/{escaped_path}{anchor}"
            return (
                f'<a href="{file_url}" target="_blank" rel="noopener noreferrer">'
                f"#L {issue.start_line}-{issue.end_line}"
                f"</a>"
            )
        return f"lines {issue.start_line}-{issue.end_line}"

    def build_position() -> dict[str, Any] | None:
        if (
            file_diff
            and base_sha
            and head_sha
            and start_sha
            and file_diff.old_path
            and file_diff.new_path
        ):
            return create_file_position(
                base_sha=base_sha,
                head_sha=head_sha,
                start_sha=start_sha,
                old_path=file_diff.old_path,
                new_path=file_diff.new_path,
            )
        return None

    discussion_title = ""
    discussion_body = "\n\n"
    first_issue, *remaining_issues = issues
    for issue in [first_issue]:
        color = severity_color_pairs[issue.severity].strip("#")
        location_line = build_location_line(issue)
        discussion_body += (
            f'<img src="https://img.shields.io/badge/{issue.severity.value.upper()}-{color}?style=flat-square" />\n\n'
            f"**{issue.title}** ({location_line})\n\n"
            f"{issue.description}\n"
        )
        if issue.suggestion:
            discussion_body += f"\n{issue.suggestion}\n"
        discussion_body += "\n"

    position = build_position()
    if position:
        console.print(f"[dim]Position object for {file_path}:[/dim]")
        import json

        console.print(f"[dim]{json.dumps(position, indent=2)}[/dim]")

    try:
        discussion_id, note_id = create_discussion(
            title=discussion_title,
            body=discussion_body,
            gitlab_config=gitlab_config,
            position=position,
        )
        first_issue.discussion_id = discussion_id
        first_issue.note_id = note_id
        for issue in remaining_issues:
            reply_body = ""
            color = severity_color_pairs[issue.severity].strip("#")
            location_line = build_location_line(issue)
            reply_body += (
                f'<img src="https://img.shields.io/badge/{issue.severity.value.upper()}-{color}?style=flat-square" /> \n\n'
                f"**{issue.title}** ({location_line})\n\n"
                f"{issue.description}\n"
            )
            if issue.suggestion:
                reply_body += f"\n{issue.suggestion}\n"
            note_id = reply_to_discussion(
                discussion_id=discussion_id,
                body=reply_body,
                gitlab_config=gitlab_config,
            )
            issue.discussion_id = discussion_id
            issue.note_id = note_id
        console.print(f"[green]✓ Created discussion for {file_path} (ID: {discussion_id})[/green]")
    except Exception as e:
        if position:
            console.print(
                f"[yellow]Failed with position for {file_path}, retrying without position: {e}[/yellow]"
            )
            try:
                discussion_id, note_id = create_discussion(
                    title=discussion_title,
                    body=discussion_body,
                    gitlab_config=gitlab_config,
                    position=None,
                )
                first_issue.discussion_id = discussion_id
                first_issue.note_id = note_id
                for issue in remaining_issues:
                    reply_body = ""
                    color = severity_color_pairs[issue.severity].strip("#")
                    location_line = build_location_line(issue)
                    reply_body += (
                        f'<img src="https://img.shields.io/badge/{issue.severity.value.upper()}-{color}?style=flat-square" /> \n\n'
                        f"**{issue.title}** ({location_line})\n\n"
                        f"{issue.description}\n"
                    )
                    if issue.suggestion:
                        reply_body += f"\n{issue.suggestion}\n"
                    note_id = reply_to_discussion(
                        discussion_id=discussion_id,
                        body=reply_body,
                        gitlab_config=gitlab_config,
                    )
                    issue.discussion_id = discussion_id
                    issue.note_id = note_id
                console.print(
                    f"[green]✓ Created discussion without position for {file_path} (ID: {discussion_id})[/green]"
                )
                return
            except Exception as e2:
                console.print(f"[red]✗ Failed to create discussion for {file_path}: {e2}[/red]")
                import traceback

                traceback.print_exc()
                return

        console.print(f"[red]✗ Failed to create discussion for {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()


def create_discussion(
    title: str,
    body: str,
    gitlab_config: GitProviderConfig,
    position: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """
    Create a discussion with title and body.

    Args:
        title: Discussion title
        body: Discussion body content
        gitlab_config: GitLab API configuration
        position: Optional position object for file-based discussions

    Returns:
        Tuple of (discussion_id, note_id)
    """
    # GitLab discussions don't have separate titles in the API,
    # so we include the title in the body with markdown formatting

    # post_discussion returns (discussion_id, note_id), we only need discussion_id
    discussion_id, note_id = post_discussion(
        api_v4=gitlab_config.get_api_base_url(),
        token=gitlab_config.token.get_secret_value(),
        project_id=gitlab_config.get_project_identifier(),
        mr_iid=gitlab_config.get_pr_identifier(),
        body=body,
        position=position,
    )

    return discussion_id, note_id


def reply_to_discussion(
    discussion_id: str,
    body: str,
    gitlab_config: GitProviderConfig,
) -> str | None:
    """
    Reply to an existing discussion.

    Args:
        discussion_id: ID of the discussion to reply to
        body: Content of the reply
        gitlab_config: GitLab API configuration
    """
    return post_discussion_reply(
        api_v4=gitlab_config.get_api_base_url(),
        token=gitlab_config.token.get_secret_value(),
        project_id=gitlab_config.get_project_identifier(),
        merge_request_id=gitlab_config.get_pr_identifier(),
        discussion_id=discussion_id,
        body=body,
    )
