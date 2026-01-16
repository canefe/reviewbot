from typing import Any, NamedTuple
from urllib.parse import quote

from ido_agents.agents.tool_runner import ToolCallerSettings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.func import task  # type: ignore
from rich.console import Console  # type: ignore

from reviewbot.agent.workflow.config import GitProviderConfig
from reviewbot.core.agent import Agent
from reviewbot.core.config import Config
from reviewbot.core.issues import Issue, IssueSeverity
from reviewbot.core.reviews.review import Acknowledgment
from reviewbot.core.reviews.review_model import ReviewSummary
from reviewbot.infra.gitlab.diff import FileDiff
from reviewbot.infra.gitlab.note import (
    get_all_discussions,
    post_discussion,
    post_merge_request_note,
    update_discussion_note,
)
from reviewbot.prompts.get_prompt import get_prompt


class AcknowledgmentResult(NamedTuple):
    discussion_id: str
    note_id: str


console = Console()


@task
async def post_review_acknowledgment(
    *, gitlab: GitProviderConfig, diffs: list[FileDiff], model: BaseChatModel, config: Config
) -> AcknowledgmentResult | None:
    """
    Posts an initial acknowledgment discussion for the MR review.

    Reads:
      - CodebaseState from store

    Writes:
      - acknowledgment ids (returned)

    Returns:
      AcknowledgmentResult if created, otherwise None
    """

    """
    Post a surface-level summary acknowledging the review is starting.
    Creates a discussion so it can be updated later.
    Only posts if no acknowledgment already exists to prevent duplicates.

    Args:
        api_v4: GitLab API v4 base URL
        token: GitLab API token
        project_id: Project ID
        mr_iid: Merge request IID
        agent: The agent to use for generating summary
        diffs: List of file diffs

    Returns:
        Tuple of (discussion_id, note_id) if created, None if already exists or failed
    """
    api_v4 = gitlab.get_api_base_url()
    token = gitlab.token.get_secret_value()
    project_id = gitlab.get_project_identifier()
    mr_iid = gitlab.get_pr_identifier()
    # Check if an acknowledgment already exists
    try:
        discussions = get_all_discussions(
            api_v4=api_v4,
            token=token,
            project_id=project_id,
            mr_iid=mr_iid,
        )

        # Only reuse "Starting" acknowledgments (in-progress reviews)
        # Don't reuse "Complete" acknowledgments - create new ones for new review runs
        starting_marker = (
            '<img src="https://img.shields.io/badge/Code_Review-Starting-blue?style=flat-square" />'
        )

        # Find ALL "Starting" acknowledgments, then pick the most recent one
        found_acknowledgments: list[Acknowledgment] = []
        for discussion in discussions:
            notes = discussion.get("notes", [])
            for note in notes:
                body = note.get("body", "")
                # Only check for "Starting" marker - this means review is in progress
                if starting_marker in body:
                    discussion_id = discussion.get("id")
                    note_id = note.get("id")
                    created_at = note.get("created_at", "")
                    if discussion_id and note_id:
                        found_acknowledgments.append(
                            Acknowledgment(
                                discussion_id=str(discussion_id),
                                note_id=str(note_id),
                                created_at=created_at,
                            )
                        )

        # If we found any in-progress acknowledgments, use the most recent one
        if found_acknowledgments:
            # Sort by created_at timestamp (most recent first)
            found_acknowledgments.sort(key=lambda x: x.created_at, reverse=True)
            most_recent = found_acknowledgments[0]
            console.print(
                f"[dim]Found {len(found_acknowledgments)} in-progress review(s), reusing most recent[/dim]"
            )
            return AcknowledgmentResult(most_recent.discussion_id, most_recent.note_id)

        # No in-progress reviews found - will create a new acknowledgment
        console.print("[dim]No in-progress reviews found, will create new acknowledgment[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not check for existing acknowledgment: {e}[/yellow]")
        # Continue anyway - better to post a duplicate than miss it

    # Get list of files being reviewed
    file_list = [diff.new_path for diff in diffs if diff.new_path]
    files_summary = "\n".join([f"- `{f}`" for f in file_list[:10]])  # Limit to first 10
    if len(file_list) > 10:
        files_summary += f"\n- ... and {len(file_list) - 10} more files"

    # Load prompt template
    prompt_template = get_prompt("gitlab/acknowledgment", config)

    # Generate a simple summary with very limited tool calls
    messages: list[BaseMessage] = prompt_template.format_messages(files_summary=files_summary)

    try:
        # Get response with no tool calls allowed
        from ido_agents.agents.ido_agent import create_ido_agent

        summary_settings = ToolCallerSettings(max_tool_calls=0)
        ido_agent = create_ido_agent(model=model, tools=[])
        summary = await ido_agent.with_tool_caller(summary_settings).ainvoke(messages)

        # Post as a discussion (so we can update it later)
        acknowledgment_body = f"""<img src="https://img.shields.io/badge/Code_Review-Starting-blue?style=flat-square" />

{summary}

---
*Review powered by ReviewBot*
"""

        # post_discussion now returns both discussion_id and note_id
        discussion_id, note_id = post_discussion(
            api_v4=api_v4,
            token=token,
            project_id=project_id,
            mr_iid=mr_iid,
            body=acknowledgment_body,
        )

        if not note_id:
            console.print("[yellow]Discussion created but no note ID returned[/yellow]")
            return None

        console.print(
            f"[green]✓ Posted review acknowledgment (discussion: {discussion_id}, note: {note_id})[/green]"
        )
        return AcknowledgmentResult(str(discussion_id), str(note_id))

    except Exception as e:
        console.print(f"[yellow]Failed to post acknowledgment: {e}[/yellow]")
        # Don't fail the whole review if acknowledgment fails
        return None


@task
async def update_review_summary(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    discussion_id: str,
    note_id: str,
    issues: list[Issue],
    diffs: list[FileDiff],
    diff_refs: dict[str, str],
    agent: Agent,
    config: Config,
    model: BaseChatModel | None = None,
    tools: list[Any] | None = None,
) -> None:
    """
    Update the acknowledgment note with a summary of the review results.
    Uses LLM to generate reasoning and summary.

    Args:
        api_v4: GitLab API v4 base URL
        token: GitLab API token
        project_id: Project ID
        mr_iid: Merge request IID
        discussion_id: Discussion ID of the acknowledgment
        note_id: Note ID to update
        issues: List of issues found during review
        diffs: List of file diffs that were reviewed
        diff_refs: Diff references including project_web_url
        agent: The agent to use for generating summary
    """
    # Count issues by severity
    high_count = sum(1 for issue in issues if issue.severity == IssueSeverity.HIGH)
    medium_count = sum(1 for issue in issues if issue.severity == IssueSeverity.MEDIUM)
    low_count = sum(1 for issue in issues if issue.severity == IssueSeverity.LOW)

    # Group issues by file
    issues_by_file: dict[str, list[Issue]] = {}
    for issue in issues:
        if issue.file_path not in issues_by_file:
            issues_by_file[issue.file_path] = []
        issues_by_file[issue.file_path].append(issue)

    # Build structured statistics
    total_files = len(diffs)
    files_with_issues = len(issues_by_file)

    # Build change overview for context
    new_files: list[str] = []
    deleted_files: list[str] = []
    renamed_files: list[tuple[str, str]] = []
    modified_files: list[str] = []

    for diff in diffs:
        if diff.is_renamed:
            renamed_files.append((diff.old_path or "unknown", diff.new_path or "unknown"))
        elif diff.is_new_file:
            new_files.append(diff.new_path or diff.old_path or "unknown")
        elif diff.is_deleted_file:
            deleted_files.append(diff.old_path or diff.new_path or "unknown")
        else:
            modified_files.append(diff.new_path or diff.old_path or "unknown")

    change_stats = (
        "Files changed: "
        f"{total_files} (new: {len(new_files)}, modified: {len(modified_files)}, "
        f"renamed: {len(renamed_files)}, deleted: {len(deleted_files)})"
    )
    change_overview_lines: list[str] = []
    change_overview_lines.extend(f"- {path} (new)" for path in new_files)
    change_overview_lines.extend(f"- {path} (deleted)" for path in deleted_files)
    change_overview_lines.extend(f"- {old} -> {new} (renamed)" for old, new in renamed_files)
    change_overview_lines.extend(f"- {path} (modified)" for path in modified_files)

    max_change_lines = 12
    if len(change_overview_lines) > max_change_lines:
        remaining = len(change_overview_lines) - max_change_lines
        change_overview_lines = change_overview_lines[:max_change_lines]
        change_overview_lines.append(f"- ... and {remaining} more file(s)")

    change_overview_text = (
        "\n".join(change_overview_lines) if change_overview_lines else "- No files listed."
    )

    # Prepare issue details for LLM
    issues_summary: list[str] = []
    for issue in issues:
        issues_summary.append(
            f"- **{issue.severity.value.upper()}** in `{issue.file_path}` (lines {issue.start_line}-{issue.end_line}): {issue.description}"
        )

    issues_text = "\n".join(issues_summary) if issues_summary else "No issues found."

    # Generate LLM summary with reasoning
    try:
        from ido_agents.agents.ido_agent import create_ido_agent

        # Load prompt template
        prompt_template = get_prompt("gitlab/summary", config)

        # Format messages with placeholders
        messages: list[BaseMessage] = prompt_template.format_messages(
            total_files=total_files,
            files_with_issues=files_with_issues,
            total_issues=len(issues),
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            change_stats=change_stats,
            change_overview_text=change_overview_text,
            issues_text=issues_text,
        )

        if model is None:
            raise ValueError("model parameter is required for ido-agents migration")

        ido_agent = create_ido_agent(model=model, tools=tools or [])

        summary_settings = ToolCallerSettings(max_tool_calls=5)
        llm_summary = await (
            ido_agent.with_structured_output(ReviewSummary)
            .with_tool_caller(summary_settings)
            .ainvoke(messages)
        )

        llm_summary = llm_summary.summary if llm_summary else "Review completed successfully."

    except Exception as e:
        console.print(f"[yellow]Failed to generate LLM summary: {e}[/yellow]")
        llm_summary = "Review completed successfully."

    # Build final summary combining statistics and LLM reasoning
    summary_parts = [
        '<img src="https://img.shields.io/badge/Code_Review-Complete-green?style=flat-square" />\n\n',
        f"Reviewed **{total_files}** file(s), found **{len(issues)}** issue(s) across **{files_with_issues}** file(s).\n\n",
        "**Summary**\n\n",
        f"{llm_summary}\n\n",
    ]

    if issues:
        project_web_url = diff_refs.get("project_web_url")

        summary_parts.append("**Issue Breakdown**\n\n")
        if high_count > 0:
            summary_parts.append(
                f'<img src="https://img.shields.io/badge/High-{high_count}-red?style=flat-square" /> \n'
            )
        if medium_count > 0:
            summary_parts.append(
                f'<img src="https://img.shields.io/badge/Medium-{medium_count}-orange?style=flat-square" /> \n'
            )
        if low_count > 0:
            summary_parts.append(
                f'<img src="https://img.shields.io/badge/Low-{low_count}-green?style=flat-square" /> \n'
            )

        summary_parts.append("\n<br>\n<br>\n\n")

        if issues:
            summary_parts.append("---\n\n")

            severity_badge_colors = {
                IssueSeverity.HIGH: "red",
                IssueSeverity.MEDIUM: "orange",
                IssueSeverity.LOW: "green",
            }

            for file_path, file_issues in sorted(issues_by_file.items()):
                summary_parts.append(f"####  `{file_path}`\n\n")
                severity_order = {
                    IssueSeverity.HIGH: 0,
                    IssueSeverity.MEDIUM: 1,
                    IssueSeverity.LOW: 2,
                }
                for issue in sorted(file_issues, key=lambda item: severity_order[item.severity]):
                    label = issue.severity.value.upper()
                    badge_color = severity_badge_colors[issue.severity]
                    note_url = None
                    if project_web_url and issue.note_id:
                        note_url = (
                            f"{project_web_url}/-/merge_requests/{mr_iid}#note_{issue.note_id}"
                        )
                    if note_url:
                        link_html = (
                            f'<a href="{note_url}" target="_blank" rel="noopener noreferrer">'
                            f"{issue.title}</a>"
                        )
                    else:
                        link_html = "Comment link unavailable"

                    summary_parts.append(
                        f'<img style="margin-left:51px" src="https://img.shields.io/badge/{quote(label)}-{badge_color}?style=flat-square" />&nbsp;&nbsp;'
                        f'<span style="margin-left:51px">{link_html}</span>\n\n'
                    )
                summary_parts.append("\n")
    else:
        summary_parts.append(
            '<img src="https://img.shields.io/badge/No_Issues Found-brightgreen?style=flat-square" />\n'
        )

    summary_parts.append("\n---\n*Review powered by ReviewBot*")

    summary_body = "".join(summary_parts)

    console.print(
        f"[dim]Updating discussion {discussion_id}, note {note_id} with review summary...[/dim]"
    )
    try:
        update_discussion_note(
            api_v4=api_v4,
            token=token,
            project_id=project_id,
            mr_iid=mr_iid,
            discussion_id=discussion_id,
            note_id=note_id,
            body=summary_body,
        )
        console.print("[green]✓ Updated review acknowledgment with summary[/green]")
    except Exception as e:
        console.print(f"[yellow]Failed to update acknowledgment: {e}[/yellow]")
        import traceback

        traceback.print_exc()
        # Don't fail the whole review if update fails


def post_mr_note(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    body: str,
) -> None:
    """
    Post a standalone note (comment) to a merge request without creating a discussion.

    Args:
        api_v4: GitLab API v4 base URL
        token: GitLab API token
        project_id: Project ID
        mr_iid: Merge request IID
        body: Note content
    """
    post_merge_request_note(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
        body=body,
    )
