from typing import Any, NamedTuple
from urllib.parse import quote

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.func import task  # type: ignore
from rich.console import Console  # type: ignore

from reviewbot.agent.workflow.config import GitProviderConfig
from reviewbot.core.agent import Agent
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


class AcknowledgmentResult(NamedTuple):
    discussion_id: str
    note_id: str


console = Console()


@task
def post_review_acknowledgment(
    *, gitlab: GitProviderConfig, diffs: list[FileDiff], model: BaseChatModel
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
    from langchain_core.messages import HumanMessage, SystemMessage

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

    # Generate a simple summary with very limited tool calls
    messages = [
        SystemMessage(
            content="""You are a code review assistant. Generate a brief, friendly acknowledgment that a code review is starting.

IMPORTANT:
- Keep it SHORT (2-3 sentences max)
- Be surface-level - this is just an acknowledgment, not the actual review
- DO NOT analyze code yet
- DO NOT use any tools
- Just acknowledge what files are being reviewed"""
        ),
        HumanMessage(
            content=f"""A merge request code review is starting for the following files:

{files_summary}

Write a brief acknowledgment message (2-3 sentences) letting the developer know the review is in progress. Be friendly and professional."""
        ),
    ]

    try:
        # Get response with no tool calls allowed
        from ido_agents.agents.ido_agent import create_ido_agent
        from ido_agents.agents.tool_runner import ToolCallerSettings

        summary_settings = ToolCallerSettings(max_tool_calls=0)
        ido_agent = create_ido_agent(model=model, tools=[])
        summary = ido_agent.with_tool_caller(summary_settings).invoke(messages)

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


def update_review_summary(
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
    from langchain_core.messages import HumanMessage, SystemMessage

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

        messages = [
            SystemMessage(
                content="""You are a Merge Request reviewer. Generate a concise, professional summary of a code review with reasoning.

IMPORTANT:
- Use EXACTLY two paragraphs, each wrapped in <p> tags.
- Provide reasoning about the overall merge request purpose and code quality.
- Highlight key concerns or positive aspects
- Be constructive and professional
- DO NOT use any tools
- Use paragraphs with readable flow. Use two paragrahs with 3-5 sentences.
Paragraphs should be wrapped with <p> tags. Use new <p> tag for a newline.
Example
<p>
paragraph
</p>
<br>
<p>
paragraph2
</p>
- Focus on the big picture, not individual issue details"""
            ),
            HumanMessage(
                content=f"""A code review has been completed with the following results:

**Statistics:**
- Files reviewed: {total_files}
- Files with issues: {files_with_issues}
- Total issues: {len(issues)}
  - High severity: {high_count}
  - Medium severity: {medium_count}
  - Low severity: {low_count}

**Issues found:**
{issues_text}

- Use EXACTLY two paragraphs, each wrapped in <p> tags.
1. Provides overall assessment of the purpose of the merge request purpose and code quality.
2. Highlights the most important concerns (if any)
3. Gives reasoning about the review findings
4. Is constructive and actionable      """
            ),
        ]

        if model is None:
            raise ValueError("model parameter is required for ido-agents migration")

        ido_agent = create_ido_agent(model=model, tools=tools or [])
        llm_summary = ido_agent.with_structured_output(ReviewSummary).invoke(messages)

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
