import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from langchain.agents import create_agent  # type: ignore
from langchain.agents.middleware import (  # type: ignore
    AgentState,
    before_agent,
    before_model,
)
from langgraph.pregel.main import Runtime  # type: ignore
from rich.console import Console  # type: ignore

from reviewbot.agent.base import (  # type: ignore
    AgentRunnerInput,
    agent_runner,  # type: ignore
)
from reviewbot.agent.tasks.core import ToolCallerSettings
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.agent import Agent
from reviewbot.core.config import Config
from reviewbot.core.issues import Issue, IssueSeverity
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import FileDiff, fetch_mr_diffs, get_mr_branch
from reviewbot.infra.gitlab.note import (
    get_all_discussions,
    post_discussion,
    post_discussion_reply,
    post_merge_request_note,
    update_discussion_note,
)
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore
from reviewbot.models.gpt import get_gpt_model, get_gpt_model_low_effort
from reviewbot.tools import (
    get_diff,
    read_file,
    think,
)

console = Console()

# Global blacklist for common files that typically don't need code review
GLOBAL_REVIEW_BLACKLIST = [
    # Dependency management files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "Pipfile.lock",
    "poetry.lock",
    "composer.lock",
    "go.sum",
    "go.mod",
    "Cargo.lock",
    # Build and distribution files
    "*.min.js",
    "*.min.css",
    "*.map",
    "dist/*",
    "build/*",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.o",
    "*.a",
    # Generated files
    "*.generated.*",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "*.pb.go",
    # Documentation and assets
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    # IDE and editor files
    ".vscode/*",
    ".idea/*",
    "*.swp",
    "*.swo",
    "*~",
]


def parse_reviewignore(repo_path: Path) -> list[str]:
    """
    Parse .reviewignore file from the repository.

    Args:
        repo_path: Path to the repository root

    Returns:
        List of glob patterns to ignore
    """
    reviewignore_path = repo_path / ".reviewignore"
    patterns = []

    if not reviewignore_path.exists():
        console.print("[dim].reviewignore file not found, using global blacklist only[/dim]")
        return patterns

    try:
        with open(reviewignore_path, encoding="utf-8") as f:
            for line in f:
                # Strip whitespace
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)

        console.print(f"[dim]Loaded {len(patterns)} patterns from .reviewignore[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read .reviewignore: {e}[/yellow]")

    return patterns


def should_ignore_file(file_path: str, reviewignore_patterns: list[str]) -> bool:
    """
    Check if a file should be ignored based on .reviewignore patterns and global blacklist.

    Args:
        file_path: Path to the file (relative to repo root)
        reviewignore_patterns: Patterns from .reviewignore file

    Returns:
        True if the file should be ignored, False otherwise
    """
    # Normalize the file path (remove leading ./ or /)
    normalized_path = file_path.lstrip("./")

    # Check against global blacklist
    for pattern in GLOBAL_REVIEW_BLACKLIST:
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Also check just the filename for non-path patterns
        if "/" not in pattern and fnmatch.fnmatch(Path(normalized_path).name, pattern):
            return True

    # Check against .reviewignore patterns
    for pattern in reviewignore_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Also check just the filename for non-path patterns
        if "/" not in pattern and fnmatch.fnmatch(Path(normalized_path).name, pattern):
            return True

    return False


def filter_diffs(diffs: list[FileDiff], reviewignore_patterns: list[str]) -> list[FileDiff]:
    """
    Filter out diffs for files that should be ignored.

    Args:
        diffs: List of file diffs
        reviewignore_patterns: Patterns from .reviewignore file

    Returns:
        Filtered list of diffs
    """
    filtered = []
    ignored_count = 0

    for diff in diffs:
        # Use new_path if available, otherwise use old_path
        file_path = diff.new_path or diff.old_path

        if file_path and should_ignore_file(file_path, reviewignore_patterns):
            console.print(f"[dim]⊘ Ignoring {file_path}[/dim]")
            ignored_count += 1
        else:
            filtered.append(diff)

    if ignored_count > 0:
        console.print(f"[cyan]Filtered out {ignored_count} file(s) based on ignore patterns[/cyan]")

    return filtered


def _extract_code_from_diff(diff_text: str, line_start: int, line_end: int) -> str:
    hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    lines = diff_text.splitlines()

    extracted = []
    current_new = 0
    in_hunk = False

    for line in lines:
        match = hunk_header_pattern.match(line)
        if match:
            current_new = int(match.group(3))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # We only care about the lines in the NEW file (the result of the change)
        if line.startswith("+"):
            if line_start <= current_new <= line_end:
                extracted.append(line[1:])  # Remove '+'
            current_new += 1
        elif line.startswith("-"):
            # Skip deleted lines for code extraction of the 'new' state
            continue
        else:
            # Context line
            if line_start <= current_new <= line_end:
                extracted.append(line[1:] if line else "")
            current_new += 1

        # FIX: Exit early if we've passed the end of our requested range
        if current_new > line_end:
            if extracted:  # Only break if we actually found lines
                break

    return "\n".join(extracted)


@dataclass
class GitLabConfig:
    """GitLab API configuration"""

    api_v4: str
    token: str
    project_id: str
    mr_iid: str


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    messages = state["messages"]
    console.print("[blue]Before modelMessages:[/blue]")
    console.print(messages[-5:])
    console.print("[blue]Before model messages end.[/blue]")
    return None


@before_agent(can_jump_to=["end"])
def check_agent_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    messages = state["messages"]
    console.print("[red]Before agent messages:[/red]")
    console.print(messages[-5:])
    console.print("[red]Before agent messages end.[/red]")
    return None


def post_review_acknowledgment(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    agent: Agent,
    diffs: list[FileDiff],
) -> tuple[str, str] | None:
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
        found_acknowledgments = []
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
                            {
                                "discussion_id": str(discussion_id),
                                "note_id": str(note_id),
                                "created_at": created_at,
                            }
                        )

        # If we found any in-progress acknowledgments, use the most recent one
        if found_acknowledgments:
            # Sort by created_at timestamp (most recent first)
            found_acknowledgments.sort(key=lambda x: x["created_at"], reverse=True)
            most_recent = found_acknowledgments[0]
            console.print(
                f"[dim]Found {len(found_acknowledgments)} in-progress review(s), reusing most recent[/dim]"
            )
            return (most_recent["discussion_id"], most_recent["note_id"])

        # No in-progress reviews found - will create a new acknowledgment
        console.print("[dim]No in-progress reviews found, will create new acknowledgment[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not check for existing acknowledgment: {e}[/yellow]")
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
        from reviewbot.agent.tasks.core import ToolCallerSettings, tool_caller

        summary_settings = ToolCallerSettings(max_tool_calls=0, max_iterations=1)
        summary = tool_caller(agent, messages, summary_settings)

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
            console.print("[yellow]⚠ Discussion created but no note ID returned[/yellow]")
            return None

        console.print(
            f"[green]✓ Posted review acknowledgment (discussion: {discussion_id}, note: {note_id})[/green]"
        )
        return (str(discussion_id), str(note_id))

    except Exception as e:
        console.print(f"[yellow]⚠ Failed to post acknowledgment: {e}[/yellow]")
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
        diff_refs: Diff references including head_sha and project_web_url
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
    issues_summary = []
    for issue in issues:
        issues_summary.append(
            f"- **{issue.severity.value.upper()}** in `{issue.file_path}` (lines {issue.start_line}-{issue.end_line}): {issue.description}"
        )

    issues_text = "\n".join(issues_summary) if issues_summary else "No issues found."

    # Generate LLM summary with reasoning
    try:
        from reviewbot.agent.tasks.core import ToolCallerSettings, tool_caller

        messages = [
            SystemMessage(
                content="""You are a Merge Request reviewer. Generate a concise, professional summary of a code review with reasoning.

IMPORTANT:
- Use EXACTLY two paragraphs, each wrapped in <p> tags.
- Provide reasoning about the overall merge request purpose and code quality.
- Highlight key concerns or positive aspects
- Be constructive and professional
- DO NOT use any tools
- Use paragraphs with readable flow.
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

        summary_settings = ToolCallerSettings(max_tool_calls=0, max_iterations=1)
        llm_summary = tool_caller(agent, messages, summary_settings)

    except Exception as e:
        console.print(f"[yellow]⚠ Failed to generate LLM summary: {e}[/yellow]")
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
        head_sha = diff_refs.get("head_sha")

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

        non_dedicated_issues = [issue for issue in issues if not issue.discussion_id]
        if non_dedicated_issues:
            issues_by_file: dict[str, list[Issue]] = {}
            for issue in non_dedicated_issues:
                issues_by_file.setdefault(issue.file_path, []).append(issue)

            severity_badge_colors = {
                IssueSeverity.HIGH: "red",
                IssueSeverity.MEDIUM: "orange",
                IssueSeverity.LOW: "green",
            }

            for file_path, file_issues in sorted(issues_by_file.items()):
                summary_parts.append(f"###  📂 {file_path}\n\n")
                for issue in file_issues:
                    file_diff = next((fd for fd in diffs if fd.new_path == issue.file_path), None)
                    code_snippet = ""
                    if file_diff:
                        code_snippet = _extract_code_from_diff(
                            file_diff.patch,
                            issue.start_line,
                            issue.end_line,
                        )
                    if not code_snippet:
                        code_snippet = "(no diff context available)"

                    label = issue.severity.value.upper()
                    badge_color = severity_badge_colors[issue.severity]
                    file_url = None
                    if project_web_url and head_sha:
                        escaped_path = quote(issue.file_path, safe="/")
                        if issue.start_line == issue.end_line:
                            anchor = f"#L{issue.start_line}"
                        else:
                            anchor = f"#L{issue.start_line}-L{issue.end_line}"
                        file_url = f"{project_web_url}/-/blob/{head_sha}/{escaped_path}{anchor}"
                    if file_url:
                        location_line = (
                            f'<a href="{file_url}" target="_blank" rel="noopener noreferrer">'
                            f"#L {issue.start_line}-{issue.end_line}"
                            f"</a>"
                        )
                    else:
                        location_line = f"#L {issue.start_line}-{issue.end_line}"

                    issue_body = f"""{issue.description}
"""
                    summary_parts.append(
                        f"""<details>
<summary><img style="margin-left:51px" src="https://img.shields.io/badge/{quote(label)}-{badge_color}?style=flat-square" />&nbsp;&nbsp;<span style="margin-left:51px">{issue.title} ({location_line})</span></summary>

{issue_body}


</details>
"""
                    )
                summary_parts.append("\n")
    else:
        summary_parts.append(
            '</details>\n<img src="https://img.shields.io/badge/No_Issues Found-brightgreen?style=flat-square" />\n'
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
        console.print(f"[yellow]⚠ Failed to update acknowledgment: {e}[/yellow]")
        import traceback

        traceback.print_exc()
        # Don't fail the whole review if update fails


def work_agent(config: Config, project_id: str, mr_iid: str) -> str:
    api_v4 = config.gitlab_api_v4 + "/api/v4"
    token = config.gitlab_token
    model = get_gpt_model(config.llm_model_name, config.llm_api_key, config.llm_base_url)

    clone_url = build_clone_url(api_v4, project_id, token)

    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)

    # Limit tool calls to prevent agent from wandering
    # For diff review: get_diff (1) + maybe read_file for context (1-2) = 3 max
    settings = ToolCallerSettings(max_tool_calls=5, max_iterations=10)

    # Only provide essential tools - remove search tools to prevent wandering
    tools = [
        get_diff,  # Primary tool: get the diff for the file
        read_file,  # Optional: get additional context if needed
        think,  # Internal reasoning and thought process
    ]

    agent: Agent = create_agent(
        model=model,
        tools=tools,
        # middleware=[check_message_limit, check_agent_messages],  # type: ignore
    )
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    repo_path = clone_repo_persistent(clone_url, branch=branch)
    repo_path = Path(repo_path).resolve()
    repo_tree = tree(repo_path)

    # Parse .reviewignore and filter diffs
    reviewignore_patterns = parse_reviewignore(repo_path)
    filtered_diffs = filter_diffs(diffs, reviewignore_patterns)
    console.print(f"[cyan]Reviewing {len(filtered_diffs)} out of {len(diffs)} changed files[/cyan]")

    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    manager.set_tree(repo_tree)
    manager.set_diffs(filtered_diffs)  # Use filtered diffs instead of all diffs
    manager.get_store()

    issue_store = InMemoryIssueStore()
    token_ctx = store_manager_ctx.set(Context(store_manager=manager, issue_store=issue_store))

    context = store_manager_ctx.get()

    # Create GitLab configuration
    gitlab_config = GitLabConfig(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
    )

    # Create a low-effort agent for simple tasks like acknowledgments and quick scans
    low_effort_model = get_gpt_model_low_effort(
        config.llm_model_name, config.llm_api_key, config.llm_base_url
    )
    low_effort_agent: Agent = create_agent(
        model=low_effort_model,
        tools=[get_diff, think],  # Only needs get_diff for quick scanning
    )

    # Post acknowledgment that review is starting
    console.print("[dim]Posting review acknowledgment...[/dim]")
    acknowledgment_ids = post_review_acknowledgment(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
        agent=low_effort_agent,
        diffs=filtered_diffs,
    )
    if acknowledgment_ids:
        console.print(
            f"[dim]Acknowledgment created: discussion={acknowledgment_ids[0]}, note={acknowledgment_ids[1]}[/dim]"
        )
    else:
        console.print("[yellow]⚠ Failed to create acknowledgment (returned None)[/yellow]")

    try:
        # Define callback to create discussions as each file's review completes
        def on_file_review_complete(file_path: str, issues: list[Any]) -> None:
            """Callback called when a file's review completes."""
            if not issues:
                console.print(f"[dim]No issues found in {file_path}, skipping discussion[/dim]")
                return
            if not config.create_threads:
                console.print(
                    f"[dim]Thread creation disabled, deferring issues in {file_path} to summary[/dim]"
                )
                return

            # Convert IssueModel to Issue domain objects
            from reviewbot.core.issues.issue_model import IssueModel

            domain_issues = [issue.to_domain() for issue in issues if isinstance(issue, IssueModel)]
            handle_file_issues(file_path, domain_issues, gitlab_config, filtered_diffs, diff_refs)

        # Pass the callback to the agent runner
        issues: list[Issue] = agent_runner.invoke(  # type: ignore
            AgentRunnerInput(
                agent=agent,
                context=context,
                settings=settings,
                on_file_complete=on_file_review_complete,
                quick_scan_agent=low_effort_agent,
            )
        )

        console.print(f"[bold cyan]📊 Total issues found: {len(issues)}[/bold cyan]")

        # Update the acknowledgment note with summary
        console.print(f"[dim]Checking acknowledgment_ids: {acknowledgment_ids}[/dim]")
        if acknowledgment_ids:
            discussion_id, note_id = acknowledgment_ids
            console.print(
                f"[dim]Calling update_review_summary for discussion {discussion_id}, note {note_id}...[/dim]"
            )
            update_review_summary(
                api_v4=api_v4,
                token=token,
                project_id=project_id,
                mr_iid=mr_iid,
                discussion_id=discussion_id,
                note_id=note_id,
                issues=issues,
                diffs=filtered_diffs,
                diff_refs=diff_refs,
                agent=low_effort_agent,
            )
            console.print("[dim]update_review_summary completed[/dim]")
        else:
            console.print(
                "[yellow]⚠ No acknowledgment to update (initial acknowledgment may have failed)[/yellow]"
            )

        # Discussions are now created as reviews complete, but we still need to
        # handle any files that might have been processed but had no issues
        # (though the callback already handles this case)

        console.print("[bold green]🎉 All reviews completed and discussions created![/bold green]")
        return "Review completed successfully"

    except Exception as e:
        console.print(f"[bold red]❌ Error during review: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise
    finally:
        store_manager_ctx.reset(token_ctx)


def handle_file_issues(
    file_path: str,
    issues: list[Issue],
    gitlab_config: GitLabConfig,
    file_diffs: list[FileDiff],  # Add this parameter
    diff_refs: dict[str, str],  # Add this parameter (contains base_sha, head_sha, start_sha)
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

    # Severity, Color pairs
    severity_color_pairs = {
        IssueSeverity.HIGH: "#d73a49",  # red
        IssueSeverity.MEDIUM: "#fbca04",  # yellow/orange
        IssueSeverity.LOW: "#28a745",  # green
    }

    max_dedicated_threads = 3
    dedicated_issues: list[Issue] = []
    reply_issues: list[Issue] = []

    for issue in issues:
        needs_dedicated = issue.suggestion is not None or issue.severity == IssueSeverity.HIGH
        if needs_dedicated and len(dedicated_issues) < max_dedicated_threads:
            dedicated_issues.append(issue)
        else:
            reply_issues.append(issue)

    def build_position(issue: Issue) -> dict[str, Any] | None:
        if (
            file_diff
            and base_sha
            and head_sha
            and start_sha
            and file_diff.old_path
            and file_diff.new_path
        ):
            return create_position_for_issue(
                diff_text=file_diff.patch,
                issue_line_start=issue.start_line,
                issue_line_end=issue.end_line,
                base_sha=base_sha,
                head_sha=head_sha,
                start_sha=start_sha,
                old_path=file_diff.old_path,
                new_path=file_diff.new_path,
            )
        return None

    def create_discussion_for_issue(issue: Issue, include_suggestion: bool = True) -> str | None:
        discussion_title = ""
        color = severity_color_pairs[issue.severity].strip("#")
        discussion_body = f"""<img src="https://img.shields.io/badge/{issue.severity.value.upper()}-{color}?style=flat-square" />

{issue.description}
"""
        if include_suggestion and issue.suggestion:
            discussion_body += f"""

{issue.suggestion}
"""

        position = build_position(issue)
        if position:
            console.print(
                f"[dim]Position object for lines {issue.start_line}-{issue.end_line}:[/dim]"
            )
            import json

            console.print(f"[dim]{json.dumps(position, indent=2)}[/dim]")

        try:
            discussion_id = create_discussion(
                title=discussion_title,
                body=discussion_body,
                gitlab_config=gitlab_config,
                position=position,
            )
            issue.discussion_id = discussion_id
            console.print(
                f"[green]✓ Created discussion for issue at lines {issue.start_line}-{issue.end_line} (ID: {discussion_id})[/green]"
            )
            return discussion_id
        except Exception as e:
            if position:
                console.print(
                    f"[yellow]Failed with position for lines {issue.start_line}-{issue.end_line}, retrying without position: {e}[/yellow]"
                )
                try:
                    discussion_id = create_discussion(
                        title=discussion_title,
                        body=discussion_body,
                        gitlab_config=gitlab_config,
                        position=None,
                    )
                    issue.discussion_id = discussion_id
                    console.print(
                        f"[green]✓ Created discussion without position (ID: {discussion_id})[/green]"
                    )
                    return discussion_id
                except Exception as e2:
                    console.print(
                        f"[red]✗ Failed to create discussion for issue at lines {issue.start_line}-{issue.end_line}: {e2}[/red]"
                    )
                    import traceback

                    traceback.print_exc()
                    return None

            console.print(
                f"[red]✗ Failed to create discussion for issue at lines {issue.start_line}-{issue.end_line}: {e}[/red]"
            )
            import traceback

            traceback.print_exc()
            return None

    for issue in dedicated_issues:
        create_discussion_for_issue(issue, include_suggestion=True)

    if reply_issues:
        console.print(
            f"[dim]Leaving {len(reply_issues)} non-dedicated issue(s) for the summary[/dim]"
        )


def generate_line_code(file_path: str, old_line: int | None, new_line: int | None) -> str:
    """
    Generates a GitLab-compatible line_code.
    Format: sha1(path) + "_" + old_line + "_" + new_line
    """
    path_hash = hashlib.sha1(file_path.encode()).hexdigest()
    old_s = str(old_line) if old_line is not None else ""
    new_s = str(new_line) if new_line is not None else ""
    return f"{path_hash}_{old_s}_{new_s}"


def create_position_for_issue(
    diff_text: str,
    issue_line_start: int,
    issue_line_end: int,
    base_sha: str,
    head_sha: str,
    start_sha: str,
    old_path: str,
    new_path: str,
) -> dict[str, Any] | None:
    hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    lines = diff_text.splitlines()

    current_old, current_new = 0, 0
    in_hunk = False

    # Track the actual lines found in the diff to build the range
    matched_lines = []  # List of (old_line, new_line)

    for line in lines:
        match = hunk_header_pattern.match(line)
        if match:
            current_old, current_new = int(match.group(1)), int(match.group(3))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # Logic to determine if this specific line is within our target range
        if line.startswith("+"):
            if issue_line_start <= current_new <= issue_line_end:
                matched_lines.append((None, current_new))
            current_new += 1
        elif line.startswith("-"):
            if issue_line_start <= current_old <= issue_line_end:
                matched_lines.append((current_old, None))
            current_old += 1
        else:
            if issue_line_start <= current_new <= issue_line_end:
                matched_lines.append((current_old, current_new))
            current_old += 1
            current_new += 1

        # FIX: Optimization to prevent "sticky" hunk matching.
        # If we have passed the end_line in the NEW file, we stop.
        if current_new > issue_line_end and not line.startswith("-"):
            if matched_lines:
                break

    if not matched_lines:
        return None

    # We anchor the comment to the LAST line of the range so the code is visible
    start_old, start_new = matched_lines[0]
    end_old, end_new = matched_lines[-1]

    # Calculate line codes for the range
    start_code = generate_line_code(new_path if start_new else old_path, start_old, start_new)
    end_code = generate_line_code(new_path if end_new else old_path, end_old, end_new)

    position = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "start_sha": start_sha,
        "position_type": "text",
        "old_path": old_path,
        "new_path": new_path,
        # Anchor the main comment on the end of the range
        "new_line": end_new,
        "old_line": end_old,
        "line_range": {
            "start": {
                "line_code": start_code,
                "type": "new" if start_new else "old",
                "new_line": start_new,
                "old_line": start_old,
            },
            "end": {
                "line_code": end_code,
                "type": "new" if end_new else "old",
                "new_line": end_new,
                "old_line": end_old,
            },
        },
    }

    # Cleanup: GitLab doesn't like None values in the schema
    if position["new_line"] is None:
        del position["new_line"]
    if position["old_line"] is None:
        del position["old_line"]

    return position


def create_discussion(
    title: str,
    body: str,
    gitlab_config: GitLabConfig,
    position: dict[str, Any] | None = None,
) -> str:
    """
    Create a discussion with title and body.

    Args:
        title: Discussion title
        body: Discussion body content
        gitlab_config: GitLab API configuration
        position: Optional position object for file-based discussions

    Returns:
        Discussion ID from GitLab
    """
    # GitLab discussions don't have separate titles in the API,
    # so we include the title in the body with markdown formatting

    # post_discussion returns (discussion_id, note_id), we only need discussion_id
    discussion_id, _ = post_discussion(
        api_v4=gitlab_config.api_v4,
        token=gitlab_config.token,
        project_id=gitlab_config.project_id,
        mr_iid=gitlab_config.mr_iid,
        body=body,
        position=position,
    )

    return discussion_id


def reply_to_discussion(
    discussion_id: str,
    body: str,
    gitlab_config: GitLabConfig,
) -> None:
    """
    Reply to an existing discussion.

    Args:
        discussion_id: ID of the discussion to reply to
        body: Content of the reply
        gitlab_config: GitLab API configuration
    """
    post_discussion_reply(
        api_v4=gitlab_config.api_v4,
        token=gitlab_config.token,
        project_id=gitlab_config.project_id,
        merge_request_id=gitlab_config.mr_iid,
        discussion_id=discussion_id,
        body=body,
    )


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
