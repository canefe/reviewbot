import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _extract_code_from_diff(diff_patch: str, line_start: int, line_end: int) -> str:
    """
    Extract code lines from a unified diff patch for a given line range.

    Args:
        diff_patch: The unified diff patch string
        line_start: Starting line number (1-indexed, in the new file)
        line_end: Ending line number (1-indexed, in the new file)

    Returns:
        String containing the code lines from the diff
    """
    import re

    lines = diff_patch.splitlines(keepends=True)
    result_lines: list[str] = []
    current_new_line = 0
    current_old_line = 0
    in_target_range = False

    # Pattern to match hunk headers: @@ -old_start,old_count +new_start,new_count @@
    hunk_header_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    for line in lines:
        # Check if this is a hunk header
        match = hunk_header_re.match(line)
        if match:
            # new_start is the line number in the new file where this hunk starts
            new_start = int(match.group(3))
            old_start = int(match.group(1))
            current_new_line = new_start
            current_old_line = old_start
            # Check if this hunk overlaps with our target range
            new_count = int(match.group(4)) if match.group(4) else 1
            in_target_range = new_start <= line_end and (new_start + new_count) >= line_start
            continue

        # Skip diff header lines
        if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
            continue

        # Process diff lines - keep the prefixes to show the actual diff
        # Include context lines to show proper indentation and structure
        if line.startswith("+"):
            # Added line - this is in the new file
            if current_new_line >= line_start and current_new_line <= line_end:
                # Ensure space after '+' for proper markdown diff formatting
                if len(line) > 1 and line[1] != " ":
                    formatted_line = "+ " + line[1:]
                else:
                    formatted_line = line
                result_lines.append(formatted_line)
            current_new_line += 1
        elif line.startswith("-"):
            continue
            # Removed line - include it to show what was removed
            # Include removals that are in the same hunk as our target range
            # Also include nearby removals for context
            if in_target_range or (
                current_old_line >= line_start - 3 and current_old_line <= line_end + 3
            ):
                # Ensure space after '-' for proper markdown diff formatting
                if len(line) > 1 and line[1] != " ":
                    formatted_line = "- " + line[1:]
                else:
                    formatted_line = line
                result_lines.append(formatted_line)
            current_old_line += 1
        elif line.startswith(" "):
            # Context line - this exists in both old and new files
            # Include context lines within the range and a few lines before/after for structure
            if current_new_line >= line_start - 2 and current_new_line <= line_end + 2:
                # Context lines already have space prefix
                result_lines.append(line)
            current_new_line += 1
            current_old_line += 1
        elif line.startswith("\\"):
            # End of file marker - skip
            continue

    return "".join(result_lines)


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

        # Check if any discussion contains the acknowledgment marker
        acknowledgment_marker = "🤖 **Code Review Starting**"
        for discussion in discussions:
            notes = discussion.get("notes", [])
            for note in notes:
                body = note.get("body", "")
                if acknowledgment_marker in body:
                    console.print(
                        "[dim]Acknowledgment already exists, returning existing IDs[/dim]"
                    )
                    # Return the existing discussion and note IDs
                    discussion_id = discussion.get("id")
                    note_id = note.get("id")
                    if discussion_id and note_id:
                        return (str(discussion_id), str(note_id))
                    return None
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
        acknowledgment_body = f"""🤖 **Code Review Starting**

{summary}

---
*Review powered by ReviewBot*
"""

        discussion_id = post_discussion(
            api_v4=api_v4,
            token=token,
            project_id=project_id,
            mr_iid=mr_iid,
            body=acknowledgment_body,
        )

        # Get the note ID from the discussion
        # The first note in a discussion is the original note
        discussions = get_all_discussions(
            api_v4=api_v4,
            token=token,
            project_id=project_id,
            mr_iid=mr_iid,
        )

        note_id = None
        for discussion in discussions:
            if str(discussion.get("id")) == str(discussion_id):
                notes = discussion.get("notes", [])
                if notes:
                    note_id = str(notes[0].get("id"))
                    break

        if not note_id:
            console.print("[yellow]⚠ Created discussion but could not find note ID[/yellow]")
            return None

        console.print("[green]✓ Posted review acknowledgment[/green]")
        return (str(discussion_id), note_id)

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
                content="""You are a code review assistant. Generate a concise, professional summary of a code review with reasoning.

IMPORTANT:
- Keep it SHORT (3-5 sentences max)
- Provide reasoning about the overall code quality
- Highlight key concerns or positive aspects
- Be constructive and professional
- DO NOT use any tools
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

Generate a brief summary (3-5 sentences) that:
1. Provides overall assessment of the code quality
2. Highlights the most important concerns (if any)
3. Gives reasoning about the review findings
4. Is constructive and actionable

If no issues were found, celebrate the good code quality."""
            ),
        ]

        summary_settings = ToolCallerSettings(max_tool_calls=0, max_iterations=1)
        llm_summary = tool_caller(agent, messages, summary_settings)

    except Exception as e:
        console.print(f"[yellow]⚠ Failed to generate LLM summary: {e}[/yellow]")
        llm_summary = "Review completed successfully."

    # Build final summary combining statistics and LLM reasoning
    summary_parts = [
        "🤖 **Code Review Complete**\n\n",
        f"Reviewed **{total_files}** file(s), found **{len(issues)}** issue(s) across **{files_with_issues}** file(s).\n\n",
        "**Summary:**\n",
        f"{llm_summary}\n\n",
    ]

    if issues:
        summary_parts.append("**Issue Breakdown:**\n")
        if high_count > 0:
            summary_parts.append(f"- 🔴 **{high_count}** High severity\n")
        if medium_count > 0:
            summary_parts.append(f"- 🟠 **{medium_count}** Medium severity\n")
        if low_count > 0:
            summary_parts.append(f"- 🟢 **{low_count}** Low severity\n")

        summary_parts.append("\n**Issues by File:**\n")
        for file_path, file_issues in sorted(issues_by_file.items()):
            high = sum(1 for i in file_issues if i.severity == IssueSeverity.HIGH)
            medium = sum(1 for i in file_issues if i.severity == IssueSeverity.MEDIUM)
            low = sum(1 for i in file_issues if i.severity == IssueSeverity.LOW)
            severity_badges = []
            if high > 0:
                severity_badges.append(f"🔴 {high}")
            if medium > 0:
                severity_badges.append(f"🟠 {medium}")
            if low > 0:
                severity_badges.append(f"🟢 {low}")
            badges = " ".join(severity_badges)
            summary_parts.append(f"- `{file_path}`: {len(file_issues)} issue(s) {badges}\n")
    else:
        summary_parts.append("\n✅ **No issues found!**\n")

    summary_parts.append("\n---\n*Review powered by ReviewBot*")

    summary_body = "".join(summary_parts)

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
        tools=[get_diff],  # Only needs get_diff for quick scanning
    )

    # Post acknowledgment that review is starting
    acknowledgment_ids = post_review_acknowledgment(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
        agent=low_effort_agent,
        diffs=filtered_diffs,
    )

    try:
        # Define callback to create discussions as each file's review completes
        def on_file_review_complete(file_path: str, issues: list[Any]) -> None:
            """Callback called when a file's review completes."""
            if not issues:
                console.print(f"[dim]No issues found in {file_path}, skipping discussion[/dim]")
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
        if acknowledgment_ids:
            discussion_id, note_id = acknowledgment_ids
            update_review_summary(
                api_v4=api_v4,
                token=token,
                project_id=project_id,
                mr_iid=mr_iid,
                discussion_id=discussion_id,
                note_id=note_id,
                issues=issues,
                diffs=filtered_diffs,
                agent=low_effort_agent,
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
    Create one discussion per file with the first issue, and reply with subsequent issues.

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
        IssueSeverity.HIGH: "red",
        IssueSeverity.MEDIUM: "orange",
        IssueSeverity.LOW: "green",
    }

    discussion_id = None

    # Process the first issue - create a discussion with position
    first_issue = issues[0]
    discussion_title = ""

    # Build the discussion body with optional suggestion
    discussion_body = f"""<img src="https://badgen.net/badge/issue/{first_issue.severity.value.upper()}/{severity_color_pairs[first_issue.severity]}" />

{first_issue.description}
"""

    # Add suggestion if available (GitLab will render it as an applicable suggestion)
    if first_issue.suggestion:
        discussion_body += f"""

```suggestion
{first_issue.suggestion}
```
"""

    # Create position for the first issue
    position = None
    if (
        file_diff
        and base_sha
        and head_sha
        and start_sha
        and file_diff.old_path
        and file_diff.new_path
    ):
        position = create_position_for_issue(
            diff_text=file_diff.patch,
            issue_line_start=first_issue.start_line,
            issue_line_end=first_issue.end_line,
            base_sha=base_sha,
            head_sha=head_sha,
            start_sha=start_sha,
            old_path=file_diff.old_path,
            new_path=file_diff.new_path,
        )

    # Create discussion for the first issue
    try:
        discussion_id = create_discussion(
            title=discussion_title,
            body=discussion_body,
            gitlab_config=gitlab_config,
            position=position,
        )
        console.print(
            f"[green]✓ Created discussion for issue at lines {first_issue.start_line}-{first_issue.end_line} (ID: {discussion_id})[/green]"
        )
    except Exception as e:
        if position:
            # If position was provided and it failed, try without position
            console.print(
                f"[yellow]Failed with position for lines {first_issue.start_line}-{first_issue.end_line}, retrying without position: {e}[/yellow]"
            )
            try:
                discussion_id = create_discussion(
                    title=discussion_title,
                    body=discussion_body,
                    gitlab_config=gitlab_config,
                    position=None,
                )
                console.print(
                    f"[green]✓ Created discussion without position (ID: {discussion_id})[/green]"
                )
            except Exception as e2:
                console.print(
                    f"[red]✗ Failed to create discussion for issue at lines {first_issue.start_line}-{first_issue.end_line}: {e2}[/red]"
                )
                import traceback

                traceback.print_exc()
                return  # Can't proceed without a discussion
        else:
            console.print(
                f"[red]✗ Failed to create discussion for issue at lines {first_issue.start_line}-{first_issue.end_line}: {e}[/red]"
            )
            import traceback

            traceback.print_exc()
            return  # Can't proceed without a discussion

    # Process remaining issues - reply to the discussion with diff blocks
    for issue in issues[1:]:
        if not discussion_id:
            console.print(
                f"[yellow]⚠ Skipping issue at lines {issue.start_line}-{issue.end_line} (no discussion created)[/yellow]"
            )
            continue

        # Extract the relevant code from the diff
        code_snippet = ""
        if file_diff:
            code_snippet = _extract_code_from_diff(
                file_diff.patch,
                issue.start_line,
                issue.end_line,
            )

        # Format the reply with a diff block and optional suggestion
        reply_body = f"""<img src="https://badgen.net/badge/issue/{issue.severity.value.upper()}/{severity_color_pairs[issue.severity]}" />

{issue.description}
"""

        # Add suggestion if available (GitLab will render it as an applicable suggestion)
        if issue.suggestion:
            reply_body += f"""

```suggestion
{issue.suggestion}
```
"""
        else:
            # If no suggestion, show the diff context
            reply_body += f"""

```diff
{code_snippet}
```
"""

        # Reply to the discussion
        try:
            reply_to_discussion(
                discussion_id=discussion_id,
                body=reply_body,
                gitlab_config=gitlab_config,
            )
            console.print(
                f"[green]✓ Added reply for issue at lines {issue.start_line}-{issue.end_line}[/green]"
            )
        except Exception as e:
            console.print(
                f"[red]✗ Failed to reply for issue at lines {issue.start_line}-{issue.end_line}: {e}[/red]"
            )
            import traceback

            traceback.print_exc()


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
    """
    Create a GitLab position object for a specific issue line range.

    Args:
        diff_text: The full diff text for the file
        issue_line_start: Start line number of the issue (in new file)
        issue_line_end: End line number of the issue (in new file)
        base_sha, head_sha, start_sha: GitLab diff refs
        old_path, new_path: File paths

    Returns:
        Position dict for GitLab API, or None if line not found in diff
    """
    hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    lines = diff_text.splitlines()
    current_old = 0
    current_new = 0
    in_hunk = False

    # Track all candidate lines in the range
    # Priority: added lines > context lines > deleted lines
    added_lines = []
    context_lines = []
    deleted_lines = []

    for line in lines:
        # Check for hunk header
        match = hunk_header_pattern.match(line)
        if match:
            current_old = int(match.group(1))
            current_new = int(match.group(3))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # Collect all matching lines in the range
        if line.startswith("-"):
            # Deletion - only has old line number
            if current_old >= issue_line_start and current_old <= issue_line_end:
                deleted_lines.append((current_old, None))
            current_old += 1
        elif line.startswith("+"):
            # Addition - only has new line number
            if current_new >= issue_line_start and current_new <= issue_line_end:
                added_lines.append((None, current_new))
            current_new += 1
        else:
            # Context line - has both
            if current_new >= issue_line_start and current_new <= issue_line_end:
                context_lines.append((current_old, current_new))
            current_old += 1
            current_new += 1

    # Choose the best line to anchor the discussion:
    # 1. Prefer the first added line (issues are usually about new code)
    # 2. Fall back to middle context line
    # 3. Finally use deleted line or start line
    found_old_line = None
    found_new_line = None

    if added_lines:
        # Use the first added line in the range
        found_old_line, found_new_line = added_lines[0]
    elif context_lines:
        # Use the middle context line
        mid_idx = len(context_lines) // 2
        found_old_line, found_new_line = context_lines[mid_idx]
    elif deleted_lines:
        # Use the first deleted line
        found_old_line, found_new_line = deleted_lines[0]

    # If we didn't find any line in the diff, return None
    if found_old_line is None and found_new_line is None:
        return None

    # Create position object
    position = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "start_sha": start_sha,
        "old_path": old_path,
        "new_path": new_path,
        "position_type": "text",
    }

    if found_new_line is not None:
        position["new_line"] = found_new_line

    if found_old_line is not None:
        position["old_line"] = found_old_line

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

    discussion_id = post_discussion(
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
