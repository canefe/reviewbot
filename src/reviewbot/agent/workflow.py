import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from reviewbot.infra.gitlab.note import post_discussion, post_discussion_reply
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore
from reviewbot.models.gpt import get_gpt_model
from reviewbot.tools import (
    get_diff,
    read_file,
    search_codebase,
    search_codebase_semantic_search,
)

console = Console()


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
    result_lines: List[str] = []
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
            in_target_range = (
                new_start <= line_end and (new_start + new_count) >= line_start
            )
            continue

        # Skip diff header lines
        if (
            line.startswith("diff --git")
            or line.startswith("---")
            or line.startswith("+++")
        ):
            continue

        # Process diff lines - keep the prefixes to show the actual diff
        # Include context lines to show proper indentation and structure
        if line.startswith("+"):
            # Added line - this is in the new file
            if current_new_line >= line_start and current_new_line <= line_end:
                # Keep the '+' prefix and the entire line with original whitespace
                result_lines.append(line)
            current_new_line += 1
        elif line.startswith("-"):
            continue
            # Removed line - include it to show what was removed
            # Include removals that are in the same hunk as our target range
            # Also include nearby removals for context
            if in_target_range or (
                current_old_line >= line_start - 3 and current_old_line <= line_end + 3
            ):
                # Keep the '-' prefix and the entire line with original whitespace
                result_lines.append(line)
            current_old_line += 1
        elif line.startswith(" "):
            # Context line - this exists in both old and new files
            # Include context lines within the range and a few lines before/after for structure
            if current_new_line >= line_start - 2 and current_new_line <= line_end + 2:
                # Keep the space prefix and the entire line with original whitespace
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


def work_agent(config: Config, project_id: str, mr_iid: str) -> str:
    api_v4 = config.gitlab_api_v4 + "/api/v4"
    token = config.gitlab_token
    print(config.llm_model_name, config.llm_api_key, config.llm_base_url)
    input("Press Enter to continue...")
    model = get_gpt_model(
        config.llm_model_name, config.llm_api_key, config.llm_base_url
    )

    clone_url = build_clone_url(api_v4, project_id, token)

    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)

    settings = ToolCallerSettings(max_tool_calls=10, max_iterations=50)

    tools = [
        search_codebase,
        get_diff,
        read_file,
        search_codebase_semantic_search,
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

    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    manager.set_tree(repo_tree)
    manager.set_diffs(diffs)
    manager.get_store()

    issue_store = InMemoryIssueStore()
    token_ctx = store_manager_ctx.set(
        Context(store_manager=manager, issue_store=issue_store)
    )

    context = store_manager_ctx.get()

    # Create GitLab configuration
    gitlab_config = GitLabConfig(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
    )

    try:
        # Define callback to create discussions as each file's review completes
        def on_file_review_complete(file_path: str, issues: List[Any]) -> None:
            """Callback called when a file's review completes."""
            if not issues:
                console.print(
                    f"[dim]No issues found in {file_path}, skipping discussion[/dim]"
                )
                return

            # Convert IssueModel to Issue domain objects
            from reviewbot.core.issues.issue_model import IssueModel

            domain_issues = [
                issue.to_domain() for issue in issues if isinstance(issue, IssueModel)
            ]
            handle_file_issues(
                file_path, domain_issues, gitlab_config, diffs, diff_refs
            )

        # Pass the callback to the agent runner
        issues: List[Issue] = agent_runner.invoke(  # type: ignore
            AgentRunnerInput(
                agent=agent,
                context=context,
                settings=settings,
                on_file_complete=on_file_review_complete,
            )
        )

        console.print(f"[bold cyan]📊 Total issues found: {len(issues)}[/bold cyan]")

        # Discussions are now created as reviews complete, but we still need to
        # handle any files that might have been processed but had no issues
        # (though the callback already handles this case)

        console.print(
            "[bold green]🎉 All reviews completed and discussions created![/bold green]"
        )
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
    issues: List[Issue],
    gitlab_config: GitLabConfig,
    file_diffs: List[FileDiff],  # Add this parameter
    diff_refs: Dict[
        str, str
    ],  # Add this parameter (contains base_sha, head_sha, start_sha)
) -> None:
    """
    Create one discussion per issue in the file.

    Args:
        file_path: Path to the file being reviewed
        issues: List of issues found in this file
        gitlab_config: GitLab API configuration
        file_diffs: List of file diffs from the MR
        diff_refs: Dict with base_sha, head_sha, start_sha
    """
    if not issues:
        return

    console.print(f"[cyan]Creating {len(issues)} discussions for {file_path}[/cyan]")

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

    # Create one discussion per issue
    for issue in issues:
        discussion_title = ""
        discussion_body = f"""
<img src="https://badgen.net/badge/issue/{issue.severity.value.upper()}/{severity_color_pairs[issue.severity]}" />

{issue.description}
"""

        # Create position specific to this issue
        position = None
        if file_diff and base_sha and head_sha and start_sha:
            position = create_position_for_issue(
                diff_text=file_diff.patch,
                issue_line_start=issue.start_line,  # Your start line
                issue_line_end=issue.end_line,  # Your end line
                base_sha=base_sha,
                head_sha=head_sha,
                start_sha=start_sha,
                old_path=file_diff.old_path,
                new_path=file_diff.new_path,
            )

        # Create discussion for this specific issue
        try:
            discussion_id = create_discussion(
                title=discussion_title,
                body=discussion_body,
                gitlab_config=gitlab_config,
                position=position,
            )
            console.print(
                f"[green]✓ Created discussion for issue at lines {issue.start_line}-{issue.end_line} (ID: {discussion_id})[/green]"
            )
        except Exception as e:
            if position:
                # If position was provided and it failed, try without position
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
                    console.print(
                        f"[green]✓ Created discussion without position (ID: {discussion_id})[/green]"
                    )
                except Exception as e2:
                    console.print(
                        f"[red]✗ Failed to create discussion for issue at lines {issue.start_line}-{issue.end_line}: {e2}[/red]"
                    )
                    import traceback

                    traceback.print_exc()
            else:
                console.print(
                    f"[red]✗ Failed to create discussion for issue at lines {issue.start_line}-{issue.end_line}: {e}[/red]"
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
) -> Optional[Dict[str, Any]]:
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
    hunk_header_pattern = re.compile(
        r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
    )

    lines = diff_text.splitlines()
    current_old = 0
    current_new = 0
    in_hunk = False

    # Track if we've found the target line
    found_old_line = None
    found_new_line = None

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

        # Check if current line matches our target
        if line.startswith("-"):
            # Deletion - only has old line number
            if current_old == issue_line_start:
                found_old_line = current_old
                found_new_line = None
                break
            current_old += 1
        elif line.startswith("+"):
            # Addition - only has new line number
            if current_new >= issue_line_start and current_new <= issue_line_end:
                found_old_line = None
                found_new_line = current_new
                break
            current_new += 1
        else:
            # Context line - has both
            if current_new >= issue_line_start and current_new <= issue_line_end:
                found_old_line = current_old
                found_new_line = current_new
                break
            current_old += 1
            current_new += 1

    # If we didn't find the line in the diff, return None
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
    position: Dict[str, Any] | None = None,
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
