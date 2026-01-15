import asyncio
import time
from collections.abc import Callable
from typing import Any

from ido_agents.agents.ido_agent import create_ido_agent
from ido_agents.agents.tool_runner import ToolCallerSettings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.func import BaseStore, task  # type: ignore
from pydantic import BaseModel, Field
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState, store
from reviewbot.core.issues import IssueModel
from reviewbot.core.issues.issue_model import IssueModelList
from reviewbot.tools.diff import get_diff_from_file
from reviewbot.tools.read_file import read_file_from_store

console = Console()


# Pydantic models for structured outputs
class QuickScanResult(BaseModel):
    """Result of quick scanning a file to determine if it needs deep review."""

    needs_review: bool = Field(
        description="True if the file needs deep review, False if it can be skipped"
    )


class ValidationResult(BaseModel):
    """Result of validating issues against the diff."""

    valid_issues: list[IssueModel] = Field(
        description="Issues that are confirmed to be valid based on the diff"
    )
    removed: list[dict[str, Any]] = Field(
        description="Issues that were removed, each with 'issue' and 'reason' fields"
    )


def get_reasoning_context(store: BaseStore | None) -> str:
    """
    Retrieve stored reasoning history from the store.

    Returns:
        Formatted string of previous reasoning, or empty string if none exists.
    """
    if not store:
        return ""

    try:
        NS = ("reasoning",)
        existing = store.get(NS, "history")

        if not existing:
            return ""

        history_data = existing.value if hasattr(existing, "value") else existing
        if not history_data or not (isinstance(history_data, dict) and history_data.get("items")):
            return ""

        reasoning_history = history_data["items"]
        if not reasoning_history:
            return ""

        # Format reasoning history for context
        formatted = "\n\n**Your Previous Reasoning:**\n"
        for i, reasoning in enumerate(reasoning_history, 1):
            formatted += f"{i}. {reasoning}\n"

        return formatted
    except Exception:
        return ""


@task
async def identify_issues(
    *,
    settings: ToolCallerSettings,
    on_file_complete: Callable[[str, list[IssueModel]], None] | None = None,
    agent: Any,
    quick_scan_agent: Any | None = None,
    model: Any | None = None,
    tools: list[Any] | None = None,
    quick_scan_model: Any | None = None,
    quick_scan_tools: list[Any] | None = None,
    acknowledgment_info: tuple[str, str, Any] | None = None,
) -> list[IssueModel]:
    """
    Identify issues in the codebase using concurrent agents per file.

    Reads CodebaseState from store and runs concurrent reviews.
    Returns list of IssueModel objects.
    """
    # Read codebase state from store
    NS = ("codebase",)
    raw = store.get(NS, "state")
    if not raw:
        raise ValueError("Codebase state not found in store")

    codebase_data = raw.value if hasattr(raw, "value") else raw
    codebase = CodebaseState.model_validate(codebase_data)
    diffs = codebase.diffs

    # Run concurrent reviews
    all_issues = await run_concurrent_reviews(
        agent,
        diffs,
        settings,
        on_file_complete=on_file_complete,
        quick_scan_agent=quick_scan_agent,
        model=model,
        tools=tools,
        quick_scan_model=quick_scan_model,
        quick_scan_tools=quick_scan_tools,
        acknowledgment_info=acknowledgment_info,
    )

    return all_issues


def format_progress_message(
    all_files: list[str],
    completed_files: set[str],
    in_progress_files: dict[asyncio.Future[Any], str],
    max_workers: int,
) -> str:
    """Format a progress message for the acknowledgment note."""
    total = len(all_files)
    completed = len(completed_files)
    in_progress = len(in_progress_files)
    pending = total - completed - in_progress

    # Progress badge
    progress_badge = '<img src="https://img.shields.io/badge/Code_Review-In_Progress-orange?style=flat-square" />'
    progress_text = f"{completed}/{total} files reviewed"

    # Build message
    lines = [progress_badge, "", f"**Review Progress: {progress_text}**", ""]

    # Worker status
    lines.append(f"**Active Workers ({in_progress}/{max_workers}):**")
    if in_progress_files:
        for task, file_path in in_progress_files.items():
            if not task.done():
                lines.append(f"- Reviewing: `{file_path}`")
    else:
        lines.append("- (All workers idle)")

    lines.append("")

    # Completed files
    if completed_files:
        lines.append(f"**Completed ({completed}):**")
        for file_path in sorted(completed_files):
            lines.append(f"- `{file_path}`")
        lines.append("")

    # Pending files
    if pending > 0:
        pending_list = [
            f for f in all_files if f not in completed_files and f not in in_progress_files.values()
        ]
        lines.append(f"**Pending ({pending}):**")
        for file_path in pending_list[:5]:  # Show first 5
            lines.append(f"- `{file_path}`")
        if pending > 5:
            lines.append(f"- ... and {pending - 5} more")

    return "\n".join(lines)


async def monitor_progress(
    task_to_file: dict[asyncio.Future[Any], str],
    start_times: dict[asyncio.Future[Any], float],
    stop_event: asyncio.Event,
    task_timeout: int = 300,  # 5 minutes per task
    acknowledgment_info: tuple[str, str, Any] | None = None,
    all_files: list[str] | None = None,
    completed_files: set[str] | None = None,
    max_workers: int = 3,
):
    """
    Monitor coroutine that logs the status of ongoing tasks and updates acknowledgment.
    """
    update_interval = 10  # Update every 10 seconds
    last_update = time.time()

    while not stop_event.is_set():
        await asyncio.sleep(5)  # Check every 5 seconds

        current_time = time.time()

        # Log status to console
        for io_task, file_path in task_to_file.items():
            if not io_task.done():
                start_time = start_times.get(io_task)
                if start_time is None:
                    continue
                elapsed = current_time - start_time
                if elapsed > task_timeout:
                    console.print(
                        f"[red]TIMEOUT WARNING: {file_path} has been running for {elapsed:.0f}s[/red]"
                    )

        # Update acknowledgment note every 10 seconds
        if acknowledgment_info and all_files and completed_files is not None:
            if current_time - last_update >= update_interval:
                try:
                    discussion_id, note_id, gitlab_config = acknowledgment_info
                    progress_message = format_progress_message(
                        all_files, completed_files, task_to_file, max_workers
                    )

                    # Import here to avoid circular dependency
                    from reviewbot.infra.gitlab.note import async_update_discussion_note

                    await async_update_discussion_note(
                        api_v4=gitlab_config.get_api_base_url(),
                        token=gitlab_config.token.get_secret_value(),
                        project_id=gitlab_config.get_project_identifier(),
                        mr_iid=gitlab_config.get_pr_identifier(),
                        discussion_id=discussion_id,
                        note_id=note_id,
                        body=progress_message,
                    )
                    last_update = current_time
                except Exception as e:
                    console.print(f"[yellow]Failed to update progress note: {e}[/yellow]")


async def run_concurrent_reviews(
    agent: Any,
    diffs: list[Any],
    settings: ToolCallerSettings,
    max_workers: int = 3,  # Limit concurrency to avoid rate limit issues
    task_timeout: int = 160,  # 5 minutes timeout per file
    on_file_complete: Callable[[str, list[IssueModel]], None] | None = None,
    quick_scan_agent: Any | None = None,
    model: Any | None = None,
    tools: list[Any] | None = None,
    quick_scan_model: Any | None = None,
    quick_scan_tools: list[Any] | None = None,
    acknowledgment_info: tuple[str, str, Any] | None = None,
) -> list[IssueModel]:
    """
    Run concurrent reviews of all diff files with monitoring.

    Args:
        agent: The agent to use for reviews
        diffs: List of diff objects
        settings: Tool caller settings
        max_workers: Maximum number of concurrent workers
        task_timeout: Timeout per task in seconds
        on_file_complete: Optional callback function called when each file's review completes.
                         Receives (file_path, issues) as arguments.
        quick_scan_agent: Optional low-effort agent for quick prerequisite scanning.
                         If provided, files are scanned first to determine if deep review is needed.
    """
    diff_file_paths = [diff.new_path for diff in diffs]

    console.print(f"[bold]Starting concurrent review of {len(diff_file_paths)} files[/bold]")
    console.print(f"[dim]Files: {', '.join(diff_file_paths)}[/dim]\n")

    all_issues: list[IssueModel] = []
    completed_files: set[str] = set()

    semaphore = asyncio.Semaphore(max_workers)
    task_to_file: dict[asyncio.Future[Any], str] = {}
    start_times: dict[asyncio.Future[Any], float] = {}

    # Wrapper to track file_path with result
    async def review_with_tracking(file_path: str) -> tuple[str, list[IssueModel]]:
        async with semaphore:
            task = asyncio.current_task()
            if task is not None:
                start_times[task] = time.time()
                task_to_file[task] = file_path

            result = await asyncio.wait_for(
                review_single_file_wrapper(
                    file_path=file_path,
                    agent=agent,
                    settings=settings,
                    quick_scan_agent=quick_scan_agent,
                    model=model,
                    tools=tools,
                    quick_scan_model=quick_scan_model,
                    quick_scan_tools=quick_scan_tools,
                ),
                timeout=task_timeout,
            )
            return file_path, result

    # Create tasks
    tasks = [asyncio.create_task(review_with_tracking(fp)) for fp in diff_file_paths]

    stop_monitor = asyncio.Event()
    monitor_task = asyncio.create_task(
        monitor_progress(
            task_to_file,
            start_times,
            stop_monitor,
            task_timeout,
            acknowledgment_info=acknowledgment_info,
            all_files=diff_file_paths,
            completed_files=completed_files,
            max_workers=max_workers,
        )
    )

    try:
        for coro in asyncio.as_completed(tasks):
            try:
                file_path, issues = await coro
                all_issues.extend(issues)
                completed_files.add(file_path)
                console.print(f"[green]✓[/green] Processed {file_path}: {len(issues)} issues")

                # Call the callback if provided, allowing immediate discussion creation
                if on_file_complete:
                    try:
                        on_file_complete(file_path, issues)
                    except Exception as e:
                        console.print(
                            f"[red]Error in on_file_complete callback for {file_path}: {e}[/red]"
                        )
                        import traceback

                        traceback.print_exc()
            except TimeoutError:
                console.print(f"[red]✗[/red] TIMEOUT: A file took longer than {task_timeout}s")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed: {e}")
                import traceback

                traceback.print_exc()
    finally:
        stop_monitor.set()
        await monitor_task

    console.print(
        f"\n[bold green]Review complete! Total issues found: {len(all_issues)}[/bold green]"
    )

    if all_issues:
        console.print("\n[bold]Issues by file:[/bold]")
        file_issue_count: dict[str, int] = {}
        for issue in all_issues:
            file_path = getattr(issue, "file_path", "unknown")
            file_issue_count[file_path] = file_issue_count.get(file_path, 0) + 1

        for file_path, count in sorted(file_issue_count.items()):
            console.print(f"  • {file_path}: {count} issues")

    return all_issues


async def quick_scan_file(
    agent: Any,
    file_path: str,
    settings: ToolCallerSettings,
    model: Any | None = None,
    tools: list[Any] | None = None,
) -> bool:
    """
    Quick scan with low-effort agent to determine if file needs deep review.
    Returns True if file needs deep review, False otherwise.
    """
    # Fetch the diff first to include in prompt

    try:
        diff_content = get_diff_from_file(agent.store, file_path)
        file_content = read_file_from_store(agent.store, file_path)
    except Exception as e:
        console.print(f"[yellow]Could not fetch diff for {file_path}: {e}[/yellow]")
        return True  # If can't get diff, do deep review to be safe

    messages: list[BaseMessage] = [
        SystemMessage(
            content="""You are a code review triage assistant. Your job is to quickly determine if a file change needs deep review.

Review the diff and decide if this file needs detailed analysis. Set needs_review=true if ANY of these apply:
- New code that implements business logic
- Changes to security-sensitive code (auth, permissions, data validation)
- Database queries or migrations
- API endpoint changes
- Complex algorithms or data structures
- Error handling changes
- Configuration changes that affect behavior
- Use tool 'think' to reason. You must reason at least 10 times before giving an answer

Set needs_review=false if:
- Only formatting/whitespace changes
- Simple refactoring (renaming variables/functions)
- Adding/updating comments or documentation only
- Import reordering
- Trivial changes (typo fixes in strings, adding logging)"""
        ),
        HumanMessage(
            content=f"""Quickly scan this file and determine if it needs deep review: {file_path}


Here is the file:
{file_content}

Here is the diff:
{diff_content}
"""
        ),
    ]

    try:
        console.print(f"[dim]Quick scanning: {file_path}[/dim]")

        if model is None:
            raise ValueError("model parameter is required for ido-agents migration")

        ido_agent = create_ido_agent(model=model, tools=tools or [])
        result = await (
            ido_agent.with_structured_output(QuickScanResult)
            .with_tool_caller(settings)
            .with_retry(max_retries=3)
            .ainvoke(messages)
        )

        if result.needs_review:
            console.print(f"[yellow]✓ Needs deep review: {file_path}[/yellow]")
        else:
            console.print(f"[dim]⊘ Skipping deep review: {file_path}[/dim]")

        return result.needs_review
    except Exception as e:
        console.print(
            f"[yellow]Quick scan failed for {file_path}, defaulting to deep review: {e}[/yellow]"
        )
        return True  # If scan fails, do deep review to be safe


async def review_single_file_wrapper(
    file_path: str,
    agent: Any,
    settings: ToolCallerSettings,
    quick_scan_agent: Any | None = None,
    model: Any | None = None,
    tools: list[Any] | None = None,
    quick_scan_model: Any | None = None,
    quick_scan_tools: list[Any] | None = None,
) -> list[IssueModel]:
    """
    Wrapper for reviewing a single file with optional quick scan.
    This runs per async task.
    """
    try:
        # Quick scan first if agent provided
        if quick_scan_agent:
            needs_deep_review = await quick_scan_file(
                quick_scan_agent, file_path, settings, quick_scan_model, quick_scan_tools
            )
            if not needs_deep_review:
                console.print(f"[dim]Skipping deep review for: {file_path}[/dim]")
                return []

        # Now call the actual review function
        return await review_single_file(agent, file_path, settings, model, tools)
    except Exception as e:
        console.print(f"[red]Exception in task for {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []


async def review_single_file(
    agent: Any,
    file_path: str,
    settings: ToolCallerSettings,
    model: Any | None = None,
    tools: list[Any] | None = None,
) -> list[IssueModel]:
    """
    Review a single diff file and return issues found.
    """
    # Get any previous reasoning context
    reasoning_context = get_reasoning_context(agent.store)

    # Force a reasoning pass to ensure think() is invoked during deep review
    diff_content = get_diff_from_file(agent.store, file_path)
    file_content = read_file_from_store(agent.store, file_path)
    if not file_content:
        file_content = ""

    messages: list[BaseMessage] = [
        SystemMessage(
            content=f"""You are a senior code reviewer analyzing code changes for bugs, security issues, and logic errors.

AVAILABLE TOOLS:
- `think()` - Record your internal reasoning (use this to analyze the code)
- `get_diff(file_path)` - Get the diff for the file being reviewed
- `read_file(file_path)` - Read the COMPLETE file to see full context beyond the diff
- `read_file(file_path, line_start, line_end)` - Read specific line ranges
- `ls_dir(dir_path)` - List contents of a directory to explore the codebase structure

IMPORTANT: CONTEXT LIMITATIONS
The diff shows only the changed lines, not the full file. When you need to verify something outside the diff (like imports, variable declarations, or function definitions), use `read_file()` to see the complete context.

Use `read_file()` when:
- You suspect undefined variables/imports but they might exist elsewhere in the file
- You need to understand surrounding code to assess impact
- The change references code not shown in the diff

HANDLING NEW FILES:
If `read_file()` returns an error stating the file is NEW:
- This file doesn't exist yet in the repository
- You can only see what's in the diff
- Be lenient about imports/definitions (assume they're complete in the actual PR)
- Focus on logic bugs, security issues, and clear errors in the visible code

REASONING TOOL:
- Use `think()` to record your analysis process{reasoning_context}
- Call `think()` before producing your final output
- Document your reasoning about each potential issue

Your task: Review the file '{file_path}' and identify actionable issues.

WHAT TO REPORT:
- **Critical bugs** - Code that will crash, throw errors, or produce incorrect results
- **Security vulnerabilities** - SQL injection, XSS, authentication bypass, etc.
- **Logic errors** - Incorrect algorithms, wrong conditions, broken business logic
- **Data corruption risks** - Code that could corrupt data or cause inconsistent state
- **Performance problems** - Clear bottlenecks like O(n²) where O(n) is possible
- **Breaking changes** - Changes that break existing APIs or functionality

WHAT NOT TO REPORT:
- Code style preferences (naming, formatting, organization)
- Missing documentation or comments
- Minor refactoring suggestions that don't fix bugs
- Hypothetical edge cases without evidence they're relevant
- Issues based on assumptions about the environment (e.g., "X might not be installed")
- Version numbers or package versions you're unfamiliar with (they may be newer than your training)
- Import paths or APIs you don't recognize (they may have changed since your training)

IMPORTANT:
- Do NOT invent issues to justify the review
- Only report issues with direct evidence in the code shown

SEVERITY GUIDELINES:
- **HIGH**: Crashes, security vulnerabilities, data corruption, broken functionality
- **MEDIUM**: Logic errors, performance issues, likely bugs in edge cases
- **LOW**: Minor issues that could cause problems in rare scenarios

SUGGESTIONS:
When you identify an issue with a clear fix, provide a `suggestion` field with the corrected code.
Format as a diff showing the old and new code:
- Lines starting with `-` show old code to remove
- Lines starting with `+` show new code to add
- Preserve exact indentation from the original

OUTPUT:
Return a JSON array of issues. If no issues are found, return an empty array: []
Each issue must have: title, description, severity, file_path, start_line, end_line, and optionally suggestion.

Be specific and reference exact line numbers from the diff."""
        ),
        HumanMessage(
            content=f"""Review the merge request diff for the file: {file_path}

File content:
{file_content}

Diff:
{diff_content}
"""
        ),
    ]

    try:
        console.print(f"[cyan]Starting review of: {file_path}[/cyan]")

        if model is None:
            raise ValueError("model parameter is required for ido-agents migration")

        # Use retry logic for the LLM call with structured output
        ido_agent = create_ido_agent(model=model, tools=tools or [])
        issues_result = await (
            ido_agent.with_structured_output(IssueModelList)
            .with_tool_caller(settings)
            .with_retry(max_retries=3)
            .ainvoke(messages)
        )

        # Extract the actual list from the RootModel
        issues = issues_result.root

        console.print(f"[green]Completed review of: {file_path}[/green]")
        console.print(f"Found {len(issues)} potential issues")

        if issues:
            # Validate issues against the diff to reduce hallucinations before creating notes.
            issues = await validate_issues_for_file(
                agent, file_path, issues, settings, model, tools
            )

        console.print(f"[blue]Found {len(issues)} issues in {file_path}[/blue]")
        return issues

    except Exception as e:
        console.print(f"[red]Error reviewing {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []


async def validate_issues_for_file(
    agent: Any,
    file_path: str,
    issues: list[IssueModel],
    settings: ToolCallerSettings,
    model: Any | None = None,
    tools: list[Any] | None = None,
) -> list[IssueModel]:
    if not issues:
        return []

    try:
        diff_content = get_diff_from_file(agent.store, file_path)
    except Exception as e:
        console.print(f"[yellow]Issue validation skipped for {file_path}: {e}[/yellow]")
        return []

    # Use JSON-friendly payload so enums serialize cleanly.
    issues_payload = [issue.model_dump(mode="json") for issue in issues]

    # Import json module for dumps
    import json

    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are an issue validator. Your job is to remove FALSE POSITIVES while keeping real bugs.\n\n"
                "The codebase already has been linted, built, formatted and compiled successfully. Make sure to remove 'issues' that claim otherwise."
                "AVAILABLE TOOLS:\n"
                "- `read_file(file_path)` - Read the complete file to verify issues\n"
                "- `ls_dir(dir_path)` - List directory contents to verify file structure\n\n"
                "WHAT TO REMOVE (false positives):\n"
                "- 'Variable X undefined' - when X is actually defined elsewhere in the file\n"
                "- 'Import Y missing' - when Y exists at the top of the file\n"
                "- 'Function Z not declared' - when Z is defined in the complete file\n\n"
                "- 'Compile error' - the codebase is already compiled successfully.\n"
                "WHAT TO KEEP (real issues):\n"
                "- Logic errors - wrong conditions, broken algorithms, incorrect business logic\n"
                "- Security vulnerabilities - SQL injection, XSS, auth bypass, etc.\n"
                "- Bugs that will crash or produce wrong results\n"
                "- Data corruption risks\n"
                "- Performance problems\n\n"
                "RULES:\n"
                "- KEEP issues about logic, bugs, security, and functionality\n"
                "- ONLY remove issues that are provably false (use read_file to verify)\n"
                "- When in doubt, KEEP the issue - don't filter out real bugs\n"
                "- Do NOT create new issues\n"
                "- Do NOT modify issue fields"
            )
        ),
        HumanMessage(
            content=f"""File: {file_path}

Diff (shows only changes):
```diff
{diff_content}
```

Issues to validate:
{json.dumps(issues_payload, indent=2)}

TASK:
1. For issues about "undefined/missing" code, use `read_file("{file_path}")` to check if the code actually exists elsewhere
2. Remove ONLY clear false positives
3. Keep all logic bugs, security issues, and real functionality problems

Return a ValidationResult with:
- valid_issues: confirmed real issues
- removed: false positives with reason for removal"""
        ),
    ]

    validation_settings = ToolCallerSettings(max_tool_calls=10)  # Allow tool calls for validation

    try:
        if model is None:
            raise ValueError("model parameter is required for ido-agents migration")

        ido_agent = create_ido_agent(model=model, tools=tools or [])
        result = await (
            ido_agent.with_structured_output(ValidationResult)
            .with_tool_caller(validation_settings)
            .with_retry(max_retries=3)
            .ainvoke(messages)
        )
    except Exception as e:
        console.print(f"[yellow]Issue validation failed for {file_path}: {e}[/yellow]")
        return issues

    if result.removed:
        console.print(
            f"[dim]Issue validation removed {len(result.removed)} issue(s) in {file_path}[/dim]"
        )
        for entry in result.removed:
            reason = entry.get("reason", "").strip()
            issue = entry.get("issue", {})
            title = issue.get("title", "Untitled issue")
            if reason:
                console.print(f"[dim]- {title}: {reason}[/dim]")
            else:
                console.print(f"[dim]- {title}: no reason provided[/dim]")

    return result.valid_issues
