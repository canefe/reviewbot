import asyncio
import time
from collections.abc import Callable
from typing import Any

from idoagents.agents.ido_agent import create_ido_agent
from idoagents.agents.tool_runner import ToolCallerSettings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.func import BaseStore, task  # type: ignore
from pydantic import BaseModel, Field
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState, store
from reviewbot.core.issues import IssueModel
from reviewbot.core.issues.issue_model import IssueModelList
from reviewbot.tools.diff import get_diff_from_file

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
        if not history_data or not history_data.get("items"):
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
    )

    return all_issues


async def monitor_progress(
    task_to_file: dict[asyncio.Future[Any], str],
    start_times: dict[asyncio.Future[Any], float],
    stop_event: asyncio.Event,
    task_timeout: int = 300,  # 5 minutes per task
):
    """
    Monitor coroutine that logs the status of ongoing tasks.
    """
    while not stop_event.is_set():
        await asyncio.sleep(10)  # Check every 10 seconds

        current_time = time.time()
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
                else:
                    console.print(
                        f"[yellow]Still processing: {file_path} ({elapsed:.0f}s elapsed)[/yellow]"
                    )


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

    semaphore = asyncio.Semaphore(max_workers)
    task_to_file: dict[asyncio.Future[Any], str] = {}
    start_times: dict[asyncio.Future[Any], float] = {}

    async def review_with_semaphore(file_path: str) -> list[IssueModel]:
        async with semaphore:
            task = asyncio.current_task()
            if task is not None:
                start_times[task] = time.time()
            return await asyncio.wait_for(
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
        monitor_progress(task_to_file, start_times, stop_monitor, task_timeout)
    )

    try:
        for coro in asyncio.as_completed(tasks):
            try:
                file_path, issues = await coro
                all_issues.extend(issues)
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

Here is the diff:

```diff
{diff_content}
```"""
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
    try:
        diff_content = get_diff_from_file(agent.store, file_path)
        think_messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a senior code reviewer. You must think and review. "
                    "with 2-5 sentences of reasoning about the provided diff. "
                    "Do not use any other tools. Once finished, reply with the "
                    "single word DONE."
                )
            ),
            HumanMessage(
                content=f"""Diff for {file_path}:

```diff
{diff_content}
```
""",
            ),
        ]
        if model:
            think_settings = ToolCallerSettings(max_tool_calls=40)
            ido_agent = create_ido_agent(model=model, tools=tools or [])
            await ido_agent.with_tool_caller(think_settings).ainvoke(think_messages)
    except Exception as e:
        console.print(f"[yellow]⚠ Failed to record reasoning for {file_path}: {e}[/yellow]")

    messages: list[BaseMessage] = [
        SystemMessage(
            content=f"""You are a senior code reviewer analyzing a specific file change.

REASONING TOOL:
- You have access to a `think()` tool for recording your internal reasoning
- Use it to plan your approach, analyze patterns, or reason about potential issues
- Your reasoning is stored and will be available in subsequent requests
- This helps maintain context and improves review quality{reasoning_context}
 - During deep reviews, you MUST call think() before producing your JSON output

Your task: Review ONLY the file '{file_path}' from the merge request diff.

IMPORTANT GUIDELINES:
- Be CONSERVATIVE: Only report real, actionable issues - not stylistic preferences or nitpicks
- If there are NO legitimate issues, return an empty array: []
- Do NOT invent issues to justify the review
- Only report issues with clear negative impact (bugs, security risks, performance problems, logic errors)
- Avoid reporting issues about code style, formatting, or personal preferences unless they violate critical standards
- Medium/High severity issues should be reserved for actual bugs, security vulnerabilities, or broken functionality
- The `description` field MUST include a short plain-text explanation (1-3 sentences).

CRITICAL - KNOWLEDGE CUTOFF AWARENESS:
Your training data has a cutoff date. The code you're reviewing may use:
- Package versions released AFTER your training (e.g., v2, v3 of libraries)
- Language versions you don't know about (e.g., Go 1.23+, Python 3.13+)
- Import paths that have changed since your training
- APIs that have been updated

DO NOT FLAG as issues:
Version numbers (e.g., "Go 1.25 doesn't exist" - it might now!)
Import paths you don't recognize (e.g., "should be v1 not v2" - v2 might be correct!)
Package versions (e.g., "mongo-driver/v2" - newer versions exist!)
Language features you don't recognize (they might be new)
API methods you don't know (they might have been added)

ONLY flag version/import issues if:
There's an obvious typo (e.g., "monggo" instead of "mongo")
The code itself shows an error (e.g., import fails in the diff)
There's a clear pattern mismatch (e.g., mixing v1 and v2 imports inconsistently)

When in doubt about versions/imports: ASSUME THE DEVELOPER IS CORRECT and skip it.

SUGGESTIONS:
- When a fix is simple, provide a "suggestion" field.
- **GitLab Syntax Requirement**: You must format the suggestion using relative line offsets based on your `start_line` and `end_line`.
- **The Formula**:
1. The header MUST be: ```diff
- **Content**: The suggestion must include the full corrected code for every line from `start_line` to `end_line`.
- **Indentation**: You MUST preserve the exact leading whitespace of the original code.
- Format:
```diff
[CORRECTED CODE BLOCK]
```

Focus ONLY on:
1. **Critical bugs** - Code that will crash or produce incorrect results
2. **Security vulnerabilities** - Actual exploitable security issues (SQL injection, XSS, etc.)
3. **Logic errors** - Incorrect business logic or algorithm implementation
4. **Performance problems** - Clear performance bottlenecks (O(n²) where O(n) is possible, memory leaks, etc.)
5. **Breaking changes** - Code that breaks existing functionality or APIs

DO NOT report:
- Stylistic preferences (variable naming, code organization) unless they severely impact readability
- Missing comments or documentation
- Minor code smells that don't impact functionality
- Hypothetical edge cases without evidence they're relevant
- Refactoring suggestions unless current code is broken
- Version numbers, import paths, or package versions you're unfamiliar with
- Missing imports

Be specific and reference exact line numbers from the diff."""
        ),
        HumanMessage(
            content=f"""Review the merge request diff for the file: {file_path}

INSTRUCTIONS:
1. Use the get_diff("{file_path}") tool ONCE to retrieve the diff
2. Review the diff content directly - read other files if absolutely necessary for more context
3. Return your findings as a list of issues

Analyze ONLY this file's diff. If you find legitimate issues, return them.
If there are no real issues, return an empty list.
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
                "You are an issue checker. Validate each issue strictly against the diff.\n"
                "Keep an issue ONLY if the diff provides direct evidence that the issue is real.\n"
                "Do NOT create new issues and do NOT modify fields. For any removed issue, provide\n"
                "a short reason grounded in the diff. Do not use tools."
            )
        ),
        HumanMessage(
            content=f"""File: {file_path}

Diff:
```diff
{diff_content}
```

Issues to validate:
{json.dumps(issues_payload, indent=2)}

Validate each issue and return a ValidationResult with:
- valid_issues: subset of input issues that are confirmed valid
- removed: list of entries with 'issue' (the removed issue object) and 'reason' (why it was removed)"""
        ),
    ]

    validation_settings = ToolCallerSettings(max_tool_calls=0)

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
