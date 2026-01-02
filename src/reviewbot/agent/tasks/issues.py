import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.func import task
from rich.console import Console

from reviewbot.agent.tasks.core import ToolCallerSettings, tool_caller
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.agent import Agent
from reviewbot.core.issues import Issue, IssueModel

console = Console()


def with_retry(func: Callable, settings: ToolCallerSettings, *args, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute
        settings: Settings containing retry configuration
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    max_retries = settings.max_retries
    retry_delay = settings.retry_delay
    retry_max_delay = settings.retry_max_delay

    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # If this was the last attempt, raise the exception
            if attempt >= max_retries:
                console.print(f"[red]All {max_retries} retries failed. Last error: {e}[/red]")
                raise

            # Calculate delay with exponential backoff
            delay = min(retry_delay * (2**attempt), retry_max_delay)

            # Check if it's a retryable error
            error_msg = str(e).lower()
            is_retryable = any(
                keyword in error_msg
                for keyword in [
                    "rate limit",
                    "timeout",
                    "connection",
                    "network",
                    "502",
                    "503",
                    "504",
                    "429",
                ]
            )

            if not is_retryable:
                console.print(f"[yellow]Non-retryable error encountered: {e}[/yellow]")
                raise

            console.print(f"[yellow]Attempt {attempt + 1}/{max_retries + 1} failed: {e}[/yellow]")
            console.print(f"[yellow]Retrying in {delay:.1f} seconds...[/yellow]")
            time.sleep(delay)

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


@dataclass
class IssuesInput:
    agent: Agent
    context: Context
    settings: ToolCallerSettings
    on_file_complete: Optional[Callable[[str, List[IssueModel]], None]] = None
    quick_scan_agent: Optional[Agent] = None


@task
def identify_issues(ctx: IssuesInput) -> List[Issue]:
    """
    Identify the issues in the codebase using concurrent agents per file.
    """
    agent = ctx.agent
    context = ctx.context
    settings = ctx.settings
    on_file_complete = ctx.on_file_complete
    quick_scan_agent = ctx.quick_scan_agent

    issue_store = context.get("issue_store")
    if not issue_store:
        raise ValueError("Issue store not found")

    manager = context.get("store_manager")
    if not manager:
        raise ValueError("Store manager not found")

    store = manager.get_store()
    if not store:
        raise ValueError("Store not found")

    tree = manager.get_tree()
    diffs = manager.get_diffs()

    if not tree or not diffs:
        raise ValueError("Tree or diffs not found")

    # Run concurrent reviews - pass the context values and callback
    all_issues = run_concurrent_reviews(
        agent,
        diffs,
        settings,
        context,
        on_file_complete=on_file_complete,
        quick_scan_agent=quick_scan_agent,
    )

    # Convert to domain objects
    return [issue.to_domain() for issue in all_issues]


def monitor_progress(
    future_to_file: dict,
    stop_event: threading.Event,
    task_timeout: int = 300,  # 5 minutes per task
):
    """
    Monitor thread that logs the status of ongoing tasks.
    """
    start_times = {future: time.time() for future in future_to_file.keys()}

    while not stop_event.is_set():
        time.sleep(10)  # Check every 10 seconds

        current_time = time.time()
        for future, file_path in future_to_file.items():
            if not future.done():
                elapsed = current_time - start_times[future]
                if elapsed > task_timeout:
                    console.print(
                        f"[red]TIMEOUT WARNING: {file_path} has been running for {elapsed:.0f}s[/red]"
                    )
                else:
                    console.print(
                        f"[yellow]Still processing: {file_path} ({elapsed:.0f}s elapsed)[/yellow]"
                    )


def run_concurrent_reviews(
    agent: Any,
    diffs: List[Any],
    settings: ToolCallerSettings,
    context: Context,
    max_workers: int = 3,  # Serial processing to avoid thread safety and rate limit issues
    task_timeout: int = 160,  # 5 minutes timeout per file
    on_file_complete: Optional[Callable[[str, List[IssueModel]], None]] = None,
    quick_scan_agent: Any | None = None,
) -> List[IssueModel]:
    """
    Run concurrent reviews of all diff files with context propagation and monitoring.

    Args:
        agent: The agent to use for reviews
        diffs: List of diff objects
        settings: Tool caller settings
        context: Context object
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

    all_issues: List[IssueModel] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with context baked in
        review_with_context = partial(
            review_single_file_with_context,
            agent=agent,
            settings=settings,
            context=context,
            quick_scan_agent=quick_scan_agent,
        )

        # Submit tasks
        future_to_file = {
            executor.submit(review_with_context, file_path): file_path
            for file_path in diff_file_paths
        }

        # Start monitoring thread
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(future_to_file, stop_monitor, task_timeout),
            daemon=True,
        )
        monitor_thread.start()

        # Process results with timeout
        for future in as_completed(future_to_file, timeout=task_timeout * len(diff_file_paths)):
            file_path = future_to_file[future]
            try:
                # Get result with per-task timeout
                issues = future.result(timeout=task_timeout)
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
                console.print(f"[red]✗[/red] TIMEOUT: {file_path} took longer than {task_timeout}s")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed {file_path}: {e}")
                import traceback

                traceback.print_exc()

        # Stop monitoring thread
        stop_monitor.set()
        monitor_thread.join(timeout=1)

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


def quick_scan_file(
    agent: Any,
    file_path: str,
    settings: ToolCallerSettings,
) -> bool:
    """
    Quick scan with low-effort agent to determine if file needs deep review.
    Returns True if file needs deep review, False otherwise.
    """
    messages: List[BaseMessage] = [
        SystemMessage(
            content="""You are a code review triage assistant. Your job is to quickly determine if a file change needs deep review.

Review the diff and decide if this file needs detailed analysis. Return TRUE if ANY of these apply:
- New code that implements business logic
- Changes to security-sensitive code (auth, permissions, data validation)
- Database queries or migrations
- API endpoint changes
- Complex algorithms or data structures
- Error handling changes
- Configuration changes that affect behavior

Return FALSE if:
- Only formatting/whitespace changes
- Simple refactoring (renaming variables/functions)
- Adding/updating comments or documentation only
- Import reordering
- Trivial changes (typo fixes in strings, adding logging)

Output ONLY "true" or "false" (lowercase, no quotes)."""
        ),
        HumanMessage(
            content=f"""Quickly scan this file and determine if it needs deep review: {file_path}

Use get_diff("{file_path}") to see the changes, then respond with ONLY "true" or "false"."""
        ),
    ]

    try:
        console.print(f"[dim]Quick scanning: {file_path}[/dim]")
        raw = with_retry(tool_caller, settings, agent, messages, settings)
        result = str(raw).strip().lower()

        needs_review = "true" in result
        if needs_review:
            console.print(f"[yellow]✓ Needs deep review: {file_path}[/yellow]")
        else:
            console.print(f"[dim]⊘ Skipping deep review: {file_path}[/dim]")

        return needs_review
    except Exception as e:
        console.print(
            f"[yellow]Quick scan failed for {file_path}, defaulting to deep review: {e}[/yellow]"
        )
        return True  # If scan fails, do deep review to be safe


def review_single_file_with_context(
    file_path: str,
    agent: Any,
    settings: ToolCallerSettings,
    context: Context,
    quick_scan_agent: Any | None = None,
) -> List[IssueModel]:
    """
    Wrapper that sets context before reviewing.
    This runs in each worker thread.
    """
    try:
        # Set the context var for this thread
        store_manager_ctx.set(context)

        console.print(f"[dim]Context set for thread processing: {file_path}[/dim]")

        # Quick scan first if agent provided
        if quick_scan_agent:
            needs_deep_review = quick_scan_file(quick_scan_agent, file_path, settings)
            if not needs_deep_review:
                console.print(f"[dim]Skipping deep review for: {file_path}[/dim]")
                return []

        # Now call the actual review function
        return review_single_file(agent, file_path, settings)
    except Exception as e:
        console.print(f"[red]Exception in thread for {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []


def review_single_file(
    agent: Any,
    file_path: str,
    settings: ToolCallerSettings,
) -> List[IssueModel]:
    """
    Review a single diff file and return issues found.
    """
    messages: List[BaseMessage] = [
        SystemMessage(
            content=f"""You are a senior code reviewer analyzing a specific file change.

Your task: Review ONLY the file '{file_path}' from the merge request diff.

IMPORTANT GUIDELINES:
- Be CONSERVATIVE: Only report real, actionable issues - not stylistic preferences or nitpicks
- If there are NO legitimate issues, return an empty array: []
- Do NOT invent issues to justify the review
- Only report issues with clear negative impact (bugs, security risks, performance problems, logic errors)
- Avoid reporting issues about code style, formatting, or personal preferences unless they violate critical standards
- Medium/High severity issues should be reserved for actual bugs, security vulnerabilities, or broken functionality

CRITICAL - KNOWLEDGE CUTOFF AWARENESS:
⚠️ Your training data has a cutoff date. The code you're reviewing may use:
- Package versions released AFTER your training (e.g., v2, v3 of libraries)
- Language versions you don't know about (e.g., Go 1.23+, Python 3.13+)
- Import paths that have changed since your training
- APIs that have been updated

DO NOT FLAG as issues:
❌ Version numbers (e.g., "Go 1.25 doesn't exist" - it might now!)
❌ Import paths you don't recognize (e.g., "should be v1 not v2" - v2 might be correct!)
❌ Package versions (e.g., "mongo-driver/v2" - newer versions exist!)
❌ Language features you don't recognize (they might be new)
❌ API methods you don't know (they might have been added)

ONLY flag version/import issues if:
✅ There's an obvious typo (e.g., "monggo" instead of "mongo")
✅ The code itself shows an error (e.g., import fails in the diff)
✅ There's a clear pattern mismatch (e.g., mixing v1 and v2 imports inconsistently)

When in doubt about versions/imports: ASSUME THE DEVELOPER IS CORRECT and skip it.

SUGGESTIONS:
- When the fix is OBVIOUS and simple, include a "suggestion" field with the corrected code
- The suggestion should contain ONLY the fixed code (not diff markers like +/-)
- Only include suggestions for simple fixes (typos, obvious bugs, missing fields, etc.)
- Do NOT include suggestions for complex refactorings or architectural changes
- DO NOT suggest version/import changes unless there's an obvious typo
- Format: just the corrected code, no explanations

Output format: JSON array of issue objects following this schema:
{IssueModel.model_json_schema()}

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

CONTEXT AWARENESS:
- If you need to verify package versions, you can use read_file to check:
  - Go: go.mod, go.sum
  - Python: requirements.txt, pyproject.toml, Pipfile
  - Node: package.json, package-lock.json
  - Rust: Cargo.toml, Cargo.lock
- Use this to understand what versions are ACTUALLY being used in the project
- Trust the dependency files over your training data

Be specific and reference exact line numbers from the diff."""
        ),
        HumanMessage(
            content=f"""Review the merge request diff for the file: {file_path}

INSTRUCTIONS:
1. Use the get_diff("{file_path}") tool ONCE to retrieve the diff
2. Review the diff content directly - DO NOT search for other files or read other files unless absolutely necessary
3. Output your findings immediately in JSON format

Analyze ONLY this file's diff. If you find legitimate issues, output them in JSON format.
If there are no real issues, output an empty array: []

Be efficient with your tool calls, they are limited, so use them wisely."""
        ),
    ]

    try:
        console.print(f"[cyan]Starting review of: {file_path}[/cyan]")

        # Use retry logic for the LLM call
        raw = with_retry(tool_caller, settings, agent, messages, settings)

        console.print(f"[green]Completed review of: {file_path}[/green]")
        console.print(
            f"Raw response: {raw[:200]}..." if len(str(raw)) > 200 else f"Raw response: {raw}"
        )

        # Parse issues from response
        issues: List[IssueModel] = []
        if isinstance(raw, str):
            try:
                import json

                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for issue_data in parsed:
                        try:
                            issues.append(IssueModel.model_validate(issue_data))
                        except Exception as e:
                            console.print(f"[yellow]Failed to validate issue: {e}[/yellow]")
                elif isinstance(parsed, dict):
                    try:
                        issues.append(IssueModel.model_validate(parsed))
                    except Exception as e:
                        console.print(f"[yellow]Failed to validate issue: {e}[/yellow]")
            except json.JSONDecodeError as e:
                console.print(f"[red]Failed to parse JSON for {file_path}: {e}[/red]")

        console.print(f"[blue]Found {len(issues)} issues in {file_path}[/blue]")
        return issues

    except Exception as e:
        console.print(f"[red]Error reviewing {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []
