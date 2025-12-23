import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.func import task
from rich.console import Console

from reviewbot.agent.tasks.core import ToolCallerSettings, tool_caller
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.agent import Agent
from reviewbot.core.issues import Issue, IssueModel

console = Console()


@dataclass
class IssuesInput:
    agent: Agent
    context: Context
    settings: ToolCallerSettings


@task
def identify_issues(ctx: IssuesInput) -> List[Issue]:
    """
    Identify the issues in the codebase using concurrent agents per file.
    """
    agent = ctx.agent
    context = ctx.context
    settings = ctx.settings

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

    # Run concurrent reviews - pass the context values
    all_issues = run_concurrent_reviews(agent, diffs, settings, context)

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
        time.sleep(3)  # Check every 10 seconds

        current_time = time.time()
        for future, file_path in future_to_file.items():
            if not future.done():
                elapsed = current_time - start_times[future]
                if elapsed > task_timeout:
                    console.print(
                        f"[red]⚠️  TIMEOUT WARNING: {file_path} has been running for {elapsed:.0f}s[/red]"
                    )
                else:
                    console.print(
                        f"[yellow]⏳ Still processing: {file_path} ({elapsed:.0f}s elapsed)[/yellow]"
                    )


def run_concurrent_reviews(
    agent: Any,
    diffs: List[Any],
    settings: ToolCallerSettings,
    context: Context,
    max_workers: int = 3,
    task_timeout: int = 300,  # 5 minutes timeout per file
) -> List[IssueModel]:
    """
    Run concurrent reviews of all diff files with context propagation and monitoring.
    """
    diff_file_paths = [diff.new_path for diff in diffs]

    console.print(
        f"[bold]🚀 Starting concurrent review of {len(diff_file_paths)} files[/bold]"
    )
    console.print(f"[dim]Files: {', '.join(diff_file_paths)}[/dim]\n")

    all_issues: List[IssueModel] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with context baked in
        review_with_context = partial(
            review_single_file_with_context,
            agent=agent,
            settings=settings,
            context=context,
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
        for future in as_completed(
            future_to_file, timeout=task_timeout * len(diff_file_paths)
        ):
            file_path = future_to_file[future]
            try:
                # Get result with per-task timeout
                issues = future.result(timeout=task_timeout)
                all_issues.extend(issues)
                console.print(
                    f"[green]✓[/green] Processed {file_path}: {len(issues)} issues"
                )
            except TimeoutError:
                console.print(
                    f"[red]✗[/red] TIMEOUT: {file_path} took longer than {task_timeout}s"
                )
            except Exception as e:
                console.print(f"[red]✗[/red] Failed {file_path}: {e}")
                import traceback

                traceback.print_exc()

        # Stop monitoring thread
        stop_monitor.set()
        monitor_thread.join(timeout=1)

    console.print(
        f"\n[bold green]🎉 Review complete! Total issues found: {len(all_issues)}[/bold green]"
    )

    if all_issues:
        console.print("\n[bold]📋 Issues by file:[/bold]")
        file_issue_count: dict[str, int] = {}
        for issue in all_issues:
            file_path = getattr(issue, "file_path", "unknown")
            file_issue_count[file_path] = file_issue_count.get(file_path, 0) + 1

        for file_path, count in sorted(file_issue_count.items()):
            console.print(f"  • {file_path}: {count} issues")

    return all_issues


def review_single_file_with_context(
    file_path: str,
    agent: Any,
    settings: ToolCallerSettings,
    context: Context,
) -> List[IssueModel]:
    """
    Wrapper that sets context before reviewing.
    This runs in each worker thread.
    """
    try:
        # Set the context var for this thread
        store_manager_ctx.set(context)

        console.print(f"[dim]🔧 Context set for thread processing: {file_path}[/dim]")

        # Now call the actual review function
        return review_single_file(agent, file_path, settings)
    except Exception as e:
        console.print(f"[red]❌ Exception in thread for {file_path}: {e}[/red]")
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
            content=f"""You are a code reviewer analyzing a specific file change.
            
Your task: Review ONLY the file '{file_path}' from the merge request diff.

Output format: JSON array of issue objects following this schema:
{IssueModel.model_json_schema()}

Focus on:
- Code quality issues
- Potential bugs
- Security vulnerabilities
- Performance problems
- Best practice violations
- Logic errors

Be specific and reference line numbers from the diff."""
        ),
        HumanMessage(
            content=f"""Review the merge request diff for the file: {file_path}

Use the get_diff("{file_path}") tool to retrieve the diff content for this specific file.

Analyze ONLY this file and output issues in JSON format."""
        ),
    ]

    try:
        console.print(f"[cyan]🔍 Starting review of: {file_path}[/cyan]")

        raw = tool_caller(agent, messages, settings)

        console.print(f"[green]✅ Completed review of: {file_path}[/green]")
        console.print(
            f"Raw response: {raw[:200]}..."
            if len(str(raw)) > 200
            else f"Raw response: {raw}"
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
                            console.print(
                                f"[yellow]⚠️  Failed to validate issue: {e}[/yellow]"
                            )
                elif isinstance(parsed, dict):
                    try:
                        issues.append(IssueModel.model_validate(parsed))
                    except Exception as e:
                        console.print(
                            f"[yellow]⚠️  Failed to validate issue: {e}[/yellow]"
                        )
            except json.JSONDecodeError as e:
                console.print(
                    f"[red]❌ Failed to parse JSON for {file_path}: {e}[/red]"
                )

        console.print(f"[blue]📊 Found {len(issues)} issues in {file_path}[/blue]")
        return issues

    except Exception as e:
        console.print(f"[red]❌ Error reviewing {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return []
