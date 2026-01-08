from pathlib import Path
from typing import Any

from idoagents.agents.tool_runner import ToolCallerSettings
from langchain.agents import create_agent  # type: ignore
from rich.console import Console  # type: ignore

from reviewbot.agent.base import (  # type: ignore
    AgentRunnerInput,
    agent_runner,  # type: ignore
)
from reviewbot.agent.workflow.config import GitLabConfig
from reviewbot.agent.workflow.discussions import handle_file_issues
from reviewbot.agent.workflow.gitlab_notes import (
    post_review_acknowledgment,
    update_review_summary,
)
from reviewbot.agent.workflow.ignore import filter_diffs, parse_reviewignore
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.agent import Agent
from reviewbot.core.config import Config
from reviewbot.core.issues import Issue
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import fetch_mr_diffs, get_mr_branch
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore
from reviewbot.models.gpt import get_gpt_model, get_gpt_model_low_effort
from reviewbot.tools import get_diff, read_file, think

console = Console()


def work_agent(config: Config, project_id: str, mr_iid: str) -> str:
    api_v4 = config.gitlab_api_v4 + "/api/v4"
    token = config.gitlab_token
    model = get_gpt_model(config.llm_model_name, config.llm_api_key, config.llm_base_url)

    clone_url = build_clone_url(api_v4, project_id, token)

    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)

    # Limit tool calls to prevent agent from wandering
    # For diff review: get_diff (1) + maybe read_file for context (1-2) = 3 max
    settings = ToolCallerSettings(max_tool_calls=40)

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
        model=low_effort_model,
        tools=[get_diff, think],
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

            handle_file_issues(file_path, issues, gitlab_config, filtered_diffs, diff_refs)

        # Pass the callback to the agent runner
        issues: list[Issue] = agent_runner(
            AgentRunnerInput(
                agent=agent,
                context=context,
                settings=settings,
                on_file_complete=on_file_review_complete,
                quick_scan_agent=low_effort_agent,
                model=model,
                tools=tools,
                quick_scan_model=low_effort_model,
                quick_scan_tools=[get_diff, think],
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
                model=low_effort_model,
                tools=[get_diff, think],
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
