from pathlib import Path
from typing import Any

from ido_agents.agents.tool_runner import ToolCallerSettings
from ido_agents.models.openai import OpenAIModelConfig, build_chat_model
from langchain.agents import create_agent  # type: ignore
from langgraph.func import entrypoint  # type: ignore
from pydantic import BaseModel, SecretStr
from rich.console import Console  # type: ignore

from reviewbot.agent.tasks.data import clone_and_tree, fetch_gitlab_data
from reviewbot.agent.tasks.issues import identify_issues
from reviewbot.agent.workflow.config import GitLabConfig
from reviewbot.agent.workflow.discussions import handle_file_issues
from reviewbot.agent.workflow.gitlab_notes import (
    post_review_acknowledgment,
    update_review_summary,
)
from reviewbot.agent.workflow.ignore import filter_diffs, parse_reviewignore
from reviewbot.agent.workflow.state import CodebaseState, checkpointer, store
from reviewbot.core.agent import Agent
from reviewbot.core.config import Config
from reviewbot.infra.git.clone import get_repo_name
from reviewbot.models.gpt import get_gpt_model_low_effort
from reviewbot.tools import get_diff, think

console = Console()


class WorkAgentInput(BaseModel):
    config: Config
    project_id: str
    mr_iid: str

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }


@entrypoint(checkpointer=checkpointer, store=store)
async def work_agent(inputs: dict[Any, Any]) -> str:
    data = WorkAgentInput.model_validate(inputs)

    config = data.config
    project_id = data.project_id
    mr_iid = data.mr_iid

    api_v4 = config.gitlab_api_v4 + "/api/v4"
    token = config.gitlab_token

    modelCfg = OpenAIModelConfig(
        model=config.llm_model_name,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0.0,
        reasoning_effort="medium",
    )

    model = build_chat_model(modelCfg)

    data = await fetch_gitlab_data(api_v4, project_id, mr_iid, token)
    repo = await clone_and_tree(data.clone_url, data.branch)

    diffs = data.diffs
    diff_refs = data.diff_refs

    repo_path = Path(repo.repo_path).resolve()
    repo_tree = repo.repo_tree

    # Parse .reviewignore and filter diffs
    reviewignore_patterns = parse_reviewignore(repo_path)
    filtered_diffs = filter_diffs(diffs, reviewignore_patterns)
    console.print(f"[cyan]Reviewing {len(filtered_diffs)} out of {len(diffs)} changed files[/cyan]")

    NS = ("codebase",)
    state = CodebaseState(
        repo_root=str(repo_path),
        repo_name=get_repo_name(repo_path),
        repo_tree=repo_tree,
        diffs=filtered_diffs,
    )
    store.put(
        NS,
        "state",
        state.model_dump(),
    )

    # Create GitLab configuration
    gitlab_config = GitLabConfig(
        api_v4=api_v4,
        token=SecretStr(token),
        project_id=project_id,
        mr_iid=mr_iid,
    )

    # Create main agent for code review
    main_agent: Agent = create_agent(
        model=model,
        tools=[get_diff, think],
        store=store,
    )

    # Create a low-effort agent for simple tasks like acknowledgments and quick scans
    low_effort_model = get_gpt_model_low_effort(
        config.llm_model_name, config.llm_api_key, config.llm_base_url
    )
    low_effort_agent: Agent = create_agent(
        model=low_effort_model,
        tools=[get_diff, think],
        store=store,
    )

    # Create settings for tool calling
    settings = ToolCallerSettings(max_tool_calls=100)

    # Post acknowledgment that review is starting
    console.print("[dim]Posting review acknowledgment...[/dim]")
    ack = await post_review_acknowledgment(
        gitlab=gitlab_config,
        diffs=filtered_diffs,
        model=low_effort_model,
    )

    if ack is not None:
        console.print(
            f"[dim]Acknowledgment created: discussion={ack.discussion_id}, note={ack.note_id}[/dim]"
        )
    else:
        console.print(
            "[yellow]⚠ Failed to create acknowledgment (returned None), stopping... [/yellow]"
        )
        return "Review failed: acknowledgment creation returned None"

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

        # Call identify_issues task directly
        issue_models = await identify_issues(
            settings=settings,
            on_file_complete=on_file_review_complete,
            agent=main_agent,
            quick_scan_agent=low_effort_agent,
            model=model,
            tools=[get_diff, think],
            quick_scan_model=low_effort_model,
            quick_scan_tools=[get_diff, think],
        )

        # Convert IssueModel to domain Issue objects
        issues = [im.to_domain() for im in issue_models]

        console.print(f"[bold cyan]Total issues found: {len(issues)}[/bold cyan]")

        # Update the acknowledgment note with summary
        console.print(f"[dim]Checking acknowledgment_ids: {ack.discussion_id} {ack.note_id}[/dim]")
        if ack.discussion_id and ack.note_id:
            discussion_id, note_id = ack.discussion_id, ack.note_id
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

        console.print("[bold green]All reviews completed and discussions created![/bold green]")
        return "Review completed successfully"

    except Exception as e:
        console.print(f"[bold red]Error during review: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise
