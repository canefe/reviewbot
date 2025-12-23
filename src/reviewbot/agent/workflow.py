from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
from reviewbot.core.issues import Issue
from reviewbot.core.issues.issue_model import IssueModel
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import fetch_mr_diffs, get_mr_branch
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

    diffs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)

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
        issues: List[Issue] = agent_runner.invoke(  # type: ignore
            AgentRunnerInput(agent=agent, context=context, settings=settings)
        )

        console.print(f"[bold cyan]📊 Total issues found: {len(issues)}[/bold cyan]")

        # Group issues by file_path
        issues_by_file: Dict[str, List[Issue]] = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)

        console.print(
            f"[bold cyan]📁 Issues grouped into {len(issues_by_file)} files[/bold cyan]\n"
        )

        # Create one discussion per file with all issues as replies
        for file_path, file_issues in issues_by_file.items():
            handle_file_issues(file_path, file_issues, gitlab_config)

        console.print(
            "[bold green]🎉 All discussions created successfully![/bold green]"
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
    file_path: str, issues: List[Issue], gitlab_config: GitLabConfig
) -> None:
    """
    Create one discussion for a file and add each issue as a reply.

    Args:
        file_path: Path to the file being reviewed
        issues: List of issues found in this file
        gitlab_config: GitLab API configuration
    """
    if not issues:
        return

    console.print(
        f"[cyan]📝 Creating discussion for {file_path} with {len(issues)} issues[/cyan]"
    )

    # Create the main discussion summary
    discussion_title = f"Code Review: {file_path}"
    discussion_body = f"""## 📋 Review Summary for `{file_path}`

Found **{len(issues)}** issue(s) in this file.

### Issues Overview:
{chr(10).join(f"{idx}. {IssueModel.model_validate(issue).title}" for idx, issue in enumerate(issues, 1))}

---

*Each issue is detailed in the replies below.*
"""

    # Create discussion and get its ID
    try:
        discussion_id = create_discussion(
            title=discussion_title,
            body=discussion_body,
            gitlab_config=gitlab_config,
        )
        console.print(
            f"[green]✓ Created discussion: {discussion_title} (ID: {discussion_id})[/green]"
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to create discussion for {file_path}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return

    # Add each issue as a reply to the discussion
    for idx, issue in enumerate(issues, 1):
        try:
            issue_model = IssueModel.model_validate(issue)
            console.print(
                f"[dim]  Adding issue {idx}/{len(issues)}: {issue_model.title}[/dim]"
            )

            # Get specific lines for this issue
            try:
                issue_content = read_file.invoke(
                    input={
                        "path": issue_model.file_path,
                        "line_start": issue_model.line_number,
                        "line_end": issue_model.column_number
                        or issue_model.line_number,
                    }
                )
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not read issue lines: {e}[/yellow]")
                issue_content = (
                    f"Lines {issue_model.line_number}-{issue_model.column_number}"
                )

            # Determine severity emoji
            severity = getattr(issue_model, "severity", "medium")
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(str(severity).lower(), "⚪")

            # Format the reply with issue details
            line_info = issue_model.line_number
            if (
                issue_model.column_number
                and issue_model.column_number != issue_model.line_number
            ):
                line_info = f"{issue_model.line_number}-{issue_model.column_number}"

            reply_body = f"""### {severity_emoji} Issue #{idx}: {issue_model.title}

**Description:**
{issue_model.description}

**Location:** `{file_path}` - Line {line_info}

**Code Context:**
```
{issue_content}
```

**Severity:** {severity}
"""

            # Add reply to the discussion
            reply_to_discussion(
                discussion_id=discussion_id,
                body=reply_body,
                gitlab_config=gitlab_config,
            )
            console.print(f"[green]  ✓ Added issue: {issue_model.title}[/green]")

        except Exception as e:
            console.print(
                f"[red]  ✗ Failed to add issue reply for '{issue_model.title}': {e}[/red]"
            )
            import traceback

            traceback.print_exc()
            # Continue with other issues even if one fails

    console.print(f"[bold green]✅ Completed discussion for {file_path}[/bold green]\n")


def create_discussion(
    title: str,
    body: str,
    gitlab_config: GitLabConfig,
) -> str:
    """
    Create a discussion with title and body.

    Args:
        title: Discussion title
        body: Discussion body content
        gitlab_config: GitLab API configuration

    Returns:
        Discussion ID from GitLab
    """
    # GitLab discussions don't have separate titles in the API,
    # so we include the title in the body with markdown formatting
    full_body = f"## {title}\n\n{body}"

    discussion_id = post_discussion(
        api_v4=gitlab_config.api_v4,
        token=gitlab_config.token,
        project_id=gitlab_config.project_id,
        mr_iid=gitlab_config.mr_iid,
        body=full_body,
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
