from typing import Any

from langchain.agents import create_agent  # type: ignore
from langchain.agents.middleware import AgentState, after_model, before_model
from langchain_core.messages import HumanMessage
from langgraph.pregel.main import Runtime  # type: ignore
from rich.console import Console

from reviewbot.agent.tasks.core import ToolCallerSettings
from reviewbot.context import Context, store_manager_ctx
from reviewbot.core.agent import Agent
from reviewbot.infra.config.env import load_env
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import fetch_mr_diffs, get_mr_branch
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore
from reviewbot.models.gpt import get_gpt_model
from reviewbot.tools.diff import get_diff, get_tree
from reviewbot.tools.search_codebase import read_file, search_codebase_semantic_search

console = Console()

MESSAGE = []


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    global MESSAGE
    MESSAGE = state["messages"]
    return None


@after_model(can_jump_to=["end"])
def after_model_check(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    last_message = state["messages"][-1]
    if last_message.content is not None:
        if isinstance(last_message.content, list):
            if last_message.content[-1].get("type") == "reasoning":
                # print(f"Reasoning: {last_message.content[-1].get('content')}")
                messages = state["messages"]
                # delete the last message
                # messages.pop()
                # messages.append(
                #    HumanMessage(
                #         content="You attempted an invalid tool call. Please avoid this in future. Your faulty tool call was: "
                #        + str(last_message.content[-1].get("content", "Unknown"))
                #   )
                # )
                print("Faulty tool call!")
                MESSAGE = messages
                return {"messages": messages}
    return None


def test_agent():
    global MESSAGE
    config = load_env()
    api_v4 = config.gitlab_api_v4
    token = config.gitlab_token
    project_id = "29"
    mr_iid = "5"

    model = get_gpt_model(config.llm_model_name, config.llm_api_key, config.llm_base_url)

    clone_url = build_clone_url(api_v4, project_id, token)

    diffs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)

    settings = ToolCallerSettings(max_tool_calls=10, max_iterations=50)

    tools = [
        search_codebase_semantic_search,
        get_diff,
        read_file,
        get_tree,
    ]

    agent: Agent = create_agent(
        model=model,
        tools=tools,
        middleware=[check_message_limit, after_model_check],  # type: ignore
    )  # type: ignore
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    repo_path = clone_repo_persistent(clone_url, branch=branch)
    repo_tree = tree(repo_path)

    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    manager.set_tree(repo_tree)
    manager.set_diffs(diffs)
    manager.get_store()

    issue_store = InMemoryIssueStore()
    token = store_manager_ctx.set(Context(store_manager=manager, issue_store=issue_store))

    context = store_manager_ctx.get()

    diff_file_paths = " ".join([diff.new_path for diff in diffs if diff.new_path is not None])
    try:
        response = agent.invoke(  # type: ignore
            {
                "messages": [
                    HumanMessage(
                        content="Check my merge request diff and code review it. Use the tools provided, you got all you need you don't have to ask questions."
                        + "The codebase tree is: "
                        + repo_tree
                        + "The merge request diff file paths are: "
                        + diff_file_paths
                    )
                ]
            }
        )

        if isinstance(response, str):
            assert response is not None
        else:
            print(f"Response: {response['messages'][-1]}")

    except Exception as e:
        console.print(f"Error: {e}")
        open("errors.txt", "w").write(str(MESSAGE))
        raise e

    finally:
        store_manager_ctx.reset(token)
