import json
from typing import Any

from langchain.tools import ToolRuntime, tool  # type: ignore
from langgraph.func import BaseStore  # type: ignore
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState

console = Console()


def get_diff_from_file(
    store: BaseStore,
    file_path: str,
) -> str:
    NS = ("codebase",)
    raw = store.get(NS, "state")
    if not raw:
        raise ValueError("Codebase state not found in store")

    codebase = CodebaseState.model_validate(raw.value)
    diffs = codebase.diffs

    diff = next((diff for diff in diffs if diff.new_path == file_path), None)
    if not diff:
        raise ValueError(f"Diff not found for file: {file_path}")

    return json.dumps(diff.patch)[1:-1]


@tool
def get_diff(
    runtime: ToolRuntime[None, dict[str, Any]],
    file_path: str,
) -> str:
    """
    Get the diff of the file.

    Args:
        file_path: path to the file to get the diff for
    Returns:
        string with the diff of the file
    """
    if not runtime.store:
        raise ValueError("Store not found in runtime")

    return get_diff_from_file(
        runtime.store,
        file_path,
    )


@tool
def get_tree(runtime: ToolRuntime[None, dict[str, Any]]) -> str:
    """
    Get the tree of the codebase.
    """
    if not runtime.store:
        raise ValueError("Store not found in runtime")

    NS = ("codebase",)
    raw = runtime.store.get(NS, "state")
    if not raw:
        raise ValueError("Codebase state not found in store")

    codebase = CodebaseState.model_validate(raw.value)
    return codebase.repo_tree
