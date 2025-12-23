import json

from langchain.tools import tool  # type: ignore
from rich.console import Console

from reviewbot.context import store_manager_ctx

console = Console()


@tool
def get_diff(file_path: str) -> str:
    """
    Get the diff of the file.

    Args:
        file_path: path to the file to get the diff for
    Returns:
        string with the diff of the file
    """
    context = store_manager_ctx.get()
    store = context.get("store_manager")
    if not store:
        raise ValueError("Store manager not found")

    diffs = store.get_diffs()
    if not diffs:
        raise ValueError("Diff not found")

    diff = next((diff for diff in diffs if diff.new_path == file_path), None)
    if not diff:
        raise ValueError(f"Diff not found for file: {file_path}")

    return json.dumps(diff.patch)[1:-1]


@tool
def get_tree() -> str:
    """
    Get the tree of the codebase.
    """
    context = store_manager_ctx.get()
    store = context.get("store_manager")
    if not store:
        raise ValueError("Store manager not found")
    return store.get_tree()
