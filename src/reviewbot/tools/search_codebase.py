from __future__ import annotations

import shlex
import subprocess

from langchain.tools import tool  # type: ignore
from rich.console import Console

from reviewbot.context import store_manager_ctx

console = Console()


@tool
def search_codebase(query: str) -> str:
    """
    Search the codebase using Unix find + grep.

    Args:
        query: string or regex to search for
    Returns:
        grep-style matches: file:line:content
    """
    # path is relative to the repo root
    context = store_manager_ctx.get()
    store = context.get("store_manager")
    if not store:
        raise ValueError("Store manager not found")

    repo_root = store.get_store().repo_root.resolve()

    base = repo_root  # or a validated subpath if you support `path`

    max_lines = 200
    cmd = [
        "bash",
        "-lc",
        (
            f"find {shlex.quote(base.as_posix())} -type f "
            f"! -path '*/.git/*' ! -path '*/node_modules/*' ! -path '*/.venv/*' "
            f"-print0 | "
            f"xargs -0 grep -nH --color=never -I {shlex.quote(query)} | "
            f"head -n {max_lines}"
        ),
    ]

    result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    )
    print(cmd)
    print("================================================")
    print(result.stdout.strip())
    print("================================================")

    if result.returncode == 1:
        return "No matches found."
    elif result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


@tool
def search_codebase_semantic_search(query: str, path: str | None = None) -> str:
    """Search the codebase for the given query. If a path is provided, search the codebase for the given query in the given path.

    Args:
        query: string to search the codebase for
        path: path to the file to search the codebase for (optional)
    Returns:
        string with the results of the search
    """
    context = store_manager_ctx.get()
    store = context.get("store_manager")
    if not store:
        raise ValueError("Store manager not found")

    store = store.get_store()
    if not store:
        raise ValueError("Store not found")

    return store.search(query, top_k=5, path=path)  # type: ignore


@tool
def read_file(path: str, line_start: int | None = None, line_end: int | None = None) -> str:
    """Read the file at the given path.

    Args:
        path: path to the file to read
        line_start: line number to start reading from (optional)
        line_end: line number to stop reading at (optional)
    Returns:
        string with the contents of the file
    """
    context = store_manager_ctx.get()
    store = context.get("store_manager")
    if not store:
        raise ValueError("Store manager not found")

    store = store.get_store()
    if not store:
        raise ValueError("Store not found")

    result = store.read_file(path, line_start=line_start, line_end=line_end)
    return result
