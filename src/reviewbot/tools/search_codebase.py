from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool  # type: ignore
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState
from reviewbot.infra.embeddings.openai import CodebaseVectorStore

console = Console()

# Cache for the vector store to avoid rebuilding
_vector_store_cache: CodebaseVectorStore | None = None


def _get_codebase_state(store: Any) -> CodebaseState:
    """Helper to get CodebaseState from store."""
    NS = ("codebase",)
    raw = store.get(NS, "state")
    if not raw:
        raise ValueError("Codebase state not found in store")
    return CodebaseState.model_validate(raw.value)


def _get_vector_store(store: Any) -> CodebaseVectorStore:
    """Get or create vector store for semantic search."""
    global _vector_store_cache

    codebase = _get_codebase_state(store)
    repo_root = Path(codebase.repo_root)
    repo_name = codebase.repo_name

    # Create new instance if not cached or repo changed
    if _vector_store_cache is None or _vector_store_cache.repo_root != repo_root:
        _vector_store_cache = CodebaseVectorStore(repo_root, repo_name)
        if not _vector_store_cache.load():
            _vector_store_cache.build()

    return _vector_store_cache


@tool
def search_codebase(query: str, runtime: ToolRuntime) -> str:
    """
    Search the codebase using Unix find + grep.

    Args:
        query: string or regex to search for
    Returns:
        grep-style matches: file:line:content
    """
    codebase = _get_codebase_state(runtime.store)
    repo_root = Path(codebase.repo_root).resolve()

    max_lines = 200
    cmd = [
        "bash",
        "-lc",
        (
            f"find {shlex.quote(repo_root.as_posix())} -type f "
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
def search_codebase_semantic_search(
    query: str, runtime: ToolRuntime, path: str | None = None
) -> str:
    """Search the codebase for the given query. If a path is provided, search the codebase for the given query in the given path.

    Args:
        query: string to search the codebase for
        path: path to the file to search the codebase for (optional)
    Returns:
        string with the results of the search
    """
    vector_store = _get_vector_store(runtime.store)
    return vector_store.search(query, top_k=5, path=path)  # type: ignore


@tool
def read_file(
    path: str, runtime: ToolRuntime, line_start: int | None = None, line_end: int | None = None
) -> str:
    """Read the file at the given path.

    Args:
        path: path to the file to read
        line_start: line number to start reading from (optional)
        line_end: line number to stop reading at (optional)
    Returns:
        string with the contents of the file
    """
    vector_store = _get_vector_store(runtime.store)
    return vector_store.read_file(path, line_start=line_start, line_end=line_end)
