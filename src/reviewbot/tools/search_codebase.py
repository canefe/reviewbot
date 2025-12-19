from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from rich.console import Console

from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
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
    store = store_manager_ctx.get().get_store()
    repo_root = Path(store.repo_root).resolve()

    base = repo_root  # or a validated subpath if you support `path`

    cmd = [
        "bash",
        "-lc",
        (
            f"find {shlex.quote(base.as_posix())} -type f "
            f"! -path '*/.git/*' "
            f"! -path '*/node_modules/*' "
            f"! -path '*/.venv/*' "
            f"-print0 | "
            f"xargs -0 grep -nH --color=never -I {shlex.quote(query)}"
        ),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(result.stdout.strip())

    if result.returncode == 1:
        return "No matches found."
    elif result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


@tool
def search_codebase_semantic_search(query: str, path: Optional[str] = None) -> str:
    """Search the codebase for the given query. If a path is provided, search the codebase for the given query in the given path.

    Args:
        query: string to search the codebase for
        path: path to the file to search the codebase for (optional)
    Returns:
        string with the results of the search
    """
    store = store_manager_ctx.get().get_store()
    results = store.search(query, top_k=5, path=path)
    print("tool called")
    if not results:
        return "No matches found."

    lines = []
    for r in results:
        lines.append(f"{r['path']} (score={r['similarity']:.3f})\n{r['text']}")

    return "\n\n---\n\n".join(lines)


@tool
def read_file(
    path: str, line_start: Optional[int] = None, line_end: Optional[int] = None
) -> str:
    """Read the file at the given path.

    Args:
        path: path to the file to read
        line_start: line number to start reading from (optional)
        line_end: line number to stop reading at (optional)
    Returns:
        string with the contents of the file
    """
    store = store_manager_ctx.get().get_store()
    result = store.read_file(path, line_start=line_start, line_end=line_end)
    console.print(result)
    return result
