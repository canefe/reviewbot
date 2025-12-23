import subprocess
from pathlib import Path
from typing import Any


def tree(repo_path: Path) -> str:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()

    root: dict[str, Any] = {}
    for file in out:
        parts = file.split("/")
        cur = root
        for p in parts:
            cur = cur.setdefault(p, {})

    def render(node: dict[str, Any], prefix: str = "") -> str:
        s = ""
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            s += prefix + ("└── " if last else "├── ") + name + "\n"
            if child:
                s += render(child, prefix + ("    " if last else "│   "))
        return s

    return render(root)
