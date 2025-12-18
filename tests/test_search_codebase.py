# tests/test_search_codebase.py

import subprocess
import tempfile
from pathlib import Path

import dotenv

from reviewbot.tools.search_codebase import search_codebase, set_repo_root


def main() -> None:
    dotenv.load_dotenv()
    with tempfile.TemporaryDirectory(prefix="reviewbot-test-") as tmp:
        repo_dir = Path(tmp) / "repo"

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/canefe/npcdrops",
                str(repo_dir),
            ],
            check=True,
        )

        set_repo_root(repo_dir)

        print("=== SEARCH: npcDrops ===")
        res = search_codebase.invoke("how does npcdrops trigger the drop chance?")
        print(res)

        print("\n=== SEARCH: drop chance ===")
        res = search_codebase.invoke("where is the hook that triggers npc death")
        print(res)


if __name__ == "__main__":
    main()
