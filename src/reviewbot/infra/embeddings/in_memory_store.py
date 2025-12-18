from pathlib import Path
from typing import Optional

from reviewbot.infra.embeddings.openai import CodebaseVectorStore

# single lazy-loaded store (process lifetime)
_store: Optional[CodebaseVectorStore] = None
_repo_root: Optional[Path] = None
_repo_name: Optional[str] = None


def set_repo_root(path: str | Path) -> None:
    global _repo_root
    _repo_root = Path(path).resolve()


def set_repo_name(name: str) -> None:
    global _repo_name
    _repo_name = name


def get_store() -> CodebaseVectorStore:
    global _store
    if _store is not None:
        return _store

    if _repo_root is None:
        raise ValueError("Repository root not set")

    if _repo_name is None:
        raise ValueError("Repository name not set")

    store = CodebaseVectorStore(_repo_root, _repo_name)

    if not store.load():
        store.build()

    _store = store
    return store
