from pathlib import Path
from typing import Optional

from reviewbot.infra.embeddings.openai import CodebaseVectorStore


class CodebaseStoreManager:
    def __init__(self) -> None:
        self._store: Optional[CodebaseVectorStore] = None
        self._repo_root: Optional[Path] = None
        self._repo_name: Optional[str] = None

    def set_repo_root(self, path: str | Path) -> None:
        self._repo_root = Path(path).resolve()
        self._store = None  # invalidate cache

    def set_repo_name(self, name: str) -> None:
        self._repo_name = name
        self._store = None  # invalidate cache

    def get_store(self) -> CodebaseVectorStore:
        if self._store is not None:
            return self._store

        if self._repo_root is None:
            raise ValueError("Repository root not set")

        if self._repo_name is None:
            raise ValueError("Repository name not set")

        store = CodebaseVectorStore(self._repo_root, self._repo_name)

        if not store.load():
            store.build()

        self._store = store
        return store