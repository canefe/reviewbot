from pathlib import Path
from typing import List, Optional

from reviewbot.infra.embeddings.openai import CodebaseVectorStore
from reviewbot.infra.gitlab.diff import FileDiff


class CodebaseStoreManager:
    def __init__(self) -> None:
        self._store: Optional[CodebaseVectorStore] = None
        self._repo_root: Optional[Path] = None
        self._repo_name: Optional[str] = None
        self._tree: Optional[str] = None
        self._diffs: Optional[List[FileDiff]] = None

    def set_repo_root(self, path: str | Path) -> None:
        self._repo_root = Path(path).resolve()
        self._store = None  # invalidate cache
        self._tree = None  # invalidate cache
        self._diffs = None  # invalidate cache

    def set_repo_name(self, name: str) -> None:
        self._repo_name = name
        self._store = None  # invalidate cache
        self._tree = None  # invalidate cache
        self._diffs = None  # invalidate cache

    def set_tree(self, tree: str) -> None:
        self._tree = tree

    def set_diffs(self, diffs: List[FileDiff]) -> None:
        self._diffs = diffs

    def get_tree(self) -> str:
        if self._tree is None:
            raise ValueError("Tree not set")
        return self._tree

    def get_diffs(self) -> List[FileDiff]:
        if self._diffs is None:
            raise ValueError("Diff not set")
        return self._diffs

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
