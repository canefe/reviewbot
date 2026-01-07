from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()
CODE_EXTENSIONS = {
    ".py",
    ".go",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".kt",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".lua",
}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "logs",
    "cache",
    "temp",
    "tmp",
    "tempdata",
    "tempfiles",
}


class CodebaseVectorStore:
    def __init__(
        self,
        repo_root: str | Path,
        repo_name: str,
        *,
        vectors_dir: str | Path = "vectors/codebase",
        embeddings_model: str | None = None,
        embeddings_base_url: str | None = None,
        embeddings_api_key: str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        if embeddings_model is None:
            embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
        if embeddings_base_url is None:
            embeddings_base_url = os.getenv("EMBEDDINGS_BASE_URL")
        if embeddings_api_key is None:
            embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY", "dummy")

        self.embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            base_url=embeddings_base_url,
            api_key=embeddings_api_key,
            tiktoken_enabled=False,
            chunk_size=64,
            max_retries=3,
            request_timeout=120,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
        )

        self.faiss_path = self.vectors_dir / "faiss" / repo_name
        self.metadata_path = self.vectors_dir / "metadata.json"

        self.vector_store: FAISS | None = None
        self.metadata_index: dict[str, list[str]] = {}

    def _iter_source_files(self) -> Iterable[Path]:
        print(f"Iterating source files in {self.repo_root}")
        for path in self.repo_root.rglob("*"):
            if path.is_dir():
                if path.name in EXCLUDE_DIRS:
                    continue
            if path.suffix in CODE_EXTENSIONS:
                if any(part in EXCLUDE_DIRS for part in path.parts):
                    continue
                yield path

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []

        for file in self._iter_source_files():
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not text.strip():
                continue

            print(f"Embedding file: {file}")
            print(f"Text: {text}")
            print(f"Text length: {len(text)}")
            print(f"Text chunks: {len(self.splitter.split_text(text))}")
            rel = file.relative_to(self.repo_root)

            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "type": "code",
                            "path": str(rel),
                            "filename": file.name,
                            "extension": file.suffix,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    )
                )

        return docs

    def build(self) -> None:
        docs = self._load_documents()
        if not docs:
            raise RuntimeError("No source files found to embed")

        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self._build_metadata_index()
        self.save()

    def compile(self, command: str) -> str:
        return subprocess.run(command, shell=True, capture_output=True, text=True).stdout

    def _build_metadata_index(self) -> None:
        self.metadata_index = {}
        if not self.vector_store:
            return

        for doc_id, doc in self.vector_store.docstore._dict.items():
            path = doc.metadata.get("path")
            if path:
                self.metadata_index.setdefault(path, []).append(doc_id)

    def save(self) -> None:
        if not self.vector_store:
            return

        self.vector_store.save_local(str(self.faiss_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_index, f, indent=2)

    def load(self) -> bool:
        if not self.faiss_path.exists():
            return False

        self.vector_store = FAISS.load_local(
            str(self.faiss_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        if self.metadata_path.exists():
            with open(self.metadata_path, encoding="utf-8") as f:
                self.metadata_index = json.load(f)

        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        path: str | None = None,
    ) -> list[dict]:
        if not self.vector_store:
            raise RuntimeError("Vector store not loaded")

        filter = {}
        if path:
            filter["path"] = path
        results = self.vector_store.similarity_search_with_score(query, k=top_k, filter=filter)
        out = []
        for doc, score in results:
            out.append(
                {
                    "similarity": float(1 - score),
                    "path": doc.metadata.get("path"),
                    "filename": doc.metadata.get("filename"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "text": doc.page_content,
                }
            )
        console.print(out)
        return out

    def read_file(
        self,
        path: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        file_path = Path(path)
        # the path is relative to the repo root so add the repo root to the path
        print(f"Reading file: {file_path}")
        file_path = self.repo_root / file_path
        if not file_path.exists():
            raise FileNotFoundError(path)

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

        # Git / editors are 1-based; Python is 0-based
        if line_start is not None:
            line_start = max(line_start - 1, 0)
        else:
            line_start = 0

        if line_end is not None:
            line_end = min(line_end, len(lines))
        else:
            line_end = len(lines)

        return "\n".join(lines[line_start:line_end])
