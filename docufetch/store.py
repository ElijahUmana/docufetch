"""DocuFetch store - vector storage and retrieval using ChromaDB."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from .chunker import Chunk

DEFAULT_DB_PATH = Path.home() / ".docufetch" / "chromadb"


class VectorStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_source(source: str) -> str:
        """If source looks like a URL, extract the domain."""
        if source.startswith(("http://", "https://")):
            from urllib.parse import urlparse
            return urlparse(source).netloc
        return source

    @staticmethod
    def _collection_name(source: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", source).strip("_-")[:63]
        if len(name) < 3:
            name += "_col"
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, source_name: str, chunks: list[Chunk]) -> int:
        source_name = self._normalize_source(source_name)
        col = self._collection_name(source_name)

        # Remove existing collection for this source
        try:
            self.client.delete_collection(col)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=col,
            metadata={"source": source_name},
        )

        batch = 100
        for i in range(0, len(chunks), batch):
            b = chunks[i : i + batch]
            collection.add(
                documents=[c.content for c in b],
                metadatas=[c.metadata for c in b],
                ids=[f"{col}_{i + j}" for j, _ in enumerate(b)],
            )

        return len(chunks)

    def query(self, source_name: str, question: str, n_results: int = 5) -> dict:
        source_name = self._normalize_source(source_name)
        col = self._collection_name(source_name)
        collection = self.client.get_collection(col)
        return collection.query(query_texts=[question], n_results=n_results)

    def list_sources(self) -> list[tuple[str, str]]:
        return [
            (c.name, c.metadata.get("source", c.name))
            for c in self.client.list_collections()
        ]

    def delete(self, source_name: str) -> None:
        source_name = self._normalize_source(source_name)
        self.client.delete_collection(self._collection_name(source_name))

    def clear(self) -> None:
        for c in self.client.list_collections():
            self.client.delete_collection(c.name)

    def get_collection(self, source_name: str):
        source_name = self._normalize_source(source_name)
        return self.client.get_collection(self._collection_name(source_name))
