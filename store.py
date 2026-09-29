"""Lightweight local vector store for RAG.

Embeddings are computed locally with sentence-transformers -- no API key,
no per-call cost, works offline once the model is cached. BAAI/bge-m3 is
the default: strong multilingual retrieval quality, including Arabic,
and it's the same PyTorch stack already pulled in by silero VAD, so
there's no extra heavy dependency beyond the sentence-transformers
package itself.

Embeddings are stored as a single float32 numpy array on disk, alongside
a JSONL sidecar with chunk text and metadata. Swap this out for
pgvector/Qdrant/Milvus later if you outgrow it -- the interface
(add / save / load / search) stays the same, so agent.py doesn't need
to change.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


@lru_cache(maxsize=1)
def _get_model():
    # Loaded once per process and cached -- the first call downloads and
    # caches the model under ~/.cache/huggingface, subsequent runs are
    # fully offline.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@dataclass
class Chunk:
    id: str
    text: str
    source: str


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    async def _embed(self, texts: list[str]) -> np.ndarray:
        # model.encode() is CPU/GPU-bound and synchronous -- run it off
        # the event loop so it doesn't block the voice pipeline.
        def _run() -> np.ndarray:
            model = _get_model()
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vecs.astype(np.float32)

        return await asyncio.to_thread(_run)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = await self._embed([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])

    async def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q = (await self._embed([query]))[0]
        scores = self.embeddings @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arr = self.embeddings if self.embeddings is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(path / "embeddings.npy", arr)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"id": c.id, "text": c.text, "source": c.source}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            arr = np.load(emb_path)
            store.embeddings = arr if arr.size else None
        chunks_path = path / "chunks.jsonl"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        store.chunks.append(Chunk(**d))
        return store