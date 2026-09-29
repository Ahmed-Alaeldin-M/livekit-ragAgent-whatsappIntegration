"""Build the vector index from files in --source.

Usage:
    python -m rag.ingest --source ./knowledge_base --out ./index

Reads .txt, .md, and .pdf files (recursively), splits them into
overlapping chunks, embeds each chunk, and writes the index to --out.
Re-running overwrites the index from scratch -- fine for a knowledge
base that's rebuilt on demand rather than updated incrementally.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from store import Chunk, VectorStore

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./knowledge_base", help="Directory of source docs")
    parser.add_argument("--out", default="./index", help="Directory to write the index to")
    args = parser.parse_args()

    source = Path(args.source)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        print(f"No {SUPPORTED_SUFFIXES} files found under {source}")
        return

    store = VectorStore()
    for path in files:
        text = read_text(path)
        pieces = chunk_text(text)
        chunks = [Chunk(id=str(uuid.uuid4()), text=piece, source=path.name) for piece in pieces]
        await store.add(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    store.save(args.out)
    print(f"Saved {len(store.chunks)} chunks from {len(files)} files to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
