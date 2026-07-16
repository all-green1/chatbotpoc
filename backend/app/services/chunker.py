from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

from chonkie import Pipeline


def chunk_text(
    *,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    article_id: str,
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Chunk a raw text string and attach minimal payload metadata needed downstream.
    """
    docs = (
        Pipeline()
        .fetch_from("text", texts=[text])
        .process_with("text")
        .chunk_with("recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        .run()
    )

    chunks: List[Dict[str, Any]] = []
    for d in docs:
        payload = getattr(d, "payload", {}) or {}
        content = payload.get("content") or getattr(d, "text", "") or str(d)

        chunks.append(
            {
                "content": content,
                "article_id": article_id,
                "source_type": source_type,
            }
        )

    return chunks


class ChunkerService:
    """
    Minimal chunker service for the POC.
    Reads chunk sizing from env to avoid coupling to the old project's Settings model.
    """

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

    async def chunk_article_text(
        self,
        *,
        article_id: str,
        source_type: str,
        text: str,
    ) -> List[Dict[str, Any]]:
        return chunk_text(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            article_id=article_id,
            source_type=source_type,
        )


@lru_cache(maxsize=1)
def get_chunker_service() -> ChunkerService:
    """
    Dependency provider.

    Env vars (POC-friendly defaults):
      - CHUNK_SIZE (default 800)
      - CHUNK_OVERLAP (default 120)
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    return ChunkerService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)