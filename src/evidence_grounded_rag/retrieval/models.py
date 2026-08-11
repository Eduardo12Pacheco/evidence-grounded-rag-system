"""Typed models shared by the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the rank and raw score produced by one retriever."""

    chunk_id: str
    document_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked chunks plus committed configuration metadata.

    ``config`` and ``versions`` are plain mappings so every retriever can
    report exactly what it did without a shared closed schema.
    """

    chunks: tuple[ScoredChunk, ...]
    config: dict[str, object]
    versions: dict[str, str]
    empty: bool = False
