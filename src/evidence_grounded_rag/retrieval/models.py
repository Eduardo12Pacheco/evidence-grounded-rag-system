"""Retrieval result and scored-chunk models with rank and trace metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[ScoredChunk, ...]
    config: dict[str, object]
    versions: dict[str, str]
    empty: bool = False
