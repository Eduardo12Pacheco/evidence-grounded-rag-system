"""Interfaces for the dense embedding layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity between two equal-length vectors."""
    if len(left) != len(right):
        raise EmbeddingDimensionError(
            f"cosine similarity dimension mismatch: {len(left)} != {len(right)}"
        )
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class EmbeddingError(Exception):
    """Base class for embedding-layer failures."""

    message: str


@dataclass(frozen=True)
class EmbeddingDimensionError(EmbeddingError):
    message: str = "embedding dimension mismatch"

    def __str__(self) -> str:
        return self.message


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract every embedding provider must satisfy.

    ``semantic`` MUST be False for any embedding that does not encode learned
    meaning. The deterministic test embedding is NOT semantic and is never
    labeled as such.
    """

    model_id: str
    mode: str
    semantic: bool
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_one(self, text: str) -> Vector: ...


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    document_id: str
    rank: int
    score: float
