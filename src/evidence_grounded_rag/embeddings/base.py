from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise EmbeddingDimensionError(
            f"cosine similarity dimension mismatch: {len(left)} != {len(right)}"
        )
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class EmbeddingError(Exception):
    message: str


@dataclass(frozen=True)
class EmbeddingDimensionError(EmbeddingError):
    message: str = "embedding dimension mismatch"

    def __str__(self) -> str:
        return self.message


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_id: str
    mode: str
    semantic: bool
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_one(self, text: str) -> Vector: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    document_id: str
    rank: int
    score: float
