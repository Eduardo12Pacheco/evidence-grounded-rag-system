from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from evidence_grounded_rag.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    Vector,
)
from evidence_grounded_rag.text.tokenizer import tokenize
from evidence_grounded_rag.versions import EMBEDDING_MODEL_ID

MODE = "offline"
DEFAULT_DIMENSION = 256


class DeterministicEmbedding(EmbeddingProvider):
    model_id: str = EMBEDDING_MODEL_ID
    mode: str = MODE
    semantic: bool = False

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if not isinstance(dimension, int) or dimension < 1:
            raise EmbeddingError(f"dimension must be a positive int, got {dimension!r}")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> Vector:
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if (digest[8] & 1) == 0 else -1.0
            vector[index] += sign * count
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "semantic": self.semantic,
            "dimension": self.dimension,
        }
