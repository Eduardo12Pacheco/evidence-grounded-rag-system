"""Dense embedding interface with a deterministic offline test embedding.

The bundled ``DeterministicEmbedding`` is a hashed bag-of-words embedding with
``semantic=False``: it is never described as semantic, downloads no weights,
and makes no network calls. A real embedding model can be plugged in behind the
same ``EmbeddingProvider`` protocol.
"""

from __future__ import annotations

from evidence_grounded_rag.embeddings.base import (
    DenseSearchResult,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProvider,
    Vector,
    cosine_similarity,
)
from evidence_grounded_rag.embeddings.deterministic import DeterministicEmbedding
from evidence_grounded_rag.embeddings.index import DenseIndex
from evidence_grounded_rag.versions import EMBEDDING_MODEL_ID

__all__ = [
    "EMBEDDING_MODEL_ID",
    "DenseIndex",
    "DenseSearchResult",
    "DeterministicEmbedding",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "Vector",
    "cosine_similarity",
]
