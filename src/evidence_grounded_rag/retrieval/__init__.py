"""Independent sparse/hybrid retrieval layer.

Provides a deterministic, committed Okapi BM25 implementation and a hybrid
retriever that fuses sparse and dense rankings with reciprocal rank fusion
(``k = 60``).
"""

from __future__ import annotations

from evidence_grounded_rag.retrieval.bm25 import BM25, BM25Config
from evidence_grounded_rag.retrieval.fusion import (
    FUSION_NAME,
    FusionConfig,
    reciprocal_rank_fusion,
)
from evidence_grounded_rag.retrieval.hybrid import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    VALID_MODES,
    HybridResult,
    HybridRetriever,
    HybridTrace,
)
from evidence_grounded_rag.retrieval.models import RetrievalResult, ScoredChunk
from evidence_grounded_rag.text.tokenizer import (
    TOKENIZER_VERSION,
    key_tokens,
    stopwords,
    tokenize,
)

__all__ = [
    "BM25",
    "FUSION_NAME",
    "MODE_DENSE",
    "MODE_HYBRID",
    "MODE_SPARSE",
    "TOKENIZER_VERSION",
    "VALID_MODES",
    "BM25Config",
    "FusionConfig",
    "HybridResult",
    "HybridRetriever",
    "HybridTrace",
    "RetrievalResult",
    "ScoredChunk",
    "key_tokens",
    "reciprocal_rank_fusion",
    "stopwords",
    "tokenize",
]
