"""Shared deterministic text processing.

The tokenizer is shared by sparse retrieval, the deterministic embedding, and
the extractive generator, so it lives in a dependency-free subpackage to avoid
import cycles between the retrieval and embeddings layers.
"""

from __future__ import annotations

from evidence_grounded_rag.text.tokenizer import (
    TOKENIZER_VERSION,
    key_tokens,
    stopwords,
    tokenize,
)

__all__ = ["TOKENIZER_VERSION", "key_tokens", "stopwords", "tokenize"]
