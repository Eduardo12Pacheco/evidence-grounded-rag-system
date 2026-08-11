"""Deterministic tokenization shared by retrieval, embeddings, and generation.

The tokenizer is committed and versioned. It performs formatting-noise removal
(NFKC, case folding) and splits on any run of non-alphanumeric characters.
Unicode letters and digits count as alphanumeric, so this works for the
English fixture corpus and degrades predictably elsewhere. There is no
stemming: "review" and "reviewed" are different tokens, which is a documented
limit that fixture and query wording must respect.
"""

from __future__ import annotations

import re
import unicodedata

from evidence_grounded_rag.versions import TOKENIZER_VERSION

__all__ = ["TOKENIZER_VERSION", "key_tokens", "stopwords", "tokenize"]

_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return the deterministic token list for *text*."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    composed = unicodedata.normalize("NFKC", text)
    folded = composed.casefold()
    tokens = _TOKEN_SPLIT_RE.split(folded)
    return [token for token in tokens if token]


def stopwords() -> frozenset[str]:
    """Return the committed stopword set used for key-token extraction.

    Stopwords are excluded from query key tokens (used to decide whether
    evidence actually addresses the query) but are still indexed and scored by
    BM25. The set is small and committed so behavior is reproducible.
    """
    return frozenset(
        {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "of", "for", "to", "in", "on", "at", "by", "with", "and", "or",
            "not", "no", "do", "does", "did", "how", "what", "when", "where",
            "which", "who", "why", "this", "that", "it", "its", "from", "as",
            "about", "per", "into", "during", "up", "after", "before",
            "between", "through", "out", "so", "if", "than", "too", "can",
            "may", "will", "would", "should", "also", "have", "has", "had",
            "must", "much", "many", "more", "most", "some", "any", "own",
        }
    )


def key_tokens(text: str) -> list[str]:
    """Return the non-stopword tokens of *text* in order of appearance."""
    return [token for token in tokenize(text) if token not in stopwords()]
