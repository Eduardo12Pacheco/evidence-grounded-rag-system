"""Deterministic, versioned tokenizer and key-token extraction."""

from __future__ import annotations

import re
import unicodedata

from evidence_grounded_rag.versions import TOKENIZER_VERSION

__all__ = ["TOKENIZER_VERSION", "key_tokens", "stopwords", "tokenize"]

_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    composed = unicodedata.normalize("NFKC", text)
    folded = composed.casefold()
    tokens = _TOKEN_SPLIT_RE.split(folded)
    return [token for token in tokens if token]


def stopwords() -> frozenset[str]:
    return frozenset(
        {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "of",
            "for",
            "to",
            "in",
            "on",
            "at",
            "by",
            "with",
            "and",
            "or",
            "not",
            "no",
            "do",
            "does",
            "did",
            "how",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "this",
            "that",
            "it",
            "its",
            "from",
            "as",
            "about",
            "per",
            "into",
            "during",
            "up",
            "after",
            "before",
            "between",
            "through",
            "out",
            "so",
            "if",
            "than",
            "too",
            "can",
            "may",
            "will",
            "would",
            "should",
            "also",
            "have",
            "has",
            "had",
            "must",
            "much",
            "many",
            "more",
            "most",
            "some",
            "any",
            "own",
        }
    )


def key_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in stopwords()]
