"""Deterministic text normalization for the evidence-grounded RAG system.

The normalization here documents formatting noise (Unicode normalization,
whitespace collapse, control-character removal) and is used for checksums,
duplicate detection, and exact excerpt comparison. It is deliberately *not*
the aggressive case-folding used by the retrieval tokenizer: checksums must be
stable and deterministic across Python versions.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Return a deterministic normalized form of *text*.

    The steps are documented so checksums are reproducible anywhere:

    1. Normalize to Unicode NFKC (compatibility composition).
    2. Remove ASCII control characters (keeps ``\\n`` and ``\\t`` so chunk
       boundaries survive).
    3. Collapse all runs of whitespace to a single space.
    4. Strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    composed = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_RE.sub("", composed)
    collapsed = _WHITESPACE_RE.sub(" ", cleaned)
    return collapsed.strip()


def normalized_equals(left: str, right: str) -> bool:
    """Return whether two strings are equal under formatting-noise removal."""
    return normalize_text(left) == normalize_text(right)


def sha256_hex(text: str) -> str:
    """Return the lowercase SHA-256 hex digest of the normalized text."""
    import hashlib

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
