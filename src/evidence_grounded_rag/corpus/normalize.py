from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    composed = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_RE.sub("", composed)
    collapsed = _WHITESPACE_RE.sub(" ", cleaned)
    return collapsed.strip()


def normalized_equals(left: str, right: str) -> bool:
    return normalize_text(left) == normalize_text(right)


def sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
