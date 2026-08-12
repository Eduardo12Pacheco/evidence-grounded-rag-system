from __future__ import annotations

import re

_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_CHUNK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}::\d+$")


def derive_chunk_id(document_id: str, seq: int) -> str:
    validate_document_id(document_id)
    if not isinstance(seq, int) or seq < 0:
        raise ValueError(f"chunk seq must be a non-negative int, got {seq!r}")
    return f"{document_id}::{seq}"


def synthetic_location(corpus_id: str, document_id: str, seq: int) -> str:
    return f"synthetic://{corpus_id}/{document_id}/chunk/{seq}"


def local_location(corpus_id: str, document_id: str, seq: int) -> str:
    return f"local://{corpus_id}/{document_id}/chunk/{seq}"


def validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not _DOC_ID_RE.match(document_id):
        raise ValueError(f"invalid document id: {document_id!r}")


def validate_chunk_id(chunk_id: str) -> None:
    if not isinstance(chunk_id, str) or not _CHUNK_ID_RE.match(chunk_id):
        raise ValueError(f"invalid chunk id: {chunk_id!r}")
