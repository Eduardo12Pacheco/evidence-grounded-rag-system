"""Typed models for the license-gated corpus layer.

Every object carries its provenance forward so downstream traces, citations,
and evaluation reports can record where a piece of evidence came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_grounded_rag.corpus.ids import (
    derive_chunk_id,
    local_location,
    synthetic_location,
)
from evidence_grounded_rag.corpus.normalize import sha256_hex


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of text with a stable identity and location."""

    chunk_id: str
    document_id: str
    seq: int
    text: str
    location: str
    corpus_version: str
    checksum_sha256: str = ""


@dataclass(frozen=True)
class Document:
    """A source document with full provenance metadata."""

    doc_id: str
    title: str
    version: str
    captured_at: str
    source_name: str
    source_url: str | None
    attribution: str
    license_note: str
    checksum_sha256: str
    chunks: tuple[Chunk, ...] = field(default_factory=tuple)
    corpus_version: str = ""


@dataclass(frozen=True)
class LicenseInfo:
    """License terms attached to a corpus manifest."""

    status: str
    terms: str
    redistribution: str


@dataclass(frozen=True)
class RealCorpusGate:
    """State of the real-corpus gate for this manifest.

    A manifest that declares ``real_corpus`` must prove its license has been
    independently confirmed. No such corpus is authorized today.
    """

    blocked: bool
    blocker: str


@dataclass(frozen=True)
class Corpus:
    """A fully validated, license-gated corpus ready for retrieval."""

    corpus_id: str
    corpus_version: str
    provenance: str
    synthetic: bool
    real_corpus: bool
    license_info: LicenseInfo
    real_corpus_gate: RealCorpusGate
    documents: tuple[Document, ...]

    def chunks(self) -> list[Chunk]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def document_by_id(self, document_id: str) -> Document | None:
        for doc in self.documents:
            if doc.doc_id == document_id:
                return doc
        return None


def build_chunks(
    corpus_id: str,
    corpus_version: str,
    document_id: str,
    texts: list[str],
) -> tuple[Chunk, ...]:
    """Build chunk records with derived IDs and stable ``synthetic://`` locations."""
    return _build_chunks_with_location(
        corpus_id, corpus_version, document_id, texts, synthetic_location
    )


def build_local_chunks(
    corpus_id: str,
    corpus_version: str,
    document_id: str,
    texts: list[str],
) -> tuple[Chunk, ...]:
    """Build chunk records with derived IDs and stable ``local://`` locations.

    Used by the real-local corpus boundary. Locations use the ``local://``
    scheme so real-local content is never mistaken for synthetic fixtures.
    """
    return _build_chunks_with_location(
        corpus_id, corpus_version, document_id, texts, local_location
    )


def _build_chunks_with_location(
    corpus_id: str,
    corpus_version: str,
    document_id: str,
    texts: list[str],
    location_fn,
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for seq, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=derive_chunk_id(document_id, seq),
                document_id=document_id,
                seq=seq,
                text=text,
                location=location_fn(corpus_id, document_id, seq),
                corpus_version=corpus_version,
                checksum_sha256=sha256_hex(text),
            )
        )
    return tuple(chunks)
