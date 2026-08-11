"""License-gated corpus layer.

Loads versioned synthetic fixture manifests, derives stable document/chunk
identities, validates checksums, rejects duplicates, and fails closed on any
blocked or unapproved corpus. No real corpus is authorized yet.
"""

from __future__ import annotations

from evidence_grounded_rag.corpus.errors import (
    BlockedRealCorpusError,
    ChecksumMismatchError,
    CorpusError,
    DuplicateChunkIdError,
    DuplicateDocumentIdError,
    ManifestFormatError,
    UnapprovedLicenseError,
)
from evidence_grounded_rag.corpus.ids import (
    derive_chunk_id,
    synthetic_location,
    validate_chunk_id,
    validate_document_id,
)
from evidence_grounded_rag.corpus.license_gate import (
    REAL_CORPUS_BLOCKER,
    GateVerdict,
    gate_manifest,
)
from evidence_grounded_rag.corpus.loader import load_corpus
from evidence_grounded_rag.corpus.models import (
    Chunk,
    Corpus,
    Document,
    LicenseInfo,
    RealCorpusGate,
    build_chunks,
)
from evidence_grounded_rag.corpus.normalize import (
    normalize_text,
    normalized_equals,
    sha256_hex,
)

__all__ = [
    "REAL_CORPUS_BLOCKER",
    "BlockedRealCorpusError",
    "ChecksumMismatchError",
    "Chunk",
    "Corpus",
    "CorpusError",
    "Document",
    "DuplicateChunkIdError",
    "DuplicateDocumentIdError",
    "GateVerdict",
    "LicenseInfo",
    "ManifestFormatError",
    "RealCorpusGate",
    "UnapprovedLicenseError",
    "build_chunks",
    "derive_chunk_id",
    "gate_manifest",
    "load_corpus",
    "normalize_text",
    "normalized_equals",
    "sha256_hex",
    "synthetic_location",
    "validate_chunk_id",
    "validate_document_id",
]
