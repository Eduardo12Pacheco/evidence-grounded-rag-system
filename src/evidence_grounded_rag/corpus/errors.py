"""Errors raised by the corpus layer, each carrying a stable error code.

The codes are the stable contract the rest of the system maps onto abstention
reasons (``blocked_corpus``, ``invalid_corpus``, ``stale_corpus``).
"""

from __future__ import annotations


class CorpusError(Exception):
    """Base class for corpus-layer failures. ``code`` is the stable error code."""

    code: str = "invalid_corpus"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ManifestFormatError(CorpusError):
    code = "unsupported_format"


class ChecksumMismatchError(CorpusError):
    code = "checksum_mismatch"


class DuplicateDocumentIdError(CorpusError):
    code = "duplicate_document_id"


class DuplicateChunkIdError(CorpusError):
    code = "duplicate_chunk_id"


class BlockedRealCorpusError(CorpusError):
    code = "blocked_real_corpus"


class UnapprovedLicenseError(CorpusError):
    code = "unapproved_license"
