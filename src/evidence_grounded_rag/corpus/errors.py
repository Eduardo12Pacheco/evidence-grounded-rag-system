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


class SourceManifestError(CorpusError):
    """Base class for source-manifest audit failures.

    These errors describe the separate corpus source-manifest boundary used
    for the bounded, explicitly approved real-local corpus. They do NOT loosen
    the synthetic fixture gate and are never mapped onto the serving pipeline.
    """

    code = "invalid_source_manifest"


class MalformedSourceManifestError(SourceManifestError):
    code = "malformed_source_manifest"


class UnsupportedSourceManifestVersionError(SourceManifestError):
    code = "unsupported_source_manifest_version"


class UnapprovedSourceError(SourceManifestError):
    code = "unapproved_source"


class BlockedCorpusStatusError(SourceManifestError):
    code = "blocked_corpus_status"


class MissingSourceEvidenceError(SourceManifestError):
    code = "missing_source_evidence"


class RevisionMismatchError(SourceManifestError):
    code = "revision_mismatch"


class LicenseMismatchError(SourceManifestError):
    code = "license_mismatch"


class PrivacyAuditError(SourceManifestError):
    code = "privacy_audit_failed"


class InvalidRedistributionStatusError(SourceManifestError):
    code = "invalid_redistribution_status"


class MissingRedistributionJustificationError(SourceManifestError):
    code = "missing_redistribution_justification"


class SourceFileError(SourceManifestError):
    code = "source_file_error"


class SourceFileMissingError(SourceFileError):
    code = "source_file_missing"


class SourceFileSizeMismatchError(SourceFileError):
    code = "source_file_size_mismatch"


class SourceFileChecksumMismatchError(SourceFileError):
    code = "source_file_checksum_mismatch"
