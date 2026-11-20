"""Audit helpers for corpus source manifests, statuses, and approvals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_grounded_rag.corpus.errors import (
    BlockedCorpusStatusError,
    InvalidRedistributionStatusError,
    LicenseMismatchError,
    MalformedSourceManifestError,
    MissingRedistributionJustificationError,
    MissingSourceEvidenceError,
    PrivacyAuditError,
    RevisionMismatchError,
    SourceFileChecksumMismatchError,
    SourceFileError,
    SourceFileMissingError,
    SourceFileSizeMismatchError,
    UnapprovedSourceError,
    UnsupportedSourceManifestVersionError,
)
from evidence_grounded_rag.corpus.normalize import sha256_bytes
from evidence_grounded_rag.versions import SOURCE_MANIFEST_VERSION

SUPPORTED_SOURCE_MANIFEST_VERSION = SOURCE_MANIFEST_VERSION

STATUS_APPROVED_LOCAL = "approved-for-local-development"
STATUS_APPROVED_REDISTRIBUTION = "approved-for-redistribution"
STATUS_BLOCKED = "blocked"
STATUS_UNDER_REVIEW = "under-review"
ALLOWED_STATUSES = frozenset({STATUS_APPROVED_LOCAL, STATUS_APPROVED_REDISTRIBUTION})

REDISTRIBUTION_LOCAL_ONLY = "not-redistributed"
ALLOWED_REDISTRIBUTION_STATUSES = frozenset(
    {REDISTRIBUTION_LOCAL_ONLY, STATUS_APPROVED_REDISTRIBUTION}
)

PRIVACY_STATUS_CLEAN = "no-private-content-detected"
PRIVACY_STATUS_EXCLUDED = "private-content-excluded"
ALLOWED_PRIVACY_STATUSES = frozenset({PRIVACY_STATUS_CLEAN, PRIVACY_STATUS_EXCLUDED})

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HTTP_URL_RE = re.compile(r"^https?://")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = (
    "manifest_version",
    "corpus_id",
    "corpus_version",
    "status",
    "publisher",
    "official_docs_url",
    "source_url",
    "license",
    "revision",
    "revision_tag",
    "access_date",
    "acquisition_method",
    "expected_files",
    "document_count",
    "redistribution_status",
    "privacy_status",
    "verification_timestamp",
)


@dataclass(frozen=True)
class ApprovedCorpus:
    corpus_id: str
    corpus_version: str
    revision: str
    revision_tag: str
    license_name: str
    license_version: str


APPROVED_CORPORA: tuple[ApprovedCorpus, ...] = (
    ApprovedCorpus(
        corpus_id="python-docs-3.12.13",
        corpus_version="1.0.0",
        revision="3bb231a6a5dc02b95658877318bf61501a7209e9",
        revision_tag="v3.12.13",
        license_name="Python Software Foundation License Version 2",
        license_version="2",
    ),
)


@dataclass(frozen=True)
class SourceLicense:
    name: str
    version: str
    url: str
    evidence_excerpt: str


@dataclass(frozen=True)
class SourceFileEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceManifest:
    manifest_version: str
    corpus_id: str
    corpus_version: str
    status: str
    publisher: str
    official_docs_url: str
    source_url: str
    license: SourceLicense
    revision: str
    revision_tag: str
    access_date: str
    acquisition_method: str
    expected_files: tuple[SourceFileEntry, ...]
    document_count: int
    redistribution_status: str
    privacy_status: str
    verification_timestamp: str
    redistribution_justification: str | None = None


@dataclass(frozen=True)
class AuditVerdict:
    allowed: bool
    reasons: tuple[str, ...]


def load_source_manifest(path: str | Path) -> SourceManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SourceFileMissingError(f"source manifest file not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedSourceManifestError(
            f"malformed source manifest JSON: {exc}"
        ) from exc
    return parse_source_manifest(data)


def parse_source_manifest(data: Any) -> SourceManifest:
    if not isinstance(data, dict):
        raise MalformedSourceManifestError("source manifest must be a JSON object")

    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        raise MalformedSourceManifestError(
            f"source manifest missing required fields: {', '.join(missing)}"
        )

    version = data["manifest_version"]
    if not isinstance(version, str) or version != SUPPORTED_SOURCE_MANIFEST_VERSION:
        raise UnsupportedSourceManifestVersionError(
            f"unsupported source manifest version {version!r}; "
            f"this implementation supports {SUPPORTED_SOURCE_MANIFEST_VERSION!r}"
        )

    for field in (
        "corpus_id",
        "corpus_version",
        "status",
        "publisher",
        "official_docs_url",
        "source_url",
        "revision",
        "revision_tag",
        "access_date",
        "acquisition_method",
        "redistribution_status",
        "privacy_status",
        "verification_timestamp",
    ):
        if not isinstance(data[field], str):
            raise MalformedSourceManifestError(f"field {field!r} must be a string")

    if data["official_docs_url"] and not _HTTP_URL_RE.match(data["official_docs_url"]):
        raise MalformedSourceManifestError("official_docs_url must be an http(s) URL")
    if data["source_url"] and not _HTTP_URL_RE.match(data["source_url"]):
        raise MalformedSourceManifestError("source_url must be an http(s) URL")
    if not _COMMIT_RE.fullmatch(data["revision"]):
        raise MalformedSourceManifestError(
            "revision must be a full 40-character hex commit SHA "
            "(exact pinned revision, never a mutable branch)"
        )
    if not _DATE_RE.fullmatch(data["access_date"]):
        raise MalformedSourceManifestError(
            f"access_date must be YYYY-MM-DD, got {data['access_date']!r}"
        )

    license_raw = data["license"]
    if not isinstance(license_raw, dict):
        raise MalformedSourceManifestError("'license' must be an object")
    license_missing = [
        key
        for key in ("name", "version", "url", "evidence_excerpt")
        if key not in license_raw
    ]
    if license_missing:
        raise MalformedSourceManifestError(
            f"'license' missing required fields: {', '.join(license_missing)}"
        )
    for key in ("name", "version", "evidence_excerpt"):
        if not isinstance(license_raw[key], str):
            raise MalformedSourceManifestError(f"license.{key} must be a string")
    if not isinstance(license_raw["url"], str):
        raise MalformedSourceManifestError("license.url must be a string")
    if license_raw["url"] and not _HTTP_URL_RE.match(license_raw["url"]):
        raise MalformedSourceManifestError("license.url must be an http(s) URL")

    files_raw = data["expected_files"]
    if not isinstance(files_raw, list):
        raise MalformedSourceManifestError("'expected_files' must be a list")
    entries = [_parse_file_entry(raw, i) for i, raw in enumerate(files_raw)]
    if not entries:
        raise MalformedSourceManifestError(
            "source manifest must declare at least one expected file"
        )

    if not isinstance(data["document_count"], int) or data["document_count"] < 0:
        raise MalformedSourceManifestError(
            "document_count must be a non-negative integer"
        )
    if data["document_count"] == 0:
        raise MalformedSourceManifestError(
            "source manifest document_count must be greater than zero"
        )
    if data["document_count"] != len(entries):
        raise MalformedSourceManifestError(
            "document_count must equal the number of expected files"
        )

    if "redistribution_justification" in data and not isinstance(
        data["redistribution_justification"], str
    ):
        raise MalformedSourceManifestError(
            "redistribution_justification must be a string"
        )

    return SourceManifest(
        manifest_version=version,
        corpus_id=data["corpus_id"],
        corpus_version=data["corpus_version"],
        status=data["status"],
        publisher=data["publisher"],
        official_docs_url=data["official_docs_url"],
        source_url=data["source_url"],
        license=SourceLicense(
            name=license_raw["name"],
            version=license_raw["version"],
            url=license_raw["url"],
            evidence_excerpt=license_raw["evidence_excerpt"],
        ),
        revision=data["revision"],
        revision_tag=data["revision_tag"],
        access_date=data["access_date"],
        acquisition_method=data["acquisition_method"],
        expected_files=tuple(entries),
        document_count=data["document_count"],
        redistribution_status=data["redistribution_status"],
        privacy_status=data["privacy_status"],
        verification_timestamp=data["verification_timestamp"],
        redistribution_justification=data.get("redistribution_justification"),
    )


def _parse_file_entry(raw: Any, index: int) -> SourceFileEntry:
    if not isinstance(raw, dict):
        raise MalformedSourceManifestError(f"expected_files[{index}] must be an object")
    for key in ("path", "sha256"):
        if not isinstance(raw.get(key), str):
            raise MalformedSourceManifestError(
                f"expected_files[{index}].{key} must be a string"
            )
    path = raw["path"]
    if path.startswith(("/", "\\")) or ".." in path.split("/"):
        raise MalformedSourceManifestError(
            f"expected_files[{index}].path must be a relative path inside the raw "
            f"directory, got {path!r}"
        )
    if not isinstance(raw.get("size"), int) or raw["size"] < 0:
        raise MalformedSourceManifestError(
            f"expected_files[{index}].size must be a non-negative integer"
        )
    if not _SHA256_RE.fullmatch(raw["sha256"].lower()):
        raise MalformedSourceManifestError(
            f"expected_files[{index}].sha256 must be a 64-character hex SHA-256"
        )
    return SourceFileEntry(path=path, size=raw["size"], sha256=raw["sha256"].lower())


def audit_source_manifest(manifest: SourceManifest) -> AuditVerdict:
    reasons = [message for _, message in _iter_audit_failures(manifest)]
    return AuditVerdict(allowed=not reasons, reasons=tuple(reasons))


def assert_audit_allowed(manifest: SourceManifest) -> None:
    for error_type, message in _iter_audit_failures(manifest):
        raise error_type(message)


def _iter_audit_failures(manifest: SourceManifest):
    if manifest.status == STATUS_BLOCKED:
        yield BlockedCorpusStatusError, "corpus status is 'blocked'"
    elif manifest.status == STATUS_UNDER_REVIEW:
        yield BlockedCorpusStatusError, "corpus status is 'under-review'"
    elif manifest.status not in ALLOWED_STATUSES:
        yield (
            BlockedCorpusStatusError,
            (
                f"corpus status {manifest.status!r} is not approved; only "
                f"{STATUS_APPROVED_LOCAL!r} or {STATUS_APPROVED_REDISTRIBUTION!r}"
            ),
        )

    approved = _find_approved(manifest.corpus_id, manifest.corpus_version)
    if approved is None:
        yield (
            UnapprovedSourceError,
            (
                f"corpus {manifest.corpus_id} v{manifest.corpus_version} is not an "
                "approved source corpus"
            ),
        )
    else:
        if manifest.revision != approved.revision:
            yield (
                RevisionMismatchError,
                (
                    f"revision {manifest.revision!r} does not match the approved "
                    f"revision {approved.revision!r}"
                ),
            )
        if manifest.revision_tag != approved.revision_tag:
            yield (
                RevisionMismatchError,
                (
                    f"revision tag {manifest.revision_tag!r} does not match the "
                    f"approved tag {approved.revision_tag!r}"
                ),
            )
        if manifest.license.name != approved.license_name:
            yield (
                LicenseMismatchError,
                (
                    f"license {manifest.license.name!r} does not match the approved "
                    f"license {approved.license_name!r}"
                ),
            )
        if manifest.license.version != approved.license_version:
            yield (
                LicenseMismatchError,
                (
                    f"license version {manifest.license.version!r} does not match the "
                    f"approved license version {approved.license_version!r}"
                ),
            )

    for field, value in (
        ("official_docs_url", manifest.official_docs_url),
        ("source_url", manifest.source_url),
        ("publisher", manifest.publisher),
        ("acquisition_method", manifest.acquisition_method),
        ("license.evidence_excerpt", manifest.license.evidence_excerpt),
        ("verification_timestamp", manifest.verification_timestamp),
    ):
        if not value.strip():
            yield MissingSourceEvidenceError, f"missing source evidence: {field}"

    if manifest.privacy_status not in ALLOWED_PRIVACY_STATUSES:
        yield (
            PrivacyAuditError,
            (
                f"privacy_status {manifest.privacy_status!r} is not a recorded audit "
                "result (must be 'no-private-content-detected' or 'private-content-excluded')"
            ),
        )

    if manifest.redistribution_status not in ALLOWED_REDISTRIBUTION_STATUSES:
        yield (
            InvalidRedistributionStatusError,
            (
                f"redistribution_status {manifest.redistribution_status!r} is not a "
                f"known value (must be {REDISTRIBUTION_LOCAL_ONLY!r} or "
                f"{STATUS_APPROVED_REDISTRIBUTION!r})"
            ),
        )
    elif (
        manifest.status == STATUS_APPROVED_LOCAL
        and manifest.redistribution_status != REDISTRIBUTION_LOCAL_ONLY
    ):
        yield (
            InvalidRedistributionStatusError,
            (
                f"status {STATUS_APPROVED_LOCAL!r} cannot claim "
                f"redistribution_status {manifest.redistribution_status!r}; local-only "
                f"development requires {REDISTRIBUTION_LOCAL_ONLY!r}"
            ),
        )
    elif (
        manifest.status == STATUS_APPROVED_REDISTRIBUTION
        and manifest.redistribution_status != STATUS_APPROVED_REDISTRIBUTION
    ):
        yield (
            InvalidRedistributionStatusError,
            (
                f"status {STATUS_APPROVED_REDISTRIBUTION!r} requires the matching "
                f"redistribution_status {STATUS_APPROVED_REDISTRIBUTION!r}"
            ),
        )

    if (
        manifest.status == STATUS_APPROVED_REDISTRIBUTION
        and not (manifest.redistribution_justification or "").strip()
    ):
        yield (
            MissingRedistributionJustificationError,
            (
                f"status {STATUS_APPROVED_REDISTRIBUTION!r} requires an explicit "
                "redistribution_justification"
            ),
        )


def _find_approved(corpus_id: str, corpus_version: str) -> ApprovedCorpus | None:
    for approved in APPROVED_CORPORA:
        if (
            approved.corpus_id == corpus_id
            and approved.corpus_version == corpus_version
        ):
            return approved
    return None


def safe_raw_path(raw_dir: str | Path, rel_path: str) -> Path:
    root = Path(raw_dir).resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SourceFileError(
            f"source file path escapes the raw directory: {rel_path!r}"
        ) from exc
    return candidate


def verify_source_files(manifest: SourceManifest, raw_dir: str | Path) -> None:
    root = Path(raw_dir)
    if not root.is_dir():
        raise SourceFileMissingError(f"raw directory not found: {root}")

    for entry in manifest.expected_files:
        path = safe_raw_path(root, entry.path)
        if not path.is_file():
            raise SourceFileMissingError(f"source file missing: {entry.path}")
        size = path.stat().st_size
        if size != entry.size:
            raise SourceFileSizeMismatchError(
                f"source file {entry.path}: actual size {size} != declared {entry.size}"
            )
        actual = sha256_bytes(path.read_bytes())
        if actual != entry.sha256:
            raise SourceFileChecksumMismatchError(
                f"source file {entry.path}: actual sha256 {actual} != declared {entry.sha256}"
            )
