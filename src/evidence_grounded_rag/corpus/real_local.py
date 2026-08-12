"""Real-local corpus derivation from an audited source manifest.

This is the minimum real-local integration for the bounded corpus-onboarding
work unit (Python 3.12.13 official documentation). It takes an audited source
manifest plus the local raw files and derives an in-memory corpus with:

- stable document IDs derived from source paths;
- stable chunk IDs (``derive_chunk_id``) and ``local://`` locations;
- raw (byte-exact) and normalized SHA-256 checksums carried forward;
- source provenance, license, and revision carried forward on every document;
- privacy exclusion enforced: flagged documents never enter derived data.

The derived corpus is EXPLICITLY local-only. It is never the default corpus,
never the demo corpus, and never part of synthetic evaluation. No real text is
committed: the raw files live only under the gitignored ``data/raw/`` tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from evidence_grounded_rag.corpus.audit import (
    PRIVACY_STATUS_EXCLUDED,
    SourceManifest,
    assert_audit_allowed,
    safe_raw_path,
    verify_source_files,
)
from evidence_grounded_rag.corpus.errors import (
    PrivacyAuditError,
    SourceFileError,
)
from evidence_grounded_rag.corpus.models import Chunk, build_local_chunks
from evidence_grounded_rag.corpus.normalize import (
    normalize_text,
    sha256_hex,
)
from evidence_grounded_rag.corpus.privacy import PrivacyAuditResult, audit_privacy_texts

#: Committed maximum characters per derived chunk. Chunking is deterministic:
#: paragraphs first, then hard splits at the limit for over-long paragraphs.
DEFAULT_CHUNK_MAX_CHARS = 2000

_PATH_ID_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class RealLocalDocument:
    """One derived document carrying its full source provenance forward."""

    doc_id: str
    source_path: str
    title: str
    raw_size: int
    raw_sha256: str
    normalized_sha256: str
    official_docs_url: str
    source_url: str
    publisher: str
    license_name: str
    license_version: str
    license_url: str
    revision: str
    revision_tag: str
    access_date: str
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class RealLocalCorpus:
    """A fully audited, verified, privacy-screened local corpus."""

    corpus_id: str
    corpus_version: str
    status: str
    manifest: SourceManifest
    documents: tuple[RealLocalDocument, ...]
    excluded_document_ids: frozenset[str]
    privacy: PrivacyAuditResult = field(repr=False)

    def chunks(self) -> list[Chunk]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def document_by_id(self, document_id: str) -> RealLocalDocument | None:
        for doc in self.documents:
            if doc.doc_id == document_id:
                return doc
        return None


def derive_source_doc_id(source_path: str) -> str:
    """Derive a stable document ID from a relative source path.

    The ID is a deterministic transformation of the path, so it is stable
    across reloads as long as the captured file layout does not change.
    """
    stem = _PATH_ID_RE.sub("-", source_path.lower()).strip(".-_")
    if not stem or not stem[0].isalnum():
        stem = f"doc-{stem}"
    return stem[:127]


def chunk_paragraphs(text: str, *, max_chars: int = DEFAULT_CHUNK_MAX_CHARS) -> list[str]:
    """Split normalized text into deterministic chunks (paragraphs first)."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            chunks.append(paragraph[start:end])
            start = end
    return chunks


def build_real_local_corpus(
    source_manifest: SourceManifest,
    raw_dir: str | Path,
    *,
    excluded_document_ids: frozenset[str] = frozenset(),
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> RealLocalCorpus:
    """Audit, verify, privacy-screen, and derive the real-local corpus.

    Raises the corresponding ``SourceManifestError`` subclass on any audit,
    verification, privacy, or derivation failure.
    """
    assert_audit_allowed(source_manifest)
    verify_source_files(source_manifest, raw_dir)
    root = Path(raw_dir)

    texts_by_path: dict[str, str] = {}
    for entry in source_manifest.expected_files:
        path = safe_raw_path(root, entry.path)
        try:
            texts_by_path[entry.path] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SourceFileError(
                f"source file {entry.path} is not valid UTF-8 text: {exc}"
            ) from exc

    doc_ids = {
        entry.path: derive_source_doc_id(entry.path)
        for entry in source_manifest.expected_files
    }
    _check_unique_doc_ids(doc_ids)

    privacy_documents = [
        (doc_ids[entry.path], entry.path, texts_by_path[entry.path])
        for entry in source_manifest.expected_files
    ]
    privacy = audit_privacy_texts(privacy_documents)

    declared_excluded = source_manifest.privacy_status == PRIVACY_STATUS_EXCLUDED
    if privacy.clean and declared_excluded:
        raise PrivacyAuditError(
            "manifest declares 'private-content-excluded' but the privacy scan "
            "found no private content; update privacy_status"
        )
    if not privacy.clean and not declared_excluded:
        raise PrivacyAuditError(
            "privacy scan found private content not recorded in the manifest; "
            "set privacy_status to 'private-content-excluded' before use"
        )

    excluded = privacy.excluded_document_ids | set(excluded_document_ids)

    documents: list[RealLocalDocument] = []
    for entry in source_manifest.expected_files:
        text = texts_by_path[entry.path]
        doc_id = doc_ids[entry.path]
        if doc_id in excluded:
            continue
        normalized = normalize_text(text)
        chunks = build_local_chunks(
            source_manifest.corpus_id,
            source_manifest.corpus_version,
            doc_id,
            chunk_paragraphs(normalized, max_chars=chunk_max_chars),
        )
        documents.append(
            RealLocalDocument(
                doc_id=doc_id,
                source_path=entry.path,
                title=f"{entry.path} (python-docs-3.12.13)",
                raw_size=entry.size,
                raw_sha256=entry.sha256,
                normalized_sha256=sha256_hex(normalized),
                official_docs_url=source_manifest.official_docs_url,
                source_url=source_manifest.source_url,
                publisher=source_manifest.publisher,
                license_name=source_manifest.license.name,
                license_version=source_manifest.license.version,
                license_url=source_manifest.license.url,
                revision=source_manifest.revision,
                revision_tag=source_manifest.revision_tag,
                access_date=source_manifest.access_date,
                chunks=chunks,
            )
        )

    return RealLocalCorpus(
        corpus_id=source_manifest.corpus_id,
        corpus_version=source_manifest.corpus_version,
        status=source_manifest.status,
        manifest=source_manifest,
        documents=tuple(documents),
        excluded_document_ids=frozenset(excluded),
        privacy=privacy,
    )


def _check_unique_doc_ids(doc_ids: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for path, doc_id in doc_ids.items():
        if doc_id in seen:
            raise SourceFileError(
                f"derived document id {doc_id!r} is ambiguous: {seen[doc_id]!r} "
                f"and {path!r} normalize to the same id"
            )
        seen[doc_id] = path
