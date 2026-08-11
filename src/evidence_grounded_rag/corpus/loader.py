"""Synthetic fixture manifest loader with checksum validation.

Supports two committed formats:

- ``.json``: a single object with ``manifest`` (metadata) and ``documents``
  (list of document records).
- ``.jsonl``: the first line carries ``{"manifest": {...}}`` and each following
  line carries one document record as ``{"document": {...}}``.

Loading always:

1. parses and validates the manifest schema and format version;
2. applies the license gate (fails closed on blocked real corpora / unapproved
   licenses);
3. derives stable chunk IDs from document identity and position;
4. recomputes every document SHA-256 checksum over the normalized text and
   rejects any mismatch;
5. rejects duplicate document IDs and duplicate chunk IDs.

Every loaded corpus is the result of the full gate: no bypass path exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidence_grounded_rag.corpus.errors import (
    ChecksumMismatchError,
    DuplicateChunkIdError,
    DuplicateDocumentIdError,
    ManifestFormatError,
)
from evidence_grounded_rag.corpus.ids import validate_chunk_id, validate_document_id
from evidence_grounded_rag.corpus.license_gate import assert_allowed
from evidence_grounded_rag.corpus.models import (
    Chunk,
    Corpus,
    Document,
    LicenseInfo,
    RealCorpusGate,
    build_chunks,
)
from evidence_grounded_rag.corpus.normalize import sha256_hex
from evidence_grounded_rag.versions import CORPUS_FORMAT_VERSION

_REQUIRED_MANIFEST_FIELDS = ("corpus_id", "corpus_version", "provenance")


def load_corpus(path: str | Path) -> Corpus:
    """Load, gate, and validate a synthetic fixture manifest from *path*."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestFormatError(f"manifest file not found: {manifest_path}")
    try:
        if manifest_path.suffix.lower() == ".jsonl":
            manifest, documents = _parse_jsonl(manifest_path)
        elif manifest_path.suffix.lower() == ".json":
            manifest, documents = _parse_json(manifest_path)
        else:
            raise ManifestFormatError(
                f"unsupported manifest suffix {manifest_path.suffix!r}; "
                "expected .json or .jsonl"
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestFormatError(f"malformed manifest JSON: {exc}") from exc

    _validate_manifest_schema(manifest)
    assert_allowed(manifest)

    corpus_id = str(manifest["corpus_id"])
    corpus_version = str(manifest["corpus_version"])

    license_info = manifest.get("license") or {}
    real_gate_raw = manifest.get("real_corpus_gate") or {}

    documents = _build_documents(corpus_id, corpus_version, documents)

    return Corpus(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        provenance=str(manifest["provenance"]),
        synthetic=bool(manifest.get("synthetic", False)),
        real_corpus=bool(manifest.get("real_corpus", False)),
        license_info=LicenseInfo(
            status=str(license_info.get("status", "unknown")),
            terms=str(license_info.get("terms", "")),
            redistribution=str(license_info.get("redistribution", "")),
        ),
        real_corpus_gate=RealCorpusGate(
            blocked=bool(real_gate_raw.get("blocked", True)),
            blocker=str(real_gate_raw.get("blocker", "")),
        ),
        documents=documents,
    )


def _parse_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestFormatError("JSON manifest root must be an object")
    if "manifest" not in raw or "documents" not in raw:
        raise ManifestFormatError("JSON manifest must contain 'manifest' and 'documents'")
    documents = raw["documents"]
    if not isinstance(documents, list):
        raise ManifestFormatError("JSON 'documents' must be a list")
    manifest = raw["manifest"]
    if not isinstance(manifest, dict):
        raise ManifestFormatError("JSON 'manifest' must be an object")
    return manifest, documents


def _parse_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest: dict[str, Any] | None = None
    documents: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ManifestFormatError(f"JSONL line {line_no} must be an object")
        if "manifest" in record:
            if manifest is not None:
                raise ManifestFormatError("JSONL must contain exactly one manifest header line")
            manifest = record["manifest"]
        elif "document" in record:
            documents.append(record["document"])
        else:
            raise ManifestFormatError(
                f"JSONL line {line_no} must be a manifest header or a document record"
            )
    if manifest is None:
        raise ManifestFormatError("JSONL is missing the manifest header line")
    return manifest, documents


def _validate_manifest_schema(manifest: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ManifestFormatError(f"manifest missing required fields: {', '.join(missing)}")
    declared_format = str(manifest.get("format_version", ""))
    if declared_format != CORPUS_FORMAT_VERSION:
        raise ManifestFormatError(
            f"unsupported corpus format_version {declared_format!r}; "
            f"this implementation supports {CORPUS_FORMAT_VERSION!r}"
        )


def _build_documents(
    corpus_id: str,
    corpus_version: str,
    records: list[dict[str, Any]],
) -> tuple[Document, ...]:
    documents: list[Document] = []
    seen_doc_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ManifestFormatError(f"document #{index} must be an object")
        doc_id = record.get("doc_id")
        if not isinstance(doc_id, str):
            raise ManifestFormatError(f"document #{index} missing string 'doc_id'")
        validate_document_id(doc_id)
        if doc_id in seen_doc_ids:
            raise DuplicateDocumentIdError(f"duplicate document id: {doc_id}")
        seen_doc_ids.add(doc_id)

        chunk_records = record.get("chunks")
        if not isinstance(chunk_records, list):
            raise ManifestFormatError(f"document {doc_id} missing 'chunks' list")
        texts: list[str] = []
        for seq, chunk_record in enumerate(chunk_records):
            if not isinstance(chunk_record, dict) or not isinstance(
                chunk_record.get("text"), str
            ):
                raise ManifestFormatError(
                    f"document {doc_id} chunk #{seq} must have a string 'text'"
                )
            declared_seq = chunk_record.get("seq")
            if declared_seq is not None and declared_seq != seq:
                raise ManifestFormatError(
                    f"document {doc_id} chunk #{seq} declares seq {declared_seq!r}; "
                    "chunk positions are derived from order"
                )
            texts.append(chunk_record["text"])

        chunks = build_chunks(corpus_id, corpus_version, doc_id, texts)
        _check_duplicate_chunk_ids(chunks)

        declared_checksum = record.get("checksum_sha256")
        document_text = "\n".join(texts)
        actual_checksum = sha256_hex(document_text)
        if not isinstance(declared_checksum, str) or declared_checksum.lower() != actual_checksum:
            raise ChecksumMismatchError(
                f"document {doc_id} checksum mismatch: declared "
                f"{declared_checksum!r}, computed {actual_checksum}"
            )

        documents.append(
            Document(
                doc_id=doc_id,
                title=str(record.get("title", doc_id)),
                version=str(record.get("version", "1.0.0")),
                captured_at=str(record.get("captured_at", "")),
                source_name=str(record.get("source_name", "project-generated-synthetic")),
                source_url=record.get("source_url"),
                attribution=str(record.get("attribution", "")),
                license_note=str(record.get("license", "")),
                checksum_sha256=actual_checksum,
                chunks=chunks,
                corpus_version=corpus_version,
            )
        )
    return tuple(documents)


def _check_duplicate_chunk_ids(chunks: tuple[Chunk, ...]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        validate_chunk_id(chunk.chunk_id)
        if chunk.chunk_id in seen:
            raise DuplicateChunkIdError(f"duplicate chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
