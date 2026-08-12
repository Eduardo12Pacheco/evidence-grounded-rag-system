from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import (
    BlockedRealCorpusError,
    ChecksumMismatchError,
    DuplicateDocumentIdError,
    ManifestFormatError,
    UnapprovedLicenseError,
    load_corpus,
)
from evidence_grounded_rag.corpus.ids import derive_chunk_id, synthetic_location
from evidence_grounded_rag.corpus.normalize import (
    normalize_text,
    normalized_equals,
    sha256_hex,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def fixture_corpus():
    return load_corpus(FIXTURE_MANIFEST)


def _manifest_documents():
    raw = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    return raw["documents"]


def test_fixture_corpus_loads_with_expected_shape(fixture_corpus):
    assert fixture_corpus.corpus_id == "synthetic-fixtures"
    assert fixture_corpus.corpus_version == "1.0.0"
    assert fixture_corpus.synthetic is True
    assert fixture_corpus.real_corpus is False
    assert fixture_corpus.provenance == "project-generated-synthetic"
    assert fixture_corpus.license_info.status == "approved"
    assert len(fixture_corpus.documents) == 6
    assert len(fixture_corpus.chunks()) == 11


def test_chunk_ids_derived_from_stable_identity_and_position(fixture_corpus):
    first = fixture_corpus.documents[0]
    assert first.doc_id == "doc-aurora-park-permits"
    for seq, chunk in enumerate(first.chunks):
        assert chunk.chunk_id == derive_chunk_id(first.doc_id, seq)
        assert chunk.chunk_id == f"{first.doc_id}::{seq}"
        assert chunk.location == synthetic_location(
            "synthetic-fixtures", first.doc_id, seq
        )
        assert chunk.corpus_version == "1.0.0"


def test_checksum_is_sha256_of_normalized_text(fixture_corpus):
    for document in fixture_corpus.documents:
        expected = sha256_hex("\n".join(chunk.text for chunk in document.chunks))
        assert document.checksum_sha256 == expected


def test_checksum_mismatch_fails_closed(tmp_path):
    records = _manifest_documents()
    records[0]["checksum_sha256"] = "f" * 64
    manifest = {
        "manifest": _header(),
        "documents": records,
    }
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ChecksumMismatchError):
        load_corpus(path)


def test_duplicate_document_id_fails_closed(tmp_path):
    records = _manifest_documents()
    records.append(dict(records[0]))
    manifest = {"manifest": _header(), "documents": records}
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DuplicateDocumentIdError):
        load_corpus(path)


def test_blocked_real_corpus_fails_closed(tmp_path):
    header = _header()
    header["synthetic"] = False
    header["real_corpus"] = True
    header["license"]["status"] = "unapproved"
    header["real_corpus_gate"] = {
        "blocked": True,
        "blocker": "No external license has been independently confirmed.",
    }
    manifest = {"manifest": header, "documents": _manifest_documents()}
    path = tmp_path / "blocked-real.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BlockedRealCorpusError) as excinfo:
        load_corpus(path)
    assert "No external license" in str(excinfo.value)


def test_real_corpus_blocker_is_exact_gobec_gate(tmp_path):
    header = _header()
    header["synthetic"] = False
    header["real_corpus"] = True
    header["license"]["status"] = "approved"
    header["real_corpus_gate"] = {"blocked": True}
    manifest = {"manifest": header, "documents": _manifest_documents()}
    path = tmp_path / "approved-but-real.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BlockedRealCorpusError) as excinfo:
        load_corpus(path)
    message = str(excinfo.value)
    assert "Gob.ec" in message
    assert "by/3.0/ec" in message
    assert "written license confirmation" in message


def test_unapproved_license_fails_closed(tmp_path):
    header = _header()
    header["license"]["status"] = "under_review"
    manifest = {"manifest": header, "documents": _manifest_documents()}
    path = tmp_path / "under-review.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(UnapprovedLicenseError):
        load_corpus(path)


def test_unsupported_format_version_fails_closed(tmp_path):
    header = _header()
    header["format_version"] = "9.9"
    manifest = {"manifest": header, "documents": _manifest_documents()}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestFormatError):
        load_corpus(path)


def test_jsonl_manifest_loads_identically(tmp_path):
    raw = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    lines = [json.dumps({"manifest": raw["manifest"]})]
    for document in raw["documents"]:
        lines.append(json.dumps({"document": document}))
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_corpus(path)
    json_corpus = load_corpus(FIXTURE_MANIFEST)
    assert loaded.corpus_id == json_corpus.corpus_id
    assert loaded.chunks() == json_corpus.chunks()


def test_unknown_suffix_fails_closed(tmp_path):
    path = tmp_path / "manifest.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestFormatError):
        load_corpus(path)


def test_normalization_is_deterministic_and_documents_whitespace_noise():
    messy = "  The\t Central  Library\nis open  "
    clean = normalize_text(messy)
    assert clean == "The Central Library is open"
    assert normalize_text(messy) == clean
    assert normalized_equals("  a\tb ", "a b")
    assert not normalized_equals("ab", "ba")
    assert len(sha256_hex("hello")) == 64


def test_derived_ids_validate():
    assert derive_chunk_id("doc-abc", 3) == "doc-abc::3"
    assert synthetic_location("c", "doc-x", 2) == "synthetic://c/doc-x/chunk/2"
    with pytest.raises(ValueError):
        derive_chunk_id("Doc With Spaces", 0)
    with pytest.raises(ValueError):
        derive_chunk_id("doc-ok", -1)


def _header() -> dict[str, object]:
    return {
        "manifest_id": "test-manifest",
        "format_version": "1.0",
        "corpus_id": "synthetic-fixtures",
        "corpus_version": "1.0.0",
        "synthetic": True,
        "real_corpus": False,
        "provenance": "project-generated-synthetic",
        "license": {
            "status": "approved",
            "terms": "synthetic",
            "redistribution": "test only",
        },
        "real_corpus_gate": {"blocked": True, "blocker": "test blocker"},
    }
