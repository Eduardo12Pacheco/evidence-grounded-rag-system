from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_grounded_rag.corpus.normalize import sha256_bytes
from evidence_grounded_rag.embeddings.manifest import (
    MODEL_ARCHITECTURE,
    MODEL_CARD_URL,
    MODEL_DIMENSION,
    MODEL_ID,
    MODEL_LICENSE_NAME,
    MODEL_LICENSE_URL,
    MODEL_LICENSE_VERSION,
    MODEL_MAX_INPUT_TOKENS,
    MODEL_NORMALIZATION,
    MODEL_POOLING,
    MODEL_PUBLISHER,
    MODEL_REVISION,
    MODEL_REVISION_TAG,
    MODEL_TOKENIZER,
    MODEL_VERSION,
    EmbeddingModelManifestError,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
    parse_embedding_model_manifest,
    verify_embedding_model_files,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "data" / "manifests" / "embedding-model-v1.0.0.json"


def _manifest_data(**overrides) -> dict[str, object]:
    data: dict[str, object] = {
        "manifest_version": "1.0",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "publisher": MODEL_PUBLISHER,
        "official_url": MODEL_CARD_URL,
        "model_card_url": MODEL_CARD_URL,
        "revision": MODEL_REVISION,
        "revision_tag": MODEL_REVISION_TAG,
        "license": {
            "name": MODEL_LICENSE_NAME,
            "version": MODEL_LICENSE_VERSION,
            "url": MODEL_LICENSE_URL,
            "evidence_excerpt": "official model card quotes the MIT license",
        },
        "architecture": dict(MODEL_ARCHITECTURE),
        "dimension": MODEL_DIMENSION,
        "max_input_tokens": MODEL_MAX_INPUT_TOKENS,
        "tokenizer": dict(MODEL_TOKENIZER),
        "pooling": dict(MODEL_POOLING),
        "normalization": dict(MODEL_NORMALIZATION),
        "device": "cpu",
        "model_status": "approved-for-local-development",
        "acquisition_method": "pinned HF resolve URLs",
        "cache_policy": "local-only under data/raw/models; never committed",
        "redistribution_status": "not-redistributed",
        "expected_files": [
            {"path": "config.json", "size": 743, "sha256": "a" * 64},
        ],
        "verification_timestamp": "2026-08-11T00:00:00Z",
    }
    data.update(overrides)
    return data


def _write_manifest(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "model-manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_cache(tmp_path: Path, files: dict[str, str]) -> Path:
    cache_dir = tmp_path / "cache"
    for rel, text in files.items():
        path = cache_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return cache_dir


def test_committed_model_manifest_parses_and_audits():
    manifest = load_embedding_model_manifest(COMMITTED_MANIFEST)
    audit_embedding_model_manifest(manifest)
    assert manifest.model_id == MODEL_ID
    assert manifest.revision == MODEL_REVISION
    assert manifest.dimension == MODEL_DIMENSION
    assert manifest.max_input_tokens == MODEL_MAX_INPUT_TOKENS
    assert manifest.license.name == MODEL_LICENSE_NAME
    assert manifest.expected_files


def test_missing_manifest_file_raises(tmp_path):
    with pytest.raises(EmbeddingModelManifestError, match="manifest_file_missing"):
        load_embedding_model_manifest(tmp_path / "nope.json")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(EmbeddingModelManifestError, match="malformed_manifest"):
        load_embedding_model_manifest(path)


def test_unsupported_manifest_version_rejected(tmp_path):
    path = _write_manifest(tmp_path, _manifest_data(manifest_version="9.9"))
    with pytest.raises(
        EmbeddingModelManifestError, match="unsupported_manifest_version"
    ):
        load_embedding_model_manifest(path)


def test_missing_required_field_rejected(tmp_path):
    data = _manifest_data()
    del data["cache_policy"]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="missing_required_field"):
        load_embedding_model_manifest(path)


def test_invalid_revision_rejected(tmp_path):
    path = _write_manifest(tmp_path, _manifest_data(revision="not-a-commit"))
    with pytest.raises(EmbeddingModelManifestError, match="invalid_revision"):
        load_embedding_model_manifest(path)


def test_file_path_escape_rejected(tmp_path):
    data = _manifest_data()
    data["expected_files"] = [{"path": "../escape.bin", "size": 1, "sha256": "a" * 64}]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="expected_files"):
        load_embedding_model_manifest(path)


def test_backslash_traversal_rejected(tmp_path):
    data = _manifest_data()
    data["expected_files"] = [
        {"path": r"..\..\escape.bin", "size": 1, "sha256": "a" * 64}
    ]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="expected_files"):
        load_embedding_model_manifest(path)


def test_mixed_slash_traversal_rejected(tmp_path):
    data = _manifest_data()
    data["expected_files"] = [
        {"path": "sub/../../escape.bin", "size": 1, "sha256": "a" * 64}
    ]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="expected_files"):
        load_embedding_model_manifest(path)


def test_leading_backslash_absolute_path_rejected(tmp_path):
    data = _manifest_data()
    data["expected_files"] = [
        {"path": "\\abs\\escape.bin", "size": 1, "sha256": "a" * 64}
    ]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="expected_files"):
        load_embedding_model_manifest(path)


def test_invalid_sha256_rejected(tmp_path):
    data = _manifest_data()
    data["expected_files"] = [{"path": "config.json", "size": 743, "sha256": "xyz"}]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(EmbeddingModelManifestError, match="sha256"):
        load_embedding_model_manifest(path)


def test_audit_rejects_other_model(tmp_path):
    data = _manifest_data(model_id="BAAI/other-model")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="unapproved_model"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_model_version_mismatch(tmp_path):
    data = _manifest_data(model_version="9.9")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="model_version_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_publisher_mismatch(tmp_path):
    data = _manifest_data(publisher="Some Other Org")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="publisher_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_official_url_mismatch(tmp_path):
    data = _manifest_data(official_url="https://example.org/not-the-model")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="official_url_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_model_card_url_mismatch(tmp_path):
    data = _manifest_data(model_card_url="https://example.org/fake-card")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="model_card_url_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_device_mismatch(tmp_path):
    data = _manifest_data(device="cuda")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="device_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_architecture_mismatch(tmp_path):
    data = _manifest_data()
    data["architecture"] = dict(MODEL_ARCHITECTURE, hidden_size=768)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="architecture_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_tokenizer_mismatch(tmp_path):
    data = _manifest_data()
    data["tokenizer"] = dict(MODEL_TOKENIZER, lowercase=False)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="tokenizer_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_pooling_mismatch(tmp_path):
    data = _manifest_data()
    data["pooling"] = dict(MODEL_POOLING, method="mean")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="pooling_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_normalization_mismatch(tmp_path):
    data = _manifest_data()
    data["normalization"] = dict(MODEL_NORMALIZATION, l2_normalize=False)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="normalization_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_revision_mismatch(tmp_path):
    data = _manifest_data(revision="f" * 40, revision_tag="f" * 40)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="revision_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_license_mismatch(tmp_path):
    data = _manifest_data()
    data["license"] = {
        "name": "Apache-2.0",
        "version": "2.0",
        "url": "https://example.org",
        "evidence_excerpt": "wrong",
    }
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="license_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_dimension_mismatch(tmp_path):
    data = _manifest_data(dimension=768)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="dimension_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_max_input_mismatch(tmp_path):
    data = _manifest_data(max_input_tokens=256)
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="max_input_mismatch"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_blocked_status(tmp_path):
    data = _manifest_data(model_status="blocked")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="model_status_blocked"):
        audit_embedding_model_manifest(manifest)


def test_audit_rejects_redistribution(tmp_path):
    data = _manifest_data(redistribution_status="approved-for-redistribution")
    manifest = parse_embedding_model_manifest(data)
    with pytest.raises(EmbeddingModelManifestError, match="redistribution_mismatch"):
        audit_embedding_model_manifest(manifest)


def _verified_manifest_data(files: dict[str, str]) -> dict[str, object]:
    entries = [
        {
            "path": rel,
            "size": len(text.encode("utf-8")),
            "sha256": sha256_bytes(text.encode("utf-8")),
        }
        for rel, text in files.items()
    ]
    return _manifest_data(expected_files=entries)


def test_verify_files_byte_exact(tmp_path):
    files = {"config.json": '{"hidden_size": 384}', "model.safetensors": "fake-weights"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    audit_embedding_model_manifest(manifest)
    verify_embedding_model_files(manifest, cache_dir)


def test_verify_missing_cache_dir(tmp_path):
    files = {"config.json": "x"}
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    with pytest.raises(EmbeddingModelManifestError, match="cache_dir_missing"):
        verify_embedding_model_files(manifest, tmp_path / "absent")


def test_verify_missing_file(tmp_path):
    files = {"config.json": "x"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    cache_dir.joinpath("config.json").unlink()
    with pytest.raises(EmbeddingModelManifestError, match="file_missing"):
        verify_embedding_model_files(manifest, cache_dir)


def test_verify_size_mismatch(tmp_path):
    files = {"config.json": "x"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    cache_dir.joinpath("config.json").write_text("tampered-longer", encoding="utf-8")
    with pytest.raises(EmbeddingModelManifestError, match="file_size_mismatch"):
        verify_embedding_model_files(manifest, cache_dir)


def test_verify_checksum_mismatch(tmp_path):
    files = {"config.json": "x"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    cache_dir.joinpath("config.json").write_text("y", encoding="utf-8")
    with pytest.raises(EmbeddingModelManifestError, match="file_checksum_mismatch"):
        verify_embedding_model_files(manifest, cache_dir)


def test_verify_extra_cache_file_rejected(tmp_path):
    files = {"config.json": "x", "model.safetensors": "fake-weights"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    (cache_dir / "sneaky.bin").write_bytes(b"undeclared")
    with pytest.raises(EmbeddingModelManifestError, match="extra_cache_file"):
        verify_embedding_model_files(manifest, cache_dir)


def test_verify_extra_nested_cache_file_rejected(tmp_path):
    files = {"config.json": "x", "model.safetensors": "fake-weights"}
    cache_dir = _write_cache(tmp_path, files)
    manifest = parse_embedding_model_manifest(_verified_manifest_data(files))
    nested = cache_dir / "1_Pooling"
    nested.mkdir()
    (nested / "config.json").write_text(
        '{"pooling_mode_cls_token": true}', encoding="utf-8"
    )
    with pytest.raises(EmbeddingModelManifestError, match="extra_cache_file"):
        verify_embedding_model_files(manifest, cache_dir)
