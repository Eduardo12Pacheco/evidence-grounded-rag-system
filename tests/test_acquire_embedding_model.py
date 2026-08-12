from __future__ import annotations

import importlib.util
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
    REQUIRED_FILES,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
    verify_embedding_model_files,
)

ROOT = Path(__file__).resolve().parents[1]

_FILES: dict[str, bytes] = {
    "config.json": b'{"hidden_size": 384}',
    "model.safetensors": b"fake-weights",
}

_BOOTSTRAP_FILES: dict[str, bytes] = {
    rel: f"content:{rel}".encode() for rel in REQUIRED_FILES
}


def _load_acquire_module():
    spec = importlib.util.spec_from_file_location(
        "acquire_embedding_model", ROOT / "scripts" / "acquire_embedding_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


acquire = _load_acquire_module()


def _manifest_doc(expected_files: list[dict[str, object]]) -> dict[str, object]:
    return {
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
        "expected_files": expected_files,
        "verification_timestamp": "2026-08-11T00:00:00Z",
    }


def _entries(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {"path": rel, "size": len(data), "sha256": sha256_bytes(data)}
        for rel, data in files.items()
    ]


def _write_trusted_manifest(tmp_path: Path, files: dict[str, bytes]) -> Path:
    path = tmp_path / "embedding-model.json"
    path.write_text(json.dumps(_manifest_doc(_entries(files))), encoding="utf-8")
    return path


def _cache_snapshot(cache_dir: Path) -> dict[str, bytes]:
    return {
        rel.relative_to(cache_dir).as_posix(): rel.read_bytes()
        for rel in cache_dir.rglob("*")
        if rel.is_file()
    }


@pytest.fixture()
def fake_download(monkeypatch):
    def install(source: dict[str, bytes]):
        def _download(model_id: str, revision: str, rel_path: str) -> bytes:
            try:
                return source[rel_path]
            except KeyError as exc:
                raise AssertionError(
                    f"download asked for undeclared path {rel_path!r}"
                ) from exc

        monkeypatch.setattr(acquire, "_download_file_bytes", _download)
        return _download

    return install


def test_normal_acquire_uses_committed_manifest_and_never_rewrites_it(
    tmp_path, fake_download
):
    manifest_path = _write_trusted_manifest(tmp_path, _FILES)
    before = manifest_path.read_bytes()
    fake_download(dict(_FILES))
    cache_dir = tmp_path / "cache"

    code = acquire.main(
        ["--manifest", str(manifest_path), "--cache-dir", str(cache_dir)]
    )

    assert code == 0
    assert manifest_path.read_bytes() == before, (
        "committed manifest must not be rewritten"
    )
    assert _cache_snapshot(cache_dir) == _FILES
    loaded = load_embedding_model_manifest(manifest_path)
    audit_embedding_model_manifest(loaded)
    verify_embedding_model_files(loaded, cache_dir)


def test_normal_acquire_missing_manifest_fails_clearly(tmp_path):
    missing = tmp_path / "nope.json"
    cache_dir = tmp_path / "cache"
    code = acquire.main(["--manifest", str(missing), "--cache-dir", str(cache_dir)])
    assert code == 2
    assert not cache_dir.exists()


def test_normal_acquire_tampered_download_rejected_manifest_unchanged(
    tmp_path, fake_download
):
    manifest_path = _write_trusted_manifest(tmp_path, _FILES)
    before = manifest_path.read_bytes()
    tampered = dict(_FILES)
    tampered["config.json"] = b"tampered-different-bytes"
    fake_download(tampered)
    cache_dir = tmp_path / "cache"

    code = acquire.main(
        ["--manifest", str(manifest_path), "--cache-dir", str(cache_dir)]
    )

    assert code == 1
    assert manifest_path.read_bytes() == before, (
        "committed manifest must not be rewritten"
    )
    assert not cache_dir.exists() or _cache_snapshot(cache_dir) != _FILES


def test_normal_acquire_staging_failure_leaves_existing_cache_untouched(
    tmp_path, fake_download, monkeypatch
):
    manifest_path = _write_trusted_manifest(tmp_path, _FILES)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for rel, data in _FILES.items():
        out = cache_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    before = _cache_snapshot(cache_dir)

    def _boom(model_id, revision, rel_path):
        raise RuntimeError("network down")

    monkeypatch.setattr(acquire, "_download_file_bytes", _boom)

    code = acquire.main(
        ["--manifest", str(manifest_path), "--cache-dir", str(cache_dir)]
    )

    assert code == 1
    assert _cache_snapshot(cache_dir) == before, (
        "existing verified cache must be untouched"
    )


def test_bootstrap_refused_when_manifest_exists(tmp_path, fake_download):
    manifest_path = _write_trusted_manifest(tmp_path, _FILES)
    cache_dir = tmp_path / "cache"
    code = acquire.main(
        ["--bootstrap", "--manifest", str(manifest_path), "--cache-dir", str(cache_dir)]
    )
    assert code == 2
    assert not cache_dir.exists()


def test_bootstrap_writes_new_manifest_only_when_manifest_absent(
    tmp_path, fake_download, capsys
):
    manifest_path = tmp_path / "embedding-model.json"
    assert not manifest_path.exists()
    fake_download(dict(_BOOTSTRAP_FILES))
    cache_dir = tmp_path / "cache"

    code = acquire.main(
        ["--bootstrap", "--manifest", str(manifest_path), "--cache-dir", str(cache_dir)]
    )

    assert code == 0
    assert manifest_path.exists()
    assert "BOOTSTRAP" in capsys.readouterr().out
    loaded = load_embedding_model_manifest(manifest_path)
    audit_embedding_model_manifest(loaded)
    verify_embedding_model_files(loaded, cache_dir)
    assert _cache_snapshot(cache_dir) == _BOOTSTRAP_FILES
