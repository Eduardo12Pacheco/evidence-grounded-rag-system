from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_grounded_rag.embeddings import (
    EmbeddingModelManifestError,
    SemanticModelUnavailableError,
)
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
)
from evidence_grounded_rag.evaluation.semantic_materialize import (
    HASH_MODE_LABEL,
    HYBRID_MODE_LABEL,
    SEMANTIC_MODE_LABEL,
    SPARSE_MODE_LABEL,
    semantic_evaluate,
)
from evidence_grounded_rag.evaluation.semantic_materialize import (
    main as semantic_main,
)
from tests.fake_provider import FakeSemanticProvider

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "corpus-source-v1.0.0.json"
QUERY_SET = ROOT / "data" / "queries" / "python-docs-real-v1.0.0.json"
RAW_DIR = ROOT / "data" / "raw" / "python-docs-3.12.13"
PRIOR_REPORT = ROOT / "data" / "evaluations" / "python-docs-real-evaluation-v1.0.0.json"
MODULE = (
    ROOT / "src" / "evidence_grounded_rag" / "evaluation" / "semantic_materialize.py"
)

REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

pytestmark = pytest.mark.skipif(
    not RAW_DIR.is_dir(),
    reason="ignored real corpus capture is absent (run scripts/acquire_corpus.py first)",
)


def _model_manifest_data(**overrides) -> dict[str, object]:
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


@pytest.fixture()
def model_manifest(tmp_path):
    path = tmp_path / "embedding-model.json"
    path.write_text(json.dumps(_model_manifest_data()), encoding="utf-8")
    return path


@pytest.fixture()
def cache_dir(tmp_path):
    return tmp_path / "cache"


def _read(path: Path) -> bytes:
    return path.read_bytes()


def test_semantic_evaluate_reports_four_separate_modes(
    tmp_path, model_manifest, cache_dir
):
    report, out_path = semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    data = report.data
    assert data["report_type"] == "real-local-semantic-evaluation"
    assert data["report_version"] == "1.0.0"
    assert set(data["metrics"].keys()) == {"sparse", "hash", "semantic", "hybrid"}
    assert set(data["per_query"].keys()) == {"sparse", "hash", "semantic", "hybrid"}
    assert set(data["ranking_stability"].keys()) == {
        "sparse",
        "hash",
        "semantic",
        "hybrid",
    }
    assert data["labels"]["sparse_mode"] == SPARSE_MODE_LABEL
    assert data["labels"]["hash_mode"] == HASH_MODE_LABEL
    assert data["labels"]["semantic_mode"] == SEMANTIC_MODE_LABEL
    assert data["labels"]["hybrid_mode"] == HYBRID_MODE_LABEL
    assert out_path.is_file()


def test_hash_mode_label_is_exact_and_semantic_never_hash(
    tmp_path, model_manifest, cache_dir
):
    report, _ = semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    assert report.data["labels"]["hash_mode"] == (
        "Deterministic hash embedding diagnostic — not semantic retrieval"
    )
    assert "no_hash_fallback" in report.data["labels"]


def test_semantic_evaluate_leaves_prior_reports_untouched(
    tmp_path, model_manifest, cache_dir
):
    before_report = _read(PRIOR_REPORT)
    before_queries = _read(QUERY_SET)
    before_corpus = _read(MANIFEST)
    semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    assert _read(PRIOR_REPORT) == before_report
    assert _read(QUERY_SET) == before_queries
    assert _read(MANIFEST) == before_corpus


def test_semantic_report_model_facts(tmp_path, model_manifest, cache_dir):
    report, _ = semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    model = report.data["model"]
    assert model["model_id"] == "BAAI/bge-small-en-v1.5"
    assert model["revision"] == REVISION
    assert model["license"]["name"] == "MIT License"
    assert model["dimension"] == 384
    assert "cache_path" in model
    assert model["local_files_only"] is True
    assert "max_input_tokens" in model


def test_semantic_report_schema_and_rankings(tmp_path, model_manifest, cache_dir):
    report, _ = semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    data = report.data
    for mode in ("sparse", "hash", "semantic", "hybrid"):
        overall = data["metrics"][mode]["overall"]
        for key in (
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@5",
            "no_result_rate",
        ):
            assert key in overall
        stability = data["ranking_stability"][mode]
        assert stability["num_queries"] == 18
        assert 0 <= stability["fraction_identical"] <= 1
        for record in data["per_query"][mode].values():
            assert "retrieved_chunk_ids" in record
            assert "top_k_hit" in record
            assert "first_hit_rank" in record
            assert "retrieval_empty" in record
    assert data["query_set"]["num_queries"] == 18
    assert data["corpus"]["chunk_count"] == 70


def test_semantic_report_has_digest_and_stores_no_chunk_text(
    tmp_path, model_manifest, cache_dir, real_corpus
):
    report, _ = semantic_evaluate(
        MANIFEST,
        QUERY_SET,
        RAW_DIR,
        model_manifest_path=model_manifest,
        cache_dir=cache_dir,
        out_path=tmp_path / "report.json",
        provider=FakeSemanticProvider(dimension=8),
    )
    payload = {
        k: v
        for k, v in report.data.items()
        if k not in ("timestamp_utc", "deterministic_digest", "runtime_observations")
    }
    import hashlib

    blob = (json.dumps(report.data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    for chunk in real_corpus.chunks():
        assert chunk.text not in blob.decode("utf-8")
    expected = hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    assert report.data["deterministic_digest"] == expected


def test_missing_model_fails_without_hash_fallback(tmp_path, model_manifest, cache_dir):
    with pytest.raises(
        SemanticModelUnavailableError, match="cache directory not found"
    ):
        semantic_evaluate(
            MANIFEST,
            QUERY_SET,
            RAW_DIR,
            model_manifest_path=model_manifest,
            cache_dir=cache_dir,
            out_path=tmp_path / "report.json",
            provider=None,
        )
    assert not (tmp_path / "report.json").exists()


def test_invalid_model_manifest_refused(tmp_path, model_manifest, cache_dir):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(_model_manifest_data(revision="f" * 40, revision_tag="f" * 40)),
        encoding="utf-8",
    )
    with pytest.raises(EmbeddingModelManifestError, match="revision_mismatch"):
        semantic_evaluate(
            MANIFEST,
            QUERY_SET,
            RAW_DIR,
            model_manifest_path=bad,
            cache_dir=cache_dir,
            out_path=tmp_path / "report.json",
            provider=None,
        )


def test_cli_fails_clearly_on_invalid_model_manifest(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(_model_manifest_data(revision="f" * 40, revision_tag="f" * 40)),
        encoding="utf-8",
    )
    code = semantic_main(
        [
            "--manifest",
            str(MANIFEST),
            "--queries",
            str(QUERY_SET),
            "--raw-dir",
            str(RAW_DIR),
            "--model-manifest",
            str(bad),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "report.json").exists()


def test_semantic_materialize_module_is_network_free():
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in (
        "urllib",
        "requests",
        "httpx",
        "socket.",
        "aiohttp",
        "hf_hub_download",
    ):
        assert forbidden not in source, (
            f"semantic_materialize must not use {forbidden!r}"
        )


@pytest.fixture(scope="module")
def real_corpus():
    from evidence_grounded_rag.corpus import load_source_manifest
    from evidence_grounded_rag.corpus.real_local import build_real_local_corpus

    manifest = load_source_manifest(MANIFEST)
    return build_real_local_corpus(manifest, RAW_DIR)
