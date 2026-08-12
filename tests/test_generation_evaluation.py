"""Tests for the separate real-local generation evaluation.

Fully offline: manifests, tampered copies, and raw-file copies are created in
temporary directories; the only real corpus content used comes from the
existing gitignored ``data/raw/python-docs-3.12.13/`` capture. Coverage:

- versioned answer-evaluation fixture loading and cross-validation;
- separate labeled offline report with generation metrics and per-query fields;
- deterministic content across two runs (timestamp/runtime excluded);
- no claim/chunk text stored in the report;
- live generation explicitly unmeasured / no provider selected;
- offline generation produces zero invalid/provider errors;
- network-free module.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_source_manifest
from evidence_grounded_rag.corpus.real_local import build_real_local_corpus
from evidence_grounded_rag.evaluation.generation_materialize import (
    GENERATION_LABEL,
    LIVE_GENERATION_LABEL,
    REPORT_TYPE,
    generation_evaluate,
    load_answer_evaluation_fixture,
)
from evidence_grounded_rag.evaluation.real_queries import load_real_query_set
from evidence_grounded_rag.versions import (
    ANSWER_EVALUATION_FIXTURE_VERSION,
    CLAIM_SUPPORT_VERSION,
    GENERATION_EVALUATION_VERSION,
    GENERATION_PROMPT_VERSION,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "corpus-source-v1.0.0.json"
QUERY_SET = ROOT / "data" / "evaluation" / "queries" / "python-docs-real-v1.0.0.json"
FIXTURE = ROOT / "data" / "fixtures" / "answer-evaluation-v1.0.0.json"
RAW_DIR = ROOT / "data" / "raw" / "python-docs-3.12.13"
GEN_MODULE = ROOT / "src" / "evidence_grounded_rag" / "evaluation" / "generation_materialize.py"

pytestmark = pytest.mark.skipif(
    not RAW_DIR.is_dir(),
    reason="ignored real corpus capture is absent (run scripts/acquire_corpus.py first)",
)


def _json_bytes(data: dict[str, object]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _deterministic_digest(data: dict[str, object]) -> str:
    import hashlib

    payload = {
        k: v
        for k, v in data.items()
        if k not in ("timestamp_utc", "deterministic_digest", "runtime_seconds")
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


@pytest.fixture()
def raw_copy(tmp_path):
    target = tmp_path / "raw"
    shutil.copytree(RAW_DIR, target)
    return target


@pytest.fixture()
def real_corpus():
    manifest = load_source_manifest(MANIFEST)
    return build_real_local_corpus(manifest, RAW_DIR)


# --------------------------------------------------------------------------- #
# fixture cross-validation
# --------------------------------------------------------------------------- #


def test_fixture_references_unchanged_18_real_queries(real_corpus):
    query_set = load_real_query_set(QUERY_SET, real_corpus, manifest_id=MANIFEST.stem)
    raw = load_answer_evaluation_fixture(FIXTURE, real_corpus, query_set)
    assert raw["fixture_version"] == ANSWER_EVALUATION_FIXTURE_VERSION
    assert len(raw["queries"]) == 18
    assert {q["query_id"] for q in raw["queries"]} == {
        q.query_id for q in query_set.queries
    }
    assert len({q["query_id"] for q in raw["queries"]}) == 18


def test_fixture_rejects_modified_query_ids(tmp_path, real_corpus):
    query_set = load_real_query_set(QUERY_SET, real_corpus, manifest_id=MANIFEST.stem)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["queries"] = raw["queries"][:17]
    path = tmp_path / "short.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the unchanged real query set"):
        load_answer_evaluation_fixture(path, real_corpus, query_set)


def test_fixture_rejects_changed_gold_chunks(tmp_path, real_corpus):
    query_set = load_real_query_set(QUERY_SET, real_corpus, manifest_id=MANIFEST.stem)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["queries"][0]["gold_chunk_ids"] = ["doc-tutorial-introduction.rst::2"]
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_answer_evaluation_fixture(path, real_corpus, query_set)


def test_fixture_rejects_unknown_chunk(tmp_path, real_corpus):
    query_set = load_real_query_set(QUERY_SET, real_corpus, manifest_id=MANIFEST.stem)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["queries"][0]["gold_chunk_ids"] = ["doc-does-not-exist::0"]
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown chunk"):
        load_answer_evaluation_fixture(path, real_corpus, query_set)


# --------------------------------------------------------------------------- #
# end-to-end generation evaluation
# --------------------------------------------------------------------------- #


def test_generation_evaluate_writes_separate_labeled_report(tmp_path, raw_copy):
    _, out_path = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert raw["report_type"] == REPORT_TYPE
    assert raw["report_version"] == GENERATION_EVALUATION_VERSION
    assert raw["corpus"]["corpus_id"] == "python-docs-3.12.13"
    assert raw["query_set"]["num_queries"] == 18
    assert raw["labels"]["generation"] == GENERATION_LABEL
    assert raw["labels"]["live_generation"] == LIVE_GENERATION_LABEL
    assert "not an LLM result" in raw["labels"]["generation"]
    assert raw["labels"]["no_live_claim"]
    assert raw["fixture"]["fixture_version"] == ANSWER_EVALUATION_FIXTURE_VERSION


def test_generation_report_metrics_and_configs(tmp_path, raw_copy):
    report, _ = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    data = report.data
    metrics = data["metrics"]
    assert "answer_coverage" in metrics
    assert "abstention_precision" in metrics
    assert "abstention_recall" in metrics
    assert "correct_abstention_count" in metrics
    assert "citation_precision" in metrics
    assert "citation_completeness" in metrics
    assert "citation_validation_failure_rate" in metrics
    assert metrics["claim_support"]["rule"] == CLAIM_SUPPORT_VERSION
    assert metrics["invalid_output_count"] == 0
    assert metrics["provider_error_count"] == 0
    configs = data["configs"]
    assert configs["generation"]["mode"] == "offline"
    assert configs["generation"]["is_llm_generated"] is False
    assert configs["generation"]["prompt_version"] == GENERATION_PROMPT_VERSION
    assert configs["generation"]["schema_version"] == STRUCTURED_ANSWER_SCHEMA_VERSION
    assert configs["claim_support_rule"] == CLAIM_SUPPORT_VERSION
    assert data["metrics"]["claim_support"]["total"] > 0


def test_generation_report_per_query_fields(tmp_path, raw_copy):
    report, _ = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    per_query = report.data["per_query"]
    assert set(per_query.keys()) == {f"real-q-{i:03d}" for i in range(1, 19)}
    for qid, record in per_query.items():
        for field in (
            "query_id", "case", "expected_status", "state", "abstention_reason",
            "answerable", "answered", "abstained", "correct_abstention",
            "retrieval_empty", "retrieved_chunk_ids", "claim_support",
            "claim_count", "citation_completeness", "validation", "generation",
            "provider_error", "invalid_output",
        ):
            assert field in record, f"{qid} missing {field}"
        assert record["generation"]["mode"] == "offline"
        assert record["generation"]["is_llm_generated"] is False
        assert record["generation"]["provider_id"] == "extractive-template"
        for support in record["claim_support"]:
            assert support["support"] in ("supported", "unsupported", "unknown")
        assert record["state"] in ("answered", "abstained", "conflicting")


def test_generation_report_deterministic_across_runs(tmp_path, raw_copy):
    _, first_path = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "r1.json",
    )
    _, second_path = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "r2.json",
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    p1 = {k: v for k, v in first.items() if k not in ("timestamp_utc", "deterministic_digest", "runtime_seconds")}
    p2 = {k: v for k, v in second.items() if k not in ("timestamp_utc", "deterministic_digest", "runtime_seconds")}
    assert _json_bytes(p1) == _json_bytes(p2)
    assert first["deterministic_digest"] == second["deterministic_digest"]
    assert first["deterministic_digest"] == _deterministic_digest(first)
    assert first["metrics"] == second["metrics"]
    assert first["per_query"] == second["per_query"]


def test_generation_report_stores_no_claim_or_chunk_text(tmp_path, raw_copy, real_corpus):
    report, _ = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    blob = _json_bytes(report.data).decode("utf-8")
    for chunk in real_corpus.chunks():
        assert chunk.text not in blob
        assert chunk.text[:200] not in blob
    for record in report.data["per_query"].values():
        assert "answer_text" not in record
        for claim in record["claims"]:
            assert "text" not in claim


def test_generation_report_live_explicitly_unmeasured(tmp_path, raw_copy):
    report, _ = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    limitations = report.data["limitations"]
    assert any("live generation is NOT measured" in item for item in limitations)
    assert any("not an LLM result" in item for item in limitations)


def test_generation_report_failures_and_support(tmp_path, raw_copy):
    report, _ = generation_evaluate(
        MANIFEST, QUERY_SET, raw_copy,
        fixture_path=FIXTURE, out_path=tmp_path / "gen.json",
    )
    data = report.data
    kinds = {failure["kind"] for failure in data["failures"]}
    assert "wrong_answer" in kinds
    assert "answerable_abstention" in kinds
    assert all(failure["kind"] not in ("provider_error", "invalid_output") for failure in data["failures"])
    for failure in data["failures"]:
        assert failure["mode"] == "hybrid"
    assert data["metrics"]["claim_support"]["supported"] > 0
    assert data["metrics"]["claim_support"]["support_rate"] > 0


# --------------------------------------------------------------------------- #
# network-free behavior
# --------------------------------------------------------------------------- #


def test_generation_materialize_module_is_network_free():
    source = GEN_MODULE.read_text(encoding="utf-8")
    for forbidden in ("urllib", "requests", "httpx", "socket.", "aiohttp"):
        assert forbidden not in source, f"generation_materialize must not use {forbidden!r}"
