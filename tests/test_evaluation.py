"""Tests for the evaluation layer: metric math, query set, materialize."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_grounded_rag.evaluation import (
    QUERY_CATEGORIES,
    evaluate_answers,
    load_query_set,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from evidence_grounded_rag.evaluation.materialize import materialize
from evidence_grounded_rag.pipeline import EvidenceRagPipeline, PipelineConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"
QUERY_SET = ROOT / "data" / "fixtures" / "queries" / "synthetic_queries_v1.json"


def test_recall_metric_formula():
    assert recall_at_k(["a", "b", "c"], ["a"], 1) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], 1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 2) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], [], 5) == 0.0


def test_mrr_formula():
    assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0
    assert mean_reciprocal_rank(["a", "b"], ["a", "b"]) == 1.0


def test_ndcg_formula():
    # DCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309 = 2.6309; IDCG = same → 1.0
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)
    # Reversed order halves the first term contribution.
    score = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2)
    assert 0.0 < score < 1.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0


def test_query_set_covers_all_categories():
    query_set = load_query_set(QUERY_SET)
    assert query_set.query_set_id == "synthetic-queries-v1"
    assert query_set.query_set_version == "1.0.0"
    assert query_set.synthetic is True
    assert query_set.categories_present() == QUERY_CATEGORIES


def test_answered_queries_require_relevance_judgments(tmp_path):
    bad = {
        "query_set_id": "bad",
        "query_set_version": "1.0.0",
        "corpus_id": "c",
        "corpus_version": "1.0.0",
        "synthetic": True,
        "queries": [
            {
                "query_id": "q-001",
                "category": "answerable",
                "question": "?",
                "answerable": True,
                "expected_state": "answered",
                "expected_reason": None,
                "relevance": [],
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="must declare relevance"):
        load_query_set(path)


def _tiny_pipeline_and_query_set():
    pipeline = EvidenceRagPipeline(FIXTURE_MANIFEST, config=PipelineConfig(top_k=10))
    query_set = load_query_set(QUERY_SET)
    return pipeline, query_set


def test_evaluation_metrics_meet_committed_targets():
    pipeline, query_set = _tiny_pipeline_and_query_set()
    records = [pipeline.answer(q.question, top_k=10) for q in query_set.queries]
    report = evaluate_answers(query_set, records)
    metrics = report.metrics
    assert metrics["answerable_coverage"] == pytest.approx(1.0)
    assert metrics["abstention_precision"] == pytest.approx(1.0)
    assert metrics["abstention_recall"] == pytest.approx(1.0)
    assert metrics["unsupported_claim_rate"] == pytest.approx(0.0)
    assert metrics["citation_precision"] == pytest.approx(1.0)
    assert metrics["citation_completeness"] == pytest.approx(1.0)
    assert 0.0 < metrics["abstention_rate"] < 1.0
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_evaluation_is_deterministic():
    pipeline, query_set = _tiny_pipeline_and_query_set()
    first = evaluate_answers(
        query_set, [pipeline.answer(q.question, top_k=10) for q in query_set.queries]
    ).metrics
    second = evaluate_answers(
        query_set, [pipeline.answer(q.question, top_k=10) for q in query_set.queries]
    ).metrics
    assert first == second


def test_materialize_writes_labeled_report(tmp_path):
    _, out_path = materialize(
        FIXTURE_MANIFEST,
        QUERY_SET,
        tmp_path / "report.json",
        top_k=10,
        mode="hybrid",
    )
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert raw["labels"]["synthetic_evaluation"] is True
    assert raw["labels"]["corpus_provenance"] == "project-generated-synthetic"
    assert raw["labels"]["real_corpus_gate_blocked"] is True
    assert raw["versions"]["corpus_version"] == "1.0.0"
    assert raw["versions"]["query_set_version"] == "1.0.0"
    assert raw["generation"]["is_llm_generated"] is False
    assert len(raw["limitations"]) >= 3
    assert "timestamp_utc" in raw
    assert raw["metrics"]["answerable_coverage"] == 1.0


def test_materialize_is_deterministic_in_metrics(tmp_path):
    first, _ = materialize(
        FIXTURE_MANIFEST, QUERY_SET, tmp_path / "r1.json", top_k=10
    )
    second, _ = materialize(
        FIXTURE_MANIFEST, QUERY_SET, tmp_path / "r2.json", top_k=10
    )
    assert first.metrics == second.metrics
