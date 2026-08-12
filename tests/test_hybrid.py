from __future__ import annotations

from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    FusionConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from evidence_grounded_rag.retrieval.models import RetrievalResult, ScoredChunk

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE_MANIFEST)


@pytest.fixture(scope="module")
def retriever(corpus):
    return HybridRetriever(corpus)


def test_fusion_formula_is_committed_reciprocal_rank():
    sparse = RetrievalResult(
        chunks=(
            ScoredChunk("chunk-a", "doc-a", 1, 1.0),
            ScoredChunk("chunk-b", "doc-b", 2, 0.5),
        ),
        config={},
        versions={"sparse": "v1"},
    )
    dense = RetrievalResult(
        chunks=(
            ScoredChunk("chunk-b", "doc-b", 1, 1.0),
            ScoredChunk("chunk-c", "doc-c", 2, 0.8),
        ),
        config={},
        versions={"dense": "v1"},
    )
    fused = reciprocal_rank_fusion(sparse, dense)
    assert fused.config["k"] == 60
    assert fused.config["name"] == "reciprocal_rank_fusion"
    scores = {item.chunk_id: item.score for item in fused.chunks}
    assert scores["chunk-a"] == pytest.approx(1 / 61)
    assert scores["chunk-b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["chunk-c"] == pytest.approx(1 / 62)
    assert [item.chunk_id for item in fused.chunks] == ["chunk-b", "chunk-a", "chunk-c"]


def test_fusion_deduplicates_and_ties_break_by_chunk_id():
    sparse = RetrievalResult(
        chunks=(ScoredChunk("z-chunk", "doc", 1, 1.0),),
        config={},
        versions={},
    )
    dense = RetrievalResult(
        chunks=(ScoredChunk("a-chunk", "doc", 1, 1.0),),
        config={},
        versions={},
    )
    fused = reciprocal_rank_fusion(sparse, dense)
    ids = [item.chunk_id for item in fused.chunks]
    assert len(ids) == 2
    assert ids == ["a-chunk", "z-chunk"]


def test_hybrid_fusion_config_is_fixed():
    assert FusionConfig().k == 60
    assert FusionConfig().to_dict() == {"name": "reciprocal_rank_fusion", "k": 60}


def test_hybrid_result_trace_is_complete(retriever):
    result = retriever.search("free trail access", top_k=5)
    trace = result.trace
    assert trace.mode == MODE_HYBRID
    assert trace.query == "free trail access"
    assert trace.sparse.chunks
    assert trace.dense.chunks
    assert trace.fused.chunks
    assert trace.versions["bm25"] == "okapi-bm25-v1"
    assert trace.versions["embedding"] == "deterministic-hashed-bow-v1"
    assert trace.versions["corpus_version"] == "1.0.0"
    assert trace.config["fusion_k"] == 60
    assert trace.config["top_k"] == 5
    serialized = trace.to_dict()
    assert "sparse" in serialized and "dense" in serialized and "fused" in serialized


def test_hybrid_ranking_combines_both_systems(retriever):
    result = retriever.search("free trail access", top_k=5)
    top = result.fused.chunks[0]
    assert top.chunk_id == "doc-east-river-trail::0"


def test_sparse_mode_passthrough(retriever):
    result = retriever.search("permit", top_k=3, mode=MODE_SPARSE)
    assert result.fused.config["retriever"] == "bm25"
    assert result.trace.dense.chunks


def test_dense_mode_passthrough(retriever):
    result = retriever.search("permit", top_k=3, mode=MODE_DENSE)
    assert result.fused.config["retriever"] == "dense"
    assert result.trace.sparse.chunks


def test_invalid_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("permit", mode="bogus")


def test_min_score_threshold_filters_fused(retriever):
    without_threshold = retriever.search("permit", top_k=10)
    thresholded = retriever.search("permit", top_k=10, min_score=0.032)
    assert thresholded.fused.chunks
    assert len(thresholded.fused.chunks) < len(without_threshold.fused.chunks)
    assert all(item.score >= 0.032 for item in thresholded.fused.chunks)


def test_top_k_zero_returns_empty(retriever):
    result = retriever.search("permit", top_k=0)
    assert result.fused.empty is True


def test_hybrid_is_deterministic(retriever):
    first = [
        item.chunk_id for item in retriever.search("library open saturday").fused.chunks
    ]
    second = [
        item.chunk_id for item in retriever.search("library open saturday").fused.chunks
    ]
    assert first == second
