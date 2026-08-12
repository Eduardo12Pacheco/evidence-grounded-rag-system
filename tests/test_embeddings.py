from __future__ import annotations

import math
from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.embeddings import (
    DenseIndex,
    DeterministicEmbedding,
    EmbeddingDimensionError,
    cosine_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE_MANIFEST)


def test_embedding_is_deterministic():
    provider = DeterministicEmbedding(dimension=64)
    first = provider.embed_one("The cost of a public event permit")
    second = provider.embed_one("The cost of a public event permit")
    assert first == second
    assert len(first) == 64


def test_embedding_is_l2_normalized():
    provider = DeterministicEmbedding(dimension=64)
    vector = provider.embed_one("some text tokens")
    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_embedding_never_claims_semantics():
    provider = DeterministicEmbedding()
    assert provider.semantic is False
    assert provider.mode == "offline"
    assert provider.model_id == "deterministic-hashed-bow-v1"
    info = provider.to_dict()
    assert info["semantic"] is False


def test_embedding_distinguishes_disjoint_text():
    provider = DeterministicEmbedding(dimension=1024)
    vector_a = provider.embed_one("permit code of conduct")
    vector_b = provider.embed_one("river trail flood season")
    assert cosine_similarity(vector_a, vector_b) < 1.0


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(EmbeddingDimensionError):
        cosine_similarity((0.1, 0.2), (0.1, 0.2, 0.3))


def test_cosine_similarity_identical_is_one():
    assert cosine_similarity((0.6, 0.8), (0.6, 0.8)) == pytest.approx(1.0)


def test_dense_index_empty_corpus_returns_empty():
    provider = DeterministicEmbedding()
    index = DenseIndex([], provider)
    assert index.size == 0
    result = index.search("anything")
    assert result.empty is True
    assert result.chunks == ()


def test_dense_index_ranks_exact_match_first(corpus):
    provider = DeterministicEmbedding()
    index = DenseIndex(corpus.chunks(), provider)
    exact_text = corpus.document_by_id("doc-central-library").chunks[1].text
    result = index.search(exact_text, top_k=3)
    assert result.chunks[0].chunk_id == "doc-central-library::1"
    assert result.chunks[0].score == pytest.approx(1.0)


def test_dense_index_top_k_and_metadata(corpus):
    provider = DeterministicEmbedding(dimension=128)
    index = DenseIndex(corpus.chunks(), provider)
    result = index.search("permit application", top_k=3)
    assert len(result.chunks) == 3
    assert result.config["metric"] == "cosine"
    assert result.config["dimension"] == 128
    assert result.config["semantic"] is False
    assert result.versions["embedding"] == "deterministic-hashed-bow-v1"


def test_dense_index_top_k_zero_returns_empty(corpus):
    index = DenseIndex(corpus.chunks(), DeterministicEmbedding())
    assert index.search("permit", top_k=0).empty is True


def test_dense_index_deterministic(corpus):
    provider = DeterministicEmbedding()
    index = DenseIndex(corpus.chunks(), provider)
    first = [item.chunk_id for item in index.search("free trail", top_k=5).chunks]
    second = [item.chunk_id for item in index.search("free trail", top_k=5).chunks]
    assert first == second
