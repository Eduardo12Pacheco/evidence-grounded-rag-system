from __future__ import annotations

from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.retrieval import BM25
from evidence_grounded_rag.text.tokenizer import key_tokens, stopwords, tokenize

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE_MANIFEST)


@pytest.fixture(scope="module")
def bm25(corpus):
    return BM25(corpus.chunks())


def test_tokenizer_is_deterministic_and_aggressive():
    tokens = tokenize("  The Central Library, is open!  ")
    assert tokens == tokenize("the central library is open")
    assert tokens == ["the", "central", "library", "is", "open"]
    assert tokenize("costs") == ["costs"]
    assert tokenize("cost") == ["cost"]


def test_no_stemming_is_a_documented_limit():
    assert tokenize("reviewed") != tokenize("review")


def test_key_tokens_exclude_committed_stopwords():
    assert key_tokens("How much is the cost of a public event permit?") == [
        "cost",
        "public",
        "event",
        "permit",
    ]
    assert "and" in stopwords()
    assert "must" in stopwords()


def test_empty_query_returns_empty_result(bm25):
    result = bm25.search("   ")
    assert result.empty is True
    assert result.chunks == ()


def test_empty_corpus_returns_empty_result():
    empty = BM25([])
    result = empty.search("anything")
    assert result.empty is True
    assert result.chunks == ()


def test_top_k_zero_or_negative_returns_empty(bm25):
    assert bm25.search("permit", top_k=0).empty is True
    assert bm25.search("permit", top_k=-1).empty is True


def test_top_k_greater_than_corpus_returns_everything(bm25):
    result = bm25.search("the permit is open", top_k=10_000)
    assert not result.empty
    assert len(result.chunks) == len(bm25._chunks)


def test_chunk_with_more_query_terms_ranks_higher(bm25):
    result = bm25.search("public event permit", top_k=10)
    top = result.chunks[0]
    assert top.chunk_id == "doc-aurora-park-permits::0"


def test_ranking_is_deterministic(bm25):
    first = [item.chunk_id for item in bm25.search("permit application review").chunks]
    second = [item.chunk_id for item in bm25.search("permit application review").chunks]
    assert first == second


def test_result_metadata_records_committed_config(bm25):
    result = bm25.search("permit", top_k=3)
    assert result.config["k1"] == 1.5
    assert result.config["b"] == 0.75
    assert result.config["idf_mode"] == "robertson-sparck-jones"
    assert result.versions["bm25"] == "okapi-bm25-v1"
    assert len(result.chunks) == 3


def test_no_result_terms_yield_empty(bm25):
    result = bm25.search("zzzqqqxxx", top_k=5)
    assert result.empty is True
