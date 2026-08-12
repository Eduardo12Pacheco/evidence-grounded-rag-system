from __future__ import annotations

from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)
from evidence_grounded_rag.validation import CitationValidator

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE_MANIFEST)


@pytest.fixture(scope="module")
def validator(corpus):
    return CitationValidator(corpus)


def _chunk_record(corpus, chunk_id):
    chunk = corpus.chunk_by_id(chunk_id)
    assert chunk is not None
    return chunk


def _answered(citations, claims):
    return GeneratedAnswer(
        state=AnswerState.ANSWERED,
        answer_text="answer",
        claims=tuple(claims),
        citations=tuple(citations),
        provider_id="test",
        model_id="test",
        is_llm_generated=False,
    )


def _valid_answer(corpus):
    chunk = _chunk_record(corpus, "doc-aurora-park-permits::0")
    citation = Citation(
        citation_id="cit-1",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        corpus_version=chunk.corpus_version,
        excerpt=chunk.text,
        location=chunk.location,
    )
    claim = Claim(
        claim_id="claim-1",
        text=chunk.text,
        citation_chunk_ids=(chunk.chunk_id,),
    )
    return _answered([citation], [claim])


def test_valid_answer_passes(corpus, validator):
    answer = _valid_answer(corpus)
    result = validator.validate(
        answer, exposed_chunk_ids={answer.citations[0].chunk_id}
    )
    assert result.valid is True
    assert result.skipped is False
    assert result.issues == ()
    assert result.checked_citations == 1
    assert result.checked_claims == 1


def test_abstained_answer_is_skipped_not_verified(corpus, validator):
    answer = GeneratedAnswer(
        state=AnswerState.ABSTAINED,
        answer_text=None,
        abstention_reason=None,
        provider_id="test",
        model_id="test",
        is_llm_generated=False,
    )
    result = validator.validate(answer, exposed_chunk_ids=set())
    assert result.skipped is True


def test_unknown_chunk_fails(corpus, validator):
    answer = _valid_answer(corpus)
    forged = Citation(
        citation_id="cit-x",
        chunk_id="doc-never-exists::0",
        document_id="doc-never-exists",
        corpus_version="1.0.0",
        excerpt="whatever",
        location="synthetic://synthetic-fixtures/doc-never-exists/chunk/0",
    )
    answer = _answered([forged], answer.claims)
    result = validator.validate(answer, exposed_chunk_ids={forged.chunk_id})
    assert result.valid is False
    assert "citation_not_found" in result.issue_codes


def test_unexposed_citation_fails(corpus, validator):
    answer = _valid_answer(corpus)
    result = validator.validate(answer, exposed_chunk_ids=set())
    assert result.valid is False
    assert "citation_not_exposed" in result.issue_codes


def test_corpus_version_mismatch_fails(corpus, validator):
    answer = _valid_answer(corpus)
    stale = _replace(answer.citations[0], corpus_version="0.9.0")
    answer = _answered([stale], answer.claims)
    result = validator.validate(answer, exposed_chunk_ids={stale.chunk_id})
    assert result.valid is False
    assert "corpus_version_mismatch" in result.issue_codes


def test_excerpt_mismatch_fails(corpus, validator):
    answer = _valid_answer(corpus)
    altered = _replace(answer.citations[0], excerpt="A different sentence entirely.")
    answer = _answered([altered], answer.claims)
    result = validator.validate(answer, exposed_chunk_ids={altered.chunk_id})
    assert result.valid is False
    assert "excerpt_mismatch" in result.issue_codes


def test_invalid_location_fails(corpus, validator):
    answer = _valid_answer(corpus)
    moved = _replace(answer.citations[0], location="synthetic://somewhere/else")
    answer = _answered([moved], answer.claims)
    result = validator.validate(answer, exposed_chunk_ids={moved.chunk_id})
    assert result.valid is False
    assert "invalid_location" in result.issue_codes


def test_claim_without_citation_fails(corpus, validator):
    chunk = _chunk_record(corpus, "doc-aurora-park-permits::0")
    citation = Citation(
        citation_id="cit-1",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        corpus_version=chunk.corpus_version,
        excerpt=chunk.text,
        location=chunk.location,
    )
    claim = Claim(claim_id="claim-1", text=chunk.text, citation_chunk_ids=())
    answer = _answered([citation], [claim])
    result = validator.validate(answer, exposed_chunk_ids={chunk.chunk_id})
    assert result.valid is False
    assert "claim_without_citation" in result.issue_codes


def test_unused_citation_fails(corpus, validator):
    answer = _valid_answer(corpus)
    chunk = _chunk_record(corpus, "doc-aurora-park-permits::1")
    orphan = Citation(
        citation_id="cit-orphan",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        corpus_version=chunk.corpus_version,
        excerpt=chunk.text,
        location=chunk.location,
    )
    answer = _answered([*answer.citations, orphan], answer.claims)
    result = validator.validate(
        answer, exposed_chunk_ids={answer.citations[0].chunk_id, chunk.chunk_id}
    )
    assert result.valid is False
    assert "unused_citation" in result.issue_codes


def test_never_verified_on_failure(corpus, validator):
    answer = _valid_answer(corpus)
    result = validator.validate(answer, exposed_chunk_ids=set())
    assert result.valid is False
    assert result.skipped is False
    assert not result.valid


def _replace(citation: Citation, **kwargs) -> Citation:
    return Citation(
        citation_id=kwargs.get("citation_id", citation.citation_id),
        chunk_id=kwargs.get("chunk_id", citation.chunk_id),
        document_id=kwargs.get("document_id", citation.document_id),
        corpus_version=kwargs.get("corpus_version", citation.corpus_version),
        excerpt=kwargs.get("excerpt", citation.excerpt),
        location=kwargs.get("location", citation.location),
    )
