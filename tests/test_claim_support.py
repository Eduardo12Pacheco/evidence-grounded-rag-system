from __future__ import annotations

from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.generation import (
    AnswerState,
    ClaimSupport,
    ClaimSupportEvaluator,
    EvidencePack,
    GeneratedAnswer,
    normalize,
)
from evidence_grounded_rag.generation.models import Citation, Claim
from evidence_grounded_rag.versions import CLAIM_SUPPORT_VERSION


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        seq=0,
        text=text,
        location=f"synthetic://fixtures/{chunk_id}",
        corpus_version="1.0.0",
    )


def _pack(*chunks: Chunk) -> EvidencePack:
    return EvidencePack.from_chunks(chunks, corpus_id="c", corpus_version="1.0.0")


def _answered(*claims: Claim, citations: tuple[Citation, ...] = ()) -> GeneratedAnswer:
    return GeneratedAnswer(
        state=AnswerState.ANSWERED,
        answer_text="answer",
        claims=claims,
        citations=citations,
        provider_id="test",
        model_id="test",
        is_llm_generated=False,
    )


def _claim(claim_id: str, text: str, *chunk_ids: str) -> Claim:
    return Claim(claim_id=claim_id, text=text, citation_chunk_ids=chunk_ids)


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  The   Park \tis Open. ") == "the park is open."
    assert normalize("Café") == "café"


def test_supported_when_claim_contained_in_cited_chunk():
    chunk = _chunk("doc-a::0", "The park is open from June to August.")
    answer = _answered(
        _claim("c-1", "The park is open from June to August.", "doc-a::0")
    )
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.SUPPORTED
    assert evaluation.results[0].matched_chunk_id == "doc-a::0"
    assert evaluation.supported == 1
    assert evaluation.support_rate == 1.0


def test_unsupported_when_claim_not_contained_anywhere():
    chunk = _chunk("doc-a::0", "The library opens on Monday.")
    answer = _answered(_claim("c-1", "The park is open on Sunday.", "doc-a::0"))
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.UNSUPPORTED
    assert evaluation.unsupported == 1


def test_unknown_when_claim_has_no_citation():
    chunk = _chunk("doc-a::0", "The park is open.")
    answer = _answered(_claim("c-1", "The park is open."))
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.UNKNOWN


def test_unknown_when_cited_chunk_not_in_pack():
    chunk = _chunk("doc-a::0", "The park is open.")
    answer = _answered(_claim("c-1", "The park is open.", "doc-unexposed::0"))
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.UNKNOWN


def test_supported_when_any_of_multiple_cited_chunks_contains_claim():
    chunk_a = _chunk("doc-a::0", "Unrelated content here.")
    chunk_b = _chunk("doc-b::0", "The answer to everything is 42.")
    answer = _answered(
        _claim("c-1", "The answer to everything is 42.", "doc-a::0", "doc-b::0")
    )
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk_a, chunk_b))
    assert evaluation.results[0].support is ClaimSupport.SUPPORTED
    assert evaluation.results[0].matched_chunk_id == "doc-b::0"


def test_case_insensitive_containment():
    chunk = _chunk("doc-a::0", "PERMITS require a signed Code of Conduct.")
    answer = _answered(
        _claim("c-1", "permits require a signed code of conduct.", "doc-a::0")
    )
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.SUPPORTED


def test_rule_version_is_committed():
    evaluation = ClaimSupportEvaluator().evaluate(
        _answered(_claim("c-1", "x", "doc-a::0")), _pack()
    )
    assert evaluation.rule == CLAIM_SUPPORT_VERSION


def test_evaluation_rates_and_serialization():
    chunk_a = _chunk("doc-a::0", "The park is open.")
    chunk_b = _chunk("doc-b::0", "The library is closed.")
    answer = _answered(
        _claim("c-1", "The park is open.", "doc-a::0"),
        _claim("c-2", "The park is closed.", "doc-b::0"),
        _claim("c-3", "Uncited statement."),
    )
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk_a, chunk_b))
    assert (evaluation.supported, evaluation.unsupported, evaluation.unknown) == (
        1,
        1,
        1,
    )
    assert evaluation.total == 3
    assert evaluation.support_rate == 1 / 3
    serialized = evaluation.to_dict()
    assert serialized["supported"] == 1
    assert serialized["unknown"] == 1
    assert len(serialized["claims"]) == 3
    assert evaluation.by_claim_id("c-2").support is ClaimSupport.UNSUPPORTED
    assert evaluation.by_claim_id("c-9") is None


def test_support_is_distinct_from_structural_validation():
    chunk = _chunk("doc-a::0", "The library opens at nine.")
    citation = Citation(
        citation_id="cit-1",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        corpus_version=chunk.corpus_version,
        excerpt=chunk.text,
        location=chunk.location,
    )
    answer = _answered(
        _claim("c-1", "The library opens at midnight.", "doc-a::0"),
        citations=(citation,),
    )
    evaluation = ClaimSupportEvaluator().evaluate(answer, _pack(chunk))
    assert evaluation.results[0].support is ClaimSupport.UNSUPPORTED
