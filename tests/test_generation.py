from __future__ import annotations

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.generation import (
    ENV_VAR,
    AnswerState,
    EnvConfiguredGenerator,
    ExtractiveTemplateGenerator,
    split_predicate_value,
    split_sentences,
    values_conflict,
)


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        seq=0,
        text=text,
        location=f"synthetic://synthetic-fixtures/{document_id}/chunk/0",
        corpus_version="1.0.0",
    )


def test_empty_evidence_abstains_insufficient():
    generator = ExtractiveTemplateGenerator()
    answer = generator.generate("Where is the park?", [])
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_empty_query_abstains_unsupported():
    generator = ExtractiveTemplateGenerator()
    answer = generator.generate(
        "   ", [_chunk("doc-a::0", "doc-a", "The park is open.")]
    )
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.UNSUPPORTED_QUERY


def test_query_terms_absent_abstains_unsupported():
    generator = ExtractiveTemplateGenerator()
    evidence = [_chunk("doc-a::0", "doc-a", "The park is open from June to August.")]
    answer = generator.generate("What is the fireworks fee?", evidence)
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.UNSUPPORTED_QUERY


def test_single_passage_answers_with_claim_and_citation():
    generator = ExtractiveTemplateGenerator()
    evidence = [
        _chunk("doc-a::0", "doc-a", "The special event fee is 120 dollars per day.")
    ]
    answer = generator.generate("What is the special event fee?", evidence)
    assert answer.state is AnswerState.ANSWERED
    assert answer.answer_text
    assert "120 dollars per day" in answer.answer_text
    assert len(answer.claims) == 1
    assert answer.claims[0].citation_chunk_ids == ("doc-a::0",)
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.chunk_id == "doc-a::0"
    assert citation.excerpt == evidence[0].text
    assert citation.location == evidence[0].location


def test_generator_receives_only_exposed_evidence():
    generator = ExtractiveTemplateGenerator()
    exposed = [
        _chunk("doc-a::0", "doc-a", "The pool is open in summer."),
        _chunk("doc-b::0", "doc-b", "The pool is open in winter."),
    ]
    answer = generator.generate("When is the pool open and is it cold?", exposed)
    cited = {citation.chunk_id for citation in answer.citations}
    assert cited <= {chunk.chunk_id for chunk in exposed}


def test_conflicting_scalar_values_abstain_conflicting():
    generator = ExtractiveTemplateGenerator()
    evidence = [
        _chunk(
            "fee-2024::0", "fee-2024", "The special event fee is 120 dollars per day."
        ),
        _chunk(
            "fee-2025::0", "fee-2025", "The special event fee is 50 dollars per day."
        ),
    ]
    answer = generator.generate("How much is the special event fee?", evidence)
    assert answer.state is AnswerState.CONFLICTING
    assert answer.abstention_reason is AbstentionReason.CONFLICTING_EVIDENCE


def test_negation_pair_abstains_conflicting():
    generator = ExtractiveTemplateGenerator()
    evidence = [
        _chunk("lib-a::0", "lib-a", "The library is open on Sunday."),
        _chunk("lib-b::0", "lib-b", "The library is not open on Sunday."),
    ]
    answer = generator.generate("Is the library open on Sunday?", evidence)
    assert answer.state is AnswerState.CONFLICTING
    assert answer.abstention_reason is AbstentionReason.CONFLICTING_EVIDENCE


def test_ambiguous_same_value_different_subjects_abstains():
    generator = ExtractiveTemplateGenerator()
    evidence = [
        _chunk(
            "pools-north::0",
            "pools-north",
            "The North Pool is open from June to August.",
        ),
        _chunk(
            "pools-south::0",
            "pools-south",
            "The South Pool is open from June to August.",
        ),
    ]
    answer = generator.generate("When is the pool open?", evidence)
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_multi_aspect_query_answers_multiple_passages():
    generator = ExtractiveTemplateGenerator()
    evidence = [
        _chunk(
            "doc-a::0",
            "doc-a",
            "A permit application must include a signed code of conduct.",
        ),
        _chunk(
            "doc-a::1",
            "doc-a",
            "Permit requests have a review time of up to 10 working days.",
        ),
    ]
    answer = generator.generate(
        "What must a permit application include and how long does the review take?",
        evidence,
    )
    assert answer.state is AnswerState.ANSWERED
    assert len(answer.claims) == 2
    assert len(answer.citations) == 2


def test_provider_metadata_never_claims_llm():
    generator = ExtractiveTemplateGenerator()
    answer = generator.generate(
        "What must a permit application include?",
        [
            _chunk(
                "doc-a::0",
                "doc-a",
                "A permit application must include a code of conduct.",
            )
        ],
    )
    assert generator.is_llm_generated is False
    assert answer.is_llm_generated is False
    assert answer.provider_id == "extractive-template"
    assert answer.model_id == "extractive-template-v1"


def test_env_adapter_default_uses_offline_provider():
    adapter = EnvConfiguredGenerator(env={})
    answer = adapter.generate(
        "What must a permit application include?",
        [
            _chunk(
                "doc-a::0",
                "doc-a",
                "A permit application must include a code of conduct.",
            )
        ],
    )
    assert answer.state is AnswerState.ANSWERED


def test_env_adapter_remote_provider_fails_closed():
    adapter = EnvConfiguredGenerator(env={ENV_VAR: "claude"})
    answer = adapter.generate(
        "anything",
        [_chunk("doc-a::0", "doc-a", "Some sentence that matches the query.")],
    )
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE


def test_env_adapter_unknown_provider_fails_closed():
    adapter = EnvConfiguredGenerator(env={ENV_VAR: "mystery-model"})
    answer = adapter.generate("anything", [])
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE


def test_sentence_and_predicate_helpers():
    assert split_sentences("First sentence. Second! Third?") == [
        "First sentence.",
        "Second!",
        "Third?",
    ]
    predicate, value = split_predicate_value(
        "The North Pool is open from June to August."
    )
    assert predicate == "north pool"
    assert value == "open from june to august."
    assert split_predicate_value("No separator here") == (None, None)
    assert values_conflict("120 dollars", "50 dollars")
    assert not values_conflict("120 dollars", "120 dollars")
    assert values_conflict("open on sunday", "not open on sunday")
    assert not values_conflict("open monday", "open saturday")
