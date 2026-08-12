"""Tests for the strict structured parser/validator and the live adapter.

Covers schema parsing (malformed JSON, missing/duplicate ids, invalid
statuses, uncited claims), evidence-scoped citation rules (unknown/unexposed,
stale version, excerpt/location mismatch), deterministic conversion to
``GeneratedAnswer``, and the live adapter's fail-closed behavior with fake
success / timeout / unavailable / malformed providers and NO offline fallback.
"""

from __future__ import annotations

import json

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.generation import (
    AnswerState,
    EvidencePack,
    LiveGenerator,
    StructuredAnswer,
    StructuredAnswerParser,
    StructuredAnswerStatus,
    StructuredAnswerValidator,
    to_generated_answer,
)


def _chunk(chunk_id: str, text: str, location: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        seq=0,
        text=text,
        location=location or f"synthetic://fixtures/{chunk_id}",
        corpus_version="1.0.0",
    )


def _pack(*chunks: Chunk) -> EvidencePack:
    return EvidencePack.from_chunks(chunks, corpus_id="c", corpus_version="1.0.0")


def _valid_payload(chunk: Chunk) -> dict[str, object]:
    return {
        "status": "answered",
        "answer_text": "The park is open.",
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "The park is open.",
                "citation_chunk_ids": [chunk.chunk_id],
            }
        ],
        "citations": [
            {
                "citation_id": "cit-1",
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "corpus_version": chunk.corpus_version,
                "excerpt": chunk.text,
                "location": chunk.location,
            }
        ],
        "abstention_reason": None,
        "abstention_detail": None,
    }


PARSER = StructuredAnswerParser()
VALIDATOR = StructuredAnswerValidator()


# --------------------------------------------------------------------------- #
# parser: schema
# --------------------------------------------------------------------------- #


def test_parser_rejects_malformed_json():
    result = PARSER.parse("{not json")
    assert not result.ok
    assert result.answer is None
    assert result.issues[0].code == "invalid_json"


def test_parser_rejects_non_object_root():
    result = PARSER.parse("[1, 2]")
    assert result.issues[0].code == "invalid_json"


def test_parser_rejects_missing_status():
    result = PARSER.parse(json.dumps({"answer_text": "x"}))
    assert any(issue.code == "missing_field" for issue in result.issues)


def test_parser_rejects_invalid_status():
    payload = _valid_payload(_chunk("doc-a::0", "The park is open."))
    payload["status"] = "hallucinated"
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "invalid_status" for issue in result.issues)


def test_parser_rejects_duplicate_claim_ids():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["claims"].append(dict(payload["claims"][0]))
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "duplicate_claim_id" for issue in result.issues)


def test_parser_rejects_missing_claim_id():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    del payload["claims"][0]["claim_id"]
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "missing_field" for issue in result.issues)


def test_parser_rejects_uncited_claim():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["claims"][0]["citation_chunk_ids"] = []
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "claim_without_citation" for issue in result.issues)


def test_parser_rejects_answered_without_answer_text():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["answer_text"] = None
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "missing_answer_text" for issue in result.issues)


def test_parser_rejects_abstained_with_payload():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["status"] = "abstained"
    payload["abstention_reason"] = "insufficient_evidence"
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "unexpected_payload" for issue in result.issues)


def test_parser_rejects_invalid_abstention_reason():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["status"] = "abstained"
    payload["claims"] = []
    payload["citations"] = []
    payload["answer_text"] = None
    payload["abstention_reason"] = "maybe"
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "invalid_abstention_reason" for issue in result.issues)


def test_parser_accepts_valid_abstention():
    payload = {
        "status": "abstained",
        "answer_text": None,
        "claims": [],
        "citations": [],
        "abstention_reason": "unsupported_query",
        "abstention_detail": "out of scope",
    }
    result = PARSER.parse(json.dumps(payload))
    assert result.ok
    assert result.answer is not None
    assert result.answer.status is StructuredAnswerStatus.ABSTAINED


def test_parser_accepts_valid_answered():
    chunk = _chunk("doc-a::0", "The park is open.")
    result = PARSER.parse(json.dumps(_valid_payload(chunk)))
    assert result.ok
    assert result.answer is not None
    assert result.answer.status is StructuredAnswerStatus.ANSWERED
    assert len(result.answer.claims) == 1
    assert len(result.answer.citations) == 1


def test_parser_rejects_claim_citing_missing_citation_record():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["claims"][0]["citation_chunk_ids"] = ["doc-other::0"]
    result = PARSER.parse(json.dumps(payload))
    assert any(issue.code == "citation_record_missing" for issue in result.issues)


# --------------------------------------------------------------------------- #
# validator: citation rules against the evidence pack
# --------------------------------------------------------------------------- #


def _parsed(payload: dict[str, object]) -> StructuredAnswer:
    result = PARSER.parse(json.dumps(payload))
    assert result.ok, [i.code for i in result.issues]
    return result.answer


def test_validator_accepts_citations_to_exposed_chunks():
    chunk = _chunk("doc-a::0", "The park is open.")
    answer = _parsed(_valid_payload(chunk))
    issues = VALIDATOR.validate(answer, _pack(chunk))
    assert issues == ()


def test_validator_rejects_unexposed_citation():
    chunk = _chunk("doc-a::0", "The park is open.")
    answer = _parsed(_valid_payload(chunk))
    issues = VALIDATOR.validate(answer, _pack())
    assert any(issue.code == "citation_not_exposed" for issue in issues)


def test_validator_rejects_stale_corpus_version():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["citations"][0]["corpus_version"] = "0.9.0"
    answer = _parsed(payload)
    issues = VALIDATOR.validate(answer, _pack(chunk))
    assert any(issue.code == "corpus_version_mismatch" for issue in issues)


def test_validator_rejects_excerpt_mismatch():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["citations"][0]["excerpt"] = "A different sentence."
    answer = _parsed(payload)
    issues = VALIDATOR.validate(answer, _pack(chunk))
    assert any(issue.code == "excerpt_mismatch" for issue in issues)


def test_validator_rejects_location_mismatch():
    chunk = _chunk("doc-a::0", "The park is open.")
    payload = _valid_payload(chunk)
    payload["citations"][0]["location"] = "synthetic://elsewhere"
    answer = _parsed(payload)
    issues = VALIDATOR.validate(answer, _pack(chunk))
    assert any(issue.code == "invalid_location" for issue in issues)


def test_validator_rejects_unused_citation():
    chunk_a = _chunk("doc-a::0", "The park is open.")
    chunk_b = _chunk("doc-b::0", "The library is open.")
    payload = _valid_payload(chunk_a)
    payload["citations"].append(
        {
            "citation_id": "cit-orphan",
            "chunk_id": chunk_b.chunk_id,
            "document_id": chunk_b.document_id,
            "corpus_version": chunk_b.corpus_version,
            "excerpt": chunk_b.text,
            "location": chunk_b.location,
        }
    )
    answer = _parsed(payload)
    issues = VALIDATOR.validate(answer, _pack(chunk_a, chunk_b))
    assert any(issue.code == "unused_citation" for issue in issues)


def test_validator_skips_non_answered_statuses():
    payload = {
        "status": "abstained",
        "answer_text": None,
        "claims": [],
        "citations": [],
        "abstention_reason": "insufficient_evidence",
        "abstention_detail": None,
    }
    answer = _parsed(payload)
    assert VALIDATOR.validate(answer, _pack()) == ()


# --------------------------------------------------------------------------- #
# conversion to GeneratedAnswer
# --------------------------------------------------------------------------- #


def test_conversion_maps_all_statuses():
    chunk = _chunk("doc-a::0", "The park is open.")

    answered = to_generated_answer(_parsed(_valid_payload(chunk)), provider_id="p", model_id="m", mode="live")
    assert answered.state is AnswerState.ANSWERED
    assert answered.is_llm_generated is True
    assert answered.mode == "live"

    abstained_payload = {
        "status": "abstained", "answer_text": None, "claims": [], "citations": [],
        "abstention_reason": "unsupported_query", "abstention_detail": "scope",
    }
    abstained = to_generated_answer(_parsed(abstained_payload), mode="live")
    assert abstained.state is AnswerState.ABSTAINED
    assert abstained.abstention_reason is AbstentionReason.UNSUPPORTED_QUERY

    conflicting_payload = {
        "status": "conflicting", "answer_text": None, "claims": [], "citations": [],
        "abstention_reason": "conflicting_evidence", "abstention_detail": "A vs B",
    }
    conflicting = to_generated_answer(_parsed(conflicting_payload), mode="live")
    assert conflicting.state is AnswerState.CONFLICTING
    assert conflicting.abstention_reason is AbstentionReason.CONFLICTING_EVIDENCE

    invalid_payload = {
        "status": "invalid", "answer_text": None, "claims": [], "citations": [],
        "abstention_reason": None, "abstention_detail": "could not produce JSON",
    }
    invalid = to_generated_answer(_parsed(invalid_payload), mode="live")
    assert invalid.state is AnswerState.ABSTAINED
    assert invalid.abstention_reason is AbstentionReason.MALFORMED_OUTPUT


# --------------------------------------------------------------------------- #
# live adapter: fake providers, fail closed, no fallback
# --------------------------------------------------------------------------- #


def _live_pack() -> EvidencePack:
    return _pack(_chunk("doc-a::0", "The park is open."))


def test_live_generator_success_parses_and_converts():
    chunk = _chunk("doc-a::0", "The park is open.")

    def fake_transport(prompt: str) -> str:
        assert "<evidence>" in prompt and "</evidence>" in prompt
        return json.dumps(_valid_payload(chunk))

    gen = LiveGenerator(fake_transport)
    answer = gen.generate("Is the park open?", _live_pack())
    assert answer.state is AnswerState.ANSWERED
    assert answer.is_llm_generated is True
    assert answer.mode == "live"
    assert answer.provider_id == "injected-live-provider"


def test_live_generator_without_transport_fails_closed():
    gen = LiveGenerator()
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert "not fall back" in (answer.abstention_detail or "").lower()


def test_live_generator_timeout_fails_closed():
    def timeout_transport(prompt: str) -> str:
        raise TimeoutError("took too long")

    gen = LiveGenerator(timeout_transport, timeout_seconds=1)
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert "timed out" in (answer.abstention_detail or "")


def test_live_generator_unavailable_fails_closed():
    def broken_transport(prompt: str) -> str:
        raise ConnectionError("provider down")

    gen = LiveGenerator(broken_transport)
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE


def test_live_generator_malformed_output_fails_closed():
    def malformed_transport(prompt: str) -> str:
        return "not json at all"

    gen = LiveGenerator(malformed_transport)
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.MALFORMED_OUTPUT


def test_live_generator_schema_failure_fails_closed():
    def bad_transport(prompt: str) -> str:
        return json.dumps({"status": "answered", "answer_text": "x", "claims": [], "citations": []})

    gen = LiveGenerator(bad_transport)
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.MALFORMED_OUTPUT


def test_live_generator_non_text_output_fails_closed():
    gen = LiveGenerator(lambda prompt: b"\x00bytes")
    answer = gen.generate("anything", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.MALFORMED_OUTPUT


def test_live_generator_never_falls_back_to_offline():
    """Under the live label, unavailability must NEVER invoke the extractive
    generator: the result is a structured abstention, not an offline answer."""

    def broken_transport(prompt: str) -> str:
        raise RuntimeError("boom")

    gen = LiveGenerator(broken_transport, model_id="live-v1")
    assert gen.is_llm_generated is True
    answer = gen.generate("Is the park open?", _live_pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert answer.answer_text is None
    assert answer.is_llm_generated is True
    assert answer.mode == "live"


def test_live_generator_empty_evidence_abstains_insufficient():
    gen = LiveGenerator(lambda prompt: "{}")
    answer = gen.generate("anything", _pack())
    assert answer.state is AnswerState.ABSTAINED
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
