from __future__ import annotations

from evidence_grounded_rag.abstention import AbstentionReason, reason_detail

REQUIRED_CODES = [
    "insufficient_evidence",
    "unsupported_query",
    "conflicting_evidence",
    "citation_validation_failed",
    "provider_unavailable",
    "malformed_output",
    "blocked_corpus",
    "invalid_corpus",
    "stale_corpus",
]


def test_all_required_reason_codes_exist():
    values = {reason.value for reason in AbstentionReason}
    for code in REQUIRED_CODES:
        assert code in values, f"missing reason code {code}"


def test_every_reason_has_human_readable_detail():
    for reason in AbstentionReason:
        detail = reason_detail(reason)
        assert isinstance(detail, str)
        assert len(detail) > 20


def test_conflicting_evidence_is_a_distinct_state():
    from evidence_grounded_rag.generation import AnswerState

    assert AbstentionReason.CONFLICTING_EVIDENCE.value == "conflicting_evidence"
    assert AnswerState.CONFLICTING.value == "conflicting"
