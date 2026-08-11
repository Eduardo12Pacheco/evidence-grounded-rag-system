"""Mandatory abstention reason codes.

An abstention is never silent: every refusal carries one of these stable
reason codes plus a human-readable detail. These codes are the contract
between the generator, the pipeline, and evaluation.
"""

from __future__ import annotations

from enum import Enum


class AbstentionReason(str, Enum):
    """Stable reason codes for every abstention path."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_QUERY = "unsupported_query"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_OUTPUT = "malformed_output"
    BLOCKED_CORPUS = "blocked_corpus"
    INVALID_CORPUS = "invalid_corpus"
    STALE_CORPUS = "stale_corpus"


REASON_DETAILS: dict[AbstentionReason, str] = {
    AbstentionReason.INSUFFICIENT_EVIDENCE: (
        "Retrieved evidence is empty, scores below the committed threshold, or "
        "the query is ambiguous across the evidence."
    ),
    AbstentionReason.UNSUPPORTED_QUERY: (
        "The query's key terms appear in no retrieved passage; the question is "
        "out of scope for this corpus."
    ),
    AbstentionReason.CONFLICTING_EVIDENCE: (
        "Retrieved passages directly contradict each other on the same fact; "
        "no answer can be grounded."
    ),
    AbstentionReason.CITATION_VALIDATION_FAILED: (
        "Generated citations failed validation (nonexistent, unexposed, "
        "wrong version, or excerpt mismatch); the answer was refused."
    ),
    AbstentionReason.PROVIDER_UNAVAILABLE: (
        "The configured generation provider is unavailable or requires "
        "network/credentials that are not present."
    ),
    AbstentionReason.MALFORMED_OUTPUT: (
        "The generation provider returned malformed output that could not be "
        "parsed into a structured answer."
    ),
    AbstentionReason.BLOCKED_CORPUS: (
        "The corpus is license-blocked: no real corpus license has been "
        "independently confirmed."
    ),
    AbstentionReason.INVALID_CORPUS: (
        "The corpus manifest failed validation (checksum, format, or "
        "duplicate IDs)."
    ),
    AbstentionReason.STALE_CORPUS: (
        "The loaded corpus version does not match the version this pipeline "
        "was configured for."
    ),
}


def reason_detail(reason: AbstentionReason) -> str:
    return REASON_DETAILS[reason]
