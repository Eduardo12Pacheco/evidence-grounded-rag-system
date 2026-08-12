"""Typed models for structured generation output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evidence_grounded_rag.abstention.reasons import AbstentionReason


class AnswerState(str, Enum):
    """Final state of a generated answer."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class Citation:
    """A citation to exactly one evidence chunk."""

    citation_id: str
    chunk_id: str
    document_id: str
    corpus_version: str
    excerpt: str
    location: str


@dataclass(frozen=True)
class Claim:
    """One grounded statement in an answer."""

    claim_id: str
    text: str
    citation_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    predicate: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured output of a generation provider.

    ``state`` supports the three documented states: answered, abstained, and
    conflicting (conflicting evidence is surfaced as an abstention with reason
    ``conflicting_evidence``). ``is_llm_generated`` is False for the bundled
    deterministic offline provider and is recorded explicitly so no report can
    claim LLM generation.
    """

    state: AnswerState
    answer_text: str | None
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    abstention_reason: AbstentionReason | None = None
    abstention_detail: str | None = None
    provider_id: str = ""
    model_id: str = ""
    is_llm_generated: bool = False
    mode: str = ""
