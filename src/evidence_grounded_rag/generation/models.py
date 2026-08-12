from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evidence_grounded_rag.abstention.reasons import AbstentionReason


class AnswerState(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    document_id: str
    corpus_version: str
    excerpt: str
    location: str


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citation_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    predicate: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
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
