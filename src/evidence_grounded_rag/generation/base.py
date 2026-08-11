"""Provider interface for structured generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.generation.models import GeneratedAnswer


@runtime_checkable
class Generator(Protocol):
    """Contract every generation provider must satisfy.

    A generator receives ONLY the retrieved evidence chunks — never the full
    corpus — and returns a structured, validated-in-shape answer. Providers
    must fail closed: any malformed or unavailable provider returns an
    abstained ``GeneratedAnswer`` with a reason code rather than raising into
    the caller with a partial answer.
    """

    provider_id: str
    model_id: str
    is_llm_generated: bool

    def generate(self, query: str, evidence: Sequence[Chunk]) -> GeneratedAnswer: ...
