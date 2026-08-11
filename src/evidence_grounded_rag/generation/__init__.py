"""Structured generation with citations, abstention, and fail-closed adapters.

The bundled provider is a deterministic offline extractive/template generator
(``is_llm_generated=False``). The environment-configured adapter fails closed:
no remote provider is ever contacted.
"""

from __future__ import annotations

from evidence_grounded_rag.generation.adapter import (
    ENV_VAR,
    EnvConfiguredGenerator,
)
from evidence_grounded_rag.generation.base import Generator
from evidence_grounded_rag.generation.extractive import (
    ExtractiveTemplateGenerator,
    split_predicate_value,
    split_sentences,
    values_conflict,
)
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)

__all__ = [
    "ENV_VAR",
    "AnswerState",
    "Citation",
    "Claim",
    "EnvConfiguredGenerator",
    "ExtractiveTemplateGenerator",
    "GeneratedAnswer",
    "Generator",
    "split_predicate_value",
    "split_sentences",
    "values_conflict",
]
