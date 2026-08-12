"""Structured generation with citations, abstention, and fail-closed adapters.

The bundled provider is a deterministic offline extractive/template generator
(``is_llm_generated=False``). The environment-configured adapter fails closed:
no remote provider is ever contacted. An explicit optional live adapter
(``LiveGenerator``) exists but requires an injected transport — no
provider/model/credentials are configured, and live generation never falls
back to offline.
"""

from __future__ import annotations

from evidence_grounded_rag.generation.adapter import (
    ENV_VAR,
    EnvConfiguredGenerator,
)
from evidence_grounded_rag.generation.base import Generator
from evidence_grounded_rag.generation.evidence import (
    DEFAULT_MAX_ITEMS,
    EvidenceItem,
    EvidencePack,
)
from evidence_grounded_rag.generation.extractive import (
    ExtractiveTemplateGenerator,
    split_predicate_value,
    split_sentences,
    values_conflict,
)
from evidence_grounded_rag.generation.live import LiveGenerator
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)
from evidence_grounded_rag.generation.prompt import (
    PROMPT_VERSION,
    build_grounded_answer_prompt,
)
from evidence_grounded_rag.generation.structured import (
    SCHEMA_VERSION,
    StructuredAnswer,
    StructuredAnswerParser,
    StructuredAnswerStatus,
    StructuredAnswerValidator,
    StructuredParseIssue,
    StructuredParseResult,
    to_generated_answer,
)
from evidence_grounded_rag.generation.support import (
    CLAIM_SUPPORT_VERSION,
    ClaimSupport,
    ClaimSupportEvaluation,
    ClaimSupportEvaluator,
    ClaimSupportResult,
    normalize,
)

__all__ = [
    "CLAIM_SUPPORT_VERSION",
    "DEFAULT_MAX_ITEMS",
    "ENV_VAR",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "AnswerState",
    "Citation",
    "Claim",
    "ClaimSupport",
    "ClaimSupportEvaluation",
    "ClaimSupportEvaluator",
    "ClaimSupportResult",
    "EnvConfiguredGenerator",
    "EvidenceItem",
    "EvidencePack",
    "ExtractiveTemplateGenerator",
    "GeneratedAnswer",
    "Generator",
    "LiveGenerator",
    "StructuredAnswer",
    "StructuredAnswerParser",
    "StructuredAnswerStatus",
    "StructuredAnswerValidator",
    "StructuredParseIssue",
    "StructuredParseResult",
    "build_grounded_answer_prompt",
    "normalize",
    "split_predicate_value",
    "split_sentences",
    "to_generated_answer",
    "values_conflict",
]
