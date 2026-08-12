from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.generation.base import Generator
from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.extractive import ExtractiveTemplateGenerator
from evidence_grounded_rag.generation.models import (
    AnswerState,
    GeneratedAnswer,
)

ENV_VAR = "EGRAG_GENERATION_PROVIDER"

OFFLINE_PROVIDER_IDS = frozenset(
    {"", "deterministic-offline", "extractive-template", "deterministic-template"}
)

REMOTE_PROVIDER_IDS = frozenset({"claude", "anthropic", "openai", "remote"})

LIVE_PROVIDER_IDS = frozenset({"live", "live-provider", "injected-live-provider"})

PROVIDER_ID = "env-configured-adapter"
MODEL_ID = "env-adapter-v1"
MODE = "offline"


class EnvConfiguredGenerator(Generator):
    provider_id: str = PROVIDER_ID
    model_id: str = MODEL_ID
    is_llm_generated: bool = False
    mode: str = MODE

    def __init__(
        self,
        *,
        offline_generator: Generator | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._offline_generator = offline_generator or ExtractiveTemplateGenerator()
        self._env = env if env is not None else os.environ

    def generate(
        self, query: str, evidence: EvidencePack | Sequence[Any] | Any
    ) -> GeneratedAnswer:
        selected = (self._env.get(ENV_VAR) or "").strip().lower()
        if selected in OFFLINE_PROVIDER_IDS:
            return self._offline_generator.generate(query, evidence)
        if selected in LIVE_PROVIDER_IDS:
            return self._abstain(
                f"Generation provider {selected!r} requires an explicitly "
                "injected live transport; none is configured and live "
                "generation does not fall back to offline."
            )
        if selected in REMOTE_PROVIDER_IDS:
            return self._abstain(
                f"Generation provider {selected!r} requires network access and "
                "credentials that are not configured; refusing in offline mode."
            )
        return self._abstain(
            f"Unknown generation provider {selected!r}; refusing in offline mode."
        )

    @staticmethod
    def _abstain(detail: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            state=AnswerState.ABSTAINED,
            answer_text=None,
            abstention_reason=AbstentionReason.PROVIDER_UNAVAILABLE,
            abstention_detail=detail,
            provider_id=PROVIDER_ID,
            model_id=MODEL_ID,
            is_llm_generated=False,
            mode=MODE,
        )
