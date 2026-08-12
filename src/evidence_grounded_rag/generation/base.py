from __future__ import annotations

from typing import Protocol, runtime_checkable

from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import GeneratedAnswer


@runtime_checkable
class Generator(Protocol):
    provider_id: str
    model_id: str
    is_llm_generated: bool
    mode: str

    def generate(self, query: str, evidence: EvidencePack) -> GeneratedAnswer: ...
