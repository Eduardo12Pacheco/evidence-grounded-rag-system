from __future__ import annotations

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.pipeline import (
    AnswerRecord,
    EvidenceRagPipeline,
    PipelineConfig,
)
from evidence_grounded_rag.versions import PACKAGE_VERSION

__version__ = PACKAGE_VERSION

__all__ = [
    "PACKAGE_VERSION",
    "AbstentionReason",
    "AnswerRecord",
    "EvidenceRagPipeline",
    "PipelineConfig",
]
