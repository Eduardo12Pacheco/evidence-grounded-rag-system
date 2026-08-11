"""Evidence-grounded RAG system.

Hybrid retrieval (sparse + dense), evidence-grounded generation with
citations, mandatory abstention, and faithfulness evaluation. The bundled
corpus is a small synthetic fixture corpus; no real corpus is implemented or
authorized (see ``docs/corpus-license-contract.md``).
"""

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
