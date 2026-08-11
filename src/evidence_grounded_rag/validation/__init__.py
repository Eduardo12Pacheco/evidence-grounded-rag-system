"""Citation validation layer: strict, deterministic, fail-closed."""

from __future__ import annotations

from evidence_grounded_rag.validation.citations import (
    CitationIssue,
    CitationValidationResult,
    CitationValidator,
)

__all__ = ["CitationIssue", "CitationValidationResult", "CitationValidator"]
