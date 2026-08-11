"""License gate for corpus manifests.

The gate FAILS CLOSED. Any manifest that cannot prove its license is approved
and independently confirmed is refused. Real corpora are blocked today: no
external license has been independently confirmed, and the Gob.ec corpus is
explicitly unauthorized with an unresolved license discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence_grounded_rag.corpus.errors import (
    BlockedRealCorpusError,
    UnapprovedLicenseError,
)

#: Exact blocker recorded for any real-corpus manifest. This text is the
#: current state of the real-corpus gate and is surfaced verbatim in reports
#: and the demo.
REAL_CORPUS_BLOCKER = (
    "No external corpus license has been independently confirmed. The Gob.ec "
    "corpus is explicitly unauthorized: its 'Creative Commons Attribution 4.0 "
    "International' label conflicts with the linked by/3.0/ec terms, and the "
    "license discrepancy is unresolved. Real-corpus ingestion stays blocked "
    "until a written license confirmation is obtained and recorded."
)

APPROVED_LICENSE_STATUSES = frozenset({"approved"})


@dataclass(frozen=True)
class GateVerdict:
    """Result of gating a manifest. ``allowed`` is False unless every check passed."""

    allowed: bool
    reasons: tuple[str, ...]
    blocker: str | None = None


def gate_manifest(manifest: dict[str, Any]) -> GateVerdict:
    """Gate a parsed manifest, failing closed on any unapproved or real corpus.

    ``manifest`` must expose ``synthetic``, ``real_corpus`` and a ``license``
    mapping with a ``status`` string.
    """
    reasons: list[str] = []
    blocker: str | None = None

    synthetic = bool(manifest.get("synthetic", False))
    real_corpus = bool(manifest.get("real_corpus", False))

    license_info = manifest.get("license") or {}
    license_status = str(license_info.get("status", "unknown")).lower()

    if real_corpus:
        blocked = bool((manifest.get("real_corpus_gate") or {}).get("blocked", True))
        if blocked:
            reasons.append("real_corpus: license not independently confirmed")
            blocker = str(
                (manifest.get("real_corpus_gate") or {}).get("blocker")
                or REAL_CORPUS_BLOCKER
            )
        elif license_status not in APPROVED_LICENSE_STATUSES:
            reasons.append("real_corpus: license status not approved")
        if not reasons and license_status not in APPROVED_LICENSE_STATUSES:
            reasons.append("real_corpus: license status not approved")

    if not real_corpus and license_status not in APPROVED_LICENSE_STATUSES:
        reasons.append("license status not approved")

    if not synthetic and not real_corpus:
        reasons.append("corpus neither synthetic nor clearly real_corpus")

    return GateVerdict(allowed=not reasons, reasons=tuple(reasons), blocker=blocker)


def assert_allowed(manifest: dict[str, Any]) -> None:
    """Raise the appropriate fail-closed error when a manifest is blocked."""
    verdict = gate_manifest(manifest)
    if verdict.allowed:
        return
    real_corpus = bool(manifest.get("real_corpus", False))
    if real_corpus:
        raise BlockedRealCorpusError(
            f"real corpus blocked: {verdict.blocker or 'no independent license'}",
            code="blocked_real_corpus",
        )
    raise UnapprovedLicenseError(
        f"license not approved: {'; '.join(verdict.reasons)}",
        code="unapproved_license",
    )
