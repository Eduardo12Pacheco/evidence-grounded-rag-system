from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import GeneratedAnswer
from evidence_grounded_rag.generation.structured import StructuredAnswer
from evidence_grounded_rag.versions import CLAIM_SUPPORT_VERSION

RULE_VERSION = CLAIM_SUPPORT_VERSION


class ClaimSupport(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


@dataclass(frozen=True)
class ClaimSupportResult:
    claim_id: str
    support: ClaimSupport
    cited_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    matched_chunk_id: str | None = None
    rule: str = RULE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "support": self.support.value,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "matched_chunk_id": self.matched_chunk_id,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class ClaimSupportEvaluation:
    results: tuple[ClaimSupportResult, ...] = field(default_factory=tuple)
    rule: str = RULE_VERSION

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def supported(self) -> int:
        return sum(1 for r in self.results if r.support is ClaimSupport.SUPPORTED)

    @property
    def unsupported(self) -> int:
        return sum(1 for r in self.results if r.support is ClaimSupport.UNSUPPORTED)

    @property
    def unknown(self) -> int:
        return sum(1 for r in self.results if r.support is ClaimSupport.UNKNOWN)

    @property
    def support_rate(self) -> float:
        return self.supported / self.total if self.total else 0.0

    def by_claim_id(self, claim_id: str) -> ClaimSupportResult | None:
        for result in self.results:
            if result.claim_id == claim_id:
                return result
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "total": self.total,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "unknown": self.unknown,
            "support_rate": self.support_rate,
            "claims": [result.to_dict() for result in self.results],
        }


class ClaimSupportEvaluator:
    def evaluate(
        self,
        answer: GeneratedAnswer | StructuredAnswer,
        pack: EvidencePack,
    ) -> ClaimSupportEvaluation:
        claims = getattr(answer, "claims", ())
        results: list[ClaimSupportResult] = []
        for claim in claims:
            cited = tuple(claim.citation_chunk_ids)
            if not cited:
                results.append(
                    ClaimSupportResult(
                        claim_id=claim.claim_id,
                        support=ClaimSupport.UNKNOWN,
                        cited_chunk_ids=(),
                    )
                )
                continue

            claim_norm = normalize(claim.text)
            matched: str | None = None
            all_present = True
            for chunk_id in cited:
                item = pack.by_chunk_id(chunk_id)
                if item is None:
                    all_present = False
                    continue
                if claim_norm and claim_norm in normalize(item.text):
                    matched = chunk_id
                    break

            if matched is not None:
                support = ClaimSupport.SUPPORTED
            elif all_present:
                support = ClaimSupport.UNSUPPORTED
            else:
                support = ClaimSupport.UNKNOWN
            results.append(
                ClaimSupportResult(
                    claim_id=claim.claim_id,
                    support=support,
                    cited_chunk_ids=cited,
                    matched_chunk_id=matched,
                )
            )
        return ClaimSupportEvaluation(results=tuple(results))
