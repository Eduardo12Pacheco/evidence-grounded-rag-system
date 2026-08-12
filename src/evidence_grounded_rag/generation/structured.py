from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from evidence_grounded_rag.abstention.reasons import AbstentionReason
from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)
from evidence_grounded_rag.versions import STRUCTURED_ANSWER_SCHEMA_VERSION

SCHEMA_VERSION = STRUCTURED_ANSWER_SCHEMA_VERSION


class StructuredAnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    CONFLICTING = "conflicting"
    INVALID = "invalid"


_ANSWERED_STATUSES = frozenset({StructuredAnswerStatus.ANSWERED})

_VALID_ABSTENTION_STRINGS = frozenset(
    {
        "insufficient_evidence",
        "unsupported_query",
        "conflicting_evidence",
    }
)

_REQUIRED_CLAIM_FIELDS = ("claim_id", "text", "citation_chunk_ids")
_REQUIRED_CITATION_FIELDS = (
    "citation_id",
    "chunk_id",
    "document_id",
    "corpus_version",
    "excerpt",
    "location",
)


@dataclass(frozen=True)
class StructuredParseIssue:
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "field": self.field}


@dataclass(frozen=True)
class StructuredAnswer:
    status: StructuredAnswerStatus
    answer_text: str | None
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    abstention_reason: str | None = None
    abstention_detail: str | None = None
    provider_id: str = ""
    model_id: str = ""
    mode: str = ""
    trace: dict[str, object] = field(default_factory=dict)
    corpus: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredParseResult:
    answer: StructuredAnswer | None = None
    issues: tuple[StructuredParseIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.answer is not None and not self.issues


class StructuredAnswerParser:
    def parse(self, raw: str) -> StructuredParseResult:
        if not isinstance(raw, str):
            return StructuredParseResult(
                issues=(
                    StructuredParseIssue(
                        code="invalid_json",
                        message=(
                            "Provider output is not text; expected a JSON string."
                        ),
                    ),
                )
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return StructuredParseResult(
                issues=(
                    StructuredParseIssue(
                        code="invalid_json",
                        message=f"Provider output is not valid JSON: {exc.msg}",
                    ),
                )
            )
        if not isinstance(payload, dict):
            return StructuredParseResult(
                issues=(
                    StructuredParseIssue(
                        code="invalid_json",
                        message="Provider output JSON root must be an object.",
                    ),
                )
            )

        issues: list[StructuredParseIssue] = []
        status = self._parse_status(payload, issues)
        claims = self._parse_claims(payload, issues)
        citations = self._parse_citations(payload, issues)

        answer_text = payload.get("answer_text")
        if status is StructuredAnswerStatus.ANSWERED:
            if not isinstance(answer_text, str) or not answer_text.strip():
                issues.append(
                    StructuredParseIssue(
                        code="missing_answer_text",
                        message="answered status requires non-empty answer_text.",
                        field="answer_text",
                    )
                )
            if not claims:
                issues.append(
                    StructuredParseIssue(
                        code="missing_claims",
                        message="answered status requires at least one claim.",
                        field="claims",
                    )
                )
            for claim in claims:
                if not claim.citation_chunk_ids:
                    issues.append(
                        StructuredParseIssue(
                            code="claim_without_citation",
                            message=f"claim {claim.claim_id!r} carries no citation.",
                            field="claims",
                        )
                    )
        else:
            if answer_text not in (None, ""):
                issues.append(
                    StructuredParseIssue(
                        code="unexpected_answer_text",
                        message=(f"{status.value} status must not carry answer_text."),
                        field="answer_text",
                    )
                )
            if claims or citations:
                issues.append(
                    StructuredParseIssue(
                        code="unexpected_payload",
                        message=(
                            f"{status.value} status must not carry claims or citations."
                        ),
                        field="claims",
                    )
                )

        if status in (
            StructuredAnswerStatus.ANSWERED,
            StructuredAnswerStatus.CONFLICTING,
        ):
            cited_chunks = {citation.chunk_id for citation in citations}
            seen: set[str] = set()
            for citation in citations:
                if citation.citation_id in seen:
                    issues.append(
                        StructuredParseIssue(
                            code="duplicate_citation_id",
                            message=(
                                f"duplicate citation id {citation.citation_id!r}."
                            ),
                            field="citations",
                        )
                    )
                seen.add(citation.citation_id)
            for claim in claims:
                for chunk_id in claim.citation_chunk_ids:
                    if chunk_id not in cited_chunks:
                        issues.append(
                            StructuredParseIssue(
                                code="citation_record_missing",
                                message=(
                                    f"claim {claim.claim_id!r} cites chunk "
                                    f"{chunk_id!r} but no citation record carries "
                                    "that chunk."
                                ),
                                field="claims",
                            )
                        )

        abstention_reason = self._parse_abstention(payload, issues, status)

        if issues:
            return StructuredParseResult(issues=tuple(issues))

        return StructuredParseResult(
            answer=StructuredAnswer(
                status=status,
                answer_text=answer_text if isinstance(answer_text, str) else None,
                claims=claims,
                citations=citations,
                abstention_reason=abstention_reason,
                abstention_detail=(
                    payload.get("abstention_detail")
                    if isinstance(payload.get("abstention_detail"), str)
                    else None
                ),
            )
        )

    def _parse_status(
        self, payload: dict[str, object], issues: list[StructuredParseIssue]
    ) -> StructuredAnswerStatus:
        raw_status = payload.get("status")
        if not isinstance(raw_status, str):
            issues.append(
                StructuredParseIssue(
                    code="missing_field",
                    message="missing required field 'status'.",
                    field="status",
                )
            )
            return StructuredAnswerStatus.INVALID
        try:
            return StructuredAnswerStatus(raw_status)
        except ValueError:
            issues.append(
                StructuredParseIssue(
                    code="invalid_status",
                    message=(
                        f"invalid status {raw_status!r}; expected one of "
                        f"{[s.value for s in StructuredAnswerStatus]}."
                    ),
                    field="status",
                )
            )
            return StructuredAnswerStatus.INVALID

    def _parse_claims(
        self, payload: dict[str, object], issues: list[StructuredParseIssue]
    ) -> tuple[Claim, ...]:
        raw_claims = payload.get("claims")
        if raw_claims is None:
            return ()
        if not isinstance(raw_claims, list):
            issues.append(
                StructuredParseIssue(
                    code="invalid_field_type",
                    message="'claims' must be a list.",
                    field="claims",
                )
            )
            return ()
        claims: list[Claim] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                issues.append(
                    StructuredParseIssue(
                        code="invalid_field_type",
                        message=f"claim #{index} must be an object.",
                        field="claims",
                    )
                )
                continue
            missing = [f for f in _REQUIRED_CLAIM_FIELDS if f not in item]
            if missing:
                issues.append(
                    StructuredParseIssue(
                        code="missing_field",
                        message=(
                            f"claim #{index} missing required fields: "
                            + ", ".join(missing)
                        ),
                        field="claims",
                    )
                )
                continue
            claim_id = str(item["claim_id"])
            if not claim_id:
                issues.append(
                    StructuredParseIssue(
                        code="missing_claim_id",
                        message=f"claim #{index} has an empty claim_id.",
                        field="claims",
                    )
                )
            if claim_id in seen:
                issues.append(
                    StructuredParseIssue(
                        code="duplicate_claim_id",
                        message=f"duplicate claim id {claim_id!r}.",
                        field="claims",
                    )
                )
            seen.add(claim_id)
            citation_ids = item["citation_chunk_ids"]
            if not isinstance(citation_ids, list) or not all(
                isinstance(cid, str) for cid in citation_ids
            ):
                issues.append(
                    StructuredParseIssue(
                        code="invalid_field_type",
                        message=(
                            f"claim {claim_id!r} citation_chunk_ids must be a "
                            "list of strings."
                        ),
                        field="claims",
                    )
                )
                citation_ids = []
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=str(item["text"]),
                    citation_chunk_ids=tuple(citation_ids),
                )
            )
        return tuple(claims)

    def _parse_citations(
        self, payload: dict[str, object], issues: list[StructuredParseIssue]
    ) -> tuple[Citation, ...]:
        raw_citations = payload.get("citations")
        if raw_citations is None:
            return ()
        if not isinstance(raw_citations, list):
            issues.append(
                StructuredParseIssue(
                    code="invalid_field_type",
                    message="'citations' must be a list.",
                    field="citations",
                )
            )
            return ()
        citations: list[Citation] = []
        for index, item in enumerate(raw_citations):
            if not isinstance(item, dict):
                issues.append(
                    StructuredParseIssue(
                        code="invalid_field_type",
                        message=f"citation #{index} must be an object.",
                        field="citations",
                    )
                )
                continue
            missing = [f for f in _REQUIRED_CITATION_FIELDS if f not in item]
            if missing:
                issues.append(
                    StructuredParseIssue(
                        code="missing_field",
                        message=(
                            f"citation #{index} missing required fields: "
                            + ", ".join(missing)
                        ),
                        field="citations",
                    )
                )
                continue
            citations.append(
                Citation(
                    citation_id=str(item["citation_id"]),
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    corpus_version=str(item["corpus_version"]),
                    excerpt=str(item["excerpt"]),
                    location=str(item["location"]),
                )
            )
        return tuple(citations)

    def _parse_abstention(
        self,
        payload: dict[str, object],
        issues: list[StructuredParseIssue],
        status: StructuredAnswerStatus,
    ) -> str | None:
        if status in _ANSWERED_STATUSES:
            return None
        reason = payload.get("abstention_reason")
        if reason is None:
            if status is StructuredAnswerStatus.INVALID:
                return None
            issues.append(
                StructuredParseIssue(
                    code="missing_abstention_reason",
                    message=(f"{status.value} status requires abstention_reason."),
                    field="abstention_reason",
                )
            )
            return None
        if not isinstance(reason, str) or reason not in _VALID_ABSTENTION_STRINGS:
            issues.append(
                StructuredParseIssue(
                    code="invalid_abstention_reason",
                    message=(
                        f"invalid abstention_reason {reason!r}; expected one of "
                        f"{sorted(_VALID_ABSTENTION_STRINGS)}."
                    ),
                    field="abstention_reason",
                )
            )
            return None
        return reason


class StructuredAnswerValidator:
    def validate(
        self,
        answer: StructuredAnswer,
        pack: EvidencePack,
    ) -> tuple[StructuredParseIssue, ...]:
        if answer.status is not StructuredAnswerStatus.ANSWERED:
            return ()
        issues: list[StructuredParseIssue] = []

        for citation in answer.citations:
            item = pack.by_chunk_id(citation.chunk_id)
            if item is None:
                issues.append(
                    StructuredParseIssue(
                        code="citation_not_exposed",
                        message=(
                            f"citation {citation.citation_id!r} cites chunk "
                            f"{citation.chunk_id!r} which was not exposed to "
                            "the generator (not in the evidence pack)."
                        ),
                        field="citations",
                    )
                )
                continue
            if citation.corpus_version != pack.corpus_version:
                issues.append(
                    StructuredParseIssue(
                        code="corpus_version_mismatch",
                        message=(
                            f"citation {citation.citation_id!r} cites corpus "
                            f"version {citation.corpus_version!r} but the served "
                            f"corpus version is {pack.corpus_version!r}."
                        ),
                        field="citations",
                    )
                )
            if citation.excerpt != item.text:
                issues.append(
                    StructuredParseIssue(
                        code="excerpt_mismatch",
                        message=(
                            f"citation {citation.citation_id!r} excerpt is not "
                            "byte-identical to the exposed chunk text."
                        ),
                        field="citations",
                    )
                )
            if citation.location != item.location:
                issues.append(
                    StructuredParseIssue(
                        code="invalid_location",
                        message=(
                            f"citation {citation.citation_id!r} location "
                            f"{citation.location!r} does not match chunk location "
                            f"{item.location!r}."
                        ),
                        field="citations",
                    )
                )

        used_chunk_ids = {
            chunk_id for claim in answer.claims for chunk_id in claim.citation_chunk_ids
        }
        for citation in answer.citations:
            if citation.chunk_id not in used_chunk_ids:
                issues.append(
                    StructuredParseIssue(
                        code="unused_citation",
                        message=(
                            f"citation {citation.citation_id!r} is not referenced "
                            "by any claim."
                        ),
                        field="citations",
                    )
                )

        return tuple(issues)


_STATUS_TO_STATE = {
    StructuredAnswerStatus.ANSWERED: AnswerState.ANSWERED,
    StructuredAnswerStatus.ABSTAINED: AnswerState.ABSTAINED,
    StructuredAnswerStatus.CONFLICTING: AnswerState.CONFLICTING,
    StructuredAnswerStatus.INVALID: AnswerState.ABSTAINED,
}


def to_generated_answer(
    answer: StructuredAnswer,
    *,
    provider_id: str = "",
    model_id: str = "",
    mode: str = "",
) -> GeneratedAnswer:
    state = _STATUS_TO_STATE[answer.status]
    abstention_reason: AbstentionReason | None = None
    abstention_detail: str | None = answer.abstention_detail

    if state is AnswerState.ANSWERED:
        pass
    elif state is AnswerState.CONFLICTING:
        abstention_reason = AbstentionReason.CONFLICTING_EVIDENCE
    elif answer.status is StructuredAnswerStatus.INVALID:
        abstention_reason = AbstentionReason.MALFORMED_OUTPUT
        abstention_detail = (
            abstention_detail
            or "Generation provider returned status 'invalid'; no structured "
            "answer could be produced."
        )
    else:
        try:
            abstention_reason = (
                AbstentionReason(answer.abstention_reason)
                if answer.abstention_reason
                else None
            )
        except ValueError:
            abstention_reason = None
            abstention_detail = (
                abstention_detail
                or "Generation provider declared an unknown abstention reason."
            )
        if abstention_reason is None:
            abstention_reason = AbstentionReason.INSUFFICIENT_EVIDENCE

    return GeneratedAnswer(
        state=state,
        answer_text=answer.answer_text,
        claims=answer.claims,
        citations=answer.citations,
        abstention_reason=abstention_reason,
        abstention_detail=abstention_detail,
        provider_id=provider_id or answer.provider_id,
        model_id=model_id or answer.model_id,
        mode=mode or answer.mode,
        is_llm_generated=True,
    )
