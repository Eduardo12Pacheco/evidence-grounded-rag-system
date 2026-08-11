"""Citation validation: strict, deterministic, fail-closed.

An answered generation is only ever reported as verified if EVERY check below
passes. On any failure ``valid`` is False and the pipeline refuses the answer
(converting it to an abstention with reason ``citation_validation_failed``).
Validation is skipped (``skipped=True``) for abstained and conflicting answers:
a refusal carries no citations, so there is nothing to verify and it is never
presented as a verified answer.

Rules (each maps to a stable issue code):

- ``citation_not_found``: citation chunk does not exist in the corpus.
- ``citation_not_exposed``: citation chunk was not part of the retrieved
  evidence handed to the generator.
- ``corpus_version_mismatch``: citation chunk belongs to a different corpus
  version than the one being served.
- ``excerpt_mismatch``: the cited excerpt is not byte-for-byte identical to the
  chunk text.
- ``invalid_location``: the citation's location is missing or does not match
  the chunk's stable ``synthetic://`` location.
- ``claim_without_citation``: a non-abstained claim cites nothing.
- ``unused_citation``: a citation is not referenced by any claim (orphaned /
  forged citation).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from evidence_grounded_rag.corpus.models import Corpus
from evidence_grounded_rag.generation.models import AnswerState, GeneratedAnswer


@dataclass(frozen=True)
class CitationIssue:
    code: str
    message: str
    chunk_id: str | None = None
    claim_id: str | None = None


@dataclass(frozen=True)
class CitationValidationResult:
    valid: bool
    skipped: bool
    issues: tuple[CitationIssue, ...] = field(default_factory=tuple)
    checked_citations: int = 0
    checked_claims: int = 0

    @property
    def issue_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


class CitationValidator:
    """Validates generated citations against the served corpus and evidence."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def validate(
        self,
        answer: GeneratedAnswer,
        exposed_chunk_ids: Iterable[str],
    ) -> CitationValidationResult:
        exposed = set(exposed_chunk_ids)

        if answer.state is not AnswerState.ANSWERED:
            return CitationValidationResult(valid=True, skipped=True)

        issues: list[CitationIssue] = []

        for claim in answer.claims:
            if not claim.citation_chunk_ids:
                issues.append(
                    CitationIssue(
                        code="claim_without_citation",
                        message=f"claim {claim.claim_id} carries no citation",
                        claim_id=claim.claim_id,
                    )
                )

        for citation in answer.citations:
            chunk = self.corpus.chunk_by_id(citation.chunk_id)
            if chunk is None:
                issues.append(
                    CitationIssue(
                        code="citation_not_found",
                        message=f"citation {citation.citation_id} refers to "
                        f"unknown chunk {citation.chunk_id!r}",
                        chunk_id=citation.chunk_id,
                    )
                )
                continue
            if citation.chunk_id not in exposed:
                issues.append(
                    CitationIssue(
                        code="citation_not_exposed",
                        message=f"citation {citation.citation_id} cites chunk "
                        f"{citation.chunk_id!r} which was not exposed to the generator",
                        chunk_id=citation.chunk_id,
                    )
                )
            if citation.corpus_version != self.corpus.corpus_version:
                issues.append(
                    CitationIssue(
                        code="corpus_version_mismatch",
                        message=f"citation {citation.citation_id} cites corpus "
                        f"version {citation.corpus_version!r} but served corpus "
                        f"version is {self.corpus.corpus_version!r}",
                        chunk_id=citation.chunk_id,
                    )
                )
            if citation.excerpt != chunk.text:
                issues.append(
                    CitationIssue(
                        code="excerpt_mismatch",
                        message=f"citation {citation.citation_id} excerpt does not "
                        "match the chunk text exactly",
                        chunk_id=citation.chunk_id,
                    )
                )
            if citation.location != chunk.location:
                issues.append(
                    CitationIssue(
                        code="invalid_location",
                        message=f"citation {citation.citation_id} location "
                        f"{citation.location!r} does not match chunk location "
                        f"{chunk.location!r}",
                        chunk_id=citation.chunk_id,
                    )
                )

        used_chunk_ids = {
            chunk_id for claim in answer.claims for chunk_id in claim.citation_chunk_ids
        }
        for citation in answer.citations:
            if citation.chunk_id not in used_chunk_ids:
                issues.append(
                    CitationIssue(
                        code="unused_citation",
                        message=f"citation {citation.citation_id} is not "
                        "referenced by any claim",
                        chunk_id=citation.chunk_id,
                    )
                )

        return CitationValidationResult(
            valid=not issues,
            skipped=False,
            issues=tuple(issues),
            checked_citations=len(answer.citations),
            checked_claims=len(answer.claims),
        )
