from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evidence_grounded_rag.abstention import AbstentionReason, reason_detail
from evidence_grounded_rag.corpus import Corpus, CorpusError, load_corpus
from evidence_grounded_rag.embeddings import EmbeddingProvider
from evidence_grounded_rag.generation import (
    EnvConfiguredGenerator,
    Generator,
)
from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)
from evidence_grounded_rag.retrieval import (
    MODE_HYBRID,
    HybridRetriever,
    HybridTrace,
    ScoredChunk,
)
from evidence_grounded_rag.validation import CitationValidationResult, CitationValidator
from evidence_grounded_rag.versions import PACKAGE_VERSION

MIN_FUSED_SCORE = 0.01

_GATE_ERROR_TO_REASON: dict[str, AbstentionReason] = {
    "blocked_real_corpus": AbstentionReason.BLOCKED_CORPUS,
    "unapproved_license": AbstentionReason.BLOCKED_CORPUS,
    "unsupported_format": AbstentionReason.INVALID_CORPUS,
    "checksum_mismatch": AbstentionReason.INVALID_CORPUS,
    "duplicate_document_id": AbstentionReason.INVALID_CORPUS,
    "duplicate_chunk_id": AbstentionReason.INVALID_CORPUS,
}


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 5
    mode: str = MODE_HYBRID
    min_fused_score: float = MIN_FUSED_SCORE
    expected_corpus_version: str | None = None


@dataclass(frozen=True)
class AnswerRecord:
    query: str
    state: AnswerState
    answer_text: str | None
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    abstention_reason: AbstentionReason | None = None
    abstention_detail: str | None = None
    evidence: tuple[ScoredChunk, ...] = field(default_factory=tuple)
    trace: HybridTrace | None = None
    validation: CitationValidationResult | None = None
    generation: dict[str, object] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)

    @property
    def is_answered(self) -> bool:
        return self.state is AnswerState.ANSWERED

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "state": self.state.value,
            "answer_text": self.answer_text,
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "citation_chunk_ids": list(claim.citation_chunk_ids),
                }
                for claim in self.claims
            ],
            "citations": [
                {
                    "citation_id": citation.citation_id,
                    "chunk_id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "corpus_version": citation.corpus_version,
                    "excerpt": citation.excerpt,
                    "location": citation.location,
                }
                for citation in self.citations
            ],
            "abstention_reason": (
                self.abstention_reason.value if self.abstention_reason else None
            ),
            "abstention_detail": self.abstention_detail,
            "evidence": [
                {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "rank": item.rank,
                    "score": item.score,
                }
                for item in self.evidence
            ],
            "trace": self.trace.to_dict() if self.trace else None,
            "validation": _validation_to_dict(self.validation),
            "generation": self.generation,
            "versions": self.versions,
            "config": self.config,
        }


def _validation_to_dict(
    validation: CitationValidationResult | None,
) -> dict[str, object] | None:
    if validation is None:
        return None
    return {
        "valid": validation.valid,
        "skipped": validation.skipped,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "chunk_id": issue.chunk_id,
                "claim_id": issue.claim_id,
            }
            for issue in validation.issues
        ],
        "checked_citations": validation.checked_citations,
        "checked_claims": validation.checked_claims,
    }


class EvidenceRagPipeline:
    def __init__(
        self,
        corpus_or_path: Corpus | str | Path,
        *,
        retriever: HybridRetriever | None = None,
        generator: Generator | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.corpus: Corpus | None = None
        self.retriever: HybridRetriever | None = None
        self.validator: CitationValidator | None = None
        self._gate_failure: tuple[AbstentionReason, str] | None = None

        if isinstance(corpus_or_path, Corpus):
            self.corpus = corpus_or_path
        else:
            try:
                self.corpus = load_corpus(corpus_or_path)
            except CorpusError as exc:
                reason = _GATE_ERROR_TO_REASON.get(
                    exc.code, AbstentionReason.INVALID_CORPUS
                )
                self._gate_failure = (reason, str(exc))

        if self.corpus is not None:
            if (
                self.config.expected_corpus_version
                and self.corpus.corpus_version != self.config.expected_corpus_version
            ):
                self._gate_failure = (
                    AbstentionReason.STALE_CORPUS,
                    (
                        f"loaded corpus version {self.corpus.corpus_version!r} "
                        f"does not match expected {self.config.expected_corpus_version!r}"
                    ),
                )
            else:
                self.retriever = retriever or HybridRetriever(
                    self.corpus, embedding_provider=embedding_provider
                )
                self.validator = CitationValidator(self.corpus)

        self.generator = generator or EnvConfiguredGenerator()

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> AnswerRecord:
        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_mode = mode if mode is not None else self.config.mode

        if self._gate_failure is not None:
            return self._abstained_record(
                query, *self._gate_failure, evidence=(), trace=None
            )
        assert self.corpus is not None
        assert self.retriever is not None
        assert self.validator is not None

        hybrid = self.retriever.search(
            query,
            top_k=effective_top_k,
            mode=effective_mode,
            min_score=self.config.min_fused_score,
        )

        evidence: list[ScoredChunk] = list(hybrid.fused.chunks)
        evidence_pack = EvidencePack.from_scored_chunks(
            self.corpus, evidence, trace=hybrid.trace
        )
        exposed_ids = {item.chunk_id for item in evidence}

        generated = self.generator.generate(query, evidence_pack)

        if generated.state is AnswerState.ANSWERED and (
            not generated.answer_text or not generated.claims
        ):
            return self._abstained_record(
                query,
                AbstentionReason.MALFORMED_OUTPUT,
                "The generation provider returned an answered state without "
                "answer text or claims.",
                evidence=tuple(evidence),
                trace=hybrid.trace,
                generation=self._generation_info(generated),
                versions={
                    "package": PACKAGE_VERSION,
                    **hybrid.trace.versions,
                    "generation": generated.model_id,
                    "generator_provider": generated.provider_id,
                },
                config={
                    **hybrid.trace.config,
                    "expected_corpus_version": self.config.expected_corpus_version,
                },
            )

        validation = self.validator.validate(generated, exposed_ids)

        versions = {
            "package": PACKAGE_VERSION,
            **hybrid.trace.versions,
            "generation": generated.model_id,
            "generator_provider": generated.provider_id,
        }
        config = {
            **hybrid.trace.config,
            "expected_corpus_version": self.config.expected_corpus_version,
        }

        if generated.state is not AnswerState.ANSWERED:
            reason = (
                generated.abstention_reason or AbstentionReason.INSUFFICIENT_EVIDENCE
            )
            return self._record(
                query,
                state=generated.state,
                answer_text=None,
                claims=generated.claims,
                citations=generated.citations,
                abstention_reason=reason,
                abstention_detail=generated.abstention_detail or reason_detail(reason),
                evidence=tuple(evidence),
                trace=hybrid.trace,
                validation=validation,
                generation=self._generation_info(generated),
                versions=versions,
                config=config,
            )

        if not validation.valid:
            detail = (
                "Generated citations failed validation; answer refused. "
                + "; ".join(
                    f"{issue.code}: {issue.message}" for issue in validation.issues
                )
            )
            return self._abstained_record(
                query,
                AbstentionReason.CITATION_VALIDATION_FAILED,
                detail,
                evidence=tuple(evidence),
                trace=hybrid.trace,
                validation=validation,
                generation=self._generation_info(generated),
                versions=versions,
                config=config,
            )

        return self._record(
            query,
            state=AnswerState.ANSWERED,
            answer_text=generated.answer_text,
            claims=generated.claims,
            citations=generated.citations,
            evidence=tuple(evidence),
            trace=hybrid.trace,
            validation=validation,
            generation=self._generation_info(generated),
            versions=versions,
            config=config,
        )

    def corpus_status(self) -> dict[str, object]:
        if self.corpus is None or self._gate_failure is not None:
            reason, detail = self._gate_failure or (
                AbstentionReason.INVALID_CORPUS,
                "Corpus not loaded.",
            )
            return {
                "loaded": False,
                "reason": reason.value,
                "detail": detail,
            }
        return {
            "loaded": True,
            "corpus_id": self.corpus.corpus_id,
            "corpus_version": self.corpus.corpus_version,
            "synthetic": self.corpus.synthetic,
            "real_corpus": self.corpus.real_corpus,
            "provenance": self.corpus.provenance,
            "license": {
                "status": self.corpus.license_info.status,
                "terms": self.corpus.license_info.terms,
                "redistribution": self.corpus.license_info.redistribution,
            },
            "real_corpus_gate": {
                "blocked": self.corpus.real_corpus_gate.blocked,
                "blocker": self.corpus.real_corpus_gate.blocker,
            },
            "num_documents": len(self.corpus.documents),
            "num_chunks": len(self.corpus.chunks()),
        }

    @staticmethod
    def _generation_info(generated: GeneratedAnswer) -> dict[str, object]:
        return {
            "provider_id": generated.provider_id,
            "model_id": generated.model_id,
            "mode": generated.mode,
            "is_llm_generated": generated.is_llm_generated,
        }

    def _abstained_record(
        self,
        query: str,
        reason: AbstentionReason,
        detail: str,
        *,
        evidence: tuple[ScoredChunk, ...],
        trace: HybridTrace | None,
        validation: CitationValidationResult | None = None,
        generation: dict[str, object] | None = None,
        versions: dict[str, str] | None = None,
        config: dict[str, object] | None = None,
    ) -> AnswerRecord:
        return self._record(
            query,
            state=AnswerState.ABSTAINED,
            answer_text=None,
            abstention_reason=reason,
            abstention_detail=detail,
            evidence=evidence,
            trace=trace,
            validation=validation,
            generation=generation
            or {"provider_id": "none", "model_id": "none", "is_llm_generated": False},
            versions=versions
            or {
                "package": PACKAGE_VERSION,
                "corpus_id": self.corpus.corpus_id if self.corpus else "",
                "corpus_version": self.corpus.corpus_version if self.corpus else "",
            },
            config=config
            or {
                "mode": self.config.mode,
                "top_k": self.config.top_k,
                "min_fused_score": self.config.min_fused_score,
                "expected_corpus_version": self.config.expected_corpus_version,
            },
        )

    @staticmethod
    def _record(
        query: str,
        *,
        state: AnswerState,
        answer_text: str | None,
        claims: Sequence[Claim] = (),
        citations: Sequence[Citation] = (),
        abstention_reason: AbstentionReason | None = None,
        abstention_detail: str | None = None,
        evidence: tuple[ScoredChunk, ...] = (),
        trace: HybridTrace | None = None,
        validation: CitationValidationResult | None = None,
        generation: dict[str, object] | None = None,
        versions: dict[str, str] | None = None,
        config: dict[str, object] | None = None,
    ) -> AnswerRecord:
        return AnswerRecord(
            query=query,
            state=state,
            answer_text=answer_text,
            claims=tuple(claims),
            citations=tuple(citations),
            abstention_reason=abstention_reason,
            abstention_detail=abstention_detail,
            evidence=tuple(evidence),
            trace=trace,
            validation=validation,
            generation=generation or {},
            versions=versions or {},
            config=config or {},
        )
