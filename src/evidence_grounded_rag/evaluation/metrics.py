from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.evaluation.queries import EvaluationQuery, QuerySet
from evidence_grounded_rag.generation.models import AnswerState
from evidence_grounded_rag.pipeline import AnswerRecord

NDCG_KS = (5, 10)
RECALL_KS = (1, 5, 10)


def recall_at_k(
    ranked_ids: Sequence[str], relevant: Sequence[str] | set[str], k: int
) -> float:
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(
    ranked_ids: Sequence[str], relevant: Sequence[str] | set[str]
) -> float:
    relevant = set(relevant)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    if not relevance or k <= 0:
        return 0.0
    dcg = 0.0
    for index, chunk_id in enumerate(ranked_ids[:k], start=1):
        dcg += relevance.get(chunk_id, 0.0) / math.log2(index + 1)
    ideal = sorted(relevance.values(), reverse=True)
    idcg = sum(
        grade / math.log2(index + 1) for index, grade in enumerate(ideal[:k], start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    category: str
    answerable: bool
    expected_state: str
    expected_reason: str | None
    state: AnswerState
    abstention_reason: AbstentionReason | None
    answered: bool
    abstained: bool
    correct_abstention: bool
    reason_match: bool
    retrieval_ids: tuple[str, ...]
    relevance: tuple[str, ...]

    @property
    def has_relevance(self) -> bool:
        return bool(self.relevance)


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict[str, float]
    per_query: tuple[QueryOutcome, ...]
    versions: dict[str, str]
    config: dict[str, object]
    labels: dict[str, object]
    limitations: tuple[str, ...]
    timestamp_utc: str
    generation: dict[str, object] = field(default_factory=dict)
    answers: dict[str, AnswerRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "per_query": [
                {
                    "query_id": outcome.query_id,
                    "category": outcome.category,
                    "answerable": outcome.answerable,
                    "expected_state": outcome.expected_state,
                    "expected_reason": outcome.expected_reason,
                    "state": outcome.state.value,
                    "abstention_reason": (
                        outcome.abstention_reason.value
                        if outcome.abstention_reason
                        else None
                    ),
                    "answered": outcome.answered,
                    "abstained": outcome.abstained,
                    "correct_abstention": outcome.correct_abstention,
                    "reason_match": outcome.reason_match,
                    "retrieval_ids": list(outcome.retrieval_ids),
                    "relevance": list(outcome.relevance),
                }
                for outcome in self.per_query
            ],
            "versions": self.versions,
            "config": self.config,
            "labels": self.labels,
            "limitations": list(self.limitations),
            "timestamp_utc": self.timestamp_utc,
            "generation": self.generation,
        }


def evaluate_answers(
    query_set: QuerySet,
    records: Sequence[AnswerRecord],
) -> EvaluationReport:
    if len(records) != len(query_set.queries):
        raise ValueError(
            f"expected {len(query_set.queries)} answers, got {len(records)}"
        )

    outcomes = [
        _outcome(query, record) for query, record in zip(query_set.queries, records)
    ]

    with_relevance = [o for o in outcomes if o.has_relevance]

    def macro_recall(k: int) -> float:
        values = [
            recall_at_k(record.retrieval_ids, record.relevance, k)
            for record in with_relevance
        ]
        return _mean(values)

    def macro_mrr() -> float:
        return _mean(
            mean_reciprocal_rank(record.retrieval_ids, record.relevance)
            for record in with_relevance
        )

    def macro_ndcg(k: int) -> float:
        values = [
            ndcg_at_k(
                record.retrieval_ids,
                {
                    chunk_id: float(grade)
                    for chunk_id, grade in _relevance_grades(
                        query_set, record.query_id
                    ).items()
                },
                k,
            )
            for record in with_relevance
        ]
        return _mean(values)

    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    abstentions = [o for o in outcomes if o.abstained]
    correct_abstentions = [o for o in outcomes if o.correct_abstention]

    answered_claims = sum(len(record.claims) for record in records)
    unsupported_claims = sum(_count_unsupported_claims(record) for record in records)
    citations_checked = sum(
        record.validation.checked_citations for record in records if record.validation
    )
    valid_citations = sum(
        record.validation.checked_citations
        for record in records
        if record.validation and record.validation.valid
    )
    claims_with_citation = sum(
        sum(1 for claim in record.claims if claim.citation_chunk_ids)
        for record in records
    )

    metrics: dict[str, float] = {
        **{f"recall@{k}": macro_recall(k) for k in RECALL_KS},
        "mrr": macro_mrr(),
        **{f"ndcg@{k}": macro_ndcg(k) for k in NDCG_KS},
        "answerable_coverage": _ratio(
            sum(1 for o in answerable if o.answered), len(answerable)
        ),
        "abstention_rate": _ratio(len(abstentions), len(outcomes)),
        "abstention_precision": _ratio(len(correct_abstentions), len(abstentions)),
        "abstention_recall": _ratio(len(correct_abstentions), len(unanswerable)),
        "unsupported_claim_rate": _ratio(unsupported_claims, answered_claims),
        "citation_precision": _ratio(valid_citations, citations_checked),
        "citation_completeness": _ratio(claims_with_citation, answered_claims),
    }

    return EvaluationReport(
        metrics=metrics,
        per_query=tuple(outcomes),
        versions={},
        config={},
        labels={
            "synthetic_evaluation": True,
            "corpus_provenance": query_set.synthetic and "project-generated-synthetic",
        },
        limitations=(
            (
                "Synthetic fixture evaluation only: numbers describe the "
                "committed fixture corpus and query set, not any real corpus."
            ),
            (
                "The deterministic embedding is non-semantic; dense scores "
                "encode token overlap only."
            ),
            (
                "No claims of real-world validation, perfect metrics, or "
                "hallucination-free behavior are made."
            ),
        ),
        timestamp_utc="",
        generation={},
        answers={record.query: record for record in records},
    )


def _outcome(query: EvaluationQuery, record: AnswerRecord) -> QueryOutcome:
    answered = record.state is AnswerState.ANSWERED
    abstained = record.state in (AnswerState.ABSTAINED, AnswerState.CONFLICTING)
    reason_match = (
        query.expected_reason is None
        or (record.abstention_reason.value if record.abstention_reason else None)
        == query.expected_reason
    )
    correct_abstention = (not query.answerable) and abstained and reason_match
    return QueryOutcome(
        query_id=query.query_id,
        category=query.category,
        answerable=query.answerable,
        expected_state=query.expected_state,
        expected_reason=query.expected_reason,
        state=record.state,
        abstention_reason=record.abstention_reason,
        answered=answered,
        abstained=abstained,
        correct_abstention=correct_abstention,
        reason_match=reason_match,
        retrieval_ids=tuple(item.chunk_id for item in record.evidence),
        relevance=tuple(sorted(query.relevant_chunk_ids)),
    )


def _relevance_grades(query_set: QuerySet, query_id: str) -> dict[str, int]:
    query = query_set.by_id(query_id)
    return {
        chunk_id: grade
        for chunk_id, grade in (query.relevance_by_chunk.items() if query else {})
    }


def _count_unsupported_claims(record: AnswerRecord) -> int:
    if not record.validation or record.validation.skipped:
        return 0
    return sum(1 for claim in record.claims if not claim.citation_chunk_ids) + sum(
        1 for issue in record.validation.issues if issue.code == "excerpt_mismatch"
    )


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
