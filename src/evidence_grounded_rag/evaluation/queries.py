"""Versioned, committed evaluation query set.

The query set is a fixture artifact: it is committed, versioned, and frozen
against a specific corpus version. It deliberately covers answerable,
unanswerable, ambiguous, conflicting, citation-boundary, numeric, and
multi-passage cases so abstention behavior is measured, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUERY_CATEGORIES = frozenset(
    {
        "answerable",
        "unanswerable",
        "ambiguous",
        "conflicting",
        "citation_boundary",
        "numeric",
        "multi_passage",
    }
)

_EXPECTED_STATES = frozenset({"answered", "abstained"})


@dataclass(frozen=True)
class RelevanceJudgment:
    chunk_id: str
    grade: int


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    category: str
    question: str
    answerable: bool
    expected_state: str
    expected_reason: str | None
    relevance: tuple[RelevanceJudgment, ...] = field(default_factory=tuple)

    @property
    def relevant_chunk_ids(self) -> set[str]:
        return {judgment.chunk_id for judgment in self.relevance}

    @property
    def relevance_by_chunk(self) -> dict[str, int]:
        return {judgment.chunk_id: judgment.grade for judgment in self.relevance}


@dataclass(frozen=True)
class QuerySet:
    query_set_id: str
    query_set_version: str
    corpus_id: str
    corpus_version: str
    synthetic: bool
    queries: tuple[EvaluationQuery, ...]

    def by_id(self, query_id: str) -> EvaluationQuery | None:
        for query in self.queries:
            if query.query_id == query_id:
                return query
        return None

    def categories_present(self) -> set[str]:
        return {query.category for query in self.queries}


def load_query_set(path: str | Path) -> QuerySet:
    query_path = Path(path)
    if not query_path.is_file():
        raise FileNotFoundError(f"query set file not found: {query_path}")
    raw = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("query set root must be an object")

    for field_name in ("query_set_id", "query_set_version", "corpus_id", "corpus_version"):
        if field_name not in raw:
            raise ValueError(f"query set missing required field {field_name!r}")

    queries_raw = raw.get("queries")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ValueError("query set must contain a non-empty 'queries' list")

    queries = tuple(_parse_query(index, record) for index, record in enumerate(queries_raw))
    _validate_unique_ids(queries)
    _validate_categories(queries)
    return QuerySet(
        query_set_id=str(raw["query_set_id"]),
        query_set_version=str(raw["query_set_version"]),
        corpus_id=str(raw["corpus_id"]),
        corpus_version=str(raw["corpus_version"]),
        synthetic=bool(raw.get("synthetic", False)),
        queries=queries,
    )


def _parse_query(index: int, record: dict[str, Any]) -> EvaluationQuery:
    for field_name in ("query_id", "category", "question", "expected_state"):
        if field_name not in record:
            raise ValueError(f"query #{index} missing required field {field_name!r}")
    category = str(record["category"])
    if category not in QUERY_CATEGORIES:
        raise ValueError(
            f"query {record['query_id']!r} has unknown category {category!r}; "
            f"expected one of {sorted(QUERY_CATEGORIES)}"
        )
    expected_state = str(record["expected_state"])
    if expected_state not in _EXPECTED_STATES:
        raise ValueError(
            f"query {record['query_id']!r} has invalid expected_state "
            f"{expected_state!r}"
        )
    relevance_raw = record.get("relevance") or []
    relevance = tuple(
        RelevanceJudgment(
            chunk_id=str(judgment.get("chunk_id", "")),
            grade=int(judgment.get("grade", 1)),
        )
        for judgment in relevance_raw
        if isinstance(judgment, dict)
    )
    if expected_state == "answered" and not relevance:
        raise ValueError(
            f"answered query {record['query_id']!r} must declare relevance judgments"
        )
    return EvaluationQuery(
        query_id=str(record["query_id"]),
        category=category,
        question=str(record["question"]),
        answerable=bool(record.get("answerable", False)),
        expected_state=expected_state,
        expected_reason=(
            str(record["expected_reason"]) if record.get("expected_reason") else None
        ),
        relevance=relevance,
    )


def _validate_unique_ids(queries: tuple[EvaluationQuery, ...]) -> None:
    seen: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            raise ValueError(f"duplicate query_id {query.query_id!r}")
        seen.add(query.query_id)


def _validate_categories(queries: tuple[EvaluationQuery, ...]) -> None:
    present = {query.category for query in queries}
    missing = sorted(QUERY_CATEGORIES - present)
    if missing:
        raise ValueError(
            "query set must cover every category; missing: " + ", ".join(missing)
        )
