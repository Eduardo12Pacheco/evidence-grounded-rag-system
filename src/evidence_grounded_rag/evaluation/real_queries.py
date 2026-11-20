"""Real-local query-set models, validation, and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evidence_grounded_rag.corpus.ids import validate_chunk_id

MIN_REAL_QUERIES = 15

VALID_EXPECTED_STATUSES = frozenset(
    {"answerable", "unanswerable", "ambiguous", "abstain"}
)

REQUIRED_CASES = frozenset(
    {
        "single_passage",
        "multi_passage",
        "exact_api",
        "terminology",
        "numeric_parameter",
        "unanswerable",
        "ambiguous",
        "abstain",
        "citation_boundary",
        "section_distinction",
    }
)

VALID_CASES = frozenset(
    {
        "single_passage",
        "multi_passage",
        "exact_api",
        "terminology",
        "numeric_parameter",
        "unanswerable",
        "ambiguous",
        "abstain",
        "citation_boundary",
        "section_distinction",
    }
)

VALID_ABSTENTION_REASONS = frozenset(
    {
        "insufficient_evidence",
        "unsupported_query",
        "conflicting_evidence",
        "citation_validation_failed",
    }
)

_TOP_LEVEL_FIELDS = (
    "query_set_id",
    "query_set_version",
    "corpus_id",
    "corpus_version",
    "corpus_manifest_id",
    "synthetic",
    "real_local",
    "queries",
)
_QUERY_FIELDS = (
    "query_id",
    "case",
    "question",
    "expected_status",
    "gold_relevant",
    "expected_citation_count",
    "expected_abstention",
    "notes",
)


@dataclass(frozen=True)
class RealGoldJudgment:
    chunk_id: str
    grade: int


@dataclass(frozen=True)
class CitationExpectation:
    min: int
    max: int


@dataclass(frozen=True)
class RealEvaluationQuery:
    query_id: str
    case: str
    question: str
    expected_status: str
    gold_relevant: tuple[RealGoldJudgment, ...] = field(default_factory=tuple)
    expected_citation_count: CitationExpectation | None = None
    expected_abstention: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    annotation: dict[str, str] = field(default_factory=dict)

    @property
    def gold_chunk_ids(self) -> set[str]:
        return {judgment.chunk_id for judgment in self.gold_relevant}

    @property
    def is_answerable(self) -> bool:
        return self.expected_status == "answerable"

    @property
    def relevance_by_chunk(self) -> dict[str, int]:
        return {judgment.chunk_id: judgment.grade for judgment in self.gold_relevant}


@dataclass(frozen=True)
class RealQuerySet:
    query_set_id: str
    query_set_version: str
    corpus_id: str
    corpus_version: str
    corpus_manifest_id: str
    synthetic: bool
    real_local: bool
    queries: tuple[RealEvaluationQuery, ...]

    def by_id(self, query_id: str) -> RealEvaluationQuery | None:
        for query in self.queries:
            if query.query_id == query_id:
                return query
        return None

    def cases_present(self) -> set[str]:
        return {query.case for query in self.queries}

    def statuses_present(self) -> set[str]:
        return {query.expected_status for query in self.queries}


def load_real_query_set(
    path: str | Path,
    corpus: Any,
    *,
    manifest_id: str | None = None,
) -> RealQuerySet:
    query_path = Path(path)
    if not query_path.is_file():
        raise FileNotFoundError(f"real query set file not found: {query_path}")
    raw = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("real query set root must be an object")

    missing = [field for field in _TOP_LEVEL_FIELDS if field not in raw]
    if missing:
        raise ValueError(
            "real query set missing required fields: " + ", ".join(missing)
        )

    queries_raw = raw.get("queries")
    if not isinstance(queries_raw, list):
        raise TypeError("real query set 'queries' must be a list")
    if len(queries_raw) < MIN_REAL_QUERIES:
        raise ValueError(
            f"real query set must contain at least {MIN_REAL_QUERIES} queries, "
            f"got {len(queries_raw)}"
        )

    queries = tuple(
        _parse_real_query(index, record, corpus)
        for index, record in enumerate(queries_raw)
    )
    _validate_unique_query_ids(queries)
    _validate_case_coverage(queries)

    corpus_id = str(raw["corpus_id"])
    corpus_version = str(raw["corpus_version"])
    corpus_manifest_id = str(raw["corpus_manifest_id"])

    if corpus_id != getattr(corpus, "corpus_id", None):
        raise ValueError(
            f"query set corpus_id {corpus_id!r} does not match corpus "
            f"{getattr(corpus, 'corpus_id', None)!r}"
        )
    if corpus_version != getattr(corpus, "corpus_version", None):
        raise ValueError(
            f"query set corpus_version {corpus_version!r} does not match corpus "
            f"version {getattr(corpus, 'corpus_version', None)!r}"
        )
    if manifest_id is not None and corpus_manifest_id != manifest_id:
        raise ValueError(
            f"query set corpus_manifest_id {corpus_manifest_id!r} does not match "
            f"the manifest id {manifest_id!r}"
        )

    if bool(raw.get("synthetic", False)):
        raise ValueError("a real query set must declare synthetic: false")
    if not bool(raw.get("real_local", False)):
        raise ValueError("a real query set must declare real_local: true")

    return RealQuerySet(
        query_set_id=str(raw["query_set_id"]),
        query_set_version=str(raw["query_set_version"]),
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        corpus_manifest_id=corpus_manifest_id,
        synthetic=False,
        real_local=True,
        queries=queries,
    )


def _parse_real_query(
    index: int,
    record: Any,
    corpus: Any,
) -> RealEvaluationQuery:
    if not isinstance(record, dict):
        raise TypeError(f"real query #{index} must be an object")
    missing = [field for field in _QUERY_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"real query #{index} missing required fields: " + ", ".join(missing)
        )

    query_id = str(record["query_id"])
    case = str(record["case"])
    if case not in VALID_CASES:
        raise ValueError(
            f"query {query_id!r} has unknown case {case!r}; expected one of "
            f"{sorted(VALID_CASES)}"
        )
    status = str(record["expected_status"])
    if status not in VALID_EXPECTED_STATUSES:
        raise ValueError(
            f"query {query_id!r} has invalid expected_status {status!r}; expected "
            f"one of {sorted(VALID_EXPECTED_STATUSES)}"
        )

    gold = tuple(
        sorted(
            (
                _parse_gold_judgment(query_id, item, corpus)
                for item in (record.get("gold_relevant") or [])
            ),
            key=lambda judgment: judgment.chunk_id,
        )
    )
    if status == "answerable" and not gold:
        raise ValueError(
            f"answerable query {query_id!r} must declare at least one gold relevant chunk"
        )
    if status != "answerable" and gold:
        raise ValueError(
            f"non-answerable query {query_id!r} must not declare gold relevant chunks"
        )

    expected_citation_count = _parse_citation_expectation(
        query_id, record.get("expected_citation_count")
    )
    if status != "answerable" and expected_citation_count not in (
        None,
        CitationExpectation(0, 0),
    ):
        raise ValueError(
            f"non-answerable query {query_id!r} must expect zero citations"
        )

    abstention = tuple(
        sorted(str(r) for r in (record.get("expected_abstention") or []))
    )
    unknown_reasons = [r for r in abstention if r not in VALID_ABSTENTION_REASONS]
    if unknown_reasons:
        raise ValueError(
            f"query {query_id!r} declares unknown abstention reasons: "
            + ", ".join(unknown_reasons)
        )
    if status != "answerable" and not abstention:
        raise ValueError(
            f"non-answerable query {query_id!r} must declare expected abstention reason(s)"
        )
    if status == "answerable" and abstention:
        raise ValueError(
            f"answerable query {query_id!r} must not declare expected abstention reasons"
        )

    annotation_raw = record.get("annotation") or {}
    annotation = {
        str(key): str(value)
        for key, value in annotation_raw.items()
        if isinstance(annotation_raw, dict)
    }

    return RealEvaluationQuery(
        query_id=query_id,
        case=case,
        question=str(record["question"]),
        expected_status=status,
        gold_relevant=gold,
        expected_citation_count=expected_citation_count,
        expected_abstention=abstention,
        notes=str(record.get("notes", "")),
        annotation=annotation,
    )


def _parse_gold_judgment(
    query_id: str,
    item: Any,
    corpus: Any,
) -> RealGoldJudgment:
    if not isinstance(item, dict) or not isinstance(item.get("chunk_id"), str):
        raise TypeError(f"query {query_id!r} has a malformed gold relevance entry")
    chunk_id = item["chunk_id"]
    validate_chunk_id(chunk_id)
    grade_raw = item.get("grade", 1)
    if not isinstance(grade_raw, int) or not (1 <= grade_raw <= 3):
        raise ValueError(
            f"query {query_id!r} gold chunk {chunk_id!r} must have an integer grade "
            f"in 1..3, got {grade_raw!r}"
        )
    if corpus.chunk_by_id(chunk_id) is None:
        raise ValueError(
            f"query {query_id!r} references unknown chunk {chunk_id!r}; the chunk "
            "does not exist in the real-local corpus"
        )
    return RealGoldJudgment(chunk_id=chunk_id, grade=grade_raw)


def _parse_citation_expectation(query_id: str, raw: Any) -> CitationExpectation | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"query {query_id!r} expected_citation_count must be an object")
    try:
        minimum = int(raw.get("min", 0))
        maximum = int(raw.get("max", minimum))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"query {query_id!r} expected_citation_count must be integers"
        ) from exc
    if minimum < 0 or maximum < minimum:
        raise ValueError(
            f"query {query_id!r} expected_citation_count must satisfy 0 <= min <= max"
        )
    return CitationExpectation(min=minimum, max=maximum)


def _validate_unique_query_ids(queries: tuple[RealEvaluationQuery, ...]) -> None:
    seen: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            raise ValueError(f"duplicate real query_id {query.query_id!r}")
        seen.add(query.query_id)


def _validate_case_coverage(queries: tuple[RealEvaluationQuery, ...]) -> None:
    present_cases = {query.case for query in queries}
    missing_cases = sorted(REQUIRED_CASES - present_cases)
    if missing_cases:
        raise ValueError(
            "real query set must cover every required case; missing: "
            + ", ".join(missing_cases)
        )
    present_statuses = {query.expected_status for query in queries}
    missing_statuses = sorted(VALID_EXPECTED_STATUSES - present_statuses)
    if missing_statuses:
        raise ValueError(
            "real query set must cover every expected_status; missing: "
            + ", ".join(missing_statuses)
        )
