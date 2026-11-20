"""CLI that materializes the real-corpus retrieval evaluation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_grounded_rag.corpus import (
    CorpusError,
    SourceManifestError,
    load_source_manifest,
)
from evidence_grounded_rag.corpus.normalize import sha256_hex
from evidence_grounded_rag.corpus.real_local import (
    RealLocalCorpus,
    build_real_local_corpus,
)
from evidence_grounded_rag.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from evidence_grounded_rag.evaluation.real_queries import (
    RealEvaluationQuery,
    RealQuerySet,
    load_real_query_set,
)
from evidence_grounded_rag.generation.extractive import ExtractiveTemplateGenerator
from evidence_grounded_rag.generation.models import AnswerState
from evidence_grounded_rag.pipeline import MIN_FUSED_SCORE
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    HybridRetriever,
)
from evidence_grounded_rag.validation import CitationValidator
from evidence_grounded_rag.versions import (
    BM25_VERSION,
    CORPUS_FORMAT_VERSION,
    EMBEDDING_MODEL_ID,
    FUSION_VERSION,
    GENERATOR_VERSION,
    NORMALIZATION_VERSION,
    PACKAGE_VERSION,
    SOURCE_MANIFEST_VERSION,
    TOKENIZER_VERSION,
)

REPORT_TYPE = "real-local-evaluation"
REPORT_VERSION = "1.0.0"

DENSE_MODE_LABEL = "Deterministic hash embedding diagnostic — not semantic retrieval"
HYBRID_MODE_LABEL = (
    "Hybrid diagnostic using BM25 plus deterministic non-semantic hash embeddings"
)
GENERATION_LABEL = "Offline deterministic extractive evaluation — not an LLM result"

MODE_LABELS: dict[str, str] = {
    MODE_SPARSE: "Sparse diagnostic using committed Okapi BM25",
    MODE_DENSE: DENSE_MODE_LABEL,
    MODE_HYBRID: HYBRID_MODE_LABEL,
}
VALID_MODES = (MODE_SPARSE, MODE_DENSE, MODE_HYBRID)

EMBEDDING_MODE = {
    "model_id": EMBEDDING_MODEL_ID,
    "mode": "offline",
    "semantic": False,
    "dimension": 256,
}
GENERATION_MODE = {
    "provider_id": "extractive-template",
    "model_id": GENERATOR_VERSION,
    "mode": "offline",
    "is_llm_generated": False,
}

DEFAULT_MANIFEST = Path("data/manifests/corpus-source-v1.0.0.json")
DEFAULT_QUERIES = Path("data/queries/python-docs-real-v1.0.0.json")
DEFAULT_RAW_DIR = Path("data/raw/python-docs-3.12.13")
DEFAULT_OUT = Path("data/evaluations/python-docs-real-evaluation-v1.0.0.json")

RECALL_KS = (1, 5, 10)
TOP_HIT_KS = (1, 5, 10)

GROUP_ANSWERABLE = "answerable"
GROUP_SINGLE_PASSAGE = "single_passage"
GROUP_MULTI_PASSAGE = "multi_passage"
GROUP_UNANSWERABLE = "unanswerable"
GROUP_AMBIGUOUS = "ambiguous"
GROUP_ABSTAIN = "abstain"
RETRIEVAL_GROUPS = (
    GROUP_ANSWERABLE,
    GROUP_SINGLE_PASSAGE,
    GROUP_MULTI_PASSAGE,
    GROUP_UNANSWERABLE,
    GROUP_AMBIGUOUS,
    GROUP_ABSTAIN,
)


@dataclass(frozen=True)
class QueryModeRecord:
    query_id: str
    case: str
    expected_status: str
    state: str
    abstention_reason: str | None
    abstention_detail: str | None
    retrieval_empty: bool
    retrieved_chunk_ids: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]
    claims: tuple[dict[str, object], ...]
    citations: tuple[dict[str, object], ...]
    validation: dict[str, object] | None
    claim_count: int
    cited_claim_count: int
    citation_completeness: float
    correct_abstention: bool
    reason_match: bool
    wrong_answer: bool
    trace_id: str
    corpus_manifest_id: str
    embedding_mode: dict[str, object]
    generation_mode: dict[str, object]

    @property
    def is_answered(self) -> bool:
        return self.state == AnswerState.ANSWERED.value

    @property
    def is_abstained(self) -> bool:
        return self.state in (
            AnswerState.ABSTAINED.value,
            AnswerState.CONFLICTING.value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "case": self.case,
            "expected_status": self.expected_status,
            "state": self.state,
            "abstention_reason": self.abstention_reason,
            "abstention_detail": self.abstention_detail,
            "retrieval_empty": self.retrieval_empty,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "claims": list(self.claims),
            "citations": list(self.citations),
            "validation": self.validation,
            "claim_count": self.claim_count,
            "cited_claim_count": self.cited_claim_count,
            "citation_completeness": self.citation_completeness,
            "correct_abstention": self.correct_abstention,
            "reason_match": self.reason_match,
            "wrong_answer": self.wrong_answer,
            "trace_id": self.trace_id,
            "corpus_manifest_id": self.corpus_manifest_id,
            "embedding_mode": self.embedding_mode,
            "generation_mode": self.generation_mode,
        }


def build_real_local(
    manifest_path: str | Path,
    raw_dir: str | Path,
) -> RealLocalCorpus:
    try:
        manifest = load_source_manifest(manifest_path)
        return build_real_local_corpus(manifest, raw_dir)
    except (SourceManifestError, CorpusError) as exc:
        raise RealLocalEvaluationError(f"{exc.code}: {exc}") from exc


class RealLocalEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RealEvaluationReport:
    data: dict[str, object]

    def deterministic_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.data.items()
            if key not in ("timestamp_utc", "deterministic_digest")
        }

    def deterministic_bytes(self) -> bytes:
        return _json_bytes(self.deterministic_payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.deterministic_bytes()).hexdigest()


def real_evaluate(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    query_set_path: str | Path = DEFAULT_QUERIES,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    out_path: str | Path | None = None,
    top_k: int = 10,
    modes: Sequence[str] = VALID_MODES,
    timestamp_utc: str | None = None,
) -> tuple[RealEvaluationReport, Path]:
    for mode in modes:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")

    corpus = build_real_local(manifest_path, raw_dir)
    manifest_id = Path(manifest_path).stem
    query_set = load_real_query_set(query_set_path, corpus, manifest_id=manifest_id)

    retriever = HybridRetriever(corpus)
    generator = ExtractiveTemplateGenerator()
    validator = CitationValidator(corpus)

    per_query: dict[str, dict[str, dict[str, object]]] = {mode: {} for mode in modes}
    traces: dict[str, dict[str, dict[str, object]]] = {mode: {} for mode in modes}
    failures: list[dict[str, object]] = []

    for mode in modes:
        for query in query_set.queries:
            record, trace = _run_query_mode(
                corpus,
                retriever,
                generator,
                validator,
                query,
                mode,
                top_k=top_k,
                min_score=MIN_FUSED_SCORE,
                corpus_manifest_id=query_set.corpus_manifest_id,
            )
            per_query[mode][query.query_id] = record.to_dict()
            traces[mode][query.query_id] = _quantize_trace(trace.to_dict())

    metrics = {
        mode: _compute_mode_metrics(
            query_set, [per_query[mode][q.query_id] for q in query_set.queries]
        )
        for mode in modes
    }
    failures.extend(_collect_failures(modes, per_query))

    data: dict[str, object] = {
        "report_type": REPORT_TYPE,
        "report_version": REPORT_VERSION,
        "labels": _labels(),
        "corpus": _corpus_facts(corpus, manifest_id),
        "query_set": _query_set_facts(query_set),
        "configs": _configs(top_k=top_k, modes=modes),
        "metrics": metrics,
        "per_query": per_query,
        "traces": traces,
        "failures": failures,
        "implementation_commit": _current_commit(),
        "timestamp_utc": timestamp_utc
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "deterministic_digest": "",
        "limitations": _limitations(),
    }
    report = RealEvaluationReport(data)
    report.data["deterministic_digest"] = report.digest

    resolved_out = Path(out_path or DEFAULT_OUT)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(_json_bytes(report.data).decode("utf-8"), encoding="utf-8")
    return report, resolved_out


def _run_query_mode(
    corpus: RealLocalCorpus,
    retriever: HybridRetriever,
    generator: ExtractiveTemplateGenerator,
    validator: CitationValidator,
    query: RealEvaluationQuery,
    mode: str,
    *,
    top_k: int,
    min_score: float,
    corpus_manifest_id: str,
) -> tuple[QueryModeRecord, Any]:
    hybrid = retriever.search(
        query.question, top_k=top_k, mode=mode, min_score=min_score
    )
    evidence = list(hybrid.fused.chunks)
    evidence_chunks = [
        chunk
        for item in evidence
        if (chunk := corpus.chunk_by_id(item.chunk_id)) is not None
    ]
    exposed_ids = {item.chunk_id for item in evidence}

    generated = generator.generate(query.question, evidence_chunks)

    if generated.state is AnswerState.ANSWERED and (
        not generated.answer_text or not generated.claims
    ):
        state = AnswerState.ABSTAINED.value
        reason = "malformed_output"
        detail = "Generation returned an answered state without answer text or claims."
        validation = None
    else:
        validation = validator.validate(generated, exposed_ids)
        if generated.state is not AnswerState.ANSWERED:
            state = generated.state.value
            reason = (
                generated.abstention_reason.value
                if generated.abstention_reason
                else None
            )
            detail = generated.abstention_detail
        elif not validation.valid:
            state = AnswerState.ABSTAINED.value
            reason = "citation_validation_failed"
            detail = (
                "Generated citations failed validation; answer refused. "
                + "; ".join(
                    f"{issue.code}: {issue.message}" for issue in validation.issues
                )
            )
        else:
            state = AnswerState.ANSWERED.value
            reason = None
            detail = None

    claims = [
        {
            "claim_id": claim.claim_id,
            "citation_chunk_ids": list(claim.citation_chunk_ids),
        }
        for claim in generated.claims
    ]
    citations = [
        _citation_to_dict(citation, corpus) for citation in generated.citations
    ]
    selected = sorted({citation.chunk_id for citation in generated.citations})
    cited_claims = sum(1 for claim in generated.claims if claim.citation_chunk_ids)
    total_claims = len(generated.claims)

    answered = state == AnswerState.ANSWERED.value
    abstained = state in (AnswerState.ABSTAINED.value, AnswerState.CONFLICTING.value)
    reason_match = (
        not query.is_answerable
        and reason is not None
        and reason in query.expected_abstention
    )
    correct_abstention = (not query.is_answerable) and abstained and reason_match
    wrong_answer = answered and not query.is_answerable

    return (
        QueryModeRecord(
            query_id=query.query_id,
            case=query.case,
            expected_status=query.expected_status,
            state=state,
            abstention_reason=reason,
            abstention_detail=(detail[:300] + "…")
            if detail and len(detail) > 300
            else detail,
            retrieval_empty=hybrid.fused.empty,
            retrieved_chunk_ids=tuple(item.chunk_id for item in evidence),
            selected_chunk_ids=tuple(selected),
            claims=tuple(claims),
            citations=tuple(citations),
            validation=_validation_to_dict(validation),
            claim_count=total_claims,
            cited_claim_count=cited_claims,
            citation_completeness=(cited_claims / total_claims)
            if total_claims
            else 0.0,
            correct_abstention=correct_abstention,
            reason_match=reason_match,
            wrong_answer=wrong_answer,
            trace_id=f"{query.query_id}::{mode}::trace",
            corpus_manifest_id=corpus_manifest_id,
            embedding_mode=dict(EMBEDDING_MODE),
            generation_mode=dict(GENERATION_MODE),
        ),
        hybrid.trace,
    )


def _citation_to_dict(citation: Any, corpus: RealLocalCorpus) -> dict[str, object]:
    chunk = corpus.chunk_by_id(citation.chunk_id)
    return {
        "citation_id": citation.citation_id,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "corpus_version": citation.corpus_version,
        "location": citation.location,
        "excerpt_sha256": sha256_hex(citation.excerpt),
        "excerpt_prefix": citation.excerpt[:120],
        "excerpt_matches": bool(chunk is not None and citation.excerpt == chunk.text),
    }


def _validation_to_dict(validation: Any) -> dict[str, object] | None:
    if validation is None:
        return None
    return {
        "valid": validation.valid,
        "skipped": validation.skipped,
        "issue_codes": list(validation.issue_codes),
        "checked_citations": validation.checked_citations,
        "checked_claims": validation.checked_claims,
    }


SCORE_QUANTIZATION_DECIMALS = 12


def _quantize_trace(trace: dict[str, object]) -> dict[str, object]:
    if isinstance(trace, dict):
        return {
            key: (_round_score(value) if key == "score" else _quantize_trace(value))
            for key, value in trace.items()
        }
    if isinstance(trace, list):
        return [_quantize_trace(item) for item in trace]
    return trace


def _round_score(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, SCORE_QUANTIZATION_DECIMALS)
    if isinstance(value, int):
        return value
    return value


def _compute_mode_metrics(
    query_set: RealQuerySet,
    records: list[dict[str, object]],
) -> dict[str, object]:
    overall = _group_retrieval_metrics(query_set, records, _all_in_group)
    groups = {
        group: _group_retrieval_metrics(query_set, records, _group_membership(group))
        for group in RETRIEVAL_GROUPS
    }
    return {
        "overall": overall,
        "groups": groups,
        "faithfulness": _faithfulness_metrics(query_set, records),
    }


def _all_in_group(_query: RealEvaluationQuery) -> bool:
    return True


def _group_membership(group: str):
    def predicate(query: RealEvaluationQuery) -> bool:
        if group == GROUP_ANSWERABLE:
            return query.is_answerable
        if group in (GROUP_UNANSWERABLE, GROUP_AMBIGUOUS, GROUP_ABSTAIN):
            return query.expected_status == group
        if group == GROUP_SINGLE_PASSAGE:
            return query.case == "single_passage"
        if group == GROUP_MULTI_PASSAGE:
            return query.case == "multi_passage"
        return False

    return predicate


def _group_retrieval_metrics(
    query_set: RealQuerySet,
    records: list[dict[str, object]],
    membership,
) -> dict[str, object]:
    members = [
        record for record in records if membership(query_set.by_id(record["query_id"]))
    ]
    with_relevance = [
        record
        for record in members
        if query_set.by_id(record["query_id"]).gold_chunk_ids
    ]

    def relevant(record: dict[str, object]) -> set[str]:
        return query_set.by_id(record["query_id"]).gold_chunk_ids

    ranked = [record["retrieved_chunk_ids"] for record in with_relevance]

    def macro_recall(k: int) -> float:
        values = [
            recall_at_k(ids, relevant(record), k)
            for record, ids in zip(with_relevance, ranked)
        ]
        return _mean(values)

    def macro_mrr() -> float:
        return _mean(
            mean_reciprocal_rank(ids, relevant(record))
            for record, ids in zip(with_relevance, ranked)
        )

    def macro_ndcg(k: int) -> float:
        values = [
            ndcg_at_k(
                ids,
                {
                    chunk_id: float(grade)
                    for chunk_id, grade in query_set.by_id(
                        record["query_id"]
                    ).relevance_by_chunk.items()
                },
                k,
            )
            for record, ids in zip(with_relevance, ranked)
        ]
        return _mean(values)

    top_hits = {
        k: sum(
            1
            for record in with_relevance
            if relevant(record) & set(record["retrieved_chunk_ids"][:k])
        )
        for k in TOP_HIT_KS
    }
    no_result = _ratio(
        sum(1 for record in members if record["retrieval_empty"]), len(members)
    )
    return {
        "num_queries": len(members),
        "num_with_relevance": len(with_relevance),
        **{f"recall@{k}": macro_recall(k) for k in RECALL_KS},
        "mrr": macro_mrr(),
        "ndcg@5": macro_ndcg(5),
        "no_result_rate": no_result,
        **{f"top{k}_hit_count": top_hits[k] for k in TOP_HIT_KS},
    }


def _faithfulness_metrics(
    query_set: RealQuerySet,
    records: list[dict[str, object]],
) -> dict[str, object]:
    answerable = [r for r in records if query_set.by_id(r["query_id"]).is_answerable]
    non_answerable = [
        r for r in records if not query_set.by_id(r["query_id"]).is_answerable
    ]
    abstentions = [
        r
        for r in records
        if r["state"] in (AnswerState.ABSTAINED.value, AnswerState.CONFLICTING.value)
    ]
    correct_abstentions = [r for r in records if r["correct_abstention"]]

    total_claims = sum(r["claim_count"] for r in records)
    unsupported_claims = sum(r["claim_count"] - r["cited_claim_count"] for r in records)
    checked_citations = sum(
        (r["validation"] or {}).get("checked_citations", 0) for r in records
    )
    valid_citations = sum(
        (r["validation"] or {}).get("checked_citations", 0)
        for r in records
        if r["validation"] is not None and r["validation"].get("valid")
    )
    validation_failures = sum(
        1
        for r in records
        if r["validation"] is not None and not r["validation"].get("valid")
    )

    return {
        "answerable_coverage": _ratio(
            sum(1 for r in answerable if r["state"] == AnswerState.ANSWERED.value),
            len(answerable),
        ),
        "abstention_precision": _ratio(len(correct_abstentions), len(abstentions)),
        "abstention_recall": _ratio(len(correct_abstentions), len(non_answerable)),
        "unsupported_claim_rate": _ratio(unsupported_claims, total_claims),
        "citation_precision": _ratio(valid_citations, checked_citations),
        "citation_completeness": _ratio(
            sum(r["cited_claim_count"] for r in records), total_claims
        ),
        "citation_validation_failure_rate": _ratio(validation_failures, len(records)),
        "wrong_answer_count": sum(1 for r in records if r["wrong_answer"]),
        "correct_abstention_count": len(correct_abstentions),
        "denominators": {
            "answerable": len(answerable),
            "non_answerable": len(non_answerable),
            "abstentions": len(abstentions),
            "checked_citations": checked_citations,
            "total_claims": total_claims,
            "queries": len(records),
        },
    }


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _labels() -> dict[str, str]:
    return {
        "dense_mode": DENSE_MODE_LABEL,
        "hybrid_mode": HYBRID_MODE_LABEL,
        "generation": GENERATION_LABEL,
        "bounded_scope": (
            "Bounded evaluation over three Python 3.12.13 documentation files and "
            "70 chunks; it does not represent all Python documentation."
        ),
        "non_semantic": (
            "Dense mode uses deterministic non-semantic hash embeddings; scores "
            "encode token overlap only, not meaning."
        ),
        "diagnostic_only": (
            "Generation is deterministic extraction, not an LLM; results are "
            "diagnostic, not production readiness."
        ),
        "separate_from_synthetic": (
            "This report is a separate real-local evaluation and is never combined "
            "with the synthetic report."
        ),
    }


def _corpus_facts(corpus: RealLocalCorpus, manifest_id: str) -> dict[str, object]:
    manifest = corpus.manifest
    documents = sorted(corpus.documents, key=lambda doc: doc.doc_id)
    canonical_checksum = hashlib.sha256(
        "\n".join(f"{doc.doc_id}:{doc.normalized_sha256}" for doc in documents).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "corpus_id": corpus.corpus_id,
        "corpus_version": corpus.corpus_version,
        "status": corpus.status,
        "manifest_id": manifest_id,
        "manifest_version": manifest.manifest_version,
        "publisher": manifest.publisher,
        "official_docs_url": manifest.official_docs_url,
        "source_url": manifest.source_url,
        "license": {
            "name": manifest.license.name,
            "version": manifest.license.version,
            "url": manifest.license.url,
        },
        "revision": manifest.revision,
        "revision_tag": manifest.revision_tag,
        "access_date": manifest.access_date,
        "acquisition_method": manifest.acquisition_method,
        "raw_files": [
            {
                "path": entry.path,
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for entry in manifest.expected_files
        ],
        "canonical_checksum": canonical_checksum,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "source_path": doc.source_path,
                "raw_size": doc.raw_size,
                "raw_sha256": doc.raw_sha256,
                "normalized_sha256": doc.normalized_sha256,
                "chunk_count": len(doc.chunks),
            }
            for doc in documents
        ],
        "document_count": len(corpus.documents),
        "chunk_count": len(corpus.chunks()),
        "chunks_per_document": {
            doc.doc_id: len(doc.chunks) for doc in corpus.documents
        },
        "redistribution_status": manifest.redistribution_status,
        "privacy_status": manifest.privacy_status,
    }


def _query_set_facts(query_set: RealQuerySet) -> dict[str, object]:
    return {
        "query_set_id": query_set.query_set_id,
        "query_set_version": query_set.query_set_version,
        "corpus_id": query_set.corpus_id,
        "corpus_version": query_set.corpus_version,
        "corpus_manifest_id": query_set.corpus_manifest_id,
        "synthetic": query_set.synthetic,
        "real_local": query_set.real_local,
        "num_queries": len(query_set.queries),
        "cases": _counts(query_set, "case"),
        "expected_statuses": _counts(query_set, "expected_status"),
    }


def _counts(query_set: RealQuerySet, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in query_set.queries:
        key = getattr(query, attr)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _configs(*, top_k: int, modes: Sequence[str]) -> dict[str, object]:
    return {
        "top_k": top_k,
        "min_fused_score": MIN_FUSED_SCORE,
        "bm25": {
            "version": BM25_VERSION,
            "k1": 1.5,
            "b": 0.75,
            "tokenizer_version": TOKENIZER_VERSION,
        },
        "fusion": {"version": FUSION_VERSION, "k": 60},
        "embedding": dict(EMBEDDING_MODE),
        "generation": dict(GENERATION_MODE),
        "modes": list(modes),
        "versions": {
            "package": PACKAGE_VERSION,
            "corpus_format": CORPUS_FORMAT_VERSION,
            "source_manifest": SOURCE_MANIFEST_VERSION,
            "normalization": NORMALIZATION_VERSION,
        },
    }


def _limitations() -> list[str]:
    return [
        (
            "Bounded evaluation over three Python 3.12.13 documentation files and "
            "70 chunks; it does not represent all Python documentation."
        ),
        (
            "Dense mode uses deterministic non-semantic hash embeddings; scores "
            "encode token overlap only, not meaning."
        ),
        "Generation is deterministic extractive/template output, not an LLM result.",
        "Results are diagnostic, not production readiness; no quality claim is made.",
        (
            "No chunk-level or claim-level text is stored in this report; citations "
            "carry excerpt checksums and short prefixes only, and abstention "
            "details are truncated."
        ),
        (
            "Denominators are small (up to 12 answerable and 6 non-answerable "
            "queries per mode); metric differences of one query move percentages "
            "by up to 8.3 percentage points."
        ),
        (
            "Raw retrieval scores in traces are quantized to 12 decimal places so "
            "two process runs serialize byte-identically; BM25's last-ulp float "
            "variance from hash-randomized summation order never changes rankings."
        ),
        (
            "Query wording is constrained to exact token forms present in the gold "
            "sentences because the committed extractive generator has no stemming."
        ),
    ]


def _collect_failures(
    modes: Sequence[str],
    per_query: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for mode in modes:
        for query_id, record in per_query[mode].items():
            if record["retrieval_empty"]:
                failures.append(
                    {
                        "kind": "empty_retrieval",
                        "mode": mode,
                        "query_id": query_id,
                    }
                )
            validation = record.get("validation")
            if validation is not None and not validation.get("valid"):
                failures.append(
                    {
                        "kind": "citation_validation_failed",
                        "mode": mode,
                        "query_id": query_id,
                        "issue_codes": validation.get("issue_codes"),
                    }
                )
            if record["wrong_answer"]:
                failures.append(
                    {
                        "kind": "wrong_answer",
                        "mode": mode,
                        "query_id": query_id,
                        "expected_status": record["expected_status"],
                    }
                )
            if record["expected_status"] == "answerable" and record["state"] in (
                AnswerState.ABSTAINED.value,
                AnswerState.CONFLICTING.value,
            ):
                failures.append(
                    {
                        "kind": "answerable_abstention",
                        "mode": mode,
                        "query_id": query_id,
                        "reason": record["abstention_reason"],
                    }
                )
    return failures


def _current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _json_bytes(data: dict[str, object]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the separate real-local diagnostic evaluation over the approved "
            "Python 3.12.13 documentation corpus and write its own report. Never "
            "uses the network and never falls back to the synthetic corpus."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=Path)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), type=Path)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), type=Path)
    parser.add_argument("--output", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        print("error: --top-k must be a positive integer", file=sys.stderr)
        return 2

    try:
        report, out_path = real_evaluate(
            args.manifest,
            args.queries,
            args.raw_dir,
            out_path=args.output,
            top_k=args.top_k,
        )
    except (RealLocalEvaluationError, ValueError, OSError) as exc:
        print(f"REAL EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"wrote real-local evaluation report: {out_path}")
    print(
        f"corpus: {report.data['corpus']['corpus_id']} v{report.data['corpus']['corpus_version']} "
        f"({report.data['corpus']['document_count']} docs / {report.data['corpus']['chunk_count']} chunks)"
    )
    print(
        f"query set: {report.data['query_set']['query_set_id']} v{report.data['query_set']['query_set_version']} "
        f"— {report.data['query_set']['num_queries']} queries"
    )
    for mode, metrics in report.data["metrics"].items():
        overall = metrics["overall"]
        faith = metrics["faithfulness"]
        print(f"[{mode}]")
        print(
            f"  recall@1={overall['recall@1']:.4f} recall@5={overall['recall@5']:.4f} "
            f"recall@10={overall['recall@10']:.4f} mrr={overall['mrr']:.4f} ndcg@5={overall['ndcg@5']:.4f}"
        )
        print(
            f"  no_result_rate={overall['no_result_rate']:.4f} "
            f"answerable_coverage={faith['answerable_coverage']:.4f} "
            f"abstention_recall={faith['abstention_recall']:.4f} "
            f"wrong_answer_count={faith['wrong_answer_count']}"
        )
    print(f"digest={report.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
