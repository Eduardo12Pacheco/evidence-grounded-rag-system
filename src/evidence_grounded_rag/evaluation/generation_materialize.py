from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.evaluation.real_materialize import (
    RealLocalEvaluationError,
    _citation_to_dict,
    _corpus_facts,
    _current_commit,
    _json_bytes,
    _query_set_facts,
    build_real_local,
)
from evidence_grounded_rag.evaluation.real_queries import (
    RealQuerySet,
    load_real_query_set,
)
from evidence_grounded_rag.generation import (
    EvidencePack,
    ExtractiveTemplateGenerator,
)
from evidence_grounded_rag.generation.models import AnswerState
from evidence_grounded_rag.generation.support import (
    ClaimSupport,
    ClaimSupportEvaluation,
    ClaimSupportEvaluator,
)
from evidence_grounded_rag.pipeline import MIN_FUSED_SCORE
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    HybridRetriever,
)
from evidence_grounded_rag.validation import CitationValidator
from evidence_grounded_rag.versions import (
    ANSWER_EVALUATION_FIXTURE_VERSION,
    BM25_VERSION,
    CLAIM_SUPPORT_VERSION,
    CORPUS_FORMAT_VERSION,
    EMBEDDING_MODEL_ID,
    FUSION_VERSION,
    GENERATION_EVALUATION_VERSION,
    GENERATION_PROMPT_VERSION,
    GENERATOR_VERSION,
    NORMALIZATION_VERSION,
    PACKAGE_VERSION,
    SOURCE_MANIFEST_VERSION,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    TOKENIZER_VERSION,
)

REPORT_TYPE = "real-local-generation-evaluation"
REPORT_VERSION = GENERATION_EVALUATION_VERSION

GENERATION_LABEL = "Offline deterministic extractive evaluation — not an LLM result"
LIVE_GENERATION_LABEL = (
    "Live generation NOT measured — no provider/model/credentials are "
    "configured and no provider is selected; the offline report covers "
    "deterministic extractive generation only."
)

DEFAULT_MANIFEST = Path("data/manifests/corpus-source-v1.0.0.json")
DEFAULT_QUERIES = Path("data/evaluation/queries/python-docs-real-v1.0.0.json")
DEFAULT_RAW_DIR = Path("data/raw/python-docs-3.12.13")
DEFAULT_FIXTURE = Path("data/fixtures/answer-evaluation-v1.0.0.json")
DEFAULT_OUT = Path(
    "data/evaluations/python-docs-real-generation-evaluation-v1.0.0.json"
)

_FIXTURE_FIELDS = (
    "fixture_id",
    "fixture_version",
    "corpus_id",
    "corpus_version",
    "query_set_id",
    "query_set_version",
    "queries",
)
_FIXTURE_QUERY_FIELDS = ("query_id", "expected_status", "gold_chunk_ids")

SCORE_QUANTIZATION_DECIMALS = 12


@dataclass(frozen=True)
class GenerationEvaluationReport:
    data: dict[str, object]

    def deterministic_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.data.items()
            if key not in ("timestamp_utc", "deterministic_digest", "runtime_seconds")
        }

    def deterministic_bytes(self) -> bytes:
        return _json_bytes(self.deterministic_payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.deterministic_bytes()).hexdigest()


def load_answer_evaluation_fixture(
    path: str | Path,
    corpus: Any,
    query_set: RealQuerySet,
) -> dict[str, object]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"answer-evaluation fixture file not found: {fixture_path}"
        )
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("answer-evaluation fixture root must be an object")
    missing = [f for f in _FIXTURE_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            "answer-evaluation fixture missing required fields: " + ", ".join(missing)
        )
    if str(raw["fixture_version"]) != ANSWER_EVALUATION_FIXTURE_VERSION:
        raise ValueError(
            f"fixture version {raw['fixture_version']!r} does not match "
            f"expected {ANSWER_EVALUATION_FIXTURE_VERSION!r}"
        )
    if str(raw["corpus_id"]) != getattr(corpus, "corpus_id", None):
        raise ValueError(
            f"fixture corpus_id {raw['corpus_id']!r} does not match corpus "
            f"{getattr(corpus, 'corpus_id', None)!r}"
        )
    if str(raw["corpus_version"]) != getattr(corpus, "corpus_version", None):
        raise ValueError(
            f"fixture corpus_version {raw['corpus_version']!r} does not match "
            f"corpus version {getattr(corpus, 'corpus_version', None)!r}"
        )
    if str(raw["query_set_id"]) != query_set.query_set_id:
        raise ValueError(
            f"fixture query_set_id {raw['query_set_id']!r} does not match "
            f"query set {query_set.query_set_id!r}"
        )
    if str(raw["query_set_version"]) != query_set.query_set_version:
        raise ValueError(
            f"fixture query_set_version {raw['query_set_version']!r} does not "
            f"match query set {query_set.query_set_version!r}"
        )

    fixture_queries = raw.get("queries")
    if not isinstance(fixture_queries, list):
        raise TypeError("answer-evaluation fixture 'queries' must be a list")

    fixture_by_id: dict[str, dict[str, object]] = {}
    for index, record in enumerate(fixture_queries):
        if not isinstance(record, dict):
            raise TypeError(f"fixture query #{index} must be an object")
        for field in _FIXTURE_QUERY_FIELDS:
            if field not in record:
                raise ValueError(f"fixture query #{index} missing field {field!r}")
        query_id = str(record["query_id"])
        if query_id in fixture_by_id:
            raise ValueError(f"duplicate fixture query_id {query_id!r}")
        fixture_by_id[query_id] = record

    fixture_ids = set(fixture_by_id)
    query_set_ids = {query.query_id for query in query_set.queries}
    if fixture_ids != query_set_ids:
        missing_ids = sorted(query_set_ids - fixture_ids)
        extra_ids = sorted(fixture_ids - query_set_ids)
        raise ValueError(
            "answer-evaluation fixture must reference exactly the unchanged "
            f"real query set; missing from fixture: {missing_ids}; "
            f"not in query set: {extra_ids}"
        )

    for query in query_set.queries:
        record = fixture_by_id[query.query_id]
        expected_status = str(record["expected_status"])
        if expected_status != query.expected_status:
            raise ValueError(
                f"fixture expected_status for {query.query_id!r} is "
                f"{expected_status!r} but the query set declares "
                f"{query.expected_status!r}"
            )
        gold_raw = record.get("gold_chunk_ids")
        if not isinstance(gold_raw, list):
            raise TypeError(
                f"fixture query {query.query_id!r} gold_chunk_ids must be a list"
            )
        gold = {str(chunk_id) for chunk_id in gold_raw}
        for chunk_id in gold:
            if corpus.chunk_by_id(chunk_id) is None:
                raise ValueError(
                    f"fixture query {query.query_id!r} references unknown chunk "
                    f"{chunk_id!r}"
                )
        if gold != query.gold_chunk_ids:
            raise ValueError(
                f"fixture gold chunks for {query.query_id!r} do not match the "
                "unchanged query set"
            )
    return raw


def generation_evaluate(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    query_set_path: str | Path = DEFAULT_QUERIES,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    fixture_path: str | Path = DEFAULT_FIXTURE,
    out_path: str | Path | None = None,
    top_k: int = 10,
    mode: str = MODE_HYBRID,
    timestamp_utc: str | None = None,
) -> tuple[GenerationEvaluationReport, Path]:
    if mode not in (MODE_SPARSE, MODE_DENSE, MODE_HYBRID):
        raise ValueError(
            f"unknown mode {mode!r}; expected one of "
            f"{[MODE_SPARSE, MODE_DENSE, MODE_HYBRID]}"
        )

    started = time.monotonic()
    corpus = build_real_local(manifest_path, raw_dir)
    manifest_id = Path(manifest_path).stem
    query_set = load_real_query_set(query_set_path, corpus, manifest_id=manifest_id)
    load_answer_evaluation_fixture(fixture_path, corpus, query_set)

    retriever = HybridRetriever(corpus)
    generator = ExtractiveTemplateGenerator()
    validator = CitationValidator(corpus)
    support_evaluator = ClaimSupportEvaluator()

    per_query: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []

    for query in query_set.queries:
        record, record_failures = _run_query(
            corpus,
            retriever,
            generator,
            validator,
            support_evaluator,
            query,
            mode,
            top_k=top_k,
            min_score=MIN_FUSED_SCORE,
            corpus_manifest_id=manifest_id,
        )
        per_query[query.query_id] = record
        failures.extend(record_failures)

    metrics = _compute_metrics(query_set, per_query)
    runtime_seconds = time.monotonic() - started

    data: dict[str, object] = {
        "report_type": REPORT_TYPE,
        "report_version": REPORT_VERSION,
        "labels": _labels(),
        "corpus": _corpus_facts(corpus, manifest_id),
        "query_set": _query_set_facts(query_set),
        "fixture": _fixture_facts(fixture_path),
        "configs": _configs(top_k=top_k, mode=mode),
        "metrics": metrics,
        "per_query": per_query,
        "failures": failures,
        "implementation_commit": _current_commit(),
        "timestamp_utc": timestamp_utc
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "deterministic_digest": "",
        "limitations": _limitations(),
    }
    report = GenerationEvaluationReport(data)
    report.data["deterministic_digest"] = report.digest

    resolved_out = Path(out_path or DEFAULT_OUT)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(_json_bytes(report.data).decode("utf-8"), encoding="utf-8")
    return report, resolved_out


def _run_query(
    corpus: Any,
    retriever: HybridRetriever,
    generator: ExtractiveTemplateGenerator,
    validator: CitationValidator,
    support_evaluator: ClaimSupportEvaluator,
    query: Any,
    mode: str,
    *,
    top_k: int,
    min_score: float,
    corpus_manifest_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    hybrid = retriever.search(
        query.question, top_k=top_k, mode=mode, min_score=min_score
    )
    evidence = list(hybrid.fused.chunks)
    pack = EvidencePack.from_scored_chunks(corpus, evidence, trace=hybrid.trace)
    exposed_ids = {item.chunk_id for item in evidence}

    generated = generator.generate(query.question, pack)

    failures: list[dict[str, object]] = []
    provider_error = (
        generated.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    )
    invalid_output = generated.abstention_reason is AbstentionReason.MALFORMED_OUTPUT

    validation = validator.validate(generated, exposed_ids)
    if generated.state is AnswerState.ANSWERED and (
        not generated.answer_text or not generated.claims
    ):
        state = AnswerState.ABSTAINED.value
        reason = "malformed_output"
        detail = "Generation returned an answered state without answer text or claims."
        validation = None
        invalid_output = True
    elif generated.state is not AnswerState.ANSWERED:
        state = generated.state.value
        reason = (
            generated.abstention_reason.value if generated.abstention_reason else None
        )
        detail = generated.abstention_detail
    elif not validation.valid:
        state = AnswerState.ABSTAINED.value
        reason = "citation_validation_failed"
        detail = "Generated citations failed validation; answer refused. " + "; ".join(
            f"{issue.code}: {issue.message}" for issue in validation.issues
        )
    else:
        state = AnswerState.ANSWERED.value
        reason = None
        detail = None

    support = (
        support_evaluator.evaluate(generated, pack)
        if not provider_error
        else ClaimSupportEvaluation()
    )

    answered = state == AnswerState.ANSWERED.value
    abstained = state in (AnswerState.ABSTAINED.value, AnswerState.CONFLICTING.value)
    reason_match = (
        not query.is_answerable
        and reason is not None
        and reason in query.expected_abstention
    )
    correct_abstention = (not query.is_answerable) and abstained and reason_match
    wrong_answer = answered and not query.is_answerable

    claims = [
        {
            "claim_id": claim.claim_id,
            "citation_chunk_ids": list(claim.citation_chunk_ids),
        }
        for claim in generated.claims
    ]
    claim_support = [
        {
            "claim_id": result.claim_id,
            "support": result.support.value,
            "cited_chunk_ids": list(result.cited_chunk_ids),
            "matched_chunk_id": result.matched_chunk_id,
        }
        for result in support.results
    ]
    citations = [
        _citation_to_dict(citation, corpus) for citation in generated.citations
    ]
    total_claims = len(generated.claims)
    cited_claims = sum(1 for claim in generated.claims if claim.citation_chunk_ids)

    if hybrid.fused.empty:
        failures.append(
            {"kind": "empty_retrieval", "mode": mode, "query_id": query.query_id}
        )
    if validation is not None and not validation.valid:
        failures.append(
            {
                "kind": "citation_validation_failed",
                "mode": mode,
                "query_id": query.query_id,
                "issue_codes": list(validation.issue_codes),
            }
        )
    if provider_error:
        failures.append(
            {
                "kind": "provider_error",
                "mode": mode,
                "query_id": query.query_id,
                "reason": reason,
            }
        )
    if invalid_output:
        failures.append(
            {
                "kind": "invalid_output",
                "mode": mode,
                "query_id": query.query_id,
                "reason": reason,
            }
        )
    if wrong_answer:
        failures.append(
            {
                "kind": "wrong_answer",
                "mode": mode,
                "query_id": query.query_id,
                "expected_status": query.expected_status,
            }
        )
    if query.expected_status == "answerable" and state in (
        AnswerState.ABSTAINED.value,
        AnswerState.CONFLICTING.value,
    ):
        failures.append(
            {
                "kind": "answerable_abstention",
                "mode": mode,
                "query_id": query.query_id,
                "reason": reason,
            }
        )

    record: dict[str, object] = {
        "query_id": query.query_id,
        "case": query.case,
        "expected_status": query.expected_status,
        "state": state,
        "abstention_reason": reason,
        "abstention_detail": (detail[:300] + "…")
        if detail and len(detail) > 300
        else detail,
        "answerable": query.is_answerable,
        "answered": answered,
        "abstained": abstained,
        "correct_abstention": correct_abstention,
        "reason_match": reason_match,
        "wrong_answer": wrong_answer,
        "provider_error": provider_error,
        "invalid_output": invalid_output,
        "retrieval_empty": hybrid.fused.empty,
        "retrieved_chunk_ids": [item.chunk_id for item in evidence],
        "evidence_ranks": [
            {
                "chunk_id": item.chunk_id,
                "rank": item.rank,
                "score": round(item.score, SCORE_QUANTIZATION_DECIMALS),
            }
            for item in evidence
        ],
        "claims": claims,
        "claim_support": claim_support,
        "claim_support_rule": support.rule,
        "citations": citations,
        "validation": _validation_to_dict(validation),
        "claim_count": total_claims,
        "cited_claim_count": cited_claims,
        "citation_completeness": (cited_claims / total_claims) if total_claims else 0.0,
        "generation": {
            "provider_id": generated.provider_id,
            "model_id": generated.model_id,
            "mode": generated.mode,
            "is_llm_generated": generated.is_llm_generated,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        },
        "corpus_manifest_id": corpus_manifest_id,
    }
    return record, failures


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


def _compute_metrics(
    query_set: RealQuerySet,
    per_query: dict[str, dict[str, object]],
) -> dict[str, object]:
    records = [per_query[q.query_id] for q in query_set.queries]
    answerable = [r for r in records if r["answerable"]]
    non_answerable = [r for r in records if not r["answerable"]]
    abstentions = [r for r in records if r["abstained"]]
    correct_abstentions = [r for r in records if r["correct_abstention"]]
    answered = [r for r in records if r["answered"]]

    total_claims = sum(r["claim_count"] for r in records)
    cited_claims = sum(r["cited_claim_count"] for r in records)
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

    supported_claims = sum(
        1
        for r in records
        for c in r["claim_support"]
        if c["support"] == ClaimSupport.SUPPORTED.value
    )
    unsupported_claims = sum(
        1
        for r in records
        for c in r["claim_support"]
        if c["support"] == ClaimSupport.UNSUPPORTED.value
    )
    unknown_claims = sum(
        1
        for r in records
        for c in r["claim_support"]
        if c["support"] == ClaimSupport.UNKNOWN.value
    )

    return {
        "answer_coverage": _ratio(len(answered), len(answerable)),
        "abstention_rate": _ratio(len(abstentions), len(records)),
        "abstention_precision": _ratio(len(correct_abstentions), len(abstentions)),
        "abstention_recall": _ratio(len(correct_abstentions), len(non_answerable)),
        "correct_abstention_count": len(correct_abstentions),
        "wrong_answer_count": sum(1 for r in records if r["wrong_answer"]),
        "citation_precision": _ratio(valid_citations, checked_citations),
        "citation_completeness": _ratio(cited_claims, total_claims),
        "citation_validation_failure_rate": _ratio(validation_failures, len(records)),
        "claim_support": {
            "rule": CLAIM_SUPPORT_VERSION,
            "supported": supported_claims,
            "unsupported": unsupported_claims,
            "unknown": unknown_claims,
            "total": total_claims,
            "support_rate": _ratio(supported_claims, total_claims),
            "evaluated_rate": _ratio(
                supported_claims + unsupported_claims, total_claims
            ),
        },
        "invalid_output_count": sum(1 for r in records if r["invalid_output"]),
        "provider_error_count": sum(1 for r in records if r["provider_error"]),
        "denominators": {
            "answerable": len(answerable),
            "non_answerable": len(non_answerable),
            "abstentions": len(abstentions),
            "checked_citations": checked_citations,
            "total_claims": total_claims,
            "queries": len(records),
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _labels() -> dict[str, str]:
    return {
        "generation": GENERATION_LABEL,
        "live_generation": LIVE_GENERATION_LABEL,
        "bounded_scope": (
            "Bounded evaluation over three Python 3.12.13 documentation files "
            "and 70 chunks; it does not represent all Python documentation."
        ),
        "non_semantic": (
            "Dense mode uses deterministic non-semantic hash embeddings; scores "
            "encode token overlap only, not meaning."
        ),
        "diagnostic_only": (
            "Generation is deterministic extraction, not an LLM; results are "
            "diagnostic, not production readiness."
        ),
        "separate_from_other_reports": (
            "This report is a separate real-local generation evaluation and is "
            "never combined with the synthetic, hash-materializer, or semantic "
            "materializer reports."
        ),
        "no_live_claim": (
            "Live generation is not measured here: no provider/model/"
            "credentials are configured, so no live result exists to report."
        ),
    }


def _fixture_facts(fixture_path: Path) -> dict[str, object]:
    raw = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    return {
        "fixture_id": raw.get("fixture_id"),
        "fixture_version": raw.get("fixture_version"),
        "query_set_id": raw.get("query_set_id"),
        "query_set_version": raw.get("query_set_version"),
        "num_queries": len(raw.get("queries") or []),
    }


def _configs(*, top_k: int, mode: str) -> dict[str, object]:
    return {
        "top_k": top_k,
        "min_fused_score": MIN_FUSED_SCORE,
        "mode": mode,
        "bm25": {
            "version": BM25_VERSION,
            "k1": 1.5,
            "b": 0.75,
            "tokenizer_version": TOKENIZER_VERSION,
        },
        "fusion": {"version": FUSION_VERSION, "k": 60},
        "embedding": {
            "model_id": EMBEDDING_MODEL_ID,
            "mode": "offline",
            "semantic": False,
        },
        "generation": {
            "provider_id": "extractive-template",
            "model_id": GENERATOR_VERSION,
            "mode": "offline",
            "is_llm_generated": False,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        },
        "claim_support_rule": CLAIM_SUPPORT_VERSION,
        "versions": {
            "package": PACKAGE_VERSION,
            "corpus_format": CORPUS_FORMAT_VERSION,
            "source_manifest": SOURCE_MANIFEST_VERSION,
            "normalization": NORMALIZATION_VERSION,
            "answer_evaluation_fixture": ANSWER_EVALUATION_FIXTURE_VERSION,
        },
    }


def _limitations() -> list[str]:
    return [
        (
            "Bounded evaluation over three Python 3.12.13 documentation files "
            "and 70 chunks; it does not represent all Python documentation."
        ),
        (
            "Generation is deterministic extractive/template output, not an "
            "LLM result; live generation is NOT measured because no provider/"
            "model/credentials are configured."
        ),
        (
            "The claim-support rule is exact normalized containment only: a "
            "claim is supported iff its normalized text is a substring of a "
            "cited chunk's normalized text. This measures verbatim grounding, "
            "not semantic entailment."
        ),
        "Results are diagnostic, not production readiness; no quality claim is made.",
        (
            "No chunk-level or claim-level text is stored in this report; "
            "claims carry ids and support verdicts, citations carry excerpt "
            "checksums and short prefixes, and abstention details are truncated."
        ),
        (
            "Denominators are small (12 answerable and 6 non-answerable "
            "queries); one query moves percentages by up to 8.3 percentage points."
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the separate real-local generation evaluation over the "
            "approved Python 3.12.13 documentation corpus and write its own "
            "report. Fully offline, deterministic extraction; live generation "
            "is not measured. Never falls back to the synthetic corpus."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=Path)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), type=Path)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), type=Path)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), type=Path)
    parser.add_argument("--output", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mode",
        default=MODE_HYBRID,
        choices=(MODE_SPARSE, MODE_DENSE, MODE_HYBRID),
    )
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        print("error: --top-k must be a positive integer", file=sys.stderr)
        return 2

    try:
        report, out_path = generation_evaluate(
            args.manifest,
            args.queries,
            args.raw_dir,
            fixture_path=args.fixture,
            out_path=args.output,
            top_k=args.top_k,
            mode=args.mode,
        )
    except (RealLocalEvaluationError, ValueError, TypeError, OSError) as exc:
        print(f"GENERATION EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"wrote real-local generation evaluation report: {out_path}")
    print(
        f"corpus: {report.data['corpus']['corpus_id']} "
        f"v{report.data['corpus']['corpus_version']}"
    )
    print(
        f"query set: {report.data['query_set']['query_set_id']} "
        f"v{report.data['query_set']['query_set_version']} "
        f"— {report.data['query_set']['num_queries']} queries"
    )
    metrics = report.data["metrics"]
    print(f"  answer_coverage={metrics['answer_coverage']:.4f}")
    print(
        f"  abstention_precision={metrics['abstention_precision']:.4f} "
        f"abstention_recall={metrics['abstention_recall']:.4f}"
    )
    print(
        f"  citation_precision={metrics['citation_precision']:.4f} "
        f"citation_completeness={metrics['citation_completeness']:.4f}"
    )
    support = metrics["claim_support"]
    print(
        f"  claim_support={support['supported']}/{support['total']} "
        f"(rate={support['support_rate']:.4f})"
    )
    print(
        f"  invalid_output_count={metrics['invalid_output_count']} "
        f"provider_error_count={metrics['provider_error_count']}"
    )
    print(f"digest={report.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
