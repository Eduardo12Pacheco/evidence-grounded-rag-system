"""Deterministic evaluation materialization.

Runs the full pipeline over the committed synthetic query set against the
committed synthetic fixture corpus and writes a versioned JSON report into
``data/manifests/``. The report records versions, configuration, model
identities, generation provenance, per-query outcomes, metrics, a timestamp,
and explicit limitations, and labels every number as synthetic-fixture
evaluation.

Command:

    uv run python -m evidence_grounded_rag.evaluation.materialize

Optional flags:

    --manifest <path>   corpus manifest (default: committed synthetic fixture)
    --queries   <path>  query set (default: committed synthetic query set)
    --out       <path>  report output path (default: data/manifests/...)
    --top-k     <int>   retrieval depth (default 10)
    --mode      <str>   hybrid | sparse | dense (default hybrid)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from evidence_grounded_rag.evaluation.metrics import EvaluationReport, evaluate_answers
from evidence_grounded_rag.evaluation.queries import load_query_set
from evidence_grounded_rag.pipeline import (
    AnswerRecord,
    EvidenceRagPipeline,
    PipelineConfig,
)
from evidence_grounded_rag.versions import (
    BM25_VERSION,
    CORPUS_FORMAT_VERSION,
    EMBEDDING_MODEL_ID,
    FUSION_VERSION,
    GENERATOR_VERSION,
    PACKAGE_VERSION,
)

DEFAULT_MANIFEST = Path("data/fixtures/synthetic_manifest.json")
DEFAULT_QUERIES = Path("data/fixtures/queries/synthetic_queries_v1.json")
DEFAULT_OUT = Path("data/manifests/evaluation-synthetic-v1.0.0.json")


def materialize(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    query_set_path: str | Path = DEFAULT_QUERIES,
    out_path: str | Path | None = None,
    *,
    top_k: int = 10,
    mode: str = "hybrid",
) -> tuple[EvaluationReport, Path]:
    """Run the deterministic evaluation and write the report; return it plus path."""
    pipeline = EvidenceRagPipeline(
        manifest_path,
        config=PipelineConfig(top_k=top_k, mode=mode),
    )
    status = pipeline.corpus_status()
    if not status["loaded"]:
        raise RuntimeError(
            f"corpus gate failed ({status['reason']}): {status['detail']}"
        )

    query_set = load_query_set(query_set_path)
    records = [
        pipeline.answer(query.question, top_k=top_k, mode=mode)
        for query in query_set.queries
    ]
    report = evaluate_answers(query_set, records)
    report = _enrich_report(report, pipeline, records, query_set_version=query_set.query_set_version)

    resolved_out = Path(out_path or DEFAULT_OUT)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report, resolved_out


def _enrich_report(
    report: EvaluationReport,
    pipeline: EvidenceRagPipeline,
    records: list[AnswerRecord],
    *,
    query_set_version: str,
) -> EvaluationReport:
    status = pipeline.corpus_status()
    first = next((record for record in records if record.trace is not None), None)
    versions = {
        "package": PACKAGE_VERSION,
        "corpus_format": CORPUS_FORMAT_VERSION,
        "corpus_id": status.get("corpus_id", ""),
        "corpus_version": status.get("corpus_version", ""),
        "query_set_version": query_set_version,
        "bm25": BM25_VERSION,
        "embedding_model": EMBEDDING_MODEL_ID,
        "fusion": FUSION_VERSION,
        "generation": GENERATOR_VERSION,
    }
    if first is not None:
        versions.update({key: value for key, value in first.versions.items() if key not in versions})

    config = {
        "top_k": pipeline.config.top_k,
        "mode": pipeline.config.mode,
        "min_fused_score": pipeline.config.min_fused_score,
    }
    if first is not None:
        config.update({key: value for key, value in first.config.items() if key not in config})

    generation = first.generation if first is not None else {}
    generation["is_llm_generated"] = False

    labels = {
        **report.labels,
        "synthetic_evaluation": True,
        "corpus_provenance": "project-generated-synthetic",
        "corpus_real": status.get("real_corpus", False),
        "real_corpus_gate_blocked": status.get("real_corpus_gate", {}).get("blocked", True),
    }

    return replace(
        report,
        versions=versions,
        config=config,
        generation=generation,
        labels=labels,
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic synthetic-fixture evaluation materializer."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="hybrid", choices=("hybrid", "sparse", "dense"))
    args = parser.parse_args(argv)

    report, out_path = materialize(
        args.manifest,
        args.queries,
        args.out,
        top_k=args.top_k,
        mode=args.mode,
    )
    print(f"wrote evaluation report: {out_path}")
    print(f"corpus: {report.versions.get('corpus_id')} v{report.versions.get('corpus_version')} (synthetic fixture)")
    print(f"query set: v{report.versions.get('query_set_version')} — {len(report.per_query)} queries")
    for metric, value in report.metrics.items():
        print(f"  {metric:<26} {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
