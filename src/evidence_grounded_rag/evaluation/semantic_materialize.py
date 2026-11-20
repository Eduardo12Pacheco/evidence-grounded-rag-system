"""CLI that materializes the real-corpus semantic retrieval report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_grounded_rag.embeddings import (
    EmbeddingModelManifestError,
    EmbeddingProvider,
    SemanticLocalEmbedding,
    SemanticModelUnavailableError,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
)
from evidence_grounded_rag.evaluation.real_materialize import (
    RETRIEVAL_GROUPS,
    RealLocalEvaluationError,
    _corpus_facts,
    _current_commit,
    _group_membership,
    _group_retrieval_metrics,
    _query_set_facts,
    build_real_local,
)
from evidence_grounded_rag.evaluation.real_queries import (
    RealQuerySet,
    load_real_query_set,
)
from evidence_grounded_rag.pipeline import MIN_FUSED_SCORE
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    HybridRetriever,
)
from evidence_grounded_rag.versions import (
    BM25_VERSION,
    CORPUS_FORMAT_VERSION,
    FUSION_VERSION,
    NORMALIZATION_VERSION,
    PACKAGE_VERSION,
    SOURCE_MANIFEST_VERSION,
    TOKENIZER_VERSION,
)

REPORT_TYPE = "real-local-semantic-evaluation"
REPORT_VERSION = "1.0.0"

MODE_SPARSE_KEY = "sparse"
MODE_HASH_KEY = "hash"
MODE_SEMANTIC_KEY = "semantic"
MODE_HYBRID_KEY = "hybrid"
VALID_MODES = (MODE_SPARSE_KEY, MODE_HASH_KEY, MODE_SEMANTIC_KEY, MODE_HYBRID_KEY)

SPARSE_MODE_LABEL = "BM25 sparse baseline"
HASH_MODE_LABEL = "Deterministic hash embedding diagnostic — not semantic retrieval"
SEMANTIC_MODE_LABEL = "Local semantic embedding model"
HYBRID_MODE_LABEL = "Hybrid BM25 plus local semantic embeddings"

MODE_LABELS: dict[str, str] = {
    MODE_SPARSE_KEY: SPARSE_MODE_LABEL,
    MODE_HASH_KEY: HASH_MODE_LABEL,
    MODE_SEMANTIC_KEY: SEMANTIC_MODE_LABEL,
    MODE_HYBRID_KEY: HYBRID_MODE_LABEL,
}

_MODE_SEARCH_MODE = {
    MODE_SPARSE_KEY: MODE_SPARSE,
    MODE_HASH_KEY: MODE_DENSE,
    MODE_SEMANTIC_KEY: MODE_DENSE,
    MODE_HYBRID_KEY: MODE_HYBRID,
}

_SEMANTIC_MODES = frozenset({MODE_SEMANTIC_KEY, MODE_HYBRID_KEY})

DEFAULT_MANIFEST = Path("data/manifests/corpus-source-v1.0.0.json")
DEFAULT_QUERIES = Path("data/queries/python-docs-real-v1.0.0.json")
DEFAULT_RAW_DIR = Path("data/raw/python-docs-3.12.13")
DEFAULT_MODEL_MANIFEST = Path("data/manifests/embedding-model-v1.0.0.json")
DEFAULT_CACHE_DIR = Path("data/raw/models/bge-small-en-v1.5")
DEFAULT_OUT = Path("data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json")


@dataclass(frozen=True)
class SemanticEvaluationReport:
    data: dict[str, object]

    def deterministic_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.data.items()
            if key
            not in (
                "timestamp_utc",
                "deterministic_digest",
                "runtime_observations",
            )
        }

    def deterministic_bytes(self) -> bytes:
        return _json_bytes(self.deterministic_payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.deterministic_bytes()).hexdigest()


def _load_semantic_provider(
    model_manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    provider: EmbeddingProvider | None = None,
) -> tuple[EmbeddingProvider, Any]:
    manifest = load_embedding_model_manifest(model_manifest_path)
    audit_embedding_model_manifest(manifest)
    if provider is not None:
        return provider, manifest
    return SemanticLocalEmbedding(cache_dir, manifest=manifest), manifest


def semantic_evaluate(
    corpus_manifest_path: str | Path = DEFAULT_MANIFEST,
    query_set_path: str | Path = DEFAULT_QUERIES,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    model_manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    out_path: str | Path | None = None,
    top_k: int = 10,
    modes: Sequence[str] = VALID_MODES,
    provider: EmbeddingProvider | None = None,
    timestamp_utc: str | None = None,
) -> tuple[SemanticEvaluationReport, Path]:
    for mode in modes:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")

    corpus = build_real_local(corpus_manifest_path, raw_dir)
    manifest_id = Path(corpus_manifest_path).stem
    query_set = load_real_query_set(query_set_path, corpus, manifest_id=manifest_id)

    semantic_provider, model_manifest = _load_semantic_provider(
        model_manifest_path, cache_dir, provider=provider
    )

    hash_retriever = HybridRetriever(corpus)
    semantic_retriever = HybridRetriever(corpus, embedding_provider=semantic_provider)

    per_query: dict[str, dict[str, dict[str, object]]] = {mode: {} for mode in modes}
    ranking_stability: dict[str, dict[str, object]] = {}
    runtime_seconds: dict[str, float] = {}

    for mode in modes:
        retriever = semantic_retriever if mode in _SEMANTIC_MODES else hash_retriever
        search_mode = _MODE_SEARCH_MODE[mode]
        started = time.monotonic()
        rankings: dict[str, tuple[str, ...]] = {}
        empties: dict[str, bool] = {}
        for query in query_set.queries:
            hybrid = retriever.search(
                query.question,
                top_k=top_k,
                mode=search_mode,
                min_score=MIN_FUSED_SCORE,
            )
            rankings[query.query_id] = tuple(
                item.chunk_id for item in hybrid.fused.chunks
            )
            empties[query.query_id] = hybrid.fused.empty

        second_pass: dict[str, tuple[str, ...]] = {}
        for query in query_set.queries:
            hybrid = retriever.search(
                query.question,
                top_k=top_k,
                mode=search_mode,
                min_score=MIN_FUSED_SCORE,
            )
            second_pass[query.query_id] = tuple(
                item.chunk_id for item in hybrid.fused.chunks
            )
        runtime_seconds[mode] = time.monotonic() - started

        identical = sum(1 for qid in rankings if second_pass[qid] == rankings[qid])
        ranking_stability[mode] = {
            "num_queries": len(rankings),
            "identical_rankings": identical,
            "fraction_identical": (identical / len(rankings) if rankings else 0.0),
        }

        for query in query_set.queries:
            ids = list(rankings[query.query_id])
            gold = query.gold_chunk_ids
            top_hit = bool(gold & set(ids[:top_k])) if gold else None
            first_hit = None
            if gold:
                for index, chunk_id in enumerate(ids, start=1):
                    if chunk_id in gold:
                        first_hit = index
                        break
            per_query[mode][query.query_id] = {
                "query_id": query.query_id,
                "case": query.case,
                "expected_status": query.expected_status,
                "retrieval_empty": empties[query.query_id],
                "retrieved_chunk_ids": ids,
                "num_relevant": len(gold),
                "top_k_hit": top_hit,
                "first_hit_rank": first_hit,
            }

    records_by_mode = {
        mode: [per_query[mode][q.query_id] for q in query_set.queries] for mode in modes
    }
    metrics = {
        mode: _retrieval_metrics(query_set, records_by_mode[mode]) for mode in modes
    }

    data: dict[str, object] = {
        "report_type": REPORT_TYPE,
        "report_version": REPORT_VERSION,
        "labels": _labels(),
        "model": _model_facts(model_manifest, semantic_provider),
        "corpus": _corpus_facts(corpus, manifest_id),
        "query_set": _query_set_facts(query_set),
        "configs": _configs(top_k=top_k, modes=modes),
        "metrics": metrics,
        "per_query": per_query,
        "ranking_stability": ranking_stability,
        "runtime_observations": _runtime_observations(runtime_seconds),
        "implementation_commit": _current_commit(),
        "timestamp_utc": timestamp_utc
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "deterministic_digest": "",
        "limitations": _limitations(),
    }
    report = SemanticEvaluationReport(data)
    report.data["deterministic_digest"] = report.digest

    resolved_out = Path(out_path or DEFAULT_OUT)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(_json_bytes(report.data).decode("utf-8"), encoding="utf-8")
    return report, resolved_out


def _retrieval_metrics(
    query_set: RealQuerySet,
    records: list[dict[str, object]],
) -> dict[str, object]:
    overall = _group_retrieval_metrics(query_set, records, lambda _query: True)
    groups = {
        group: _group_retrieval_metrics(query_set, records, _group_membership(group))
        for group in RETRIEVAL_GROUPS
    }
    return {"overall": overall, "groups": groups}


def _model_facts(manifest: Any, provider: EmbeddingProvider) -> dict[str, object]:
    provider_dict = provider.to_dict()
    return {
        "manifest_version": manifest.manifest_version,
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "publisher": manifest.publisher,
        "official_url": manifest.official_url,
        "model_card_url": manifest.model_card_url,
        "revision": manifest.revision,
        "revision_tag": manifest.revision_tag,
        "license": {
            "name": manifest.license.name,
            "version": manifest.license.version,
            "url": manifest.license.url,
        },
        "architecture": manifest.architecture,
        "dimension": manifest.dimension,
        "max_input_tokens": manifest.max_input_tokens,
        "tokenizer": manifest.tokenizer,
        "pooling": manifest.pooling,
        "normalization": manifest.normalization,
        "device": provider_dict.get("device"),
        "cache_path": provider_dict.get("cache_path"),
        "local_files_only": provider_dict.get("local_files_only"),
        "model_status": manifest.model_status,
        "acquisition_method": manifest.acquisition_method,
        "redistribution_status": manifest.redistribution_status,
        "verification_timestamp": manifest.verification_timestamp,
        "files": [
            {
                "path": entry.path,
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for entry in manifest.expected_files
        ],
    }


def _labels() -> dict[str, str]:
    return {
        "sparse_mode": SPARSE_MODE_LABEL,
        "hash_mode": HASH_MODE_LABEL,
        "semantic_mode": SEMANTIC_MODE_LABEL,
        "hybrid_mode": HYBRID_MODE_LABEL,
        "generation": (
            "No generation in this module: retrieval evaluation only, no LLM."
        ),
        "bounded_scope": (
            "Bounded evaluation over three Python 3.12.13 documentation files "
            "and 70 chunks; it does not represent all Python documentation."
        ),
        "retrieval_not_faithfulness": (
            "Retrieval quality does not imply answer faithfulness; no "
            "generation or faithfulness metrics are reported here."
        ),
        "no_hash_fallback": (
            "Semantic modes load the local model with local_files_only=True "
            "and refuse to run if it is unavailable; hash embeddings are never "
            "used under a semantic label."
        ),
        "separate_report": (
            "This report is separate from the synthetic report and from "
            "python-docs-real-evaluation-v1.0.0.json and is never combined "
            "with either."
        ),
    }


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
        "mode_labels": {mode: MODE_LABELS[mode] for mode in modes},
        "modes": list(modes),
        "versions": {
            "package": PACKAGE_VERSION,
            "corpus_format": CORPUS_FORMAT_VERSION,
            "source_manifest": SOURCE_MANIFEST_VERSION,
            "normalization": NORMALIZATION_VERSION,
        },
    }


def _runtime_observations(
    runtime_seconds: dict[str, float],
) -> dict[str, object]:
    peak_rss_kb: int | None = None
    try:
        import resource

        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        peak_rss_kb = None
    return {
        "elapsed_seconds_per_mode": {
            mode: round(seconds, 3) for mode, seconds in runtime_seconds.items()
        },
        "process_peak_rss_kb": peak_rss_kb,
        "memory_note": (
            "Only process-wide peak RSS is recorded; per-mode memory was not measured."
        ),
    }


def _limitations() -> list[str]:
    return [
        (
            "Bounded evaluation over three Python 3.12.13 documentation files "
            "and 70 chunks; it does not represent all Python documentation and "
            "is NOT validation of Python documentation in general."
        ),
        (
            "Retrieval metrics only. No generation and no faithfulness metrics "
            "are produced by this module; retrieval quality does not imply "
            "answer faithfulness."
        ),
        (
            "No LLM is involved anywhere; generation is out of scope for this "
            "materializer."
        ),
        (
            "The semantic model (BAAI/bge-small-en-v1.5, revision "
            "5c38ec7c) is used locally with local_files_only=True from a "
            "gitignored cache; weights are never committed or redistributed. "
            "No benchmark numbers from the model card are claimed as project "
            "metrics, and no model is claimed to be universally best."
        ),
        (
            "Semantic float inference is deterministic within one CPU thread "
            "and one process; cross-process float determinism is best-effort. "
            "Ranking stability is recorded from two runs in the same process."
        ),
        (
            "Denominators are small (12 answerable and 6 non-answerable "
            "queries per mode); one query moves percentages by up to 8.3 "
            "percentage points."
        ),
        (
            "Only wall-clock seconds per mode and process-wide peak RSS are "
            "recorded; per-mode memory was not measured. Runtime observations "
            "and the timestamp are runtime provenance and are excluded from "
            "the deterministic digest."
        ),
        (
            "No chunk-level or claim-level text is stored in this report; only "
            "chunk ids, query ids, and metric numbers."
        ),
    ]


def _json_bytes(data: dict[str, object]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the separate semantic-materializer retrieval evaluation over "
            "the approved Python 3.12.13 documentation corpus with the local "
            "semantic embedding model, and write its own report. Never uses "
            "the network and never falls back to hash embeddings under a "
            "semantic label."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=Path)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), type=Path)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), type=Path)
    parser.add_argument(
        "--model-manifest", default=str(DEFAULT_MODEL_MANIFEST), type=Path
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), type=Path)
    parser.add_argument("--output", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        print("error: --top-k must be a positive integer", file=sys.stderr)
        return 2

    try:
        report, out_path = semantic_evaluate(
            args.manifest,
            args.queries,
            args.raw_dir,
            model_manifest_path=args.model_manifest,
            cache_dir=args.cache_dir,
            out_path=args.output,
            top_k=args.top_k,
        )
    except SemanticModelUnavailableError as exc:
        print(
            f"SEMANTIC EVALUATION STOPPED: local semantic model unavailable — {exc}",
            file=sys.stderr,
        )
        print(
            "No semantic metrics were produced and no hash fallback was used. "
            "Install the 'embeddings' extra and run "
            "scripts/acquire_embedding_model.py then "
            "scripts/verify_embedding_model.py first.",
            file=sys.stderr,
        )
        return 1
    except (
        EmbeddingModelManifestError,
        RealLocalEvaluationError,
        ValueError,
        OSError,
    ) as exc:
        print(f"SEMANTIC EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"wrote semantic-materializer report: {out_path}")
    print(
        f"model: {report.data['model']['model_id']} revision {report.data['model']['revision'][:12]}..."
    )
    print(
        f"corpus: {report.data['corpus']['corpus_id']} v{report.data['corpus']['corpus_version']} "
        f"({report.data['corpus']['document_count']} docs / {report.data['corpus']['chunk_count']} chunks)"
    )
    print(f"query set: {report.data['query_set']['num_queries']} queries")
    for mode, metrics in report.data["metrics"].items():
        overall = metrics["overall"]
        stability = report.data["ranking_stability"][mode]
        print(f"[{mode}] {report.data['configs']['mode_labels'][mode]}")
        print(
            f"  recall@1={overall['recall@1']:.4f} recall@5={overall['recall@5']:.4f} "
            f"recall@10={overall['recall@10']:.4f} mrr={overall['mrr']:.4f} "
            f"ndcg@5={overall['ndcg@5']:.4f} no_result_rate={overall['no_result_rate']:.4f}"
        )
        print(
            f"  ranking stability: {stability['identical_rankings']}/{stability['num_queries']} "
            f"identical ({stability['fraction_identical']:.3f})"
        )
    print(f"digest={report.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
