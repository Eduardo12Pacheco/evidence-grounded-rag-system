"""Hybrid retriever fusing BM25 with dense embeddings and traces."""

from __future__ import annotations

from dataclasses import dataclass

from evidence_grounded_rag.corpus.models import Corpus
from evidence_grounded_rag.embeddings import DeterministicEmbedding, EmbeddingProvider
from evidence_grounded_rag.retrieval.bm25 import BM25, BM25Config
from evidence_grounded_rag.retrieval.fusion import FusionConfig, reciprocal_rank_fusion
from evidence_grounded_rag.retrieval.models import RetrievalResult
from evidence_grounded_rag.versions import PACKAGE_VERSION

MODE_HYBRID = "hybrid"
MODE_SPARSE = "sparse"
MODE_DENSE = "dense"
VALID_MODES = (MODE_HYBRID, MODE_SPARSE, MODE_DENSE)


@dataclass(frozen=True)
class HybridTrace:
    query: str
    mode: str
    sparse: RetrievalResult
    dense: RetrievalResult
    fused: RetrievalResult
    fusion_config: dict[str, object]
    config: dict[str, object]
    versions: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "mode": self.mode,
            "sparse": _result_to_dict(self.sparse),
            "dense": _result_to_dict(self.dense),
            "fused": _result_to_dict(self.fused),
            "fusion": self.fusion_config,
            "config": self.config,
            "versions": self.versions,
        }


def _result_to_dict(result: RetrievalResult) -> dict[str, object]:
    return {
        "chunks": [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "rank": item.rank,
                "score": item.score,
            }
            for item in result.chunks
        ],
        "config": result.config,
        "versions": result.versions,
        "empty": result.empty,
    }


@dataclass(frozen=True)
class HybridResult:
    fused: RetrievalResult
    trace: HybridTrace


class HybridRetriever:
    def __init__(
        self,
        corpus: Corpus,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_config: BM25Config | None = None,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.embedding_provider = embedding_provider or DeterministicEmbedding()
        self.bm25_config = bm25_config or BM25Config()
        self.fusion_config = fusion_config or FusionConfig()
        chunks = corpus.chunks()
        self.bm25 = BM25(chunks, config=self.bm25_config)
        self.dense_index = self._build_dense_index(chunks)

    def _build_dense_index(self, chunks):
        from evidence_grounded_rag.embeddings.index import DenseIndex

        return DenseIndex(chunks, self.embedding_provider)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: str = MODE_HYBRID,
        min_score: float | None = None,
    ) -> HybridResult:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        sparse = self.bm25.search(query, top_k=top_k)
        dense = self.dense_index.search(query, top_k=top_k)

        if mode == MODE_SPARSE:
            fused = sparse
        elif mode == MODE_DENSE:
            fused = dense
        else:
            fused = reciprocal_rank_fusion(sparse, dense, self.fusion_config)

        if min_score is not None:
            fused = _apply_min_score(fused, min_score)

        fused = _apply_top_k(fused, top_k)

        versions = {
            "package": PACKAGE_VERSION,
            "corpus_id": self.corpus.corpus_id,
            "corpus_version": self.corpus.corpus_version,
            "bm25": self.bm25.version,
            "embedding": self.embedding_provider.model_id,
            "fusion": self.fusion_config.name,
        }

        config = {
            "mode": mode,
            "top_k": top_k,
            "min_score": min_score,
            "k1": self.bm25_config.k1,
            "b": self.bm25_config.b,
            "fusion_k": self.fusion_config.k,
        }

        trace = HybridTrace(
            query=query,
            mode=mode,
            sparse=sparse,
            dense=dense,
            fused=fused,
            fusion_config=self.fusion_config.to_dict(),
            config=config,
            versions=versions,
        )
        return HybridResult(fused=fused, trace=trace)


def _apply_min_score(result: RetrievalResult, min_score: float) -> RetrievalResult:
    kept = [item for item in result.chunks if item.score >= min_score]
    return RetrievalResult(
        chunks=tuple(kept),
        config=dict(result.config),
        versions=dict(result.versions),
        empty=not kept,
    )


def _apply_top_k(result: RetrievalResult, top_k: int) -> RetrievalResult:
    if top_k <= 0:
        return RetrievalResult(
            chunks=(),
            config=dict(result.config),
            versions=dict(result.versions),
            empty=True,
        )
    if len(result.chunks) <= top_k:
        return result
    return RetrievalResult(
        chunks=result.chunks[:top_k],
        config=dict(result.config),
        versions=dict(result.versions),
        empty=False,
    )
