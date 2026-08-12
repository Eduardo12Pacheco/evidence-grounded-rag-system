from __future__ import annotations

from collections.abc import Sequence

from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.embeddings.base import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    Vector,
    cosine_similarity,
)
from evidence_grounded_rag.retrieval.models import RetrievalResult, ScoredChunk


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._chunks = list(chunks)
        self._vectors: list[Vector] = []
        self.dimension = provider.dimension
        if self._chunks:
            self._vectors = provider.embed([chunk.text for chunk in self._chunks])

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        if top_k <= 0 or self.size == 0:
            return RetrievalResult(
                chunks=(),
                config=self._config_dict(),
                versions=self.versions(),
                empty=True,
            )
        query_vector = self.provider.embed_one(query)
        if len(query_vector) != self.dimension:
            raise EmbeddingDimensionError(
                f"query vector dimension {len(query_vector)} != indexed dimension "
                f"{self.dimension}"
            )
        scored = [
            (cosine_similarity(query_vector, vector), chunk_index)
            for chunk_index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: (-item[0], self._chunks[item[1]].chunk_id))
        results = [
            ScoredChunk(
                chunk_id=self._chunks[chunk_index].chunk_id,
                document_id=self._chunks[chunk_index].document_id,
                rank=rank,
                score=score,
            )
            for rank, (score, chunk_index) in enumerate(scored[:top_k], start=1)
        ]
        return RetrievalResult(
            chunks=tuple(results),
            config=self._config_dict(),
            versions=self.versions(),
            empty=not results,
        )

    def _config_dict(self) -> dict[str, object]:
        return {
            "retriever": "dense",
            "metric": "cosine",
            "dimension": self.dimension,
            **self.provider.to_dict(),
        }

    def versions(self) -> dict[str, str]:
        return {"embedding": self.provider.model_id}
