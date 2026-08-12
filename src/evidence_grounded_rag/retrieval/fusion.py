from __future__ import annotations

from dataclasses import dataclass

from evidence_grounded_rag.retrieval.models import RetrievalResult, ScoredChunk

FUSION_NAME = "reciprocal_rank_fusion"


@dataclass(frozen=True)
class FusionConfig:
    k: int = 60
    name: str = FUSION_NAME

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "k": self.k}


def reciprocal_rank_fusion(
    sparse: RetrievalResult,
    dense: RetrievalResult,
    config: FusionConfig | None = None,
) -> RetrievalResult:
    config = config or FusionConfig()
    scores: dict[str, tuple[float, str]] = {}
    for result in (sparse, dense):
        for item in result.chunks:
            contribution = 1.0 / (config.k + item.rank)
            chunk_id = item.chunk_id
            current = scores.get(chunk_id)
            if current is None:
                scores[chunk_id] = (contribution, item.document_id)
            else:
                scores[chunk_id] = (current[0] + contribution, current[1])

    ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
    fused = [
        ScoredChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            rank=rank,
            score=score,
        )
        for rank, (chunk_id, (score, document_id)) in enumerate(ranked, start=1)
    ]
    versions = dict(sparse.versions)
    versions.update(dense.versions)
    return RetrievalResult(
        chunks=tuple(fused),
        config={"retriever": "hybrid", **config.to_dict()},
        versions=versions,
        empty=not fused,
    )
