from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.retrieval.models import RetrievalResult, ScoredChunk
from evidence_grounded_rag.text.tokenizer import tokenize
from evidence_grounded_rag.versions import BM25_VERSION, TOKENIZER_VERSION

IDF_MODE = "robertson-sparck-jones"


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    idf_mode: str = IDF_MODE
    tokenizer_version: str = TOKENIZER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "k1": self.k1,
            "b": self.b,
            "idf_mode": self.idf_mode,
            "tokenizer_version": self.tokenizer_version,
        }


class BM25:
    version: str = BM25_VERSION

    def __init__(
        self, chunks: Sequence[Chunk], config: BM25Config | None = None
    ) -> None:
        self.config = config or BM25Config()
        self._chunks = list(chunks)
        self._doc_lengths: list[int] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        self._doc_lengths = [len(tokenize(chunk.text)) for chunk in self._chunks]
        for doc_index, chunk in enumerate(self._chunks):
            for term, count in Counter(tokenize(chunk.text)).items():
                self._postings.setdefault(term, {})[doc_index] = count

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, {}))
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def _score_document(self, term: str, doc_index: int) -> float:
        tf = self._postings.get(term, {}).get(doc_index, 0)
        if tf == 0:
            return 0.0
        k1 = self.config.k1
        b = self.config.b
        length = self._doc_lengths[doc_index]
        avgdl = self._average_document_length()
        denom = tf + k1 * (1.0 - b + b * (length / avgdl)) if avgdl else tf
        return self._idf(term) * (tf * (k1 + 1.0)) / denom

    def _average_document_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths) / len(self._doc_lengths)

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_terms = tokenize(query)
        if not query_terms or top_k <= 0 or self.size == 0:
            return RetrievalResult(
                chunks=(),
                config=self._config_dict(),
                versions=self.versions(),
                empty=True,
            )
        scores: dict[int, float] = {}
        for term in set(query_terms):
            for doc_index in self._postings.get(term, {}):
                scores[doc_index] = scores.get(doc_index, 0.0) + self._score_document(
                    term, doc_index
                )
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._chunks[item[0]].chunk_id),
        )
        results = [
            ScoredChunk(
                chunk_id=self._chunks[doc_index].chunk_id,
                document_id=self._chunks[doc_index].document_id,
                rank=rank,
                score=score,
            )
            for rank, (doc_index, score) in enumerate(ranked[:top_k], start=1)
        ]
        return RetrievalResult(
            chunks=tuple(results),
            config=self._config_dict(),
            versions=self.versions(),
            empty=not results,
        )

    def _config_dict(self) -> dict[str, object]:
        return {"retriever": "bm25", **self.config.to_dict()}

    def versions(self) -> dict[str, str]:
        return {
            "bm25": self.version,
            "tokenizer": self.config.tokenizer_version,
        }
