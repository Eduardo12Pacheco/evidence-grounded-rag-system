from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from evidence_grounded_rag.corpus.models import Chunk

DEFAULT_MAX_ITEMS = 20


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    document_id: str
    seq: int
    text: str
    location: str
    corpus_version: str
    rank: int
    score: float
    checksum_sha256: str = ""

    def to_chunk(self) -> Chunk:
        return Chunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            seq=self.seq,
            text=self.text,
            location=self.location,
            corpus_version=self.corpus_version,
            checksum_sha256=self.checksum_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "seq": self.seq,
            "location": self.location,
            "corpus_version": self.corpus_version,
            "rank": self.rank,
            "score": self.score,
        }


@dataclass(frozen=True)
class EvidencePack:
    items: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    corpus_id: str = ""
    corpus_version: str = ""
    trace: Any | None = None
    max_items: int = DEFAULT_MAX_ITEMS

    def __iter__(self) -> Iterator[EvidenceItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def empty(self) -> bool:
        return not self.items

    @property
    def exposed_chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.items)

    def by_chunk_id(self, chunk_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.chunk_id == chunk_id:
                return item
        return None

    def trace_to_dict(self) -> dict[str, object]:
        if self.trace is None:
            return {}
        to_dict = getattr(self.trace, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return {}

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "max_items": self.max_items,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_scored_chunks(
        cls,
        corpus: Any,
        scored_chunks: Iterable[Any],
        *,
        trace: Any | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> EvidencePack:
        items: list[EvidenceItem] = []
        seen: set[str] = set()
        for scored in scored_chunks:
            if len(items) >= max_items:
                break
            chunk = corpus.chunk_by_id(scored.chunk_id)
            if chunk is None or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            items.append(
                EvidenceItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    seq=chunk.seq,
                    text=chunk.text,
                    location=chunk.location,
                    corpus_version=chunk.corpus_version,
                    checksum_sha256=chunk.checksum_sha256,
                    rank=scored.rank,
                    score=scored.score,
                )
            )
        return cls(
            items=tuple(items),
            corpus_id=getattr(corpus, "corpus_id", ""),
            corpus_version=getattr(corpus, "corpus_version", ""),
            trace=trace,
            max_items=max_items,
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Chunk],
        *,
        corpus_id: str = "",
        corpus_version: str = "",
        trace: Any | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> EvidencePack:
        items: list[EvidenceItem] = []
        for rank, chunk in enumerate(chunks, start=1):
            if len(items) >= max_items:
                break
            items.append(
                EvidenceItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    seq=chunk.seq,
                    text=chunk.text,
                    location=chunk.location,
                    corpus_version=chunk.corpus_version,
                    checksum_sha256=chunk.checksum_sha256,
                    rank=rank,
                    score=0.0,
                )
            )
        return cls(
            items=tuple(items),
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            trace=trace,
            max_items=max_items,
        )
