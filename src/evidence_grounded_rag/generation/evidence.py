"""The bounded evidence surface handed to a generation provider.

A generator NEVER receives the corpus. It receives exactly one
``EvidencePack``: the retrieved, selected chunks (with rank, score, and
provenance metadata) plus the retrieval trace and corpus metadata. Any chunk
whose id is not in the pack is, by definition, not exposed to the generator,
and a citation that references a chunk outside the pack is rejected by the
strict structured validator.

``EvidencePack`` enforces a committed ``max_items`` bound: the builder never
exposes more chunks than the bound, so the prompt, the generator, and the
support evaluator all operate on a bounded surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from evidence_grounded_rag.corpus.models import Chunk

#: Committed upper bound on the number of evidence items a pack may carry.
#: The pipeline exposes at most ``top_k`` fused chunks (default 5), well under
#: this bound; the bound exists so a pack is always a bounded surface even if a
#: caller hands it more chunks.
DEFAULT_MAX_ITEMS = 20


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved, selected chunk with rank/score and provenance metadata.

    Carries the same stable fields as the source ``Chunk`` plus the rank and
    raw score it received from the fused retrieval ranking, so downstream
    claims, citations, prompts, and reports can reference the exact evidence
    the generator was exposed to.
    """

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
        """Rebuild the source chunk record (for offline extractive use)."""
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
        """Stable serialization WITHOUT the raw text.

        ``to_dict`` deliberately omits the passage text so traces, reports, and
        demo tables can surface evidence metadata without committing corpus
        text; the prompt builder reads ``text`` directly.
        """
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
    """The bounded, ordered evidence surface exposed to the generator.

    ``items`` is the ordered, deduplicated list of exposed evidence (rank
    ascending). ``corpus_id``/``corpus_version`` record which served corpus the
    evidence came from, and ``trace`` carries the retrieval trace object when
    the pipeline produced one (serialized lazily via ``trace_to_dict``).
    """

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
        """Build a pack from a corpus and scored (ranked) chunks.

        *corpus* needs only ``corpus_id``, ``corpus_version``, and
        ``chunk_by_id``. Chunks whose id cannot be resolved in the corpus are
        skipped (they cannot be exposed). Ordering follows the scored ranking.
        """
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
        """Build a pack directly from chunk records (tests and offline use).

        Ranks follow the given order (1-based); scores default to ``0.0`` so
        deterministic offline callers that only care about the exposed surface
        do not need synthetic scores.
        """
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
