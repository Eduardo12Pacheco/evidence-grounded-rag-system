from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from typing import Any

from evidence_grounded_rag.abstention import AbstentionReason, reason_detail
from evidence_grounded_rag.corpus.models import Chunk
from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import (
    AnswerState,
    Citation,
    Claim,
    GeneratedAnswer,
)
from evidence_grounded_rag.text.tokenizer import key_tokens, tokenize
from evidence_grounded_rag.versions import GENERATOR_VERSION

PROVIDER_ID = "extractive-template"
MODEL_ID = GENERATOR_VERSION
IS_LLM_GENERATED = False
MODE = "offline"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_SCALAR_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?\b")
_NEGATION_PREFIXES = ("not ", "no ", "never ")

_MULTI_ASPECT_MARKERS = frozenset({"and", "or", "also"})
_WH_WORDS = frozenset({"what", "when", "where", "which", "who", "how", "why"})


class ExtractiveTemplateGenerator:
    provider_id: str = PROVIDER_ID
    model_id: str = MODEL_ID
    is_llm_generated: bool = IS_LLM_GENERATED
    mode: str = MODE

    def __init__(
        self,
        *,
        evidence_limit: int = 5,
        max_cited_claims: int = 3,
    ) -> None:
        self.evidence_limit = evidence_limit
        self.max_cited_claims = max_cited_claims

    def generate(
        self, query: str, evidence: EvidencePack | Sequence[Chunk] | Any
    ) -> GeneratedAnswer:
        chunks = _evidence_to_chunks(evidence)
        return self._generate_chunks(query, chunks)

    def _generate_chunks(
        self, query: str, evidence: Sequence[Chunk]
    ) -> GeneratedAnswer:
        if not isinstance(query, str) or not query.strip():
            return self._abstain(
                AbstentionReason.UNSUPPORTED_QUERY,
                "The query is empty; there is nothing to answer.",
            )

        if not evidence:
            return self._abstain(
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                "No evidence was retrieved or the retrieved evidence fell below "
                "the committed threshold.",
            )

        query_keys = set(key_tokens(query))
        matched = self._extract_matched_claims(query, evidence, query_keys)
        if not matched:
            return self._abstain(
                AbstentionReason.UNSUPPORTED_QUERY,
                "Key terms of the query appear in no retrieved passage; the "
                "question is out of scope for this corpus.",
            )

        matched = matched[: self.max_cited_claims]

        full_matches = [claim for claim in matched if _is_full_match(claim, query_keys)]
        conflict_candidates = full_matches if full_matches else matched
        conflict = self._find_conflict(conflict_candidates)
        if conflict is not None:
            return self._abstain(
                AbstentionReason.CONFLICTING_EVIDENCE,
                conflict,
            )

        if not self._is_multi_aspect(query):
            ambiguous = self._find_ambiguity(matched, query_keys)
            if ambiguous is not None:
                return self._abstain(
                    AbstentionReason.INSUFFICIENT_EVIDENCE,
                    ambiguous,
                )
            selected = self._select_single_aspect_claims(matched, query_keys)
            if selected is None:
                return self._abstain(
                    AbstentionReason.INSUFFICIENT_EVIDENCE,
                    "No single retrieved passage fully supports the query and "
                    "the matched passages address different predicates.",
                )
            matched = selected

        return self._answered(query, evidence, matched)

    def _extract_matched_claims(
        self,
        query: str,
        evidence: Sequence[Chunk],
        query_keys: set[str],
    ) -> list[Claim]:
        claims: list[Claim] = []
        for chunk in evidence[: self.evidence_limit]:
            for index, sentence in enumerate(split_sentences(chunk.text)):
                sentence_tokens = set(tokenize(sentence))
                if sentence_tokens & query_keys:
                    claims.append(self._claim_for(query, chunk, index, sentence))
        return claims

    def _claim_for(
        self,
        query: str,
        chunk: Chunk,
        sentence_index: int,
        sentence: str,
    ) -> Claim:
        predicate, value = split_predicate_value(sentence)
        return Claim(
            claim_id=f"{chunk.chunk_id}::claim-{sentence_index}",
            text=sentence,
            citation_chunk_ids=(chunk.chunk_id,),
            predicate=predicate,
            value=value,
        )

    def _find_conflict(self, claims: list[Claim]) -> str | None:
        for left, right in itertools.combinations(claims, 2):
            if left.citation_chunk_ids == right.citation_chunk_ids:
                continue
            if left.predicate is None or right.predicate is None:
                continue
            if left.predicate != right.predicate:
                continue
            if left.value == right.value:
                continue
            if values_conflict(left.value, right.value):
                return (
                    "Retrieved passages directly contradict each other: "
                    f"{left.text!r} vs {right.text!r}. No answer can be grounded."
                )
        return None

    def _find_ambiguity(self, claims: list[Claim], query_keys: set[str]) -> str | None:
        full_matches = [claim for claim in claims if _is_full_match(claim, query_keys)]
        if not full_matches:
            predicates = {claim.predicate for claim in claims if claim.predicate}
            if len(predicates) > 1:
                return (
                    "No single retrieved passage contains every key term of the "
                    "query, and the matched passages address different predicates; "
                    "the query is ambiguous as posed."
                )
            return None

        values_by_predicate: dict[str, set[str]] = {}
        for claim in full_matches:
            if claim.predicate and claim.value:
                values_by_predicate.setdefault(claim.predicate, set()).add(claim.value)

        distinct = [
            (predicate, value)
            for predicate, values in values_by_predicate.items()
            for value in values
        ]
        if len(distinct) > 1 and len({value for _, value in distinct}) == 1:
            return (
                "Multiple passages fully match the query but refer to different "
                "subjects with the same value; the intended subject is "
                "ambiguous."
            )
        return None

    def _select_single_aspect_claims(
        self, claims: list[Claim], query_keys: set[str]
    ) -> list[Claim] | None:
        full_matches = [claim for claim in claims if _is_full_match(claim, query_keys)]
        if full_matches:
            return [full_matches[0]]
        predicates = {claim.predicate for claim in claims if claim.predicate}
        if len(predicates) == 1:
            return [claims[0]]
        return None

    @staticmethod
    def _is_multi_aspect(query: str) -> bool:
        tokens = set(tokenize(query))
        markers = tokens & _MULTI_ASPECT_MARKERS
        wh_count = len(tokens & _WH_WORDS)
        return bool(markers) or wh_count >= 2

    def _answered(
        self,
        query: str,
        evidence: Sequence[Chunk],
        claims: list[Claim],
    ) -> GeneratedAnswer:
        evidence_by_id = {chunk.chunk_id: chunk for chunk in evidence}
        citations: list[Citation] = []
        seen: set[str] = set()
        for claim in claims:
            for chunk_id in claim.citation_chunk_ids:
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                chunk = evidence_by_id.get(chunk_id)
                if chunk is None:
                    continue
                citations.append(
                    Citation(
                        citation_id=f"cit-{len(citations) + 1}-{chunk_id}",
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        corpus_version=chunk.corpus_version,
                        excerpt=chunk.text,
                        location=chunk.location,
                    )
                )
        answer_text = " ".join(
            f"{claim.text} [{index + 1}]" for index, claim in enumerate(claims)
        )
        return GeneratedAnswer(
            state=AnswerState.ANSWERED,
            answer_text=answer_text,
            claims=tuple(claims),
            citations=tuple(citations),
            provider_id=self.provider_id,
            model_id=self.model_id,
            is_llm_generated=self.is_llm_generated,
            mode=self.mode,
        )

    def _abstain(
        self,
        reason: AbstentionReason,
        detail: str,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            state=AnswerState.CONFLICTING
            if reason is AbstentionReason.CONFLICTING_EVIDENCE
            else AnswerState.ABSTAINED,
            answer_text=None,
            abstention_reason=reason,
            abstention_detail=detail or reason_detail(reason),
            provider_id=self.provider_id,
            model_id=self.model_id,
            is_llm_generated=self.is_llm_generated,
            mode=self.mode,
        )


def _evidence_to_chunks(evidence: EvidencePack | Sequence[Chunk] | Any) -> list[Chunk]:
    if isinstance(evidence, EvidencePack):
        return [item.to_chunk() for item in evidence.items]
    if isinstance(evidence, Sequence):
        return [chunk for chunk in evidence if isinstance(chunk, Chunk)]
    return []


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def split_predicate_value(sentence: str) -> tuple[str | None, str | None]:
    normalized = " ".join(sentence.casefold().split())
    marker = " is "
    if marker not in normalized:
        return None, None
    predicate, value = normalized.split(marker, 1)
    predicate = _LEADING_ARTICLE_RE.sub("", predicate).strip()
    return (predicate or None, value.strip() or None)


def values_conflict(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    l_stripped = _LEADING_ARTICLE_RE.sub("", left).strip()
    r_stripped = _LEADING_ARTICLE_RE.sub("", right).strip()
    if _SCALAR_RE.match(l_stripped) and _SCALAR_RE.match(r_stripped):
        return l_stripped != r_stripped
    l_neg = l_stripped.startswith(_NEGATION_PREFIXES)
    r_neg = r_stripped.startswith(_NEGATION_PREFIXES)
    return l_neg != r_neg


def _is_full_match(claim: Claim, query_keys: set[str]) -> bool:
    if not query_keys:
        return False
    sentence_tokens = set(tokenize(claim.text))
    return query_keys.issubset(sentence_tokens)
