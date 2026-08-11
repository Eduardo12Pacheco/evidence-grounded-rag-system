# Design — evidence-grounded-rag-system

Status: **implementation in progress — offline synthetic-fixture evaluation only.**

## Problem

RAG systems are judged on perceived fluency, not on whether the answer is
supported. This project builds a RAG pipeline where every answer is bound to
evidence: hybrid retrieval surfaces passages, generation cites them, abstention
refuses to fabricate, and faithfulness evaluation measures the gap between
claim and evidence.

## Architecture

```text
src/evidence_grounded_rag/
├── corpus/        # license-gated corpus loading, IDs, checksums, normalization
├── text/          # shared deterministic tokenizer (used by retrieval/embeddings)
├── retrieval/     # BM25 + dense + RRF fusion + hybrid engine and traces
├── embeddings/    # embedding interface + deterministic non-semantic embedding
├── generation/    # claim/citation models, extractive provider, env adapter
├── validation/    # strict citation validation (fail-closed)
├── abstention/    # mandatory abstention reason codes
├── evaluation/    # committed metrics, versioned query set, materialize command
└── pipeline.py    # end-to-end EvidenceRagPipeline (retrieve → generate → validate)
```

Each layer is behind a documented interface and is replaceable.

## Key decisions (implemented)

| Topic | Decision |
| --- | --- |
| Language | Python >=3.11, `uv`, hatchling src layout; stdlib for the core |
| Sparse retrieval | Okapi BM25, `k1 = 1.5`, `b = 0.75`, Robertson-Sparck Jones idf, deterministic tokenizer (`v1`, no stemming) |
| Dense retrieval | `EmbeddingProvider` protocol; bundled `DeterministicEmbedding` is a hashed bag-of-words vector with `semantic = False` (offline, no weights, never labeled semantic) |
| Fusion | Reciprocal rank fusion, `k = 60` (committed; changing it changes every number) |
| Generation | Deterministic offline extractive/template provider (`is_llm_generated = False`); environment-configured adapter fails closed to `provider_unavailable` for any non-offline provider |
| Abstention | Required behavior, tested; every refusal carries a stable reason code |
| Evidence | Claims map to retrieved passages only; generator never sees the full corpus |
| Citations | Strict validation: exists, exposed, current corpus version, exact excerpt, stable location, referenced by a claim; unverified answers are never surfaced |
| Corpus gate | Fails closed: blocked real corpora, unapproved licenses, checksum/format/duplicate errors, and stale versions all refuse to serve |
| Evaluation | Committed formulas + committed versioned query set; one command reproduces all numbers |
| Demo | Local Streamlit app over committed fixtures only |

## Decision procedure of the extractive generator

The generator receives only the retrieved evidence chunks and applies a
committed, fully deterministic procedure:

1. Empty evidence → abstain `insufficient_evidence`.
2. Extract matched claims (sentences containing a key token of the query). None
   → abstain `unsupported_query`.
3. Conflict: among claims that fully match the query's key tokens (falling back
   to all matched claims when none are full), two claims from different chunks
   with the same normalized predicate but different scalar values (or explicit
   negation) → `conflicting_evidence`.
4. Multi-aspect query (coordination marker or two wh-words) → answered with the
   top matched claims.
5. Otherwise (single-aspect): full matches from different chunks with different
   subjects but identical values → ambiguous → `insufficient_evidence`; else
   answered with the top full match (or the top claim when all matched claims
   share one predicate).

The conflict/ambiguity rules are deliberately narrow and documented in
`src/evidence_grounded_rag/generation/extractive.py`; they are not semantic
understanding.

## No-goals

- No production search service or public deployment.
- No unlicensed corpus use, ever (Gob.ec is not authorized).
- No copying of the existing BM25 prototype code.
- No claims of benchmark parity, real-world validation, perfect metrics,
  hallucination-free behavior, trustworthiness, or production readiness.

## Risks

- Corpus license unresolved → real-corpus layer stays blocked by design.
- Hallucination → mitigated by citation verification, mandatory abstention,
  and fail-closed validation (unverified answers are never surfaced).
- Metric gaming → fixed protocols, committed formulas, and committed query sets.

## Definition of done (for the project, not just the current milestone)

- [x] License-gated corpus layer over synthetic fixtures (real corpus blocked).
- [x] Hybrid retrieval (BM25 + deterministic dense + RRF) implemented and tested.
- [x] Citation, abstention, and strict validation implemented and tested.
- [x] Deterministic evaluation over the committed query set, reproduced by one command.
- [x] Demo runs from fixtures with no network access.
- [x] No unsupported claims in README or docs.
- [ ] Real corpus license contract satisfied for any real corpus (blocked).
