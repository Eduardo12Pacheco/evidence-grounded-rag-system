# Retrieval & Evaluation Protocol — evidence-grounded-rag-system

Status: **committed formulas, committed query set, deterministic synthetic-fixture numbers.**

## Purpose

Fix the retrieval design and the evaluation protocol so quality claims are
reproducible rather than asserted. Every number below is produced by one
command against committed fixtures and a committed query set.

## Retrieval design

### Sparse — Okapi BM25 (`retrieval/bm25.py`)

Scoring for a query `q` and document `d`:

    score(q, d) = Σ_{t ∈ q} idf(t) · (tf(t,d) · (k1 + 1)) / (tf(t,d) + k1 · (1 − b + b · |d| / avgdl))

with Robertson-Sparck Jones idf:

    idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Committed parameters: `k1 = 1.5`, `b = 0.75`. Tokenization: `v1` (NFKC,
case-fold, split on non-alphanumeric runs, **no stemming** — "review" and
"reviewed" are distinct tokens). Ties break by (score desc, chunk id
ascending). Empty query, empty corpus, `top_k <= 0`, and no matching terms all
return an empty result.

### Dense — embedding interface (`embeddings/`)

`EmbeddingProvider` is the replaceable interface. The bundled
`DeterministicEmbedding` is a hashed bag-of-words vector (signed hashing, L2
normalized, dimension 256) with `semantic = False`, `mode = offline`. It is
NEVER labeled semantic: dense scores encode token overlap only. Ranking is
cosine similarity; ties break by (score desc, chunk id ascending); a query
vector whose dimension differs from the indexed dimension raises
`EmbeddingDimensionError`; an empty index returns an empty result.

### Fusion — reciprocal rank fusion (`retrieval/fusion.py`)

    rrf(chunk) = Σ_{systems s} 1 / (k + rank_s(chunk))

Committed parameter: `k = 60`. Candidates are deduplicated; ties break by
(fused score desc, chunk id ascending). A `min_score` threshold (default
`0.01`, see `PipelineConfig`) is applied to the fused ranking before evidence
is exposed to the generator. `top_k <= 0` yields an empty result.

### Modes

`hybrid` (default), `sparse`, and `dense` are supported by the hybrid engine.
In `sparse`/`dense` mode the answer is grounded on that single ranking, but the
trace always records both systems' candidates plus the fused ranking, the
committed versions, and the configuration.

## Evaluation protocol

### Query set (`data/fixtures/queries/synthetic_queries_v1.json`)

Version `1.0.0`, frozen against corpus `synthetic-fixtures` v1.0.0. 11
committed queries covering all seven required categories:

| Category | Queries | Expected |
| --- | --- | --- |
| answerable | q-002, q-005, q-006, q-011 | answered |
| unanswerable | q-009, q-010 | abstain `unsupported_query` |
| ambiguous | q-008 | abstain `insufficient_evidence` |
| conflicting | q-007 | abstain `conflicting_evidence` |
| citation_boundary | q-003 | answered, citation only to exposed chunk |
| numeric | q-001 | answered with the exact number |
| multi_passage | q-004 | answered with multiple claims/citations |

Each query declares `answerable`, `expected_state`, `expected_reason`, and
graded relevance judgments (chunk id + grade). The query set loader rejects
answered queries without judgments and requires every category to be covered.

### Retrieval metrics (macro-averaged over queries with ≥1 relevant judgment)

- **Recall@k** (k = 1, 5, 10): `|relevant ∩ top-k| / |relevant|`.
- **MRR**: `1 / rank(first relevant)`, else 0.
- **nDCG@k** (k = 5, 10): `DCG@k / IDCG@k` with `DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)`.

### Faithfulness metrics (over all generated records)

- **Answerable coverage**: answered / answerable queries.
- **Abstention rate**: abstained (including conflicting) / all queries
  (descriptive).
- **Abstention precision**: correct abstentions / abstentions.
- **Abstention recall**: correct abstentions / unanswerable queries. A
  rejection is "correct" when the query is not answerable AND the recorded
  reason matches the committed expected reason (when declared).
- **Unsupported-claim rate**: claims that failed citation validation / total
  claims.
- **Citation precision**: valid citations / citations checked.
- **Citation completeness**: claims with at least one valid citation / total
  claims.

### Citation validation rules

An answered generation is only reported as verified if every check passes;
otherwise the pipeline refuses the answer (abstention
`citation_validation_failed`) and never surfaces it:

- `citation_not_found` — chunk does not exist in the served corpus.
- `citation_not_exposed` — chunk was not part of the evidence handed to the generator.
- `corpus_version_mismatch` — citation belongs to another corpus version.
- `excerpt_mismatch` — cited excerpt differs from the chunk text.
- `invalid_location` — location missing or not the chunk's `synthetic://` location.
- `claim_without_citation` — a non-abstained claim cites nothing.
- `unused_citation` — a citation is not referenced by any claim (forged/orphan).
- Refusals are `skipped` by validation (never presented as verified).

### Abstention reasons (mandatory, stable codes)

`insufficient_evidence`, `unsupported_query`, `conflicting_evidence`,
`citation_validation_failed`, `provider_unavailable`, `malformed_output`,
`blocked_corpus`, `invalid_corpus`, `stale_corpus`. Human-readable details are
committed in `src/evidence_grounded_rag/abstention/reasons.py`.

## Materializing the report

```bash
uv run python -m evidence_grounded_rag.evaluation.materialize
# optional: --manifest, --queries, --out, --top-k, --mode
```

Writes `data/manifests/evaluation-synthetic-v1.0.0.json` with versions,
configuration, model identities, per-query outcomes, metrics, a timestamp,
explicit limitations, and the labels `synthetic_evaluation: true`,
`corpus_provenance: project-generated-synthetic`,
`real_corpus_gate_blocked: true`. Metrics are deterministic; the timestamp is
recorded provenance.

## Current synthetic-fixture numbers (committed report)

```text
recall@1  0.8333   recall@5  1.0000   recall@10  1.0000
mrr       1.0000   ndcg@5    1.0000   ndcg@10    1.0000
answerable_coverage      1.0000
abstention_rate          0.3636
abstention_precision     1.0000
abstention_recall        1.0000
unsupported_claim_rate   0.0000
citation_precision       1.0000
citation_completeness    1.0000
```

Recall@1 is below 1 because multi-relevant queries (q-004, q-007, q-008) count
all their relevant chunks in the denominator. These numbers describe the
committed synthetic fixtures and query set only.

## Honesty rules

- Every number is reproducible from one command and a committed query set.
- No metric is reported before its formula is committed in this doc.
- Synthetic-fixture evaluation is labeled as such in every report and in the demo.
- No claim of real-world validation, perfect metrics, hallucination-free
  behavior, trustworthiness, or production readiness.

## Definition of done for evaluation

- [x] Query sets and metric formulas are committed.
- [x] One command reproduces all reported numbers.
- [x] Abstention behavior is tested, not optional.
