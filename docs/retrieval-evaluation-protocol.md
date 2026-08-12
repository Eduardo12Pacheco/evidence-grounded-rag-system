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

`EmbeddingProvider` is the replaceable interface (protocol with `model_id`,
`mode`, `semantic`, `dimension`, `embed`, `embed_one`, and `to_dict`). Two
providers exist:

- **`DeterministicEmbedding`** (default) is a hashed bag-of-words vector
  (signed hashing, L2 normalized, dimension 256) with `semantic = False`,
  `mode = offline`. It is NEVER labeled semantic: dense scores encode token
  overlap only.
- **`SemanticLocalEmbedding`** is optional, explicit, local-only
  (BAAI/bge-small-en-v1.5, `semantic = True`, `mode = semantic-local`, 384
  dimensions, CLS pooling, L2 normalized). It loads ONLY from a verified local
  cache with `local_files_only=True` and raises `SemanticModelUnavailableError`
  when unavailable — it never silently falls back to the hash embedding. See
  `docs/embedding-model-contract.md`.

Ranking is cosine similarity; ties break by (score desc, chunk id ascending); a
query vector whose dimension differs from the indexed dimension raises
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

## Real-local evaluation (separate, diagnostic)

A separate bounded evaluation over the single approved real-local corpus
(Python 3.12.13 documentation, 3 files / 70 chunks). It is NEVER combined with
the synthetic report: separate query set, separate command, separate report
file, separate labels.

### Real query set (`data/evaluation/queries/python-docs-real-v1.0.0.json`)

Version `1.0.0`, frozen against corpus `python-docs-3.12.13` v1.0.0 via source
manifest `corpus-source-v1.0.0` (revision `3bb231a6...`). 18 committed queries
covering every required case:

| Case | Queries |
| --- | --- |
| single_passage | real-q-001, real-q-012 |
| multi_passage | real-q-006, real-q-007, real-q-008 |
| exact_api | real-q-004, real-q-010 |
| terminology | real-q-003, real-q-009 |
| numeric_parameter | real-q-002, real-q-011 |
| section_distinction | real-q-005 |
| citation_boundary | real-q-009 |
| unanswerable | real-q-013, real-q-014 |
| ambiguous | real-q-015, real-q-016 |
| abstain | real-q-017, real-q-018 |

Each query declares `query_id`, `case`, `question`, `expected_status`
(`answerable | unanswerable | ambiguous | abstain`), graded gold chunks,
expected citation-count range, expected abstention reason(s), notes, and an
annotation record. The loader (`evaluation/real_queries.py`) validates schema,
duplicate IDs, corpus id/version and manifest reference, chunk existence in the
derived corpus, case/status coverage, and deterministic load. The synthetic
seven-category loader is untouched.

### Annotation procedure (reproducible)

1. Build the derived corpus in memory from the approved source manifest and
   local raw files (`build_real_local_corpus`); chunk IDs are
   `derive_chunk_id(document_id, seq)`, verified against the manifest revision.
2. For each question, locate the sentence(s) that answer it and record the
   derived chunk ID(s); grade 2 = primary, 1 = secondary.
3. Phrase the question with the exact token forms present in the gold sentence
   because the committed extractive generator has no stemming and requires
   every key token of the query in one sentence for a full match.
4. Declare the expected status from the annotator's judgment of the corpus
   content, independent of any retrieval mode.

### Command

```bash
uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json

uv run python -m evidence_grounded_rag.evaluation.real_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --output data/evaluations/python-docs-real-evaluation-v1.0.0.json
```

The command verifies the manifest, audit, files, sizes, raw SHA-256, and the
privacy scan first, then builds the real-local corpus. On ANY failure it exits
non-zero with the stable error code and never falls back to the synthetic
corpus. It makes no network calls.

### Report (`data/evaluations/python-docs-real-evaluation-v1.0.0.json`)

Records corpus/source/license/manifest/revision/raw checksums/canonical
checksum/doc and chunk counts, query set, configs and thresholds, per-mode
metrics, per-query records, retrieval traces, failures, timestamp,
implementation commit, and limitations. Required labels:

- Dense mode: **`Deterministic hash embedding diagnostic — not semantic retrieval`**
- Hybrid mode: **`Hybrid diagnostic using BM25 plus deterministic non-semantic hash embeddings`**
- Generation: **`Offline deterministic extractive evaluation — not an LLM result`**
- The report explicitly states it is a bounded evaluation over three Python
  3.12.13 documentation files and 70 chunks, does not represent all Python
  documentation, uses non-semantic hash embeddings, deterministic (non-LLM)
  generation, and is diagnostic, not production readiness.

No chunk-level or claim-level text is stored in the report; citations carry
excerpt checksums and short prefixes only.

### Metrics (computed per retrieval mode: sparse, dense, hybrid)

Retrieval metrics (same formulas as the synthetic protocol, macro-averaged over
queries with ≥1 gold judgment), reported overall and grouped by answerable,
single_passage, multi_passage, unanswerable, ambiguous, and abstain:

- Recall@1/5/10, MRR, nDCG@5 (graded).
- No-result rate (queries whose fused result is empty after the threshold).
- Top-k hit counts (queries whose top-k contains ≥1 gold chunk).

Faithfulness metrics over all 18 queries per mode:

- Answerable coverage: answered / answerable queries.
- Abstention precision: correct abstentions / abstentions.
- Abstention recall: correct abstentions / non-answerable queries (a rejection
  is correct when the query is not answerable AND the recorded reason is one of
  the query's declared expected abstention reason(s)).
- Citation precision: valid citations / citations checked.
- Citation completeness: claims with ≥1 valid citation / total claims.
- Unsupported-claim rate: claims without a valid citation / total claims.
- Citation-validation failure rate: answered generations refused by validation
  / total queries.
- Wrong-answer count: answered queries whose expected status is not answerable.

### Timestamp and determinism policy

`timestamp_utc` is runtime provenance and the only field allowed to differ
between runs. All other fields are deterministic and covered by
`deterministic_digest` (SHA-256 of the canonical serialization without the
timestamp and digest keys); two runs produce byte-identical deterministic
portions. Raw retrieval scores are quantized to 12 decimal places in the report
because BM25's in-memory summation adds term contributions in hash-randomized
dict/set order, which can move the last ulp of a float between processes;
rankings (which drive every metric) are unchanged.

### Honesty for the real-local evaluation

- The real report is a diagnostic over 3 files / 70 chunks of Python 3.12.13
  documentation; it is NOT validation of Python documentation in general.
- Dense mode is a deterministic non-semantic hash embedding, not semantic
  retrieval; generation is deterministic extraction, not an LLM.
- Results are diagnostic, not production readiness.
- Denominators are small (12 answerable / 6 non-answerable queries per mode);
  one query moves percentages by up to 8.3 points.

## Semantic-materializer evaluation (separate, retrieval-only)

A SEPARATE retrieval evaluation over the same real-local corpus with the local
semantic embedding model (BAAI/bge-small-en-v1.5). It is NEVER combined with
the synthetic report or with the hash-materializer report
(`python-docs-real-evaluation-v1.0.0.json`); it never falls back to hash
embeddings under a semantic label. See `docs/embedding-model-contract.md` for
the model audit, license, revision, and checksum policy.

### Model acquisition and verification (explicit; network only here)

```bash
uv sync --dev --extra demo --extra embeddings
uv run python scripts/acquire_embedding_model.py \
    --manifest data/manifests/embedding-model-v1.0.0.json
# downloads the pinned revision into a staging dir, verifies it byte-exact
# against the committed manifest, then atomically swaps it into
# data/raw/models/bge-small-en-v1.5/ (gitignored); the committed manifest is
# never rewritten (first-time bootstrap requires the explicit --bootstrap flag)

uv run python scripts/verify_embedding_model.py \
    --manifest data/manifests/embedding-model-v1.0.0.json
# offline: re-checks manifest schema/version, model id/version, publisher,
# URLs, revision, license, dimension (384), max input (512), device,
# architecture/tokenizer/pooling/normalization metadata, status,
# redistribution, then file existence + exact sizes + SHA-256 and rejects
# extra files beyond expected_files; any mismatch stops semantic enablement
```

### Command

```bash
uv run python -m evidence_grounded_rag.evaluation.semantic_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --model-manifest data/manifests/embedding-model-v1.0.0.json \
    --cache-dir data/raw/models/bge-small-en-v1.5 \
    --output data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json
```

The command verifies the real corpus first, then loads the semantic model only
from the local cache (`local_files_only=True`). If the model snapshot cannot be
verified/loaded it raises `SemanticModelUnavailableError` and writes no report;
semantic modes are refused and never degrade to the hash embedding.

### Four separated modes

| Mode key | Label | What serves it |
| --- | --- | --- |
| `sparse` | `BM25 sparse baseline` | committed Okapi BM25 only |
| `hash` | `Deterministic hash embedding diagnostic — not semantic retrieval` | deterministic non-semantic hash embedding |
| `semantic` | `Local semantic embedding model` | BAAI/bge-small-en-v1.5 dense, local-only |
| `hybrid` | `Hybrid BM25 plus local semantic embeddings` | BM25 + local semantic embeddings fused with committed RRF (`k = 60`) |

All existing thresholds are unchanged: `min_fused_score = 0.01`, `top_k = 10`,
BM25 `k1 = 1.5` / `b = 0.75`, RRF `k = 60`.

### Report (`data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json`)

Retrieval metrics only (no generation, no LLM). It records the model
facts/provenance/license/tokenizer/dimension/normalization/device/cache path
and per-file checksums, corpus and query-set facts, all four modes, retrieval
metrics (same formulas as the real protocol: Recall@1/5/10, MRR, nDCG@5,
no-result rate, top-k hit counts, grouped by answerable/single-passage/
multi-passage/unanswerable/ambiguous/abstain), per-query rankings and hit
ranks, ranking stability across two in-process runs, runtime observations
(wall-clock seconds per mode; process peak RSS if measured; per-mode memory
"unavailable"), timestamp, code revision, and limitations.

Determinism policy: metrics, per-query rankings, model facts, and configs are
deterministic and covered by `deterministic_digest`; only the timestamp and
runtime observations differ between runs (runtime is measured provenance, not
content). The semantic report explicitly states: bounded 3 files / 70 chunks,
no general-Python claim, no LLM, retrieval quality does not imply answer
faithfulness, and no model-card benchmark is claimed as a project metric.

Run the command twice and compare `deterministic_digest` and per-query
`retrieved_chunk_ids` for cross-run stability; both runs in the same process
are additionally reported under `ranking_stability`.

## Generation evaluation (separate, offline, diagnostic)

A SEPARATE offline evaluation of the deterministic extractive generator over
the same approved real-local corpus and the UNCHANGED real query set
(`python-docs-real-v1`, 18 queries). It is never combined with the synthetic,
hash-materializer, or semantic-materializer reports and never falls back to the
synthetic corpus. Generation policy is committed in
`docs/generation-contract.md` (prompt `grounded-answer-prompt-v1.0.0`,
structured schema `structured-answer-v1`, provider selection blocker, live
adapter that fails closed).

### Command

```bash
uv run python -m evidence_grounded_rag.evaluation.generation_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --fixture data/fixtures/answer-evaluation-v1.0.0.json \
    --output data/evaluations/python-docs-real-generation-evaluation-v1.0.0.json
```

The command verifies the real corpus first, then loads the unchanged real query
set and cross-validates it against the committed answer-evaluation fixture
(`data/fixtures/answer-evaluation-v1.0.0.json`), which references the 18 query
IDs and their gold chunks WITHOUT copying any passage. It then runs the offline
deterministic generator with strict citation validation and the claim-support
evaluator, mirroring the serving pipeline's fail-closed rules. No network, no
LLM, no synthetic fallback.

### Report (`data/evaluations/python-docs-real-generation-evaluation-v1.0.0.json`)

Records corpus/query-set/fixture facts, configs (BM25/RRF/thresholds unchanged,
`prompt_version`, `schema_version`, claim-support rule), metrics, per-query
records, failures, timestamp, runtime, implementation commit, and limitations.

Metrics (all over the 18 queries in the evaluated mode, hybrid by default):

- **Answer coverage**: answered / answerable queries.
- **Abstention precision**: correct abstentions / abstentions.
- **Abstention recall**: correct abstentions / non-answerable queries.
- **Abstention rate** (descriptive) and **correct abstention count**.
- **Citation precision**: valid citations / citations checked.
- **Citation completeness**: claims with ≥1 valid citation / total claims.
- **Citation-validation failure rate**: answered generations refused by
  validation / total queries.
- **Claim support**: supported / unsupported / unknown claim counts and
  `support_rate` (supported / total claims) under the committed
  `normalized-exact-containment-v1` rule.
- **Invalid-output and provider-error counts** (zero for the offline run).
- Per-query fields: query id/case/expected status, state, abstention reason and
  truncated detail, answerable/answered/abstained flags, correct abstention,
  wrong answer, retrieved chunk ids and quantized ranks/scores, claim ids with
  support verdicts, citation checksums/prefixes, validation summary, and
  generation provenance (`provider_id`, `model_id`, `mode=offline`,
  `is_llm_generated=False`).

Required labels: generation is labeled exactly
**`Offline deterministic extractive evaluation — not an LLM result`**, and live
generation is labeled **`Live generation NOT measured`** because no
provider/model/credentials are configured. The report stores no claim or chunk
text; citations carry excerpt checksums and short prefixes.

### Timestamp and determinism policy

`timestamp_utc` and `runtime_seconds` are runtime provenance — the ONLY fields
allowed to differ between runs. Every other field is deterministic and covered
by `deterministic_digest` (SHA-256 of the canonical serialization without those
keys); two runs produce byte-identical deterministic portions. Raw retrieval
scores are quantized to 12 decimal places (same policy as the hash materializer).

## Definition of done for evaluation

- [x] Query sets and metric formulas are committed.
- [x] One command reproduces all reported numbers.
- [x] Abstention behavior is tested, not optional.
