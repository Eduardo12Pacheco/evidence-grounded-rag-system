# Evidence-Grounded RAG System

Status: **implementation in progress — synthetic-fixture evaluation is the
default; one real-local corpus (Python 3.12.13 documentation) is approved for
local development only.**

A greenfield RAG system where answers are bound to evidence: hybrid retrieval
(sparse + dense), embeddings, generation with citations, mandatory abstention,
and faithfulness evaluation. This README is the project contract and records
exactly what is implemented, what is blocked, and what is not claimed.

## What this is

- **Hybrid retrieval** — sparse (Okapi BM25) and dense (deterministic,
  explicitly *non-semantic* embedding) fused by reciprocal rank fusion
  (RRF, `k = 60`). All parameters are committed and versioned.
- **Structured generation** — a deterministic offline extractive/template
  provider (`is_llm_generated = False`) that turns retrieved passages into
  claims, each cited to the exact chunk.
- **Mandatory abstention** — every refusal carries a stable reason code
  (insufficient evidence, unsupported query, conflicting evidence, citation
  validation failure, provider unavailable, malformed output, blocked/invalid/
  stale corpus).
- **Strict citation validation** — a citation is only valid if it exists, was
  exposed to the generator, belongs to the served corpus version, matches the
  excerpt exactly, and is referenced by a claim. An unverified answer is never
  surfaced.
- **Faithfulness evaluation** — committed metric formulas (Recall@k, MRR,
  nDCG@k, answerable coverage, abstention precision/recall, unsupported-claim
  rate, citation precision/completeness) over a committed, versioned query set
  covering answerable / unanswerable / ambiguous / conflicting /
  citation-boundary / numeric / multi-passage cases.

## What this is NOT

- **No served real corpus.** The serving gate is BLOCKED for every real corpus
  except one explicitly approved local-only source: Python 3.12.13
  documentation (`approved-for-local-development`, see below). Every number
  reported here describes **synthetic fixtures only**; the real-local corpus
  never feeds the default demo or synthetic evaluation.
- **Gob.ec is not used.** The Gob.ec corpus is explicitly unauthorized: its
  "Creative Commons Attribution 4.0 International" label conflicts with the
  linked `by/3.0/ec` terms, and the license discrepancy is unresolved. See
  `docs/corpus-license-contract.md` for why the Python docs were selected and
  Gob.ec was not.
- **No claims of quality.** No real-world validation, no "perfect" metrics, no
  hallucination-free guarantee, no trustworthiness or production-readiness
  claim. The deterministic embedding is not semantic.
- No API, no deployment, no unlicensed corpus indexing.

## Relationship to the existing BM25 prototype

This is an independent project. It does not copy the BM25 prototype's code or
structure, and it does not present the Gob.ec corpus as authorized. The license
gate, abstention discipline, and synthetic-fixture honesty principles are
aligned with the project's own contracts, not derived from any prototype.

## Architecture

```text
src/evidence_grounded_rag/
├── corpus/        # synthetic gate+loader AND source-manifest audit, real-local derivation, privacy
├── text/          # shared deterministic tokenizer (used by retrieval/embeddings)
├── retrieval/     # BM25 + dense + RRF fusion + hybrid engine and traces
├── embeddings/    # embedding interface + deterministic non-semantic embedding
├── generation/    # claim/citation models, extractive provider, env adapter
├── validation/    # strict citation validation (fail-closed)
├── abstention/    # mandatory abstention reason codes
├── evaluation/    # committed metrics, versioned query set, materialize command
└── pipeline.py    # end-to-end EvidenceRagPipeline (retrieve → generate → validate)
```

### Committed components and versions

| Component | ID / version | Notes |
| --- | --- | --- |
| Corpus format | `1.0` | JSON / JSONL manifests, SHA-256 checksums |
| Source manifest | `1.0` | source-manifest audit boundary for the approved local corpus |
| Tokenizer | `v1` | NFKC + case-fold, no stemming |
| Sparse retriever | `okapi-bm25-v1` | `k1 = 1.5`, `b = 0.75`, Robertson-Sparck Jones idf |
| Embedding | `deterministic-hashed-bow-v1` | `semantic = False`, offline, no weights |
| Fusion | `rrf-v1` | RRF with `k = 60` |
| Generator | `extractive-template-v1` | not an LLM |
| Query set | `synthetic-queries-v1` | 11 committed queries, 7 categories |

### Current synthetic-fixture numbers

Deterministic evaluation over the committed fixtures and query set
(`data/fixtures/*`); reproduced with one command:

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

These describe 6 fictional documents about a fictional "Aurora City". They
are **not** evidence about any real institution or regulation.

## Running the system

```bash
uv sync --dev --extra demo
uv run pytest                      # real behavior tests, offline
uv run ruff check .                # lint
uv run python scripts/materialize_fixtures.py   # regenerate committed fixtures
uv run python -m evidence_grounded_rag.evaluation.materialize   # deterministic report
uv run --extra demo streamlit run scripts/demo_retrieval.py     # offline demo
```

The demo runs only on committed synthetic fixtures, makes no network calls,
and uses no credentials. See `docs/demo-script.md` for the walkthrough.

## Real-local evaluation (separate, diagnostic)

There is a SEPARATE bounded evaluation over the approved real-local corpus
(Python 3.12.13 documentation, 3 files / 70 chunks). It uses its own committed
query set (`data/evaluation/queries/python-docs-real-v1.0.0.json`, 18 queries),
its own command, and its own report
(`data/evaluations/python-docs-real-evaluation-v1.0.0.json`). It is NEVER
combined with the synthetic report and never falls back to the synthetic corpus.

```bash
# 1. Verify the approved local corpus against its source manifest (offline)
uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json

# 2. Run the real-local evaluation (offline, no synthetic fallback)
uv run python -m evidence_grounded_rag.evaluation.real_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --output data/evaluations/python-docs-real-evaluation-v1.0.0.json
```

Every number in the real report is a **diagnostic** result: the dense mode is a
deterministic non-semantic hash embedding (never labeled semantic), generation
is deterministic extraction (not an LLM), and the report covers three files and
70 chunks — it is NOT validation of Python documentation in general and is NOT
production readiness. The real report stores no chunk or claim text; citations
carry excerpt checksums and short prefixes. Timestamp policy: only
`timestamp_utc` differs between runs; the deterministic portions are covered by
`deterministic_digest`.

Synthetic evaluation (unchanged): `uv run python -m
evidence_grounded_rag.evaluation.materialize` writes
`data/manifests/evaluation-synthetic-v1.0.0.json`.

## Real-local corpus (approved, local-only, not the default)

The **only** approved real source is the Python 3.12.13 official documentation
(publisher: Python Software Foundation; license: Python Software Foundation
License Version 2). The approval, candidate/rejection record, license
evidence, and pinned revision live in `docs/corpus-license-contract.md`.

Three modes exist and are deliberately distinct:

| Mode | What serves it | Status |
| --- | --- | --- |
| **synthetic** (default) | `data/fixtures/synthetic_manifest.json` via the synthetic gate | demo + evaluation + README numbers |
| **real-local** | `data/raw/python-docs-3.12.13/` via the source-manifest audit | explicit opt-in; offline scripts only; never the demo or synthetic evaluation |
| **blocked** | any other real corpus (incl. Gob.ec) | refused by both gates |

Acquisition and verification (offline, standard library only):

```bash
uv run python scripts/acquire_corpus.py \
    # downloads 3 commit-pinned files → data/raw/python-docs-3.12.13/ (gitignored)
    # writes data/manifests/corpus-source-v1.0.0.json with real sizes + raw SHA-256

uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json
    # offline: re-checks manifest/version/revision, files, exact bytes, sizes,
    # SHA-256, privacy scan, then derives the corpus in memory and prints a summary
```

Raw data policy:

- Raw capture lives only under `data/raw/` (gitignored, never committed); the
  committed artifact is the small source manifest with byte sizes and raw
  SHA-256 values, not the text.
- Raw SHA-256 is computed over the exact captured bytes; normalized SHA-256
  over the normalized text; chunk SHA-256 per derived chunk. They are distinct
  by design and all recorded on the derived documents.
- Every captured file is privacy-audited (`corpus/privacy.py`); flagged
  documents are excluded and cannot enter derived data.
- The real-local corpus is derived in memory by
  `corpus/real_local.py::build_real_local_corpus` and carries source URL,
  license, revision, and checksums forward on every document.

## Evidence requirements (unchanged contract)

- Corpus license contract satisfied before any real corpus
  (`docs/corpus-license-contract.md`).
- Retrieval and evaluation protocol committed before results
  (`docs/retrieval-evaluation-protocol.md`).
- Every number reproducible from one command and a committed query set.
- Abstention is tested as a required, non-optional feature.

## Data and license limits

- Third-party documents retain their own terms; this repo's MIT license covers
  only original code and the synthetic fixtures.
- The approved real-local corpus (Python 3.12.13 docs) is used for local
  development only; its raw text is never committed, packaged, or published.
- No unlicensed corpus is indexed, ever.
- `data/raw/` and `data/processed/` are never committed.
- Small synthetic fixtures and their manifests are committed for offline
  development. The source manifest
  (`data/manifests/corpus-source-v1.0.0.json`) is committed; the raw capture
  it describes is not.
