# Evidence-Grounded RAG System

Status: **scaffold / implementation not started.**

A greenfield RAG system where answers are bound to evidence: hybrid retrieval,
embeddings, generation with citations, mandatory abstention, and faithfulness
evaluation. This README is the project contract and will be updated as the
system becomes real.

## What this will be

- Hybrid retrieval (sparse + dense) with a documented fusion rule.
- Generation that cites the retrieved passages behind every claim.
- Mandatory abstention when evidence is insufficient or conflicting.
- A faithfulness evaluation protocol that measures claim-to-passage support.

## What this is NOT yet

- No code beyond the workspace scaffold.
- No corpus, no metrics, no claimed results.
- **No real corpus is implemented until its license is confirmed.** The Gob.ec
  corpus is NOT authorized and is NOT presented as such
  (see `docs/corpus-license-contract.md`).

## Relationship to the existing BM25 prototype

This is an independent project. It does not copy the BM25 prototype's code and
does not present the Gob.ec corpus as authorized. Any baselines are reproduced
in-repo.

## Planned architecture

```text
src/evidence_grounded_rag/
├── corpus/        # corpus loading + license-gated ingestion
├── retrieval/     # hybrid: sparse + dense fusion
├── embeddings/    # dense embedding interface
├── generation/    # generation with citation + abstention logic
└── evaluation/    # retrieval quality and faithfulness metrics
```

Design decisions and no-goals: `DESIGN.md`. Product intent: `PRODUCT.md`.

## Evidence that will be required

- Corpus license contract satisfied (`docs/corpus-license-contract.md`).
- Retrieval and evaluation protocol committed BEFORE results
  (`docs/retrieval-evaluation-protocol.md`).
- Every number reproducible from one command and a committed query set.
- Abstention behavior tested as a required, non-optional feature.

## Data and license limits

- Third-party documents retain their own terms; this repo's MIT license covers
  only original code.
- No unlicensed corpus is indexed, ever.
- `data/raw/` and `data/processed/` are never committed.
- Small synthetic fixtures are committed for offline development.

## Running the scaffold

```bash
uv sync --dev
uv run pytest          # passes: workspace contract tests only
uv run ruff check .
```

The demo command will be added when the demo exists
(`docs/demo-script.md`).
