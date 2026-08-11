# AGENTS.md — evidence-grounded-rag-system

## Objective

Build hybrid retrieval (sparse + dense embeddings), evidence-grounded
generation with citations, abstention when evidence is insufficient, and
faithfulness evaluation. Independent, greenfield project. It MUST NOT copy the
existing BM25 prototype and MUST NOT present the Gob.ec corpus as authorized.

## Status

Scaffold / implementation not started. No metrics, no claims of results.

## Boundaries

- Do NOT modify these sibling projects: `ecuador-job-market-intelligence`,
  `ecuador-mobility-reliability`, `demand-inventory-optimizer`,
  `ecuador-public-information-evidence-assistant`, `eduardo-github-profile`.
- Do NOT push, publish, or create GitHub remotes.
- **No real corpus is implemented until its license is confirmed**
  (see `docs/corpus-license-contract.md`).

## Structure

```text
src/evidence_grounded_rag/     # package under src layout
tests/                         # pytest; real tests only, no fake coverage
docs/                          # corpus/license contract, retrieval & evaluation protocol, demo script
data/fixtures/                 # small versioned fixtures (never a real corpus)
data/manifests/                # versioned manifests of captured artifacts
data/raw/ data/processed/      # gitignored runtime output
deploy/                        # deployment notes (later)
```

## Expected commands

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run --extra demo streamlit run scripts/demo_retrieval.py
```

## Data policy

- The Gob.ec corpus is NOT implemented and NOT claimed as authorized until the
  license contract is satisfied.
- Never commit `data/raw/` or `data/processed/`.
- Commit small fixtures and manifests only.

## CodeGraph

Use `.codegraph/` index for structural queries. Never commit its contents.

## Testing

Tests must verify real behavior that exists. No placeholder tests pretending
features exist.
