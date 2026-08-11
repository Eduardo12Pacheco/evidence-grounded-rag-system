# Demo Script — evidence-grounded-rag-system

Status: **scaffold / implementation not started**

## What the demo will show

A user asks a question and sees:

1. The retrieved evidence passages that were actually used.
2. The generated answer with inline citations to those passages.
3. A clear abstention case: a question the system refuses to answer because
   evidence is insufficient.

## Constraint

The demo runs ONLY on synthetic fixtures or corpus whose license contract has
been satisfied. It performs no live fetches and no unlicensed corpus use.

## Run book (once implemented)

```bash
uv sync --dev --extra demo
uv run --extra demo streamlit run scripts/demo_retrieval.py
```

## What the demo must NOT claim

- No real accuracy numbers before the evaluation protocol runs.
- No Gob.ec corpus use unless the corpus/license contract is satisfied.
