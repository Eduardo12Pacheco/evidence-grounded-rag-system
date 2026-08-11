# Design — evidence-grounded-rag-system

Status: **scaffold / implementation not started**

## Problem

RAG systems are judged on perceived fluency, not on whether the answer is
supported. This project builds a RAG pipeline where every answer is bound to
evidence: hybrid retrieval surfaces passages, generation cites them, abstention
refuses to fabricate, and faithfulness evaluation measures the gap between
claim and evidence.

## Planned architecture

```text
src/evidence_grounded_rag/
├── corpus/        # corpus loading + license-gated ingestion
├── retrieval/     # hybrid: sparse + dense fusion
├── embeddings/    # dense embedding interface
├── generation/    # generation with citation + abstention logic
└── evaluation/    # retrieval quality and faithfulness metrics
```

Each layer is behind a documented interface and is replaceable.

## Key decisions (scaffold phase)

| Topic | Decision |
| --- | --- |
| Language | Python >=3.11, `uv`, hatchling src layout |
| Retrieval | Hybrid sparse+dense with committed fusion rule |
| Abstention | Required behavior, tested; never optional |
| Evidence | Claims map to retrieved passages; citations verified |
| Demo | Local Streamlit app over fixtures only |

## No-goals

- No production search service or public deployment yet.
- No unlicensed corpus use, ever (Gob.ec is not authorized).
- No copying of the existing BM25 prototype code.
- No claims of benchmark parity without reproduced baselines.

## Risks

- Corpus license unresolved → corpus layer stays empty by design.
- Hallucination → mitigated by citation verification and mandatory abstention.
- Metric gaming → fixed protocols and committed query sets.

## Definition of done (for the project, not just scaffold)

- [ ] Corpus license contract satisfied for any used corpus.
- [ ] Hybrid retrieval, citation, and abstention implemented and tested.
- [ ] Faithfulness and retrieval evaluation executed and reported.
- [ ] Demo runs from fixtures with no network access.
- [ ] No unsupported claims in README or docs.
