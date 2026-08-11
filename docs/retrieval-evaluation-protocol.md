# Retrieval & Evaluation Protocol — evidence-grounded-rag-system

Status: **scaffold / implementation not started**

## Purpose

Fix the retrieval design and the faithfulness evaluation protocol BEFORE any
implementation, so quality claims are reproducible rather than asserted.

## Retrieval design

1. **Hybrid retrieval** — sparse (BM25-style) and dense (embeddings) ranked
   results merged by a documented fusion rule. The fusion parameters are
   committed.
2. **Evidence passages** — every generated claim must map to retrieved passages
   that support it.
3. **Abstention** — when retrieved evidence is insufficient or contradicts the
   question, the system must abstain rather than fabricate.

## Evaluation protocol

- **Retrieval quality** — standard metrics (recall@k, nDCG@k, MRR) over a
  labeled query set; the labeled set is versioned.
- **Faithfulness** — generated claims are checked against retrieved passages;
  unsupported claims are counted (precision of supported claims, and rate of
  abstentions).
- **Citation grounding** — a claim is grounded only if its cited passage
  actually supports it; citation errors are measured.

## Honesty rules

- Every number is reproducible from one command and a committed query set.
- No metric is reported before its formula is committed in this doc.
- The BM25 prototype is not a benchmark claim source; baselines are reproduced
  in-repo.

## Definition of done for evaluation

- [ ] Query sets, splits, and metric formulas are committed.
- [ ] One command reproduces all reported numbers.
- [ ] Abstention behavior is tested, not optional.

## Demo script

See `docs/demo-script.md` for the human-facing walkthrough.
