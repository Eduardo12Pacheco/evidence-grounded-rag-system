# Corpus & License Contract — evidence-grounded-rag-system

Status: **scaffold / implementation not started**

## Purpose

Define the conditions under which ANY document corpus may be used. This is a
hard gate: **no real corpus is implemented until its license is confirmed.**
The Gob.ec corpus is explicitly NOT authorized and is NOT presented as such.

## Mandatory gate

- Corpus acquisition and indexing is BLOCKED until the license terms of the
  source documents are audited, recorded here, and judged compatible.
- A corpus whose terms are unknown or incompatible is never indexed.
- The existing BM25 prototype's approach may inform design but its code and
  corpus handling are NOT copied.

## Required per corpus

| Property | Requirement |
| --- | --- |
| Provenance | Source URL(s) and retrieval date recorded |
| License | Terms documented verbatim with citation |
| Reuse rights | Explicit: indexing, transformation, redistribution |
| Attribution | Required credit recorded |
| Volume limits | Size or rate constraints documented |
| Versioning | Corpus snapshot is reproducible and versioned |

## Small-fixture exception

Small hand-authored fixtures for offline development are allowed under this
repository's own MIT license, but they must never represent a real corpus.

## Acceptance

- [ ] License terms documented verbatim for any candidate corpus.
- [ ] Compatibility judgment recorded with reasoning.
- [ ] Fixtures (if any) are clearly synthetic and small.
- [ ] No Gob.ec or other unlicensed corpus is present in the repository.
