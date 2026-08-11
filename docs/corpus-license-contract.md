# Corpus & License Contract — evidence-grounded-rag-system

Status: **gate implemented; real corpus BLOCKED. Only synthetic fixtures are served.**

## Purpose

Define the conditions under which ANY document corpus may be used, and record
the implemented gate that enforces them in code. This is a hard gate: **no real
corpus is implemented until its license is confirmed.** The Gob.ec corpus is
explicitly NOT authorized and is NOT presented as such.

## Mandatory gate (enforced in code)

The gate lives in `src/evidence_grounded_rag/corpus/license_gate.py` and is
applied by `src/evidence_grounded_rag/corpus/loader.py` on every load. It
FAILS CLOSED:

- A manifest that declares `real_corpus: true` is refused with error code
  `blocked_real_corpus` unless the gate records an independently confirmed
  license. Today it never does: every real corpus is blocked.
- A manifest whose license status is not `approved` is refused with error code
  `unapproved_license` (a synthetic fixture that stops declaring approval is
  also refused).
- A manifest whose checksums do not match, whose IDs are duplicated, or whose
  format/version is unsupported is refused with the corresponding `invalid`
  code.
- The pipeline maps these failures onto abstention reasons
  (`blocked_corpus`, `invalid_corpus`, `stale_corpus`) and never serves an
  answer from a gated-out corpus.

## The real-corpus blocker (exact, current)

No external corpus license has been independently confirmed. The Gob.ec corpus
is explicitly unauthorized: its "Creative Commons Attribution 4.0
International" label conflicts with the linked `by/3.0/ec` terms, and the
license discrepancy is unresolved. Real-corpus ingestion stays blocked until a
written license confirmation is obtained and recorded.

This text is committed as `REAL_CORPUS_BLOCKER` in
`src/evidence_grounded_rag/corpus/license_gate.py`, surfaced verbatim in
`corpus_status()` of the pipeline, and shown in the demo.

## Required per corpus (future real corpus)

| Property | Requirement |
| --- | --- |
| Provenance | Source URL(s) and retrieval date recorded |
| License | Terms documented verbatim with citation |
| Reuse rights | Explicit: indexing, transformation, redistribution |
| Attribution | Required credit recorded |
| Volume limits | Size or rate constraints documented |
| Versioning | Corpus snapshot is reproducible and versioned |

## Synthetic-fixture exception (current status)

Small hand-authored fixtures for offline development are allowed under this
repository's own MIT license, and they are clearly synthetic:

- Committed fixture: `data/fixtures/synthetic_manifest.json` (corpus id
  `synthetic-fixtures`, version `1.0.0`), 6 fictional documents / 11 chunks
  about the fictional "Aurora City". Regenerated deterministically by
  `scripts/materialize_fixtures.py`.
- The manifest states `provenance: project-generated-synthetic`,
  `synthetic: true`, `real_corpus: false`, license status `approved`
  (project-generated content covered by the repo's MIT license).
- Every result, report, and the demo label themselves **synthetic/offline**.
- Chunk locations use the `synthetic://` scheme; no external locators.

## Acceptance

- [x] License gate implemented, fail-closed, and tested (blocked real corpus,
      unapproved license, checksum mismatch, duplicate IDs, format errors).
- [x] Fixtures are clearly synthetic, small, and versioned; never represent a
      real corpus.
- [x] No Gob.ec or other unlicensed corpus is present in the repository.
- [ ] License terms documented verbatim for any candidate real corpus.
- [ ] Compatibility judgment recorded with reasoning.
