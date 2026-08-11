# Corpus & License Contract — evidence-grounded-rag-system

Status: **synthetic fixtures served; one real-local corpus approved for local
development only (Python 3.12.13 documentation). All other real corpora remain
BLOCKED.**

## Purpose

Define the conditions under which ANY document corpus may be used, record the
implemented gates that enforce them in code, and record the one approved
exception. This is a hard gate: **no real corpus is used until its license is
confirmed and recorded here.** The Gob.ec corpus is explicitly NOT authorized
and is NOT presented as such.

There are two separate, non-interchangeable boundaries:

1. **Synthetic-fixture gate** (`corpus/license_gate.py` + `corpus/loader.py`) —
   serves the committed fictional fixture corpus. Unchanged.
2. **Source-manifest audit** (`corpus/audit.py`, `corpus/real_local.py`,
   `corpus/privacy.py`) — a separate, explicit, opt-in boundary for the single
   approved real-local corpus. It never loosens the synthetic gate, never feeds
   the default demo, and never feeds synthetic evaluation.

## Decision — the one approved corpus (recorded 2026-08-11)

**Python 3.12.13 official documentation is approved for local development
only** (`status: approved-for-local-development`). The Python software and its
documentation are released under the **Python Software Foundation License
Version 2**, which grants reproduction, analysis, testing, display, derivative
works, distribution, and use while requiring that the license agreement and
copyright notice be retained and that distributed derivatives summarize the
changes made.

| Property | Value |
| --- | --- |
| Corpus id / version | `python-docs-3.12.13` / `1.0.0` |
| Publisher | Python Software Foundation |
| Official docs | https://docs.python.org/3.12/ |
| License | Python Software Foundation License Version 2 |
| License URL (docs) | https://docs.python.org/3.12/license.html |
| License URL (PSF) | https://www.python.org/psf/license/ |
| Source | https://github.com/python/cpython/tree/v3.12.13/Doc |
| Pinned revision | commit `3bb231a6a5dc02b95658877318bf61501a7209e9` (tag `v3.12.13`) |
| Access date | 2026-08-11 |
| Captured files | `Doc/tutorial/introduction.rst`, `Doc/library/functions.rst`, `Doc/reference/lexical_analysis.rst` |
| Audit status | `approved-for-local-development` |
| Redistribution | `not-redistributed` — raw capture is local-only, never committed |
| Privacy audit | `no-private-content-detected` |

### License evidence (PSF License Version 2, quoted from the official page)

> Section 2: "PSF hereby grants Licensee a nonexclusive, royalty-free,
> world-wide license to reproduce, analyze, test, perform and/or display
> publicly, prepare derivative works, distribute, and otherwise use Python
> alone or in any derivative version, provided, however, that PSF's License
> Agreement and PSF's notice of copyright ... are retained in Python alone or
> in any derivative version prepared by Licensee."
>
> Section 3: "In the event Licensee prepares a derivative work that is based
> on or incorporates Python or any part thereof, and wants to make the
> derivative work available to others ... Licensee hereby agrees to include in
> any such work a brief summary of the changes made to Python."

### Why redistribution is NOT claimed here

The PSF license permits redistribution with retained notices and a summary of
changes. This work unit deliberately takes the narrower
`approved-for-local-development` status: the captured raw files stay in the
gitignored `data/raw/` tree and are never committed, packaged, or published.
If a future work unit wants to redistribute, it must raise the status to
`approved-for-redistribution` with an explicit `redistribution_justification`.

## Candidates considered and rejected

| Candidate | Reason rejected |
| --- | --- |
| Rust Book | Mutable `main` / edition-scope complexity — the canonical source moves between editions and the "current" revision is not a single stable pinned snapshot. |
| Kubernetes website | License / provenance scope is not sufficiently stable for this work unit — the site aggregates many contributors and repos with per-page licensing; no single clean pinned revision + license pair. |
| Gob.ec corpus | Explicitly unauthorized (see below); not a candidate. |

## The real-corpus blocker (exact, current)

Every corpus EXCEPT the approved `python-docs-3.12.13` local capture remains
blocked. The Gob.ec corpus is explicitly unauthorized: its "Creative Commons
Attribution 4.0 International" label conflicts with the linked `by/3.0/ec`
terms, and the license discrepancy is unresolved.

This text is committed as `REAL_CORPUS_BLOCKER` in
`src/evidence_grounded_rag/corpus/license_gate.py`, surfaced verbatim in
`corpus_status()` of the pipeline, and shown in the demo.

## Mandatory gate 1 — synthetic fixtures (enforced in code, unchanged)

The gate lives in `src/evidence_grounded_rag/corpus/license_gate.py` and is
applied by `src/evidence_grounded_rag/corpus/loader.py` on every load. It
FAILS CLOSED:

- A manifest that declares `real_corpus: true` is refused with error code
  `blocked_real_corpus` unless the gate records an independently confirmed
  license. Today it never does — every real corpus is blocked by this gate.
- A manifest whose license status is not `approved` is refused with error code
  `unapproved_license` (a synthetic fixture that stops declaring approval is
  also refused).
- A manifest whose checksums do not match, whose IDs are duplicated, or whose
  format/version is unsupported is refused with the corresponding `invalid`
  code.
- The pipeline maps these failures onto abstention reasons
  (`blocked_corpus`, `invalid_corpus`, `stale_corpus`) and never serves an
  answer from a gated-out corpus.

## Mandatory gate 2 — source-manifest audit (new boundary)

The source-manifest boundary is a SEPARATE, explicit, opt-in local-development
path. It is exercised only by `scripts/acquire_corpus.py` and
`scripts/verify_corpus.py`; it is not wired into `load_corpus`, the pipeline,
the demo, or synthetic evaluation. It FAILS CLOSED:

- A source manifest must be schema-valid (all evidence fields present and
  well-typed, at least one expected file, `document_count > 0`) or it is
  refused (`malformed_source_manifest`,
  `unsupported_source_manifest_version`).
- Only `approved-for-local-development` or `approved-for-redistribution` are
  accepted; `blocked`, `under-review`, and unknown statuses are refused
  (`blocked_corpus_status`).
- Only the corpus committed in `APPROVED_CORPORA` (`python-docs-3.12.13`) is
  accepted; any other corpus id/version is refused (`unapproved_source`).
- The revision and the license name AND version must match the committed
  approval exactly (`revision_mismatch`, `license_mismatch`); a mutable
  revision is malformed.
- `redistribution_status` must be a known value and consistent with the corpus
  status: `approved-for-local-development` cannot claim redistribution, and
  `approved-for-redistribution` must use the matching redistribution status
  and carry an explicit justification (`invalid_redistribution_status`,
  `missing_redistribution_justification`).
- Expected files must exist with exact byte sizes and raw SHA-256 values
  (`source_file_missing`, `source_file_size_mismatch`,
  `source_file_checksum_mismatch`).
- The privacy scan result must match the manifest's `privacy_status`
  (`privacy_audit_failed`).

Stable codes are defined in `src/evidence_grounded_rag/corpus/errors.py`
(`SourceManifestError` hierarchy).

## Privacy audit (source-manifest boundary)

`src/evidence_grounded_rag/corpus/privacy.py` scans every captured file for
high-precision markers of private or sensitive content (emails, phones, SSNs,
credit-card and passport patterns, financial/medical/credential/secrets
keywords). Findings never carry the matched text. Any document with a finding
is **excluded** from derived data — exclusion is fail-safe. The audit result
must match the manifest's `privacy_status` or verification fails.

The approved files were scanned at acquisition (2026-08-11):
`no-private-content-detected`, zero excluded documents.

## Required per corpus (any future real corpus)

| Property | Requirement |
| --- | --- |
| Provenance | Source URL(s) and retrieval date recorded |
| License | Terms documented verbatim with citation |
| Reuse rights | Explicit: indexing, transformation, redistribution |
| Attribution | Required credit recorded |
| Volume limits | Size or rate constraints documented |
| Versioning | Corpus snapshot is reproducible and versioned |

A future corpus that cannot be pinned to a single exact revision with a single
stable license must be rejected under the same reasoning used for Rust Book and
Kubernetes above.

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
- [x] One real-local corpus approved (Python 3.12.13 docs), local-development
      only, with candidates/rejections and license evidence recorded here.
- [x] Source-manifest audit implemented, fail-closed, offline-tested, and kept
      separate from the synthetic gate and the serving pipeline.
- [x] Privacy audit implemented; excluded documents cannot enter derived data.
- [x] No Gob.ec or other unlicensed corpus is present in the repository.
- [ ] License terms documented verbatim for any candidate real corpus.
- [ ] Compatibility judgment recorded with reasoning.
