# Demo Script — evidence-grounded-rag-system

Status: **implemented — offline Streamlit demo over committed synthetic fixtures.**

## What the demo shows

A user asks a question and sees, in one screen:

1. The **corpus/license panel**: corpus id/version, provenance
   (`project-generated-synthetic`), license status, and the exact
   real-corpus blocker (Gob.ec unauthorized; license discrepancy unresolved).
2. A **SYNTHETIC / OFFLINE** banner — all content is fictional fixture data,
   no network calls, no credentials.
3. **Modes**: hybrid (default), sparse, dense; plus a `top_k` slider.
4. **Sparse / dense / hybrid results**: ranked chunks with rank, score, chunk
   id, and snippet.
5. **Passage / source metadata** for the top evidence chunk (document id,
   title, version, captured date, source, license note, SHA-256 checksum,
   `synthetic://` location).
6. **Structured answer** with numbered claims and their citations.
7. **Citation validation**: valid, or the exact failing issue codes (answer is
   refused, never shown as verified).
8. **Abstention reason** when the system refuses (e.g. conflicting evidence),
   with the stable reason code and detail.
9. **Versions & configuration** and the full **trace** (sparse/dense/fused
   candidates, versions, config) as JSON.

Failures are visible: gate failures, validation failures, and empty queries
render as explicit error/warning panels, never as silent empty screens.

## Constraint

The demo runs ONLY on the committed synthetic fixture corpus
(`data/fixtures/synthetic_manifest.json`). It performs no live fetches, uses
no credentials, and never touches an unlicensed corpus. The real-corpus gate
is shown as BLOCKED.

## Real-local corpus is NOT part of the demo

The approved real-local corpus (Python 3.12.13 documentation,
`approved-for-local-development`) is intentionally absent from the demo. It is
an explicit, opt-in, local-development boundary exercised only by the offline
scripts — the demo, the default pipeline, and synthetic evaluation are
unchanged and remain synthetic-only:

```bash
uv run python scripts/acquire_corpus.py \
    # writes data/raw/python-docs-3.12.13/ (gitignored) + the source manifest
uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json
    # offline verification of manifest, revision, files, sizes, SHA-256, privacy
```

See `docs/corpus-license-contract.md` for the license decision and why Gob.ec
is not used, and `README.md` for the synthetic / real-local / blocked modes.

## Run book

```bash
uv sync --dev --extra demo
uv run --extra demo streamlit run scripts/demo_retrieval.py
```

Open the printed local URL. The app builds the pipeline in memory from the
committed fixture manifest (gate + checksum validated on load).

## Walkthrough (60–90 seconds)

1. **Open** — the SYNTHETIC/OFFLINE banner and the corpus/license panel are
   immediately visible, including the real-corpus blocker.
2. **Supported query** — the default query `What must a permit application
   include?` answers with a cited claim:
   - Answer: "A permit application must include a signed code of conduct and a
     contact phone number. [1]"
   - Claim cites `doc-aurora-park-permits::1` at its `synthetic://` location.
   - Citation validation shows VALID.
3. **Conflicting evidence** — ask `How much is the special event fee per day?`:
   the corpus contains two contradictory fee schedules, so the system shows
   `CONFLICTING EVIDENCE: conflicting_evidence` and explains the contradiction.
4. **Out-of-corpus** — ask `What is the fine for littering in Millbrook?`:
   the system abstains with `unsupported_query`.
5. **Modes** — switch to sparse or dense and re-run to compare rankings; the
   answer is grounded on the selected mode while both systems' candidates stay
   visible.
6. **Close** — the trace and versions/config panels let any claim be traced
   back to a specific chunk, rank, score, and committed version.

## What the demo must NOT claim

- No real accuracy numbers: the committed metrics describe synthetic fixtures
  only and are labeled as such.
- No Gob.ec corpus use, ever, until the corpus/license contract is satisfied.
- No hallucination-free, trustworthy, or production-ready wording.
