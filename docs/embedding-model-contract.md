# Embedding Model Contract — BAAI/bge-small-en-v1.5 (local, opt-in)

Status: **`approved-for-local-development`** — pinned revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, acquired locally, never committed, never redistributed.

## Quick path

```bash
# 1. Install the optional dependency stack (torch/transformers are a large footprint)
uv sync --dev --extra demo --extra embeddings

# 2. Acquire the pinned model into the gitignored cache (the ONLY network step)
uv run python scripts/acquire_embedding_model.py --manifest data/manifests/embedding-model-v1.0.0.json

Normal acquisition treats the **committed manifest as the trust anchor**: it
loads and audits the committed manifest, downloads exactly the manifest's
pinned files into a temporary staging directory, verifies them byte-exact
against the manifest's declared sizes/SHA-256 **before** touching the cache,
then atomically renames the verified staging directory over the cache. A
failed or partial run can never leave a seemingly-valid cache, and the
committed manifest is **never rewritten** — a compromised download cannot
self-certify by overwriting the committed checksums. First-time bootstrap (no
manifest yet) is an explicit, separate action:

```bash
uv run python scripts/acquire_embedding_model.py --bootstrap \
    --manifest data/manifests/embedding-model-v1.0.0.json
```

Bootstrap is the only mode that writes a new manifest, it requires the manifest
to be absent, and its checksums are self-certified (no trust anchor existed):
the command prints a loud warning that the new manifest must be reviewed and
committed before it is treated as approved.

# 3. Verify the local snapshot byte-exact (offline, no downloads)
uv run python scripts/verify_embedding_model.py --manifest data/manifests/embedding-model-v1.0.0.json

# 4. Run the separate semantic-materializer evaluation
uv run python -m evidence_grounded_rag.evaluation.semantic_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --model-manifest data/manifests/embedding-model-v1.0.0.json \
    --cache-dir data/raw/models/bge-small-en-v1.5 \
    --output data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json
```

The semantic model is **opt-in and explicit**. The default embedding remains the
deterministic, non-semantic hash embedding (`semantic = False`, no weights, no
network). There is **no silent fallback**: if the local model or its verified
snapshot is unavailable, semantic retrieval raises
`SemanticModelUnavailableError` and is refused.

## Candidate audit (recorded 2026-08-11)

All six candidates are official Hugging Face model pages. Selection is
reproducible from official evidence; rejection reasons are recorded so the
decision does not have to be re-derived.

| Candidate (official page) | Rejected? | Reason |
| --- | --- | --- |
| **BAAI/bge-small-en-v1.5** | **SELECTED** | English retrieval-oriented, MIT licensed, 384-dim, 512-token input, no mandatory query prefix for v1.5, compact and CPU-feasible, public and non-gated. |
| sentence-transformers/all-MiniLM-L6-v2 | Rejected | 256-token truncation conflicts with the corpus's 2000-char chunks (would silently cut evidence). |
| sentence-transformers/all-mpnet-base-v2 | Rejected | Larger footprint (~420 MB) than needed for a 70-chunk bounded evaluation. |
| Snowflake/snowflake-arctic-embed-xs | Rejected | Mandatory query prefix (`query:`) adds a brittle preprocessing contract. |
| intfloat/multilingual-e5-small | Rejected | Mandatory prefix conventions (`query:`/`passage:`) and heavier multilingual weights; the current corpus is English. |
| nomic-ai/nomic-embed-text-v1 | Rejected | Custom `trust_remote_code` remote code path — more fragile and harder to pin than a plain BERT WordPiece model. |

No candidate is claimed to be universally best, and no model-card benchmark
number is ever reported as a project metric.

## Selected model — official evidence

| Field | Value |
| --- | --- |
| Model id | `BAAI/bge-small-en-v1.5` |
| Publisher | Beijing Academy of Artificial Intelligence (BAAI) |
| Official model page | https://huggingface.co/BAAI/bge-small-en-v1.5 |
| Official model card (raw) | https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/README.md |
| Official API / config | https://huggingface.co/api/models/BAAI/bge-small-en-v1.5/revision/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| Pinned revision (commit SHA) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Revision tag | none (pinned by commit SHA; never a mutable branch) |
| License | MIT License (`license: mit` in the card's YAML; "FlagEmbedding is licensed under the MIT License. The released models can be used for commercial purposes free of charge.") |
| License text | https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE |
| Acquisition date | 2026-08-11 |
| Model status | `approved-for-local-development` |

## Model facts (from the pinned config files)

| Fact | Value | Source |
| --- | --- | --- |
| Architecture | BERT, `BertModel`, 12 layers, 12 heads | `config.json` |
| Embedding dimension | **384** (`hidden_size` = 384) | `config.json`, `1_Pooling/config.json` |
| Max input | **512** tokens (`max_seq_length` = 512, `max_position_embeddings` = 512) | `sentence_bert_config.json`, `config.json` |
| Pooling | **CLS** — `1_Pooling/config.json` sets `pooling_mode_cls_token: true`; the official README instructs "Perform pooling. In this case, cls pooling. `sentence_embeddings = model_output[0][:, 0]`" | `1_Pooling/config.json`, official README |
| Normalization | **L2** — official README: "normalize embeddings `torch.nn.functional.normalize(..., p=2, dim=1)`"; the sentence-transformers module chain adds a `2_Normalize` module | official README, `modules.json` |
| Tokenizer | WordPiece, `bert-base-uncased` (vocab 30522, lowercasing), `special_tokens_map.json` `[CLS] [MASK] [PAD] [SEP] [UNK]` | `tokenizer.json`, `vocab.txt`, `config.json` |
| Weights format | single `model.safetensors` (float32) | repo file tree |
| Device | CPU (single thread for deterministic inference) | our provider contract |

Pooling note: the official BGE README shows CLS pooling for direct
`transformers` usage. The sentence-transformers wrapper manages pooling and
normalization through its own `modules.json` chain (Transformer →
`1_Pooling` CLS → `2_Normalize`); the wrapper's output is the 384-dim L2
normalized vector.

## Permitted use and redistribution policy

- The MIT license permits local reproduction, analysis, testing, display, and
  derivative use; retained for local development of this repository.
- **Redistribution policy is `not-redistributed`** (see
  `data/manifests/embedding-model-v1.0.0.json`): the weights are downloaded
  once into the gitignored `data/raw/models/bge-small-en-v1.5/` cache and are
  **never committed, packaged, or published** by this repository.
- The committed artifact is the small manifest
  (`data/manifests/embedding-model-v1.0.0.json`) containing metadata and the
  **actual SHA-256 checksums and byte sizes** of every acquired file — never the
  weights.
- Third-party artifacts retain their own terms (MIT for this model); this
  repo's MIT license covers only original code and the synthetic fixtures.

## Checksum policy

- Every acquired file is recorded in the manifest with its byte-exact size and
  raw SHA-256 (`sha256_bytes` over the exact downloaded bytes).
- `model.safetensors` was cross-checked against the Hub's LFS blob hash
  `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`.
- Verification (`scripts/verify_embedding_model.py`) re-checks schema/version,
  model id/version, publisher, URLs, revision, license, dimension, max input,
  device, architecture/tokenizer/pooling/normalization metadata, status,
  redistribution, then file existence + exact size + exact SHA-256, and
  rejects any extra file in the cache beyond the manifest's `expected_files`.
  Any mismatch stops semantic enablement.

## Inference behavior

- **Local-only**: `SentenceTransformer(local_path, device="cpu",
  local_files_only=True)`. No network, no Hub access, no fallback revision.
- **Deterministic CPU settings**: single thread (`torch.set_num_threads(1)`),
  fixed `batch_size=16`, stable input order, float32 inference, L2 via the
  model's `2_Normalize` module plus `encode(normalize_embeddings=True)`.
- **Dimension check**: the loaded model's embedding dimension is verified equal
  to the committed 384 before any scoring; mismatch raises
  `SemanticModelUnavailableError`.
- **Empty input**: `embed([])` returns `[]`.
- **Long input**: inputs are truncated by the tokenizer at 512 tokens, exactly
  as the official model behaves; documented, not silent.

## Expected memory (approximate, qualitative)

Qualitative only — no invented measurements. BGE-small is a 12-layer, 384-wide
BERT (~130 MB of float32 weights). On CPU the model plus tokenizer plus a
70-chunk index is expected to fit comfortably within a few hundred MB of RAM;
the 70-chunk dense index itself is negligible compared to the loaded model. The
evaluation records the process peak RSS it actually observed (see the report's
`runtime_observations`); no per-mode memory figure is claimed.

## Dependency and install footprint

- `sentence-transformers>=3.0` behind the optional extra
  `[project.optional-dependencies].embeddings` (MIT/Apache-licensed ecosystem).
- Installing it pulls the PyTorch/Transformers stack (torch, transformers,
  tokenizers, safetensors, scikit-learn, numpy), which is a large disk/install
  footprint. That is why it is optional and never required by the core
  pipeline, the demo default, or the test suite.
- Tests never download weights; real inference is exercised only when the
  optional dependency and the verified local snapshot are present.

## Unavailable behavior (explicit, no fallback)

- **The provider enforces the boundary itself.** `SemanticLocalEmbedding`
  requires an audited manifest: construction audits the manifest against the
  approval constants and byte-verifies the cache (existence, exact size, exact
  SHA-256, and **no extra files beyond the manifest's `expected_files`**)
  before anything is loaded. Caller-supplied metadata is never trusted and no
  hardcoded critical-file list is used.
- Model cache missing or incomplete / any file missing, tampered, or extra →
  `SemanticModelUnavailableError` listing the exact blocker.
- `sentence-transformers` not installed → `SemanticModelUnavailableError` with
  the install command.
- `local_files_only=False` or a non-approved device → refused
  (`SemanticModelUnavailableError`); `local_files_only=True` and CPU are
  invariants.
- Loaded model dimension unknown or ≠ 384 → `SemanticModelUnavailableError`
  (the dimension is never silently skipped).
- Manifest schema/audit-policy failures (missing manifest, unapproved model,
  revision/license/metadata mismatch, blocked status) → `EmbeddingModelManifestError`
  with a stable code; these propagate unwrapped.
- In every case the semantic mode is refused and **never** degrades to the
  deterministic hash embedding under a semantic label.

## Limitations

- This is a bounded evaluation contract for a 3-file / 70-chunk corpus; it does
  not validate Python documentation in general.
- Retrieval quality does not imply answer faithfulness; the semantic
  materializer reports retrieval metrics only (no generation, no LLM).
- No model is claimed to be universally best; model-card benchmarks are not
  project metrics.
