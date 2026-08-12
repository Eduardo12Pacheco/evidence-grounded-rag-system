"""Acquire the approved local semantic embedding model (BAAI/bge-small-en-v1.5).

Run (normal, trust-anchor mode):

    uv run python scripts/acquire_embedding_model.py \
        --manifest data/manifests/embedding-model-v1.0.0.json \
        --cache-dir data/raw/models/bge-small-en-v1.5

Normal mode REQUIRES the committed manifest and treats it as the trust anchor:

- The committed manifest (``data/manifests/embedding-model-v1.0.0.json``) is
  loaded, schema-validated, and audited against the approval policy. A missing
  manifest fails clearly and is NOT bootstrapped implicitly.
- Exactly the manifest's pinned files are downloaded (model id + revision read
  from the audited manifest) into a temporary STAGING directory on the same
  filesystem — never directly into the cache.
- The staged files are verified byte-exact against the manifest's declared
  sizes and SHA-256 checksums BEFORE the cache is touched.
- Only then is the staging directory atomically renamed over the cache. A
  partial or failed run can never leave a seemingly-valid cache: the old cache
  is only swapped once the full snapshot is verified.
- The approved manifest is NEVER rewritten in normal mode — a compromised
  download cannot self-certify by overwriting the committed checksums.

First-time bootstrap (explicit, separate action) for a missing manifest:

    uv run python scripts/acquire_embedding_model.py --bootstrap \
        --manifest data/manifests/embedding-model-v1.0.0.json \
        --cache-dir data/raw/models/bge-small-en-v1.5

``--bootstrap`` is the ONLY mode that writes a new manifest, and it requires
the manifest to be absent. The checksums it records are self-certified (there
is no trust anchor yet): the command prints a loud warning that the new
manifest must be reviewed and committed before it is treated as approved.

Standard library only (``urllib``, ``hashlib``, ``json``). The raw cache stays
gitignored (``data/raw/``); no secrets are ever logged or written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from evidence_grounded_rag.corpus.audit import (
    REDISTRIBUTION_LOCAL_ONLY,
    STATUS_APPROVED_LOCAL,
)
from evidence_grounded_rag.corpus.normalize import sha256_bytes
from evidence_grounded_rag.embeddings.manifest import (
    MODEL_ACQUISITION_METHOD,
    MODEL_ARCHITECTURE,
    MODEL_CACHE_POLICY,
    MODEL_CARD_URL,
    MODEL_DIMENSION,
    MODEL_ID,
    MODEL_LICENSE_EVIDENCE_EXCERPT,
    MODEL_LICENSE_NAME,
    MODEL_LICENSE_URL,
    MODEL_LICENSE_VERSION,
    MODEL_MAX_INPUT_TOKENS,
    MODEL_NORMALIZATION,
    MODEL_OFFICIAL_URL,
    MODEL_POOLING,
    MODEL_PUBLISHER,
    MODEL_REVISION,
    MODEL_REVISION_TAG,
    MODEL_TOKENIZER,
    MODEL_VERSION,
    REQUIRED_FILES,
    SUPPORTED_EMBEDDING_MODEL_MANIFEST_VERSION,
    EmbeddingModelManifestError,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
    verify_embedding_model_files,
)

#: Committed model facts recorded on the manifest.
REDISTRIBUTION_STATUS = REDISTRIBUTION_LOCAL_ONLY
MODEL_STATUS = STATUS_APPROVED_LOCAL

BOOTSTRAP_WARNING = (
    "BOOTSTRAP: the model manifest did not exist, so a NEW manifest was written "
    "with checksums computed from this download. This is self-certified: the "
    "download could have been tampered with and no committed trust anchor "
    "existed to catch it. REVIEW the manifest before committing it, then treat "
    "the committed manifest as the trust anchor for all normal acquisition and "
    "verification from now on."
)


def _download_file_bytes(model_id: str, revision: str, rel_path: str) -> bytes:
    """Download one pinned model file. Returns the raw bytes (network only here)."""
    import urllib.request

    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{rel_path}"
    print(f"  fetching {rel_path}")
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _make_staging_dir(cache_dir: Path) -> Path:
    """Create a temporary staging directory beside the cache (same filesystem)."""
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=cache_dir.parent))


def _write_staged_file(staging_dir: Path, rel_path: str, data: bytes) -> None:
    out = staging_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def _atomically_replace_cache(cache_dir: Path, staging_dir: Path) -> None:
    """Rename *staging_dir* over *cache_dir*; the swap is all-or-nothing.

    The staging dir is on the same filesystem, so the final rename is atomic.
    If an old cache exists it is first moved aside to a backup; if the new
    rename fails the old cache is restored. A partial run can never leave a
    seemingly-valid cache in place.
    """
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        backup = cache_dir.with_name(
            f"{cache_dir.name}.old-{int(time.time() * 1000)}"
        )
        cache_dir.rename(backup)
        try:
            staging_dir.rename(cache_dir)
        except Exception:
            backup.rename(cache_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        staging_dir.rename(cache_dir)


def build_manifest(entries: list[dict[str, object]], timestamp: str) -> dict[str, object]:
    """Assemble the full model manifest from committed metadata + capture facts."""
    return {
        "manifest_version": SUPPORTED_EMBEDDING_MODEL_MANIFEST_VERSION,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "publisher": MODEL_PUBLISHER,
        "official_url": MODEL_OFFICIAL_URL,
        "model_card_url": MODEL_CARD_URL,
        "revision": MODEL_REVISION,
        "revision_tag": MODEL_REVISION_TAG,
        "license": {
            "name": MODEL_LICENSE_NAME,
            "version": MODEL_LICENSE_VERSION,
            "url": MODEL_LICENSE_URL,
            "evidence_excerpt": MODEL_LICENSE_EVIDENCE_EXCERPT,
        },
        "architecture": MODEL_ARCHITECTURE,
        "dimension": MODEL_DIMENSION,
        "max_input_tokens": MODEL_MAX_INPUT_TOKENS,
        "tokenizer": MODEL_TOKENIZER,
        "pooling": MODEL_POOLING,
        "normalization": MODEL_NORMALIZATION,
        "device": "cpu",
        "model_status": MODEL_STATUS,
        "acquisition_method": MODEL_ACQUISITION_METHOD,
        "cache_policy": MODEL_CACHE_POLICY,
        "redistribution_status": REDISTRIBUTION_STATUS,
        "expected_files": entries,
        "verification_timestamp": timestamp,
    }


def _acquire_from_trusted_manifest(args) -> int:
    """Normal mode: download pinned files and verify against the committed manifest.

    Never rewrites the manifest. Returns the process exit code.
    """
    try:
        manifest = load_embedding_model_manifest(args.manifest)
    except EmbeddingModelManifestError as exc:
        print(
            f"ACQUISITION REFUSED: committed model manifest unavailable — {exc}",
            file=sys.stderr,
        )
        print(
            "Normal acquisition uses the committed manifest as its trust anchor. "
            "If this is a first-time bootstrap, run this command explicitly with "
            "--bootstrap (a separate, explicit action).",
            file=sys.stderr,
        )
        return 2

    try:
        audit_embedding_model_manifest(manifest)
    except EmbeddingModelManifestError as exc:
        print(
            f"ACQUISITION REFUSED: committed model manifest failed audit — {exc}",
            file=sys.stderr,
        )
        return 2

    staging = _make_staging_dir(args.cache_dir)
    try:
        for entry in manifest.expected_files:
            data = _download_file_bytes(manifest.model_id, manifest.revision, entry.path)
            _write_staged_file(staging, entry.path, data)
        verify_embedding_model_files(manifest, staging)
    except Exception as exc:  # noqa: BLE001 - any failure must leave the cache untouched
        shutil.rmtree(staging, ignore_errors=True)
        print(
            f"ACQUISITION FAILED: {exc} — the cache was left untouched "
            "(downloads are staged and verified before any swap).",
            file=sys.stderr,
        )
        return 1

    _atomically_replace_cache(args.cache_dir, staging)

    print(f"\nAcquired {manifest.model_id} v{manifest.model_version} at {manifest.revision}")
    print(f"  cache dir: {args.cache_dir} (gitignored, never committed)")
    print("  staged, verified byte-exact against the committed manifest, atomically swapped")
    print(f"  manifest: {args.manifest} (NOT rewritten — committed manifest is the trust anchor)")
    total = sum(entry.size for entry in manifest.expected_files)
    print(f"  files: {len(manifest.expected_files)}  total bytes: {total}")
    for entry in manifest.expected_files:
        print(f"    {entry.path}  {entry.size} bytes  {entry.sha256[:16]}...")
    return 0


def _bootstrap(args) -> int:
    """First-time bootstrap: only allowed when no manifest exists. Self-certifying."""
    if args.manifest.exists():
        print(
            "BOOTSTRAP REFUSED: a model manifest already exists at "
            f"{args.manifest}. Bootstrap is only for first-time acquisition "
            "without a trust anchor; use the normal mode instead.",
            file=sys.stderr,
        )
        return 2

    print(f"BOOTSTRAP: no committed manifest; acquiring {MODEL_ID} v{MODEL_VERSION} "
          f"at {MODEL_REVISION} self-certified.")
    staging = _make_staging_dir(args.cache_dir)
    entries: list[dict[str, object]] = []
    try:
        for rel_path in REQUIRED_FILES:
            data = _download_file_bytes(MODEL_ID, MODEL_REVISION, rel_path)
            _write_staged_file(staging, rel_path, data)
            entries.append(
                {"path": rel_path, "size": len(data), "sha256": sha256_bytes(data)}
            )
    except Exception as exc:  # noqa: BLE001 - any failure must leave the cache untouched
        shutil.rmtree(staging, ignore_errors=True)
        print(
            f"BOOTSTRAP FAILED: {exc} — no manifest was written and the cache "
            "was left untouched.",
            file=sys.stderr,
        )
        return 1

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_doc = build_manifest(entries, timestamp)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")

    try:
        loaded = load_embedding_model_manifest(args.manifest)
        audit_embedding_model_manifest(loaded)
    except EmbeddingModelManifestError as exc:
        print(f"BOOTSTRAP FAILED: newly written manifest invalid — {exc}", file=sys.stderr)
        return 1

    _atomically_replace_cache(args.cache_dir, staging)
    verify_embedding_model_files(loaded, args.cache_dir)

    print(f"\nwrote {args.manifest}")
    print(BOOTSTRAP_WARNING)
    print(f"  cache dir: {args.cache_dir} (gitignored, never committed)")
    total = sum(entry["size"] for entry in entries)
    print(f"  files: {len(entries)}  total bytes: {total}")
    for entry in manifest_doc["expected_files"]:
        print(f"    {entry['path']}  {entry['size']} bytes  {entry['sha256'][:16]}...")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the approved local semantic embedding model. Normal mode "
            "uses the committed manifest as the trust anchor and never rewrites "
            "it; --bootstrap is the explicit, separate first-time action."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/embedding-model-v1.0.0.json",
        type=Path,
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/models/bge-small-en-v1.5",
        type=Path,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="explicit, separate first-time action: write a NEW self-certified "
        "manifest when none exists (never used for normal acquisition)",
    )
    args = parser.parse_args(argv)

    if args.bootstrap:
        return _bootstrap(args)
    return _acquire_from_trusted_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
