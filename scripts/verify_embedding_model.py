"""Verify the approved local semantic embedding model against its manifest.

Run:

    uv run python scripts/verify_embedding_model.py \
        --manifest data/manifests/embedding-model-v1.0.0.json \
        --cache-dir data/raw/models/bge-small-en-v1.5

An offline, local-only check (never downloads). It validates the model manifest
schema and version, audits the model id/revision/license/dimension/max-input/
status/redistribution policy, then verifies every expected file exists with
exact byte sizes and raw SHA-256 checksums. Exits non-zero with a stable error
code on any failure. If the exact pinned snapshot cannot be verified, semantic
enablement is stopped honestly — nothing silently degrades to hash embeddings.

Standard library only. No secrets are logged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evidence_grounded_rag.embeddings.manifest import (
    EmbeddingModelManifestError,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
    verify_embedding_model_files,
)

DEFAULT_MANIFEST = Path("data/manifests/embedding-model-v1.0.0.json")
DEFAULT_CACHE_DIR = Path("data/raw/models/bge-small-en-v1.5")


def _print_summary(manifest) -> None:
    print(f"model: {manifest.model_id} v{manifest.model_version}")
    print(f"status: {manifest.model_status}")
    print(f"publisher: {manifest.publisher}")
    print(f"official: {manifest.official_url}")
    print(f"revision: {manifest.revision} ({manifest.revision_tag})")
    print(
        f"license: {manifest.license.name} v{manifest.license.version} "
        f"({manifest.license.url})"
    )
    print(
        f"dimension: {manifest.dimension} | max_input_tokens: "
        f"{manifest.max_input_tokens} | device: {manifest.device}"
    )
    print(f"pooling: {manifest.pooling.get('method')} | l2_normalize: {manifest.normalization.get('l2_normalize')}")
    print(
        f"files: {len(manifest.expected_files)} | total bytes: "
        f"{sum(entry.size for entry in manifest.expected_files)}"
    )
    for entry in manifest.expected_files:
        print(
            f"  {entry.path}  {entry.size} bytes  "
            f"sha256={entry.sha256[:16]}..."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the approved local semantic embedding model against its manifest."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="path to the embedding-model manifest",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        type=Path,
        help="path to the model cache directory",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_embedding_model_manifest(args.manifest)
        audit_embedding_model_manifest(manifest)
        verify_embedding_model_files(manifest, args.cache_dir)
    except EmbeddingModelManifestError as exc:
        print(f"VERIFICATION FAILED [{exc.code}]: {exc}", file=sys.stderr)
        print(
            "Semantic enablement stopped: the exact pinned model snapshot is not "
            "verified. Re-run scripts/acquire_embedding_model.py to fetch it "
            "(network is allowed only in that explicit command).",
            file=sys.stderr,
        )
        return 1

    _print_summary(manifest)
    print("OK: manifest schema/version OK, audit passed, files byte-exact, sha256 matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
