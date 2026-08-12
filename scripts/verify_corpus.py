from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evidence_grounded_rag.corpus import (
    CorpusError,
    SourceManifestError,
    load_source_manifest,
)
from evidence_grounded_rag.corpus.real_local import build_real_local_corpus

DEFAULT_RAW_DIR = Path("data/raw/python-docs-3.12.13")
DEFAULT_MANIFEST = Path("data/manifests/corpus-source-v1.0.0.json")


def _print_summary(corpus) -> None:
    print(f"corpus: {corpus.corpus_id} v{corpus.corpus_version}")
    print(f"status: {corpus.status}")
    print(f"revision: {corpus.manifest.revision} ({corpus.manifest.revision_tag})")
    print(f"source: {corpus.manifest.source_url}")
    print(f"official docs: {corpus.manifest.official_docs_url}")
    print(
        f"license: {corpus.manifest.license.name} "
        f"v{corpus.manifest.license.version} ({corpus.manifest.license.url})"
    )
    print(f"privacy: {corpus.manifest.privacy_status}")
    if corpus.privacy.findings:
        print(f"  findings: {len(corpus.privacy.findings)}")
        for finding in corpus.privacy.findings:
            print(
                f"    {finding.source_path}:{finding.line} "
                f"(category={finding.category})"
            )
    print(f"excluded documents: {sorted(corpus.excluded_document_ids)}")
    print(f"documents: {len(corpus.documents)} | chunks: {len(corpus.chunks())}")
    for document in corpus.documents:
        print(
            f"  {document.doc_id}  raw={document.raw_size}B "
            f"raw_sha256={document.raw_sha256[:16]}... "
            f"normalized_sha256={document.normalized_sha256[:16]}... "
            f"chunks={len(document.chunks)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the approved local corpus against its source manifest."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="path to the source manifest (default: data/manifests/corpus-source-v1.0.0.json)",
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        type=Path,
        help="path to the raw capture directory (default: data/raw/python-docs-3.12.13)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_source_manifest(args.manifest)
        corpus = build_real_local_corpus(manifest, args.raw_dir)
    except SourceManifestError as exc:
        print(f"VERIFICATION FAILED [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except CorpusError as exc:
        print(f"VERIFICATION FAILED [{exc.code}]: {exc}", file=sys.stderr)
        return 1

    _print_summary(corpus)
    print("OK: audit passed, files byte-exact, privacy audited, derivation clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
