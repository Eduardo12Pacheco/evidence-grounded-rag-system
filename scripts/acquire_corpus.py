from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from evidence_grounded_rag.corpus import (
    assert_audit_allowed,
    load_source_manifest,
    sha256_bytes,
    verify_source_files,
)
from evidence_grounded_rag.corpus.privacy import audit_privacy_texts
from evidence_grounded_rag.corpus.real_local import derive_source_doc_id

RAW_ROOT = Path("data/raw/python-docs-3.12.13")
MANIFEST_PATH = Path("data/manifests/corpus-source-v1.0.0.json")

CORPUS_ID = "python-docs-3.12.13"
CORPUS_VERSION = "1.0.0"
MANIFEST_VERSION = "1.0"
STATUS = "approved-for-local-development"
PUBLISHER = "Python Software Foundation"
OFFICIAL_DOCS_URL = "https://docs.python.org/3.12/"
SOURCE_URL = "https://github.com/python/cpython/tree/v3.12.13/Doc"
LICENSE_NAME = "Python Software Foundation License Version 2"
LICENSE_VERSION = "2"
LICENSE_URL_DOCS = "https://docs.python.org/3.12/license.html"
LICENSE_URL_PSF = "https://www.python.org/psf/license/"
REVISION = "3bb231a6a5dc02b95658877318bf61501a7209e9"
REVISION_TAG = "v3.12.13"
ACCESS_DATE = "2026-08-11"
ACQUISITION_METHOD = (
    "commit-pinned raw GitHub URLs "
    "(https://raw.githubusercontent.com/python/cpython/<commit>/<path>); "
    "no mirror, no mutable branch"
)
REDISTRIBUTION_STATUS = "not-redistributed"

LICENSE_EVIDENCE_EXCERPT = (
    'Section 2: "PSF hereby grants Licensee a nonexclusive, royalty-free, '
    "world-wide license to reproduce, analyze, test, perform and/or display "
    "publicly, prepare derivative works, distribute, and otherwise use Python "
    "alone or in any derivative version, provided, however, that PSF's "
    "License Agreement and PSF's notice of copyright ... are retained in "
    'Python alone or in any derivative version prepared by Licensee." '
    'Section 3: "In the event Licensee prepares a derivative work that is '
    "based on or incorporates Python or any part thereof, and wants to make "
    "the derivative work available to others ... Licensee hereby agrees to "
    'include in any such work a brief summary of the changes made to Python."'
)

FILES: tuple[str, ...] = (
    "Doc/tutorial/introduction.rst",
    "Doc/library/functions.rst",
    "Doc/reference/lexical_analysis.rst",
)


def _raw_url(rel_path: str) -> str:
    return f"https://raw.githubusercontent.com/python/cpython/{REVISION}/{rel_path}"


def _acquire_file(rel_path: str) -> dict[str, object]:
    import urllib.request

    url = _raw_url(rel_path)
    print(f"  fetching {rel_path}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    out = RAW_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "path": rel_path,
        "size": len(data),
        "sha256": sha256_bytes(data),
    }


def build_manifest(
    entries: list[dict[str, object]], timestamp: str
) -> dict[str, object]:
    texts = {
        entry["path"]: (RAW_ROOT / entry["path"]).read_text(encoding="utf-8")
        for entry in entries
    }
    privacy_docs = [
        (derive_source_doc_id(entry["path"]), entry["path"], texts[entry["path"]])
        for entry in entries
    ]
    privacy = audit_privacy_texts(privacy_docs)
    privacy_status = (
        "no-private-content-detected" if privacy.clean else "private-content-excluded"
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "corpus_id": CORPUS_ID,
        "corpus_version": CORPUS_VERSION,
        "status": STATUS,
        "publisher": PUBLISHER,
        "official_docs_url": OFFICIAL_DOCS_URL,
        "source_url": SOURCE_URL,
        "license": {
            "name": LICENSE_NAME,
            "version": LICENSE_VERSION,
            "url": LICENSE_URL_DOCS,
            "url_psf": LICENSE_URL_PSF,
            "evidence_excerpt": LICENSE_EVIDENCE_EXCERPT,
        },
        "revision": REVISION,
        "revision_tag": REVISION_TAG,
        "access_date": ACCESS_DATE,
        "acquisition_method": ACQUISITION_METHOD,
        "expected_files": entries,
        "document_count": len(entries),
        "redistribution_status": REDISTRIBUTION_STATUS,
        "privacy_status": privacy_status,
        "verification_timestamp": timestamp,
    }


def main() -> None:
    print(f"Acquiring {CORPUS_ID} v{CORPUS_VERSION} at {REVISION}")
    entries = [_acquire_file(rel) for rel in FILES]

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(entries, timestamp)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    loaded = load_source_manifest(MANIFEST_PATH)
    assert_audit_allowed(loaded)
    verify_source_files(loaded, RAW_ROOT)

    print(f"\nwrote {MANIFEST_PATH}")
    print(f"  raw dir: {RAW_ROOT} (gitignored, never committed)")
    print("  verified: audit passed, files byte-exact, sha256 matches")
    for entry in manifest["expected_files"]:
        print(f"    {entry['path']}  {entry['size']} bytes  {entry['sha256'][:16]}...")


if __name__ == "__main__":
    main()
