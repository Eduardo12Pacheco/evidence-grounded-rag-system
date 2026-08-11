"""Regenerate the committed synthetic fixture corpus manifest.

Run:  uv run python scripts/materialize_fixtures.py

Writes ``data/fixtures/synthetic_manifest.json`` and verifies it loads through
the real license gate and loader (including checksum validation). The output is
deterministic: identical content produces identical bytes.

All content is fictional, project-generated, and explicitly labeled synthetic.
It does not represent any real institution, regulation, or dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.corpus.normalize import sha256_hex

OUT_PATH = Path("data/fixtures/synthetic_manifest.json")

#: Exact blocker text for the real-corpus gate, mirrored from the license gate.
REAL_CORPUS_BLOCKER = (
    "No external corpus license has been independently confirmed. The Gob.ec "
    "corpus is explicitly unauthorized: its 'Creative Commons Attribution 4.0 "
    "International' label conflicts with the linked by/3.0/ec terms, and the "
    "license discrepancy is unresolved. Real-corpus ingestion stays blocked "
    "until a written license confirmation is obtained and recorded."
)

CAPTURED_AT = "2026-08-11"

_DOCUMENTS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "doc-aurora-park-permits",
        "Aurora City Park Permits Guide",
        "1.0.0",
        "Synthetic guide describing a fictional city's park permit rules.",
        [
            "The cost of a public event permit in Aurora Park is 120 dollars per day.",
            "A permit application must include a signed code of conduct and a contact phone number.",
            "Permit requests have a review time of up to 10 working days after submission.",
        ],
    ),
    (
        "doc-east-river-trail",
        "East River Trail Access Rules",
        "1.0.0",
        "Synthetic access rules for a fictional river trail.",
        [
            "Access to the East River trail is free for residents between sunrise and sunset.",
            "The East River trail is closed during flood season in April.",
        ],
    ),
    (
        "doc-central-library",
        "Central Library Opening Hours",
        "1.0.0",
        "Synthetic opening hours for a fictional public library.",
        [
            "The Central Library is open Monday through Friday from 8 in the morning to 8 in the evening.",
            "The Central Library is open Saturday from 9 in the morning to 5 in the evening.",
        ],
    ),
    (
        "doc-event-fee-2024",
        "Special Event Fee Schedule (2024 Edition)",
        "1.0.0",
        "Synthetic fee schedule edition 2024.",
        ["The special event fee is 120 dollars per day."],
    ),
    (
        "doc-event-fee-2025",
        "Special Event Fee Schedule (2025 Draft)",
        "1.0.0",
        "Synthetic fee schedule draft 2025; directly contradicts the 2024 edition.",
        ["The special event fee is 50 dollars per day."],
    ),
    (
        "doc-aurora-pools",
        "Aurora City Pools Opening Season",
        "1.0.0",
        "Synthetic opening season for two fictional city pools.",
        [
            "The North Pool is open from June to August.",
            "The South Pool is open from June to August.",
        ],
    ),
]


def build_manifest() -> dict[str, object]:
    documents = []
    for doc_id, title, version, attribution, texts in _DOCUMENTS:
        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "version": version,
                "captured_at": CAPTURED_AT,
                "source_name": "project-generated-synthetic",
                "source_url": None,
                "attribution": attribution,
                "license": "Project-generated synthetic content; covered by this "
                "repository's MIT license as original content.",
                "chunks": [{"seq": seq, "text": text} for seq, text in enumerate(texts)],
                "checksum_sha256": sha256_hex("\n".join(texts)),
            }
        )
    return {
        "manifest": {
            "manifest_id": "synthetic-fixtures-2026-08-11",
            "format_version": "1.0",
            "corpus_id": "synthetic-fixtures",
            "corpus_version": "1.0.0",
            "synthetic": True,
            "real_corpus": False,
            "provenance": "project-generated-synthetic",
            "provenance_detail": (
                "Hand-authored fictional content about the fictional Aurora City. "
                "No real institution, regulation, person, or dataset is represented."
            ),
            "license": {
                "status": "approved",
                "terms": "Project-generated synthetic content authored for this "
                "repository. Covered by the repository MIT license as original "
                "content. Not derived from any third-party source.",
                "redistribution": "Allowed within this repository as committed "
                "fixtures only.",
            },
            "real_corpus_gate": {
                "blocked": True,
                "blocker": REAL_CORPUS_BLOCKER,
            },
        },
        "documents": documents,
    }


def main() -> None:
    manifest = build_manifest()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    corpus = load_corpus(OUT_PATH)
    print(f"wrote {OUT_PATH} ({corpus.corpus_id} v{corpus.corpus_version})")
    print(f"  documents: {len(corpus.documents)} | chunks: {len(corpus.chunks())}")
    print("  verified: license gate passed, checksums valid, ids unique")


if __name__ == "__main__":
    main()
