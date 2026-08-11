"""Version identifiers for every replaceable component of the system.

All versions are committed here so evaluation reports and traces can record
exactly which component versions produced a result. Nothing here implies any
external license or model approval.
"""

from __future__ import annotations

#: Version of the ``evidence_grounded_rag`` package itself.
PACKAGE_VERSION = "0.1.0"

#: Version of the corpus manifest schema this implementation understands.
CORPUS_FORMAT_VERSION = "1.0"

#: Version of the deterministic text normalization used for checksums and
#: exact excerpt comparison.
NORMALIZATION_VERSION = "v1"

#: Version of the deterministic tokenizer used by sparse retrieval.
TOKENIZER_VERSION = "v1"

#: Version of the committed Okapi BM25 configuration.
BM25_VERSION = "okapi-bm25-v1"

#: Identifier of the deterministic, non-semantic embedding used for offline
#: development. Never label this model "semantic": it is a hashed bag-of-words
#: embedding with no learned meaning.
EMBEDDING_MODEL_ID = "deterministic-hashed-bow-v1"

#: Version of the committed reciprocal rank fusion configuration.
FUSION_VERSION = "rrf-v1"

#: Version of the deterministic offline extractive/template generator.
GENERATOR_VERSION = "extractive-template-v1"

#: Version of the committed synthetic query set.
QUERY_SET_VERSION = "synthetic-queries-v1"
