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

#: Version of the source-manifest schema used by the corpus audit boundary
#: (the bounded, explicitly approved real-local corpus). This is a separate
#: schema from ``CORPUS_FORMAT_VERSION``, which describes synthetic fixtures.
SOURCE_MANIFEST_VERSION = "1.0"

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

#: Identifier of the optional local semantic embedding model (BAAI bge-small-en
#: v1.5). The model is NOT bundled and is never downloaded implicitly: it is
#: acquired explicitly into a gitignored cache and verified against the
#: committed model manifest before any semantic use.
SEMANTIC_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

#: Version of the embedding-model manifest schema
#: (data/manifests/embedding-model-v1.0.0.json).
EMBEDDING_MODEL_MANIFEST_VERSION = "1.0"

#: Version of the committed reciprocal rank fusion configuration.
FUSION_VERSION = "rrf-v1"

#: Version of the deterministic offline extractive/template generator.
GENERATOR_VERSION = "extractive-template-v1"

#: Version of the provider-neutral live adapter. No provider/model/credentials
#: are configured; live generation requires an explicitly injected transport.
LIVE_GENERATOR_VERSION = "injected-live-provider-v1"

#: Version of the committed grounded-answer prompt. The prompt text is frozen
#: under ``docs/generation-contract.md`` (version ``grounded-answer-prompt-v1.0.0``).
GENERATION_PROMPT_VERSION = "grounded-answer-prompt-v1.0.0"

#: Version of the structured answer JSON schema produced by live providers and
#: parsed by the strict parser (see ``generation/structured.py`` and
#: ``docs/generation-contract.md``).
STRUCTURED_ANSWER_SCHEMA_VERSION = "structured-answer-v1"

#: Version of the deterministic claim-support containment rule
#: (see ``generation/support.py`` and ``docs/generation-contract.md``).
CLAIM_SUPPORT_VERSION = "normalized-exact-containment-v1"

#: Version of the committed answer-evaluation fixture that references the
#: unchanged real query IDs and gold chunks without copying any passage.
ANSWER_EVALUATION_FIXTURE_VERSION = "1.0.0"

#: Version of the separate real-local generation evaluation report.
GENERATION_EVALUATION_VERSION = "1.0.0"

#: Version of the committed synthetic query set.
QUERY_SET_VERSION = "synthetic-queries-v1"
