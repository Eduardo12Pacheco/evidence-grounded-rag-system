"""Dense embedding interface with a deterministic offline test embedding.

The bundled ``DeterministicEmbedding`` is a hashed bag-of-words embedding with
``semantic=False``: it is never described as semantic, downloads no weights,
and makes no network calls. A real embedding model can be plugged in behind the
same ``EmbeddingProvider`` protocol.

``SemanticLocalEmbedding`` is the optional semantic provider: it loads an
explicitly acquired local sentence-transformers model with
``local_files_only=True`` and is the only provider allowed to report
``semantic=True``. It is NEVER a silent fallback: if the local model is
unavailable, ``SemanticModelUnavailableError`` is raised instead of degrading
to hash embeddings.
"""

from __future__ import annotations

from evidence_grounded_rag.embeddings.base import (
    DenseSearchResult,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProvider,
    Vector,
    cosine_similarity,
)
from evidence_grounded_rag.embeddings.deterministic import DeterministicEmbedding
from evidence_grounded_rag.embeddings.index import DenseIndex
from evidence_grounded_rag.embeddings.manifest import (
    EmbeddingModelManifest,
    EmbeddingModelManifestError,
    ModelFileEntry,
    ModelLicense,
    audit_embedding_model_manifest,
    load_embedding_model_manifest,
    parse_embedding_model_manifest,
    safe_model_path,
    verify_embedding_model_files,
)
from evidence_grounded_rag.embeddings.semantic import (
    MODE as SEMANTIC_EMBEDDING_MODE,
)
from evidence_grounded_rag.embeddings.semantic import (
    SemanticLocalEmbedding,
    SemanticModelUnavailableError,
)
from evidence_grounded_rag.versions import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_MANIFEST_VERSION,
)

__all__ = [
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_MANIFEST_VERSION",
    "SEMANTIC_EMBEDDING_MODE",
    "DenseIndex",
    "DenseSearchResult",
    "DeterministicEmbedding",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingModelManifest",
    "EmbeddingModelManifestError",
    "EmbeddingProvider",
    "ModelFileEntry",
    "ModelLicense",
    "SemanticLocalEmbedding",
    "SemanticModelUnavailableError",
    "Vector",
    "audit_embedding_model_manifest",
    "cosine_similarity",
    "load_embedding_model_manifest",
    "parse_embedding_model_manifest",
    "safe_model_path",
    "verify_embedding_model_files",
]
