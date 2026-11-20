"""Opt-in local semantic embeddings with no network or hash fallback."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evidence_grounded_rag.embeddings.base import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    Vector,
)
from evidence_grounded_rag.embeddings.manifest import (
    MODEL_DEVICE,
    EmbeddingModelManifest,
    EmbeddingModelManifestError,
    audit_embedding_model_manifest,
    verify_embedding_model_files,
)

MODE = "semantic-local"

DEFAULT_THREAD_COUNT = 1

DEFAULT_BATCH_SIZE = 16


class SemanticModelUnavailableError(RuntimeError):
    pass


def _import_sentence_transformers():
    if importlib.util.find_spec("sentence_transformers") is None:
        raise SemanticModelUnavailableError(
            "sentence-transformers is not installed. Install the optional "
            "'embeddings' extra (e.g. 'uv sync --dev --extra demo --extra "
            "embeddings') before using the local semantic model."
        )
    import sentence_transformers

    return sentence_transformers


class SemanticLocalEmbedding(EmbeddingProvider):
    mode: str = MODE
    semantic: bool = True

    def __init__(
        self,
        cache_dir: str | Any,
        manifest: EmbeddingModelManifest | None = None,
        *,
        device: str = MODEL_DEVICE,
        local_files_only: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thread_count: int = DEFAULT_THREAD_COUNT,
    ) -> None:
        if manifest is None:
            raise EmbeddingModelManifestError(
                "manifest_required",
                "SemanticLocalEmbedding requires an embedding model manifest; "
                "the provider never trusts hardcoded approval constants alone. "
                "Load and audit the committed manifest "
                "(data/manifests/embedding-model-v1.0.0.json) and pass it in.",
            )
        if not local_files_only:
            raise SemanticModelUnavailableError(
                "local_files_only must be True: the semantic model is only ever "
                "loaded from the verified local cache, never fetched from the "
                "network at inference."
            )
        if device != MODEL_DEVICE:
            raise SemanticModelUnavailableError(
                f"device {device!r} is not approved; the semantic model runs "
                f"only on the approved device {MODEL_DEVICE!r}."
            )

        self._cache_dir = str(Path(cache_dir))
        self._device = MODEL_DEVICE
        self._local_files_only = True
        self._batch_size = batch_size
        self._thread_count = thread_count

        audit_embedding_model_manifest(manifest)
        self._assert_cache_verified(manifest)

        self._manifest = manifest
        self._model_id = manifest.model_id
        self._model_version = manifest.model_version
        self._publisher = manifest.publisher
        self._official_url = manifest.official_url
        self._revision = manifest.revision
        self._revision_tag = manifest.revision_tag
        self._dimension = manifest.dimension
        self._max_input_tokens = manifest.max_input_tokens
        self._license = {
            "name": manifest.license.name,
            "version": manifest.license.version,
            "url": manifest.license.url,
        }
        self._acquisition_method = manifest.acquisition_method
        self._model = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def license(self) -> dict[str, str]:
        return dict(self._license)

    @property
    def cache_path(self) -> str:
        return self._cache_dir

    @property
    def device(self) -> str:
        return self._device

    @property
    def normalization(self) -> bool:
        return True

    def _assert_cache_verified(self, manifest: EmbeddingModelManifest) -> None:
        try:
            verify_embedding_model_files(manifest, self._cache_dir)
        except EmbeddingModelManifestError as exc:
            raise SemanticModelUnavailableError(
                f"semantic model cache failed verification at {self._cache_dir}: "
                f"{exc} — re-run scripts/acquire_embedding_model.py and "
                "scripts/verify_embedding_model.py to restore the exact pinned "
                "snapshot; semantic retrieval is NOT enabled without it."
            ) from exc

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        st = _import_sentence_transformers()
        import torch

        torch.set_num_threads(self._thread_count)

        try:
            self._model = st.SentenceTransformer(
                self._cache_dir,
                device=self._device,
                local_files_only=True,
            )
        except Exception as exc:
            raise SemanticModelUnavailableError(
                f"failed to load the local semantic model from "
                f"{self._cache_dir} (local_files_only=True): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if getter is None:
            raise SemanticModelUnavailableError(
                "could not determine the loaded semantic model's embedding "
                "dimension; refusing to use an unverifiable model."
            )
        try:
            actual_dimension = int(getter())
        except Exception as exc:
            raise SemanticModelUnavailableError(
                "could not determine the loaded semantic model's embedding "
                "dimension; refusing to use an unverifiable model."
            ) from exc
        if actual_dimension != self._dimension:
            raise SemanticModelUnavailableError(
                f"semantic model embedding dimension {actual_dimension} does not "
                f"match the approved dimension {self._dimension}; refusing to "
                "use the model."
            )
        return self._model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        model = self._ensure_loaded()
        embeddings = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self.normalization,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[Vector] = []
        for index, vector in enumerate(embeddings):
            row = tuple(float(value) for value in vector)
            if len(row) != self._dimension:
                raise EmbeddingDimensionError(
                    f"semantic vector {index} has dimension {len(row)}, expected "
                    f"{self._dimension}"
                )
            vectors.append(row)
        return vectors

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self._model_id,
            "model_version": self._model_version,
            "publisher": self._publisher,
            "official_url": self._official_url,
            "mode": self.mode,
            "semantic": self.semantic,
            "dimension": self._dimension,
            "revision": self._revision,
            "revision_tag": self._revision_tag,
            "license": dict(self._license),
            "max_input_tokens": self._max_input_tokens,
            "pooling": "cls",
            "normalization": self.normalization,
            "device": self._device,
            "cache_path": self._cache_dir,
            "local_files_only": self._local_files_only,
            "acquisition_method": self._acquisition_method,
        }
