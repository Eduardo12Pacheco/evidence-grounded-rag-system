from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from evidence_grounded_rag.embeddings.base import EmbeddingProvider, Vector

MODEL_ID = "fake-semantic-provider-v1"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class FakeSemanticProvider(EmbeddingProvider):
    model_id = MODEL_ID
    mode = "semantic-local"
    semantic = True

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> Vector:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [(digest[index] - 127.0) / 128.0 for index in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def embed_one(self, text: str) -> Vector:
        return self._vector(text)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": MODEL_ID,
            "model_version": "fake",
            "publisher": "test-fixture",
            "official_url": "https://huggingface.co/fake",
            "mode": self.mode,
            "semantic": self.semantic,
            "dimension": self._dimension,
            "revision": REVISION,
            "revision_tag": REVISION,
            "license": {
                "name": "MIT License",
                "version": "1.0",
                "url": "https://huggingface.co/fake",
            },
            "max_input_tokens": 512,
            "pooling": "cls",
            "normalization": True,
            "device": "cpu",
            "cache_path": "tests/fake-cache",
            "local_files_only": True,
            "acquisition_method": "test fixture",
        }
