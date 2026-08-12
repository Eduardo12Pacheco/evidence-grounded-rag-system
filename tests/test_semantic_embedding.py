from __future__ import annotations

import types
from pathlib import Path

import pytest

from evidence_grounded_rag.corpus import load_corpus
from evidence_grounded_rag.corpus.normalize import sha256_bytes
from evidence_grounded_rag.embeddings import (
    DenseIndex,
    DeterministicEmbedding,
    EmbeddingModelManifestError,
    SemanticLocalEmbedding,
    SemanticModelUnavailableError,
)
from evidence_grounded_rag.embeddings import semantic as semantic_module
from evidence_grounded_rag.embeddings.manifest import (
    MODEL_ARCHITECTURE,
    MODEL_CARD_URL,
    MODEL_DIMENSION,
    MODEL_ID,
    MODEL_LICENSE_NAME,
    MODEL_LICENSE_URL,
    MODEL_LICENSE_VERSION,
    MODEL_MAX_INPUT_TOKENS,
    MODEL_NORMALIZATION,
    MODEL_POOLING,
    MODEL_REVISION,
    MODEL_REVISION_TAG,
    MODEL_TOKENIZER,
    parse_embedding_model_manifest,
)
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    HybridRetriever,
)
from tests.fake_provider import MODEL_ID as FAKE_MODEL_ID
from tests.fake_provider import FakeSemanticProvider

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"
SEMANTIC_SOURCE = ROOT / "src" / "evidence_grounded_rag" / "embeddings" / "semantic.py"

REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_CACHE_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
    "README.md",
)


def _write_cache(tmp_path: Path, files: tuple[str, ...] = _CACHE_FILES) -> Path:
    cache_dir = tmp_path / "cache"
    for rel in files:
        path = cache_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode("utf-8"))
    return cache_dir


def _cache_entries(files: tuple[str, ...] = _CACHE_FILES) -> list[dict[str, object]]:
    return [
        {
            "path": rel,
            "size": len(rel.encode("utf-8")),
            "sha256": sha256_bytes(rel.encode("utf-8")),
        }
        for rel in files
    ]


def _manifest_doc(expected_files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "manifest_version": "1.0",
        "model_id": MODEL_ID,
        "model_version": "1.5",
        "publisher": "Beijing Academy of Artificial Intelligence (BAAI)",
        "official_url": MODEL_CARD_URL,
        "model_card_url": MODEL_CARD_URL,
        "revision": MODEL_REVISION,
        "revision_tag": MODEL_REVISION_TAG,
        "license": {
            "name": MODEL_LICENSE_NAME,
            "version": MODEL_LICENSE_VERSION,
            "url": MODEL_LICENSE_URL,
            "evidence_excerpt": "official model card quotes the MIT license",
        },
        "architecture": dict(MODEL_ARCHITECTURE),
        "dimension": MODEL_DIMENSION,
        "max_input_tokens": MODEL_MAX_INPUT_TOKENS,
        "tokenizer": dict(MODEL_TOKENIZER),
        "pooling": dict(MODEL_POOLING),
        "normalization": dict(MODEL_NORMALIZATION),
        "device": "cpu",
        "model_status": "approved-for-local-development",
        "acquisition_method": "pinned HF resolve URLs",
        "cache_policy": "local-only; never committed",
        "redistribution_status": "not-redistributed",
        "expected_files": expected_files,
        "verification_timestamp": "2026-08-11T00:00:00Z",
    }


def _make_manifest(
    expected_files: list[dict[str, object]] | None = None,
) -> object:
    return parse_embedding_model_manifest(
        _manifest_doc(
            expected_files if expected_files is not None else _cache_entries()
        )
    )


def _make_fake_st(dimension: int = 384, has_dimension_getter: bool = True):
    instances: list = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs
            self.encode_calls = []
            instances.append(self)

        def get_sentence_embedding_dimension(self):
            if not has_dimension_getter:
                raise AttributeError("no dimension getter")
            return dimension

        def encode(self, inputs, **kwargs):
            self.encode_calls.append((list(inputs), kwargs))
            return [
                [0.1 * (index + 1) for _ in range(dimension)]
                for index in range(len(inputs))
            ]

    fake = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer, instances=instances
    )
    return fake


@pytest.fixture()
def fake_st(monkeypatch):
    fake = _make_fake_st()
    monkeypatch.setattr(semantic_module, "_import_sentence_transformers", lambda: fake)
    return fake


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE_MANIFEST)


def test_provider_requires_manifest(tmp_path):
    cache_dir = _write_cache(tmp_path)
    with pytest.raises(EmbeddingModelManifestError, match="manifest_required"):
        SemanticLocalEmbedding(cache_dir)


def test_missing_cache_directory_raises(tmp_path):
    with pytest.raises(
        SemanticModelUnavailableError, match="cache directory not found"
    ):
        SemanticLocalEmbedding(tmp_path / "absent", manifest=_make_manifest())


def test_incomplete_cache_raises_with_missing_files(tmp_path):
    cache_dir = _write_cache(
        tmp_path, tuple(rel for rel in _CACHE_FILES if rel != "model.safetensors")
    )
    with pytest.raises(SemanticModelUnavailableError, match="model.safetensors"):
        SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())


def test_extra_cache_file_rejected(tmp_path):
    cache_dir = _write_cache(tmp_path)
    (cache_dir / "sneaky.bin").write_bytes(b"extra")
    with pytest.raises(SemanticModelUnavailableError, match="extra_cache_file"):
        SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())


def test_tampered_cache_bytes_rejected(tmp_path):
    cache_dir = _write_cache(tmp_path)
    (cache_dir / "config.json").write_bytes(b"tampered-not-the-approved-config")
    with pytest.raises(SemanticModelUnavailableError, match="failed verification"):
        SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())


def test_missing_dependency_raises(monkeypatch, tmp_path):
    cache_dir = _write_cache(tmp_path)

    def _boom():
        raise SemanticModelUnavailableError("sentence-transformers is not installed")

    monkeypatch.setattr(semantic_module, "_import_sentence_transformers", _boom)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    with pytest.raises(SemanticModelUnavailableError, match="not installed"):
        provider.embed(["hello"])


def test_dimension_mismatch_raises(monkeypatch, tmp_path):
    fake = _make_fake_st(dimension=768)
    monkeypatch.setattr(semantic_module, "_import_sentence_transformers", lambda: fake)
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    with pytest.raises(SemanticModelUnavailableError, match="dimension"):
        provider.embed(["hello"])


def test_unknown_loaded_dimension_fails(monkeypatch, tmp_path):
    fake = _make_fake_st(dimension=384, has_dimension_getter=False)
    monkeypatch.setattr(semantic_module, "_import_sentence_transformers", lambda: fake)
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    with pytest.raises(SemanticModelUnavailableError, match="could not determine"):
        provider.embed(["hello"])


def test_provider_never_uses_network_imports():
    source = SEMANTIC_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "snapshot_download",
        "hf_hub_download",
        "socket.",
    ):
        assert forbidden not in source, f"semantic.py must not use {forbidden!r}"


def test_provider_refuses_local_files_only_false(tmp_path):
    cache_dir = _write_cache(tmp_path)
    with pytest.raises(
        SemanticModelUnavailableError, match="local_files_only must be True"
    ):
        SemanticLocalEmbedding(
            cache_dir, manifest=_make_manifest(), local_files_only=False
        )


def test_provider_refuses_off_approved_device(tmp_path):
    cache_dir = _write_cache(tmp_path)
    with pytest.raises(SemanticModelUnavailableError, match="approved device"):
        SemanticLocalEmbedding(cache_dir, manifest=_make_manifest(), device="cuda")


def test_provider_loads_with_local_files_only(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    provider.embed(["hello"])
    transformer = fake_st.instances[0]
    assert transformer.init_kwargs["local_files_only"] is True
    assert transformer.init_kwargs["device"] == "cpu"
    assert transformer.init_args[0] == str(cache_dir)


def test_provider_embeds_with_normalization_and_no_download(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    provider.embed_one("hello")
    inputs, kwargs = fake_st.instances[0].encode_calls[0]
    assert inputs == ["hello"]
    assert kwargs["normalize_embeddings"] is True
    assert "local_files_only" not in kwargs
    assert kwargs["show_progress_bar"] is False


def test_provider_metadata(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    assert provider.model_id == "BAAI/bge-small-en-v1.5"
    assert provider.mode == "semantic-local"
    assert provider.semantic is True
    assert provider.dimension == 384
    assert provider.revision == REVISION
    assert provider.license["name"] == "MIT License"
    assert provider.cache_path == str(cache_dir)
    assert provider.device == "cpu"
    assert provider.normalization is True
    info = provider.to_dict()
    assert info["model_id"] == "BAAI/bge-small-en-v1.5"
    assert info["mode"] == "semantic-local"
    assert info["semantic"] is True
    assert info["dimension"] == 384
    assert info["revision"] == REVISION
    assert info["normalization"] is True
    assert info["cache_path"] == str(cache_dir)
    assert info["local_files_only"] is True


def test_provider_empty_input_returns_empty(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    assert provider.embed([]) == []


def test_provider_embed_shape_and_order(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    vectors = provider.embed(["first", "second"])
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 384
    assert vectors[0] != vectors[1]


def test_provider_long_text_is_handled(fake_st, tmp_path):
    cache_dir = _write_cache(tmp_path)
    provider = SemanticLocalEmbedding(cache_dir, manifest=_make_manifest())
    long_text = ("python function definition " * 400)[:5000]
    vector = provider.embed_one(long_text)
    assert len(vector) == 384


def test_dense_index_uses_semantic_provider_metadata(corpus):
    provider = FakeSemanticProvider(dimension=16)
    index = DenseIndex(corpus.chunks(), provider)
    result = index.search("permit application", top_k=3)
    assert not result.empty
    assert result.config["semantic"] is True
    assert result.config["dimension"] == 16
    assert result.config["mode"] == "semantic-local"
    assert result.versions["embedding"] == FAKE_MODEL_ID


def test_hybrid_semantic_modes_trace_provider(corpus):
    provider = FakeSemanticProvider(dimension=16)
    retriever = HybridRetriever(corpus, embedding_provider=provider)
    dense = retriever.search("permit", top_k=3, mode=MODE_DENSE)
    hybrid = retriever.search("permit", top_k=3, mode=MODE_HYBRID)
    assert dense.trace.versions["embedding"] == FAKE_MODEL_ID
    assert hybrid.trace.versions["embedding"] == FAKE_MODEL_ID
    assert dense.trace.dense.config["semantic"] is True
    assert not dense.fused.empty
    assert not hybrid.fused.empty


def test_sparse_mode_does_not_need_semantic_provider(corpus):
    retriever = HybridRetriever(corpus, embedding_provider=DeterministicEmbedding())
    result = retriever.search("permit", top_k=3, mode=MODE_SPARSE)
    assert result.fused.config["retriever"] == "bm25"
    assert result.trace.versions["embedding"] == "deterministic-hashed-bow-v1"
