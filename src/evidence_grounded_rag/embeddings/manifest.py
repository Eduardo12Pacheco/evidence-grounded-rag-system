from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_grounded_rag.corpus.audit import (
    REDISTRIBUTION_LOCAL_ONLY,
    STATUS_APPROVED_LOCAL,
)
from evidence_grounded_rag.corpus.normalize import sha256_bytes
from evidence_grounded_rag.versions import (
    EMBEDDING_MODEL_MANIFEST_VERSION,
    SEMANTIC_EMBEDDING_MODEL_ID,
)

SUPPORTED_EMBEDDING_MODEL_MANIFEST_VERSION = EMBEDDING_MODEL_MANIFEST_VERSION

MODEL_ID = SEMANTIC_EMBEDDING_MODEL_ID
MODEL_VERSION = "1.5"
MODEL_PUBLISHER = "Beijing Academy of Artificial Intelligence (BAAI)"
MODEL_OFFICIAL_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5"
MODEL_CARD_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_REVISION_TAG = MODEL_REVISION
MODEL_LICENSE_NAME = "MIT License"
MODEL_LICENSE_VERSION = "1.0"
MODEL_LICENSE_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5"
MODEL_LICENSE_EVIDENCE_EXCERPT = (
    "Official model card (BAAI/bge-small-en-v1.5, README.md, pinned revision "
    '5c38ec7c): "FlagEmbedding is licensed under the MIT License. The released '
    'models can be used for commercial purposes free of charge." The card also '
    'declares "license: mit" in its YAML metadata.'
)
MODEL_DIMENSION = 384
MODEL_MAX_INPUT_TOKENS = 512
MODEL_DEVICE = "cpu"
MODEL_ACQUISITION_METHOD = (
    "Hugging Face resolve URLs pinned to the exact revision "
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a "
    "(https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/<revision>/<file>); "
    "no mutable branch, no mirror, no fallback"
)

MODEL_ARCHITECTURE: dict[str, object] = {
    "name": "bert",
    "model_type": "BertModel",
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": MODEL_DIMENSION,
    "intermediate_size": 1536,
    "vocab_size": 30522,
    "torch_dtype": "float32",
    "max_position_embeddings": 512,
}

MODEL_TOKENIZER: dict[str, object] = {
    "name": "WordPiece (bert-base-uncased)",
    "vocab_size": 30522,
    "lowercase": True,
    "max_length": MODEL_MAX_INPUT_TOKENS,
    "special_tokens": ["[CLS]", "[MASK]", "[PAD]", "[SEP]", "[UNK]"],
}

MODEL_POOLING: dict[str, object] = {
    "method": "cls",
    "note": (
        "1_Pooling/config.json sets pooling_mode_cls_token=true; the "
        "sentence-transformers module chain also includes a 2_Normalize "
        "L2-normalization module."
    ),
}

MODEL_NORMALIZATION: dict[str, object] = {
    "l2_normalize": True,
    "apply_at": (
        "sentence-transformers 2_Normalize module plus "
        "encode(normalize_embeddings=True)"
    ),
}

MODEL_CACHE_POLICY = (
    "local-only under data/raw/models/bge-small-en-v1.5 (gitignored); weights "
    "are never committed, packaged, or redistributed"
)

REQUIRED_FILES: tuple[str, ...] = (
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

ALLOWED_CACHE_EXTRAS: frozenset[str] = frozenset()

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HTTP_URL_RE = re.compile(r"^https?://")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = (
    "manifest_version",
    "model_id",
    "model_version",
    "publisher",
    "official_url",
    "model_card_url",
    "revision",
    "revision_tag",
    "license",
    "architecture",
    "dimension",
    "max_input_tokens",
    "tokenizer",
    "pooling",
    "normalization",
    "device",
    "model_status",
    "acquisition_method",
    "cache_policy",
    "redistribution_status",
    "expected_files",
    "verification_timestamp",
)


class EmbeddingModelManifestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class ModelLicense:
    name: str
    version: str
    url: str
    evidence_excerpt: str


@dataclass(frozen=True)
class ModelFileEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class EmbeddingModelManifest:
    manifest_version: str
    model_id: str
    model_version: str
    publisher: str
    official_url: str
    model_card_url: str
    revision: str
    revision_tag: str
    license: ModelLicense
    architecture: dict[str, object]
    dimension: int
    max_input_tokens: int
    tokenizer: dict[str, object]
    pooling: dict[str, object]
    normalization: dict[str, object]
    device: str
    model_status: str
    acquisition_method: str
    cache_policy: str
    redistribution_status: str
    expected_files: tuple[ModelFileEntry, ...]
    verification_timestamp: str


def load_embedding_model_manifest(path: str | Path) -> EmbeddingModelManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise EmbeddingModelManifestError(
            "manifest_file_missing",
            f"embedding model manifest not found: {manifest_path}",
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EmbeddingModelManifestError(
            "malformed_manifest", f"malformed embedding model manifest JSON: {exc}"
        ) from exc
    return parse_embedding_model_manifest(data)


def parse_embedding_model_manifest(data: Any) -> EmbeddingModelManifest:
    if not isinstance(data, dict):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "embedding model manifest must be a JSON object"
        )

    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        raise EmbeddingModelManifestError(
            "missing_required_field",
            "embedding model manifest missing required fields: " + ", ".join(missing),
        )

    version = data["manifest_version"]
    if (
        not isinstance(version, str)
        or version != SUPPORTED_EMBEDDING_MODEL_MANIFEST_VERSION
    ):
        raise EmbeddingModelManifestError(
            "unsupported_manifest_version",
            f"unsupported embedding model manifest version {version!r}; this "
            f"implementation supports {SUPPORTED_EMBEDDING_MODEL_MANIFEST_VERSION!r}",
        )

    for field in (
        "model_id",
        "model_version",
        "publisher",
        "official_url",
        "model_card_url",
        "revision",
        "revision_tag",
        "device",
        "model_status",
        "acquisition_method",
        "cache_policy",
        "redistribution_status",
        "verification_timestamp",
    ):
        if not isinstance(data[field], str):
            raise EmbeddingModelManifestError(
                "malformed_manifest", f"field {field!r} must be a string"
            )

    if data["official_url"] and not _HTTP_URL_RE.match(data["official_url"]):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "official_url must be an http(s) URL"
        )
    if not _COMMIT_RE.fullmatch(data["revision"]):
        raise EmbeddingModelManifestError(
            "invalid_revision",
            "revision must be a full 40-character hex commit SHA "
            "(exact pinned revision, never a mutable branch)",
        )

    license_raw = data["license"]
    if not isinstance(license_raw, dict):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "'license' must be an object"
        )
    license_missing = [
        key
        for key in ("name", "version", "url", "evidence_excerpt")
        if key not in license_raw
    ]
    if license_missing:
        raise EmbeddingModelManifestError(
            "malformed_manifest",
            f"'license' missing required fields: {', '.join(license_missing)}",
        )
    for key in ("name", "version", "url", "evidence_excerpt"):
        if not isinstance(license_raw[key], str):
            raise EmbeddingModelManifestError(
                "malformed_manifest", f"license.{key} must be a string"
            )
    if license_raw["url"] and not _HTTP_URL_RE.match(license_raw["url"]):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "license.url must be an http(s) URL"
        )

    for field in ("architecture", "tokenizer", "pooling", "normalization"):
        if not isinstance(data[field], dict):
            raise EmbeddingModelManifestError(
                "malformed_manifest", f"{field!r} must be an object"
            )

    if not isinstance(data["dimension"], int) or data["dimension"] < 1:
        raise EmbeddingModelManifestError(
            "malformed_manifest", "dimension must be a positive integer"
        )
    if not isinstance(data["max_input_tokens"], int) or data["max_input_tokens"] < 1:
        raise EmbeddingModelManifestError(
            "malformed_manifest", "max_input_tokens must be a positive integer"
        )

    files_raw = data["expected_files"]
    if not isinstance(files_raw, list):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "'expected_files' must be a list"
        )
    entries = [_parse_file_entry(raw, i) for i, raw in enumerate(files_raw)]
    if not entries:
        raise EmbeddingModelManifestError(
            "malformed_manifest",
            "embedding model manifest must declare at least one expected file",
        )

    return EmbeddingModelManifest(
        manifest_version=version,
        model_id=data["model_id"],
        model_version=data["model_version"],
        publisher=data["publisher"],
        official_url=data["official_url"],
        model_card_url=data["model_card_url"],
        revision=data["revision"],
        revision_tag=data["revision_tag"],
        license=ModelLicense(
            name=license_raw["name"],
            version=license_raw["version"],
            url=license_raw["url"],
            evidence_excerpt=license_raw["evidence_excerpt"],
        ),
        architecture=dict(data["architecture"]),
        dimension=data["dimension"],
        max_input_tokens=data["max_input_tokens"],
        tokenizer=dict(data["tokenizer"]),
        pooling=dict(data["pooling"]),
        normalization=dict(data["normalization"]),
        device=data["device"],
        model_status=data["model_status"],
        acquisition_method=data["acquisition_method"],
        cache_policy=data["cache_policy"],
        redistribution_status=data["redistribution_status"],
        expected_files=tuple(entries),
        verification_timestamp=data["verification_timestamp"],
    )


def _parse_file_entry(raw: Any, index: int) -> ModelFileEntry:
    if not isinstance(raw, dict):
        raise EmbeddingModelManifestError(
            "malformed_manifest", f"expected_files[{index}] must be an object"
        )
    for key in ("path", "sha256"):
        if not isinstance(raw.get(key), str):
            raise EmbeddingModelManifestError(
                "malformed_manifest",
                f"expected_files[{index}].{key} must be a string",
            )
    path = raw["path"]
    if not _is_safe_relative_model_path(path):
        raise EmbeddingModelManifestError(
            "malformed_manifest",
            f"expected_files[{index}].path must be a plain relative path inside "
            f"the cache directory (no leading / or \\, no ''/'..' components, no "
            f"empty or absolute path), got {path!r}",
        )
    if not isinstance(raw.get("size"), int) or raw["size"] < 0:
        raise EmbeddingModelManifestError(
            "malformed_manifest",
            f"expected_files[{index}].size must be a non-negative integer",
        )
    if not _SHA256_RE.fullmatch(raw["sha256"].lower()):
        raise EmbeddingModelManifestError(
            "malformed_manifest",
            f"expected_files[{index}].sha256 must be a 64-character hex SHA-256",
        )
    return ModelFileEntry(path=path, size=raw["size"], sha256=raw["sha256"].lower())


def _is_safe_relative_model_path(path: str) -> bool:
    if not path or "\x00" in path:
        return False
    if path.startswith(("/", "\\")):
        return False
    parts = path.replace("\\", "/").split("/")
    return bool(parts) and all(part not in ("", ".", "..") for part in parts)


def audit_embedding_model_manifest(manifest: EmbeddingModelManifest) -> None:
    if manifest.model_status != STATUS_APPROVED_LOCAL:
        raise EmbeddingModelManifestError(
            "model_status_blocked",
            f"model_status {manifest.model_status!r} is not approved; only "
            f"{STATUS_APPROVED_LOCAL!r} is accepted",
        )

    if manifest.model_id != MODEL_ID:
        raise EmbeddingModelManifestError(
            "unapproved_model",
            f"model {manifest.model_id!r} is not the approved model {MODEL_ID!r}",
        )
    if manifest.model_version != MODEL_VERSION:
        raise EmbeddingModelManifestError(
            "model_version_mismatch",
            f"model version {manifest.model_version!r} does not match the "
            f"approved version {MODEL_VERSION!r}",
        )
    if manifest.publisher != MODEL_PUBLISHER:
        raise EmbeddingModelManifestError(
            "publisher_mismatch",
            f"publisher {manifest.publisher!r} does not match the approved "
            f"publisher {MODEL_PUBLISHER!r}",
        )
    if manifest.official_url != MODEL_OFFICIAL_URL:
        raise EmbeddingModelManifestError(
            "official_url_mismatch",
            f"official_url {manifest.official_url!r} does not match the approved "
            f"official_url {MODEL_OFFICIAL_URL!r}",
        )
    if manifest.model_card_url != MODEL_CARD_URL:
        raise EmbeddingModelManifestError(
            "model_card_url_mismatch",
            f"model_card_url {manifest.model_card_url!r} does not match the "
            f"approved model card URL {MODEL_CARD_URL!r}",
        )
    if manifest.revision != MODEL_REVISION:
        raise EmbeddingModelManifestError(
            "revision_mismatch",
            f"revision {manifest.revision!r} does not match the approved revision "
            f"{MODEL_REVISION!r}",
        )
    if manifest.revision_tag != MODEL_REVISION_TAG:
        raise EmbeddingModelManifestError(
            "revision_mismatch",
            f"revision tag {manifest.revision_tag!r} does not match the approved "
            f"revision tag {MODEL_REVISION_TAG!r}",
        )
    if manifest.license.name != MODEL_LICENSE_NAME:
        raise EmbeddingModelManifestError(
            "license_mismatch",
            f"license {manifest.license.name!r} does not match the approved "
            f"license {MODEL_LICENSE_NAME!r}",
        )
    if manifest.license.version != MODEL_LICENSE_VERSION:
        raise EmbeddingModelManifestError(
            "license_mismatch",
            f"license version {manifest.license.version!r} does not match the "
            f"approved version {MODEL_LICENSE_VERSION!r}",
        )
    if manifest.license.url != MODEL_LICENSE_URL:
        raise EmbeddingModelManifestError(
            "license_mismatch",
            f"license url {manifest.license.url!r} does not match the approved "
            f"url {MODEL_LICENSE_URL!r}",
        )
    if manifest.dimension != MODEL_DIMENSION:
        raise EmbeddingModelManifestError(
            "dimension_mismatch",
            f"dimension {manifest.dimension} does not match the approved model "
            f"dimension {MODEL_DIMENSION}",
        )
    if manifest.max_input_tokens != MODEL_MAX_INPUT_TOKENS:
        raise EmbeddingModelManifestError(
            "max_input_mismatch",
            f"max_input_tokens {manifest.max_input_tokens} does not match the "
            f"approved model max input {MODEL_MAX_INPUT_TOKENS}",
        )
    if manifest.device != MODEL_DEVICE:
        raise EmbeddingModelManifestError(
            "device_mismatch",
            f"device {manifest.device!r} does not match the approved device "
            f"{MODEL_DEVICE!r}",
        )
    if manifest.architecture != MODEL_ARCHITECTURE:
        raise EmbeddingModelManifestError(
            "architecture_mismatch",
            "architecture metadata does not match the approved model architecture",
        )
    if manifest.tokenizer != MODEL_TOKENIZER:
        raise EmbeddingModelManifestError(
            "tokenizer_mismatch",
            "tokenizer metadata does not match the approved model tokenizer",
        )
    if manifest.pooling != MODEL_POOLING:
        raise EmbeddingModelManifestError(
            "pooling_mismatch",
            "pooling metadata does not match the approved model pooling",
        )
    if manifest.normalization != MODEL_NORMALIZATION:
        raise EmbeddingModelManifestError(
            "normalization_mismatch",
            "normalization metadata does not match the approved model normalization",
        )
    if manifest.redistribution_status != REDISTRIBUTION_LOCAL_ONLY:
        raise EmbeddingModelManifestError(
            "redistribution_mismatch",
            f"redistribution_status {manifest.redistribution_status!r} must be "
            f"{REDISTRIBUTION_LOCAL_ONLY!r} for a local-only development model",
        )
    if not (manifest.acquisition_method or "").strip():
        raise EmbeddingModelManifestError(
            "missing_evidence", "missing acquisition_method evidence"
        )
    if not (manifest.cache_policy or "").strip():
        raise EmbeddingModelManifestError(
            "missing_evidence", "missing cache_policy evidence"
        )


def safe_model_path(cache_dir: str | Path, rel_path: str) -> Path:
    root = Path(cache_dir).resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EmbeddingModelManifestError(
            "file_path_escape",
            f"model file path escapes the cache directory: {rel_path!r}",
        ) from exc
    return candidate


def verify_embedding_model_files(
    manifest: EmbeddingModelManifest, cache_dir: str | Path
) -> None:
    root = Path(cache_dir)
    if not root.is_dir():
        raise EmbeddingModelManifestError(
            "cache_dir_missing", f"cache directory not found: {root}"
        )

    declared = {Path(entry.path).as_posix() for entry in manifest.expected_files}
    for current_dir, _subdirs, files in os.walk(root):
        for name in files:
            rel = Path(current_dir).relative_to(root).joinpath(name).as_posix()
            if rel not in declared and rel not in ALLOWED_CACHE_EXTRAS:
                raise EmbeddingModelManifestError(
                    "extra_cache_file",
                    f"undeclared file in model cache: {rel} (the cache must "
                    "contain exactly the manifest's expected_files)",
                )

    for entry in manifest.expected_files:
        path = safe_model_path(root, entry.path)
        if not path.is_file():
            raise EmbeddingModelManifestError(
                "file_missing", f"model file missing: {entry.path}"
            )
        size = path.stat().st_size
        if size != entry.size:
            raise EmbeddingModelManifestError(
                "file_size_mismatch",
                f"model file {entry.path}: actual size {size} != declared {entry.size}",
            )
        actual = sha256_bytes(path.read_bytes())
        if actual != entry.sha256:
            raise EmbeddingModelManifestError(
                "file_checksum_mismatch",
                f"model file {entry.path}: actual sha256 {actual} != declared "
                f"{entry.sha256}",
            )
