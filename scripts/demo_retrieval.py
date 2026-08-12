"""Offline Streamlit demo for the evidence-grounded RAG system.

Two explicit corpus sources:

- **Synthetic fixture** (default): runs entirely from committed synthetic
  fixtures. No credentials, no network, no model downloads. Every panel is
  failure-visible: gate failures, validation failures, and abstentions are
  shown explicitly.
- **Real local corpus**: the approved Python 3.12.13 documentation corpus,
  retrieval only. Retrieval modes are explicitly labeled and never silently
  swap: ``BM25 sparse baseline``, ``Hash embedding diagnostic``, ``Local
  semantic embeddings``, and ``Semantic hybrid``. The semantic modes load the
  local model with ``local_files_only=True`` and show an actionable warning
  instead of falling back to hash embeddings when it is unavailable.

Run:  uv run --extra demo streamlit run scripts/demo_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evidence_grounded_rag import EvidenceRagPipeline, PipelineConfig
from evidence_grounded_rag.embeddings import (
    SemanticLocalEmbedding,
    SemanticModelUnavailableError,
    load_embedding_model_manifest,
)
from evidence_grounded_rag.evaluation.real_materialize import build_real_local
from evidence_grounded_rag.generation import AnswerState
from evidence_grounded_rag.retrieval import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    HybridRetriever,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"

REAL_MANIFEST = ROOT / "data" / "manifests" / "corpus-source-v1.0.0.json"
REAL_RAW_DIR = ROOT / "data" / "raw" / "python-docs-3.12.13"
MODEL_MANIFEST = ROOT / "data" / "manifests" / "embedding-model-v1.0.0.json"
MODEL_CACHE_DIR = ROOT / "data" / "raw" / "models" / "bge-small-en-v1.5"

CORPUS_SOURCE_SYNTHETIC = "Synthetic fixture"
CORPUS_SOURCE_REAL = "Real local corpus"
CORPUS_SOURCES = (CORPUS_SOURCE_SYNTHETIC, CORPUS_SOURCE_REAL)

REAL_MODE_SPARSE = "BM25 sparse baseline"
REAL_MODE_HASH = "Hash embedding diagnostic"
REAL_MODE_SEMANTIC = "Local semantic embeddings"
REAL_MODE_HYBRID = "Semantic hybrid"
REAL_MODES = (
    REAL_MODE_SPARSE,
    REAL_MODE_HASH,
    REAL_MODE_SEMANTIC,
    REAL_MODE_HYBRID,
)
_SEMANTIC_REAL_MODES = (REAL_MODE_SEMANTIC, REAL_MODE_HYBRID)

DEFAULT_QUERY = "What must a permit application include?"


@st.cache_resource
def load_pipeline() -> EvidenceRagPipeline:
    return EvidenceRagPipeline(MANIFEST, config=PipelineConfig(top_k=5))


@st.cache_resource
def load_real_corpus():
    return build_real_local(REAL_MANIFEST, REAL_RAW_DIR)


def _semantic_status() -> tuple[object | None, str | None]:
    """Return (provider, error) for the local semantic model.

    The provider itself enforces the boundary: the committed manifest is loaded
    and passed to ``SemanticLocalEmbedding``, which audits it against the
    approval constants and byte-verifies the cache before it is usable. Any
    manifest, cache, or dimension failure surfaces here as an actionable
    unavailable message. Never falls back to hash embeddings: either a verified
    local provider is returned or an actionable error message is returned.
    """
    if st.session_state.get("force_semantic_unavailable"):
        return None, (
            "Semantic model marked unavailable for demo verification; the "
            "local cache was deliberately not used. Semantic retrieval is NOT "
            "enabled and will not fall back to hash embeddings."
        )
    try:
        manifest = load_embedding_model_manifest(MODEL_MANIFEST)
        return SemanticLocalEmbedding(MODEL_CACHE_DIR, manifest=manifest), None
    except Exception as exc:  # noqa: BLE001 - demo must surface the exact blocker
        return None, (
            f"Local semantic model is unavailable: {exc} "
            "(install the 'embeddings' extra and run "
            "scripts/acquire_embedding_model.py then "
            "scripts/verify_embedding_model.py first). Semantic retrieval is "
            "NOT enabled and will not fall back to hash embeddings."
        )


def _chunk_snippet(corpus, chunk_id: str) -> str:
    if corpus is None:
        return ""
    chunk = corpus.chunk_by_id(chunk_id)
    return chunk.text if chunk else "(chunk not found)"


def _result_table(corpus, title: str, result) -> None:
    st.markdown(f"#### {title}")
    if result.empty:
        st.caption("no candidates")
        return
    rows = []
    for item in result.chunks:
        rows.append(
            {
                "rank": item.rank,
                "chunk_id": item.chunk_id,
                "score": round(item.score, 4),
                "snippet": _chunk_snippet(corpus, item.chunk_id)[:80],
            }
        )
    st.table(rows)


def _show_answer(record) -> None:
    st.markdown("### Structured answer")
    if record.state is AnswerState.ANSWERED:
        st.success("ANSWERED")
        st.markdown(record.answer_text)
        st.markdown("**Claims**")
        for claim in record.claims:
            refs = ", ".join(
                f"[{index + 1}]"
                for index, citation in enumerate(record.citations)
                if citation.chunk_id in claim.citation_chunk_ids
            )
            st.markdown(f"- `{claim.claim_id}` — {claim.text} {refs}")
        st.markdown("**Claim citations**")
        st.table(
            [
                {
                    "citation_id": citation.citation_id,
                    "chunk_id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "location": citation.location,
                }
                for citation in record.citations
            ]
        )
    else:
        state_label = (
            "CONFLICTING EVIDENCE"
            if record.state is AnswerState.CONFLICTING
            else "ABSTAINED"
        )
        st.warning(f"{state_label}: `{record.abstention_reason.value if record.abstention_reason else None}`")
        st.markdown(record.abstention_detail or "")


def _show_validation(record) -> None:
    st.markdown("### Citation validation")
    if record.validation is None:
        st.caption("no validation performed")
        return
    if record.validation.skipped:
        st.info("validation skipped: no citations on a refusal (never presented as verified)")
        return
    if record.validation.valid:
        st.success(
            f"VALID — {record.validation.checked_citations} citations, "
            f"{record.validation.checked_claims} claims"
        )
    else:
        st.error("VALIDATION FAILED — answer was refused")
        for issue in record.validation.issues:
            st.markdown(f"- `{issue.code}`: {issue.message}")


def _show_versions_config(record) -> None:
    st.markdown("### Versions & configuration")
    st.json(
        {
            "versions": record.versions,
            "config": record.config,
            "generation": record.generation,
        }
    )


def _show_trace(record) -> None:
    st.markdown("### Trace")
    if record.trace is None:
        st.caption("no retrieval trace (corpus gate failed)")
        return
    st.json(record.trace.to_dict())


def _show_synthetic() -> None:
    st.markdown(
        "### SYNTHETIC / OFFLINE EVALUATION\n"
        "All content is **fictional, project-generated synthetic fixture data** "
        "(`synthetic://`). No real corpus is used and the real-corpus gate is "
        "**BLOCKED**. The demo makes no network calls and uses no credentials."
    )

    pipeline = load_pipeline()
    status = pipeline.corpus_status()

    with st.expander("Corpus & license", expanded=True):
        if not status["loaded"]:
            st.error(f"corpus not loaded: {status['reason']} — {status['detail']}")
        else:
            st.markdown(
                f"- **corpus:** `{status['corpus_id']}` v{status['corpus_version']} "
                f"({status['num_documents']} documents, {status['num_chunks']} chunks)"
            )
            st.markdown(f"- **provenance:** `{status['provenance']}` — synthetic: {status['synthetic']}")
            st.markdown(f"- **license status:** `{status['license']['status']}`")
            st.markdown(f"- **real corpus:** `{status['real_corpus']}` — gate blocked: {status['real_corpus_gate']['blocked']}")
            st.markdown(f"**Real-corpus blocker:** {status['real_corpus_gate']['blocker']}")

    col_mode, col_k, _ = st.columns([2, 1, 3])
    with col_mode:
        mode = st.selectbox(
            "Retrieval mode",
            (MODE_HYBRID, MODE_SPARSE, MODE_DENSE),
            key="mode",
        )
    with col_k:
        top_k = st.slider("top_k", 1, 10, 5, key="top_k")

    query = st.text_input("Query", value=DEFAULT_QUERY, key="query_input")
    run = st.button("Run retrieval", key="run_button", type="primary")

    if run:
        if not query.strip():
            st.error("query is empty")
            return

        if pipeline.retriever is None:
            record = pipeline.answer(query, top_k=top_k, mode=mode)
            _show_answer(record)
            _show_validation(record)
            _show_versions_config(record)
            _show_trace(record)
            return

        try:
            record = pipeline.answer(query, top_k=top_k, mode=mode)
            sparse = pipeline.retriever.search(query, top_k=top_k, mode=MODE_SPARSE)
            dense = pipeline.retriever.search(query, top_k=top_k, mode=MODE_DENSE)
            fused = pipeline.retriever.search(query, top_k=top_k, mode=MODE_HYBRID)
        except Exception as exc:  # noqa: BLE001 - demo must surface failures
            st.error(f"pipeline failed: {exc}")
            return

        st.markdown("## Retrieval results")
        _result_table(pipeline.corpus, "Sparse (BM25)", sparse.fused)
        _result_table(pipeline.corpus, "Dense (deterministic, non-semantic)", dense.fused)
        _result_table(pipeline.corpus, f"Hybrid (RRF, mode={mode})", fused.fused)

        if mode != MODE_HYBRID:
            st.caption(f"Answer is grounded on the **{mode}** ranking above.")

        if record.evidence:
            top_chunk_id = record.evidence[0].chunk_id
            document = pipeline.corpus.document_by_id(record.evidence[0].document_id)
            if document is not None:
                st.markdown("### Passage / source metadata (top evidence)")
                chunk = pipeline.corpus.chunk_by_id(top_chunk_id)
                st.table(
                    {
                        "document_id": document.doc_id,
                        "title": document.title,
                        "document_version": document.version,
                        "captured_at": document.captured_at,
                        "source_name": document.source_name,
                        "source_url": document.source_url,
                        "license_note": document.license_note,
                        "checksum_sha256": document.checksum_sha256,
                        "chunk_id": top_chunk_id,
                        "location": chunk.location if chunk else "",
                    }
                )

        _show_answer(record)
        _show_validation(record)
        _show_versions_config(record)
        _show_trace(record)
    else:
        st.caption("Enter a query above and press **Run retrieval**.")


def _show_real_local() -> None:
    st.markdown(
        "### REAL LOCAL EVALUATION — approved local-only corpus\n"
        "**Retrieval only.** Corpus: Python 3.12.13 official documentation "
        "(3 files / 70 chunks), approved for local development. The demo runs "
        "offline; semantic modes use the local model with `local_files_only=True` "
        "and never fall back to hash embeddings. For the full labeled evaluation "
        "run `uv run python -m "
        "evidence_grounded_rag.evaluation.semantic_materialize`."
    )

    try:
        corpus = load_real_corpus()
    except Exception as exc:  # noqa: BLE001 - demo must surface failures
        st.error(f"real local corpus unavailable: {exc}")
        return

    with st.expander("Corpus & license", expanded=True):
        st.markdown(
            f"- **corpus:** `{corpus.corpus_id}` v{corpus.corpus_version} "
            f"({len(corpus.documents)} documents, {len(corpus.chunks())} chunks)"
        )
        st.markdown(
            f"- **license:** {corpus.manifest.license.name} v"
            f"{corpus.manifest.license.version} ({corpus.manifest.license.url})"
        )
        st.markdown(
            f"- **revision:** `{corpus.manifest.revision}` "
            f"({corpus.manifest.revision_tag})"
        )

    col_mode, col_k, _ = st.columns([2, 1, 3])
    with col_mode:
        real_mode = st.selectbox("Retrieval mode", REAL_MODES, key="mode")
    with col_k:
        top_k = st.slider("top_k", 1, 10, 5, key="top_k")

    query = st.text_input("Query", value=DEFAULT_QUERY, key="query_input")
    run = st.button("Run retrieval", key="run_button", type="primary")

    if not run:
        st.caption("Enter a query above and press **Run retrieval**.")
        return
    if not query.strip():
        st.error("query is empty")
        return

    semantic_selected = real_mode in _SEMANTIC_REAL_MODES
    if semantic_selected:
        provider, error = _semantic_status()
        if provider is None:
            st.error(f"**Semantic retrieval NOT enabled** — {error}")
            return
        retriever = HybridRetriever(corpus, embedding_provider=provider)
        search_mode = MODE_DENSE if real_mode == REAL_MODE_SEMANTIC else MODE_HYBRID
    else:
        retriever = HybridRetriever(corpus)
        search_mode = MODE_SPARSE if real_mode == REAL_MODE_SPARSE else MODE_DENSE

    try:
        sparse = retriever.search(query, top_k=top_k, mode=MODE_SPARSE)
        dense = retriever.search(query, top_k=top_k, mode=MODE_DENSE)
        fused = retriever.search(query, top_k=top_k, mode=search_mode)
    except SemanticModelUnavailableError as exc:
        st.error(
            f"**Semantic retrieval NOT enabled** — the local semantic model "
            f"failed while loading or embedding: {exc} Semantic retrieval does "
            "NOT fall back to hash embeddings."
        )
        return
    except Exception as exc:  # noqa: BLE001 - demo must surface failures
        st.error(f"retrieval failed: {exc}")
        return

    st.markdown("## Retrieval results")
    _result_table(corpus, "Sparse (BM25)", sparse.fused)
    _result_table(
        corpus,
        "Dense — " + ("semantic (local model)" if semantic_selected else "deterministic hash, non-semantic"),
        dense.fused,
    )
    _result_table(corpus, f"Fused — {real_mode}", fused.fused)
    st.caption(f"Answer is grounded on the **{real_mode}** ranking above.")


def main() -> None:
    st.set_page_config(page_title="Evidence-Grounded RAG — retrieval demo", layout="wide")
    st.title("Evidence-Grounded RAG — offline demo")

    corpus_source = st.selectbox(
        "Corpus source",
        CORPUS_SOURCES,
        index=0,
        key="corpus_source",
        help=(
            "Synthetic fixture (default, offline, fictional) or the approved "
            "real local corpus (Python 3.12.13 docs, retrieval only)."
        ),
    )

    if corpus_source == CORPUS_SOURCE_REAL:
        _show_real_local()
    else:
        _show_synthetic()


main()
