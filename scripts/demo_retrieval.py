"""Offline Streamlit demo for the evidence-grounded RAG system.

Runs entirely from committed synthetic fixtures. No credentials, no network,
no model downloads. Every panel is failure-visible: gate failures, validation
failures, and abstentions are shown explicitly.

Run:  uv run --extra demo streamlit run scripts/demo_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evidence_grounded_rag import EvidenceRagPipeline, PipelineConfig
from evidence_grounded_rag.generation import AnswerState
from evidence_grounded_rag.retrieval import MODE_DENSE, MODE_HYBRID, MODE_SPARSE

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"

DEFAULT_QUERY = "What must a permit application include?"


@st.cache_resource
def load_pipeline() -> EvidenceRagPipeline:
    return EvidenceRagPipeline(MANIFEST, config=PipelineConfig(top_k=5))


def _chunk_snippet(pipeline: EvidenceRagPipeline, chunk_id: str) -> str:
    if pipeline.corpus is None:
        return ""
    chunk = pipeline.corpus.chunk_by_id(chunk_id)
    return chunk.text if chunk else "(chunk not found)"


def _result_table(pipeline: EvidenceRagPipeline, title: str, result) -> None:
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
                "snippet": _chunk_snippet(pipeline, item.chunk_id)[:80],
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


def main() -> None:
    st.set_page_config(page_title="Evidence-Grounded RAG — synthetic demo", layout="wide")
    st.title("Evidence-Grounded RAG — offline demo")

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
        _result_table(pipeline, "Sparse (BM25)", sparse.fused)
        _result_table(pipeline, "Dense (deterministic, non-semantic)", dense.fused)
        _result_table(pipeline, f"Hybrid (RRF, mode={mode})", fused.fused)

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


main()
