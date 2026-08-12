"""Streamlit demo smoke test via AppTest (requires the demo extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_retrieval.py"


def test_demo_renders_offline_and_answers_default_query():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    assert not app.exception, app.exception

    app.text_input(key="query_input").set_value("What must a permit application include?")
    app.selectbox(key="mode").set_value("hybrid")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception

    body = " ".join(str(item.value) for item in app.markdown)
    assert "SYNTHETIC" in body or "synthetic" in body
    success_text = " ".join(str(item.value) for item in app.success)
    assert "ANSWERED" in success_text


def test_demo_surfaces_abstention():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.text_input(key="query_input").set_value("How much is the special event fee per day?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    warning_text = " ".join(str(item.value) for item in app.warning)
    assert "CONFLICTING" in warning_text
    assert "conflicting_evidence" in warning_text


def test_demo_always_shows_corpus_license_panel():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    body = " ".join(str(item.value) for item in app.markdown)
    assert "real corpus" in body
    assert "blocker" in body or "BLOCKED" in body
    assert not app.exception, app.exception


# --------------------------------------------------------------------------- #
# corpus source selector
# --------------------------------------------------------------------------- #


def test_demo_defaults_to_synthetic_fixture():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    assert app.selectbox(key="corpus_source").value == "Synthetic fixture"
    body = " ".join(str(item.value) for item in app.markdown)
    assert "SYNTHETIC" in body
    assert not app.exception, app.exception


# --------------------------------------------------------------------------- #
# generation state
# --------------------------------------------------------------------------- #


def test_demo_defaults_to_offline_generation():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    assert app.selectbox(key="generation_mode").value == "Offline deterministic extractive"
    body = " ".join(str(item.value) for item in app.markdown)
    assert "Offline" in body
    assert not app.exception, app.exception


def test_demo_offline_generation_answers_and_shows_support():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.text_input(key="query_input").set_value("What must a permit application include?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    success_text = " ".join(str(item.value) for item in app.success)
    assert "ANSWERED" in success_text
    body = " ".join(str(item.value) for item in app.markdown)
    assert "Evidence pack" in body
    assert "Claim support" in body
    assert "normalized-exact-containment-v1" in body
    assert "Generation state" in body
    assert "Offline deterministic extractive" in body


def test_demo_live_generation_shows_actionable_unavailable_and_no_fallback():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.selectbox(key="generation_mode").set_value("Live provider (not configured)")
    app.text_input(key="query_input").set_value("What must a permit application include?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    error_text = " ".join(str(item.value) for item in app.error)
    assert "Live generation is not available" in error_text
    assert "not fall back" in error_text.lower()
    warning_text = " ".join(str(item.value) for item in app.warning)
    assert "provider_unavailable" in warning_text


def test_demo_always_shows_faithfulness_warning():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    warning_text = " ".join(str(item.value) for item in app.warning)
    assert "Retrieval quality does not prove answer faithfulness." in warning_text
    assert not app.exception, app.exception


def test_demo_real_local_mode_labels_present():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.selectbox(key="corpus_source").set_value("Real local corpus")
    app.run()
    options = app.selectbox(key="mode").options
    assert "BM25 sparse baseline" in options
    assert "Hash embedding diagnostic" in options
    assert "Local semantic embeddings" in options
    assert "Semantic hybrid" in options
    assert not app.exception, app.exception


RAW_REAL = ROOT / "data" / "raw" / "python-docs-3.12.13"


@pytest.mark.skipif(
    not RAW_REAL.is_dir(),
    reason="ignored real corpus capture is absent (run scripts/acquire_corpus.py first)",
)
def test_demo_real_local_semantic_unavailable_shows_actionable_warning():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.session_state["force_semantic_unavailable"] = True
    app.run()
    app.selectbox(key="corpus_source").set_value("Real local corpus")
    app.run()
    app.selectbox(key="mode").set_value("Local semantic embeddings")
    app.text_input(key="query_input").set_value("What is a generator expression?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    error_text = " ".join(str(item.value) for item in app.error)
    assert "Semantic retrieval NOT enabled" in error_text
    assert "not fall back to hash embeddings" in error_text
    body = " ".join(str(item.value) for item in app.markdown)
    assert "Semantic retrieval" not in body or "NOT enabled" in error_text


@pytest.mark.skipif(
    not RAW_REAL.is_dir(),
    reason="ignored real corpus capture is absent (run scripts/acquire_corpus.py first)",
)
def test_demo_semantic_manifest_failure_shows_actionable_warning(monkeypatch):
    from evidence_grounded_rag.embeddings import EmbeddingModelManifestError

    def _boom(*args, **kwargs):
        raise EmbeddingModelManifestError(
            "malformed_manifest", "simulated tampered model manifest"
        )

    monkeypatch.setattr(
        "evidence_grounded_rag.embeddings.load_embedding_model_manifest", _boom
    )
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.selectbox(key="corpus_source").set_value("Real local corpus")
    app.run()
    app.selectbox(key="mode").set_value("Local semantic embeddings")
    app.text_input(key="query_input").set_value("What is a generator expression?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    error_text = " ".join(str(item.value) for item in app.error)
    assert "Semantic retrieval NOT enabled" in error_text
    assert "not fall back to hash embeddings" in error_text


@pytest.mark.skipif(
    not RAW_REAL.is_dir(),
    reason="ignored real corpus capture is absent (run scripts/acquire_corpus.py first)",
)
def test_demo_real_local_hash_mode_renders_results():
    app = AppTest.from_file(str(DEMO), default_timeout=30)
    app.run()
    app.selectbox(key="corpus_source").set_value("Real local corpus")
    app.run()
    app.selectbox(key="mode").set_value("Hash embedding diagnostic")
    app.text_input(key="query_input").set_value("What is a generator expression?")
    app.button(key="run_button").click()
    app.run()
    assert not app.exception, app.exception
    body = " ".join(str(item.value) for item in app.markdown)
    assert "deterministic hash" in body
    assert "REAL LOCAL EVALUATION" in body
