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
