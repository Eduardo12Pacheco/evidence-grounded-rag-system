from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_grounded_rag import EvidenceRagPipeline, PipelineConfig
from evidence_grounded_rag.abstention import AbstentionReason
from evidence_grounded_rag.generation import AnswerState, GeneratedAnswer
from evidence_grounded_rag.generation.models import Citation, Claim

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "data" / "fixtures" / "synthetic_manifest.json"


@pytest.fixture(scope="module")
def pipeline():
    return EvidenceRagPipeline(FIXTURE_MANIFEST, config=PipelineConfig(top_k=10))


def test_pipeline_answers_supported_query(pipeline):
    record = pipeline.answer("What must a permit application include?", top_k=10)
    assert record.is_answered
    assert record.state is AnswerState.ANSWERED
    assert record.answer_text
    assert record.validation.valid is True
    assert record.validation.skipped is False
    assert record.citations
    assert all(
        citation.chunk_id in {item.chunk_id for item in record.evidence}
        for citation in record.citations
    )


def test_pipeline_abstains_on_conflicting_evidence(pipeline):
    record = pipeline.answer("How much is the special event fee per day?", top_k=10)
    assert record.state is AnswerState.CONFLICTING
    assert record.abstention_reason is AbstentionReason.CONFLICTING_EVIDENCE
    assert record.answer_text is None


def test_pipeline_abstains_on_unsupported_query(pipeline):
    record = pipeline.answer("What is the fine for littering in Millbrook?", top_k=10)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.UNSUPPORTED_QUERY


def test_pipeline_abstains_on_ambiguous_query(pipeline):
    record = pipeline.answer("When is the pool open?", top_k=10)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_generator_never_sees_unexposed_evidence():
    seen: list[list[str]] = []

    class RecordingGenerator:
        provider_id = "recording"
        model_id = "recording-v1"
        is_llm_generated = False

        def generate(self, query, evidence):
            seen.append([chunk.chunk_id for chunk in evidence])
            exposed = _single_chunk_evidence(query)
            return exposed

    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(top_k=5),
        generator=RecordingGenerator(),
    )
    record = pipe.answer("What must a permit application include?", top_k=5)
    exposed_ids = {item.chunk_id for item in record.evidence}
    assert set(seen[-1]) <= exposed_ids


def test_stale_corpus_abstains_with_stale_reason():
    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(expected_corpus_version="999.0.0"),
    )
    record = pipe.answer("anything")
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.STALE_CORPUS
    status = pipe.corpus_status()
    assert status["loaded"] is False
    assert status["reason"] == "stale_corpus"


def test_blocked_real_corpus_abstains_with_blocked_reason(tmp_path):
    manifest_path = _real_corpus_manifest(tmp_path)
    pipe = EvidenceRagPipeline(manifest_path)
    record = pipe.answer("anything")
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.BLOCKED_CORPUS
    assert "Gob.ec" in (record.abstention_detail or "")


def test_invalid_corpus_abstains_with_invalid_reason(tmp_path):
    raw = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    raw["documents"][0]["checksum_sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    pipe = EvidenceRagPipeline(path)
    record = pipe.answer("anything")
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.INVALID_CORPUS


def test_forged_citation_fails_closed_to_citation_validation():
    class ForgingGenerator:
        provider_id = "forging"
        model_id = "forging-v1"
        is_llm_generated = False

        def generate(self, query, evidence):
            if not evidence:
                return GeneratedAnswer(
                    state=AnswerState.ABSTAINED,
                    answer_text=None,
                    provider_id=self.provider_id,
                    model_id=self.model_id,
                    is_llm_generated=False,
                )
            forged = Citation(
                citation_id="cit-forged",
                chunk_id="doc-never-exists::0",
                document_id="doc-never-exists",
                corpus_version="1.0.0",
                excerpt="forged excerpt",
                location="synthetic://nowhere",
            )
            claim = Claim(
                claim_id="claim-1",
                text="forged claim",
                citation_chunk_ids=("doc-never-exists::0",),
            )
            return GeneratedAnswer(
                state=AnswerState.ANSWERED,
                answer_text="forged answer",
                claims=(claim,),
                citations=(forged,),
                provider_id=self.provider_id,
                model_id=self.model_id,
                is_llm_generated=False,
            )

    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(top_k=5),
        generator=ForgingGenerator(),
    )
    record = pipe.answer("What must a permit application include?", top_k=5)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.CITATION_VALIDATION_FAILED
    assert record.validation.valid is False


def test_malformed_provider_fails_closed():
    class MalformedGenerator:
        provider_id = "malformed"
        model_id = "malformed-v1"
        is_llm_generated = False

        def generate(self, query, evidence):
            return GeneratedAnswer(
                state=AnswerState.ANSWERED,
                answer_text=None,
                provider_id=self.provider_id,
                model_id=self.model_id,
                is_llm_generated=False,
            )

    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(top_k=5),
        generator=MalformedGenerator(),
    )
    record = pipe.answer("What must a permit application include?", top_k=5)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.MALFORMED_OUTPUT


def test_corpus_status_reports_loaded_gate(pipeline):
    status = pipeline.corpus_status()
    assert status["loaded"] is True
    assert status["synthetic"] is True
    assert status["real_corpus"] is False
    assert status["real_corpus_gate"]["blocked"] is True
    assert "Gob.ec" in status["real_corpus_gate"]["blocker"]
    assert status["num_chunks"] == 11


def test_answer_record_contains_complete_trace(pipeline):
    record = pipeline.answer("Is access to the East River trail free?", top_k=5)
    assert record.trace is not None
    trace = record.trace.to_dict()
    assert trace["sparse"]["chunks"]
    assert trace["dense"]["chunks"]
    assert trace["fused"]["chunks"]
    assert "corpus_version" in trace["versions"]
    assert record.versions["generation"] == "extractive-template-v1"
    assert record.generation["is_llm_generated"] is False


def test_missing_manifest_file_reports_invalid_corpus(tmp_path):
    pipe = EvidenceRagPipeline(tmp_path / "nope.json")
    record = pipe.answer("anything")
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.INVALID_CORPUS


def test_answer_record_generation_metadata_includes_mode(pipeline):
    record = pipeline.answer("What must a permit application include?", top_k=5)
    assert record.generation["provider_id"] == "extractive-template"
    assert record.generation["model_id"] == "extractive-template-v1"
    assert record.generation["mode"] == "offline"
    assert record.generation["is_llm_generated"] is False


def test_env_adapter_live_label_fails_closed_in_pipeline():
    from evidence_grounded_rag.generation import ENV_VAR, EnvConfiguredGenerator

    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(top_k=5),
        generator=EnvConfiguredGenerator(env={ENV_VAR: "live"}),
    )
    record = pipe.answer("What must a permit application include?", top_k=5)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert record.generation["mode"] == "offline"
    assert record.answer_text is None


def test_live_generator_without_transport_abstains_through_pipeline():
    from evidence_grounded_rag.generation import LiveGenerator

    pipe = EvidenceRagPipeline(
        FIXTURE_MANIFEST,
        config=PipelineConfig(top_k=5),
        generator=LiveGenerator(),
    )
    record = pipe.answer("What must a permit application include?", top_k=5)
    assert record.state is AnswerState.ABSTAINED
    assert record.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert record.generation["mode"] == "live"
    assert record.generation["is_llm_generated"] is True
    assert "not fall back" in (record.abstention_detail or "").lower()


def _single_chunk_evidence(query: str) -> GeneratedAnswer:
    from evidence_grounded_rag.corpus import load_corpus

    corpus = load_corpus(FIXTURE_MANIFEST)
    chunk = corpus.chunks()[0]
    citation = Citation(
        citation_id="cit-1",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        corpus_version=chunk.corpus_version,
        excerpt=chunk.text,
        location=chunk.location,
    )
    claim = Claim(
        claim_id="claim-1", text=chunk.text, citation_chunk_ids=(chunk.chunk_id,)
    )
    return GeneratedAnswer(
        state=AnswerState.ANSWERED,
        answer_text=chunk.text,
        claims=(claim,),
        citations=(citation,),
        provider_id="recording",
        model_id="recording-v1",
        is_llm_generated=False,
    )


def _real_corpus_manifest(tmp_path: Path) -> Path:
    raw = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    raw["manifest"]["synthetic"] = False
    raw["manifest"]["real_corpus"] = True
    raw["manifest"]["license"] = {
        "status": "unapproved",
        "terms": "unknown",
        "redistribution": "unknown",
    }
    raw["manifest"]["real_corpus_gate"] = {
        "blocked": True,
        "blocker": "No external license has been independently confirmed. The Gob.ec corpus is explicitly unauthorized and its license discrepancy is unresolved.",
    }
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path
