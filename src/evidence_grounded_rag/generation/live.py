"""Explicit optional live generation adapter — provider-neutral, fail-closed.

This adapter exists so the system can OPTIONALLY call a live generation
provider WITHOUT naming one: no provider, model, or credentials are configured
anywhere, and nothing in this module opens a network connection. A live run
requires the caller to inject a callable *transport* that takes the built
prompt text and returns the provider's raw output text.

Fail-closed contract (never falls back to offline, ever):

- no transport injected  -> ``provider_unavailable``
- transport raises ``TimeoutError``  -> ``provider_unavailable`` (timeout)
- transport raises anything else  -> ``provider_unavailable``
- transport returns non-text  -> ``malformed_output``
- transport returns malformed JSON / schema violations  -> ``malformed_output``
- parsed answer fails evidence validation  -> ``malformed_output``

Under the live label the adapter never invokes the deterministic offline
provider; unavailability is surfaced as a structured abstention instead.
"""

from __future__ import annotations

from collections.abc import Callable

from evidence_grounded_rag.abstention import AbstentionReason, reason_detail
from evidence_grounded_rag.generation.evidence import EvidencePack
from evidence_grounded_rag.generation.models import AnswerState, GeneratedAnswer
from evidence_grounded_rag.generation.prompt import (
    DEFAULT_LIVE_TIMEOUT_SECONDS,
    PROMPT_VERSION,
    build_grounded_answer_prompt,
)
from evidence_grounded_rag.generation.structured import (
    SCHEMA_VERSION,
    StructuredAnswerParser,
    StructuredAnswerValidator,
    to_generated_answer,
)
from evidence_grounded_rag.versions import LIVE_GENERATOR_VERSION

PROVIDER_ID = "injected-live-provider"
MODEL_ID = LIVE_GENERATOR_VERSION
MODE = "live"

#: A transport is a callable ``prompt_text -> raw output text``. It owns any
#: network/credentials it needs; this system never provides them.
Transport = Callable[[str], str]


class LiveGenerator:
    """Provider-neutral live adapter requiring an injected transport."""

    provider_id: str = PROVIDER_ID
    model_id: str = MODEL_ID
    is_llm_generated: bool = True
    mode: str = MODE

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        provider_id: str = PROVIDER_ID,
        model_id: str = MODEL_ID,
        prompt_version: str = PROMPT_VERSION,
        timeout_seconds: int = DEFAULT_LIVE_TIMEOUT_SECONDS,
        parser: StructuredAnswerParser | None = None,
        validator: StructuredAnswerValidator | None = None,
    ) -> None:
        self._transport = transport
        self.provider_id = provider_id
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds
        self._parser = parser or StructuredAnswerParser()
        self._validator = validator or StructuredAnswerValidator()

    @property
    def transport_configured(self) -> bool:
        return self._transport is not None

    def generate(
        self,
        query: str,
        evidence: EvidencePack,
    ) -> GeneratedAnswer:
        if self._transport is None:
            return self._abstain(
                AbstentionReason.PROVIDER_UNAVAILABLE,
                (
                    "Live generation is not configured: no provider/model/"
                    "credentials exist and no transport was injected. Live "
                    "generation does NOT fall back to the offline provider."
                ),
            )

        if not isinstance(query, str) or not query.strip():
            return self._abstain(
                AbstentionReason.UNSUPPORTED_QUERY,
                "The query is empty; there is nothing to answer.",
            )
        if evidence.empty:
            return self._abstain(
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                "No evidence was exposed to the generator.",
            )

        prompt = build_grounded_answer_prompt(query, evidence)
        try:
            raw = self._transport(prompt)
        except TimeoutError:
            return self._abstain(
                AbstentionReason.PROVIDER_UNAVAILABLE,
                (
                    f"Live provider timed out after {self.timeout_seconds}s; "
                    "no answer was produced and generation did not fall back "
                    "to offline."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - provider failures are data
            return self._abstain(
                AbstentionReason.PROVIDER_UNAVAILABLE,
                (
                    f"Live provider call failed ({exc.__class__.__name__}: {exc}); "
                    "no answer was produced and generation did not fall back "
                    "to offline."
                ),
            )

        if not isinstance(raw, str):
            return self._abstain(
                AbstentionReason.MALFORMED_OUTPUT,
                "Live provider returned non-text output; expected a JSON string.",
            )

        parsed = self._parser.parse(raw)
        if not parsed.ok:
            detail = "; ".join(issue.message for issue in parsed.issues)
            return self._abstain(
                AbstentionReason.MALFORMED_OUTPUT,
                f"Live provider output failed structured parsing: {detail}",
            )

        assert parsed.answer is not None
        validation_issues = self._validator.validate(parsed.answer, evidence)
        if validation_issues:
            detail = "; ".join(issue.message for issue in validation_issues)
            return self._abstain(
                AbstentionReason.MALFORMED_OUTPUT,
                f"Live provider answer failed evidence validation: {detail}",
            )

        return to_generated_answer(
            parsed.answer,
            provider_id=self.provider_id,
            model_id=self.model_id,
            mode=self.mode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "mode": self.mode,
            "is_llm_generated": self.is_llm_generated,
            "prompt_version": self.prompt_version,
            "schema_version": SCHEMA_VERSION,
            "timeout_seconds": self.timeout_seconds,
            "transport_configured": self.transport_configured,
        }

    def _abstain(
        self,
        reason: AbstentionReason,
        detail: str,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            state=AnswerState.ABSTAINED,
            answer_text=None,
            abstention_reason=reason,
            abstention_detail=detail or reason_detail(reason),
            provider_id=self.provider_id,
            model_id=self.model_id,
            is_llm_generated=self.is_llm_generated,
            mode=self.mode,
        )
