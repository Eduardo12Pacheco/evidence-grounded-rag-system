from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]\d{3}[-. ]\d{4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
)
PASSPORT_RE = re.compile(r"\b[A-Z][0-9]{6,9}\b")

_KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (category, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
    for category, phrase in (
        ("confidential", "confidential"),
        ("classified", "classified"),
        ("top_secret", "top secret"),
        ("secret_key", "secret key"),
        ("api_key", "api key"),
        ("api_key", "api_key"),
        ("auth_token", "auth token"),
        ("bearer_token", "bearer token"),
        ("private_key", "private key"),
        ("credentials", "credentials"),
        ("credential", "credential"),
        ("password", "password"),
        ("passwd", "passwd"),
        ("full_name", "full name"),
        ("home_address", "home address"),
        ("mailing_address", "mailing address"),
        ("street_address", "street address"),
        ("date_of_birth", "date of birth"),
        ("national_id", "national id"),
        ("tax_id", "tax id"),
        ("tax_identification", "tax identification"),
        ("social_security", "social security"),
        ("credit_card", "credit card"),
        ("card_number", "card number"),
        ("bank_account", "bank account"),
        ("routing_number", "routing number"),
        ("medical_record", "medical record"),
        ("health_record", "health record"),
        ("health_information", "health information"),
        ("patient_data", "patient data"),
        ("personal_data", "personal data"),
        ("personal_information", "personal information"),
        ("personally_identifiable", "personally identifiable"),
        ("ssn", "ssn"),
        ("unsafe_content", "death threat"),
        ("unsafe_content", "bomb threat"),
        ("unsafe_content", "shooting threat"),
        ("unsafe_content", "terrorist attack"),
        ("unsafe_content", "sexual abuse"),
    )
)

_PATTERN_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", EMAIL_RE),
    ("phone", PHONE_RE),
    ("social_security_number", SSN_RE),
    ("credit_card", CREDIT_CARD_RE),
    ("passport", PASSPORT_RE),
)

_MAX_FINDINGS_PER_FILE = 100


@dataclass(frozen=True)
class PrivacyFinding:
    doc_id: str
    source_path: str
    category: str
    line: int


@dataclass(frozen=True)
class PrivacyAuditResult:
    clean: bool
    findings: tuple[PrivacyFinding, ...]
    excluded_document_ids: frozenset[str]


def scan_text(doc_id: str, source_path: str, text: str) -> tuple[PrivacyFinding, ...]:
    findings: list[PrivacyFinding] = []
    seen: set[tuple[str, int]] = set()

    def add(category: str, match: re.Match[str]) -> None:
        if len(findings) >= _MAX_FINDINGS_PER_FILE:
            return
        line = text.count("\n", 0, match.start()) + 1
        if (category, line) in seen:
            return
        seen.add((category, line))
        findings.append(
            PrivacyFinding(
                doc_id=doc_id, source_path=source_path, category=category, line=line
            )
        )

    for category, pattern in _PATTERN_CATEGORIES:
        for match in pattern.finditer(text):
            add(category, match)
    for category, pattern in _KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            add(category, match)
    return tuple(findings)


def audit_privacy_texts(
    documents: list[tuple[str, str, str]],
) -> PrivacyAuditResult:
    findings: list[PrivacyFinding] = []
    excluded: set[str] = set()
    for doc_id, source_path, text in documents:
        file_findings = scan_text(doc_id, source_path, text)
        findings.extend(file_findings)
        if file_findings:
            excluded.add(doc_id)
    return PrivacyAuditResult(
        clean=not findings,
        findings=tuple(findings),
        excluded_document_ids=frozenset(excluded),
    )
