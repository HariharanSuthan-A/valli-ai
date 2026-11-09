"""
test_kb.py – Unit tests for app/kb.py.

Tests:
  - Valid knowledge.json loads and validates
  - Missing required field fails
  - Duplicate id fails
  - Non-https official_site fails
  - Disallowed domain fails
  - Currency mention in fields fails
  - strict mode fails on verified=false
  - lookup() finds the right entry
  - get_by_id() returns the right entry
"""
import json
import pytest
from pathlib import Path
from app import kb as kb_module
from app.kb import KBError, validate, lookup, get_by_id


# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_ENTRY = {
    "id": "test_service",
    "name_en": "Test Service",
    "name_ta": "சோதனை சேவை",
    "what_it_is": "A test government service.",
    "who_can_apply": "Any Indian citizen.",
    "documents": ["Aadhaar card", "Proof of address"],
    "where_to_apply": "Online at india.gov.in",
    "steps": ["Step 1", "Step 2"],
    "official_site": "https://india.gov.in",
    "last_checked": "2026-10-01",
    "verified": False,
}


def make_entry(**overrides) -> dict:
    return {**VALID_ENTRY, **overrides}


# ── validate() ────────────────────────────────────────────────────────────────

def test_valid_entry_passes():
    warnings = validate([VALID_ENTRY])
    # Should warn about verified=false but not raise
    assert any("verified=false" in w for w in warnings)


def test_missing_required_field_raises():
    entry = {k: v for k, v in VALID_ENTRY.items() if k != "official_site"}
    with pytest.raises(KBError, match="missing required field"):
        validate([entry])


def test_duplicate_id_raises():
    with pytest.raises(KBError, match="duplicate id"):
        validate([VALID_ENTRY, VALID_ENTRY])


def test_non_https_site_raises():
    entry = make_entry(official_site="http://india.gov.in")
    with pytest.raises(KBError, match="must start with https://"):
        validate([entry])


def test_disallowed_domain_raises():
    entry = make_entry(official_site="https://example.com/passport")
    with pytest.raises(KBError, match="domain not in allowed list"):
        validate([entry])


def test_currency_in_what_it_is_raises():
    entry = make_entry(what_it_is="This service has a fee of ₹100.")
    with pytest.raises(KBError, match="currency/fee mention"):
        validate([entry])


def test_currency_in_documents_raises():
    entry = make_entry(documents=["ID proof", "Pay ₹200 stamp paper"])
    with pytest.raises(KBError, match="currency/fee mention"):
        validate([entry])


def test_strict_mode_fails_on_unverified():
    with pytest.raises(KBError, match="strict mode"):
        validate([VALID_ENTRY], strict=True)


def test_strict_mode_passes_when_verified():
    entry = make_entry(verified=True)
    warnings = validate([entry], strict=True)
    assert warnings == []


# ── load() ────────────────────────────────────────────────────────────────────

def test_load_real_knowledge_file():
    kb_module.invalidate_cache()
    entries = kb_module.load()
    assert isinstance(entries, list)
    assert len(entries) >= 8  # T4 requires at least 8-10 entries


def test_load_requires_list(tmp_path):
    kb_module.invalidate_cache()
    p = tmp_path / "kb.json"
    p.write_text('{"id": "not_a_list"}', encoding="utf-8")
    with pytest.raises(KBError, match="JSON array"):
        kb_module.load(path=p)


# ── lookup() ─────────────────────────────────────────────────────────────────

def test_lookup_passport():
    kb_module.invalidate_cache()
    result = lookup("passport apply")
    assert result is not None
    assert result["id"] == "passport"


def test_lookup_aadhaar():
    kb_module.invalidate_cache()
    result = lookup("aadhaar update address")
    assert result is not None
    assert result["id"] == "aadhaar_update"


def test_lookup_no_match_returns_none():
    kb_module.invalidate_cache()
    result = lookup("zzzyyyxxx_no_match_possible")
    assert result is None


def test_lookup_empty_query_returns_none():
    kb_module.invalidate_cache()
    result = lookup("")
    assert result is None


# ── get_by_id() ───────────────────────────────────────────────────────────────

def test_get_by_id_found():
    kb_module.invalidate_cache()
    entry = get_by_id("passport")
    assert entry is not None
    assert entry["id"] == "passport"


def test_get_by_id_not_found():
    kb_module.invalidate_cache()
    entry = get_by_id("nonexistent_id_12345")
    assert entry is None

