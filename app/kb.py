"""
kb.py – Load, validate and look up the knowledge base (data/knowledge.json).

Schema per entry:
  id            str   unique slug
  name_en       str
  name_ta       str
  what_it_is    str
  who_can_apply str
  documents     list[str]
  where_to_apply str
  steps         list[str]
  official_site str   must be https:// on an allow-listed gov domain
  last_checked  str   ISO date
  verified      bool  only the human sets true

kb.lookup(query_text) returns the best-matching entry or None.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import KNOWLEDGE_FILE


# ── Allowed government domains ────────────────────────────────────────────────

ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "passportindia.gov.in",
        "parivahan.gov.in",
        "uidai.gov.in",
        "incometax.gov.in",
        "onlineservices.ec.in",        # Voter ID / ECI
        "nvsp.in",
        "eci.gov.in",
        "tnpds.gov.in",                # TN ration card
        "epds.tn.gov.in",
        "tnrd.gov.in",
        "cms.tn.gov.in",
        "tn.gov.in",
        "socialsecurity.tn.gov.in",    # TN pension schemes
        "pmkisan.gov.in",              # PM-KISAN
        "pmsby.gov.in",
        "pmjay.gov.in",                # Ayushman Bharat
        "nha.gov.in",
        "scholarships.gov.in",
        "india.gov.in",
        "digilocker.gov.in",
        "mha.gov.in",
        "bvialo.gov.in",
        "mohua.gov.in",
        "labour.gov.in",
        "epfindia.gov.in",
    }
)

REQUIRED_FIELDS: list[str] = [
    "id",
    "name_en",
    "name_ta",
    "what_it_is",
    "who_can_apply",
    "documents",
    "where_to_apply",
    "steps",
    "official_site",
    "last_checked",
    "verified",
]

# Currency amount detection: reject ₹, explicit rupee amounts, or ₹-adjacent digits.
# The word "fee" or "charge" alone (without an amount) is acceptable in process steps.
_CURRENCY_RE = re.compile(
    r"₹\s*[\d,]+|[\d,]+\s*(?:rs|inr|rupees?)\b|\brupees?\s*[\d,]+",
    re.IGNORECASE,
)


# ── Validation ────────────────────────────────────────────────────────────────


class KBError(Exception):
    pass


def _check_domain(url: str) -> bool:
    m = re.match(r"https?://([^/]+)", url)
    if not m:
        return False
    host = m.group(1).lower()
    for domain in ALLOWED_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
    return False


def validate(entries: list[dict[str, Any]], *, strict: bool = False) -> list[str]:
    """
    Validate a list of KB entries.  Returns a list of warning strings.
    Raises KBError on hard errors.  With strict=True, also raises if any
    entry has verified=False.
    """
    warnings: list[str] = []
    ids_seen: set[str] = set()

    for i, entry in enumerate(entries):
        ref = f"Entry #{i} (id={entry.get('id', '?')!r})"

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in entry:
                raise KBError(f"{ref}: missing required field {field!r}")

        # Unique IDs
        eid = entry["id"]
        if eid in ids_seen:
            raise KBError(f"{ref}: duplicate id {eid!r}")
        ids_seen.add(eid)

        # official_site must be https:// on an allowed domain
        site = entry["official_site"]
        if not site.startswith("https://"):
            raise KBError(f"{ref}: official_site must start with https://")
        if not _check_domain(site):
            raise KBError(
                f"{ref}: official_site domain not in allowed list: {site}"
            )

        # No currency amounts
        for field in ("what_it_is", "who_can_apply", "where_to_apply"):
            val = entry.get(field, "")
            if isinstance(val, str) and _CURRENCY_RE.search(val):
                raise KBError(f"{ref}: currency/fee mention in field {field!r}")
        for doc in entry.get("documents", []):
            if _CURRENCY_RE.search(str(doc)):
                raise KBError(f"{ref}: currency/fee mention in documents list")
        for step in entry.get("steps", []):
            if _CURRENCY_RE.search(str(step)):
                raise KBError(f"{ref}: currency/fee mention in steps list")

        # Unverified warning
        if not entry.get("verified", False):
            warnings.append(f"{ref}: verified=false (human must review)")
            if strict:
                raise KBError(f"{ref}: strict mode – verified=false entries not allowed")

    return warnings


# ── Loading ───────────────────────────────────────────────────────────────────

_kb_cache: list[dict[str, Any]] | None = None


def load(path: Path | None = None, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load and validate knowledge.json.  Caches after first load."""
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache

    p = path or KNOWLEDGE_FILE
    if not p.exists():
        raise KBError(f"knowledge.json not found at {p}")

    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise KBError("knowledge.json must be a JSON array of entries")

    warnings = validate(raw, strict=strict)
    for w in warnings:
        import logging
        logging.getLogger(__name__).warning(w)

    _kb_cache = raw
    return _kb_cache


def invalidate_cache() -> None:
    """Force a reload on the next call to load()."""
    global _kb_cache
    _kb_cache = None


# ── Lookup ────────────────────────────────────────────────────────────────────

_NAME_FIELDS = ("name_ta", "name_en", "id")
_DESC_FIELDS = ("what_it_is", "who_can_apply")


def lookup(query: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """
    Keyword match: score each entry by how many query words appear in its text fields.
    Preserves Tamil Unicode characters (including combining virama/vowels).
    Returns the highest-scoring entry or None if no match.
    """
    if entries is None:
        entries = load()

    # Split on whitespace/punctuation without breaking Tamil Unicode graphemes
    q_words = set(re.findall(r"[^\s.,!?।\"'():;]+", query.lower()))
    # Filter out single-character punctuation artifacts
    q_words = {w for w in q_words if len(w) > 1}
    if not q_words:
        return None

    best_entry = None
    best_score = 0.0

    for entry in entries:
        score = 0.0
        # High weight for service names and ID
        for field in _NAME_FIELDS:
            val = str(entry.get(field, "")).lower()
            for w in q_words:
                if w in val:
                    score += 5.0

        # Medium weight for descriptions
        for field in _DESC_FIELDS:
            val = str(entry.get(field, "")).lower()
            for w in q_words:
                if w in val:
                    score += 2.0

        # Lower weight for documents and steps
        for item in entry.get("documents", []) + entry.get("steps", []):
            val = str(item).lower()
            for w in q_words:
                if w in val:
                    score += 0.5

        if score > best_score:
            best_score = score
            best_entry = entry

    # Require score threshold to avoid spurious results
    return best_entry if best_score >= 2.0 else None


def get_by_id(service_id: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return the entry with the given id, or None."""
    if entries is None:
        entries = load()
    for entry in entries:
        if entry.get("id") == service_id:
            return entry
    return None
