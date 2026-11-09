"""
llm.py – Evon LLM client (OpenAI-compatible, streaming).

Real mode:  calls EVON_BASE_URL with the openai SDK.
Mock mode:  yields a canned Tamil token stream (no network).

Public API:
    async for token in stream(messages):
        ...

Raises EvonError on hard failures (after one retry on transient errors).
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

from app.config import (
    EVON_API_KEY,
    EVON_BASE_URL,
    EVON_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    MOCK,
)

log = logging.getLogger(__name__)

# ── Error types ───────────────────────────────────────────────────────────────


class EvonError(Exception):
    """Hard failure from the Evon endpoint."""


# ── Mock response ─────────────────────────────────────────────────────────────

def _make_mock_tokens(messages: list[dict]) -> list[str]:
    """Generate contextual mock tokens based on the last user message."""
    from app import kb as kb_module

    # Get the last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    query_lower = last_user_msg.lower()

    # Check for prompt injection
    if any(x in query_lower for x in ["ignore all", "system prompt", "instructions"]):
        return [
            "@@service=none@@ ",
            "<pauses> ",
            "மன்னிக்கவும், நான் அரசு சேவைகள் பற்றி மட்டுமே பதில் சொல்ல முடியும். ",
            "வேறு என்ன உதவி வேண்டும்?",
        ]

    # Check for visa question
    if "visa" in query_lower:
        return [
            "@@service=none@@ ",
            "<pauses> ",
            "வீசா பற்றி எனக்கு விவரங்கள் இல்லை. ",
            "அந்த நாட்டின் embassy website check பண்ணுங்க.",
        ]

    # Check for fees
    if any(x in query_lower for x in ["fee", "கட்டணம்", "evvalavu", "how much"]):
        entry = kb_module.lookup(last_user_msg)
        sid = entry["id"] if entry else "none"
        return [
            f"@@service={sid}@@ ",
            "<pauses> ",
            "கட்டண விவரங்களுக்கு official website check பண்ணுங்க.",
        ]

    # Check for off-topic
    if any(x in query_lower for x in ["recipe", "sambar", "cricket", "movie"]):
        return [
            "@@service=none@@ ",
            "<pauses> ",
            "நான் அரசு சேவைகள் பற்றி மட்டுமே தகவல் தருகிறேன்.",
        ]

    # Normal KB lookup
    entry = kb_module.lookup(last_user_msg)

    # In multi-turn context: if query has no strong match or just asks for docs/clarification,
    # inherit the service_id from the preceding assistant message
    if not entry or any(w in query_lower for w in ["acceptable", "enna venum", "documents"]):
        for msg in reversed(messages[:-1]):
            if msg.get("role") == "assistant":
                m = re.search(r"@@service=([^@]*)@@", msg.get("content", ""))
                if m and m.group(1).strip() != "none":
                    inherited = kb_module.get_by_id(m.group(1).strip())
                    if inherited:
                        entry = inherited
                        break
    if entry:
        sid = entry["id"]
        name = entry.get("name_ta", entry.get("name_en", ""))
        docs = ", ".join(entry.get("documents", [])[:2])
        return [
            f"@@service={sid}@@ ",
            "<pauses> ",
            f"{name} பத்தி சொல்றேன். ",
            "<slow> ",
            f"முக்கியமா {docs} வேணும்.",
        ]

    return [
        "@@service=none@@ ",
        "<pauses> ",
        "நான் உங்களுக்கு help பண்ண ready. ",
        "என்ன service வேணும்?",
    ]


async def _mock_stream(messages: list[dict]) -> AsyncIterator[str]:
    tokens = _make_mock_tokens(messages)
    for tok in tokens:
        await asyncio.sleep(0.01)  # simulate ~10 ms per token
        yield tok


# ── Service-marker parsing helpers ───────────────────────────────────────────

_MARKER_RE = re.compile(r"@@service=([^@]+)@@")


def extract_service_marker(text: str) -> tuple[str | None, str]:
    """
    If *text* starts with a @@service=...@@ marker, return (service_id, rest).
    Otherwise return (None, text).
    """
    m = _MARKER_RE.match(text.lstrip())
    if m:
        sid = m.group(1).strip()
        rest = text[text.index(m.group(0)) + len(m.group(0)):]
        return sid, rest.lstrip()
    return None, text


# ── Real streaming client ─────────────────────────────────────────────────────

_evon_client = None
_groq_client = None


def _get_evon_client():
    global _evon_client
    if _evon_client is None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as e:
            raise EvonError("openai package not installed") from e
        _evon_client = AsyncOpenAI(
            base_url=EVON_BASE_URL,
            api_key=EVON_API_KEY,
            timeout=2.0,
            max_retries=0,
        )
    return _evon_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as e:
            raise EvonError("openai package not installed") from e
        if not GROQ_API_KEY:
            return None
        _groq_client = AsyncOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
            timeout=15.0,
            max_retries=2,
        )
    return _groq_client


async def _evon_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Stream tokens from primary Evon endpoint."""
    client = _get_evon_client()
    try:
        response = await client.chat.completions.create(
            model=EVON_MODEL,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as exc:
        raise EvonError(f"Evon streaming error: {exc}") from exc


async def _groq_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Fallback stream from Groq."""
    client = _get_groq_client()
    if client is None:
        raise EvonError("Groq client not configured (GROQ_API_KEY missing)")
    try:
        log.info("Using Groq fallback model: %s", GROQ_MODEL)
        # openai/gpt-oss-20b is a reasoning model; set reasoning_effort='low'
        # and provide sufficient tokens so reasoning does not exhaust the token budget.
        kwargs: dict = {
            "model": GROQ_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": max(LLM_MAX_TOKENS, 600),
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        if "gpt-oss" in GROQ_MODEL or "20b" in GROQ_MODEL:
            kwargs["reasoning_effort"] = "low"

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as exc:
        raise EvonError(f"Groq fallback streaming error: {exc}") from exc


# ── Public API ────────────────────────────────────────────────────────────────


async def stream(messages: list[dict]) -> AsyncIterator[str]:
    """
    Yield LLM tokens for the given message history.
    Tries Evon first; falls back to Groq if Evon fails or is unreachable.
    """
    if MOCK:
        async for tok in _mock_stream(messages):
            yield tok
        return

    # Attempt 1: Evon LLM
    try:
        async for tok in _evon_stream(messages):
            yield tok
        return
    except Exception as evon_err:
        log.warning("Evon primary endpoint failed: %s", evon_err)

    # Attempt 2: Groq Fallback
    try:
        async for tok in _groq_stream(messages):
            yield tok
        return
    except Exception as groq_err:
        log.error("Groq fallback also failed: %s", groq_err)
        raise EvonError(f"All LLM providers failed. Evon error, Groq error: {groq_err}")

