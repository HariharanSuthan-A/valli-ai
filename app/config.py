"""
config.py – All settings loaded from .env.
Never log the values of secret fields.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of app/)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Required env var {name!r} is not set. See .env.example.")
    return val


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default)


# ── Gnani API ──────────────────────────────────────────────────────────────────
GNANI_API_KEY: str = _optional("GNANI_API_KEY", "")

# ── STT ───────────────────────────────────────────────────────────────────────
STT_LANGUAGE: str = _optional("STT_LANGUAGE", "ta-IN")
STT_STREAM_URL: str = _optional(
    "STT_STREAM_URL", "wss://api.vachana.ai/stt/v3/stream"
)
STT_REST_URL: str = _optional("STT_REST_URL", "https://api.vachana.ai/stt/v3")

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_MODEL: str = _optional("TTS_MODEL", "timbre-v3.0")
TTS_VOICE: str = _optional("TTS_VOICE", "Lakshmi")
TTS_SAMPLE_RATE: int = int(_optional("TTS_SAMPLE_RATE", "16000"))
TTS_SSE_URL: str = _optional(
    "TTS_SSE_URL", "https://api.vachana.ai/api/v1/tts/sse"
)
TTS_REST_URL: str = _optional(
    "TTS_REST_URL", "https://api.vachana.ai/api/v1/tts/inference"
)
TTS_WS_URL: str = _optional(
    "TTS_WS_URL", "wss://api.vachana.ai/api/v1/tts"
)

# ── LLM (Evon, OpenAI-compatible) ─────────────────────────────────────────────
EVON_BASE_URL: str = _optional("EVON_BASE_URL", "http://localhost:8001/v1")
EVON_MODEL: str = _optional("EVON_MODEL", "gnani/gnani-evon-v3.3-30B-A3B")
EVON_API_KEY: str = _optional("EVON_API_KEY", "not-needed-for-local")

# ── LLM Fallback (Groq) ───────────────────────────────────────────────────────
GROQ_API_KEY: str = _optional("GROQ_API_KEY", "")
GROQ_MODEL: str = _optional("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL: str = _optional("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# ── Feature flags ─────────────────────────────────────────────────────────────
MOCK: bool = _optional("MOCK", "0").strip() == "1"

# ── Tuning knobs ──────────────────────────────────────────────────────────────
LLM_MAX_TOKENS: int = int(_optional("LLM_MAX_TOKENS", "220"))
LLM_TEMPERATURE: float = float(_optional("LLM_TEMPERATURE", "0.6"))
LLM_TOP_P: float = float(_optional("LLM_TOP_P", "0.95"))
HISTORY_TURNS: int = int(_optional("HISTORY_TURNS", "8"))  # last N turns kept

# Filler timeout: if real answer starts within this window, skip filler (seconds)
FILLER_GRACE_S: float = float(_optional("FILLER_GRACE_S", "0.4"))

# Silence timeouts for guardrails (seconds): wait 45s for user to talk, then say "Hello?"
SILENCE_WARN_S: float = float(_optional("SILENCE_WARN_S", "45.0"))
SILENCE_END_S: float = float(_optional("SILENCE_END_S", "30.0"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _root / "data"
STATIC_DIR: Path = _root / "static"
KNOWLEDGE_FILE: Path = DATA_DIR / "knowledge.json"

