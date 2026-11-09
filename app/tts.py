"""
tts.py – Gnani Timbre TTS wrapper.

Supports:
  1. Realtime WebSocket (GnaniTTSRealtimeClient) for lowest latency streaming
  2. REST / SSE (POST /api/v1/tts/sse and /inference)
  3. Mock mode (silent WAV for unit tests)

Public API:
    async def synthesize(segment: Segment) -> bytes
        Returns raw audio bytes (WAV, 16 kHz mono).

Raises GnaniError on failure (one retry on transient errors).
"""
from __future__ import annotations

import asyncio
import io
import logging
import struct
import wave
from typing import Any

import httpx

from app.config import (
    GNANI_API_KEY,
    MOCK,
    TTS_MODEL,
    TTS_REST_URL,
    TTS_SAMPLE_RATE,
    TTS_VOICE,
)
from app.stt import GnaniError
from app.tags import Segment

log = logging.getLogger(__name__)

# Compatibility patch for gnani-vachana across websockets versions (<13, 13, 14+)
try:
    import inspect
    import websockets
    import gnani.tts.client

    def _robust_ws_header_kwargs(headers: dict[str, str]) -> dict[str, Any]:
        try:
            sig = inspect.signature(websockets.connect)
            if "additional_headers" in sig.parameters:
                return {"additional_headers": headers}
            if "extra_headers" in sig.parameters:
                return {"extra_headers": headers}
        except Exception:
            pass
        try:
            major = int(websockets.__version__.split(".", 1)[0])
            return {"additional_headers": headers} if major >= 14 else {"extra_headers": headers}
        except Exception:
            return {"additional_headers": headers}

    gnani.tts.client._ws_header_kwargs = _robust_ws_header_kwargs
except Exception:
    pass


# ── Silent WAV factory (for mock mode) ───────────────────────────────────────


def _silent_wav(duration_ms: int = 300, sample_rate: int = 16000) -> bytes:
    """Return a minimal silent WAV buffer."""
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


_SILENT_WAV = _silent_wav()


# ── Mock ──────────────────────────────────────────────────────────────────────


async def _mock_synthesize(segment: Segment) -> bytes:
    await asyncio.sleep(0.05)
    return _SILENT_WAV


# ── Text Builder ──────────────────────────────────────────────────────────────

MAX_TTS_PAYLOAD_CHARS: int = 780  # Gnani Timbre strictly requires <= 800 chars


def _build_tts_text(segment: Segment) -> str:
    """Compose the text to send to Timbre (preserving valid audio tags)."""
    text = segment["text"]
    tag = segment.get("tag")
    if tag:
        return f"{tag} {text}"
    return text


def _combine_wavs(wav_list: list[bytes]) -> bytes:
    """Combine multiple 16kHz WAV byte streams into a single seamless WAV byte stream."""
    valid_wavs = [w for w in wav_list if w and len(w) > 44]
    if not valid_wavs:
        return b""
    if len(valid_wavs) == 1:
        return valid_wavs[0]

    out_buf = io.BytesIO()
    out_wf = None
    for wb in valid_wavs:
        try:
            with wave.open(io.BytesIO(wb), "rb") as in_wf:
                if out_wf is None:
                    out_wf = wave.open(out_buf, "wb")
                    out_wf.setparams(in_wf.getparams())
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))
        except Exception as exc:
            log.warning("Error reading WAV chunk during combine: %s", exc)

    if out_wf is not None:
        out_wf.close()
        return out_buf.getvalue()
    return valid_wavs[0]


# ── Real TTS (SDK Realtime WebSocket preferred, fallback to REST/SSE) ────────


async def _realtime_ws_synthesize(segment: Segment, language: str = "ta-IN") -> bytes:
    """Use GnaniTTSRealtimeClient for lowest latency audio generation."""
    from gnani.tts import AudioConfig, GnaniTTSRealtimeClient

    tts_text = _build_tts_text(segment)
    cfg = AudioConfig(
        sample_rate=TTS_SAMPLE_RATE,
        encoding="linear_pcm",
        container="wav",
        num_channels=1,
        sample_width=2,
    )

    async with GnaniTTSRealtimeClient(api_key=GNANI_API_KEY) as client:
        return await client.synthesize_and_collect(
            tts_text,
            voice=TTS_VOICE,
            model=TTS_MODEL,
            language=language,
            audio_config=cfg,
        )


async def _rest_synthesize(
    segment: Segment, *, client: httpx.AsyncClient, language: str = "ta-IN"
) -> bytes:
    """Fallback REST/SSE endpoint."""
    tts_text = _build_tts_text(segment)

    payload: dict[str, Any] = {
        "text": tts_text,
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "language": language,
        "speed": 1,
        "audio_config": {
            "sample_rate": TTS_SAMPLE_RATE,
            "encoding": "linear_pcm",
            "container": "wav",
            "num_channels": 1,
            "sample_width": 2,
        },
    }

    headers = {
        "X-API-Key-ID": GNANI_API_KEY,
        "Content-Type": "application/json",
    }

    resp = await client.post(TTS_REST_URL, json=payload, headers=headers, timeout=15.0)

    if resp.status_code != 200:
        raise GnaniError(f"Timbre TTS error {resp.status_code}: {resp.text[:200]}")

    return resp.content


# ── Public API ────────────────────────────────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client


async def _synthesize_single(segment: Segment, language: str = "ta-IN") -> bytes:
    """Synthesize a single verified <= 800 char segment with retry and fallback."""
    for attempt in range(2):
        try:
            try:
                return await _realtime_ws_synthesize(segment, language=language)
            except Exception as ws_err:
                log.debug("Realtime WS TTS fallback to REST: %s", ws_err)
                client = _get_http_client()
                return await _rest_synthesize(segment, client=client, language=language)
        except Exception as exc:
            if attempt == 0:
                log.warning("Timbre transient error (retrying): %s", exc)
                await asyncio.sleep(0.5)
            else:
                raise GnaniError(f"Timbre synthesis failed: {exc}") from exc
    raise GnaniError("Timbre: unreachable")


async def synthesize(segment: Segment, language: str = "ta-IN") -> bytes:
    """
    Synthesize *segment* and return audio bytes (WAV).
    Enforces Gnani Timbre's <= 800 character limit per request.
    Oversized segments are automatically chunked and stitched.
    """
    if MOCK:
        return await _mock_synthesize(segment)

    tts_text = _build_tts_text(segment)
    if len(tts_text) > MAX_TTS_PAYLOAD_CHARS:
        log.info("Segment length %d exceeds max %d chars; chunking for Timbre TTS",
                 len(tts_text), MAX_TTS_PAYLOAD_CHARS)
        from app.tags import _chunk_long_text
        sub_texts = _chunk_long_text(segment["text"], max_chars=700)
        wav_chunks: list[bytes] = []
        for i, sub_text in enumerate(sub_texts):
            sub_seg: Segment = {
                "tag": segment.get("tag") if i == 0 else None,
                "text": sub_text,
            }
            wav_chunk = await _synthesize_single(sub_seg, language=language)
            wav_chunks.append(wav_chunk)
        return _combine_wavs(wav_chunks)

    return await _synthesize_single(segment, language=language)


async def close() -> None:
    """Close any shared HTTP client gracefully."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


