"""
stt.py – Gnani Prisma STT streaming wrapper using gnani-vachana SDK.

Real mode:  connects to Prisma via GnaniSTTStreamClient, forwards PCM16 frames,
            yields STTEvent objects (partial / final / error).
Mock mode:  reads from pcm_queue and yields a canned Tamil transcript event.

Public API:
    async for event in stream(pcm_queue: asyncio.Queue[bytes | None]):
        # event.type in {"partial", "final", "error"}
        # event.text: str
        # event.segment_id: str
        # event.latency_ms: float | None

    None in pcm_queue signals end of stream.

Raises GnaniError on hard failures (after one retry on transient errors).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.config import GNANI_API_KEY, MOCK, STT_LANGUAGE

log = logging.getLogger(__name__)


# ── Error types ───────────────────────────────────────────────────────────────


class GnaniError(Exception):
    """Hard failure from a Gnani API call."""


# ── Event type ────────────────────────────────────────────────────────────────


@dataclass
class STTEvent:
    type: str           # "partial" | "final" | "error"
    text: str = ""
    segment_id: str = ""
    latency_ms: float | None = None
    raw: dict = field(default_factory=dict)


# ── Mock mode ─────────────────────────────────────────────────────────────────

_MOCK_RESPONSE = "பாஸ்போர்ட் எப்படி apply பண்றது?"


async def _mock_stream(pcm_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
    """Drain the queue and yield one final transcript event."""
    while True:
        chunk = await pcm_queue.get()
        if chunk is None:
            break
    yield STTEvent(type="partial", text=_MOCK_RESPONSE, segment_id="mock-1")
    await asyncio.sleep(0.1)
    yield STTEvent(type="final", text=_MOCK_RESPONSE, segment_id="mock-1", latency_ms=120.0)


# ── Real streaming mode (using official SDK) ──────────────────────────────────


async def _real_stream(pcm_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
    """Connect to Prisma WebSocket stream using GnaniSTTStreamClient and yield events."""
    try:
        from gnani.stt import (
            GnaniSTTStreamClient,
            StreamErrorEvent,
            StreamProcessingEvent,
            StreamTranscriptEvent,
        )
    except ImportError as exc:
        raise GnaniError("gnani-vachana package not installed") from exc

    try:
        client = GnaniSTTStreamClient(
            api_key=GNANI_API_KEY,
            language_code=STT_LANGUAGE,
            sample_rate=16000,
        )
        await client.connect()

        async def _sender():
            while True:
                chunk = await pcm_queue.get()
                if chunk is None:
                    break
                try:
                    await client.send_audio(chunk)
                except Exception as exc:
                    log.warning("STT send error: %s", exc)
                    break

        send_task = asyncio.create_task(_sender())

        try:
            async for event in client.events():
                if isinstance(event, StreamTranscriptEvent):
                    yield STTEvent(
                        type="final",
                        text=event.text,
                        segment_id=str(getattr(event, "segment_id", event.segment_index)),
                        latency_ms=getattr(event, "latency", None),
                        raw=getattr(event, "raw", {}),
                    )
                elif isinstance(event, StreamProcessingEvent):
                    # Interim processing event from VAD
                    yield STTEvent(type="partial", text="")
                elif isinstance(event, StreamErrorEvent):
                    yield STTEvent(type="error", text=event.message)
                    break
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await client.close()

    except Exception as exc:
        raise GnaniError(f"Prisma STT stream error: {exc}") from exc


# ── Public API ────────────────────────────────────────────────────────────────


async def stream(pcm_queue: asyncio.Queue) -> AsyncIterator[STTEvent]:
    """
    Yield STTEvent objects from Prisma for audio in *pcm_queue*.
    One automatic retry on transient errors.
    """
    if MOCK:
        async for evt in _mock_stream(pcm_queue):
            yield evt
        return

    for attempt in range(2):
        try:
            async for evt in _real_stream(pcm_queue):
                yield evt
            return
        except GnaniError as exc:
            if attempt == 0:
                log.warning("Prisma transient error (retrying): %s", exc)
                await asyncio.sleep(1.0)
            else:
                raise


