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


# Prisma streaming expects fixed-size frames: 512 PCM16 samples = 1024 bytes
# (32 ms at 16 kHz).  Older clients send 20 ms chunks, so re-frame defensively.
STREAM_FRAME_BYTES = 1024


async def _real_stream(
    pcm_queue: asyncio.Queue, language: str = STT_LANGUAGE
) -> AsyncIterator[STTEvent]:
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

    client = GnaniSTTStreamClient(
        api_key=GNANI_API_KEY,
        language_code=language,
        sample_rate=16000,
    )
    try:
        await client.connect()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise GnaniError(f"Prisma STT connect error: {exc}") from exc

    send_error: list[BaseException] = []

    async def _sender() -> None:
        """Forward mic PCM to Prisma in exact 1024-byte frames."""
        buf = bytearray()
        try:
            while True:
                chunk = await pcm_queue.get()
                if chunk is None:
                    break
                buf.extend(chunk)
                while len(buf) >= STREAM_FRAME_BYTES:
                    frame = bytes(buf[:STREAM_FRAME_BYTES])
                    del buf[:STREAM_FRAME_BYTES]
                    await client.send_audio(frame)

            # Flush a trailing partial frame (padded with silence) so the VAD
            # can finalise the last utterance instead of discarding it.
            if buf:
                buf.extend(b"\x00" * (STREAM_FRAME_BYTES - len(buf)))
                await client.send_audio(bytes(buf))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced after the event loop ends
            send_error.append(exc)
            log.warning("STT send error: %s", exc)
            try:
                await client.close()
            except Exception:
                pass

    send_task = asyncio.create_task(_sender(), name="stt-sender")

    try:
        async for event in client.events():
            if isinstance(event, StreamTranscriptEvent):
                segment_id = str(
                    getattr(event, "segment_id", "")
                    or getattr(event, "segment_index", "")
                )
                yield STTEvent(
                    type="final",
                    text=event.text,
                    segment_id=segment_id,
                    latency_ms=getattr(event, "latency", None),
                    raw=getattr(event, "raw", {}),
                )
            elif isinstance(event, StreamProcessingEvent):
                # VAD lifecycle / processing signal.  An empty partial is used
                # only to refresh the caller-activity/silence timer.
                yield STTEvent(type="partial", text="")
            elif isinstance(event, StreamErrorEvent):
                yield STTEvent(type="error", text=event.message)
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise GnaniError(f"Prisma STT stream error: {exc}") from exc
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        try:
            await client.close()
        except Exception:
            pass

    if send_error:
        raise GnaniError(
            f"Prisma STT send error: {send_error[0]}"
        ) from send_error[0]


# ── Public API ────────────────────────────────────────────────────────────────


async def stream(
    pcm_queue: asyncio.Queue, language: str = STT_LANGUAGE
) -> AsyncIterator[STTEvent]:
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
            async for evt in _real_stream(pcm_queue, language=language):
                yield evt
            return
        except GnaniError as exc:
            if attempt == 0:
                log.warning("Prisma transient error (retrying): %s", exc)
                await asyncio.sleep(1.0)
            else:
                raise


