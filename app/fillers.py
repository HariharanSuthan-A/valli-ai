"""
fillers.py – Pre-synthesised short filler audio clips.

Fillers are played immediately (~400 ms) after the user finishes speaking,
while the LLM is thinking, to avoid dead air.  Clips are synthesised at
startup and cached as bytes.

rotate_filler() returns the next filler's audio bytes in round-robin order.
"""
from __future__ import annotations

import asyncio
import itertools
import logging

from app.tags import Segment
import app.tts as tts_module

log = logging.getLogger(__name__)

# Filler texts – short, human-sounding acknowledgements in Tamil/Tanglish.
# Each has a leading <pauses> tag so it sounds natural before the response.
_FILLER_TEXTS: list[str] = [
    "<pauses> ஒரு நிமிஷம்...",
    "<pauses> ஆமா...",
    "<pauses> hmm...",
    "<pauses> சரி...",
]

_oru_nimisham_audio: bytes = b""
_cache: list[bytes] = []
_cycle: itertools.cycle | None = None
_lock = asyncio.Lock()


async def preload() -> None:
    """
    Synthesise all filler clips at startup.  Call once from main.py lifespan.
    Safe to call multiple times (no-op if already loaded).
    """
    global _cache, _cycle, _oru_nimisham_audio
    async with _lock:
        if _cache:
            return  # already loaded

        log.info("Pre-synthesising %d filler clips…", len(_FILLER_TEXTS))
        clips: list[bytes] = []
        for text in _FILLER_TEXTS:
            clean_text = text.replace("<pauses> ", "").strip()
            seg = Segment(tag="<pauses>", text=clean_text)
            try:
                audio = await tts_module.synthesize(seg)
                clips.append(audio)
                if "ஒரு நிமிஷம்" in clean_text and not _oru_nimisham_audio:
                    _oru_nimisham_audio = audio
            except Exception as exc:
                log.warning("Could not synthesise filler %r: %s", text, exc)
                # Push an empty placeholder so the rotation stays consistent
                clips.append(b"")

        _cache = clips
        _cycle = itertools.cycle(_cache)
        log.info("Filler clips ready.")


def get_oru_nimisham_audio() -> bytes:
    """
    Return the pre-synthesised audio bytes for 'ஒரு நிமிஷம்...'.
    Falls back to rotate_filler() if not yet loaded.
    """
    if _oru_nimisham_audio:
        return _oru_nimisham_audio
    return rotate_filler()


def rotate_filler() -> bytes:
    """
    Return the next filler audio bytes in rotation.
    Returns empty bytes if fillers have not been pre-loaded.
    """
    if _cycle is None:
        return b""
    return next(_cycle)

