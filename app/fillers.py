"""
fillers.py – Pre-synthesised short filler audio clips.

Fillers are played a fraction of a second after the user finishes speaking,
while the LLM is thinking, to avoid dead air.  Clips are synthesised once per
language at startup and cached as bytes.

Public API:
    await preload()                       – synthesise all clips (startup)
    await get_filler_audio(language)      – next (audio_bytes, caption_text)
    rotate_filler()                       – compatibility: next Tamil clip
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Iterable

from app.tags import Segment
import app.tts as tts_module

log = logging.getLogger(__name__)

# Filler texts – short, human-sounding acknowledgements.  Each is prefixed with
# a leading <pauses> tag by the synthesiser so it sounds natural.
FILLER_TEXTS: dict[str, list[str]] = {
    "ta-IN": ["ஒரு நிமிஷம்...", "ஆமா...", "hmm...", "சரி..."],
    "hi-IN": ["एक मिनट...", "हाँ...", "ठीक है..."],
    "te-IN": ["ఒక నిమిషం...", "అవును...", "సరే..."],
    "ml-IN": ["ഒരു മിനിറ്റ്...", "അതെ...", "ശരി..."],
    "kn-IN": ["ಒಂದು ನಿಮಿಷ...", "ಹೌದು...", "ಸರಿ..."],
}

# language -> list of (audio_bytes, spoken_text)
_cache: dict[str, list[tuple[bytes, str]]] = {}
_cycles: dict[str, itertools.cycle] = {}
# A module-global cache must be safe across successive event loops (tests,
# reloads).  asyncio.Lock objects bind to the loop that first uses them, so keep
# one lock per loop instead of a single global lock.
_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
_MAX_CONCURRENT_SYNTH = 4


def _get_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _locks[loop] = lock
    return lock


async def _synthesize_one(
    language: str, text: str, sem: asyncio.Semaphore
) -> tuple[bytes, str]:
    async with sem:
        seg = Segment(tag="<pauses>", text=text)
        try:
            audio = await tts_module.synthesize(seg, language=language)
        except Exception as exc:  # noqa: BLE001 - a failed clip must not kill startup
            log.warning("Could not synthesise %s filler %r: %s", language, text, exc)
            audio = b""
        return audio, text


async def preload(languages: Iterable[str] | None = None) -> None:
    """
    Synthesise filler clips for the given languages (all known by default).

    Safe to call repeatedly; already-loaded languages are skipped.  A language
    whose clips all fail is cached as empty so we do not retry on every turn.
    """
    langs = list(languages) if languages is not None else list(FILLER_TEXTS)

    async with _get_lock():
        todo_langs = [lang for lang in langs if lang not in _cache]
        if not todo_langs:
            return

        log.info("Pre-synthesising filler clips for %s…", ", ".join(todo_langs))
        sem = asyncio.Semaphore(_MAX_CONCURRENT_SYNTH)
        jobs: list[tuple[str, str]] = []
        for lang in todo_langs:
            for text in FILLER_TEXTS.get(lang, []):
                jobs.append((lang, text))

        results = await asyncio.gather(
            *(_synthesize_one(lang, text, sem) for lang, text in jobs)
        )

        by_lang: dict[str, list[tuple[bytes, str]]] = {lang: [] for lang in todo_langs}
        for (lang, _text), result in zip(jobs, results):
            by_lang[lang].append(result)

        for lang in todo_langs:
            clips = [(audio, text) for audio, text in by_lang[lang] if audio]
            _cache[lang] = clips
            if clips:
                _cycles[lang] = itertools.cycle(clips)
                log.info("Filler clips ready for %s (%d).", lang, len(clips))
            else:
                log.warning("No filler clips available for %s.", lang)

        log.info("Filler preload complete.")


async def get_filler_audio(language: str = "ta-IN") -> tuple[bytes, str]:
    """
    Return the next filler clip for *language* as ``(audio, spoken_text)``.

    The first call for a language lazily synthesises its clips if startup
    preload did not already cover it.
    """
    if language not in _cache:
        await preload([language])

    cycle = _cycles.get(language)
    if cycle is None:
        return b"", ""
    return next(cycle)


def get_oru_nimisham_audio() -> bytes:
    """Compatibility helper: return the Tamil 'ஒரு நிமிஷம்...' clip."""
    clips = _cache.get("ta-IN") or []
    return clips[0][0] if clips else b""


def rotate_filler() -> bytes:
    """Compatibility helper: return the next Tamil filler audio bytes."""
    cycle = _cycles.get("ta-IN")
    if cycle is None:
        return b""
    return next(cycle)[0]
