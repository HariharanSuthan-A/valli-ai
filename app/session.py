"""
session.py – CallSession: state machine and async task orchestration for one call.

State machine:
    idle -> listening -> thinking -> speaking -> listening (loop)
    any state -> ended (on disconnect or end-call)

One asyncio TaskGroup manages:
    1. STT task:   streams PCM to Prisma; fires on final transcript
    2. Turn task:  LLM + TTS pipeline; feeds audio back to browser
    3. Silence watcher: timeout guardrails

WebSocket protocol:
    Binary frames up: PCM16 mono 16 kHz
    JSON frames up:   {"type": "start"} | {"type": "mute", "on": bool} | {"type": "end"}
    JSON frames down: state, user_text, agent_text, service, stop_audio, error
    Binary frames down: audio chunks (preceded by audio metadata JSON)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any

from app import fillers, kb as kb_module
from app import llm as llm_module
from app import stt as stt_module
from app import tts as tts_module
from app.config import (
    FILLER_GRACE_S,
    HISTORY_TURNS,
    MOCK,
    SILENCE_END_S,
    SILENCE_WARN_S,
    STT_LANGUAGE,
)
from app.persona import build_system_prompt
from app.tags import Segment, TagBuffer, parse, strip

log = logging.getLogger(__name__)

# ── Greetings & Multilingual Defaults ──────────────────────────────────────────

GREETINGS: dict[str, str] = {
    "ta-IN": (
        "@@service=none@@ Vanakkam, Naan Valli pesuren. "
        "Government services pathi enna help venum?"
    ),
    "hi-IN": (
        "@@service=none@@ Namaste! Main Valli bol rahi hoon. "
        "Sarkari sevaon ke baare mein aapko kya madad chahiye?"
    ),
    "te-IN": (
        "@@service=none@@ Namaskaram! Nenu Valli matladuthunnanu. "
        "Prabhutva sevala gurinchi meeku emi sahayam kavali?"
    ),
    "ml-IN": (
        "@@service=none@@ Namaskaram! Njan Valli samsarikunnu. "
        "Sarkar sevanangale patti enthanu ariyendathu?"
    ),
    "kn-IN": (
        "@@service=none@@ Namaskara! Naanu Valli mathaduthiddene. "
        "Sarkari sevegalu bagge nimage enu sahaya beku?"
    ),
}

SILENCE_MESSAGES: dict[str, dict[str, str]] = {
    "ta-IN": {"warn": "Hello? Ungalukku ketkutha?", "end": "Sari, piragu pesalam. Nandri!"},
    "hi-IN": {"warn": "Hello? Kya aap mujhe sun sakte hain?", "end": "Theek hai, baad mein baat karte hain. Dhanyavad!"},
    "te-IN": {"warn": "Hello? Meeku vinipisthunda?", "end": "Sare, tarvata matladadam. Dhanyavadhalu!"},
    "ml-IN": {"warn": "Hello? Kelkkunnundo?", "end": "Sheri, pinne samsarikam. Nanni!"},
    "kn-IN": {"warn": "Hello? Kelusthidiya?", "end": "Sari, amele mathadona. Dhanyavada!"},
}

THANK_YOU_RESPONSES: dict[str, str] = {
    "ta-IN": "வள்ளியைத் தொடர்பு கொண்டதற்கு நன்றி, மேலும் விவரங்கள் தேவைப்பட்டால் என்னைத் தொடர்பு கொள்ளவும்.",
    "hi-IN": "वल्ली से संपर्क करने के लिए धन्यवाद, यदि आपको कोई अन्य विवरण चाहिए तो मुझसे संपर्क करें।",
    "te-IN": "వల్లిని సంప్రదించినందుకు ధన్యవాదాలు, మీకు ఏవైనా మరిన్ని వివరాలు కావాలంటే నన్ను సంప్రదించండి.",
    "ml-IN": "വള്ളിയെ ബന്ധപ്പെട്ടതിന് നന്ദി, കൂടുതൽ വിവരങ്ങൾ ആവശ്യമുണ്ടെങ്കിൽ എന്നെ ബന്ധപ്പെടുക.",
    "kn-IN": "ವಳ್ಳಿಯನ್ನು ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು, ಹೆಚ್ಚಿನ ವಿವರಗಳು ಬೇಕಾದರೆ ನನ್ನನ್ನು ಸಂಪರ್ಕಿಸಿ.",
}

# Spoken when the LLM fails before producing any answer (never leave silence).
LLM_ERROR_REPLIES: dict[str, str] = {
    "ta-IN": "மன்னிக்கவும், ஒரு சிறிய தொழில்நுட்ப சிக்கல். மீண்டும் கேட்கவும்.",
    "hi-IN": "क्षमा करें, थोड़ी तकनीकी समस्या है। कृपया फिर से पूछें।",
    "te-IN": "క్షమించండి, ఒక చిన్న సాంకేతిక సమస్య. దయచేసి మళ్లీ అడగండి.",
    "ml-IN": "ക്ഷമിക്കണം, ഒരു ചെറിയ സാങ്കേതിക പ്രശ്നം. വീണ്ടും ചോദിക്കുക.",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ಸಣ್ಣ ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಕೇಳಿ.",
}

def _is_thank_you_phrase(text: str) -> bool:
    cleaned = text.lower().strip()
    english_patterns = [
        r"\b(thank|thanks|thankyou|thx)\b",
        r"\b(nandri|dhanyavad|dhanyavaad|dhanyavadalu|nandi|dhanyavadagalu|shukriya)\b",
    ]
    if any(re.search(p, cleaned) for p in english_patterns):
        return True

    indic_keywords = [
        "நன்றி",
        "धन्यवाद", "शुक्रिया",
        "ధన్యవాదాలు", "ధన్యవాదములు", "థాంక్స్",
        "നന്ദി", "താങ്ക്സ്",
        "ಧನ್ಯವಾದಗಳು", "ಧನ್ಯವಾದ", "ಥ್ಯಾಂಕ್ಸ್",
    ]
    return any(w in cleaned for w in indic_keywords)

GREETING_TEXT = GREETINGS["ta-IN"]

# ── State enum ────────────────────────────────────────────────────────────────

class CallState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"


# ── Service marker regex ──────────────────────────────────────────────────────

_MARKER_RE = re.compile(r"@@\s*service\s*=\s*([^@]*)@@", re.IGNORECASE)

# Silence-watcher timing (module-level so tests can shorten the tick/pause).
_SILENCE_TICK_S = 1.0
_END_PAUSE_S = 3.5
_GOODBYE_WAIT_S = 6.0


def _extract_marker(text: str) -> tuple[str | None, str]:
    """
    Return (service_id, text_without_marker).

    The marker may appear anywhere in *text* (models occasionally emit it after
    a stray tag); every marker token is removed so it can never be spoken.
    ``service_id`` is None when no marker was present and ``"none"`` is mapped
    to None as well.
    """
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None, text
    sid = matches[0].group(1).strip()
    rest = _MARKER_RE.sub("", text).lstrip()
    return (None if sid.lower() == "none" else sid), rest


# ── CallSession ───────────────────────────────────────────────────────────────


class CallSession:
    """
    Manages the full lifecycle of a single phone call WebSocket session.

    Usage:
        session = CallSession(websocket)
        await session.run()
    """

    def __init__(self, websocket: Any) -> None:
        default_lang = STT_LANGUAGE if STT_LANGUAGE in GREETINGS else "ta-IN"
        self.ws = websocket
        self.state = CallState.IDLE
        self.muted = False
        self.language: str = default_lang

        # PCM audio queue fed by the WebSocket receiver and consumed by STT
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

        # Conversation history (alternating user/assistant, last HISTORY_TURNS*2)
        self._history: list[dict[str, str]] = []

        # System prompt (built for selected language)
        self._system_prompt: str = build_system_prompt(language=self.language)

        # Active service ID tracked across follow-up turns
        self._active_service_id: str | None = None

        # Flag: agent audio is currently being produced / played (blocks echo)
        self._agent_speaking = asyncio.Event()
        self._agent_speaking.clear()

        # Flag: client browser is currently playing audio chunks
        self._client_playing_audio: bool = False
        self._client_playing_since: float = 0.0

        # Cancellation token for the current turn pipeline
        self._turn_cancel: asyncio.Event = asyncio.Event()

        # Active turn task (so a barge-in can cancel it immediately)
        self._turn_task: asyncio.Task | None = None

        # Wake-up for the STT loop when the caller changes language mid-call
        self._stt_restart: asyncio.Event = asyncio.Event()
        self._stt_stream_task: asyncio.Task | None = None

        # Serialises JSON+binary audio frame pairs and barge-in handling
        self._audio_send_lock = asyncio.Lock()
        self._bargein_lock = asyncio.Lock()

        # Bumped on barge-in so stale TTS results are never sent after a flush
        self._audio_epoch: int = 0

        # Silence guardrails
        self._last_audio_ts: float = time.monotonic()
        self._silence_warned: bool = False

        # Per-stage timing log
        self._timings: dict[str, float] = {}

    def set_language(self, language: str) -> None:
        """Update active language for STT, TTS and LLM prompt."""
        if language not in GREETINGS:
            return
        changed = language != self.language
        self.language = language
        self._system_prompt = build_system_prompt(language=self.language)
        self._timings = {}
        log.info("CallSession language set to: %s", self.language)
        if changed:
            # Reconnect the STT stream with the new language on the next loop.
            self._stt_restart.set()
            task = self._stt_stream_task
            if task and not task.done():
                task.cancel()

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    async def _send_json(self, obj: dict) -> bool:
        try:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
            return True
        except Exception as exc:
            log.debug("WS send_json error: %s", exc)
            return False

    async def _send_binary(self, data: bytes) -> bool:
        try:
            await self.ws.send_bytes(data)
            return True
        except Exception as exc:
            log.debug("WS send_bytes error: %s", exc)
            return False

    async def _set_state(self, state: CallState) -> None:
        self.state = state
        if state == CallState.LISTENING:
            self._last_audio_ts = time.monotonic()
        await self._send_json({"type": "state", "value": state.value})

    # ── Audio output ──────────────────────────────────────────────────────────

    async def _send_audio(
        self, audio_bytes: bytes, text: str = "", sample_rate: int = 16000
    ) -> bool:
        """
        Send audio metadata (with paired transcript text) then the raw audio
        binary frame.  The lock guarantees that two concurrent speakers (a
        filler and the real answer) cannot interleave their metadata/binary
        frames and corrupt the browser's playback queue.
        """
        if not audio_bytes:
            return False
        async with self._audio_send_lock:
            ok = await self._send_json(
                {"type": "audio", "format": "wav", "rate": sample_rate, "text": text}
            )
            if not ok:
                return False
            ok = await self._send_binary(audio_bytes)
            if ok:
                self._client_playing_audio = True
                self._client_playing_since = time.monotonic()
            return ok

    # ── History management ────────────────────────────────────────────────────

    def _push_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        # Keep only the last HISTORY_TURNS*2 messages (each turn = user+assistant)
        max_msgs = HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _build_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            *self._history,
        ]

    # ── Greeting ──────────────────────────────────────────────────────────────

    async def _play_greeting(self) -> None:
        """Synthesise and stream the agent greeting (TTS failure is non-fatal)."""
        greeting_text = GREETINGS.get(self.language, GREETINGS["ta-IN"])
        _, text_no_marker = _extract_marker(greeting_text)
        segments = parse(text_no_marker)
        for seg in segments:
            if self._turn_cancel.is_set():
                break
            try:
                audio = await tts_module.synthesize(seg, language=self.language)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Greeting TTS failed: %s", exc)
                audio = b""
            if audio:
                # Block mic echo while the greeting is audibly playing; the
                # client's playback_done clears this flag.
                self._agent_speaking.set()
                await self._send_audio(audio, text=seg["text"])
        # Captions are shown even if audio synthesis failed.
        await self._send_json({"type": "agent_text", "text": strip(text_no_marker)})
        await self._send_json({"type": "audio_end"})

    # ── Filler ────────────────────────────────────────────────────────────────

    async def _play_filler(self, cancel_event: asyncio.Event) -> None:
        """
        Play a short localized acknowledgement if the real answer has not
        started within FILLER_GRACE_S seconds.  This keeps the call from
        feeling dead while the LLM is thinking.
        """
        try:
            await asyncio.sleep(FILLER_GRACE_S)
            if cancel_event.is_set() or self._turn_cancel.is_set():
                return

            audio, text = await fillers.get_filler_audio(self.language)
            if not audio or cancel_event.is_set() or self._turn_cancel.is_set():
                return
            if self.state in (CallState.IDLE, CallState.ENDED):
                return

            if self.state != CallState.SPEAKING:
                await self._set_state(CallState.SPEAKING)
            # Re-check after the await: the real answer may have arrived.
            if cancel_event.is_set():
                return

            self._agent_speaking.set()
            await self._send_audio(audio, text=text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Filler playback failed: %s", exc)

    # ── Fixed reply (thank-you / closing) ─────────────────────────────────────

    async def _speak_fixed_reply(self, reply_text: str) -> None:
        """Speak a short non-LLM reply (e.g. the thank-you closing)."""
        self._push_history("assistant", reply_text)
        try:
            if self.state != CallState.SPEAKING:
                await self._set_state(CallState.SPEAKING)
            self._agent_speaking.set()
            seg: Segment = {"tag": None, "text": reply_text}
            audio = await tts_module.synthesize(seg, language=self.language)
            if audio and not self._turn_cancel.is_set():
                await self._send_audio(audio, text=reply_text)
                await self._send_json({"type": "agent_text", "text": reply_text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Fixed reply TTS failed: %s", exc)
            await self._send_json({"type": "agent_text", "text": reply_text})
        finally:
            self._agent_speaking.clear()

        if not self._turn_cancel.is_set():
            await self._send_json({"type": "audio_end"})
            await self._set_state(CallState.LISTENING)

    # ── LLM + TTS pipeline for one turn ──────────────────────────────────────

    async def _run_turn(self, user_text: str) -> None:
        """
        Full pipeline: LLM stream -> sentence chunker -> TTS -> audio send.
        Respects self._turn_cancel for barge-in and always leaves the session
        in a usable state, even if the LLM/TTS provider fails mid-turn.
        """
        t_turn_start = time.monotonic()
        self._turn_cancel.clear()
        self._silence_warned = False

        # Push user message into history
        self._push_history("user", user_text)

        fallback_reply: str | None = None

        try:
            await self._set_state(CallState.THINKING)

            # Fast-path for caller thank you / gratitude / closing
            if _is_thank_you_phrase(user_text):
                reply_text = THANK_YOU_RESPONSES.get(
                    self.language, THANK_YOU_RESPONSES["ta-IN"]
                )
                await self._speak_fixed_reply(reply_text)
                return

            # Start filler; skip it automatically when the answer is fast.
            filler_cancel = asyncio.Event()
            filler_task = asyncio.create_task(
                self._play_filler(filler_cancel), name="call-filler"
            )

            # Collect the full LLM reply (streamed) while chunking into sentences
            full_reply_tokens: list[str] = []
            tag_buffer = TagBuffer()
            service_sent = False
            service_notified = False
            first_audio = False
            tag_budget = 2
            t_first_token: float | None = None

            async def _send_segment(seg_text: str) -> None:
                nonlocal first_audio, service_sent, service_notified, tag_budget
                if self._turn_cancel.is_set():
                    return

                # Pull the service marker out of whichever sentence carries it.
                sid, seg_text = _extract_marker(seg_text)
                if not service_sent:
                    service_sent = True
                    if sid is None:
                        matched = kb_module.lookup(user_text)
                        if matched:
                            sid = matched["id"]
                        elif self._active_service_id:
                            sid = self._active_service_id
                    if sid:
                        self._active_service_id = sid

                if sid and not service_notified:
                    service_notified = True
                    entry = kb_module.get_by_id(sid)
                    if entry:
                        await self._send_json({"type": "service", "entry": entry})

                if not seg_text.strip():
                    return

                segments = parse(seg_text, max_tags=tag_budget)
                tag_budget = max(
                    0, tag_budget - sum(1 for s in segments if s["tag"])
                )

                for seg in segments:
                    if self._turn_cancel.is_set():
                        return

                    if not first_audio:
                        filler_cancel.set()
                        if self.state != CallState.SPEAKING:
                            await self._set_state(CallState.SPEAKING)
                        first_audio = True
                        self._timings["first_audio"] = time.monotonic()
                        log.info(
                            "[timing] thinking→first_audio: %.3f s",
                            self._timings["first_audio"] - t_turn_start,
                        )

                    try:
                        audio = await tts_module.synthesize(
                            seg, language=self.language
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.warning("TTS failed for a segment: %s", exc)
                        audio = b""

                    if audio and not self._turn_cancel.is_set():
                        # Mark speaking only when audio is actually going out.
                        self._agent_speaking.set()
                        await self._send_audio(audio, text=seg["text"])
                        await self._send_json(
                            {"type": "agent_text", "text": seg["text"]}
                        )

            try:
                async for token in llm_module.stream(self._build_messages()):
                    if self._turn_cancel.is_set():
                        break
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                        log.info(
                            "[timing] thinking→first_token: %.3f s",
                            t_first_token - t_turn_start,
                        )

                    full_reply_tokens.append(token)

                    for sentence_text in tag_buffer.feed(token):
                        await _send_segment(sentence_text)

                # Flush remaining buffer
                for sentence_text in tag_buffer.flush():
                    await _send_segment(sentence_text)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("LLM error during turn: %s", exc)
                await self._send_json(
                    {"type": "error", "message": "Answer generation failed"}
                )
                # If nothing was spoken yet, apologise instead of dead air.
                if not full_reply_tokens and not first_audio:
                    fallback_reply = LLM_ERROR_REPLIES.get(
                        self.language, LLM_ERROR_REPLIES["ta-IN"]
                    )
            finally:
                filler_cancel.set()
                if not filler_task.done():
                    filler_task.cancel()
                try:
                    await filler_task
                except (asyncio.CancelledError, Exception):
                    pass

            full_reply = "".join(full_reply_tokens).strip()
            if full_reply and not self._turn_cancel.is_set():
                # Keep memory clean: no service markers or audio tags in history.
                _, clean_reply = _extract_marker(full_reply)
                clean_reply = strip(clean_reply).strip()
                if clean_reply:
                    self._push_history("assistant", clean_reply)

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # last-resort guard: never kill the call
            log.error("Unhandled turn error: %s", exc, exc_info=True)
            await self._send_json({"type": "error", "message": "Call turn failed"})
        finally:
            self._agent_speaking.clear()

        if self._turn_cancel.is_set():
            return

        if fallback_reply is not None:
            await self._speak_fixed_reply(fallback_reply)
            return

        # Notify browser that all audio segments for this turn have been sent
        await self._send_json({"type": "audio_end"})
        await self._set_state(CallState.LISTENING)
        log.info(
            "[timing] total turn: %.3f s",
            time.monotonic() - t_turn_start,
        )

    # ── Barge-in ──────────────────────────────────────────────────────────────

    async def _handle_bargein(self) -> None:
        """Cancel any in-flight turn immediately and return to listening."""
        async with self._bargein_lock:
            log.info("Barge-in detected; cancelling turn.")
            self._turn_cancel.set()
            self._silence_warned = False
            self._agent_speaking.clear()
            self._client_playing_audio = False
            self._client_playing_since = 0.0
            self._audio_epoch += 1

            # Flush the browser's audio queue first for an instant stop.
            await self._send_json({"type": "stop_audio"})

            task = self._turn_task
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._turn_task = None

            self._last_audio_ts = time.monotonic()
            await self._set_state(CallState.LISTENING)

    # ── Silence watcher ───────────────────────────────────────────────────────

    async def _speak_silence_notice(self, text: str, tag: str | None = None) -> None:
        """Speak a silence-guardrail notice without ever crashing the call."""
        try:
            if self.state != CallState.LISTENING or self._agent_speaking.is_set():
                return
            epoch = self._audio_epoch
            seg: Segment = {"tag": tag, "text": text}
            audio = await tts_module.synthesize(seg, language=self.language)
            # The caller may have started talking (barge-in) while TTS ran.
            if (
                epoch != self._audio_epoch
                or self.state != CallState.LISTENING
                or self._agent_speaking.is_set()
            ):
                return
            if audio:
                await self._send_audio(audio, text=text)
                await self._send_json({"type": "audio_end"})
            await self._send_json({"type": "agent_text", "text": text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Silence notice failed: %s", exc)

    async def _silence_watcher(self) -> None:
        """
        Periodically checks for prolonged silence while listening.
        First timeout (SILENCE_WARN_S, 45s): gentle prompt in active language.
        Second timeout (SILENCE_END_S, 30s after warning): polite conclusion.

        The timer is frozen while the agent is thinking/speaking or while the
        browser is still playing audio.  Crucially, only *speech activity*
        (STT events) refreshes the timer – continuous silent mic frames must
        not keep the call alive forever.
        """
        while self.state != CallState.ENDED:
            await asyncio.sleep(_SILENCE_TICK_S)
            now = time.monotonic()

            # Safety valve: never freeze forever if a playback_done is lost.
            if (
                self._client_playing_audio
                and self._client_playing_since
                and now - self._client_playing_since > 60.0
            ):
                log.warning("playback_done not received; clearing playback flag.")
                self._client_playing_audio = False
                self._client_playing_since = 0.0
                self._agent_speaking.clear()
                self._last_audio_ts = now

            if (
                self.state != CallState.LISTENING
                or self._agent_speaking.is_set()
                or self._client_playing_audio
            ):
                # Thinking/speaking/playing: keep the patience timer fresh but
                # preserve the fact that a warning was already issued.
                self._last_audio_ts = now
                continue

            elapsed = time.monotonic() - self._last_audio_ts
            silence_cfg = SILENCE_MESSAGES.get(
                self.language, SILENCE_MESSAGES["ta-IN"]
            )

            if not self._silence_warned:
                if elapsed >= SILENCE_WARN_S:
                    self._silence_warned = True
                    log.info(
                        "Silence warning triggered (elapsed=%.1fs >= %.1fs)",
                        elapsed,
                        SILENCE_WARN_S,
                    )
                    await self._speak_silence_notice(silence_cfg["warn"])
                    self._last_audio_ts = time.monotonic()
            else:
                if elapsed >= SILENCE_END_S:
                    log.info(
                        "Silence end triggered (elapsed=%.1fs post-warning)",
                        elapsed,
                    )
                    await self._speak_silence_notice(
                        silence_cfg["end"], tag="<sigh>"
                    )
                    await asyncio.sleep(_END_PAUSE_S)
                    # Let the browser finish the goodbye before closing the
                    # socket (otherwise handleCallEnded flushes it mid-word).
                    deadline = time.monotonic() + _GOODBYE_WAIT_S
                    while self._client_playing_audio and time.monotonic() < deadline:
                        await asyncio.sleep(0.1)
                    await self._set_state(CallState.ENDED)
                    break

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main coroutine: receive WebSocket frames, orchestrate STT/LLM/TTS.
        """
        await self._set_state(CallState.IDLE)

        # Wait for {"type": "start"} from the client
        try:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout=30.0)
            msg = json.loads(raw)
            if msg.get("type") != "start":
                await self._send_json({"type": "error", "message": "Expected start"})
                return
            if msg.get("language"):
                self.set_language(msg["language"])
        except asyncio.TimeoutError:
            await self._send_json({"type": "error", "message": "Start timeout"})
            return
        except Exception as exc:
            log.warning("Handshake error: %s", exc)
            return

        # Greeting
        await self._set_state(CallState.SPEAKING)
        await self._play_greeting()
        await self._set_state(CallState.LISTENING)

        # Launch silence watcher
        silence_task = asyncio.create_task(self._silence_watcher())

        # Active turn task handle (so barge-in can cancel it)
        turn_task: asyncio.Task | None = None

        try:
            while self.state != CallState.ENDED:
                # Receive the next frame
                try:
                    frame = await asyncio.wait_for(
                        self._receive_frame(), timeout=SILENCE_WARN_S + SILENCE_END_S + 30
                    )
                except asyncio.TimeoutError:
                    break
                except Exception:
                    break

                if frame is None:
                    break

                frame_type, payload = frame

                if frame_type == "binary":
                    # PCM audio from the mic
                    if self.muted or self.state != CallState.LISTENING or self._agent_speaking.is_set():
                        continue
                    self._last_audio_ts = time.monotonic()

                    # Forward to STT queue (drop if full to avoid back-pressure)
                    try:
                        self._pcm_queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass

                elif frame_type == "json":
                    msg_type = payload.get("type")
                    if msg_type == "playback_done":
                        log.debug("Client audio playback finished.")
                        self._client_playing_audio = False
                        self._agent_speaking.clear()
                        self._last_audio_ts = time.monotonic()
                        await self._set_state(CallState.LISTENING)
                    elif msg_type == "mute":
                        self.muted = bool(payload.get("on", False))
                        if self.muted:
                            # Signal end-of-stream to STT
                            await self._pcm_queue.put(None)
                    elif msg_type == "end":
                        break

                # Check if STT has produced a final transcript and start a turn
                # (We use a lightweight polling approach here; for a full async
                #  pipeline you would run the STT as a separate task.)

        finally:
            silence_task.cancel()
            if turn_task and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
            await self._pcm_queue.put(None)  # signal STT to stop
            await self._set_state(CallState.ENDED)
            log.info("CallSession ended.")

    async def _receive_frame(self):
        """
        Wait for the next WebSocket frame.
        Returns ("binary", bytes) or ("json", dict) or None on disconnect.
        """
        from starlette.websockets import WebSocketDisconnect  # type: ignore

        try:
            # Try binary first
            data = await self.ws.receive()
            if data["type"] == "websocket.disconnect":
                return None
            if "bytes" in data and data["bytes"]:
                return ("binary", data["bytes"])
            if "text" in data and data["text"]:
                try:
                    return ("json", json.loads(data["text"]))
                except json.JSONDecodeError:
                    return ("json", {})
        except Exception:
            return None


# ── Integrated full-pipeline session (used by main.py) ───────────────────────


class FullCallSession(CallSession):
    """
    Extends CallSession with the full async STT->LLM->TTS pipeline running
    concurrently in separate tasks.
    """

    async def run(self) -> None:
        await self._set_state(CallState.IDLE)

        # Wait for {"type": "start"} from the client
        try:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout=30.0)
            msg = json.loads(raw)
        except asyncio.TimeoutError:
            await self._send_json({"type": "error", "message": "Start timeout"})
            return
        except Exception as exc:
            log.warning("Handshake error: %s", exc)
            return

        if msg.get("type") != "start":
            await self._send_json({"type": "error", "message": "Expected start"})
            return
        if msg.get("language"):
            self.set_language(msg["language"])
        # The language was just applied; no need to restart the first stream.
        self._stt_restart.clear()

        silence_task = asyncio.create_task(
            self._silence_watcher(), name="call-silence"
        )
        stt_task = asyncio.create_task(self._stt_loop(), name="call-stt")
        ws_task = asyncio.create_task(
            self._ws_receive_loop(), name="call-ws"
        )

        try:
            # Greeting first, then listen
            await self._set_state(CallState.SPEAKING)
            await self._play_greeting()
            await self._set_state(CallState.LISTENING)

            # The call ends when the websocket closes or the silence watcher
            # decides to hang up.  STT keeps running independently.
            await asyncio.wait(
                {silence_task, ws_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Session loop error: %s", exc, exc_info=True)
        finally:
            tasks = [silence_task, stt_task, ws_task, self._turn_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            # Never block hang-up on a provider/stream that ignores cancellation.
            pending = [task for task in tasks if task is not None]
            if pending:
                done, still_pending = await asyncio.wait(pending, timeout=2.0)
                for task in still_pending:
                    log.warning(
                        "Task %s did not stop within 2s of hang-up.",
                        task.get_name(),
                    )
            self._turn_task = None
            self._signal_stt_end()
            await self._set_state(CallState.ENDED)
            log.info("FullCallSession ended.")

    # ── Queue helpers ─────────────────────────────────────────────────────────

    def _signal_stt_end(self) -> None:
        """Non-blocking end-of-stream signal (never hangs on a full queue)."""
        try:
            self._pcm_queue.put_nowait(None)
            return
        except asyncio.QueueFull:
            pass
        # Make room for the sentinel; stale mic frames are useless at hang-up.
        try:
            self._pcm_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self._pcm_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def _on_turn_done(self, task: asyncio.Task) -> None:
        if self._turn_task is task:
            self._turn_task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("Turn task failed: %s", exc, exc_info=exc)

    # ── WebSocket receive loop ────────────────────────────────────────────────

    async def _ws_receive_loop(self) -> None:
        """Read WebSocket frames and route audio to the PCM queue / handle control."""
        while self.state != CallState.ENDED:
            frame = await self._receive_frame()
            if frame is None:
                self._signal_stt_end()
                return

            frame_type, payload = frame
            if frame_type == "binary":
                if self.muted:
                    continue
                if self.state in (CallState.IDLE, CallState.ENDED):
                    continue
                # Do not feed the agent's own voice back into STT.  We *do*
                # accept audio while thinking/synthesising so a real barge-in
                # can be transcribed.
                if self._agent_speaking.is_set():
                    continue

                try:
                    self._pcm_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass  # drop rather than apply back-pressure to the socket

            elif frame_type == "json":
                msg_type = payload.get("type")

                if msg_type == "playback_done":
                    log.debug("Client audio playback completed.")
                    self._client_playing_audio = False
                    self._client_playing_since = 0.0
                    self._agent_speaking.clear()
                    self._last_audio_ts = time.monotonic()
                    # A stale playback_done must not interrupt an active turn.
                    if self.state not in (CallState.THINKING, CallState.SPEAKING):
                        await self._set_state(CallState.LISTENING)

                elif msg_type == "bargein":
                    await self._handle_bargein()

                elif msg_type == "set_language":
                    lang = payload.get("language")
                    if lang:
                        self.set_language(lang)

                elif msg_type == "mute":
                    self.muted = bool(payload.get("on", False))
                    if self.muted:
                        log.info("Caller muted the microphone.")

                elif msg_type == "end":
                    self._signal_stt_end()
                    return

    # ── STT loop ──────────────────────────────────────────────────────────────

    async def _stt_loop(self) -> None:
        """
        Drive STT continuously.  Turns run in a separate task so transcript
        events keep flowing while the agent thinks/speaks (required for
        barge-in).  The stream is reconnected when the caller changes language
        or the provider closes it.
        """
        while self.state != CallState.ENDED:
            task = asyncio.create_task(
                self._consume_stt(self.language), name="call-stt-stream"
            )
            self._stt_stream_task = task
            try:
                await task
            except asyncio.CancelledError:
                if self.state == CallState.ENDED:
                    break
                if self._stt_restart.is_set():
                    # Language changed: reconnect with the new language.
                    self._stt_restart.clear()
                    continue
                raise
            except Exception as exc:
                if self.state == CallState.ENDED:
                    break
                log.warning("STT stream reset: %s", exc)
                await asyncio.sleep(0.5)
            else:
                if self.state != CallState.ENDED:
                    # Provider closed the stream cleanly: reconnect shortly.
                    await asyncio.sleep(0.2)
            finally:
                if self._stt_stream_task is task:
                    self._stt_stream_task = None

    async def _consume_stt(self, language: str) -> None:
        """Consume one STT stream until it ends, dispatching transcript events."""
        async for event in stt_module.stream(self._pcm_queue, language=language):
            if self.state == CallState.ENDED:
                break

            if event.type in ("partial", "final"):
                # Only real speech activity should refresh the silence timer.
                self._last_audio_ts = time.monotonic()
                self._silence_warned = False

            if event.type == "partial":
                if event.text:
                    await self._send_json(
                        {"type": "user_text", "text": event.text, "final": False}
                    )

            elif event.type == "final":
                text = event.text.strip()
                if not text:
                    continue
                await self._send_json(
                    {"type": "user_text", "text": text, "final": True}
                )
                log.info(
                    "[STT final] %s (latency=%.1f ms)",
                    text,
                    event.latency_ms or 0,
                )

                if self._turn_task and not self._turn_task.done():
                    # Duplicate/echo transcript while a turn is already running.
                    log.debug("Ignoring transcript while turn in flight: %s", text)
                    continue

                task = asyncio.create_task(
                    self._run_turn(text), name="call-turn"
                )
                self._turn_task = task
                task.add_done_callback(self._on_turn_done)

            elif event.type == "error":
                await self._send_json(
                    {"type": "error", "message": event.text}
                )
                break

