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
)
from app.persona import build_system_prompt
from app.tags import TagBuffer, parse, strip

log = logging.getLogger(__name__)

# ── Silence & Timeout Constants ───────────────────────────────────────────────

_SILENCE_TICK_S: float = 1.0
_END_PAUSE_S: float = 3.5

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

APOLOGY_MESSAGES: dict[str, str] = {
    "ta-IN": "மன்னிக்கவும், ஒரு தொழில்நுட்ப கோளாறு ஏற்பட்டுள்ளது. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    "hi-IN": "क्षमा करें, एक तकनीकी समस्या आ गई है। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
    "te-IN": "క్షమించండి, సాంకేతిక సమస్య ఏర్పడింది. దయచేసి కాసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి.",
    "ml-IN": "ക്ഷമിക്കണം, ഒരു സാങ്കേതിക തകരാർ സംഭവിച്ചു. ദയവായി അല്പം കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ದೋಷ ಉಂಟಾಗಿದೆ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
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

_MARKER_RE = re.compile(r"@@service=([^@]*)@@")


def _extract_marker(text: str) -> tuple[str | None, str]:
    """Return (service_id, text_without_marker). service_id may be 'none'."""
    matches = _MARKER_RE.findall(text)
    sid: str | None = None
    for m in matches:
        clean = m.strip()
        if clean and clean != "none":
            sid = clean
            break
    clean_text = _MARKER_RE.sub("", text)
    clean_text = re.sub(r"[ \t]+", " ", clean_text).strip()
    return sid, clean_text


# ── CallSession ───────────────────────────────────────────────────────────────


class CallSession:
    """
    Manages the full lifecycle of a single phone call WebSocket session.

    Usage:
        session = CallSession(websocket)
        await session.run()
    """

    def __init__(self, websocket: Any) -> None:
        self.ws = websocket
        self.state = CallState.IDLE
        self.muted = False
        self.language: str = "ta-IN"

        # PCM audio queue fed by the WebSocket receiver and consumed by STT
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

        # Conversation history (alternating user/assistant, last HISTORY_TURNS*2)
        self._history: list[dict[str, str]] = []

        # System prompt (built for selected language)
        self._system_prompt: str = build_system_prompt(language=self.language)

        # Active service ID tracked across follow-up turns
        self._active_service_id: str | None = None

        # Flag: agent is currently playing audio (for barge-in detection)
        self._agent_speaking = asyncio.Event()
        self._agent_speaking.clear()

        # Flag: client browser is currently playing audio chunks
        self._client_playing_audio: bool = False
        self._client_playing_since: float = 0.0

        # Cancellation token and task for the current turn pipeline
        self._turn_cancel: asyncio.Event = asyncio.Event()
        self._turn_task: asyncio.Task | None = None

        # Last time we received user audio (for silence timeout)
        self._last_audio_ts: float = time.monotonic()

        # Per-stage timing log
        self._timings: dict[str, float] = {}

    def set_language(self, language: str) -> None:
        """Update active language for STT, TTS and LLM prompt."""
        if language in GREETINGS:
            self.language = language
            self._system_prompt = build_system_prompt(language=self.language)
            log.info("CallSession language set to: %s", self.language)

        # Per-stage timing log
        self._timings: dict[str, float] = {}

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    async def _send_json(self, obj: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception as exc:
            log.debug("WS send_json error: %s", exc)

    async def _send_binary(self, data: bytes) -> None:
        try:
            await self.ws.send_bytes(data)
        except Exception as exc:
            log.debug("WS send_bytes error: %s", exc)

    async def _set_state(self, state: CallState) -> None:
        self.state = state
        if state == CallState.LISTENING:
            self._last_audio_ts = time.monotonic()
        await self._send_json({"type": "state", "value": state.value})

    # ── Audio output ──────────────────────────────────────────────────────────

    async def _send_audio(self, audio_bytes: bytes, text: str = "", sample_rate: int = 16000) -> None:
        """Send audio metadata (with paired transcript text) then the raw audio binary frame."""
        self._client_playing_audio = True
        self._client_playing_since = time.monotonic()
        await self._send_json(
            {"type": "audio", "format": "wav", "rate": sample_rate, "text": text}
        )
        await self._send_binary(audio_bytes)

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
        """Synthesise and stream the agent greeting."""
        greeting_text = GREETINGS.get(self.language, GREETINGS["ta-IN"])
        _, text_no_marker = _extract_marker(greeting_text)
        segments = parse(text_no_marker)
        for seg in segments:
            if self._turn_cancel.is_set():
                break
            audio = await tts_module.synthesize(seg, language=self.language)
            if audio:
                await self._send_audio(audio, text=seg["text"])
        await self._send_json(
            {"type": "agent_text", "text": strip(text_no_marker)}
        )
        await self._send_json({"type": "audio_end"})

    # ── Filler ────────────────────────────────────────────────────────────────

    async def _play_filler(self, cancel_event: asyncio.Event) -> None:
        """
        Play the localized 'One minute...' filler clip based on selected language
        while the response is being generated.
        """
        await asyncio.sleep(0.15)
        if cancel_event.is_set():
            return
        audio, text = await fillers.get_filler_audio(self.language)
        if audio and not cancel_event.is_set():
            if self.state != CallState.SPEAKING:
                await self._set_state(CallState.SPEAKING)
            self._agent_speaking.set()
            await self._send_audio(audio, text=text)

    # ── LLM + TTS pipeline for one turn ──────────────────────────────────────

    async def _run_turn(self, user_text: str) -> None:
        """
        Full pipeline: LLM stream -> sentence chunker -> TTS -> audio send.
        Respects self._turn_cancel for barge-in.
        """
        t_turn_start = time.monotonic()
        self._turn_cancel.clear()
        await self._set_state(CallState.THINKING)

        # Push user message into history
        self._push_history("user", user_text)

        # Fast-path for caller thank you / gratitude / closing
        if _is_thank_you_phrase(user_text):
            reply_text = THANK_YOU_RESPONSES.get(self.language, THANK_YOU_RESPONSES["ta-IN"])
            if self.state != CallState.SPEAKING:
                await self._set_state(CallState.SPEAKING)
            self._agent_speaking.set()
            self._push_history("assistant", reply_text)
            seg = {"tags": [], "text": reply_text}
            audio = await tts_module.synthesize(seg, language=self.language)
            if audio and not self._turn_cancel.is_set():
                await self._send_audio(audio, text=reply_text)
                await self._send_json({"type": "agent_text", "text": reply_text})
            await self._send_json({"type": "audio_end"})
            self._agent_speaking.clear()
            if not self._turn_cancel.is_set():
                await self._set_state(CallState.LISTENING)
            return

        # Start filler; cancel it when the first audio chunk is ready
        filler_cancel = asyncio.Event()
        filler_task = asyncio.create_task(self._play_filler(filler_cancel))

        # Collect the full LLM reply (streamed) while also chunking into sentences
        full_reply_tokens: list[str] = []
        tag_buffer = TagBuffer()
        service_id: str | None = None
        service_sent = False
        first_audio = False
        t_first_token: float | None = None

        async def _send_segment(seg_text: str) -> None:
            nonlocal first_audio, service_id, service_sent
            if self._turn_cancel.is_set():
                return

            sid, clean_seg_text = _extract_marker(seg_text)
            if sid and sid != "none":
                self._active_service_id = sid
                service_id = sid
                if not service_sent:
                    service_sent = True
                    entry = kb_module.get_by_id(sid)
                    if entry:
                        await self._send_json({"type": "service", "entry": entry})

            if not clean_seg_text:
                return

            segments = parse(clean_seg_text)
            for seg in segments:
                if self._turn_cancel.is_set():
                    return
                # Cancel filler and start speaking state on first audio
                if not first_audio:
                    filler_cancel.set()
                    filler_task.cancel()
                    if self.state != CallState.SPEAKING:
                        await self._set_state(CallState.SPEAKING)
                    self._agent_speaking.set()
                    first_audio = True
                    self._timings["first_audio"] = time.monotonic()
                    log.info(
                        "[timing] thinking→first_audio: %.3f s",
                        self._timings["first_audio"] - t_turn_start,
                    )

                audio = await tts_module.synthesize(seg, language=self.language)
                if audio and not self._turn_cancel.is_set():
                    await self._send_audio(audio, text=seg["text"])
                    await self._send_json({"type": "agent_text", "text": seg["text"]})

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

            # Fallback to KB lookup or active service if no explicit marker was given
            if not service_sent:
                matched = kb_module.lookup(user_text)
                sid = matched["id"] if matched else self._active_service_id
                if sid and sid != "none":
                    self._active_service_id = sid
                    service_sent = True
                    entry = kb_module.get_by_id(sid)
                    if entry:
                        await self._send_json({"type": "service", "entry": entry})

        except Exception as exc:
            log.error("LLM error during turn: %s", exc)
            await self._send_json({"type": "error", "message": "LLM error"})
            filler_cancel.set()
            filler_task.cancel()
            if not self._turn_cancel.is_set():
                apology_text = APOLOGY_MESSAGES.get(self.language, APOLOGY_MESSAGES["ta-IN"])
                try:
                    seg = {"tags": [], "text": apology_text}
                    audio = await tts_module.synthesize(seg, language=self.language)
                    if audio and not self._turn_cancel.is_set():
                        await self._send_audio(audio, text=apology_text)
                        await self._send_json({"type": "agent_text", "text": apology_text})
                except Exception as tts_err:
                    log.error("Failed to speak apology: %s", tts_err)
        finally:
            filler_cancel.set()
            filler_task.cancel()
            try:
                await filler_task
            except asyncio.CancelledError:
                pass
            self._agent_speaking.clear()

        full_reply = "".join(full_reply_tokens).strip()
        if full_reply:
            self._push_history("assistant", full_reply)

        # Notify browser that all audio segments for this turn have been sent
        await self._send_json({"type": "audio_end"})

        self._agent_speaking.clear()
        if not self._turn_cancel.is_set():
            await self._set_state(CallState.LISTENING)
        log.info(
            "[timing] total turn: %.3f s",
            time.monotonic() - t_turn_start,
        )

    # ── Barge-in ──────────────────────────────────────────────────────────────

    async def _handle_bargein(self) -> None:
        """Cancel any in-flight turn and return to listening."""
        log.info("Barge-in detected; cancelling turn.")
        self._turn_cancel.set()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._turn_task = None
        self._agent_speaking.clear()
        self._client_playing_audio = False
        await self._send_json({"type": "stop_audio"})
        await self._set_state(CallState.LISTENING)

    # ── Silence watcher ───────────────────────────────────────────────────────

    async def _silence_watcher(self) -> None:
        """
        Periodically checks for prolonged silence while listening.
        First timeout (SILENCE_WARN_S, 45s): gentle prompt in active language.
        Second timeout (SILENCE_END_S, 30s after warning): polite conclusion.
        Timer is kept frozen while the caller is listening to agent audio playback.
        """
        warned = False
        while self.state != CallState.ENDED:
            await asyncio.sleep(_SILENCE_TICK_S)

            # Safety valve: unstick if playback_done was lost for > 60s
            if self._client_playing_audio and (time.monotonic() - self._client_playing_since > 60.0):
                log.warning("Playback done timeout (>60s) - resetting client playing state")
                self._client_playing_audio = False
                self._agent_speaking.clear()
                self._last_audio_ts = time.monotonic()

            if self.state not in (CallState.LISTENING,) or self._agent_speaking.is_set() or self._client_playing_audio:
                # When thinking, speaking, in greeting, or client is playing audio: keep timer fresh
                self._last_audio_ts = time.monotonic()
                warned = False
                continue

            elapsed = time.monotonic() - self._last_audio_ts
            silence_cfg = SILENCE_MESSAGES.get(self.language, SILENCE_MESSAGES["ta-IN"])

            if not warned:
                if elapsed >= SILENCE_WARN_S:  # Full 45-second patience window for caller
                    warned = True
                    log.info("Silence warning triggered (elapsed=%.1fs >= %.1fs)", elapsed, SILENCE_WARN_S)
                    hello_seg = {"tags": [], "text": silence_cfg["warn"]}
                    audio = await tts_module.synthesize(hello_seg, language=self.language)
                    if audio:
                        await self._send_audio(audio, text=hello_seg["text"])
                        await self._send_json({"type": "audio_end"})
                    await self._send_json(
                        {"type": "agent_text", "text": silence_cfg["warn"]}
                    )
                    self._last_audio_ts = time.monotonic()
            else:
                # Only check for ending call AFTER the warning has been issued!
                if elapsed >= SILENCE_END_S:  # Additional 30s after warning
                    log.info("Silence end triggered (elapsed=%.1fs post-warning)", elapsed)
                    bye_seg = {"tags": ["<sigh>"], "text": silence_cfg["end"]}
                    audio = await tts_module.synthesize(bye_seg, language=self.language)
                    if audio:
                        await self._send_audio(audio, text=bye_seg["text"])
                        await self._send_json({"type": "audio_end"})
                    await self._send_json(
                        {"type": "agent_text", "text": silence_cfg["end"]}
                    )
                    await asyncio.sleep(_END_PAUSE_S)
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
    concurrently in an asyncio task group.
    """

    async def run(self) -> None:
        await self._set_state(CallState.IDLE)

        try:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout=30.0)
            msg = json.loads(raw)
            if msg.get("type") != "start":
                await self._send_json({"type": "error", "message": "Expected start"})
                return
            if msg.get("language"):
                self.set_language(msg["language"])
        except (asyncio.TimeoutError, Exception) as exc:
            await self._send_json({"type": "error", "message": str(exc)})
            return

        silence_task = asyncio.create_task(self._silence_watcher())
        stt_task = asyncio.create_task(self._stt_loop())
        ws_task = asyncio.create_task(self._ws_receive_loop())

        await self._set_state(CallState.SPEAKING)
        await self._play_greeting()
        await self._set_state(CallState.LISTENING)

        try:
            done, pending = await asyncio.wait(
                {silence_task, ws_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        finally:
            stt_task.cancel()
            try:
                await stt_task
            except asyncio.CancelledError:
                pass
            await self._pcm_queue.put(None)
            await self._set_state(CallState.ENDED)
            log.info("FullCallSession ended.")

    async def _ws_receive_loop(self) -> None:
        """Read WebSocket frames and route audio to PCM queue or handle control msgs."""
        while True:
            frame = await self._receive_frame()
            if frame is None:
                await self._pcm_queue.put(None)
                break

            frame_type, payload = frame
            if frame_type == "binary":
                if self.muted:
                    continue
                # Do NOT listen to microphone while agent is speaking or thinking!
                # Wait until all TTS speech finishes playing and state is LISTENING.
                if self.state != CallState.LISTENING or self._agent_speaking.is_set():
                    continue

                self._last_audio_ts = time.monotonic()
                try:
                    self._pcm_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
            elif frame_type == "json":
                msg_type = payload.get("type")
                if msg_type == "playback_done":
                    log.info("Client audio playback completed -> Switched to LISTENING. 45s timer armed.")
                    self._client_playing_audio = False
                    self._agent_speaking.clear()
                    self._last_audio_ts = time.monotonic()
                    await self._set_state(CallState.LISTENING)
                elif msg_type == "set_language":
                    lang = payload.get("language")
                    if lang:
                        self.set_language(lang)
                elif msg_type == "mute":
                    self.muted = bool(payload.get("on", False))
                    if self.muted:
                        await self._pcm_queue.put(None)
                elif msg_type == "end":
                    await self._pcm_queue.put(None)
                    return

    async def _stt_loop(self) -> None:
        """Drive STT and trigger a turn on each final utterance; loop continuously across turns."""
        while self.state != CallState.ENDED:
            try:
                async for event in stt_module.stream(self._pcm_queue, language=self.language):
                    if self.state == CallState.ENDED:
                        break
                    if event.type == "partial":
                        await self._send_json(
                            {"type": "user_text", "text": event.text, "final": False}
                        )
                    elif event.type == "final" and event.text.strip():
                        await self._send_json(
                            {"type": "user_text", "text": event.text, "final": True}
                        )
                        log.info("[STT final] %s (latency=%.1f ms)",
                                 event.text, event.latency_ms or 0)
                        self._turn_task = asyncio.create_task(self._run_turn(event.text))
                        try:
                            await self._turn_task
                        except asyncio.CancelledError:
                            pass
                        finally:
                            self._turn_task = None
                    elif event.type == "error":
                        await self._send_json({"type": "error", "message": event.text})
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self.state == CallState.ENDED:
                    break
                log.warning("STT loop stream reset: %s", exc)
                await asyncio.sleep(0.5)

