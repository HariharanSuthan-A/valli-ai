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

# ── Greeting ──────────────────────────────────────────────────────────────────

GREETING_TEXT = (
    "@@service=none@@ Vanakkam, Naan Valli pesuren. "
    "Government services pathi enna help venum?"
)

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
    m = _MARKER_RE.match(text.lstrip())
    if m:
        sid = m.group(1).strip()
        rest = text[text.index(m.group(0)) + len(m.group(0)):].lstrip()
        return (None if sid == "none" else sid), rest
    return None, text


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

        # PCM audio queue fed by the WebSocket receiver and consumed by STT
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

        # Conversation history (alternating user/assistant, last HISTORY_TURNS*2)
        self._history: list[dict[str, str]] = []

        # System prompt (built once)
        self._system_prompt: str = build_system_prompt()

        # Active service ID tracked across follow-up turns
        self._active_service_id: str | None = None

        # Flag: agent is currently playing audio (for barge-in detection)
        self._agent_speaking = asyncio.Event()
        self._agent_speaking.clear()

        # Cancellation token for the current turn pipeline
        self._turn_cancel: asyncio.Event = asyncio.Event()

        # Last time we received user audio (for silence timeout)
        self._last_audio_ts: float = time.monotonic()

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
        _, text_no_marker = _extract_marker(GREETING_TEXT)
        segments = parse(text_no_marker)
        for seg in segments:
            if self._turn_cancel.is_set():
                break
            audio = await tts_module.synthesize(seg)
            if audio:
                await self._send_audio(audio, text=seg["text"])
        await self._send_json(
            {"type": "agent_text", "text": strip(text_no_marker)}
        )
        await self._send_json({"type": "audio_end"})

    # ── Filler ────────────────────────────────────────────────────────────────

    async def _play_filler(self, cancel_event: asyncio.Event) -> None:
        """
        Play the 'ஒரு நிமிஷம்...' filler clip while the response is being generated.
        """
        await asyncio.sleep(0.15)
        if cancel_event.is_set():
            return
        audio = fillers.get_oru_nimisham_audio()
        if audio and not cancel_event.is_set():
            if self.state != CallState.SPEAKING:
                await self._set_state(CallState.SPEAKING)
            self._agent_speaking.set()
            await self._send_audio(audio, text="ஒரு நிமிஷம்...")

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

            # On the very first segment, extract the @@service=...@@ marker
            if not service_sent:
                sid, seg_text = _extract_marker(seg_text)
                if sid is None:
                    # In follow-up turns, if user query didn't explicitly trigger another service,
                    # retain the active service; otherwise try lookup
                    matched = kb_module.lookup(user_text)
                    if matched:
                        sid = matched["id"]
                    elif self._active_service_id:
                        sid = self._active_service_id
                if sid and sid != "none":
                    self._active_service_id = sid

                service_id = sid
                service_sent = True
                if sid and sid != "none":
                    entry = kb_module.get_by_id(sid)
                    if entry:
                        await self._send_json({"type": "service", "entry": entry})

            segments = parse(seg_text)
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

                audio = await tts_module.synthesize(seg)
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

        except Exception as exc:
            log.error("LLM error during turn: %s", exc)
            await self._send_json({"type": "error", "message": "LLM error"})
        finally:
            filler_cancel.set()
            filler_task.cancel()
            try:
                await filler_task
            except asyncio.CancelledError:
                pass

        full_reply = "".join(full_reply_tokens).strip()
        self._push_history("assistant", full_reply)

        # Notify browser that all audio segments for this turn have been sent
        await self._send_json({"type": "audio_end"})

        self._agent_speaking.clear()
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
        self._agent_speaking.clear()
        await self._send_json({"type": "stop_audio"})
        await self._set_state(CallState.LISTENING)

    # ── Silence watcher ───────────────────────────────────────────────────────

    async def _silence_watcher(self) -> None:
        """
        Periodically checks for prolonged silence while listening.
        First timeout: gentle prompt.
        Second timeout: end call.
        """
        warned = False
        while self.state != CallState.ENDED:
            await asyncio.sleep(2.0)
            if self.state not in (CallState.LISTENING,):
                # When thinking, speaking, or in greeting: continuously keep timer fresh
                self._last_audio_ts = time.monotonic()
                warned = False
                continue
            elapsed = time.monotonic() - self._last_audio_ts
            if not warned and elapsed >= SILENCE_WARN_S:
                warned = True
                hello_seg = {"tag": None, "text": "Hello?"}
                audio = await tts_module.synthesize(hello_seg)
                if audio:
                    await self._send_audio(audio, text=hello_seg["text"])
                    await self._send_json({"type": "audio_end"})
                await self._send_json(
                    {"type": "agent_text", "text": "Hello?"}
                )
                self._last_audio_ts = time.monotonic()
            elif elapsed >= SILENCE_END_S:
                bye_seg = {"tag": "<sigh>", "text": "Sari, piragu pesalam. Nandri!"}
                audio = await tts_module.synthesize(bye_seg)
                if audio:
                    await self._send_audio(audio, text=bye_seg["text"])
                    await self._send_json({"type": "audio_end"})
                await self._send_json(
                    {"type": "agent_text", "text": "Sari, piragu pesalam. Nandri!"}
                )
                await asyncio.sleep(3.5)
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
                        self._receive_frame(), timeout=SILENCE_END_S + 5
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
                {silence_task, stt_task, ws_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        finally:
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
                    log.debug("Client audio playback finished.")
                    self._last_audio_ts = time.monotonic()
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
                async for event in stt_module.stream(self._pcm_queue):
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
                        await self._run_turn(event.text)
                    elif event.type == "error":
                        await self._send_json({"type": "error", "message": event.text})
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self.state == CallState.ENDED:
                    break
                log.warning("STT loop stream reset: %s", exc)
                await asyncio.sleep(0.5)

