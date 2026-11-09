"""
test_regressions.py – Regression tests for call-breaking bugs.

Covered:
  - Slow LLM turn plays a filler without crashing (``get_filler_audio`` API).
  - Service markers are never spoken or shown in captions.
  - Barge-in cancels an in-flight turn and sends ``stop_audio``.
  - Silence watcher warns, then ends the call, without relying on mic frames.
"""
import asyncio
import json
import os

import pytest

os.environ["MOCK"] = "1"

from app import fillers
import app.session as session_module
from app.session import CallState, FullCallSession, GREETINGS, SILENCE_MESSAGES


class MockWebSocket:
    """Minimal fake WebSocket (no inbound frames needed for these tests)."""

    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.client = ("127.0.0.1", 1234)

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self) -> None:
        pass

    def messages(self) -> list[dict]:
        return [json.loads(t) for t in self.sent_text]

    def types(self) -> list[str]:
        return [m.get("type") for m in self.messages()]


@pytest.mark.asyncio
async def test_slow_turn_plays_filler_without_crashing(monkeypatch):
    """A reply that takes longer than FILLER_GRACE_S must not freeze the call."""
    assert callable(fillers.get_filler_audio)

    # Warm the cache so the test is not dominated by mock TTS latency.
    await fillers.preload(["ta-IN"])

    async def slow_stream(_messages):
        await asyncio.sleep(0.2)  # longer than the filler grace below
        yield "இது ஒரு பதில். "

    monkeypatch.setattr(session_module, "FILLER_GRACE_S", 0.0)
    monkeypatch.setattr(session_module.llm_module, "stream", slow_stream)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    session.set_language("ta-IN")
    await session._set_state(CallState.LISTENING)

    await asyncio.wait_for(session._run_turn("aadhaar பற்றி சொல்லுங்க"), timeout=3.0)

    types = ws.types()
    assert "error" not in types
    assert session.state == CallState.LISTENING
    assert not session._agent_speaking.is_set()
    assert types[-1] == "state"
    # Filler + real answer => at least two audio chunks were dispatched.
    assert len(ws.sent_bytes) >= 2


@pytest.mark.asyncio
async def test_service_marker_never_spoken(monkeypatch):
    """Markers emitted after the first sentence must be stripped everywhere."""

    async def marker_stream(_messages):
        yield "முதல் வாக்கியம். "
        yield "@@service=aadhaar_update@@ "
        yield "ஆதார் புதுப்பிக்க இந்த இணைப்பை பயன்படுத்துங்கள்."

    monkeypatch.setattr(session_module.llm_module, "stream", marker_stream)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    session.set_language("ta-IN")
    await session._set_state(CallState.LISTENING)

    await asyncio.wait_for(session._run_turn("hello there"), timeout=3.0)

    msgs = ws.messages()
    assert any(m.get("type") == "service" for m in msgs)

    spoken_texts = [
        m.get("text", "")
        for m in msgs
        if m.get("type") in ("audio", "agent_text")
    ]
    assert spoken_texts, "expected some audio/text output"
    assert all("@@" not in t for t in spoken_texts)


@pytest.mark.asyncio
async def test_bargein_cancels_turn_and_sends_stop_audio(monkeypatch):
    """A barge-in must cancel the active turn and flush browser playback."""

    async def stalled_stream(_messages):
        yield "First part. "
        await asyncio.sleep(10)
        yield "Never reached."

    monkeypatch.setattr(session_module.llm_module, "stream", stalled_stream)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    session.set_language("ta-IN")
    await session._set_state(CallState.LISTENING)

    task = asyncio.create_task(session._run_turn("tell me about aadhaar"))
    session._turn_task = task

    # Wait until the agent has actually started speaking.
    for _ in range(200):
        if session.state == CallState.SPEAKING:
            break
        await asyncio.sleep(0.01)
    assert session.state == CallState.SPEAKING

    await asyncio.wait_for(session._handle_bargein(), timeout=2.0)

    assert task.cancelled() or task.done()
    assert session._turn_task is None
    assert session.state == CallState.LISTENING
    assert not session._agent_speaking.is_set()
    assert not session._client_playing_audio
    assert "stop_audio" in ws.types()


@pytest.mark.asyncio
async def test_llm_total_failure_speaks_apology(monkeypatch):
    """A dead LLM must produce a spoken apology, not silent dead air."""

    async def failing_stream(_messages):
        raise RuntimeError("provider down")
        yield ""  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(session_module.llm_module, "stream", failing_stream)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    session.set_language("ta-IN")
    await session._set_state(CallState.LISTENING)

    await asyncio.wait_for(session._run_turn("passport எப்படி?"), timeout=3.0)

    assert session.state == CallState.LISTENING
    assert not session._agent_speaking.is_set()
    assert ws.sent_bytes, "apology audio must be dispatched"
    assert "error" in ws.types()
    spoken = [m.get("text", "") for m in ws.messages() if m.get("type") == "agent_text"]
    assert any("தொழில்நுட்ப" in t for t in spoken), spoken


@pytest.mark.asyncio
async def test_lost_playback_done_safety_valve(monkeypatch):
    """A missing playback_done must not block the mic forever."""
    import time

    monkeypatch.setattr(session_module, "_SILENCE_TICK_S", 0.02)
    monkeypatch.setattr(session_module, "SILENCE_WARN_S", 999.0)
    monkeypatch.setattr(session_module, "SILENCE_END_S", 999.0)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    await session._set_state(CallState.LISTENING)
    session._client_playing_audio = True
    session._client_playing_since = time.monotonic() - 61.0
    session._agent_speaking.set()

    task = asyncio.create_task(session._silence_watcher())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert not session._client_playing_audio
    assert not session._agent_speaking.is_set()


@pytest.mark.asyncio
async def test_silence_watcher_warns_then_ends(monkeypatch):
    """Silence must eventually warn and hang up without any mic activity."""
    monkeypatch.setattr(session_module, "_SILENCE_TICK_S", 0.02)
    monkeypatch.setattr(session_module, "_END_PAUSE_S", 0.01)
    monkeypatch.setattr(session_module, "SILENCE_WARN_S", 0.0)
    monkeypatch.setattr(session_module, "SILENCE_END_S", 0.0)

    ws = MockWebSocket()
    session = FullCallSession(ws)
    session.set_language("ta-IN")

    # Pretend the browser finished playback instantly after each notice.
    async def fake_send_audio(audio, text="", sample_rate=16000):
        ws.sent_bytes.append(audio)
        return True

    session._send_audio = fake_send_audio  # type: ignore[method-assign]

    await session._set_state(CallState.LISTENING)
    session._last_audio_ts = 0.0  # long ago

    await asyncio.wait_for(session._silence_watcher(), timeout=2.0)

    assert session.state == CallState.ENDED
    cfg = SILENCE_MESSAGES["ta-IN"]
    spoken = [
        m.get("text", "")
        for m in ws.messages()
        if m.get("type") == "agent_text"
    ]
    assert cfg["warn"] in spoken
    assert cfg["end"] in spoken


def test_extract_marker_removes_every_occurrence():
    from app.session import _extract_marker

    sid, text = _extract_marker(
        "Before. @@service=aadhaar_update@@ Middle "
        "@@service=passport@@ End"
    )
    assert sid == "aadhaar_update"
    assert "@@" not in text
    assert text.startswith("Before.")
    assert "Middle" in text and "End" in text


def test_greetings_exist_for_every_supported_language():
    # Guards against a language selection producing a KeyError at call start.
    for lang in ("ta-IN", "hi-IN", "te-IN", "ml-IN", "kn-IN"):
        assert lang in GREETINGS
        assert GREETINGS[lang].strip()
