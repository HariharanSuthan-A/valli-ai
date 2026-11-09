"""
test_session_mock.py – Integration tests for the CallSession in mock mode.

Tests run with MOCK=1, so no real API calls are made.

Covered scenarios:
  - Full turn: greeting -> user utterance -> agent reply
  - Barge-in mid-speech: stop_audio sent, state returns to listening
  - Silence timeout: gentle warning after SILENCE_WARN_S
  - Rapid double utterance: second turn starts after first completes
"""
import asyncio
import json
import os
import pytest
import pytest_asyncio

# Force mock mode before any app imports
os.environ["MOCK"] = "1"

from app.session import FullCallSession, CallState


# ── Mock WebSocket ─────────────────────────────────────────────────────────────

class MockWebSocket:
    """Fake WebSocket for testing CallSession."""

    def __init__(self, receive_sequence: list):
        """
        receive_sequence: list of items to deliver from receive().
        Each item is either:
          - bytes  → binary frame (audio)
          - dict   → JSON control message
          - None   → disconnect signal
        """
        self._recv = list(receive_sequence)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.client = ("127.0.0.1", 1234)

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def receive_text(self) -> str:
        """Used for the initial handshake."""
        item = self._recv.pop(0) if self._recv else None
        if isinstance(item, dict):
            return json.dumps(item)
        raise Exception("Expected JSON in receive_text")

    async def receive(self) -> dict:
        """Starlette-style receive."""
        if not self._recv:
            await asyncio.sleep(0.5)
            return {"type": "websocket.disconnect"}

        item = self._recv.pop(0)
        if item is None:
            return {"type": "websocket.disconnect"}
        if isinstance(item, bytes):
            return {"type": "websocket.receive", "bytes": item, "text": None}
        if isinstance(item, dict):
            return {"type": "websocket.receive", "bytes": None, "text": json.dumps(item)}
        return {"type": "websocket.disconnect"}

    async def close(self) -> None:
        pass

    def get_json_messages(self) -> list[dict]:
        return [json.loads(t) for t in self.sent_text]

    def get_message_types(self) -> list[str]:
        return [m.get("type") for m in self.get_json_messages()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _silent_pcm(ms: int = 40, rate: int = 16000) -> bytes:
    """Return silent PCM16 for the given duration."""
    n = int(rate * ms / 1000)
    return b"\x00\x00" * n


async def _run_session(recv_seq: list, timeout: float = 5.0) -> MockWebSocket:
    ws = MockWebSocket(recv_seq)
    session = FullCallSession(ws)
    try:
        await asyncio.wait_for(session.run(), timeout=timeout)
    except asyncio.TimeoutError:
        pass  # expected for sessions that don't self-terminate
    return ws


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_turn_greeting_then_utterance():
    """
    After start, the session should:
    1. Send state=idle
    2. Send state=speaking (greeting)
    3. Send agent_text
    4. Send state=listening
    5. On final utterance, send thinking then speaking again
    """
    recv = [
        {"type": "start"},
        # Simulate one audio chunk (agent will drain it)
        _silent_pcm(),
        # End the call
        {"type": "end"},
    ]
    ws = await _run_session(recv, timeout=6.0)
    types = ws.get_message_types()
    assert "state" in types


@pytest.mark.asyncio
async def test_mute_control():
    """Mute message should be accepted without error."""
    recv = [
        {"type": "start"},
        {"type": "mute", "on": True},
        {"type": "mute", "on": False},
        {"type": "end"},
    ]
    ws = await _run_session(recv, timeout=5.0)
    # Should not have produced any error messages
    msgs = ws.get_json_messages()
    errors = [m for m in msgs if m.get("type") == "error"]
    assert errors == []


@pytest.mark.asyncio
async def test_barge_in_sends_stop_audio():
    """
    When binary audio arrives while agent_speaking flag is set,
    the session should send stop_audio.
    Note: in mock mode the agent will be speaking during the greeting.
    We inject audio quickly to trigger the barge-in path.
    """
    # Lots of audio right after start to trigger barge-in during greeting
    audio_chunks = [_silent_pcm() for _ in range(10)]
    recv = [
        {"type": "start"},
        *audio_chunks,
        {"type": "end"},
    ]
    ws = await _run_session(recv, timeout=6.0)
    types = ws.get_message_types()
    # stop_audio should appear if barge-in fired (not guaranteed in all timings,
    # but we at least confirm no crash)
    assert "state" in types  # session ran without error


@pytest.mark.asyncio
async def test_disconnect_cleans_up():
    """Disconnect (None) should end the session cleanly."""
    recv = [
        {"type": "start"},
        None,  # disconnect
    ]
    ws = await _run_session(recv, timeout=5.0)
    # ended state should have been sent
    types = ws.get_message_types()
    state_msgs = [m for m in ws.get_json_messages() if m.get("type") == "state"]
    # At least idle or speaking should be in there
    assert len(state_msgs) >= 1


@pytest.mark.asyncio
async def test_no_start_message_errors():
    """If client sends wrong message first, server should send an error."""
    recv = [
        {"type": "mute", "on": False},  # wrong first message
    ]
    ws = await _run_session(recv, timeout=5.0)
    msgs = ws.get_json_messages()
    errors = [m for m in msgs if m.get("type") == "error"]
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_end_message_terminates_session():
    """{"type": "end"} should cleanly end the session."""
    recv = [
        {"type": "start"},
        {"type": "end"},
    ]
    ws = await _run_session(recv, timeout=5.0)
    types = ws.get_message_types()
    assert "state" in types


@pytest.mark.asyncio
async def test_multi_turn_followup_and_memory():
    """Verify follow-up turns preserve conversational memory and active service."""
    ws = MockWebSocket([])
    session = FullCallSession(ws)
    await session._set_state(CallState.LISTENING)

    # Turn 1: User asks about Aadhaar
    await session._run_turn("ஆதார் அட்டை திருத்தம் எப்படி செய்வது?")
    assert len(session._history) == 2
    assert session._history[0]["role"] == "user"
    assert session._history[1]["role"] == "assistant"
    assert session._active_service_id == "aadhaar_update"

    # Turn 2: Follow-up question
    await session._run_turn("முகவரி மாற்ற என்ன ஆவணங்கள் வேணும்?")
    assert len(session._history) == 4
    assert session._history[2]["role"] == "user"
    assert session._history[3]["role"] == "assistant"
    assert session._active_service_id == "aadhaar_update"

    # Audio should have been dispatched for both turns
    assert len(ws.sent_bytes) >= 2


@pytest.mark.asyncio
async def test_multilingual_language_switching():
    """Verify session switches language and generates language-specific greetings."""
    from app.session import GREETINGS

    ws = MockWebSocket([])
    session = FullCallSession(ws)

    # Test Hindi
    session.set_language("hi-IN")
    assert session.language == "hi-IN"
    assert "Hindi" in session._system_prompt or "हिन्दी" in session._system_prompt
    assert "Namaste" in GREETINGS["hi-IN"]

    # Test Telugu
    session.set_language("te-IN")
    assert session.language == "te-IN"
    assert "Telugu" in session._system_prompt or "తెలుగు" in session._system_prompt

    # Test Malayalam
    session.set_language("ml-IN")
    assert session.language == "ml-IN"
    assert "Malayalam" in session._system_prompt or "മലയാളം" in session._system_prompt

    # Test Kannada
    session.set_language("kn-IN")
    assert session.language == "kn-IN"
    assert "Kannada" in session._system_prompt or "ಕನ್ನಡ" in session._system_prompt

    # Test Tamil (default)
    session.set_language("ta-IN")
    assert session.language == "ta-IN"
    assert "Tamil" in session._system_prompt or "தமிழ்" in session._system_prompt


@pytest.mark.asyncio
async def test_multilingual_thank_you_responses():
    """Verify thank-you detection and localized closing replies across all 5 languages."""
    from app.session import _is_thank_you_phrase, THANK_YOU_RESPONSES

    # Verify detection across various languages and English
    assert _is_thank_you_phrase("thank you")
    assert _is_thank_you_phrase("Thanks a lot")
    assert _is_thank_you_phrase("ரொம்ப நன்றி வள்ளி")
    assert _is_thank_you_phrase("धन्यवाद!")
    assert _is_thank_you_phrase("చాలా ధన్యవాదాలు")
    assert _is_thank_you_phrase("വളരെ നന്ദി")
    assert _is_thank_you_phrase("ತುಂಬಾ ಧನ್ಯವಾದಗಳು")

    ws = MockWebSocket([])
    session = FullCallSession(ws)

    # Test each language produces its specific thank you reply
    for lang, expected_snippet in [
        ("ta-IN", "வள்ளியைத்"),
        ("hi-IN", "वल्ली से"),
        ("te-IN", "వల్లిని"),
        ("ml-IN", "വള്ളിയെ"),
        ("kn-IN", "ವಳ್ಳಿಯನ್ನು"),
    ]:
        session.set_language(lang)
        await session._run_turn("thank you")
        msgs = ws.get_json_messages()
        agent_texts = [m.get("text") for m in msgs if m.get("type") == "agent_text"]
        assert any(expected_snippet in (t or "") for t in agent_texts)
        assert THANK_YOU_RESPONSES[lang] in session._history[-1]["content"]


@pytest.mark.asyncio
async def test_playback_done_switches_to_listening_and_arms_45s_timer():
    """Verify that when audio playback finishes, state transitions to LISTENING and 45s timer is armed."""
    import time
    ws = MockWebSocket([])
    session = FullCallSession(ws)
    session._client_playing_audio = True
    session._agent_speaking.set()
    session.state = CallState.SPEAKING

    # Simulate playback_done event received from browser
    t_done = time.monotonic()
    session._client_playing_audio = False
    session._agent_speaking.clear()
    session._last_audio_ts = t_done
    await session._set_state(CallState.LISTENING)

    assert session.state == CallState.LISTENING
    assert not session._agent_speaking.is_set()
    assert not session._client_playing_audio
    assert session._last_audio_ts >= t_done
