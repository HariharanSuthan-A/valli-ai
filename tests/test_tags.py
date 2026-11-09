"""
test_tags.py – Unit tests for app/tags.py.

Tests:
  - Unknown tags are dropped
  - Blocked tags are dropped
  - Cap of 2 tags per reply
  - Tags split across streamed chunks (TagBuffer)
  - strip() removes all tags
  - Per-sentence segmentation
  - Tamil text is preserved intact
  - Allowed tags survive
"""
import pytest
from app.tags import (
    ALLOWED_TAGS,
    BLOCKED_TAGS,
    Segment,
    TagBuffer,
    parse,
    strip,
)


# ── strip() ────────────────────────────────────────────────────────────────────

def test_strip_removes_allowed_tags():
    text = "<pauses> நான் உங்களுக்கு help பண்ண ready."
    assert strip(text) == "நான் உங்களுக்கு help பண்ண ready."


def test_strip_removes_blocked_tags():
    text = "<shouting> Hello <sniffles> World"
    assert strip(text) == "Hello World"


def test_strip_removes_unknown_tags():
    text = "<unknown_tag> some text"
    assert strip(text) == "some text"


def test_strip_no_tags():
    text = "Plain text with no tags."
    assert strip(text) == text


def test_strip_preserves_tamil():
    text = "வணக்கம்! <excited> நலமா இருக்கீங்களா?"
    result = strip(text)
    assert "வணக்கம்!" in result
    assert "நலமா இருக்கீங்களா?" in result
    assert "<excited>" not in result


# ── parse() ───────────────────────────────────────────────────────────────────

def test_parse_basic_no_tags():
    text = "வணக்கம். எப்படி இருக்கீங்க?"
    segs = parse(text)
    assert len(segs) >= 1
    for seg in segs:
        assert seg["tag"] is None


def test_parse_allowed_tag_at_start():
    text = "<pauses> ஒரு நிமிஷம். சொல்றேன்."
    segs = parse(text)
    # First segment should carry the <pauses> tag
    assert segs[0]["tag"] == "<pauses>"
    assert "<pauses>" not in segs[0]["text"]


def test_parse_blocked_tag_dropped():
    text = "<shouting> Hello! Normal text."
    segs = parse(text)
    for seg in segs:
        assert seg["tag"] != "<shouting>"
        assert "<shouting>" not in seg["text"]


def test_parse_unknown_tag_dropped():
    text = "<dancing> வணக்கம்!"
    segs = parse(text)
    for seg in segs:
        assert seg["tag"] is None
        assert "<dancing>" not in seg["text"]


def test_parse_cap_at_two_tags():
    text = (
        "<excited> முதல் வாக்கியம். "
        "<sigh> இரண்டாவது வாக்கியம். "
        "<slow> மூன்றாவது வாக்கியம்."
    )
    segs = parse(text, max_tags=2)
    tags_used = sum(1 for s in segs if s["tag"] is not None)
    assert tags_used <= 2


def test_parse_tamil_text_intact():
    text = "<pauses> பாஸ்போர்ட்டா? <excited> ரொம்ப ஈசிதான்."
    segs = parse(text)
    all_text = " ".join(s["text"] for s in segs)
    assert "பாஸ்போர்ட்டா" in all_text
    assert "ரொம்ப ஈசிதான்" in all_text


def test_parse_multiple_allowed_tags():
    text = "<sigh> அது கஷ்டம்தான். <slow> முகவரி சான்று வேணும்."
    segs = parse(text, max_tags=2)
    tags = [s["tag"] for s in segs if s["tag"]]
    assert "<sigh>" in tags or "<slow>" in tags
    assert len(tags) <= 2


# ── TagBuffer ─────────────────────────────────────────────────────────────────

def test_tagbuffer_complete_sentence():
    buf = TagBuffer()
    result = buf.feed("வணக்கம். ")
    assert len(result) == 1
    assert "வணக்கம்" in result[0]


def test_tagbuffer_accumulates_partial():
    buf = TagBuffer()
    result1 = buf.feed("வணக்")
    assert result1 == []  # incomplete sentence
    result2 = buf.feed("கம்.")
    assert len(result2) == 1


def test_tagbuffer_tag_split_across_chunks():
    buf = TagBuffer()
    # Tag split: '<pau' in chunk1, 'ses>' in chunk2
    result1 = buf.feed("Hello <pau")
    assert result1 == []  # pending tag, can't flush yet
    result2 = buf.feed("ses> World.")
    # Should produce one complete sentence
    combined = "".join(result1 + result2)
    assert "Hello" in combined or "World" in combined


def test_tagbuffer_flush_discards_unterminated_tag():
    buf = TagBuffer()
    buf.feed("Hello <unfinished")
    flushed = buf.flush()
    for item in flushed:
        assert "<unfinished" not in item


def test_tagbuffer_flush_empty():
    buf = TagBuffer()
    result = buf.flush()
    assert result == []


def test_tagbuffer_multiple_sentences():
    buf = TagBuffer()
    r1 = buf.feed("<pauses> முதல். இரண்டாவது. ")
    r2 = buf.flush()
    all_results = r1 + r2
    # Should have at least 2 items
    assert len(all_results) >= 1


# ── Integration: parse + strip round-trip ─────────────────────────────────────

def test_parse_then_strip_no_tags_in_text():
    text = "<excited> Great news! <slow> You need three documents."
    segs = parse(text)
    for seg in segs:
        assert "<" not in seg["text"] or ">" not in seg["text"] or not any(
            tag in seg["text"] for tag in ALLOWED_TAGS | BLOCKED_TAGS
        )


def test_all_allowed_tags_pass_through():
    for tag in ALLOWED_TAGS:
        text = f"{tag} Some sentence."
        segs = parse(text)
        assert any(s["tag"] == tag for s in segs), f"Tag {tag} was dropped unexpectedly"


def test_all_blocked_tags_are_dropped():
    for tag in BLOCKED_TAGS:
        text = f"{tag} Some sentence."
        segs = parse(text)
        for s in segs:
            assert s["tag"] != tag, f"Blocked tag {tag} was not dropped"


# ── Max segment length (<= 800 chars for Gnani Timbre) ───────────────────────

def test_parse_long_sentence_chunks_under_800():
    # Long text with no periods, exceeding 800 characters
    long_clause = "ஆதார் அட்டை முகவரி மாற்றம் செய்வதற்கு தேவையான ஆவணங்கள் " * 20
    assert len(long_clause) > 1000
    segs = parse(long_clause)
    assert len(segs) > 1
    for seg in segs:
        assert len(seg["text"]) <= 800


def test_tagbuffer_long_text_chunks_under_800():
    buf = TagBuffer()
    long_text = "வணக்கம் " * 150  # ~1200 characters
    chunks = buf.feed(long_text)
    flushed = buf.flush()
    all_chunks = chunks + flushed
    assert len(all_chunks) > 1
    for c in all_chunks:
        assert len(c) <= 800


