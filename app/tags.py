"""
tags.py – Audio-tag parser, validator, segmenter and stripper.

Timbre audio tags are inline tokens in angle brackets that modify TTS delivery.
Tags are case-sensitive.  This module:
  1. parse(text) -> list[Segment]   – split into per-sentence segments, each
                                      carrying at most one leading tag.
  2. strip(text) -> str             – remove all tags (for captions and logs).
  3. Drops unknown or blocked tags.
  4. Caps at 2 tags per full reply (keeps the first 2 valid ones).
  5. Handles tags that may arrive split across streamed chunks.

A Segment is a TypedDict {tag: str | None, text: str}.
"""
from __future__ import annotations

import re
from typing import TypedDict


# ── Tag sets ──────────────────────────────────────────────────────────────────

ALLOWED_TAGS: frozenset[str] = frozenset(
    {
        "<laugh>",
        "<giggle>",
        "<sigh>",
        "<exhales>",
        "<coughs>",
        "<clears throat>",
        "<whispers>",
        "<excited>",
        "<slow>",
        "<pauses>",
    }
)

BLOCKED_TAGS: frozenset[str] = frozenset(
    {
        "<shouting>",
        "<sniffles>",
        "<yawns>",
    }
)

# All known tags (for detecting stray/unknown angle-bracket tokens)
ALL_KNOWN_TAGS: frozenset[str] = ALLOWED_TAGS | BLOCKED_TAGS

# Regex that matches any angle-bracket token (greedy but not crossing >)
_TAG_RE = re.compile(r"<[^>]+>")

# Sentence-ending punctuation (Tamil and Latin)
_SENT_END_RE = re.compile(r"(?<=[.!?।])\s+|(?<=[.!?।])$")


# ── Public types ──────────────────────────────────────────────────────────────

MAX_SEGMENT_CHARS: int = 700  # Timbre TTS supports max 800 chars per segment


class Segment(TypedDict):
    tag: str | None  # the Timbre tag that opens this segment, or None
    text: str        # sentence text with tags already stripped


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_valid_tag(tok: str) -> bool:
    """Return True if tok is an allowed tag (not blocked, not unknown)."""
    return tok in ALLOWED_TAGS


def _chunk_long_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """
    Split text that exceeds max_chars into smaller chunks.
    Prefers splitting on clause boundaries (, ; : \n -) or whitespace.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    while len(text) > max_chars:
        sub = text[:max_chars]
        split_idx = -1
        # Look for commas, semicolons, colons, newlines, dashes
        for delim in (",", ";", ":", "\n", "—", "- "):
            idx = sub.rfind(delim)
            if idx >= max_chars // 3:
                split_idx = max(split_idx, idx + len(delim))

        # If no clause delimiter found, look for whitespace
        if split_idx == -1:
            idx = sub.rfind(" ")
            if idx >= max_chars // 3:
                split_idx = idx + 1

        # Hard cut if no delimiter or whitespace found
        if split_idx <= 0:
            split_idx = max_chars

        part = text[:split_idx].strip()
        if part:
            chunks.append(part)
        text = text[split_idx:].strip()

    if text:
        chunks.append(text)
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on Tamil/Latin punctuation boundaries, chunking oversized parts."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    sentences: list[str] = []
    for p in parts:
        p_str = p.strip()
        if not p_str:
            continue
        if len(p_str) > MAX_SEGMENT_CHARS:
            sentences.extend(_chunk_long_text(p_str, MAX_SEGMENT_CHARS))
        else:
            sentences.append(p_str)
    return sentences


def _extract_leading_tag(sentence: str) -> tuple[str | None, str]:
    """
    If sentence starts with a valid tag, return (tag, rest_of_sentence).
    Otherwise return (None, sentence).
    """
    m = _TAG_RE.match(sentence.lstrip())
    if m:
        tok = m.group(0)
        if _is_valid_tag(tok):
            rest = sentence[sentence.index(tok) + len(tok):].lstrip()
            return tok, rest
    return None, sentence


# ── Public API ────────────────────────────────────────────────────────────────


def strip(text: str) -> str:
    """Remove all angle-bracket tag tokens from *text* (for captions/logs)."""
    result = _TAG_RE.sub("", text)
    # Collapse multiple spaces left behind by tag removal
    result = re.sub(r" {2,}", " ", result)
    return result.strip()


def parse(text: str, *, max_tags: int = 2) -> list[Segment]:
    """
    Parse *text* into a list of Segments, each holding one sentence and at
    most one valid leading tag.  Blocked and unknown tags are silently dropped.
    At most *max_tags* tags survive across the whole reply (default 2).

    Steps:
      1. Replace blocked tags with nothing.
      2. Replace unknown tags with nothing.
      3. Split into sentences.
      4. For each sentence: extract a leading valid tag if present.
      5. Enforce global tag cap (keep only the first max_tags).
    """
    # 1 & 2: strip disallowed tags (blocked + unknown)
    def _replace_tag(m: re.Match) -> str:
        tok = m.group(0)
        if tok in BLOCKED_TAGS or tok not in ALLOWED_TAGS:
            return ""
        return tok  # keep valid tags for now

    cleaned = _TAG_RE.sub(_replace_tag, text)

    # 3: split into sentences
    sentences = _split_sentences(cleaned)
    if not sentences:
        # Fallback: treat the whole cleaned text as one sentence
        sentences = [cleaned.strip()] if cleaned.strip() else []

    # 4 & 5: assign tags to segments, respecting the global cap
    segments: list[Segment] = []
    tags_used = 0

    for sent in sentences:
        tag, body = _extract_leading_tag(sent)
        if tag and tags_used < max_tags:
            tags_used += 1
        else:
            tag = None  # over cap or no valid tag
        # Strip any remaining stray tags from body (shouldn't be any after step 1&2)
        body = strip(body)
        if body:
            segments.append(Segment(tag=tag, text=body))

    return segments


# ── Streamed-chunk buffer ──────────────────────────────────────────────────────


class TagBuffer:
    """
    Accumulates streamed LLM tokens, yielding complete sentences (with tags)
    as they arrive, handling tags that may be split across chunk boundaries.

    Usage:
        buf = TagBuffer()
        for chunk in token_stream:
            for segment_text in buf.feed(chunk):
                process(segment_text)
        for segment_text in buf.flush():
            process(segment_text)
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        """
        Append *chunk* to the internal buffer.  Return a list of complete
        sentence strings ready for `parse()`.  A sentence is complete when
        it ends with `.`, `!`, `?`, or `।` (optionally followed by space).
        An incomplete `<...` at the end of the buffer is left pending.
        """
        self._buf += chunk
        return self._flush_complete()

    def flush(self) -> list[str]:
        """
        Force-flush whatever remains in the buffer (end of turn).
        Any unterminated `<` is discarded (the tag was never closed).
        Oversized chunks are split to guarantee <= MAX_SEGMENT_CHARS.
        """
        # Discard trailing unterminated tag opener
        idx = self._buf.rfind("<")
        if idx != -1 and ">" not in self._buf[idx:]:
            self._buf = self._buf[:idx]
        raw = self._buf.strip()
        self._buf = ""
        if not raw:
            return []
        if len(raw) > MAX_SEGMENT_CHARS:
            return _chunk_long_text(raw, MAX_SEGMENT_CHARS)
        return [raw]

    def _flush_complete(self) -> list[str]:
        """Return all complete sentences from the buffer, leaving the rest."""
        complete: list[str] = []
        i = 0
        in_tag = False
        last_cut = 0

        while i < len(self._buf):
            ch = self._buf[i]
            if ch == "<":
                in_tag = True
            elif ch == ">":
                in_tag = False
            elif ch in ".!?।" and not in_tag:
                next_i = i + 1
                is_end_of_buf = (next_i >= len(self._buf))
                next_is_separator = (
                    not is_end_of_buf
                    and (self._buf[next_i] in " \t\n" or self._buf[next_i] in ".!?।")
                )
                if is_end_of_buf or next_is_separator:
                    sentence = self._buf[last_cut : i + 1].strip()
                    if sentence:
                        if len(sentence) > MAX_SEGMENT_CHARS:
                            complete.extend(_chunk_long_text(sentence, MAX_SEGMENT_CHARS))
                        else:
                            complete.append(sentence)
                    last_cut = i + 1
            elif not in_tag:
                # Early clause break when segment reaches a good length (>= 300 chars)
                seg_len = i - last_cut
                if seg_len >= 300 and ch in ",;\n":
                    next_i = i + 1
                    if next_i >= len(self._buf) or self._buf[next_i] in " \t\n":
                        clause = self._buf[last_cut : i + 1].strip()
                        if clause:
                            complete.append(clause)
                        last_cut = i + 1
                elif seg_len >= MAX_SEGMENT_CHARS and ch in " \t\n":
                    # Hard ceiling break on whitespace to prevent exceeding 800 chars
                    clause = self._buf[last_cut:i].strip()
                    if clause:
                        complete.append(clause)
                    last_cut = i + 1
            i += 1

        # Check for a dangling open tag near the end – don't cut past it
        remaining = self._buf[last_cut:]
        open_idx = remaining.rfind("<")
        if open_idx != -1 and ">" not in remaining[open_idx:]:
            self._buf = remaining
        else:
            self._buf = remaining

        return complete
