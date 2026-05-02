"""Transcript output formats: SRT, plain prose, time remapping."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from mixing.transcript.fillers import (
    DEFAULT_AUDIO_EVENTS_TO_CUT,
    DEFAULT_FILLER_TOKENS,
    is_filler,
)

_SENTENCE_ENDINGS_DEFAULT = ".?!"


def fmt_srt_time(t: float) -> str:
    """Format a time in seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600 * 1000)
    m, ms = divmod(ms, 60 * 1000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(
    words: Sequence[dict],
    *,
    max_chars: int = 80,
    sentence_endings: str = _SENTENCE_ENDINGS_DEFAULT,
) -> str:
    """Render an SRT from a Scribe word list (no filler removal, no remapping)."""
    chunks = list(_chunkify(words, max_chars=max_chars, sentence_endings=sentence_endings))
    return _chunks_to_srt(chunks)


def words_to_srt_remapped(
    words: Sequence[dict],
    cuts: Sequence[dict],
    *,
    max_chars: int = 80,
    sentence_endings: str = _SENTENCE_ENDINGS_DEFAULT,
    drop_fillers: bool = True,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
) -> str:
    """SRT aligned to a post-cut timeline.

    Each word's timestamp is shifted earlier by the cumulative duration
    of all cuts that ended before it, so the resulting SRT drops in over
    the cleaned media.
    """
    chunks = list(
        _chunkify(
            words,
            max_chars=max_chars,
            sentence_endings=sentence_endings,
            drop_fillers=drop_fillers,
            fillers=fillers,
            audio_events=audio_events,
            time_fn=lambda t: remap_time_after_cuts(t, cuts),
        )
    )
    return _chunks_to_srt(chunks)


def words_to_prose(
    words: Sequence[dict],
    *,
    paragraph_pause: float = 1.2,
    drop_fillers: bool = False,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
) -> str:
    """Render a Scribe word list as plain prose, with paragraph breaks on long pauses.

    Args:
        words: Scribe ``words`` array.
        paragraph_pause: Insert a blank line when the gap between two
            consecutive non-filler words exceeds this many seconds.
        drop_fillers: If ``True``, omit filler words and removable audio events.
        fillers / audio_events: Override the default filler / event sets.
    """
    out: list[str] = []
    last_end: float | None = None
    for w in words:
        if w.get("type") != "word":
            continue
        if drop_fillers and is_filler(w, fillers=fillers, audio_events=audio_events):
            continue
        if last_end is not None and w["start"] - last_end > paragraph_pause:
            out.append("\n\n")
        out.append(w["text"])
        last_end = w["end"]
    prose = " ".join(out)
    # tidy whitespace + orphan punctuation, but preserve paragraph breaks (\n\n)
    prose = re.sub(r" +\n\n +", "\n\n", prose)
    prose = re.sub(r"[ \t]+([,.;:!?])", r"\1", prose)
    prose = re.sub(r",\s*([.,;:!?])", r"\1", prose)
    prose = re.sub(r"[ \t]{2,}", " ", prose)
    prose = re.sub(r"\n\n[ \t]+", "\n\n", prose)
    return prose.strip() + "\n"


def remap_time_after_cuts(t: float, cuts: Sequence[dict]) -> float:
    """Map ``t`` from the original timeline onto the post-cut timeline.

    If ``t`` falls inside a cut, snaps to the moment that cut would have
    started in the post-cut timeline.
    """
    cut_before = sum(c["end"] - c["start"] for c in cuts if c["end"] <= t)
    for c in cuts:
        if c["start"] <= t < c["end"]:
            return c["start"] - cut_before
    return t - cut_before


def _chunkify(
    words: Sequence[dict],
    *,
    max_chars: int,
    sentence_endings: str,
    drop_fillers: bool = False,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
    time_fn=None,
):
    endings = tuple(sentence_endings)
    cur: list[str] = []
    cur_s: float | None = None
    cur_e: float = 0.0
    for w in words:
        if w.get("type") == "spacing":
            continue
        if drop_fillers and is_filler(w, fillers=fillers, audio_events=audio_events):
            continue
        s = time_fn(w["start"]) if time_fn else w["start"]
        e = time_fn(w["end"]) if time_fn else w["end"]
        if cur_s is None:
            cur_s = s
        cur_e = e
        cur.append(w["text"])
        joined = " ".join(cur)
        if w["text"].rstrip().endswith(endings) or len(joined) >= max_chars:
            yield (cur_s, cur_e, joined)
            cur, cur_s = [], None
    if cur and cur_s is not None:
        yield (cur_s, cur_e, " ".join(cur))


def _chunks_to_srt(chunks) -> str:
    parts: list[str] = []
    for i, (s, e, text) in enumerate(chunks, 1):
        parts.append(f"{i}\n{fmt_srt_time(s)} --> {fmt_srt_time(e)}\n{text}\n")
    return "\n".join(parts)
