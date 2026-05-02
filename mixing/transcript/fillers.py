"""Filler-word detection over Scribe-shaped word lists.

A "word list" here is the ``words`` array in the ElevenLabs Scribe JSON
response — each entry has ``text``, ``start``, ``end``, and ``type``
(one of ``"word"``, ``"spacing"``, ``"audio_event"``).

The two main outputs are:
- :func:`build_cuts`: time ranges to REMOVE (fillers + optional audio events).
- :func:`keeps_from_cuts`: complementary KEEP ranges, suitable for ffmpeg.
"""

from __future__ import annotations

from typing import Iterable, Sequence

DEFAULT_FILLER_TOKENS: frozenset[str] = frozenset(
    {
        "uh",
        "um",
        "umm",
        "mm",
        "mmm",
        "mhm",
        "hmm",
        "ah",
        "er",
        "erm",
        "eh",
        "uhh",
        "uhm",
    }
)
"""Default filler tokens (lowercased, alpha-only, see :func:`normalize_token`)."""

DEFAULT_AUDIO_EVENTS_TO_CUT: frozenset[str] = frozenset({"(coughs)"})
"""Default audio-event tags to remove. ``(laughs)`` is intentionally kept."""


def normalize_token(text: str) -> str:
    """Lowercase and strip non-alpha so ``"Uh,"`` -> ``"uh"``."""
    return "".join(ch for ch in text.lower() if ch.isalpha())


def is_filler(
    item: dict,
    *,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
) -> bool:
    """Return ``True`` if ``item`` is a filler word or a removable audio event."""
    fillers_set = set(fillers)
    audio_events_set = set(audio_events)
    item_type = item.get("type")
    if item_type == "word":
        return normalize_token(item["text"]) in fillers_set
    if item_type == "audio_event":
        return item["text"].strip() in audio_events_set
    return False


def build_cuts(
    words: Sequence[dict],
    *,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
    absorb_trailing_space: bool = True,
    merge_gap: float = 0.08,
) -> list[dict]:
    """Compute time ranges to REMOVE.

    Args:
        words: Scribe ``words`` array.
        fillers: Override the default filler set.
        audio_events: Override the default audio-event set.
        absorb_trailing_space: Extend each cut to the end of the trailing
            ``"spacing"`` token, which avoids leaving a stranded pause.
        merge_gap: Merge adjacent cuts when their gap is below this many seconds.

    Returns:
        List of ``{"start": float, "end": float, "label": str}`` dicts,
        sorted by start time and non-overlapping.
    """
    raw: list[dict] = []
    n = len(words)
    for i, w in enumerate(words):
        if not is_filler(w, fillers=fillers, audio_events=audio_events):
            continue
        start = w["start"]
        end = w["end"]
        if absorb_trailing_space and i + 1 < n:
            nxt = words[i + 1]
            if nxt.get("type") == "spacing":
                end = nxt["end"]
        raw.append({"start": start, "end": end, "label": w["text"]})
    return _merge_adjacent(raw, gap=merge_gap)


def keeps_from_cuts(cuts: Sequence[dict], duration: float) -> list[dict]:
    """Return ranges to KEEP given the cut ranges and total duration."""
    keeps: list[dict] = []
    cursor = 0.0
    for c in cuts:
        if c["start"] > cursor:
            keeps.append({"start": cursor, "end": c["start"]})
        cursor = max(cursor, c["end"])
    if cursor < duration:
        keeps.append({"start": cursor, "end": duration})
    return keeps


def _merge_adjacent(cuts: Sequence[dict], *, gap: float) -> list[dict]:
    merged: list[dict] = []
    for c in cuts:
        if merged and c["start"] - merged[-1]["end"] < gap:
            merged[-1]["end"] = c["end"]
            merged[-1]["label"] = merged[-1]["label"] + " + " + c["label"]
        else:
            merged.append(dict(c))
    return merged
