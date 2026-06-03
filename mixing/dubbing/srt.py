"""SRT subtitle parsing, serialization, and translation.

The transcript side of :mod:`mixing` *generates* SRT from word timestamps
(:func:`mixing.transcript.words_to_srt`); this module reads it back into
structured cues, writes cues out again, and translates the cue text into
another language while preserving timings.

Translation is pluggable: pass any ``translate_fn`` that maps a list of
strings to a list of the same length. The default uses an LLM via the
``aix`` package (provider-agnostic) when it is importable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from mixing.transcript.formats import fmt_srt_time

#: A translator maps (texts, target_language, source_language) -> texts.
TranslateFn = Callable[[Sequence[str], str, "str | None"], list[str]]

_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class Cue:
    """One SRT subtitle cue.

    Attributes:
        index: 1-based cue number.
        start: Start time in seconds.
        end: End time in seconds.
        text: Cue text (may contain embedded newlines).
    """

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Cue duration in seconds (never negative)."""
        return max(0.0, self.end - self.start)


def srt_time_to_seconds(s: str) -> float:
    """Parse an ``HH:MM:SS,mmm`` (or ``.mmm``) SRT timestamp to seconds."""
    h, m, sec, ms = re.split(r"[:,.]", s.strip())
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(srt_text: str) -> list[Cue]:
    """Parse SRT text into a list of :class:`Cue` objects.

    Tolerant of blank-line spacing variations and of either ``,`` or ``.``
    as the millisecond separator. Cues without a valid time line are skipped.
    """
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # The first line may be the numeric index; the time line is the one
        # that matches _TIME_RE (usually line 0 or 1).
        time_idx = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), None)
        if time_idx is None:
            continue
        m = _TIME_RE.search(lines[time_idx])
        start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4].ljust(3, "0")) / 1000.0
        end = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8].ljust(3, "0")) / 1000.0
        idx_line = lines[time_idx - 1] if time_idx > 0 else ""
        index = int(idx_line) if idx_line.strip().isdigit() else len(cues) + 1
        text = "\n".join(lines[time_idx + 1 :]).strip()
        cues.append(Cue(index=index, start=start, end=end, text=text))
    return cues


def dump_srt(cues: Iterable[Cue]) -> str:
    """Serialize cues back to SRT text, renumbering from 1."""
    parts: list[str] = []
    for i, c in enumerate(cues, 1):
        parts.append(
            f"{i}\n{fmt_srt_time(c.start)} --> {fmt_srt_time(c.end)}\n{c.text}\n"
        )
    return "\n".join(parts)


def translate_srt(
    srt: str | Sequence[Cue],
    target_language: str,
    *,
    source_language: str | None = None,
    translate_fn: TranslateFn | None = None,
) -> str:
    """Translate the cue text of an SRT to ``target_language``, keeping timings.

    Args:
        srt: SRT text or a list of :class:`Cue` objects.
        target_language: Human-readable target language (e.g. ``"French"``).
        source_language: Optional source language hint.
        translate_fn: A callable ``(texts, target_language, source_language)
            -> list[str]`` returning one translation per input text, in
            order. Defaults to :func:`default_translate_fn` (LLM-backed).

    Returns:
        Translated SRT text with the original cue timings.

    Raises:
        ValueError: The translator returned a different number of segments
            than it was given.
    """
    cues = parse_srt(srt) if isinstance(srt, str) else list(srt)
    translate_fn = translate_fn or default_translate_fn
    texts = [c.text for c in cues]
    translated = translate_fn(texts, target_language, source_language)
    if len(translated) != len(texts):
        raise ValueError(
            f"Translator returned {len(translated)} segments for {len(texts)} cues; "
            "translate_fn must preserve segment count."
        )
    out = [
        Cue(index=c.index, start=c.start, end=c.end, text=t.strip())
        for c, t in zip(cues, translated)
    ]
    return dump_srt(out)


def default_translate_fn(
    texts: Sequence[str],
    target_language: str,
    source_language: str | None = None,
    *,
    model: str | None = None,
) -> list[str]:
    """LLM-backed translator (segment-count preserving) using ``aix.chat``.

    Translates all segments in a single call, returning a JSON array so the
    one-to-one mapping with the input cues is preserved. Falls back to a
    clear error if ``aix`` is not importable — pass your own ``translate_fn``
    in that case.
    """
    try:
        from aix import chat  # provider-agnostic LLM interface
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "default_translate_fn needs the 'aix' package for LLM translation. "
            "Install aix, or pass a custom translate_fn to translate_srt()."
        ) from e

    import json as _json

    src = f" from {source_language}" if source_language else ""
    numbered = _json.dumps(list(texts), ensure_ascii=False, indent=2)
    prompt = (
        f"You are a professional subtitle translator{src} into {target_language}.\n"
        f"Translate each string in the following JSON array into {target_language}.\n"
        "Preserve meaning and tone (this is marketing copy — keep it natural and "
        "persuasive, not literal). Keep proper nouns and brand names unchanged.\n"
        "Return ONLY a JSON array of the same length, one translated string per "
        "input string, in the same order. No commentary, no code fences.\n\n"
        f"{numbered}"
    )
    raw = chat(prompt, model=model, temperature=0.3)
    return _parse_json_array(raw, expected=len(texts))


def _parse_json_array(raw: str, *, expected: int) -> list[str]:
    """Extract a JSON array of strings from a (possibly fenced) LLM response."""
    import json as _json

    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Grab the outermost [...] if there's surrounding prose.
    if not text.startswith("["):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            text = m[0]
    arr = _json.loads(text)
    if not isinstance(arr, list):
        raise ValueError("Translator response was not a JSON array.")
    return [str(x) for x in arr]
