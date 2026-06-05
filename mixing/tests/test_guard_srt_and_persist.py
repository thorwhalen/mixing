"""Characterization tests pinning SRT/time round-trips and ``srt_for_media`` persistence.

These tests lock in the CURRENT observable behavior of two soon-to-be-consolidated
concerns so an upcoming refactor cannot silently change semantics:

(A) :func:`mixing.transcript.persist.srt_for_media` — the "transcribe once, persist,
    reuse" rule. Transcription is INJECTED by monkeypatching
    ``mixing.transcript.persist.transcribe`` to return a tiny fake Scribe dict, so
    NO network call (and no ElevenLabs API key) is ever needed.

(B) The SRT <-> seconds formatting/parsing helpers that exist in THREE modules today
    (``mixing.dubbing.srt``, ``mixing.transcript.formats``,
    ``mixing.video.video_subtitles``). These are about to be consolidated, so we pin
    their exact current outputs, the parse∘format round-trips, and the
    ``parse_srt(dump_srt(cues))`` cue round-trip.

Surprising current behavior encoded below: ``video_subtitles.to_srt_time`` TRUNCATES
the millisecond field (``int(...)``) instead of rounding, so for some inputs (e.g.
``2592.187``) it yields a timestamp one millisecond *earlier* than the rounding-based
``fmt_srt_time`` / ``seconds_to_srt_time``. The refactor should preserve or
deliberately fix this divergence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mixing.transcript import persist
from mixing.transcript.persist import srt_for_media
from mixing.dubbing import srt as dubbing_srt
from mixing.dubbing.srt import Cue, parse_srt, dump_srt
from mixing.transcript import formats
from mixing.video import video_subtitles as vs


# --------------------------------------------------------------------------------------
# Fake Scribe response injected in place of the network transcription call.
# --------------------------------------------------------------------------------------

_FAKE_SCRIBE = {
    "language_code": "eng",
    "text": "Hello world.",
    "words": [
        {"text": "Hello", "type": "word", "start": 0.0, "end": 0.5, "confidence": 0.99},
        {"text": "world.", "type": "word", "start": 0.5, "end": 1.0, "confidence": 0.95},
    ],
}


def _install_fake_transcribe(monkeypatch, *, response=_FAKE_SCRIBE):
    """Replace ``persist.transcribe`` with a no-network stub; return its call log."""
    calls: list = []

    def _fake_transcribe(media, **kwargs):
        calls.append((Path(media), kwargs))
        return response

    monkeypatch.setattr(persist, "transcribe", _fake_transcribe)
    return calls


# --------------------------------------------------------------------------------------
# (A) srt_for_media — persistence / reuse semantics (transcription injected)
# --------------------------------------------------------------------------------------


def test_srt_for_media_writes_next_to_media_then_reuses(tmp_path, monkeypatch):
    """First call transcribes + writes ``<media>.srt``; second call reuses it."""
    calls = _install_fake_transcribe(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not really a video")

    srt_text, srt_path = srt_for_media(media)

    # Written next to the media (same folder, same stem, .srt suffix).
    assert srt_path == media.with_suffix(".srt")
    assert srt_path.exists()
    assert srt_path.read_text(encoding="utf-8") == srt_text
    # Content reflects the injected words (single cue, both words, period ending).
    assert "Hello world." in srt_text
    assert "00:00:00,000 --> 00:00:01,000" in srt_text
    assert len(calls) == 1  # transcription happened exactly once

    # Second call: reuse=True (default) and the .srt exists -> no re-transcription.
    srt_text2, srt_path2 = srt_for_media(media)
    assert srt_text2 == srt_text
    assert srt_path2 == srt_path
    assert len(calls) == 1, "second call must reuse the on-disk SRT, not re-transcribe"


def test_srt_for_media_reuse_returns_existing_srt_untouched(tmp_path, monkeypatch):
    """An existing .srt is returned byte-for-byte without transcribing."""
    calls = _install_fake_transcribe(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    existing = media.with_suffix(".srt")
    hand_corrected = "1\n00:00:00,000 --> 00:00:02,000\nHAND CORRECTED\n"
    existing.write_text(hand_corrected, encoding="utf-8")

    srt_text, srt_path = srt_for_media(media)

    assert srt_path == existing
    assert srt_text == hand_corrected  # untouched, even though words differ
    assert calls == [], "reuse must not call transcribe when the SRT already exists"


def test_srt_for_media_honors_explicit_srt_path(tmp_path, monkeypatch):
    """A provided srt_path is where the SRT is written and later reused."""
    calls = _install_fake_transcribe(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    custom = tmp_path / "subs" / "custom.srt"  # different folder + stem

    srt_text, srt_path = srt_for_media(media, srt_path=custom)

    assert srt_path == custom
    assert custom.exists()
    # The default sibling path was NOT created.
    assert not media.with_suffix(".srt").exists()
    assert len(calls) == 1

    # Reused on a second call against the same explicit path.
    srt_text2, srt_path2 = srt_for_media(media, srt_path=custom)
    assert srt_text2 == srt_text
    assert srt_path2 == custom
    assert len(calls) == 1


def test_srt_for_media_refresh_retranscribes_and_overwrites(tmp_path, monkeypatch):
    """refresh=True forces re-transcription even when the SRT exists."""
    calls = _install_fake_transcribe(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    existing = media.with_suffix(".srt")
    existing.write_text("STALE", encoding="utf-8")

    srt_text, srt_path = srt_for_media(media, refresh=True)

    assert srt_path == existing
    assert srt_text != "STALE"
    assert "Hello world." in srt_text
    assert len(calls) == 1, "refresh must re-transcribe over the existing file"


def test_srt_for_media_reuse_false_retranscribes(tmp_path, monkeypatch):
    """reuse=False ignores an existing SRT and re-transcribes."""
    calls = _install_fake_transcribe(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    existing = media.with_suffix(".srt")
    existing.write_text("STALE", encoding="utf-8")

    srt_text, srt_path = srt_for_media(media, reuse=False)

    assert srt_path == existing
    assert "Hello world." in srt_text
    assert len(calls) == 1


def test_srt_for_media_returns_str_and_path_types(tmp_path, monkeypatch):
    """Return is a (str, Path) tuple."""
    _install_fake_transcribe(monkeypatch)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    result = srt_for_media(media)
    assert isinstance(result, tuple) and len(result) == 2
    srt_text, srt_path = result
    assert isinstance(srt_text, str)
    assert isinstance(srt_path, Path)


def test_srt_for_media_accepts_string_media_path(tmp_path, monkeypatch):
    """A str media argument works and still yields a Path srt_path."""
    _install_fake_transcribe(monkeypatch)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    srt_text, srt_path = srt_for_media(str(media))
    assert isinstance(srt_path, Path)
    assert srt_path == media.with_suffix(".srt")
    assert "Hello world." in srt_text


# --------------------------------------------------------------------------------------
# (B) SRT <-> seconds formatting, parsing, and round-trips across the three modules.
# --------------------------------------------------------------------------------------

# A known seconds value and its canonical SRT timestamp (rounding-based formatters).
_KNOWN_SECONDS = 2592.187
_KNOWN_SRT = "00:43:12,187"


def test_fmt_srt_time_known_value():
    """transcript.formats.fmt_srt_time rounds to the canonical HH:MM:SS,mmm."""
    assert formats.fmt_srt_time(_KNOWN_SECONDS) == _KNOWN_SRT
    assert formats.fmt_srt_time(1.5) == "00:00:01,500"
    assert formats.fmt_srt_time(0.0) == "00:00:00,000"
    assert formats.fmt_srt_time(65.123) == "00:01:05,123"


def test_seconds_to_srt_time_known_value():
    """video_subtitles.seconds_to_srt_time (rounding-based) matches fmt_srt_time."""
    assert vs.seconds_to_srt_time(_KNOWN_SECONDS) == _KNOWN_SRT
    assert vs.seconds_to_srt_time(1.5) == "00:00:01,500"
    assert vs.seconds_to_srt_time(65.123) == "00:01:05,123"


def test_to_srt_time_truncates_milliseconds_quirk():
    """video_subtitles.to_srt_time TRUNCATES ms (current quirk), off-by-one vs rounding.

    For 2592.187, float representation makes ``int((x-int(x))*1000)`` yield 186, not
    187 — one millisecond *earlier* than the rounding-based formatters. This is
    surprising current behavior the refactor should preserve or deliberately fix.
    """
    assert vs.to_srt_time(_KNOWN_SECONDS) == "00:43:12,186"
    # Diverges from the rounding-based formatters for this input.
    assert vs.to_srt_time(_KNOWN_SECONDS) != formats.fmt_srt_time(_KNOWN_SECONDS)
    assert vs.to_srt_time(_KNOWN_SECONDS) != vs.seconds_to_srt_time(_KNOWN_SECONDS)
    # But agrees for "clean" inputs.
    assert vs.to_srt_time(1.5) == "00:00:01,500"
    assert vs.to_srt_time(65.123) == "00:01:05,123"


def test_srt_time_to_seconds_dubbing_known_value():
    """dubbing.srt.srt_time_to_seconds parses the canonical timestamp."""
    assert dubbing_srt.srt_time_to_seconds(_KNOWN_SRT) == pytest.approx(_KNOWN_SECONDS)
    assert dubbing_srt.srt_time_to_seconds("00:00:01,500") == pytest.approx(1.5)
    # Tolerant of a '.' millisecond separator too.
    assert dubbing_srt.srt_time_to_seconds("00:00:01.500") == pytest.approx(1.5)


def test_srt_time_to_seconds_video_subtitles_known_value():
    """video_subtitles.srt_time_to_seconds parses the canonical timestamp."""
    assert vs.srt_time_to_seconds(_KNOWN_SRT) == pytest.approx(_KNOWN_SECONDS)
    assert vs.srt_time_to_seconds("00:00:01,500") == pytest.approx(1.5)


def test_video_subtitles_invalid_time_raises():
    """video_subtitles.srt_time_to_seconds raises ValueError on a non-timestamp."""
    with pytest.raises(ValueError):
        vs.srt_time_to_seconds("not-a-timestamp")


@pytest.mark.parametrize("seconds", [0.0, 1.5, 3.25, 12.75, 65.123, _KNOWN_SECONDS])
def test_parse_after_format_round_trip_dubbing(seconds):
    """seconds -> fmt_srt_time -> dubbing.srt_time_to_seconds recovers the value."""
    formatted = formats.fmt_srt_time(seconds)
    recovered = dubbing_srt.srt_time_to_seconds(formatted)
    assert recovered == pytest.approx(seconds, abs=1e-3)


@pytest.mark.parametrize("seconds", [0.0, 1.5, 3.25, 12.75, 65.123, _KNOWN_SECONDS])
def test_parse_after_format_round_trip_video_subtitles(seconds):
    """seconds -> seconds_to_srt_time -> srt_time_to_seconds recovers the value."""
    formatted = vs.seconds_to_srt_time(seconds)
    recovered = vs.srt_time_to_seconds(formatted)
    assert recovered == pytest.approx(seconds, abs=1e-3)


def test_dump_srt_then_parse_srt_preserves_cues():
    """parse_srt(dump_srt(cues)) preserves cue count, text, and timestamps."""
    cues = [
        Cue(index=1, start=0.0, end=1.5, text="Hello world"),
        Cue(index=2, start=2.0, end=3.25, text="Second line\nwith two rows"),
        Cue(index=3, start=10.0, end=12.75, text="Third"),
    ]
    dumped = dump_srt(cues)
    reparsed = parse_srt(dumped)

    assert len(reparsed) == len(cues)
    for original, got in zip(cues, reparsed):
        assert got.text == original.text
        assert got.start == pytest.approx(original.start, abs=1e-3)
        assert got.end == pytest.approx(original.end, abs=1e-3)
    # dump_srt renumbers cues from 1.
    assert [c.index for c in reparsed] == [1, 2, 3]


def test_dump_srt_renumbers_from_one_ignoring_input_index():
    """dump_srt always renumbers from 1 regardless of the cues' own index field."""
    cues = [
        Cue(index=99, start=0.0, end=1.0, text="A"),
        Cue(index=7, start=1.0, end=2.0, text="B"),
    ]
    dumped = dump_srt(cues)
    # The serialized index lines are 1 and 2, not 99 and 7.
    assert dumped.startswith("1\n")
    assert "\n2\n" in dumped
    reparsed = parse_srt(dumped)
    assert [c.index for c in reparsed] == [1, 2]


def test_words_to_srt_known_words_format():
    """transcript.formats.words_to_srt renders a single cue from two words."""
    words = [
        {"text": "Hello", "type": "word", "start": 0.0, "end": 0.5},
        {"text": "world.", "type": "word", "start": 0.5, "end": 1.0},
    ]
    srt = formats.words_to_srt(words, max_chars=80)
    assert srt == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"


def test_words_to_srt_then_parse_srt_round_trip():
    """words_to_srt output parses back into one cue with matching timing/text."""
    words = [
        {"text": "Hello", "type": "word", "start": 0.0, "end": 0.5},
        {"text": "world.", "type": "word", "start": 0.5, "end": 1.0},
    ]
    srt = formats.words_to_srt(words, max_chars=80)
    cues = parse_srt(srt)
    assert len(cues) == 1
    assert cues[0].text == "Hello world."
    assert cues[0].start == pytest.approx(0.0, abs=1e-3)
    assert cues[0].end == pytest.approx(1.0, abs=1e-3)


def test_cue_duration_property_non_negative():
    """Cue.duration is end-start, clamped at 0 (pins the current property)."""
    assert Cue(index=1, start=1.0, end=3.5, text="x").duration == pytest.approx(2.5)
    # Inverted times clamp to 0 rather than going negative.
    assert Cue(index=1, start=3.0, end=1.0, text="x").duration == 0.0
