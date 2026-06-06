"""Characterization tests pinning current behavior of free functions in
``mixing/audio/audio_ops.py``.

These tests lock in the *observable* behavior of the public audio-editing
convenience functions ahead of a refactor:

- ``crop_audio`` — ``output=`` keyword, returned ``Path``, duration of the
  cropped clip, and ``time_unit`` handling (seconds vs milliseconds).
- ``fade_in`` / ``fade_out`` — return ``Audio`` when no ``output`` is
  given, return a ``Path`` (to an existing file) when one is, and preserve the
  source duration.
- ``concatenate_audio`` — concatenating N tones yields ~sum of durations, and a
  positive ``crossfade`` reduces the total. Returns ``Audio`` by default, a
  ``Path`` when ``output=`` is given.
- ``overlay_audio`` — ``mix_ratio`` and ``position`` semantics produce an
  ``Audio`` / a file; overlay never extends the background duration.
- ``save_audio_clip`` — ``output=`` keyword writes a file and returns its
  ``Path``.
- ``find_audio_offset`` — returns a ``float`` offset (seconds) that correctly
  locates a known sub-segment placed at a known offset inside a longer signal.

All audio is synthesized locally via the shared ``make_tone_audio`` fixture; no
network or API keys are touched. ``pydub`` / ``scipy`` are required and the
tests skip gracefully if unavailable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydub")

from mixing.audio import (
    Audio,
    crop_audio,
    fade_in,
    fade_out,
    concatenate_audio,
    overlay_audio,
    save_audio_clip,
    find_audio_offset,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _duration(path) -> float:
    """Duration (seconds) of an audio file via the Audio facade."""
    return Audio(str(path)).duration


def _tmp(suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(name)


# --------------------------------------------------------------------------- #
# crop_audio
# --------------------------------------------------------------------------- #


def test_crop_audio_returns_path_with_expected_duration(make_tone_audio):
    """crop_audio(output=...) returns a Path to an existing file whose
    duration matches the requested [start, end) window (in seconds)."""
    src = make_tone_audio(2.0)
    out = _tmp(".mp3")
    try:
        result = crop_audio(str(src), 0.5, 1.5, output=str(out))
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(1.0, abs=0.15)
    finally:
        out.unlink(missing_ok=True)


def test_crop_audio_time_unit_milliseconds(make_tone_audio):
    """time_unit='milliseconds' interprets start/end as ms: 0..500ms -> ~0.5s."""
    src = make_tone_audio(2.0)
    out = _tmp(".mp3")
    try:
        result = crop_audio(
            str(src), 0, 500, time_unit="milliseconds", output=str(out)
        )
        assert _duration(result) == pytest.approx(0.5, abs=0.15)
    finally:
        out.unlink(missing_ok=True)


def test_crop_audio_time_unit_samples(make_tone_audio):
    """time_unit='samples' interprets start/end as sample indices:
    0..22050 samples at 44.1kHz -> ~0.5s."""
    src = make_tone_audio(2.0, sample_rate=44100)
    out = _tmp(".mp3")
    try:
        result = crop_audio(
            str(src), 0, 22050, time_unit="samples", output=str(out)
        )
        assert _duration(result) == pytest.approx(0.5, abs=0.15)
    finally:
        out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# fade_in / fade_out
# --------------------------------------------------------------------------- #


def test_fade_in_returns_audio_without_output_path(make_tone_audio):
    """fade_in with no output returns an Audio instance (not a Path),
    preserving the source duration."""
    src = make_tone_audio(1.0)
    result = fade_in(str(src), duration=0.2)
    assert isinstance(result, Audio)
    assert result.duration == pytest.approx(1.0, abs=0.15)


def test_fade_in_returns_path_with_output_path(make_tone_audio):
    """fade_in with output writes a file and returns its Path."""
    src = make_tone_audio(1.0)
    out = _tmp(".mp3")
    try:
        result = fade_in(str(src), duration=0.2, output=str(out))
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(1.0, abs=0.15)
    finally:
        out.unlink(missing_ok=True)


def test_fade_out_returns_audio_without_output_path(make_tone_audio):
    """fade_out with no output returns an Audio instance."""
    src = make_tone_audio(1.0)
    result = fade_out(str(src), duration=0.2)
    assert isinstance(result, Audio)
    assert result.duration == pytest.approx(1.0, abs=0.15)


def test_fade_out_returns_path_with_output_path(make_tone_audio):
    """fade_out with output writes a file and returns its Path."""
    src = make_tone_audio(1.0)
    out = _tmp(".mp3")
    try:
        result = fade_out(str(src), duration=0.2, output=str(out))
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(1.0, abs=0.15)
    finally:
        out.unlink(missing_ok=True)


def test_fade_accepts_audio_instance(make_tone_audio):
    """fade_in/fade_out accept an Audio instance as src, not only a path."""
    src = make_tone_audio(1.0)
    audio = Audio(str(src))
    assert isinstance(fade_in(audio, duration=0.1), Audio)
    assert isinstance(fade_out(audio, duration=0.1), Audio)


# --------------------------------------------------------------------------- #
# concatenate_audio
# --------------------------------------------------------------------------- #


def test_concatenate_audio_sum_of_durations(make_tone_audio):
    """Concatenating N tones yields ~ the sum of their durations (Audio)."""
    a = make_tone_audio(0.5)
    b = make_tone_audio(0.7)
    c = make_tone_audio(0.3)
    result = concatenate_audio(str(a), str(b), str(c))
    assert isinstance(result, Audio)
    assert result.duration == pytest.approx(1.5, abs=0.2)


def test_concatenate_audio_crossfade_reduces_duration(make_tone_audio):
    """A positive crossfade overlaps adjacent segments, reducing the total
    duration below the plain sum."""
    a = make_tone_audio(1.0)
    b = make_tone_audio(1.0)
    plain = concatenate_audio(str(a), str(b))
    faded = concatenate_audio(str(a), str(b), crossfade=0.4)
    assert plain.duration == pytest.approx(2.0, abs=0.2)
    # crossfade of 0.4s removes ~0.4s of overlap
    assert faded.duration < plain.duration - 0.2
    assert faded.duration == pytest.approx(1.6, abs=0.2)


def test_concatenate_audio_output_path_returns_path(make_tone_audio):
    """concatenate_audio(output=...) writes a file and returns its Path."""
    a = make_tone_audio(0.5)
    b = make_tone_audio(0.5)
    out = _tmp(".mp3")
    try:
        result = concatenate_audio(str(a), str(b), output=str(out))
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(1.0, abs=0.2)
    finally:
        out.unlink(missing_ok=True)


def test_concatenate_audio_no_sources_raises():
    """Calling concatenate_audio with no sources raises ValueError."""
    with pytest.raises(ValueError):
        concatenate_audio()


# --------------------------------------------------------------------------- #
# overlay_audio
# --------------------------------------------------------------------------- #


def test_overlay_audio_default_mix_returns_audio(make_tone_audio):
    """overlay_audio returns an Audio by default; overlaying onto a background
    never extends the background's duration (mix_ratio=0.5)."""
    bg = make_tone_audio(2.0, freq=220)
    ov = make_tone_audio(1.0, freq=660)
    result = overlay_audio(str(bg), str(ov))
    assert isinstance(result, Audio)
    # overlay is shorter than/equal to background; duration stays the background's
    assert result.duration == pytest.approx(2.0, abs=0.2)


def test_overlay_audio_position_keeps_background_duration(make_tone_audio):
    """Positioning the overlay later in the background does not extend the
    (longer) background's total duration."""
    bg = make_tone_audio(3.0, freq=220)
    ov = make_tone_audio(1.0, freq=660)
    result = overlay_audio(str(bg), str(ov), position=1.0)
    assert isinstance(result, Audio)
    assert result.duration == pytest.approx(3.0, abs=0.2)


def test_overlay_audio_mix_ratio_and_output_path(make_tone_audio):
    """overlay_audio with a non-default mix_ratio and an output writes a
    file and returns its Path."""
    bg = make_tone_audio(2.0, freq=220)
    ov = make_tone_audio(1.0, freq=660)
    out = _tmp(".mp3")
    try:
        result = overlay_audio(
            str(bg), str(ov), mix_ratio=0.3, output=str(out)
        )
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(2.0, abs=0.2)
    finally:
        out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# save_audio_clip
# --------------------------------------------------------------------------- #


def test_save_audio_clip_saveas_returns_path(make_tone_audio):
    """save_audio_clip(audio_src, start, end, output=...) writes a file and
    returns its Path; the clip spans [start, end) seconds."""
    src = make_tone_audio(2.0)
    out = _tmp(".mp3")
    try:
        result = save_audio_clip(str(src), 0.25, 1.25, output=str(out))
        assert isinstance(result, Path)
        assert result.exists()
        assert _duration(result) == pytest.approx(1.0, abs=0.2)
    finally:
        out.unlink(missing_ok=True)


def test_save_audio_clip_no_end_goes_to_end(make_tone_audio):
    """With end=None the clip runs from start to the end of the source."""
    src = make_tone_audio(2.0)
    out = _tmp(".mp3")
    try:
        result = save_audio_clip(str(src), 1.0, None, output=str(out))
        assert _duration(result) == pytest.approx(1.0, abs=0.2)
    finally:
        out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# find_audio_offset
# --------------------------------------------------------------------------- #


def test_find_audio_offset_locates_known_subsegment(make_tone_audio):
    """Build a longer reference signal whose middle contains a known query clip
    at a known offset, and assert the returned offset (seconds) matches it.

    The reference is [lead-in tone @330Hz | query tone @550Hz | tail tone
    @770Hz]; the distinct frequencies guarantee the cross-correlation peak sits
    at the start of the embedded query rather than elsewhere.
    """
    scipy = pytest.importorskip("scipy")  # noqa: F841

    lead = make_tone_audio(1.5, freq=330)
    query = make_tone_audio(1.0, freq=550)
    tail = make_tone_audio(1.0, freq=770)

    # Reference = lead + query + tail, so the query starts at ~1.5s.
    reference = concatenate_audio(str(lead), str(query), str(tail))
    ref_path = _tmp(".wav")
    query_path = _tmp(".wav")
    try:
        reference.save(str(ref_path), format="wav")
        # re-save the query as a standalone file for the search
        Audio(str(query)).save(str(query_path), format="wav")

        offset = find_audio_offset(str(ref_path), str(query_path))
        assert isinstance(offset, float)
        assert offset == pytest.approx(1.5, abs=0.05)
    finally:
        ref_path.unlink(missing_ok=True)
        query_path.unlink(missing_ok=True)


def test_find_audio_offset_zero_when_query_at_start(make_tone_audio):
    """When the query is the prefix of the reference, the offset is ~0s."""
    pytest.importorskip("scipy")

    query = make_tone_audio(1.0, freq=550)
    tail = make_tone_audio(1.0, freq=770)
    reference = concatenate_audio(str(query), str(tail))

    ref_path = _tmp(".wav")
    query_path = _tmp(".wav")
    try:
        reference.save(str(ref_path), format="wav")
        Audio(str(query)).save(str(query_path), format="wav")

        offset = find_audio_offset(str(ref_path), str(query_path))
        assert isinstance(offset, float)
        assert offset == pytest.approx(0.0, abs=0.05)
    finally:
        ref_path.unlink(missing_ok=True)
        query_path.unlink(missing_ok=True)
