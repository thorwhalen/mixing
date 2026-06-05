"""Pytest configuration and shared media fixtures for the ``mixing`` test suite.

Provides lightweight, auto-cleaned factories for synthetic audio and video so
individual test modules don't each re-implement temp-file plumbing:

- ``make_tone_audio`` / ``tone_audio`` — a sine-tone audio file (pydub).
- ``make_color_video`` / ``color_video`` — a solid-color video clip (moviepy).

Each factory returns a :class:`pathlib.Path` and registers the file for
deletion at test teardown. Tests needing a backend that isn't installed are
skipped (not failed) via :func:`pytest.importorskip`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Demos and the (soon-to-be-relocated) widget test pull heavy/optional UI deps;
# keep them out of default collection.
collect_ignore_glob = [
    "mixing/audio/*widget*",
    "mixing/audio/demo_*",
]


def _new_tempfile(suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(name)


@pytest.fixture
def make_tone_audio():
    """Factory: write a sine-tone audio file and return its path.

    ``make_tone_audio(duration_s=1.0, freq=440, fmt="mp3", sample_rate=44100,
    channels=1, volume_db=-20)`` -> Path. All created files are removed at
    teardown. Skips the test if ``pydub`` is unavailable.
    """
    pydub = pytest.importorskip("pydub")
    from pydub.generators import Sine

    created: list[Path] = []

    def _make(
        duration_s: float = 1.0,
        *,
        freq: float = 440.0,
        fmt: str = "mp3",
        sample_rate: int = 44100,
        channels: int = 1,
        volume_db: float = -20.0,
    ) -> Path:
        seg = Sine(freq, sample_rate=sample_rate).to_audio_segment(
            duration=duration_s * 1000, volume=volume_db
        )
        if channels == 2:
            seg = seg.set_channels(2)
        path = _new_tempfile(f".{fmt}")
        seg.export(str(path), format=fmt)
        created.append(path)
        return path

    yield _make

    for p in created:
        p.unlink(missing_ok=True)


@pytest.fixture
def tone_audio(make_tone_audio):
    """A 1-second 440 Hz mono mp3 tone (path)."""
    return make_tone_audio(1.0)


@pytest.fixture
def make_color_video():
    """Factory: write a solid-color video clip and return its path.

    ``make_color_video(duration_s=1.0, fps=24, size=(320, 240),
    color=(255, 0, 0), with_audio=False, fmt="mp4")`` -> Path. All created
    files are removed at teardown. Skips the test if ``moviepy`` is unavailable.
    """
    pytest.importorskip("moviepy")
    import moviepy as mp
    from moviepy.video.VideoClip import ColorClip

    created: list[Path] = []

    def _make(
        duration_s: float = 1.0,
        *,
        fps: int = 24,
        size: tuple[int, int] = (320, 240),
        color: tuple[int, int, int] = (255, 0, 0),
        with_audio: bool = False,
        fmt: str = "mp4",
    ) -> Path:
        clip = ColorClip(size=size, color=color, duration=duration_s).with_fps(fps)
        path = _new_tempfile(f".{fmt}")
        if with_audio:
            from moviepy.audio.AudioClip import AudioArrayClip
            import numpy as np

            sr = 44100
            n = int(duration_s * sr)
            t = np.linspace(0, duration_s, n, endpoint=False)
            wave = 0.1 * np.sin(2 * np.pi * 440 * t)
            stereo = np.column_stack([wave, wave])
            clip = clip.with_audio(AudioArrayClip(stereo, fps=sr))
            clip.write_videofile(
                str(path), codec="libx264", audio_codec="aac", logger=None
            )
        else:
            clip.write_videofile(str(path), codec="libx264", audio=False, logger=None)
        clip.close()
        created.append(path)
        return path

    yield _make

    for p in created:
        p.unlink(missing_ok=True)


@pytest.fixture
def color_video(make_color_video):
    """A 1-second 24 fps 320x240 red video without audio (path)."""
    return make_color_video(1.0)
