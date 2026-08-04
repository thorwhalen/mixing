"""Tests for mixing.audio.beat_grid — the librosa-backed beat/onset primitive.

librosa is the ``mixing[beats]`` extra; these tests ``importorskip`` it so the base
test env skips cleanly. ``import mixing.audio`` must NOT pull librosa (lazy import).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_import_mixing_audio_does_not_pull_librosa():
    # A clean subprocess: importing mixing.audio must not import librosa (lazy).
    import subprocess
    import sys

    code = (
        "import sys, mixing.audio; "
        "assert 'librosa' not in sys.modules, 'librosa leaked into import mixing.audio'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


def test_beat_grid_recovers_a_steady_pulse():
    pytest.importorskip("librosa")
    from mixing.audio import beat_grid, BeatGrid

    sr = 22050
    dur = 8.0
    period = 0.5  # 120 BPM
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    # A sharp click every `period` seconds → an unambiguous pulse.
    for onset in np.arange(0.0, dur, period):
        i = int(onset * sr)
        x[i : i + 220] += np.hanning(220) * np.sin(2 * np.pi * 1200 * t[i : i + 220])

    bg = beat_grid(x, sample_rate=sr)
    assert isinstance(bg, BeatGrid)
    # Tempo near 120 BPM (allow the 2x/half-time octave the tracker may pick).
    assert bg.tempo_bpm == pytest.approx(120, abs=8) or bg.tempo_bpm == pytest.approx(
        60, abs=8
    ) or bg.tempo_bpm == pytest.approx(240, abs=16)
    assert len(bg.beat_times) >= 5  # a steady pulse yields several beats
    # onset envelope is a dense per-hop array; frame k is at k*onset_hop_s
    assert bg.onset_env.ndim == 1 and len(bg.onset_env) > 0
    assert bg.onset_hop_s == pytest.approx(512 / sr)
    # beats are ascending and within the clip
    bt = bg.beat_times
    assert np.all(np.diff(bt) > 0)
    assert bt[0] >= 0 and bt[-1] <= dur


def test_beat_grid_rejects_unsupported_backend():
    pytest.importorskip("librosa")
    from mixing.audio import beat_grid

    with pytest.raises(ValueError, match="unsupported beat backend"):
        beat_grid(np.zeros(22050), backend="madmom")


def test_beat_grid_to_dict_is_jsonable():
    pytest.importorskip("librosa")
    import json

    from mixing.audio import beat_grid

    sr = 22050
    x = np.random.default_rng(0).normal(0, 0.1, sr * 3)
    bg = beat_grid(x, sample_rate=sr)
    s = json.dumps(bg.to_dict())  # must not raise (no NaN, arrays → lists)
    assert "beat_times" in json.loads(s)
