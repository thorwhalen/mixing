"""Tests for the alignment INGEST path — decode/resample quality (issue #25).

``AudioOffset.confidence`` is documented as comparable ACROSS clips. That promise
only holds if the score is a property of the signal, not of the route the bytes
took to reach the aligner. It was not: ``_load_mono_samples`` resampled via
pydub's ``set_frame_rate`` (= ``audioop.ratecv``, linear interpolation with NO
anti-alias filter), so downsampling a 48 kHz camera track to the 16 kHz analysis
rate folded everything above 8 kHz back over the band. Measured on real multicam
footage, the same clip scored roughly HALF the confidence when ingested from the
.mov (pydub decode + naive resample) versus an ffmpeg-extracted 16 kHz WAV —
identical offsets, incomparable scores.

These tests pin the fixed behaviour and are mutation-tests against the naive
path: reintroducing ``set_frame_rate`` in ``_load_mono_samples`` fails BOTH
``test_downsampling_is_anti_aliased`` (70% aliased RMS vs <5%) and
``test_confidence_stable_across_decode_paths`` (~2x confidence gap vs <5%).

All fixtures are synthesized at test time — no committed media. The
container-path test shells out to ffmpeg (a documented system requirement of
this package) and skips when it is absent.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import find_audio_offset_detailed
from mixing.audio.audio_ops import _load_mono_samples

SR_ANALYSIS = 16000
SR_NATIVE = 48000


def _run_ffmpeg(*args: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


def _broadband_reference(seconds: float = 20.0) -> np.ndarray:
    """A non-periodic 'song' at 48 kHz with substantial energy ABOVE 8 kHz.

    The high band is the point: content above the target Nyquist is what a
    naive (non-anti-aliased) 48 kHz -> 16 kHz downsample folds back over the
    band as garbage. A proper resampler removes it; ``audioop.ratecv`` mirrors
    it into the analysis band, decorrelating the two views of the same signal.
    """
    t = np.arange(int(seconds * SR_NATIVE)) / SR_NATIVE
    x = np.zeros_like(t)
    # In-band chirps (the part every decode path should preserve) ...
    for f0, f1 in [(180, 520), (440, 130), (700, 3000), (1200, 6500)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    # ... plus high-band chirps living entirely above the 8 kHz target Nyquist.
    for f0, f1 in [(9000, 20000), (16000, 10000), (21000, 12000)]:
        x += 0.8 * np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)  # rhythmic AM -> onset structure
    return x / np.max(np.abs(x))


def test_downsampling_is_anti_aliased(tmp_path):
    """A tone above the target Nyquist must (almost) vanish, not alias in-band.

    A 10 kHz tone at 48 kHz is above the 8 kHz Nyquist of the 16 kHz analysis
    rate. A proper resampler filters it out (measured RMS ratio ~0.002); pydub's
    ``set_frame_rate`` mirrors it to 6 kHz at ~0.71 RMS ratio. The 0.05 gate
    puts this test squarely between the two — it fails on the naive path.
    """
    seconds = 2.0
    t = np.arange(int(seconds * SR_NATIVE)) / SR_NATIVE
    tone = 0.5 * np.sin(2 * np.pi * 10000 * t)
    p = tmp_path / "tone48.wav"
    wavfile.write(str(p), SR_NATIVE, (tone * 32767).astype(np.int16))

    out = _load_mono_samples(p, SR_ANALYSIS)

    full_scale = 0.5 * 32767  # the tone's own RMS-of-peak scale in int16 units
    rms_ratio = np.sqrt(np.mean(out**2)) / full_scale
    assert rms_ratio < 0.05, (
        f"supra-Nyquist energy survived the downsample (rms ratio {rms_ratio:.3f}) "
        "— the resampler is not anti-aliased"
    )
    # Resampling must preserve duration.
    assert len(out) == pytest.approx(seconds * SR_ANALYSIS, abs=2)


def test_native_rate_load_is_lossless(tmp_path):
    """A file already at the analysis rate passes through untouched (no filter)."""
    rng = np.random.default_rng(0)
    samples = (rng.normal(0, 0.2, SR_ANALYSIS) * 32767).astype(np.int16)
    p = tmp_path / "native16.wav"
    wavfile.write(str(p), SR_ANALYSIS, samples)

    out = _load_mono_samples(p, SR_ANALYSIS)

    np.testing.assert_array_equal(out, samples.astype(np.float64))


def test_confidence_stable_across_decode_paths(tmp_path):
    """Issue #25's acceptance criterion: same audio, WAV vs video container,
    same offset AND confidences within a few percent.

    Mirrors the real pipeline: the reference is a 16 kHz mono WAV in both runs;
    only the QUERY's decode path varies — (a) an ffmpeg-extracted 16 kHz mono
    WAV (no resample on ingest) vs (b) the same audio inside a .mov at its
    native 48 kHz stereo (pydub decode, ingest resamples 48 kHz -> 16 kHz).
    On the naive path (b) scored roughly HALF of (a) on real footage; the fixed
    path must keep them within 5%.
    """
    ref = _broadband_reference()
    offset_s, dur_s = 7.3, 8.0

    # The clip: a segment of the reference with mild independent noise, stereo.
    rng = np.random.default_rng(25)
    seg = ref[int(offset_s * SR_NATIVE) : int((offset_s + dur_s) * SR_NATIVE)].copy()
    p_sig = np.mean(seg**2)
    seg = seg + rng.normal(0, np.sqrt(p_sig / (10**1.5)), len(seg))  # ~15 dB SNR
    seg = seg / (np.max(np.abs(seg)) + 1e-9)
    stereo = np.stack([seg, seg], axis=1)

    ref48 = tmp_path / "ref48.wav"
    clip48 = tmp_path / "clip48.wav"
    wavfile.write(str(ref48), SR_NATIVE, (ref * 32767).astype(np.int16))
    wavfile.write(str(clip48), SR_NATIVE, (stereo * 32767).astype(np.int16))

    # One reference for both runs, extracted the high-quality way.
    ref16 = tmp_path / "ref16.wav"
    _run_ffmpeg("-i", str(ref48), "-ar", str(SR_ANALYSIS), "-ac", "1", str(ref16))
    # Path (a): ffmpeg-extracted 16 kHz mono WAV of the clip.
    clip16 = tmp_path / "clip16.wav"
    _run_ffmpeg("-i", str(clip48), "-ar", str(SR_ANALYSIS), "-ac", "1", str(clip16))
    # Path (b): the clip in a video container at native 48 kHz stereo (PCM, so
    # the codec adds nothing — the decode/resample path is the only variable).
    clip_mov = tmp_path / "clip.mov"
    _run_ffmpeg("-i", str(clip48), "-c:a", "pcm_s16le", str(clip_mov))

    via_wav = find_audio_offset_detailed(
        ref16, clip16, sample_rate=SR_ANALYSIS, feature="envelope"
    )
    via_mov = find_audio_offset_detailed(
        ref16, clip_mov, sample_rate=SR_ANALYSIS, feature="envelope"
    )

    # Offsets: exact (they already were — do not regress them).
    assert via_wav.offset_s == pytest.approx(offset_s, abs=0.05)
    assert via_mov.offset_s == pytest.approx(offset_s, abs=0.05)
    # A real match clears the gate on the reference path.
    assert via_wav.confidence > 0.3
    # THE fix: the score is a property of the signal, not of the decode path.
    rel_gap = abs(via_wav.confidence - via_mov.confidence) / via_wav.confidence
    assert rel_gap < 0.05, (
        f"confidence depends on the decode path: wav={via_wav.confidence:.3f} "
        f"mov={via_mov.confidence:.3f} (relative gap {rel_gap:.1%})"
    )
