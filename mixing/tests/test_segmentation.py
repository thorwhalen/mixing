"""Smoke tests for mixing.audio.segmentation.

We build synthetic signals where we know the ground-truth boundaries and
check that each strategy lands close enough to them. Tolerances are loose:
the goal is to catch outright bugs, not to evaluate algorithm quality.
"""

import pytest
import numpy as np

pydub = pytest.importorskip("pydub")
pytest.importorskip("scipy")

from pydub import AudioSegment

from mixing.audio import (
    Segment,
    find_segments,
    extract_segments,
    segment_by_silence,
    segment_by_self_similarity,
    segment_by_speech_music,
)


SR = 22050


def _samples_to_segment(samples: np.ndarray, sr: int = SR) -> AudioSegment:
    """Wrap a float [-1, 1] mono numpy array as a pydub AudioSegment."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)


def _tone(freq: float, duration_s: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _noise(duration_s: float, sr: int = SR, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(duration_s * sr))).astype(np.float32)


def test_segment_dataclass_basics():
    s = Segment(start=2.0, end=5.0, label="x")
    assert s.duration == pytest.approx(3.0)
    assert s.offset == 2.0
    assert s.as_start_end() == (2.0, 5.0)
    assert s.as_offset_duration() == (2.0, 3.0)


def test_silence_strategy_finds_three_tracks():
    # 3 tones separated by 2 s of silence each
    parts = [
        _tone(440, 4.0),
        _silence(2.0),
        _tone(660, 4.0),
        _silence(2.0),
        _tone(880, 4.0),
    ]
    audio = _samples_to_segment(np.concatenate(parts))
    segs = find_segments(
        audio,
        strategy="silence",
        silence_thresh_db=-50,
        min_silence_len=1.0,
    )
    assert len(segs) == 3
    # Tones are roughly 4 s long, allow some slack from the silence detector.
    for s in segs:
        assert 3.0 <= s.duration <= 5.0
        assert s.label == "non_silent"


def test_self_similarity_finds_spectral_change():
    # Two long, spectrally different blocks with constant background noise
    # but no silence between them. Energy-only methods would fail here.
    n = _noise(40.0)
    block1 = _tone(220, 20.0) + n[: 20 * SR]
    block2 = _tone(880, 20.0) + n[20 * SR :]
    audio = _samples_to_segment(np.concatenate([block1, block2]))
    segs = find_segments(
        audio,
        strategy="self_similarity",
        kernel_seconds=6.0,
        min_peak_distance_seconds=10.0,
        peak_threshold_factor=0.5,
    )
    # Expect at least 2 segments and a boundary near 20 s.
    assert len(segs) >= 2
    boundaries = [s.end for s in segs[:-1]]
    assert any(abs(b - 20.0) < 4.0 for b in boundaries), (
        f"expected boundary near 20 s, got {boundaries}"
    )


def test_speech_music_tags_blocks():
    # "Speech": amplitude-modulated noise bursts (mimics syllable structure).
    # "Music": steady tone — low LEFR, low ZCR variance.
    rng = np.random.default_rng(0)
    sr = SR
    speech_dur = 8.0
    t = np.arange(int(speech_dur * sr)) / sr
    # 4 Hz amplitude modulation gives the syllable-rate dips speech has.
    envelope = (np.sin(2 * np.pi * 4 * t) > 0).astype(np.float32)
    speech = (0.3 * rng.standard_normal(len(t)) * envelope).astype(np.float32)
    music = _tone(440, 8.0)
    audio = _samples_to_segment(np.concatenate([speech, music]))

    segs = find_segments(
        audio,
        strategy="speech_music",
        smooth_seconds=2.0,
        min_segment_duration=1.0,
    )
    labels = [s.label for s in segs]
    assert "speech" in labels and "music" in labels, labels
    # The first half should be predominantly speech, second half music.
    first_half_speech = sum(
        s.duration for s in segs if s.label == "speech" and s.start < 8.0
    )
    second_half_music = sum(
        s.duration for s in segs if s.label == "music" and s.start >= 6.0
    )
    assert first_half_speech > 4.0
    assert second_half_music > 4.0


def test_callable_strategy_is_accepted():
    audio = _samples_to_segment(_tone(440, 6.0))

    def custom(seg, *, n=3):
        dur = len(seg) / 1000.0
        return [
            Segment(start=i * dur / n, end=(i + 1) * dur / n, label="custom")
            for i in range(n)
        ]

    segs = find_segments(audio, strategy=custom, n=2)
    assert len(segs) == 2
    assert all(s.label == "custom" for s in segs)


def test_extract_segments_writes_files(tmp_path):
    parts = [_tone(440, 3.0), _silence(2.0), _tone(660, 3.0)]
    audio = _samples_to_segment(np.concatenate(parts))
    paths = extract_segments(
        audio,
        segments=[(0.0, 3.0), (5.0, 8.0)],
        output_dir=tmp_path,
        format="wav",
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_post_processing_min_segment_duration():
    # Shave off short segments via the absorber.
    parts = [_tone(440, 0.3), _silence(2.0), _tone(660, 5.0)]
    audio = _samples_to_segment(np.concatenate(parts))
    segs = find_segments(
        audio,
        strategy="silence",
        silence_thresh_db=-50,
        min_silence_len=1.0,
        min_segment_duration=2.0,
    )
    # The 0.3 s tone gets absorbed into its only neighbor.
    for s in segs:
        assert s.duration >= 2.0 - 1e-6


def test_max_segment_duration_splits():
    audio = _samples_to_segment(_tone(440, 12.0))
    segs = find_segments(
        audio,
        strategy=segment_by_silence,
        silence_thresh_db=-80,
        max_segment_duration=5.0,
    )
    assert len(segs) >= 3
    for s in segs:
        assert s.duration <= 5.0 + 1e-6
