"""`overlay_audio` mix_ratio now correctly controls *overlay prominence*.

Before the fix, ``mix_ratio`` was inverted/no-op (0.0 played only the overlay,
>0.5 did nothing). These tests pin the corrected linear-amplitude crossfade by
measuring loudness (dBFS) of the result.
"""

import pytest

from mixing.audio import Audio, overlay_audio


def _dbfs(audio: Audio) -> float:
    return audio._get_segment().dBFS


def _bg_ov(make_tone_audio):
    # Quiet background tone + loud overlay tone, fully overlapping.
    bg = make_tone_audio(2.0, freq=220, volume_db=-30, fmt="wav")
    ov = make_tone_audio(2.0, freq=880, volume_db=-5, fmt="wav")
    return bg, ov


def test_mix_ratio_out_of_range_raises(make_tone_audio):
    bg, ov = _bg_ov(make_tone_audio)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            overlay_audio(str(bg), str(ov), mix_ratio=bad)


def test_mix_ratio_zero_is_background_only(make_tone_audio):
    bg, ov = _bg_ov(make_tone_audio)
    result = overlay_audio(str(bg), str(ov), mix_ratio=0.0)
    # Only the (quiet) background — NOT the loud overlay (the old bug).
    assert _dbfs(result) == pytest.approx(_dbfs(Audio(str(bg))), abs=1.5)


def test_mix_ratio_one_is_overlay_dominant(make_tone_audio):
    bg, ov = _bg_ov(make_tone_audio)
    result = overlay_audio(str(bg), str(ov), mix_ratio=1.0)
    # Overlay dominates the (full) overlap; background is ducked to silence.
    assert _dbfs(result) == pytest.approx(_dbfs(Audio(str(ov))), abs=2.0)


def test_overlay_prominence_increases_with_mix_ratio(make_tone_audio):
    bg, ov = _bg_ov(make_tone_audio)
    low = overlay_audio(str(bg), str(ov), mix_ratio=0.2)
    high = overlay_audio(str(bg), str(ov), mix_ratio=0.8)
    # The loud overlay contributes more as mix_ratio rises (was a no-op above 0.5).
    assert _dbfs(high) > _dbfs(low) + 2.0
