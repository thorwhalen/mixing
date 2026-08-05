"""Tests for the ambient-bed primitives: ``loop_audio``, ``duck_audio``,
``overlay_ambient_bed``.

All media is synthesized in-process (sine tones / silence) — no fixtures to
fetch, no network, no paid API. The video test needs moviepy + ffmpeg and
skips cleanly without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pydub = pytest.importorskip("pydub")

from mixing.audio import (  # noqa: E402
    Audio,
    DEFAULT_DUCK_DB,
    duck_audio,
    loop_audio,
)

# A source whose last sample sits near +peak while its first sits at 0: splicing
# it head-to-tail without a crossfade produces a one-sample jump of ~full scale,
# which is exactly the click `loop_audio`'s crossfade exists to remove.
_SEAM_SAMPLE_RATE = 8000
_SEAM_FREQ = 100.0  # 80 samples per cycle at 8 kHz
_SEAM_N_SAMPLES = 8020  # 100.2375 cycles -> ends at sin(0.2375 turn) ~ +0.997


def _segment_from_wave(wave: np.ndarray, *, sample_rate: int) -> "pydub.AudioSegment":  # noqa: F821
    """Float wave in [-1, 1] -> a mono 16-bit pydub AudioSegment."""
    samples = np.clip(np.rint(wave * 32767.0), -32768, 32767).astype(np.int16)
    return pydub.AudioSegment(
        data=samples.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1
    )


def _tone(
    duration_s: float, *, freq: float = 440.0, amplitude: float = 0.5, rate=44100
):
    t = np.arange(int(round(duration_s * rate))) / rate
    return _segment_from_wave(
        amplitude * np.sin(2 * np.pi * freq * t), sample_rate=rate
    )


def _silence(duration_s: float, *, rate: int = 44100):
    return _segment_from_wave(np.zeros(int(round(duration_s * rate))), sample_rate=rate)


def _seam_source() -> Audio:
    t = np.arange(_SEAM_N_SAMPLES) / _SEAM_SAMPLE_RATE
    wave = 0.9 * np.sin(2 * np.pi * _SEAM_FREQ * t)
    return Audio(_segment_from_wave(wave, sample_rate=_SEAM_SAMPLE_RATE))


def _mono_floats(audio: Audio) -> np.ndarray:
    seg = audio._get_segment()
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels).mean(axis=1)
    return samples / float(1 << (8 * seg.sample_width - 1))


def _max_step(audio: Audio) -> float:
    """Largest sample-to-sample jump — a hard splice shows up here."""
    return float(np.max(np.abs(np.diff(_mono_floats(audio)))))


# --------------------------------------------------------------------------
# loop_audio
# --------------------------------------------------------------------------


def test_loop_audio_reaches_target_duration():
    bed = Audio(_tone(0.5))
    looped = loop_audio(bed, 3.0, crossfade_s=0.05)
    assert looped.duration == pytest.approx(3.0, abs=0.01)


def test_loop_audio_trims_a_source_longer_than_target():
    looped = loop_audio(Audio(_tone(4.0)), 1.5)
    assert looped.duration == pytest.approx(1.5, abs=0.01)


def test_loop_audio_crossfade_removes_the_splice_click():
    source = _seam_source()
    # Sanity: the source really does end far from where it starts.
    raw = _mono_floats(source)
    assert abs(raw[-1] - raw[0]) > 0.5

    hard = loop_audio(source, 3.0, crossfade_s=0.0)
    faded = loop_audio(source, 3.0, crossfade_s=0.1)

    # Hard splice: a ~full-scale one-sample jump at every loop join.
    assert _max_step(hard) > 0.5
    # Crossfaded: no jump beyond the tone's own slope (2*pi*f/rate ~ 0.08).
    assert _max_step(faded) < _max_step(hard) / 5
    assert _max_step(faded) < 0.2


def test_loop_audio_clamps_crossfade_to_half_the_source():
    # A crossfade longer than the source would never advance the timeline;
    # clamping keeps it terminating (and correct).
    looped = loop_audio(Audio(_tone(0.2)), 1.0, crossfade_s=5.0)
    assert looped.duration == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("bad_target", [0.0, -1.0])
def test_loop_audio_rejects_non_positive_target(bad_target):
    with pytest.raises(ValueError):
        loop_audio(Audio(_tone(1.0)), bad_target)


def test_loop_audio_rejects_negative_crossfade():
    with pytest.raises(ValueError):
        loop_audio(Audio(_tone(1.0)), 2.0, crossfade_s=-0.1)


def test_loop_audio_honors_the_output_protocol(tmp_path):
    target = tmp_path / "bed.wav"
    returned = loop_audio(Audio(_tone(0.5)), 2.0, output=str(target))
    assert Path(returned) == target and target.exists()
    assert Audio(target).duration == pytest.approx(2.0, abs=0.02)


# --------------------------------------------------------------------------
# duck_audio
# --------------------------------------------------------------------------


def _dialogue_sidechain() -> Audio:
    """Silence, then a loud 1 s "word", then silence again."""
    return Audio(_silence(1.0) + _tone(1.0, freq=300, amplitude=0.7) + _silence(1.0))


def _rms_db(audio: Audio, start_s: float, end_s: float) -> float:
    return audio[start_s:end_s]._get_segment().dBFS


def test_duck_audio_lowers_the_bed_while_the_sidechain_is_loud():
    bed = Audio(_tone(3.0, freq=110, amplitude=0.4))
    ducked = duck_audio(bed, _dialogue_sidechain(), release_s=0.1, hold_s=0.05)

    quiet_region = _rms_db(ducked, 0.2, 0.8)  # sidechain silent
    ducked_region = _rms_db(ducked, 1.3, 1.9)  # sidechain loud
    assert ducked_region == pytest.approx(quiet_region + DEFAULT_DUCK_DB, abs=1.5)


def test_duck_audio_leaves_the_bed_alone_when_the_sidechain_is_silent():
    bed = Audio(_tone(2.0, freq=110, amplitude=0.4))
    ducked = duck_audio(bed, Audio(_silence(2.0)))
    assert ducked._get_segment().dBFS == pytest.approx(bed._get_segment().dBFS, abs=0.3)


def test_duck_audio_preserves_the_bed_duration_with_a_shorter_sidechain():
    bed = Audio(_tone(3.0, freq=110))
    ducked = duck_audio(bed, Audio(_tone(0.5, freq=300, amplitude=0.7)))
    assert ducked.duration == pytest.approx(3.0, abs=0.01)
    # The tail, past the end of the sidechain, is back at full level.
    assert _rms_db(ducked, 2.4, 2.9) == pytest.approx(_rms_db(bed, 2.4, 2.9), abs=0.3)


def test_duck_audio_depth_follows_duck_db():
    bed = Audio(_tone(3.0, freq=110, amplitude=0.4))
    shallow = duck_audio(bed, _dialogue_sidechain(), duck_db=-6.0, release_s=0.1)
    deep = duck_audio(bed, _dialogue_sidechain(), duck_db=-18.0, release_s=0.1)
    assert _rms_db(deep, 1.3, 1.9) < _rms_db(shallow, 1.3, 1.9) - 8.0


@pytest.mark.parametrize("sample_width", [1, 2, 3, 4])
@pytest.mark.parametrize("channels", [1, 2])
def test_duck_audio_survives_any_bed_format(sample_width, channels):
    # The gain envelope goes through numpy, so the bed's sample width / channel
    # count / sample rate must all round-trip — and the sidechain is allowed to
    # differ from the bed in every one of them. (`sample_width=3` is included
    # deliberately: pydub widens 24-bit to 32-bit, so it must still work.)
    bed_segment = _tone(2.0, freq=110, amplitude=0.4)
    bed_segment = bed_segment.set_channels(channels).set_sample_width(sample_width)
    ducked = duck_audio(
        Audio(bed_segment),
        Audio(
            _silence(0.5, rate=22050) + _tone(1.5, freq=300, amplitude=0.7, rate=22050)
        ),
        release_s=0.1,
        attack_s=0.02,
    )
    result = ducked._get_segment()
    assert result.channels == channels
    assert ducked.duration == pytest.approx(2.0, abs=0.01)
    # The duck lands where the sidechain is, not smeared across the timeline —
    # which a mis-deinterleaved stereo envelope would break.
    assert _rms_db(ducked, 0.05, 0.4) == pytest.approx(
        Audio(bed_segment)._get_segment().dBFS, abs=0.5
    )
    assert _rms_db(ducked, 1.0, 1.9) < _rms_db(ducked, 0.05, 0.4) - 6.0


def test_duck_audio_ramps_instead_of_stepping():
    # attack_s / release_s exist so the bed *slides* down and back up. With a
    # slow attack the bed is still near unity just after the sidechain starts
    # and only reaches full depth later; an un-smoothed (instantaneous) gain
    # would be at full depth immediately — audible as a click.
    sidechain = Audio(_silence(0.5) + _tone(2.5, freq=300, amplitude=0.7))
    bed = Audio(_tone(3.0, freq=110, amplitude=0.4))
    ducked = duck_audio(bed, sidechain, attack_s=0.5, hold_s=0.0, release_s=0.5)

    just_after_onset = _rms_db(ducked, 0.5, 0.6)
    well_into_speech = _rms_db(ducked, 2.4, 2.9)
    assert just_after_onset > well_into_speech + 6.0
    # …and the slide really is gradual: the midpoint sits between the two.
    midway = _rms_db(ducked, 1.0, 1.1)
    assert well_into_speech + 1.0 < midway < just_after_onset - 1.0


def test_duck_audio_rejects_a_boosting_duck():
    with pytest.raises(ValueError):
        duck_audio(Audio(_tone(1.0)), Audio(_tone(1.0)), duck_db=3.0)


def test_duck_audio_hold_bridges_a_gap_between_words():
    # Two words with a 0.15 s gap: with hold, the bed stays down across it.
    sidechain = Audio(
        _silence(0.5)
        + _tone(0.4, freq=300, amplitude=0.7)
        + _silence(0.15)
        + _tone(0.4, freq=300, amplitude=0.7)
        + _silence(0.5)
    )
    bed = Audio(_tone(1.95, freq=110, amplitude=0.4))
    common = dict(release_s=0.05, attack_s=0.01)
    with_hold = duck_audio(bed, sidechain, hold_s=0.3, **common)
    without_hold = duck_audio(bed, sidechain, hold_s=0.0, **common)
    gap = (0.92, 1.03)
    assert _rms_db(with_hold, *gap) < _rms_db(without_hold, *gap) - 3.0


# --------------------------------------------------------------------------
# overlay_ambient_bed
# --------------------------------------------------------------------------


def _write(segment, path: Path) -> Path:
    segment.export(str(path), format="wav")
    return path


def test_overlay_ambient_bed_on_audio_loops_to_the_base_duration(tmp_path):
    from mixing.video import overlay_ambient_bed

    base = _write(_tone(3.0, freq=300, amplitude=0.5), tmp_path / "dialogue.wav")
    bed = _write(_tone(0.4, freq=90, amplitude=0.5), tmp_path / "room.wav")

    out = overlay_ambient_bed(base, bed, output=str(tmp_path / "mixed.wav"))
    result = Audio(out)
    assert result.duration == pytest.approx(3.0, abs=0.02)
    # The bed reaches the end: the tail carries the bed's low tone, which the
    # un-looped base does not (a 0.4 s bed would otherwise stop at 0.4 s).
    assert _bed_energy(result, 2.0, 2.9) > _bed_energy(Audio(base), 2.0, 2.9) * 5


def _bed_energy(audio: Audio, start_s: float, end_s: float) -> float:
    """Energy near the bed's 90 Hz tone, via a crude band measurement."""
    samples = _mono_floats(audio[start_s:end_s])
    seg = audio._get_segment()
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / seg.frame_rate)
    band = (freqs > 60) & (freqs < 130)
    return float(np.sum(spectrum[band] ** 2))


def test_overlay_ambient_bed_without_loop_leaves_the_tail_bare(tmp_path):
    from mixing.video import overlay_ambient_bed

    base = _write(_tone(3.0, freq=300, amplitude=0.5), tmp_path / "dialogue.wav")
    bed = _write(_tone(0.4, freq=90, amplitude=0.5), tmp_path / "room.wav")

    looped = Audio(overlay_ambient_bed(base, bed, output=str(tmp_path / "a.wav")))
    once = Audio(
        overlay_ambient_bed(base, bed, loop=False, output=str(tmp_path / "b.wav"))
    )
    assert _bed_energy(once, 2.0, 2.9) < _bed_energy(looped, 2.0, 2.9) / 5


def test_overlay_ambient_bed_fits_a_bed_longer_than_the_media(tmp_path):
    from mixing.video import overlay_ambient_bed

    base = _write(_tone(1.0, freq=300, amplitude=0.5), tmp_path / "dialogue.wav")
    bed = _write(_tone(3.0, freq=90, amplitude=0.5), tmp_path / "room.wav")

    out = Audio(overlay_ambient_bed(base, bed, output=str(tmp_path / "mixed.wav")))
    assert out.duration == pytest.approx(1.0, abs=0.02)


def test_overlay_ambient_bed_default_output_is_beside_the_input(tmp_path):
    from mixing.video import overlay_ambient_bed

    base = _write(_tone(1.0, freq=300), tmp_path / "cut.wav")
    _write(_tone(0.3, freq=90), tmp_path / "room.wav")
    out = Path(overlay_ambient_bed(base, tmp_path / "room.wav"))
    assert out == tmp_path / "cut_ambient.wav" and out.exists()


def test_overlay_ambient_bed_ducks_under_the_existing_dialogue(tmp_path):
    from mixing.video import overlay_ambient_bed

    # Base: silence, loud "dialogue", silence. Bed: a steady low tone.
    base = _write(
        _silence(1.0) + _tone(1.0, freq=300, amplitude=0.7) + _silence(1.0),
        tmp_path / "dialogue.wav",
    )
    bed = _write(_tone(0.5, freq=90, amplitude=0.5), tmp_path / "room.wav")

    plain = Audio(overlay_ambient_bed(base, bed, output=str(tmp_path / "plain.wav")))
    ducked = Audio(
        overlay_ambient_bed(
            base, bed, duck_under_dialogue=True, output=str(tmp_path / "ducked.wav")
        )
    )
    # Under the dialogue the bed's band is quieter when ducking is on ...
    assert _bed_energy(ducked, 1.3, 1.9) < _bed_energy(plain, 1.3, 1.9) / 3
    # ... and unchanged where there is no dialogue.
    assert _bed_energy(ducked, 0.2, 0.8) == pytest.approx(
        _bed_energy(plain, 0.2, 0.8), rel=0.15
    )


def test_overlay_ambient_bed_on_video(tmp_path, make_color_video):
    from mixing.util import has_ffmpeg

    if not has_ffmpeg():
        pytest.skip("ffmpeg not available")
    from mixing.video import overlay_ambient_bed

    video = make_color_video(2.0, with_audio=True)
    bed = _write(_tone(0.4, freq=90, amplitude=0.5), tmp_path / "room.wav")

    out = Path(overlay_ambient_bed(str(video), bed, output=str(tmp_path / "out.mp4")))
    assert out.exists()

    import moviepy as mp

    with mp.VideoFileClip(str(out)) as clip:
        assert clip.audio is not None
        assert clip.duration == pytest.approx(2.0, abs=0.2)


def test_overlay_ambient_bed_on_a_silent_video(tmp_path, make_color_video):
    from mixing.util import has_ffmpeg

    if not has_ffmpeg():
        pytest.skip("ffmpeg not available")
    from mixing.video import overlay_ambient_bed

    video = make_color_video(1.5, with_audio=False)
    bed = _write(_tone(0.4, freq=90, amplitude=0.5), tmp_path / "room.wav")

    out = Path(overlay_ambient_bed(str(video), bed, output=str(tmp_path / "out.mp4")))
    import moviepy as mp

    with mp.VideoFileClip(str(out)) as clip:
        assert clip.audio is not None  # the bed became the track
