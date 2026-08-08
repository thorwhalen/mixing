"""Tests for the alignment FEATURE choice — ``onset_envelope`` and the confidence it gives.

The point of these tests is one measured fact: a raw-waveform cross-correlation coefficient
is not a usable trust gate for CROSS-DEVICE alignment. Two microphones in a room are not
sample-correlated even when the alignment is exact, so the coefficient understates a correct
alignment several-fold — on a real 6-device shoot it scored provably-correct alignments at
0.064-0.148, below any threshold a caller would set, while an onset-envelope score put the
same pairs at 0.43-0.56 and a genuine non-match at 0.018.

``_cross_device`` below simulates that: same source, different colouration, different noise,
different "room". It is a *mild* simulation — mild enough that the waveform feature still
works on it — so these tests assert the RELATIVE improvement the change guarantees, and
leave the absolute gate behaviour where it was actually observed: on real footage. See
:class:`TestCrossDeviceConfidence` for why that distinction is kept rather than engineered
away.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import lfilter

from mixing.audio import (
    align_clips_to_reference,
    find_audio_offset_detailed,
    onset_envelope,
)

SR = 16000
#: The gate muvid's connector applies to a clip's alignment confidence.
GATE = 0.3


def _percussive(seconds: float = 20.0, seed: int = 0) -> np.ndarray:
    """A signal with real ONSETS — the thing an onset envelope exists to track.

    Deliberately NOT the smooth-AM chirp used by test_audio_alignment: a signal whose energy
    is smoothly modulated has no onsets, so its envelope is periodic and envelope-based
    LOCATION is ambiguous on it. (That is why the waveform, not the envelope, chooses the
    lag — see ``_envelope_then_waveform``.) Here we test the confidence, which needs onsets.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros(n)
    t = np.arange(n) / SR
    # Irregularly spaced percussive hits (exponentially-decaying noise bursts).
    hit_times = np.cumsum(rng.uniform(0.12, 0.45, size=int(seconds * 4)))
    for ht in hit_times[hit_times < seconds - 0.2]:
        i = int(ht * SR)
        env = np.exp(-np.arange(SR // 8) / (SR * 0.02))
        x[i : i + len(env)] += rng.normal(0, 1, len(env)) * env
    # A little tonal content so it is not pure noise.
    x += 0.25 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    return x / (np.max(np.abs(x)) + 1e-9)


def _room_impulse_response(rng, *, rt60_s: float = 0.4, direct_gain: float = 1.0) -> np.ndarray:
    """A crude but physically-shaped room impulse response: direct path + a decaying tail.

    Reverberation is the DOMINANT decorrelator between two microphones in one room, and it
    is what a single early reflection under-models: the tail is a dense, position-specific
    random process, so two positions produce waveforms that share a spectrum but not
    samples. This is the mechanism behind the measured 0.06-0.15 waveform coefficients on
    real footage.
    """
    n = int(rt60_s * SR)
    tail = rng.normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)  # -60 dB over rt60
    tail[0] = direct_gain
    return tail / np.sqrt(np.sum(tail**2))


def _cross_device(ref: np.ndarray, *, start_s: float, dur_s: float, seed: int) -> np.ndarray:
    """A different DEVICE's recording of ``ref`` — the case the waveform feature fails on.

    Models the three things that decorrelate two recordings of one sound: a different
    microphone response (a random FIR colouration), a different position in the room (its
    own impulse response — see :func:`_room_impulse_response`), and independent ambient
    noise. The reverberation term is the one that matters; without it the simulation is far
    milder than reality and the waveform coefficient stays misleadingly high.
    """
    rng = np.random.default_rng(seed)
    seg = ref[int(start_s * SR) : int((start_s + dur_s) * SR)].copy()
    # Different position in the room: convolve with this device's own impulse response.
    rir = _room_impulse_response(rng, rt60_s=rng.uniform(0.15, 0.25))
    seg = np.convolve(seg, rir, mode="full")[: len(seg)]
    # Different mic response: a random 6-tap FIR colouration.
    seg = lfilter(rng.normal(0, 1, 6), [1.0], seg)
    # Ambient noise, independent per device.
    seg = seg + rng.normal(0, np.sqrt(np.mean(seg**2)) * 0.35, len(seg))
    return seg / (np.max(np.abs(seg)) + 1e-9)


def _write(tmp_path, name, samples):
    p = tmp_path / name
    wavfile.write(str(p), SR, (samples * 32767).astype(np.int16))
    return p


@pytest.fixture
def reference(tmp_path):
    ref = _percussive()
    return _write(tmp_path, "ref.wav", ref), ref


class TestOnsetEnvelope:
    def test_shape_and_rate(self):
        env, rate = onset_envelope(_percussive(4.0), SR)
        assert rate == pytest.approx(100.0)  # 160-sample hop at 16 kHz
        assert env.size == pytest.approx(4.0 * rate, rel=0.05)

    def test_standardized(self):
        env, _ = onset_envelope(_percussive(4.0), SR)
        assert env.mean() == pytest.approx(0.0, abs=1e-6)
        assert env.std() == pytest.approx(1.0, abs=1e-3)

    def test_onsets_only(self):
        """Decays carry no timing information, so the flux is rectified."""
        env, _ = onset_envelope(_percussive(4.0), SR)
        assert env.max() > 0  # something survived rectification


class TestCrossDeviceConfidence:
    """The regression the whole change exists for.

    **These assert RELATIVE claims deliberately.** A synthetic room cannot reproduce the
    absolute coefficients real footage produced (0.064-0.148 waveform against 0.43-0.56
    envelope, at a 0.3 gate) without the fixture being tuned until it agrees with the
    conclusion — which would prove nothing. What the change actually guarantees, and what
    is therefore asserted here, is that the envelope scores a correct cross-device
    alignment *higher* than the waveform does, and *separates* match from non-match better.
    The absolute gate behaviour is a measured property of real material, recorded in the
    module docstring rather than faked here.
    """

    @pytest.mark.parametrize("seed,start", [(1, 3.0), (2, 6.0), (3, 2.0)])
    def test_envelope_scores_a_correct_alignment_higher_than_the_waveform(
        self, reference, seed, start
    ):
        ref_p, ref = reference
        p = _write(reference[0].parent, f"d{seed}.wav", _cross_device(ref, start_s=start, dur_s=10.0, seed=seed))
        wav = find_audio_offset_detailed(ref_p, p, sample_rate=SR, feature="waveform")
        env = find_audio_offset_detailed(ref_p, p, sample_rate=SR, feature="envelope")
        assert wav.offset_s == pytest.approx(start, abs=0.05), "the LOCATION is right either way"
        assert env.confidence > wav.confidence, "the SCORE is what the waveform gets wrong"

    def test_features_agree_on_the_offset(self, reference):
        """The envelope changes the SCORE, never the reported offset."""
        ref_p, ref = reference
        p = _write(reference[0].parent, "same.wav", _cross_device(ref, start_s=6.0, dur_s=9.0, seed=2))
        wav = find_audio_offset_detailed(ref_p, p, sample_rate=SR, feature="waveform")
        env = find_audio_offset_detailed(ref_p, p, sample_rate=SR, feature="envelope")
        assert env.offset_s == wav.offset_s

    def test_envelope_clears_the_gate_and_noise_does_not(self, reference):
        """A real match must be gate-separable from unrelated audio on the envelope score.

        **What this test deliberately does NOT assert**, having been tried and found false:
        that the envelope's match/noise *ratio* beats the waveform's. On this synthetic
        room the two ratios are indistinguishable (~20 each) — the waveform separates
        perfectly well here. Its failure is specific to REAL cross-device recordings, where
        decorrelation is far stronger than a 6-tap FIR and a 0.2 s tail can reproduce: on a
        real 6-device shoot the waveform put a correct alignment at 0.148 and an unrelated
        clip at 0.006, a spread no threshold can exploit, while the envelope gave 0.431 and
        0.018. Making this fixture reproduce that would mean tuning it until it agreed with
        the conclusion, which would prove nothing — so the real-data claim stays a
        documented measurement and this test asserts only what it can honestly show.
        """
        ref_p, ref = reference
        match = _write(reference[0].parent, "m.wav", _cross_device(ref, start_s=2.0, dur_s=10.0, seed=5))
        rng = np.random.default_rng(99)
        noise = _write(reference[0].parent, "n.wav", rng.normal(0, 0.3, 10 * SR))
        c_match = find_audio_offset_detailed(ref_p, match, sample_rate=SR, feature="envelope").confidence
        c_noise = find_audio_offset_detailed(ref_p, noise, sample_rate=SR, feature="envelope").confidence
        assert c_match > GATE > c_noise


class TestAlignClipsFeature:
    def test_envelope_is_the_default_for_the_multidevice_primitive(self, reference):
        """align_clips_to_reference exists FOR the cross-device case, so it must default
        to the feature that works there."""
        ref_p, ref = reference
        clips = [
            _write(reference[0].parent, f"c{i}.wav", _cross_device(ref, start_s=s, dur_s=8.0, seed=i))
            for i, s in enumerate([1.0, 5.0, 9.0])
        ]
        default = align_clips_to_reference(ref_p, clips, sample_rate=SR)
        explicit = align_clips_to_reference(ref_p, clips, sample_rate=SR, feature="envelope")
        assert [a.confidence for a in default] == [a.confidence for a in explicit]
        waveform = align_clips_to_reference(ref_p, clips, sample_rate=SR, feature="waveform")
        for d, w in zip(default, waveform):
            assert d.offset_s == w.offset_s
            assert d.confidence > w.confidence

    def test_waveform_remains_available(self, reference):
        ref_p, ref = reference
        clips = [_write(reference[0].parent, "w.wav", _cross_device(ref, start_s=4.0, dur_s=8.0, seed=7))]
        (a,) = align_clips_to_reference(ref_p, clips, sample_rate=SR, feature="waveform")
        (b,) = align_clips_to_reference(ref_p, clips, sample_rate=SR, feature="envelope")
        assert a.confidence != b.confidence  # the old scoring is still reachable
        assert a.offset_s == b.offset_s

    @pytest.mark.parametrize("bad", ["", "chroma", "Envelope", None])
    def test_unknown_feature_is_refused_by_name(self, reference, bad):
        ref_p, ref = reference
        clips = [_write(reference[0].parent, "x.wav", _cross_device(ref, start_s=1.0, dur_s=5.0, seed=8))]
        with pytest.raises(ValueError, match="unknown feature"):
            align_clips_to_reference(ref_p, clips, sample_rate=SR, feature=bad)


def test_no_divide_by_zero_warning_on_a_silent_overlap(reference):
    """A real recording reaches denom == 0 at its extreme lags.

    The quotient used to be evaluated for EVERY lag before the mask was applied, so every
    call emitted RuntimeWarning: divide by zero.
    """
    ref_p, ref = reference
    clip = np.concatenate([np.zeros(2 * SR), _cross_device(ref, start_s=0.0, dur_s=6.0, seed=4)])
    p = _write(reference[0].parent, "sil.wav", clip)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        align_clips_to_reference(ref_p, [p], sample_rate=SR)
