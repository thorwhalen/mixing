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
    LOCATION is ambiguous on it — which is why a lag the envelope nominates has to win on
    score against the waveform's rather than by fiat (see ``_feature_candidates``, and the
    decoy section at the end of this file). Here we test the confidence, which needs onsets.
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
        """Where both features can locate, the feature choice changes only the SCORE.

        This is the common case and it must stay boring: a mild cross-device simulation
        has an unambiguous peak in either domain, so the envelope's nomination and the
        waveform's are the same lag. That the choice CAN move the offset, on material
        where the two domains disagree, is asserted by
        ``test_the_feature_choice_moves_the_offset_and_not_only_the_confidence``.
        """
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


class TestBlindSpots:
    """Each feature is blind in a regime the other sees; the confidence must cover both.

    These pin why the score is the MAX of two correlations rather than the envelope's alone.
    """

    def test_same_source_copy_scores_high_despite_a_useless_envelope(self, tmp_path):
        """An exact slice of the reference must score ~1, even with no onsets to track.

        Onset-free content (smoothly-modulated tones) has no flux structure, so its envelope
        correlation is near noise — measured at 0.08 for an EXACT copy. Scoring on the
        envelope alone would report 0.08 for a perfect match, and at a 0.3 gate that deletes
        the user's footage.
        """
        # Onset-free but NON-periodic: band-limited noise (a sharp, unambiguous
        # autocorrelation peak) under a smooth amplitude contour (no onsets to track).
        # A tone with sinusoidal AM would be onset-free but periodic, so every AM period is
        # an equally valid lag and the offset assertion becomes meaningless.
        rng = np.random.default_rng(11)
        t = np.arange(20 * SR) / SR
        carrier = lfilter(np.ones(24) / 24, [1.0], rng.normal(0, 1, len(t)))
        smooth = carrier * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t))
        smooth /= np.max(np.abs(smooth))
        ref_p = _write(tmp_path, "smooth_ref.wav", smooth)
        clip_p = _write(tmp_path, "smooth_clip.wav", smooth[int(3.0 * SR) : int(13.0 * SR)])
        got = find_audio_offset_detailed(ref_p, clip_p, sample_rate=SR, feature="envelope")
        assert got.offset_s == pytest.approx(3.0, abs=0.05)
        assert got.confidence > 0.9, "a same-source copy must not be scored by its blind spot"

    def test_unrelated_audio_is_low_under_both_features(self, reference):
        """The maximum must not become a way for noise to sneak past.

        Both correlations are low on unrelated material, so their max is low too — on a real
        non-match the pair was max(0.010, 0.021).
        """
        ref_p, _ = reference
        rng = np.random.default_rng(4242)
        noise = _write(reference[0].parent, "pure_noise.wav", rng.normal(0, 0.3, 12 * SR))
        got = find_audio_offset_detailed(ref_p, noise, sample_rate=SR, feature="envelope")
        assert got.confidence < GATE

# --------------------------------------------------------------------------
# WHICH DOMAIN NOMINATES THE CANDIDATE LAGS (issue #30, reopened)
#
# Everything above measures the CONFIDENCE the feature choice produces. This section
# measures the OFFSET it produces, which is the thing that was not actually wired up:
# candidate lags were generated on the raw waveform and the envelope was used only to
# re-score them, so `feature='envelope'` and `feature='waveform'` returned byte-identical
# offsets on every input that exists, and a waveform-domain bias — shared by every window,
# so consensus ratifies rather than cancels it — reached the caller as `support=1.0` for
# an offset 15 s wrong. Measured on real cross-device material in issue #30.
# --------------------------------------------------------------------------

#: The tiling period of the fixture's backing bed, in seconds. Every partial in the bed is
#: an exact multiple of ``1 / BED_LOOP_S`` Hz, so tiling it is seamless — no boundary
#: transient, and therefore no onset the envelope could use to count tiles.
BED_LOOP_S = 4.0
#: Length of the fixture's "song".
DECOY_REFERENCE_S = 60.0
#: Where the fixture's clip really belongs on that song's timeline.
DECOY_TRUE_OFFSET_S = 33.0
#: How much of the song the clip covers.
DECOY_CLIP_S = 10.0
#: Which tile of the bed is mixed the way the clip's device hears it — the passage whose
#: WAVEFORM the clip resembles most, and nowhere near where the clip belongs.
DECOY_TILE = 2
#: The clip's device rolls off above this; the decoy tile is mixed the same way, which is
#: what makes it the waveform's best answer under per-lag normalization.
DEVICE_TOP_HZ = 900.0
#: Silence before the song starts. A recording that begins from silence puts a large onset
#: in the envelope's first frames, and an envelope correlation that is not normalized per
#: lag locks onto it — the trap named in issue #30.
DECOY_LEAD_IN_S = 1.0
#: Analysis windows for the fixture, matching the reproduction in issue #30.
DECOY_WINDOW_S, DECOY_HOP_S = 5.0, 1.25


def _tiling_bed(
    seconds: float, seed: int, *, top_hz: float, loop_s: float = BED_LOOP_S
) -> np.ndarray:
    """A stationary tonal bed that tiles seamlessly at ``loop_s``.

    Stationary is the point: it carries a lot of waveform energy and almost no spectral
    flux, so it dominates a waveform correlation and is invisible to an onset envelope.
    Tiling it makes the reference's WAVEFORM repeat verbatim — issue #30's material — while
    ``top_hz`` models a device that rolls off the highs.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for harmonic in rng.choice(np.arange(400, 8000), size=60, replace=False):
        frequency = harmonic / loop_s
        if frequency > top_hz:
            continue
        x += np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
    return x / np.max(np.abs(x))


def _onset_track(seconds: float, times: np.ndarray, *, hz: float) -> np.ndarray:
    """Short decaying bursts at ``times`` — pure onsets, at a device-specific timbre.

    Two devices playing the same part give bursts at the SAME times with DIFFERENT
    waveforms, which is precisely the split this section exists to exercise: the onset
    envelope sees the times, the waveform correlation sees almost nothing.
    """
    x = np.zeros(int(seconds * SR))
    burst_t = np.arange(int(0.04 * SR)) / SR
    burst = np.sin(2 * np.pi * hz * burst_t) * np.exp(-burst_t / 0.01)
    for onset in times:
        i = int(onset * SR)
        x[i : i + len(burst)] += burst[: len(x) - i]
    return x


@pytest.fixture(scope="module")
def waveform_decoy(tmp_path_factory) -> "tuple[str, str]":
    """``(reference, clip)`` where the waveform's best answer is far from the truth.

    The reference is a stationary bed tiled every :data:`BED_LOOP_S` — so its waveform
    repeats verbatim — with an irregular onset track over it. The clip is another device's
    recording of :data:`DECOY_TRUE_OFFSET_S`: same onset TIMES, different burst timbre,
    and a bed rolled off at :data:`DEVICE_TOP_HZ`. One tile of the reference,
    :data:`DECOY_TILE`, is mixed with that same roll-off, so it — not the truth — is where
    the waveform correlates best.

    Nothing here is tuned to a conclusion: each half follows from what the two features
    claim to read. The waveform reads sample similarity, and the clip's samples genuinely
    do resemble the decoy tile more than they resemble their own home, because a different
    device recorded them. The envelope reads onset TIMING, which the two devices share, and
    the onset pattern is irregular, so it occurs exactly once.
    """
    tiles = int(DECOY_REFERENCE_S / BED_LOOP_S)
    bed = _tiling_bed(BED_LOOP_S, seed=11, top_hz=SR / 2)
    device_bed = _tiling_bed(BED_LOOP_S, seed=11, top_hz=DEVICE_TOP_HZ)
    reference_bed = np.tile(bed, tiles)
    reference_bed[DECOY_TILE * len(bed) : (DECOY_TILE + 1) * len(bed)] = device_bed

    rng = np.random.default_rng(7)
    onsets = np.cumsum(rng.uniform(0.25, 0.9, 300))
    onsets = onsets[onsets < DECOY_REFERENCE_S - 0.1]
    reference = reference_bed + _onset_track(DECOY_REFERENCE_S, onsets, hz=3000)
    reference[: int(DECOY_LEAD_IN_S * SR)] = 0.0

    end_s = DECOY_TRUE_OFFSET_S + DECOY_CLIP_S
    clip_bed = np.tile(device_bed, tiles)[
        int(DECOY_TRUE_OFFSET_S * SR) : int(end_s * SR)
    ]
    heard = onsets[(onsets >= DECOY_TRUE_OFFSET_S) & (onsets < end_s)]
    clip = clip_bed + _onset_track(DECOY_CLIP_S, heard - DECOY_TRUE_OFFSET_S, hz=300)

    out = tmp_path_factory.mktemp("decoy")
    return (
        str(_write(out, "decoy_ref.wav", reference / np.max(np.abs(reference)))),
        str(_write(out, "decoy_clip.wav", clip / np.max(np.abs(clip)))),
    )


def _decoy_alignment(waveform_decoy, **kw):
    reference, clip = waveform_decoy
    (got,) = align_clips_to_reference(
        reference,
        [clip],
        sample_rate=SR,
        window_s=DECOY_WINDOW_S,
        hop_s=DECOY_HOP_S,
        **kw,
    )
    return got


def _decoy_windows(waveform_decoy):
    from mixing.audio.audio_ops import _load_mono_samples, _window_offsets

    reference, clip = waveform_decoy
    return _window_offsets(
        _load_mono_samples(reference, SR),
        _load_mono_samples(clip, SR),
        SR,
        window_s=DECOY_WINDOW_S,
        hop_s=DECOY_HOP_S,
        feature="envelope",
        min_overlap_ratio=0.5,
    )


def test_the_envelope_domain_reaches_an_offset_the_waveform_cannot(waveform_decoy):
    """The headline, and the shape measured on the real material in issue #30.

    Every window's ballot used to be generated on the waveform, so the true offset was on
    none of them and the vote could only ratify the decoy. With the envelope nominating,
    the windows reach the truth unaided — ``support`` of 1.0 here is a real unanimity, the
    thing the 1.0 reported before this fix was pretending to be.
    """
    got = _decoy_alignment(waveform_decoy, feature="envelope")
    assert got.offset_s == pytest.approx(DECOY_TRUE_OFFSET_S, abs=0.05)
    assert got.support == 1.0, "and every window found it on its own"


def test_the_feature_choice_moves_the_offset_and_not_only_the_confidence(
    waveform_decoy,
):
    """The regression test issue #30 asked for by name.

    Before the fix these two were byte-identical on every input that exists, which is the
    one-line proof that ``feature=`` was documented as choosing a locator and in fact chose
    only a scorer. The waveform's own answer is left unpinned beyond "not the truth": which
    wrong peak it lands on is not the contract, that it is free to land on one is.
    """
    envelope = _decoy_alignment(waveform_decoy, feature="envelope")
    waveform = _decoy_alignment(waveform_decoy, feature="waveform")
    assert envelope.offset_s != waveform.offset_s
    assert abs(waveform.offset_s - DECOY_TRUE_OFFSET_S) > DECOY_CLIP_S, (
        "the fixture is only meaningful while the waveform really is misled"
    )


def test_the_references_opening_onset_does_not_capture_the_windows(waveform_decoy):
    """Per-lag normalization, in the envelope domain, is load-bearing (issue #30).

    An envelope built from a recording that starts in silence opens with a large onset. A
    correlation normalized once for the whole surface lets that one spike outscore real
    structure at every lag, and each window then reports that it begins at the start of the
    song. Candidate generation goes through ``_xcorr_surface``, which divides each lag by
    the energy of ITS OWN overlap, so the spike is worth no more than what it overlaps.
    """
    windows = _decoy_windows(waveform_decoy)
    assert len(windows) > 1, "a single window would make unanimity vacuous"
    for window in windows:
        assert window.vote_offset_s == pytest.approx(DECOY_TRUE_OFFSET_S, abs=0.05), (
            f"the window at {window.clip_start_s}s did not find the truth unaided"
        )


def test_a_windows_candidates_are_ordered_by_the_confidence_they_carry(waveform_decoy):
    """The ballot must be sorted by the score printed on it.

    Candidates used to be ordered by the waveform while carrying ``max(waveform,
    envelope)``, so ``candidates[0]`` was not the highest-confidence entry in its own list
    — measured on real material at ``[(53.68, 0.482), (38.56, 0.484)]``. Two features
    nominate now, so there is no single domain whose order could stand in for the score,
    and the score is the order.
    """
    windows = _decoy_windows(waveform_decoy)
    assert any(len(w.candidates) > 1 for w in windows), "nothing to be ordered"
    for window in windows:
        carried = [confidence for _, confidence in window.candidates]
        assert carried == sorted(carried, reverse=True), window.candidates


# --------------------------------------------------------------------------
# The ballot has a fixed number of seats (MAX_CANDIDATE_LAGS)
#
# Found by review of the fix above. A cap that is a plain budget re-creates the defect it
# was part of fixing: on a reference that repeats verbatim the waveform can fill every
# seat with near-tied aliases of one peak, all of them scoring above a cross-device
# envelope match, and the envelope's only nominee falls off the end. The ballot is then
# waveform-only again, silently — no warning, no confidence drop.
# --------------------------------------------------------------------------

#: Bed loop for the eviction fixture. Short enough that a :data:`DECOY_REFERENCE_S`
#: reference holds more verbatim repeats than the ballot has seats — which is the whole
#: point: the waveform must be ABLE to fill it.
ALIAS_LOOP_S = 2.0
#: Tiles left un-rolled-off around the truth, so the passage the clip actually belongs to
#: is not one of the aliases the clip's device response matches.
CLEAN_TILES_AROUND_TRUTH = 5


@pytest.fixture(scope="module")
def alias_flood(tmp_path_factory) -> "tuple[np.ndarray, np.ndarray]":
    """``(reference, clip)`` where the waveform alone can fill the whole ballot.

    Same construction as :func:`waveform_decoy`, wound up: the bed loops every
    :data:`ALIAS_LOOP_S` and EVERY tile except the few around the truth is mixed with the
    clip's device roll-off. So the clip's waveform matches ~25 places equally well and its
    own home not as well, while its onset pattern still occurs exactly once.

    Returned as arrays, not files: the assertions are about what one window nominates, not
    about the decoded-file path.
    """
    tiles = int(DECOY_REFERENCE_S / ALIAS_LOOP_S)
    full = _tiling_bed(ALIAS_LOOP_S, seed=11, top_hz=SR / 2, loop_s=ALIAS_LOOP_S)
    device = _tiling_bed(ALIAS_LOOP_S, seed=11, top_hz=DEVICE_TOP_HZ, loop_s=ALIAS_LOOP_S)
    reference_bed = np.tile(full, tiles)
    home = int(DECOY_TRUE_OFFSET_S / ALIAS_LOOP_S)
    clean = range(home, home + CLEAN_TILES_AROUND_TRUTH)
    for tile in (t for t in range(tiles) if t not in clean):
        reference_bed[tile * len(full) : (tile + 1) * len(full)] = device

    rng = np.random.default_rng(7)
    onsets = np.cumsum(rng.uniform(0.25, 0.9, 300))
    onsets = onsets[onsets < DECOY_REFERENCE_S - 0.1]
    reference = reference_bed + _onset_track(DECOY_REFERENCE_S, onsets, hz=3000)
    reference[: int(DECOY_LEAD_IN_S * SR)] = 0.0

    end_s = DECOY_TRUE_OFFSET_S + DECOY_CLIP_S
    clip_bed = np.tile(device, tiles)[int(DECOY_TRUE_OFFSET_S * SR) : int(end_s * SR)]
    heard = onsets[(onsets >= DECOY_TRUE_OFFSET_S) & (onsets < end_s)]
    clip = clip_bed + _onset_track(DECOY_CLIP_S, heard - DECOY_TRUE_OFFSET_S, hz=300)
    return reference, clip


def test_the_envelopes_nominee_is_not_evicted_by_a_flood_of_waveform_aliases(
    alias_flood,
):
    """The reviewer's case, constructed: the waveform CAN fill the ballot, and does.

    What is asserted is that the envelope's answer is on the ballot at all — not that it
    wins. On a reference where 25 of 30 passages are equally good waveform matches the
    vote is genuinely ambiguous and ``support`` says so; the contract broken before this
    fix was narrower and worse, namely that the answer was not even nominated, so no
    downstream stage could ever choose it.
    """
    from mixing.audio.audio_ops import MAX_CANDIDATE_LAGS, _feature_candidates

    reference, clip = alias_flood
    window = clip[: int(DECOY_WINDOW_S * SR)]
    candidates = _feature_candidates(
        reference,
        window,
        SR,
        feature="envelope",
        min_overlap_ratio=0.5,
        near_tie_ratio=0.05,
        min_separation=int(round(0.25 * SR)),
    )
    assert len(candidates) == MAX_CANDIDATE_LAGS, (
        "the fixture is only meaningful while the ballot is actually full"
    )
    offsets = [lag / SR for lag, _ in candidates]
    assert any(abs(o - DECOY_TRUE_OFFSET_S) < 0.05 for o in offsets), offsets
    # ...and it survived only because a seat was reserved: it is the lowest-scoring entry,
    # so a plain by-score cut at MAX_CANDIDATE_LAGS would have dropped it.
    truth = next(c for c in candidates if abs(c[0] / SR - DECOY_TRUE_OFFSET_S) < 0.05)
    assert truth == min(candidates, key=lambda c: c[1])


def test_a_reserved_seat_cannot_be_spent_by_a_louder_domain():
    """`_best_separated` unit: the same eviction, with the audio taken out of it.

    One domain nominating ``MAX_CANDIDATE_LAGS`` lags that all outscore the other
    domain's single nominee is exactly the shape a verbatim-tiling reference produces.
    Passed as ONE list the loser is dropped; passed as two it is kept, and the cap is
    honoured by dropping the weakest of the flooding domain instead.
    """
    from mixing.audio.audio_ops import MAX_CANDIDATE_LAGS, _best_separated

    step = 1000
    flood = [(i * step, 0.9 - i / 1000) for i in range(1, MAX_CANDIDATE_LAGS + 2)]
    lone = [(500_000, 0.4)]

    flat = _best_separated([flood + lone], min_separation=step)
    assert lone[0] not in flat, "one list is the pre-fix behaviour: the quiet one is cut"

    reserved = _best_separated([flood, lone], min_separation=step)
    assert len(reserved) == MAX_CANDIDATE_LAGS, "the cap is still a cap"
    assert lone[0] in reserved
    carried = [confidence for _, confidence in reserved]
    assert carried == sorted(carried, reverse=True), "and the result stays score-ordered"


def test_a_domain_whose_peak_is_already_on_the_ballot_gets_no_extra_seat():
    """Reservation must not become a way to smuggle a second-choice lag onto the ballot.

    When both domains nominate the same peak the second one is already represented, so it
    is dropped as the duplicate it is rather than promoted to its next candidate.
    """
    from mixing.audio.audio_ops import _best_separated

    agreed = (10_000, 0.8)
    got = _best_separated(
        [[agreed, (30_000, 0.7)], [(10_050, 0.6), (90_000, 0.5)]],
        min_separation=1000,
        max_candidates=2,
    )
    assert got == [agreed, (30_000, 0.7)]
