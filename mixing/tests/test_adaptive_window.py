"""The windowing `align_clips_to_reference` fits to the clip it is measuring (issue #41).

At the shipped default a 10 s clip against a long reference used to be a single window:
no vote, no second opinion, and a ``support`` of ``None`` that honestly said so — while
the offset it returned was whatever one correlation happened to say. Measured on a real
cross-device shoot, that put one of three clips 102 s from the truth at confidence 0.834,
where every window from 3 s to 6 s put all three on their true offsets.

So the window is now fitted to the clip. These tests pin the two halves of that: a short
clip gets a real vote and a measured support, and a clip long enough to hold the default
window is measured at the default window *exactly* — same offset, same confidence, same
support, to the bit.

Issue #43 closed the other side of the same door. Fitting the window is what the DEFAULT
does; an EXPLICIT window is still never overridden, but one the clip cannot hold is now
refused (:class:`~mixing.errors.WindowTooWideForClip`) instead of quietly delivering the
single-window answer this module exists to have replaced. Several tests below pin that
boundary, and the calls they make are the ones that used to return ``support=None``.
"""

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import align_clips_to_reference
from mixing.errors import WindowTooWideForClip
from mixing.audio.audio_ops import (
    ADAPTIVE_HOP_RATIO,
    MAX_SUPPORT_OVERLAP,
    ADAPTIVE_WINDOW_MIN_FRAMES,
    ENVELOPE_HOP,
    MIN_WINDOWS_FOR_SUPPORT,
    MIN_WINDOWS_FOR_SUPPORT_TARGET,
    SPAN_HOP_S,
    SPAN_WINDOW_S,
    _clip_window_and_hop,
    _independent_windows,
    _window_offsets,
)

SR = 16000

#: The bed tiles at this period, so the reference's WAVEFORM repeats verbatim — the
#: material issue #30 is about, and what makes a single uncorroborated correlation a
#: coin flip rather than a measurement.
BED_LOOP_S = 4.0
#: Long enough that the clips below are a small fraction of it, as on real material.
REFERENCE_S = 96.0
#: Where the short clip really belongs.
TRUE_OFFSET_S = 33.0
#: Short: fewer than one default window, so it had no vote at all before.
SHORT_CLIP_S = 10.0
#: Long: at least :data:`MIN_WINDOWS_FOR_SUPPORT_TARGET` default windows, so the rule
#: leaves it at the default and its answer must not move by one bit.
LONG_CLIP_S = SPAN_WINDOW_S * MIN_WINDOWS_FOR_SUPPORT_TARGET


def _write(path, name, x):
    p = path / name
    wavfile.write(str(p), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return str(p)


def _repeating_bed(seconds: float, *, seed: int, loop_s: float = BED_LOOP_S):
    """A stationary tonal bed that tiles seamlessly at ``loop_s``.

    Every partial is an exact multiple of ``1 / loop_s`` Hz, so the tiling has no seam
    and the waveform genuinely repeats — no window can tell one tile from another.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for harmonic in rng.choice(np.arange(4, 200), size=40, replace=False):
        x += np.sin(2 * np.pi * (harmonic / loop_s) * t + rng.uniform(0, 2 * np.pi))
    return x / np.max(np.abs(x))


def _onset_track(seconds: float, times, *, hz: float):
    """Short decaying bursts at ``times`` — the only thing that is NOT periodic here."""
    x = np.zeros(int(seconds * SR))
    burst_t = np.arange(int(0.04 * SR)) / SR
    burst = np.sin(2 * np.pi * hz * burst_t) * np.exp(-burst_t / 0.01)
    for onset in times:
        i = int(onset * SR)
        x[i : i + len(burst)] += burst[: len(x) - i]
    return x


@pytest.fixture(scope="module")
def repeating_reference(tmp_path_factory):
    """``(reference_path, reference_samples)`` — a bed that repeats, under real onsets.

    A clip cut from this correlates excellently at its true offset AND at every other
    tile of the bed; only the irregular onset pattern separates them. That is what makes
    "how many independent looks agree" the question worth asking, and a single look no
    answer to it.
    """
    tiles = int(REFERENCE_S / BED_LOOP_S)
    bed = np.tile(_repeating_bed(BED_LOOP_S, seed=3), tiles)
    rng = np.random.default_rng(5)
    onsets = np.cumsum(rng.uniform(0.25, 0.9, 400))
    onsets = onsets[onsets < REFERENCE_S - 0.1]
    reference = bed + _onset_track(REFERENCE_S, onsets, hz=3000)
    reference /= np.max(np.abs(reference))
    out = tmp_path_factory.mktemp("adaptive")
    return _write(out, "reference.wav", reference), reference


def _excerpt(path, reference, name, start_s, duration_s, *, seed):
    """Another device's recording of ``[start_s, start_s + duration_s)``.

    Same content, plus a little independent noise — enough that the match is not the
    trivially perfect one, not enough to hide it.
    """
    rng = np.random.default_rng(seed)
    take = reference[int(start_s * SR) : int((start_s + duration_s) * SR)]
    return _write(path, name, take + rng.normal(0, 0.02, len(take)))


# --------------------------------------------------------------------------
# The rule itself: what window a clip of a given length is measured at.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clip_duration_s, expect_window_s",
    [
        (250.0, SPAN_WINDOW_S),  # the real reference's length: the default, untouched
        # exactly three default windows — still the default
        (LONG_CLIP_S, SPAN_WINDOW_S),
        (LONG_CLIP_S - 1.0, (LONG_CLIP_S - 1.0) / MIN_WINDOWS_FOR_SUPPORT_TARGET),
        (10.0, 10.0 / MIN_WINDOWS_FOR_SUPPORT_TARGET),  # issue #41's clip: 3.33 s
        (6.0, ADAPTIVE_WINDOW_MIN_FRAMES * ENVELOPE_HOP / SR),  # floored, not 2.0
        # floored, and longer than the clip itself
        (1.0, ADAPTIVE_WINDOW_MIN_FRAMES * ENVELOPE_HOP / SR),
    ],
)
def test_the_window_a_clip_is_measured_at(clip_duration_s, expect_window_s):
    """``min(default, clip / target)``, floored at what the onset envelope can carry.

    The floor is the half that keeps this honest. Dividing all the way down would hand
    any clip a unanimous vote of windows too short to carry an onset pattern — support
    bought by making each opinion worthless. Under three floor-lengths the window stops
    shrinking with the clip, and a clip no longer than the floor itself is left as one
    window — which reports ``support=None`` and says so.
    """
    window_s, hop_s = _clip_window_and_hop(
        clip_duration_s, SR, window_s=None, hop_s=None
    )
    assert window_s == pytest.approx(expect_window_s)
    assert hop_s == pytest.approx(window_s * ADAPTIVE_HOP_RATIO)


def test_the_floor_follows_the_analysis_rate_not_the_clock():
    """The floor is a count of envelope FRAMES, so it moves with the sample rate.

    :data:`ENVELOPE_HOP` is a number of samples, so at half the analysis rate the same
    number of frames is twice as many seconds. A floor written in seconds would silently
    become a different number of frames — the quantity that actually decides whether a
    window carries enough onset structure to correlate.
    """
    at_full, _ = _clip_window_and_hop(1.0, SR, window_s=None, hop_s=None)
    at_half, _ = _clip_window_and_hop(1.0, SR // 2, window_s=None, hop_s=None)
    assert at_half == pytest.approx(2 * at_full)


def test_an_explicit_window_is_never_overridden():
    """The caller has said what a window means for their material; the rule has not.

    Stated on a clip the window FITS. It used to be stated on a 5 s clip, where the
    window could not be honored at all — the rule left it alone there too, and the
    result was a single-window answer with ``support=None`` and nothing saying the
    window was the reason. That case is now a refusal (issue #43, below); what survives
    unchanged is the part that was ever meaningful: where the window can be honored, it
    is, exactly as given.
    """
    assert _clip_window_and_hop(60.0, SR, window_s=20.0, hop_s=10.0) == (20.0, 10.0)
    assert _clip_window_and_hop(60.0, SR, window_s=20.0, hop_s=None) == (
        20.0,
        20.0 * ADAPTIVE_HOP_RATIO,
    )


def test_an_explicit_window_the_clip_cannot_hold_is_refused():
    """Issue #43: the boundary of "never overridden" is a refusal, not a silent fallback.

    Above :func:`_max_supportable_window_s` the clip holds no second INDEPENDENT window,
    so the vote the caller named cannot be held. Neither honoring it (one correlation,
    ``support=None``, no reason given) nor clamping it (a support on a scale nobody asked
    for) is reportable, so the call refuses and the error carries the largest window that
    would work.

    **The bound is a function of the clip and the overlap rule, NOT of the hop.** A
    hop-based bound is the obvious guess and it is wrong in both directions — the two
    counterexamples below were measured against the support
    ``align_clips_to_reference`` actually returns, and they are pinned here because the
    guess is what a future reader will reach for again.
    """
    supportable = 15.0 / (1 + MAX_SUPPORT_OVERLAP)  # 10.0 s

    # At the bound and below: honored exactly as given, whatever the hop.
    assert _clip_window_and_hop(15.0, SR, window_s=supportable, hop_s=3.0) == (
        supportable,
        3.0,
    )

    with pytest.raises(WindowTooWideForClip) as excinfo:
        _clip_window_and_hop(15.0, SR, window_s=11.0, hop_s=1.0, clip_index=3)
    err = excinfo.value
    assert (err.clip_index, err.window_s, err.hop_s) == (3, 11.0, 1.0)
    assert err.clip_duration_s == 15.0
    assert err.max_window_s == pytest.approx(supportable)
    assert "clip 3" in str(err)

    # Counterexample 1 — a TINY hop does not buy a second look. `window_s=11, hop_s=1`
    # on a 15 s clip passes `window <= duration - hop` and returns support=None: the
    # windows the small hop produces overlap far too much to count as second opinions.
    # (Same call as above; stated separately because this is what it is pinning.)
    assert 11.0 <= 15.0 - 1.0, "the hop-based bound would have accepted this"

    # Counterexample 2 — a hop LONGER than the clip does not cost one. `window_s=6,
    # hop_s=12` on a 15 s clip fails `window <= duration - hop`, yet measures
    # support=1.0: the grid always appends a tail window at `len(clip) - n_win`, so a
    # second look exists no matter how large the hop.
    assert 6.0 > 15.0 - 12.0, "the hop-based bound would have refused this"
    assert _clip_window_and_hop(15.0, SR, window_s=6.0, hop_s=12.0) == (6.0, 12.0)

    # The hop does not enter the ceiling either — same clip, same window, four hops.
    ceilings = {
        _clip_window_and_hop_error(15.0, window_s=14.0, hop_s=h).max_window_s
        for h in (0.5, 3.0, 7.0, 30.0, None)
    }
    assert len(ceilings) == 1 and ceilings.pop() == pytest.approx(supportable)


def _clip_window_and_hop_error(clip_duration_s, **kwargs):
    """The :class:`WindowTooWideForClip` a refused call raises — a test-local helper."""
    with pytest.raises(WindowTooWideForClip) as excinfo:
        _clip_window_and_hop(clip_duration_s, SR, **kwargs)
    return excinfo.value


def test_the_refused_window_is_exactly_the_one_that_would_have_no_support():
    """The guard's verdict must equal what the estimator would actually report.

    A guard that is merely *nearly* right is worse than none: it refuses calls that
    would have measured a support (a false refusal the caller cannot work around) and
    lets through calls that return the unexplained ``None`` issue #43 is about. So the
    predicate is not an approximation of the rule — it IS the rule
    (:func:`_independent_windows`' own test on the two windows that always exist),
    evaluated on the same rounded sample counts the grid is built from.

    Checked here against :func:`_window_offsets` + :func:`_independent_windows`
    directly, over a grid of durations, windows and hops, rather than against the
    formula restated.
    """
    for clip_duration_s in (6.0, 10.0, 12.5, 15.0, 22.0):
        reference = np.zeros(int(90 * SR))
        clip = np.zeros(int(clip_duration_s * SR))
        for window_s in (1.0, 3.33, 5.0, 7.0, 9.0, 11.0, 20.0):
            for hop_s in (0.5, 2.0, 8.0, 30.0):
                windows = _window_offsets(
                    reference,
                    clip,
                    SR,
                    window_s=window_s,
                    hop_s=hop_s,
                    feature="envelope",
                    min_overlap_ratio=0.5,
                )
                would_support = (
                    len(_independent_windows(windows)) >= MIN_WINDOWS_FOR_SUPPORT
                )
                try:
                    _clip_window_and_hop(
                        clip_duration_s, SR, window_s=window_s, hop_s=hop_s
                    )
                    accepted = True
                except WindowTooWideForClip:
                    accepted = False
                assert accepted is would_support, (
                    f"guard says {accepted} but {clip_duration_s} s at "
                    f"window_s={window_s}, hop_s={hop_s} would "
                    f"{'have' if would_support else 'have no'} support"
                )


def test_the_ceiling_the_error_names_is_one_the_call_accepts():
    """The remedy must work, and half a sample of rounding is enough to break it.

    ``max_window_s`` exists to be retried at, so a value even one rounding step outside
    the bound sends the caller straight back into the exception. The seconds expression
    does exactly that at some durations: on a 10 s clip at 16 kHz, ``10 / 1.5`` rounds
    UP to 106667 samples where 106666 is the most that fits. The bound is therefore
    taken in samples, and this pins the property rather than the expression.
    """
    for clip_duration_s in (6.0, 8.0, 10.0, 12.5, 15.0, 22.0, 30.0, 47.3):
        err = _clip_window_and_hop_error(
            clip_duration_s, window_s=clip_duration_s, hop_s=1.0
        )
        assert err.max_window_s > 0, "some window always holds a second look"
        # The ceiling itself is accepted...
        _clip_window_and_hop(clip_duration_s, SR, window_s=err.max_window_s, hop_s=1.0)
        # ...and it really is the ceiling: one sample more is not.
        with pytest.raises(WindowTooWideForClip):
            _clip_window_and_hop(
                clip_duration_s, SR, window_s=err.max_window_s + 1 / SR, hop_s=1.0
            )
        # The number the MESSAGE names is rounded down, so it is admissible too.
        advised = float(str(err).split("pass window_s <= ")[1].split(",")[0])
        _clip_window_and_hop(clip_duration_s, SR, window_s=advised, hop_s=1.0)


#: The smallest clip that can hold two half-window-separated looks at the floor window:
#: one window, then a second starting a hop later. Below it no windowing can produce a
#: quorum and ``support=None`` is the true answer, not a miss.
SUPPORTABLE_FROM_S = (ADAPTIVE_WINDOW_MIN_FRAMES * ENVELOPE_HOP / SR) * (
    1 + ADAPTIVE_HOP_RATIO
)


@pytest.mark.parametrize(
    "clip_duration_s", [SUPPORTABLE_FROM_S, 6.0, 10.0, 22.0, 24.0, 26.0, 30.0, 250.0]
)
def test_a_fitted_grid_always_holds_a_quorum_of_independent_windows(clip_duration_s):
    """The guarantee, stated on the windows themselves rather than on a clip's answer.

    ``support`` is ``None`` whenever fewer than :data:`MIN_WINDOWS_FOR_SUPPORT` windows are
    INDEPENDENT (:data:`MAX_SUPPORT_OVERLAP`), and that bites well above one window's
    length: at the default 20 s/10 s grid a clip needs about ``window_s + hop_s`` before a
    second look exists at all, so 22, 24 and 26 s clips all read ``None`` — measured, with
    the 22 s one landing 64 s from the truth at confidence 0.279 while ``window_s=10`` got
    it right. A rule that only rescued clips shorter than one window would have missed
    every one of them.

    Counted here on real windowing (:func:`_window_offsets` over silence, whose CONTENT is
    irrelevant — only the grid is under test), not on arithmetic repeated from the rule.
    """
    window_s, hop_s = _clip_window_and_hop(
        clip_duration_s, SR, window_s=None, hop_s=None
    )
    reference = np.zeros(int(max(2 * clip_duration_s, 90) * SR))
    windows = _window_offsets(
        reference,
        np.zeros(int(clip_duration_s * SR)),
        SR,
        window_s=window_s,
        hop_s=hop_s,
        feature="waveform",
        min_overlap_ratio=0.5,
    )
    assert len(_independent_windows(windows)) >= MIN_WINDOWS_FOR_SUPPORT


def test_below_the_supportable_length_no_window_can_hold_a_quorum():
    """And there the honest answer is still ``None`` — the floor is not a failure.

    A clip shorter than one floor-window plus a hop cannot hold two looks that are not
    each other's echo, whatever window it is measured at. Shrinking below the floor to
    manufacture a vote would buy the quorum with windows too short to carry an onset
    pattern, which is the one thing worse than reporting nothing.
    """
    window_s, hop_s = _clip_window_and_hop(
        SUPPORTABLE_FROM_S - 0.5, SR, window_s=None, hop_s=None
    )
    reference = np.zeros(int(90 * SR))
    windows = _window_offsets(
        reference,
        np.zeros(int((SUPPORTABLE_FROM_S - 0.5) * SR)),
        SR,
        window_s=window_s,
        hop_s=hop_s,
        feature="waveform",
        min_overlap_ratio=0.5,
    )
    assert len(_independent_windows(windows)) < MIN_WINDOWS_FOR_SUPPORT


def test_the_target_leaves_room_for_the_excluded_tail_window():
    """Aiming AT the quorum would miss it whenever a clip has a tail window.

    The tail window :func:`_window_offsets` appends for a clip that is not a whole number
    of hops starts less than half a window after its neighbour, so the support tally
    excludes it. A target equal to :data:`MIN_WINDOWS_FOR_SUPPORT` would therefore leave
    such a clip one look short of a quorum and report ``None`` again.
    """
    assert MIN_WINDOWS_FOR_SUPPORT_TARGET > MIN_WINDOWS_FOR_SUPPORT


# --------------------------------------------------------------------------
# What that does to an actual alignment.
# --------------------------------------------------------------------------


def test_a_short_clip_now_gets_a_vote_and_a_measured_support(
    repeating_reference, tmp_path
):
    """Issue #41's acceptance: the true offset, with a support that is a number.

    At the default the clip is measured at a window fitted to it, so several independent
    windows look at it and agree. The comparison is with the same call at the default
    *pair* — which is what this clip used to get. That call returned one window and an
    honest ``support=None``; since issue #43 it is refused outright, which is the same
    statement made where the caller can see it: this clip cannot be measured at 20 s.
    """
    reference_path, reference = repeating_reference
    clip = _excerpt(
        tmp_path, reference, "short.wav", TRUE_OFFSET_S, SHORT_CLIP_S, seed=17
    )

    (fitted,) = align_clips_to_reference(reference_path, [clip], sample_rate=SR)
    with pytest.raises(WindowTooWideForClip):
        align_clips_to_reference(
            reference_path,
            [clip],
            sample_rate=SR,
            window_s=SPAN_WINDOW_S,
            hop_s=SPAN_HOP_S,
        )

    assert fitted.offset_s == pytest.approx(TRUE_OFFSET_S, abs=0.05)
    assert fitted.support is not None
    assert 0.0 < fitted.support <= 1.0


def test_a_clip_just_LONGER_than_one_default_window_is_rescued_too(
    repeating_reference, tmp_path
):
    """The case that is easy to think is already handled, and is not.

    22 s is longer than the default 20 s window, so it is not "a clip shorter than one
    window" — and it still had no support, because the second window the grid gives it
    starts 2 s after the first and is that first window's echo, not a second opinion.
    Measured at the default on real material: 22, 24 and 26 s clips against a 90 s
    reference all read ``None``, and the 22 s one landed 64 s from the truth at confidence
    0.279 where ``window_s=10`` was right to 3 ms. The rule is written on independent
    LOOKS for exactly this reason, not on whether the clip fits in a window.
    """
    reference_path, reference = repeating_reference
    clip = _excerpt(tmp_path, reference, "just_over.wav", TRUE_OFFSET_S, 22.0, seed=29)

    (fitted,) = align_clips_to_reference(reference_path, [clip], sample_rate=SR)
    # Longer than a window, still only one INDEPENDENT look — and since issue #43 that
    # is refused rather than answered with an unexplained ``support=None``.
    with pytest.raises(WindowTooWideForClip) as excinfo:
        align_clips_to_reference(
            reference_path,
            [clip],
            sample_rate=SR,
            window_s=SPAN_WINDOW_S,
            hop_s=SPAN_HOP_S,
        )
    assert excinfo.value.max_window_s < SPAN_WINDOW_S

    assert fitted.offset_s == pytest.approx(TRUE_OFFSET_S, abs=0.05)
    assert fitted.support is not None
    assert 0.0 < fitted.support <= 1.0


def test_a_long_clip_is_measured_at_the_default_window_exactly(
    repeating_reference, tmp_path
):
    """A clip that holds the default window is not adapted at all — bit for bit.

    Not ``approx``: the rule must be a no-op above its threshold, and an offset that
    agrees to five decimals would hide a window that quietly moved.
    """
    reference_path, reference = repeating_reference
    clip = _excerpt(tmp_path, reference, "long.wav", 5.0, LONG_CLIP_S, seed=19)

    (fitted,) = align_clips_to_reference(reference_path, [clip], sample_rate=SR)
    (pinned,) = align_clips_to_reference(
        reference_path, [clip], sample_rate=SR, window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S
    )

    assert (fitted.offset_s, fitted.confidence, fitted.support) == (
        pinned.offset_s,
        pinned.confidence,
        pinned.support,
    )
    assert fitted.offset_s == pytest.approx(5.0, abs=0.05)


def test_the_window_support_is_relative_to_comes_back_with_it(
    repeating_reference, tmp_path
):
    """A fitted scale the caller cannot see is a support fraction they cannot read.

    Support is relative to ``window_s`` and deliberately not normalised, so a gate on it
    is relative to ``window_s`` too. Fitting the window per clip therefore takes that
    scale out of the caller's own arguments — and a threshold applied across clips
    measured at different windows compares numbers that answer different questions.
    Measured on real cross-device material: 21 alignments that were all CORRECT reported
    support anywhere from 0.00 to 1.00, largely by clip length. So the window each clip
    was measured at is reported back with the answer.
    """
    reference_path, reference = repeating_reference
    short = _excerpt(
        tmp_path, reference, "scale_short.wav", TRUE_OFFSET_S, SHORT_CLIP_S, seed=31
    )
    long_clip = _excerpt(
        tmp_path, reference, "scale_long.wav", 5.0, LONG_CLIP_S, seed=33
    )

    short_result, long_result = align_clips_to_reference(
        reference_path, [short, long_clip], sample_rate=SR
    )

    assert short_result.window_s == pytest.approx(
        SHORT_CLIP_S / MIN_WINDOWS_FOR_SUPPORT_TARGET
    )
    assert long_result.window_s == SPAN_WINDOW_S
    assert short_result.window_s != long_result.window_s, (
        "two clips in one call, two scales — which is exactly why it is reported"
    )


def test_the_reported_window_is_the_one_asked_for_when_one_is_asked_for():
    """An explicit window is reported unchanged, so the field never lies about the scale."""
    reference = np.zeros(int(300 * SR))
    clip = np.zeros(int(120 * SR))
    (pinned,) = align_clips_to_reference(
        reference, [clip], sample_rate=SR, window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S
    )
    assert pinned.window_s == SPAN_WINDOW_S


def test_no_vote_means_no_window_to_be_relative_to():
    """``consensus=False`` holds no vote, so there is no scale — ``None``, like support.

    A number here would be as misleading as a manufactured support: it would name a
    window that nothing was measured over.
    """
    reference = np.zeros(int(90 * SR))
    clip = np.zeros(int(30 * SR))
    (single,) = align_clips_to_reference(
        reference, [clip], sample_rate=SR, consensus=False
    )
    assert (single.support, single.window_s) == (None, None)


def test_each_clip_in_a_set_is_windowed_for_itself(repeating_reference, tmp_path):
    """A short clip in a set must not inherit a long one's window, or the call's order.

    The windowing is decided per clip, so the same clip gets the same answer whether it
    is measured alone or alongside a clip five times its length.
    """
    reference_path, reference = repeating_reference
    short = _excerpt(
        tmp_path, reference, "set_short.wav", TRUE_OFFSET_S, SHORT_CLIP_S, seed=21
    )
    long_clip = _excerpt(tmp_path, reference, "set_long.wav", 5.0, LONG_CLIP_S, seed=23)

    together = align_clips_to_reference(
        reference_path, [long_clip, short], sample_rate=SR
    )
    (alone,) = align_clips_to_reference(reference_path, [short], sample_rate=SR)

    assert [a.index for a in together] == [0, 1]
    measured_in_a_set = (together[1].offset_s, together[1].support)
    assert measured_in_a_set == (alone.offset_s, alone.support)
