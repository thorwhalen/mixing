"""``margin`` — how far the answer's support sits above the RUNNER-UP offset's.

Issue #47. ``support`` answers "how much of the clip's own evidence reaches this
offset". What nothing reported was the other half of the question a trust gate actually
asks: **"and how much reaches somewhere else instead?"** Three scalars have now been
measured NOT to separate a correct alignment from a wrong one on real cross-device
material (thorwhalen/muvid#59):

- ``confidence`` — the wrong offset's peak scored 0.987-0.993 of the right one's, and on
  one clip the wrong offset scored HIGHER;
- ``support`` after grading (issue #45) — an ambiguous tiling lands mid-scale *by
  construction*, because every alias genuinely is reached by the same evidence;
- the window rule (issue #48) — deleted, having been measured not to separate them
  either.

A fraction cannot see a runner-up. A margin can, and it is nearly free: the same graded
tally, read at the best offset outside ``offset_tolerance_s`` of the answer, subtracted.

The deterministic half of this file builds :class:`_WindowMeasurement` directly, because
what is being pinned — the scale, the ``None`` rule, the sign, what a rival is — is a
property of the tally and not of any audio. The synthetic-audio half pins the three cases
the issue names, and the characterization at the end pins that nothing else moved.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import align_clips_to_reference, aligned_spans
from mixing.audio.audio_ops import (
    BALLOT_VOTE_WEIGHT,
    MIN_WINDOWS_FOR_SUPPORT,
    AlignedSpan,
    _merge_margin,
    _rival_offsets,
    _support_fraction,
    _support_margin,
    _WindowMeasurement,
)

SR = 16000

#: The offset every constructed window below is asked about.
TRUTH = 20.0
#: A rival offset far enough away that no tolerance confuses the two.
RIVAL = 65.0
#: A second rival, equally far from both.
OTHER = 110.0
#: Tolerance used throughout the constructed cases — the shipped default.
TOL = 0.25
#: Window length and stride for the constructed cases: a half-window stride sits exactly
#: on ``MAX_SUPPORT_OVERLAP``, so every constructed window counts as independent.
WIN, HOP = 10.0, 5.0


def _window(
    index: float, candidates: "list[tuple[float, float]]"
) -> _WindowMeasurement:
    """One window at ``index * HOP``, whose ballot is ``candidates`` best-first."""
    start = index * HOP
    return _WindowMeasurement(
        clip_start_s=start,
        clip_end_s=start + WIN,
        candidates=tuple(candidates),
        vote_offset_s=candidates[0][0],
    )


def _margin(windows: "list[_WindowMeasurement]", offset_s: float = TRUTH):
    return _support_margin(windows, offset_s, offset_tolerance_s=TOL)


def _support(windows: "list[_WindowMeasurement]", offset_s: float = TRUTH):
    return _support_fraction(windows, offset_s, offset_tolerance_s=TOL)


# --------------------------------------------------------------------------
# The scale
# --------------------------------------------------------------------------


def test_an_offset_nothing_rivals_keeps_its_whole_tally():
    """The top of the scale: no window nominated anything else, so nothing is subtracted.

    This is what "1.0" has to mean for the number to be readable at all — every
    independent window found the offset unaided AND no other offset was even put
    forward. A rival that nobody nominated is not a runner-up; it is a number.
    """
    unanimous = [_window(i, [(TRUTH, 0.9)]) for i in range(4)]
    assert _support(unanimous) == 1.0
    assert _margin(unanimous) == 1.0


def test_a_rival_every_window_carries_eats_into_the_margin_support_keeps():
    """Support is untouched by a rival; the margin is not. That divergence is the issue.

    Every window's own argmax is ``TRUTH``, so support is a flat 1.0 and stays there
    however strong the rival gets. But every window also could not separate ``RIVAL``
    from it, so ``RIVAL`` is backed by :data:`BALLOT_VOTE_WEIGHT` of the same evidence —
    and the answer is only that much clear of it.
    """
    tied = [_window(i, [(TRUTH, 1.0), (RIVAL, 1.0)]) for i in range(4)]
    assert _support(tied) == 1.0, "each window's own argmax is TRUTH"
    assert _margin(tied) == pytest.approx(1.0 - BALLOT_VOTE_WEIGHT)


def test_an_evenly_split_vote_leaves_no_margin_at_all():
    """The bottom of the useful scale: two offsets, one tally, nothing to separate them.

    Half the windows answer each way and every window carries both, so the tally is
    identical at ``TRUTH`` and at ``RIVAL``. Support reports a respectable half at
    either — honestly, that much evidence really does reach each — and only the margin
    says the half is shared and the answer was a free choice.
    """
    split = [_window(i, [(TRUTH, 1.0), (RIVAL, 1.0)]) for i in range(2)] + [
        _window(i, [(RIVAL, 1.0), (TRUTH, 1.0)]) for i in (2, 3)
    ]
    assert _support(split) == pytest.approx(_support(split, RIVAL))
    assert _support(split) > BALLOT_VOTE_WEIGHT, "support alone still reads well"
    assert _margin(split) == pytest.approx(0.0)


def test_the_margin_is_negative_when_the_tally_prefers_somewhere_else():
    """Not clamped. A number that cannot go below zero cannot report this.

    Asking about an offset the evidence ranks second gets the honest answer: the tally
    here is behind. Callers see this when the vote's headcount and the graded tally over
    the independent subset do not rank identically — rare, and the strongest "do not
    trust this" the field can carry, so it must not be rounded away.
    """
    leaning = [_window(i, [(RIVAL, 1.0), (TRUTH, 1.0)]) for i in range(4)]
    assert _margin(leaning) == pytest.approx(BALLOT_VOTE_WEIGHT - 1.0)
    assert _margin(leaning) < 0.0


def test_strengthening_a_rival_lowers_the_margin_while_support_does_not_move():
    """The separation support is structurally unable to make.

    Each rung leaves every window's evidence FOR ``TRUTH`` untouched — its argmax is
    ``TRUTH`` throughout — so support is pinned at 1.0 the whole way up. What changes is
    how much of the same evidence also reaches ``RIVAL``, and only the margin sees it.
    """
    ladder = [
        [(TRUTH, 1.0)],  # no rival at all
        [(TRUTH, 1.0), (RIVAL, 0.2)],  # a distant nomination
        [(TRUTH, 1.0), (RIVAL, 0.99)],  # a near-tie
    ]
    supports = [_support([_window(i, rung) for i in range(4)]) for rung in ladder]
    margins = [_margin([_window(i, rung) for i in range(4)]) for rung in ladder]

    assert supports == [1.0, 1.0, 1.0], supports
    assert margins == sorted(margins, reverse=True), margins
    assert margins[0] > margins[-1], "the ladder must actually descend"


# --------------------------------------------------------------------------
# What counts as a runner-up
# --------------------------------------------------------------------------


def test_one_peaks_shoulders_are_one_rival_and_not_several():
    """Rivals are clustered on ``offset_tolerance_s`` — the vote's own equivalence.

    Two nominations a tenth of a tolerance apart are one hypothesis seen twice. Listing
    both would change no maximum here, but it would make ``_rival_offsets`` a different
    statement about the ballot than the one the vote reads, and the margin is a
    difference between two readings that must be the same reading.
    """
    windows = [
        _window(0, [(TRUTH, 1.0), (RIVAL, 0.9), (RIVAL + TOL / 10, 0.8)]),
        _window(1, [(TRUTH, 1.0), (RIVAL - TOL / 10, 0.9)]),
    ]
    rivals = _rival_offsets(windows, TRUTH, offset_tolerance_s=TOL)
    assert len(rivals) == 1, rivals
    assert rivals[0] == pytest.approx(RIVAL, abs=TOL)


def test_an_offset_inside_the_tolerance_is_the_answer_and_not_a_rival():
    """A candidate the vote would have grouped WITH the answer cannot run against it."""
    windows = [_window(i, [(TRUTH, 1.0), (TRUTH + TOL / 2, 0.99)]) for i in range(3)]
    assert _rival_offsets(windows, TRUTH, offset_tolerance_s=TOL) == []
    assert _margin(windows) == 1.0


def test_the_best_rival_wins_even_when_it_is_not_the_second_voting_group():
    """The decision issue #47 left open, settled: best OFFSET outside tolerance.

    Here ``RIVAL`` is on every window's ballot at a near-tie while ``OTHER`` is the
    argmax of a single window. The second-best voting *group* is ``OTHER`` — one window
    landed there unaided — but the offset backed by the most evidence is ``RIVAL``, and
    that is what a caller asking "does anything else have an equal claim" needs to hear.
    """
    windows = [_window(i, [(TRUTH, 1.0), (RIVAL, 1.0)]) for i in range(3)] + [
        _window(3, [(OTHER, 1.0), (TRUTH, 1.0)])
    ]
    assert _support(windows, RIVAL) > _support(windows, OTHER)
    assert _margin(windows) == pytest.approx(
        _support(windows) - _support(windows, RIVAL)
    )


# --------------------------------------------------------------------------
# When there is nothing to measure
# --------------------------------------------------------------------------


def test_one_independent_look_has_no_runner_up_and_reports_None():
    """The same quorum as ``support``, for the same reason: never manufactured.

    A lone window cannot be ahead of anything — it has no second opinion to be ahead
    of — and a margin invented there would VOUCH for whatever it is attached to, which
    is precisely the failure the whole line of work exists to prevent.
    """
    assert MIN_WINDOWS_FOR_SUPPORT == 2, "this test is written against a quorum of two"
    lone = [_window(0, [(TRUTH, 1.0), (RIVAL, 0.6)])]
    assert _support(lone) is None
    assert _margin(lone) is None


def test_margin_is_None_exactly_where_support_is():
    """One rule, not two: a caller that has learned support's ``None`` knows margin's."""
    for windows in ([], [_window(0, [(TRUTH, 1.0)])]):
        assert (_support(windows) is None) == (_margin(windows) is None)
    paired = [_window(i, [(TRUTH, 1.0)]) for i in range(2)]
    assert _support(paired) is not None and _margin(paired) is not None


def test_merging_two_spans_averages_the_margin_and_unmeasured_wins():
    """Merged spans follow support's rule — duration-weighted, ``None`` if either is.

    A span whose margin was never measured must not inherit its neighbour's separation:
    the merge joins two runs across a gap nothing verified, so there is no combined
    ballot a runner-up could be read off.
    """
    long_span = AlignedSpan(0.0, 30.0, offset_s=5.0, confidence=0.9, margin=1.0)
    short_span = AlignedSpan(31.0, 37.0, offset_s=5.0, confidence=0.9, margin=0.0)
    unmeasured = AlignedSpan(31.0, 37.0, offset_s=5.0, confidence=0.9, margin=None)

    assert _merge_margin(long_span, short_span) == pytest.approx(30.0 / 36.0)
    assert _merge_margin(long_span, unmeasured) is None
    assert _merge_margin(unmeasured, long_span) is None


# --------------------------------------------------------------------------
# The same thing on synthetic audio, through the shipped entry points
#
# The three cases issue #47 names, built to be the three cases and nothing else. Every
# builder below is byte-for-byte the one the corresponding shipped fixture uses, so the
# numbers pinned in the characterization at the end are the numbers 0.0.50 returned.
# --------------------------------------------------------------------------

#: One motif in the repetitive references, seconds. Six of them make a 90 s "song", the
#: same length as the non-repetitive reference, so repetition is the only variable.
MOTIF_S = 15.0
#: The tiling reference's period. Nine copies of one 10 s motif: every offset that is
#: right is right nine ways, 10 s apart.
TILE_S = 10.0
#: The clip taken from each 90 s reference — one continuous take, and where it belongs.
TAKE_S = (20.0, 70.0)
#: The bed's tiling period, seconds. Every partial is an exact multiple of ``1 / loop``
#: Hz, so the waveform genuinely repeats and no window can tell one tile from another.
BED_LOOP_S = 2.0
#: Long enough that the short clip is a small fraction of it, as on real material.
BED_REFERENCE_S = 96.0
#: Mean spacing of the bed's onsets — the only non-periodic thing in it. Sparse enough
#: that a fitted window of a few seconds holds only a couple of them.
ONSET_GAP_S = 2.5
#: How loud those onsets are against the bed.
ONSET_GAIN = 0.15
#: Recording noise on the short clip, as a fraction of full scale — "another device".
CLIP_NOISE = 0.15
#: Where the short clip really belongs, and how long it is. Issue #45's band: long
#: enough to be fitted to a window of a few seconds, short enough that no window's
#: argmax resolves the tile.
SHORT_CLIP_START_S, SHORT_CLIP_S = 33.0, 16.0
#: Signal-to-noise of a take from a 90 s reference, in dB.
TAKE_SNR_DB = 15.0
#: What "a large margin" means for a reference with one sharp autocorrelation peak, and
#: what "near zero" means for an exactly tiling one. Bands, not pins: which alias a coin
#: flip lands on is exactly what is not stable, and pinning it would fail for being right.
LARGE_MARGIN, NEAR_ZERO_MARGIN = 0.9, 0.1


def _write(path, x: np.ndarray) -> str:
    wavfile.write(str(path), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return str(path)


def _write_normalized(path, x: np.ndarray) -> str:
    return _write(path, x / np.max(np.abs(x)))


def _plain_reference(seconds: float = 90.0) -> np.ndarray:
    """A non-periodic broadband 'song' — one sharp autocorrelation peak, no ambiguity."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in [(180, 520), (440, 130), (700, 900), (110, 250)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    return x / np.max(np.abs(x))


def _motif(seconds: float, seed: int) -> np.ndarray:
    """A short broadband phrase with percussive onsets, so both features have a grip."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in rng.uniform(120, 900, (4, 2)):
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / max(t[-1], 1e-9))) * t)
    beat = np.zeros_like(t)
    for i in range(0, len(t), int(0.5 * SR)):
        n = min(400, len(t) - i)
        beat[i : i + n] += np.hanning(400)[:n]
    return (x / np.max(np.abs(x))) * 0.7 + beat * 0.5


def _repeating_bed(seconds: float, *, seed: int, loop_s: float) -> np.ndarray:
    """A stationary tonal bed that tiles seamlessly at ``loop_s``."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for harmonic in rng.choice(np.arange(4, 200), size=40, replace=False):
        x += np.sin(2 * np.pi * (harmonic / loop_s) * t + rng.uniform(0, 2 * np.pi))
    return x / np.max(np.abs(x))


def _onset_track(seconds: float, times, *, hz: float) -> np.ndarray:
    """Short decaying bursts at ``times`` — the only thing that is NOT periodic here."""
    x = np.zeros(int(seconds * SR))
    burst_t = np.arange(int(0.04 * SR)) / SR
    burst = np.sin(2 * np.pi * hz * burst_t) * np.exp(-burst_t / 0.01)
    for onset in times:
        i = int(onset * SR)
        x[i : i + len(burst)] += burst[: len(x) - i]
    return x


def _take(reference: np.ndarray, a: float, b: float, rng) -> np.ndarray:
    """Another device's recording of ``[a, b)`` at :data:`TAKE_SNR_DB`."""
    segment = reference[int(a * SR) : int(b * SR)].copy()
    power = np.mean(segment**2)
    noise = np.sqrt(power / (10 ** (TAKE_SNR_DB / 10)))
    return segment + rng.normal(0, noise, len(segment))


@pytest.fixture(scope="module")
def references() -> "dict[str, np.ndarray]":
    """The three 90 s references, differing only in how much they repeat."""
    a, b, c = (_motif(MOTIF_S, seed) for seed in (1, 2, 3))
    return {
        "plain": _plain_reference(),
        "verse/chorus": np.concatenate([a, b, c, b, a, b]),
        "exact tiling": np.concatenate([_motif(TILE_S, 11)] * 9),
    }


@pytest.fixture(scope="module")
def bed() -> np.ndarray:
    """A repeating bed under onsets too sparse for a short window to resolve."""
    tiled = np.tile(
        _repeating_bed(BED_LOOP_S, seed=3, loop_s=BED_LOOP_S),
        int(BED_REFERENCE_S / BED_LOOP_S),
    )
    rng = np.random.default_rng(5)
    onsets = np.cumsum(rng.uniform(ONSET_GAP_S * 0.5, ONSET_GAP_S * 1.5, 400))
    onsets = onsets[onsets < BED_REFERENCE_S - 0.1]
    reference = tiled + ONSET_GAIN * _onset_track(BED_REFERENCE_S, onsets, hz=3000)
    return reference / np.max(np.abs(reference))


def _take_case(tmp_path, references, name: str):
    """``(reference_path, clip_path)`` for one continuous take of ``references[name]``."""
    material = references[name]
    slug = name.replace("/", "_")
    reference_path = _write(tmp_path / f"ref_{slug}.wav", material)
    clip_path = _write_normalized(
        tmp_path / f"take_{slug}.wav",
        _take(material, *TAKE_S, np.random.default_rng(31)),
    )
    return reference_path, clip_path


def _longest_span(reference_path: str, clip_path: str) -> AlignedSpan:
    """The take's span. On the tiling reference a tail window can peel off."""
    spans = aligned_spans(
        reference_path, clip_path, sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    return max(spans, key=lambda s: s.duration_s)


def _aligned(reference_path: str, clip_path: str, **grid):
    (measured,) = align_clips_to_reference(
        reference_path, [clip_path], sample_rate=SR, **grid
    )
    return measured


def _short_bed_clip(tmp_path, bed, *, seed: int):
    """``(reference_path, clip_path)`` for issue #45's case: a short clip on the bed."""
    reference_path = _write(tmp_path / "bed.wav", bed)
    segment = bed[
        int(SHORT_CLIP_START_S * SR) : int((SHORT_CLIP_START_S + SHORT_CLIP_S) * SR)
    ]
    noise = np.random.default_rng(seed).normal(0, CLIP_NOISE, len(segment))
    return reference_path, _write(tmp_path / f"short_{seed}.wav", segment + noise)


def test_a_unique_offset_has_a_large_margin(references, tmp_path):
    """Nothing else has any claim, so the margin is the whole of the support."""
    span = _longest_span(*_take_case(tmp_path, references, "plain"))

    assert span.offset_s == pytest.approx(TAKE_S[0], abs=0.05)
    assert span.support == 1.0
    assert span.margin > LARGE_MARGIN, span.margin


def test_an_exactly_tiling_reference_has_a_margin_near_zero(references, tmp_path):
    """Both answers are TRUE, and that is what the margin is for.

    Nine aliases 10 s apart, every one of them a correct description of where the clip
    sits. The span reports one and its support reads a respectable half — earned, not
    inflated: that much of the clip's evidence really does reach the reported offset.
    The margin is what says the same is true of eight other offsets.
    """
    span = _longest_span(*_take_case(tmp_path, references, "exact tiling"))

    assert (span.offset_s - TAKE_S[0]) % TILE_S == pytest.approx(0.0, abs=0.05), (
        "the reported offset must be one of the aliases — all of them are true"
    )
    assert span.support > 2 * NEAR_ZERO_MARGIN, (
        f"support alone cannot report the ambiguity, got {span.support}"
    )
    assert abs(span.margin) < NEAR_ZERO_MARGIN, span.margin


def test_a_verse_chorus_reference_keeps_a_clear_margin(references, tmp_path):
    """The middle case: rivals exist and are genuinely behind.

    The distinction the field has to make is not "repetitive or not" — it is "is the
    runner-up level with the answer". A B C B A B repeats, but an offset into it is
    still unique, so the margin stays wide while the exactly tiling one collapses.
    """
    verse = _longest_span(*_take_case(tmp_path, references, "verse/chorus"))
    tiling = _longest_span(*_take_case(tmp_path, references, "exact tiling"))

    assert verse.offset_s == pytest.approx(TAKE_S[0], abs=0.05)
    assert verse.margin > NEAR_ZERO_MARGIN
    assert verse.margin > tiling.margin


@pytest.mark.parametrize("seed", [19, 23, 29])
def test_a_short_clip_on_a_repeating_bed_has_a_positive_margin(tmp_path, bed, seed):
    """Issue #45's case, and the row a support-only gate gets wrong.

    A 16 s clip fitted to a window of a few seconds, on a bed that repeats every 2 s. No
    window's argmax resolves the tile, so the graded tally sits near
    ``BALLOT_VOTE_WEIGHT`` — thin evidence — and the offset is nonetheless exactly
    right. The margin is what says so: the thin evidence is undisputed.
    """
    measured = _aligned(*_short_bed_clip(tmp_path, bed, seed=seed))

    assert measured.offset_s == pytest.approx(SHORT_CLIP_START_S, abs=0.05)
    assert measured.support == pytest.approx(BALLOT_VOTE_WEIGHT, abs=0.2)
    assert measured.margin > 0.0, measured.margin


def test_margin_separates_two_clips_that_support_cannot(references, tmp_path, bed):
    """The acceptance, as one comparison: same support, opposite verdict.

    A correct short clip on a repeating bed and an ambiguous take of an exactly tiling
    reference report support within a few hundredths of each other — both around a half,
    both honestly. No threshold on support can pass the first and refuse the second. The
    margin does: clearly positive for the one whose runner-up is behind, and level with
    zero — which side of it is not a fact about anything — where nine offsets are equal.
    """
    correct = _aligned(*_short_bed_clip(tmp_path, bed, seed=19))
    ambiguous = _aligned(
        *_take_case(tmp_path, references, "exact tiling"), window_s=WIN, hop_s=HOP
    )

    assert abs(correct.support - ambiguous.support) < NEAR_ZERO_MARGIN, (
        "the two must be indistinguishable on support for this to prove anything: "
        f"{correct.support} vs {ambiguous.support}"
    )
    assert correct.margin > 0.0, (
        f"the correct clip's runner-up is behind it: {correct.margin}"
    )
    assert abs(ambiguous.margin) < NEAR_ZERO_MARGIN, (
        f"nine offsets are level here, so nothing is ahead: {ambiguous.margin}"
    )
    assert correct.margin > ambiguous.margin


def test_without_a_vote_there_is_no_margin(references, tmp_path):
    """``consensus=False`` puts nothing to a vote, so there is no runner-up to be ahead of."""
    measured = _aligned(
        *_take_case(tmp_path, references, "plain"), consensus=False, feature="waveform"
    )
    assert measured.support is None
    assert measured.margin is None


# --------------------------------------------------------------------------
# Characterization: this field is ADDITIVE
#
# The numbers below were measured on 0.0.50's tree — the released version this was built
# on — over exactly the material the builders above produce, and are reproduced here to
# the digit. `margin` is a new field computed after the fact from the same windows; if
# any of it ever reaches back into the estimate, this is what fails.
#
# Compared to a relative tolerance rather than by `==`: an FFT correlation's last bits
# are a function of the numpy/scipy build, while any behavioural change moves these by
# 1e-3 or more. Offsets, windows and hops are exact — they come from integer lags and
# from the grid — so those are pinned outright.
# --------------------------------------------------------------------------

#: ``0.0.50``'s ``(offset_s, confidence, support)`` for the longest span of one
#: continuous take of each reference, at ``window_s=10, hop_s=5``.
SPANS_0_0_50 = {
    "plain": (20.0, 0.984476713061276, 1.0),
    "verse/chorus": (20.0, 0.984572367514042, 0.7222125224346537),
    "exact tiling": (10.0, 0.9845022902002918, 0.49998510877499713),
}

#: ``0.0.50``'s ``(offset_s, confidence, support, window_s, hop_s)`` for the short clip
#: on the repeating bed, at the FITTED grid (``window_s=None``).
BED_SHORT_0_0_50 = (
    33.0,
    0.8864321776265371,
    0.49976406980351934,
    5.333333333333333,
    2.6666666666666665,
)

#: How far a reproduced number may drift from the pinned one and still be the same
#: number. Far below any behavioural change, far above FFT last-bit noise.
CHARACTERIZATION_REL = 1e-9


@pytest.mark.parametrize("name", list(SPANS_0_0_50))
def test_aligned_spans_returns_what_0_0_50_returned(references, tmp_path, name):
    """Adding ``margin`` moved no span's offset, confidence or support."""
    span = _longest_span(*_take_case(tmp_path, references, name))
    offset_s, confidence, support = SPANS_0_0_50[name]

    assert span.offset_s == offset_s
    assert span.confidence == pytest.approx(confidence, rel=CHARACTERIZATION_REL)
    assert span.support == pytest.approx(support, rel=CHARACTERIZATION_REL)
    assert span.margin is not None, "and the new field is populated"


def test_align_clips_to_reference_returns_what_0_0_50_returned(tmp_path, bed):
    """Same, on the fitted-window path, where ``window_s`` is itself an output."""
    measured = _aligned(*_short_bed_clip(tmp_path, bed, seed=19))
    offset_s, confidence, support, window_s, hop_s = BED_SHORT_0_0_50

    assert measured.offset_s == offset_s
    assert (measured.window_s, measured.hop_s) == (window_s, hop_s)
    assert measured.confidence == pytest.approx(confidence, rel=CHARACTERIZATION_REL)
    assert measured.support == pytest.approx(support, rel=CHARACTERIZATION_REL)
    assert measured.margin is not None
