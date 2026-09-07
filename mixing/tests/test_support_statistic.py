"""``support`` is a GRADED tally of the windows' evidence, not an argmax headcount.

Issue #45. Once the analysis window was fitted to the clip (issue #41) a short clip got a
real vote — and a confidence statistic that fell apart underneath it. ``support`` counted
windows whose INDEPENDENT ARGMAX reached the offset, a definition written for 20 s
windows where an argmax is a real opinion. At the ~4 s window a 12 s clip is fitted to,
the argmax is close to a coin flip while the *vote*, which pools each window's near-tied
candidates, is still right. Measured on real cross-device material: 21 alignments that
were ALL correct on the material reported support from 0.00 to 1.00, six of them 0.00, so
a single gate at 0.25 refused about half of them. The estimator got more reliable as the
window shrank and its confidence statistic got less — backwards, for a trust gate.

So a window that put the winning offset on its BALLOT now counts for something: a full
1.0 for an independent argmax, up to :data:`BALLOT_VOTE_WEIGHT` for a ballot mention
scaled by how close it scored, nothing for an offset the window never considered. Two
consequences worth stating rather than discovering:

- **The old question is still answerable.** Ballot mentions alone cannot carry the tally
  past 0.5, so ``support > 0.5`` means at least one independent window got there unaided.
- **The number now means "how much of the clip's evidence reaches this offset"**, not
  "how uniquely does the clip belong here". Those came apart in the old definition only by
  accident, and the accident is what broke on short clips.

The deterministic half of this file builds :class:`_WindowMeasurement` directly, because
the properties being pinned — grading, monotonicity, the 0.5 boundary — are properties of
the tally and not of any audio. The synthetic-audio half pins that the fix reaches the
shipped entry point on material that reproduces #45's signature.
"""

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import align_clips_to_reference
from mixing.audio import audio_ops
from mixing.audio.audio_ops import (
    BALLOT_VOTE_WEIGHT,
    MIN_WINDOWS_FOR_SUPPORT,
    SPAN_HOP_S,
    SPAN_WINDOW_S,
    _support_fraction,
    _window_agreement,
    _WindowMeasurement,
)

SR = 16000

#: The offset every constructed window below is asked about.
TRUTH = 20.0
#: A rival offset far enough away that no tolerance confuses the two.
RIVAL = 65.0
#: Tolerance used throughout the constructed cases — the shipped default.
TOL = 0.25
#: Window length and stride for the constructed cases: a half-window stride sits exactly
#: on :data:`MAX_SUPPORT_OVERLAP`, so every constructed window counts as independent.
WIN, HOP = 10.0, 5.0


def _window(index: int, candidates: "list[tuple[float, float]]") -> _WindowMeasurement:
    """One window at ``index * HOP``, whose ballot is ``candidates`` best-first.

    ``candidates[0]`` is the window's own argmax, which is what ``vote_offset_s`` keeps —
    the same invariant :func:`_window_offsets` establishes.
    """
    start = index * HOP
    return _WindowMeasurement(
        clip_start_s=start,
        clip_end_s=start + WIN,
        candidates=tuple(candidates),
        vote_offset_s=candidates[0][0],
    )


def _support(windows: "list[_WindowMeasurement]") -> "float | None":
    return _support_fraction(windows, TRUTH, offset_tolerance_s=TOL)


# --------------------------------------------------------------------------
# What one window's evidence is worth
# --------------------------------------------------------------------------


def test_an_independent_argmax_is_worth_a_whole_window():
    """The strongest thing a window can say, and the grade the old tally counted."""
    window = _window(0, [(TRUTH, 0.9), (RIVAL, 0.88)])
    assert _window_agreement(window, TRUTH, offset_tolerance_s=TOL) == 1.0


def test_a_ballot_mention_is_worth_less_than_an_argmax_and_more_than_nothing():
    """The grade that did not exist before — the whole of issue #45.

    The window's answer was ``RIVAL``; it could not separate ``TRUTH`` from it, and said
    so by putting it on its ballot. That is weaker evidence than finding it unaided and
    much stronger than never considering it, and the old tally scored it as a flat
    disagreement.
    """
    window = _window(0, [(RIVAL, 0.90), (TRUTH, 0.90)])
    mention = _window_agreement(window, TRUTH, offset_tolerance_s=TOL)
    assert 0.0 < mention < 1.0
    assert mention == pytest.approx(BALLOT_VOTE_WEIGHT)


def test_a_ballot_mention_is_weighted_by_how_close_it_scored():
    """A near-tie is nearly a full mention; a distant nomination is nearly nothing.

    This is the "weight the tally by score" half of the issue: an offset the window rated
    at a tenth of its own best is on the ballot, but it is not the same evidence as one
    the window could not separate from its answer.
    """
    near = _window(0, [(RIVAL, 1.0), (TRUTH, 0.98)])
    far = _window(0, [(RIVAL, 1.0), (TRUTH, 0.1)])
    assert _window_agreement(near, TRUTH, offset_tolerance_s=TOL) == pytest.approx(
        BALLOT_VOTE_WEIGHT * 0.98
    )
    assert _window_agreement(far, TRUTH, offset_tolerance_s=TOL) == pytest.approx(
        BALLOT_VOTE_WEIGHT * 0.1
    )


def test_an_offset_nobody_nominated_is_worth_nothing():
    """No vote can be read out of an offset the window never put forward."""
    window = _window(0, [(RIVAL, 0.9)])
    assert _window_agreement(window, TRUTH, offset_tolerance_s=TOL) == 0.0


# --------------------------------------------------------------------------
# The properties the tally must have
# --------------------------------------------------------------------------


def test_more_agreeing_evidence_never_lowers_the_support():
    """Monotone in the evidence — the property the old tally did not have.

    A statistic a caller gates on must not fall when a window says *more*. The old one
    could not fall, but it could not rise either: everything short of an argmax was a
    zero, so a window moving from "never considered it" to "could not separate it from
    its own answer" bought nothing. Here each step up the ladder raises the number, and
    no step ever lowers it.
    """
    ladder = [
        [(RIVAL, 1.0)],  # never considered
        [(RIVAL, 1.0), (TRUTH, 0.20)],  # nominated, far behind
        [(RIVAL, 1.0), (TRUTH, 0.99)],  # a near-tie it could not separate
        [(TRUTH, 1.0), (RIVAL, 0.99)],  # its own answer
    ]
    fixed = [_window(i, [(TRUTH, 0.9)]) for i in (1, 2)]
    measured = [_support([_window(0, rung)] + fixed) for rung in ladder]

    assert measured == sorted(measured), measured
    assert measured[0] < measured[-1], "the ladder must actually climb"


def test_ballot_mentions_alone_cannot_reach_past_half():
    """``support > 0.5`` still means some window found the offset unaided.

    The old statistic's whole question, kept answerable: :data:`BALLOT_VOTE_WEIGHT` is
    the ceiling of ballot-only evidence, so the top half of the range is reserved for
    independent agreement. A caller that wants exactly what the old number promised gates
    above 0.5 rather than re-deriving it.
    """
    unanimous_on_the_ballot = [
        _window(i, [(RIVAL, 1.0), (TRUTH, 1.0)]) for i in range(4)
    ]
    assert _support(unanimous_on_the_ballot) == pytest.approx(BALLOT_VOTE_WEIGHT)

    one_found_it_alone = unanimous_on_the_ballot[:-1] + [
        _window(3, [(TRUTH, 1.0), (RIVAL, 1.0)])
    ]
    assert _support(one_found_it_alone) > 0.5


def test_a_genuinely_split_vote_still_reads_as_a_split():
    """Half the windows one way, half the other, and neither has the other on its ballot.

    A 50/50 is the case grading must NOT inflate: there is no hidden agreement to
    recover, so the number stays where the headcount had it. Half is also exactly the
    boundary of what ballots alone can buy, which is why an inflated split would be
    indistinguishable from a unanimous near-tie.
    """
    split = [_window(i, [(TRUTH, 0.9)]) for i in range(2)] + [
        _window(i, [(RIVAL, 0.9)]) for i in (2, 3)
    ]
    assert _support(split) == pytest.approx(0.5)


def test_where_every_window_found_it_unaided_the_graded_tally_is_the_old_one():
    """Grading is a no-op wherever the argmax was already decisive.

    The compatibility half of the change: a long clip at the default window, whose
    windows all reach the offset on their own, must report exactly what it reported
    before — 1.0, not 1.0-and-a-bit. Nothing above an argmax exists to add.
    """
    unanimous = [_window(i, [(TRUTH, 0.9), (RIVAL, 0.89)]) for i in range(4)]
    assert _support(unanimous) == 1.0


def test_too_few_independent_windows_is_still_None_not_a_number():
    """Grading does not lower the bar for having measured anything at all.

    A single look agrees with itself, and a manufactured number VOUCHES. ``None`` is the
    only honest answer, and it stays ``None`` however generous the grades get.
    """
    assert MIN_WINDOWS_FOR_SUPPORT == 2, "this test is written against a quorum of two"
    lone = [_window(0, [(RIVAL, 1.0), (TRUTH, 1.0)])]
    assert _support(lone) is None


def test_a_window_that_scored_nothing_cannot_lend_its_ballot_weight():
    """A degenerate all-zero surface has no evidence to grade, not evidence of zero.

    Dividing a match's score by a best of zero would be a ratio with no meaning; the
    honest reading is that the window said nothing.
    """
    silent = _window(0, [(TRUTH, 0.0), (RIVAL, 0.0)])
    assert _window_agreement(silent, RIVAL, offset_tolerance_s=TOL) == 0.0


# --------------------------------------------------------------------------
# The same thing, through the shipped entry point, on synthetic audio
#
# The "before" is measured, not remembered: `BALLOT_VOTE_WEIGHT = 0` reduces the graded
# tally to exactly the argmax headcount it replaced, so each case below runs the same
# code over the same windows twice and compares. That is the same device
# `near_tie_ratio=0.0` gives for the pre-consensus argmax path (issue #30).
# --------------------------------------------------------------------------

#: The bed tiles at this period, so the reference's WAVEFORM repeats verbatim — the
#: material issue #30 is about. Only the onsets are unique.
BED_LOOP_S = 2.0
#: Long enough that the clips below are a small fraction of it, as on real material.
BED_REFERENCE_S = 96.0
#: A non-repeating reference, long enough to hold a 60 s clip well inside it.
PLAIN_REFERENCE_S = 120.0
#: Where the short clip really belongs.
TRUE_OFFSET_S = 33.0
#: Mean spacing of the reference's onsets, seconds. Sparse enough that a fitted window of
#: a few seconds holds only a couple of them, which is what makes its argmax a coin flip.
ONSET_GAP_S = 2.5
#: How loud those onsets are against the bed. Quiet enough that the bed's periodicity
#: dominates any single short window's correlation.
ONSET_GAIN = 0.15
#: Recording noise on a clip, as a fraction of full scale — "another device", not a copy.
CLIP_NOISE = 0.15
#: Long enough to be fitted to a window of a few seconds, short enough that no window's
#: argmax resolves the tile. This is #45's band.
SHORT_CLIP_S = 16.0
#: Three default windows — the length at which nothing is adapted (issue #41).
LONG_CLIP_S = SPAN_WINDOW_S * 3
#: Where the long clip sits, comfortably inside both references.
LONG_CLIP_START_S = 5.0
#: The two takes of the split-vote clip, in reference time.
SPLIT_TAKES_S = ((5.0, 30.0), (60.0, 85.0))
#: Amplitude of the white-noise clip that is not the reference at all.
NOISE_LEVEL = 0.3


def _write(path, name: str, x: np.ndarray) -> str:
    p = path / name
    wavfile.write(str(p), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return str(p)


def _repeating_bed(seconds: float, *, seed: int, loop_s: float) -> np.ndarray:
    """A stationary tonal bed that tiles seamlessly at ``loop_s``.

    Every partial is an exact multiple of ``1 / loop_s`` Hz, so the tiling has no seam and
    the waveform genuinely repeats — no window can tell one tile from another.
    """
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


@pytest.fixture(scope="module")
def bed(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """``(path, samples)`` — a repeating bed under onsets too sparse for a short window."""
    tiles = int(BED_REFERENCE_S / BED_LOOP_S)
    tiled = np.tile(_repeating_bed(BED_LOOP_S, seed=3, loop_s=BED_LOOP_S), tiles)
    rng = np.random.default_rng(5)
    onsets = np.cumsum(rng.uniform(ONSET_GAP_S * 0.5, ONSET_GAP_S * 1.5, 400))
    onsets = onsets[onsets < BED_REFERENCE_S - 0.1]
    reference = tiled + ONSET_GAIN * _onset_track(BED_REFERENCE_S, onsets, hz=3000)
    reference /= np.max(np.abs(reference))
    out = tmp_path_factory.mktemp("support_bed")
    return _write(out, "bed.wav", reference), reference


@pytest.fixture(scope="module")
def plain(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """A non-periodic broadband 'song' — one sharp autocorrelation peak, no ambiguity.

    The control for the bed: here every window reaches the offset unaided, so the graded
    tally has nothing to add and must report what the headcount reported.
    """
    t = np.arange(int(PLAIN_REFERENCE_S * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in [(180, 520), (440, 130), (700, 900), (110, 250)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    reference = x / np.max(np.abs(x))
    out = tmp_path_factory.mktemp("support_plain")
    return _write(out, "plain.wav", reference), reference


def _excerpt(path, reference, name, start_s, duration_s, *, seed) -> str:
    """Another device's recording of ``[start_s, start_s + duration_s)``."""
    rng = np.random.default_rng(seed)
    take = reference[int(start_s * SR) : int((start_s + duration_s) * SR)]
    return _write(path, name, take + rng.normal(0, CLIP_NOISE, len(take)))


def _aligned(reference_path: str, clip: str, **grid):
    (measured,) = align_clips_to_reference(
        reference_path, [clip], sample_rate=SR, **grid
    )
    return measured


def _as_argmax_headcount(reference_path: str, clip: str, **grid):
    """The same alignment with the statistic this replaced in force.

    ``BALLOT_VOTE_WEIGHT = 0`` makes a ballot mention worth nothing again, which is
    precisely the old definition: the fraction of independent windows whose own argmax
    reached the offset. Everything else — windows, ballots, the vote, the offset — is
    untouched, so the two runs differ in the statistic and in nothing else.
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(audio_ops, "BALLOT_VOTE_WEIGHT", 0.0)
        return _aligned(reference_path, clip, **grid)


@pytest.mark.parametrize("seed", [19, 23, 29])
def test_a_correct_short_clip_on_a_repeating_bed_is_no_longer_refused(
    bed, tmp_path, seed
):
    """Issue #45's acceptance, at synthetic scale: the offset is right and so is the gate.

    A 16 s clip is fitted to a window of a few seconds. Every window correlates just as
    well against every other tile of the bed, so its argmax lands wherever the noise sends
    it — and the old headcount reads 0.00 for an offset that is exactly right. The graded
    tally reads about a half: no window found it unaided, but every one of them put it on
    its ballot, which is what the vote read and what a caller is entitled to see.

    Asserted as bands rather than pinned: which tile a coin flip lands on is precisely
    what is not stable here, and pinning it would make this test fail for being right.
    """
    reference_path, reference = bed
    clip = _excerpt(
        tmp_path, reference, f"short_{seed}.wav", TRUE_OFFSET_S, SHORT_CLIP_S, seed=seed
    )

    measured = _aligned(reference_path, clip)
    before = _as_argmax_headcount(reference_path, clip)

    assert measured.offset_s == pytest.approx(TRUE_OFFSET_S, abs=0.05)
    assert before.support <= 0.25, (
        "the fixture must still reproduce the defect: a gate at 0.25 refused this "
        f"correct alignment, got {before.support}"
    )
    assert measured.support > 0.25, (
        f"the correct alignment must clear the gate that refused it, got "
        f"{measured.support}"
    )
    assert measured.support == pytest.approx(BALLOT_VOTE_WEIGHT, abs=0.2), (
        "and it must land where ballot-only evidence belongs, not at unanimity"
    )


def test_only_the_statistic_moved_and_never_the_estimate(bed, tmp_path):
    """The offset and the confidence are not a function of the tally, and must not become one.

    Support is computed after the vote, from the same windows the vote read. If changing
    what a ballot mention is worth could move where the clip lands, the number would be
    steering the estimate it is supposed to describe.
    """
    reference_path, reference = bed
    clip = _excerpt(
        tmp_path, reference, "unmoved.wav", TRUE_OFFSET_S, SHORT_CLIP_S, seed=19
    )

    measured = _aligned(reference_path, clip)
    before = _as_argmax_headcount(reference_path, clip)

    assert (measured.offset_s, measured.confidence) == (
        before.offset_s,
        before.confidence,
    )
    assert (measured.window_s, measured.hop_s) == (before.window_s, before.hop_s)
    assert measured.support != before.support, "and the statistic did move"


def test_a_long_clip_at_the_default_window_reports_the_old_number_exactly(
    plain, tmp_path
):
    """The compatibility guarantee, measured rather than assumed.

    Above the fitting threshold nothing about this clip is adapted, and on a reference
    that does not repeat every window reaches the offset unaided. Grading has nothing to
    add there — no window has a rival it could not separate — so ``support`` must equal
    the argmax headcount it replaced, to the bit rather than to five decimals.
    """
    reference_path, reference = plain
    clip = _excerpt(
        tmp_path, reference, "long.wav", LONG_CLIP_START_S, LONG_CLIP_S, seed=19
    )
    grid = dict(window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S)

    measured = _aligned(reference_path, clip, **grid)
    before = _as_argmax_headcount(reference_path, clip, **grid)

    assert measured.window_s == SPAN_WINDOW_S
    assert measured.offset_s == pytest.approx(LONG_CLIP_START_S, abs=0.05)
    assert measured.support == before.support
    assert measured.support == 1.0, (
        "on a reference that does not repeat, every window gets there on its own"
    )


def test_white_noise_does_not_acquire_support_it_did_not_earn(bed, tmp_path):
    """Grading must not manufacture agreement where there is nothing to agree about.

    A clip that is not the reference at all still lands somewhere — the correlation always
    has an argmax — and the fear with a more generous tally is that the somewhere starts
    looking corroborated. It does not: a window's ballot is its own near-ties, and noise
    has none in common with another window's noise.
    """
    reference_path, _ = bed
    noise = np.random.default_rng(99).normal(0, NOISE_LEVEL, int(50 * SR))
    clip = _write(tmp_path, "noise.wav", noise)
    grid = dict(window_s=WIN, hop_s=HOP)

    measured = _aligned(reference_path, clip, **grid)
    before = _as_argmax_headcount(reference_path, clip, **grid)

    assert measured.confidence < 0.25, "the fixture must not accidentally match"
    assert measured.support is None or measured.support < 0.25, (
        f"noise must stay unsupported, got {measured.support}"
    )
    assert measured.support == pytest.approx(before.support, abs=0.05), (
        "and grading must not have moved it much either"
    )


def test_a_split_vote_is_not_inflated_towards_unanimity(bed, tmp_path):
    """Two takes from two places: the report stays a report of a split.

    The synthetic-audio counterpart of the constructed 50/50. Grading raises the number —
    the windows of the take that lost did have the winner on their ballots, because the
    bed repeats — but nowhere near the unanimity a caller reads as "the whole clip
    agrees".
    """
    reference_path, reference = bed
    rng = np.random.default_rng(53)
    takes = [reference[int(a * SR) : int(b * SR)] for a, b in SPLIT_TAKES_S]
    clip = _write(
        tmp_path,
        "split.wav",
        np.concatenate([t + rng.normal(0, CLIP_NOISE, len(t)) for t in takes]),
    )

    measured = _aligned(reference_path, clip, window_s=SPAN_WINDOW_S, hop_s=SPAN_HOP_S)

    assert measured.support is not None
    assert 0.2 < measured.support < 0.7, (
        f"a split must read as a split, got {measured.support}"
    )
