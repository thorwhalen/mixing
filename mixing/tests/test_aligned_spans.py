"""``mixing.audio.aligned_spans`` — which PARTS of a clip align, and at what offset each.

``align_clips_to_reference`` answers "where does this clip sit?" with one number. That is
the right answer only for a clip that is one continuous take. A clip that was stopped and
restarted has no such number, and the single-offset model does not say so — it returns the
offset of whichever part correlated best and describes the rest of the clip wrongly, with a
confidence that looks fine. Measured below: a clip holding two takes of the same song
scores 0.494 on the single-offset path, which passes any sane trust gate.

The fixtures are small on purpose. Each window is an FFT cross-correlation against the
whole reference, so window count is the cost axis and these run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from mixing.audio import (
    AlignedSpan,
    align_clips_to_reference,
    aligned_spans,
    find_audio_offset_detailed,
)

SR = 16000
WIN, HOP = 10.0, 5.0


def _reference(seconds: float = 90.0) -> np.ndarray:
    """A non-periodic broadband 'song' — one sharp autocorrelation peak, no ambiguity."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for f0, f1 in [(180, 520), (440, 130), (700, 900), (110, 250)]:
        x += np.sin(2 * np.pi * (f0 + (f1 - f0) * (t / t[-1])) * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    return x / np.max(np.abs(x))


@pytest.fixture(scope="module")
def ref() -> np.ndarray:
    return _reference()


@pytest.fixture(scope="module")
def song(tmp_path_factory, ref) -> str:
    p = tmp_path_factory.mktemp("spans") / "song.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p)


def _noisy(seg: np.ndarray, rng, snr_db: float = 15.0) -> np.ndarray:
    p_sig = np.mean(seg**2)
    return seg + rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), len(seg))


def _write(tmp_path, name: str, samples: np.ndarray) -> str:
    p = tmp_path / name
    wavfile.write(
        str(p), SR, ((samples / np.max(np.abs(samples))) * 32767).astype(np.int16)
    )
    return str(p)


def _take(ref, a: float, b: float, rng) -> np.ndarray:
    return _noisy(ref[int(a * SR) : int(b * SR)].copy(), rng)


def _spans(song, clip, **kw):
    return aligned_spans(song, clip, sample_rate=SR, window_s=WIN, hop_s=HOP, **kw)


# --------------------------------------------------------------------------
# The thing it exists for
# --------------------------------------------------------------------------


def test_a_stopped_and_restarted_clip_yields_one_span_per_take(song, ref, tmp_path):
    """The headline. Two takes, two spans, each with its OWN offset."""
    rng = np.random.default_rng(0)
    # Recorded song[5..30], stopped, then recorded song[60..85].
    clip = _write(
        tmp_path,
        "two.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 60, 85, rng)]),
    )
    spans = _spans(song, clip)

    assert len(spans) == 2, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    a, b = spans
    # abs=0.05, not abs=WIN: the observed error is 0.0000, and a WIN-wide tolerance
    # would let every reported offset be wrong by a whole window with the suite green.
    assert a.offset_s == pytest.approx(5.0, abs=0.05)
    assert b.offset_s == pytest.approx(35.0, abs=0.05)  # clip t=25 -> song t=60
    assert a.reference_span[0] == pytest.approx(5.0, abs=0.05)
    assert b.reference_span[1] == pytest.approx(85.0, abs=0.05)


def test_the_single_offset_model_describes_that_clip_wrongly_and_looks_confident(
    song, ref, tmp_path
):
    """Why this function had to exist, stated as a measurement rather than a claim.

    The single-offset answer is not merely imprecise — it is *right about one take and
    wrong about the other*, at a confidence that clears any threshold a caller would set.
    """
    rng = np.random.default_rng(0)
    clip = _write(
        tmp_path,
        "two.wav",
        np.concatenate([_take(ref, 5, 30, rng), _take(ref, 60, 85, rng)]),
    )
    one = find_audio_offset_detailed(song, clip, sample_rate=SR, feature="envelope")
    assert one.confidence > 0.3, "the wrong answer does not announce itself"
    spans = _spans(song, clip)
    # It matches ONE of the two spans and contradicts the other.
    matches = [s for s in spans if abs(s.offset_s - one.offset_s) < WIN]
    assert len(matches) == 1


def test_a_continuous_take_returns_exactly_one_span(song, ref, tmp_path):
    """The compatibility property every caller migrating off the single offset needs.

    Asserted rather than assumed: if a clip that IS one take came back as several, the
    migration would turn every healthy project into a fragmented one.
    """
    rng = np.random.default_rng(1)
    clip = _write(tmp_path, "one.wav", _take(ref, 20, 70, rng))
    spans = _spans(song, clip)

    assert len(spans) == 1
    assert spans[0].clip_start_s == pytest.approx(0.0, abs=0.01)
    # The most valuable assertion here — it cross-checks the windowed path against an
    # INDEPENDENT implementation. At abs=WIN it could not fire.
    single = find_audio_offset_detailed(song, clip, sample_rate=SR, feature="envelope")
    assert spans[0].offset_s == pytest.approx(single.offset_s, abs=0.05)


def test_spans_are_ordered_and_never_overlap(song, ref, tmp_path):
    """A clip instant claimed by two different offsets is not a fact about anything.

    Windows overlap by ``window - hop``, so the window straddling a boundary belongs to
    both takes; untrimmed, consecutive spans overlapped by a full hop.
    """
    rng = np.random.default_rng(2)
    clip = _write(
        tmp_path,
        "three.wav",
        np.concatenate(
            [_take(ref, 5, 25, rng), _take(ref, 55, 75, rng), _take(ref, 30, 50, rng)]
        ),
    )
    spans = _spans(song, clip)
    for a, b in zip(spans, spans[1:]):
        assert a.clip_end_s <= b.clip_start_s + 1e-9, (a, b)
        assert a.clip_start_s < a.clip_end_s


def test_a_clip_that_matches_nothing_returns_no_spans(song, tmp_path):
    """Empty is a real answer, not a failure."""
    rng = np.random.default_rng(3)
    clip = _write(tmp_path, "noise.wav", rng.normal(0, 1, int(30 * SR)))
    assert _spans(song, clip) == []


# --------------------------------------------------------------------------
# The parts that are easy to get wrong
# --------------------------------------------------------------------------


def test_the_offset_is_relative_to_the_clip_not_the_window(song, ref, tmp_path):
    """`offset = lag - window_start`, and dropping the subtraction is invisible at t=0.

    A clip whose match begins well INTO it is what makes the conversion observable: every
    window would otherwise report a different offset and no two would ever group, so a
    continuous take would come back shattered into one span per window.
    """
    rng = np.random.default_rng(4)
    clip = _write(tmp_path, "late.wav", _take(ref, 10, 60, rng))
    spans = _spans(song, clip)
    assert len(spans) == 1, (
        "the per-window offsets did not agree — check `lag - start_s`"
    )
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_silence_mid_take_does_not_read_as_a_stop_and_restart(song, ref, tmp_path):
    """One continuous take with a dead patch must come back as ONE span.

    Measured: 12 s of hard silence collapses the windows inside it to a confidence of
    exactly 0.0, the run breaks, and without the merge the result is two spans BOTH
    reporting offset 10.00 — a discontinuity that never happened. A stop and a restart
    cannot resume in sync, so two spans AGREEING on the offset are positive evidence of
    continuity.

    This replaces a hysteresis test that could not fail: on real correlations a degraded
    window does not dip into a band, it collapses, so no keep-threshold was ever what
    decided.
    """
    rng = np.random.default_rng(5)
    take = _take(ref, 10, 60, rng)
    take[int(20 * SR) : int(32 * SR)] = 0.0  # twelve dead seconds mid-take
    clip = _write(tmp_path, "dip.wav", take)

    spans = _spans(song, clip)
    assert len(spans) == 1, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(10.0, abs=0.05)
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_a_long_unverified_gap_is_not_merged_away(song, ref, tmp_path):
    """The bound on the merge, and why it is not optional.

    A clip that records the song, then a stretch of something else, then the song again
    AT THE SAME OFFSET would otherwise be merged into one span claiming the middle
    aligns. Correlation cannot tell "quiet" from "different material", so past the gap
    bound both are reported and the caller decides.
    """
    rng = np.random.default_rng(11)
    head = _take(ref, 10, 25, rng)
    middle = rng.normal(
        0, 0.5, int(30 * SR)
    )  # unrelated material, far longer than a window
    tail = _take(ref, 55, 70, rng)  # offset 10 again: clip t=45 -> song t=55
    clip = _write(tmp_path, "far.wav", np.concatenate([head, middle, tail]))

    spans = _spans(song, clip)
    assert len(spans) == 2, [(s.clip_start_s, s.clip_end_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(spans[1].offset_s, abs=1.0), (
        "the fixture is only meaningful if both spans DO agree on the offset — "
        "otherwise the offset rule would have kept them apart and the gap bound "
        "would not be what this test measures"
    )


def test_the_merge_is_bounded_by_merge_gap_s_and_the_caller_can_widen_it(
    song, ref, tmp_path
):
    """Same fixture, wider bound -> one span. The knob is what is under test."""
    rng = np.random.default_rng(11)
    clip = _write(
        tmp_path,
        "far.wav",
        np.concatenate(
            [
                _take(ref, 10, 25, rng),
                rng.normal(0, 0.5, int(30 * SR)),
                _take(ref, 55, 70, rng),
            ]
        ),
    )
    assert len(_spans(song, clip, merge_gap_s=60.0)) == 1


def test_a_clip_shorter_than_one_window_still_works(song, ref, tmp_path):
    rng = np.random.default_rng(6)
    clip = _write(tmp_path, "short.wav", _take(ref, 30, 36, rng))
    spans = _spans(song, clip)
    assert len(spans) == 1
    assert spans[0].offset_s == pytest.approx(30.0, abs=0.05)


def test_the_tail_of_a_clip_is_covered(song, ref, tmp_path):
    """A clip whose length is not a whole number of hops must not lose its end.

    Without the final catch-up window, up to ``window_s`` of every such clip is never
    looked at and a span ending there is silently truncated.
    """
    rng = np.random.default_rng(7)
    clip = _write(tmp_path, "tail.wav", _take(ref, 10, 47.5, rng))  # 37.5s on a 5s hop
    spans = _spans(song, clip)
    assert len(spans) == 1
    assert spans[0].clip_end_s == pytest.approx(37.5, abs=0.2)


def test_the_signals_are_decoded_once_not_once_per_window(
    song, ref, tmp_path, monkeypatch
):
    """Re-decoding per window would be N times the cost AND reintroduce issue #25.

    pydub's rate conversion has no anti-alias filter, so a per-window decode path is a
    per-window chance to halve the confidence. Counting the calls is the only way to keep
    a future refactor from quietly moving the decode inside the loop.
    """
    import mixing.audio.audio_ops as ops

    calls = []
    real = ops._load_mono_samples
    monkeypatch.setattr(
        ops, "_load_mono_samples", lambda src, sr: calls.append(1) or real(src, sr)
    )
    rng = np.random.default_rng(8)
    clip = _write(tmp_path, "count.wav", _take(ref, 10, 60, rng))
    ops.aligned_spans(song, clip, sample_rate=SR, window_s=WIN, hop_s=HOP)
    assert len(calls) == 2, "one decode for the reference, one for the clip — no more"


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_reference_span_uses_the_same_offset_convention_as_ClipAlignment():
    """``reference_time = clip_time + offset_s``, as `ClipAlignment.offset_s` documents."""
    s = AlignedSpan(clip_start_s=4.0, clip_end_s=9.0, offset_s=100.0, confidence=0.5)
    assert s.reference_span == (104.0, 109.0)
    assert s.duration_s == 5.0


def test_an_unknown_feature_is_refused(song, tmp_path):
    rng = np.random.default_rng(9)
    clip = _write(tmp_path, "f.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="unknown feature"):
        aligned_spans(song, clip, feature="vibes")


@pytest.mark.parametrize("kw", [{"window_s": 0.0}, {"hop_s": -1.0}])
def test_a_non_positive_window_or_hop_is_refused(song, tmp_path, kw):
    rng = np.random.default_rng(10)
    clip = _write(tmp_path, "w.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="must be positive"):
        aligned_spans(song, clip, **kw)


# --------------------------------------------------------------------------
# Found by adversarial review: spans must lie ON the reference
# --------------------------------------------------------------------------


def test_a_span_never_reports_reference_time_the_reference_does_not_have(
    song, ref, tmp_path
):
    """The reviewer's reproducer, and the worst failure mode in this module.

    A window is admitted while it overlaps the reference by `min_overlap_ratio` of
    ITSELF, so a span could run past a reference edge by `(1-ratio)*window_s`. Measured
    before the fix on a 30 s reference: `reference_span == (-10.0, 40.0)` — a 50 s span
    of a 30 s reference, which is not imprecise but impossible.

    What made it dangerous is that it fails SILENTLY and in the wrong direction. A
    caller slicing `ref[int(-10.0*sr):int(40.0*sr)]` gets numpy's negative-index
    resolution: the start wraps to reference time 20 s and the end clamps to 30 s, so
    they receive the reference's TAIL for a span whose true content is its ENTIRE
    length. No exception either way.
    """
    rng = np.random.default_rng(20)
    # clip = 20 s of junk, then the whole reference, then 20 s of junk
    clip = _write(
        tmp_path,
        "overrun.wav",
        np.concatenate(
            [
                rng.normal(0, 0.5, int(20 * SR)),
                _noisy(ref.copy(), rng),
                rng.normal(0, 0.5, int(20 * SR)),
            ]
        ),
    )
    ref_dur = len(ref) / SR
    spans = _spans(song, clip)
    assert spans
    for sp in spans:
        a, b = sp.reference_span
        assert a >= -1e-6, f"reference start {a} is before the reference begins"
        assert b <= ref_dur + 1e-6, (
            f"reference end {b} is past the reference's {ref_dur}"
        )
        assert sp.duration_s <= ref_dur + 1e-6, (
            f"a {sp.duration_s}s span cannot align to a {ref_dur}s reference at one offset"
        )


def test_a_clip_that_starts_before_the_reference_is_trimmed_not_negated(
    song, ref, tmp_path
):
    """The preroll case — the common one, and the one that slices to EMPTY."""
    rng = np.random.default_rng(21)
    clip = _write(
        tmp_path,
        "preroll.wav",
        np.concatenate([rng.normal(0, 0.5, int(20 * SR)), _take(ref, 0, 30, rng)]),
    )
    spans = _spans(song, clip)
    assert spans
    assert spans[0].reference_span[0] >= -1e-6
    # ...and the CLIP edge moved with it, so the documented identity still holds.
    for sp in spans:
        assert sp.reference_span == pytest.approx(
            (sp.clip_start_s + sp.offset_s, sp.clip_end_s + sp.offset_s)
        )


def test_hop_wider_than_the_window_is_refused(song, ref, tmp_path):
    """It would leave clip time no correlation ever examined inside a single span."""
    rng = np.random.default_rng(22)
    clip = _write(tmp_path, "hop.wav", _take(ref, 10, 60, rng))
    with pytest.raises(ValueError, match="must not exceed"):
        aligned_spans(song, clip, sample_rate=SR, window_s=5.0, hop_s=55.0)


def test_the_earlier_spans_end_wins_the_disputed_overlap():
    """`_disjoin`'s documented rule, pinned directly.

    A pure list operation, so this needs no audio and no tolerance — it pins the POLICY
    rather than the correlation's accuracy, which is why the end-to-end tests (with their
    window-wide tolerances) could not.
    """
    from mixing.audio.audio_ops import _disjoin

    a = AlignedSpan(clip_start_s=0.0, clip_end_s=25.0, offset_s=5.0, confidence=0.9)
    b = AlignedSpan(clip_start_s=20.0, clip_end_s=50.0, offset_s=35.0, confidence=0.9)
    assert [(s.clip_start_s, s.clip_end_s) for s in _disjoin([a, b])] == [
        (0.0, 25.0),
        (25.0, 50.0),
    ]


def test_the_shipped_defaults_are_exercised(song, ref, tmp_path):
    """Every other test overrides window_s/hop_s; nothing ran what users get."""
    from mixing.audio.audio_ops import SPAN_HOP_S, SPAN_WINDOW_S

    rng = np.random.default_rng(23)
    clip = _write(tmp_path, "defaults.wav", _take(ref, 5, 85, rng))
    spans = aligned_spans(song, clip, sample_rate=SR)  # no overrides at all
    assert len(spans) == 1
    assert spans[0].offset_s == pytest.approx(5.0, abs=0.05)
    assert (SPAN_WINDOW_S, SPAN_HOP_S) == (20.0, 10.0)


def test_reference_duration_can_be_stated_by_the_caller(song, ref, tmp_path):
    """Parity with `align_clips_to_reference`, and it must actually bound the spans."""
    rng = np.random.default_rng(24)
    clip = _write(tmp_path, "rd.wav", _take(ref, 10, 80, rng))
    spans = _spans(song, clip, reference_duration=40.0)
    assert spans
    for sp in spans:
        assert sp.reference_span[1] <= 40.0 + 1e-6


# --------------------------------------------------------------------------
# Issue #30: a reference that repeats itself
#
# The fixtures above are deliberately non-repetitive ("one sharp autocorrelation peak,
# no alignment ambiguity"), which is why the suite was green while this was live. Music
# is the material this function exists for, and music repeats.
# --------------------------------------------------------------------------


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


#: Length of one motif in the repetitive fixtures, seconds. Six of them make a 90 s
#: "song" — the same length as the non-repetitive reference above, so the two are
#: comparable and the only variable is the repetition.
MOTIF_S = 15.0


@pytest.fixture(scope="module")
def repeating(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """Verse/chorus structure: A B C B A B. An offset into it is NOT unique."""
    a, b, c = (_motif(MOTIF_S, s) for s in (1, 2, 3))
    ref = np.concatenate([a, b, c, b, a, b])
    p = tmp_path_factory.mktemp("rep") / "repeating.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


@pytest.fixture(scope="module")
def tiling(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """One motif, nine times. Every offset that is right is right nine ways."""
    ref = np.concatenate([_motif(10.0, 11)] * 9)
    p = tmp_path_factory.mktemp("tile") / "tiling.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


def test_a_repeating_reference_no_longer_fragments_one_continuous_take(
    repeating, tmp_path
):
    """The headline of issue #30, as a before/after on the same clip.

    ``near_tie_ratio=0.0`` is the shipped-before behaviour — each window picks its lag by
    an independent argmax — and it is exercised here rather than described, because a
    regression test for a coin flip has to show the coin.

    Measured: the argmax path returns 4 spans for one continuous take, of which three
    carry a WRONG offset (80.0, 50.0 and -40.0 against a truth of 20.0) — and the worst
    of them scores 0.982, HIGHER than the span that is right. That is the failure a
    consumer cannot filter out. The count is asserted as a floor rather than pinned:
    which way a near-tie falls is exactly what is not stable here, and pinning it would
    make this test fail for being right.
    """
    song, ref = repeating
    rng = np.random.default_rng(30)
    clip = _write(
        tmp_path, "rep_one.wav", _take(ref, 20, 70, rng)
    )  # one take, offset 20

    argmax = _spans(song, clip, near_tie_ratio=0.0)
    assert len(argmax) >= 3, "the fixture must still reproduce the defect"
    wrong = [s for s in argmax if abs(s.offset_s - 20.0) > 1.0]
    assert wrong, "the defect is a WRONG offset, not merely a split"
    assert max(s.confidence for s in wrong) > 0.9, (
        "and it is wrong at a confidence no caller would filter out"
    )

    spans = _spans(song, clip)
    assert len(spans) == 1, [(s.clip_start_s, s.offset_s) for s in spans]
    assert spans[0].offset_s == pytest.approx(20.0, abs=0.05)
    assert spans[0].duration_s == pytest.approx(50.0, abs=WIN)


def test_support_says_the_windows_needed_help_and_the_confidence_does_not(
    repeating, tiling, song, ref, tmp_path
):
    """The acceptance line: the offset is right AND the honest doubt is reported.

    ``confidence`` answers "how well does the clip match where we put it", and on a
    repeating reference that question has several excellent answers. Measured across
    these three references it reads 0.985 / 0.978 / 0.980 — it barely moves, and it does
    not even move the right way: the exactly tiling reference, where the offset is a
    free choice among nine, scores HIGHER than the verse/chorus one. A caller gating on
    it alone cannot tell them apart, which is the complaint in issue #30. ``support`` —
    the fraction of windows that reached the offset unaided — is what moves, and it
    moves with the repetition: 1.000 / 0.444 / 0.000.
    """
    rng = np.random.default_rng(31)
    cases = {}
    for name, (path, material) in {
        "none": (song, ref),
        "verse/chorus": repeating,
        "exact tiling": tiling,
    }.items():
        clip = _write(tmp_path, f"sup_{len(cases)}.wav", _take(material, 20, 70, rng))
        # The take is one span on the first two; on the tiling reference the answer is
        # arbitrary enough that a tail window can peel off, so take the longest.
        cases[name] = max(_spans(path, clip), key=lambda s: s.duration_s)

    assert cases["none"].support == 1.0, "no repetition, no disagreement"
    assert cases["verse/chorus"].support < 1.0, (
        "some windows correlated just as well against the wrong chorus"
    )
    assert cases["exact tiling"].support < cases["verse/chorus"].support, (
        "and where every offset is equally true, nothing agrees unaided"
    )
    # The point of the field: the confidence cannot make this distinction.
    assert min(s.confidence for s in cases.values()) > 0.9


def test_a_stop_and_restart_still_departs_on_a_repeating_reference(repeating, tmp_path):
    """The feature the fix must not eat.

    Consensus could trivially return one span for everything. It does not, because no
    window is ever moved to a lag its own correlation did not already rate a near-tie —
    and a restarted take has no such lag near the old offset. Both takes here span a
    motif boundary whose SEQUENCE occurs once (…C and B,A…), so each offset is unique
    and this asserts the answer rather than the tie-break.
    """
    song, ref = repeating
    rng = np.random.default_rng(32)
    clip = _write(
        tmp_path,
        "rep_two.wav",
        np.concatenate([_take(ref, 25, 45, rng), _take(ref, 55, 75, rng)]),
    )
    spans = _spans(song, clip)
    assert len(spans) == 2, [(s.clip_start_s, s.offset_s) for s in spans]
    a, b = spans
    assert a.offset_s == pytest.approx(25.0, abs=0.05)
    assert b.offset_s == pytest.approx(35.0, abs=0.05)  # clip t=20 -> song t=55


def test_near_tie_ratio_zero_is_the_old_behaviour_exactly(song, ref, tmp_path):
    """The knob is a seam, so it has to be pinned at both ends.

    On material with no repetition the consensus pass has nothing to decide, so the two
    settings must agree exactly — otherwise the default silently perturbs every existing
    caller's numbers, and the before/after test above would be measuring the wrong thing.
    """
    rng = np.random.default_rng(33)
    clip = _write(tmp_path, "seam.wav", _take(ref, 10, 60, rng))
    assert _spans(song, clip) == _spans(song, clip, near_tie_ratio=0.0)


def test_a_negative_near_tie_ratio_is_refused(song, tmp_path):
    rng = np.random.default_rng(34)
    clip = _write(tmp_path, "neg.wav", rng.normal(0, 1, int(5 * SR)))
    with pytest.raises(ValueError, match="near_tie_ratio"):
        aligned_spans(song, clip, sample_rate=SR, near_tie_ratio=-0.1)


# --------------------------------------------------------------------------
# The same defect in the whole-clip estimator (`align_clips_to_reference`)
#
# Same correlation, same coin flip — and it is the entry point the named consumer
# actually calls, so the fix has to reach it or it reaches no one.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_halves(tmp_path_factory) -> "tuple[str, np.ndarray]":
    """A 90 s reference that is one 45 s half, twice."""
    ref = np.concatenate([_motif(45.0, 7)] * 2)
    p = tmp_path_factory.mktemp("halves") / "halves.wav"
    wavfile.write(str(p), SR, (ref * 32767).astype(np.int16))
    return str(p), ref


def test_the_whole_clip_estimator_reaches_the_consensus_offset(two_halves, tmp_path):
    """Measured: the argmax puts this clip 45 s out, and its coefficient does not flinch.

    Offset 65 is not merely wrong, it is impossible — the clip is 50 s and the reference
    ends at 90 s, so the clip's last 25 s would sit past the end of the song. One number
    over the whole clip cannot notice that; the windows can, because the ones covering
    that tail have no candidate there at all.
    """
    song, ref = two_halves
    rng = np.random.default_rng(35)
    clip = _write(tmp_path, "halves_clip.wav", _take(ref, 20, 70, rng))  # offset 20

    (old,) = align_clips_to_reference(song, [clip], sample_rate=SR, consensus=False)
    assert abs(old.offset_s - 20.0) > 1.0, "the fixture must still reproduce the defect"
    assert old.confidence > 0.9, "at a confidence that clears any sane gate"
    assert old.support == 1.0, "and with nothing else to warn the caller"

    (new,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    assert new.offset_s == pytest.approx(20.0, abs=0.05)
    assert new.support < 1.0, "half of this reference genuinely does fit twice"


def test_support_localises_a_clip_that_is_only_partly_the_song(song, ref, tmp_path):
    """The second thing a coefficient cannot say: how MUCH of the clip is the reference.

    Half song, half unrelated noise. The confidence is measured where the clip matches,
    so it stays high — and should, the alignment really is that good. ``support`` is the
    field that reports that only half the clip voted for it.
    """
    rng = np.random.default_rng(36)
    clip = _write(
        tmp_path,
        "halfjunk.wav",
        np.concatenate([_take(ref, 10, 40, rng), rng.normal(0, 0.5, int(30 * SR))]),
    )
    (a,) = align_clips_to_reference(
        song, [clip], sample_rate=SR, window_s=WIN, hop_s=HOP
    )
    assert a.offset_s == pytest.approx(10.0, abs=0.05)
    assert a.confidence > 0.9
    assert 0.3 < a.support < 0.8, f"about half the clip should agree, got {a.support}"


def test_a_clip_shorter_than_one_window_takes_the_whole_clip_answer(
    song, ref, tmp_path
):
    """Consensus must degrade to the thing it replaces, not to a different thing.

    A clip shorter than ``window_s`` is a single window, so there is nothing to vote on
    and the answer — offset, confidence and support — must equal the single-correlation
    path exactly. That is what leaves every existing short-clip caller unmoved, and it is
    why turning consensus on by default does not perturb the rest of this suite.
    """
    rng = np.random.default_rng(37)
    clip = _write(tmp_path, "short_clip.wav", _take(ref, 30, 45, rng))
    (new,) = align_clips_to_reference(song, [clip], sample_rate=SR)  # window is 20 s
    (old,) = align_clips_to_reference(song, [clip], sample_rate=SR, consensus=False)
    assert (new.offset_s, new.confidence, new.support) == (
        old.offset_s,
        old.confidence,
        old.support,
    )
