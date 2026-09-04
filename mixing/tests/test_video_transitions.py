"""Video transition primitives — importable top-level + working under moviepy 2.x.

Two things this locks in:

- ``slow_motion_blend`` used ``clip.with_speed(...)``, removed in moviepy 2.x
  (renamed ``with_speed_scaled``), so it raised ``AttributeError`` on every
  call. Nothing exercised it, so the break went unnoticed. The parametrized
  render test below fails without the fix.
- The transition helpers (``crossfade_transition`` / ``fade_through_black`` /
  …) were only reachable via a deep ``mixing.video.video_concat`` import. They
  are now top-level ``mixing.*`` names so consumers (e.g. reelee's animatic
  assembly) can pass them to ``concatenate_videos(transform_clips=…)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_transitions_are_importable_top_level():
    from mixing import (
        crossfade_transition,
        fade_through_black,
        overlap_blend,
        slow_motion_blend,
        trim_and_crossfade,
    )

    for fn in (
        crossfade_transition,
        fade_through_black,
        slow_motion_blend,
        trim_and_crossfade,
        overlap_blend,
    ):
        assert callable(fn)


def _tiny_clips(d: Path, n: int = 3) -> list[str]:
    try:
        from moviepy import ColorClip
    except Exception:  # pragma: no cover - moviepy<2 layout
        from moviepy.editor import ColorClip

    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)][:n]
    paths: list[str] = []
    for i, color in enumerate(colors):
        p = d / f"c{i}.mp4"
        ColorClip(size=(64, 64), color=color, duration=0.6).with_fps(10).write_videofile(
            str(p), logger=None
        )
        paths.append(str(p))
    return paths


@pytest.mark.parametrize(
    "name",
    [
        "crossfade_transition",
        "fade_through_black",
        "slow_motion_blend",  # the regression: with_speed → with_speed_scaled
        "trim_and_crossfade",
        "overlap_blend",
    ],
)
def test_transition_assembles_a_real_video(name: str, tmp_path: Path):
    import mixing
    from mixing import concatenate_videos, has_ffmpeg

    if not has_ffmpeg():
        pytest.skip("ffmpeg is required for a real concat")
    try:
        clips = _tiny_clips(tmp_path)
    except Exception:  # pragma: no cover
        pytest.skip("moviepy is required for a real concat")

    fn = getattr(mixing, name)
    out = tmp_path / f"{name}.mp4"
    result = concatenate_videos(clips, transform_clips=fn, output=str(out))
    try:
        assert out.exists() and out.stat().st_size > 0, "a real mp4 was written"
        assert getattr(result, "duration", 0) > 0
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- #
# The join: does a crossfade actually blend a pixel? (issue #33)
# --------------------------------------------------------------------------- #
#
# The tests above assert that an mp4 exists with a nonzero duration. Both are
# true of a hard cut, which is exactly what `crossfade_transition` and
# `overlap_blend` used to render: moviepy 2.x joins with method="chain",
# padding=0, under which the CrossFade masks are never composited and the clips
# never overlap. Measured before the fix, red 251 -> green 252 in a single
# frame with nothing in between. These tests decode the boundary instead.

BLEND_FLOOR = 12  # 8-bit channel value above which a colour is "present"


def _two_flat_clips(d: Path, fps: int = 10, duration: float = 1.0) -> list[str]:
    """One solid red clip then one solid green clip — nothing else in frame.

    Flat colours make the assertion unambiguous: any frame carrying *both* red
    and green can only have come from compositing the two clips.
    """
    try:
        from moviepy import ColorClip
    except Exception:  # pragma: no cover - moviepy<2 layout
        from moviepy.editor import ColorClip

    paths = []
    for name, color in (("red.mp4", (255, 0, 0)), ("green.mp4", (0, 255, 0))):
        p = d / name
        ColorClip(size=(32, 32), color=color, duration=duration).with_fps(
            fps
        ).write_videofile(str(p), logger=None)
        paths.append(str(p))
    return paths


def _centre_pixels(path: str) -> tuple[float, list[tuple[int, int, int]]]:
    """``(duration, [rgb of the centre pixel, one per frame])`` of a rendered mp4."""
    try:
        from moviepy import VideoFileClip
    except Exception:  # pragma: no cover - moviepy<2 layout
        from moviepy.editor import VideoFileClip

    clip = VideoFileClip(path)
    try:
        n_frames = int(round(clip.duration * clip.fps))
        rows = []
        for i in range(n_frames):
            t = min((i + 0.5) / clip.fps, clip.duration - 1e-6)
            frame = clip.get_frame(t)
            h, w = frame.shape[:2]
            rows.append(tuple(int(v) for v in frame[h // 2][w // 2]))
        return clip.duration, rows
    finally:
        clip.close()


def _render(transform, tmp_path: Path, name: str):
    """Concatenate two flat clips through ``transform`` and decode the result."""
    import mixing

    clips = _two_flat_clips(tmp_path)
    out = tmp_path / f"{name}.mp4"
    result = mixing.concatenate_videos(
        clips, transform_clips=transform, output=str(out), normalize_dimensions=False
    )
    try:
        return _centre_pixels(str(out))
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            close()


def _blended_frames(rows) -> list[int]:
    """Indices of frames carrying red *and* green — i.e. actually composited."""
    return [i for i, (r, g, _b) in enumerate(rows) if r > BLEND_FLOOR and g > BLEND_FLOOR]


@pytest.fixture
def real_media():
    from mixing import has_ffmpeg

    if not has_ffmpeg():
        pytest.skip("ffmpeg is required for a real concat")
    pytest.importorskip("moviepy")


@pytest.mark.parametrize(
    "name", ["crossfade_transition", "trim_and_crossfade", "overlap_blend"]
)
def test_crossfade_transitions_render_a_blend_not_a_hard_cut(
    name: str, tmp_path: Path, real_media
):
    """A declared-overlap transition puts intermediate colours on screen."""
    import mixing

    transform = getattr(mixing, name)
    _duration, rows = _render(transform, tmp_path, name)

    blended = _blended_frames(rows)
    assert blended, (
        f"{name} rendered a hard cut: no frame carries both red and green. "
        f"centre pixels: {rows}"
    )


def test_crossfade_shortens_the_output_by_the_overlap(tmp_path: Path, real_media):
    """Two 1 s clips crossfaded over 0.5 s make 1.5 s, not 2 s.

    A working crossfade *consumes* the overlap; a duration equal to the sum of
    the inputs means the clips were butted and nothing blended.
    """
    import mixing

    duration, _rows = _render(mixing.crossfade_transition, tmp_path, "shorten")

    assert abs(duration - 1.5) < 0.15, f"expected ~1.5s, got {duration}s"


def test_the_overlap_follows_the_duration_the_caller_asked_for(
    tmp_path: Path, real_media
):
    """A longer fade requested via ``partial`` produces a longer blend.

    This is what makes the declaration a *parameter name* rather than a number:
    there is no second place to remember when the caller changes the duration.

    The blended-frame count alone does **not** show that. It is set by the mask
    length the transform applies to its own clips, which the caller's ``partial``
    reaches directly — so it stays ordered even when the *join* has stopped
    tracking the caller entirely (measured: pinning the padding at a constant
    0.5 s leaves both counts ordered and both outputs 1.5 s long). The output
    duration is the only observable of the padding the join actually chose:
    two 1 s clips overlapped by *d* render ``2 - d`` seconds.
    """
    import functools

    import mixing

    short = functools.partial(mixing.crossfade_transition, duration=0.3)
    long = functools.partial(mixing.crossfade_transition, duration=0.8)

    short_dir, long_dir = tmp_path / "s", tmp_path / "l"
    short_dir.mkdir()
    long_dir.mkdir()

    d_short, rows_short = _render(short, short_dir, "short")
    d_long, rows_long = _render(long, long_dir, "long")

    assert len(_blended_frames(rows_long)) > len(_blended_frames(rows_short))
    assert abs(d_short - 1.7) < 0.15, f"a 0.3s fade should give ~1.7s, got {d_short}s"
    assert abs(d_long - 1.2) < 0.15, f"a 0.8s fade should give ~1.2s, got {d_long}s"


@pytest.mark.parametrize("name", ["fade_through_black", "slow_motion_blend"])
def test_bake_in_transitions_are_not_given_an_overlap(name: str):
    """Transitions that bake their effect into their own frames declare nothing.

    ``FadeIn``/``FadeOut`` and speed ramps change each clip's own pixels, so a
    back-to-back join renders them correctly; overlapping them would eat
    footage for no blend. Guarding the negative matters as much as the
    positive — an over-eager declaration silently shortens their output.
    """
    import mixing

    assert mixing.crossfade_overlap(getattr(mixing, name)) is None


def test_declared_overlaps_are_the_transitions_own_defaults():
    """The declaration reads the function's parameter, not a hard-coded copy."""
    import mixing

    assert mixing.crossfade_overlap(mixing.crossfade_transition) == 0.5
    assert mixing.crossfade_overlap(mixing.trim_and_crossfade) == 0.4
    assert mixing.crossfade_overlap(mixing.overlap_blend) == 0.5
    assert mixing.crossfade_overlap(None) is None
    assert mixing.crossfade_overlap(lambda clips: clips) is None


def test_a_new_transition_declares_its_overlap_without_editing_concat():
    """Adding a transition is a decoration, not a branch somewhere else.

    The open-closed half of the fix: ``concatenate_videos`` never learns any
    transition by name.
    """
    import mixing

    @mixing.needs_crossfade_overlap("fade_seconds")
    def seventh_transition(clips, *, fade_seconds=0.25):
        return clips

    assert mixing.crossfade_overlap(seventh_transition) == 0.25


def test_explicit_concat_kwargs_win_over_the_chosen_join(tmp_path: Path, real_media):
    """A caller who passes ``method``/``padding`` keeps them.

    The join is a default, not an override — forcing ``chain`` back on gets the
    old back-to-back behaviour (and, deliberately, the hard cut with it).
    """
    import mixing

    clips = _two_flat_clips(tmp_path)
    out = tmp_path / "forced.mp4"
    result = mixing.concatenate_videos(
        clips,
        transform_clips=mixing.crossfade_transition,
        output=str(out),
        normalize_dimensions=False,
        method="chain",
        padding=0,
    )
    try:
        duration, rows = _centre_pixels(str(out))
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            close()

    assert abs(duration - 2.0) < 0.15, "the caller's chain join was overridden"
    assert not _blended_frames(rows)


# --------------------------------------------------------------------------- #
# The audio at the join: crossfaded, or summed on top of itself?
# --------------------------------------------------------------------------- #
#
# An overlapped join composites the overlapping AUDIO too, with moviepy's
# `CompositeAudioClip`, which SUMS it. `vfx.CrossFadeIn`/`CrossFadeOut` are
# video-mask effects and touch no sound. So the first version of the overlapped
# join played both tracks at FULL level through every join: measured +3.01 dB,
# and on ordinary material (-3 dBFS source) 20.7 % of the overlap's samples
# pinned against the +-0.99 rail with 4.2 % third-harmonic distortion
# manufactured out of nothing. These tests decode the samples with ffmpeg.

SAMPLE_RATE = 44100
OUTGOING_HZ, INCOMING_HZ = 440.0, 1000.0  # distinct, so each is measurable alone
TONE_PEAK = 0.7  # an ordinary mastering level (-3.1 dBFS), well clear of the rail
RAMP_RATIO = 3.0  # how far a track must fall across the overlap to count as a fade


def _tone_clips(d: Path, *, tones=None, duration: float = 1.0, fps: int = 10):
    """One flat-colour clip per tone, each a single pure sine, written losslessly.

    Distinct frequencies are the point: a lock-in at one of them measures that
    clip's own contribution to the mix, rather than inferring it from an
    aggregate level that two tracks could reach in more than one way.
    """
    import numpy as np

    try:
        from moviepy import ColorClip
    except Exception:  # pragma: no cover - moviepy<2 layout
        from moviepy.editor import ColorClip
    from moviepy.audio.AudioClip import AudioArrayClip

    tones = tones or (OUTGOING_HZ, INCOMING_HZ)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    paths = []
    for i, hz in enumerate(tones):
        t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
        wave = TONE_PEAK * np.sin(2 * np.pi * hz * t)
        clip = ColorClip(size=(32, 32), color=colors[i], duration=duration).with_fps(fps)
        clip = clip.with_audio(
            AudioArrayClip(np.column_stack([wave, wave]), fps=SAMPLE_RATE)
        )
        p = d / f"tone{i}.mov"
        clip.write_videofile(str(p), audio_codec="pcm_s16le", logger=None)
        clip.close()
        paths.append(str(p))
    return paths


def _decode_audio(path):
    """Mono float samples of ``path``, decoded by ffmpeg alone (never moviepy)."""
    import subprocess

    import numpy as np

    from mixing.util import ffmpeg_exe

    raw = subprocess.run(
        [
            ffmpeg_exe(), "-v", "error", "-i", str(path),
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0


def _tone_amplitude(samples, hz: float) -> float:
    """Amplitude of the ``hz`` component of ``samples`` — phase-independent."""
    import numpy as np

    if not len(samples):
        return 0.0
    t = np.arange(len(samples)) / SAMPLE_RATE
    return float(
        np.hypot(
            2 * (samples * np.cos(2 * np.pi * hz * t)).mean(),
            2 * (samples * np.sin(2 * np.pi * hz * t)).mean(),
        )
    )


@pytest.mark.parametrize(
    "name", ["crossfade_transition", "trim_and_crossfade", "overlap_blend"]
)
def test_the_overlapping_audio_is_crossfaded_not_summed(
    name: str, tmp_path: Path, real_media
):
    """Through the overlap each track ramps, and the sum never gets louder.

    Two rails, because they fail for different reasons. A *level* assertion
    alone would pass a join that ducked both tracks by 6 dB and still played
    them simultaneously; a *ramp* assertion alone would pass a join that
    ramped and then clipped. Together they are what "crossfade" means.
    """
    import numpy as np

    import mixing

    transform = getattr(mixing, name)
    overlap = mixing.crossfade_overlap(transform)
    clip_seconds = 1.0
    clips = _tone_clips(tmp_path, duration=clip_seconds)
    out = tmp_path / f"{name}-audio.mov"
    result = mixing.concatenate_videos(
        clips,
        transform_clips=transform,
        output=str(out),
        normalize_dimensions=False,
        audio_codec="pcm_s16le",
    )
    try:
        samples = _decode_audio(out)
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            close()

    def window(t0, t1):
        return samples[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]

    # The first clip is never re-timed, so it spans [0, clip_seconds] and the
    # overlap is its last `overlap` seconds — whatever the second clip's length.
    join = clip_seconds - overlap
    solo = window(0.05, join - 0.05)
    over = window(join + 0.02, clip_seconds - 0.02)
    eighth = (overlap - 0.04) / 8
    head = window(join + 0.02, join + 0.02 + eighth)
    tail = window(clip_seconds - 0.02 - eighth, clip_seconds - 0.02)

    solo_peak = float(np.abs(solo).max())
    over_peak = float(np.abs(over).max())

    # 1. A crossfade never gets LOUDER than either track playing alone. The
    #    summing bug measured 1.99x here (+3.01 dB), or a rail full of clipped
    #    samples once the source was hot enough for the sum to exceed 1.0.
    assert over_peak <= solo_peak * 1.05, (
        f"{name}: the overlap peaks at {over_peak:.4f} against {solo_peak:.4f} "
        f"for the same material playing alone — the tracks were summed, not "
        f"crossfaded"
    )
    assert np.abs(samples).max() < 0.985, f"{name}: samples are clipped"

    # 2. …and each track actually ramps across the overlap. Summing leaves both
    #    FLAT at their solo amplitude, which is how the bug sounded: two
    #    speakers talking over each other rather than one handing over.
    out_head = _tone_amplitude(head, OUTGOING_HZ)
    out_tail = _tone_amplitude(tail, OUTGOING_HZ)
    in_head = _tone_amplitude(head, INCOMING_HZ)
    in_tail = _tone_amplitude(tail, INCOMING_HZ)
    assert out_head > out_tail * RAMP_RATIO, (
        f"{name}: the outgoing tone is {out_head:.3f} -> {out_tail:.3f} across "
        f"the overlap; it should fade out, not hold level"
    )
    assert in_tail > in_head * RAMP_RATIO, (
        f"{name}: the incoming tone is {in_head:.3f} -> {in_tail:.3f} across "
        f"the overlap; it should fade in, not hold level"
    )


# --------------------------------------------------------------------------- #
# The overlap versus the clips: nothing may be dropped, and never in silence
# --------------------------------------------------------------------------- #
#
# An overlap longer than the clips can carry does not render a longer
# crossfade. moviepy lays clips out at `cumsum(durations) + padding*arange`, so
# clip i and clip i+2 land on the same instant and the later paints over the
# middle one; and a middle clip's CrossFadeIn and CrossFadeOut masks multiply,
# capping its opacity at (d / 2*overlap)**2. Measured on the fixture below,
# three 0.6 s clips came out as a 0.5 s video in which two of the three never
# appeared. The render succeeded. Nothing warned.

FULL_STRENGTH = 200  # 8-bit channel mean at which a flat clip is "on screen"
FLAT_CLIP_SIZE = (64, 48)  # the frames' own size, so decoding needs no ffprobe


def _flat_clips(d: Path, *, duration: float, fps: int = 20) -> list[str]:
    """Three clips, each a flat saturated primary — disjoint in channel space."""
    try:
        from moviepy import ColorClip
    except Exception:  # pragma: no cover - moviepy<2 layout
        from moviepy.editor import ColorClip

    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = d / f"flat{i}.mp4"
        ColorClip(size=FLAT_CLIP_SIZE, color=color, duration=duration).with_fps(
            fps
        ).write_videofile(str(p), logger=None)
        paths.append(str(p))
    return paths


def _frame_means(path, size=FLAT_CLIP_SIZE):
    """Whole-frame RGB mean of every decoded frame, via ffmpeg alone."""
    import subprocess

    import numpy as np

    from mixing.util import ffmpeg_exe

    w, h = size
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    flat = np.frombuffer(raw, dtype=np.uint8)
    n = len(flat) // (w * h * 3)
    assert n, f"{path}: decoded no frames at {w}x{h}"
    return flat[: n * w * h * 3].reshape(n, h, w, 3).astype(float).mean(axis=(1, 2))


def _clips_that_reached_the_screen(means) -> set:
    """Indices of the source primaries that are ever on screen at full strength."""
    shown = set()
    for mean in means:
        channel = int(mean.argmax())
        if mean[channel] >= FULL_STRENGTH:
            shown.add(channel)
    return shown


@pytest.mark.parametrize(
    "name", ["crossfade_transition", "trim_and_crossfade", "overlap_blend"]
)
def test_no_clip_is_dropped_when_it_is_shorter_than_the_declared_overlap(
    name: str, tmp_path: Path, real_media
):
    """Every clip reaches the screen, and the clamp that made it so says so."""
    import mixing

    import warnings as _warnings

    clips = _flat_clips(tmp_path, duration=0.6)  # under 2x every declared overlap
    out = tmp_path / f"{name}-short.mp4"
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = mixing.concatenate_videos(
            clips,
            transform_clips=getattr(mixing, name),
            output=str(out),
            normalize_dimensions=False,
        )
        close = getattr(result, "close", None)
        if callable(close):
            close()

    means = _frame_means(out)
    shown = _clips_that_reached_the_screen(means)
    assert shown == {0, 1, 2}, (
        f"{name}: source clips {sorted({0, 1, 2} - shown)} never reach the "
        f"screen — footage was silently dropped. Frame means: {means.tolist()}"
    )
    assert any(
        "never appear in the output" in str(w.message) for w in caught
    ), f"{name}: the overlap was clamped without saying so; warnings: {caught}"


@pytest.mark.parametrize(
    "name", ["crossfade_transition", "trim_and_crossfade", "overlap_blend"]
)
def test_clips_long_enough_for_the_overlap_are_joined_without_a_word(
    name: str, tmp_path: Path, real_media
):
    """The negative control, plus: the join eats the overlap ONCE per join.

    A guard that always fires guards nothing, so these clips are comfortably
    longer than every declared overlap and nothing may be clamped. The
    footage-budget assertion is the other half: ``overlap_blend`` used to trim
    ``overlap`` off each clip's head *as well as* overlapping the join, so every
    clip after the first paid twice.
    """
    import mixing

    import warnings as _warnings

    clip_seconds, fps = 1.4, 20  # over 2x every declared overlap
    clips = _flat_clips(tmp_path, duration=clip_seconds, fps=fps)
    out = tmp_path / f"{name}-long.mp4"
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = mixing.concatenate_videos(
            clips,
            transform_clips=getattr(mixing, name),
            output=str(out),
            normalize_dimensions=False,
        )
        close = getattr(result, "close", None)
        if callable(close):
            close()

    means = _frame_means(out)
    assert _clips_that_reached_the_screen(means) == {0, 1, 2}

    joins = len(clips) - 1
    overlap = mixing.crossfade_overlap(getattr(mixing, name))
    consumed = len(clips) * clip_seconds - len(means) / fps
    # One overlap per join — plus, for `trim_and_crossfade`, the single frame it
    # drops off each clip after the first, and a frame of encode rounding.
    budget = joins * overlap + (joins + 1) / fps
    assert consumed <= budget, (
        f"{name}: {consumed:.3f}s of source went missing across {joins} joins, "
        f"which can only account for {budget:.3f}s"
    )

    clamped = [w for w in caught if "never appear in the output" in str(w.message)]
    assert not clamped, f"{name}: clamped clips that did not need it: {clamped}"


def test_two_clips_may_overlap_further_than_three_can(tmp_path: Path, real_media):
    """The ceiling is per-clip, not "half the shortest".

    A crossfade eats the overlap off each end it touches, so the first and last
    clip pay once and everything between them pays twice. "Half the shortest
    clip" is the three-clip case of that rule, and applying it to a pair would
    refuse a perfectly renderable 0.8 s dissolve between two 1 s clips.
    """
    import functools

    import mixing

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        duration, rows = _render(
            functools.partial(mixing.crossfade_transition, duration=0.8),
            tmp_path,
            "wide",
        )

    assert abs(duration - 1.2) < 0.15, f"expected ~1.2s, got {duration}s"
    assert _blended_frames(rows)
    assert not [w for w in caught if "never appear in the output" in str(w.message)]


# --------------------------------------------------------------------------- #
# A declaration that cannot be read must not restore the hard cut in silence
# --------------------------------------------------------------------------- #


def test_an_unreadable_declaration_is_refused_where_it_is_written():
    """A typo or a ``**kwargs`` signature fails at import, not at render.

    ``crossfade_overlap`` has nothing to read in either case, so it returns
    ``None`` and the join goes back to back — which is exactly the hard cut
    issue #33 was filed for, restored silently by a one-character mistake.
    """
    import mixing

    with pytest.raises(TypeError, match="no such parameter"):

        @mixing.needs_crossfade_overlap("dur")  # the parameter is 'duration'
        def typo(clips, *, duration=0.5):
            return clips

    with pytest.raises(TypeError, match="no such parameter"):

        @mixing.needs_crossfade_overlap("duration")  # nothing to read it from
        def catch_all(clips, **kwargs):
            return clips


def test_a_declaration_with_no_value_to_read_warns():
    """Declared, legal at decoration time, unresolvable at render time."""
    import mixing

    @mixing.needs_crossfade_overlap("fade")
    def no_default(clips, *, fade):
        return clips

    with pytest.warns(UserWarning, match="no value reached it"):
        assert mixing.crossfade_overlap(no_default) is None


def test_a_declaration_bound_to_something_that_is_not_seconds_warns():
    """``partial(fn, fade='half a second')`` is a hard cut waiting to happen."""
    import functools

    import mixing

    @mixing.needs_crossfade_overlap("fade")
    def fading(clips, *, fade=0.5):
        return clips

    with pytest.warns(UserWarning, match="is not a number of seconds"):
        assert (
            mixing.crossfade_overlap(functools.partial(fading, fade="half")) is None
        )


def test_a_partial_binds_the_overlap_positionally_too():
    """``partial`` is seen through by both halves, not just its keywords."""
    import functools

    import mixing

    @mixing.needs_crossfade_overlap("fade")
    def positional_fade(clips=None, fade=0.5):
        return clips

    assert mixing.crossfade_overlap(positional_fade) == 0.5
    assert mixing.crossfade_overlap(functools.partial(positional_fade, None, 0.9)) == 0.9
    assert (
        mixing.crossfade_overlap(functools.partial(positional_fade, fade=0.7)) == 0.7
    )


THIRD_HZ = 1600.0  # a third tone, for the three-clip clamp test below
MIDPOINT_LEVEL = 0.85  # share of solo level the middle clip must reach


def test_a_clamped_overlap_reaches_the_transform_not_only_the_join(
    tmp_path: Path, real_media
):
    """Clamping the padding alone leaves the ramps too long, and it is audible.

    The transform bakes the caller's overlap into its own fade ramps *before*
    the join is chosen, so a clamp applied only to ``padding`` leaves a
    mismatched pair: 0.8 s ramps across a 0.5 s overlap. The picture survives it
    (the composite renormalises over a transparent background — measured, both
    variants reach full colour), so the only instrument that sees it is the
    sound.

    Three 1 s clips asked to overlap by 0.8 s clamp to 0.5 s, which puts the
    middle clip alone and at full level at exactly ``t = 1.0``: its neighbours'
    ramps both cross zero there. Measured 0.98x its solo level when the clamp
    reaches the transform, 0.43x when only the padding is clamped.
    """
    import functools
    import warnings as _warnings

    import numpy as np

    import mixing

    tones = (OUTGOING_HZ, INCOMING_HZ, THIRD_HZ)
    clips = _tone_clips(tmp_path, tones=tones, duration=1.0)
    out = tmp_path / "clamped.mov"
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = mixing.concatenate_videos(
            clips,
            transform_clips=functools.partial(mixing.crossfade_transition, duration=0.8),
            output=str(out),
            normalize_dimensions=False,
            audio_codec="pcm_s16le",
        )
        close = getattr(result, "close", None)
        if callable(close):
            close()

    assert any("never appear in the output" in str(w.message) for w in caught)

    samples = _decode_audio(out)

    def window(t0, t1):
        return samples[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]

    solo = _tone_amplitude(window(0.05, 0.45), OUTGOING_HZ)
    midpoint = window(0.98, 1.02)
    middle = _tone_amplitude(midpoint, INCOMING_HZ)
    neighbours = max(
        _tone_amplitude(midpoint, OUTGOING_HZ), _tone_amplitude(midpoint, THIRD_HZ)
    )

    assert middle >= solo * MIDPOINT_LEVEL, (
        f"the middle clip peaks at {middle:.4f} against a solo level of "
        f"{solo:.4f} ({middle / solo:.2f}x) — its fade ramps are longer than the "
        f"overlap the join actually used, so it never reaches full level"
    )
    assert neighbours < solo * 0.15, (
        f"at the middle clip's peak its neighbours are still at "
        f"{neighbours:.4f} against {solo:.4f}: the ramps do not line up with "
        f"the join"
    )
    assert np.abs(samples).max() < 0.985, "samples are clipped"
