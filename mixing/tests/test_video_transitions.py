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
    """
    import functools

    import mixing

    short = functools.partial(mixing.crossfade_transition, duration=0.3)
    long = functools.partial(mixing.crossfade_transition, duration=0.8)

    short_dir, long_dir = tmp_path / "s", tmp_path / "l"
    short_dir.mkdir()
    long_dir.mkdir()

    _d1, rows_short = _render(short, short_dir, "short")
    _d2, rows_long = _render(long, long_dir, "long")

    assert len(_blended_frames(rows_long)) > len(_blended_frames(rows_short))


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
