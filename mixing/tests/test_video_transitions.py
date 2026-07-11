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
