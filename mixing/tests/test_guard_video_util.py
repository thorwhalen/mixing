"""Characterization tests for ``mixing.video.video_util``.

These tests PIN the current observable behavior of the three public helpers in
``mixing/video/video_util.py`` so an upcoming refactor cannot silently change
their semantics:

- ``get_video_dimensions(video)`` -> ``(width, height)`` tuple read off the clip.
- ``resize_to_dimensions(video, target_width, target_height, *, method, bg_color)``
  -> a clip whose ``(w, h)`` equals the requested target for every supported
  ``method`` ('stretch', 'fit', 'fill', 'social'); unknown methods raise
  ``ValueError``.
- ``normalize_video_dimensions(videos, *, reference_video, target_width,
  target_height, method, bg_color)`` -> a list of clips all sharing one target
  size, resolved from an index reference, a clip reference, or explicit
  dimensions; an empty input returns ``[]``.

All clips are derived from the shared ``make_color_video`` factory (synthetic,
auto-cleaned). No network or API keys are involved. The module is skipped if
``moviepy`` is unavailable.
"""

import pytest

pytest.importorskip("moviepy")

import moviepy as mp  # noqa: E402

from mixing.video.video_util import (  # noqa: E402
    get_video_dimensions,
    resize_to_dimensions,
    normalize_video_dimensions,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def open_clip():
    """Factory that opens a ``VideoFileClip`` from a path and auto-closes it.

    ``video_util`` operates on loaded clips, while the shared factories yield
    file *paths*; this bridges the two and guarantees cleanup of the moviepy
    readers spawned during the test.
    """
    opened = []

    def _open(path):
        clip = mp.VideoFileClip(str(path))
        opened.append(clip)
        return clip

    yield _open

    for c in opened:
        try:
            c.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# get_video_dimensions
# --------------------------------------------------------------------------- #


def test_get_video_dimensions_returns_width_height_tuple(make_color_video, open_clip):
    """Returns a plain ``(width, height)`` int tuple matching the source size."""
    path = make_color_video(1.0, size=(320, 240))
    clip = open_clip(path)

    dims = get_video_dimensions(clip)

    assert isinstance(dims, tuple)
    assert len(dims) == 2
    assert dims == (320, 240)
    # Order is (width, height), not (height, width).
    width, height = dims
    assert width == 320
    assert height == 240


def test_get_video_dimensions_matches_distinct_size(make_color_video, open_clip):
    """A differently-sized source yields its own dimensions (not a constant)."""
    clip = open_clip(make_color_video(1.0, size=(200, 200)))
    assert get_video_dimensions(clip) == (200, 200)


# --------------------------------------------------------------------------- #
# resize_to_dimensions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["stretch", "fit", "fill", "social"])
def test_resize_to_dimensions_hits_target_for_each_method(
    method, make_color_video, open_clip
):
    """Every supported method returns a clip at exactly the requested size.

    Source is 320x240 (4:3); target is 200x200 (1:1) so each method must
    actively reshape rather than no-op. ``stretch`` / ``fill`` keep the
    ``VideoFileClip`` type while ``fit`` / ``social`` wrap in a
    ``CompositeVideoClip`` — but all expose the target ``(w, h)``.
    """
    clip = open_clip(make_color_video(1.0, size=(320, 240)))

    resized = resize_to_dimensions(clip, 200, 200, method=method)

    assert (resized.w, resized.h) == (200, 200)
    # Duration is preserved (within encoder tolerance).
    assert abs(resized.duration - clip.duration) < 0.2


def test_resize_to_dimensions_default_method_is_fit(make_color_video, open_clip):
    """The default ``method`` is 'fit', producing a CompositeVideoClip here.

    With a 4:3 source padded into a 1:1 target, 'fit' must letterbox via a
    ``CompositeVideoClip`` composite, so the default call matches an explicit
    ``method='fit'`` in both type and size.
    """
    from moviepy import CompositeVideoClip

    clip = open_clip(make_color_video(1.0, size=(320, 240)))

    default = resize_to_dimensions(clip, 200, 200)

    assert isinstance(default, CompositeVideoClip)
    assert (default.w, default.h) == (200, 200)


def test_resize_to_dimensions_stretch_preserves_videofileclip_type(
    make_color_video, open_clip
):
    """'stretch' returns the same VideoFileClip kind, resized to target."""
    clip = open_clip(make_color_video(1.0, size=(320, 240)))

    resized = resize_to_dimensions(clip, 160, 120, method="stretch")

    assert isinstance(resized, mp.VideoFileClip)
    assert (resized.w, resized.h) == (160, 120)


def test_resize_to_dimensions_unknown_method_raises_valueerror(
    make_color_video, open_clip
):
    """An unsupported ``method`` raises ValueError naming the valid options."""
    clip = open_clip(make_color_video(1.0, size=(320, 240)))

    with pytest.raises(ValueError) as exc_info:
        resize_to_dimensions(clip, 100, 100, method="bogus")

    msg = str(exc_info.value)
    assert "Unknown method" in msg
    assert "stretch" in msg and "fit" in msg and "fill" in msg and "social" in msg


def test_resize_to_dimensions_accepts_bg_color_for_fit_padding(
    make_color_video, open_clip
):
    """'fit' accepts a custom ``bg_color`` and still hits the target size.

    Pins that padded letterboxing is reachable with a non-default background
    color (the padding branch only runs when the fitted clip is smaller than
    the target on at least one axis, as here: 4:3 source into 1:1 target).
    """
    clip = open_clip(make_color_video(1.0, size=(320, 240)))

    resized = resize_to_dimensions(
        clip, 200, 200, method="fit", bg_color=(10, 20, 30)
    )

    assert (resized.w, resized.h) == (200, 200)


# --------------------------------------------------------------------------- #
# normalize_video_dimensions
# --------------------------------------------------------------------------- #


def test_normalize_empty_list_returns_empty_list():
    """An empty ``videos`` input short-circuits to an empty list."""
    assert normalize_video_dimensions([]) == []


def test_normalize_uses_first_video_as_default_reference(
    make_color_video, open_clip
):
    """Default ``reference_video=0`` targets the first clip's dimensions.

    The already-correct first clip is returned unchanged (identity), while the
    second is resized to match it.
    """
    v1 = open_clip(make_color_video(1.0, size=(320, 240)))
    v2 = open_clip(make_color_video(1.0, size=(200, 200)))

    out = normalize_video_dimensions([v1, v2], method="fit")

    assert len(out) == 2
    assert all((c.w, c.h) == (320, 240) for c in out)
    # The reference clip (already at target size) is passed through untouched.
    assert out[0] is v1


def test_normalize_with_explicit_target_dimensions(make_color_video, open_clip):
    """Explicit ``target_width``/``target_height`` override the reference."""
    v1 = open_clip(make_color_video(1.0, size=(320, 240)))
    v2 = open_clip(make_color_video(1.0, size=(200, 200)))

    out = normalize_video_dimensions(
        [v1, v2], target_width=160, target_height=120, method="stretch"
    )

    assert [(c.w, c.h) for c in out] == [(160, 120), (160, 120)]


def test_normalize_with_clip_instance_as_reference(make_color_video, open_clip):
    """A ``VideoFileClip`` passed as ``reference_video`` sets the target size.

    The clip that equals the reference size passes through by identity.
    """
    v1 = open_clip(make_color_video(1.0, size=(320, 240)))
    v2 = open_clip(make_color_video(1.0, size=(200, 200)))

    out = normalize_video_dimensions([v1, v2], reference_video=v2, method="fit")

    assert all((c.w, c.h) == (200, 200) for c in out)
    # v2 already matches the (its own) reference size and is returned as-is.
    assert out[1] is v2


def test_normalize_reference_index_selects_other_video(make_color_video, open_clip):
    """``reference_video`` as an int index picks that clip's dimensions."""
    v1 = open_clip(make_color_video(1.0, size=(320, 240)))
    v2 = open_clip(make_color_video(1.0, size=(200, 200)))

    out = normalize_video_dimensions(
        [v1, v2], reference_video=1, method="fit"
    )

    assert all((c.w, c.h) == (200, 200) for c in out)
    assert out[1] is v2
