"""Characterization tests for ``mixing.video.video_util``.

These tests PIN the current observable behavior of the three public helpers in
``mixing/video/video_util.py`` so an upcoming refactor cannot silently change
their semantics:

- ``get_video_dimensions(video)`` -> ``(width, height)`` tuple read off the clip.
  It stays in ``mixing`` on purpose: it is a *probe*, not geometry, and ``paces``
  calls it at five sites behind an explicit ``mixing`` floor.
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
    SOCIAL_SIZES,
    get_video_dimensions,
    resize_to_dimensions,
    normalize_video_dimensions,
)


def test_social_sizes_presets_and_export():
    """``SOCIAL_SIZES`` holds the expected (w, h) presets and is re-exported."""
    assert SOCIAL_SIZES == {
        "youtube": (1920, 1080),
        "shorts": (1080, 1920),
        "square": (1080, 1080),
        "story": (1080, 1920),
        "tiktok": (1080, 1920),
    }
    # Re-exported from the subpackage namespace (same object).
    import mixing.video as mv

    assert mv.SOCIAL_SIZES is SOCIAL_SIZES


def test_social_sizes_is_looks_dict_not_a_copy():
    """The presets are `looks`' vocabulary, imported — not a second copy here.

    Identity, not equality: two dicts that happen to agree today are exactly
    what drifts tomorrow, and the whole point of the port is that there is one
    place saying what a "shorts" is.
    """
    from looks.geometry import SOCIAL_SIZES as looks_social_sizes

    assert SOCIAL_SIZES is looks_social_sizes


def test_social_backdrop_constants_are_looks_values():
    """The blur sigma and dim factor of the ``social`` backdrop come from `looks`.

    They were transcribed *out of* this module into `looks`; importing them
    back is what stops the two copies drifting. The values are pinned here too,
    because a silent change to either is a visible change to every
    ``method='social'`` render.

    Names only — that the *render* reads them is the next two tests' job, and
    it has to be a separate test because the imported values equal the literals
    they replaced, so nothing observable changes if the code stops using them.
    """
    from looks import geometry
    from mixing.video import video_util

    assert video_util.DFLT_BACKDROP_BLUR_SIGMA is geometry.DFLT_BACKDROP_BLUR_SIGMA
    assert video_util.DFLT_BACKDROP_DIM is geometry.DFLT_BACKDROP_DIM
    assert video_util.DFLT_BACKDROP_BLUR_SIGMA == 15.0
    assert video_util.DFLT_BACKDROP_DIM == 0.7


def _textured_clip(width=320, height=240, *, fps=10, duration=0.4):
    """A high-frequency checkerboard clip — blur and dim are both visible on it.

    In memory, no encode: the ``social`` branch only ever asks a clip for
    ``w``/``h``/``duration``/``fps`` and the frame-level operations.
    """
    import numpy as np
    from moviepy import ImageClip

    ys, xs = np.mgrid[0:height, 0:width]
    checker = (((xs // 4) + (ys // 4)) % 2).astype("uint8") * 255
    frame = np.dstack([checker, checker, checker])
    return ImageClip(frame, duration=duration).with_fps(fps)


def _social_backdrop_row(monkeypatch=None, **overrides):
    """Render ``method='social'`` 4:3 -> 1:1 and return a row of pure backdrop.

    A 4:3 source fitted into a square leaves the top and bottom bands showing
    only the blurred, dimmed backdrop, so row 5 is backdrop and nothing else.
    """
    from mixing.video import video_util

    clip = _textured_clip()
    try:
        for name, value in overrides.items():
            monkeypatch.setattr(video_util, name, value)
        out = video_util.resize_to_dimensions(clip, 200, 200, method="social")
        return out.get_frame(0.1)[5].astype(int)
    finally:
        clip.close()


def test_the_social_render_reads_the_imported_blur_sigma(monkeypatch):
    """Changing ``DFLT_BACKDROP_BLUR_SIGMA`` changes the backdrop pixels.

    The behavioural half of the port: a hard-coded ``radius=15`` renders
    identically to the imported ``15.0`` today, so only moving the constant can
    tell the two apart. A checkerboard blurred at sigma 15 is nearly uniform;
    at sigma 0.5 it is still a checkerboard.
    """
    import numpy as np

    blurred = _social_backdrop_row()
    sharp = _social_backdrop_row(monkeypatch, DFLT_BACKDROP_BLUR_SIGMA=0.5)

    assert np.std(sharp) > np.std(blurred) + 20, (
        "the social backdrop ignored DFLT_BACKDROP_BLUR_SIGMA "
        f"(std blurred={np.std(blurred):.1f}, std sharp={np.std(sharp):.1f})"
    )


def test_the_social_render_reads_the_imported_dim_factor(monkeypatch):
    """Changing ``DFLT_BACKDROP_DIM`` changes how dark the backdrop is."""
    import numpy as np

    dimmed = _social_backdrop_row()
    undimmed = _social_backdrop_row(monkeypatch, DFLT_BACKDROP_DIM=1.0)

    assert np.mean(undimmed) > np.mean(dimmed) + 10, (
        "the social backdrop ignored DFLT_BACKDROP_DIM "
        f"(mean dimmed={np.mean(dimmed):.1f}, mean undimmed={np.mean(undimmed):.1f})"
    )


def test_the_resize_method_vocabulary_is_looks_plus_social():
    """``ResizeMethod`` is `looks`' three modes plus ``social`` — not a 4th literal.

    ``social`` is ``fit`` over a blurred, dimmed copy of the source rather than
    a solid colour, which is why `looks` (pure arithmetic) has three names and
    this module has four. Written as a ``Union`` of the two ``Literal``s so the
    three come from `looks`; the assertion below is what fails if someone
    flattens it back into one hand-written four-name literal.

    What this cannot check: ``typing.Literal`` is interned, so a hand-written
    ``Literal['stretch', 'fit', 'fill']`` **is** `looks`' ``FitMode`` object.
    Identity proves nothing here — the ``Union`` shape is the only observable
    difference, and it is what is asserted.
    """
    from typing import get_args

    from looks.geometry import FitMode
    from mixing.video.video_util import ResizeMethod

    assert set(get_args(FitMode)) == {"stretch", "fit", "fill"}

    union_members = get_args(ResizeMethod)
    assert FitMode in union_members, (
        "ResizeMethod no longer composes looks' FitMode — the three mode names "
        f"have been restated locally: {union_members}"
    )
    names = {name for member in union_members for name in get_args(member)}
    assert names == {"stretch", "fit", "fill", "social"}


def test_get_video_dimensions_stays_in_mixing():
    """The probe did NOT move to `looks` — ``paces`` imports it from here.

    `looks.geometry` is pure arithmetic with no file open and nothing decoded;
    ``get_video_dimensions`` opens a file (or reads a live clip), so it has no
    home there. Pinned at the top-level facade, which is the spelling ``paces``
    uses (``mixing.get_video_dimensions``).
    """
    import looks.geometry
    import mixing

    assert callable(mixing.get_video_dimensions)
    assert not hasattr(looks.geometry, "get_video_dimensions")


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
