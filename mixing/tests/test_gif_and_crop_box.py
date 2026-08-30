"""Tests for spatial ``crop_box`` on :func:`crop_video` and :func:`make_gif`.

Both features run on the pip-bundled imageio-ffmpeg binary (via moviepy for
mp4, via subprocess for gif), so these tests need no system ffmpeg.
"""

import pytest

pytest.importorskip("moviepy")
pytest.importorskip("cv2")

from pathlib import Path

from mixing.video import crop_video, make_gif
from mixing.video._helpers import _validated_crop_box


def _mp4_dimensions(path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


# --------------------------- _validated_crop_box ---------------------------


def test_validated_crop_box_accepts_and_even_rounds():
    assert _validated_crop_box((10, 20, 100, 50), (320, 240)) == (10, 20, 100, 50)
    # odd width/height are floored to even, origin untouched
    assert _validated_crop_box((10, 20, 101, 51), (320, 240)) == (10, 20, 100, 50)


@pytest.mark.parametrize(
    "box",
    [
        (10, 20, 400, 50),  # exceeds width
        (10, 200, 100, 50),  # exceeds height
        (-1, 0, 100, 50),  # negative origin
        (0, 0, 0, 50),  # zero width
        (0, 0, 1, 50),  # under 2px after even-rounding
        ("a", 0, 10, 10),  # malformed
        (0, 0, 10),  # wrong arity
    ],
)
def test_validated_crop_box_rejects(box):
    with pytest.raises(ValueError):
        _validated_crop_box(box, (320, 240))


# ------------------------------- crop_video --------------------------------


def test_crop_video_with_crop_box_crops_spatially(make_color_video, tmp_path):
    src = make_color_video(1.0, size=(320, 240))
    out = crop_video(
        str(src), 0.2, 0.8, crop_box=(10, 20, 100, 60), output=tmp_path / "cut.mp4"
    )
    assert Path(out).exists()
    assert _mp4_dimensions(out) == (100, 60)


def test_crop_video_crop_box_even_rounding(make_color_video, tmp_path):
    src = make_color_video(0.5, size=(320, 240))
    out = crop_video(
        str(src), 0, 0.4, crop_box=(0, 0, 101, 61), output=tmp_path / "odd.mp4"
    )
    assert _mp4_dimensions(out) == (100, 60)


def test_crop_video_crop_box_out_of_bounds_raises(make_color_video, tmp_path):
    src = make_color_video(0.5, size=(320, 240))
    with pytest.raises(ValueError, match="320x240"):
        crop_video(
            str(src), 0, 0.4, crop_box=(300, 0, 100, 60), output=tmp_path / "x.mp4"
        )


def test_crop_video_crop_box_single_frame_raises(make_color_video):
    src = make_color_video(0.5)
    with pytest.raises(ValueError, match="single-frame"):
        crop_video(str(src), 0.1, 0.1, crop_box=(0, 0, 100, 60))


def test_crop_video_without_crop_box_unchanged(make_color_video, tmp_path):
    src = make_color_video(1.0, size=(320, 240))
    out = crop_video(str(src), 0.2, 0.8, output=tmp_path / "plain.mp4")
    assert _mp4_dimensions(out) == (320, 240)


# -------------------------------- make_gif ---------------------------------


def test_make_gif_produces_animated_gif(make_color_video, tmp_path):
    src = make_color_video(1.0, fps=24, size=(320, 240))
    out = make_gif(str(src), 0.0, 0.8, width=160, output=tmp_path / "loop.gif")
    assert Path(out).exists()
    data = Path(out).read_bytes()
    assert data[:6] in (b"GIF87a", b"GIF89a")

    from PIL import Image

    with Image.open(out) as im:
        assert im.size[0] == 160
        assert getattr(im, "n_frames", 1) >= 2
        # loop=0 (forever) is recorded in the GIF's netscape extension
        assert im.info.get("loop", None) == 0


def test_make_gif_with_crop_box_scales_from_crop_aspect(make_color_video, tmp_path):
    src = make_color_video(0.6, size=(320, 240))
    out = make_gif(
        str(src), crop_box=(10, 10, 100, 200), width=50, output=tmp_path / "c.gif"
    )
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (50, 100)


def test_make_gif_native_size_when_width_none(make_color_video, tmp_path):
    src = make_color_video(0.5, size=(64, 48))
    out = make_gif(str(src), width=None, output=tmp_path / "n.gif")
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (64, 48)


def test_make_gif_default_path_beside_input(make_color_video):
    src = make_color_video(0.5)
    out = make_gif(str(src), 0.0, 0.4)
    try:
        assert out == src.with_stem(f"{src.stem}_0_0").with_suffix(".gif")
        assert out.exists()
    finally:
        Path(out).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(colors=1), "colors"),
        (dict(start=2.0, end=1.0), "after start"),
        (dict(crop_box=(0, 0, 999, 999)), "frame"),
    ],
)
def test_make_gif_rejects_bad_inputs(make_color_video, kwargs, match):
    src = make_color_video(0.5, size=(320, 240))
    with pytest.raises((ValueError,), match=match):
        make_gif(str(src), **kwargs)
