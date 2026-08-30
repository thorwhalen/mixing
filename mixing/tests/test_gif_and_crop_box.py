"""Tests for spatial ``crop_box`` on :func:`crop_video` and :func:`make_gif`.

Both features run on the pip-bundled imageio-ffmpeg binary (via moviepy for
mp4, via subprocess for gif), so these tests need no system ffmpeg. The
fixtures here are spatially NON-uniform on purpose: a solid-color clip cannot
distinguish a correct crop origin from a swapped one, and an adversarial
review demonstrated exactly that mutation surviving a dimensions-only suite.
"""

import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("moviepy")
pytest.importorskip("cv2")

from mixing.video import crop_video, make_gif
from mixing.video._helpers import _validated_crop_box

#: The rect fixture's geometry: green rect on blue, deliberately x != y so an
#: x/y swap in either crop path moves the asserted pixels off the rect.
RECT_XYWH = (60, 40, 100, 80)
RECT_SIZE = (320, 240)


def _rect_frame():
    import numpy as np

    width, height = RECT_SIZE
    x, y, w, h = RECT_XYWH
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)  # blue background (RGB)
    frame[y : y + h, x : x + w] = (0, 255, 0)  # green subject
    return frame


def _write_clip(path, frames, fps):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(path), codec="libx264", audio=False, logger=None)
    clip.close()
    return path


@pytest.fixture
def rect_video(tmp_path):
    """1 s, 24 fps: a green rectangle at a known, asymmetric position."""
    return _write_clip(tmp_path / "rect.mp4", [_rect_frame()] * 24, fps=24)


@pytest.fixture
def gradient_video(tmp_path):
    """0.5 s of many-colored frames — palette assertions need real entropy."""
    import numpy as np

    width, height = 64, 48
    frames = []
    for i in range(12):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        frame[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        frame[..., 2] = i * 20
        frames.append(frame)
    return _write_clip(tmp_path / "gradient.mp4", frames, fps=24)


def _mp4_dimensions(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def _mp4_frame_rgb(path, col, row):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        assert ok, f"could not read a frame from {path}"
        b, g, r = frame[row, col]
        return int(r), int(g), int(b)
    finally:
        cap.release()


def _is_green(rgb):
    r, g, b = rgb
    return g > 180 and r < 80 and b < 80


def _is_blue(rgb):
    r, g, b = rgb
    return b > 180 and r < 80 and g < 80


def _gif_frames(path):
    from PIL import Image

    with Image.open(path) as im:
        return getattr(im, "n_frames", 1)


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
        (float("inf"), 0, 10, 10),  # non-finite (used to leak OverflowError)
    ],
)
def test_validated_crop_box_rejects(box):
    with pytest.raises(ValueError):
        _validated_crop_box(box, (320, 240))


# ------------------------------- crop_video --------------------------------


def test_crop_video_crop_box_lands_on_the_subject(rect_video, tmp_path):
    out = crop_video(
        str(rect_video), 0.2, 0.8, crop_box=RECT_XYWH, output=tmp_path / "cut.mp4"
    )
    assert _mp4_dimensions(out) == (RECT_XYWH[2], RECT_XYWH[3])
    # the crop IS the rectangle: an interior pixel near the top-left is green
    # (an x/y-swapped crop would start at (40, 60) and read background here)
    assert _is_green(_mp4_frame_rgb(out, 5, 5))
    assert _is_green(_mp4_frame_rgb(out, 94, 74))


def test_crop_video_crop_box_away_from_subject_is_background(rect_video, tmp_path):
    x, y, w, h = RECT_XYWH
    out = crop_video(
        str(rect_video), 0, 0.4, crop_box=(x + w, y, w, h), output=tmp_path / "bg.mp4"
    )
    assert _is_blue(_mp4_frame_rgb(out, w // 2, h // 2))


def test_crop_video_crop_box_even_rounding(rect_video, tmp_path):
    out = crop_video(
        str(rect_video), 0, 0.4, crop_box=(0, 0, 101, 61), output=tmp_path / "odd.mp4"
    )
    assert _mp4_dimensions(out) == (100, 60)


def test_crop_video_crop_box_with_frames_time_unit(rect_video, tmp_path):
    out = crop_video(
        str(rect_video),
        2,
        20,
        time_unit="frames",
        crop_box=RECT_XYWH,
        output=tmp_path / "frames.mp4",
    )
    assert _mp4_dimensions(out) == (RECT_XYWH[2], RECT_XYWH[3])


def test_crop_video_crop_box_out_of_bounds_raises(rect_video, tmp_path):
    with pytest.raises(ValueError, match="320x240"):
        crop_video(
            str(rect_video),
            0,
            0.4,
            crop_box=(300, 0, 100, 60),
            output=tmp_path / "x.mp4",
        )


def test_crop_video_crop_box_single_frame_raises(rect_video):
    with pytest.raises(ValueError, match="single-frame"):
        crop_video(str(rect_video), 0.1, 0.1, crop_box=(0, 0, 100, 60))


def test_crop_video_single_frame_rejects_segment_kwargs(rect_video):
    with pytest.raises(ValueError, match="segment export"):
        crop_video(str(rect_video), 0.1, 0.1, codec="libx264")


def test_crop_video_single_frame_still_takes_frame_kwargs(rect_video, tmp_path):
    out = crop_video(
        str(rect_video), 0.1, 0.1, image_format="jpg", output=str(tmp_path / "f.jpg")
    )
    assert Path(out).suffix == ".jpg" and Path(out).exists()


def test_crop_video_default_path_gets_crop_suffix(rect_video):
    out = crop_video(str(rect_video), 0.2, 0.8, crop_box=RECT_XYWH)
    try:
        assert out.name == f"{rect_video.stem}_0_0_crop.mp4"
        assert out.exists()
    finally:
        Path(out).unlink(missing_ok=True)


def test_crop_video_without_crop_box_unchanged(rect_video, tmp_path):
    out = crop_video(str(rect_video), 0.2, 0.8, output=tmp_path / "plain.mp4")
    assert _mp4_dimensions(out) == RECT_SIZE


# -------------------------------- make_gif ---------------------------------


def test_make_gif_windows_and_paces_honestly(rect_video, tmp_path):
    # 1 s source; window 0.5..0.75 at 12.5 fps -> ~3 frames. Catches: the
    # window being ignored (~12), `-t end` instead of `-t end-start` (~9),
    # and a wrong fps (4x -> ~12).
    out = make_gif(str(rect_video), 0.5, 0.75, output=tmp_path / "w.gif")
    assert 2 <= _gif_frames(out) <= 4
    data = Path(out).read_bytes()
    assert data[:6] in (b"GIF87a", b"GIF89a")


def test_make_gif_default_fps_frame_count(rect_video, tmp_path):
    # 0.8 s at the default 12.5 fps -> 10 frames (+/- 1)
    out = make_gif(str(rect_video), 0.0, 0.8, width=160, output=tmp_path / "d.gif")
    assert 9 <= _gif_frames(out) <= 11

    from PIL import Image

    with Image.open(out) as im:
        assert im.size[0] == 160
        assert im.info.get("loop") == 0  # forever, the default


def test_make_gif_negative_times_count_from_the_end(rect_video, tmp_path):
    # 1 s source: start=-0.4 -> 0.6..end at 10 fps -> ~4 frames
    out = make_gif(str(rect_video), -0.4, None, fps=10, output=tmp_path / "n.gif")
    assert 3 <= _gif_frames(out) <= 5
    # end=-0.5 -> 0..0.5 -> ~5 frames
    out2 = make_gif(str(rect_video), None, -0.5, fps=10, output=tmp_path / "n2.gif")
    assert 4 <= _gif_frames(out2) <= 6


def test_make_gif_loop_parameter_is_written(rect_video, tmp_path):
    from PIL import Image

    out = make_gif(str(rect_video), 0.0, 0.4, loop=3, output=tmp_path / "l.gif")
    with Image.open(out) as im:
        assert im.info.get("loop") == 3


def test_make_gif_crop_box_lands_on_the_subject(rect_video, tmp_path):
    from PIL import Image

    out = make_gif(
        str(rect_video), crop_box=RECT_XYWH, width=None, output=tmp_path / "c.gif"
    )
    with Image.open(out) as im:
        assert im.size == (RECT_XYWH[2], RECT_XYWH[3])
        rgb = im.convert("RGB").getpixel((5, 5))
        assert _is_green(rgb)


def test_make_gif_crop_box_scales_from_crop_aspect(rect_video, tmp_path):
    from PIL import Image

    out = make_gif(
        str(rect_video),
        crop_box=(10, 10, 100, 200),
        width=50,
        output=tmp_path / "s.gif",
    )
    with Image.open(out) as im:
        assert im.size == (50, 100)


def test_make_gif_colors_shapes_the_palette(gradient_video, tmp_path):
    # colors=4 on a many-colored source: the frame may use at most 4 colors.
    # Catches both a hardcoded palette size and a single-pass regression
    # (no palettegen -> ffmpeg's generic 256-color table).
    from PIL import Image

    out = make_gif(str(gradient_video), colors=4, width=None, output=tmp_path / "p.gif")
    with Image.open(out) as im:
        unique = im.convert("RGB").getcolors(maxcolors=4096)
        assert unique is not None and len(unique) <= 4


def test_make_gif_native_size_when_width_none(rect_video, tmp_path):
    from PIL import Image

    out = make_gif(str(rect_video), width=None, output=tmp_path / "nat.gif")
    with Image.open(out) as im:
        assert im.size == RECT_SIZE


def test_make_gif_default_path_keeps_subsecond_precision(rect_video):
    # int-truncated names made (0.2, 0.8) and (0.3, 0.85) silently clobber
    # each other; the default name now carries the exact window
    out = make_gif(str(rect_video), 0.0, 0.4)
    try:
        assert out.name == f"{rect_video.stem}_0_0.4.gif"
        assert out.exists()
    finally:
        Path(out).unlink(missing_ok=True)
    out2 = make_gif(str(rect_video), 0.25)
    try:
        assert out2.name == f"{rect_video.stem}_0.25_end.gif"
    finally:
        Path(out2).unlink(missing_ok=True)


def test_make_gif_from_gif_source_default_does_not_collide(rect_video, tmp_path):
    src_gif = make_gif(str(rect_video), 0.0, 0.4, output=tmp_path / "src.gif")
    out = make_gif(str(src_gif), width=32)
    try:
        assert Path(out) != Path(src_gif)
        assert out.name == "src_gif.gif"
        assert out.exists()
    finally:
        Path(out).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(colors=2), "colors"),  # palettegen's real minimum is 3
        (dict(colors=257), "colors"),
        (dict(width=1), "width"),
        (dict(fps=0), "fps"),
        (dict(start=0.3, end=0.2), "empty"),
        (dict(start=5.0), "outside"),  # start past a 1 s source
        (dict(crop_box=(0, 0, 999, 999)), "frame"),
    ],
)
def test_make_gif_rejects_bad_inputs(rect_video, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_gif(str(rect_video), **kwargs)


# --------------------------- error contracts -------------------------------


def test_make_gif_ffmpeg_failure_surfaces_stderr_and_cleans_palette(
    rect_video, tmp_path, monkeypatch
):
    from mixing.video import gif as gif_module

    created = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(name)
        return fd, name

    monkeypatch.setattr(gif_module.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(gif_module, "_PALETTEUSE", "nosuchfilter")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        gif_module.make_gif(str(rect_video), 0.0, 0.4, output=tmp_path / "x.gif")
    assert created, "the palette temp file was never created"
    assert not Path(created[-1]).exists(), "palette temp file leaked on failure"


def test_make_gif_cleans_palette_on_success(rect_video, tmp_path, monkeypatch):
    from mixing.video import gif as gif_module

    created = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(name)
        return fd, name

    monkeypatch.setattr(gif_module.tempfile, "mkstemp", recording_mkstemp)
    out = gif_module.make_gif(str(rect_video), 0.0, 0.4, output=tmp_path / "ok.gif")
    assert Path(out).exists()
    assert created and not Path(created[-1]).exists()


def test_ffmpeg_exe_falls_back_to_path_and_reports_absence(monkeypatch):
    import shutil as shutil_module

    from mixing import util

    # imageio_ffmpeg unavailable -> PATH lookup wins
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(shutil_module, "which", lambda name: "/fake/ffmpeg")
    assert util.ffmpeg_exe() == "/fake/ffmpeg"
    # ...and with no PATH ffmpeg either, an informative error names remedies
    monkeypatch.setattr(shutil_module, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        util.ffmpeg_exe()


def test_get_video_dimensions_accepts_a_path(rect_video):
    from mixing.video import get_video_dimensions

    assert tuple(get_video_dimensions(str(rect_video))) == RECT_SIZE
    with pytest.raises(ValueError, match="Cannot open"):
        get_video_dimensions("no/such/file.mp4")
