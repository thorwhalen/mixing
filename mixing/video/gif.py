"""Animated-GIF export via ffmpeg's two-pass palette pipeline.

GIF quality lives or dies on the palette: a single-pass encode dithers against
a generic 256-color table and produces large, banded files. The two-pass
recipe here — ``palettegen`` (per-clip palette, ``stats_mode=diff`` so colors
go to what *changes*) then ``paletteuse`` with ordered bayer dithering —
produces the small, stable loops this module exists for. moviepy's
``write_gif`` (imageio/pillow single-pass) is deliberately not used.

The ffmpeg binary is resolved via :func:`mixing.util.ffmpeg_exe`, which
prefers the pip-bundled ``imageio-ffmpeg`` binary — so this works on machines
with no system ffmpeg installed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from ..egress import Output, write_egress
from ..util import ffmpeg_exe
from ._helpers import _auto_video_path, _validated_crop_box

#: Defaults proven out on real dance-loop media: 12.5 fps halves a 25 fps
#: source evenly; 460 px width keeps a portrait crop under ~2 MB for a
#: few-second loop; 160 colors is where palette banding stopped being visible.
DFLT_GIF_FPS = 12.5
DFLT_GIF_WIDTH = 460
DFLT_GIF_COLORS = 160

#: Filter fragments of the two-pass palette recipe. ``stats_mode=diff``
#: spends palette entries on moving content; bayer dithering avoids the
#: "crawling ants" of error-diffusion dither on looping video.
_PALETTEGEN = "palettegen=max_colors={colors}:stats_mode=diff"
_PALETTEUSE = "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"


def _video_frame_size(video_src: str) -> tuple[int, int]:
    """Return ``(width, height)`` of ``video_src`` via cv2 (no ffmpeg needed)."""
    import cv2

    cap = cv2.VideoCapture(str(video_src))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_src}")
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def _run_ffmpeg(args: list[str]) -> None:
    """Run one ffmpeg invocation, surfacing stderr on failure."""
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")


def make_gif(
    video_src: str,
    start: float | None = None,
    end: float | None = None,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    fps: float = DFLT_GIF_FPS,
    width: int | None = DFLT_GIF_WIDTH,
    colors: int = DFLT_GIF_COLORS,
    loop: int = 0,
    output: Output = None,
) -> Path:
    """Encode a (windowed, optionally cropped) video as a looping GIF.

    Args:
        video_src: Path to source video.
        start: Window start in seconds (None = beginning).
        end: Window end in seconds (None = end of video).
        crop_box: Optional spatial crop as ``(x, y, w, h)`` pixels from the
            top-left, validated against the source frame.
        fps: GIF frame rate.
        width: Output width in pixels; height follows the (cropped) aspect.
            ``None`` keeps the source/crop size.
        colors: Palette size (2-256).
        loop: GIF loop count — 0 means loop forever.
        output: Where to put the result — None (save beside the input), a file
            path, a directory (auto-named), or a callable sink. See
            mixing.egress.

    Returns:
        Path to the saved GIF.

    Examples:
        >>> make_gif("video.mp4", 95.8, 110.6)  # doctest: +SKIP
        >>> make_gif("video.mp4", 10, 15, crop_box=(549, 102, 309, 386))  # doctest: +SKIP
    """
    if not 2 <= colors <= 256:
        raise ValueError(f"colors must be in 2..256, got {colors}")
    if end is not None and start is not None and end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    filters = []
    if crop_box is not None:
        x, y, w, h = _validated_crop_box(crop_box, _video_frame_size(video_src))
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.append(f"fps={fps}")
    if width is not None:
        # -2: derive height from aspect, rounded to even (harmless for GIF,
        # required if the same filter chain is ever reused for mp4).
        filters.append(f"scale={width}:-2:flags=lanczos")
    vf = ",".join(filters)

    window = []
    if start is not None:
        window += ["-ss", str(start)]
    if end is not None:
        window += ["-t", str(end - (start or 0))]

    if start is not None or end is not None:
        suffix = f"{int(start or 0)}_{'end' if end is None else int(end)}"
        default_path = _auto_video_path(video_src, suffix, ext=".gif")
    else:
        default_path = Path(video_src).with_suffix(".gif")

    exe = ffmpeg_exe()

    def _write(path: Path) -> None:
        fd, palette = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            _run_ffmpeg(
                [exe, "-y", *window, "-i", str(video_src)]
                + ["-vf", f"{vf},{_PALETTEGEN.format(colors=colors)}"]
                + ["-loglevel", "error", palette]
            )
            _run_ffmpeg(
                [exe, "-y", *window, "-i", str(video_src), "-i", palette]
                + ["-lavfi", f"{vf}[x];[x][1:v]{_PALETTEUSE}"]
                + ["-loop", str(loop), "-loglevel", "error", str(path)]
            )
        finally:
            Path(palette).unlink(missing_ok=True)

    return write_egress(output, default_path=default_path, write=_write)
