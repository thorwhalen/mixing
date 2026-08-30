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

#: palettegen reserves a transparent slot by default, so its real minimum
#: for ``max_colors`` is 3 — ffmpeg refuses 2 outright.
MIN_GIF_COLORS = 3
MAX_GIF_COLORS = 256

#: How much of ffmpeg's stderr to surface when an invocation fails.
STDERR_TAIL_CHARS = 2000

#: Filter fragments of the two-pass palette recipe. ``stats_mode=diff``
#: spends palette entries on moving content; bayer dithering avoids the
#: "crawling ants" of error-diffusion dither on looping video.
_PALETTEGEN = "palettegen=max_colors={colors}:stats_mode=diff"
_PALETTEUSE = "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"


def _probe_video(video_src: str) -> tuple[int, int, float]:
    """``(width, height, duration_s)`` of ``video_src`` via cv2 (no ffmpeg)."""
    import cv2

    cap = cv2.VideoCapture(str(video_src))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = frames / fps if fps > 0 else 0.0
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            duration,
        )
    finally:
        cap.release()


def _run_ffmpeg(args: list[str]) -> None:
    """Run one ffmpeg invocation, surfacing stderr on failure."""
    result = subprocess.run(
        args, capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-STDERR_TAIL_CHARS:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")


def _resolve_window(
    start: float | None, end: float | None, duration: float
) -> tuple[float, float | None]:
    """Normalise the (start, end) window against the source duration.

    Negative values count from the end — the module's one convention for
    times (``Video.__getitem__``, ``crop_video``). Raises ``ValueError``,
    naming the numbers, for a window outside the media or empty: ffmpeg
    would otherwise emit a zero-frame palette that only fails one pass
    later, blaming a temp .png the caller never made.
    """
    if duration <= 0:
        # cv2 could not measure this container; pass the window through
        # unvalidated rather than refusing valid media.
        if (start or 0) < 0 or (end or 0) < 0:
            raise ValueError(
                "negative (from-the-end) times need a measurable duration, "
                f"which cv2 could not read from this media (start={start}, "
                f"end={end})"
            )
        return (float(start or 0.0), None if end is None else float(end))
    resolved_start = 0.0 if start is None else float(start)
    if resolved_start < 0:
        resolved_start = duration + resolved_start
    resolved_end = None if end is None else float(end)
    if resolved_end is not None and resolved_end < 0:
        resolved_end = duration + resolved_end
    if not 0 <= resolved_start < duration:
        raise ValueError(
            f"start ({start}) resolves to {resolved_start:g}s, outside the "
            f"{duration:g}s media"
        )
    if resolved_end is not None and resolved_end <= resolved_start:
        raise ValueError(
            f"window is empty: start ({start}) resolves to "
            f"{resolved_start:g}s, end ({end}) to {resolved_end:g}s "
            f"in {duration:g}s media"
        )
    return resolved_start, resolved_end


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
        start: Window start in seconds (None = beginning; negative = from
            the end, like ``crop_video``).
        end: Window end in seconds (None = end of video; negative = from
            the end).
        crop_box: Optional spatial crop as ``(x, y, w, h)`` pixels from the
            top-left, validated against the source frame.
        fps: GIF frame rate. A window shorter than one frame at this rate
            yields a single-frame (static) GIF.
        width: Output width in pixels; height follows the (cropped) aspect.
            ``None`` keeps the source/crop size.
        colors: Palette size (3-256 — palettegen's transparent-slot
            reservation makes 3 the real ffmpeg minimum).
        loop: GIF loop count written to the file — 0 means loop forever.
        output: Where to put the result — None (save beside the input), a
            file path, a directory (auto-named), or a callable sink. See
            mixing.egress.

    Returns:
        Path to the saved GIF.

    Examples:
        >>> make_gif("video.mp4", 95.8, 110.6)  # doctest: +SKIP
        >>> make_gif("video.mp4", 10, 15, crop_box=(549, 102, 309, 386))  # doctest: +SKIP
    """
    if not MIN_GIF_COLORS <= colors <= MAX_GIF_COLORS:
        raise ValueError(
            f"colors must be in {MIN_GIF_COLORS}..{MAX_GIF_COLORS}, got {colors}"
        )
    if width is not None and width < 2:
        raise ValueError(f"width must be at least 2 pixels (or None), got {width}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    frame_w, frame_h, duration = _probe_video(video_src)
    start_s, end_s = _resolve_window(start, end, duration)

    filters = []
    if crop_box is not None:
        x, y, w, h = _validated_crop_box(crop_box, (frame_w, frame_h))
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.append(f"fps={fps}")
    if width is not None:
        # -2: derive height from aspect, rounded to even (harmless for GIF,
        # required if the same filter chain is ever reused for mp4).
        filters.append(f"scale={width}:-2:flags=lanczos")
    vf = ",".join(filters)

    window = []
    if start_s > 0:
        window += ["-ss", f"{start_s:g}"]
    if end_s is not None:
        window += ["-t", f"{end_s - start_s:g}"]

    if start is not None or end is not None:
        suffix = f"{start_s:g}_{'end' if end_s is None else f'{end_s:g}'}"
        default_path = _auto_video_path(video_src, suffix, ext=".gif")
    elif Path(video_src).suffix.lower() == ".gif":
        # a bare .with_suffix('.gif') would BE the input path
        default_path = _auto_video_path(video_src, "gif")
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
            if Path(palette).stat().st_size == 0:
                # backstop: palettegen exits 0 for a zero-frame window
                raise RuntimeError(
                    f"the palette pass produced no frames for window "
                    f"{start_s:g}-{end_s if end_s is not None else 'end'} "
                    f"of {video_src} ({duration:g}s) — is the window inside "
                    "the media?"
                )
            _run_ffmpeg(
                [exe, "-y", *window, "-i", str(video_src), "-i", palette]
                + ["-lavfi", f"{vf}[x];[x][1:v]{_PALETTEUSE}"]
                + ["-loop", str(loop), "-loglevel", "error", str(path)]
            )
        finally:
            Path(palette).unlink(missing_ok=True)

    return write_egress(output, default_path=default_path, write=_write)
