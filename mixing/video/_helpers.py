"""Internal helper functions for video operations."""

import os
from pathlib import Path

#: File extensions treated as video across ``mixing.video``. Single source of
#: truth — every "is this a video?" check reads this set.
_VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".3gp",
        ".mpg",
        ".mpeg",
    }
)


def _is_video_file(path: str | os.PathLike) -> bool:
    """True if ``path``'s extension is a known video extension (case-insensitive).

    Examples:
        >>> _is_video_file("cut.MP4")
        True
        >>> _is_video_file("bed.wav")
        False
    """
    return Path(os.fspath(path)).suffix.lower() in _VIDEO_EXTENSIONS


def _auto_video_path(src_path: str, suffix: str, *, ext: str | None = None) -> Path:
    """
    Generate output video path with suffix.

    Args:
        src_path: Source video file path
        suffix: Suffix to add to stem (e.g., 'normalized', 'loop3')
        ext: Optional extension override (e.g., '.mp4')

    Returns:
        Path with format: {stem}_{suffix}{ext}

    Examples:
        >>> str(_auto_video_path("video.mp4", "cropped"))  # doctest: +SKIP
        'video_cropped.mp4'
        >>> str(_auto_video_path("video.mp4", "normalized", ext=".mov"))  # doctest: +SKIP
        'video_normalized.mov'
    """
    src = Path(src_path)
    output = src.with_stem(f"{src.stem}_{suffix}")
    if ext:
        output = output.with_suffix(ext)
    return output


def _auto_frame_path(
    src_path: str, frame_idx: int, *, image_format: str = "png"
) -> Path:
    """
    Generate output image path for frame.

    Args:
        src_path: Source video file path
        frame_idx: Frame index number
        image_format: Image format extension (default: png)

    Returns:
        Path with format: {stem}_{frame_idx:06d}.{format}

    Examples:
        >>> str(_auto_frame_path("video.mp4", 42))  # doctest: +SKIP
        'video_000042.png'
        >>> str(_auto_frame_path("video.mp4", 100, image_format="jpg"))  # doctest: +SKIP
        'video_000100.jpg'
    """
    src = Path(src_path)
    return src.parent / f"{src.stem}_{frame_idx:06d}.{image_format}"


def _set_default_codecs(
    kwargs: dict, *, codec: str = "libx264", audio_codec: str = "aac", **extras
) -> dict:
    """
    Set default codec parameters if not specified.

    Args:
        kwargs: Keyword arguments dictionary to update
        codec: Default video codec
        audio_codec: Default audio codec
        **extras: Additional default parameters

    Returns:
        Updated kwargs dictionary (modified in-place and returned)

    Examples:
        >>> kwargs = {}
        >>> _set_default_codecs(kwargs)
        {'codec': 'libx264', 'audio_codec': 'aac'}
        >>> kwargs = {'codec': 'libx265'}
        >>> _set_default_codecs(kwargs, bitrate='5000k')
        {'codec': 'libx265', 'audio_codec': 'aac', 'bitrate': '5000k'}
    """
    kwargs.setdefault("codec", codec)
    kwargs.setdefault("audio_codec", audio_codec)
    for key, value in extras.items():
        kwargs.setdefault(key, value)
    return kwargs


def _validated_crop_box(
    crop_box, frame_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Validate an ``(x, y, w, h)`` pixel crop box against ``frame_size=(W, H)``.

    Returns the box with ``w``/``h`` floored to even values — libx264 rejects
    odd frame dimensions, and a 1-px shave is the deterministic repair (the
    box's origin is untouched). Raises ``ValueError`` for a malformed,
    non-positive, or out-of-bounds box, naming the box and the frame.

    Examples:
        >>> _validated_crop_box((10, 20, 100, 50), (320, 240))
        (10, 20, 100, 50)
        >>> _validated_crop_box((10, 20, 101, 51), (320, 240))
        (10, 20, 100, 50)
        >>> _validated_crop_box((10, 20, 400, 50), (320, 240))
        Traceback (most recent call last):
        ...
        ValueError: crop_box (10, 20, 400, 50) exceeds the 320x240 frame
    """
    try:
        x, y, w, h = (int(round(float(v))) for v in crop_box)
    except (TypeError, ValueError):
        raise ValueError(
            f"crop_box must be four numbers (x, y, w, h), got {crop_box!r}"
        )
    if w <= 0 or h <= 0:
        raise ValueError(f"crop_box width/height must be positive, got {crop_box!r}")
    frame_w, frame_h = frame_size
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise ValueError(
            f"crop_box ({x}, {y}, {w}, {h}) exceeds the {frame_w}x{frame_h} frame"
        )
    w -= w % 2
    h -= h % 2
    if w == 0 or h == 0:
        raise ValueError(
            f"crop_box {crop_box!r} is under 2 pixels wide/tall after even-rounding"
        )
    return x, y, w, h


def _ensure_output_path(path: str | Path) -> Path:
    """
    Convert to Path and ensure parent directory exists.

    Args:
        path: File path as string or Path object

    Returns:
        Path object with parent directory created

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> temp_dir = Path(tempfile.mkdtemp())
        >>> output = _ensure_output_path(temp_dir / "subdir" / "file.mp4")
        >>> output.parent.exists()
        True
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_path(
    src_path: str, requested_path: str | None, suffix: str, *, ext: str | None = None
) -> Path:
    """
    Resolve output path: use requested or auto-generate with suffix.

    Args:
        src_path: Source file path
        requested_path: User-provided output path (None = auto-generate)
        suffix: Suffix to add to stem if auto-generating
        ext: Extension override (e.g., '.mp4')

    Returns:
        Resolved output path

    Examples:
        >>> str(_resolve_output_path("video.mp4", None, "cropped"))  # doctest: +SKIP
        'video_cropped.mp4'
        >>> str(_resolve_output_path("video.mp4", "output.mp4", "cropped"))
        'output.mp4'
    """
    if requested_path is None:
        return _auto_video_path(src_path, suffix, ext=ext)
    return Path(requested_path)
