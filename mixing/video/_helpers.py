"""Internal helper functions for video operations."""

from pathlib import Path


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
    kwargs.setdefault('codec', codec)
    kwargs.setdefault('audio_codec', audio_codec)
    for key, value in extras.items():
        kwargs.setdefault(key, value)
    return kwargs


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
