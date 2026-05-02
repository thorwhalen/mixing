"""ffmpeg-backed audio extraction and timeline-cut application.

These functions shell out to ``ffmpeg`` (which must be on PATH); they
have no Python dependencies beyond the stdlib.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence, Union

PathLike = Union[str, Path]


def extract_audio(
    input_path: PathLike,
    output_path: PathLike,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    bitrate: str = "96k",
    overwrite: bool = True,
) -> Path:
    """Extract a mono mp3 from ``input_path`` (suitable for STT).

    Returns the output path.
    """
    ffmpeg = _ffmpeg_required()
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(output_path)


def apply_keeps(
    input_path: PathLike,
    output_path: PathLike,
    keeps: Sequence[dict],
    *,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 20,
    preset: str = "medium",
    audio_bitrate: str = "160k",
    faststart: bool = True,
    overwrite: bool = True,
) -> Path:
    """Re-encode ``input_path`` keeping only the listed time ranges.

    Args:
        input_path: Input audio/video file.
        output_path: Output file (extension picks the container).
        keeps: ``[{"start": float, "end": float}, ...]`` time ranges in seconds.
        video_codec / audio_codec / crf / preset / audio_bitrate: Re-encode
            knobs forwarded to ffmpeg.
        faststart: Add ``-movflags +faststart`` (mp4 web-streaming).
        overwrite: Pass ``-y`` to ffmpeg.

    Returns:
        The output path.
    """
    if not keeps:
        raise ValueError("keeps must contain at least one range")
    ffmpeg = _ffmpeg_required()
    expr = "+".join(f"between(t,{k['start']:.3f},{k['end']:.3f})" for k in keeps)
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{expr}',asetpts=N/SR/TB"
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-af",
        af,
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
    ]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(output_path)


def _ffmpeg_required() -> str:
    bin_path = shutil.which("ffmpeg")
    if not bin_path:
        raise RuntimeError(
            "ffmpeg is required on PATH. Install:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Linux:  sudo apt-get install ffmpeg"
        )
    return bin_path
