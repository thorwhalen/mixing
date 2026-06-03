"""Reuse-or-create the SRT that lives next to a media file.

Encodes the "transcribe once, persist, reuse" rule: the SRT lives alongside
the media (same folder, same basename, ``.srt``). If it already exists it is
returned untouched — re-transcribing costs API credits and the file may have
been hand-corrected. Otherwise it is generated via ElevenLabs Scribe (whose
raw response is itself cached on disk) and written next to the media.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from mixing.transcript.scribe import transcribe
from mixing.transcript.formats import words_to_srt

PathLike = Union[str, Path]


def srt_for_media(
    media: PathLike,
    *,
    srt_path: PathLike | None = None,
    reuse: bool = True,
    refresh: bool = False,
    cache: bool = True,
    max_chars: int = 80,
    **transcribe_kwargs,
) -> tuple[str, Path]:
    """Return ``(srt_text, srt_path)`` for ``media``, transcribing if needed.

    Args:
        media: Audio or video file (Scribe extracts audio from video).
        srt_path: Where the SRT lives/should be written. Defaults to the media
            path with an ``.srt`` suffix.
        reuse: When ``True`` (default) and the SRT already exists, read and
            return it without re-transcribing.
        refresh: Force re-transcription even if the SRT exists (overwrites it).
        cache: Pass-through to :func:`mixing.transcript.transcribe`'s on-disk
            response cache.
        max_chars: Max characters per SRT cue when generating.
        transcribe_kwargs: Extra args forwarded to ``transcribe`` (e.g.
            ``language_code``, ``diarize``).

    Returns:
        The SRT text and the path it lives at.
    """
    media = Path(media)
    srt_path = Path(srt_path) if srt_path is not None else media.with_suffix(".srt")

    if reuse and not refresh and srt_path.exists():
        return srt_path.read_text(encoding="utf-8"), srt_path

    resp = transcribe(media, cache=cache, **transcribe_kwargs)
    srt = words_to_srt(resp.get("words", []), max_chars=max_chars)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(srt, encoding="utf-8")
    return srt, srt_path
