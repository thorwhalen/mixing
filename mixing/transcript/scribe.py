"""ElevenLabs Scribe (speech-to-text) HTTP client.

Stdlib-only — no ``requests`` or ``elevenlabs`` SDK dependency, so this
module adds nothing to the project's required deps. Returns the raw
JSON response which includes word-level timestamps when
``timestamps_granularity="word"`` (the default).
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Union

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ENV_KEY = "ELEVENLABS_API_KEY"

PathLike = Union[str, Path]
AudioInput = Union[PathLike, bytes]


def transcribe(
    audio: AudioInput,
    *,
    api_key: str | None = None,
    model_id: str = "scribe_v1",
    timestamps_granularity: str = "word",
    tag_audio_events: bool = True,
    diarize: bool = False,
    language_code: str | None = None,
    extra_fields: Mapping[str, str] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Transcribe ``audio`` with ElevenLabs Scribe.

    Args:
        audio: Path to an audio/video file, or raw bytes.
        api_key: API key. Falls back to env var ``ELEVENLABS_API_KEY``.
        model_id: Scribe model id (currently ``scribe_v1``).
        timestamps_granularity: ``"word"`` (default), ``"character"``, or ``"none"``.
        tag_audio_events: Whether to surface ``(laughs)`` / ``(coughs)`` etc.
        diarize: Speaker diarization on/off.
        language_code: Optional BCP-47 hint to skip language detection.
        extra_fields: Additional multipart fields to send (forward compatible).
        timeout: Total request timeout in seconds.

    Returns:
        The raw JSON response from ElevenLabs (a dict).

    Raises:
        RuntimeError: No API key supplied and ``ELEVENLABS_API_KEY`` unset.
        urllib.error.HTTPError: Request failed (4xx/5xx).
    """
    api_key = api_key or os.environ.get(ENV_KEY)
    if not api_key:
        raise RuntimeError(f"No ElevenLabs API key. Pass api_key= or set {ENV_KEY}.")

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        audio_bytes = path.read_bytes()
        filename = path.name
    else:
        audio_bytes = audio
        filename = "audio.bin"

    fields: dict[str, str] = {
        "model_id": model_id,
        "timestamps_granularity": timestamps_granularity,
        "tag_audio_events": str(tag_audio_events).lower(),
        "diarize": str(diarize).lower(),
    }
    if language_code:
        fields["language_code"] = language_code
    if extra_fields:
        fields.update({k: str(v) for k, v in extra_fields.items()})

    body, content_type = _multipart_encode(fields, filename, audio_bytes)
    req = urllib.request.Request(
        ELEVENLABS_STT_URL,
        data=body,
        method="POST",
        headers={"xi-api-key": api_key, "Content-Type": content_type},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _multipart_encode(
    fields: Mapping[str, str], filename: str, file_bytes: bytes
) -> tuple[bytes, str]:
    boundary = "----mixingTranscript" + os.urandom(8).hex()
    head_parts: list[bytes] = []
    for k, v in fields.items():
        head_parts.append(f"--{boundary}\r\n".encode())
        head_parts.append(
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    file_mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    head_parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {file_mime}\r\n\r\n"
        ).encode()
    )
    closing = f"\r\n--{boundary}--\r\n".encode()
    body = b"".join(head_parts) + file_bytes + closing
    return body, f"multipart/form-data; boundary={boundary}"
