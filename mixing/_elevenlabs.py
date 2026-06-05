"""Shared ElevenLabs HTTP client helpers (internal).

The Scribe (STT) and TTS clients both target ElevenLabs over stdlib HTTP and
share the same auth convention: an ``xi-api-key`` taken from the call or the
``ELEVENLABS_API_KEY`` environment variable. Centralized here so both clients
resolve credentials identically.
"""

from __future__ import annotations

import os

#: Environment variable holding the ElevenLabs API key.
ENV_KEY = "ELEVENLABS_API_KEY"


def resolve_api_key(api_key: str | None) -> str:
    """Return ``api_key`` or fall back to ``$ELEVENLABS_API_KEY``.

    Raises:
        RuntimeError: Neither an explicit key nor the env var is set.
    """
    key = api_key or os.environ.get(ENV_KEY)
    if not key:
        raise RuntimeError(f"No ElevenLabs API key. Pass api_key= or set {ENV_KEY}.")
    return key
