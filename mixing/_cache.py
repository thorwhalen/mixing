"""Generic on-disk content cache (internal).

A tiny sharded key→bytes store used by the ElevenLabs clients
(:mod:`mixing.transcript.scribe`, :mod:`mixing.dubbing.tts`) to avoid paying
for the same API call twice. Values are stored as bytes under a 2-character
shard directory; callers supply ``loads``/``dumps`` to cache structured data
(e.g. JSON) on top.

Not part of the public API — import the per-client ``default_cache_dir`` /
``transcribe(cache=...)`` / ``text_to_speech(cache=...)`` surfaces instead.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Union

CacheArg = Union[bool, str, Path]


def default_cache_dir(subdir: str, *, env_key: str) -> Path:
    """Default cache directory for ``mixing/<subdir>``.

    Honors ``$<env_key>``, then ``$XDG_CACHE_HOME``, then ``~/.cache/``; the
    final path segments are always ``mixing/<subdir>`` (except when overridden
    by ``$<env_key>``, which is used verbatim).
    """
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base).expanduser().resolve() / "mixing" / subdir
    return Path.home() / ".cache" / "mixing" / subdir


def resolve_cache_dir(
    cache: CacheArg, *, default_factory: Callable[[], Path]
) -> Path | None:
    """Turn a ``cache`` argument into a directory (created) or ``None``.

    ``False`` → ``None`` (disabled); ``True`` → ``default_factory()``; a path →
    that directory. The chosen directory is created if missing.
    """
    if cache is False:
        return None
    directory = (
        default_factory() if cache is True else Path(cache).expanduser().resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_key(*parts: Union[bytes, str, None]) -> str:
    """SHA-256 hex digest over ``parts`` (``None`` → empty, str → utf-8), \\0-delimited."""
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            part = b""
        elif isinstance(part, str):
            part = part.encode()
        h.update(part)
        h.update(b"\0")
    return h.hexdigest()


def _entry_path(cache_dir: Path, key: str, suffix: str) -> Path:
    # 2-char shard avoids one giant directory.
    return cache_dir / key[:2] / f"{key}{suffix}"


def read_cache(
    cache_dir: Path,
    key: str,
    *,
    suffix: str,
    loads: Callable[[bytes], object] | None = None,
):
    """Return the cached value for ``key`` (decoded via ``loads``), or ``None``."""
    path = _entry_path(cache_dir, key, suffix)
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if loads is None:
        return data
    try:
        return loads(data)
    except Exception:
        return None


def write_cache(
    cache_dir: Path,
    key: str,
    value,
    *,
    suffix: str,
    dumps: Callable[[object], bytes] | None = None,
) -> None:
    """Atomically write ``value`` (encoded via ``dumps``) under ``key``."""
    path = _entry_path(cache_dir, key, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if dumps is None else dumps(value)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
