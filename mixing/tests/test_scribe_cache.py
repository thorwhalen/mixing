"""Cache layer in ``mixing.transcript.scribe.transcribe``.

These tests stub the network layer (``urllib.request.urlopen``) so they
never actually hit ElevenLabs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mixing.transcript import scribe
from mixing.transcript.scribe import default_cache_dir, transcribe


_FAKE_RESPONSE = {
    "language_code": "eng",
    "language_probability": 1.0,
    "text": "hello world",
    "words": [
        {"text": "hello", "type": "word", "start": 0.0, "end": 0.5, "confidence": 0.99},
        {"text": "world", "type": "word", "start": 0.5, "end": 1.0, "confidence": 0.95},
    ],
}


class _FakeResp:
    def __init__(self, payload):
        self._buf = io.BytesIO(json.dumps(payload).encode())

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_urlopen(payload, calls: list):
    def _stub(req, timeout=None):
        calls.append(req)
        return _FakeResp(payload)

    return _stub


def test_transcribe_no_cache_calls_every_time(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 1024)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        r1 = transcribe(audio)
        r2 = transcribe(audio)
    assert r1 == _FAKE_RESPONSE
    assert r2 == _FAKE_RESPONSE
    assert len(calls) == 2


def test_transcribe_cache_skips_second_call(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 1024)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        r1 = transcribe(audio, cache=cache_dir)
        r2 = transcribe(audio, cache=cache_dir)
    assert r1 == r2 == _FAKE_RESPONSE
    assert len(calls) == 1, "second call should hit cache, not the network"


def test_transcribe_cache_distinct_keys_for_different_params(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 1024)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        transcribe(audio, cache=cache_dir, language_code="eng")
        transcribe(audio, cache=cache_dir, language_code="fra")
        transcribe(audio, cache=cache_dir, language_code="eng")  # cached
    assert len(calls) == 2


def test_transcribe_cache_distinct_for_different_audio(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    a.write_bytes(b"\x00" * 1024)
    b = tmp_path / "b.wav"
    b.write_bytes(b"\xff" * 1024)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        transcribe(a, cache=cache_dir)
        transcribe(b, cache=cache_dir)
        transcribe(a, cache=cache_dir)  # cached
    assert len(calls) == 2


def test_transcribe_refresh_overwrites_cache(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 1024)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        transcribe(audio, cache=cache_dir)
        transcribe(audio, cache=cache_dir, refresh=True)
    assert len(calls) == 2


def test_transcribe_cache_true_uses_default_dir(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 32)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("MIXING_TRANSCRIPT_CACHE_DIR", str(tmp_path / "default_cache"))
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        transcribe(audio, cache=True)
        transcribe(audio, cache=True)
    assert len(calls) == 1
    # Cache populated under the env-pointed location.
    cached_files = list((tmp_path / "default_cache").rglob("*.json"))
    assert cached_files


def test_default_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MIXING_TRANSCRIPT_CACHE_DIR", str(tmp_path / "x"))
    assert default_cache_dir() == (tmp_path / "x").resolve()


def test_transcribe_cache_does_not_require_api_key(tmp_path, monkeypatch):
    """Cached responses should be returned even with no API key set."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00" * 32)
    cache_dir = tmp_path / "cache"
    # Prime the cache.
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    calls: list = []
    with patch.object(scribe.urllib.request, "urlopen", _stub_urlopen(_FAKE_RESPONSE, calls)):
        transcribe(audio, cache=cache_dir)

    # Now wipe the key — second call should still succeed via cache.
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    r = transcribe(audio, cache=cache_dir)
    assert r == _FAKE_RESPONSE
