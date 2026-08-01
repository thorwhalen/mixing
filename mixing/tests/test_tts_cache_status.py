"""``return_cache_status`` on :func:`mixing.dubbing.text_to_speech`.

The on-disk cache means an identical re-synthesis makes no ElevenLabs call (no
spend). ``return_cache_status=True`` surfaces that hit/miss so a caller can
attribute real cost. The default return stays ``bytes`` (backward-compatible).
"""

from unittest.mock import patch

from mixing.dubbing import text_to_speech
from mixing.dubbing import tts as tts_mod


class _FakeResp:
    def read(self):
        return b"AUDIO"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_return_cache_status_miss_then_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "K")
    calls: list = []

    def _stub(req, timeout=None):
        calls.append(req)
        return _FakeResp()

    # 1st call: cache miss → live synth → (audio, False), exactly one HTTP call
    with patch.object(tts_mod.urllib.request, "urlopen", _stub):
        audio, was_cached = text_to_speech(
            "hi", "voice-x", cache=tmp_path, return_cache_status=True
        )
    assert audio == b"AUDIO" and was_cached is False and len(calls) == 1

    # 2nd call: identical args → cache hit → (audio, True), NO new HTTP call
    with patch.object(tts_mod.urllib.request, "urlopen", _stub):
        audio2, was_cached2 = text_to_speech(
            "hi", "voice-x", cache=tmp_path, return_cache_status=True
        )
    assert audio2 == b"AUDIO" and was_cached2 is True
    assert len(calls) == 1  # still one — the hit did not call ElevenLabs


def test_default_return_is_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "K")
    with patch.object(
        tts_mod.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp()
    ):
        out = text_to_speech("hi", "voice-x", cache=tmp_path)
    assert isinstance(out, bytes) and out == b"AUDIO"  # unchanged default shape
