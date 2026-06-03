"""Text-to-speech dubbing: re-voice or translate a video from its SRT.

The transcript side of :mod:`mixing` turns speech into SRT (ElevenLabs
Scribe); this subpackage closes the loop — turning SRT back into speech with
an ElevenLabs voice and muxing it over the video, optionally translating the
SRT first to produce a foreign-language dub.

Building blocks:
    - :func:`text_to_speech` / :func:`synthesize_to_file` — ElevenLabs TTS.
    - :func:`list_voices` / :func:`find_voice` — discover account voices.
    - :func:`parse_srt` / :func:`dump_srt` / :func:`translate_srt` — SRT I/O
      and LLM-backed translation.
    - :func:`dub_video_from_srt` — the end-to-end pipeline.

The HTTP clients use stdlib only (no ``requests`` / ``elevenlabs`` SDK), so
this adds no required deps. Needs ffmpeg on PATH and an ElevenLabs API key
(env var ``ELEVENLABS_API_KEY`` or explicit ``api_key=``). Translation needs
either the ``aix`` package or a custom ``translate_fn``.

Quick start:
    >>> from mixing.dubbing import dub_video_from_srt, translate_srt  # doctest: +SKIP
    >>> dub_video_from_srt("promo.mp4", "promo.srt", voice_id="...", output="promo.en.mp4")  # doctest: +SKIP
    >>> fr_srt = translate_srt(open("promo.srt").read(), "French")  # doctest: +SKIP
"""

from mixing.dubbing.tts import (
    ELEVENLABS_TTS_URL,
    ELEVENLABS_VOICES_URL,
    DFLT_MODEL_ID,
    DFLT_OUTPUT_FORMAT,
    DFLT_VOICE_SETTINGS,
    default_cache_dir,
    text_to_speech,
    synthesize_to_file,
    list_voices,
    find_voice,
    search_shared_voices,
    add_shared_voice,
)
from mixing.dubbing.srt import (
    Cue,
    parse_srt,
    dump_srt,
    srt_time_to_seconds,
    translate_srt,
    default_translate_fn,
)
from mixing.dubbing.dub import dub_video_from_srt

__all__ = [
    "ELEVENLABS_TTS_URL",
    "ELEVENLABS_VOICES_URL",
    "DFLT_MODEL_ID",
    "DFLT_OUTPUT_FORMAT",
    "DFLT_VOICE_SETTINGS",
    "default_cache_dir",
    "text_to_speech",
    "synthesize_to_file",
    "list_voices",
    "find_voice",
    "search_shared_voices",
    "add_shared_voice",
    "Cue",
    "parse_srt",
    "dump_srt",
    "srt_time_to_seconds",
    "translate_srt",
    "default_translate_fn",
    "dub_video_from_srt",
]
