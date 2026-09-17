# mixing.dubbing.dub

SRT-driven dubbing: replace a video’s audio with TTS narration.

Given a video and an SRT transcript, synthesize each cue with an ElevenLabs
voice, lay the clips onto a silent track at their cue start times, and mux the
result back over the video — producing a re-voiced (or translated) version
that stays aligned to the original timeline.

Timing fit (`fit="speed"`, the default): when a synthesized line is longer
than its cue window, it is gently time-compressed (capped at `max_speedup`)
so downstream cues keep their start times; when shorter, the slot is padded
with silence. This keeps lip/scene sync without clipping speech. Use
`fit="natural"` to never alter speed (lines may drift later in the timeline).

Quick start:

```pycon
>>> from mixing.dubbing import dub_video_from_srt
>>> dub_video_from_srt(
...     "promo.mp4", "promo.srt",
...     voice_id="<elevenlabs-voice-id>",
...     output="promo.en.mp4",
... )
```

### Module Attributes

| [`SynthFn`](#mixing.dubbing.dub.SynthFn)   | A synthesizer maps (text, out_path) -> written path.   |
|------------------------------------------------------------|--------------------------------------------------------|

### Functions

| [`dub_video_from_srt`](#mixing.dubbing.dub.dub_video_from_srt)(video, srt, \*, voice_id)   | Replace `video`'s audio with TTS narration built from `srt`.   |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------|

### mixing.dubbing.dub.SynthFn

A synthesizer maps (text, out_path) -> written path. Lets callers swap the
TTS backend (or inject a stub in tests) without touching the pipeline.

alias of `Callable`[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]

### mixing.dubbing.dub.dub_video_from_srt(video, srt, , voice_id, output=None, api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, fit='speed', max_speedup=1.5, keep_original_audio=0.0, work_dir=None, keep_work=False, cache=True, synth_fn=None)

Replace `video`’s audio with TTS narration built from `srt`.

* **Parameters:**
  * **video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to the source video.
  * **srt** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]]) – An `.srt` file path, raw SRT text, or a list of `Cue`.
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs voice id for the narration.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output video path. Defaults to
    `<video-stem>.<language_code or 'dub'>.mp4` next to the source.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs API key (falls back to `ELEVENLABS_API_KEY`).
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model id (default `eleven_multilingual_v2`).
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS audio format for the per-cue clips.
  * **voice_settings** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Override default ElevenLabs voice settings.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional ISO-639-1 hint forwarded to the TTS model and
    used in the default output filename.
  * **fit** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"speed"` (default) time-compresses overlong lines (capped at
    `max_speedup`) to hold sync; `"natural"` never changes speed.
  * **max_speedup** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Maximum tempo factor when `fit="speed"` (clamped to
    ffmpeg’s single-`atempo` ceiling of 2.0).
  * **keep_original_audio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fraction (0..1) of the original audio to keep
    mixed under the narration. `0.0` (default) replaces it entirely;
    e.g. `0.15` keeps a little background music/ambience.
  * **work_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Directory for intermediate audio. Defaults to a temp dir.
  * **keep_work** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Keep the intermediate per-cue clips and assembled track.
  * **cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Cache TTS calls on disk (skips re-synthesis of identical lines).
  * **synth_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]]) – Override the synthesizer `(text, out_path) -> path`. When
    `None`, uses ElevenLabs via [`mixing.dubbing.tts.synthesize_to_file()`](mixing.dubbing.tts.html.md#mixing.dubbing.tts.synthesize_to_file).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the dubbed video.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `fit` is not a recognized strategy.
