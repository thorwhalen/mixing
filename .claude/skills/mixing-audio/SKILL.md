---
name: mixing-audio
description: >
  Use for AUDIO editing with the `mixing` package: trim/crop a clip, fade in or
  fade out, join/concatenate clips (with optional crossfade), mix music under a
  voice or overlay one track on another, normalize loudness, convert to mono,
  resample, peek at raw samples, align two recordings of the same performance,
  or split a long recording (concert, DJ mix, radio show, podcast) into
  segments. Trigger on phrasings like "trim audio", "fade out the end", "join
  these audio clips", "add background music under the narration", "make it
  louder / normalize", "convert to mono", "align two recordings", "find the
  offset between these takes", "split a concert into songs", "separate speech
  from music". For video audio (replace/normalize a video's track) use
  mixing-video; this skill is audio-file → audio-file.
---

# mixing-audio — edit audio files

`mixing.audio` gives you a sliceable `Audio` view plus free functions for
fades, crop, concat, overlay, normalize, alignment, and segmentation. Time is
**seconds (float)** by default. Needs ffmpeg + the `[audio]` extra
(`pip install mixing[audio]` → pydub/soundfile).

```python
import mixing
assert mixing.has_ffmpeg(), "install ffmpeg (brew install ffmpeg / apt install ffmpeg)"
from mixing.audio import Audio  # or facade: mixing.Audio
```

## The `output` protocol (applies to every writer here)

`output=None` → object producers return the in-memory `Audio`; `Audio.save`
and the `crop_*`/`save_audio_clip` file producers save **beside the input**.
A file path writes there; a directory auto-names; a callable is a sink. All
return a `Path` (or the sink's value). Never `saveas=`/`output_path=`.

## `Audio` — sliceable, chainable, in-memory

```python
a = Audio("song.mp3")                       # time_unit defaults to "seconds"
a.duration; a.full_duration                 # this view vs the whole source (s)
a.sample_rate; a.channels; a.sample_count   # 44100, 2, ...

a[10:30]                                     # 10s–30s → new Audio (lazy view, no copy)
a[-30:]                                      # last 30s
Audio("song.mp3", time_unit="samples")[0:44100]      # 1s @ 44.1kHz (kw-only)
Audio("song.mp3", time_unit="milliseconds")[500:1500]

# chain effects, then save (each returns a new Audio)
a[5:120].fade_in(2).fade_out(3).save(output="final.mp3")   # -> Path
a[10:30].save()                              # output=None → beside input, returns Path
a1 = Audio("p1.mp3"); a2 = Audio("p2.mp3")
(a1 + a2).save(output="joined.mp3")          # __add__ concatenates

# transforms (pure pydub, no extra deps) — all return a new Audio:
a.normalize()                                # peak-normalize; normalize(headroom=0.1)
a.to_mono()                                  # downmix to 1 channel
a.resample(16000)                            # change sample rate (Hz)
a.overlay(other, position=5.0, gain_during_overlay=-6)   # mix `other` on top

# context manager (frees the in-memory buffer on exit):
with Audio("song.mp3") as a:
    dur = a.duration

# raw samples (collections.abc.Mapping[int, float], normalized to [-1, 1]):
s = a.samples
s[0]; s[-1]; s[1000:2000]                    # negative index + slicing supported
```

`save(output=None, *, format=None, bitrate="192k", **export_kwargs)` —
`format` auto-detected from the extension when `None`.

## Free functions (file-or-object in, `output`-governed out)

```python
from mixing.audio import (
    crop_audio, fade_in, fade_out, concatenate_audio,
    overlay_audio, save_audio_clip, find_audio_offset,
)

# trim — first 3 positional, rest keyword-only:
crop_audio("song.mp3", 10, 30, output="clip.mp3")          # -> Path
crop_audio("song.mp3", 44100, 88200, time_unit="samples")  # output=None → beside input

# fades accept a path OR an Audio; output=None returns the Audio object:
fade_in("intro.mp3", 2.0, output="faded.mp3")              # -> Path
faded = fade_out("song.mp3", 3.0)                          # -> Audio (not saved yet)

# join any number of clips; crossfade is in seconds:
concatenate_audio("intro.mp3", "main.mp3", "outro.mp3", output="full.mp3")
concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)         # 500ms crossfade → Audio

# music under voice: mix_ratio is the PROMINENCE OF THE OVERLAY
#   1.0 = only overlay, 0.0 = only background, 0.5 = equal blend
overlay_audio("music.mp3", "voice.mp3", position=5.0, mix_ratio=0.35, output="mix.mp3")

# extract a clip (audio_src=None pulls a path from the clipboard):
save_audio_clip("song.mp3", 10, 30, output="clip.mp3", format="mp3")

# align two recordings of the same take (cross-correlation) → offset in SECONDS:
offset = find_audio_offset("camera_audio.wav", "studio.mp3")   # float, sample_rate=16000
# positive => query (studio) starts `offset`s into the reference (camera)
```

## Segmentation — split a long recording into pieces

```python
from mixing.audio import find_segments, extract_segments, Segment

segs = find_segments("concert.wav", strategy="self_similarity")   # -> list[Segment]
for s in segs:
    s.start, s.end, s.duration, s.label, s.score   # all seconds; label/score optional
    s.as_start_end()          # (start, end)
    s.as_offset_duration()    # (start, duration)

# discover AND export in one call (output is a DIRECTORY, one file per segment):
paths = extract_segments("concert.wav", strategy="self_similarity",
                         output="songs/", format="mp3")           # -> list[Path]
# or pass timestamps you already have ((start, end) tuples or Segments):
extract_segments("mix.mp3", segments=[(0, 245), (247, 445)], output="tracks/")
```

Strategies (pick by regime; pass tuning kwargs straight through to `find_segments`):

| `strategy=` | When | Key kwargs |
|---|---|---|
| `"silence"` (default) | clean gaps (DJ mix, audiobook) | `silence_thresh_db=-40`, `min_silence_len=1.0` |
| `"energy_novelty"` | fade-outs / quiet moments, no true silence | `valley_threshold_factor=0.5`, `min_peak_distance_seconds=30` |
| `"self_similarity"` | concert/live: spectral content changes, level steady | `kernel_seconds=12.0`, `min_peak_distance_seconds=30` |
| `"speech_music"` | radio/podcast: tag spoken vs musical regions | `low_energy_threshold=0.5`, `min_segment_duration=3.0` |

`find_segments(..., strategy=, min_segment_duration=, max_segment_duration=,
merge_gap=, pad_start=, pad_end=, **strategy_kwargs)`. You can also pass your
own `(AudioSegment, **kwargs) -> list[Segment]` callable as `strategy`. The
`segment_by_silence/energy/self_similarity/speech_music` functions are exported
too if you want to call one directly on a pydub `AudioSegment`.

## Common recipes

```python
# Trim, normalize, fade, save:
Audio("raw.wav")[2:65].normalize().fade_in(1).fade_out(2).save(output="clean.mp3")

# 16kHz mono for an STT/ML pipeline:
Audio("voice.mp3").to_mono().resample(16000).save(output="voice_16k.wav")

# Duck background music under narration (music at ~30% under the voice):
overlay_audio("music.mp3", "narration.mp3", position=0.0,
              mix_ratio=0.3, output="podcast.mp3")

# Align a clean studio take to a camera recording, then crop the camera to match:
off = find_audio_offset("camera.wav", "studio.mp3")
crop_audio("camera.wav", off, off + Audio("studio.mp3").duration, output="aligned.wav")

# Split a radio show into speech vs music regions and save each:
extract_segments("show.mp3", strategy="speech_music", output="parts/")
```

## Gotchas

- `Audio(..., time_unit=...)` is **keyword-only**. Slicing uses that unit;
  `step` in a slice (`a[::2]`) is rejected.
- Slicing with a single index (`a[100]`) returns a 1-sample-long `Audio`, not a
  number. For raw values use `a.samples[100]`.
- `output=None` for `Audio.save`/`crop_audio`/`save_audio_clip` writes **next to
  the input** (auto-named) — pass an explicit `output` to control location.
- `overlay_audio`'s `mix_ratio` is the overlay's prominence (1.0 = only overlay).
  Don't confuse it with `Audio.overlay`'s `gain_during_overlay` (a dB number).
- `find_audio_offset` returns **seconds** (positive = query starts inside the
  reference); the two recordings need only a shared correlated component.
- `extract_segments(output=...)` treats `output` as a **directory** (it emits
  many files), unlike the single-file writers. Default dir is the source's
  parent (or cwd for non-path input).
- Segmentation is a tuned toolbox, not magic — expect to adjust the per-strategy
  kwargs (thresholds, distances) for your material.
```
