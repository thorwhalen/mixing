# mixing

Tools for video and audio editing.

`mixing` is organized into focused subpackages, each with a clear dependency
footprint so you only pay for what you import:

- [`mixing.audio`](mixing.audio.md#module-mixing.audio) — sliceable [`Audio`](mixing.audio.md#mixing.audio.Audio), fades, crop,
  concat, overlay, alignment, segmentation (needs `pydub`).
- [`mixing.video`](mixing.video.md#module-mixing.video) — sliceable `Video`, crop, loop,
  audio replace/normalize, Ken Burns, thumbnails, subtitles (needs `moviepy`
  / `opencv`).
- [`mixing.transcript`](mixing.transcript.md#module-mixing.transcript) — speech-to-text (ElevenLabs Scribe, stdlib HTTP),
  filler removal, transcript formats (no heavy deps).
- [`mixing.dubbing`](mixing.dubbing.md#module-mixing.dubbing) — text-to-speech re-voicing / translation from SRT.
- [`mixing.srt`](mixing.srt.md#module-mixing.srt) — canonical SRT/timeline parsing & formatting (pure).
- [`mixing.chapters`](mixing.chapters.md#module-mixing.chapters) — transcript → chapter markers (pure; LLM optional).

**Lazy by design.** Importing `mixing` (or a light submodule such as
`mixing.chapters` / `mixing.srt`) does **not** import `moviepy` or
`opencv`. The heavy backends load only when you first touch a name that needs
them — e.g. `mixing.Video` or `from mixing.video import replace_audio`.

The top-level namespace is a curated, lazily-resolved facade over the
subpackages; see `__all__`. For the full surface of a subsystem, import
it directly (`from mixing.audio import ...`).

### Functions

| [`assemble_audio_track`](#mixing.assemble_audio_track)(segments, \*, output[, ...])   | Assemble per-segment `(audio, duration_s)` pairs into one audio track.                                                 |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [`change_speed`](#mixing.change_speed)(video_src, speed_factor, \*[, ...])    | Change the playback speed of a video.                                                                                  |
| [`concatenate_audio`](#mixing.concatenate_audio)(\*sources[, output, crossfade])   | Concatenate multiple audio files/segments.                                                                             |
| [`concatenate_videos`](#mixing.concatenate_videos)(videos, \*[, ...])               | Concatenate multiple videos with optional clip transformation and dimension normalization.                             |
| [`crop_audio`](#mixing.crop_audio)(src_path[, start, end, ...])             | Convenience function to crop and save an audio segment.                                                                |
| [`crop_video`](#mixing.crop_video)(video_src[, start, end, ...])            | Convenience function to crop and save a video segment or frame.                                                        |
| [`crossfade_overlap`](#mixing.crossfade_overlap)(transform)                        | The overlap in seconds a `transform_clips` callable needs, or `None`.                                                  |
| [`crossfade_transition`](#mixing.crossfade_transition)(clips, \*[, duration])         | Crossfade between clips to smoothly blend spatial and temporal discontinuities.                                        |
| [`detect_chapters`](#mixing.detect_chapters)(transcript, \*[, duration, ...])    | Detect chapter markers from a `transcript`.                                                                            |
| [`dub_video_from_srt`](#mixing.dub_video_from_srt)(video, srt, \*, voice_id)        | Replace `video`'s audio with TTS narration built from `srt`.                                                           |
| [`duck_audio`](#mixing.duck_audio)(bed, sidechain, \*[, duck_db, ...])      | Duck `bed` wherever `sidechain` is loud (sidechain ducking).                                                           |
| [`dump_srt`](#mixing.dump_srt)(cues)                                      | Serialize cues back to SRT text, renumbering from 1.                                                                   |
| [`extract_segments`](#mixing.extract_segments)(audio[, segments, output, ...])    | Save each segment as its own audio file.                                                                               |
| [`fade_in`](#mixing.fade_in)(src[, duration, output])                    | Apply fade-in effect to audio.                                                                                         |
| [`fade_out`](#mixing.fade_out)(src[, duration, output])                   | Apply fade-out effect to audio.                                                                                        |
| [`fade_through_black`](#mixing.fade_through_black)(clips, \*[, duration])           | Fade out to black, then fade in from black between clips.                                                              |
| [`find_audio_offset`](#mixing.find_audio_offset)(reference_audio, ...[, ...])      | Find the time offset where query_audio best aligns within reference_audio.                                             |
| [`find_segments`](#mixing.find_segments)(audio, \*[, strategy, ...])           | Find segment boundaries in `audio` using a chosen strategy.                                                            |
| [`get_video_dimensions`](#mixing.get_video_dimensions)(video)                         | Get the (width, height) dimensions of a video — a moviepy clip, or a file path (probed via cv2, so no clip is opened). |
| [`has_ffmpeg`](#mixing.has_ffmpeg)()                                        | Return True if the `ffmpeg` binary is available on `PATH`.                                                             |
| [`ken_burns_film`](#mixing.ken_burns_film)(panels, \*, output[, fps, ...])      | Stitch `(image, BurnsPath, duration_s)` panels into one film (wraps `burns`).                                          |
| [`ken_burns_video`](#mixing.ken_burns_video)(image[, path, duration, ...])       | Render one image into a pan/zoom video (thin wrapper over `burns`).                                                    |
| [`loop_audio`](#mixing.loop_audio)(source, target_duration_s, \*[, ...])    | Tile an audio source until it fills `target_duration_s`, seamlessly.                                                   |
| [`loop_video`](#mixing.loop_video)(video_src[, n_loops, output])            | Create a video by looping/repeating another video.                                                                     |
| [`make_gif`](#mixing.make_gif)(video_src[, start, end, crop_box, ...])    | Encode a (windowed, optionally cropped) video as a looping GIF.                                                        |
| [`make_thumbnail`](#mixing.make_thumbnail)(video, \*[, at_time, text, ...])     | Create a thumbnail image from a frame of `video`.                                                                      |
| [`max_overlap_for_clips`](#mixing.max_overlap_for_clips)(clips)                        | The largest overlap `clips` can be joined with, or `None` if unknowable.                                               |
| [`needs_crossfade_overlap`](#mixing.needs_crossfade_overlap)(param)                      | Declare that a `transform_clips` callable needs overlapped compositing.                                                |
| [`normalize_audio`](#mixing.normalize_audio)(video_src, \*[, output])            | Normalize audio levels in a video to reduce volume fluctuations.                                                       |
| [`normalize_video_dimensions`](#mixing.normalize_video_dimensions)(videos, \*[, ...])       | Normalize all videos to the same dimensions.                                                                           |
| [`overlap_blend`](#mixing.overlap_blend)(clips, \*[, overlap])                 | Overlap clips and crossfade the overlapping region.                                                                    |
| [`overlay_ambient_bed`](#mixing.overlay_ambient_bed)(media, ambient, \*[, ...])      | Lay a looping ambient bed under a cut, optionally ducked under dialogue.                                               |
| [`overlay_audio`](#mixing.overlay_audio)(background, overlay[, ...])           | Overlay/mix two audio sources.                                                                                         |
| [`parse_srt`](#mixing.parse_srt)(srt_text)                                 | Parse SRT text into a list of [`Cue`](#mixing.Cue) objects.                            |
| [`remove_fillers`](#mixing.remove_fillers)(input_media, output_dir, \*[, ...])  | End-to-end filler removal.                                                                                             |
| [`replace_audio`](#mixing.replace_audio)(video_src, audio_src, \*[, ...])      | Replace or mix audio in a video with new audio.                                                                        |
| [`resize_to_dimensions`](#mixing.resize_to_dimensions)(video, target_width, ...)      | Resize a video to target dimensions with different methods.                                                            |
| [`save_audio_clip`](#mixing.save_audio_clip)([audio_src, start, end, ...])       | Extract and save an audio clip.                                                                                        |
| [`save_frame`](#mixing.save_frame)([video_src, time_or_frame, ...])         | Extract and save a frame from a video file.                                                                            |
| [`seconds_to_srt_time`](#mixing.seconds_to_srt_time)(seconds)                        | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                                           |
| [`slow_motion_blend`](#mixing.slow_motion_blend)(clips, \*[, ramp_duration])       | Slow down the end of each clip and beginning of next for smoother motion transition.                                   |
| [`srt_for_media`](#mixing.srt_for_media)(media, \*[, srt_path, reuse, ...])    | Return `(srt_text, srt_path)` for `media`, transcribing if needed.                                                     |
| [`srt_time_to_seconds`](#mixing.srt_time_to_seconds)(timestamp)                      | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                                                          |
| [`text_to_speech`](#mixing.text_to_speech)(text, voice_id, \*[, api_key, ...])  | Synthesize `text` to speech with ElevenLabs and return audio bytes.                                                    |
| [`transcribe`](#mixing.transcribe)(audio, \*[, api_key, model_id, ...])     | Transcribe `audio` with ElevenLabs Scribe.                                                                             |
| [`translate_srt`](#mixing.translate_srt)(srt, target_language, \*[, ...])      | Translate the cue text of an SRT to `target_language`, keeping timings.                                                |
| [`trim_and_crossfade`](#mixing.trim_and_crossfade)(clips, \*[, duration])           | Trim first frame from subsequent clips, then crossfade.                                                                |
| [`trim_first_frame_from_subsequent_clips`](#mixing.trim_first_frame_from_subsequent_clips)(clips)       | Keep first clip intact, trim first frame from subsequent clips.                                                        |
| [`write_subtitles_in_video`](#mixing.write_subtitles_in_video)(video[, subtitles, ...])   | Write subtitles in a video, preserving audio and video quality.                                                        |

### Classes

| [`Audio`](#mixing.Audio)(src_path, \*[, time_unit, start_time, ...])   | Sliceable interface for audio supporting time-based operations.                                          |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`AudioSamples`](#mixing.AudioSamples)(audio_src[, start_sample, ...])        | Mapping interface to access audio samples by index.                                                      |
| [`Chapter`](#mixing.Chapter)(start, title)                               | A chapter marker: a start time (seconds) and a short title.                                              |
| [`Cue`](#mixing.Cue)(index, start, end, text)                        | One SRT subtitle cue.                                                                                    |
| [`FillerRemovalResult`](#mixing.FillerRemovalResult)(cleaned_media, ...[, ...])      | Paths and computed ranges produced by [`remove_fillers()`](#mixing.remove_fillers). |
| [`Segment`](#mixing.Segment)(start, end[, label, score])                 | A time-bounded slice of an audio file.                                                                   |
| [`Video`](#mixing.Video)(video_src, \*[, time_unit, start_time, ...])  | Sliceable interface for videos supporting both time ranges and frame extraction.                         |
| [`VideoFrames`](#mixing.VideoFrames)(video_src[, start_frame, end_frame])    | Mapping interface to access video frames by index.                                                       |

### Exceptions

| [`MixingError`](#mixing.MixingError)                               | Base for every error `mixing` raises on its own behalf.                |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`WindowTooWideForClip`](#mixing.WindowTooWideForClip)(\*, clip_index, ...) | An explicit analysis window leaves a clip no second, independent look. |

### *class* mixing.Audio(src_path, , time_unit='seconds', start_time=None, end_time=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sliceable interface for audio supporting time-based operations.

Provides lazy views into audio segments using slice notation. Slicing returns
new Audio instances (not copies), enabling chained operations.

* **Parameters:**
  * **src_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `AudioSegment`]) – Path to source audio file or AudioSegment
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]) – Unit for slice indices (‘seconds’, ‘samples’, ‘milliseconds’)
  * **start_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time in seconds (for creating sub-views)
  * **end_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time in seconds (for creating sub-views)

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>>
>>> # Get segment from 10s to 20s (returns Audio)
>>> segment = audio[10:20]
>>> segment.save("clip.mp3")
>>>
>>> # Use sample numbers as unit
>>> audio_samples = Audio("song.mp3", time_unit="samples")
>>> segment = audio_samples[44100:88200]  # 1 second at 44.1kHz
>>>
>>> # Get last 30 seconds
>>> ending = audio[-30:]
>>>
>>> # Chain operations
>>> trimmed = audio[5:120]  # Trim to 5s-120s
>>> faded = trimmed.fade_in(2).fade_out(3)  # Apply fades
>>> faded.save("final.mp3")
```

#### *property* channels *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Number of audio channels.

#### close()

Release the reference to the in-memory audio (no OS handles to free).

`Audio` is fully in-memory (a decoded `AudioSegment`), so there is
nothing OS-level to close. `close` simply drops the reference so the
data can be garbage-collected promptly; the object should not be used
afterwards.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of this audio/segment in seconds.

#### *property* end_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

End time in seconds (audio duration if not set).

#### fade_in(duration=1.0)

Apply fade-in effect.

* **Parameters:**
  **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with fade applied

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> faded = audio.fade_in(2.0)  # 2 second fade in
```

#### fade_out(duration=1.0)

Apply fade-out effect.

* **Parameters:**
  **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with fade applied

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> faded = audio.fade_out(3.0)  # 3 second fade out
```

#### *property* full_duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of the source audio in seconds.

#### normalize(, headroom=0.1)

Peak-normalize the audio (via `pydub.effects.normalize`).

Boosts (or attenuates) the segment so its loudest peak sits `headroom`
dB below 0 dBFS. Pure pydub — adds no new dependency.

* **Parameters:**
  **headroom** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Target peak distance below 0 dBFS, in dB (keyword-only).
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with normalization applied.

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> louder = audio.normalize()
```

#### overlay(other, position=0.0, , gain_during_overlay=0.0)

Overlay another audio on top of this one.

* **Parameters:**
  * **other** ([`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)) – Audio to overlay
  * **position** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Position in seconds where overlay starts
  * **gain_during_overlay** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Volume adjustment in dB during overlay
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with overlay applied

### Examples

```pycon
>>> bg = Audio("background.mp3")
>>> voice = Audio("voice.mp3")
>>> mixed = bg.overlay(voice, position=5.0, gain_during_overlay=-6)
```

#### resample(sample_rate)

Change the sample rate (via pydub `set_frame_rate`).

* **Parameters:**
  **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target sample rate in Hz (e.g. `16000`, `44100`).
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio at the requested sample rate. Pure pydub — adds no new
  dependency.

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> downsampled = audio.resample(16000)
```

#### *property* sample_count *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Total number of samples in this segment.

#### *property* sample_rate *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Sample rate in Hz.

#### *property* samples *: [AudioSamples](mixing.audio.audio_ops.md#mixing.audio.audio_ops.AudioSamples)*

Get sample-by-sample Mapping interface for this audio.

#### save(output=None, , format=None, bitrate='192k', \*\*export_kwargs)

Save this audio/segment to a new audio file.

* **Parameters:**
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input with
    an auto-derived name), a file path, a directory (auto-named), or
    a callable sink. See mixing.egress.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Audio format (mp3, wav, etc.). Auto-detected from extension if None.
  * **bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bitrate for compressed formats
  * **\*\*export_kwargs** – Additional arguments for pydub export
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved file

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> audio[10:30].save("clip.mp3")
>>> audio[10:30].save("clip.wav", format="wav")
```

#### *property* start_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Start time in seconds (0.0 if not set).

#### to_mono()

Downmix to a single channel (via pydub `set_channels(1)`).

* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New mono Audio. Pure pydub — adds no new dependency.

### Examples

```pycon
>>> audio = Audio("stereo.mp3")
>>> mono = audio.to_mono()
```

### *class* mixing.AudioSamples(audio_src, start_sample=0, end_sample=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

Mapping interface to access audio samples by index.

Provides dictionary-like access to audio samples with support for negative
indexing and slicing. Samples are returned as normalized float values.

* **Parameters:**
  * **audio_src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `AudioSegment`]) – Path to audio file or AudioSegment
  * **start_sample** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Starting sample index (for segments)
  * **end_sample** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ending sample index (for segments)

### Examples

```pycon
>>> audio_samples = AudioSamples("test_audio.mp3")
>>> sample = audio_samples[0]  # Get first sample
>>> last_sample = audio_samples[-1]  # Get last sample
>>> samples = list(audio_samples[1000:2000])  # Get samples 1000-1999
```

### *class* mixing.Chapter(start, title)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A chapter marker: a start time (seconds) and a short title.

### *class* mixing.Cue(index, start, end, text)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One SRT subtitle cue.

#### index

1-based cue number.

#### start

Start time in seconds.

#### end

End time in seconds.

#### text

Cue text (may contain embedded newlines).

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Cue duration in seconds (never negative).

### *class* mixing.FillerRemovalResult(cleaned_media, transcript_md, transcript_srt, cleaned_md, cleaned_srt, scribe_json, cuts_json, keeps_json, duration, cuts=<factory>, keeps=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Paths and computed ranges produced by [`remove_fillers()`](#mixing.remove_fillers).

### *exception* mixing.MixingError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base for every error `mixing` raises on its own behalf.

### *class* mixing.Segment(start, end, label=None, score=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A time-bounded slice of an audio file.

#### start

Segment start in seconds.

#### end

Segment end in seconds.

#### label

Optional tag (e.g. `"speech"`, `"music"`, `"song"`).

#### score

Optional confidence/novelty value. Higher = stronger boundary.

#### as_offset_duration()

Return `(offset, duration)` in seconds.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

#### as_start_end()

Return `(start, end)` in seconds.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Segment duration in seconds.

#### *property* offset *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Alias for `start` — read-only.

### *class* mixing.Video(video_src, , time_unit='seconds', start_time=None, end_time=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sliceable interface for videos supporting both time ranges and frame extraction.

Provides lazy views into video segments using slice notation, and direct frame
access using integer indexing. Slicing returns new Video instances
(not copies), enabling chained operations.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video file
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'frames'`, `'milliseconds'`]) – Unit for slice indices (‘seconds’, ‘frames’, ‘milliseconds’)
  * **start_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time in seconds (for creating sub-views)
  * **end_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time in seconds (for creating sub-views)

### Examples

```pycon
>>> video = Video("movie.mp4")
>>>
>>> # Get segment from 10s to 20s (returns Video)
>>> segment = video[10:20]
>>> segment.save("clip.mp4")
>>>
>>> # Extract single frame (returns numpy array)
>>> frame = video[100]  # Frame at 100 seconds
>>>
>>> # Use frame numbers as unit
>>> video_frames = Video("movie.mp4", time_unit="frames")
>>> segment = video_frames[100:500]  # Frames 100-500
>>> single_frame = video_frames[250]  # Single frame
>>>
>>> # Get last 30 seconds
>>> ending = video[-30:]
>>>
>>> # Chain operations with moviepy
>>> with video[5:15].to_clip() as clip:
...     reversed_clip = clip.fx(mp.vfx.time_mirror)
...     reversed_clip.write_videofile("reversed.mp4")
```

#### close()

Release any backend handles (moviepy clip / cv2 capture) held.

`Video` is path-backed: most operations open a moviepy clip or cv2
capture inside a `with` / `try-finally` block and release it
immediately. `close` defensively releases any handle that *was*
cached on the instance (`_clip` / `_cap`), so it is safe to call
even when nothing is open, and future-proof if a handle is ever cached.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of this video/segment in seconds.

#### *property* end_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

End time in seconds (video duration if not set).

#### *property* fps *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Frames per second of video.

#### *property* frame_count *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Total number of frames in source video.

#### *property* frames *: [VideoFrames](mixing.video.video_ops.md#mixing.video.video_ops.VideoFrames)*

Get frame-by-frame Mapping interface for this video/segment.

#### *property* full_duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of the source video in seconds.

#### save(output=None, , codec='libx264', audio_codec='aac', crop_box=None, \*\*write_kwargs)

Save this video/segment to a new video file.

* **Parameters:**
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input with
    an auto-derived name), a file path, a directory (auto-named), or
    a callable sink. See mixing.egress.
  * **codec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video codec to use
  * **audio_codec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio codec to use
  * **crop_box** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional spatial crop as `(x, y, w, h)` pixels from the
    top-left, applied in the same encode pass as the temporal
    slice. Width/height are floored to even values (libx264
    rejects odd dimensions); an out-of-bounds box raises
    `ValueError`.
  * **\*\*write_kwargs** – Additional arguments for write_videofile
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved file

#### save_frame(time_or_frame=None, output=None, , image_format='png', copy_to_clipboard=False)

Save a single frame as an image and/or copy to clipboard.

* **Parameters:**
  * **time_or_frame** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Time/frame index (None = start of segment)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Where to put the frame — None (save beside the input with an
    auto-derived name), a file path, a directory (auto-named), or a
    callable sink. `False` means “don’t save to file” (clipboard
    only). See mixing.egress.
  * **image_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image format (png, jpg, etc.)
  * **copy_to_clipboard** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, copy image to system clipboard
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Path to saved image, or None if only copied to clipboard

### Examples

```pycon
>>> video = Video("movie.mp4")
>>> video.save_frame(10.5)  # Save frame at 10.5s
>>> video.save_frame(10.5, copy_to_clipboard=True)  # Save and copy
>>> video.save_frame(10.5, output=False, copy_to_clipboard=True)  # Clipboard only
```

#### *property* start_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Start time in seconds (0.0 if not set).

#### to_clip()

Get a moviepy VideoFileClip for this segment.

#### NOTE
Caller is responsible for closing the clip.

* **Return type:**
  `VideoFileClip`

### *class* mixing.VideoFrames(video_src, start_frame=0, end_frame=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`int`](https://docs.python.org/3/builtins/functions.html#int), `ndarray`]

Mapping interface to access video frames by index.

Provides dictionary-like access to video frames with support for negative
indexing and slicing. Frames are returned as numpy arrays (BGR format).

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to the video file
  * **start_frame** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Starting frame index (for segments)
  * **end_frame** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ending frame index (for segments)

### Examples

```pycon
>>> vf = VideoFrames("test_video.mp4")
>>> frame = vf[0]  # Get first frame
>>> last_frame = vf[-1]  # Get last frame
>>> frames = list(vf[10:20])  # Get frames 10-19
```

### *exception* mixing.WindowTooWideForClip(, clip_index, clip_duration_s, window_s, hop_s, max_window_s)

Bases: [`MixingError`](mixing.errors.md#mixing.errors.MixingError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

An explicit analysis window leaves a clip no second, independent look.

Raised by [`mixing.audio.align_clips_to_reference()`](mixing.audio.md#mixing.audio.align_clips_to_reference) when the caller passes a
`window_s` wider than [`max_window_s`](#mixing.WindowTooWideForClip.max_window_s) — the length above which the clip holds
no second INDEPENDENT window, so the consensus vote has nothing to arbitrate and
`support` comes back `None`.

Why an error and not a clamp or a silent single window: the caller asked for a
windowed vote and would otherwise receive a single whole-clip correlation whose
only signal is the ABSENCE of a support number — indistinguishable from a clip
that is genuinely too short to support at any window (issue #43). Clamping would
be worse still: an explicit `window_s` is the caller saying what a window means
for their material, and silently measuring at a different one makes `support`
incomparable across a set for a reason nothing reports.

It subclasses [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) because it is an argument that cannot be honored,
so existing `except ValueError` handlers around alignment keep working.

#### clip_index

Position of the offending clip in the `clips` sequence, or
`None` when the caller did not identify one.

#### clip_duration_s

The clip’s duration in seconds.

#### window_s

The window that was asked for.

#### hop_s

The hop in force — the caller’s, or the one derived from `window_s`.
Reported because it is part of the grid that was asked for; it is NOT part
of the bound, which does not depend on it (see
`_max_supportable_window_s()`).

#### max_window_s

The largest window that still leaves this clip a second,
independent look. The bound is inclusive, so for any clip long enough to be
worth aligning a retry at exactly this window measures — which is what makes
it worth reporting. A degenerate clip of a sample or two has no such window
at all; the value is floored at one sample there and is a lower bound rather
than a promise.

### mixing.assemble_audio_track(segments, , output, sample_rate=44100)

Assemble per-segment `(audio, duration_s)` pairs into one audio track.

Each segment occupies *exactly* `duration_s` seconds of the output:
its audio clip (when given) followed by silence padding up to the
segment duration — or pure silence when `audio` is `None`. An audio
clip longer than its slot is trimmed to fit. The sections are
concatenated in order, so the result aligns slot-for-slot with a video
timeline built from the same per-segment durations — e.g. the panels of
[`ken_burns_film()`](#mixing.ken_burns_film), where each panel holds for its segment duration
and the matching narration plays over it.

This is the audio counterpart to stitching a multi-shot film: it lets a
caller mux one pre-built track rather than attaching audio per shot.

* **Parameters:**
  * **segments** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – ordered `(audio_path_or_None, duration_s)` pairs, one per
    shot. `audio_path_or_None` is a local audio file (any format
    ffmpeg reads) or `None` for a silent slot.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the assembled track — a file path, a directory
    (auto-named `audio_track.wav`), or a callable sink. See
    mixing.egress. Written as WAV (`pcm_s16le`) — universally
    muxable; a downstream mp4 encode re-encodes to aac. Required: there
    is no input file to derive a default location from.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – sample rate of the generated silence, in Hz.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  The path to the written track (or the sink’s return value), or `None`
  when no segment carries audio (the track would be wholly silent —
  nothing is written, callers can skip muxing).

### Examples

```pycon
>>> assemble_audio_track(
...     [("voice1.mp3", 5.0), (None, 3.0), ("voice2.mp3", 4.0)],
...     output="film_audio.wav",
... )
```

### mixing.change_speed(video_src, speed_factor, , output=None, \*\*save_kwargs)

Change the playback speed of a video.

Creates a new video that plays faster or slower than the original while
preserving audio pitch (audio is also sped up/slowed down proportionally).

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to input video file
  * **speed_factor** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – 

    Speed multiplier (e.g., 2.0 = 2x faster, 0.5 = half speed)
    - > 1.0: speeds up the video (e.g., 2.0 = twice as fast)
    - < 1.0: slows down the video (e.g., 0.5 = half speed)
    - 1.0: no change
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for write_videofile (e.g., codec, fps)
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the output video file

### Examples

```pycon
>>> # Create slow-motion video at half speed
>>> change_speed("action.mp4", 0.5)
>>> # Output: action_speed_0.5x.mp4
```

```pycon
>>> # Speed up video 2x
>>> change_speed("lecture.mp4", 2.0, output="fast_lecture.mp4")
```

```pycon
>>> # Extreme slow motion
>>> change_speed("jump.mp4", 0.25)
>>> # Output: jump_speed_0.25x.mp4
```

### mixing.concatenate_audio(\*sources, output=None, crossfade=0.0, \*\*save_kwargs)

Concatenate multiple audio files/segments.

* **Parameters:**
  * **\*sources** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio sources (filepaths or Audio instances)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **crossfade** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Crossfade duration in seconds between segments
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> concatenate_audio("intro.mp3", "main.mp3", "outro.mp3")
>>> concatenate_audio("a.mp3", "b.mp3", output="combined.mp3")
>>> concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)  # 500ms crossfade
```

### mixing.concatenate_videos(videos, , transform_clips=None, normalize_dimensions='social', target_width=None, target_height=None, output=None, codec='libx264', audio_codec='aac', \*\*concat_kwargs)

Concatenate multiple videos with optional clip transformation and dimension normalization.

Audio from all input videos is automatically concatenated and included in the output.
Automatically manages resource cleanup for clips created from sources.

* **Parameters:**
  * **videos** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `VideoFileClip`, [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)]]]) – 

    Can be:
    - A folder path (str or Path) containing video files (sorted order)
    - An iterable of video sources (file paths, VideoFileClip instances,
      bytes, BytesIO, or file-like objects)
  * **transform_clips** ([`None`](https://docs.python.org/3/builtins/constants.html#None) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`VideoFileClip`]], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]]) – Optional function to transform clips before concatenation.
    Receives a list of clips, returns an iterable of clips.
  * **normalize_dimensions** (`Union`[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'stretch'`, `'fit'`, `'fill'`, `'social'`]]) – 

    How to handle videos with different dimensions:
    - False: No normalization (may cause issues if dimensions differ)
    - ’stretch’: Stretch all videos to match first video’s dimensions
    - ’fit’: Scale to fit inside dimensions with padding (letterbox/pillarbox)
    - ’fill’: Scale to fill dimensions (may crop edges)
    - ’social’: Scale with blurred/zoomed background (social media style) [DEFAULT]
    - True: Same as ‘social’
  * **target_width** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit target width (overrides first video’s dimensions)
  * **target_height** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit target height (overrides first video’s dimensions)
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional path to save the concatenated video file.
    If True and videos is a folder path, generates filename from folder name.
    If True and videos is not a folder, raises ValueError.
  * **codec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video codec to use when writing file (default: ‘libx264’)
  * **audio_codec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio codec to use when writing file (default: ‘aac’)
  * **\*\*concat_kwargs** – Additional arguments passed to moviepy’s
    concatenate_videoclips. Anything given here wins over the join this
    function would otherwise pick (see below).
* **Return type:**
  *VideoFileClip*

How the clips are joined:
: A crossfade lives in the **join**, not in either clip, so the join is
  chosen from what `transform_clips` declares it needs (via
  [`needs_crossfade_overlap()`](#mixing.needs_crossfade_overlap)) rather than defaulted for everything:
  <br/>
  - A transform declaring an overlap of *d* seconds
    (`crossfade_transition`, `trim_and_crossfade`, `overlap_blend`)
    is joined with `method='compose', padding=-d`, so the clips overlap
    and their crossfade masks are actually composited. The result is
    therefore *shorter* than the sum of its clips by *d* per join, and the
    overlapping audio is crossfaded (the transforms pair every video mask
    with an `afx` gain ramp; without them the composited audio is
    *summed* at full level).
  - Everything else (no transform, or one that bakes its effect into its
    own frames, like `fade_through_black` / `slow_motion_blend`) keeps
    moviepy’s back-to-back default.
  <br/>
  An overlap larger than the clips can carry
  ([`max_overlap_for_clips()`](#mixing.max_overlap_for_clips)) does not render a longer crossfade — it
  **deletes clips**, because they land on top of one another. It is
  therefore clamped to the ceiling, with a `UserWarning` naming what was
  asked for and what was used, and the clamped value is fed back through
  the transform as well as into the padding so the ramps and the join stay
  one number. Passing `padding=` yourself takes the join over entirely:
  no ceiling, no clamp, no warning.

* **Return type:**
  `VideoFileClip`
* **Returns:**
  Concatenated VideoFileClip with audio. Caller is responsible for closing this clip.

### Examples

```pycon
>>> # From folder path with auto dimension handling (social media style)
>>> final = concatenate_videos('/path/to/videos/')
```

```pycon
>>> # From folder path with explicit output and letterboxing
>>> final = concatenate_videos(
...     '/path/to/videos/',
...     normalize_dimensions='fit',
...     output='/path/output.mp4'
... )
```

```pycon
>>> # From list of paths with transformation and specific dimensions
>>> def trim_first_frame(clips):
...     '''Keep first clip intact, trim first frame from rest.'''
...     yield clips[0]
...     for clip in clips[1:]:
...         yield clip.subclipped(1 / clip.fps)
>>> paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
>>> final = concatenate_videos(
...     paths,
...     transform_clips=trim_first_frame,
...     target_width=1920,
...     target_height=1080,
...     normalize_dimensions='social'
... )
>>> final.write_videofile('output.mp4')
>>> final.close()
```

### mixing.crop_audio(src_path, start=None, end=None, , time_unit='seconds', output=None, \*\*save_kwargs)

Convenience function to crop and save an audio segment.

* **Parameters:**
  * **src_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source audio
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time (None = beginning)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time (None = end of audio)
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]) – Unit for start/end values
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for save operation
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved cropped audio

### Examples

```pycon
>>> crop_audio("song.mp3", 10, 30)  # Crop 10s-30s
>>> crop_audio("song.mp3", 44100, 88200, time_unit="samples")
```

### mixing.crop_video(video_src, start=None, end=None, , time_unit='seconds', crop_box=None, output=None, \*\*save_kwargs)

Convenience function to crop and save a video segment or frame.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time/frame (None = beginning)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time/frame (None = end of video)
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'frames'`, `'milliseconds'`]) – Unit for start/end values
  * **crop_box** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional spatial crop as `(x, y, w, h)` pixels from the
    top-left, applied in the same encode pass as the temporal slice
    (see [`Video.save()`](#mixing.Video.save)). Not supported for single-frame
    extraction (`start == end`).
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for save operation
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved cropped video

### Examples

```pycon
>>> crop_video("video.mp4", 10, 30)  # Crop 10s-30s
>>> crop_video("video.mp4", 100, 500, time_unit="frames")
>>> crop_video("video.mp4", 10, 10)  # Single frame at 10s
>>> crop_video("video.mp4", 10, 15, crop_box=(549, 102, 309, 386))
```

### mixing.crossfade_overlap(transform)

The overlap in seconds a `transform_clips` callable needs, or `None`.

`None` means “join these clips back to back” — either the transform did
not declare an overlap, or it declared one that resolved to a non-positive
number. Anything a caller wrapped in [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial) is seen
through — keyword **and** positional binds — so the overlap tracks the
duration the caller actually chose.

A declaration that *is* present but cannot be resolved to a number
**warns**, because the alternative is the silent hard cut this whole
mechanism exists to prevent. One case stays structurally invisible: a
wrapper built without [`functools.wraps()`](https://docs.python.org/3/library/functools.html#functools.wraps) copies neither `__dict__`
nor the declaration, so nothing is left here to warn about — wrap a
transition with `functools.wraps`.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### Examples

```pycon
>>> crossfade_overlap(crossfade_transition)
0.5
>>> crossfade_overlap(trim_and_crossfade)
0.4
>>> crossfade_overlap(overlap_blend)
0.5
>>> import functools
>>> crossfade_overlap(functools.partial(crossfade_transition, duration=0.8))
0.8
```

A positional bind counts as much as a keyword one:

```pycon
>>> @needs_crossfade_overlap('fade')
... def positional_fade(clips=None, fade=0.5):
...     return clips
>>> crossfade_overlap(functools.partial(positional_fade, None, 0.9))
0.9
```

Transforms that bake their effect into their own frames need no
overlap, and neither does an undecorated callable:

```pycon
>>> crossfade_overlap(fade_through_black) is None
True
>>> crossfade_overlap(lambda clips: clips) is None
True
>>> crossfade_overlap(None) is None
True
```

A declaration nothing can supply a value for says so out loud:

```pycon
>>> @needs_crossfade_overlap('fade')
... def no_default(clips, *, fade):
...     return clips
>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter('always')
...     print(crossfade_overlap(no_default))
...     print(caught[0].message)
None
'no_default' declares 'fade' as its crossfade overlap but no value...
```

### mixing.crossfade_transition(clips, , duration=0.5)

Crossfade between clips to smoothly blend spatial and temporal discontinuities.

Best for hiding both pixel differences and motion changes.
Recommended: duration=0.3 to 0.8 seconds.

Uses CrossFadeOut on end of clips and CrossFadeIn on start of clips. Those
are masks, so they only render as a blend when the clips are composited
with a `duration`-second overlap — which is what the
[`needs_crossfade_overlap()`](#mixing.needs_crossfade_overlap) declaration tells [`concatenate_videos()`](#mixing.concatenate_videos)
to do. Each join therefore shortens the result by `duration`.

The audio is crossfaded too (`_crossfade_effects()`) — an overlapped
join *sums* the overlapping audio, so without the matching fades both
tracks play at full level through every join and the sum clips.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.detect_chapters(transcript, , duration=None, min_chapters=3, max_chapters=8, min_spacing=10.0, target_count=None, segment_fn=None, model=None)

Detect chapter markers from a `transcript`.

* **Parameters:**
  * **transcript** – One of — a Scribe response `dict` (with `"words"`), a
    Scribe `words` list, SRT text, or a list of cue dicts/objects
    exposing `start`/`end`/`text`.
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Media duration in seconds. Inferred from the transcript’s
    last timestamp when omitted; used to choose a sensible chapter
    count and to bound the final marker.
  * **min_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum chapters worth showing. If fewer survive the
    constraints, an **empty list** is returned.
  * **max_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Upper bound on chapter count.
  * **min_spacing** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum seconds between consecutive chapters (players such
    as YouTube require >= 10s).
  * **target_count** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Desired chapter count. When omitted, scales with
    `duration` (roughly one chapter per ~1.5 min, clamped to
    `[min_chapters, max_chapters]`).
  * **segment_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Pluggable segmenter `(segments, target_count) -> [{start,
    title}]`. Defaults to `default_segment_fn()` (LLM-backed).
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional LLM model override for the default segmenter.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Chapter`](mixing.chapters.md#mixing.chapters.Chapter)]
* **Returns:**
  A list of [`Chapter`](#mixing.Chapter), first at `0:00`, spaced by at least
  `min_spacing` — or `[]` when the media can’t support
  `min_chapters` (e.g. it is too short).

### mixing.dub_video_from_srt(video, srt, , voice_id, output=None, api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, fit='speed', max_speedup=1.5, keep_original_audio=0.0, work_dir=None, keep_work=False, cache=True, synth_fn=None)

Replace `video`’s audio with TTS narration built from `srt`.

* **Parameters:**
  * **video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to the source video.
  * **srt** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.md#mixing.srt.Cue)]]) – An `.srt` file path, raw SRT text, or a list of [`Cue`](#mixing.Cue).
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
    `None`, uses ElevenLabs via [`mixing.dubbing.tts.synthesize_to_file()`](mixing.dubbing.tts.md#mixing.dubbing.tts.synthesize_to_file).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the dubbed video.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `fit` is not a recognized strategy.

### mixing.duck_audio(bed, sidechain, , duck_db=-12.0, threshold_db=-40.0, attack_s=0.05, release_s=0.4, hold_s=0.2, frame_s=0.02, output=None, \*\*save_kwargs)

Duck `bed` wherever `sidechain` is loud (sidechain ducking).

**What it does.** A level-detector sidechain ducker: the `sidechain`
(typically the dialogue track) is framed at `frame_s`, each frame’s RMS
is compared to `threshold_db`, active frames are extended by `hold_s`,
and the resulting on/off signal drives a gain envelope on `bed` that
falls to `duck_db` with time constant `attack_s` and returns to unity
with `release_s`. The envelope is interpolated to sample resolution
before it is applied, so there is no zipper noise. The result always has
the **bed’s** duration; a shorter sidechain simply leaves the tail
un-ducked.

**What it does not do.** It is not a full compressor: there is no ratio,
knee, or makeup gain — the duck depth is the fixed `duck_db`, not a
function of how loud the sidechain is. There is no lookahead, so with a
short `attack_s` the first few milliseconds of a sudden word can sneak
through at full bed level. Detection is **energy-based, not speech-aware**:
any loud sidechain content (music, a door slam, hiss above
`threshold_db`) ducks the bed just as dialogue would. And the bed is
attenuated, never EQ’d — it does not carve a vocal-band notch.

* **Parameters:**
  * **bed** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – The audio to be ducked (filepath or [`Audio`](#mixing.Audio)).
  * **sidechain** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – The audio that triggers ducking — the dialogue track.
  * **duck_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Gain (dB, `<= 0`) held while the sidechain is active.
  * **threshold_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sidechain frames above this RMS dBFS count as active.
  * **attack_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time constant for reaching the ducked level.
  * **release_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time constant for returning to unity.
  * **hold_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How long the duck persists after the last active frame.
  * **frame_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Level-detector frame size (envelope time resolution).
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments.
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file.

### Examples

```pycon
>>> quiet_bed = duck_audio("bed.wav", "dialogue.wav")
>>> duck_audio("bed.wav", "vo.wav", duck_db=-18)  # deeper duck
```

### mixing.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.extract_segments(audio, segments=None, , output=None, name_template='{stem}_{idx:03d}{ext}', format='mp3', bitrate='192k', strategy='silence', \*\*strategy_kwargs)

Save each segment as its own audio file.

If `segments` is `None`, this calls [`find_segments()`](#mixing.find_segments) first using
`strategy` and `strategy_kwargs`.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **segments** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Either a list of `Segment` objects, or a list of
    `(start, end)` tuples in seconds. If `None`, segments are
    discovered automatically.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Destination directory to write the per-segment files to
    (created if missing). This multi-file producer treats `output`
    strictly as a directory — one file per segment is written into it.
    Defaults to the source file’s parent, or the current directory
    if `audio` isn’t a path.
  * **name_template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Filename template. Available fields: `{stem}`,
    `{idx}`, `{label}`, `{start}`, `{end}`, `{ext}`.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output audio format (e.g. `"mp3"`, `"wav"`).
  * **bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bitrate for compressed formats.
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]]]) – Used only when `segments` is `None`.
  * **\*\*strategy_kwargs** – Used only when `segments` is `None`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  List of paths to the saved files, in segment order.

### Examples

```pycon
>>> from mixing.audio import extract_segments
>>> # Auto-detect and save in one call
>>> paths = extract_segments(
...     "concert.wav", strategy="self_similarity",
...     output="songs/", format="mp3",
...     kernel_seconds=14.0,
... )
>>>
>>> # Or pass timestamps you already have
>>> paths = extract_segments(
...     "mix.mp3",
...     segments=[(0, 245), (247, 445), (445, 600)],
...     output="tracks/",
... )
```

### mixing.fade_in(src, duration=1.0, , output=None, \*\*save_kwargs)

Apply fade-in effect to audio.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio source (filepath or Audio instance)
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> fade_in("song.mp3", 2.0, output="faded.mp3")
>>> audio = fade_in("song.mp3", 2.0)  # Returns Audio instance
```

### mixing.fade_out(src, duration=1.0, , output=None, \*\*save_kwargs)

Apply fade-out effect to audio.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio source (filepath or Audio instance)
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> fade_out("song.mp3", 3.0, output="faded.mp3")
>>> audio = fade_out("song.mp3", 3.0)  # Returns Audio instance
```

### mixing.fade_through_black(clips, , duration=0.3)

Fade out to black, then fade in from black between clips.

More dramatic transition - clearly separates scenes.

Deliberately **not** declared with [`needs_crossfade_overlap()`](#mixing.needs_crossfade_overlap):
`FadeIn` / `FadeOut` bake the effect into each clip’s own frames, so a
back-to-back join renders it correctly and an overlapped one would eat
footage for nothing.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.find_audio_offset(reference_audio, query_audio, , sample_rate=16000)

Find the time offset where query_audio best aligns within reference_audio.

Uses FFT-based cross-correlation to find the position in reference_audio
where query_audio starts. This is useful for aligning different recordings
of the same performance — for example, aligning a studio recording (voice

+ instruments) with a camera recording (voice only).

The two audio signals don’t need to be identical; they just need to share
a correlated component (e.g., the same voice in both).

* **Parameters:**
  * **reference_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The longer audio to search within (e.g., extracted
    from a video). Accepts a file path, numpy array, or AudioSegment.
  * **query_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The shorter audio to align (e.g., a studio recording).
    Accepts a file path, numpy array, or AudioSegment.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate for analysis. Lower values are faster but
    less precise. Default 16000 Hz gives ~0.06ms precision, which is
    more than sufficient for alignment purposes.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Offset in seconds — the position in reference_audio where query_audio
  starts. Positive means query begins after the start of reference.

### Examples

```pycon
>>> from mixing.audio import find_audio_offset
>>> # Find where a studio recording aligns with a camera recording
>>> offset = find_audio_offset("camera_audio.wav", "studio.mp3")
>>> print(f"Studio recording starts at {offset:.2f}s in the camera audio")
```

### mixing.find_segments(audio, , strategy='silence', min_segment_duration=0.0, max_segment_duration=None, merge_gap=0.0, pad_start=0.0, pad_end=0.0, \*\*strategy_kwargs)

Find segment boundaries in `audio` using a chosen strategy.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]]]) – Strategy name (`"silence"`, `"energy_novelty"`,
    `"self_similarity"`, `"speech_music"`) or a callable
    `(AudioSegment, **kwargs) -> list[Segment]`.
  * **min_segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Drop or merge segments shorter than this.
  * **max_segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Split segments longer than this into equal pieces.
  * **merge_gap** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Merge consecutive same-label segments separated by less
    than this gap (seconds).
  * **pad_start** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Extend each segment backwards by this many seconds.
  * **pad_end** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Extend each segment forwards by this many seconds.
  * **\*\*strategy_kwargs** – Forwarded to the chosen strategy.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of `Segment` instances. Use `.as_start_end()` or
  `.as_offset_duration()` to get plain timestamp tuples.

### Examples

```pycon
>>> from mixing.audio import find_segments
>>> segs = find_segments("mix.mp3", strategy="silence",
...                       silence_thresh_db=-45)
>>> [s.as_offset_duration() for s in segs]
[(0.0, 245.3), (247.1, 198.4), ...]
```

### mixing.get_video_dimensions(video)

Get the (width, height) dimensions of a video — a moviepy clip, or a
file path (probed via cv2, so no clip is opened).

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

```pycon
>>> width, height = get_video_dimensions('video.mp4')
>>> clip = VideoFileClip('video.mp4')
>>> width, height = get_video_dimensions(clip)
```

### mixing.has_ffmpeg()

Return True if the `ffmpeg` binary is available on `PATH`.

Most real video work in `mixing` (concatenation, trim/pad, dimension
normalization, audio replacement) shells out to ffmpeg via moviepy.
Without it, some operations degrade silently — producing wrong-duration
or lower-quality output instead of failing. Callers should check this at
startup and refuse to run rather than emit broken media.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> isinstance(has_ffmpeg(), bool)
True
```

### mixing.ken_burns_film(panels, , output, fps=30, audio_path=None, \*\*write_kwargs)

Stitch `(image, BurnsPath, duration_s)` panels into one film (wraps `burns`).

* **Parameters:**
  * **panels** – ordered `(image, BurnsPath, duration_s)` triples.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the film — a file path, a directory (auto-named),
    or a callable sink. See [`mixing.egress`](mixing.egress.md#module-mixing.egress). Required.
  * **fps** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – frames per second.
  * **audio_path** – optional pre-built audio track to mux over the film.
  * **write_kwargs** – forwarded to the ffmpeg write.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The `Path` written (or the sink’s return value).

### mixing.ken_burns_video(image, path=BurnsPath(keyframes=((0.0, Rect(x=0.0, y=0.0, w=1.0, h=1.0)), (1.0, Rect(x=0.11538461538461542, y=0.11538461538461542, w=0.7692307692307692, h=0.7692307692307692))), easing='ease-in-out', interp='linear', output_aspect=None, version=1), , duration=2.0, fps=30, output=None, output_size=None, \*\*write_kwargs)

Render one image into a pan/zoom video (thin wrapper over `burns`).

* **Parameters:**
  * **image** – path / `PIL.Image` / `np.ndarray`.
  * **path** – the `burns.BurnsPath` motion spec (defaults to a 2s push-in).
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – clip length in seconds.
  * **fps** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – frames per second.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — `None` (burns auto-names beside the
    image), a file path, a directory (auto-named), or a callable sink.
    See [`mixing.egress`](mixing.egress.md#module-mixing.egress).
  * **output_size** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `(w, h)` to render at.
  * **write_kwargs** – forwarded to the ffmpeg write (codec, audio_codec, …).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The `Path` written (or the sink’s return value).

### mixing.loop_audio(source, target_duration_s, , crossfade_s=0.5, output=None, \*\*save_kwargs)

Tile an audio source until it fills `target_duration_s`, seamlessly.

The source is appended to itself with a `crossfade_s` crossfade at every
join (pydub’s equal-gain fade-out/fade-in), then trimmed to *exactly*
`target_duration_s`. This is the “make a 20 s ambient bed last 4 minutes”
primitive: without the crossfade, every loop point is a hard splice and a
waveform discontinuity you can hear as a click.

A source **longer** than the target is simply trimmed — looping is only
ever additive, never a no-op guard the caller has to write.

Because each join consumes `crossfade_s` of timeline, the crossfade is
clamped to half the source’s duration; otherwise a long crossfade over a
short source would never advance.

Note the tail is cut wherever `target_duration_s` lands (mid-loop is
normal), and no fade-out is applied — chain [`fade_out()`](#mixing.fade_out) if the bed
ends exposed.

* **Parameters:**
  * **source** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio to loop (filepath or [`Audio`](#mixing.Audio)).
  * **target_duration_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Duration of the result, in seconds (> 0).
  * **crossfade_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Crossfade at each loop join, in seconds. `0` gives hard
    splices. Clamped to half the source duration.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments.
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file.

### Examples

```pycon
>>> bed = loop_audio("room_tone.wav", 90.0)  # 90s bed
>>> loop_audio("waves.wav", 240.0, output="bed.wav")
```

### mixing.loop_video(video_src, n_loops=2, , output=None, \*\*save_kwargs)

Create a video by looping/repeating another video.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video
  * **n_loops** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of times to repeat the video
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for video export
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved looped video

### Examples

```pycon
>>> loop_video("intro.mp4", 3)  # Repeat 3 times
>>> loop_video("short_clip.mp4", 5, output="extended.mp4")
```

### mixing.make_gif(video_src, start=None, end=None, , crop_box=None, fps=12.5, width=460, colors=160, loop=0, output=None)

Encode a (windowed, optionally cropped) video as a looping GIF.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video.
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Window start in seconds (None = beginning; negative = from
    the end, like `crop_video`).
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Window end in seconds (None = end of video; negative = from
    the end).
  * **crop_box** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional spatial crop as `(x, y, w, h)` pixels from the
    top-left, validated against the source frame.
  * **fps** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – GIF frame rate. A window shorter than one frame at this rate
    yields a single-frame (static) GIF.
  * **width** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output width in pixels; height follows the (cropped) aspect.
    `None` keeps the source/crop size.
  * **colors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Palette size (3-256 — palettegen’s transparent-slot
    reservation makes 3 the real ffmpeg minimum).
  * **loop** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – GIF loop count written to the file — 0 means loop forever.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a
    file path, a directory (auto-named), or a callable sink. See
    mixing.egress.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the saved GIF.

### Examples

```pycon
>>> make_gif("video.mp4", 95.8, 110.6)
>>> make_gif("video.mp4", 10, 15, crop_box=(549, 102, 309, 386))
```

### mixing.make_thumbnail(video, , at_time=None, text=None, output=None, size=(1280, 720))

Create a thumbnail image from a frame of `video`.

* **Parameters:**
  * **video** (PathLike) – Source video path.
  * **at_time** (float | None) – Timestamp (seconds) of the frame to grab. Defaults to 85% of
    the video duration (typically the closing brand/logo shot).
  * **text** (str | None) – Optional short overlay text (e.g. the title). Rendered bottom-left
    over a dark gradient band for legibility.
  * **output** (Output) – Where to put the result — None (save beside the input as
    `<video-stem>.thumb.jpg`), a file path, a directory (auto-named),
    or a callable sink. See mixing.egress.
  * **size** (tuple[int, int]) – Output size (width, height). Defaults to 1280x720.
* **Return type:**
  Path
* **Returns:**
  Path to the written JPEG.

### mixing.max_overlap_for_clips(clips)

The largest overlap `clips` can be joined with, or `None` if unknowable.

Note for anyone writing a `transform_clips`: when the declared overlap
exceeds this ceiling it is clamped, and the transform is invoked a **second**
time with the clamped value. It must therefore be pure — a transform that
accumulates state, or that may only run once, will be applied twice on that
path.

A crossfade eats `overlap` seconds off *each end it touches*: the first
and last clip are touched once, every clip between them twice. So each clip
affords `duration / (number of joins it takes part in)` and the join can
only be as long as the tightest of those budgets. Past it two things go
wrong at once, both of them silent:

- a middle clip carries both a `CrossFadeIn` and a `CrossFadeOut`
  (audio ramps too) whose ramps *multiply*, so its peak opacity falls to
  `(duration / (2 * overlap)) ** 2` and it never reaches full strength;
- moviepy lays the clips out at `cumsum(durations) + padding * arange`,
  so clip *i* and clip *i+2* start at the same instant — or in the wrong
  order, once the `maximum(0, …)` clamp bites — and the later one paints
  over the middle one. The render succeeds; the footage is simply gone.

`None` when there is no join to bound (fewer than two clips) or when any
clip’s duration is unknown — a ceiling derived from a partial view would be
wrong in the dangerous direction.

### Examples

```pycon
>>> class _Clip:
...     def __init__(self, duration):
...         self.duration = duration
```

Two clips are both end clips, so each affords its whole duration:

```pycon
>>> max_overlap_for_clips([_Clip(1.0), _Clip(1.0)])
1.0
```

Add a third and the middle one pays twice:

```pycon
>>> max_overlap_for_clips([_Clip(1.0), _Clip(1.0), _Clip(1.0)])
0.5
>>> max_overlap_for_clips([_Clip(3.0), _Clip(1.0), _Clip(3.0)])
0.5
>>> max_overlap_for_clips([_Clip(1.0), _Clip(3.0), _Clip(3.0)])
1.0
```

```pycon
>>> max_overlap_for_clips([_Clip(2.0), _Clip(None)]) is None
True
>>> max_overlap_for_clips([_Clip(2.0)]) is None
True
```

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### mixing.needs_crossfade_overlap(param)

Declare that a `transform_clips` callable needs overlapped compositing.

A crossfade is a property of the **join**, not of either clip: moviepy’s
`CrossFadeIn`/`CrossFadeOut` only set a mask, and a mask does nothing
unless the clips are composited *and* overlap in time. A transform that
relies on that has to say so, or [`concatenate_videos()`](#mixing.concatenate_videos) cannot know —
and moviepy 2.x’s defaults (`method="chain"`, `padding=0`) satisfy
neither condition, so the transition silently renders a hard cut.

Declaring the *parameter name* rather than a number is what keeps the two
in step: a caller who asks for a longer fade gets a longer overlap, with no
second place to remember.

The declaration is checked **here**, at decoration time, so a typo or a
`**kwargs` signature fails at import instead of silently restoring the
hard cut at render time (there is nothing to read, so
[`crossfade_overlap()`](#mixing.crossfade_overlap) would return `None` and the join would go back
to back with nothing said).

* **Parameters:**
  **param** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the name of the decorated function’s parameter holding the
  required overlap, in seconds.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if the decorated callable has no such parameter, or names
      its `*args` / `**kwargs` catch-all.
* **Return type:**
  [*Callable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

### Examples

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

```pycon
>>> @needs_crossfade_overlap('duration')
... def my_transition(clips, *, duration=0.5):
...     "Yields clips that must overlap by ``duration`` seconds."
...     return clips
>>> crossfade_overlap(my_transition)
0.5
```

A declaration that cannot be read is refused where it is written:

```pycon
>>> @needs_crossfade_overlap('dur')          # the parameter is 'duration'
... def typo(clips, *, duration=0.5):
...     return clips
Traceback (most recent call last):
  ...
TypeError: @needs_crossfade_overlap('dur') on 'typo': no such parameter...
>>> @needs_crossfade_overlap('duration')     # nothing to read it from
... def catch_all(clips, **kwargs):
...     return clips
Traceback (most recent call last):
  ...
TypeError: ...on 'catch_all': no such parameter. ...choices: ['clips'].
```

### mixing.normalize_audio(video_src, , output=None, \*\*save_kwargs)

Normalize audio levels in a video to reduce volume fluctuations.

This function adjusts the audio so that the loudest parts reach a consistent
level, reducing the variation between quiet and loud sections. This is
particularly useful for videos with narration that varies in volume.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to input video file
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for write_videofile (e.g., codec, audio_codec)
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the output video file with normalized audio

### Examples

```pycon
>>> # Normalize audio in a video with varying narrator volume
>>> normalize_audio("lecture.mp4")
>>> # Output: lecture_normalized.mp4
```

```pycon
>>> # Specify custom output path
>>> normalize_audio("interview.mp4", output="interview_fixed.mp4")
```

### mixing.normalize_video_dimensions(videos, , reference_video=0, target_width=None, target_height=None, method='social', bg_color=(0, 0, 0))

Normalize all videos to the same dimensions.

* **Parameters:**
  * **videos** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`VideoFileClip`]) – List of video clips to normalize
  * **reference_video** (`Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), `VideoFileClip`, [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Index of video to use as reference (default: 0 = first video)
    or a VideoFileClip instance to use as reference
  * **target_width** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Explicit target width (overrides reference_video)
  * **target_height** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Explicit target height (overrides reference_video)
  * **method** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'stretch'`, `'fit'`, `'fill'`], [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'social'`]]) – Resizing method (‘stretch’, ‘fit’, ‘fill’, ‘social’)
  * **bg_color** ([`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Background color for padding
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`VideoFileClip`]
* **Returns:**
  List of normalized VideoFileClip instances

### Examples

```pycon
>>> # Normalize all to first video's dimensions
>>> normalized = normalize_video_dimensions(clips)
```

```pycon
>>> # Normalize all to specific dimensions with social media style
>>> normalized = normalize_video_dimensions(
...     clips, target_width=1920, target_height=1080, method='social'
... )
```

```pycon
>>> # Normalize to second video's dimensions
>>> normalized = normalize_video_dimensions(clips, reference_video=1)
```

### mixing.overlap_blend(clips, , overlap=0.5)

Overlap clips and crossfade the overlapping region.

The gentler sibling of [`crossfade_transition()`](#mixing.crossfade_transition): the incoming clip
dissolves in over an outgoing clip that stays fully opaque (no
`CrossFadeOut`), so the cut reads as a wipe-through rather than a dip.
Every frame of every clip is used — the `overlap` is consumed by the join
itself, exactly once.

#### NOTE
This used to *also* trim `overlap` seconds off the head of every clip
after the first. That trim was a stand-in for an overlap back when the
join was back-to-back; once the join genuinely overlaps, keeping it
charged `2 x overlap` per clip and **deleted whole clips** — three
1 s clips at the default came out as a 1 s video with the middle clip
nowhere in it. The audio fades are here for the same reason the video
ones are: an overlapped join sums the overlapping audio.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.overlay_ambient_bed(media, ambient, , mix_ratio=0.25, loop=True, crossfade_s=None, duck_under_dialogue=False, duck_db=None, output=None, \*\*save_kwargs)

Lay a looping ambient bed under a cut, optionally ducked under dialogue.

This is the one-call version of “soften the cuts with room tone”: the
`ambient` clip (usually 10–30 s) is looped with crossfades to the exact
length of `media` ([`mixing.audio.loop_audio()`](mixing.audio.md#mixing.audio.loop_audio)), optionally ducked
against the existing dialogue ([`mixing.audio.duck_audio()`](mixing.audio.md#mixing.audio.duck_audio)), and mixed
under the existing track ([`mixing.audio.overlay_audio()`](mixing.audio.md#mixing.audio.overlay_audio)).

`media` may be a **video** (recognised by extension) or an **audio**
file; the result is written in kind — a video keeps its picture and gets a
new mixed audio track, an audio file becomes the mixed track. It is
*file-first*: `output=None` writes beside the input.

`mix_ratio` is the bed’s prominence, exactly as in
[`overlay_audio()`](mixing.audio.md#mixing.audio.overlay_audio) — the bed plays at
`20·log10(mix_ratio)` dB and the existing track is attenuated by
`20·log10(1 - mix_ratio)`. The bed is attenuated by `mix_ratio` even
when the media has no audio at all (a silent base is synthesized), so the
parameter means one thing everywhere.

`duck_under_dialogue=True` runs the bed through
[`duck_audio()`](mixing.audio.md#mixing.audio.duck_audio) with the media’s own audio as the
sidechain; read that function’s docstring for what the ducker does and does
not do (energy-based detection, fixed depth, no lookahead). With no
existing audio track there is nothing to duck against, so it is a no-op.

With `loop=False` a bed shorter than the media is laid once at the start
and the rest of the timeline has no bed; a bed longer than the media is
always trimmed to fit, looping or not.

* **Parameters:**
  * **media** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Video or audio file to lay the bed under.
  * **ambient** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)) – The ambient/room-tone clip (a path or an [`Audio`](#mixing.Audio)).
  * **mix_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Bed prominence in `[0.0, 1.0]` (see above).
  * **loop** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Loop the bed to the media’s duration when it is shorter.
  * **crossfade_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Crossfade at each loop join; `None` uses
    `mixing.audio.DEFAULT_LOOP_CROSSFADE_S`.
  * **duck_under_dialogue** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Duck the bed under the media’s existing audio.
  * **duck_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Duck depth in dB; `None` uses
    `mixing.audio.DEFAULT_DUCK_DB`.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Extra encode arguments — `write_videofile` kwargs for a
    video input, `Audio.save` kwargs for an audio input.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the written file (or the sink’s return value).

### Examples

```pycon
>>> overlay_ambient_bed("cut.mp4", "room_tone.wav", mix_ratio=0.2)
>>> overlay_ambient_bed(  # duck under the dialogue
...     "cut.mp4", "rain.wav", duck_under_dialogue=True
... )
```

### mixing.overlay_audio(background, overlay, position=0.0, , mix_ratio=0.5, output=None, \*\*save_kwargs)

Overlay/mix two audio sources.

`mix_ratio` is the prominence of the *overlay*, modeled as a
linear-amplitude crossfade between background-only and overlay-only: the
overlay plays at gain `20·log10(mix_ratio)` and the background is ducked
by `20·log10(1 - mix_ratio)` for the overlap’s duration. So `0.0` =
only the background, `1.0` = only the overlay (during the overlap),
`0.5` = an equal blend (both ~-6 dB).

* **Parameters:**
  * **background** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Background audio (filepath or Audio instance)
  * **overlay** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio to overlay (filepath or Audio instance)
  * **position** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Position in seconds where overlay starts
  * **mix_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Prominence of the overlay in `[0.0, 1.0]` (see above).
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> overlay_audio("music.mp3", "voice.mp3", position=5.0)
>>> overlay_audio("bg.mp3", "sfx.mp3", mix_ratio=0.3)  # 30% overlay, 70% bg
```

### mixing.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](mixing.srt.md#mixing.srt.Cue)]

### mixing.remove_fillers(input_media, output_dir, , output_media=None, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), api_key=None, scribe_kwargs=None, apply_keeps_kwargs=None, keep_intermediate=True, scribe_data=None)

End-to-end filler removal.

* **Parameters:**
  * **input_media** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Video or audio file to clean.
  * **output_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory to write transcripts and intermediate files into.
    Created if missing.
  * **output_media** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Output cleaned media path. Defaults to
    `output_dir / f"{stem}.cleaned.mp4"`.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs API key (else env `ELEVENLABS_API_KEY`).
  * **scribe_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to [`transcribe()`](#mixing.transcribe).
  * **apply_keeps_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to `apply_keeps()`.
  * **keep_intermediate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `False`, delete the extracted audio file.
  * **scribe_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If provided, skip the network call and use this dict
    (must match Scribe’s response shape). Useful for tests / replays.
* **Return type:**
  [`FillerRemovalResult`](mixing.transcript.pipeline.md#mixing.transcript.pipeline.FillerRemovalResult)
* **Returns:**
  A [`FillerRemovalResult`](#mixing.FillerRemovalResult) with file paths and the computed
  cut/keep ranges.

### mixing.replace_audio(video_src, audio_src, , mix_ratio=1.0, output=None, match_duration=True, \*\*save_kwargs)

Replace or mix audio in a video with new audio.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video
  * **audio_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to audio file to add/mix
  * **mix_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Audio mixing ratio (0.0 = keep only original, 1.0 = replace completely,
    0.5 = mix both equally). Values between 0 and 1 blend the audio tracks.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **match_duration** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, adjust audio duration to match video
  * **\*\*save_kwargs** – Additional arguments for video export
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved video with new/mixed audio

### Examples

```pycon
>>> replace_audio("video.mp4", "music.mp3")  # Replace audio completely
>>> replace_audio("video.mp4", "bgm.mp3", mix_ratio=0.5)  # Equal mix
>>> replace_audio("video.mp4", "voice.mp3", mix_ratio=0.7)  # 70% new, 30% original
```

### mixing.resize_to_dimensions(video, target_width, target_height, , method='fit', bg_color=(0, 0, 0))

Resize a video to target dimensions with different methods.

* **Parameters:**
  * **video** (`VideoFileClip`) – Input video clip
  * **target_width** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target width in pixels
  * **target_height** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target height in pixels
  * **method** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'stretch'`, `'fit'`, `'fill'`], [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'social'`]]) – 

    Resizing method:
    - ’stretch’: Stretch to fit (may distort aspect ratio)
    - ’fit’: Scale to fit inside target (maintains aspect ratio, adds padding)
    - ’fill’: Scale to fill target (maintains aspect ratio, may crop)
    - ’social’: Like ‘fill’ but uses blurred/zoomed background (social media style)
  * **bg_color** ([`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Background color for padding (RGB tuple, 0-255)
* **Return type:**
  `VideoFileClip`
* **Returns:**
  Resized VideoFileClip

### Examples

```pycon
>>> # Stretch to exact dimensions (may distort)
>>> resized = resize_to_dimensions(clip, 1920, 1080, method='stretch')
```

```pycon
>>> # Fit inside dimensions with black padding
>>> resized = resize_to_dimensions(clip, 1920, 1080, method='fit')
```

```pycon
>>> # Fill dimensions (may crop edges)
>>> resized = resize_to_dimensions(clip, 1920, 1080, method='fill')
```

```pycon
>>> # Social media style with blurred background
>>> resized = resize_to_dimensions(clip, 1920, 1080, method='social')
```

### mixing.save_audio_clip(audio_src=None, start=0, end=None, , time_unit=None, output=None, format='mp3')

Extract and save an audio clip.

* **Parameters:**
  * **audio_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to audio file. If None, gets from clipboard.
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Start time/sample (default: 0)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time/sample (None = end of audio)
  * **time_unit** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]]) – Unit for start/end (‘seconds’, ‘samples’, ‘milliseconds’)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output format
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved audio file

### Examples

```pycon
>>> save_audio_clip("song.mp3", 10, 30)  # Save 10s-30s
>>> save_audio_clip(start=5, end=15)  # From clipboard
```

### mixing.save_frame(video_src=None, time_or_frame=0, , time_unit=None, output=None, image_format='png', copy_to_clipboard=False)

Extract and save a frame from a video file.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to the video file. If None, gets from clipboard.
  * **time_or_frame** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time/frame index of the frame to extract (default: 0)
  * **time_unit** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'frames'`, `'milliseconds'`]]) – Unit for time_or_frame (‘seconds’, ‘frames’, ‘milliseconds’).
    If None, defaults to ‘seconds’, unless time_or_frame is a negative integer,
    in which case it defaults to ‘frames’.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Where to put the frame — None (save beside the input with an
    auto-derived name), a file path, a directory (auto-named), or a
    callable sink. See mixing.egress. Two extra string shorthands are
    honored: `""` (same as None), a value starting with `"."` (used
    as the image extension), and `"/TMP"` (save to the temp dir).
    `False` means “don’t save to file” (requires copy_to_clipboard).
  * **image_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Default image format if not specified in output
  * **copy_to_clipboard** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, copy image to system clipboard
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Path to the saved image file, or None if only copied to clipboard

### Examples

```pycon
>>> save_frame("video.mp4")  # Saves frame 0
>>> save_frame("video.mp4", 10)  # Saves frame at 10s
>>> save_frame("video.mp4", -1)  # Saves last frame (frame-based)
>>> save_frame("video.mp4", 100, time_unit="frames")  # Frame 100
>>> save_frame("video.mp4", 5, output=".jpg")
>>> save_frame("video.mp4", 5, output="/TMP")
>>> save_frame(time_or_frame=3, copy_to_clipboard=True)  # From clipboard
>>> save_frame(time_or_frame=3, output=False, copy_to_clipboard=True)  # Clipboard only
```

### mixing.seconds_to_srt_time(seconds)

Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.

Milliseconds are *rounded* (not truncated), with carry handled correctly,
and negative inputs clamp to zero.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> seconds_to_srt_time(2592.187)
'00:43:12,187'
>>> seconds_to_srt_time(1.5)
'00:00:01,500'
>>> seconds_to_srt_time(-3)
'00:00:00,000'
```

### mixing.slow_motion_blend(clips, , ramp_duration=0.5)

Slow down the end of each clip and beginning of next for smoother motion transition.

Helps with motion discontinuity by creating a speed buffer zone.

#### NOTE
This changes timing, so final video will be slightly longer.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.srt_for_media(media, , srt_path=None, reuse=True, refresh=False, cache=True, max_chars=80, \*\*transcribe_kwargs)

Return `(srt_text, srt_path)` for `media`, transcribing if needed.

* **Parameters:**
  * **media** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Audio or video file (Scribe extracts audio from video).
  * **srt_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Where the SRT lives/should be written. Defaults to the media
    path with an `.srt` suffix.
  * **reuse** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` (default) and the SRT already exists, read and
    return it without re-transcribing.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Force re-transcription even if the SRT exists (overwrites it).
  * **cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass-through to [`mixing.transcript.transcribe()`](mixing.transcript.md#mixing.transcript.transcribe)’s on-disk
    response cache.
  * **max_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max characters per SRT cue when generating.
  * **transcribe_kwargs** – Extra args forwarded to `transcribe` (e.g.
    `language_code`, `diarize`).
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  The SRT text and the path it lives at.

### mixing.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

### mixing.text_to_speech(text, voice_id, , api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, timeout=600.0, cache=True, refresh=False, return_cache_status=False)

Synthesize `text` to speech with ElevenLabs and return audio bytes.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text to speak.
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs voice id (see `list_voices()`).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY`.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model id. Defaults to `eleven_multilingual_v2`.
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs `output_format` query value, e.g.
    `mp3_44100_128` (default), `mp3_44100_192` (needs a paid
    tier), or `pcm_44100`.
  * **voice_settings** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Override the default voice settings (stability,
    similarity_boost, style, use_speaker_boost). Merged over
    `DFLT_VOICE_SETTINGS`.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional ISO-639-1 hint (e.g. `"fr"`). Only some
    models honor it; ignored by others.
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** (`Union`[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `True` (default) → use `default_cache_dir()`. `False`
    → no cache. A path → use that directory. The cache key is a
    SHA-256 of the text plus every parameter that affects the audio.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` and `cache` is enabled, force a re-call and
    overwrite the cached entry.
  * **return_cache_status** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, return `(audio, was_cached)` instead
    of just `audio`. `was_cached` is `True` iff the bytes came from the
    on-disk cache (no ElevenLabs call = no spend), so a caller can attribute
    real cost. Default `False` keeps the `bytes` return.
* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]
* **Returns:**
  Raw audio bytes in `output_format` (MP3 by default) — or, when
  `return_cache_status` is `True`, a `(audio, was_cached)` tuple.
* **Raises:**
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` unset.
  * [**HTTPError**](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) – Request failed (4xx/5xx).

### mixing.transcribe(audio, , api_key=None, model_id='scribe_v1', timestamps_granularity='word', tag_audio_events=True, diarize=False, language_code=None, extra_fields=None, timeout=600.0, cache=False, refresh=False)

Transcribe `audio` with ElevenLabs Scribe.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Path to an audio/video file, or raw bytes.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY`.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Scribe model id (currently `scribe_v1`).
  * **timestamps_granularity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"word"` (default), `"character"`, or `"none"`.
  * **tag_audio_events** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to surface `(laughs)` / `(coughs)` etc.
  * **diarize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Speaker diarization on/off.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional BCP-47 hint to skip language detection.
  * **extra_fields** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Additional multipart fields to send (forward compatible).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** (`Union`[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `False` (default) → no cache. `True` → use
    `default_cache_dir()`. A path → use that directory.
    The cache key is the SHA-256 of the audio bytes plus all
    request parameters that affect the response.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` and `cache` is enabled, force a re-call
    and overwrite the cached entry. Useful for invalidating
    stale entries when Scribe upgrades its model.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  The raw JSON response from ElevenLabs (a dict). When
  `timestamps_granularity="word"` (default), the response contains
  a `"words"` list where each entry has at minimum
  `{"text", "start", "end", "type", "confidence"}`. Non-word
  events (`type != "word"`) include `(laughs)` etc. when
  `tag_audio_events=True`.
* **Raises:**
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` unset.
  * [**HTTPError**](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) – Request failed (4xx/5xx).

### mixing.translate_srt(srt, target_language, , source_language=None, translate_fn=None)

Translate the cue text of an SRT to `target_language`, keeping timings.

* **Parameters:**
  * **srt** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.md#mixing.srt.Cue)]]) – SRT text or a list of [`Cue`](mixing.srt.md#mixing.srt.Cue) objects.
  * **target_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Human-readable target language (e.g. `"French"`).
  * **source_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional source language hint.
  * **translate_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – A callable `(texts, target_language, source_language)
    -> list[str]` returning one translation per input text, in order.
    Defaults to `default_translate_fn()` (LLM-backed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Translated SRT text with the original cue timings.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – The translator returned a different number of segments than
      it was given.

### mixing.trim_and_crossfade(clips, , duration=0.4)

Trim first frame from subsequent clips, then crossfade.

Combines frame removal with smooth blending. Like
[`crossfade_transition()`](#mixing.crossfade_transition), the blend only happens because the declared
`duration` overlap reaches the join, and the audio is crossfaded to match.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.trim_first_frame_from_subsequent_clips(clips)

Keep first clip intact, trim first frame from subsequent clips.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.write_subtitles_in_video(video, subtitles=None, output=None, , embed_subtitles=True, style=None, use_ffmpeg=True, auto_detect_audio_start=False, start_time=None, \*\*subtitle_kwargs)

Write subtitles in a video, preserving audio and video quality.

Uses FFmpeg by default for 100x speedup over MoviePy’s CompositeVideoClip.

* **Parameters:**
  * **video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to input video file
  * **subtitles** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to SRT file, SRT content string, or None (auto-detect)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input,
    auto-named), a file path, a directory (auto-named), or a callable
    sink. See mixing.egress.
  * **embed_subtitles** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to embed subtitles (default True)
  * **style** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`SubtitleStyle`](mixing.video.video_subtitles.md#mixing.video.video_subtitles.SubtitleStyle)]) – SubtitleStyle configuration (optional)
  * **use_ffmpeg** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use FFmpeg directly for speed (default True, recommended)
  * **auto_detect_audio_start** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, align subtitles with detected audio start (default False)
  * **start_time** (`Union`[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

    Target start time for first subtitle, or True for auto-detect, or callable.
    - None (default): No change, keep original subtitle timestamps
    - float: Shift subtitles so first one starts at this time (in seconds)
    - True: Auto-detect audio start and align first subtitle to that time
    - callable: Use this function to find the target start time
  * **\*\*subtitle_kwargs** – Additional kwargs for subtitle styling (MoviePy only)
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the output video file

### Modules

| [`audio`](mixing.audio.md#module-mixing.audio)           | Audio mixing and editing functionality.                                            |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`chapters`](mixing.chapters.md#module-mixing.chapters)     | Detect chapter markers from a transcript — platform-agnostic.                      |
| [`dubbing`](mixing.dubbing.md#module-mixing.dubbing)       | Text-to-speech dubbing: re-voice or translate a video from its SRT.                |
| [`egress`](mixing.egress.md#module-mixing.egress)         | Canonical *egress*: route a produced result to its destination.                    |
| [`errors`](mixing.errors.md#module-mixing.errors)         | Typed errors `mixing` raises — the home for exceptions a caller may want to catch. |
| [`srt`](mixing.srt.md#module-mixing.srt)               | Canonical SRT / timeline parsing and formatting (pure, dependency-free).           |
| [`transcript`](mixing.transcript.md#module-mixing.transcript) | Transcript-driven media editing.                                                   |
| [`util`](mixing.util.md#module-mixing.util)             | General utilities for mixing video, audio, etc.                                    |
| [`video`](mixing.video.md#module-mixing.video)           | Video tools                                                                        |
