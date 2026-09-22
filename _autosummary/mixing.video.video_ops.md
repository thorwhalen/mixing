# mixing.video.video_ops

Video cropping via slicing interface.

Provides a lazy view into video segments without copying the underlying file.

This module provides:

- `Video`: A sliceable video interface using `video[start:end]` or `video[idx]` notation
- `crop_video()`: Convenience function for direct cropping operations
- Flexible time units: seconds, frames, or milliseconds
- Integration with moviepy and cv2 for further processing
- Frame extraction via integer indexing

### Examples

```pycon
>>> video = Video("my_video.mp4")
>>> segment = video[10:20]  # Lazy view, no copying
>>> segment.save("clip.mp4")  # Only then does it process
```

```pycon
>>> # Extract single frame
>>> frame = video[100]  # Returns numpy array (frame 100)
>>> # Or as a video segment
>>> frame_video = video[10.5:10.5]  # Single frame at 10.5s
```

```pycon
>>> # Use frame numbers instead
>>> video = Video("movie.mp4", time_unit="frames")
>>> segment = video[100:500]  # Use frame numbers
```

```pycon
>>> # Get a clip for further processing
>>> with video[5:15].to_clip() as clip:
...     reversed = clip.fx(mp.vfx.time_mirror)
...     reversed.write_videofile("output.mp4")
```

Design principles:

- Lazy evaluation: Slicing creates views, not copies
- Facade pattern: Clean interface over moviepy/cv2 complexity
- Standard library interfaces: Uses Python’s slice notation
- Dependency injection: Configurable time units and codecs
- Open-closed: Extensible via keyword arguments
- Single source of truth: One class handles both time ranges and frames

### Module Attributes

| [`DEFAULT_AMBIENT_MIX_RATIO`](#mixing.video.video_ops.DEFAULT_AMBIENT_MIX_RATIO)   | Default prominence of the ambient bed in the final mix.   |
|------------------------------------------------------------------------------|-----------------------------------------------------------|

### Functions

| [`assemble_audio_track`](#mixing.video.video_ops.assemble_audio_track)(segments, \*, output[, ...])   | Assemble per-segment `(audio, duration_s)` pairs into one audio track.        |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`change_speed`](#mixing.video.video_ops.change_speed)(video_src, speed_factor, \*[, ...])    | Change the playback speed of a video.                                         |
| [`crop_video`](#mixing.video.video_ops.crop_video)(video_src[, start, end, ...])            | Convenience function to crop and save a video segment or frame.               |
| [`ken_burns_film`](#mixing.video.video_ops.ken_burns_film)(panels, \*, output[, fps, ...])      | Stitch `(image, BurnsPath, duration_s)` panels into one film (wraps `burns`). |
| [`ken_burns_video`](#mixing.video.video_ops.ken_burns_video)(image[, path, duration, ...])       | Render one image into a pan/zoom video (thin wrapper over `burns`).           |
| [`loop_video`](#mixing.video.video_ops.loop_video)(video_src[, n_loops, output])            | Create a video by looping/repeating another video.                            |
| [`normalize_audio`](#mixing.video.video_ops.normalize_audio)(video_src, \*[, output])            | Normalize audio levels in a video to reduce volume fluctuations.              |
| [`overlay_ambient_bed`](#mixing.video.video_ops.overlay_ambient_bed)(media, ambient, \*[, ...])      | Lay a looping ambient bed under a cut, optionally ducked under dialogue.      |
| [`replace_audio`](#mixing.video.video_ops.replace_audio)(video_src, audio_src, \*[, ...])      | Replace or mix audio in a video with new audio.                               |
| [`save_frame`](#mixing.video.video_ops.save_frame)([video_src, time_or_frame, ...])         | Extract and save a frame from a video file.                                   |

### Classes

| [`Video`](#mixing.video.video_ops.Video)(video_src, \*[, time_unit, start_time, ...])   | Sliceable interface for videos supporting both time ranges and frame extraction.   |
|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`VideoFrames`](#mixing.video.video_ops.VideoFrames)(video_src[, start_frame, end_frame])     | Mapping interface to access video frames by index.                                 |

### mixing.video.video_ops.DEFAULT_AMBIENT_MIX_RATIO *= 0.25*

Default prominence of the ambient bed in the final mix. Interpreted exactly
as [`mixing.audio.overlay_audio()`](mixing.audio.md#mixing.audio.overlay_audio)’s `mix_ratio`: the bed plays at
`20·log10(mix_ratio)` dB (≈ -12 dB at 0.25) under the existing track.

### *class* mixing.video.video_ops.Video(video_src, , time_unit='seconds', start_time=None, end_time=None)

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

#### *property* frames *: [VideoFrames](#mixing.video.video_ops.VideoFrames)*

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

### *class* mixing.video.video_ops.VideoFrames(video_src, start_frame=0, end_frame=None)

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

### mixing.video.video_ops.assemble_audio_track(segments, , output, sample_rate=44100)

Assemble per-segment `(audio, duration_s)` pairs into one audio track.

Each segment occupies *exactly* `duration_s` seconds of the output:
its audio clip (when given) followed by silence padding up to the
segment duration — or pure silence when `audio` is `None`. An audio
clip longer than its slot is trimmed to fit. The sections are
concatenated in order, so the result aligns slot-for-slot with a video
timeline built from the same per-segment durations — e.g. the panels of
[`ken_burns_film()`](#mixing.video.video_ops.ken_burns_film), where each panel holds for its segment duration
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

### mixing.video.video_ops.change_speed(video_src, speed_factor, , output=None, \*\*save_kwargs)

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

### mixing.video.video_ops.crop_video(video_src, start=None, end=None, , time_unit='seconds', crop_box=None, output=None, \*\*save_kwargs)

Convenience function to crop and save a video segment or frame.

* **Parameters:**
  * **video_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source video
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time/frame (None = beginning)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time/frame (None = end of video)
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'frames'`, `'milliseconds'`]) – Unit for start/end values
  * **crop_box** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional spatial crop as `(x, y, w, h)` pixels from the
    top-left, applied in the same encode pass as the temporal slice
    (see [`Video.save()`](#mixing.video.video_ops.Video.save)). Not supported for single-frame
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

### mixing.video.video_ops.ken_burns_film(panels, , output, fps=30, audio_path=None, \*\*write_kwargs)

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

### mixing.video.video_ops.ken_burns_video(image, path=BurnsPath(keyframes=((0.0, Rect(x=0.0, y=0.0, w=1.0, h=1.0)), (1.0, Rect(x=0.11538461538461542, y=0.11538461538461542, w=0.7692307692307692, h=0.7692307692307692))), easing='ease-in-out', interp='linear', output_aspect=None, version=1), , duration=2.0, fps=30, output=None, output_size=None, \*\*write_kwargs)

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

### mixing.video.video_ops.loop_video(video_src, n_loops=2, , output=None, \*\*save_kwargs)

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

### mixing.video.video_ops.normalize_audio(video_src, , output=None, \*\*save_kwargs)

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

### mixing.video.video_ops.overlay_ambient_bed(media, ambient, , mix_ratio=0.25, loop=True, crossfade_s=None, duck_under_dialogue=False, duck_db=None, output=None, \*\*save_kwargs)

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
  * **ambient** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)) – The ambient/room-tone clip (a path or an `Audio`).
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

### mixing.video.video_ops.replace_audio(video_src, audio_src, , mix_ratio=1.0, output=None, match_duration=True, \*\*save_kwargs)

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

### mixing.video.video_ops.save_frame(video_src=None, time_or_frame=0, , time_unit=None, output=None, image_format='png', copy_to_clipboard=False)

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
