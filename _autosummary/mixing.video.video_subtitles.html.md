# mixing.video.video_subtitles

Video utilities for subtitle embedding and video processing.

This module provides utilities for embedding subtitles into videos with two approaches:

1. Fast FFmpeg-based approach (recommended for production)
2. MoviePy CompositeVideoClip approach (for compatibility)

### Functions

| [`auto_shift_srt_to_start`](#mixing.video.video_subtitles.auto_shift_srt_to_start)(srt_content, \*[, ...])    | Automatically shift SRT timestamps to align with a target start time.     |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`find_audio_start_offset`](#mixing.video.video_subtitles.find_audio_start_offset)(video_path, \*[, ...])     | Determine when speaking/audio actually starts in a video.                 |
| [`fix_srt_file`](#mixing.video.video_subtitles.fix_srt_file)(input_srt[, output_srt, ...])         | Fix an SRT file by shifting timestamps to align with a target start time. |
| [`generate_subtitle_clips`](#mixing.video.video_subtitles.generate_subtitle_clips)(subtitles, video_clip, \*) | Generate subtitle clips from SRT content.                                 |
| [`write_subtitles_in_video`](#mixing.video.video_subtitles.write_subtitles_in_video)(video[, subtitles, ...])  | Write subtitles in a video, preserving audio and video quality.           |

### Classes

| [`SubtitleStyle`](#mixing.video.video_subtitles.SubtitleStyle)([font_size, color, position, ...])   | Configuration for subtitle appearance.   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------|

### *class* mixing.video.video_subtitles.SubtitleStyle(font_size=24, color='white', position=('center', 'bottom'), font_name='Arial')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for subtitle appearance.

#### to_ffmpeg_style()

Convert to FFmpeg subtitle style string.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### to_text_clip_kwargs()

Convert style to TextClip kwargs.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### mixing.video.video_subtitles.auto_shift_srt_to_start(srt_content, , video_path=None, auto_detect_audio_start=False, start_time=None)

Automatically shift SRT timestamps to align with a target start time.

By default, keeps subtitles at their original timestamps. Can optionally align
to a specific start time, or auto-detect when audio actually begins.

* **Parameters:**
  * **srt_content** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – SRT file content as string
  * **video_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to video file (required if auto_detect_audio_start=True or start_time=True)
  * **auto_detect_audio_start** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, detect actual audio start time (default False)
  * **start_time** (`Union`[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

    Target start time for first subtitle, or True for auto-detect, or callable.
    - None (default): No change, keep original subtitle timestamps
    - float: Shift subtitles so first one starts at this time (in seconds)
    - True: Auto-detect audio start and align first subtitle to that time
    - callable: Use this function to find the target start time
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Modified SRT content with shifted timestamps

### Examples

```pycon
>>> srt = '''1
... 00:43:12,187 --> 00:43:13,817
... First subtitle
...
... 2
... 00:43:20,557 --> 00:43:22,087
... Second subtitle'''
>>> shifted = auto_shift_srt_to_start(srt, start_time=0.0)
>>> '00:00:00,000 --> 00:00:01,630' in shifted
True
>>> '00:00:08,370 --> 00:00:09,900' in shifted
True
```

#### NOTE
The shift amount is calculated as: shift = start_time - current_first_subtitle_time
Example: If subtitles start at 10s and start_time=3s, shift will be -7s,
moving all timestamps back by 7 seconds so first subtitle starts at 3s.

### mixing.video.video_subtitles.find_audio_start_offset(video_path, , first_n_seconds_to_sample=20.0, intensity_threshold=0.001, variation_multiplier=2.0)

Determine when speaking/audio actually starts in a video.

Analyzes audio peaks and finds the first significant audio activity,
typically indicating when speech begins.

* **Parameters:**
  * **video_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to video file
  * **first_n_seconds_to_sample** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Duration to analyze (default 20 seconds)
  * **intensity_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum intensity to consider as “audio start” (default 0.001)
  * **variation_multiplier** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Multiplier for variation-based detection
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Time offset in seconds where audio starts (0.0 if no audio detected)

#### NOTE
The default threshold of 0.001 is calibrated to detect speech start reliably
across various audio levels. Adjust if needed for very quiet or loud audio.

### mixing.video.video_subtitles.fix_srt_file(input_srt, output_srt=None, , video_path=None, auto_detect_audio_start=False, start_time=None)

Fix an SRT file by shifting timestamps to align with a target start time.

By default, keeps subtitles at their original timestamps. Can optionally align
to a specific start time, or auto-detect when audio actually begins.

* **Parameters:**
  * **input_srt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to input SRT file
  * **output_srt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to output SRT file (default: add suffix based on operation)
  * **video_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to video file (for audio-based start time detection)
  * **auto_detect_audio_start** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, align with detected audio start (default False)
  * **start_time** (`Union`[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

    Target start time for first subtitle, or True for auto-detect, or callable.
    - None (default): No change, keep original subtitle timestamps
    - float: Shift subtitles so first one starts at this time (in seconds)
    - True: Auto-detect audio start and align first subtitle to that time
    - callable: Use this function to find the target start time
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the fixed SRT file

### Examples

Basic usage (no change): fix_srt_file(‘subtitles.srt’)
Shift to start at 0: fix_srt_file(‘subtitles.srt’, start_time=0.0)
With audio detection: fix_srt_file(‘subtitles.srt’, video_path=’video.mp4’,

> auto_detect_audio_start=True)

Explicit start time: fix_srt_file(‘subtitles.srt’, start_time=3.5)

### mixing.video.video_subtitles.generate_subtitle_clips(subtitles, video_clip, , style=None, \*\*text_clip_kwargs)

Generate subtitle clips from SRT content.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`TextClip`]

### mixing.video.video_subtitles.write_subtitles_in_video(video, subtitles=None, output=None, , embed_subtitles=True, style=None, use_ffmpeg=True, auto_detect_audio_start=False, start_time=None, \*\*subtitle_kwargs)

Write subtitles in a video, preserving audio and video quality.

Uses FFmpeg by default for 100x speedup over MoviePy’s CompositeVideoClip.

* **Parameters:**
  * **video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to input video file
  * **subtitles** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to SRT file, SRT content string, or None (auto-detect)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input,
    auto-named), a file path, a directory (auto-named), or a callable
    sink. See mixing.egress.
  * **embed_subtitles** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to embed subtitles (default True)
  * **style** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`SubtitleStyle`](#mixing.video.video_subtitles.SubtitleStyle)]) – SubtitleStyle configuration (optional)
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
