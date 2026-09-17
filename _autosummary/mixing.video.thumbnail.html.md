# mixing.video.thumbnail

Thumbnail / cover-image generation from a video frame.

Extracts a representative frame and renders it to a 16:9 1280x720 image
(a good default for video thumbnails and podcast-cover-over-video alike),
optionally overlaying a short title with a legible gradient band. Uses ffmpeg
for frame extraction and Pillow for compositing. Platform-neutral — the size
is a sensible default, not a YouTube-specific coupling.

### Module Attributes

| [`THUMBNAIL_SIZE`](#mixing.video.thumbnail.THUMBNAIL_SIZE)              | 9 thumbnail/cover resolution (also YouTube's recommended size).                                                                                                                                    |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`YOUTUBE_THUMB_SIZE`](#mixing.video.thumbnail.YOUTUBE_THUMB_SIZE)          | Backwards-compatible alias.                                                                                                                                                                        |
| [`DEFAULT_FRAME_TIME_FRACTION`](#mixing.video.thumbnail.DEFAULT_FRAME_TIME_FRACTION) | Fraction of the video duration used for the default frame grab when `at_time` is omitted — 85% lands near the end, typically on the closing brand/logo shot, which makes a good default thumbnail. |

### Functions

| [`make_thumbnail`](#mixing.video.thumbnail.make_thumbnail)(video, \*[, at_time, text, ...])   | Create a thumbnail image from a frame of `video`.   |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------|

### mixing.video.thumbnail.DEFAULT_FRAME_TIME_FRACTION *= 0.85*

Fraction of the video duration used for the default frame grab when
`at_time` is omitted — 85% lands near the end, typically on the closing
brand/logo shot, which makes a good default thumbnail.

### mixing.video.thumbnail.THUMBNAIL_SIZE *= (1280, 720)*

9 thumbnail/cover resolution (also YouTube’s recommended size).

* **Type:**
  Default 16

### mixing.video.thumbnail.YOUTUBE_THUMB_SIZE *= (1280, 720)*

Backwards-compatible alias.

### mixing.video.thumbnail.make_thumbnail(video, , at_time=None, text=None, output=None, size=(1280, 720))

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
