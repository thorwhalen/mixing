# mixing.video.gif

Animated-GIF export via ffmpeg’s two-pass palette pipeline.

GIF quality lives or dies on the palette: a single-pass encode dithers against
a generic 256-color table and produces large, banded files. The two-pass
recipe here — `palettegen` (per-clip palette, `stats_mode=diff` so colors
go to what *changes*) then `paletteuse` with ordered bayer dithering —
produces the small, stable loops this module exists for. moviepy’s
`write_gif` (imageio/pillow single-pass) is deliberately not used.

The ffmpeg binary is resolved via [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe), which
prefers the pip-bundled `imageio-ffmpeg` binary — so this works on machines
with no system ffmpeg installed.

### Module Attributes

| [`DFLT_GIF_FPS`](#mixing.video.gif.DFLT_GIF_FPS)      | 12.5 fps halves a 25 fps source evenly; 460 px width keeps a portrait crop under ~2 MB for a few-second loop; 160 colors is where palette banding stopped being visible.   |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`MIN_GIF_COLORS`](#mixing.video.gif.MIN_GIF_COLORS)    | palettegen reserves a transparent slot by default, so its real minimum for `max_colors` is 3 — ffmpeg refuses 2 outright.                                                  |
| [`STDERR_TAIL_CHARS`](#mixing.video.gif.STDERR_TAIL_CHARS) | How much of ffmpeg's stderr to surface when an invocation fails.                                                                                                           |

### Functions

| [`make_gif`](#mixing.video.gif.make_gif)(video_src[, start, end, crop_box, ...])   | Encode a (windowed, optionally cropped) video as a looping GIF.   |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|

### mixing.video.gif.DFLT_GIF_FPS *= 12.5*

12.5 fps halves a 25 fps
source evenly; 460 px width keeps a portrait crop under ~2 MB for a
few-second loop; 160 colors is where palette banding stopped being visible.

* **Type:**
  Defaults proven out on real dance-loop media

### mixing.video.gif.MIN_GIF_COLORS *= 3*

palettegen reserves a transparent slot by default, so its real minimum
for `max_colors` is 3 — ffmpeg refuses 2 outright.

### mixing.video.gif.STDERR_TAIL_CHARS *= 2000*

How much of ffmpeg’s stderr to surface when an invocation fails.

### mixing.video.gif.make_gif(video_src, start=None, end=None, , crop_box=None, fps=12.5, width=460, colors=160, loop=0, output=None)

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
