# mixing.video.video_util

Video utils.

Utils that will be useful in multiple modules of the video package.

**The geometry vocabulary comes from** `looks.geometry` **; the
implementations stay here.** `SOCIAL_SIZES`, the `stretch`/`fit`/`fill`
mode names and the two constants that parameterise the `social` backdrop are
*imported*, not re-declared, so there is exactly one place that says what a
“shorts” is or how blurred a social backdrop gets. What is deliberately **not**
imported is the arithmetic: this module resizes with moviepy, and the `social`
branch is a composite (a scaled, centre-cropped, Gaussian-blurred, dimmed copy
of the input behind the fitted foreground), not a formula. `looks` is
stdlib-only, so importing the vocabulary costs nothing at import time.

[`get_video_dimensions()`](#mixing.video.video_util.get_video_dimensions) stays here on purpose — it is a *probe* (it opens
a file, or reads a live clip), not geometry, so it has no home in a pure-
arithmetic module.

### Module Attributes

| [`ResizeMethod`](#mixing.video.video_util.ResizeMethod)   | How [`resize_to_dimensions()`](#mixing.video.video_util.resize_to_dimensions) places a source frame in a target frame.   |
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`get_video_dimensions`](#mixing.video.video_util.get_video_dimensions)(video)                    | Get the (width, height) dimensions of a video — a moviepy clip, or a file path (probed via cv2, so no clip is opened).   |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`normalize_video_dimensions`](#mixing.video.video_util.normalize_video_dimensions)(videos, \*[, ...])  | Normalize all videos to the same dimensions.                                                                             |
| [`resize_to_dimensions`](#mixing.video.video_util.resize_to_dimensions)(video, target_width, ...) | Resize a video to target dimensions with different methods.                                                              |

### mixing.video.video_util.ResizeMethod

How [`resize_to_dimensions()`](#mixing.video.video_util.resize_to_dimensions) places a source frame in a target frame.
The first three names are `looks`’ `FitMode`; `social`
is **not** a fourth mode but `fit` over a blurred, dimmed copy of the source
instead of a solid colour — which is why it lives here (it is a composite)
while the other three are arithmetic.

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘stretch’, ‘fit’, ‘fill’] | [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘social’]

### mixing.video.video_util.get_video_dimensions(video)

Get the (width, height) dimensions of a video — a moviepy clip, or a
file path (probed via cv2, so no clip is opened).

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

```pycon
>>> width, height = get_video_dimensions('video.mp4')
>>> clip = VideoFileClip('video.mp4')
>>> width, height = get_video_dimensions(clip)
```

### mixing.video.video_util.normalize_video_dimensions(videos, , reference_video=0, target_width=None, target_height=None, method='social', bg_color=(0, 0, 0))

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

### mixing.video.video_util.resize_to_dimensions(video, target_width, target_height, , method='fit', bg_color=(0, 0, 0))

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
