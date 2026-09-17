# mixing.video.video_concat

Video concatenation utilities with smooth transitions.

Provides tools for concatenating videos from various sources (file paths, clips,
bytes, BytesIO) with configurable transition effects to handle AI-generated video
discontinuities.

Key Features:

- Flexible video source handling (paths, clips, bytes, file objects)
- Multiple transition types (crossfade, fade through black, overlap blend)
- Frame continuity verification between videos
- Automatic resource management for created clips

Basic Usage:

```pycon
>>> paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
>>> stitch_with_trim_and_crossfade(paths, 'output.mp4', duration=0.4)
```

Transition Options:

> - crossfade_transition: Simple blend between clips
> - trim_and_crossfade: Remove duplicate frames then blend
> - fade_through_black: Fade out/in through black
> - overlap_blend: Dissolve the incoming clip in over a still-opaque outgoing one

The first three of those live in the **join**, not in either clip: they declare
the overlap they need with [`needs_crossfade_overlap()`](#mixing.video.video_concat.needs_crossfade_overlap), which is what tells
[`concatenate_videos()`](#mixing.video.video_concat.concatenate_videos) to composite rather than butt, and they fade picture
*and* sound (an overlapped join sums the overlapping audio otherwise). The
overlap is bounded by what the clips can carry — [`max_overlap_for_clips()`](#mixing.video.video_concat.max_overlap_for_clips) —
because past that ceiling clips paint over one another and some never appear.

Frame Verification:

```pycon
>>> match, diff = verify_frame_continuity('v1.mp4', 'v2.mp4',
...   save_comparison='comp.png')
```

### Functions

| [`concatenate_videos`](#mixing.video.video_concat.concatenate_videos)(videos, \*[, ...])              | Concatenate multiple videos with optional clip transformation and dimension normalization.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`crossfade_overlap`](#mixing.video.video_concat.crossfade_overlap)(transform)                       | The overlap in seconds a `transform_clips` callable needs, or `None`.                        |
| [`crossfade_transition`](#mixing.video.video_concat.crossfade_transition)(clips, \*[, duration])        | Crossfade between clips to smoothly blend spatial and temporal discontinuities.              |
| [`ensure_videoclip_iterable`](#mixing.video.video_concat.ensure_videoclip_iterable)(videos)                  | Normalize various video input formats to an iterable of VideoFileClip instances.             |
| [`fade_through_black`](#mixing.video.video_concat.fade_through_black)(clips, \*[, duration])          | Fade out to black, then fade in from black between clips.                                    |
| [`max_overlap_for_clips`](#mixing.video.video_concat.max_overlap_for_clips)(clips)                       | The largest overlap `clips` can be joined with, or `None` if unknowable.                     |
| [`needs_crossfade_overlap`](#mixing.video.video_concat.needs_crossfade_overlap)(param)                     | Declare that a `transform_clips` callable needs overlapped compositing.                      |
| [`overlap_blend`](#mixing.video.video_concat.overlap_blend)(clips, \*[, overlap])                | Overlap clips and crossfade the overlapping region.                                          |
| [`slow_motion_blend`](#mixing.video.video_concat.slow_motion_blend)(clips, \*[, ramp_duration])      | Slow down the end of each clip and beginning of next for smoother motion transition.         |
| [`trim_and_crossfade`](#mixing.video.video_concat.trim_and_crossfade)(clips, \*[, duration])          | Trim first frame from subsequent clips, then crossfade.                                      |
| [`trim_first_frame_from_subsequent_clips`](#mixing.video.video_concat.trim_first_frame_from_subsequent_clips)(clips)      | Keep first clip intact, trim first frame from subsequent clips.                              |
| [`verify_frame_continuity`](#mixing.video.video_concat.verify_frame_continuity)(video1, video2, \*[, ...]) | Verify that the last frame of video1 matches the first frame of video2.                      |

### mixing.video.video_concat.concatenate_videos(videos, , transform_clips=None, normalize_dimensions='social', target_width=None, target_height=None, output=None, codec='libx264', audio_codec='aac', \*\*concat_kwargs)

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
  [`needs_crossfade_overlap()`](#mixing.video.video_concat.needs_crossfade_overlap)) rather than defaulted for everything:
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
  ([`max_overlap_for_clips()`](#mixing.video.video_concat.max_overlap_for_clips)) does not render a longer crossfade — it
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

### mixing.video.video_concat.crossfade_overlap(transform)

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

### mixing.video.video_concat.crossfade_transition(clips, , duration=0.5)

Crossfade between clips to smoothly blend spatial and temporal discontinuities.

Best for hiding both pixel differences and motion changes.
Recommended: duration=0.3 to 0.8 seconds.

Uses CrossFadeOut on end of clips and CrossFadeIn on start of clips. Those
are masks, so they only render as a blend when the clips are composited
with a `duration`-second overlap — which is what the
[`needs_crossfade_overlap()`](#mixing.video.video_concat.needs_crossfade_overlap) declaration tells [`concatenate_videos()`](#mixing.video.video_concat.concatenate_videos)
to do. Each join therefore shortens the result by `duration`.

The audio is crossfaded too (`_crossfade_effects()`) — an overlapped
join *sums* the overlapping audio, so without the matching fades both
tracks play at full level through every join and the sum clips.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.video.video_concat.ensure_videoclip_iterable(videos)

Normalize various video input formats to an iterable of VideoFileClip instances.

Handles:

- Folder paths (str/Path): Iterates through video files in sorted order
- Iterables of VideoSource: Converts each item to VideoFileClip

* **Parameters:**
  **videos** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `VideoFileClip`, [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)]]]) – 

  Can be:
  - A folder path (str or Path) containing video files
  - An iterable of VideoSource items (paths, clips, bytes, etc.)
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]
* **Returns:**
  Iterable of VideoFileClip instances

### Examples

```pycon
>>> # From folder path
>>> clips = ensure_videoclip_iterable('/path/to/videos/')
```

```pycon
>>> # From list of paths
>>> clips = ensure_videoclip_iterable(['v1.mp4', 'v2.mp4'])
```

```pycon
>>> # From existing clips
>>> existing_clips = [VideoFileClip('v1.mp4')]
>>> clips = ensure_videoclip_iterable(existing_clips)
```

### mixing.video.video_concat.fade_through_black(clips, , duration=0.3)

Fade out to black, then fade in from black between clips.

More dramatic transition - clearly separates scenes.

Deliberately **not** declared with [`needs_crossfade_overlap()`](#mixing.video.video_concat.needs_crossfade_overlap):
`FadeIn` / `FadeOut` bake the effect into each clip’s own frames, so a
back-to-back join renders it correctly and an overlapped one would eat
footage for nothing.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.video.video_concat.max_overlap_for_clips(clips)

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

### mixing.video.video_concat.needs_crossfade_overlap(param)

Declare that a `transform_clips` callable needs overlapped compositing.

A crossfade is a property of the **join**, not of either clip: moviepy’s
`CrossFadeIn`/`CrossFadeOut` only set a mask, and a mask does nothing
unless the clips are composited *and* overlap in time. A transform that
relies on that has to say so, or [`concatenate_videos()`](#mixing.video.video_concat.concatenate_videos) cannot know —
and moviepy 2.x’s defaults (`method="chain"`, `padding=0`) satisfy
neither condition, so the transition silently renders a hard cut.

Declaring the *parameter name* rather than a number is what keeps the two
in step: a caller who asks for a longer fade gets a longer overlap, with no
second place to remember.

The declaration is checked **here**, at decoration time, so a typo or a
`**kwargs` signature fails at import instead of silently restoring the
hard cut at render time (there is nothing to read, so
[`crossfade_overlap()`](#mixing.video.video_concat.crossfade_overlap) would return `None` and the join would go back
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

### mixing.video.video_concat.overlap_blend(clips, , overlap=0.5)

Overlap clips and crossfade the overlapping region.

The gentler sibling of [`crossfade_transition()`](#mixing.video.video_concat.crossfade_transition): the incoming clip
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

### mixing.video.video_concat.slow_motion_blend(clips, , ramp_duration=0.5)

Slow down the end of each clip and beginning of next for smoother motion transition.

Helps with motion discontinuity by creating a speed buffer zone.

#### NOTE
This changes timing, so final video will be slightly longer.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.video.video_concat.trim_and_crossfade(clips, , duration=0.4)

Trim first frame from subsequent clips, then crossfade.

Combines frame removal with smooth blending. Like
[`crossfade_transition()`](#mixing.video.video_concat.crossfade_transition), the blend only happens because the declared
`duration` overlap reaches the join, and the audio is crossfaded to match.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.video.video_concat.trim_first_frame_from_subsequent_clips(clips)

Keep first clip intact, trim first frame from subsequent clips.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`VideoFileClip`]

### mixing.video.video_concat.verify_frame_continuity(video1, video2, , tolerance=0.0, save_comparison=None)

Verify that the last frame of video1 matches the first frame of video2.

* **Parameters:**
  * **video1** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `VideoFileClip`, [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)]) – First video source
  * **video2** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `VideoFileClip`, [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO)]) – Second video source
  * **tolerance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Maximum allowed difference (0.0-1.0) for frames to be considered equal.
    0.0 = exact match, 1.0 = completely different
  * **save_comparison** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional path to save a side-by-side comparison image
* **Returns:**
  bool, difference_score: float)
  difference_score is the mean absolute difference normalized to [0, 1]
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### Example

```pycon
>>> match, diff = verify_frame_continuity('video1.mp4', 'video2.mp4',
...                                       save_comparison='comparison.png')
>>> print(f"Frames match: {match}, difference: {diff:.4f}")
```
