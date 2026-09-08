"""
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
    >>> paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    >>> stitch_with_trim_and_crossfade(paths, 'output.mp4', duration=0.4)  # doctest: +SKIP

Transition Options:
    - crossfade_transition: Simple blend between clips
    - trim_and_crossfade: Remove duplicate frames then blend
    - fade_through_black: Fade out/in through black
    - overlap_blend: Dissolve the incoming clip in over a still-opaque outgoing one

The first three of those live in the **join**, not in either clip: they declare
the overlap they need with :func:`needs_crossfade_overlap`, which is what tells
:func:`concatenate_videos` to composite rather than butt, and they fade picture
*and* sound (an overlapped join sums the overlapping audio otherwise). The
overlap is bounded by what the clips can carry — :func:`max_overlap_for_clips` —
because past that ceiling clips paint over one another and some never appear.

Frame Verification:
    >>> match, diff = verify_frame_continuity('v1.mp4', 'v2.mp4',
    ...   save_comparison='comp.png')    # doctest: +SKIP
"""

from typing import Optional, Union, Literal
from collections.abc import Iterable, Callable
from pathlib import Path
from io import BytesIO
import functools
import inspect
import logging
import os
import warnings
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, afx, vfx

from ._helpers import _is_video_file

logger = logging.getLogger(__name__)

VideoSource = Union[str, Path, VideoFileClip, bytes, BytesIO]

#: Attribute a ``transform_clips`` callable carries to declare that its output
#: only means anything when the clips are **composited with an overlap**. Its
#: value is the *name of the parameter* that carries that overlap in seconds,
#: not the number — so a caller who changes the duration changes the join too.
#: Set it with :func:`needs_crossfade_overlap`; read it with
#: :func:`crossfade_overlap`.
_OVERLAP_PARAM_ATTR = "crossfade_overlap_param"

#: Relative slack on the :func:`max_overlap_for_clips` ceiling, so a clip whose
#: decoded duration reads 0.99999 s instead of 1 s is not clamped (and warned
#: about) for a shortfall no viewer can see.
_OVERLAP_TOLERANCE = 1e-3


def needs_crossfade_overlap(param: str) -> Callable:
    """Declare that a ``transform_clips`` callable needs overlapped compositing.

    A crossfade is a property of the **join**, not of either clip: moviepy's
    ``CrossFadeIn``/``CrossFadeOut`` only set a mask, and a mask does nothing
    unless the clips are composited *and* overlap in time. A transform that
    relies on that has to say so, or :func:`concatenate_videos` cannot know —
    and moviepy 2.x's defaults (``method="chain"``, ``padding=0``) satisfy
    neither condition, so the transition silently renders a hard cut.

    Declaring the *parameter name* rather than a number is what keeps the two
    in step: a caller who asks for a longer fade gets a longer overlap, with no
    second place to remember.

    The declaration is checked **here**, at decoration time, so a typo or a
    ``**kwargs`` signature fails at import instead of silently restoring the
    hard cut at render time (there is nothing to read, so
    :func:`crossfade_overlap` would return ``None`` and the join would go back
    to back with nothing said).

    Args:
        param: the name of the decorated function's parameter holding the
            required overlap, in seconds.

    Raises:
        TypeError: if the decorated callable has no such parameter, or names
            its ``*args`` / ``**kwargs`` catch-all.

    Examples:
        >>> @needs_crossfade_overlap('duration')
        ... def my_transition(clips, *, duration=0.5):
        ...     "Yields clips that must overlap by ``duration`` seconds."
        ...     return clips
        >>> crossfade_overlap(my_transition)
        0.5

        A declaration that cannot be read is refused where it is written:

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
    """

    def declare(func):
        _check_overlap_declaration(func, param)
        setattr(func, _OVERLAP_PARAM_ATTR, param)
        return func

    return declare


def _check_overlap_declaration(func: Callable, param: str) -> None:
    """Raise if ``func`` has no readable parameter called ``param``."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - un-introspectable callable
        return  # nothing to check against; the read side warns if it can't resolve

    spec = params.get(param)
    catch_alls = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    if spec is not None and spec.kind not in catch_alls:
        return

    nameable = [n for n, p in params.items() if p.kind not in catch_alls]
    raise TypeError(
        f"@needs_crossfade_overlap({param!r}) on "
        f"{getattr(func, '__name__', func)!r}: no such parameter. Name the "
        f"parameter holding the overlap in seconds; choices: {nameable}."
    )


def _partial_bindings(transform: Callable) -> tuple[dict, Callable]:
    """``({parameter name: bound value}, innermost func)`` for a partial chain.

    Both halves of a ``partial`` are read. Keywords are collected outermost
    first, because the outermost bind is what calling the chain applies.
    Positional arguments accumulate innermost first — ``partial(partial(f, a),
    b)`` calls ``f(a, b)`` — and are then resolved to parameter *names* against
    the innermost signature, so a positionally bound overlap is seen too.
    """
    layers: list = []
    func = transform
    while isinstance(func, functools.partial):
        layers.append(func)
        func = func.func

    bound: dict = {}
    for layer in layers:  # outermost first: the outermost bind wins
        for key, value in layer.keywords.items():
            bound.setdefault(key, value)

    positional: list = []
    for layer in reversed(layers):  # innermost first: that is the call order
        positional.extend(layer.args)
    if positional:
        try:
            partial_bind = inspect.signature(func).bind_partial(*positional)
        except (TypeError, ValueError):
            pass
        else:
            for key, value in partial_bind.arguments.items():
                bound.setdefault(key, value)  # an explicit keyword still wins

    return bound, func


def crossfade_overlap(transform: Optional[Callable]) -> Optional[float]:
    """The overlap in seconds a ``transform_clips`` callable needs, or ``None``.

    ``None`` means "join these clips back to back" — either the transform did
    not declare an overlap, or it declared one that resolved to a non-positive
    number. Anything a caller wrapped in :func:`functools.partial` is seen
    through — keyword **and** positional binds — so the overlap tracks the
    duration the caller actually chose.

    A declaration that *is* present but cannot be resolved to a number
    **warns**, because the alternative is the silent hard cut this whole
    mechanism exists to prevent. One case stays structurally invisible: a
    wrapper built without :func:`functools.wraps` copies neither ``__dict__``
    nor the declaration, so nothing is left here to warn about — wrap a
    transition with ``functools.wraps``.

    Examples:
        >>> crossfade_overlap(crossfade_transition)
        0.5
        >>> crossfade_overlap(trim_and_crossfade)
        0.4
        >>> crossfade_overlap(overlap_blend)
        0.5
        >>> import functools
        >>> crossfade_overlap(functools.partial(crossfade_transition, duration=0.8))
        0.8

        A positional bind counts as much as a keyword one:

        >>> @needs_crossfade_overlap('fade')
        ... def positional_fade(clips=None, fade=0.5):
        ...     return clips
        >>> crossfade_overlap(functools.partial(positional_fade, None, 0.9))
        0.9

        Transforms that bake their effect into their own frames need no
        overlap, and neither does an undecorated callable:

        >>> crossfade_overlap(fade_through_black) is None
        True
        >>> crossfade_overlap(lambda clips: clips) is None
        True
        >>> crossfade_overlap(None) is None
        True

        A declaration nothing can supply a value for says so out loud:

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
    """
    if transform is None:
        return None

    bound, func = _partial_bindings(transform)

    param = getattr(func, _OVERLAP_PARAM_ATTR, None)
    if param is None:
        return None

    name = getattr(func, "__name__", func)
    if param in bound:
        value = bound[param]
    else:
        try:
            value = inspect.signature(func).parameters[param].default
        except (TypeError, ValueError, KeyError):  # pragma: no cover - defensive
            value = inspect.Parameter.empty
        if value is inspect.Parameter.empty:
            warnings.warn(
                f"{name!r} declares {param!r} as its crossfade overlap but no "
                f"value reached it and the parameter has no default; joining the "
                f"clips back to back, which renders a hard cut. Bind it, e.g. "
                f"functools.partial({name}, {param}=0.5).",
                stacklevel=2,
            )
            return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        warnings.warn(
            f"{name!r} declares {param!r} as its crossfade overlap but its value "
            f"{value!r} is not a number of seconds; joining the clips back to "
            f"back, which renders a hard cut.",
            stacklevel=2,
        )
        return None
    return seconds if seconds > 0 else None


def max_overlap_for_clips(clips) -> Optional[float]:
    """The largest overlap ``clips`` can be joined with, or ``None`` if unknowable.

    Note for anyone writing a ``transform_clips``: when the declared overlap
    exceeds this ceiling it is clamped, and the transform is invoked a **second**
    time with the clamped value. It must therefore be pure — a transform that
    accumulates state, or that may only run once, will be applied twice on that
    path.

    A crossfade eats ``overlap`` seconds off *each end it touches*: the first
    and last clip are touched once, every clip between them twice. So each clip
    affords ``duration / (number of joins it takes part in)`` and the join can
    only be as long as the tightest of those budgets. Past it two things go
    wrong at once, both of them silent:

    - a middle clip carries both a ``CrossFadeIn`` and a ``CrossFadeOut``
      (audio ramps too) whose ramps *multiply*, so its peak opacity falls to
      ``(duration / (2 * overlap)) ** 2`` and it never reaches full strength;
    - moviepy lays the clips out at ``cumsum(durations) + padding * arange``,
      so clip *i* and clip *i+2* start at the same instant — or in the wrong
      order, once the ``maximum(0, …)`` clamp bites — and the later one paints
      over the middle one. The render succeeds; the footage is simply gone.

    ``None`` when there is no join to bound (fewer than two clips) or when any
    clip's duration is unknown — a ceiling derived from a partial view would be
    wrong in the dangerous direction.

    Examples:
        >>> class _Clip:
        ...     def __init__(self, duration):
        ...         self.duration = duration

        Two clips are both end clips, so each affords its whole duration:

        >>> max_overlap_for_clips([_Clip(1.0), _Clip(1.0)])
        1.0

        Add a third and the middle one pays twice:

        >>> max_overlap_for_clips([_Clip(1.0), _Clip(1.0), _Clip(1.0)])
        0.5
        >>> max_overlap_for_clips([_Clip(3.0), _Clip(1.0), _Clip(3.0)])
        0.5
        >>> max_overlap_for_clips([_Clip(1.0), _Clip(3.0), _Clip(3.0)])
        1.0

        >>> max_overlap_for_clips([_Clip(2.0), _Clip(None)]) is None
        True
        >>> max_overlap_for_clips([_Clip(2.0)]) is None
        True
    """
    durations = [getattr(clip, "duration", None) for clip in clips]
    if len(durations) < 2 or any(d is None for d in durations):
        return None
    joins = [2] * len(durations)
    joins[0] = joins[-1] = 1  # the ends are touched by one join each
    return min(d / j for d, j in zip(durations, joins))


def _with_crossfade_overlap(transform: Callable, overlap: float) -> Callable:
    """``transform`` with its declared overlap parameter bound to ``overlap``.

    The declaration names a *parameter*, which is exactly what makes this
    possible: a clamped overlap can be handed back to the transform, so the
    ramps it builds and the padding the join uses stay the same number.
    """
    _bound, func = _partial_bindings(transform)
    param = getattr(func, _OVERLAP_PARAM_ATTR, None)
    if param is None:  # pragma: no cover - callers check first
        return transform
    return functools.partial(transform, **{param: overlap})


def _ensure_video_clip(video_src: VideoSource) -> VideoFileClip:
    """
    Convert various video source types to a VideoFileClip instance.

    Handles file paths (str/Path), existing VideoFileClip instances,
    bytes, BytesIO objects, and file-like objects.
    """
    if isinstance(video_src, VideoFileClip):
        return video_src
    elif isinstance(video_src, (str, Path)):
        return VideoFileClip(str(video_src))
    elif isinstance(video_src, bytes):
        return VideoFileClip(BytesIO(video_src))
    elif isinstance(video_src, BytesIO):
        return VideoFileClip(video_src)
    else:
        # Assume it's a file-like object
        return VideoFileClip(video_src)


def _iter_video_files(folder_path: Path) -> Iterable[Path]:
    """
    Generate video file paths from a folder in sorted order.

    >>> # Example (doctest would require actual folder):
    >>> # list(_iter_video_files(Path('/path/to/videos')))
    """
    for filepath in sorted(folder_path.iterdir()):
        if filepath.is_file() and _is_video_file(filepath):
            yield filepath


def ensure_videoclip_iterable(
    videos: Union[str, Path, Iterable[VideoSource]],
) -> Iterable[VideoFileClip]:
    """
    Normalize various video input formats to an iterable of VideoFileClip instances.

    Handles:
    - Folder paths (str/Path): Iterates through video files in sorted order
    - Iterables of VideoSource: Converts each item to VideoFileClip

    Args:
        videos: Can be:
            - A folder path (str or Path) containing video files
            - An iterable of VideoSource items (paths, clips, bytes, etc.)

    Returns:
        Iterable of VideoFileClip instances

    Examples:
        >>> # From folder path
        >>> clips = ensure_videoclip_iterable('/path/to/videos/')  # doctest: +SKIP

        >>> # From list of paths
        >>> clips = ensure_videoclip_iterable(['v1.mp4', 'v2.mp4'])  # doctest: +SKIP

        >>> # From existing clips
        >>> existing_clips = [VideoFileClip('v1.mp4')]  # doctest: +SKIP
        >>> clips = ensure_videoclip_iterable(existing_clips)  # doctest: +SKIP
    """
    # Check if it's a folder path
    if isinstance(videos, (str, Path)):
        folder_path = Path(videos)
        if folder_path.is_dir():
            return map(_ensure_video_clip, _iter_video_files(folder_path))
        else:
            # Single file path - wrap in iterable
            return map(_ensure_video_clip, [videos])

    # It's an iterable of VideoSource items
    return map(_ensure_video_clip, videos)


def verify_frame_continuity(
    video1: VideoSource,
    video2: VideoSource,
    *,
    tolerance: float = 0.0,
    save_comparison: str | None = None,
) -> tuple[bool, float]:
    """
    Verify that the last frame of video1 matches the first frame of video2.

    Args:
        video1: First video source
        video2: Second video source
        tolerance: Maximum allowed difference (0.0-1.0) for frames to be considered equal.
                  0.0 = exact match, 1.0 = completely different
        save_comparison: Optional path to save a side-by-side comparison image

    Returns:
        Tuple of (frames_match: bool, difference_score: float)
        difference_score is the mean absolute difference normalized to [0, 1]

    Example:
        >>> match, diff = verify_frame_continuity('video1.mp4', 'video2.mp4',
        ...                                       save_comparison='comparison.png')  # doctest: +SKIP
        >>> print(f"Frames match: {match}, difference: {diff:.4f}")  # doctest: +SKIP
    """
    clips_to_close = []

    try:
        # Get clips
        if not isinstance(video1, VideoFileClip):
            clip1 = _ensure_video_clip(video1)
            clips_to_close.append(clip1)
        else:
            clip1 = video1

        if not isinstance(video2, VideoFileClip):
            clip2 = _ensure_video_clip(video2)
            clips_to_close.append(clip2)
        else:
            clip2 = video2

        # Extract frames
        last_frame = clip1.get_frame(clip1.duration - 1 / clip1.fps)
        first_frame = clip2.get_frame(0)

        # Compare frames (normalized mean absolute difference)
        difference = (
            np.mean(np.abs(last_frame.astype(float) - first_frame.astype(float)))
            / 255.0
        )
        frames_match = difference <= tolerance

        # Save comparison if requested
        if save_comparison is not None:
            _save_frame_comparison(last_frame, first_frame, difference, save_comparison)

        return frames_match, difference

    finally:
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception:
                pass


def _save_frame_comparison(
    frame1: np.ndarray,
    frame2: np.ndarray,
    difference_score: float,
    output: str,
) -> None:
    """
    Save a side-by-side comparison of two frames with difference heatmap.

    Creates an image showing: [Frame 1] [Frame 2] [Difference Heatmap]
    """
    from PIL import Image, ImageDraw, ImageFont

    # Convert numpy arrays to PIL Images
    img1 = Image.fromarray(frame1)
    img2 = Image.fromarray(frame2)

    # Calculate absolute difference and create heatmap
    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)  # Average across RGB
    diff_img = Image.fromarray(diff_gray).convert("RGB")

    # Create side-by-side composite
    width, height = img1.size
    composite = Image.new("RGB", (width * 3, height + 40))
    composite.paste(img1, (0, 40))
    composite.paste(img2, (width, 40))
    composite.paste(diff_img, (width * 2, 40))

    # Add labels
    draw = ImageDraw.Draw(composite)
    try:
        # Try to use a decent font, fall back to default if not available
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 10), "Last Frame (Video 1)", fill="white", font=font)
    draw.text((width + 10, 10), "First Frame (Video 2)", fill="white", font=font)
    draw.text(
        (width * 2 + 10, 10), f"Diff: {difference_score:.1%}", fill="white", font=font
    )

    composite.save(output)
    logger.info("Comparison saved to: %s", output)


def concatenate_videos(
    videos: Union[str, Path, Iterable[VideoSource]],
    *,
    transform_clips: None
    | (Callable[[list[VideoFileClip]], Iterable[VideoFileClip]]) = None,
    normalize_dimensions: bool | Literal["stretch", "fit", "fill", "social"] = "social",
    target_width: int | None = None,
    target_height: int | None = None,
    output: str | bool | None = None,
    codec: str = "libx264",
    audio_codec: str = "aac",
    **concat_kwargs,
) -> VideoFileClip:
    """
    Concatenate multiple videos with optional clip transformation and dimension normalization.

    Audio from all input videos is automatically concatenated and included in the output.
    Automatically manages resource cleanup for clips created from sources.

    Args:
        videos: Can be:
            - A folder path (str or Path) containing video files (sorted order)
            - An iterable of video sources (file paths, VideoFileClip instances,
              bytes, BytesIO, or file-like objects)
        transform_clips: Optional function to transform clips before concatenation.
                        Receives a list of clips, returns an iterable of clips.
        normalize_dimensions: How to handle videos with different dimensions:
            - False: No normalization (may cause issues if dimensions differ)
            - 'stretch': Stretch all videos to match first video's dimensions
            - 'fit': Scale to fit inside dimensions with padding (letterbox/pillarbox)
            - 'fill': Scale to fill dimensions (may crop edges)
            - 'social': Scale with blurred/zoomed background (social media style) [DEFAULT]
            - True: Same as 'social'
        target_width: Explicit target width (overrides first video's dimensions)
        target_height: Explicit target height (overrides first video's dimensions)
        output: Optional path to save the concatenated video file.
                    If True and videos is a folder path, generates filename from folder name.
                    If True and videos is not a folder, raises ValueError.
        codec: Video codec to use when writing file (default: 'libx264')
        audio_codec: Audio codec to use when writing file (default: 'aac')
        **concat_kwargs: Additional arguments passed to moviepy's
            concatenate_videoclips. Anything given here wins over the join this
            function would otherwise pick (see below).

    How the clips are joined:
        A crossfade lives in the **join**, not in either clip, so the join is
        chosen from what ``transform_clips`` declares it needs (via
        :func:`needs_crossfade_overlap`) rather than defaulted for everything:

        - A transform declaring an overlap of *d* seconds
          (``crossfade_transition``, ``trim_and_crossfade``, ``overlap_blend``)
          is joined with ``method='compose', padding=-d``, so the clips overlap
          and their crossfade masks are actually composited. The result is
          therefore *shorter* than the sum of its clips by *d* per join, and the
          overlapping audio is crossfaded (the transforms pair every video mask
          with an ``afx`` gain ramp; without them the composited audio is
          *summed* at full level).
        - Everything else (no transform, or one that bakes its effect into its
          own frames, like ``fade_through_black`` / ``slow_motion_blend``) keeps
          moviepy's back-to-back default.

        An overlap larger than the clips can carry
        (:func:`max_overlap_for_clips`) does not render a longer crossfade — it
        **deletes clips**, because they land on top of one another. It is
        therefore clamped to the ceiling, with a ``UserWarning`` naming what was
        asked for and what was used, and the clamped value is fed back through
        the transform as well as into the padding so the ramps and the join stay
        one number. Passing ``padding=`` yourself takes the join over entirely:
        no ceiling, no clamp, no warning.

    Returns:
        Concatenated VideoFileClip with audio. Caller is responsible for closing this clip.

    Examples:
        >>> # From folder path with auto dimension handling (social media style)
        >>> final = concatenate_videos('/path/to/videos/')  # doctest: +SKIP

        >>> # From folder path with explicit output and letterboxing
        >>> final = concatenate_videos(
        ...     '/path/to/videos/',
        ...     normalize_dimensions='fit',
        ...     output='/path/output.mp4'
        ... )  # doctest: +SKIP

        >>> # From list of paths with transformation and specific dimensions
        >>> def trim_first_frame(clips):
        ...     '''Keep first clip intact, trim first frame from rest.'''
        ...     yield clips[0]
        ...     for clip in clips[1:]:
        ...         yield clip.subclipped(1 / clip.fps)  # doctest: +SKIP
        >>> paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']  # doctest: +SKIP
        >>> final = concatenate_videos(
        ...     paths,
        ...     transform_clips=trim_first_frame,
        ...     target_width=1920,
        ...     target_height=1080,
        ...     normalize_dimensions='social'
        ... )  # doctest: +SKIP
        >>> final.write_videofile('output.mp4')  # doctest: +SKIP
        >>> final.close()  # doctest: +SKIP
    """
    clips_to_close = []

    try:
        # Handle auto-generated output path from folder name
        if output is True:
            if isinstance(videos, (str, Path)):
                folder_path = Path(videos)
                if folder_path.is_dir():
                    # Generate output filename from folder name
                    output = str(folder_path.parent / f"{folder_path.name}.mp4")
                else:
                    raise ValueError("output=True requires videos to be a folder path")
            else:
                raise ValueError("output=True requires videos to be a folder path")

        # Normalize input to iterable of VideoFileClips
        # All clips created by ensure_videoclip_iterable need to be closed by us
        clips = list(ensure_videoclip_iterable(videos))
        clips_to_close.extend(clips)

        # Normalize dimensions if requested
        if normalize_dimensions is not False:
            from mixing.video.video_util import normalize_video_dimensions

            # Handle normalize_dimensions=True -> 'social'
            method = "social" if normalize_dimensions is True else normalize_dimensions

            normalized_clips = normalize_video_dimensions(
                clips,
                target_width=target_width,
                target_height=target_height,
                method=method,
            )

            # Track newly created normalized clips for cleanup
            original_ids = {id(c) for c in clips}
            for clip in normalized_clips:
                if id(clip) not in original_ids:
                    clips_to_close.append(clip)

            clips = normalized_clips

        def _transform(fn):
            """Run a ``transform_clips`` callable, registering what it created."""
            transformed = list(fn(clips))
            # Track new clips created by transformation (e.g., trimmed clips)
            original_ids = {id(c) for c in clips}
            for clip in transformed:
                if id(clip) not in original_ids:
                    clips_to_close.append(clip)
            return transformed

        # Apply transformation if provided
        if transform_clips is not None:
            clips_to_concat = _transform(transform_clips)
        else:
            clips_to_concat = clips

        # Concatenate and optionally write.
        # A crossfade needs BOTH conditions and moviepy 2.x defaults to neither:
        # under method="chain" the clips' CrossFade masks are never composited,
        # and padding=0 leaves no overlapping region for them to act in. Setting
        # only one of the two still renders a hard cut. An explicit caller
        # kwarg wins (setdefault), and a caller who set `padding` themselves
        # owns the join outright — no ceiling, no clamp, no warning.
        #
        # An overlap larger than the clips can carry does not render a longer
        # crossfade — it DELETES clips (see `max_overlap_for_clips`), and
        # silently, because the render succeeds. Clamp to what the clips can
        # carry and say so; the clamped value goes back through the transform
        # as well as into the padding, so the ramps and the join stay one
        # number (a padding-only clamp leaves the middle clip at 0.43x level).
        overlap = crossfade_overlap(transform_clips)
        if (
            overlap is not None
            and "padding" not in concat_kwargs
            and len(clips_to_concat) > 1  # one clip has no join to overlap
        ):
            ceiling = max_overlap_for_clips(clips_to_concat)
            if ceiling is not None and overlap > ceiling * (1 + _OVERLAP_TOLERANCE):
                warnings.warn(
                    f"{getattr(transform_clips, '__name__', transform_clips)!r} "
                    f"asked to overlap the join by {overlap:g}s, but these "
                    f"{len(clips_to_concat)} clips can carry at most "
                    f"{ceiling:g}s: a crossfade eats the overlap off each end it "
                    f"touches, and every clip but the first and last is touched "
                    f"twice (shortest clip "
                    f"{min(c.duration for c in clips_to_concat):g}s). Past that "
                    f"ceiling clips land on top of one another and some never "
                    f"appear in the output at all. Overlapping by {ceiling:g}s "
                    f"instead — use longer clips, ask for a shorter fade, or "
                    f"pass `padding=` yourself to own the join.",
                    stacklevel=2,
                )
                # `overlap is not None` already means a transform declared it.
                overlap = ceiling if ceiling > 0 else None
                if overlap is not None:
                    # SECOND invocation of the caller's transform. The ceiling
                    # above was derived from the FIRST call's output, so it is
                    # only valid for this one if the transform's clip durations
                    # do not depend on the overlap it was given. No shipped
                    # transform's do — this same change removed the one that
                    # did — but "no current caller breaks it" is a fact about
                    # today's callers, so it is CHECKED rather than assumed.
                    clips_to_concat = _transform(
                        _with_crossfade_overlap(transform_clips, overlap)
                    )
                    settled = max_overlap_for_clips(clips_to_concat)
                    if settled is not None and overlap > settled * (
                        1 + _OVERLAP_TOLERANCE
                    ):
                        raise ValueError(
                            f"{getattr(transform_clips, '__name__', transform_clips)!r} "
                            f"returns clips whose durations depend on the overlap "
                            f"it is given: clamping to {overlap:g}s produced clips "
                            f"that afford only {settled:g}s. The clamp cannot "
                            "converge, and proceeding would drop footage silently "
                            "— the failure this ceiling exists to prevent. Pass "
                            "`padding=` yourself to own the join, or make the "
                            "transform's output length independent of its overlap."
                        )
        if overlap is not None:
            concat_kwargs.setdefault("method", "compose")
            concat_kwargs.setdefault("padding", -overlap)

        final_clip = concatenate_videoclips(clips_to_concat, **concat_kwargs)
        if output is not None:
            # Explicitly include audio with proper codecs
            final_clip.write_videofile(output, codec=codec, audio_codec=audio_codec)

        return final_clip

    finally:
        # Clean up clips we created
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception:
                # Silently ignore cleanup errors
                pass


# ============================================================================
# TRANSITION FUNCTIONS - Try these alternatives
# ============================================================================


def trim_first_frame_from_subsequent_clips(
    clips: list[VideoFileClip],
) -> Iterable[VideoFileClip]:
    """Keep first clip intact, trim first frame from subsequent clips."""
    yield clips[0]
    for clip in clips[1:]:
        yield clip.subclipped(1 / clip.fps)


def _crossfade_effects(
    duration: float, *, fade_in: bool, fade_out: bool, video_out: bool = True
) -> list:
    """The mask *and* gain ramps one clip needs at an overlapped join.

    ``fade_in``/``fade_out`` say whether this clip has a neighbour on that side.

    The audio ramps are not decoration. An overlapped join composites the
    overlapping audio with moviepy's ``CompositeAudioClip``, which **sums** it;
    a clip that fades its picture but not its sound therefore plays both tracks
    at full level through every join — measured at +3.0 dB and, on ordinary
    material, hard against the rail. ``vfx.CrossFadeIn``/``CrossFadeOut`` are
    video-mask effects and do nothing to audio.

    ``video_out=False`` is :func:`overlap_blend`'s character — the outgoing
    picture stays opaque while the incoming one dissolves over it. It changes
    nothing about the audio, which overlaps either way.
    """
    effects = []
    if fade_in:
        effects.append(vfx.CrossFadeIn(duration))
    if fade_out and video_out:
        effects.append(vfx.CrossFadeOut(duration))
    if fade_in:
        effects.append(afx.AudioFadeIn(duration))
    if fade_out:
        effects.append(afx.AudioFadeOut(duration))
    return effects


@needs_crossfade_overlap("duration")
def crossfade_transition(
    clips: list[VideoFileClip], *, duration: float = 0.5
) -> Iterable[VideoFileClip]:
    """
    Crossfade between clips to smoothly blend spatial and temporal discontinuities.

    Best for hiding both pixel differences and motion changes.
    Recommended: duration=0.3 to 0.8 seconds.

    Uses CrossFadeOut on end of clips and CrossFadeIn on start of clips. Those
    are masks, so they only render as a blend when the clips are composited
    with a ``duration``-second overlap — which is what the
    :func:`needs_crossfade_overlap` declaration tells :func:`concatenate_videos`
    to do. Each join therefore shortens the result by ``duration``.

    The audio is crossfaded too (:func:`_crossfade_effects`) — an overlapped
    join *sums* the overlapping audio, so without the matching fades both
    tracks play at full level through every join and the sum clips.
    """
    for i, clip in enumerate(clips):
        yield clip.with_effects(
            _crossfade_effects(duration, fade_in=i > 0, fade_out=i < len(clips) - 1)
        )


@needs_crossfade_overlap("duration")
def trim_and_crossfade(
    clips: list[VideoFileClip], *, duration: float = 0.4
) -> Iterable[VideoFileClip]:
    """
    Trim first frame from subsequent clips, then crossfade.

    Combines frame removal with smooth blending. Like
    :func:`crossfade_transition`, the blend only happens because the declared
    ``duration`` overlap reaches the join, and the audio is crossfaded to match.
    """
    for i, clip in enumerate(clips):
        # Subsequent clips: trim the duplicated first frame before blending
        clip = clip if i == 0 else clip.subclipped(1 / clip.fps)
        yield clip.with_effects(
            _crossfade_effects(duration, fade_in=i > 0, fade_out=i < len(clips) - 1)
        )


def fade_through_black(
    clips: list[VideoFileClip], *, duration: float = 0.3
) -> Iterable[VideoFileClip]:
    """
    Fade out to black, then fade in from black between clips.

    More dramatic transition - clearly separates scenes.

    Deliberately **not** declared with :func:`needs_crossfade_overlap`:
    ``FadeIn`` / ``FadeOut`` bake the effect into each clip's own frames, so a
    back-to-back join renders it correctly and an overlapped one would eat
    footage for nothing.
    """
    for i, clip in enumerate(clips):
        effects = []
        if i > 0:
            # Fade in from black at start of clip (except first)
            effects.append(vfx.FadeIn(duration))
        if i < len(clips) - 1:
            # Fade out to black at end of clip (except last)
            effects.append(vfx.FadeOut(duration))

        if effects:
            clip = clip.with_effects(effects)
        yield clip


def slow_motion_blend(
    clips: list[VideoFileClip], *, ramp_duration: float = 0.5
) -> Iterable[VideoFileClip]:
    """
    Slow down the end of each clip and beginning of next for smoother motion transition.

    Helps with motion discontinuity by creating a speed buffer zone.
    Note: This changes timing, so final video will be slightly longer.
    """
    for i, clip in enumerate(clips):
        if i == 0:
            # First clip: just slow down the end
            slow_end = clip.subclipped(clip.duration - ramp_duration).with_speed_scaled(
                0.5
            )
            main_part = clip.subclipped(0, clip.duration - ramp_duration)
            yield main_part
            yield slow_end
        else:
            # Subsequent clips: slow start, normal middle, slow end
            slow_start = clip.subclipped(0, ramp_duration).with_speed_scaled(0.5)
            yield slow_start

            if i < len(clips) - 1:
                # Not the last clip
                main_part = clip.subclipped(
                    ramp_duration, clip.duration - ramp_duration
                )
                slow_end = clip.subclipped(
                    clip.duration - ramp_duration
                ).with_speed_scaled(0.5)
                yield main_part
                yield slow_end
            else:
                # Last clip
                main_part = clip.subclipped(ramp_duration)
                yield main_part


@needs_crossfade_overlap("overlap")
def overlap_blend(
    clips: list[VideoFileClip], *, overlap: float = 0.5
) -> Iterable[VideoFileClip]:
    """
    Overlap clips and crossfade the overlapping region.

    The gentler sibling of :func:`crossfade_transition`: the incoming clip
    dissolves in over an outgoing clip that stays fully opaque (no
    ``CrossFadeOut``), so the cut reads as a wipe-through rather than a dip.
    Every frame of every clip is used — the ``overlap`` is consumed by the join
    itself, exactly once.

    .. note::
       This used to *also* trim ``overlap`` seconds off the head of every clip
       after the first. That trim was a stand-in for an overlap back when the
       join was back-to-back; once the join genuinely overlaps, keeping it
       charged ``2 x overlap`` per clip and **deleted whole clips** — three
       1 s clips at the default came out as a 1 s video with the middle clip
       nowhere in it. The audio fades are here for the same reason the video
       ones are: an overlapped join sums the overlapping audio.
    """
    for i, clip in enumerate(clips):
        yield clip.with_effects(
            _crossfade_effects(
                overlap,
                fade_in=i > 0,
                fade_out=i < len(clips) - 1,
                video_out=False,
            )
        )
