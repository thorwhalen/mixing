"""Video utils.

Utils that will be useful in multiple modules of the video package.

**The geometry vocabulary comes from** :mod:`looks.geometry` **; the
implementations stay here.** ``SOCIAL_SIZES``, the ``stretch``/``fit``/``fill``
mode names and the two constants that parameterise the ``social`` backdrop are
*imported*, not re-declared, so there is exactly one place that says what a
"shorts" is or how blurred a social backdrop gets. What is deliberately **not**
imported is the arithmetic: this module resizes with moviepy, and the ``social``
branch is a composite (a scaled, centre-cropped, Gaussian-blurred, dimmed copy
of the input behind the fitted foreground), not a formula. `looks` is
stdlib-only, so importing the vocabulary costs nothing at import time.

:func:`get_video_dimensions` stays here on purpose — it is a *probe* (it opens
a file, or reads a live clip), not geometry, so it has no home in a pure-
arithmetic module.
"""

from typing import Literal, Tuple, Optional, Union
import numpy as np
from moviepy import VideoFileClip, VideoClip, ImageClip, CompositeVideoClip

# The geometry vocabulary, imported rather than re-declared:
#
# - ``SOCIAL_SIZES`` maps a preset name to a ``(width, height)`` pixel pair,
#   handy as the ``target_width``/``target_height`` for
#   :func:`resize_to_dimensions` and friends. Landscape ``youtube`` is 16:9;
#   the vertical 9:16 presets (``shorts`` / ``story`` / ``tiktok``) and the 1:1
#   ``square`` cover the usual short-form formats.
# - ``FitMode`` is the ``stretch``/``fit``/``fill`` name set.
# - ``DFLT_BACKDROP_BLUR_SIGMA`` / ``DFLT_BACKDROP_DIM`` parameterise the
#   ``social`` backdrop below. They were transcribed *from* this module into
#   `looks`; importing them back is what stops the two copies drifting.
from looks.geometry import (
    DFLT_BACKDROP_BLUR_SIGMA,
    DFLT_BACKDROP_DIM,
    FitMode,
    SOCIAL_SIZES,  # noqa: F401  — deliberate re-export; a --fix would break the port
)

#: How :func:`resize_to_dimensions` places a source frame in a target frame.
#: The first three names are `looks`' :data:`~looks.geometry.FitMode`; ``social``
#: is **not** a fourth mode but ``fit`` over a blurred, dimmed copy of the source
#: instead of a solid colour — which is why it lives here (it is a composite)
#: while the other three are arithmetic.
ResizeMethod = Union[FitMode, Literal["social"]]


def get_video_dimensions(video) -> Tuple[int, int]:
    """
    Get the (width, height) dimensions of a video — a moviepy clip, or a
    file path (probed via cv2, so no clip is opened).

    >>> width, height = get_video_dimensions('video.mp4')  # doctest: +SKIP
    >>> clip = VideoFileClip('video.mp4')  # doctest: +SKIP
    >>> width, height = get_video_dimensions(clip)  # doctest: +SKIP
    """
    import os

    if isinstance(video, (str, os.PathLike)):
        import cv2

        cap = cv2.VideoCapture(os.fspath(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video}")
            return (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        finally:
            cap.release()
    return video.w, video.h


def resize_to_dimensions(
    video: VideoFileClip,
    target_width: int,
    target_height: int,
    *,
    method: ResizeMethod = "fit",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> VideoFileClip:
    """
    Resize a video to target dimensions with different methods.

    Args:
        video: Input video clip
        target_width: Target width in pixels
        target_height: Target height in pixels
        method: Resizing method:
            - 'stretch': Stretch to fit (may distort aspect ratio)
            - 'fit': Scale to fit inside target (maintains aspect ratio, adds padding)
            - 'fill': Scale to fill target (maintains aspect ratio, may crop)
            - 'social': Like 'fill' but uses blurred/zoomed background (social media style)
        bg_color: Background color for padding (RGB tuple, 0-255)

    Returns:
        Resized VideoFileClip

    Examples:
        >>> # Stretch to exact dimensions (may distort)
        >>> resized = resize_to_dimensions(clip, 1920, 1080, method='stretch')  # doctest: +SKIP

        >>> # Fit inside dimensions with black padding
        >>> resized = resize_to_dimensions(clip, 1920, 1080, method='fit')  # doctest: +SKIP

        >>> # Fill dimensions (may crop edges)
        >>> resized = resize_to_dimensions(clip, 1920, 1080, method='fill')  # doctest: +SKIP

        >>> # Social media style with blurred background
        >>> resized = resize_to_dimensions(clip, 1920, 1080, method='social')  # doctest: +SKIP
    """
    current_width, current_height = get_video_dimensions(video)
    target_aspect = target_width / target_height
    current_aspect = current_width / current_height

    if method == "stretch":
        # Simple stretch - may distort aspect ratio
        return video.resized(new_size=(target_width, target_height))

    elif method == "fit":
        # Scale to fit inside target dimensions, add padding if needed
        if current_aspect > target_aspect:
            # Video is wider - scale by width
            new_width = target_width
            new_height = int(target_width / current_aspect)
        else:
            # Video is taller - scale by height
            new_height = target_height
            new_width = int(target_height * current_aspect)

        # Resize video
        resized = video.resized(new_size=(new_width, new_height))

        # Add padding if needed
        if new_width != target_width or new_height != target_height:
            # Create background
            bg = ImageClip(
                np.full((target_height, target_width, 3), bg_color, dtype=np.uint8),
                duration=video.duration,
            ).with_fps(video.fps)

            # Center the resized video on the background
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            return (
                CompositeVideoClip(
                    [bg, resized.with_position((x_offset, y_offset))],
                    size=(target_width, target_height),
                )
                .with_duration(video.duration)
                .with_fps(video.fps)
            )

        return resized

    elif method == "fill":
        # Scale to fill target dimensions, may crop edges
        if current_aspect > target_aspect:
            # Video is wider - scale by height (will crop sides)
            new_height = target_height
            new_width = int(target_height * current_aspect)
        else:
            # Video is taller - scale by width (will crop top/bottom)
            new_width = target_width
            new_height = int(target_width / current_aspect)

        # Resize video
        resized = video.resized(new_size=(new_width, new_height))

        # Crop to target dimensions (center crop)
        x_offset = (new_width - target_width) // 2
        y_offset = (new_height - target_height) // 2

        return resized.cropped(
            x1=x_offset,
            y1=y_offset,
            x2=x_offset + target_width,
            y2=y_offset + target_height,
        )

    elif method == "social":
        # Social media style: blurred/zoomed background with video on top
        # Scale video to fit inside target
        if current_aspect > target_aspect:
            new_width = target_width
            new_height = int(target_width / current_aspect)
        else:
            new_height = target_height
            new_width = int(target_height * current_aspect)

        foreground = video.resized(new_size=(new_width, new_height))

        # Create blurred, zoomed background
        # Scale the original video to fill the target (will be blurred)
        if current_aspect > target_aspect:
            bg_height = target_height
            bg_width = int(target_height * current_aspect)
        else:
            bg_width = target_width
            bg_height = int(target_width / current_aspect)

        background = video.resized(new_size=(bg_width, bg_height))

        # Center crop background
        x_offset = (bg_width - target_width) // 2
        y_offset = (bg_height - target_height) // 2
        background = background.cropped(
            x1=x_offset,
            y1=y_offset,
            x2=x_offset + target_width,
            y2=y_offset + target_height,
        )

        # Apply blur to background using PIL
        def blur_frame(frame):
            """Apply Gaussian blur to a frame using PIL.

            PIL's ``GaussianBlur(radius=...)`` takes the standard deviation of
            the kernel, which is what ``DFLT_BACKDROP_BLUR_SIGMA`` names.
            """
            from PIL import Image, ImageFilter
            import numpy as np

            # Convert numpy array to PIL Image
            img = Image.fromarray(frame.astype("uint8"))
            # Apply Gaussian blur
            blurred = img.filter(
                ImageFilter.GaussianBlur(radius=DFLT_BACKDROP_BLUR_SIGMA)
            )
            # Convert back to numpy array
            return np.array(blurred)

        background = background.image_transform(blur_frame)

        # Optionally darken the background slightly for better contrast
        from moviepy import vfx

        background = background.with_effects(
            [vfx.MultiplyColor([DFLT_BACKDROP_DIM] * 3)]
        )

        # Center the foreground on the background
        x_pos = (target_width - new_width) // 2
        y_pos = (target_height - new_height) // 2

        return (
            CompositeVideoClip(
                [background, foreground.with_position((x_pos, y_pos))],
                size=(target_width, target_height),
            )
            .with_duration(video.duration)
            .with_fps(video.fps)
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: 'stretch', 'fit', 'fill', 'social'"
        )


def normalize_video_dimensions(
    videos: list[VideoFileClip],
    *,
    reference_video: Optional[int | VideoFileClip] = 0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    method: ResizeMethod = "social",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> list[VideoFileClip]:
    """
    Normalize all videos to the same dimensions.

    Args:
        videos: List of video clips to normalize
        reference_video: Index of video to use as reference (default: 0 = first video)
                        or a VideoFileClip instance to use as reference
        target_width: Explicit target width (overrides reference_video)
        target_height: Explicit target height (overrides reference_video)
        method: Resizing method ('stretch', 'fit', 'fill', 'social')
        bg_color: Background color for padding

    Returns:
        List of normalized VideoFileClip instances

    Examples:
        >>> # Normalize all to first video's dimensions
        >>> normalized = normalize_video_dimensions(clips)  # doctest: +SKIP

        >>> # Normalize all to specific dimensions with social media style
        >>> normalized = normalize_video_dimensions(
        ...     clips, target_width=1920, target_height=1080, method='social'
        ... )  # doctest: +SKIP

        >>> # Normalize to second video's dimensions
        >>> normalized = normalize_video_dimensions(clips, reference_video=1)  # doctest: +SKIP
    """
    if not videos:
        return []

    # Determine target dimensions
    if target_width is not None and target_height is not None:
        # Explicit dimensions provided
        pass
    elif isinstance(reference_video, VideoFileClip):
        # Use provided reference video
        target_width, target_height = get_video_dimensions(reference_video)
    elif isinstance(reference_video, int):
        # Use video at index as reference
        target_width, target_height = get_video_dimensions(videos[reference_video])
    else:
        raise ValueError(
            "Must provide either target_width/target_height or reference_video"
        )

    # Resize all videos to target dimensions
    normalized = []
    for video in videos:
        current_width, current_height = get_video_dimensions(video)
        if current_width == target_width and current_height == target_height:
            # Already correct dimensions
            normalized.append(video)
        else:
            # Need to resize
            normalized.append(
                resize_to_dimensions(
                    video,
                    target_width,
                    target_height,
                    method=method,
                    bg_color=bg_color,
                )
            )

    return normalized
