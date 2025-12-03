"""Video utils.

Utils that will be useful in multiple modules of the video package.
"""

from typing import Literal, Tuple, Optional
import numpy as np
from moviepy import VideoFileClip, VideoClip, ImageClip, CompositeVideoClip


def get_video_dimensions(video: VideoFileClip) -> Tuple[int, int]:
    """
    Get the (width, height) dimensions of a video.

    >>> clip = VideoFileClip('video.mp4')  # doctest: +SKIP
    >>> width, height = get_video_dimensions(clip)  # doctest: +SKIP
    """
    return video.w, video.h


def resize_to_dimensions(
    video: VideoFileClip,
    target_width: int,
    target_height: int,
    *,
    method: Literal['stretch', 'fit', 'fill', 'social'] = 'fit',
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

    if method == 'stretch':
        # Simple stretch - may distort aspect ratio
        return video.resized(new_size=(target_width, target_height))

    elif method == 'fit':
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

    elif method == 'fill':
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

    elif method == 'social':
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
            """Apply Gaussian blur to a frame using PIL."""
            from PIL import Image, ImageFilter
            import numpy as np

            # Convert numpy array to PIL Image
            img = Image.fromarray(frame.astype('uint8'))
            # Apply Gaussian blur
            blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
            # Convert back to numpy array
            return np.array(blurred)

        background = background.image_transform(blur_frame)

        # Optionally darken the background slightly for better contrast
        from moviepy import vfx

        background = background.with_effects([vfx.MultiplyColor([0.7, 0.7, 0.7])])

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
    method: Literal['stretch', 'fit', 'fill', 'social'] = 'social',
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
