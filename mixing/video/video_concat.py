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
    - overlap_blend: Overlap and blend clips aggressively

Frame Verification:
    >>> match, diff = verify_frame_continuity('v1.mp4', 'v2.mp4',
    ...   save_comparison='comp.png')    # doctest: +SKIP
"""

from typing import Iterable, Callable, Optional, Union
from pathlib import Path
from io import BytesIO
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, vfx

VideoSource = Union[str, Path, VideoFileClip, bytes, BytesIO]


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


def verify_frame_continuity(
    video1: VideoSource,
    video2: VideoSource,
    *,
    tolerance: float = 0.0,
    save_comparison: Optional[str] = None,
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
    output_path: str,
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
    diff_img = Image.fromarray(diff_gray).convert('RGB')

    # Create side-by-side composite
    width, height = img1.size
    composite = Image.new('RGB', (width * 3, height + 40))
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

    draw.text((10, 10), "Last Frame (Video 1)", fill='white', font=font)
    draw.text((width + 10, 10), "First Frame (Video 2)", fill='white', font=font)
    draw.text(
        (width * 2 + 10, 10), f"Diff: {difference_score:.1%}", fill='white', font=font
    )

    composite.save(output_path)
    print(f"Comparison saved to: {output_path}")


def concatenate_videos(
    videos: Iterable[VideoSource],
    *,
    transform_clips: Optional[
        Callable[[list[VideoFileClip]], Iterable[VideoFileClip]]
    ] = None,
    output_path: Optional[str] = None,
) -> VideoFileClip:
    """
    Concatenate multiple videos with optional clip transformation.

    Automatically manages resource cleanup for clips created from sources,
    while preserving user-provided VideoFileClip instances for caller control.

    Args:
        videos: Iterable of video sources (file paths, VideoFileClip instances,
                bytes, BytesIO, or file-like objects)
        transform_clips: Optional function to transform clips before concatenation.
                        Receives a list of clips, returns an iterable of clips.
        output_path: Optional path to save the concatenated video file

    Returns:
        Concatenated VideoFileClip. Caller is responsible for closing this clip.

    Example:
        >>> def trim_first_frame(clips):
        ...     '''Keep first clip intact, trim first frame from rest.'''
        ...     yield clips[0]
        ...     for clip in clips[1:]:
        ...         yield clip.subclipped(1 / clip.fps)  # doctest: +SKIP
        >>> paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']  # doctest: +SKIP
        >>> final = concatenate_videos(paths, transform_clips=trim_first_frame)  # doctest: +SKIP
        >>> final.write_videofile('output.mp4')  # doctest: +SKIP
        >>> final.close()  # doctest: +SKIP
    """
    clips_to_close = []

    try:
        # Convert all video sources to clips, tracking which ones we create
        clips = []
        for video_src in videos:
            if isinstance(video_src, VideoFileClip):
                # User-provided clip; they manage lifecycle
                clips.append(video_src)
            else:
                # We create this clip; we manage cleanup
                clip = _ensure_video_clip(video_src)
                clips.append(clip)
                clips_to_close.append(clip)

        # Apply transformation if provided
        if transform_clips is not None:
            transformed = list(transform_clips(clips))
            # Track new clips created by transformation (e.g., trimmed clips)
            original_ids = {id(c) for c in clips}
            for clip in transformed:
                if id(clip) not in original_ids:
                    clips_to_close.append(clip)
            clips_to_concat = transformed
        else:
            clips_to_concat = clips

        # Concatenate and optionally write
        final_clip = concatenate_videoclips(clips_to_concat)
        if output_path is not None:
            final_clip.write_videofile(output_path)

        return final_clip

    finally:
        # Clean up only clips we created
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception:
                # Silently ignore cleanup errors
                pass


# Example usage matching your original code
if __name__ == "__main__":
    video_paths = """
    /Users/thorwhalen/Downloads/cosmo_vids/00_Book_Skyscraper_City_Video.mp4
    /Users/thorwhalen/Downloads/cosmo_vids/01_Book_Towers_Collapsing_Video.mp4
    /Users/thorwhalen/Downloads/cosmo_vids/02_meh_Catastrophic_Book_City_Collapse_Video.mp4
    """.split()

    def trim_first_frame_from_subsequent_clips(
        clips: list[VideoFileClip],
    ) -> Iterable[VideoFileClip]:
        """Keep first clip intact, trim first frame from subsequent clips."""
        yield clips[0]
        for clip in clips[1:]:
            yield clip.subclipped(1 / clip.fps)

    output_path = "/Users/thorwhalen/Downloads/cosmo_vids/concatenated_video.mp4"

    final_clip = concatenate_videos(
        video_paths,
        transform_clips=trim_first_frame_from_subsequent_clips,
        output_path=output_path,
    )
    final_clip.close()

    print(f"Successfully concatenated videos and saved to {output_path}")


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


def crossfade_transition(
    clips: list[VideoFileClip], *, duration: float = 0.5
) -> Iterable[VideoFileClip]:
    """
    Crossfade between clips to smoothly blend spatial and temporal discontinuities.

    Best for hiding both pixel differences and motion changes.
    Recommended: duration=0.3 to 0.8 seconds.

    Uses CrossFadeOut on end of clips and CrossFadeIn on start of clips.
    """
    for i, clip in enumerate(clips):
        if i == 0:
            # First clip: just fade out at the end
            yield clip.with_effects([vfx.CrossFadeOut(duration)])
        elif i == len(clips) - 1:
            # Last clip: just fade in at the start
            yield clip.with_effects([vfx.CrossFadeIn(duration)])
        else:
            # Middle clips: fade in at start, fade out at end
            yield clip.with_effects(
                [vfx.CrossFadeIn(duration), vfx.CrossFadeOut(duration)]
            )


def trim_and_crossfade(
    clips: list[VideoFileClip], *, duration: float = 0.4
) -> Iterable[VideoFileClip]:
    """
    Trim first frame from subsequent clips, then crossfade.

    Combines frame removal with smooth blending.
    """
    for i, clip in enumerate(clips):
        if i == 0:
            # First clip: just fade out at the end
            yield clip.with_effects([vfx.CrossFadeOut(duration)])
        else:
            # Subsequent clips: trim first frame, then fade in/out
            trimmed = clip.subclipped(1 / clip.fps)
            if i == len(clips) - 1:
                # Last clip: just fade in
                yield trimmed.with_effects([vfx.CrossFadeIn(duration)])
            else:
                # Middle clips: fade in and out
                yield trimmed.with_effects(
                    [vfx.CrossFadeIn(duration), vfx.CrossFadeOut(duration)]
                )


def fade_through_black(
    clips: list[VideoFileClip], *, duration: float = 0.3
) -> Iterable[VideoFileClip]:
    """
    Fade out to black, then fade in from black between clips.

    More dramatic transition - clearly separates scenes.
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
            slow_end = clip.subclipped(clip.duration - ramp_duration).with_speed(0.5)
            main_part = clip.subclipped(0, clip.duration - ramp_duration)
            yield main_part
            yield slow_end
        else:
            # Subsequent clips: slow start, normal middle, slow end
            slow_start = clip.subclipped(0, ramp_duration).with_speed(0.5)
            yield slow_start

            if i < len(clips) - 1:
                # Not the last clip
                main_part = clip.subclipped(
                    ramp_duration, clip.duration - ramp_duration
                )
                slow_end = clip.subclipped(clip.duration - ramp_duration).with_speed(
                    0.5
                )
                yield main_part
                yield slow_end
            else:
                # Last clip
                main_part = clip.subclipped(ramp_duration)
                yield main_part


def overlap_blend(
    clips: list[VideoFileClip], *, overlap: float = 0.5
) -> Iterable[VideoFileClip]:
    """
    Overlap clips and crossfade the overlapping region.

    More aggressive blending - uses more footage from both clips.
    """
    for i, clip in enumerate(clips):
        if i == 0:
            yield clip
        else:
            # Trim overlap amount from start, then crossfade
            trimmed = clip.subclipped(overlap)
            yield trimmed.with_effects([vfx.CrossFadeIn(overlap)])
