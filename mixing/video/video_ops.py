"""
Video cropping via slicing interface.

Provides a lazy view into video segments without copying the underlying file.

This module provides:
- `Video`: A sliceable video interface using `video[start:end]` or `video[idx]` notation
- `crop_video()`: Convenience function for direct cropping operations
- Flexible time units: seconds, frames, or milliseconds
- Integration with moviepy and cv2 for further processing
- Frame extraction via integer indexing

Examples:
    >>> video = Video("my_video.mp4")  # doctest: +SKIP
    >>> segment = video[10:20]  # Lazy view, no copying  # doctest: +SKIP
    >>> segment.save("clip.mp4")  # Only then does it process  # doctest: +SKIP

    >>> # Extract single frame
    >>> frame = video[100]  # Returns numpy array (frame 100)  # doctest: +SKIP
    >>> # Or as a video segment
    >>> frame_video = video[10.5:10.5]  # Single frame at 10.5s  # doctest: +SKIP

    >>> # Use frame numbers instead
    >>> video = Video("movie.mp4", time_unit="frames")  # doctest: +SKIP
    >>> segment = video[100:500]  # Use frame numbers  # doctest: +SKIP

    >>> # Get a clip for further processing
    >>> with video[5:15].to_clip() as clip:  # doctest: +SKIP
    ...     reversed = clip.fx(mp.vfx.time_mirror)  # doctest: +SKIP
    ...     reversed.write_videofile("output.mp4")  # doctest: +SKIP

Design principles:
- Lazy evaluation: Slicing creates views, not copies
- Facade pattern: Clean interface over moviepy/cv2 complexity
- Standard library interfaces: Uses Python's slice notation
- Dependency injection: Configurable time units and codecs
- Open-closed: Extensible via keyword arguments
- Single source of truth: One class handles both time ranges and frames
"""

from typing import Union
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping
import io
import os
import tempfile
import numpy as np
import cv2
import moviepy as mp

from ..util import require_package, TimeUnit, to_seconds
from ._helpers import (
    _auto_video_path,
    _auto_frame_path,
    _set_default_codecs,
    _ensure_output_path,
    _resolve_output_path,
)


def _to_seconds(value: float, *, unit: TimeUnit, fps: float) -> float:
    """Convert time value to seconds based on unit (wraps util.to_seconds with fps as rate)."""
    return to_seconds(value, unit=unit, rate=fps)


def _get_video_path_from_clipboard() -> str:
    """Get video file path from clipboard and validate it."""
    from ..util import get_path_from_clipboard

    return get_path_from_clipboard()


def _copy_frame_to_clipboard(frame: np.ndarray) -> None:
    """
    Copy a cv2 frame (numpy array) to system clipboard as pasteable image.

    Requires: pillow, pyclip
    """
    from ..util import copy_to_clipboard

    PIL_Image = require_package('PIL.Image')

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = PIL_Image.fromarray(rgb_frame)

    # Convert PIL Image to bytes for clipboard
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    copy_to_clipboard(image_bytes)


class VideoFrames(Mapping[int, np.ndarray]):
    """
    Mapping interface to access video frames by index.

    Provides dictionary-like access to video frames with support for negative
    indexing and slicing. Frames are returned as numpy arrays (BGR format).

    Args:
        video_src: Path to the video file
        start_frame: Starting frame index (for segments)
        end_frame: Ending frame index (for segments)

    Examples:
        >>> vf = VideoFrames("test_video.mp4")  # doctest: +SKIP
        >>> frame = vf[0]  # Get first frame  # doctest: +SKIP
        >>> last_frame = vf[-1]  # Get last frame  # doctest: +SKIP
        >>> frames = list(vf[10:20])  # Get frames 10-19  # doctest: +SKIP
    """

    def __init__(
        self, video_src: str, start_frame: int = 0, end_frame: int | None = None
    ):
        self.video_src = video_src
        self._cap = cv2.VideoCapture(video_src)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_src}")
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap.release()

        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else total_frames
        self._frame_count = self.end_frame - self.start_frame

    def __len__(self) -> int:
        """Return number of frames in this view."""
        return self._frame_count

    def __iter__(self) -> Iterator[int]:
        """Iterate over frame indices in this view."""
        return iter(range(self.start_frame, self.end_frame))

    def _normalize_index(self, idx: int) -> int:
        """Convert negative indices to positive, relative to this view."""
        if idx < 0:
            idx = self._frame_count + idx
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        return self.start_frame + idx

    def _read_frame_at_index(self, idx: int) -> np.ndarray:
        """Read a single frame at the given absolute index."""
        cap = cv2.VideoCapture(self.video_src)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at index {idx}")
            return frame
        finally:
            cap.release()

    def _iterate_frames(
        self, start: int, stop: int, step: int = 1
    ) -> Iterator[np.ndarray]:
        """Iterate over frames in the given range."""
        # Normalize indices relative to this view
        if start < 0:
            start = max(0, self._frame_count + start)
        if stop < 0:
            stop = max(0, self._frame_count + stop)

        start = max(0, min(start, self._frame_count))
        stop = max(0, min(stop, self._frame_count))

        # Convert to absolute frame indices
        abs_start = self.start_frame + start
        abs_stop = self.start_frame + stop

        if step == 1:
            # Efficient sequential reading
            cap = cv2.VideoCapture(self.video_src)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_start)
                for _ in range(abs_start, abs_stop):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
            finally:
                cap.release()
        else:
            # Random access for non-sequential steps
            for idx in range(abs_start, abs_stop, step):
                yield self._read_frame_at_index(idx)

    def __getitem__(self, key: int | slice) -> np.ndarray | Iterator[np.ndarray]:
        """
        Get frame(s) by index or slice.

        Args:
            key: Integer index or slice object (relative to this view)

        Returns:
            Single frame (np.ndarray) for integer index,
            or iterator of frames for slice
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self._frame_count)
            return self._iterate_frames(start, stop, step or 1)
        elif isinstance(key, int):
            abs_idx = self._normalize_index(key)
            return self._read_frame_at_index(abs_idx)
        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )


class Video:
    """
    Sliceable interface for videos supporting both time ranges and frame extraction.

    Provides lazy views into video segments using slice notation, and direct frame
    access using integer indexing. Slicing returns new Video instances
    (not copies), enabling chained operations.

    Args:
        video_src: Path to source video file
        time_unit: Unit for slice indices ('seconds', 'frames', 'milliseconds')
        start_time: Start time in seconds (for creating sub-views)
        end_time: End time in seconds (for creating sub-views)

    Examples:
        >>> video = Video("movie.mp4")  # doctest: +SKIP
        >>>
        >>> # Get segment from 10s to 20s (returns Video)
        >>> segment = video[10:20]  # doctest: +SKIP
        >>> segment.save("clip.mp4")  # doctest: +SKIP
        >>>
        >>> # Extract single frame (returns numpy array)
        >>> frame = video[100]  # Frame at 100 seconds  # doctest: +SKIP
        >>>
        >>> # Use frame numbers as unit
        >>> video_frames = Video("movie.mp4", time_unit="frames")  # doctest: +SKIP
        >>> segment = video_frames[100:500]  # Frames 100-500  # doctest: +SKIP
        >>> single_frame = video_frames[250]  # Single frame  # doctest: +SKIP
        >>>
        >>> # Get last 30 seconds
        >>> ending = video[-30:]  # doctest: +SKIP
        >>>
        >>> # Chain operations with moviepy
        >>> with video[5:15].to_clip() as clip:  # doctest: +SKIP
        ...     reversed_clip = clip.fx(mp.vfx.time_mirror)  # doctest: +SKIP
        ...     reversed_clip.write_videofile("reversed.mp4")  # doctest: +SKIP
    """

    def __init__(
        self,
        video_src: str,
        *,
        time_unit: TimeUnit = "seconds",
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        self.video_src = str(video_src)
        self.time_unit = time_unit
        self._start_time = start_time  # None means start of video
        self._end_time = end_time  # None means end of video
        self._duration = None
        self._fps = None
        self._frame_count = None

    @property
    def start_time(self) -> float:
        """Start time in seconds (0.0 if not set)."""
        return self._start_time if self._start_time is not None else 0.0

    @property
    def end_time(self) -> float:
        """End time in seconds (video duration if not set)."""
        return self._end_time if self._end_time is not None else self.full_duration

    @property
    def full_duration(self) -> float:
        """Duration of the source video in seconds."""
        if self._duration is None:
            with mp.VideoFileClip(self.video_src) as clip:
                self._duration = clip.duration
        return self._duration

    @property
    def duration(self) -> float:
        """Duration of this video/segment in seconds."""
        return self.end_time - self.start_time

    @property
    def fps(self) -> float:
        """Frames per second of video."""
        if self._fps is None:
            with mp.VideoFileClip(self.video_src) as clip:
                self._fps = clip.fps
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in source video."""
        if self._frame_count is None:
            cap = cv2.VideoCapture(self.video_src)
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self._frame_count

    def _find_last_readable_frame(self) -> int:
        """Find the last actually readable frame index (some videos have corrupted end frames)."""
        cap = cv2.VideoCapture(self.video_src)
        try:
            reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Binary search for last readable frame
            left, right = 0, reported_count - 1
            last_readable = 0

            while left <= right:
                mid = (left + right) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, _ = cap.read()

                if ret:
                    last_readable = mid
                    left = mid + 1
                else:
                    right = mid - 1

            return last_readable
        finally:
            cap.release()

    def _resolve_time(self, idx: int | float | None, *, is_start: bool) -> float:
        """
        Convert index to absolute time in seconds, handling negative indices.

        Centralizes time/index resolution logic used throughout the class.

        Args:
            idx: Time/frame index (can be negative for end-relative)
            is_start: True if this is a start index, False for end

        Returns:
            Absolute time in seconds, clamped to segment bounds
        """
        if idx is None:
            return self.start_time if is_start else self.end_time

        # Convert to seconds based on time_unit
        idx_seconds = _to_seconds(idx, unit=self.time_unit, fps=self.fps)

        # Handle negative indices (from end of this segment)
        if idx_seconds < 0:
            idx_seconds = self.end_time + idx_seconds
        else:
            # Positive indices are relative to segment start
            idx_seconds = self.start_time + idx_seconds

        # Clamp to segment's valid range
        return max(self.start_time, min(idx_seconds, self.end_time))

    def _get_frame_at_time(self, time_seconds: float) -> np.ndarray:
        """Extract a single frame at the given time in seconds."""
        # Convert time to absolute frame index
        frame_idx = int(time_seconds * self.fps)
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))

        cap = cv2.VideoCapture(self.video_src)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            # Handle corrupted/unreadable frames at end of video
            if not ret:
                # Find last readable frame
                last_readable = self._find_last_readable_frame()

                if frame_idx > last_readable:
                    # Requested frame is beyond readable range, use last readable
                    cap.set(cv2.CAP_PROP_POS_FRAMES, last_readable)
                    ret, frame = cap.read()
                    if ret:
                        return frame

                # Still failed - raise error
                raise ValueError(
                    f"Failed to read frame at time {time_seconds}s (frame {frame_idx}). "
                    f"Last readable frame: {last_readable}"
                )

            return frame
        finally:
            cap.release()

    def __getitem__(self, key: int | float | slice) -> Union["Video", np.ndarray]:
        """
        Get a video segment or single frame using slice/index notation.

        Args:
            key: Integer/float index for single frame, or slice for time range

        Returns:
            Video for slice, numpy array (frame) for integer/float

        Examples:
            >>> video = Video("test.mp4")  # doctest: +SKIP
            >>> segment = video[10:20]  # 10s to 20s (Video)  # doctest: +SKIP
            >>> frame = video[15]  # Frame at 15 seconds (np.ndarray)  # doctest: +SKIP
            >>> ending = video[-30:]  # Last 30 seconds  # doctest: +SKIP
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported for video cropping")

            start = self._resolve_time(key.start, is_start=True)
            end = self._resolve_time(key.stop, is_start=False)

            if start >= end:
                raise ValueError(
                    f"Invalid time range: start ({start}s) must be before end ({end}s)"
                )

            # Return new Video instance
            return Video(
                self.src_path,
                time_unit=self.time_unit,
                start_time=start,
                end_time=end,
            )

        elif isinstance(key, (int, float)):
            # Single frame extraction
            # Convert index to absolute time
            idx_seconds = _to_seconds(key, unit=self.time_unit, fps=self.fps)

            # Handle negative indices
            if idx_seconds < 0:
                time_seconds = self.end_time + idx_seconds
            else:
                time_seconds = self.start_time + idx_seconds

            # Clamp to valid range
            time_seconds = max(self.start_time, min(time_seconds, self.end_time))

            return self._get_frame_at_time(time_seconds)

        else:
            raise TypeError(
                f"Video indexing requires int/float or slice, "
                f"got {type(key).__name__}"
            )

    def save(
        self,
        saveas: str | None = None,
        *,
        codec: str = "libx264",
        audio_codec: str = "aac",
        **write_kwargs,
    ) -> Path:
        """
        Save this video/segment to a new video file.

        Args:
            saveas: Path for output file (auto-generated if None)
            codec: Video codec to use
            audio_codec: Audio codec to use
            **write_kwargs: Additional arguments for write_videofile

        Returns:
            Path to saved file
        """
        if saveas is None:
            saveas = _auto_video_path(
                self.video_src, f"{int(self.start_time)}_{int(self.end_time)}"
            )

        output_path = _ensure_output_path(saveas)

        with mp.VideoFileClip(self.video_src) as clip:
            subclip = clip.subclipped(self.start_time, self.end_time)
            subclip.write_videofile(
                str(output_path), codec=codec, audio_codec=audio_codec, **write_kwargs
            )

        return output_path

    def save_frame(
        self,
        time_or_frame: float | None = None,
        saveas: str | None = None,
        *,
        image_format: str = "png",
        copy_to_clipboard: bool = False,
    ) -> Path | None:
        """
        Save a single frame as an image and/or copy to clipboard.

        Args:
            time_or_frame: Time/frame index (None = start of segment)
            saveas: Path for output image (auto-generated if None,
                        False = don't save to file)
            image_format: Image format (png, jpg, etc.)
            copy_to_clipboard: If True, copy image to system clipboard

        Returns:
            Path to saved image, or None if only copied to clipboard

        Examples:
            >>> video = Video("movie.mp4")  # doctest: +SKIP
            >>> video.save_frame(10.5)  # Save frame at 10.5s  # doctest: +SKIP
            >>> video.save_frame(10.5, copy_to_clipboard=True)  # Save and copy  # doctest: +SKIP
            >>> video.save_frame(10.5, saveas=False, copy_to_clipboard=True)  # Clipboard only  # doctest: +SKIP
        """
        if saveas is False and not copy_to_clipboard:
            raise ValueError(
                "Must specify at least one output: set saveas or copy_to_clipboard=True"
            )

        if time_or_frame is None:
            time_or_frame = self.start_time

        # Get the frame
        frame = self._get_frame_at_time(time_or_frame)

        # Copy to clipboard if requested
        if copy_to_clipboard:
            print("Copying image to clipboard...")
            _copy_frame_to_clipboard(frame)

        # Save to file if requested
        if saveas is not False:
            # Determine output path
            if saveas is None:
                frame_idx = int(time_or_frame * self.fps)
                saveas = _auto_frame_path(
                    self.video_src, frame_idx, image_format=image_format
                )
            else:
                saveas = Path(saveas)
                if not saveas.suffix:
                    saveas = saveas.with_suffix(f".{image_format}")

            # Ensure directory exists
            output_path = _ensure_output_path(saveas)

            # Save frame
            cv2.imwrite(str(output_path), frame)
            print(f"Saved frame to: {output_path}")

            return output_path

        return None

    def to_clip(self) -> mp.VideoFileClip:
        """
        Get a moviepy VideoFileClip for this segment.

        Note: Caller is responsible for closing the clip.
        """
        clip = mp.VideoFileClip(self.video_src)
        return clip.subclipped(self.start_time, self.end_time)

    def __repr__(self) -> str:
        if self._start_time is not None or self._end_time is not None:
            return (
                f"Video('{self.video_src}', "
                f"time_unit='{self.time_unit}', "
                f"start={self.start_time:.2f}s, "
                f"end={self.end_time:.2f}s, "
                f"duration={self.duration:.2f}s)"
            )
        else:
            return (
                f"Video('{self.video_src}', "
                f"time_unit='{self.time_unit}', "
                f"duration={self.full_duration:.2f}s)"
            )

    @property
    def frames(self) -> 'VideoFrames':
        """Get frame-by-frame Mapping interface for this video/segment."""
        return VideoFrames(
            self.video_src,
            start_frame=int(self.start_time * self.fps),
            end_frame=int(self.end_time * self.fps),
        )


def crop_video(
    video_src: str,
    start: float | int | None = None,
    end: float | int | None = None,
    *,
    time_unit: TimeUnit = "seconds",
    saveas: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Convenience function to crop and save a video segment or frame.

    Args:
        video_src: Path to source video
        start: Start time/frame (None = beginning)
        end: End time/frame (None = end of video)
        time_unit: Unit for start/end values
        saveas: Path for output (auto-generated if None or False)
        **save_kwargs: Additional arguments for save operation

    Returns:
        Path to saved cropped video

    Examples:
        >>> crop_video("video.mp4", 10, 30)  # Crop 10s-30s  # doctest: +SKIP
        >>> crop_video("video.mp4", 100, 500, time_unit="frames")  # doctest: +SKIP
        >>> crop_video("video.mp4", 10, 10)  # Single frame at 10s  # doctest: +SKIP
    """
    video = Video(video_src, time_unit=time_unit)

    # Handle single frame case
    if start is not None and end is not None and start == end:
        # Extract single frame
        return video.save_frame(time_or_frame=start, saveas=saveas, **save_kwargs)

    # Handle segment case
    segment = video[start:end]
    return segment.save(saveas, **save_kwargs)


def save_frame(
    video_src: str | None = None,
    time_or_frame: int | float = 0,
    *,
    time_unit: TimeUnit | None = None,
    saveas: str | bool | None = None,
    image_format: str = "png",
    copy_to_clipboard: bool = False,
) -> Path | None:
    """
    Extract and save a frame from a video file.

    Args:
        video_src: Path to the video file. If None, gets from clipboard.
        time_or_frame: Time/frame index of the frame to extract (default: 0)
        time_unit: Unit for time_or_frame ('seconds', 'frames', 'milliseconds').
            If None, defaults to 'seconds', unless time_or_frame is a negative integer,
            in which case it defaults to 'frames'.
        saveas: Where to save the image. If None or "", auto-generates path.
            - None or "": Auto-generate path based on video filename
            - Path starting with '.': Use as extension (e.g., '.jpg')
            - '/TMP': Save to temporary directory
            - Full filepath: Use as-is
            - False: Don't save to file (requires copy_to_clipboard=True)
        image_format: Default image format if not specified in saveas
        copy_to_clipboard: If True, copy image to system clipboard

    Returns:
        Path to the saved image file, or None if only copied to clipboard

    Examples:
        >>> save_frame("video.mp4")  # Saves frame 0  # doctest: +SKIP
        >>> save_frame("video.mp4", 10)  # Saves frame at 10s  # doctest: +SKIP
        >>> save_frame("video.mp4", -1)  # Saves last frame (frame-based)  # doctest: +SKIP
        >>> save_frame("video.mp4", 100, time_unit="frames")  # Frame 100  # doctest: +SKIP
        >>> save_frame("video.mp4", 5, saveas=".jpg")  # doctest: +SKIP
        >>> save_frame("video.mp4", 5, saveas="/TMP")  # doctest: +SKIP
        >>> save_frame(time_or_frame=3, copy_to_clipboard=True)  # From clipboard  # doctest: +SKIP
        >>> save_frame(time_or_frame=3, saveas=False, copy_to_clipboard=True)  # Clipboard only  # doctest: +SKIP
    """
    if saveas is False and not copy_to_clipboard:
        raise ValueError(
            "Must specify at least one output: set saveas or copy_to_clipboard=True"
        )

    if video_src is None:
        video_src = _get_video_path_from_clipboard()

    # Smart defaulting for time_unit
    if time_unit is None:
        if isinstance(time_or_frame, int) and time_or_frame < 0:
            time_unit = "frames"
        else:
            time_unit = "seconds"

    video = Video(video_src, time_unit=time_unit)

    # Resolve time using the Video's _resolve_time method
    time_seconds = video._resolve_time(time_or_frame, is_start=True)

    # Determine output path based on saveas parameter
    output_path = None
    if saveas is False:
        output_path = False
    elif saveas is None or saveas == "":
        # Auto-generate: video_dir/video_name_frameIdx.ext
        output_path = None  # Let Video.save_frame auto-generate
    elif saveas == "/TMP":
        # Save to temporary directory
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_src).stem
        frame_idx_int = int(time_seconds * video.fps)
        output_path = (
            Path(temp_dir) / f"{video_name}_{frame_idx_int:06d}.{image_format}"
        )
    elif saveas.startswith("."):
        # Extension provided: use it as format
        image_format = saveas[1:]
        output_path = None  # Let Video.save_frame auto-generate with new format
    else:
        # Full path provided
        output_path = saveas

    return video.save_frame(
        time_or_frame=time_seconds,
        saveas=output_path,
        image_format=image_format,
        copy_to_clipboard=copy_to_clipboard,
    )


def loop_video(
    video_src: str,
    n_loops: int = 2,
    *,
    saveas: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Create a video by looping/repeating another video.

    Args:
        video_src: Path to source video
        n_loops: Number of times to repeat the video
        saveas: Path for output (auto-generated if None)
        **save_kwargs: Additional arguments for video export

    Returns:
        Path to saved looped video

    Examples:
        >>> loop_video("intro.mp4", 3)  # Repeat 3 times  # doctest: +SKIP
        >>> loop_video("short_clip.mp4", 5, saveas="extended.mp4")  # doctest: +SKIP
    """
    if n_loops < 1:
        raise ValueError(f"n_loops must be at least 1, got {n_loops}")

    output_path = _resolve_output_path(video_src, saveas, f"loop{n_loops}")

    # Set default codecs
    _set_default_codecs(save_kwargs)

    # Load video clip
    with mp.VideoFileClip(video_src) as clip:
        # Create list of clips to concatenate
        clips = [clip] * n_loops

        # Concatenate
        looped = mp.concatenate_videoclips(clips)

        # Write output
        looped.write_videofile(str(output_path), **save_kwargs)

        # Clean up
        looped.close()

    print(f"Saved looped video to: {output_path}")
    return output_path


def replace_audio(
    video_src: str,
    audio_src: str,
    *,
    mix_ratio: float = 1.0,
    saveas: str | None = None,
    match_duration: bool = True,
    **save_kwargs,
) -> Path:
    """
    Replace or mix audio in a video with new audio.

    Args:
        video_src: Path to source video
        audio_src: Path to audio file to add/mix
        mix_ratio: Audio mixing ratio (0.0 = keep only original, 1.0 = replace completely,
                   0.5 = mix both equally). Values between 0 and 1 blend the audio tracks.
        saveas: Path for output (auto-generated if None)
        match_duration: If True, adjust audio duration to match video
        **save_kwargs: Additional arguments for video export

    Returns:
        Path to saved video with new/mixed audio

    Examples:
        >>> replace_audio("video.mp4", "music.mp3")  # Replace audio completely  # doctest: +SKIP
        >>> replace_audio("video.mp4", "bgm.mp3", mix_ratio=0.5)  # Equal mix  # doctest: +SKIP
        >>> replace_audio("video.mp4", "voice.mp3", mix_ratio=0.7)  # 70% new, 30% original  # doctest: +SKIP
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    audio_name = Path(audio_src).stem
    output_path = _resolve_output_path(video_src, saveas, f"audio_{audio_name}")

    # Set default codecs
    _set_default_codecs(save_kwargs)

    # Load video and audio
    with mp.VideoFileClip(video_src) as video_clip:
        # Load new audio
        with mp.AudioFileClip(audio_src) as new_audio:
            # Adjust audio duration if needed
            if match_duration and new_audio.duration != video_clip.duration:
                if new_audio.duration < video_clip.duration:
                    # Loop audio to match video length
                    n_loops = int(np.ceil(video_clip.duration / new_audio.duration))
                    new_audio = mp.concatenate_audioclips([new_audio] * n_loops)
                # Trim to exact video duration
                new_audio = new_audio.subclipped(0, video_clip.duration)

            # Handle audio mixing based on ratio
            if mix_ratio == 1.0:
                # Complete replacement
                final_clip = video_clip.with_audio(new_audio)
            elif mix_ratio == 0.0:
                # Keep original audio only
                if video_clip.audio is None:
                    # No original audio, add new audio anyway
                    final_clip = video_clip.with_audio(new_audio)
                else:
                    # Keep original - no change needed
                    final_clip = video_clip
            else:
                # Mix original and new audio
                if video_clip.audio is None:
                    # No original audio, just use new
                    final_clip = video_clip.with_audio(new_audio)
                else:
                    # Blend both audio tracks
                    # Adjust volumes: mix_ratio controls new audio prominence
                    original_volume = 1.0 - mix_ratio
                    new_volume = mix_ratio

                    # Apply volume adjustments using fx
                    from moviepy.audio.fx import MultiplyVolume

                    original_audio = video_clip.audio.with_effects(
                        [MultiplyVolume(original_volume)]
                    )
                    adjusted_new_audio = new_audio.with_effects(
                        [MultiplyVolume(new_volume)]
                    )

                    # Composite audio tracks
                    mixed_audio = mp.CompositeAudioClip(
                        [original_audio, adjusted_new_audio]
                    )
                    final_clip = video_clip.with_audio(mixed_audio)

            # Write output
            final_clip.write_videofile(str(output_path), **save_kwargs)

    print(f"Saved video with audio to: {output_path}")
    return output_path


def normalize_audio(
    video_src: str,
    *,
    saveas: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Normalize audio levels in a video to reduce volume fluctuations.

    This function adjusts the audio so that the loudest parts reach a consistent
    level, reducing the variation between quiet and loud sections. This is
    particularly useful for videos with narration that varies in volume.

    Args:
        video_src: Path to input video file
        saveas: Optional output path. If None, generates name with suffix
        **save_kwargs: Additional arguments for write_videofile (e.g., codec, audio_codec)

    Returns:
        Path to the output video file with normalized audio

    Examples:
        >>> # Normalize audio in a video with varying narrator volume
        >>> normalize_audio("lecture.mp4")  # doctest: +SKIP
        >>> # Output: lecture_normalized.mp4

        >>> # Specify custom output path
        >>> normalize_audio("interview.mp4", saveas="interview_fixed.mp4")  # doctest: +SKIP
    """
    output_path = _resolve_output_path(video_src, saveas, "normalized")

    # Set default codecs
    _set_default_codecs(save_kwargs)

    # Load video and normalize audio
    with mp.VideoFileClip(str(video_src)) as clip:
        if clip.audio is not None:
            # Apply audio normalization
            from moviepy.audio.fx import AudioNormalize

            normalized_audio = clip.audio.with_effects([AudioNormalize()])
            final_clip = clip.with_audio(normalized_audio)
        else:
            # No audio to normalize
            print(f"Warning: {video_src} has no audio track")
            final_clip = clip

        final_clip.write_videofile(str(output_path), **save_kwargs)

    print(f"Saved video with normalized audio to: {output_path}")
    return output_path


def change_speed(
    video_src: str,
    speed_factor: float,
    *,
    saveas: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Change the playback speed of a video.

    Creates a new video that plays faster or slower than the original while
    preserving audio pitch (audio is also sped up/slowed down proportionally).

    Args:
        video_src: Path to input video file
        speed_factor: Speed multiplier (e.g., 2.0 = 2x faster, 0.5 = half speed)
            - > 1.0: speeds up the video (e.g., 2.0 = twice as fast)
            - < 1.0: slows down the video (e.g., 0.5 = half speed)
            - 1.0: no change
        saveas: Optional output path. If None, generates name with speed suffix
        **save_kwargs: Additional arguments for write_videofile (e.g., codec, fps)

    Returns:
        Path to the output video file

    Examples:
        >>> # Create slow-motion video at half speed
        >>> change_speed("action.mp4", 0.5)  # doctest: +SKIP
        >>> # Output: action_speed_0.5x.mp4

        >>> # Speed up video 2x
        >>> change_speed("lecture.mp4", 2.0, saveas="fast_lecture.mp4")  # doctest: +SKIP

        >>> # Extreme slow motion
        >>> change_speed("jump.mp4", 0.25)  # doctest: +SKIP
        >>> # Output: jump_speed_0.25x.mp4
    """
    output_path = _resolve_output_path(video_src, saveas, f"speed_{speed_factor}x")

    # Set default codecs
    _set_default_codecs(save_kwargs)

    # Load video and change speed
    with mp.VideoFileClip(str(video_src)) as clip:
        # Apply speed change (this also affects audio)
        sped_clip = clip.with_speed_scaled(speed_factor)

        # Write output
        sped_clip.write_videofile(str(output_path), **save_kwargs)

        # Clean up
        sped_clip.close()

    print(f"Saved {speed_factor}x speed video to: {output_path}")
    return output_path


def _parse_rectangle(rect, default=(0.5, 0.5, 1.0)):
    """
    Normalize rectangle input to (cx, cy, s) tuple.
    Accepts:
        - None: returns default
        - single number: (0.5, 0.5, v)
        - pair: (cx, cy, 1.0)
        - triple: (cx, cy, s)
    """
    if rect is None:
        return default
    if isinstance(rect, (int, float)):
        return (0.5, 0.5, float(rect))
    if isinstance(rect, (list, tuple)):
        if len(rect) == 1:
            return (0.5, 0.5, float(rect[0]))
        elif len(rect) == 2:
            return (float(rect[0]), float(rect[1]), 1.0)
        elif len(rect) == 3:
            return (float(rect[0]), float(rect[1]), float(rect[2]))
    raise ValueError(f"Invalid rectangle: {rect}")


def _rect_to_box(cx, cy, s, img_w, img_h):
    """
    Convert (cx, cy, s) to pixel bounding box (xmin, ymin, xmax, ymax) in image coordinates.
    """
    w = 1.0 / s
    h = 1.0 / s
    xmin = (cx - w / 2) * img_w
    ymin = (cy - h / 2) * img_h
    xmax = (cx + w / 2) * img_w
    ymax = (cy + h / 2) * img_h
    # Clamp to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def ken_burns_video(
    image,
    *,
    duration_s: float = 3,
    fps: int = 30,
    start_rectangle=None,
    end_rectangle=None,
    saveas: str | None = None,
    codec: str = "libx264",
    audio_codec: str = "aac",
    **write_kwargs,
) -> Path:
    """
    Create a Ken Burns effect video from an image.

    Args:
        image: Path to image file or image object (PIL.Image, np.ndarray)
        duration_s: Duration of output video in seconds
        fps: Frames per second (default 30)
        start_rectangle: Ken Burns rect at start (see doc)
        end_rectangle: Ken Burns rect at end (see doc)
        saveas: Where to save video (default: image path with .mp4 extension,
            auto-incremented if exists)
        codec: Video codec (default libx264)
        audio_codec: Audio codec (default aac)
        **write_kwargs: Passed to write_videofile

    Returns:
        Path to saved video

    Rectangle parameterization:
        Rect = (cx, cy, s)
        cx, cy in [0, 1], s > 0
        s = 1: original size, <1: zoomed out, >1: zoomed in
        See module doc for details.

    Examples:
        >>> ken_burns_video("photo.jpg", duration_s=5, start_rectangle=1.5, end_rectangle=1.0)  # doctest: +SKIP
        >>> ken_burns_video("photo.jpg", duration_s=3, start_rectangle=(0.3, 0.3, 2), end_rectangle=(0.7, 0.7, 2))  # doctest: +SKIP
    """
    from dol import non_colliding_key

    # Accept image as path, PIL.Image, or np.ndarray
    PIL_Image = require_package('PIL.Image')

    # Track the original image path for output path generation
    image_path = None
    if isinstance(image, str) or isinstance(image, Path):
        image_path = Path(image)
        img = PIL_Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = PIL_Image.fromarray(image)
    elif hasattr(image, 'convert'):
        img = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    img_w, img_h = img.size
    n_frames = int(duration_s * fps)

    # Parse rectangles
    start_rect = _parse_rectangle(start_rectangle)
    end_rect = _parse_rectangle(end_rectangle)

    # Interpolate rectangles for each frame
    def lerp(a, b, t):
        return a + (b - a) * t

    rects = [
        (
            lerp(start_rect[0], end_rect[0], i / (n_frames - 1)),
            lerp(start_rect[1], end_rect[1], i / (n_frames - 1)),
            lerp(start_rect[2], end_rect[2], i / (n_frames - 1)),
        )
        for i in range(n_frames)
    ]

    # Preload as numpy array for fast cropping
    img_np = np.array(img)

    frames = []
    for cx, cy, s in rects:
        xmin, ymin, xmax, ymax = _rect_to_box(cx, cy, s, img_w, img_h)
        crop = img_np[ymin:ymax, xmin:xmax]
        # Resize to original size
        crop_img = PIL_Image.fromarray(crop).resize(
            (img_w, img_h), resample=PIL_Image.BICUBIC
        )
        frames.append(np.array(crop_img))

    # Determine output path
    if saveas is None:
        if image_path is not None:
            # Use image path with .mp4 extension
            output_path = image_path.with_suffix('.mp4')
        else:
            # Fallback to temp directory
            output_path = Path(tempfile.gettempdir()) / f"kenburns_{os.getpid()}.mp4"
    else:
        output_path = Path(saveas)
        # Ensure it has a video extension
        if not output_path.suffix or output_path.suffix.lower() not in [
            '.mp4',
            '.mov',
            '.avi',
            '.mkv',
        ]:
            output_path = output_path.with_suffix('.mp4')

    # Use non_colliding_key to avoid overwriting
    directory = output_path.parent
    filename = output_path.name
    try:
        existing_files = (
            set(os.listdir(directory)) if directory else set(os.listdir('.'))
        )
    except OSError:
        existing_files = set()

    if filename in existing_files:
        safe_filename = non_colliding_key(filename, existing_files)
        output_path = directory / safe_filename

    # Create video using moviepy with proper settings for compatibility
    clip = mp.ImageSequenceClip(frames, fps=fps)

    # Set default kwargs for better compatibility
    write_kwargs.setdefault('bitrate', '5000k')
    write_kwargs.setdefault('preset', 'medium')
    write_kwargs.setdefault('logger', None)  # Suppress verbose output

    clip.write_videofile(
        str(output_path), codec=codec, audio_codec=audio_codec, **write_kwargs
    )
    clip.close()

    print(f"Saved Ken Burns video to: {output_path}")
    return output_path
