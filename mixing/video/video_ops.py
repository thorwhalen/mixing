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

from typing import Union, Literal
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping
import io
import os
import tempfile
import numpy as np
import cv2
import moviepy as mp


TimeUnit = Literal["seconds", "frames", "milliseconds"]


def _to_seconds(value: float, *, unit: TimeUnit, fps: float) -> float:
    """Convert time value to seconds based on unit."""
    if unit == "seconds":
        return value
    elif unit == "frames":
        return value / fps
    elif unit == "milliseconds":
        return value / 1000.0
    else:
        raise ValueError(f"Invalid time unit: {unit}")


def require_package(package_name: str):
    """
    Import a package, raising an informative error if not installed.

    >>> math = require_package('math')  # doctest: +SKIP
    >>> math.pi  # doctest: +SKIP
    3.141592653589793
    """
    try:
        import importlib

        return importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(
            f"Package '{package_name}' is required for this functionality. "
            f"Please install it via 'pip install {package_name}'."
        ) from e


def _get_video_path_from_clipboard() -> str:
    """Get video file path from clipboard and validate it."""
    print("Getting video source from clipboard...")
    clipboard_content = require_package('pyclip').paste()

    # Validate it's text, not binary data
    if isinstance(clipboard_content, bytes):
        try:
            video_src = clipboard_content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(
                "Clipboard contains binary data that cannot be decoded as text. "
                "Expected a file path string."
            )
    elif isinstance(clipboard_content, str):
        video_src = clipboard_content
    else:
        raise ValueError(
            f"Clipboard content is not a valid string or bytes. "
            f"Got {type(clipboard_content).__name__}"
        )

    # Clean up the path
    video_src = os.path.expanduser(video_src.strip())

    # Validate it's a file path
    if not os.path.isfile(video_src):
        # Truncate long strings to avoid printing huge binary garbage
        display_content = video_src if len(video_src) < 100 else video_src[:100] + '...'
        raise ValueError(
            f"Clipboard content is not a valid (existing) file path: {display_content}"
        )

    print(f"... Video source: {video_src}")
    return video_src


def _copy_frame_to_clipboard(frame: np.ndarray) -> None:
    """
    Copy a cv2 frame (numpy array) to system clipboard as pasteable image.

    Requires: pillow, pyclip
    """
    pyclip = require_package('pyclip')
    PIL_Image = require_package('PIL.Image')

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = PIL_Image.fromarray(rgb_frame)

    # Convert PIL Image to bytes for clipboard
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()

    pyclip.copy(image_bytes)


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
        src_path: Path to source video file
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
        src_path: str,
        *,
        time_unit: TimeUnit = "seconds",
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        self.src_path = str(src_path)
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
            with mp.VideoFileClip(self.src_path) as clip:
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
            with mp.VideoFileClip(self.src_path) as clip:
                self._fps = clip.fps
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in source video."""
        if self._frame_count is None:
            cap = cv2.VideoCapture(self.src_path)
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self._frame_count

    def _normalize_index(self, idx: int | float | None, is_start: bool) -> float:
        """Convert slice index to seconds, handling None and negative indices."""
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

        cap = cv2.VideoCapture(self.src_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(
                    f"Failed to read frame at time {time_seconds}s (frame {frame_idx})"
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

            start = self._normalize_index(key.start, is_start=True)
            end = self._normalize_index(key.stop, is_start=False)

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
        output_path: str | None = None,
        *,
        codec: str = "libx264",
        audio_codec: str = "aac",
        **write_kwargs,
    ) -> Path:
        """
        Save this video/segment to a new video file.

        Args:
            output_path: Path for output file (auto-generated if None)
            codec: Video codec to use
            audio_codec: Audio codec to use
            **write_kwargs: Additional arguments for write_videofile

        Returns:
            Path to saved file
        """
        if output_path is None:
            src = Path(self.src_path)
            output_path = src.with_stem(
                f"{src.stem}_{int(self.start_time)}_{int(self.end_time)}"
            )

        output_path = Path(output_path)

        with mp.VideoFileClip(self.src_path) as clip:
            subclip = clip.subclipped(self.start_time, self.end_time)
            subclip.write_videofile(
                str(output_path), codec=codec, audio_codec=audio_codec, **write_kwargs
            )

        return output_path

    def save_frame(
        self,
        frame_time: float | None = None,
        output_path: str | None = None,
        *,
        image_format: str = "png",
        copy_to_clipboard: bool = False,
    ) -> Path | None:
        """
        Save a single frame as an image and/or copy to clipboard.

        Args:
            frame_time: Time in seconds (None = start of segment)
            output_path: Path for output image (auto-generated if None,
                        False = don't save to file)
            image_format: Image format (png, jpg, etc.)
            copy_to_clipboard: If True, copy image to system clipboard

        Returns:
            Path to saved image, or None if only copied to clipboard

        Examples:
            >>> video = Video("movie.mp4")  # doctest: +SKIP
            >>> video.save_frame(10.5)  # Save frame at 10.5s  # doctest: +SKIP
            >>> video.save_frame(10.5, copy_to_clipboard=True)  # Save and copy  # doctest: +SKIP
            >>> video.save_frame(10.5, output_path=False, copy_to_clipboard=True)  # Clipboard only  # doctest: +SKIP
        """
        if output_path is False and not copy_to_clipboard:
            raise ValueError(
                "Must specify at least one output: set output_path or copy_to_clipboard=True"
            )

        if frame_time is None:
            frame_time = self.start_time

        # Get the frame
        frame = self._get_frame_at_time(frame_time)

        # Copy to clipboard if requested
        if copy_to_clipboard:
            print("Copying image to clipboard...")
            _copy_frame_to_clipboard(frame)

        # Save to file if requested
        if output_path is not False:
            # Determine output path
            if output_path is None:
                src = Path(self.src_path)
                frame_idx = int(frame_time * self.fps)
                output_path = src.parent / f"{src.stem}_{frame_idx:06d}.{image_format}"
            else:
                output_path = Path(output_path)
                if not output_path.suffix:
                    output_path = output_path.with_suffix(f".{image_format}")

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

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
        clip = mp.VideoFileClip(self.src_path)
        return clip.subclipped(self.start_time, self.end_time)

    def __repr__(self) -> str:
        if self._start_time is not None or self._end_time is not None:
            return (
                f"Video('{self.src_path}', "
                f"time_unit='{self.time_unit}', "
                f"start={self.start_time:.2f}s, "
                f"end={self.end_time:.2f}s, "
                f"duration={self.duration:.2f}s)"
            )
        else:
            return (
                f"Video('{self.src_path}', "
                f"time_unit='{self.time_unit}', "
                f"duration={self.full_duration:.2f}s)"
            )

    @property
    def frames(self) -> 'VideoFrames':
        """Get frame-by-frame Mapping interface for this video/segment."""
        return VideoFrames(
            self.src_path,
            start_frame=int(self.start_time * self.fps),
            end_frame=int(self.end_time * self.fps),
        )


def crop_video(
    src_path: str,
    start: float | int | None = None,
    end: float | int | None = None,
    *,
    time_unit: TimeUnit = "seconds",
    output_path: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Convenience function to crop and save a video segment or frame.

    Args:
        src_path: Path to source video
        start: Start time/frame (None = beginning)
        end: End time/frame (None = end of video)
        time_unit: Unit for start/end values
        output_path: Path for output (auto-generated if None)
        **save_kwargs: Additional arguments for save operation

    Returns:
        Path to saved cropped video

    Examples:
        >>> crop_video("video.mp4", 10, 30)  # Crop 10s-30s  # doctest: +SKIP
        >>> crop_video("video.mp4", 100, 500, time_unit="frames")  # doctest: +SKIP
        >>> crop_video("video.mp4", 10, 10)  # Single frame at 10s  # doctest: +SKIP
    """
    video = Video(src_path, time_unit=time_unit)

    # Handle single frame case
    if start is not None and end is not None and start == end:
        # Extract single frame
        if output_path is None:
            # Auto-generate image path
            return video.save_frame(frame_time=start, **save_kwargs)
        else:
            return video.save_frame(
                frame_time=start, output_path=output_path, **save_kwargs
            )

    # Handle segment case
    segment = video[start:end]
    return segment.save(output_path, **save_kwargs)


def save_frame(
    video_src: str | None = None,
    frame_idx: int | float = 0,
    *,
    time_unit: TimeUnit | None = None,
    saveas: str | bool | None = None,
    image_format: str = "png",
    copy_to_clipboard: bool = False,
) -> Path | None:
    """
    Extract and save a frame from a video file.

    Backward compatibility wrapper with full feature parity to original.

    Args:
        video_src: Path to the video file. If None, gets from clipboard.
        frame_idx: Index/time of the frame to extract (default: 0)
        time_unit: Unit for frame_idx ('seconds', 'frames', 'milliseconds').
            If None, defaults to 'seconds', unless frame_idx is a negative integer,
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
        >>> save_frame(frame_idx=3, copy_to_clipboard=True)  # From clipboard  # doctest: +SKIP
        >>> save_frame(frame_idx=3, saveas=False, copy_to_clipboard=True)  # Clipboard only  # doctest: +SKIP
    """
    if saveas is False and not copy_to_clipboard:
        raise ValueError(
            "Must specify at least one output: set saveas or copy_to_clipboard=True"
        )

    if video_src is None:
        video_src = _get_video_path_from_clipboard()

    # Smart defaulting for time_unit
    if time_unit is None:
        if isinstance(frame_idx, int) and frame_idx < 0:
            time_unit = "frames"
        else:
            time_unit = "seconds"

    video = Video(video_src, time_unit=time_unit)

    # Handle negative indices (from end)
    if frame_idx < 0:
        if time_unit == "frames":
            # Negative frame index: count from end
            frame_time = (video.frame_count + frame_idx) / video.fps
        else:
            # Negative time: subtract from duration
            frame_time = video.full_duration + frame_idx
    else:
        # Positive index: convert to seconds
        frame_time = _to_seconds(frame_idx, unit=time_unit, fps=video.fps)

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
        frame_idx_int = int(frame_time * video.fps)
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
        frame_time=frame_time,
        output_path=output_path,
        image_format=image_format,
        copy_to_clipboard=copy_to_clipboard,
    )
