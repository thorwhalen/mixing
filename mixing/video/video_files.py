"""
Video frame access via Mapping interface.

"""

import io
import os
from typing import Union
from collections.abc import Iterator
from collections.abc import Mapping
import cv2
import numpy as np
import tempfile


# TODO: Use i2.castgraph to make video_src input flexible
class VideoFrames(Mapping[int, np.ndarray]):
    """
    Mapping interface to access video frames by index.

    Provides dictionary-like access to video frames with support for negative
    indexing and slicing. Frames are returned as numpy arrays (BGR format).

    Args:
        video_src: Path to the video file

    Examples:

        >>> import tempfile
        >>> # Create a small test video (requires actual video file to test)
        >>> vf = VideoFrames("test_video.mp4")  # doctest: +SKIP
        >>> frame = vf[0]  # Get first frame  # doctest: +SKIP
        >>> last_frame = vf[-1]  # Get last frame  # doctest: +SKIP
        >>> frames = list(vf[10:20])  # Get frames 10-19  # doctest: +SKIP

    """

    def __init__(self, video_src: str):
        self.video_src = video_src
        self._cap = cv2.VideoCapture(video_src)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_src}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap.release()

    def __len__(self) -> int:
        """Return total number of frames in the video."""
        return self._frame_count

    def __iter__(self) -> Iterator[int]:
        """Iterate over all frame indices."""
        return iter(range(self._frame_count))

    def _normalize_index(self, idx: int) -> int:
        """Convert negative indices to positive."""
        if idx < 0:
            idx = self._frame_count + idx
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        return idx

    def _read_frame_at_index(self, idx: int) -> np.ndarray:
        """Read a single frame at the given index."""
        idx = self._normalize_index(idx)
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
        # Normalize indices
        if start < 0:
            start = max(0, self._frame_count + start)
        if stop < 0:
            stop = max(0, self._frame_count + stop)

        start = max(0, min(start, self._frame_count))
        stop = max(0, min(stop, self._frame_count))

        if step == 1:
            # Efficient sequential reading
            cap = cv2.VideoCapture(self.video_src)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, stop):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
            finally:
                cap.release()
        else:
            # Random access for non-sequential steps
            for idx in range(start, stop, step):
                yield self._read_frame_at_index(idx)

    def __getitem__(self, key: int | slice) -> np.ndarray | Iterator[np.ndarray]:
        """
        Get frame(s) by index or slice.

        Args:
            key: Integer index or slice object

        Returns:
            Single frame (np.ndarray) for integer index,
            or iterator of frames for slice
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self._frame_count)
            return self._iterate_frames(start, stop, step or 1)
        elif isinstance(key, int):
            return self._read_frame_at_index(key)
        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )


# TODO: Use i2.castgraph to make video_src input flexible
def save_frame(
    video_src: str | None = None,
    frame_idx: int = 0,
    *,
    saveas: str | bool | None = None,
    image_format: str = "png",
    copy_to_clipboard: bool = False,
):
    """
    Extract and save a frame from a video file.

    Args:
        video_src: Path to the video file. If None, will attempt to get from clipboard.
        frame_idx: Index of the frame to extract (default: 0)
        saveas: Where to save the image. If None or "", auto-generates path.
            - None or "": Auto-generate path based on video filename
            - Path starting with '.': Use as extension (e.g., '.jpg')
            - '/TMP': Save to temporary directory
            - Full filepath: Use as-is
            - False: Don't save to file (requires copy_to_clipboard=True)
        image_format: Default image format if not specified in saveas (default: 'png')
        copy_to_clipboard: If True, copy image to system clipboard as pasteable image

    Returns:
        Path to the saved image file, or None if only copied to clipboard

    Examples:

        >>> save_frame("video.mp4")  # Saves as video_000000.png  # doctest: +SKIP
        >>> save_frame("video.mp4", 10, saveas=".jpg")  # Saves as video_000010.jpg  # doctest: +SKIP
        >>> save_frame("video.mp4", -1, saveas="last_frame.png")  # doctest: +SKIP
        >>> save_frame("video.mp4", 5, saveas="/TMP")  # Saves to temp dir  # doctest: +SKIP
        >>> save_frame(frame_idx=3, copy_to_clipboard=True)  # From clipboard to file and clipboard  # doctest: +SKIP
        >>> save_frame(frame_idx=3, saveas=False, copy_to_clipboard=True)  # Clipboard only  # doctest: +SKIP

    """
    if saveas is False and not copy_to_clipboard:
        raise ValueError(
            "Must specify at least one output: set saveas or copy_to_clipboard=True"
        )

    if video_src is None:
        video_src = _get_video_path_from_clipboard()

    # Get the frame
    frames = VideoFrames(video_src)
    frame = frames[frame_idx]

    if copy_to_clipboard:
        print(f"Copying image to clipboard...")
        _copy_frame_to_clipboard(frame)

    if saveas is not False:
        output_path = _determine_output_path(video_src, frame_idx, saveas, image_format)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Saved frame to: {output_path}")

        return output_path

    return None


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


def _determine_output_path(
    video_src: str,
    frame_idx: int,
    saveas: str | None,
    image_format: str,
) -> str:
    """
    Determine the output path for saving a frame.

    >>> _determine_output_path("/path/to/video.mp4", 5, None, "png")  # doctest: +SKIP
    '/path/to/video_000005.png'
    """
    video_dir = os.path.dirname(os.path.abspath(video_src))
    video_name = os.path.splitext(os.path.basename(video_src))[0]

    if saveas is None or saveas == "":
        # Auto-generate: video_dir/video_name_frameIdx.ext
        return os.path.join(video_dir, f"{video_name}_{frame_idx:06d}.{image_format}")

    elif saveas == "/TMP":
        # Save to temporary directory
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, f"{video_name}_{frame_idx:06d}.{image_format}")

    elif saveas.startswith("."):
        # Extension provided: video_dir/video_name_frameIdx.ext
        ext = saveas[1:]  # Remove the leading dot
        return os.path.join(video_dir, f"{video_name}_{frame_idx:06d}.{ext}")

    else:
        # Full path provided
        output_path = saveas
        ext = os.path.splitext(output_path)[1][1:]  # Get extension without dot
        if not ext:
            output_path = f"{output_path}.{image_format}"
        return output_path


def _copy_frame_to_clipboard(frame: np.ndarray) -> None:
    """
    Copy a cv2 frame (numpy array) to system clipboard as a pasteable image.

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


if __name__ == "__main__":
    import argh

    argh.dispatch_commands([save_frame])
