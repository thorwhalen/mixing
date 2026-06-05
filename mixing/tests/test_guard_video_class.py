"""Characterization tests pinning the CURRENT behavior of ``Video`` / ``VideoFrames``.

These guard-rail tests lock in the observable semantics of
``mixing.video.video_ops.Video`` and ``mixing.video.video_ops.VideoFrames`` so an
upcoming refactor cannot silently change them. They pin *behavior* (construction,
time-unit slicing, single-frame indexing, negative indices, properties, ``save`` /
``save_frame`` / ``to_clip``, and the ``.frames`` Mapping), not implementation
details.

Everything runs offline against synthetic clips from the shared
``make_color_video`` / ``color_video`` conftest fixtures. ``cv2`` and ``moviepy``
are required; tests skip cleanly when those backends are missing. Durations and
pixel values use tolerance-based assertions because media encoders are not exact.
"""

import os
import tempfile
from pathlib import Path

import pytest

# These backends are mandatory for the Video class; skip the whole module if absent.
pytest.importorskip("cv2")
pytest.importorskip("moviepy")
pytest.importorskip("numpy")

import numpy as np

from mixing.video.video_ops import Video, VideoFrames


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #

DUR_TOL = 0.2  # seconds; media encoders aren't frame-exact


def _tmp_out(suffix: str) -> Path:
    """Return an unused temp path with the given suffix (file not created)."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    p = Path(name)
    p.unlink(missing_ok=True)  # we want the path, not the (empty) file
    return p


# --------------------------------------------------------------------------- #
# Construction & properties
# --------------------------------------------------------------------------- #


def test_construction_accepts_path_and_stores_str_src(color_video):
    """Video(path) constructs; video_src is coerced to str; defaults are unset."""
    v = Video(color_video)
    assert isinstance(v.video_src, str)
    assert v.video_src == str(color_video)
    assert v.time_unit == "seconds"
    # Unset start/end => full clip view.
    assert v.start_time == 0.0


def test_properties_full_clip(make_color_video):
    """duration/fps/frame_count/start_time/end_time/full_duration on a full clip."""
    path = make_color_video(2.0, fps=24)
    v = Video(path)

    assert abs(v.full_duration - 2.0) < DUR_TOL
    assert abs(v.fps - 24.0) < 0.5
    # 2s * 24fps ~ 48 frames (allow encoder slack).
    assert abs(v.frame_count - 48) <= 2
    assert v.start_time == 0.0
    assert abs(v.end_time - v.full_duration) < 1e-9
    # On a full (unsliced) clip duration == full_duration.
    assert abs(v.duration - v.full_duration) < 1e-9


def test_fps_and_duration_are_cached_after_first_access(color_video):
    """Property values are memoized (private cache fields populated on access)."""
    v = Video(color_video)
    assert v._fps is None and v._duration is None and v._frame_count is None
    _ = v.fps
    _ = v.full_duration
    _ = v.frame_count
    assert v._fps is not None
    assert v._duration is not None
    assert v._frame_count is not None


# --------------------------------------------------------------------------- #
# Time-unit slicing  ->  sub-clip Video
# --------------------------------------------------------------------------- #


def test_slice_returns_subclip_video_with_right_duration(make_color_video):
    """Video[start:end] (seconds) yields a Video sub-view with the sliced duration."""
    v = Video(make_color_video(2.0, fps=24))
    seg = v[0.5:1.5]

    assert isinstance(seg, Video)
    assert seg is not v
    assert abs(seg.start_time - 0.5) < 1e-9
    assert abs(seg.end_time - 1.5) < 1e-9
    assert abs(seg.duration - 1.0) < DUR_TOL
    # The sub-view points at the same source file.
    assert seg.video_src == v.video_src


def test_slice_open_ended_start_and_stop(make_color_video):
    """Open-ended slices default to start-of-clip / end-of-clip."""
    v = Video(make_color_video(2.0, fps=24))

    head = v[:1.0]
    assert abs(head.start_time - 0.0) < 1e-9
    assert abs(head.end_time - 1.0) < 1e-9

    tail = v[1.0:]
    assert abs(tail.start_time - 1.0) < 1e-9
    assert abs(tail.end_time - v.full_duration) < DUR_TOL


def test_slice_negative_stop_is_relative_to_end(make_color_video):
    """A negative slice bound counts back from the end of the (sub)clip."""
    v = Video(make_color_video(2.0, fps=24))
    seg = v[0:-0.5]  # last 0.5s trimmed off
    assert abs(seg.start_time - 0.0) < 1e-9
    assert abs(seg.end_time - 1.5) < DUR_TOL


def test_slice_with_step_raises_value_error(color_video):
    """Slicing with a step is rejected (cropping has no notion of stride)."""
    v = Video(color_video)
    with pytest.raises(ValueError, match="Step is not supported"):
        _ = v[0:1:2]


def test_slice_inverted_range_raises_value_error(make_color_video):
    """start >= end after resolution is an error."""
    v = Video(make_color_video(2.0, fps=24))
    with pytest.raises(ValueError, match="Invalid time range"):
        _ = v[1.5:0.5]


def test_frames_time_unit_slicing(make_color_video):
    """time_unit='frames' interprets slice indices as frame numbers."""
    path = make_color_video(2.0, fps=24)
    v = Video(path, time_unit="frames")
    # frames 0..24 == seconds 0..1 at 24fps
    seg = v[0:24]
    assert isinstance(seg, Video)
    assert abs(seg.start_time - 0.0) < 1e-9
    assert abs(seg.end_time - 1.0) < DUR_TOL
    assert abs(seg.duration - 1.0) < DUR_TOL


# --------------------------------------------------------------------------- #
# Single-frame indexing  ->  numpy BGR array
# --------------------------------------------------------------------------- #


def test_single_index_returns_bgr_frame_array(make_color_video):
    """Video[t] returns one frame: uint8 ndarray shaped (H, W, 3), cv2 BGR ordering.

    The conftest ``color=(255, 0, 0)`` is RGB red. cv2 (OpenCV) reads frames back
    as BGR-ordered arrays, so the source red intensity lands in channel index 2
    while channels 0 (blue) and 1 (green) stay near zero. This pins the BGR
    convention the rest of the package relies on.
    """
    path = make_color_video(1.0, fps=24, size=(320, 240), color=(255, 0, 0))
    v = Video(path)
    frame = v[0.0]

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.shape == (240, 320, 3)  # (height, width, channels)

    b, g, r = frame[0, 0].tolist()
    # BGR: red sits at index 2 (r); blue (0) and green (1) are near zero.
    assert r > 200
    assert b < 60
    assert g < 60


def test_single_index_negative_returns_frame(make_color_video):
    """Negative index resolves from the end and returns a valid frame array."""
    path = make_color_video(1.0, fps=24)
    v = Video(path)
    last = v[-1]
    assert isinstance(last, np.ndarray)
    assert last.shape == (240, 320, 3)


def test_index_bad_type_raises_type_error(color_video):
    """Non int/float/slice index is a TypeError."""
    v = Video(color_video)
    with pytest.raises(TypeError):
        _ = v["not-an-index"]


# --------------------------------------------------------------------------- #
# save / save_frame / to_clip
# --------------------------------------------------------------------------- #


def test_save_writes_subclip_of_right_duration(make_color_video):
    """.save(output=...) writes a file; the result has the sub-clip's duration."""
    v = Video(make_color_video(2.0, fps=24))
    seg = v[0.5:1.5]
    out = _tmp_out(".mp4")
    try:
        returned = seg.save(output=str(out))
        assert isinstance(returned, Path)
        assert returned.exists()
        # Round-trip the saved file and confirm its duration.
        saved = Video(str(returned))
        assert abs(saved.full_duration - 1.0) < DUR_TOL
    finally:
        out.unlink(missing_ok=True)


def test_save_full_clip_preserves_duration(make_color_video):
    """Saving an unsliced Video preserves the source duration."""
    src = make_color_video(1.0, fps=24)
    v = Video(src)
    out = _tmp_out(".mp4")
    try:
        returned = v.save(output=str(out))
        assert returned.exists()
        assert abs(Video(str(returned)).full_duration - 1.0) < DUR_TOL
    finally:
        out.unlink(missing_ok=True)


def test_save_frame_writes_image_file(make_color_video):
    """.save_frame(t, output=...) writes an image of the frame's (H, W, 3) size."""
    pytest.importorskip("cv2")
    import cv2

    path = make_color_video(1.0, fps=24, size=(320, 240))
    v = Video(path)
    out = _tmp_out(".png")
    try:
        returned = v.save_frame(0.0, output=str(out))
        assert isinstance(returned, Path)
        assert returned.exists()
        img = cv2.imread(str(returned))
        assert img is not None
        assert img.shape == (240, 320, 3)
    finally:
        out.unlink(missing_ok=True)


def test_save_frame_requires_an_output_target(color_video):
    """save_frame with output=False and no clipboard is a ValueError."""
    v = Video(color_video)
    with pytest.raises(ValueError, match="at least one output"):
        v.save_frame(0.0, output=False, copy_to_clipboard=False)


def test_to_clip_returns_moviepy_clip_of_subduration(make_color_video):
    """.to_clip() returns a moviepy clip spanning the (sub)segment duration."""
    v = Video(make_color_video(2.0, fps=24))
    seg = v[0.5:1.5]
    clip = seg.to_clip()
    try:
        assert abs(clip.duration - 1.0) < DUR_TOL
    finally:
        clip.close()


# --------------------------------------------------------------------------- #
# .frames  ->  VideoFrames Mapping
# --------------------------------------------------------------------------- #


def test_frames_property_returns_videoframes_mapping(make_color_video):
    """.frames is a VideoFrames Mapping spanning the whole clip by default."""
    from collections.abc import Mapping

    path = make_color_video(1.0, fps=24)
    v = Video(path)
    frames = v.frames

    assert isinstance(frames, VideoFrames)
    assert isinstance(frames, Mapping)
    assert frames.start_frame == 0
    # len == number of frames in the view (~fps * duration).
    assert abs(len(frames) - 24) <= 2


def test_frames_getitem_returns_bgr_frame(make_color_video):
    """VideoFrames[i] returns a single (H, W, 3) uint8 BGR frame."""
    path = make_color_video(1.0, fps=24, size=(320, 240))
    v = Video(path)
    frames = v.frames

    first = frames[0]
    assert isinstance(first, np.ndarray)
    assert first.dtype == np.uint8
    assert first.shape == (240, 320, 3)


def test_frames_negative_index(make_color_video):
    """VideoFrames supports negative indexing (last frame)."""
    path = make_color_video(1.0, fps=24)
    frames = Video(path).frames
    last = frames[-1]
    assert isinstance(last, np.ndarray)
    assert last.shape == (240, 320, 3)


def test_frames_slicing_yields_iterator_of_frames(make_color_video):
    """VideoFrames[a:b] returns an iterator of frames (not a list/Mapping)."""
    from collections.abc import Iterator

    path = make_color_video(1.0, fps=24)
    frames = Video(path).frames

    sliced = frames[0:3]
    assert isinstance(sliced, Iterator)
    materialized = list(sliced)
    assert len(materialized) == 3
    for fr in materialized:
        assert isinstance(fr, np.ndarray)
        assert fr.shape == (240, 320, 3)


def test_frames_index_out_of_range_raises(make_color_video):
    """An integer index past the end raises IndexError."""
    path = make_color_video(1.0, fps=24)
    frames = Video(path).frames
    with pytest.raises(IndexError):
        _ = frames[len(frames) + 1000]


def test_subsegment_frames_offsets_into_source(make_color_video):
    """A sub-segment's .frames window is offset into the source frame range."""
    path = make_color_video(2.0, fps=24)
    seg = Video(path)[0.5:1.5]
    frames = seg.frames

    assert isinstance(frames, VideoFrames)
    # start_frame == int(start_time * fps) == int(0.5 * 24) == 12
    assert frames.start_frame == 12
    # ~24 frames in a 1-second window at 24fps.
    assert abs(len(frames) - 24) <= 2


def test_videoframes_direct_construction_open_video(make_color_video):
    """VideoFrames(path) opens the file and spans all frames by default."""
    path = make_color_video(1.0, fps=24)
    vf = VideoFrames(str(path))
    assert vf.start_frame == 0
    assert abs(len(vf) - 24) <= 2


def test_videoframes_bad_path_raises_value_error():
    """Opening a non-existent video raises ValueError at construction."""
    with pytest.raises(ValueError, match="Cannot open video"):
        VideoFrames("/no/such/video/file/definitely_missing.mp4")
