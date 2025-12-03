"""
Tests for new video operations (loop_video and replace_audio).

These tests verify:
- loop_video: Video looping/repetition functionality
- replace_audio: Audio replacement and mixing in videos
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_video_file():
    """Create a sample video file for testing."""
    try:
        import moviepy as mp
        from moviepy.video.VideoClip import ColorClip
    except ImportError:
        pytest.skip("moviepy not installed")

    # Create a simple 1-second red video clip
    clip = ColorClip(size=(320, 240), color=(255, 0, 0), duration=1.0)
    clip = clip.with_fps(24)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name

    clip.write_videofile(temp_path, codec='libx264', audio=False, logger=None)
    clip.close()

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        pytest.skip("pydub not installed")

    # Create 1 second of audio
    audio = Sine(440).to_audio_segment(duration=1000, volume=-20)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        temp_path = f.name

    audio.export(temp_path, format='mp3')

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLoopVideo:
    """Test loop_video functionality."""

    def test_loop_video_import(self):
        """Test that loop_video can be imported."""
        from mixing.video import loop_video

        assert loop_video is not None

    def test_loop_video_basic(self, sample_video_file):
        """Test basic video looping."""
        from mixing.video import loop_video
        import moviepy as mp

        # Get original duration
        with mp.VideoFileClip(sample_video_file) as clip:
            original_duration = clip.duration

        # Loop 3 times
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            saveas = f.name

        try:
            result = loop_video(sample_video_file, n_loops=3, saveas=saveas)

            assert isinstance(result, Path)
            assert result.exists()

            # Verify duration is approximately 3x original
            with mp.VideoFileClip(str(result)) as looped:
                assert (
                    abs(looped.duration - original_duration * 3) < 0.5
                )  # 0.5s tolerance
        finally:
            if os.path.exists(saveas):
                os.unlink(saveas)

    def test_loop_video_twice(self, sample_video_file):
        """Test looping video twice."""
        from mixing.video import loop_video
        import moviepy as mp

        with mp.VideoFileClip(sample_video_file) as clip:
            original_duration = clip.duration

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            saveas = f.name

        try:
            result = loop_video(sample_video_file, n_loops=2, saveas=saveas)

            assert result.exists()

            with mp.VideoFileClip(str(result)) as looped:
                # Should be approximately double the original duration
                assert looped.duration > original_duration * 1.8
                assert looped.duration < original_duration * 2.2
        finally:
            if os.path.exists(saveas):
                os.unlink(saveas)

    def test_loop_video_invalid_loops(self, sample_video_file):
        """Test that invalid n_loops raises error."""
        from mixing.video import loop_video

        # n_loops must be at least 1
        with pytest.raises(ValueError):
            loop_video(sample_video_file, n_loops=0)


class TestReplaceAudio:
    """Test replace_audio functionality."""

    def test_replace_audio_import(self):
        """Test that replace_audio can be imported."""
        from mixing.video import replace_audio

        assert replace_audio is not None

    def test_replace_audio_complete(self, sample_video_file, sample_audio_file):
        """Test complete audio replacement (mix_ratio=1.0)."""
        from mixing.video import replace_audio
        import moviepy as mp

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            saveas = f.name

        try:
            result = replace_audio(
                sample_video_file,
                sample_audio_file,
                mix_ratio=1.0,
                saveas=saveas,
            )

            assert isinstance(result, Path)
            assert result.exists()

            # Verify video has audio
            with mp.VideoFileClip(str(result)) as clip:
                assert clip.audio is not None
        finally:
            if os.path.exists(saveas):
                os.unlink(saveas)

    def test_replace_audio_mix_ratios(self, sample_video_file, sample_audio_file):
        """Test different mix ratios."""
        from mixing.video import replace_audio

        mix_ratios = [0.0, 0.5, 1.0]

        for ratio in mix_ratios:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                saveas = f.name

            try:
                result = replace_audio(
                    sample_video_file,
                    sample_audio_file,
                    mix_ratio=ratio,
                    saveas=saveas,
                )

                assert result.exists()
            finally:
                if os.path.exists(saveas):
                    os.unlink(saveas)

    def test_replace_audio_invalid_ratio(self, sample_video_file, sample_audio_file):
        """Test that invalid mix_ratio raises error."""
        from mixing.video import replace_audio

        # mix_ratio must be between 0 and 1
        with pytest.raises(ValueError):
            replace_audio(sample_video_file, sample_audio_file, mix_ratio=1.5)

        with pytest.raises(ValueError):
            replace_audio(sample_video_file, sample_audio_file, mix_ratio=-0.5)

    def test_replace_audio_normalize(self, sample_video_file, sample_audio_file):
        """Test audio normalization to video length."""
        from mixing.video import replace_audio
        import moviepy as mp

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            saveas = f.name

        try:
            result = replace_audio(
                sample_video_file,
                sample_audio_file,
                mix_ratio=1.0,
                match_duration=True,
                saveas=saveas,
            )

            assert result.exists()

            # Verify audio duration matches video duration
            with mp.VideoFileClip(str(result)) as clip:
                if clip.audio:
                    assert abs(clip.audio.duration - clip.duration) < 0.5
        finally:
            if os.path.exists(saveas):
                os.unlink(saveas)


class TestVideoAudioIntegration:
    """Test integration between video and audio operations."""

    def test_video_import_new_functions(self):
        """Test that new functions are accessible from video module."""
        from mixing.video import loop_video, replace_audio

        assert loop_video is not None
        assert replace_audio is not None

    def test_loop_then_replace_audio(self, sample_video_file, sample_audio_file):
        """Test chaining loop_video and replace_audio."""
        from mixing.video import loop_video, replace_audio

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            looped_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            final_path = f.name

        try:
            # First loop the video
            looped = loop_video(sample_video_file, n_loops=2, saveas=looped_path)
            assert looped.exists()

            # Then replace audio
            final = replace_audio(
                str(looped), sample_audio_file, mix_ratio=1.0, saveas=final_path
            )
            assert final.exists()
        finally:
            for path in [looped_path, final_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestUtilityIntegration:
    """Test that utilities are properly shared between audio and video."""

    def test_shared_utilities_import(self):
        """Test that shared utilities can be imported."""
        from mixing.util import require_package, to_seconds, TimeUnit

        assert require_package is not None
        assert to_seconds is not None
        assert TimeUnit is not None

    def test_time_conversion_for_video(self):
        """Test time conversion works for video (frames)."""
        from mixing.util import to_seconds

        # 24 fps, 240 frames = 10 seconds
        result = to_seconds(240, unit='frames', rate=24)
        assert abs(result - 10.0) < 0.01

    def test_time_conversion_for_audio(self):
        """Test time conversion works for audio (samples)."""
        from mixing.util import to_seconds

        # 44100 Hz, 44100 samples = 1 second
        result = to_seconds(44100, unit='samples', rate=44100)
        assert abs(result - 1.0) < 0.01


# Mark tests that require actual media files
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
