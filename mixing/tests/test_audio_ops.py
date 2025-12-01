"""
Tests for audio editing operations.

These tests verify the audio editing functionality including:
- Audio class slicing and time-based operations
- Fade in/out effects
- Audio cropping, concatenation, and overlay
- Time unit conversions
- Mapping interface for audio samples
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


# Test fixtures
@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # Generate 1 second of sine wave at 440Hz, 44.1kHz sample rate
    duration = 1.0  # seconds
    sample_rate = 44100
    frequency = 440  # Hz (A note)

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    samples = np.sin(2 * np.pi * frequency * t)

    return samples, sample_rate


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Create a temporary audio file for testing."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        pytest.skip("pydub not installed")

    samples, sample_rate = sample_audio_data

    # Create audio segment from sine wave
    # Use pydub's generator for simplicity
    audio = Sine(440).to_audio_segment(duration=1000, volume=-20)  # 1 second

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        temp_path = f.name

    audio.export(temp_path, format='mp3')

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestAudioClass:
    """Test the Audio class."""

    def test_audio_import(self):
        """Test that Audio class can be imported."""
        from mixing.audio import Audio

        assert Audio is not None

    def test_audio_creation_from_file(self, temp_audio_file):
        """Test creating Audio instance from file."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)
        assert audio is not None
        assert audio.src_path == temp_audio_file
        assert audio.duration > 0
        assert audio.sample_rate > 0

    def test_audio_slicing(self, temp_audio_file):
        """Test audio slicing creates proper segments."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)
        original_duration = audio.duration

        # Slice to first half
        segment = audio[0 : original_duration / 2]

        assert segment is not None
        assert isinstance(segment, Audio)
        assert segment.duration < original_duration
        assert abs(segment.duration - original_duration / 2) < 0.1  # Within 100ms

    def test_audio_negative_indexing(self, temp_audio_file):
        """Test negative indexing for audio slicing."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)

        # Get last 0.5 seconds
        ending = audio[-0.5:]

        assert ending is not None
        assert ending.duration <= 0.5 + 0.1  # Allow 100ms tolerance

    def test_audio_properties(self, temp_audio_file):
        """Test audio properties are accessible."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)

        assert audio.sample_rate > 0
        assert audio.channels in [1, 2]
        assert audio.duration > 0
        assert audio.sample_count > 0


class TestAudioOperations:
    """Test audio editing operations."""

    def test_fade_in_function(self, temp_audio_file):
        """Test fade_in function."""
        from mixing.audio import fade_in, Audio

        # Test with file path
        result = fade_in(temp_audio_file, duration=0.1)
        assert isinstance(result, Audio)

        # Test with Audio instance
        audio = Audio(temp_audio_file)
        faded = fade_in(audio, duration=0.1)
        assert isinstance(faded, Audio)

    def test_fade_out_function(self, temp_audio_file):
        """Test fade_out function."""
        from mixing.audio import fade_out, Audio

        result = fade_out(temp_audio_file, duration=0.1)
        assert isinstance(result, Audio)

    def test_crop_audio_function(self, temp_audio_file):
        """Test crop_audio function."""
        from mixing.audio import crop_audio, Audio

        audio = Audio(temp_audio_file)
        original_duration = audio.duration

        # Crop to middle section
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            output_path = f.name

        try:
            result = crop_audio(
                temp_audio_file, start=0.2, end=0.8, output_path=output_path
            )

            assert isinstance(result, Path)
            assert result.exists()

            # Verify cropped audio
            cropped = Audio(str(result))
            assert cropped.duration < original_duration
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_concatenate_audio_function(self, temp_audio_file):
        """Test concatenate_audio function."""
        from mixing.audio import concatenate_audio, Audio

        # Concatenate file with itself
        result = concatenate_audio(temp_audio_file, temp_audio_file)

        assert isinstance(result, Audio)

        # Duration should be approximately double
        original = Audio(temp_audio_file)
        assert result.duration > original.duration * 1.8  # Allow some tolerance


class TestTimeUnitConversion:
    """Test time unit conversion functionality."""

    def test_time_unit_seconds(self, temp_audio_file):
        """Test time unit in seconds (default)."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file, time_unit='seconds')
        segment = audio[0:0.5]

        assert abs(segment.duration - 0.5) < 0.1

    def test_time_unit_milliseconds(self, temp_audio_file):
        """Test time unit in milliseconds."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file, time_unit='milliseconds')
        segment = audio[0:500]  # 500ms

        assert abs(segment.duration - 0.5) < 0.1

    def test_time_unit_samples(self, temp_audio_file):
        """Test time unit in samples."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file, time_unit='samples')

        # Get first 22050 samples (0.5 seconds at 44.1kHz)
        segment = audio[0:22050]

        # Should be approximately 0.5 seconds
        assert abs(segment.duration - 0.5) < 0.1


class TestAudioSamples:
    """Test AudioSamples mapping interface."""

    def test_audio_samples_property(self, temp_audio_file):
        """Test that audio.samples property works."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)
        samples = audio.samples

        assert samples is not None
        assert len(samples) > 0

    def test_audio_samples_indexing(self, temp_audio_file):
        """Test indexing into audio samples."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)
        samples = audio.samples

        # Get first sample
        first_sample = samples[0]
        assert isinstance(first_sample, (float, np.floating))

        # Get last sample with negative indexing
        last_sample = samples[-1]
        assert isinstance(last_sample, (float, np.floating))


class TestUtilityFunctions:
    """Test utility functions."""

    def test_require_package(self):
        """Test require_package utility."""
        from mixing.util import require_package

        # Test with built-in package
        math = require_package('math')
        assert math is not None

        # Test with non-existent package
        with pytest.raises(ImportError) as exc_info:
            require_package('nonexistent_package_xyz123')

        assert 'pip install' in str(exc_info.value)

    def test_to_seconds_conversion(self):
        """Test to_seconds time conversion."""
        from mixing.util import to_seconds

        # Test seconds
        assert to_seconds(10, unit='seconds', rate=44100) == 10

        # Test samples
        assert to_seconds(44100, unit='samples', rate=44100) == 1.0

        # Test milliseconds
        assert to_seconds(1000, unit='milliseconds', rate=44100) == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_time_range(self, temp_audio_file):
        """Test that invalid time ranges raise errors."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)

        # Start after end should raise error
        with pytest.raises(ValueError):
            _ = audio[1.0:0.5]

    def test_audio_chaining(self, temp_audio_file):
        """Test chaining audio operations."""
        from mixing.audio import Audio

        audio = Audio(temp_audio_file)

        # Chain multiple operations
        result = audio[0.1:0.9].fade_in(0.1).fade_out(0.1)

        assert result is not None
        assert isinstance(result, Audio)


# Mark tests that require actual audio files
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
