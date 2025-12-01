"""
Tests for shared utility functions in mixing.util.

These tests verify:
- require_package: Safe package importing with helpful errors
- to_seconds: Time unit conversion for audio and video
- Clipboard utilities (when pyclip is available)
"""

import pytest
import numpy as np


class TestRequirePackage:
    """Test require_package utility."""

    def test_require_existing_package(self):
        """Test importing an existing package."""
        from mixing.util import require_package

        # Test with built-in package
        math_module = require_package('math')
        assert math_module is not None
        assert hasattr(math_module, 'pi')

    def test_require_nonexistent_package(self):
        """Test that non-existent package raises helpful error."""
        from mixing.util import require_package

        with pytest.raises(ImportError) as exc_info:
            require_package('this_package_definitely_does_not_exist_xyz123')

        error_msg = str(exc_info.value)
        assert 'pip install' in error_msg
        assert 'this_package_definitely_does_not_exist_xyz123' in error_msg


class TestToSeconds:
    """Test to_seconds time conversion."""

    def test_seconds_to_seconds(self):
        """Test that seconds remain unchanged."""
        from mixing.util import to_seconds

        result = to_seconds(10.5, unit='seconds', rate=44100)
        assert result == 10.5

    def test_frames_to_seconds(self):
        """Test converting video frames to seconds."""
        from mixing.util import to_seconds

        # At 24 fps, 240 frames = 10 seconds
        result = to_seconds(240, unit='frames', rate=24)
        assert abs(result - 10.0) < 0.001

        # At 30 fps, 300 frames = 10 seconds
        result = to_seconds(300, unit='frames', rate=30)
        assert abs(result - 10.0) < 0.001

    def test_samples_to_seconds(self):
        """Test converting audio samples to seconds."""
        from mixing.util import to_seconds

        # At 44100 Hz, 44100 samples = 1 second
        result = to_seconds(44100, unit='samples', rate=44100)
        assert abs(result - 1.0) < 0.001

        # At 48000 Hz, 48000 samples = 1 second
        result = to_seconds(48000, unit='samples', rate=48000)
        assert abs(result - 1.0) < 0.001

    def test_milliseconds_to_seconds(self):
        """Test converting milliseconds to seconds."""
        from mixing.util import to_seconds

        # 1000 ms = 1 second (rate doesn't matter for ms)
        result = to_seconds(1000, unit='milliseconds', rate=44100)
        assert abs(result - 1.0) < 0.001

        result = to_seconds(500, unit='milliseconds', rate=24)
        assert abs(result - 0.5) < 0.001

    def test_invalid_unit(self):
        """Test that invalid unit raises error."""
        from mixing.util import to_seconds

        with pytest.raises(ValueError) as exc_info:
            to_seconds(100, unit='invalid_unit', rate=44100)

        assert 'Invalid time unit' in str(exc_info.value)

    def test_fractional_conversions(self):
        """Test that fractional values work correctly."""
        from mixing.util import to_seconds

        # 2.5 seconds should remain 2.5 seconds
        result = to_seconds(2.5, unit='seconds', rate=24)
        assert abs(result - 2.5) < 0.001

        # 60 frames at 24 fps = 2.5 seconds
        result = to_seconds(60, unit='frames', rate=24)
        assert abs(result - 2.5) < 0.001


class TestTimeUnitTypes:
    """Test TimeUnit and AudioTimeUnit type definitions."""

    def test_time_unit_exists(self):
        """Test that TimeUnit type is defined."""
        from mixing.util import TimeUnit

        assert TimeUnit is not None

    def test_audio_time_unit_exists(self):
        """Test that AudioTimeUnit type is defined."""
        from mixing.util import AudioTimeUnit

        assert AudioTimeUnit is not None


class TestClipboardUtilities:
    """Test clipboard utilities (when available)."""

    def test_copy_to_clipboard_exists(self):
        """Test that copy_to_clipboard function exists."""
        from mixing.util import copy_to_clipboard

        assert copy_to_clipboard is not None

    def test_get_path_from_clipboard_exists(self):
        """Test that get_path_from_clipboard function exists."""
        from mixing.util import get_path_from_clipboard

        assert get_path_from_clipboard is not None

    @pytest.mark.skipif(
        not pytest.importorskip('pyclip', reason='pyclip not installed'),
        reason='Clipboard tests require pyclip',
    )
    def test_copy_string_to_clipboard(self):
        """Test copying string to clipboard."""
        from mixing.util import copy_to_clipboard

        test_string = "test_clipboard_content"

        # This test just verifies the function doesn't crash
        # Actual clipboard interaction is hard to test reliably
        try:
            copy_to_clipboard(test_string)
        except Exception as e:
            # May fail in headless environments
            pytest.skip(f"Clipboard not available: {e}")


class TestUtilityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_conversion(self):
        """Test converting zero values."""
        from mixing.util import to_seconds

        assert to_seconds(0, unit='seconds', rate=44100) == 0.0
        assert to_seconds(0, unit='frames', rate=24) == 0.0
        assert to_seconds(0, unit='samples', rate=44100) == 0.0
        assert to_seconds(0, unit='milliseconds', rate=44100) == 0.0

    def test_large_values(self):
        """Test converting large values."""
        from mixing.util import to_seconds

        # 1 hour in frames at 24 fps
        result = to_seconds(24 * 60 * 60, unit='frames', rate=24)
        assert abs(result - 3600.0) < 0.01

        # 1 hour in samples at 44100 Hz
        result = to_seconds(44100 * 60 * 60, unit='samples', rate=44100)
        assert abs(result - 3600.0) < 0.01

    def test_different_rates(self):
        """Test conversion with various frame/sample rates."""
        from mixing.util import to_seconds

        rates = [24, 30, 60, 44100, 48000, 96000]

        for rate in rates:
            # 1 rate unit should equal 1/rate seconds
            result = to_seconds(
                1, unit='frames' if rate < 100 else 'samples', rate=rate
            )
            expected = 1.0 / rate
            assert abs(result - expected) < 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
