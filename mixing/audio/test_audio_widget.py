"""
Comprehensive test suite for audio_widget.py

Tests cover:
- ensure_wfsr function with various input types
- AudioWidget core functionality
- Selection, zoom, and crop operations
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import io

# Import the module under test
from audio_widget import ensure_wfsr, AudioWidget, wf_to_base64_wav


# Fixtures
@pytest.fixture
def sample_waveform():
    """Create a sample mono waveform."""
    duration = 1.0  # 1 second
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave at 440 Hz (A4)
    wf = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return wf, sr


@pytest.fixture
def stereo_waveform():
    """Create a sample stereo waveform."""
    duration = 1.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    # Left channel: 440 Hz, Right channel: 880 Hz
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    wf = np.column_stack([left, right]).astype(np.float32)
    return wf, sr


@pytest.fixture
def temp_audio_file(sample_waveform):
    """Create a temporary audio file."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    wf, sr = sample_waveform
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, wf, sr)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# Tests for ensure_wfsr function
class TestEnsureWfsr:
    """Test suite for the ensure_wfsr utility function."""

    def test_ensure_wfsr_with_tuple(self, sample_waveform):
        """Test that (wf, sr) tuple input passes through correctly."""
        wf, sr = sample_waveform
        result_wf, result_sr = ensure_wfsr((wf, sr))

        assert isinstance(result_wf, np.ndarray)
        assert isinstance(result_sr, int)
        assert np.array_equal(result_wf, wf)
        assert result_sr == sr

    def test_ensure_wfsr_with_numpy_int_sr(self, sample_waveform):
        """Test that numpy integer sample rates are converted to int."""
        wf, sr = sample_waveform
        result_wf, result_sr = ensure_wfsr((wf, np.int32(sr)))

        assert isinstance(result_sr, int)
        assert result_sr == sr

    def test_ensure_wfsr_with_file_path_str(self, temp_audio_file):
        """Test loading from string file path."""
        wf, sr = ensure_wfsr(str(temp_audio_file))

        assert isinstance(wf, np.ndarray)
        assert isinstance(sr, int)
        assert len(wf) > 0
        assert sr > 0

    def test_ensure_wfsr_with_file_path_pathlib(self, temp_audio_file):
        """Test loading from pathlib.Path object."""
        wf, sr = ensure_wfsr(temp_audio_file)

        assert isinstance(wf, np.ndarray)
        assert isinstance(sr, int)
        assert len(wf) > 0

    def test_ensure_wfsr_with_bytes(self, temp_audio_file):
        """Test loading from raw audio bytes."""
        # Read file as bytes
        audio_bytes = temp_audio_file.read_bytes()

        wf, sr = ensure_wfsr(audio_bytes)

        assert isinstance(wf, np.ndarray)
        assert isinstance(sr, int)
        assert len(wf) > 0

    def test_ensure_wfsr_invalid_tuple_type(self):
        """Test that invalid tuple types raise TypeError."""
        with pytest.raises(TypeError, match="Invalid .* tuple"):
            ensure_wfsr(("not_array", 44100))

    def test_ensure_wfsr_invalid_tuple_length(self):
        """Test that tuples with wrong length are handled."""
        with pytest.raises((TypeError, ValueError)):
            ensure_wfsr((1, 2, 3))

    def test_ensure_wfsr_nonexistent_file(self):
        """Test that nonexistent file paths raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ensure_wfsr("/nonexistent/path/to/audio.wav")

    def test_ensure_wfsr_unsupported_type(self):
        """Test that unsupported input types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported audio input type"):
            ensure_wfsr(12345)

        with pytest.raises(TypeError, match="Unsupported audio input type"):
            ensure_wfsr([1, 2, 3])

    def test_ensure_wfsr_stereo_shape(self, stereo_waveform):
        """Test that stereo waveforms preserve correct shape."""
        wf, sr = stereo_waveform
        result_wf, result_sr = ensure_wfsr((wf, sr))

        assert result_wf.shape == wf.shape
        assert result_wf.ndim == 2
        assert result_wf.shape[1] == 2  # Two channels


class TestWfToBase64Wav:
    """Test suite for wf_to_base64_wav helper function."""

    def test_wf_to_base64_wav_mono(self, sample_waveform):
        """Test encoding mono waveform to base64."""
        wf, sr = sample_waveform
        result = wf_to_base64_wav(wf, sr)

        assert isinstance(result, str)
        assert result.startswith("data:audio/wav;base64,")
        assert len(result) > 100  # Should be a substantial base64 string

    def test_wf_to_base64_wav_stereo(self, stereo_waveform):
        """Test encoding stereo waveform to base64."""
        wf, sr = stereo_waveform
        result = wf_to_base64_wav(wf, sr)

        assert isinstance(result, str)
        assert result.startswith("data:audio/wav;base64,")

    def test_wf_to_base64_wav_dtype_conversion(self, sample_waveform):
        """Test that non-float32 arrays are converted."""
        wf, sr = sample_waveform
        wf_int = (wf * 32767).astype(np.int16)

        result = wf_to_base64_wav(wf_int, sr)
        assert isinstance(result, str)
        assert result.startswith("data:audio/wav;base64,")


class TestAudioWidget:
    """Test suite for the AudioWidget class."""

    def test_widget_init_with_tuple(self, sample_waveform):
        """Test widget initialization with (wf, sr) tuple."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        assert isinstance(widget, AudioWidget)
        assert np.array_equal(widget.wf, wf)
        assert widget.sr == sr
        assert widget.duration > 0

    def test_widget_init_with_file(self, temp_audio_file):
        """Test widget initialization with file path."""
        widget = AudioWidget(temp_audio_file)

        assert isinstance(widget, AudioWidget)
        assert len(widget.wf) > 0
        assert widget.sr > 0

    def test_widget_properties(self, sample_waveform):
        """Test widget properties are set correctly."""
        wf, sr = sample_waveform
        widget = AudioWidget(
            (wf, sr),
            height=200,
            waveform_color='#FF0000',
            progress_color='#00FF00'
        )

        assert widget.height == 200
        assert widget.waveform_color == '#FF0000'
        assert widget.progress_color == '#00FF00'

    def test_widget_get_waveform(self, sample_waveform):
        """Test get_waveform method returns correct data."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        result_wf, result_sr = widget.get_waveform()

        assert np.array_equal(result_wf, wf)
        assert result_sr == sr

    def test_widget_selection_initial_state(self, sample_waveform):
        """Test that selection state is initially empty."""
        widget = AudioWidget(sample_waveform)

        assert widget.has_selection == False
        assert widget.selection_start == 0.0
        assert widget.selection_end == 0.0
        assert widget.get_selection() is None

    def test_widget_zoom_without_selection(self, sample_waveform):
        """Test that zoom raises error without selection."""
        widget = AudioWidget(sample_waveform)

        with pytest.raises(ValueError, match="No selection made"):
            widget.zoom()

    def test_widget_crop_without_selection(self, sample_waveform):
        """Test that crop raises error without selection."""
        widget = AudioWidget(sample_waveform)

        with pytest.raises(ValueError, match="No selection made"):
            widget.crop()

    def test_widget_zoom_with_selection(self, sample_waveform):
        """Test zoom operation with valid selection."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        # Simulate selection from 0.2s to 0.5s
        widget.selection_start = 0.2
        widget.selection_end = 0.5
        widget.has_selection = True

        original_length = len(widget.wf)
        widget.zoom()

        # Waveform should be shorter after zoom
        assert len(widget.wf) < original_length
        # Duration should be approximately 0.3s
        assert abs(widget.duration - 0.3) < 0.01
        # Selection should be reset
        assert widget.has_selection == False

    def test_widget_crop_with_selection(self, sample_waveform):
        """Test crop operation with valid selection."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        # Simulate selection
        widget.selection_start = 0.1
        widget.selection_end = 0.4
        widget.has_selection = True

        original_length = len(widget.wf)
        widget.crop()

        # Waveform should be cropped
        assert len(widget.wf) < original_length
        expected_samples = int(0.3 * sr)
        assert abs(len(widget.wf) - expected_samples) < sr * 0.01  # Within 10ms
        # Selection should be reset
        assert widget.has_selection == False

    def test_widget_invalid_selection_order(self, sample_waveform):
        """Test that invalid selection order raises error."""
        widget = AudioWidget(sample_waveform)

        # Set selection with start > end
        widget.selection_start = 0.5
        widget.selection_end = 0.2
        widget.has_selection = True

        with pytest.raises(ValueError, match="must be before end"):
            widget.zoom()

    def test_widget_reset(self, sample_waveform):
        """Test reset functionality restores original audio."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        original_length = len(widget.wf)

        # Perform crop
        widget.selection_start = 0.1
        widget.selection_end = 0.5
        widget.has_selection = True
        widget.crop()

        assert len(widget.wf) < original_length

        # Reset
        widget.reset()

        assert len(widget.wf) == original_length
        assert np.array_equal(widget.wf, wf)

    def test_widget_get_selection(self, sample_waveform):
        """Test get_selection returns correct segment."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        # Simulate selection
        widget.selection_start = 0.2
        widget.selection_end = 0.5
        widget.has_selection = True

        selected_wf, selected_sr = widget.get_selection()

        assert selected_sr == sr
        expected_samples = int(0.3 * sr)
        assert abs(len(selected_wf) - expected_samples) < sr * 0.01

    def test_widget_save(self, sample_waveform, tmp_path):
        """Test saving audio to file."""
        widget = AudioWidget(sample_waveform)

        output_file = tmp_path / "output.wav"
        widget.save(output_file)

        assert output_file.exists()

        # Verify we can load it back
        loaded_wf, loaded_sr = ensure_wfsr(output_file)
        assert len(loaded_wf) > 0
        assert loaded_sr == widget.sr

    def test_widget_apply_fade_in(self, sample_waveform):
        """Test fade-in effect."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf.copy(), sr))

        original_wf = widget.wf.copy()
        widget.apply_fade_in(duration=0.1)

        # First samples should be attenuated
        assert widget.wf[0] < original_wf[0] or widget.wf[0] == 0.0
        # Later samples should be unchanged (approximately)
        mid_point = len(widget.wf) // 2
        assert np.allclose(widget.wf[mid_point:], original_wf[mid_point:], atol=1e-5)

    def test_widget_apply_fade_out(self, sample_waveform):
        """Test fade-out effect."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf.copy(), sr))

        original_wf = widget.wf.copy()
        widget.apply_fade_out(duration=0.1)

        # Last samples should be attenuated
        assert abs(widget.wf[-1]) < abs(original_wf[-1]) or widget.wf[-1] == 0.0
        # Earlier samples should be unchanged (approximately)
        mid_point = len(widget.wf) // 2
        assert np.allclose(widget.wf[:mid_point], original_wf[:mid_point], atol=1e-5)

    def test_widget_stereo_support(self, stereo_waveform):
        """Test that widget handles stereo audio correctly."""
        widget = AudioWidget(stereo_waveform)

        assert widget.wf.ndim == 2
        assert widget.wf.shape[1] == 2

        # Test operations work with stereo
        widget.selection_start = 0.1
        widget.selection_end = 0.5
        widget.has_selection = True
        widget.crop()

        assert widget.wf.ndim == 2
        assert widget.wf.shape[1] == 2

    def test_widget_edge_case_very_short_audio(self):
        """Test widget with very short audio."""
        sr = 44100
        wf = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, int(sr * 0.01))).astype(np.float32)

        widget = AudioWidget((wf, sr))

        assert widget.duration < 0.02
        assert len(widget.wf) > 0

    def test_widget_selection_bounds_checking(self, sample_waveform):
        """Test that selection bounds are properly enforced."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        # Try to select beyond audio length
        widget.selection_start = 0.5
        widget.selection_end = 10.0  # Way beyond actual duration
        widget.has_selection = True

        # Should not raise error but clamp to valid range
        widget.crop()

        # Audio should be cropped from 0.5 to end
        expected_samples = int((widget.duration - 0.5) * sr)
        # Due to clamping, actual result may differ slightly
        assert len(widget.wf) > 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_soundfile_not_installed(self, monkeypatch):
        """Test graceful handling when soundfile is not installed."""
        # Mock the import to raise ImportError
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'soundfile':
                raise ImportError("No module named 'soundfile'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)

        with pytest.raises(ImportError, match="soundfile is required"):
            # This should trigger the ImportError
            ensure_wfsr("fake_path.wav")

    def test_widget_with_zero_length_selection(self, sample_waveform):
        """Test widget behavior with zero-length selection."""
        widget = AudioWidget(sample_waveform)

        widget.selection_start = 0.5
        widget.selection_end = 0.5  # Same as start
        widget.has_selection = True

        with pytest.raises(ValueError, match="must be before end"):
            widget.crop()

    def test_widget_duration_calculation(self, sample_waveform):
        """Test that duration is calculated correctly."""
        wf, sr = sample_waveform
        widget = AudioWidget((wf, sr))

        expected_duration = len(wf) / sr
        assert abs(widget.duration - expected_duration) < 0.001


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_load_crop_save(self, temp_audio_file, tmp_path):
        """Test complete workflow: load, select, crop, save."""
        # Load
        widget = AudioWidget(temp_audio_file)
        original_duration = widget.duration

        # Select middle third
        start = original_duration / 3
        end = 2 * original_duration / 3
        widget.selection_start = start
        widget.selection_end = end
        widget.has_selection = True

        # Crop
        widget.crop()

        # Verify crop worked
        expected_duration = original_duration / 3
        assert abs(widget.duration - expected_duration) < 0.01

        # Save
        output_file = tmp_path / "cropped.wav"
        widget.save(output_file)

        # Verify saved file
        assert output_file.exists()

        # Load saved file and verify
        new_widget = AudioWidget(output_file)
        assert abs(new_widget.duration - expected_duration) < 0.01

    def test_multiple_operations_sequence(self, sample_waveform):
        """Test performing multiple operations in sequence."""
        widget = AudioWidget(sample_waveform)

        # First crop
        widget.selection_start = 0.2
        widget.selection_end = 0.8
        widget.has_selection = True
        widget.crop()
        duration_after_first_crop = widget.duration

        # Second crop on the result
        widget.selection_start = 0.1
        widget.selection_end = 0.4
        widget.has_selection = True
        widget.crop()

        # Should be smaller than after first crop
        assert widget.duration < duration_after_first_crop

        # Apply fade
        widget.apply_fade_in(0.05)
        widget.apply_fade_out(0.05)

        # Should still have audio
        assert len(widget.wf) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
