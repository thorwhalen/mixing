"""
Audio editing via slicing interface.

Provides lazy views into audio segments and comprehensive audio editing operations.

This module provides:
- `Audio`: A sliceable audio interface using `audio[start:end]` notation
- `fade_in()`, `fade_out()`: Apply fade effects
- `crop_audio()`: Trim audio segments
- `concatenate_audio()`: Join multiple audio files
- `overlay_audio()`: Mix audio tracks
- Flexible time units: seconds, samples, or milliseconds
- Integration with pydub for audio processing
- Clipboard support for audio file paths

Examples:
    >>> audio = Audio("my_audio.mp3")  # doctest: +SKIP
    >>> segment = audio[10:20]  # Lazy view, no copying  # doctest: +SKIP
    >>> segment.save("clip.mp3")  # Only then does it process  # doctest: +SKIP

    >>> # Apply fade effects
    >>> faded = fade_in(audio, duration=2.0)  # 2 second fade in  # doctest: +SKIP
    >>> faded.save("faded.mp3")  # doctest: +SKIP

    >>> # Concatenate audio files
    >>> combined = concatenate_audio(["intro.mp3", "main.mp3", "outro.mp3"])  # doctest: +SKIP
    >>> combined.save("full.mp3")  # doctest: +SKIP

    >>> # Overlay/mix audio
    >>> mixed = overlay_audio("background.mp3", "voice.mp3", position=5.0)  # doctest: +SKIP

Design principles:
- Lazy evaluation: Operations create views, not copies
- Facade pattern: Clean interface over pydub complexity
- Standard library interfaces: Uses Python's slice notation
- Dependency injection: Configurable time units and formats
- Open-closed: Extensible via keyword arguments
"""

from typing import Union, TYPE_CHECKING
from pathlib import Path
from collections.abc import Iterator, Mapping
import io
import os
import tempfile
import numpy as np

from ..util import require_package, AudioTimeUnit, to_seconds, get_path_from_clipboard

if TYPE_CHECKING:
    from pydub import AudioSegment


class AudioSamples(Mapping[int, float]):
    """
    Mapping interface to access audio samples by index.

    Provides dictionary-like access to audio samples with support for negative
    indexing and slicing. Samples are returned as normalized float values.

    Args:
        audio_src: Path to audio file or AudioSegment
        start_sample: Starting sample index (for segments)
        end_sample: Ending sample index (for segments)

    Examples:
        >>> audio_samples = AudioSamples("test_audio.mp3")  # doctest: +SKIP
        >>> sample = audio_samples[0]  # Get first sample  # doctest: +SKIP
        >>> last_sample = audio_samples[-1]  # Get last sample  # doctest: +SKIP
        >>> samples = list(audio_samples[1000:2000])  # Get samples 1000-1999  # doctest: +SKIP
    """

    def __init__(
        self,
        audio_src: Union[str, "AudioSegment"],
        start_sample: int = 0,
        end_sample: int | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(audio_src, str):
            self.audio_src = audio_src
            self._audio = AudioSegment.from_file(audio_src)
        else:
            self.audio_src = None
            self._audio = audio_src

        # Get samples as numpy array
        self._samples = np.array(self._audio.get_array_of_samples())
        if self._audio.channels == 2:
            # Reshape stereo to (n_samples, 2)
            self._samples = self._samples.reshape((-1, 2))

        # Normalize to [-1, 1]
        self._samples = self._samples.astype(np.float32) / 32768.0

        self.start_sample = start_sample
        self.end_sample = end_sample if end_sample is not None else len(self._samples)
        self._sample_count = self.end_sample - self.start_sample

    def __len__(self) -> int:
        """Return number of samples in this view."""
        return self._sample_count

    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices in this view."""
        return iter(range(self.start_sample, self.end_sample))

    def __getitem__(self, key: int | slice) -> float | np.ndarray:
        """
        Get sample(s) by index or slice.

        Args:
            key: Integer index or slice object (relative to this view)

        Returns:
            Single sample value or array of samples
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self._sample_count)
            abs_start = self.start_sample + start
            abs_stop = self.start_sample + stop
            return self._samples[abs_start:abs_stop:step]
        elif isinstance(key, int):
            if key < 0:
                key = self._sample_count + key
            if key < 0 or key >= self._sample_count:
                raise IndexError(
                    f"Sample index {key} out of range [0, {self._sample_count})"
                )
            abs_idx = self.start_sample + key
            return self._samples[abs_idx]
        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )


class Audio:
    """
    Sliceable interface for audio supporting time-based operations.

    Provides lazy views into audio segments using slice notation. Slicing returns
    new Audio instances (not copies), enabling chained operations.

    Args:
        src_path: Path to source audio file or AudioSegment
        time_unit: Unit for slice indices ('seconds', 'samples', 'milliseconds')
        start_time: Start time in seconds (for creating sub-views)
        end_time: End time in seconds (for creating sub-views)

    Examples:
        >>> audio = Audio("song.mp3")  # doctest: +SKIP
        >>>
        >>> # Get segment from 10s to 20s (returns Audio)
        >>> segment = audio[10:20]  # doctest: +SKIP
        >>> segment.save("clip.mp3")  # doctest: +SKIP
        >>>
        >>> # Use sample numbers as unit
        >>> audio_samples = Audio("song.mp3", time_unit="samples")  # doctest: +SKIP
        >>> segment = audio_samples[44100:88200]  # 1 second at 44.1kHz  # doctest: +SKIP
        >>>
        >>> # Get last 30 seconds
        >>> ending = audio[-30:]  # doctest: +SKIP
        >>>
        >>> # Chain operations
        >>> trimmed = audio[5:120]  # Trim to 5s-120s  # doctest: +SKIP
        >>> faded = trimmed.fade_in(2).fade_out(3)  # Apply fades  # doctest: +SKIP
        >>> faded.save("final.mp3")  # doctest: +SKIP
    """

    def __init__(
        self,
        src_path: Union[str, "AudioSegment"],
        *,
        time_unit: AudioTimeUnit = "seconds",
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(src_path, str):
            self.src_path = str(src_path)
            self._audio = AudioSegment.from_file(src_path)
        else:
            self.src_path = None
            self._audio = src_path

        self.time_unit = time_unit
        self._start_time = start_time  # None means start of audio
        self._end_time = end_time  # None means end of audio

    @property
    def start_time(self) -> float:
        """Start time in seconds (0.0 if not set)."""
        return self._start_time if self._start_time is not None else 0.0

    @property
    def end_time(self) -> float:
        """End time in seconds (audio duration if not set)."""
        return self._end_time if self._end_time is not None else self.full_duration

    @property
    def full_duration(self) -> float:
        """Duration of the source audio in seconds."""
        return len(self._audio) / 1000.0

    @property
    def duration(self) -> float:
        """Duration of this audio/segment in seconds."""
        return self.end_time - self.start_time

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._audio.frame_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._audio.channels

    @property
    def sample_count(self) -> int:
        """Total number of samples in this segment."""
        return int(self.duration * self.sample_rate)

    def _normalize_index(self, idx: int | float | None, is_start: bool) -> float:
        """Convert slice index to seconds, handling None and negative indices."""
        if idx is None:
            return self.start_time if is_start else self.end_time

        # Convert to seconds based on time_unit
        idx_seconds = to_seconds(idx, unit=self.time_unit, rate=self.sample_rate)

        # Handle negative indices (from end of this segment)
        if idx_seconds < 0:
            idx_seconds = self.end_time + idx_seconds
        else:
            # Positive indices are relative to segment start
            idx_seconds = self.start_time + idx_seconds

        # Clamp to segment's valid range
        return max(self.start_time, min(idx_seconds, self.end_time))

    def __getitem__(self, key: int | float | slice) -> "Audio":
        """
        Get an audio segment using slice notation.

        Args:
            key: Slice for time range

        Returns:
            New Audio instance representing the segment

        Examples:
            >>> audio = Audio("test.mp3")  # doctest: +SKIP
            >>> segment = audio[10:20]  # 10s to 20s  # doctest: +SKIP
            >>> ending = audio[-30:]  # Last 30 seconds  # doctest: +SKIP
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported for audio slicing")

            start = self._normalize_index(key.start, is_start=True)
            end = self._normalize_index(key.stop, is_start=False)

            if start >= end:
                raise ValueError(
                    f"Invalid time range: start ({start}s) must be before end ({end}s)"
                )

            # Return new Audio instance
            return Audio(
                self._audio,
                time_unit=self.time_unit,
                start_time=start,
                end_time=end,
            )

        elif isinstance(key, (int, float)):
            # Single sample/time point - return very short segment
            idx_seconds = to_seconds(key, unit=self.time_unit, rate=self.sample_rate)
            if idx_seconds < 0:
                time_seconds = self.end_time + idx_seconds
            else:
                time_seconds = self.start_time + idx_seconds

            time_seconds = max(self.start_time, min(time_seconds, self.end_time))
            sample_duration = 1.0 / self.sample_rate
            return Audio(
                self._audio,
                time_unit=self.time_unit,
                start_time=time_seconds,
                end_time=time_seconds + sample_duration,
            )
        else:
            raise TypeError(
                f"Audio indexing requires int/float or slice, "
                f"got {type(key).__name__}"
            )

    def _get_segment(self) -> "AudioSegment":
        """Get the AudioSegment for this time range."""
        start_ms = int(self.start_time * 1000)
        end_ms = int(self.end_time * 1000)
        return self._audio[start_ms:end_ms]

    def save(
        self,
        output_path: str | None = None,
        *,
        format: str | None = None,
        bitrate: str = "192k",
        **export_kwargs,
    ) -> Path:
        """
        Save this audio/segment to a new audio file.

        Args:
            output_path: Path for output file (auto-generated if None)
            format: Audio format (mp3, wav, etc.). Auto-detected from extension if None.
            bitrate: Bitrate for compressed formats
            **export_kwargs: Additional arguments for pydub export

        Returns:
            Path to saved file

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> audio[10:30].save("clip.mp3")  # doctest: +SKIP
            >>> audio[10:30].save("clip.wav", format="wav")  # doctest: +SKIP
        """
        if output_path is None:
            if self.src_path:
                src = Path(self.src_path)
                output_path = src.with_stem(
                    f"{src.stem}_{int(self.start_time)}_{int(self.end_time)}"
                )
            else:
                output_path = Path(
                    f"audio_{int(self.start_time)}_{int(self.end_time)}.mp3"
                )

        output_path = Path(output_path)

        # Auto-detect format from extension
        if format is None:
            format = output_path.suffix[1:] if output_path.suffix else "mp3"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export segment
        segment = self._get_segment()
        segment.export(
            str(output_path), format=format, bitrate=bitrate, **export_kwargs
        )

        print(f"Saved audio to: {output_path}")
        return output_path

    def fade_in(self, duration: float = 1.0) -> "Audio":
        """
        Apply fade-in effect.

        Args:
            duration: Fade duration in seconds

        Returns:
            New Audio with fade applied

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> faded = audio.fade_in(2.0)  # 2 second fade in  # doctest: +SKIP
        """
        segment = self._get_segment()
        duration_ms = int(duration * 1000)
        faded = segment.fade_in(duration_ms)
        return Audio(faded, time_unit=self.time_unit)

    def fade_out(self, duration: float = 1.0) -> "Audio":
        """
        Apply fade-out effect.

        Args:
            duration: Fade duration in seconds

        Returns:
            New Audio with fade applied

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> faded = audio.fade_out(3.0)  # 3 second fade out  # doctest: +SKIP
        """
        segment = self._get_segment()
        duration_ms = int(duration * 1000)
        faded = segment.fade_out(duration_ms)
        return Audio(faded, time_unit=self.time_unit)

    def __add__(self, other: "Audio") -> "Audio":
        """
        Concatenate two audio segments.

        Examples:
            >>> audio1 = Audio("part1.mp3")  # doctest: +SKIP
            >>> audio2 = Audio("part2.mp3")  # doctest: +SKIP
            >>> combined = audio1 + audio2  # doctest: +SKIP
        """
        seg1 = self._get_segment()
        seg2 = other._get_segment()
        combined = seg1 + seg2
        return Audio(combined, time_unit=self.time_unit)

    def overlay(
        self,
        other: "Audio",
        position: float = 0.0,
        *,
        gain_during_overlay: float = 0.0,
    ) -> "Audio":
        """
        Overlay another audio on top of this one.

        Args:
            other: Audio to overlay
            position: Position in seconds where overlay starts
            gain_during_overlay: Volume adjustment in dB during overlay

        Returns:
            New Audio with overlay applied

        Examples:
            >>> bg = Audio("background.mp3")  # doctest: +SKIP
            >>> voice = Audio("voice.mp3")  # doctest: +SKIP
            >>> mixed = bg.overlay(voice, position=5.0, gain_during_overlay=-6)  # doctest: +SKIP
        """
        seg1 = self._get_segment()
        seg2 = other._get_segment()
        position_ms = int(position * 1000)
        mixed = seg1.overlay(
            seg2, position=position_ms, gain_during_overlay=gain_during_overlay
        )
        return Audio(mixed, time_unit=self.time_unit)

    def __repr__(self) -> str:
        if self._start_time is not None or self._end_time is not None:
            src_info = f"'{self.src_path}'" if self.src_path else "AudioSegment"
            return (
                f"Audio({src_info}, "
                f"time_unit='{self.time_unit}', "
                f"start={self.start_time:.2f}s, "
                f"end={self.end_time:.2f}s, "
                f"duration={self.duration:.2f}s)"
            )
        else:
            src_info = f"'{self.src_path}'" if self.src_path else "AudioSegment"
            return (
                f"Audio({src_info}, "
                f"time_unit='{self.time_unit}', "
                f"duration={self.full_duration:.2f}s)"
            )

    @property
    def samples(self) -> AudioSamples:
        """Get sample-by-sample Mapping interface for this audio."""
        start_sample = int(self.start_time * self.sample_rate)
        end_sample = int(self.end_time * self.sample_rate)
        return AudioSamples(
            self._audio, start_sample=start_sample, end_sample=end_sample
        )


# Convenience functions


def crop_audio(
    src_path: str,
    start: float | int | None = None,
    end: float | int | None = None,
    *,
    time_unit: AudioTimeUnit = "seconds",
    output_path: str | None = None,
    **save_kwargs,
) -> Path:
    """
    Convenience function to crop and save an audio segment.

    Args:
        src_path: Path to source audio
        start: Start time (None = beginning)
        end: End time (None = end of audio)
        time_unit: Unit for start/end values
        output_path: Path for output (auto-generated if None)
        **save_kwargs: Additional arguments for save operation

    Returns:
        Path to saved cropped audio

    Examples:
        >>> crop_audio("song.mp3", 10, 30)  # Crop 10s-30s  # doctest: +SKIP
        >>> crop_audio("song.mp3", 44100, 88200, time_unit="samples")  # doctest: +SKIP
    """
    audio = Audio(src_path, time_unit=time_unit)
    segment = audio[start:end]
    return segment.save(output_path, **save_kwargs)


def fade_in(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output_path: str | None = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-in effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output_path: If provided, saves to file and returns Path. Otherwise returns Audio.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_in("song.mp3", 2.0, output_path="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_in("song.mp3", 2.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_in(duration)

    if output_path:
        return faded.save(output_path, **save_kwargs)
    return faded


def fade_out(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output_path: str | None = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-out effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output_path: If provided, saves to file and returns Path. Otherwise returns Audio.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_out("song.mp3", 3.0, output_path="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_out("song.mp3", 3.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_out(duration)

    if output_path:
        return faded.save(output_path, **save_kwargs)
    return faded


def concatenate_audio(
    *sources: Union[str, Audio],
    output_path: str | None = None,
    crossfade: float = 0.0,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Concatenate multiple audio files/segments.

    Args:
        *sources: Audio sources (filepaths or Audio instances)
        output_path: If provided, saves to file and returns Path. Otherwise returns Audio.
        crossfade: Crossfade duration in seconds between segments
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> concatenate_audio("intro.mp3", "main.mp3", "outro.mp3")  # doctest: +SKIP
        >>> concatenate_audio("a.mp3", "b.mp3", output_path="combined.mp3")  # doctest: +SKIP
        >>> concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)  # 500ms crossfade  # doctest: +SKIP
    """
    if not sources:
        raise ValueError("At least one audio source is required")

    # Convert all to Audio instances
    audios = [Audio(src) if isinstance(src, str) else src for src in sources]

    # Start with first audio
    result = audios[0]

    # Add remaining audios
    for audio in audios[1:]:
        if crossfade > 0:
            # Apply crossfade
            seg1 = result._get_segment()
            seg2 = audio._get_segment()
            crossfade_ms = int(crossfade * 1000)
            combined = seg1.append(seg2, crossfade=crossfade_ms)
            result = Audio(combined)
        else:
            # Simple concatenation
            result = result + audio

    if output_path:
        return result.save(output_path, **save_kwargs)
    return result


def overlay_audio(
    background: Union[str, Audio],
    overlay: Union[str, Audio],
    position: float = 0.0,
    *,
    mix_ratio: float = 0.5,
    output_path: str | None = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Overlay/mix two audio sources.

    Args:
        background: Background audio (filepath or Audio instance)
        overlay: Audio to overlay (filepath or Audio instance)
        position: Position in seconds where overlay starts
        mix_ratio: Mix ratio (0.0 = only background, 1.0 = only overlay, 0.5 = equal mix)
        output_path: If provided, saves to file and returns Path. Otherwise returns Audio.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> overlay_audio("music.mp3", "voice.mp3", position=5.0)  # doctest: +SKIP
        >>> overlay_audio("bg.mp3", "sfx.mp3", mix_ratio=0.3)  # 30% overlay, 70% bg  # doctest: +SKIP
    """
    bg_audio = Audio(background) if isinstance(background, str) else background
    ov_audio = Audio(overlay) if isinstance(overlay, str) else overlay

    # Calculate gain adjustments based on mix ratio
    # mix_ratio = 0.5 means equal mix (both at -3dB)
    # mix_ratio = 1.0 means full overlay volume, background silent
    # mix_ratio = 0.0 means full background volume, overlay silent

    if mix_ratio == 0.5:
        # Equal mix: reduce both by 3dB
        gain_during_overlay = -3.0
    elif mix_ratio < 0.5:
        # More background, less overlay
        # Overlay gain ranges from -inf (at 0.0) to -3dB (at 0.5)
        if mix_ratio == 0.0:
            gain_during_overlay = -100  # Effectively silent
        else:
            # Logarithmic scaling
            gain_during_overlay = 20 * np.log10(mix_ratio * 2)
    else:
        # More overlay, less background
        # For now, just use standard overlay (might need to adjust background volume)
        gain_during_overlay = 0.0

    mixed = bg_audio.overlay(
        ov_audio, position=position, gain_during_overlay=gain_during_overlay
    )

    if output_path:
        return mixed.save(output_path, **save_kwargs)
    return mixed


def save_audio_clip(
    audio_src: str | None = None,
    start: float = 0,
    end: float | None = None,
    *,
    time_unit: AudioTimeUnit | None = None,
    saveas: str | None = None,
    format: str = "mp3",
) -> Path:
    """
    Extract and save an audio clip.

    Args:
        audio_src: Path to audio file. If None, gets from clipboard.
        start: Start time/sample (default: 0)
        end: End time/sample (None = end of audio)
        time_unit: Unit for start/end ('seconds', 'samples', 'milliseconds')
        saveas: Output path (auto-generated if None)
        format: Output format

    Returns:
        Path to saved audio file

    Examples:
        >>> save_audio_clip("song.mp3", 10, 30)  # Save 10s-30s  # doctest: +SKIP
        >>> save_audio_clip(start=5, end=15)  # From clipboard  # doctest: +SKIP
    """
    if audio_src is None:
        audio_src = get_path_from_clipboard()

    if time_unit is None:
        time_unit = "seconds"

    audio = Audio(audio_src, time_unit=time_unit)
    segment = audio[start:end] if end is not None else audio[start:]

    return segment.save(saveas, format=format)
