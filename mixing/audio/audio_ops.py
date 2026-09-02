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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import io
import os
import tempfile
import numpy as np

from ..util import require_package, AudioTimeUnit, to_seconds, get_path_from_clipboard
from .audio_util import AudioSource, _normalize_audio_source
from ..egress import Output, deliver, is_sink, resolve_output_path

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
        audio_src: Union[str, "os.PathLike[str]", "AudioSegment"],
        start_sample: int = 0,
        end_sample: int | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(audio_src, (str, os.PathLike)):
            # Accept both ``str`` and ``pathlib.Path`` / any os.PathLike
            audio_src = os.fspath(audio_src)
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
        src_path: Union[str, "os.PathLike[str]", "AudioSegment"],
        *,
        time_unit: AudioTimeUnit = "seconds",
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(src_path, (str, os.PathLike)):
            # Accept both ``str`` and ``pathlib.Path`` / any os.PathLike
            src_path = os.fspath(src_path)
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
                f"Audio indexing requires int/float or slice, got {type(key).__name__}"
            )

    def _get_segment(self) -> "AudioSegment":
        """Get the AudioSegment for this time range."""
        start_ms = int(self.start_time * 1000)
        end_ms = int(self.end_time * 1000)
        return self._audio[start_ms:end_ms]

    def save(
        self,
        output: Output = None,
        *,
        format: str | None = None,
        bitrate: str = "192k",
        **export_kwargs,
    ) -> Path:
        """
        Save this audio/segment to a new audio file.

        Args:
            output: Where to put the result — None (save beside the input with
                an auto-derived name), a file path, a directory (auto-named), or
                a callable sink. See mixing.egress.
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
        # Auto-name (beside the source) when no explicit destination is given.
        if self.src_path:
            src = Path(self.src_path)
            default_name = (
                f"{src.stem}_{int(self.start_time)}_{int(self.end_time)}{src.suffix}"
            )
        else:
            default_name = f"audio_{int(self.start_time)}_{int(self.end_time)}.mp3"

        sink = output if is_sink(output) else None
        if output is None or sink is not None:
            # No path given (or a sink): write to the default location beside
            # the input, then hand that Path to the sink if there is one.
            if self.src_path:
                output_path = Path(self.src_path).with_name(default_name)
            else:
                output_path = Path(default_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = resolve_output_path(output, default_name=default_name)

        # Auto-detect format from extension
        if format is None:
            format = output_path.suffix[1:] if output_path.suffix else "mp3"

        # Export segment
        segment = self._get_segment()
        segment.export(
            str(output_path), format=format, bitrate=bitrate, **export_kwargs
        )

        print(f"Saved audio to: {output_path}")
        return sink(output_path) if sink is not None else output_path

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

    def normalize(self, *, headroom: float = 0.1) -> "Audio":
        """Peak-normalize the audio (via ``pydub.effects.normalize``).

        Boosts (or attenuates) the segment so its loudest peak sits ``headroom``
        dB below 0 dBFS. Pure pydub — adds no new dependency.

        Args:
            headroom: Target peak distance below 0 dBFS, in dB (keyword-only).

        Returns:
            New Audio with normalization applied.

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> louder = audio.normalize()  # doctest: +SKIP
        """
        from pydub import effects

        segment = self._get_segment()
        normalized = effects.normalize(segment, headroom=headroom)
        return Audio(normalized, time_unit=self.time_unit)

    def to_mono(self) -> "Audio":
        """Downmix to a single channel (via pydub ``set_channels(1)``).

        Returns:
            New mono Audio. Pure pydub — adds no new dependency.

        Examples:
            >>> audio = Audio("stereo.mp3")  # doctest: +SKIP
            >>> mono = audio.to_mono()  # doctest: +SKIP
        """
        segment = self._get_segment()
        mono = segment.set_channels(1)
        return Audio(mono, time_unit=self.time_unit)

    def resample(self, sample_rate: int) -> "Audio":
        """Change the sample rate (via pydub ``set_frame_rate``).

        Args:
            sample_rate: Target sample rate in Hz (e.g. ``16000``, ``44100``).

        Returns:
            New Audio at the requested sample rate. Pure pydub — adds no new
            dependency.

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> downsampled = audio.resample(16000)  # doctest: +SKIP
        """
        segment = self._get_segment()
        resampled = segment.set_frame_rate(sample_rate)
        return Audio(resampled, time_unit=self.time_unit)

    def close(self) -> None:
        """Release the reference to the in-memory audio (no OS handles to free).

        ``Audio`` is fully in-memory (a decoded ``AudioSegment``), so there is
        nothing OS-level to close. ``close`` simply drops the reference so the
        data can be garbage-collected promptly; the object should not be used
        afterwards.
        """
        self._audio = None

    def __enter__(self) -> "Audio":
        """Support ``with Audio(path) as a: ...`` — returns ``self``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release in-memory data on context exit (see :meth:`close`)."""
        self.close()

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
    output: Output = None,
    **save_kwargs,
) -> Path:
    """
    Convenience function to crop and save an audio segment.

    Args:
        src_path: Path to source audio
        start: Start time (None = beginning)
        end: End time (None = end of audio)
        time_unit: Unit for start/end values
        output: Where to put the result — None (save beside the input), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional arguments for save operation

    Returns:
        Path to saved cropped audio

    Examples:
        >>> crop_audio("song.mp3", 10, 30)  # Crop 10s-30s  # doctest: +SKIP
        >>> crop_audio("song.mp3", 44100, 88200, time_unit="samples")  # doctest: +SKIP
    """
    audio = Audio(src_path, time_unit=time_unit)
    segment = audio[start:end]
    return segment.save(output, **save_kwargs)


def fade_in(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-in effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_in("song.mp3", 2.0, output="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_in("song.mp3", 2.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_in(duration)
    return deliver(
        faded,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_fade_in.mp3",
    )


def fade_out(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-out effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_out("song.mp3", 3.0, output="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_out("song.mp3", 3.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_out(duration)
    return deliver(
        faded,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_fade_out.mp3",
    )


def concatenate_audio(
    *sources: Union[str, Audio],
    output: Output = None,
    crossfade: float = 0.0,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Concatenate multiple audio files/segments.

    Args:
        *sources: Audio sources (filepaths or Audio instances)
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        crossfade: Crossfade duration in seconds between segments
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> concatenate_audio("intro.mp3", "main.mp3", "outro.mp3")  # doctest: +SKIP
        >>> concatenate_audio("a.mp3", "b.mp3", output="combined.mp3")  # doctest: +SKIP
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

    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_concat.mp3",
    )


#: Gain (dB) below which an overlay/background contribution is treated as muted.
#: -120 dB is ~1e-6 amplitude — inaudible — and avoids ``log10(0) = -inf``.
_MIX_SILENCE_FLOOR_DB = -120.0


def _mix_amplitude_to_db(amplitude: float) -> float:
    """Linear amplitude ratio in ``[0, 1]`` → dB gain, floored at silence."""
    if amplitude <= 0.0:
        return _MIX_SILENCE_FLOOR_DB
    return max(_MIX_SILENCE_FLOOR_DB, float(20.0 * np.log10(amplitude)))


def overlay_audio(
    background: Union[str, Path, Audio],
    overlay: Union[str, Path, Audio],
    position: float = 0.0,
    *,
    mix_ratio: float = 0.5,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Overlay/mix two audio sources.

    ``mix_ratio`` is the prominence of the *overlay*, modeled as a
    linear-amplitude crossfade between background-only and overlay-only: the
    overlay plays at gain ``20·log10(mix_ratio)`` and the background is ducked
    by ``20·log10(1 - mix_ratio)`` for the overlap's duration. So ``0.0`` =
    only the background, ``1.0`` = only the overlay (during the overlap),
    ``0.5`` = an equal blend (both ~-6 dB).

    Args:
        background: Background audio (filepath or Audio instance)
        overlay: Audio to overlay (filepath or Audio instance)
        position: Position in seconds where overlay starts
        mix_ratio: Prominence of the overlay in ``[0.0, 1.0]`` (see above).
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> overlay_audio("music.mp3", "voice.mp3", position=5.0)  # doctest: +SKIP
        >>> overlay_audio("bg.mp3", "sfx.mp3", mix_ratio=0.3)  # 30% overlay, 70% bg  # doctest: +SKIP
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    bg_audio = (
        Audio(background) if isinstance(background, (str, os.PathLike)) else background
    )
    ov_audio = Audio(overlay) if isinstance(overlay, (str, os.PathLike)) else overlay

    overlay_gain_db = _mix_amplitude_to_db(mix_ratio)
    background_gain_db = _mix_amplitude_to_db(1.0 - mix_ratio)

    if overlay_gain_db <= _MIX_SILENCE_FLOOR_DB:
        # Overlay is muted — the result is just the background, untouched.
        mixed = bg_audio
    else:
        overlay_seg = ov_audio._get_segment()
        if overlay_gain_db != 0.0:
            overlay_seg = overlay_seg + overlay_gain_db  # pydub gain
        overlay_adjusted = Audio(overlay_seg, time_unit=ov_audio.time_unit)
        mixed = bg_audio.overlay(
            overlay_adjusted, position=position, gain_during_overlay=background_gain_db
        )

    return deliver(
        mixed,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_overlay.mp3",
    )


#: Default crossfade (seconds) applied at every loop join in :func:`loop_audio`.
#: Long enough to hide the waveform discontinuity where the tail meets the head,
#: short enough not to smear the bed's content.
DEFAULT_LOOP_CROSSFADE_S = 0.5

#: A loop crossfade may consume at most this fraction of the source's duration.
#: Each join must still advance the timeline, so this can never reach ``1.0``.
_MAX_LOOP_CROSSFADE_FRACTION = 0.5


def loop_audio(
    source: Union[str, Path, Audio],
    target_duration_s: float,
    *,
    crossfade_s: float = DEFAULT_LOOP_CROSSFADE_S,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Tile an audio source until it fills ``target_duration_s``, seamlessly.

    The source is appended to itself with a ``crossfade_s`` crossfade at every
    join (pydub's equal-gain fade-out/fade-in), then trimmed to *exactly*
    ``target_duration_s``. This is the "make a 20 s ambient bed last 4 minutes"
    primitive: without the crossfade, every loop point is a hard splice and a
    waveform discontinuity you can hear as a click.

    A source **longer** than the target is simply trimmed — looping is only
    ever additive, never a no-op guard the caller has to write.

    Because each join consumes ``crossfade_s`` of timeline, the crossfade is
    clamped to half the source's duration; otherwise a long crossfade over a
    short source would never advance.

    Note the tail is cut wherever ``target_duration_s`` lands (mid-loop is
    normal), and no fade-out is applied — chain :func:`fade_out` if the bed
    ends exposed.

    Args:
        source: Audio to loop (filepath or :class:`Audio`).
        target_duration_s: Duration of the result, in seconds (> 0).
        crossfade_s: Crossfade at each loop join, in seconds. ``0`` gives hard
            splices. Clamped to half the source duration.
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments.

    Returns:
        Audio instance or Path to saved file.

    Examples:
        >>> bed = loop_audio("room_tone.wav", 90.0)  # 90s bed  # doctest: +SKIP
        >>> loop_audio("waves.wav", 240.0, output="bed.wav")  # doctest: +SKIP
    """
    if target_duration_s <= 0:
        raise ValueError(f"target_duration_s must be > 0, got {target_duration_s}")
    if crossfade_s < 0:
        raise ValueError(f"crossfade_s must be >= 0, got {crossfade_s}")

    audio = source if isinstance(source, Audio) else Audio(source)
    segment = audio._get_segment()
    source_ms = len(segment)
    if source_ms <= 0:
        raise ValueError("Cannot loop an empty audio source")

    target_ms = int(round(target_duration_s * 1000))
    crossfade_ms = min(
        int(round(crossfade_s * 1000)),
        int(source_ms * _MAX_LOOP_CROSSFADE_FRACTION),
    )

    looped = segment
    while len(looped) < target_ms:
        looped = looped.append(segment, crossfade=crossfade_ms)
    looped = looped[:target_ms]

    return deliver(
        Audio(looped, time_unit=audio.time_unit),
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_loop.mp3",
    )


#: Gain (dB) applied to the bed while the sidechain is active.
DEFAULT_DUCK_DB = -12.0
#: Sidechain frames louder than this (dBFS RMS) count as "active".
DEFAULT_DUCK_THRESHOLD_DB = -40.0
#: Time constant for moving *down* to the ducked level (sidechain goes active).
DEFAULT_DUCK_ATTACK_S = 0.05
#: Time constant for coming *back up* to unity (sidechain goes quiet).
DEFAULT_DUCK_RELEASE_S = 0.4
#: How long the duck is held past the last active frame, so the bed rides
#: through the gaps between words instead of pumping on every syllable.
DEFAULT_DUCK_HOLD_S = 0.2
#: Level-detector frame size — the time resolution of the sidechain envelope.
DEFAULT_DUCK_FRAME_S = 0.02


def _frame_rms_db(segment: "AudioSegment", *, frame_n: int) -> np.ndarray:
    """Per-frame RMS level (dBFS) of ``segment``, downmixed to mono.

    ``frame_n`` is the frame length in samples-per-channel. The final partial
    frame is zero-padded, which only ever *lowers* its level — a partial frame
    can't fake activity.
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels).mean(axis=1)
    full_scale = float(1 << (8 * segment.sample_width - 1))
    samples = samples / full_scale

    n_frames = max(1, int(np.ceil(len(samples) / frame_n)))
    padded = np.zeros(n_frames * frame_n, dtype=np.float64)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame_n) ** 2, axis=1))
    floor = 10.0 ** (_MIX_SILENCE_FLOOR_DB / 20.0)
    return 20.0 * np.log10(np.maximum(rms, floor))


def _hold_active(active: np.ndarray, *, hold_frames: int) -> np.ndarray:
    """Extend each active run forward by ``hold_frames`` frames."""
    if hold_frames <= 0:
        return active
    indices = np.arange(len(active))
    last_active = np.maximum.accumulate(np.where(active, indices, -1))
    return (last_active >= 0) & ((indices - last_active) <= hold_frames)


def _duck_gain_envelope(
    active: np.ndarray,
    *,
    frame_s: float,
    duck_db: float,
    attack_s: float,
    release_s: float,
) -> np.ndarray:
    """Per-frame gain (dB) that falls to ``duck_db`` while ``active``, else 0.

    A one-pole smoother in the dB domain with separate attack (going down) and
    release (coming back up) time constants — the standard gain-smoothing shape
    of a broadcast ducker.
    """

    def _coeff(time_constant_s: float) -> float:
        if time_constant_s <= 0:
            return 1.0  # instantaneous
        return 1.0 - float(np.exp(-frame_s / time_constant_s))

    attack_coeff = _coeff(attack_s)
    release_coeff = _coeff(release_s)

    targets = np.where(active, duck_db, 0.0)
    gains = np.empty(len(targets), dtype=np.float64)
    gain = 0.0
    for i, target in enumerate(targets):
        coeff = attack_coeff if target < gain else release_coeff
        gain += (target - gain) * coeff
        gains[i] = gain
    return gains


def _apply_gain_envelope(
    segment: "AudioSegment", gain_db: np.ndarray, *, frame_s: float
) -> "AudioSegment":
    """Apply a per-frame dB gain envelope to ``segment``, sample-interpolated.

    Gains are linearly interpolated between frame centers so the envelope is
    continuous — a per-frame staircase would add its own zipper noise.

    Samples round-trip through numpy at the segment's own width; pydub only
    ever holds 1-, 2- or 4-byte samples (its ``AudioSegment`` widens 24-bit to
    32-bit on construction), so the `array` typecode always matches the frame
    width and no width conversion is needed here.
    """
    AudioSegment = require_package("pydub").AudioSegment

    samples = np.array(segment.get_array_of_samples())
    channels = segment.channels
    per_channel = len(samples) // channels
    if per_channel == 0:
        return segment

    sample_times = (np.arange(per_channel) + 0.5) / segment.frame_rate
    frame_times = (np.arange(len(gain_db)) + 0.5) * frame_s
    gain_linear = 10.0 ** (np.interp(sample_times, frame_times, gain_db) / 20.0)

    scaled = samples.astype(np.float64).reshape(-1, channels) * gain_linear[:, None]
    limits = np.iinfo(samples.dtype)
    adjusted = np.clip(np.rint(scaled), limits.min, limits.max).astype(samples.dtype)

    return AudioSegment(
        data=adjusted.reshape(-1).tobytes(),
        sample_width=segment.sample_width,
        frame_rate=segment.frame_rate,
        channels=channels,
    )


def duck_audio(
    bed: Union[str, Path, Audio],
    sidechain: Union[str, Path, Audio],
    *,
    duck_db: float = DEFAULT_DUCK_DB,
    threshold_db: float = DEFAULT_DUCK_THRESHOLD_DB,
    attack_s: float = DEFAULT_DUCK_ATTACK_S,
    release_s: float = DEFAULT_DUCK_RELEASE_S,
    hold_s: float = DEFAULT_DUCK_HOLD_S,
    frame_s: float = DEFAULT_DUCK_FRAME_S,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Duck ``bed`` wherever ``sidechain`` is loud (sidechain ducking).

    **What it does.** A level-detector sidechain ducker: the ``sidechain``
    (typically the dialogue track) is framed at ``frame_s``, each frame's RMS
    is compared to ``threshold_db``, active frames are extended by ``hold_s``,
    and the resulting on/off signal drives a gain envelope on ``bed`` that
    falls to ``duck_db`` with time constant ``attack_s`` and returns to unity
    with ``release_s``. The envelope is interpolated to sample resolution
    before it is applied, so there is no zipper noise. The result always has
    the **bed's** duration; a shorter sidechain simply leaves the tail
    un-ducked.

    **What it does not do.** It is not a full compressor: there is no ratio,
    knee, or makeup gain — the duck depth is the fixed ``duck_db``, not a
    function of how loud the sidechain is. There is no lookahead, so with a
    short ``attack_s`` the first few milliseconds of a sudden word can sneak
    through at full bed level. Detection is **energy-based, not speech-aware**:
    any loud sidechain content (music, a door slam, hiss above
    ``threshold_db``) ducks the bed just as dialogue would. And the bed is
    attenuated, never EQ'd — it does not carve a vocal-band notch.

    Args:
        bed: The audio to be ducked (filepath or :class:`Audio`).
        sidechain: The audio that triggers ducking — the dialogue track.
        duck_db: Gain (dB, ``<= 0``) held while the sidechain is active.
        threshold_db: Sidechain frames above this RMS dBFS count as active.
        attack_s: Time constant for reaching the ducked level.
        release_s: Time constant for returning to unity.
        hold_s: How long the duck persists after the last active frame.
        frame_s: Level-detector frame size (envelope time resolution).
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments.

    Returns:
        Audio instance or Path to saved file.

    Examples:
        >>> quiet_bed = duck_audio("bed.wav", "dialogue.wav")  # doctest: +SKIP
        >>> duck_audio("bed.wav", "vo.wav", duck_db=-18)  # deeper duck  # doctest: +SKIP
    """
    if duck_db > 0:
        raise ValueError(f"duck_db must be <= 0 dB (a duck attenuates), got {duck_db}")
    if frame_s <= 0:
        raise ValueError(f"frame_s must be > 0, got {frame_s}")
    for name, value in (
        ("attack_s", attack_s),
        ("release_s", release_s),
        ("hold_s", hold_s),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    bed_audio = bed if isinstance(bed, Audio) else Audio(bed)
    side_audio = sidechain if isinstance(sidechain, Audio) else Audio(sidechain)

    bed_segment = bed_audio._get_segment()
    side_segment = side_audio._get_segment()
    if len(bed_segment) <= 0:
        raise ValueError("Cannot duck an empty bed")

    n_frames = max(1, int(np.ceil((len(bed_segment) / 1000.0) / frame_s)))
    side_frame_n = max(1, int(round(frame_s * side_segment.frame_rate)))

    active = np.zeros(n_frames, dtype=bool)
    if len(side_segment) > 0:
        levels = _frame_rms_db(side_segment, frame_n=side_frame_n)
        overlap = min(n_frames, len(levels))
        active[:overlap] = levels[:overlap] > threshold_db

    gain_db = _duck_gain_envelope(
        _hold_active(active, hold_frames=int(round(hold_s / frame_s))),
        frame_s=frame_s,
        duck_db=duck_db,
        attack_s=attack_s,
        release_s=release_s,
    )
    ducked = _apply_gain_envelope(bed_segment, gain_db, frame_s=frame_s)

    return deliver(
        Audio(ducked, time_unit=bed_audio.time_unit),
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_ducked.mp3",
    )


def save_audio_clip(
    audio_src: str | None = None,
    start: float = 0,
    end: float | None = None,
    *,
    time_unit: AudioTimeUnit | None = None,
    output: Output = None,
    format: str = "mp3",
) -> Path:
    """
    Extract and save an audio clip.

    Args:
        audio_src: Path to audio file. If None, gets from clipboard.
        start: Start time/sample (default: 0)
        end: End time/sample (None = end of audio)
        time_unit: Unit for start/end ('seconds', 'samples', 'milliseconds')
        output: Where to put the result — None (save beside the input), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
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

    return segment.save(output, format=format)


def find_audio_offset(
    reference_audio: AudioSource,
    query_audio: AudioSource,
    *,
    sample_rate: int = 16000,
) -> float:
    """Find the time offset where query_audio best aligns within reference_audio.

    Uses FFT-based cross-correlation to find the position in reference_audio
    where query_audio starts. This is useful for aligning different recordings
    of the same performance — for example, aligning a studio recording (voice
    + instruments) with a camera recording (voice only).

    The two audio signals don't need to be identical; they just need to share
    a correlated component (e.g., the same voice in both).

    Args:
        reference_audio: The longer audio to search within (e.g., extracted
            from a video). Accepts a file path, numpy array, or AudioSegment.
        query_audio: The shorter audio to align (e.g., a studio recording).
            Accepts a file path, numpy array, or AudioSegment.
        sample_rate: Sample rate for analysis. Lower values are faster but
            less precise. Default 16000 Hz gives ~0.06ms precision, which is
            more than sufficient for alignment purposes.

    Returns:
        Offset in seconds — the position in reference_audio where query_audio
        starts. Positive means query begins after the start of reference.

    Examples:
        >>> from mixing.audio import find_audio_offset  # doctest: +SKIP
        >>> # Find where a studio recording aligns with a camera recording
        >>> offset = find_audio_offset("camera_audio.wav", "studio.mp3")  # doctest: +SKIP
        >>> print(f"Studio recording starts at {offset:.2f}s in the camera audio")  # doctest: +SKIP
    """
    return find_audio_offset_detailed(
        reference_audio, query_audio, sample_rate=sample_rate
    ).offset_s


@dataclass(frozen=True)
class AudioOffset:
    """The result of aligning one recording within another.

    Attributes:
        offset_s: Time in ``reference`` where ``query`` begins (seconds). Positive
            means query starts after the reference's t=0; **negative** means query
            began before it (e.g. a phone that started filming before the song).
        confidence: A scale-invariant normalized cross-correlation coefficient in
            ``[0, 1]`` at the best lag — comparable ACROSS clips of different loudness
            and length. ~0.5+ is a strong match; near 0 means no shared component.
        sample_rate: The analysis sample rate the offset was computed at.
    """

    offset_s: float
    confidence: float
    sample_rate: int


def _resample_samples(
    samples: np.ndarray, native_rate: int, target_rate: int
) -> np.ndarray:
    """Anti-aliased polyphase resample of mono ``samples`` (native → target rate).

    Uses :func:`scipy.signal.resample_poly`, which low-pass filters before
    decimating, so downsampling does not fold energy above the target Nyquist
    back into the band. Rational rate ratios (e.g. 44100 → 16000 = 160/441)
    are reduced exactly via :class:`fractions.Fraction`.
    """
    if native_rate == target_rate:
        return samples
    from fractions import Fraction

    ratio = Fraction(target_rate, native_rate)
    resample_poly = require_package("scipy.signal").resample_poly
    return resample_poly(samples, ratio.numerator, ratio.denominator)


def _load_mono_samples(source: AudioSource, sample_rate: int) -> np.ndarray:
    """Load any :data:`AudioSource` as a mono float64 array at ``sample_rate``.

    Decodes at the source's NATIVE rate (pydub → ffmpeg), downmixes to mono,
    then resamples with an anti-aliased polyphase filter
    (:func:`_resample_samples`). pydub's ``set_frame_rate`` is deliberately NOT
    used for the rate change: it delegates to ``audioop.ratecv`` — linear
    interpolation with no anti-alias filter — so downsampling a 48 kHz camera
    track to the 16 kHz analysis rate folds everything above 8 kHz back over
    the band. Measured on real multicam footage that aliasing roughly HALVED
    the alignment confidence for identical (correct) offsets, making the score
    depend on the decode path rather than the signal (issue #25).
    """
    seg = _normalize_audio_source(source, target_type="AudioSegment")
    seg = seg.set_channels(1)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    return _resample_samples(samples, seg.frame_rate, sample_rate)


def _normalized_xcorr(
    ref: np.ndarray, query: np.ndarray, *, min_overlap_ratio: float = 0.5
) -> tuple[int, float]:
    """Overlap-normalized cross-correlation of two mono signals.

    Returns ``(lag_samples, coefficient)`` where ``lag_samples`` is where ``query``
    begins within ``ref`` (may be negative) and ``coefficient`` is the normalized
    cross-correlation in ``[0, 1]`` at that lag. Normalizing each lag by the energy of
    its actual overlap makes the score scale-invariant (comparable across clips) AND
    removes the triangular-overlap argmax bias for clips that extend before/after the
    reference — the common multi-device case. Lags overlapping less than
    ``min_overlap_ratio`` of the shorter signal are excluded so a tiny sliver of overlap
    can't win.
    """
    correlate = require_package("scipy.signal").correlate
    ref = ref - ref.mean()
    query = query - query.mean()
    n_r, n_q = len(ref), len(query)
    num = correlate(ref, query, mode="full", method="fft")
    # lag L for full-output index k is L = k - (n_q - 1)
    lags = np.arange(n_r + n_q - 1) - (n_q - 1)
    overlap = np.minimum(n_r, lags + n_q) - np.maximum(0, lags)
    # windowed sum-of-squares of each signal over its overlap region, via cumsum
    cum_r = np.concatenate([[0.0], np.cumsum(ref**2)])
    cum_q = np.concatenate([[0.0], np.cumsum(query**2)])
    r_lo = np.maximum(0, lags)
    q_lo = np.maximum(0, -lags)
    energy_r = cum_r[r_lo + overlap] - cum_r[r_lo]
    energy_q = cum_q[q_lo + overlap] - cum_q[q_lo]
    denom = np.sqrt(energy_r * energy_q)
    # Divide only where the denominator is non-zero. `np.where(denom > 0, num / denom, 0)`
    # evaluates the quotient for EVERY lag first, so a silent overlap (denom == 0, which a
    # real recording reaches at its extreme lags) raised a divide-by-zero RuntimeWarning on
    # every call before the mask was applied.
    coeff = np.zeros_like(denom)
    nz = denom > 0
    np.divide(num, denom, out=coeff, where=nz)
    valid = overlap >= max(1.0, min_overlap_ratio * min(n_r, n_q))
    scored = np.where(valid, np.abs(coeff), -1.0)
    best = int(np.argmax(scored))
    return int(lags[best]), float(np.abs(coeff[best]))


#: Envelope hop, in samples at the analysis rate. 160 @ 16 kHz = 10 ms frames (100 Hz).
ENVELOPE_HOP = 160
#: STFT window for the envelope. 1024 @ 16 kHz = 64 ms — long enough to resolve a musical
#: onset, short enough not to smear it.
ENVELOPE_NFFT = 1024


def onset_envelope(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop: int = ENVELOPE_HOP,
    nfft: int = ENVELOPE_NFFT,
) -> "tuple[np.ndarray, float]":
    """A channel-robust onset/energy envelope: ``(envelope, envelope_rate_hz)``.

    Log-compressed STFT magnitude, positive first difference, summed over frequency, then
    standardized. This is *spectral flux* — it tracks WHEN energy arrives, not the waveform
    itself, so it survives the things that destroy raw-waveform similarity between two
    devices recording the same sound: different microphone responses, different positions
    (hence different room impulse responses), and the resulting phase differences.

    Why this matters, measured on a real 6-device shoot: raw-waveform correlation scored
    provably-correct alignments (three independent methods agreeing to within 10 ms) at
    0.064-0.148, while the envelope scored the same pairs at 0.441-0.634 and a genuine
    non-match at 0.102. The raw coefficient could not separate match from non-match; the
    envelope separates them by more than 4x.

    Args:
        samples: Mono samples.
        sample_rate: Their rate.
        hop: Frames advance by this many samples (sets the envelope's time resolution).
        nfft: STFT window length.

    Returns:
        ``(envelope, envelope_rate_hz)``. The envelope is zero-mean, unit-variance, so
        correlations of two envelopes are directly comparable.
    """
    stft = require_package("scipy.signal").stft
    _, _, spec = stft(
        samples,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=nfft - hop,
        padded=False,
        boundary=None,
    )
    logmag = np.log1p(np.abs(spec))
    flux = np.diff(logmag, axis=1)
    flux[flux < 0] = 0.0  # onsets only — decays carry no timing information
    env = flux.sum(axis=0)
    return (env - env.mean()) / (env.std() + 1e-9), sample_rate / hop


def _envelope_then_waveform(
    ref: np.ndarray,
    query: np.ndarray,
    sample_rate: int,
    *,
    min_overlap_ratio: float,
    ref_envelope: "tuple[np.ndarray, float] | None" = None,
) -> "tuple[int, float]":
    """Waveform picks the lag; the confidence is the better of two views of that lag.

    Returns ``(lag_samples, confidence)``.

    **The waveform locates.** Its peak *position* is reliable — on a real 6-device shoot it
    agreed with two independent methods to within 10 ms — and it has full sample resolution.
    It does not get to choose alone how good the match is, and the envelope does not get to
    choose the lag: on signals whose energy is smoothly modulated rather than percussive
    (linear chirps under a 1.7 Hz AM), the envelope autocorrelation is periodic, and
    letting it pick moved a known -2.0 s offset to -2.5 s.

    **The confidence is the larger of two correlations evaluated AT that lag** — the
    waveform's and the onset envelope's. Not a trick to inflate the score: each feature is
    *blind* in a regime the other sees clearly, and which regime you are in is not knowable
    in advance.

    - *Same source* (an export against its master; a clip cut from the reference): the
      waveform correlates near 1.0. The envelope may be near-useless here, because content
      with no percussive onsets has no flux structure to correlate — measured at 0.08 for an
      exact copy of an onset-free signal.
    - *Different devices* (the multicam case): the waveform is near-useless, because two
      microphones in a room are not sample-correlated even when the alignment is exact —
      measured at 0.055-0.188 for alignments confirmed correct by two other methods. The
      envelope reads the same pairs at 0.17-0.60.
    - *Unrelated audio*: both are low, so the maximum is low and the pair is still rejected —
      measured at max(0.010, 0.021) on a genuine non-match.

    Taking the maximum means the score answers "how similar are these at the offset we are
    reporting, by the most favourable of two complementary measures". Taking either alone
    means silently returning a near-zero score for a perfectly good alignment in the other
    feature's blind spot — which, at a downstream threshold, deletes the user's footage.
    """
    # `ref_envelope` lets a caller that aligns MANY queries against ONE reference
    # compute the reference's envelope once. Omitting it reproduces the previous
    # behaviour exactly, so `find_audio_offset_detailed` is unchanged; supplying it
    # measured 2.11x on `aligned_spans`, which was recomputing it per window.
    env_ref, env_rate = (
        onset_envelope(ref, sample_rate) if ref_envelope is None else ref_envelope
    )
    env_query, _ = onset_envelope(query, sample_rate)
    wav_lag, _ = _normalized_xcorr(ref, query, min_overlap_ratio=min_overlap_ratio)
    wav_at_lag = _correlation_at_lag(ref, query, wav_lag)
    if env_ref.size < 2 or env_query.size < 2:  # too short to have an envelope
        return wav_lag, wav_at_lag
    env_lag = int(round(wav_lag * env_rate / sample_rate))
    env_at_lag = _correlation_at_lag(env_ref, env_query, env_lag)
    return wav_lag, max(wav_at_lag, env_at_lag)


def _correlation_at_lag(ref: np.ndarray, query: np.ndarray, lag: int) -> float:
    """Normalized correlation of ``query`` against ``ref`` at exactly ``lag``, in ``[0, 1]``.

    The single-lag counterpart of :func:`_normalized_xcorr`, which searches for the best
    lag. Used to score a lag that some other feature chose.
    """
    n_r, n_q = len(ref), len(query)
    r_lo, q_lo = max(0, lag), max(0, -lag)
    overlap = min(n_r, lag + n_q) - r_lo
    if overlap <= 0:
        return 0.0
    a = ref[r_lo : r_lo + overlap]
    b = query[q_lo : q_lo + overlap]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0:
        return 0.0
    return float(abs(np.dot(a, b)) / denom)


#: Alignment features. ``'envelope'`` is coarse-to-fine (see
#: :func:`_envelope_then_waveform`); ``'waveform'`` is raw normalized cross-correlation,
#: correct when both signals come from the SAME source (re-aligning an export against its
#: master) and misleading across devices.
ALIGNMENT_FEATURES = ("envelope", "waveform")


def find_audio_offset_detailed(
    reference_audio: AudioSource,
    query_audio: AudioSource,
    *,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
    feature: str = "waveform",
) -> AudioOffset:
    """Align ``query_audio`` within ``reference_audio`` — offset **and** confidence.

    The detailed twin of :func:`find_audio_offset` (which returns just ``offset_s``).
    Uses an overlap-normalized cross-correlation so the confidence is a scale-invariant
    coefficient in ``[0, 1]`` — usable both as a per-clip trust gate and to compare
    alignments across clips (which the multi-device / multicam case needs). Unlike the
    scalar helper's assumption that ``reference`` is the longer signal, this handles a
    ``query`` that is longer than, or starts before, the reference (negative offset).

    Args:
        reference_audio: The signal to align within (e.g. the clean song).
        query_audio: The signal to locate (e.g. a phone recording of the song).
        sample_rate: Analysis sample rate (mono). 16 kHz gives ~0.06 ms precision.
        min_overlap_ratio: Reject lags overlapping less than this fraction of the
            shorter signal (guards against a tiny-overlap spurious peak).
        feature: Which similarity feature the confidence is measured on — see
            :data:`ALIGNMENT_FEATURES`. Defaults to ``'waveform'`` here because this is the
            low-level primitive and the caller knows their own signals; use ``'envelope'``
            whenever the two recordings came from **different devices**, where a waveform
            coefficient understates a correct alignment several-fold.
            :func:`align_clips_to_reference` — the multi-device primitive — defaults to
            ``'envelope'`` for that reason.

    Returns:
        An :class:`AudioOffset` (``offset_s``, ``confidence``, ``sample_rate``).
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    ref = _load_mono_samples(reference_audio, sample_rate)
    query = _load_mono_samples(query_audio, sample_rate)
    if feature == "envelope":
        lag, coeff = _envelope_then_waveform(
            ref, query, sample_rate, min_overlap_ratio=min_overlap_ratio
        )
    else:
        lag, coeff = _normalized_xcorr(ref, query, min_overlap_ratio=min_overlap_ratio)
    return AudioOffset(
        offset_s=lag / sample_rate, confidence=coeff, sample_rate=sample_rate
    )


@dataclass(frozen=True)
class ClipAlignment:
    """Where one clip sits on a reference (song) timeline.

    Attributes:
        index: The clip's position in the input sequence.
        offset_s: Reference-time where the clip's audio begins (may be negative).
        confidence: Normalized cross-correlation coefficient in ``[0, 1]``.
        duration_s: The clip's own duration (seconds).
        coverage: ``(start_s, end_s)`` — the clip's span **intersected with the
            reference timeline** ``[0, reference_duration]``. When the clip does not
            overlap the reference at all this is a degenerate ``(t, t)`` and
            :attr:`overlaps` is False; callers building an edit must skip those.
        overlaps: Whether the clip intersects the reference timeline at all. A clip that
            does not is still RETURNED, with its measured offset and confidence — see
            :func:`align_clips_to_reference` for why it is not dropped.
    """

    index: int
    offset_s: float
    confidence: float
    duration_s: float
    coverage: tuple[float, float]
    overlaps: bool = True


def align_clips_to_reference(
    reference_audio: AudioSource,
    clips: "Sequence[AudioSource]",
    *,
    reference_duration: float | None = None,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
    feature: str = "envelope",
) -> list[ClipAlignment]:
    """Align a SET of clips to one reference — the multi-device / multicam primitive.

    Aligns each clip against ``reference_audio`` (e.g. the clean song) and returns its
    offset, a scale-invariant confidence, and its **coverage clamped to the reference
    timeline** — so a downstream editor gets valid spans and never references a time the
    reference does not cover. Preserves the original ``index`` so callers can map results
    back to inputs.

    **Every clip gets a record.** A clip with no temporal overlap is returned with
    ``overlaps=False`` rather than omitted, because omission is how a source silently leaves
    the addressable set: a caller that persists this list as *the* alignment artifact ends
    up with material it can no longer reference, name, or explain — the file is still there,
    but nothing downstream can point at it. Selecting what goes into an edit is a matter of
    *referencing* sources and intervals; a source must never disappear from what can be
    referenced as a side effect of being measured. Callers building an edit filter on
    ``overlaps``; callers reporting to a human show all of them, with the reason.

    Args:
        reference_audio: The signal every clip is aligned to (the song).
        clips: The clip audio sources (paths, arrays, or ``AudioSegment``\\ s).
        reference_duration: The reference timeline length (seconds); computed from
            ``reference_audio`` when omitted.
        sample_rate: Analysis sample rate (mono).
        min_overlap_ratio: Passed through to the alignment (see
            :func:`find_audio_offset_detailed`).
        feature: Similarity feature for the confidence — see :data:`ALIGNMENT_FEATURES`.
            Defaults to ``'envelope'`` **because this function's whole purpose is the
            cross-device case**, and a raw-waveform coefficient is not a usable trust gate
            there: two microphones in a room are not sample-correlated even when the
            alignment is exact. Measured on a real 6-device shoot, the waveform coefficient
            scored provably-correct alignments at 0.064-0.148 — below any threshold a
            caller would sensibly set — while the envelope scored them 0.441-0.634 and a
            genuine non-match at 0.102. Pass ``'waveform'`` when the clips come from the
            SAME source as the reference (e.g. verifying an export against its master),
            where sample correlation is meaningful and gives finer confidence resolution.

    Returns:
        A list of :class:`ClipAlignment`, in input order (minus dropped clips).
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    ref = _load_mono_samples(reference_audio, sample_rate)
    ref_dur = (
        reference_duration if reference_duration is not None else len(ref) / sample_rate
    )
    out: list[ClipAlignment] = []
    for i, clip in enumerate(clips):
        query = _load_mono_samples(clip, sample_rate)
        if feature == "envelope":
            lag, coeff = _envelope_then_waveform(
                ref, query, sample_rate, min_overlap_ratio=min_overlap_ratio
            )
        else:
            lag, coeff = _normalized_xcorr(
                ref, query, min_overlap_ratio=min_overlap_ratio
            )
        offset_s = lag / sample_rate
        dur_s = len(query) / sample_rate
        start = max(0.0, offset_s)
        end = min(ref_dur, offset_s + dur_s)
        overlaps = end > start
        out.append(
            ClipAlignment(
                index=i,
                offset_s=offset_s,
                confidence=coeff,
                duration_s=dur_s,
                coverage=(start, end) if overlaps else (start, start),
                overlaps=overlaps,
            )
        )
    return out


#: Default analysis window for :func:`aligned_spans`, in seconds. Long enough that a
#: window of a real recording carries enough structure to correlate, short enough that a
#: span boundary is locatable. This is the knob that trades boundary resolution against
#: cost: halving it doubles the number of correlations.
SPAN_WINDOW_S = 20.0

#: Default hop between windows. Half the window, so every instant of the clip is covered
#: by two windows and a boundary can never fall in the blind spot between them.
SPAN_HOP_S = 10.0

#: A window must reach this for its span to exist at all.
SPAN_MIN_CONFIDENCE = 0.15

#: How far two windows' implied offsets may disagree and still be called the same span.
#: Wider than a frame at any sane rate (a span is not a cut point) but far tighter than
#: the drift a genuine stop/restart introduces.
SPAN_OFFSET_TOLERANCE_S = 0.25

#: How long an UNVERIFIED gap between two same-offset spans may be and still be treated
#: as one take. Defaults to one window.
#:
#: A dozen seconds of silence mid-take collapses the windows inside it to a confidence of
#: zero, so one continuous recording comes back as two spans carrying the SAME offset —
#: measured, and downstream that reads as a stop/restart that never happened. Two spans
#: agreeing on the offset across a short gap are one take with an unusable patch: the
#: camera kept rolling in sync, only the audio stopped saying so.
#:
#: The bound is what stops that from becoming a lie. A clip that records the song, then
#: five minutes of something else, then the song again at the same offset would otherwise
#: be merged into one span claiming those five minutes align. Past this gap the two are
#: reported separately, because correlation cannot tell "quiet" from "different material"
#: and inventing an answer is worse than reporting both.
SPAN_MERGE_GAP_S: float | None = None


@dataclass(frozen=True)
class AlignedSpan:
    """A maximal run of CLIP time that tracks the reference at one stable offset.

    ``align_clips_to_reference`` answers "where does this clip sit?" with a single
    number, which is the right answer only when the clip is one continuous take of the
    reference. A clip that was stopped and restarted, or that contains a chunk of
    something else, has no such number — and the single-offset model does not say so, it
    just returns the offset of whichever part correlated best and describes the rest of
    the clip wrongly.

    Attributes:
        clip_start_s: Where this span begins in the CLIP's own timeline.
        clip_end_s: Where it ends, in the clip's timeline.
        offset_s: Reference-time where this span's clip-time zero would fall — the same
            convention as :attr:`ClipAlignment.offset_s`, so
            ``reference_time = clip_time + offset_s`` holds inside the span.
        confidence: Representative confidence for the span (the median over its windows,
            not the max — a span is only as trustworthy as its typical window).
    """

    clip_start_s: float
    clip_end_s: float
    offset_s: float
    confidence: float

    @property
    def duration_s(self) -> float:
        return self.clip_end_s - self.clip_start_s

    @property
    def reference_span(self) -> "tuple[float, float]":
        """This span's extent on the REFERENCE timeline."""
        return (self.clip_start_s + self.offset_s, self.clip_end_s + self.offset_s)


def aligned_spans(
    reference_audio: AudioSource,
    clip_audio: AudioSource,
    *,
    reference_duration: float | None = None,
    sample_rate: int = 16000,
    window_s: float = SPAN_WINDOW_S,
    hop_s: float = SPAN_HOP_S,
    min_confidence: float = SPAN_MIN_CONFIDENCE,
    offset_tolerance_s: float = SPAN_OFFSET_TOLERANCE_S,
    merge_gap_s: float | None = SPAN_MERGE_GAP_S,
    feature: str = "envelope",
    min_overlap_ratio: float = 0.5,
) -> "list[AlignedSpan]":
    """The MAXIMAL spans of ``clip_audio`` that align to ``reference_audio``.

    The windowed counterpart of :func:`find_audio_offset_detailed`. Where that answers
    "where does this clip sit on the reference?" with one number, this answers "which
    PARTS of it sit there, and at what offset each?" — which is the only answerable
    question for a clip that was stopped and restarted, or that holds material the
    reference does not contain.

    A clip that is one continuous take **and that the correlation can verify
    throughout** returns exactly one span — the compatibility property a caller
    migrating from the single-offset model depends on. The qualification is load-bearing
    in two ways, both measured:

    - A stretch longer than ``merge_gap_s`` that nothing can verify (hard silence, a hand
      over the mic) splits the take into two spans reporting the SAME offset. Two
      same-offset neighbours are the *signal* that this happened, not a stop/restart.
    - **A reference that repeats itself verbatim fragments a take**, because each window
      picks its lag by an independent argmax and a repeated chorus makes two peaks
      near-tied. Measured: a verse/chorus reference split one continuous take into 3
      spans, and one of them reported a WRONG offset at 0.985 confidence. Music is the
      material this is for, so treat a short span whose offset disagrees with its
      neighbours as suspect. Giving the window chooser a continuity prior is the fix and
      is not done here.

    **Boundary resolution is ``window_s``, and no better.** A window is evidence that its
    whole extent aligns; a boundary falling inside a window degrades that window rather
    than locating itself within it. So a returned edge is accurate to roughly ±
    ``window_s``, which is ample for telling two takes apart and NOT enough to cut on.
    A caller who needs a tighter edge pays for it by shrinking ``window_s`` — the cost is
    linear in the number of windows. Stated here because a span that looks like a precise
    interval and is not is exactly the kind of number that gets used as one.

    **Decoded once.** Both signals are loaded through :func:`_load_mono_samples` a single
    time and the windows are slices of the resulting array. Re-decoding per window would
    be N times the cost *and* would reintroduce issue #25: pydub's rate conversion has no
    anti-alias filter, so a per-window decode path is a per-window chance to halve the
    confidence.

    Args:
        reference_audio: The signal to align within (e.g. the clean song).
        clip_audio: The recording to dissect.
        sample_rate: Analysis sample rate (mono).
        window_s: Analysis window — see the resolution note above.
        hop_s: Step between windows. Defaults to half a window, so every instant is
            covered twice and a boundary cannot hide between windows.
        min_confidence: A window must reach this for its span to exist at all.
        offset_tolerance_s: How far a window's implied offset may drift from its span's
            before it is treated as a different take.
        merge_gap_s: How long an unverified gap between two spans that AGREE on the
            offset may be and still be called one take. ``None`` (the default) uses one
            window. See :data:`SPAN_MERGE_GAP_S`.
        reference_duration: The reference timeline's length (seconds); computed from the
            decoded reference when omitted. Spans are trimmed to it, so a returned
            extent is always one the reference can honour. Same keyword, and the same
            purpose, as :func:`align_clips_to_reference`.
        feature: As :func:`find_audio_offset_detailed`. Defaults to ``'envelope'`` for
            the same reason :func:`align_clips_to_reference` does — the cross-device case
            is what this is for.
        min_overlap_ratio: Passed through to the correlation.

    Returns:
        Spans in clip order, non-overlapping, and each lying entirely within
        ``[0, reference_duration]`` on the reference timeline. Empty when nothing in the
        clip aligns — which is a real answer, not a failure.

    >>> import numpy as np                                  # doctest: +SKIP
    >>> spans = aligned_spans(song, phone_recording)        # doctest: +SKIP
    >>> [(round(s.clip_start_s), round(s.offset_s)) for s in spans]   # doctest: +SKIP
    [(0, 12), (95, 240)]
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    if hop_s <= 0 or window_s <= 0:
        raise ValueError(
            f"window_s and hop_s must be positive, got {window_s}, {hop_s}"
        )
    if hop_s > window_s:
        # Not a style rule: with a hop wider than the window, the clip time BETWEEN
        # consecutive windows is never looked at, yet a run spanning them reports one
        # span across the whole range — asserting alignment over instants no
        # correlation ever examined. `merge_gap_s` cannot bound it either, because the
        # gap is inside a run rather than between two spans.
        raise ValueError(
            f"hop_s ({hop_s}) must not exceed window_s ({window_s}): a larger hop "
            "leaves clip time unmeasured, and a span would then assert alignment over "
            "instants no window ever looked at."
        )
    ref = _load_mono_samples(reference_audio, sample_rate)
    clip = _load_mono_samples(clip_audio, sample_rate)
    measured = _window_offsets(
        ref,
        clip,
        sample_rate,
        window_s=window_s,
        hop_s=hop_s,
        feature=feature,
        min_overlap_ratio=min_overlap_ratio,
    )
    spans = _spans_from_windows(
        measured,
        min_confidence=min_confidence,
        offset_tolerance_s=offset_tolerance_s,
        merge_gap_s=window_s if merge_gap_s is None else merge_gap_s,
    )
    ref_dur = (
        len(ref) / sample_rate if reference_duration is None else reference_duration
    )
    return _clamp_to_reference(spans, ref_dur)


def _window_offsets(
    ref: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    *,
    window_s: float,
    hop_s: float,
    feature: str,
    min_overlap_ratio: float,
) -> "list[tuple[float, float, float, float]]":
    """One ``(clip_start_s, clip_end_s, offset_s, confidence)`` per analysis window.

    The offset conversion is the load-bearing line. The correlation reports where the
    WINDOW begins on the reference; the window itself begins ``start_s`` into the clip;
    so the offset the window implies for the clip as a whole is ``lag - start_s``.
    Skipping that subtraction would make every window of a correctly-aligned clip report
    a different offset, and no two of them would ever be grouped.
    """
    n_win = max(1, int(round(window_s * sample_rate)))
    n_hop = max(1, int(round(hop_s * sample_rate)))
    starts = list(range(0, max(1, len(clip) - n_win + 1), n_hop))
    # Cover the tail: without this, up to `window_s` of a clip whose length is not a
    # whole number of hops is never looked at, and a span ending there is truncated.
    if starts and starts[-1] + n_win < len(clip):
        starts.append(max(0, len(clip) - n_win))

    # Computed once for the whole call rather than per window — 2.11x measured, and
    # the reference does not change between windows.
    ref_env = onset_envelope(ref, sample_rate) if feature == "envelope" else None

    out = []
    for s0 in starts:
        win = clip[s0 : s0 + n_win]
        if len(win) == 0:
            continue
        if feature == "envelope":
            lag, coeff = _envelope_then_waveform(
                ref,
                win,
                sample_rate,
                min_overlap_ratio=min_overlap_ratio,
                ref_envelope=ref_env,
            )
        else:
            lag, coeff = _normalized_xcorr(
                ref, win, min_overlap_ratio=min_overlap_ratio
            )
        start_s = s0 / sample_rate
        out.append(
            (
                start_s,
                (s0 + len(win)) / sample_rate,
                lag / sample_rate - start_s,
                coeff,
            )
        )
    return out


def _spans_from_windows(
    measured: "list[tuple[float, float, float, float]]",
    *,
    min_confidence: float,
    offset_tolerance_s: float,
    merge_gap_s: float,
) -> "list[AlignedSpan]":
    """Group per-window measurements into maximal runs of one stable offset.

    A run continues while a window is confident enough AND agrees with the run's offset
    so far, taken as the median of its members — the representative value, and the same
    statistic the span reports.

    An earlier draft also carried a *hysteresis* threshold, on the theory that a window
    dipping just below the opening bar would split a good span. Mutation testing showed
    that guard could not fail, and measuring why is what produced the merge pass below:
    a window degraded by silence does not dip, it collapses to zero — far under any keep
    threshold — so hysteresis was never what decided. Merging same-offset neighbours is.
    """
    runs: list[list[tuple[float, float, float, float]]] = []
    current: list[tuple[float, float, float, float]] = []
    for w in measured:
        _, _, offset, coeff = w
        if current:
            ref_offset = float(np.median([m[2] for m in current]))
            if (
                coeff >= min_confidence
                and abs(offset - ref_offset) <= offset_tolerance_s
            ):
                current.append(w)
                continue
            runs.append(current)
            current = []
        if coeff >= min_confidence:
            current = [w]
    if current:
        runs.append(current)

    spans = [
        AlignedSpan(
            clip_start_s=run[0][0],
            clip_end_s=run[-1][1],
            offset_s=float(np.median([m[2] for m in run])),
            confidence=float(np.median([m[3] for m in run])),
        )
        for run in runs
    ]
    return _merge_same_offset(
        _disjoin(spans), offset_tolerance_s=offset_tolerance_s, merge_gap_s=merge_gap_s
    )


def _clamp_to_reference(
    spans: "list[AlignedSpan]", reference_duration: float
) -> "list[AlignedSpan]":
    """Trim every span to the clip time that actually lands ON the reference.

    A window is admitted while it overlaps the reference by ``min_overlap_ratio`` of
    itself, so a span can run past a reference edge by up to
    ``(1 - min_overlap_ratio) * window_s`` — and the reported extent then describes
    reference time that does not exist. Measured on a 30 s reference:
    ``reference_span == (-10.0, 40.0)``, a 50 s span of a 30 s reference. That is not
    imprecise, it is impossible, and no ``window_s`` tolerance makes it fit.

    The failure it produces downstream is silent and points the wrong way. A caller
    slicing ``ref[int(-10.0 * sr):int(40.0 * sr)]`` gets numpy's negative-index
    resolution: the start wraps to reference time 20 s and the end clamps to 30 s, so
    the caller receives the reference's **tail** for a span whose true content is the
    reference's **entire** length. At the head edge the same slice returns zero samples.
    Neither raises.

    **Both timelines are trimmed, not just the projection.** ``AlignedSpan`` documents
    ``reference_time = clip_time + offset_s`` as holding inside the span; clamping
    ``reference_span`` alone would leave the clip edges wide and make the two views
    disagree. Because the offset is constant within a span the edges move together, so
    trimming is exact — it recovers ground truth on both axes.

    A span left with nothing is dropped: it aligned to no part of the reference, so it
    is not an aligned span. (Unlike ``ClipAlignment``, where a non-overlapping CLIP is
    kept with ``overlaps=False`` because a *source* must never leave the addressable set
    as a side effect of being measured — a span is a measurement, not a source.)

    **Why the clamp alone, and not also a per-window coverage penalty.** The tempting
    root-cause fix is to scale each window's confidence by the fraction of it that lies
    over the reference — because the correlation normalizes by the energy of its
    OVERLAP, so a half-hanging window scores like a whole match (measured: 0.9846
    against 0.9845 for the clean interior windows, i.e. marginally HIGHER). That
    reasoning is sound at the window level and it was implemented; it was then removed,
    because it changed no observable outcome. A span's confidence is the MEDIAN over its
    windows, which absorbs the single edge window — measured, an edge-overrun span
    reported 0.9847 against an interior span's 0.9845 with the penalty applied. The
    extent is fixed here, exactly; a contaminated window is kept out of a run by the
    offset-agreement rule; and a penalty that cannot change an output is decoration.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        lo = max(span.clip_start_s, -span.offset_s)
        hi = min(span.clip_end_s, reference_duration - span.offset_s)
        if hi - lo <= 0:
            continue
        out.append(replace(span, clip_start_s=lo, clip_end_s=hi))
    return out


def _merge_same_offset(
    spans: "list[AlignedSpan]", *, offset_tolerance_s: float, merge_gap_s: float
) -> "list[AlignedSpan]":
    """Rejoin neighbours that agree on the offset across a short unverified gap.

    Two spans carrying the same offset are one take with an unusable patch, not two
    takes: a stop and a restart cannot resume in sync, because whatever the reference
    kept doing while the camera was stopped changes the offset. So a same-offset pair is
    positive evidence of continuity, and reporting it as two spans invents a discontinuity.

    Measured on a continuous 50 s take with 12 s of hard silence at its middle: the
    windows inside the silence score exactly 0.0, the run breaks, and the result is two
    spans BOTH reporting offset 10.00 — a stop/restart that never happened.

    ``merge_gap_s`` is what keeps this honest. The gap is clip time nothing verified, and
    correlation cannot tell "quiet" from "different material"; past one window the two
    spans stay separate rather than the merge claiming a range it never measured.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        if out:
            prev = out[-1]
            gap = span.clip_start_s - prev.clip_end_s
            if (
                abs(span.offset_s - prev.offset_s) <= offset_tolerance_s
                and gap <= merge_gap_s
            ):
                out[-1] = AlignedSpan(
                    clip_start_s=prev.clip_start_s,
                    clip_end_s=span.clip_end_s,
                    # Duration-weighted, so a long span is not dragged by a short one.
                    offset_s=(
                        prev.offset_s * prev.duration_s
                        + span.offset_s * span.duration_s
                    )
                    / max(prev.duration_s + span.duration_s, 1e-9),
                    confidence=min(prev.confidence, span.confidence),
                )
                continue
        out.append(span)
    return out


def _disjoin(spans: "list[AlignedSpan]") -> "list[AlignedSpan]":
    """Trim overlaps so the result partitions the aligned clip time.

    Windows overlap by ``window_s - hop_s``, so the window straddling a boundary belongs
    to both takes and gets assigned to whichever it correlated better with. Left alone
    that makes consecutive spans overlap — measured at 10 s on a 20 s/10 s grid — and a
    clip instant claimed by two different offsets is not a fact about anything.

    **The earlier span's END wins**, rather than splitting the difference. Its last
    window is a positive measurement: it correlated at that offset over its whole extent
    without straddling. The later span's first window is the contaminated one — it can
    score well while containing seconds of the previous take, because half a matching
    window is enough. Measured on a clip with a true boundary at 50 s: first-wins
    reproduces it exactly, while a midpoint split puts both edges 5 s out.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        if out and span.clip_start_s < out[-1].clip_end_s:
            start = out[-1].clip_end_s
            if span.clip_end_s - start <= 0:
                continue  # wholly swallowed by its predecessor
            span = replace(span, clip_start_s=start)
        out.append(span)
    return out
