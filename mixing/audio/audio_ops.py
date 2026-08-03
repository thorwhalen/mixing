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
from dataclasses import dataclass
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


def _load_mono_samples(source: AudioSource, sample_rate: int) -> np.ndarray:
    """Load any :data:`AudioSource` as a mono float64 array at ``sample_rate``."""
    seg = _normalize_audio_source(source, target_type="AudioSegment")
    seg = seg.set_channels(1).set_frame_rate(sample_rate)
    return np.array(seg.get_array_of_samples(), dtype=np.float64)


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
    coeff = np.where(denom > 0, num / denom, 0.0)
    valid = overlap >= max(1.0, min_overlap_ratio * min(n_r, n_q))
    scored = np.where(valid, np.abs(coeff), -1.0)
    best = int(np.argmax(scored))
    return int(lags[best]), float(np.abs(coeff[best]))


def find_audio_offset_detailed(
    reference_audio: AudioSource,
    query_audio: AudioSource,
    *,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
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

    Returns:
        An :class:`AudioOffset` (``offset_s``, ``confidence``, ``sample_rate``).
    """
    ref = _load_mono_samples(reference_audio, sample_rate)
    query = _load_mono_samples(query_audio, sample_rate)
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
            reference timeline** ``[0, reference_duration]``. Empty-coverage clips
            (no temporal overlap with the reference) are dropped by
            :func:`align_clips_to_reference`, so ``end_s > start_s`` always holds here.
    """

    index: int
    offset_s: float
    confidence: float
    duration_s: float
    coverage: tuple[float, float]


def align_clips_to_reference(
    reference_audio: AudioSource,
    clips: "Sequence[AudioSource]",
    *,
    reference_duration: float | None = None,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
) -> list[ClipAlignment]:
    """Align a SET of clips to one reference — the multi-device / multicam primitive.

    Aligns each clip against ``reference_audio`` (e.g. the clean song) and returns its
    offset, a scale-invariant confidence, and its **coverage clamped to the reference
    timeline** — so a downstream editor gets valid spans and never references a time the
    reference does not cover. Clips with no temporal overlap with the reference are
    dropped. Preserves the original ``index`` so callers can map results back to inputs.

    Args:
        reference_audio: The signal every clip is aligned to (the song).
        clips: The clip audio sources (paths, arrays, or ``AudioSegment``\\ s).
        reference_duration: The reference timeline length (seconds); computed from
            ``reference_audio`` when omitted.
        sample_rate: Analysis sample rate (mono).
        min_overlap_ratio: Passed through to the alignment (see
            :func:`find_audio_offset_detailed`).

    Returns:
        A list of :class:`ClipAlignment`, in input order (minus dropped clips).
    """
    ref = _load_mono_samples(reference_audio, sample_rate)
    ref_dur = (
        reference_duration if reference_duration is not None else len(ref) / sample_rate
    )
    out: list[ClipAlignment] = []
    for i, clip in enumerate(clips):
        query = _load_mono_samples(clip, sample_rate)
        lag, coeff = _normalized_xcorr(ref, query, min_overlap_ratio=min_overlap_ratio)
        offset_s = lag / sample_rate
        dur_s = len(query) / sample_rate
        start = max(0.0, offset_s)
        end = min(ref_dur, offset_s + dur_s)
        if end <= start:  # no overlap with the reference timeline
            continue
        out.append(
            ClipAlignment(
                index=i,
                offset_s=offset_s,
                confidence=coeff,
                duration_s=dur_s,
                coverage=(start, end),
            )
        )
    return out
