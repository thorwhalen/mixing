"""Audio mixing and editing functionality.

Main exports:
- Audio: Sliceable audio interface with time-based operations
- fade_in, fade_out: Apply fade effects
- crop_audio: Trim audio segments
- concatenate_audio: Join multiple audio files
- overlay_audio: Mix/overlay audio tracks
- save_audio_clip: Extract and save audio segments

Examples:
    >>> from mixing.audio import Audio, fade_in, concatenate_audio  # doctest: +SKIP
    >>> audio = Audio("song.mp3")  # doctest: +SKIP
    >>> segment = audio[10:30]  # 10s to 30s  # doctest: +SKIP
    >>> segment.save("clip.mp3")  # doctest: +SKIP

    >>> faded = fade_in("intro.mp3", duration=2.0)  # doctest: +SKIP
    >>> combined = concatenate_audio("part1.mp3", "part2.mp3", "part3.mp3")  # doctest: +SKIP
"""

from .audio_ops import (
    Audio,
    AudioSamples,
    fade_in,
    fade_out,
    crop_audio,
    concatenate_audio,
    overlay_audio,
    save_audio_clip,
)

__all__ = [
    "Audio",
    "AudioSamples",
    "fade_in",
    "fade_out",
    "crop_audio",
    "concatenate_audio",
    "overlay_audio",
    "save_audio_clip",
]
