"""
Example demonstrating audio-based subtitle alignment.

This shows how to use the new audio detection features to automatically
align subtitles with when speech actually starts in a video.
"""

from mixing.video.video_util import (
    fix_srt_file,
    auto_shift_srt_to_start,
    find_audio_start_offset,
    _find_audio_peaks,
)


def example_basic_shift():
    """Basic usage: shift subtitles to start at 0."""
    print("=" * 60)
    print("Example 1: Basic shift to 0")
    print("=" * 60)

    srt_content = """1
00:00:10,000 --> 00:00:12,000
Hello world

2
00:00:15,000 --> 00:00:17,000
Second subtitle"""

    shifted = auto_shift_srt_to_start(srt_content)
    print("Original first subtitle: 00:00:10,000")
    print("Shifted first subtitle: 00:00:00,000")
    print()


def example_audio_based_shift(video_path: str):
    """Audio-based shift: align with detected speech start."""
    print("=" * 60)
    print("Example 2: Audio-based alignment")
    print("=" * 60)

    # First, let's see what the audio analysis finds
    print("Analyzing audio...")
    audio_start = find_audio_start_offset(video_path, first_n_seconds_to_sample=20.0)
    print(f"Detected audio starts at: {audio_start:.2f}s")
    print()

    # Now align subtitles with the detected audio start
    srt_content = """1
00:00:10,000 --> 00:00:12,000
Hello world

2
00:00:15,000 --> 00:00:17,000
Second subtitle"""

    shifted = auto_shift_srt_to_start(
        srt_content,
        video_path=video_path,
        auto_detect_audio_start=True,
    )
    print("Result: Subtitles now aligned with audio start!")
    print()


def example_analyze_audio_peaks(video_path: str):
    """Detailed analysis: examine audio peaks directly."""
    print("=" * 60)
    print("Example 3: Detailed audio peak analysis")
    print("=" * 60)

    peaks = _find_audio_peaks(
        video_path,
        first_n_seconds_to_sample=10.0,
        window_size=0.1,
        min_peak_threshold=0.02,
    )

    print(f"Found {len(peaks)} audio peaks in first 10 seconds:")
    print()

    # Show first few peaks
    for i, peak in enumerate(peaks[:5]):
        print(f"Peak {i+1}:")
        print(f"  Time: {peak['time']:.2f}s")
        print(f"  Intensity: {peak['intensity']:.4f}")
        print(f"  Variation: {peak['variation']:.4f}")

    if len(peaks) > 5:
        print(f"... and {len(peaks) - 5} more peaks")
    print()


def example_fix_srt_file_with_audio(video_path: str, srt_path: str):
    """Complete workflow: fix SRT file with audio detection."""
    print("=" * 60)
    print("Example 4: Fix SRT file with audio detection")
    print("=" * 60)

    # Option 1: Basic fix (shift to 0)
    output_path = fix_srt_file(srt_path)
    print(f"Basic fix saved to: {output_path}")
    print()

    # Option 2: Audio-based alignment
    output_path = fix_srt_file(
        srt_path,
        video_path=video_path,
        auto_detect_audio_start=True,
    )
    print(f"Audio-aligned version saved to: {output_path}")
    print()


if __name__ == "__main__":
    # Basic examples that don't need a video file
    example_basic_shift()

    # To run audio-based examples, provide your video path:
    # video_path = "/path/to/your/video.mp4"
    # srt_path = "/path/to/your/subtitles.srt"
    #
    # example_audio_based_shift(video_path)
    # example_analyze_audio_peaks(video_path)
    # example_fix_srt_file_with_audio(video_path, srt_path)

    print("\nℹ️  To run audio-based examples, uncomment the code above")
    print("   and provide paths to your video and SRT files.")
