# mixing

Tools for video and audio editing

To install:	```pip install mixing```

# Quick Start

```python
# Audio editing
from mixing.audio import Audio, fade_in, concatenate_audio

audio = Audio("song.mp3")
segment = audio[10:30]  # Get 10s-30s segment
faded = segment.fade_in(2).fade_out(3)  # Apply fades
faded.save("edited.mp3")

# Video editing
from mixing.video import Video, loop_video, replace_audio

video = Video("clip.mp4")
trimmed = video[5:15]  # Get 5s-15s segment
looped = loop_video("intro.mp4", n_loops=3)  # Repeat 3 times
mixed = replace_audio("video.mp4", "music.mp3", mix_ratio=0.7)  # Mix audio
```

# Examples

## mixing.audio (NEW!)

### Audio: Sliceable audio editing interface

The `Audio` class provides a clean, Pythonic interface for audio editing with slice notation and chainable operations.

```python
from mixing.audio import Audio

# Load audio file
audio = Audio("song.mp3")

# Slice audio (lazy evaluation - no copying)
intro = audio[0:10]        # First 10 seconds
chorus = audio[30:60]       # 30s to 60s
outro = audio[-10:]         # Last 10 seconds

# Use different time units
audio_samples = Audio("song.mp3", time_unit="samples")
segment = audio_samples[0:44100]  # First 44100 samples (1s at 44.1kHz)

audio_ms = Audio("song.mp3", time_unit="milliseconds")
segment = audio_ms[0:5000]  # First 5000ms (5 seconds)

# Chain operations
edited = audio[10:120].fade_in(2).fade_out(3)
edited.save("edited.mp3")

# Access properties
print(f"Duration: {audio.duration}s")
print(f"Sample rate: {audio.sample_rate}Hz")
print(f"Channels: {audio.channels}")
```

### Audio Editing Functions

```python
from mixing.audio import fade_in, fade_out, crop_audio, concatenate_audio, overlay_audio

# Apply fade effects
faded_in = fade_in("song.mp3", duration=2.0)
faded_out = fade_out("song.mp3", duration=3.0)

# Crop/trim audio
crop_audio("song.mp3", start=10, end=30, output_path="segment.mp3")

# Concatenate multiple audio files
combined = concatenate_audio(
    "intro.mp3", 
    "main.mp3", 
    "outro.mp3",
    crossfade=0.5  # 500ms crossfade between segments
)
combined.save("full_song.mp3")

# Overlay/mix audio tracks
# mix_ratio controls the balance: 0.0=only background, 1.0=only overlay, 0.5=equal mix
mixed = overlay_audio(
    background="music.mp3",
    overlay="voice.mp3",
    position=5.0,      # Start overlay at 5 seconds
    mix_ratio=0.7      # 70% overlay, 30% background
)
mixed.save("mixed.mp3")
```

### AudioSamples: Sample-level access

Access individual audio samples through a Mapping interface:

```python
audio = Audio("song.mp3")
samples = audio.samples

# Access individual samples
first_sample = samples[0]
last_sample = samples[-1]

# Slice samples
sample_range = samples[1000:2000]  # Returns numpy array

# Get properties
print(f"Total samples: {len(samples)}")
```

## mixing.video

### Video: Sliceable video editing interface (Enhanced!)

```python
from mixing.video import Video

# Load and slice video
video = Video("movie.mp4")
clip = video[10:30]  # Get 10s-30s segment
clip.save("clip.mp4")

# Use frame numbers
video_frames = Video("movie.mp4", time_unit="frames")
segment = video_frames[100:500]  # Frames 100-500

# Extract single frames
frame = video[15]  # Returns numpy array (frame at 15s)
```

### NEW: Loop Video

Repeat a video multiple times:

```python
from mixing.video import loop_video

# Loop video 3 times
looped = loop_video("intro.mp4", n_loops=3)
print(f"Looped video saved to: {looped}")

# Custom output path
looped = loop_video("clip.mp4", n_loops=5, output_path="extended_clip.mp4")
```

### NEW: Replace/Mix Video Audio

Replace or mix audio in videos with fine control:

```python
from mixing.video import replace_audio

# Complete audio replacement
replace_audio("video.mp4", "new_music.mp3", mix_ratio=1.0)

# Equal mix of original and new audio
replace_audio("video.mp4", "music.mp3", mix_ratio=0.5)

# Mostly new audio with some original (70% new, 30% original)
replace_audio("video.mp4", "voice.mp3", mix_ratio=0.7)

# Keep only original audio (no change)
replace_audio("video.mp4", "music.mp3", mix_ratio=0.0)

# Auto-adjust audio length to match video
replace_audio(
    "video.mp4", 
    "short_music.mp3",
    mix_ratio=1.0,
    normalize_audio=True  # Loops/trims audio to match video length
)
```

### NEW: Normalize Audio Levels

Reduce volume fluctuations in video audio (perfect for narration with varying volume):

```python
from mixing.video import normalize_audio

# Normalize audio in a video with varying narrator volume
normalize_audio("lecture.mp4")  # Creates lecture_normalized.mp4

# Custom output path
normalize_audio("interview.mp4", output_path="interview_fixed.mp4")

# The function adjusts audio so loudest parts reach a consistent level,
# reducing variation between quiet and loud sections
```

### VideoFrames: Dictionary-like access to video frames

`VideoFrames` provides a Mapping interface to access individual frames from a video file by index. Frames are returned as numpy arrays (BGR format).

```python
from mv.util import VideoFrames

# Create frame accessor
frames = VideoFrames("my_video.mp4")

# Access frames by index
first_frame = frames[0]
last_frame = frames[-1]  # Negative indexing supported
middle_frame = frames[len(frames) // 2]

# Slice to get multiple frames (returns an iterator, not a list)
for frame in frames[10:50]:
    # Process frames 10 through 49
    pass

# Get every 5th frame
for frame in frames[::5]:
    # Process every 5th frame
    pass
```

### save_frame: Extract and save video frames as images

Convenience function to extract a single frame from a video and save it as an image.

```python
from mv.util import save_frame

# Save first frame with auto-generated filename (video_0.png)
save_frame("my_video.mp4")

# Save frame 100 as JPEG
save_frame("my_video.mp4", 100, saveas=".jpg")  # Saves as my_video_000100.jpg

# Save last frame to specific path
save_frame("my_video.mp4", -1, saveas="output/last_frame.png")

```

The `saveas` parameter accepts:
- `None` (default): Auto-generate filename in video's directory
- `.ext`: Use this extension with auto-generated filename
- Full path: Save to this specific location

### write_subtitles_in_video


Write subtitles in a video.

Example usage:

```python
>>> from mixing import write_subtitles_in_video
>>> output_path = write_subtitles_in_video("~/Downloads/some_video.mp4") 
```

Which is syntactic sugar for the more explicit:

```python
>>> output_path = write_subtitles_in_video(
...     "~/Downloads/some_video.mp4", 
...     subtitles="~/Downloads/some_video.srt",
...     output_video="~/Downloads/some_video.mp4"
... )  
```

### AI Video Generation with Google Vertex AI Veo

Generate videos using Google's state-of-the-art Veo models. The simplest approach generates videos and returns file paths in one call.

#### Quick Start

```python
from mixing.video.video_gen import generate_video

# Generate video and get file path in one call
video_path = generate_video("A serene forest at dawn with golden sunlight filtering through mist")
print(f"Video saved to: {video_path}")
```

#### Authentication Setup

Before using video generation, set up Google Cloud authentication:

```bash
# Service account (recommended for production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or use gcloud CLI (for development)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

#### Advanced Examples

```python
# Save to specific location
video_path = generate_video(
    prompt="The camera slowly zooms out revealing a vast landscape",
    first_frame="/path/to/image.jpg",
    save_video="/my/custom/video.mp4",
    duration_seconds=8
)

# Video-to-video: Use frames from existing videos
video_path = generate_video(
    prompt="Smooth transition with swirling particles",
    first_frame="video1.mp4",  # Uses last frame as start
    last_frame="video2.mp4",   # Uses first frame as end
    model="veo-2.0-generate-001"
)

# Handle multiple generated videos (some models create variations)
video_paths = generate_video(
    "A magical forest with dancing fireflies",
    save_video="/output/directory/"  # Auto-indexed files
)

# Get raw operation for custom processing
operation = generate_video("Epic landscape", save_video=False)
# Process operation.response.generated_videos as needed
```

#### Flexible Egress Control

The `save_video` parameter controls how videos are processed:

```python
# Default: Auto-save to temp files and return path(s)
path = generate_video("prompt")

# Save to specific file or directory
path = generate_video("prompt", save_video="/path/to/video.mp4")
path = generate_video("prompt", save_video="/output/dir/")

# Just get the operation (no saving)
op = generate_video("prompt", save_video=False)

# Custom processing function
def custom_processor(operation):
    # Your custom logic here
    return processed_result

result = generate_video("prompt", save_video=custom_processor)
```


# Further requirements

## Google Cloud Setup (for AI Video Generation)

To use the AI video generation features:

1. **Google Cloud Project**: Create a project with billing enabled
2. **Enable APIs**: Enable the Vertex AI API in your project  
3. **Authentication**: Set up service account credentials or use application default credentials
4. **Quotas**: Ensure sufficient Vertex AI quotas for video generation

For detailed setup instructions, see the [video generation authentication guide](mixing/video/README.md#authentication-setup).

## FFmpeg

Many of the tools also require `ffmeg`. 

To install FFmpeg on your system, follow the instructions for your operating system below.

### macOS

1. **Using Homebrew:**
   - Open Terminal.
   - Run the following command:
     ```bash
     brew install ffmpeg
     ```

For more details, visit the [FFmpeg installation page for macOS](https://ffmpeg.org/download.html#build-mac).

### Linux

1. **Using the package manager:**
   - For Debian/Ubuntu-based distributions, run:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - For Fedora, run:
     ```bash
     sudo dnf install ffmpeg
     ```
   - For Arch Linux, run:
     ```bash
     sudo pacman -S ffmpeg
     ```

For more details, visit the [FFmpeg installation page for Linux](https://ffmpeg.org/download.html#build-linux).

### Windows

1. **Using Windows builds:**
   - Download the executable from [FFmpeg for Windows](https://ffmpeg.org/download.html#build-windows).
   - Extract the downloaded files and add the `bin` directory to your system's PATH.

For more details, visit the [FFmpeg installation page for Windows](https://ffmpeg.org/download.html#build-windows).


# Optional Dependencies

For additional functionality, you can install optional dependencies:

```bash
# For testing
pip install mixing[testing]

# For clipboard functionality (get file paths from clipboard)
pip install mixing[clipboard]

# For audio editing functionality
pip install mixing[audio]

# Install multiple extras
pip install mixing[testing,clipboard,audio]
```

## Audio Editing Requirements

The audio editing features require:
- **pydub**: Python audio manipulation library
- **ffmpeg**: Audio/video processing (see installation below)

Install audio extras:
```bash
pip install mixing[audio]
```
