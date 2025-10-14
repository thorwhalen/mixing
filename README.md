
# mixing
Tools for video and audio editing


To install:	```pip install mixing```


# Examples

## mixing.video

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
