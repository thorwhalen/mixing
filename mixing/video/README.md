# mixing.video

Video processing and generation tools for Python.

## Overview

This module provides comprehensive video processing capabilities including:
- **AI Video Generation** with Google Vertex AI Veo models
- **Frame Extraction** and video file manipulation
- **Video Concatenation** with various transition effects
- **Subtitle Integration** for video content

## AI Video Generation (`video_gen`)

Generate videos using Google's state-of-the-art Veo models on Vertex AI.

### Quick Start: Generate and Get Video Files

The simplest way to generate videos is with a single function call that returns file paths:

```python
from mixing.video.video_gen import generate_video

# Generate video and get file path in one call
video_path = generate_video("A serene forest at dawn with golden sunlight filtering through mist")
print(f"Video saved to: {video_path}")
```

This automatically handles authentication, generation, and saving to get you a downloadable video file.

### Authentication Setup

Before using video generation, set up Google Cloud authentication:

#### Option 1: Service Account (Recommended for Production)
```bash
# Download service account JSON from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

#### Option 2: Application Default Credentials (Development)
```bash
# Use gcloud CLI for local development
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

#### Option 3: Environment Variables (Custom)
```bash
# Alternative environment variable names
export VEO_SERVICE_ACCOUNT_FILE="/path/to/service-account.json"
export VEO_PROJECT_ID="your-project-id"
export VEO_LOCATION="us-central1"  # Optional, defaults to us-central1
```

### Basic Video Generation

```python
from mixing.video.video_gen import generate_video

# Generate video and get file path (default behavior)
video_path = generate_video("A serene forest at dawn with golden sunlight filtering through mist")
print(f"Video saved to: {video_path}")

# Save to specific location
video_path = generate_video("A butterfly in a garden", save_video="/path/to/my_video.mp4")

# Save to directory with auto-generated name
video_path = generate_video("Ocean waves at sunset", save_video="/output/directory/")
```

### Flexible Egress Options

The `save_video` parameter controls how generated videos are processed:

```python
# Default: Auto-save to temp files and return path(s)
path = generate_video("Forest scene")

# Save with specific extension to temp files
path = generate_video("Forest scene", save_video=".webm")

# Just get the raw operation (no saving)
operation = generate_video("Forest scene", save_video=False)
# or: operation = generate_video("Forest scene", save_video=lambda x: x)

# Custom processing function
def my_video_processor(operation):
    # Save with custom prefix and settings
    from mixing.video.video_gen import save_generated_videos
    paths = save_generated_videos(operation, prefix="custom_", extension_fallback="webm")
    print(f"Processed {len(paths)} videos with custom settings")
    return paths

result = generate_video("Forest scene", save_video=my_video_processor)
```

### Advanced Video Generation Examples

#### Text-to-Video with Custom Parameters
```python
video_path = generate_video(
    prompt="A butterfly gracefully flying through a flower garden",
    model="veo-2.0-generate-001",
    aspect_ratio="16:9",
    duration_seconds=8,
    save_video="/my/custom/video.mp4",
    project_id="your-project-id"  # Override environment
)
```

#### Image-to-Video Generation
```python
# Use an image as the starting frame
video_path = generate_video(
    prompt="The camera slowly zooms out revealing a vast landscape",
    first_frame="/path/to/start_image.jpg",
    save_video="/output/zoom_out.mp4"
)
```

#### Video-to-Video Generation (Frame-based)
```python
# Use frames from existing videos as start/end points
video_path = generate_video(
    prompt="Smooth transition with swirling particles",
    first_frame="/path/to/video1.mp4",  # Extracts last frame
    last_frame="/path/to/video2.mp4",   # Extracts first frame
    save_video="/output/transition.mp4"
)
```

#### Multiple Model Output
```python
# Some models generate multiple video variations
video_paths = generate_video(
    prompt="A cosmic dance of stars and galaxies",
    model="veo-3.0-generate-001",  # May generate multiple results
    save_video="/output/directory/"  # Auto-indexed: cosmic_00.mp4, cosmic_01.mp4, etc.
)
print(f"Generated {len(video_paths)} videos: {video_paths}")
```

#### Advanced Processing Workflow
```python
# Get raw operation for custom processing
def advanced_processor(operation):
    from mixing.video.video_gen import save_generated_videos
    
    # Save multiple formats
    mp4_paths = save_generated_videos(operation, "/output/mp4/", extension_fallback="mp4")
    webm_paths = save_generated_videos(operation, "/output/webm/", extension_fallback="webm") 
    
    # Log generation details
    print(f"Generated {len(operation.response.generated_videos)} videos")
    for i, video in enumerate(operation.response.generated_videos):
        print(f"Video {i}: {getattr(video, 'mime_type', 'unknown')} format")
    
    return {"mp4": mp4_paths, "webm": webm_paths}

result = generate_video("Epic landscape timelapse", save_video=advanced_processor)
```

### Standalone Video Saving (save_generated_videos)

For more control over the saving process, you can also use `save_generated_videos` directly:

```python
# Get raw operation without auto-saving
operation = generate_video("Forest scene", save_video=False)

# Then save with full control
from mixing.video.video_gen import save_generated_videos

# Various saving options
paths = save_generated_videos(operation)  # Auto temp files
paths = save_generated_videos(operation, "/output/dir/")  # Directory
paths = save_generated_videos(operation, "/my/video.mp4")  # Specific file
paths = save_generated_videos(operation, ".webm")  # Extension

# Advanced options
paths = save_generated_videos(
    operation,
    "/output/",
    prefix="animation_",
    directory_name="my_video_",
    extension_fallback="mp4"
)
```

### Input Format Flexibility

The functions accept various input types:

```python
# Single GeneratedVideo object
save_generated_videos(video_obj)

# List of GeneratedVideo objects
save_generated_videos([video1, video2, video3])

# Operation response or full operation
save_generated_videos(operation.response.generated_videos)
save_generated_videos(operation)
```

### Error Handling and Troubleshooting

The module provides helpful error messages for common issues:

```python
try:
    video_path = generate_video("A magical forest scene")
    print(f"Video ready: {video_path}")
except Exception as e:
    # Automatic authentication help will be displayed
    # for authentication-related errors
    print(f"Error: {e}")
```

Common issues and solutions:
- **Authentication errors**: Check service account file and project ID
- **API not enabled**: Enable Vertex AI API in Google Cloud Console
- **Quota exceeded**: Check your Vertex AI quotas and usage limits
- **Invalid prompts**: Ensure prompts comply with content policies

## Frame Extraction (`video_files`)

### VideoFrames: Dictionary-like access to video frames

`VideoFrames` provides a Mapping interface to access individual frames from a video file by index. Frames are returned as numpy arrays (BGR format).

```python
from mixing.video.video_files import VideoFrames

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
from mixing.video.video_files import save_frame

# Save first frame with auto-generated filename (video_0.png)
save_frame("my_video.mp4")

# Save frame 100 as JPEG
save_frame("my_video.mp4", 100, saveas=".jpg")  # Saves as my_video_000100.jpg

# Save last frame to specific path
save_frame("my_video.mp4", -1, saveas="output/last_frame.png")

# Save to temporary file (persists)
temp_path = save_frame("my_video.mp4", 0, saveas="/TMP")
print(f"Frame saved to: {temp_path}")
```

The `saveas` parameter accepts:
- `None` (default): Auto-generate filename in video's directory
- `.ext`: Use this extension with auto-generated filename
- `/TMP`: Save to temporary directory (persists)
- Full path: Save to this specific location

## Video Processing (`video_subtitles`)

### write_subtitles_in_video

Write subtitles in a video.

Example usage:

```python
from mixing.video.video_subtitles import write_subtitles_in_video

# Auto-detect subtitle file
output_path = write_subtitles_in_video("~/Downloads/some_video.mp4") 
```

Which is syntactic sugar for the more explicit:

```python
output_path = write_subtitles_in_video(
    "~/Downloads/some_video.mp4", 
    subtitles="~/Downloads/some_video.srt",
    output_video="~/Downloads/some_video_with_subs.mp4"
)  
```

## Video Concatenation (`video_concat`)

Combine multiple videos with various transition effects and automatic dimension normalization:

### Basic Concatenation

```python
from mixing.video import concatenate_videos

# Simple concatenation from a folder (auto-normalizes dimensions)
final_video = concatenate_videos(
    "/path/to/videos/",
    output_path="/output/combined.mp4"
)
```

### Dimension Normalization

When concatenating videos with different dimensions (width/height), `concatenate_videos` automatically normalizes them. The default mode uses a **'social media style'** with blurred/zoomed background (like Instagram/TikTok):

```python
# Default: Social media style (blurred background)
concatenate_videos(
    [video1, video2, video3],  # Videos with different dimensions
    output_path="output.mp4"
    # normalize_dimensions='social' is the default
)
```

**Available normalization methods:**

- **`'social'`** (default): Scales video to fit with blurred/zoomed background fill
- **`'fit'`**: Letterbox/pillarbox - scale to fit with black bars
- **`'fill'`**: Scale to fill (may crop edges)
- **`'stretch'`**: Stretch to fit (may distort aspect ratio)
- **`False`**: No normalization (may cause issues if dimensions differ)

```python
# Letterbox style (black bars on sides/top-bottom)
concatenate_videos(
    videos,
    output_path="letterbox.mp4",
    normalize_dimensions='fit'
)

# Fill mode (may crop edges to avoid distortion)
concatenate_videos(
    videos,
    output_path="filled.mp4",
    normalize_dimensions='fill'
)

# Explicit target dimensions
concatenate_videos(
    videos,
    target_width=1920,
    target_height=1080,
    normalize_dimensions='social',
    output_path="1080p.mp4"
)
```

### Using Resize Utilities Separately

You can also use the dimension utilities independently for single videos:

```python
from mixing.video import resize_to_dimensions, get_video_dimensions
from moviepy import VideoFileClip

# Check video dimensions
video = VideoFileClip("my_video.mp4")
width, height = get_video_dimensions(video)
print(f"Original: {width}x{height}")

# Resize with social media style
resized = resize_to_dimensions(
    video,
    1920, 1080,
    method='social'
)
resized.write_videofile("resized_social.mp4")

# Or use letterbox
resized = resize_to_dimensions(
    video,
    1920, 1080,
    method='fit',
    bg_color=(0, 0, 0)  # Black background
)
```

### Transition Effects

Add smooth transitions between concatenated videos:

```python
from mixing.video.video_concat import trim_and_crossfade

# Concatenate with crossfade transitions
final = concatenate_videos(
    videos,
    transform_clips=lambda clips: trim_and_crossfade(clips, duration=0.4),
    output_path="smooth_transitions.mp4"
)
```

## Requirements

### Google Cloud Setup (for AI Video Generation)

1. **Enable APIs**: Enable the Vertex AI API in your Google Cloud project
2. **Authentication**: Set up service account or application default credentials
3. **Quotas**: Ensure sufficient Vertex AI quotas for video generation
4. **Billing**: Video generation requires a billing-enabled project

### FFmpeg

Many video processing tools require `ffmpeg`. Installation instructions:

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows
Download from [FFmpeg official site](https://ffmpeg.org/download.html#build-windows) and add to PATH.

## API Reference

For detailed API documentation, see the docstrings in individual modules:
- `video_gen.py`: AI video generation with Veo
- `video_files.py`: Frame extraction and video file utilities  
- `video_subtitles.py`: Subtitle processing and video utilities
- `video_concat.py`: Video concatenation and transitions

