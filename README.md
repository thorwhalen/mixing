
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


# Further requirements

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
