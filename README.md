
# mixing
Tools for video and audio editing


To install:	```pip install mixing```


# Examples

## mixing.video

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
