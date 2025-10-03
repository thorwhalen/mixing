"""Video util"""

from typing import Optional
from pathlib import Path
import os

import moviepy as mp

from config2py import process_path


def to_srt_time(seconds):
    """
    Convert seconds to SRT time format.

    Example usage:

    >>> to_srt_time(1.5)
    '00:00:01,500'

    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def generate_subtitle_clips(subtitles, video_clip, fontsize=24, color="white"):
    subtitle_clips = []
    for subtitle in subtitles.split("\n\n"):
        lines = subtitle.split("\n")
        if len(lines) >= 3:
            time_info = lines[1].split(" --> ")
            start_time = time_info[0].replace(",", ".")
            end_time = time_info[1].replace(",", ".")
            text = " ".join(lines[2:])
            txt_clip = mp.TextClip(
                text,
                fontsize=fontsize,
                color=color,
                size=(video_clip.w, None),
                method="caption",
            )
            txt_clip = (
                txt_clip.set_start(start_time)
                .set_end(end_time)
                .set_position(("center", "bottom"))
            )
            subtitle_clips.append(txt_clip)
    return subtitle_clips


Filepath = str


def write_subtitles_in_video(
    video: Filepath, subtitles: Optional[str] = None, output_video: Optional[str] = None
):
    """
    Write subtitles in a video.

    Example usage:

    >>> output_path = write_subtitles_in_video("~/Downloads/some_video.mp4")  # doctest: +SKIP

    Which is syntactic sugar for the more explicit:

    >>> output_path = write_subtitles_in_video(
    ...     "~/Downloads/some_video.mp4",
    ...     subtitles="~/Downloads/Ssome_video.srt",
    ...     output_video="~/Downloads/some_video.mp4"
    ... )  # doctest: +SKIP

    """
    video_path = process_path(video)

    srt_content, output_video_path = _process_inputs(
        subtitles, output_video, video_path
    )

    # Load the video file
    video_clip = mp.VideoFileClip(str(video_path))

    # Generate subtitle clips
    subtitle_clips = generate_subtitle_clips(srt_content, video_clip)

    # Create a composite video with subtitles
    video_with_subtitles = mp.CompositeVideoClip([video_clip] + subtitle_clips)

    # Export the video with subtitles
    video_with_subtitles.write_videofile(
        str(output_video_path), codec="libx264", fps=video_clip.fps
    )

    return output_video_path


def _process_inputs(subtitles, output_video, video_path):
    if subtitles is None:
        subtitles_path = video_path.with_suffix(".srt")
        srt_content = subtitles_path.read_text()
    elif os.path.isfile(subtitles):
        subtitles_path = process_path(subtitles)
        srt_content = subtitles_path.read_text()
    else:
        assert isinstance(subtitles, str), "subtitles should be a string"
        srt_content = subtitles

    if output_video is None:
        output_video_path = video_path.with_stem(video_path.stem + "_with_subtitles")
        output_video_path = output_video_path.with_suffix(video_path.suffix)
    else:
        output_video_path = Path(output_video)
    return srt_content, output_video_path
