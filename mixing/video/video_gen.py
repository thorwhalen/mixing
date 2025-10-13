"""Video gen utils"""

import time
import base64
import mimetypes
import os


def _is_video_file(path: str) -> bool:
    """Check if a file has a video extension."""
    video_extensions = {
        '.mp4',
        '.avi',
        '.mkv',
        '.mov',
        '.wmv',
        '.flv',
        '.webm',
        '.m4v',
        '.3gp',
        '.mpg',
        '.mpeg',
    }
    return os.path.splitext(path.lower())[1] in video_extensions


def _get_frame_path(media_path: str, frame_idx: int = 0) -> str:
    """
    Get frame path from media file. If it's a video, extract frame to temp file.
    If it's already an image, return the original path.
    """
    if _is_video_file(media_path):
        from .video_files import save_frame

        return save_frame(media_path, frame_idx, saveas='/TMP')
    else:
        return media_path


def _read_image_as_base64(path: str):
    """Read a local image file and return (data: str, mime_type: str)."""
    with open(path, "rb") as f:
        b = f.read()
    data_b64 = base64.b64encode(b).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"
    return data_b64, mime_type


def generate_video(
    prompt: str,
    first_frame: str | None = None,
    last_frame: str | None = None,
    model: str = "veo-2.0-generate-001",  # TODO: manage models better (when aix offers model routing tools)
    aspect_ratio: str = "16:9",
    duration_seconds: int = 5,
    output_gcs_uri: str | None = None,
):
    """
    Generate a video via Veo (Vertex AI) given an optional first and last frame.
    Returns the long-running operation result.
    """
    # TODO: manage models better (when aix offers model routing tools)
    from google import genai
    from google.genai.types import GenerateVideosConfig, Image

    client = genai.Client(vertexai=True)

    # Prepare Image objects if frames provided
    image_obj = None
    last_image_obj = None

    if first_frame:
        # Get first frame (0) if it's a video, otherwise use as-is
        frame_path = _get_frame_path(first_frame, frame_idx=0)
        b64, mime = _read_image_as_base64(frame_path)
        image_obj = Image(
            mime_type=mime,
            data=b64,
        )
    if last_frame:
        # Get last frame (-1) if it's a video, otherwise use as-is
        frame_path = _get_frame_path(last_frame, frame_idx=-1)
        b642, mime2 = _read_image_as_base64(frame_path)
        last_image_obj = Image(
            mime_type=mime2,
            data=b642,
        )

    # Build config
    config = GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        output_gcs_uri=output_gcs_uri,
    )

    # Build request
    # Note: the SDK signature may vary; check your installed version of google-genai
    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image_obj,
        last_frame=last_image_obj,
        config=config,
    )

    # Poll until done
    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)

    return operation


# if __name__ == "__main__":
#     # Example usage:
#     op = generate_video(
#         prompt="A serene forest at dawn, light beams filtering through mist",
#         first_frame="start.png",
#         last_frame="end.png",
#         model="veo-2.0-generate-001",
#         aspect_ratio="16:9",
#         duration_seconds=8,
#         output_gcs_uri="gs://mybucket/output_prefix/"
#     )
#     # Access generated videos
#     for vid in op.response.generated_videos:
#         print("Generated video GCS URI:", vid.video)
