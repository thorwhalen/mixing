# mixing.video.genai

AI Video Generation with Google Vertex AI Veo

This module provides tools for generating videos using Google’s Veo models on Vertex AI.
The main workflow is simple: **generate and get a file path** to your video.

It is an *optional* feature: it requires `google-genai` (`pip install
'mixing[gen]'`) and Google Cloud credentials. Importing it pulls
`from google import genai`, so it is not wired into the lazy `mixing.video`
facade — import it explicitly via `mixing.video.genai`.

Key features include:

• **One-Step Generation**: Generate videos and get file paths in one call
• **Flexible Egress**: Control how videos are saved with the `output` parameter
• **Multiple Input Formats**: Generate from text prompts, images, or video frames
• **Smart Authentication**: Automatic environment variable detection and helpful error messages
• **Multiple Video Support**: Handle models that generate multiple video variations

Quick Start (Generate and Save):

```python
from mixing.video.genai import generate_video

# Generate video and get file path in one call
video_path = generate_video("A serene forest at dawn with mist")
print(f"Video saved to: {video_path}")
```

Flexible Egress Options:

```python
# Default: Auto-save to temp files and return path(s)
path = generate_video("prompt")

# Save to specific location
path = generate_video("prompt", output="/path/to/my_video.mp4")

# Save to directory with auto-naming
path = generate_video("prompt", output="/path/to/output_dir/")

# Just return the operation (no saving)
op = generate_video("prompt", output=False)
# or: op = generate_video("prompt", output=lambda x: x)

# Custom processing function
def my_processor(operation):
    paths = save_generated_videos(operation, prefix="custom_")
    print(f"Saved {len(paths)} videos")
    return paths

result = generate_video("prompt", output=my_processor)
```

Authentication Setup:
Set environment variables for automatic authentication:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

Main Functions:

- generate_video(): Generate videos with flexible egress options
- save_generated_videos(): Standalone video saving utility
- Frame extraction utilities for using video frames as inputs

### Functions

| [`generate_video`](#mixing.video.genai.generate_video)(prompt[, first_frame, ...])        | Generate a video using Veo (Vertex AI) and get the file path(s).   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`save_generated_video`](#mixing.video.genai.save_generated_video)(video_input[, output, ...])  | Save generated video(s) to file(s).                                |
| [`save_generated_videos`](#mixing.video.genai.save_generated_videos)(video_input[, output, ...]) | Save generated video(s) to file(s).                                |

### mixing.video.genai.generate_video(prompt, first_frame=None, last_frame=None, \*, output=<function save_generated_videos>, model='veo-2.0-generate-001', aspect_ratio='16:9', duration_seconds=5, output_gcs_uri=None, service_account_file=None, project_id=None, location='us-central1')

Generate a video using Veo (Vertex AI) and get the file path(s).

This function provides a complete workflow: generate video(s) and get downloadable file(s).
By default, videos are automatically saved to temporary files and you get the path(s).

The `output` parameter follows the canonical [`mixing.egress`](mixing.egress.md#module-mixing.egress) protocol
(a path/dir writes there; a callable is a sink applied to the result), with
two generator-specific sentinels noted below.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text prompt for video generation
  * **first_frame** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to image/video file for first frame
  * **last_frame** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to image/video file for last frame
  * **output** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – 

    Egress target / path specification. Controls how generated videos are processed:
    - save_generated_videos (default): Auto-save to temp files, return path(s)
    - ”/path/to/file.mp4”: Save to specific file path (via mixing.egress)
    - ”/path/to/directory/”: Save to directory with auto-generated names
    - ”.mp4”: Save to temp files with specific extension
    - False: Return raw operation without saving (sentinel)
    - lambda x: x or custom callable: Sink applied to the operation
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for generation
  * **aspect_ratio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video aspect ratio
  * **duration_seconds** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Video duration
  * **output_gcs_uri** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – GCS URI for output
  * **service_account_file** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to service account JSON file
  * **project_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Google Cloud project ID
  * **location** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Google Cloud location
* **Returns:**
  - Default: File path(s) where video(s) were saved
  - output=False: Raw operation object
  - Custom function: Whatever your function returns
* **Return type:**
  Depends on output parameter

### Examples

Basic usage (generate and get file path):

```pycon
>>> path = generate_video("A serene forest at dawn")
>>> print(f"Video saved to: {path}")
```

Save to specific location:

```pycon
>>> path = generate_video("Forest scene", output="/my/video.mp4")
```

Save to directory:

```pycon
>>> path = generate_video("Forest scene", output="/output/dir/")
```

Just get the operation (no saving):

```pycon
>>> op = generate_video("Forest scene", output=False)
>>> # Process op.response.generated_videos yourself
```

Custom processing:

```pycon
>>> def my_saver(op):
...     return save_generated_videos(op, prefix="custom_", extension_fallback="webm")
>>> paths = generate_video("Forest scene", output=my_saver)
```

Environment Variables:

```default
VEO_SERVICE_ACCOUNT_FILE or GOOGLE_APPLICATION_CREDENTIALS: Service account JSON path
VEO_PROJECT_ID, GOOGLE_CLOUD_PROJECT, or GCLOUD_PROJECT: Project ID
VEO_LOCATION or GOOGLE_CLOUD_LOCATION: Cloud location (default: us-central1)
```

Authentication Setup:
: For automatic authentication, set these environment variables:
  <br/>
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
  export GOOGLE_CLOUD_PROJECT="your-project-id"
  ```

* **Raises:**
  **Authentication errors with helpful setup instructions** – 

### mixing.video.genai.save_generated_video(video_input, output=None, , prefix='generated_video_', directory_name='generated_video_', extension_fallback='mp4')

Save generated video(s) to file(s). Handles single videos, lists of videos,
operation responses, or full operations.

* **Parameters:**
  * **video_input** – 

    Can be one of:
    - A single google.genai.types.GeneratedVideo object
    - A list of GeneratedVideo objects (from op.response.generated_videos)
    - An operation response object (op.response)
    - A full operation object (op)
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    Output path specification. Can be:
    - None: Auto-generate temp file(s) with appropriate extension
    - Path starting with ‘.’: Use as extension (e.g., ‘.mp4’)
    - Full filepath: Use as-is (for single video) or add index for multiple
    - Directory path: Use directory with generated names
  * **prefix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prefix for auto-generated filenames (keyword-only)
  * **directory_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name to use when output is directory (keyword-only)
  * **extension_fallback** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Default extension if MIME type can’t be determined (keyword-only)
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Single path (str) for one video, or list of paths for multiple videos

### Examples

```pycon
>>> # Save single video to temp file
>>> path = save_generated_videos(video_obj)
>>> # Save multiple videos with index
>>> paths = save_generated_videos(op.response.generated_videos)
>>> # Save from full operation
>>> paths = save_generated_videos(op, '/path/to/videos/')
>>> # Save with specific extension
>>> path = save_generated_videos(video_obj, '.mp4')
```

### mixing.video.genai.save_generated_videos(video_input, output=None, , prefix='generated_video_', directory_name='generated_video_', extension_fallback='mp4')

Save generated video(s) to file(s). Handles single videos, lists of videos,
operation responses, or full operations.

* **Parameters:**
  * **video_input** – 

    Can be one of:
    - A single google.genai.types.GeneratedVideo object
    - A list of GeneratedVideo objects (from op.response.generated_videos)
    - An operation response object (op.response)
    - A full operation object (op)
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    Output path specification. Can be:
    - None: Auto-generate temp file(s) with appropriate extension
    - Path starting with ‘.’: Use as extension (e.g., ‘.mp4’)
    - Full filepath: Use as-is (for single video) or add index for multiple
    - Directory path: Use directory with generated names
  * **prefix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prefix for auto-generated filenames (keyword-only)
  * **directory_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name to use when output is directory (keyword-only)
  * **extension_fallback** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Default extension if MIME type can’t be determined (keyword-only)
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Single path (str) for one video, or list of paths for multiple videos

### Examples

```pycon
>>> # Save single video to temp file
>>> path = save_generated_videos(video_obj)
>>> # Save multiple videos with index
>>> paths = save_generated_videos(op.response.generated_videos)
>>> # Save from full operation
>>> paths = save_generated_videos(op, '/path/to/videos/')
>>> # Save with specific extension
>>> path = save_generated_videos(video_obj, '.mp4')
```
