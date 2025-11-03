"""
Video frame access via Mapping interface.

DEPRECATED: This module is deprecated. Use `mixing.video.video_ops` instead.

All functionality has been moved to `video_ops`:
- VideoFrames -> mixing.video.video_ops.VideoFrames
- save_frame -> mixing.video.video_ops.save_frame
- Helper functions are now internal to video_ops

This module remains for backward compatibility only.
"""

import warnings

# Deprecation warning
warnings.warn(
    "mixing.video.video_files is deprecated and will be removed in a future version. "
    "Use mixing.video.video_ops instead. "
    "Import VideoFrames and save_frame from mixing.video.video_ops.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from video_ops for backward compatibility
from mixing.video.video_ops import (
    VideoFrames,
    save_frame,
    require_package,
    _get_video_path_from_clipboard,
    _copy_frame_to_clipboard,
)

# Make available for backward compatibility
__all__ = [
    'VideoFrames',
    'save_frame',
    'require_package',
]


if __name__ == "__main__":
    import argh

    argh.dispatch_commands([save_frame])
