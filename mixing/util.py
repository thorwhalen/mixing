"""General utilities for mixing video, audio, etc."""

from typing import Literal
import importlib

TimeUnit = Literal["seconds", "frames", "milliseconds"]
AudioTimeUnit = Literal["seconds", "samples", "milliseconds"]


def require_package(package_name: str):
    """
    Import a package, raising an informative error if not installed.

    >>> math = require_package('math')  # doctest: +SKIP
    >>> math.pi  # doctest: +SKIP
    3.141592653589793
    """
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(
            f"Package '{package_name}' is required for this functionality. "
            f"Please install it via 'pip install {package_name}'."
        ) from e


def to_seconds(value: float, *, unit: TimeUnit | AudioTimeUnit, rate: float) -> float:
    """
    Convert time value to seconds based on unit.

    Args:
        value: Time value to convert
        unit: Unit of the value ('seconds', 'frames', 'milliseconds', 'samples')
        rate: Frame rate (fps) for 'frames' or sample rate (Hz) for 'samples'

    Returns:
        Time in seconds

    Examples:
        >>> to_seconds(10, unit="seconds", rate=24)  # doctest: +SKIP
        10.0
        >>> to_seconds(240, unit="frames", rate=24)  # doctest: +SKIP
        10.0
        >>> to_seconds(10000, unit="milliseconds", rate=24)  # doctest: +SKIP
        10.0
    """
    if unit == "seconds":
        return value
    elif unit in ("frames", "samples"):
        return value / rate
    elif unit == "milliseconds":
        return value / 1000.0
    else:
        raise ValueError(f"Invalid time unit: {unit}")


def get_path_from_clipboard() -> str:
    """
    Get file path from clipboard and validate it.

    Intelligently extracts file paths from various clipboard contents including:
    - Direct file paths
    - File paths within error messages or tracebacks
    - Quoted paths
    """
    import os
    import re

    print("Getting file path from clipboard...")
    clipboard_content = require_package("pyclip").paste()

    # Validate it's text, not binary data
    if isinstance(clipboard_content, bytes):
        try:
            file_path = clipboard_content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "Clipboard contains binary data that cannot be decoded as text. "
                "Expected a file path string."
            )
    elif isinstance(clipboard_content, str):
        file_path = clipboard_content
    else:
        raise ValueError(
            f"Clipboard content is not a valid string or bytes. "
            f"Got {type(clipboard_content).__name__}"
        )

    # Clean up the path
    file_path = os.path.expanduser(file_path.strip())

    # Try the path as-is first
    if os.path.isfile(file_path):
        print(f"... File path: {file_path}")
        return file_path

    # If that didn't work, try to extract paths from the text
    # Look for quoted paths or paths in error messages
    patterns = [
        r"'([^']+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))'",  # Single quoted paths
        r'"([^"]+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))"',  # Double quoted paths
        r"([/~][^\s'\"]+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))",  # Unquoted absolute paths
    ]

    for pattern in patterns:
        matches = re.findall(pattern, file_path, re.IGNORECASE)
        for match in matches:
            candidate = os.path.expanduser(match.strip())
            if os.path.isfile(candidate):
                print(f"... Extracted file path from clipboard text: {candidate}")
                return candidate

    # If still not found, give helpful error
    # Truncate long strings to avoid printing huge binary garbage
    display_content = file_path if len(file_path) < 100 else file_path[:100] + "..."
    raise ValueError(
        f"Clipboard content is not a valid (existing) file path: {display_content}\n"
        f"Tip: Copy the file path directly, not an error message containing it."
    )

    print(f"... File path: {file_path}")
    return file_path


def copy_to_clipboard(data: bytes | str) -> None:
    """
    Copy data to system clipboard.

    Args:
        data: Bytes (for binary data like images) or string to copy
    """
    pyclip = require_package("pyclip")
    pyclip.copy(data)
