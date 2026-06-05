"""Canonical *egress*: route a produced result to its destination.

Across ``mixing``, every result-producing function takes a single ``output``
parameter whose **role** is constant — "what to do with the result" — while its
**type** is open:

====================  ===================================================
``output`` value      behavior
====================  ===================================================
``None``              return the in-memory result (object producers) or
                      save next to the input (file producers); see below
a **file path**       write the result there; return the ``Path``
a **directory**       write with an auto-derived name; return the ``Path``
a **callable**        return ``output(result)`` — the general escape hatch
====================  ===================================================

There are two flavors of producer, modeled by two helpers:

- :func:`deliver` — *object-first*: ``output=None`` returns the in-memory
  object (e.g. an :class:`~mixing.audio.Audio`). Used by editing functions that
  build a value you may want to keep chaining.
- :func:`write_egress` — *file-first*: ``output=None`` saves to a default path
  (typically beside the input) and returns it. Used by file→file operations
  whose whole point is to produce a file.

Both accept the same ``output`` forms, so callers learn one mental model.
``Output`` is the shared type alias.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Union

PathLike = Union[str, "os.PathLike[str]"]

#: An egress target: nothing, a filesystem path/dir, or a result-consuming sink.
Output = Union[None, PathLike, Callable[[Any], Any]]

__all__ = [
    "Output",
    "PathLike",
    "is_path_output",
    "is_sink",
    "resolve_output_path",
    "deliver",
    "write_egress",
]


def is_path_output(output: object) -> bool:
    """True if ``output`` denotes a filesystem path (``str`` / ``os.PathLike``)."""
    return isinstance(output, (str, os.PathLike))


def is_sink(output: object) -> bool:
    """True if ``output`` is a result-consuming callable (not a path)."""
    return callable(output) and not is_path_output(output)


def _looks_like_dir(output: PathLike) -> bool:
    """Heuristic: an existing directory, or a path with a trailing separator."""
    if Path(output).is_dir():
        return True
    raw = os.fspath(output)
    return raw.endswith(("/", os.sep))


def resolve_output_path(output: PathLike, *, default_name: str) -> Path:
    """Resolve a path/dir ``output`` to a concrete file ``Path`` (parents created).

    A directory ``output`` (existing, or with a trailing separator) is joined
    with ``default_name``. Only call this when ``output`` is a path — not
    ``None`` and not a sink.
    """
    path = Path(output).expanduser()
    if _looks_like_dir(output):
        path = path / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def deliver(
    result: Any,
    output: Output,
    *,
    write: Callable[[Any, Path], Any],
    default_name: str,
) -> Any:
    """Object-first egress: route an in-memory ``result`` to ``output``.

    - ``None`` → return ``result`` unchanged (no I/O).
    - sink callable → return ``output(result)``.
    - path/dir → ``write(result, path)`` then return the ``Path``.

    Args:
        result: the produced in-memory value (e.g. an ``Audio``).
        output: the egress target (see module docstring).
        write: ``write(result, path)`` persists ``result`` to ``path``.
        default_name: filename to use when ``output`` is a directory.
    """
    if output is None:
        return result
    if is_sink(output):
        return output(result)
    path = resolve_output_path(output, default_name=default_name)
    write(result, path)
    return path


def write_egress(
    output: Output,
    *,
    default_path: PathLike,
    write: Callable[[Path], Any],
) -> Any:
    """File-first egress: for operations whose primary effect is writing a file.

    - ``None`` → write to ``default_path`` (e.g. beside the input) and return it.
    - sink callable → write to ``default_path``, then return ``output(path)``.
    - path/dir → write to that file (dir → ``default_path``'s name inside it).

    Args:
        output: the egress target (see module docstring).
        default_path: where to write when ``output`` is ``None`` / a sink, and
            the source of the auto filename when ``output`` is a directory.
        write: ``write(path)`` performs the actual encode to ``path``.
    """
    default_path = Path(default_path)
    if is_sink(output) or output is None:
        target = default_path
    elif _looks_like_dir(output):
        target = Path(output).expanduser() / default_path.name
    else:
        target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target)
    return output(target) if is_sink(output) else target
