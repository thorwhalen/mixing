"""Typed errors `mixing` raises — the home for exceptions a caller may want to catch.

Pure stdlib, imported by the lazy facade without pulling any backend, so a consumer
can ``from mixing import WindowTooWideForClip`` in a light module and still catch what
a heavy one raises.

The convention: an error class carries the MEASURED numbers as attributes, not only in
its message, so a caller can act on them (retry at the largest valid window, say) rather
than parse prose.
"""

__all__ = ["MixingError", "WindowTooWideForClip"]


class MixingError(Exception):
    """Base for every error `mixing` raises on its own behalf."""


class WindowTooWideForClip(MixingError, ValueError):
    """An explicit analysis window leaves a clip no second, independent look.

    Raised by :func:`mixing.audio.align_clips_to_reference` when the caller passes a
    ``window_s`` wider than ``clip_duration_s - hop_s`` — the length below which the
    clip holds only one window, so the consensus vote has nothing to arbitrate and
    ``support`` comes back ``None``.

    Why an error and not a clamp or a silent single window: the caller asked for a
    windowed vote and would otherwise receive a single whole-clip correlation whose
    only signal is the ABSENCE of a support number — indistinguishable from a clip
    that is genuinely too short to support at any window (issue #43). Clamping would
    be worse still: an explicit ``window_s`` is the caller saying what a window means
    for their material, and silently measuring at a different one makes ``support``
    incomparable across a set for a reason nothing reports.

    It subclasses :class:`ValueError` because it is an argument that cannot be honored,
    so existing ``except ValueError`` handlers around alignment keep working.

    Attributes:
        clip_index: Position of the offending clip in the ``clips`` sequence, or
            ``None`` when the caller did not identify one.
        clip_duration_s: The clip's duration in seconds.
        window_s: The window that was asked for.
        hop_s: The hop in force — the caller's, or the one derived from ``window_s``.
        max_window_s: The largest window that would still leave this clip a second
            look, given ``hop_s``. ``<= 0`` means no window can: the clip is shorter
            than the hop, and only ``consensus=False`` (or a smaller ``hop_s``)
            applies.
    """

    def __init__(
        self,
        *,
        clip_index: "int | None",
        clip_duration_s: float,
        window_s: float,
        hop_s: float,
        max_window_s: float,
    ) -> None:
        self.clip_index = clip_index
        self.clip_duration_s = float(clip_duration_s)
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.max_window_s = float(max_window_s)
        who = "clip" if clip_index is None else f"clip {clip_index}"
        if self.max_window_s > 0:
            remedy = (
                f"pass window_s <= {self.max_window_s:.3f}, or window_s=None to fit "
                f"the window to each clip"
            )
        else:
            remedy = (
                f"no window holds a second look at hop_s={self.hop_s:.3f}; pass a "
                f"smaller hop_s, window_s=None, or consensus=False"
            )
        super().__init__(
            f"window_s={self.window_s:.3f} is wider than {who} "
            f"({self.clip_duration_s:.3f} s) allows at hop_s={self.hop_s:.3f}: the clip "
            f"would hold a single window, so the consensus vote has nothing to "
            f"arbitrate and support would be None for a reason nothing reports — "
            f"{remedy}."
        )
