"""Characterization tests pinning ``mixing.chapters.detect_chapters`` behavior.

These guard the *constraint machinery* of chapter detection ahead of a refactor,
without touching the network or an LLM. Every test injects a deterministic
``segment_fn=(segments, target_count) -> [{"start", "title"}]`` stub, so the
LLM-backed :func:`mixing.chapters.default_segment_fn` (which needs ``aix``) is
never invoked.

Pinned behaviors:
- :class:`Chapter` dataclass fields (``start``, ``title``).
- First chapter is forced to ``0:00`` even when the segmenter returns a non-zero
  first start.
- Off-boundary chapter starts snap to the nearest real segment boundary.
- ``min_spacing`` drops markers closer than the spacing to the previous kept one.
- ``max_chapters`` bounds the kept count from above.
- ``[]`` returned when the media is too short to host ``min_chapters`` spaced
  markers (``duration < min_chapters * min_spacing``) — and the segmenter is not
  even called in that case.
- ``[]`` returned when fewer than ``min_chapters`` survive the constraints.
- The three transcript forms ``_segments_from_transcript`` accepts: a Scribe
  response dict with ``"words"``, an SRT text string, and a list of cue-like
  dicts/objects exposing ``start``/``end``/``text``.
- Empty/whitespace titles are dropped before constraint enforcement; surviving
  titles are stripped.
- ``duration`` is inferred from the last segment's ``end`` when omitted.
- ``target_count`` is clamped into ``[min_chapters, max_chapters]`` before being
  handed to the segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mixing.chapters import Chapter, detect_chapters, _segments_from_transcript


# --- shared fixtures/helpers -------------------------------------------------


def _srt(starts_ends_texts) -> str:
    """Build SRT text from ``(start_s, end_s, text)`` triples."""

    def ts(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, (start, end, text) in enumerate(starts_ends_texts, 1):
        blocks.append(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n")
    return "\n".join(blocks)


# A transcript whose segments sit at 0, 20, 45, 58 seconds.
SRT = _srt(
    [
        (0.0, 5.0, "Intro sentence."),
        (20.0, 25.0, "Middle sentence."),
        (45.0, 50.0, "Later sentence."),
        (58.0, 59.0, "Near the end."),
    ]
)


def _stub_at_segment_starts(segments, target_count):
    """Return one boundary per segment, titled ``T0``, ``T1``, ..."""
    return [
        {"start": s["start"], "title": f"T{i}"} for i, s in enumerate(segments)
    ]


# --- Chapter dataclass -------------------------------------------------------


def test_chapter_dataclass_fields():
    ch = Chapter(start=12.5, title="A title")
    assert ch.start == 12.5
    assert ch.title == "A title"
    # dataclass exposes exactly these two fields, in order.
    from dataclasses import fields

    assert [f.name for f in fields(Chapter)] == ["start", "title"]


def test_detect_chapters_returns_chapter_instances():
    chs = detect_chapters(SRT, duration=60.0, segment_fn=_stub_at_segment_starts)
    assert chs
    assert all(isinstance(c, Chapter) for c in chs)


# --- constraint machinery ----------------------------------------------------


def test_first_chapter_forced_to_zero():
    def stub(segments, target_count):
        # Deliberately non-zero first start; must be forced to 0:00.
        return [
            {"start": 20.0, "title": "B"},
            {"start": 45.0, "title": "C"},
            {"start": 58.0, "title": "D"},
        ]

    chs = detect_chapters(SRT, duration=70.0, segment_fn=stub)
    assert chs[0].start == 0.0


def test_off_boundary_starts_snap_to_nearest_segment_boundary():
    def stub(segments, target_count):
        # None of these are exact boundaries (0, 20, 45, 58).
        return [
            {"start": 2.0, "title": "A"},
            {"start": 22.0, "title": "B"},
            {"start": 43.0, "title": "C"},
        ]

    chs = detect_chapters(SRT, duration=70.0, segment_fn=stub)
    assert [round(c.start) for c in chs] == [0, 20, 45]


def test_min_spacing_drops_too_close_markers():
    # Spacing is measured against the previously *kept* marker, greedily from
    # the start. With min_spacing=30 and markers at 0, 20, 45, 58:
    #   0 kept; 20 is <30 from 0 -> dropped; 45 is >=30 from 0 -> kept;
    #   58 is <30 from 45 -> dropped. Result: [0, 45].
    def stub(segments, target_count):
        return [
            {"start": 0.0, "title": "A"},
            {"start": 20.0, "title": "B too close to A"},
            {"start": 45.0, "title": "C"},
            {"start": 58.0, "title": "D too close to C"},
        ]

    chs = detect_chapters(
        SRT, duration=70.0, min_spacing=30.0, min_chapters=2, segment_fn=stub
    )
    assert [round(c.start) for c in chs] == [0, 45]


def test_min_spacing_default_drops_only_the_close_marker():
    # Classic case: a 3s-after-0 marker is dropped, well-spaced ones survive.
    def stub(segments, target_count):
        return [
            {"start": 0.0, "title": "A"},
            {"start": 3.0, "title": "B too close"},  # <10s from kept A -> dropped
            {"start": 20.0, "title": "C"},
            {"start": 45.0, "title": "D"},
        ]

    chs = detect_chapters(SRT, duration=70.0, min_spacing=10.0, segment_fn=stub)
    assert [round(c.start) for c in chs] == [0, 20, 45]


def test_max_chapters_upper_bound():
    # A long transcript with 12 well-spaced segments at 0,30,60,...,330.
    srt = _srt([(i * 30.0, i * 30.0 + 5.0, f"Sentence {i}.") for i in range(12)])

    chs = detect_chapters(
        srt, duration=360.0, max_chapters=5, segment_fn=_stub_at_segment_starts
    )
    assert len(chs) == 5
    assert [round(c.start) for c in chs] == [0, 30, 60, 90, 120]


def test_returns_empty_when_media_too_short():
    # duration < min_chapters * min_spacing -> [] and segmenter NOT called.
    called = {"n": 0}

    def stub(segments, target_count):
        called["n"] += 1
        return [{"start": 0.0, "title": "A"}]

    chs = detect_chapters(
        SRT, duration=15.0, min_chapters=3, min_spacing=10.0, segment_fn=stub
    )
    assert chs == []
    assert called["n"] == 0  # short-circuits before segmenting


def test_returns_empty_when_below_min_chapters_after_constraints():
    def stub(segments, target_count):
        # Only two distinct survivors, but min_chapters=3.
        return [
            {"start": 0.0, "title": "A"},
            {"start": 20.0, "title": "B"},
        ]

    chs = detect_chapters(SRT, duration=70.0, min_chapters=3, segment_fn=stub)
    assert chs == []


def test_marker_at_or_beyond_duration_is_dropped():
    def stub(segments, target_count):
        return [
            {"start": 0.0, "title": "A"},
            {"start": 20.0, "title": "B"},
            {"start": 45.0, "title": "C"},
            {"start": 58.0, "title": "D at duration"},
        ]

    # duration == 58, so the boundary snapped to 58 is dropped (>= duration-1e-3).
    chs = detect_chapters(SRT, duration=58.0, segment_fn=stub)
    assert [round(c.start) for c in chs] == [0, 20, 45]


def test_empty_titles_dropped_before_constraints_and_titles_stripped():
    def stub(segments, target_count):
        return [
            {"start": 0.0, "title": "  First  "},
            {"start": 20.0, "title": ""},  # falsy title -> dropped early
            {"start": 45.0, "title": "Third"},
        ]

    chs = detect_chapters(SRT, duration=70.0, min_chapters=2, segment_fn=stub)
    # The empty-title marker at 20 vanishes; surviving titles are stripped.
    assert [(round(c.start), c.title) for c in chs] == [(0, "First"), (45, "Third")]


def test_duration_inferred_from_last_segment_end_when_omitted():
    # Last segment end is 59.0; min_chapters*min_spacing = 30 < 59, so chapters
    # survive even though `duration` was not passed.
    chs = detect_chapters(SRT, segment_fn=_stub_at_segment_starts)
    assert chs
    assert chs[0].start == 0.0


def test_target_count_clamped_into_min_max_range():
    seen = {"target": None}

    def stub(segments, target_count):
        seen["target"] = target_count
        return _stub_at_segment_starts(segments, target_count)

    # Ask for target_count=99 but cap max_chapters at 4 -> clamped to 4.
    detect_chapters(
        SRT,
        duration=70.0,
        min_chapters=2,
        max_chapters=4,
        target_count=99,
        segment_fn=stub,
    )
    assert seen["target"] == 4

    # Ask for target_count=1 but floor min_chapters at 3 -> clamped to 3.
    detect_chapters(
        SRT,
        duration=70.0,
        min_chapters=3,
        max_chapters=8,
        target_count=1,
        segment_fn=stub,
    )
    assert seen["target"] == 3


# --- _segments_from_transcript: the three accepted transcript forms ----------


def test_segments_from_scribe_response_dict_with_words():
    transcript = {
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word"},
            {"text": "world.", "start": 0.4, "end": 0.9, "type": "word"},
            {"text": "Next", "start": 30.0, "end": 30.4, "type": "word"},
            {"text": "topic.", "start": 30.4, "end": 30.9, "type": "word"},
        ]
    }
    segs = _segments_from_transcript(transcript)
    # Grouped into sentences by terminal punctuation.
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == pytest.approx(0.0)
    assert segs[0]["end"] == pytest.approx(0.9)
    assert segs[1]["text"] == "Next topic."
    assert segs[1]["start"] == pytest.approx(30.0)


def test_segments_from_srt_text_string():
    srt = _srt(
        [
            (0.0, 5.0, "Intro to the problem we are solving."),
            (20.0, 25.0, "Here is how the product works in detail."),
            (45.0, 55.0, "And finally a call to action."),
        ]
    )
    segs = _segments_from_transcript(srt)
    assert [round(s["start"]) for s in segs] == [0, 20, 45]
    assert segs[1]["text"].startswith("Here is how")
    assert segs[2]["end"] == pytest.approx(55.0)


def test_segments_from_list_of_cue_dicts():
    cues = [
        {"start": 0.0, "end": 5.0, "text": "First cue sentence here."},
        {"start": 10.0, "end": 15.0, "text": "Second cue sentence here."},
    ]
    segs = _segments_from_transcript(cues)
    assert [s["start"] for s in segs] == [0.0, 10.0]
    assert segs[0]["text"] == "First cue sentence here."
    assert segs[1]["end"] == pytest.approx(15.0)


def test_segments_from_list_of_cue_objects():
    @dataclass
    class Cue:
        start: float
        end: float
        text: str

    cues = [
        Cue(0.0, 5.0, "First object cue sentence."),
        Cue(10.0, 15.0, "Second object cue sentence."),
    ]
    segs = _segments_from_transcript(cues)
    assert [s["start"] for s in segs] == [0.0, 10.0]
    assert segs[0]["text"] == "First object cue sentence."
    assert segs[1]["text"] == "Second object cue sentence."


def test_segments_from_empty_inputs():
    # Empty / unsupported transcripts normalize to an empty segment list.
    assert _segments_from_transcript([]) == []
    assert _segments_from_transcript("") == []
    assert _segments_from_transcript({"words": []}) == []


def test_detect_chapters_returns_empty_for_empty_transcript():
    # No segments -> [] regardless of segment_fn (which is never reached).
    def stub(segments, target_count):  # pragma: no cover - must not be called
        raise AssertionError("segment_fn should not be called for empty input")

    assert detect_chapters([], duration=100.0, segment_fn=stub) == []
