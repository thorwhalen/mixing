"""Network-free tests for mixing.chapters (transcript → chapter markers)."""

from __future__ import annotations

import pytest

from mixing.chapters import Chapter, detect_chapters, _segments_from_transcript


SRT = """1
00:00:00,000 --> 00:00:05,000
Intro to the problem we are solving.

2
00:00:20,000 --> 00:00:25,000
Here is how the product works in detail.

3
00:00:45,000 --> 00:00:55,000
And finally a call to action.
"""

WORDS = {
    "words": [
        {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word"},
        {"text": "world.", "start": 0.4, "end": 0.9, "type": "word"},
        {"text": "Next", "start": 30.0, "end": 30.4, "type": "word"},
        {"text": "topic.", "start": 30.4, "end": 30.9, "type": "word"},
    ]
}


def test_segments_from_srt():
    segs = _segments_from_transcript(SRT)
    assert [round(s["start"]) for s in segs] == [0, 20, 45]
    assert segs[1]["text"].startswith("Here is how")


def test_segments_from_words_groups_sentences():
    segs = _segments_from_transcript(WORDS)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[1]["start"] == pytest.approx(30.0)


def test_detect_chapters_enforces_first_at_zero_and_spacing():
    def stub(segments, target_count):
        # Return boundaries (with a non-zero first to test 0:00 forcing).
        return [
            {"start": 5.0, "title": "The problem"},
            {"start": 20.0, "title": "How it works"},
            {"start": 45.0, "title": "Call to action"},
        ]

    chs = detect_chapters(SRT, duration=60.0, segment_fn=stub)
    assert chs[0].start == 0.0  # forced to 0:00
    assert [round(c.start) for c in chs] == [0, 20, 45]
    assert chs[1].title == "How it works"


def test_detect_chapters_drops_too_close_markers():
    def stub(segments, target_count):
        return [
            {"start": 0.0, "title": "A"},
            {"start": 3.0, "title": "B too close"},  # < min_spacing from A
            {"start": 20.0, "title": "C"},
            {"start": 45.0, "title": "D"},
        ]

    chs = detect_chapters(SRT, duration=60.0, min_spacing=10.0, segment_fn=stub)
    assert [round(c.start) for c in chs] == [0, 20, 45]  # B dropped


def test_detect_chapters_returns_empty_when_too_short():
    # duration < min_chapters * min_spacing → no chapters
    called = False

    def stub(segments, target_count):
        nonlocal called
        called = True
        return []

    chs = detect_chapters(SRT, duration=15.0, min_chapters=3, min_spacing=10.0, segment_fn=stub)
    assert chs == []
    assert called is False  # short-circuits before segmenting


def test_detect_chapters_returns_empty_when_below_min_count():
    def stub(segments, target_count):
        return [{"start": 0.0, "title": "Only one"}]

    chs = detect_chapters(SRT, duration=60.0, min_chapters=3, segment_fn=stub)
    assert chs == []
