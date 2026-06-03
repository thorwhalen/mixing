"""Network-free tests for mixing.dubbing (SRT I/O, translation glue, segment timing).

These exercise the pure logic — SRT parse/serialize round-trips, the
segment-count guarantee of :func:`translate_srt`, and the timing model of the
dub pipeline's segment builder — without calling ElevenLabs or any LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mixing.dubbing import Cue, dump_srt, parse_srt, translate_srt
from mixing.dubbing.dub import _build_segments


SAMPLE_SRT = """1
00:00:00,500 --> 00:00:02,000
Hello world.

2
00:00:03,000 --> 00:00:05,000
This is a test.
"""


def test_parse_srt_basic():
    cues = parse_srt(SAMPLE_SRT)
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.5)
    assert cues[0].end == pytest.approx(2.0)
    assert cues[0].text == "Hello world."
    assert cues[1].duration == pytest.approx(2.0)


def test_parse_srt_tolerates_dot_separator_and_no_index():
    # '.' ms separator and a missing numeric index line.
    txt = "00:00:01.250 --> 00:00:02.750\nLine without index"
    cues = parse_srt(txt)
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(1.25)
    assert cues[0].text == "Line without index"


def test_srt_round_trip():
    cues = parse_srt(SAMPLE_SRT)
    reparsed = parse_srt(dump_srt(cues))
    assert [(c.start, c.end, c.text) for c in cues] == [
        (c.start, c.end, c.text) for c in reparsed
    ]


def test_translate_srt_preserves_timing_and_uses_translate_fn():
    def fake_translate(texts, target_language, source_language=None):
        assert target_language == "French"
        return [t.upper() for t in texts]  # stand-in "translation"

    out = translate_srt(SAMPLE_SRT, "French", translate_fn=fake_translate)
    cues = parse_srt(out)
    assert cues[0].text == "HELLO WORLD."
    assert cues[0].start == pytest.approx(0.5)  # timings unchanged
    assert cues[1].end == pytest.approx(5.0)


def test_translate_srt_rejects_segment_count_mismatch():
    def bad_translate(texts, target_language, source_language=None):
        return texts[:-1]  # drops one — must raise

    with pytest.raises(ValueError):
        translate_srt(SAMPLE_SRT, "French", translate_fn=bad_translate)


def test_build_segments_inserts_gaps_and_pads_to_video_end(tmp_path):
    cues = [Cue(1, 1.0, 2.0, "a"), Cue(2, 4.0, 5.0, "b")]

    # Stub synth: write a tiny marker file; durations come from a fake map.
    durations = {}

    def synth(text, dest: Path):
        dest.write_bytes(b"x")
        durations[str(dest)] = 0.5  # each clip is 0.5s, shorter than its window
        return dest

    # Patch the duration probe used inside the builder.
    import mixing.dubbing.dub as dub_mod

    orig = dub_mod._media_duration
    dub_mod._media_duration = lambda p: durations.get(str(p), 0.5)
    try:
        segs = _build_segments(
            cues, work=tmp_path, synth_fn=synth, fit="speed",
            max_speedup=1.5, video_dur=10.0,
        )
    finally:
        dub_mod._media_duration = orig

    # Expected layout: 1.0s silence, clip(0.5), gap to 4.0 (2.5s silence),
    # clip(0.5), trailing silence to 10.0.
    silences = [d for a, d in segs if a is None]
    clips = [d for a, d in segs if a is not None]
    assert clips == [0.5, 0.5]
    assert sum(d for _, d in segs) == pytest.approx(10.0)  # fills whole video
    assert silences[0] == pytest.approx(1.0)  # leading gap to first cue


def test_build_segments_compresses_overlong_line(tmp_path):
    # One cue whose clip (3s) overruns its window (cue starts at 0, next/end at 2s).
    cues = [Cue(1, 0.0, 1.0, "long line")]

    def synth(text, dest: Path):
        dest.write_bytes(b"x")
        return dest

    import mixing.dubbing.dub as dub_mod

    calls = {"atempo_factor": None}

    def fake_media_duration(p):
        # natural clip is 3s; after atempo it's reported as window length
        return 3.0 if str(p).endswith(".mp3") else 2.0

    def fake_atempo(src, dst, factor):
        calls["atempo_factor"] = factor
        Path(dst).write_bytes(b"x")
        return Path(dst)

    orig_dur, orig_atempo = dub_mod._media_duration, dub_mod._atempo
    dub_mod._media_duration, dub_mod._atempo = fake_media_duration, fake_atempo
    try:
        segs = _build_segments(
            cues, work=tmp_path, synth_fn=synth, fit="speed",
            max_speedup=1.5, video_dur=2.0,
        )
    finally:
        dub_mod._media_duration, dub_mod._atempo = orig_dur, orig_atempo

    # Needed 3/2 = 1.5x, exactly the cap — atempo must have been invoked.
    assert calls["atempo_factor"] == pytest.approx(1.5)
    assert any(a is not None for a, _ in segs)
