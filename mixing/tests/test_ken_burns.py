"""mixing wraps the Ken Burns renderers from the ``burns`` package.

The rendering implementation lives in ``burns`` (which uses ``saveas=``); mixing
exposes thin adapters ``ken_burns_video`` / ``ken_burns_film`` that speak the
package-wide ``output`` egress protocol and delegate to burns. These tests guard
that contract by monkeypatching burns — no actual rendering (or ffmpeg) needed.
"""

from pathlib import Path

import mixing.video.video_ops as vo


def test_ken_burns_video_translates_output_to_burns_saveas(monkeypatch, tmp_path):
    calls = {}

    def fake_burns(image, path, *, duration, fps, saveas, output_size, **kw):
        calls["saveas"] = saveas
        Path(saveas).write_bytes(b"mp4")
        return Path(saveas)

    monkeypatch.setattr(vo, "_burns_ken_burns_video", fake_burns)

    out = tmp_path / "clip.mp4"
    result = vo.ken_burns_video("img.jpg", duration=3, fps=24, output=str(out))
    assert result == out and out.read_bytes() == b"mp4"
    assert calls["saveas"] == str(out)  # output -> burns saveas


def test_ken_burns_video_output_none_defers_to_burns_autoname(monkeypatch):
    calls = {}

    def fake_burns(image, path, *, duration, fps, saveas, output_size, **kw):
        calls["saveas"] = saveas
        return Path("auto_kenburns.mp4")

    monkeypatch.setattr(vo, "_burns_ken_burns_video", fake_burns)

    result = vo.ken_burns_video("img.jpg")
    assert calls["saveas"] is None  # None -> burns picks its own name
    assert result == Path("auto_kenburns.mp4")


def test_ken_burns_video_output_callable_sink(monkeypatch, tmp_path):
    def fake_burns(image, path, *, duration, fps, saveas, output_size, **kw):
        Path(saveas).write_bytes(b"mp4")
        return Path(saveas)

    monkeypatch.setattr(vo, "_burns_ken_burns_video", fake_burns)
    monkeypatch.chdir(tmp_path)  # sink path defaults to cwd; keep it in tmp
    seen = {}
    result = vo.ken_burns_video("img.jpg", output=lambda p: seen.setdefault("p", p))
    assert seen["p"] == result


def test_ken_burns_film_translates_output_to_burns_saveas(monkeypatch, tmp_path):
    calls = {}

    def fake_film(panels, *, saveas, fps, audio_path, **kw):
        calls["saveas"] = saveas
        Path(saveas).write_bytes(b"mp4")
        return Path(saveas)

    monkeypatch.setattr(vo, "_burns_ken_burns_film", fake_film)

    out = tmp_path / "film.mp4"
    result = vo.ken_burns_film([("a.jpg", None, 2.0)], output=str(out), fps=24)
    assert result == out and calls["saveas"] == str(out)
