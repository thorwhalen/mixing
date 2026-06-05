"""Tests for the canonical egress protocol (mixing.egress)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mixing.egress import (
    Output,
    deliver,
    is_sink,
    is_path_output,
    resolve_output_path,
    write_egress,
)


def _write_text(result, path: Path) -> None:
    Path(path).write_text(str(result))


# --- predicates ----------------------------------------------------------


def test_is_sink_and_is_path():
    assert is_sink(lambda r: r) is True
    assert is_sink("out.txt") is False
    assert is_sink(Path("out.txt")) is False
    assert is_sink(None) is False
    assert is_path_output("out.txt") is True
    assert is_path_output(Path("x")) is True
    assert is_path_output(lambda r: r) is False
    assert is_path_output(None) is False


# --- deliver (object-first) ----------------------------------------------


def test_deliver_none_returns_object():
    obj = object()
    assert deliver(obj, None, write=_write_text, default_name="x.txt") is obj


def test_deliver_callable_returns_sink_result():
    obj = "RESULT"
    out = deliver(obj, lambda r: f"got:{r}", write=_write_text, default_name="x.txt")
    assert out == "got:RESULT"


def test_deliver_path_writes_and_returns_path(tmp_path):
    target = tmp_path / "a.txt"
    out = deliver("hello", str(target), write=_write_text, default_name="x.txt")
    assert out == target and target.read_text() == "hello"


def test_deliver_directory_uses_default_name(tmp_path):
    out = deliver("hi", str(tmp_path), write=_write_text, default_name="auto.txt")
    assert out == tmp_path / "auto.txt"
    assert (tmp_path / "auto.txt").read_text() == "hi"


def test_deliver_trailing_sep_is_directory(tmp_path):
    d = tmp_path / "sub"
    out = deliver("hi", f"{d}/", write=_write_text, default_name="auto.txt")
    assert out == d / "auto.txt" and out.read_text() == "hi"


# --- write_egress (file-first) -------------------------------------------


def test_write_egress_none_writes_default(tmp_path):
    default = tmp_path / "default.txt"
    out = write_egress(None, default_path=default, write=lambda p: Path(p).write_text("d"))
    assert out == default and default.read_text() == "d"


def test_write_egress_explicit_path(tmp_path):
    default = tmp_path / "default.txt"
    explicit = tmp_path / "explicit.txt"
    out = write_egress(
        str(explicit), default_path=default, write=lambda p: Path(p).write_text("e")
    )
    assert out == explicit and explicit.read_text() == "e"
    assert not default.exists()


def test_write_egress_directory_uses_default_name(tmp_path):
    default = tmp_path / "name.txt"
    d = tmp_path / "outdir"
    d.mkdir()
    out = write_egress(str(d), default_path=default, write=lambda p: Path(p).write_text("x"))
    assert out == d / "name.txt" and out.read_text() == "x"


def test_write_egress_sink_gets_written_path(tmp_path):
    default = tmp_path / "default.txt"
    seen = {}

    def sink(p):
        seen["path"] = p
        return "SINK"

    out = write_egress(sink, default_path=default, write=lambda p: Path(p).write_text("s"))
    assert out == "SINK"
    assert seen["path"] == default and default.read_text() == "s"


def test_resolve_output_path_makes_parent(tmp_path):
    nested = tmp_path / "a" / "b" / "c.txt"
    resolved = resolve_output_path(str(nested), default_name="x.txt")
    assert resolved == nested and nested.parent.is_dir()
