# mixing.chapters

Detect chapter markers from a transcript — platform-agnostic.

A *chapter* is a `(start_seconds, title)` pair marking a topic shift. This
module turns a transcript (ElevenLabs Scribe response/words, SRT text, or a
list of cues) into a list of [`Chapter`](#mixing.chapters.Chapter) objects, with titles produced by
a pluggable LLM segmenter. The result is intentionally **target-neutral** —
formatting chapters into YouTube description timestamps, podcast PSC, or ID3
chapter frames is the job of a publication layer (e.g. the `yb` package).

The detector enforces the constraints common to chapter-aware players:
first chapter at `0:00`, a minimum spacing between chapters, and a minimum
count below which chapters are not worth showing (an empty list is returned so
callers can cleanly skip them — e.g. for a very short clip).

### Module Attributes

| [`SegmentFn`](#mixing.chapters.SegmentFn)                     | str}].                                                                                                                                       |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [`SECONDS_PER_CHAPTER_HEURISTIC`](#mixing.chapters.SECONDS_PER_CHAPTER_HEURISTIC) | Heuristic spacing used to pick a default chapter count when `target_count` is not given: roughly one chapter per this many seconds of media. |

### Functions

| [`default_segment_fn`](#mixing.chapters.default_segment_fn)(segments, target_count, \*)   | LLM-backed segmenter using `aix.chat`.      |
|---------------------------------------------------------------------------------------------------|---------------------------------------------|
| [`detect_chapters`](#mixing.chapters.detect_chapters)(transcript, \*[, duration, ...]) | Detect chapter markers from a `transcript`. |

### Classes

| [`Chapter`](#mixing.chapters.Chapter)(start, title)   | A chapter marker: a start time (seconds) and a short title.   |
|--------------------------------------------------------------------------|---------------------------------------------------------------|

### *class* mixing.chapters.Chapter(start, title)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A chapter marker: a start time (seconds) and a short title.

### mixing.chapters.SECONDS_PER_CHAPTER_HEURISTIC *= 90.0*

Heuristic spacing used to pick a default chapter count when `target_count`
is not given: roughly one chapter per this many seconds of media. ~90s (1.5
minutes) is a common chapter cadence for talks/tutorials.

### mixing.chapters.SegmentFn

str}].
`segments` is a list of `{"start", "end", "text"}` sentence-ish units.

* **Type:**
  A segmenter maps (segments, target_count) -> [{“start”
* **Type:**
  [*float*](https://docs.python.org/3/builtins/functions.html#float), ”title”

alias of `Callable`[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### mixing.chapters.default_segment_fn(segments, target_count, , model=None)

LLM-backed segmenter using `aix.chat`.

Presents the timestamped sentences and asks for `target_count` chapter
boundaries as a JSON array of `{"start", "title"}`. Pass your own
`segment_fn` to [`detect_chapters()`](#mixing.chapters.detect_chapters) to avoid the `aix` dependency.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### mixing.chapters.detect_chapters(transcript, , duration=None, min_chapters=3, max_chapters=8, min_spacing=10.0, target_count=None, segment_fn=None, model=None)

Detect chapter markers from a `transcript`.

* **Parameters:**
  * **transcript** – One of — a Scribe response `dict` (with `"words"`), a
    Scribe `words` list, SRT text, or a list of cue dicts/objects
    exposing `start`/`end`/`text`.
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Media duration in seconds. Inferred from the transcript’s
    last timestamp when omitted; used to choose a sensible chapter
    count and to bound the final marker.
  * **min_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum chapters worth showing. If fewer survive the
    constraints, an **empty list** is returned.
  * **max_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Upper bound on chapter count.
  * **min_spacing** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum seconds between consecutive chapters (players such
    as YouTube require >= 10s).
  * **target_count** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Desired chapter count. When omitted, scales with
    `duration` (roughly one chapter per ~1.5 min, clamped to
    `[min_chapters, max_chapters]`).
  * **segment_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Pluggable segmenter `(segments, target_count) -> [{start,
    title}]`. Defaults to [`default_segment_fn()`](#mixing.chapters.default_segment_fn) (LLM-backed).
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional LLM model override for the default segmenter.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Chapter`](#mixing.chapters.Chapter)]
* **Returns:**
  A list of [`Chapter`](#mixing.chapters.Chapter), first at `0:00`, spaced by at least
  `min_spacing` — or `[]` when the media can’t support
  `min_chapters` (e.g. it is too short).
