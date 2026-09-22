# mixing.srt

Canonical SRT / timeline parsing and formatting (pure, dependency-free).

This is the single source of truth for SubRip (`.srt`) time handling in
`mixing`. Before consolidation the same logic was reimplemented in three
places (`dubbing.srt`, `video.video_subtitles`, `transcript.formats`)
with subtly different rounding and tolerance; those modules now re-export from
here.

The vocabulary:

- A **timestamp** is the `HH:MM:SS,mmm` string SRT uses; [`seconds_to_srt_time()`](#mixing.srt.seconds_to_srt_time)
  and [`srt_time_to_seconds()`](#mixing.srt.srt_time_to_seconds) convert between it and seconds (a `float`).
- A [`Cue`](#mixing.srt.Cue) is one subtitle block: a 1-based `index`, `start`/`end`
  in seconds, and `text` (which may contain embedded newlines).

Times are always seconds (`float`) everywhere except the timestamp string.

### Module Attributes

| [`TIME_RE`](#mixing.srt.TIME_RE)               | Matches an SRT cue time line, tolerant of `,` or `.` as the ms separator.   |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`fmt_srt_time`](#mixing.srt.fmt_srt_time)(seconds) | Back-compat aliases — historically these names lived in different modules.  |

### Functions

| [`srt_time_to_seconds`](#mixing.srt.srt_time_to_seconds)(timestamp)                  | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                               |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`seconds_to_srt_time`](#mixing.srt.seconds_to_srt_time)(seconds)                    | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                |
| [`fmt_srt_time`](#mixing.srt.fmt_srt_time)(seconds)                           | Back-compat aliases — historically these names lived in different modules.                  |
| [`to_srt_time`](#mixing.srt.to_srt_time)(seconds)                            | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                |
| [`parse_srt`](#mixing.srt.parse_srt)(srt_text)                             | Parse SRT text into a list of [`Cue`](#mixing.srt.Cue) objects. |
| [`dump_srt`](#mixing.srt.dump_srt)(cues)                                  | Serialize cues back to SRT text, renumbering from 1.                                        |
| [`shift_srt_timestamps`](#mixing.srt.shift_srt_timestamps)(srt_text[, shift_seconds]) | Shift every cue time line in `srt_text` by `shift_seconds`.                                 |

### Classes

| [`Cue`](#mixing.srt.Cue)(index, start, end, text)   | One SRT subtitle cue.   |
|---------------------------------------------------------------------------------|-------------------------|

### *class* mixing.srt.Cue(index, start, end, text)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One SRT subtitle cue.

#### index

1-based cue number.

#### start

Start time in seconds.

#### end

End time in seconds.

#### text

Cue text (may contain embedded newlines).

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Cue duration in seconds (never negative).

### mixing.srt.TIME_RE *= re.compile('(\\\\d{1,2}):(\\\\d{2}):(\\\\d{2})[,.](\\\\d{1,3})\\\\s\*-->\\\\s\*(\\\\d{1,2}):(\\\\d{2}):(\\\\d{2})[,.](\\\\d{1,3})')*

Matches an SRT cue time line, tolerant of `,` or `.` as the ms separator.

### mixing.srt.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.srt.fmt_srt_time(seconds)

Back-compat aliases — historically these names lived in different modules.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.srt.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.srt.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](#mixing.srt.Cue)]

### mixing.srt.seconds_to_srt_time(seconds)

Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.

Milliseconds are *rounded* (not truncated), with carry handled correctly,
and negative inputs clamp to zero.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> seconds_to_srt_time(2592.187)
'00:43:12,187'
>>> seconds_to_srt_time(1.5)
'00:00:01,500'
>>> seconds_to_srt_time(-3)
'00:00:00,000'
```

### mixing.srt.shift_srt_timestamps(srt_text, shift_seconds=0.0)

Shift every cue time line in `srt_text` by `shift_seconds`.

Negative values shift earlier; resulting negative times clamp to zero.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> srt = '1\n00:43:12,187 --> 00:43:13,817\nHello world'
>>> '00:00:00,187 --> 00:00:01,817' in shift_srt_timestamps(srt, -2592)
True
```

### mixing.srt.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

### mixing.srt.to_srt_time(seconds)

Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.

Milliseconds are *rounded* (not truncated), with carry handled correctly,
and negative inputs clamp to zero.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> seconds_to_srt_time(2592.187)
'00:43:12,187'
>>> seconds_to_srt_time(1.5)
'00:00:01,500'
>>> seconds_to_srt_time(-3)
'00:00:00,000'
```
