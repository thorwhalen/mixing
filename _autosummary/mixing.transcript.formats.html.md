# mixing.transcript.formats

Transcript output formats: SRT, plain prose, time remapping.

### Functions

| [`remap_time_after_cuts`](#mixing.transcript.formats.remap_time_after_cuts)(t, cuts)                    | Map `t` from the original timeline onto the post-cut timeline.                  |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`words_to_prose`](#mixing.transcript.formats.words_to_prose)(words, \*[, paragraph_pause, ...]) | Render a Scribe word list as plain prose, with paragraph breaks on long pauses. |
| [`words_to_srt`](#mixing.transcript.formats.words_to_srt)(words, \*[, max_chars, ...])         | Render an SRT from a Scribe word list (no filler removal, no remapping).        |
| [`words_to_srt_remapped`](#mixing.transcript.formats.words_to_srt_remapped)(words, cuts, \*[, ...])     | SRT aligned to a post-cut timeline.                                             |

### mixing.transcript.formats.remap_time_after_cuts(t, cuts)

Map `t` from the original timeline onto the post-cut timeline.

If `t` falls inside a cut, snaps to the moment that cut would have
started in the post-cut timeline.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.transcript.formats.words_to_prose(words, , paragraph_pause=1.2, drop_fillers=False, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

Render a Scribe word list as plain prose, with paragraph breaks on long pauses.

* **Parameters:**
  * **words** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Scribe `words` array.
  * **paragraph_pause** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Insert a blank line when the gap between two
    consecutive non-filler words exceeds this many seconds.
  * **drop_fillers** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `True`, omit filler words and removable audio events.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.formats.words_to_srt(words, , max_chars=80, sentence_endings='.?!')

Render an SRT from a Scribe word list (no filler removal, no remapping).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.formats.words_to_srt_remapped(words, cuts, , max_chars=80, sentence_endings='.?!', drop_fillers=True, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

SRT aligned to a post-cut timeline.

Each word’s timestamp is shifted earlier by the cumulative duration
of all cuts that ended before it, so the resulting SRT drops in over
the cleaned media.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
