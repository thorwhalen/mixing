# mixing.transcript.fillers

Filler-word detection over Scribe-shaped word lists.

A “word list” here is the `words` array in the ElevenLabs Scribe JSON
response — each entry has `text`, `start`, `end`, and `type`
(one of `"word"`, `"spacing"`, `"audio_event"`).

The two main outputs are:

- [`build_cuts()`](#mixing.transcript.fillers.build_cuts): time ranges to REMOVE (fillers + optional audio events).
- [`keeps_from_cuts()`](#mixing.transcript.fillers.keeps_from_cuts): complementary KEEP ranges, suitable for ffmpeg.

### Module Attributes

| [`DEFAULT_FILLER_TOKENS`](#mixing.transcript.fillers.DEFAULT_FILLER_TOKENS)       | Default filler tokens (lowercased, alpha-only, see [`normalize_token()`](#mixing.transcript.fillers.normalize_token)).   |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_AUDIO_EVENTS_TO_CUT`](#mixing.transcript.fillers.DEFAULT_AUDIO_EVENTS_TO_CUT) | Default audio-event tags to remove.                                                                                       |

### Functions

| [`build_cuts`](#mixing.transcript.fillers.build_cuts)(words, \*[, fillers, ...])        | Compute time ranges to REMOVE.                                       |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`is_filler`](#mixing.transcript.fillers.is_filler)(item, \*[, fillers, audio_events]) | Return `True` if `item` is a filler word or a removable audio event. |
| [`keeps_from_cuts`](#mixing.transcript.fillers.keeps_from_cuts)(cuts, duration)              | Return ranges to KEEP given the cut ranges and total duration.       |
| [`normalize_token`](#mixing.transcript.fillers.normalize_token)(text)                        | Lowercase and strip non-alpha so `"Uh,"` -> `"uh"`.                  |

### mixing.transcript.fillers.DEFAULT_AUDIO_EVENTS_TO_CUT *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= frozenset({'(coughs)'})*

Default audio-event tags to remove. `(laughs)` is intentionally kept.

### mixing.transcript.fillers.DEFAULT_FILLER_TOKENS *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'})*

Default filler tokens (lowercased, alpha-only, see [`normalize_token()`](#mixing.transcript.fillers.normalize_token)).

### mixing.transcript.fillers.build_cuts(words, , fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), absorb_trailing_space=True, merge_gap=0.08)

Compute time ranges to REMOVE.

* **Parameters:**
  * **words** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Scribe `words` array.
  * **fillers** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler set.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default audio-event set.
  * **absorb_trailing_space** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Extend each cut to the end of the trailing
    `"spacing"` token, which avoids leaving a stranded pause.
  * **merge_gap** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Merge adjacent cuts when their gap is below this many seconds.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  List of `{"start": float, "end": float, "label": str}` dicts,
  sorted by start time and non-overlapping.

### mixing.transcript.fillers.is_filler(item, , fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

Return `True` if `item` is a filler word or a removable audio event.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.transcript.fillers.keeps_from_cuts(cuts, duration)

Return ranges to KEEP given the cut ranges and total duration.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### mixing.transcript.fillers.normalize_token(text)

Lowercase and strip non-alpha so `"Uh,"` -> `"uh"`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
