# mixing.audio.segmentation

Split a long audio into segments — for example, the individual songs in a
concert recording, a DJ mix, or a radio show.

Three things make this hard in practice:

1. Sometimes the boundaries are obvious (silence between tracks).
2. Sometimes there’s continuous background noise (audience, room tone),
   so the energy never really drops.
3. Sometimes the boundaries are between different *kinds* of audio
   (speech vs music on the radio).

This module offers a small family of pluggable strategies that cover those
three regimes, plus a single entry point `find_segments` that you can
parametrize for your particular case. It is not magic — it is a toolbox
with sensible defaults. Tuning the keyword arguments matters.

The strategies, in increasing order of sophistication:

- `"silence"` — pydub silence detection. Good for clean track separations.
- `"energy_novelty"` — adaptive RMS valley detection. Catches fade-outs and
  quiet inter-track moments that aren’t quite silence.
- `"self_similarity"` — Foote’s checkerboard novelty (ICME 2000) on
  log-spectrogram features. The standard tool for concert recordings where
  amplitude is roughly constant: it detects boundaries where the \*spectral
  content\* changes abruptly. See Foote, “Automatic audio segmentation using
  a measure of audio novelty” (ICME 2000).
- `"speech_music"` — low-energy frame ratio + 4 Hz modulation energy
  (Scheirer & Slaney 1997-style features). Splits radio audio into spoken
  and musical regions.

You can also pass your own callable as `strategy` for custom logic.

Output:

- `find_segments` returns a list of `Segment` dataclasses (start/end in
  seconds), which you can hand to a player, a transcript tool, or whatever.
- `extract_segments` does the same plus exports each segment as a file.
- `Segment.as_offset_duration()` and `.as_start_end()` give the two
  common timestamp representations.

### Examples

```pycon
>>> from mixing.audio import find_segments, extract_segments
>>>
>>> # DJ mix with silences between tracks
>>> segs = find_segments("mix.mp3", strategy="silence",
...                       silence_thresh_db=-40, min_silence_len=1.5)
>>>
>>> # Concert: continuous audience noise, songs differ in spectral content
>>> segs = find_segments("concert.wav", strategy="self_similarity",
...                       kernel_seconds=12.0,
...                       min_peak_distance_seconds=60.0)
>>>
>>> # Radio show: tag speech vs music regions
>>> segs = find_segments("radio.mp3", strategy="speech_music")
>>>
>>> # Save them to disk
>>> paths = extract_segments("concert.wav", segs, output="songs/")
```

### Functions

| [`extract_segments`](#mixing.audio.segmentation.extract_segments)(audio[, segments, output, ...])   | Save each segment as its own audio file.                           |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`find_segments`](#mixing.audio.segmentation.find_segments)(audio, \*[, strategy, ...])          | Find segment boundaries in `audio` using a chosen strategy.        |
| [`segment_by_energy`](#mixing.audio.segmentation.segment_by_energy)(audio, \*[, sample_rate, ...])   | Split audio at local energy valleys (fade-outs and quiet moments). |
| [`segment_by_self_similarity`](#mixing.audio.segmentation.segment_by_self_similarity)(audio, \*[, ...])       | Find boundaries via Foote's checkerboard novelty on the SSM.       |
| [`segment_by_silence`](#mixing.audio.segmentation.segment_by_silence)(audio, \*[, ...])               | Find non-silent regions using pydub's silence detector.            |
| [`segment_by_speech_music`](#mixing.audio.segmentation.segment_by_speech_music)(audio, \*[, ...])          | Tag regions of audio as `"speech"` or `"music"`.                   |

### Classes

| [`Segment`](#mixing.audio.segmentation.Segment)(start, end[, label, score])   | A time-bounded slice of an audio file.   |
|----------------------------------------------------------------------------------------|------------------------------------------|

### *class* mixing.audio.segmentation.Segment(start, end, label=None, score=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A time-bounded slice of an audio file.

#### start

Segment start in seconds.

#### end

Segment end in seconds.

#### label

Optional tag (e.g. `"speech"`, `"music"`, `"song"`).

#### score

Optional confidence/novelty value. Higher = stronger boundary.

#### as_offset_duration()

Return `(offset, duration)` in seconds.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

#### as_start_end()

Return `(start, end)` in seconds.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Segment duration in seconds.

#### *property* offset *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Alias for `start` — read-only.

### mixing.audio.segmentation.extract_segments(audio, segments=None, , output=None, name_template='{stem}_{idx:03d}{ext}', format='mp3', bitrate='192k', strategy='silence', \*\*strategy_kwargs)

Save each segment as its own audio file.

If `segments` is `None`, this calls [`find_segments()`](#mixing.audio.segmentation.find_segments) first using
`strategy` and `strategy_kwargs`.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **segments** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Segment`](#mixing.audio.segmentation.Segment) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Either a list of `Segment` objects, or a list of
    `(start, end)` tuples in seconds. If `None`, segments are
    discovered automatically.
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Destination directory to write the per-segment files to
    (created if missing). This multi-file producer treats `output`
    strictly as a directory — one file per segment is written into it.
    Defaults to the source file’s parent, or the current directory
    if `audio` isn’t a path.
  * **name_template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Filename template. Available fields: `{stem}`,
    `{idx}`, `{label}`, `{start}`, `{end}`, `{ext}`.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output audio format (e.g. `"mp3"`, `"wav"`).
  * **bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bitrate for compressed formats.
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]]]) – Used only when `segments` is `None`.
  * **\*\*strategy_kwargs** – Used only when `segments` is `None`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  List of paths to the saved files, in segment order.

### Examples

```pycon
>>> from mixing.audio import extract_segments
>>> # Auto-detect and save in one call
>>> paths = extract_segments(
...     "concert.wav", strategy="self_similarity",
...     output="songs/", format="mp3",
...     kernel_seconds=14.0,
... )
>>>
>>> # Or pass timestamps you already have
>>> paths = extract_segments(
...     "mix.mp3",
...     segments=[(0, 245), (247, 445), (445, 600)],
...     output="tracks/",
... )
```

### mixing.audio.segmentation.find_segments(audio, , strategy='silence', min_segment_duration=0.0, max_segment_duration=None, merge_gap=0.0, pad_start=0.0, pad_end=0.0, \*\*strategy_kwargs)

Find segment boundaries in `audio` using a chosen strategy.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]]]) – Strategy name (`"silence"`, `"energy_novelty"`,
    `"self_similarity"`, `"speech_music"`) or a callable
    `(AudioSegment, **kwargs) -> list[Segment]`.
  * **min_segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Drop or merge segments shorter than this.
  * **max_segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Split segments longer than this into equal pieces.
  * **merge_gap** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Merge consecutive same-label segments separated by less
    than this gap (seconds).
  * **pad_start** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Extend each segment backwards by this many seconds.
  * **pad_end** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Extend each segment forwards by this many seconds.
  * **\*\*strategy_kwargs** – Forwarded to the chosen strategy.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of `Segment` instances. Use `.as_start_end()` or
  `.as_offset_duration()` to get plain timestamp tuples.

### Examples

```pycon
>>> from mixing.audio import find_segments
>>> segs = find_segments("mix.mp3", strategy="silence",
...                       silence_thresh_db=-45)
>>> [s.as_offset_duration() for s in segs]
[(0.0, 245.3), (247.1, 198.4), ...]
```

### mixing.audio.segmentation.segment_by_energy(audio, , sample_rate=16000, frame_seconds=0.5, hop_seconds=0.1, smooth_seconds=2.0, valley_threshold_factor=0.5, min_peak_distance_seconds=30.0, label='segment')

Split audio at local energy valleys (fade-outs and quiet moments).

Computes frame-wise RMS, smooths it, and finds local minima that fall
below a fraction of the local maximum. Useful when tracks fade into each
other without true silence.

* **Parameters:**
  * **audio** (`AudioSegment`) – pydub `AudioSegment`.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (Hz). Lower is faster.
  * **frame_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – RMS frame length in seconds.
  * **hop_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – RMS hop length in seconds.
  * **smooth_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Smoothing window length in seconds. Bigger smooths
    out within-song dynamics.
  * **valley_threshold_factor** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – A frame is a valley candidate only if its
    RMS is below `factor * max(RMS)`. `0.5` = halfway down from
    the peak. Lower this for cleaner tracks, raise it for messier ones.
  * **min_peak_distance_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum spacing between valleys. Prevents
    multiple boundaries inside one inter-track pause and acts as a
    crude minimum-song-length filter.
  * **label** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Label assigned to every returned segment.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back 

  ```
  ``
  ```

  Segment\`\`s covering the whole audio.

### mixing.audio.segmentation.segment_by_self_similarity(audio, , sample_rate=11025, n_fft=2048, hop_seconds=0.25, n_bands=40, kernel_seconds=12.0, novelty_smooth_seconds=4.0, peak_threshold_factor=1.0, min_peak_distance_seconds=30.0, label='song')

Find boundaries via Foote’s checkerboard novelty on the SSM.

This is the standard approach for finding song boundaries in a recording
where amplitude is roughly constant (concerts, live sets, continuous
radio with steady levels). It looks at *what* the audio sounds like
rather than *how loud* it is, so it works when energy-based methods fail.

Pipeline:

> 1. log-magnitude spectrogram, averaged into log-spaced bands, z-scored
> 2. cosine self-similarity matrix
> 3. correlate the SSM diagonal with a Gaussian-tapered checkerboard
>    kernel — this measures how much frame `t` is “the end of one
>    block and the start of another”
> 4. peaks in the resulting novelty curve are segment boundaries
* **Parameters:**
  * **audio** (`AudioSegment`) – pydub `AudioSegment`.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (Hz). 11025 is plenty for novelty.
  * **n_fft** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – STFT window length in samples.
  * **hop_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – STFT hop length in seconds.
  * **n_bands** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of log-spaced frequency bands. 20-60 is reasonable.
  * **kernel_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Half-width of the checkerboard kernel times 2, i.e.
    roughly how much context (in seconds) is used on each side of a
    candidate boundary. Section-level boundaries (songs) usually
    want 8-20 s.
  * **novelty_smooth_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Smoothing applied to the novelty curve
    before peak picking. Reduces jitter from beat-level changes.
  * **peak_threshold_factor** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Adaptive threshold = mean + factor \* std of
    the novelty curve. Raise to be stricter, lower to find more
    boundaries.
  * **min_peak_distance_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum spacing between detected
    boundaries. Acts as a minimum-song-length filter.
  * **label** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Label assigned to every returned segment.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back `Segment``s covering the whole audio. Each
  segment's ``score` is the novelty value at its trailing boundary.

### mixing.audio.segmentation.segment_by_silence(audio, , silence_thresh_db=-40.0, min_silence_len=1.0, seek_step=0.01, keep_silence=0.0, label='non_silent')

Find non-silent regions using pydub’s silence detector.

Best for cleanly separated tracks (DJ mixes, audiobooks, voicemails) where
there’s a real silence gap between segments.

* **Parameters:**
  * **audio** (`AudioSegment`) – pydub `AudioSegment`.
  * **silence_thresh_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – dBFS threshold below which audio counts as silence.
    More negative = stricter silence requirement. Tune downward for
    noisy recordings (e.g. -50 dBFS).
  * **min_silence_len** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum silence duration in seconds to count as a
    boundary. Tracks separated by less than this are merged.
  * **seek_step** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Search granularity in seconds.
  * **keep_silence** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds of surrounding silence to include with each
    non-silent segment.
  * **label** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Label assigned to each returned segment.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of `Segment` covering the non-silent regions.

### mixing.audio.segmentation.segment_by_speech_music(audio, , sample_rate=16000, frame_seconds=0.05, hop_seconds=0.025, window_seconds=1.5, low_energy_threshold=0.5, speech_decision_threshold=0.5, smooth_seconds=5.0, min_segment_duration=3.0, speech_label='speech', music_label='music')

Tag regions of audio as `"speech"` or `"music"`.

Useful for radio shows and podcasts with musical interludes. Implements
a simplified version of Scheirer & Slaney (ICASSP 1997) using two of
their most discriminative features:

- **Low-energy frame ratio (LEFR)**: fraction of short frames in a
  ~1-2 s window with RMS below half the window mean. Speech has
  bursts of silence between syllables, so its LEFR is high; music
  stays loud, so its LEFR is low.
- **ZCR variance**: speech alternates voiced/unvoiced (low/high ZCR);
  music’s ZCR is steadier.

These are combined into a soft speech score, smoothed, and thresholded.
Consecutive frames with the same class become a single segment.

* **Parameters:**
  * **audio** (`AudioSegment`) – pydub `AudioSegment`.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (Hz).
  * **frame_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Short-frame length for RMS/ZCR (e.g. 50 ms).
  * **hop_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Short-frame hop length.
  * **window_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Long-window length over which LEFR and ZCR-variance
    are aggregated. ~1-2 s is the canonical choice.
  * **low_energy_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – A short frame counts as “low energy” if its
    RMS is below `low_energy_threshold * mean RMS in window`.
    Scheirer & Slaney used 0.5; lower values (~0.1) only count true
    silence frames, raise it for noisier recordings.
  * **speech_decision_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Speech score threshold in [0, 1] above
    which a window is tagged as speech. Raise to make speech-tagging
    stricter.
  * **smooth_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Smoothing applied to the soft speech score before
    thresholding. Larger = fewer, longer segments.
  * **min_segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Drop or merge segments shorter than this.
  * **speech_label** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Labels assigned to each region.
  * **music_label** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Labels assigned to each region.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back 

  ```
  ``
  ```

  Segment\`\`s tagging speech and music regions.
