# mixing.audio.beats

Beat / downbeat / onset analysis of an audio signal — the `mixing[beats]` primitive.

A small, permissively-licensed (librosa, ISC) audio-analysis primitive that any consumer
can reuse to answer “where are the beats and where is the rhythmic energy in this audio”.
Its first customer is muvid’s footage-scoring layer (thorwhalen/muvid#13), which computes a
[`beat_grid()`](#mixing.audio.beats.beat_grid) **once on the clean master song** and maps every clip onto it via the
clip’s known offset — but nothing here is footage-specific.

Design notes:

- **Lazy heavy import.** `librosa` is imported via [`mixing.util.require_package()`](mixing.util.html.md#mixing.util.require_package)
  inside the function body, so `import mixing.audio` never pulls it. Install it with the
  `mixing[beats]` extra.
- **Permissive only.** librosa is ISC (commercial-clean). A future `backend="madmom"` /
  `"beatnet"` could fill in real downbeats, but madmom’s *models* are academic-licensed,
  so librosa stays the default and the sole backend shipped in the extra.
- **Downbeats are best-effort.** librosa has no downbeat tracker, so `downbeat_times` is
  empty for `backend="librosa"`; the field exists so a stronger backend can populate it
  without a signature change. Consumers that want downbeats should fall back to beats when
  the array is empty.

### Module Attributes

| [`DEFAULT_SAMPLE_RATE`](#mixing.audio.beats.DEFAULT_SAMPLE_RATE)   | Default analysis sample rate.                              |
|------------------------------------------------------------------------|------------------------------------------------------------|
| [`DEFAULT_HOP_LENGTH`](#mixing.audio.beats.DEFAULT_HOP_LENGTH)    | Default STFT hop for the onset envelope (librosa default). |

### Functions

| [`beat_grid`](#mixing.audio.beats.beat_grid)(audio, \*[, sample_rate, ...])   | Estimate beats, (best-effort) downbeats, and the onset envelope of `audio`.   |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### Classes

| [`BeatGrid`](#mixing.audio.beats.BeatGrid)(beat_times, downbeat_times, ...)   | Rhythmic analysis of one audio signal.   |
|----------------------------------------------------------------------------------------------|------------------------------------------|

### *class* mixing.audio.beats.BeatGrid(beat_times, downbeat_times, onset_env, onset_hop_s, sample_rate, tempo_bpm)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Rhythmic analysis of one audio signal.

#### beat_times

Beat instants in seconds (ascending).

#### downbeat_times

Downbeat instants in seconds (best-effort; empty for the
`librosa` backend, which has no downbeat tracker).

#### onset_env

The onset-strength envelope (one value per STFT hop; higher = more
rhythmic onset energy). Its frame *k* is at time `k * onset_hop_s`.

#### onset_hop_s

Seconds between consecutive `onset_env` frames (`hop_length/sr`).

#### sample_rate

The analysis sample rate.

#### tempo_bpm

The estimated global tempo (beats per minute).

#### to_dict()

JSON-round-trippable view (arrays → lists; small enough to inline).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### mixing.audio.beats.DEFAULT_HOP_LENGTH *= 512*

Default STFT hop for the onset envelope (librosa default). onset_hop_s = hop/​sr.

### mixing.audio.beats.DEFAULT_SAMPLE_RATE *= 22050*

Default analysis sample rate. 22.05 kHz is librosa’s default and ample for beat/onset.

### mixing.audio.beats.beat_grid(audio, , sample_rate=22050, hop_length=512, start_bpm=120.0, backend='librosa')

Estimate beats, (best-effort) downbeats, and the onset envelope of `audio`.

* **Parameters:**
  * **audio** (AudioSource) – The audio to analyze (path, numpy array, or `AudioSegment`).
  * **sample_rate** (int) – Analysis sample rate (mono).
  * **hop_length** (int) – STFT hop for the onset envelope; `onset_hop_s = hop_length/sr`.
  * **start_bpm** (float) – Tempo prior for the beat tracker (helps on ambiguous material).
  * **backend** (str) – Only `"librosa"` (ISC, commercial-clean) is supported. `"madmom"` is
    intentionally NOT shipped — its beat models are academic-licensed.
* **Return type:**
  BeatGrid
* **Returns:**
  A [`BeatGrid`](#mixing.audio.beats.BeatGrid). `downbeat_times` is empty for the librosa backend.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unsupported `backend`.
