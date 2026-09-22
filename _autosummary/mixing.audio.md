# mixing.audio

Audio mixing and editing functionality.

Main exports:

- Audio: Sliceable audio interface with time-based operations
- fade_in, fade_out: Apply fade effects
- crop_audio: Trim audio segments
- concatenate_audio: Join multiple audio files
- overlay_audio: Mix/overlay audio tracks
- loop_audio: Tile a clip (crossfaded) to a target duration
- duck_audio: Sidechain-duck a bed under a dialogue track
- save_audio_clip: Extract and save audio segments
- find_audio_offset: Align two recordings via cross-correlation
- find_segments, extract_segments: Split a long audio into pieces (songs,
  speech vs music, etc.) using pluggable strategies

### Examples

```pycon
>>> from mixing.audio import Audio, fade_in, concatenate_audio
>>> audio = Audio("song.mp3")
>>> segment = audio[10:30]  # 10s to 30s
>>> segment.save("clip.mp3")
```

```pycon
>>> faded = fade_in("intro.mp3", duration=2.0)
>>> combined = concatenate_audio("part1.mp3", "part2.mp3", "part3.mp3")
```

```pycon
>>> from mixing.audio import find_segments, extract_segments
>>> segs = find_segments("concert.wav", strategy="self_similarity")
>>> paths = extract_segments("concert.wav", segs, output="songs/")
```

### Functions

| [`fade_in`](#mixing.audio.fade_in)(src[, duration, output])                  | Apply fade-in effect to audio.                                                 |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`fade_out`](#mixing.audio.fade_out)(src[, duration, output])                 | Apply fade-out effect to audio.                                                |
| [`crop_audio`](#mixing.audio.crop_audio)(src_path[, start, end, ...])           | Convenience function to crop and save an audio segment.                        |
| [`concatenate_audio`](#mixing.audio.concatenate_audio)(\*sources[, output, crossfade]) | Concatenate multiple audio files/segments.                                     |
| [`overlay_audio`](#mixing.audio.overlay_audio)(background, overlay[, ...])         | Overlay/mix two audio sources.                                                 |
| [`loop_audio`](#mixing.audio.loop_audio)(source, target_duration_s, \*[, ...])  | Tile an audio source until it fills `target_duration_s`, seamlessly.           |
| [`duck_audio`](#mixing.audio.duck_audio)(bed, sidechain, \*[, duck_db, ...])    | Duck `bed` wherever `sidechain` is loud (sidechain ducking).                   |
| [`save_audio_clip`](#mixing.audio.save_audio_clip)([audio_src, start, end, ...])     | Extract and save an audio clip.                                                |
| [`find_audio_offset`](#mixing.audio.find_audio_offset)(reference_audio, ...[, ...])    | Find the time offset where query_audio best aligns within reference_audio.     |
| [`find_audio_offset_detailed`](#mixing.audio.find_audio_offset_detailed)(reference_audio, ...)  | Align `query_audio` within `reference_audio` — offset **and** confidence.      |
| [`align_clips_to_reference`](#mixing.audio.align_clips_to_reference)(reference_audio, ...)    | Align a SET of clips to one reference — the multi-device / multicam primitive. |
| [`aligned_spans`](#mixing.audio.aligned_spans)(reference_audio, clip_audio, \*)    | The MAXIMAL spans of `clip_audio` that align to `reference_audio`.             |
| [`onset_envelope`](#mixing.audio.onset_envelope)(samples, sample_rate, \*[, ...])   | A channel-robust onset/energy envelope: `(envelope, envelope_rate_hz)`.        |
| [`find_segments`](#mixing.audio.find_segments)(audio, \*[, strategy, ...])         | Find segment boundaries in `audio` using a chosen strategy.                    |
| [`extract_segments`](#mixing.audio.extract_segments)(audio[, segments, output, ...])  | Save each segment as its own audio file.                                       |
| [`segment_by_silence`](#mixing.audio.segment_by_silence)(audio, \*[, ...])              | Find non-silent regions using pydub's silence detector.                        |
| [`segment_by_energy`](#mixing.audio.segment_by_energy)(audio, \*[, sample_rate, ...])  | Split audio at local energy valleys (fade-outs and quiet moments).             |
| [`segment_by_self_similarity`](#mixing.audio.segment_by_self_similarity)(audio, \*[, ...])      | Find boundaries via Foote's checkerboard novelty on the SSM.                   |
| [`segment_by_speech_music`](#mixing.audio.segment_by_speech_music)(audio, \*[, ...])         | Tag regions of audio as `"speech"` or `"music"`.                               |
| [`beat_grid`](#mixing.audio.beat_grid)(audio, \*[, sample_rate, ...])          | Estimate beats, (best-effort) downbeats, and the onset envelope of `audio`.    |

### Classes

| [`Audio`](#mixing.audio.Audio)(src_path, \*[, time_unit, start_time, ...])   | Sliceable interface for audio supporting time-based operations.            |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`AudioSamples`](#mixing.audio.AudioSamples)(audio_src[, start_sample, ...])        | Mapping interface to access audio samples by index.                        |
| [`AudioOffset`](#mixing.audio.AudioOffset)(offset_s, confidence, sample_rate)      | The result of aligning one recording within another.                       |
| [`AlignedSpan`](#mixing.audio.AlignedSpan)(clip_start_s, clip_end_s, ...[, ...])   | A maximal run of CLIP time that tracks the reference at one stable offset. |
| [`ClipAlignment`](#mixing.audio.ClipAlignment)(index, offset_s, confidence, ...)     | Where one clip sits on a reference (song) timeline.                        |
| [`Segment`](#mixing.audio.Segment)(start, end[, label, score])                 | A time-bounded slice of an audio file.                                     |
| [`BeatGrid`](#mixing.audio.BeatGrid)(beat_times, downbeat_times, ...)           | Rhythmic analysis of one audio signal.                                     |

### Exceptions

| [`MixingError`](#mixing.audio.MixingError)                               | Base for every error `mixing` raises on its own behalf.                |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`WindowTooWideForClip`](#mixing.audio.WindowTooWideForClip)(\*, clip_index, ...) | An explicit analysis window leaves a clip no second, independent look. |

### *class* mixing.audio.AlignedSpan(clip_start_s, clip_end_s, offset_s, confidence, support=None, margin=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A maximal run of CLIP time that tracks the reference at one stable offset.

`align_clips_to_reference` answers “where does this clip sit?” with a single
number, which is the right answer only when the clip is one continuous take of the
reference. A clip that was stopped and restarted, or that contains a chunk of
something else, has no such number — and the single-offset model does not say so, it
just returns the offset of whichever part correlated best and describes the rest of
the clip wrongly.

#### clip_start_s

Where this span begins in the CLIP’s own timeline.

#### clip_end_s

Where it ends, in the clip’s timeline.

#### offset_s

Reference-time where this span’s clip-time zero would fall — the same
convention as [`ClipAlignment.offset_s`](#mixing.audio.ClipAlignment.offset_s), so
`reference_time = clip_time + offset_s` holds inside the span.

#### confidence

Representative confidence for the span (the median over its windows,
not the max — a span is only as trustworthy as its typical window). It says
**how well the clip matches where this span puts it**, and on a reference
that repeats verbatim that is a question with several excellent answers —
see [`support`](#mixing.audio.AlignedSpan.support), and gate on both.

#### support

How much of the span’s INDEPENDENT windows’ own evidence
(`MAX_SUPPORT_OVERLAP`) reaches this offset, in `[0, 1]`, before the
consensus vote and relative to `window_s` the same way
[`ClipAlignment.support`](#mixing.audio.ClipAlignment.support) is — or `None` when the span was
built from fewer than `MIN_WINDOWS_FOR_SUPPORT` windows and there was
therefore nothing to agree. A window that reached the offset unaided counts
1.0 and one that only put it on its ballot counts up to
`BALLOT_VOTE_WEIGHT`, so `support > 0.5` still means some window got
there on its own (issue #45). This is the number that knows about repetition.
A reference with no repeats gives 1.0; a verse/chorus reference gives less,
because some windows correlated just as well against the wrong chorus and
only the crowd put them right; an exactly tiling reference gives very
little, because there genuinely is no unique answer and the confidence alone
would never say so. Low support with high confidence means “it fits here
beautifully, and it would fit elsewhere too”.

**\`\`None\`\` is not 1.0.** A span of one window cannot disagree with itself,
so reporting 1.0 there would be a unanimous vote of one — a number that
vouches for an offset nothing corroborated. `None` says “not measured”,
which a caller can fall back from; a manufactured 1.0 is what a caller
trusts.

#### margin

How far this span’s [`support`](#mixing.audio.AlignedSpan.support) tally sits above the tally for the
best DIFFERENT offset — the best one outside `offset_tolerance_s` of this
span’s — or `None` on the same quorum support is `None` on. Same
statistic, same scale and same guidance as
[`ClipAlignment.margin`](#mixing.audio.ClipAlignment.margin), which documents both in full: support is the
floor (“this much of the span’s evidence reaches here”), margin is the
separator (“and nothing else has an equal claim”). Gate on both.

It is the field that tells an exactly tiling reference from a merely
repetitive one. Both report a middling support; only the tiling one reports
a margin near zero, because there every alias is equally true and the span
reports one of them without having chosen (issue #47).

Two merged spans report the duration-weighted mean, and unmeasured plus
measured is unmeasured — the same rule [`support`](#mixing.audio.AlignedSpan.support) merges under.

#### *property* reference_span *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [float](https://docs.python.org/3/builtins/functions.html#float)]*

This span’s extent on the REFERENCE timeline.

### *class* mixing.audio.Audio(src_path, , time_unit='seconds', start_time=None, end_time=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sliceable interface for audio supporting time-based operations.

Provides lazy views into audio segments using slice notation. Slicing returns
new Audio instances (not copies), enabling chained operations.

* **Parameters:**
  * **src_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `AudioSegment`]) – Path to source audio file or AudioSegment
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]) – Unit for slice indices (‘seconds’, ‘samples’, ‘milliseconds’)
  * **start_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time in seconds (for creating sub-views)
  * **end_time** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time in seconds (for creating sub-views)

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>>
>>> # Get segment from 10s to 20s (returns Audio)
>>> segment = audio[10:20]
>>> segment.save("clip.mp3")
>>>
>>> # Use sample numbers as unit
>>> audio_samples = Audio("song.mp3", time_unit="samples")
>>> segment = audio_samples[44100:88200]  # 1 second at 44.1kHz
>>>
>>> # Get last 30 seconds
>>> ending = audio[-30:]
>>>
>>> # Chain operations
>>> trimmed = audio[5:120]  # Trim to 5s-120s
>>> faded = trimmed.fade_in(2).fade_out(3)  # Apply fades
>>> faded.save("final.mp3")
```

#### *property* channels *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Number of audio channels.

#### close()

Release the reference to the in-memory audio (no OS handles to free).

`Audio` is fully in-memory (a decoded `AudioSegment`), so there is
nothing OS-level to close. `close` simply drops the reference so the
data can be garbage-collected promptly; the object should not be used
afterwards.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of this audio/segment in seconds.

#### *property* end_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

End time in seconds (audio duration if not set).

#### fade_in(duration=1.0)

Apply fade-in effect.

* **Parameters:**
  **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with fade applied

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> faded = audio.fade_in(2.0)  # 2 second fade in
```

#### fade_out(duration=1.0)

Apply fade-out effect.

* **Parameters:**
  **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with fade applied

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> faded = audio.fade_out(3.0)  # 3 second fade out
```

#### *property* full_duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Duration of the source audio in seconds.

#### normalize(, headroom=0.1)

Peak-normalize the audio (via `pydub.effects.normalize`).

Boosts (or attenuates) the segment so its loudest peak sits `headroom`
dB below 0 dBFS. Pure pydub — adds no new dependency.

* **Parameters:**
  **headroom** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Target peak distance below 0 dBFS, in dB (keyword-only).
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with normalization applied.

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> louder = audio.normalize()
```

#### overlay(other, position=0.0, , gain_during_overlay=0.0)

Overlay another audio on top of this one.

* **Parameters:**
  * **other** ([`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)) – Audio to overlay
  * **position** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Position in seconds where overlay starts
  * **gain_during_overlay** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Volume adjustment in dB during overlay
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio with overlay applied

### Examples

```pycon
>>> bg = Audio("background.mp3")
>>> voice = Audio("voice.mp3")
>>> mixed = bg.overlay(voice, position=5.0, gain_during_overlay=-6)
```

#### resample(sample_rate)

Change the sample rate (via pydub `set_frame_rate`).

* **Parameters:**
  **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target sample rate in Hz (e.g. `16000`, `44100`).
* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New Audio at the requested sample rate. Pure pydub — adds no new
  dependency.

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> downsampled = audio.resample(16000)
```

#### *property* sample_count *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Total number of samples in this segment.

#### *property* sample_rate *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Sample rate in Hz.

#### *property* samples *: [AudioSamples](mixing.audio.audio_ops.md#mixing.audio.audio_ops.AudioSamples)*

Get sample-by-sample Mapping interface for this audio.

#### save(output=None, , format=None, bitrate='192k', \*\*export_kwargs)

Save this audio/segment to a new audio file.

* **Parameters:**
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input with
    an auto-derived name), a file path, a directory (auto-named), or
    a callable sink. See mixing.egress.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Audio format (mp3, wav, etc.). Auto-detected from extension if None.
  * **bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bitrate for compressed formats
  * **\*\*export_kwargs** – Additional arguments for pydub export
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved file

### Examples

```pycon
>>> audio = Audio("song.mp3")
>>> audio[10:30].save("clip.mp3")
>>> audio[10:30].save("clip.wav", format="wav")
```

#### *property* start_time *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Start time in seconds (0.0 if not set).

#### to_mono()

Downmix to a single channel (via pydub `set_channels(1)`).

* **Return type:**
  [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)
* **Returns:**
  New mono Audio. Pure pydub — adds no new dependency.

### Examples

```pycon
>>> audio = Audio("stereo.mp3")
>>> mono = audio.to_mono()
```

### *class* mixing.audio.AudioOffset(offset_s, confidence, sample_rate)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of aligning one recording within another.

#### offset_s

Time in `reference` where `query` begins (seconds). Positive
means query starts after the reference’s t=0; **negative** means query
began before it (e.g. a phone that started filming before the song).

#### confidence

A scale-invariant normalized cross-correlation coefficient in
`[0, 1]` at the best lag — comparable ACROSS clips of different loudness
and length. ~0.5+ is a strong match; near 0 means no shared component.

#### sample_rate

The analysis sample rate the offset was computed at.

### *class* mixing.audio.AudioSamples(audio_src, start_sample=0, end_sample=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

Mapping interface to access audio samples by index.

Provides dictionary-like access to audio samples with support for negative
indexing and slicing. Samples are returned as normalized float values.

* **Parameters:**
  * **audio_src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `AudioSegment`]) – Path to audio file or AudioSegment
  * **start_sample** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Starting sample index (for segments)
  * **end_sample** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ending sample index (for segments)

### Examples

```pycon
>>> audio_samples = AudioSamples("test_audio.mp3")
>>> sample = audio_samples[0]  # Get first sample
>>> last_sample = audio_samples[-1]  # Get last sample
>>> samples = list(audio_samples[1000:2000])  # Get samples 1000-1999
```

### *class* mixing.audio.BeatGrid(beat_times, downbeat_times, onset_env, onset_hop_s, sample_rate, tempo_bpm)

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

### *class* mixing.audio.ClipAlignment(index, offset_s, confidence, duration_s, coverage, overlaps=True, support=None, margin=None, window_s=None, hop_s=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Where one clip sits on a reference (song) timeline.

#### index

The clip’s position in the input sequence.

#### offset_s

Reference-time where the clip’s audio begins (may be negative).

#### confidence

Normalized cross-correlation coefficient in `[0, 1]`.

#### duration_s

The clip’s own duration (seconds).

#### coverage

`(start_s, end_s)` — the clip’s span \*\*intersected with the
reference timeline\*\* `[0, reference_duration]`. When the clip does not
overlap the reference at all this is a degenerate `(t, t)` and
[`overlaps`](#mixing.audio.ClipAlignment.overlaps) is False; callers building an edit must skip those.

#### overlaps

Whether the clip intersects the reference timeline at all. A clip that
does not is still RETURNED, with its measured offset and confidence — see
[`align_clips_to_reference()`](#mixing.audio.align_clips_to_reference) for why it is not dropped.

#### support

How much of the clip’s analysis windows’ own evidence — before the
consensus vote — reaches [`offset_s`](#mixing.audio.ClipAlignment.offset_s), in `[0, 1]`: “how much of this
clip agrees that this is where it goes”. It is a different question
from [`confidence`](#mixing.audio.ClipAlignment.confidence), which asks only how well the clip matches at the
offset reported, and it is the one that catches the two failures a
coefficient cannot: a clip that matches beautifully **somewhere else too**
(repetitive music), and a clip only PART of which is the reference at all.

**A graded tally, not a headcount** (issue #45). A window whose own argmax
landed on [`offset_s`](#mixing.audio.ClipAlignment.offset_s) contributes a full 1.0; a window that merely put
it on its ballot — could not separate it from its own answer, or had it
nominated by the other feature — contributes up to
`BALLOT_VOTE_WEIGHT`, scaled by how close it scored; a window that
never considered it contributes nothing. Ballot mentions alone cannot carry
the tally past `0.5`, so **\`\`support > 0.5\`\` means at least one independent
window reached this offset unaided** — which is exactly what the whole number
used to mean, now the top half of a range rather than all of it.

\*\*A threshold carried over from before the change no longer means what it
meant.\*\* Every value in `(0, 0.5]` is reachable with zero windows having
found the offset unaided, so a gate at 0.25 now says “some of the clip’s
evidence mentions this offset”, not “a quarter of the windows located it
themselves”. A caller who meant the latter gates at `> 0.5`, and that is
not a cosmetic re-tune: measured at the default window, an exactly tiling
reference — where the offset is a free choice among nine — now lands just
*below* 0.5 and a reference that is one half twice lands just *above* it, so
a gate at 0.5 separates them and a gate at 0.25 passes both.

**The change is a no-op only where the argmax was already decisive.** A long
clip at the default 20 s window is unchanged on a reference that does not
repeat, because every window got there on its own and there is nothing to
add. On a repeating reference the same long clip moves — measured, a 60 s
clip at `window_s=20` went from a headcount of 0.00 to about 0.50. Length
was never the thing that made the old number safe; an undisputed argmax was.

The reason it is graded: an argmax is a real opinion at a 20 s window and
close to a coin flip at 4 s, so a bare headcount got *less* confident exactly
as fitting the window to the clip (issue #41) made the estimator *more*
reliable — backwards, for a trust gate. Measured on real cross-device
material, 21 alignments that were ALL correct reported 0.00-1.00 with six at
0.00: the vote landed on the right offset while no individual window’s argmax
agreed.

Measured on three phone recordings of one commercial track, the whole-clip
argmax was 83 s, 174 s and 83 s wrong while its coefficient looked ordinary;
consensus support was 10/24, 17/37 and 45/61 and pointed at offsets three
independent methods then confirmed to within 40 ms (issue #30).
**\`\`None\`\` when it was not measured** — with `consensus=False`, for a clip
short enough that the window in force gives it fewer than two INDEPENDENT
looks — which takes about `window_s + hop_s` of clip, not one window’s worth,
so at the default grid a 26 s clip had no support either; with
`window_s=None` that threshold moves down to one floor-window plus its hop
(see `_clip_window_and_hop()`) — and when the windows overlap too heavily
to be separate opinions (`MAX_SUPPORT_OVERLAP`). None of those has a
second opinion to compare against, and a support of 1.0 there would be a
unanimous vote of one: a number that VOUCHES for an offset nothing
corroborated. That is not hypothetical — a 15 s clip truly at offset 30.0,
against a reference that is one half twice, comes back at offset 75.0 with
confidence 0.979; and on real material a 10 s clip at `window_s=9.5,
hop_s=0.5` gave two 95%-overlapping windows that agreed on an offset 102 s
wrong, which the tally reported as 1.00 before those windows were excluded.
`None` says “not measured”, which a caller can fall back from.

\*\*It is relative to `window_s`, so a gate on it is too.\*\* Support asks how
much independent evidence agreed, and a shorter window is a weaker opinion.
Grading the tally softens that — a short window’s near-miss is now worth
something rather than nothing — but it does not remove it: measured on the
same three correct cross-device alignments, the headcount this replaced read
0.45/0.64/0.73 at `window_s=20` and 0.19/0.16/0.23 at `window_s=5`. It is
deliberately not normalised — dividing by something to make the numbers look
stable would invent a statistic — so a caller that changes `window_s` must
revisit its threshold, and a caller comparing two clips must compare them at
the same window. With `window_s=None` the window is fitted to each clip, so
two clips of different lengths are NOT at the same window unless both are long
enough to sit at the default; a caller that ranks clips by support should pass
an explicit `window_s` to put them back on one scale, or read
[`window_s`](#mixing.audio.ClipAlignment.window_s) and scale its threshold per clip.

It says how much evidence reaches this offset and nothing about how much
reaches somewhere else — see [`margin`](#mixing.audio.ClipAlignment.margin) for that half.

#### margin

How far [`support`](#mixing.audio.ClipAlignment.support)’s tally for this offset sits above the tally for
the best DIFFERENT offset — the best one outside `offset_tolerance_s` of
it — or `None` when there was nothing to measure. Support says how much of
the clip’s own evidence reaches this offset; margin says \*\*whether anything
else has an equal claim\*\* (issue #47). A coefficient cannot answer that: on
real material the wrong offset’s peak scored 0.987-0.993 of the right one’s,
and support cannot answer it either, because an ambiguous tiling lands mid-
scale by construction — every alias is genuinely well supported.

**Scale.** It is a difference of two support tallies, read on
[`support`](#mixing.audio.ClipAlignment.support)’s scale and in its units. `1.0` means every independent
window found this offset unaided and no window put any other offset forward
at all. `0.0` means some rival offset is backed by exactly as much of the
clip’s evidence as the answer is.

**A wide margin is not a claim of sub-tolerance precision.** Differences
smaller than `offset_tolerance_s` are one hypothesis by construction — the
same equivalence the vote groups ballots by — so a candidate that close is
the answer, not a runner-up, and never subtracts. Measured at the default
tolerance: a rival 0.20 s away leaves the margin at 1.000, and the same
rival at 0.30 s takes it to 0.505. The field says nothing has an equal claim
*at a different offset*; how sharply the offset itself is located is
`offset_tolerance_s`, and it is the caller’s to set. Where no window nominated any rival the
runner-up’s tally is `0.0`, so the margin is simply the support — nothing
else has any claim. It can come back slightly NEGATIVE: the offset is chosen
by the vote’s headcount over every window, while the tally is graded and read
over the independent subset only (`MAX_SUPPORT_OVERLAP`), so the two
need not rank identically. A negative margin says the tally actually prefers
somewhere else, which is the strongest “do not trust this” this object
carries.

**How to combine it with support.** They answer different questions, and a
trust gate wants both:

- high support, wide margin — the evidence points here, and nowhere else;
- high support, margin near zero — the clip fits here \*and fits somewhere
  else exactly as well\*. An exactly tiling reference is the pure case: every
  alias draws the same tally, and no estimator can say which one was meant;
- low support, wide margin — thin evidence, undisputed. A short clip on a
  repeating bed lands here (issue #45): no window’s argmax resolved the tile,
  so the tally sits near `BALLOT_VOTE_WEIGHT`, and yet no other offset
  drew any of it. This is the row a support-only gate refuses and should not;
- low support, margin near zero — nothing is known.

So **support is a floor and margin is a separator**. Measured on the real
cross-device material this was built for — 24 correct alignments against 6
pure-noise clips — the separator is doing nearly all of the work:
`support > 0.5` passed 18 of 24 and refused all 6; adding
`and margin > 0` to it changed nothing (margin never refused what support
passed); and **\`\`margin > 0\`\` alone passed all 24 and still refused all 6**,
recovering every correct short clip the support floor was turning away.
Five of the six noise clips came back NEGATIVE (-0.167 to -0.317) and no
correct alignment did, so a negative margin is worth reading as a refusal
rather than merely as a failure to vouch.

**Keep a floor under it anyway.** The sixth noise clip scored exactly
`+0.000` — refused by an exact tie, which a different draw does not
guarantee — and two correct alignments passed at `+0.014` and `+0.029`
on a support of 0.34-0.35, which is the “thin evidence, undisputed” row with
nothing beneath it. A conjunction such as `support > 0.25 and margin > 0`
scored the same 24/6 there while giving five of the six noise clips a second,
independent reason to fail. Thirty cases is encouraging and is not proof.

Margin is relative to [`window_s`](#mixing.audio.ClipAlignment.window_s) for exactly the reason support is —
it is built from the same tally — so a threshold on it moves with the window
the same way.

**\`\`None\`\` means unmeasured**, never manufactured, on the same quorum as
[`support`](#mixing.audio.ClipAlignment.support) (`MIN_WINDOWS_FOR_SUPPORT`): one independent look has
no runner-up to be ahead of, and a number invented there would VOUCH.

#### window_s

The analysis window this clip’s vote was actually held at, in seconds —
\*\*the scale [`support`](#mixing.audio.ClipAlignment.support) is expressed on\*\*, reported because since the
window is fitted to the clip it is no longer something the caller can infer
from its own arguments. `None` when no vote was held (`consensus=False`),
for the same reason `support` is.

Read it whenever you gate on support. A fixed threshold applied across clips
measured at different windows compares numbers that are not comparable:
measured on real cross-device material, 21 alignments that were all CORRECT
reported the bare argmax headcount from 0.00 to 1.00 (which is why that tally
is now graded — `BALLOT_VOTE_WEIGHT`) depending mostly on clip length,
because a 4 s window on a 12 s clip is a weaker opinion than a 20 s window on
a 60 s one. With this field a caller can scale its gate to the window, or
decline to gate when the window came out small — what it cannot do is read
0.33 and 0.75 as if they answered the same question.

**Necessary, and since #45 no longer sufficient.** A rule of the form “below
the default window, treat the value as unmeasured” was a complete defence
while support was an argmax headcount, because that was the only axis that
moved it. It is not any more: what a repeating reference does to the tally it
does at `window_s=20` too — a 60 s clip there went from 0.00 to about 0.50
on a bed that tiles. So reading this field tells a caller how strong each
opinion was; it does not put the number back on a scale a pre-#45 threshold
was calibrated against. Nothing does. A threshold calibrated on the headcount
has to be re-measured against this statistic, not re-pointed at it.

\*\*And such a guard should be REMOVED by this change, not kept alongside a new
threshold\*\* — measured, not reasoned. A rule that declines to trust support
below the default window has to route those clips somewhere, and the only
thing left is [`confidence`](#mixing.audio.ClipAlignment.confidence). That fallback is weaker than the statistic
the guard was protecting against: on six pure-noise clips at a fitted 6.67 s
window, confidences of 0.017-0.173 passed a consumer’s floor of 0.1 for
**four of the six**, while graded support at `> 0.5` refused all six.
Grading is what makes support usable at a fitted window in the first place,
so a window guard in front of it now costs more than it saves.

#### hop_s

The step between those windows — the other half of the grid, reported for
the same reason and `None` in the same cases. Support counts windows that
are separated enough to be second opinions
(`MAX_SUPPORT_OVERLAP`), so which windows were *eligible* to agree
depends on the hop as much as on the window: a support figure is reproducible
from `(window_s, hop_s)` and not from either alone.

### *exception* mixing.audio.MixingError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base for every error `mixing` raises on its own behalf.

### *class* mixing.audio.Segment(start, end, label=None, score=None)

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

### *exception* mixing.audio.WindowTooWideForClip(, clip_index, clip_duration_s, window_s, hop_s, max_window_s)

Bases: [`MixingError`](mixing.errors.md#mixing.errors.MixingError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

An explicit analysis window leaves a clip no second, independent look.

Raised by [`mixing.audio.align_clips_to_reference()`](#mixing.audio.align_clips_to_reference) when the caller passes a
`window_s` wider than [`max_window_s`](#mixing.audio.WindowTooWideForClip.max_window_s) — the length above which the clip holds
no second INDEPENDENT window, so the consensus vote has nothing to arbitrate and
`support` comes back `None`.

Why an error and not a clamp or a silent single window: the caller asked for a
windowed vote and would otherwise receive a single whole-clip correlation whose
only signal is the ABSENCE of a support number — indistinguishable from a clip
that is genuinely too short to support at any window (issue #43). Clamping would
be worse still: an explicit `window_s` is the caller saying what a window means
for their material, and silently measuring at a different one makes `support`
incomparable across a set for a reason nothing reports.

It subclasses [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) because it is an argument that cannot be honored,
so existing `except ValueError` handlers around alignment keep working.

#### clip_index

Position of the offending clip in the `clips` sequence, or
`None` when the caller did not identify one.

#### clip_duration_s

The clip’s duration in seconds.

#### window_s

The window that was asked for.

#### hop_s

The hop in force — the caller’s, or the one derived from `window_s`.
Reported because it is part of the grid that was asked for; it is NOT part
of the bound, which does not depend on it (see
`_max_supportable_window_s()`).

#### max_window_s

The largest window that still leaves this clip a second,
independent look. The bound is inclusive, so for any clip long enough to be
worth aligning a retry at exactly this window measures — which is what makes
it worth reporting. A degenerate clip of a sample or two has no such window
at all; the value is floored at one sample there and is a lower bound rather
than a promise.

### mixing.audio.align_clips_to_reference(reference_audio, clips, , reference_duration=None, sample_rate=16000, min_overlap_ratio=0.5, feature='envelope', consensus=True, window_s=None, hop_s=None, offset_tolerance_s=0.25, near_tie_ratio=0.05)

Align a SET of clips to one reference — the multi-device / multicam primitive.

Aligns each clip against `reference_audio` (e.g. the clean song) and returns its
offset, a scale-invariant confidence, and its \*\*coverage clamped to the reference
timeline\*\* — so a downstream editor gets valid spans and never references a time the
reference does not cover. Preserves the original `index` so callers can map results
back to inputs.

**Every clip gets a record.** A clip with no temporal overlap is returned with
`overlaps=False` rather than omitted, because omission is how a source silently leaves
the addressable set: a caller that persists this list as *the* alignment artifact ends
up with material it can no longer reference, name, or explain — the file is still there,
but nothing downstream can point at it. Selecting what goes into an edit is a matter of
*referencing* sources and intervals; a source must never disappear from what can be
referenced as a side effect of being measured. Callers building an edit filter on
`overlaps`; callers reporting to a human show all of them, with the reason.

* **Parameters:**
  * **reference_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The signal every clip is aligned to (the song).
  * **clips** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]]) – The clip audio sources (paths, arrays, or `AudioSegment`s).
  * **reference_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The reference timeline length (seconds); computed from
    `reference_audio` when omitted.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (mono).
  * **min_overlap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Passed through to the alignment (see
    [`find_audio_offset_detailed()`](#mixing.audio.find_audio_offset_detailed)).
  * **feature** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Similarity feature the alignment is measured on — see
    `ALIGNMENT_FEATURES`. Under `consensus` it chooses the OFFSET and not
    only the confidence: `'envelope'` lets the onset envelope nominate lags the
    waveform never offers, which on cross-device material is the only way the true
    offset reaches a window’s ballot at all (issue #30).
    Defaults to `'envelope'` \*\*because this function’s whole purpose is the
    cross-device case\*\*, and a raw-waveform coefficient is not a usable trust gate
    there: two microphones in a room are not sample-correlated even when the
    alignment is exact. Measured on a real 6-device shoot, the waveform coefficient
    scored provably-correct alignments at 0.064-0.148 — below any threshold a
    caller would sensibly set — while the envelope scored them 0.441-0.634 and a
    genuine non-match at 0.102. Pass `'waveform'` when the clips come from the
    SAME source as the reference (e.g. verifying an export against its master),
    where sample correlation is meaningful and gives finer confidence resolution.
  * **consensus** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Estimate the offset by putting the clip’s analysis windows to a vote
    rather than by one argmax over the whole clip (issue #30). \*\*On by default,
    because the whole-clip argmax is measurably wrong on repetitive material\*\*:
    it minimises a correlation whose peaks are near-tied at musical periods, so
    it returns whichever repeat won a coin flip, and its coefficient does not
    drop when it does. A spurious peak lands at a different lag in every window
    while the true offset is the one they share, so the windows’ agreement is
    what separates them — and how much of the clip agrees is reported as
    [`support`](#mixing.audio.ClipAlignment.support), with how far that is ahead of the runner-up
    offset as [`margin`](#mixing.audio.ClipAlignment.margin). `False` restores the single
    whole-clip correlation exactly — same offset, same confidence, and
    `support=None`, `margin=None` because nothing was put to a vote. It
    is cheaper (one correlation instead of one per window) and is the right
    choice only when the reference is known not to repeat.
  * **window_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    Analysis window for the vote. Ignored when `consensus` is False.
    `None` (the default) **fits the window to each clip** — see
    `_clip_window_and_hop()`: `min(20 s, clip_duration / 3)`, floored at
    what the onset envelope can carry, so a clip too short to hold three default
    windows still gets a vote and a measured `support` instead of one
    uncorroborated correlation (issue #41). Whatever window each clip ends up
    measured at is reported back as [`ClipAlignment.window_s`](#mixing.audio.ClipAlignment.window_s), because it is
    the scale its `support` is on. A clip of three default windows or
    more is measured at `SPAN_WINDOW_S` exactly, so nothing about a long
    clip’s answer moves. Pass a number to fix the window yourself — an explicit
    value is never overridden.

    **An explicit window a clip cannot hold is REFUSED** with
    [`WindowTooWideForClip`](mixing.errors.md#mixing.errors.WindowTooWideForClip) (issue #43). The bound is
    `clip_duration / (1 + MAX_SUPPORT_OVERLAP)` and does NOT involve `hop_s`
    (`_max_supportable_window_s()`): above it no second INDEPENDENT window
    exists, so the vote asked for cannot be held and the result would be one
    whole-clip correlation reporting `support=None` — the same thing a clip too
    short to support at any window reports, with nothing to tell the two apart.
    The error names the clip, its duration and the largest window that still
    leaves a second look — measurable for any clip long enough to be worth
    aligning, though a degenerate clip of a sample or two has no such window at
    all. To measure such a clip, pass that window, pass `window_s=None` to fit
    the window to each clip, or pass `consensus=False` to ask for the single
    correlation outright.
  * **hop_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    Step between those windows. Ignored when `consensus` is False. `None`
    (the default) is `ADAPTIVE_HOP_RATIO` of whatever window is in force —
    half of it, the default pair’s own ratio — so an adapted grid keeps the
    default grid’s shape.

    \*\*Passing `window_s` alone now moves the hop too\*\*, and that is a change:
    before the window was fitted, an unpassed `hop_s` was a flat 10 s, so
    `window_s=5` meant the pair `(5, 10)` — a hop twice the window, which
    skips over half the clip and was never anyone’s intent. It now means
    `(5, 2.5)`. Nothing else about an explicit `window_s` moved, but a
    `support` measured at `window_s=5` on an earlier version is not
    reproducible here without passing `hop_s=10` alongside it. The pair each
    clip was actually measured at comes back as [`ClipAlignment.window_s`](#mixing.audio.ClipAlignment.window_s)
    and [`ClipAlignment.hop_s`](#mixing.audio.ClipAlignment.hop_s).
  * **offset_tolerance_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How far two windows’ offsets may differ and still count as
    the same answer.
  * **near_tie_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How close a rival peak must score to a window’s best one to join
    the vote — see `NEAR_TIE_RATIO`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ClipAlignment`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.ClipAlignment)]
* **Returns:**
  A list of [`ClipAlignment`](#mixing.audio.ClipAlignment), in input order (minus dropped clips).
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `feature` is not one of `ALIGNMENT_FEATURES`, or
        `near_tie_ratio` is negative.
  * [**WindowTooWideForClip**](mixing.errors.md#mixing.errors.WindowTooWideForClip) – `consensus` is on and an EXPLICIT
        `window_s` is wider than some clip can hold a second independent look at
        (issue #43). The error names that clip, its duration and the largest window
        that would work; it subclasses `ValueError`. Never raised on the
        `window_s=None` path, which fits the window to each clip.

### mixing.audio.aligned_spans(reference_audio, clip_audio, , reference_duration=None, sample_rate=16000, window_s=20.0, hop_s=10.0, min_confidence=0.15, offset_tolerance_s=0.25, merge_gap_s=None, feature='envelope', min_overlap_ratio=0.5, near_tie_ratio=0.05)

The MAXIMAL spans of `clip_audio` that align to `reference_audio`.

The windowed counterpart of [`find_audio_offset_detailed()`](#mixing.audio.find_audio_offset_detailed). Where that answers
“where does this clip sit on the reference?” with one number, this answers “which
PARTS of it sit there, and at what offset each?” — which is the only answerable
question for a clip that was stopped and restarted, or that holds material the
reference does not contain.

A clip that is one continuous take \*\*and that the correlation can verify
throughout\*\* returns exactly one span — the compatibility property a caller
migrating from the single-offset model depends on. The qualification is load-bearing
in two ways, both measured:

- A stretch longer than `merge_gap_s` that nothing can verify (hard silence, a hand
  over the mic) splits the take into two spans reporting the SAME offset. Two
  same-offset neighbours are the *signal* that this happened, not a stop/restart.
- A reference whose repeats are EXACT admits several true answers, and this reports
  one of them with a low [`support`](#mixing.audio.AlignedSpan.support) rather than pretending to have
  chosen (see below).

**Repetition is decided by consensus, not by argmax** (issue #30). A verse/chorus
reference used to shatter one continuous take into 3 spans — one of them reporting a
WRONG offset at 0.985 confidence — because each window picked its lag independently
and a repeated chorus makes two peaks near-tied (measured within 0.2% here, and
0.987-0.993 second-to-first on real music). So each window now puts its near-tied
rivals forward instead of only its argmax, and takes whichever of them the most
other windows can also read. A spurious peak lands at a different lag in every
window; the true offset is the one they share. Measured on a repeated motif:
verse/chorus 3 spans → 1, two identical halves 6 → 1, exact tiling 8 → 1, with the
genuine stop/restart still at 2 and the non-repetitive take still at 1 — a window is
never moved to a lag its own correlation did not already rate a near-tie, and a
restarted take has no such lag near the old offset, so it departs freely.

\*\*What that costs you is told by `support`, not by `confidence`.\*\* The
confidence says how well the clip matches where the span puts it, and on repetitive
material that question has several excellent answers. `support` — the fraction of
the span’s windows that found this offset unaided — is the one that knows: 1.0 on a
reference with no repeats, less where the crowd had to intervene, and very little on
an exactly tiling reference where there is genuinely no unique answer. \*\*Gate on
both.\*\* High confidence with low support means “it fits here beautifully, and it
would fit elsewhere too”.

\*\*And `support` in turn cannot see a runner-up\*\*, which is what
[`margin`](#mixing.audio.AlignedSpan.margin) is for (issue #47). Support measures how much of the
span’s evidence reaches this offset; on an exactly tiling reference every alias is
reached by the same evidence, so each of them would report a similar middling
support and none of them is more true than the others. The margin — this offset’s
tally minus the best other offset’s — reads near zero exactly there, and stays wide
where a short clip’s thin evidence is nonetheless undisputed. \*\*Support is a floor;
margin is a separator.\*\*

A span too short to hold a disagreement reports `support=None`, meaning \*not
measured\* — never 1.0. One window agrees with itself, and a unanimous vote of one
is exactly the kind of number a caller would trust and should not.

\*\*Boundary resolution is `window_s`, and no better.\*\* A window is evidence that its
whole extent aligns; a boundary falling inside a window degrades that window rather
than locating itself within it. So a returned edge is accurate to roughly ±
`window_s`, which is ample for telling two takes apart and NOT enough to cut on.
A caller who needs a tighter edge pays for it by shrinking `window_s` — the cost is
linear in the number of windows. Stated here because a span that looks like a precise
interval and is not is exactly the kind of number that gets used as one.

**Decoded once.** Both signals are loaded through `_load_mono_samples()` a single
time and the windows are slices of the resulting array. Re-decoding per window would
be N times the cost *and* would reintroduce issue #25: pydub’s rate conversion has no
anti-alias filter, so a per-window decode path is a per-window chance to halve the
confidence.

* **Parameters:**
  * **reference_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The signal to align within (e.g. the clean song).
  * **clip_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The recording to dissect.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (mono).
  * **window_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Analysis window — see the resolution note above.
  * **hop_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Step between windows. Defaults to half a window, so every instant is
    covered twice and a boundary cannot hide between windows.
  * **min_confidence** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – A window must reach this for its span to exist at all.
  * **offset_tolerance_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How far a window’s implied offset may drift from its span’s
    before it is treated as a different take.
  * **merge_gap_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – How long an unverified gap between two spans that AGREE on the
    offset may be and still be called one take. `None` (the default) uses one
    window. See `SPAN_MERGE_GAP_S`.
  * **reference_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The reference timeline’s length (seconds); computed from the
    decoded reference when omitted. Spans are trimmed to it, so a returned
    extent is always one the reference can honour. Same keyword, and the same
    purpose, as [`align_clips_to_reference()`](#mixing.audio.align_clips_to_reference).
  * **feature** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – As [`align_clips_to_reference()`](#mixing.audio.align_clips_to_reference) — windowed, so it chooses the
    OFFSET and not only the confidence. Defaults to `'envelope'` for the same
    reason that function does: the cross-device case is what this is for.
  * **min_overlap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Passed through to the correlation.
  * **near_tie_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How close a rival correlation peak must score to a window’s best
    one to join the vote, as a fraction of that best score — see
    `NEAR_TIE_RATIO`. `0.0` disables the consensus pass entirely and
    restores the per-window argmax this function shipped with, which is
    measurably wrong on repetitive material and is offered only for reproducing
    an older result.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`AlignedSpan`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.AlignedSpan)]
* **Returns:**
  Spans in clip order, non-overlapping, and each lying entirely within
  `[0, reference_duration]` on the reference timeline. Empty when nothing in the
  clip aligns — which is a real answer, not a failure.

```pycon
>>> import numpy as np
>>> spans = aligned_spans(song, phone_recording)
>>> [(round(s.clip_start_s), round(s.offset_s)) for s in spans]
[(0, 12), (95, 240)]
```

### mixing.audio.beat_grid(audio, , sample_rate=22050, hop_length=512, start_bpm=120.0, backend='librosa')

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
  A [`BeatGrid`](#mixing.audio.BeatGrid). `downbeat_times` is empty for the librosa backend.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unsupported `backend`.

### mixing.audio.concatenate_audio(\*sources, output=None, crossfade=0.0, \*\*save_kwargs)

Concatenate multiple audio files/segments.

* **Parameters:**
  * **\*sources** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio sources (filepaths or Audio instances)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **crossfade** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Crossfade duration in seconds between segments
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> concatenate_audio("intro.mp3", "main.mp3", "outro.mp3")
>>> concatenate_audio("a.mp3", "b.mp3", output="combined.mp3")
>>> concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)  # 500ms crossfade
```

### mixing.audio.crop_audio(src_path, start=None, end=None, , time_unit='seconds', output=None, \*\*save_kwargs)

Convenience function to crop and save an audio segment.

* **Parameters:**
  * **src_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to source audio
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Start time (None = beginning)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time (None = end of audio)
  * **time_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]) – Unit for start/end values
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional arguments for save operation
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved cropped audio

### Examples

```pycon
>>> crop_audio("song.mp3", 10, 30)  # Crop 10s-30s
>>> crop_audio("song.mp3", 44100, 88200, time_unit="samples")
```

### mixing.audio.duck_audio(bed, sidechain, , duck_db=-12.0, threshold_db=-40.0, attack_s=0.05, release_s=0.4, hold_s=0.2, frame_s=0.02, output=None, \*\*save_kwargs)

Duck `bed` wherever `sidechain` is loud (sidechain ducking).

**What it does.** A level-detector sidechain ducker: the `sidechain`
(typically the dialogue track) is framed at `frame_s`, each frame’s RMS
is compared to `threshold_db`, active frames are extended by `hold_s`,
and the resulting on/off signal drives a gain envelope on `bed` that
falls to `duck_db` with time constant `attack_s` and returns to unity
with `release_s`. The envelope is interpolated to sample resolution
before it is applied, so there is no zipper noise. The result always has
the **bed’s** duration; a shorter sidechain simply leaves the tail
un-ducked.

**What it does not do.** It is not a full compressor: there is no ratio,
knee, or makeup gain — the duck depth is the fixed `duck_db`, not a
function of how loud the sidechain is. There is no lookahead, so with a
short `attack_s` the first few milliseconds of a sudden word can sneak
through at full bed level. Detection is **energy-based, not speech-aware**:
any loud sidechain content (music, a door slam, hiss above
`threshold_db`) ducks the bed just as dialogue would. And the bed is
attenuated, never EQ’d — it does not carve a vocal-band notch.

* **Parameters:**
  * **bed** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – The audio to be ducked (filepath or [`Audio`](#mixing.audio.Audio)).
  * **sidechain** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – The audio that triggers ducking — the dialogue track.
  * **duck_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Gain (dB, `<= 0`) held while the sidechain is active.
  * **threshold_db** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sidechain frames above this RMS dBFS count as active.
  * **attack_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time constant for reaching the ducked level.
  * **release_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time constant for returning to unity.
  * **hold_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How long the duck persists after the last active frame.
  * **frame_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Level-detector frame size (envelope time resolution).
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments.
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file.

### Examples

```pycon
>>> quiet_bed = duck_audio("bed.wav", "dialogue.wav")
>>> duck_audio("bed.wav", "vo.wav", duck_db=-18)  # deeper duck
```

### mixing.audio.extract_segments(audio, segments=None, , output=None, name_template='{stem}_{idx:03d}{ext}', format='mp3', bitrate='192k', strategy='silence', \*\*strategy_kwargs)

Save each segment as its own audio file.

If `segments` is `None`, this calls [`find_segments()`](#mixing.audio.find_segments) first using
`strategy` and `strategy_kwargs`.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **segments** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Either a list of `Segment` objects, or a list of
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
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]]]) – Used only when `segments` is `None`.
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

### mixing.audio.fade_in(src, duration=1.0, , output=None, \*\*save_kwargs)

Apply fade-in effect to audio.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio source (filepath or Audio instance)
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> fade_in("song.mp3", 2.0, output="faded.mp3")
>>> audio = fade_in("song.mp3", 2.0)  # Returns Audio instance
```

### mixing.audio.fade_out(src, duration=1.0, , output=None, \*\*save_kwargs)

Apply fade-out effect to audio.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio source (filepath or Audio instance)
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fade duration in seconds
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> fade_out("song.mp3", 3.0, output="faded.mp3")
>>> audio = fade_out("song.mp3", 3.0)  # Returns Audio instance
```

### mixing.audio.find_audio_offset(reference_audio, query_audio, , sample_rate=16000)

Find the time offset where query_audio best aligns within reference_audio.

Uses FFT-based cross-correlation to find the position in reference_audio
where query_audio starts. This is useful for aligning different recordings
of the same performance — for example, aligning a studio recording (voice

+ instruments) with a camera recording (voice only).

The two audio signals don’t need to be identical; they just need to share
a correlated component (e.g., the same voice in both).

* **Parameters:**
  * **reference_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The longer audio to search within (e.g., extracted
    from a video). Accepts a file path, numpy array, or AudioSegment.
  * **query_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The shorter audio to align (e.g., a studio recording).
    Accepts a file path, numpy array, or AudioSegment.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate for analysis. Lower values are faster but
    less precise. Default 16000 Hz gives ~0.06ms precision, which is
    more than sufficient for alignment purposes.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Offset in seconds — the position in reference_audio where query_audio
  starts. Positive means query begins after the start of reference.

### Examples

```pycon
>>> from mixing.audio import find_audio_offset
>>> # Find where a studio recording aligns with a camera recording
>>> offset = find_audio_offset("camera_audio.wav", "studio.mp3")
>>> print(f"Studio recording starts at {offset:.2f}s in the camera audio")
```

### mixing.audio.find_audio_offset_detailed(reference_audio, query_audio, , sample_rate=16000, min_overlap_ratio=0.5, feature='waveform')

Align `query_audio` within `reference_audio` — offset **and** confidence.

The detailed twin of [`find_audio_offset()`](#mixing.audio.find_audio_offset) (which returns just `offset_s`).
Uses an overlap-normalized cross-correlation so the confidence is a scale-invariant
coefficient in `[0, 1]` — usable both as a per-clip trust gate and to compare
alignments across clips (which the multi-device / multicam case needs). Unlike the
scalar helper’s assumption that `reference` is the longer signal, this handles a
`query` that is longer than, or starts before, the reference (negative offset).

* **Parameters:**
  * **reference_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The signal to align within (e.g. the clean song).
  * **query_audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – The signal to locate (e.g. a phone recording of the song).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analysis sample rate (mono). 16 kHz gives ~0.06 ms precision.
  * **min_overlap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Reject lags overlapping less than this fraction of the
    shorter signal (guards against a tiny-overlap spurious peak).
  * **feature** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which similarity feature the confidence is measured on — see
    `ALIGNMENT_FEATURES`. **Here it moves only the confidence**: one
    correlation over a whole clip has no vote to hold, so the lag is the
    waveform’s either way (`_envelope_then_waveform()`). The windowed
    [`align_clips_to_reference()`](#mixing.audio.align_clips_to_reference) is where the choice also moves the offset.
    Defaults to `'waveform'` here because this is the
    low-level primitive and the caller knows their own signals; use `'envelope'`
    whenever the two recordings came from **different devices**, where a waveform
    coefficient understates a correct alignment several-fold.
    [`align_clips_to_reference()`](#mixing.audio.align_clips_to_reference) — the multi-device primitive — defaults to
    `'envelope'` for that reason.
* **Return type:**
  [`AudioOffset`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.AudioOffset)
* **Returns:**
  An [`AudioOffset`](#mixing.audio.AudioOffset) (`offset_s`, `confidence`, `sample_rate`).

### mixing.audio.find_segments(audio, , strategy='silence', min_segment_duration=0.0, max_segment_duration=None, merge_gap=0.0, pad_start=0.0, pad_end=0.0, \*\*strategy_kwargs)

Find segment boundaries in `audio` using a chosen strategy.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, `AudioSegment`]) – File path, numpy array, or pydub `AudioSegment`.
  * **strategy** (`Union`[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'silence'`, `'energy_novelty'`, `'self_similarity'`, `'speech_music'`], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]]]) – Strategy name (`"silence"`, `"energy_novelty"`,
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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
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

### mixing.audio.loop_audio(source, target_duration_s, , crossfade_s=0.5, output=None, \*\*save_kwargs)

Tile an audio source until it fills `target_duration_s`, seamlessly.

The source is appended to itself with a `crossfade_s` crossfade at every
join (pydub’s equal-gain fade-out/fade-in), then trimmed to *exactly*
`target_duration_s`. This is the “make a 20 s ambient bed last 4 minutes”
primitive: without the crossfade, every loop point is a hard splice and a
waveform discontinuity you can hear as a click.

A source **longer** than the target is simply trimmed — looping is only
ever additive, never a no-op guard the caller has to write.

Because each join consumes `crossfade_s` of timeline, the crossfade is
clamped to half the source’s duration; otherwise a long crossfade over a
short source would never advance.

Note the tail is cut wherever `target_duration_s` lands (mid-loop is
normal), and no fade-out is applied — chain [`fade_out()`](#mixing.audio.fade_out) if the bed
ends exposed.

* **Parameters:**
  * **source** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio to loop (filepath or [`Audio`](#mixing.audio.Audio)).
  * **target_duration_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Duration of the result, in seconds (> 0).
  * **crossfade_s** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Crossfade at each loop join, in seconds. `0` gives hard
    splices. Clamped to half the source duration.
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments.
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file.

### Examples

```pycon
>>> bed = loop_audio("room_tone.wav", 90.0)  # 90s bed
>>> loop_audio("waves.wav", 240.0, output="bed.wav")
```

### mixing.audio.onset_envelope(samples, sample_rate, , hop=160, nfft=1024)

A channel-robust onset/energy envelope: `(envelope, envelope_rate_hz)`.

Log-compressed STFT magnitude, positive first difference, summed over frequency, then
standardized. This is *spectral flux* — it tracks WHEN energy arrives, not the waveform
itself, so it survives the things that destroy raw-waveform similarity between two
devices recording the same sound: different microphone responses, different positions
(hence different room impulse responses), and the resulting phase differences.

Why this matters, measured on a real 6-device shoot: raw-waveform correlation scored
provably-correct alignments (three independent methods agreeing to within 10 ms) at
0.064-0.148, while the envelope scored the same pairs at 0.441-0.634 and a genuine
non-match at 0.102. The raw coefficient could not separate match from non-match; the
envelope separates them by more than 4x.

* **Parameters:**
  * **samples** (`ndarray`) – Mono samples.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Their rate.
  * **hop** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Frames advance by this many samples (sets the envelope’s time resolution).
  * **nfft** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – STFT window length.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`float`](https://docs.python.org/3/builtins/functions.html#float)]
* **Returns:**
  `(envelope, envelope_rate_hz)`. The envelope is zero-mean, unit-variance, so
  correlations of two envelopes are directly comparable.

### mixing.audio.overlay_audio(background, overlay, position=0.0, , mix_ratio=0.5, output=None, \*\*save_kwargs)

Overlay/mix two audio sources.

`mix_ratio` is the prominence of the *overlay*, modeled as a
linear-amplitude crossfade between background-only and overlay-only: the
overlay plays at gain `20·log10(mix_ratio)` and the background is ducked
by `20·log10(1 - mix_ratio)` for the overlap’s duration. So `0.0` =
only the background, `1.0` = only the overlay (during the overlap),
`0.5` = an equal blend (both ~-6 dB).

* **Parameters:**
  * **background** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Background audio (filepath or Audio instance)
  * **overlay** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio)]) – Audio to overlay (filepath or Audio instance)
  * **position** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Position in seconds where overlay starts
  * **mix_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Prominence of the overlay in `[0.0, 1.0]` (see above).
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (return the Audio object), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **\*\*save_kwargs** – Additional save arguments
* **Return type:**
  `Union`[[`Audio`](mixing.audio.audio_ops.md#mixing.audio.audio_ops.Audio), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  Audio instance or Path to saved file

### Examples

```pycon
>>> overlay_audio("music.mp3", "voice.mp3", position=5.0)
>>> overlay_audio("bg.mp3", "sfx.mp3", mix_ratio=0.3)  # 30% overlay, 70% bg
```

### mixing.audio.save_audio_clip(audio_src=None, start=0, end=None, , time_unit=None, output=None, format='mp3')

Extract and save an audio clip.

* **Parameters:**
  * **audio_src** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to audio file. If None, gets from clipboard.
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Start time/sample (default: 0)
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – End time/sample (None = end of audio)
  * **time_unit** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]]) – Unit for start/end (‘seconds’, ‘samples’, ‘milliseconds’)
  * **output** (`Union`[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Where to put the result — None (save beside the input), a file
    path, a directory (auto-named), or a callable sink. See mixing.egress.
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output format
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to saved audio file

### Examples

```pycon
>>> save_audio_clip("song.mp3", 10, 30)  # Save 10s-30s
>>> save_audio_clip(start=5, end=15)  # From clipboard
```

### mixing.audio.segment_by_energy(audio, , sample_rate=16000, frame_seconds=0.5, hop_seconds=0.1, smooth_seconds=2.0, valley_threshold_factor=0.5, min_peak_distance_seconds=30.0, label='segment')

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back 

  ```
  ``
  ```

  Segment\`\`s covering the whole audio.

### mixing.audio.segment_by_self_similarity(audio, , sample_rate=11025, n_fft=2048, hop_seconds=0.25, n_bands=40, kernel_seconds=12.0, novelty_smooth_seconds=4.0, peak_threshold_factor=1.0, min_peak_distance_seconds=30.0, label='song')

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back `Segment``s covering the whole audio. Each
  segment's ``score` is the novelty value at its trailing boundary.

### mixing.audio.segment_by_silence(audio, , silence_thresh_db=-40.0, min_silence_len=1.0, seek_step=0.01, keep_silence=0.0, label='non_silent')

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of `Segment` covering the non-silent regions.

### mixing.audio.segment_by_speech_music(audio, , sample_rate=16000, frame_seconds=0.05, hop_seconds=0.025, window_seconds=1.5, low_energy_threshold=0.5, speech_decision_threshold=0.5, smooth_seconds=5.0, min_segment_duration=3.0, speech_label='speech', music_label='music')

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](mixing.audio.segmentation.md#mixing.audio.segmentation.Segment)]
* **Returns:**
  List of back-to-back 

  ```
  ``
  ```

  Segment\`\`s tagging speech and music regions.

### Modules

| [`audio_ops`](mixing.audio.audio_ops.md#module-mixing.audio.audio_ops)       | Audio editing via slicing interface.                                                                                    |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`audio_util`](mixing.audio.audio_util.md#module-mixing.audio.audio_util)     | Audio utility functions.                                                                                                |
| [`beats`](mixing.audio.beats.md#module-mixing.audio.beats)               | Beat / downbeat / onset analysis of an audio signal — the `mixing[beats]` primitive.                                    |
| [`segmentation`](mixing.audio.segmentation.md#module-mixing.audio.segmentation) | Split a long audio into segments — for example, the individual songs in a concert recording, a DJ mix, or a radio show. |
