# mixing.audio.audio_util

Audio utility functions.

### Functions

| [`get_audio_info`](#mixing.audio.audio_util.get_audio_info)(source)   | Get audio properties (duration, sample rate, channels).   |
|---------------------------------------------------------------------------|-----------------------------------------------------------|

### mixing.audio.audio_util.get_audio_info(source)

Get audio properties (duration, sample rate, channels).

* **Parameters:**
  **source** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`, AudioSegment]) – Audio source (filepath, numpy array, or AudioSegment)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary with audio properties

### Examples

```pycon
>>> info = get_audio_info("audio.mp3")
>>> info['duration_seconds']
120.5
```
