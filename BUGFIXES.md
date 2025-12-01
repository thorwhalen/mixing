# Bug Fixes and Solutions

## Date: 2025-11-24

### Issues Encountered and Resolved

This document tracks the bugs found during development and how they were resolved.

---

## Bug #1: MoviePy API Compatibility - `verbose` Parameter

**Problem:**
Tests were failing with:
```
TypeError: write_videofile() got an unexpected keyword argument 'verbose'
```

**Root Cause:**
The `verbose` parameter was deprecated in newer versions of moviepy. The API changed from:
```python
clip.write_videofile(path, verbose=False, logger=None)
```
to just:
```python
clip.write_videofile(path, logger=None)
```

**Solution:**
- Removed `verbose=False` from all `write_videofile()` calls
- Location: `mixing/tests/test_video_audio_integration.py` line 34

**Status:** ✅ FIXED

---

## Bug #2: Audio Volume Adjustment API

**Problem:**
`replace_audio()` function crashed with:
```
AttributeError: 'AudioFileClip' object has no attribute 'with_multiply_volume'
```
when `mix_ratio` was between 0.0 and 1.0 (e.g., 0.5).

**Root Cause:**
The method `with_multiply_volume()` doesn't exist in moviepy's API. The correct way to adjust volume is using the `MultiplyVolume` effect.

**Incorrect Code:**
```python
original_audio = video_clip.audio.with_multiply_volume(original_volume)
adjusted_new_audio = new_audio.with_multiply_volume(new_volume)
```

**Corrected Code:**
```python
from moviepy.audio.fx import MultiplyVolume

original_audio = video_clip.audio.with_effects([MultiplyVolume(original_volume)])
adjusted_new_audio = new_audio.with_effects([MultiplyVolume(new_volume)])
```

**Solution:**
- Imported `MultiplyVolume` from `moviepy.audio.fx`
- Used `audio.with_effects([MultiplyVolume(factor)])` instead of non-existent method
- Location: `mixing/video/video_ops.py` lines ~844-849

**Status:** ✅ FIXED

---

## Bug #3: Missing Audio in Output Video

**Problem:**
When using `replace_audio()` with `mix_ratio=0.0` or `mix_ratio=1.0`, the output video had no audio track - completely silent.

**Root Cause:**
The audio codec wasn't explicitly specified when writing the video file, causing moviepy to either:
1. Skip audio encoding entirely
2. Use an incompatible codec that players couldn't decode

**Solution:**
Explicitly specify the audio codec when writing video:
```python
# Write output with audio codec specified
if 'codec' not in save_kwargs:
    save_kwargs['codec'] = 'libx264'
if 'audio_codec' not in save_kwargs:
    save_kwargs['audio_codec'] = 'aac'
    
final_clip.write_videofile(str(output_path), **save_kwargs)
```

**Location:** `mixing/video/video_ops.py` lines ~858-863

**Status:** ✅ FIXED

---

## Bug #4: Tests Skipped Due to Missing Dependency

**Problem:**
5 out of 14 tests in `test_video_audio_integration.py` were being skipped with:
```
pydub not installed
```

**Root Cause:**
While `pydub` was listed in `setup.cfg` under `install_requires`, it wasn't actually installed in the development environment. The fixtures needed pydub to generate test audio files.

**Solution:**
```bash
pip install pydub
```

**Note:** This is already in `setup.cfg`, so users installing the package properly will have it. This was just a development environment issue.

**Status:** ✅ FIXED

---

## Bug #5: Edge Case - Video Without Original Audio

**Problem:**
When `mix_ratio=0.0` (keep original audio) was used on a video that had no audio track, the function would keep the video silent instead of adding the new audio.

**Root Cause:**
The logic didn't handle the case where the user wants to "keep original" but there is no original audio.

**Solution:**
Added check to use new audio when original is absent:
```python
elif mix_ratio == 0.0:
    # Keep original audio only
    if video_clip.audio is None:
        # No original audio, add new audio anyway
        final_clip = video_clip.with_audio(new_audio)
    else:
        # Keep original - no change needed
        final_clip = video_clip
```

**Location:** `mixing/video/video_ops.py` lines ~828-835

**Status:** ✅ FIXED

---

## Test Results After Fixes

All tests passing:
```
50 tests collected
50 passed
0 failed
0 skipped

Test breakdown:
- test_audio_ops.py: 18 tests ✅
- test_video_audio_integration.py: 14 tests ✅
- test_util.py: 13 tests ✅
- Other tests: 5 tests ✅
```

---

## Lessons Learned

1. **Always check API documentation**: Libraries like moviepy evolve and deprecate methods
2. **Explicit is better than implicit**: Always specify codecs explicitly
3. **Test edge cases**: Consider what happens when optional data (like audio tracks) is missing
4. **Don't skip tests**: Missing dependencies should be documented and installed, not worked around
5. **Real-world testing**: Test with actual files, not just unit tests

---

## Related Files

- `mixing/video/video_ops.py` - Main video operations
- `mixing/tests/test_video_audio_integration.py` - Integration tests
- `setup.cfg` - Package dependencies
- `README.md` - User documentation
