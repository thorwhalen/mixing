---
name: mixing-video
description: >
  Use for VIDEO editing with the `mixing` Python package: trim/crop a clip,
  loop it, speed it up or down, replace or mix in a new audio track, normalize
  audio levels, pan/zoom a still image (Ken Burns), build a film from panels,
  concatenate clips, grab/extract frames, make a YouTube-style thumbnail, burn
  in subtitles, or resize for shorts/tiktok/square. Trigger on phrasings like
  "trim this video", "loop the intro", "speed up / slow down this clip",
  "replace the audio with this music", "add a pan-zoom over this photo", "make a
  thumbnail", "burn the subtitles in", "resize for YouTube Shorts", or "stitch
  these clips together". For audio-only work use mixing-audio; for transcription
  / filler removal use mixing-transcript; for TTS re-voicing use mixing-dubbing.
---

# mixing-video — edit video from Python

Slice, transform, and assemble video. Most ops need **ffmpeg** on PATH.

## Quick start

```python
import mixing
assert mixing.has_ffmpeg(), "install ffmpeg (brew install ffmpeg / apt install ffmpeg)"

from mixing.video import Video, crop_video, replace_audio, make_thumbnail

Video("clip.mp4")[5:15].save(output="cut.mp4")        # trim 5s–15s
crop_video("clip.mp4", 5, 15, output="cut.mp4")       # same, one call
replace_audio("clip.mp4", "music.mp3", output="scored.mp4")
make_thumbnail("clip.mp4", text="My Title", output="thumb.jpg")
```

`import mixing` is light (no moviepy/opencv until you touch a heavy name). Use
the facade (`mixing.Video`, `mixing.crop_video`, …) or import the subpackage
(`from mixing.video import ...`). Note `THUMBNAIL_SIZE` / `YOUTUBE_THUMB_SIZE` /
`SOCIAL_SIZES` live on `mixing.video`, not the top-level facade.

## The `output` egress protocol (applies to every writer below)

One `output` param: `None` → save beside input, return `Path`; a **file path** →
write+return `Path`; a **directory** → auto-named file; a **callable** → sink.
Time is **seconds (float)** unless noted. Exceptions flagged inline:
`ken_burns_*` and `concatenate_videos` predate the protocol (see below).

## `Video` — sliceable, lazy, path-backed

```python
v = Video("movie.mp4")                       # time_unit="seconds" (default)
v.duration, v.fps, v.frame_count             # lazy props (full_duration = source length)
v[10:20]                  -> Video           # a 10s segment view (no copy)
v[-30:]                   -> Video           # last 30s
v[15]                     -> np.ndarray       # single frame at 15s (BGR, from cv2)
v[10:20].save(output="seg.mp4")              # render the segment
v.save_frame(15, output="f.png")             # write one frame
v.to_clip()                                  # moviepy VideoFileClip (caller closes)
for frame in v.frames[0:100]: ...            # .frames -> VideoFrames Mapping[int, ndarray]

Video("movie.mp4", time_unit="frames")[100:500]   # slice by frame number instead
with Video("movie.mp4") as v: d = v.duration       # context-managed (releases handles)
```

- Slices return new `Video` views; slicing is bounds-clamped; a `step` raises.
- An int/float index returns a **frame** (numpy BGR); a slice returns a `Video`.
- `save(output=None, *, codec="libx264", audio_codec="aac", **write_kwargs)`.
- `save_frame(time_or_frame=None, output=None, *, image_format="png",
  copy_to_clipboard=False)` — `output=False` means clipboard-only (needs
  `copy_to_clipboard=True`).

## Core transforms (all file→file, honor `output`)

```python
crop_video("in.mp4", 5, 15, output="out.mp4")                 # trim; start==end -> single frame
crop_video("in.mp4", 100, 500, time_unit="frames")            # by frame number
loop_video("intro.mp4", 3, output="x3.mp4")                   # n_loops>=1, repeats the clip
change_speed("in.mp4", 2.0, output="fast.mp4")                # 2.0=2x faster, 0.5=half (audio too)
normalize_audio("lecture.mp4", output="leveled.mp4")          # even out volume swings
```

### replace_audio — swap or blend the soundtrack

```python
replace_audio("v.mp4", "music.mp3", output="out.mp4")                 # mix_ratio=1.0 (default): only new
replace_audio("v.mp4", "bgm.mp3", mix_ratio=0.0, output="out.mp4")    # keep original only
replace_audio("v.mp4", "bgm.mp3", mix_ratio=0.5, output="out.mp4")    # equal blend
replace_audio("v.mp4", "voice.mp3", mix_ratio=0.7)                    # 70% new / 30% original
```

`mix_ratio` (0.0–1.0): `1.0`=only new, `0.0`=keep original, `0.5`=blend.
`match_duration=True` (default) loops/trims the audio to the video length.

## Ken Burns (pan/zoom) — mixing wrappers over the `burns` package

mixing wraps burns' renderers so they speak the **`output` protocol** like
everything else (`output=None` lets burns auto-name beside the image).
`output_size=(w,h)` sets the render size; the pan/zoom path is a `BurnsPath`.

```python
from mixing.video import ken_burns_video, ken_burns_film

# Animate a still image into a clip (default path zooms in slightly)
ken_burns_video("photo.jpg", duration=6.0, fps=30, output_size=(1920, 1080),
                output="pan.mp4")

# Stitch a multi-panel film; mux a pre-built audio track over the panels.
# Each panel is an (image, BurnsPath, duration_s) TRIPLE — the path is required.
from burns import BurnsPath
panels = [
    ("p1.jpg", BurnsPath(), 5.0),
    ("p2.jpg", BurnsPath(), 4.0),
]
ken_burns_film(panels, fps=30, audio_path="film_audio.wav", output="film.mp4")
```

### assemble_audio_track — build the audio that lines up with a film

`output` is **required** (no input file to derive a name from). Returns `None`
and writes nothing when every slot is silent.

```python
from mixing.video import assemble_audio_track

assemble_audio_track(
    [("voice1.mp3", 5.0), (None, 3.0), ("voice2.mp3", 4.0)],  # (audio|None, duration_s) per panel
    output="film_audio.wav",          # WAV pcm_s16le; slot-for-slot with ken_burns_film panels
)
# then: ken_burns_film(panels, audio_path="film_audio.wav", output="film.mp4")
```

Each segment occupies exactly its `duration_s` (audio then silence padding; an
over-long clip is trimmed). `sample_rate=44100` by default.

## Thumbnails

```python
from mixing.video import make_thumbnail, THUMBNAIL_SIZE, YOUTUBE_THUMB_SIZE

make_thumbnail("video.mp4", output="thumb.jpg")                      # frame at 85% of duration
make_thumbnail("video.mp4", at_time=12.5, text="Episode 4", output="thumb.jpg")
make_thumbnail("video.mp4", size=YOUTUBE_THUMB_SIZE, output="t/")    # 1280x720 (== THUMBNAIL_SIZE)
```

`make_thumbnail(video, *, at_time=None, text=None, output=None, size=(1280,720))`
— all keyword-only after `video`. `text` is overlaid bottom-left on a dark
gradient band. Default `output=None` writes `<stem>.thumb.jpg` beside the video.

## Subtitles

```python
from mixing.video import write_subtitles_in_video

write_subtitles_in_video("v.mp4", "subs.srt", output="captioned.mp4")  # SRT path
write_subtitles_in_video("v.mp4", srt_content_string)                  # or raw SRT text
write_subtitles_in_video("v.mp4")                                      # None -> sibling <stem>.srt
```

`write_subtitles_in_video(video, subtitles=None, output=None, *, style=None,
use_ffmpeg=True, start_time=None, auto_detect_audio_start=False, ...)`. Burns
captions in via ffmpeg (~100x faster than the moviepy fallback). `style` is a
`SubtitleStyle(font_size=, color=, position=, font_name=)`. `start_time` (float
seconds, or `True` to auto-detect audio onset) shifts the first cue.

## Dimensions & social presets

`SOCIAL_SIZES` maps names → `(w, h)`: `youtube` (1920x1080), `shorts`/`story`/
`tiktok` (1080x1920), `square` (1080x1080). These resize helpers operate on
**moviepy clips** and **return clips** (not the `output` protocol) — get a clip
with `Video.to_clip()`, resize, then `write_videofile`.

```python
from mixing.video import Video, SOCIAL_SIZES, resize_to_dimensions

w, h = SOCIAL_SIZES["shorts"]
with Video("wide.mp4").to_clip() as clip:
    vertical = resize_to_dimensions(clip, w, h, method="social")  # blurred-bg letterbox
    vertical.write_videofile("shorts.mp4")
# methods: "fit" (pad), "fill" (crop), "stretch" (distort), "social" (blurred bg)
# get_video_dimensions(clip) -> (w, h);  normalize_video_dimensions([clips], ...) -> [clips]
```

## Concatenating clips

`concatenate_videos` uses `output=` (a path, or `True` to name from a folder),
but note it **returns a moviepy clip** (not a Path) which the caller must
`.close()`.

```python
from mixing.video import concatenate_videos

final = concatenate_videos(["a.mp4", "b.mp4", "c.mp4"], output="joined.mp4")
final.close()
final = concatenate_videos("clips_folder/", output=True)   # auto-name from folder
final.close()
```

Different sizes are reconciled via `normalize_dimensions` (`"social"` default;
also `"fit"`/`"fill"`/`"stretch"`/`False`). `transform_clips=` lets you inject
transitions (e.g. `crossfade_transition`, `fade_through_black` from the module).

## AI video generation (Veo) — optional

Needs `pip install 'mixing[gen]'` + Google Cloud auth (`GOOGLE_CLOUD_PROJECT` +
ADC or a service-account JSON). Not on the lazy facade — import explicitly.

```python
from mixing.video.genai import generate_video

path = generate_video("A serene forest at dawn", output="forest.mp4")  # output protocol + path
op   = generate_video("A serene forest at dawn", output=False)         # raw operation, no save
```

`generate_video(prompt, first_frame=None, last_frame=None, *, output=save_generated_videos,
model=..., aspect_ratio="16:9", duration_seconds=5, project_id=..., ...)`. `output`
follows the egress protocol with two sentinels: default auto-saves to temp and
returns path(s); `output=False` returns the raw operation.

## Gotchas

- `output=None` for file producers writes **next to the input** — pass an
  explicit `output` for a known location.
- `concatenate_videos` follows `output=` but **returns a moviepy clip** (not a
  Path) — you must `.close()` it. `ken_burns_film`'s `output` is required.
- `assemble_audio_track`'s `output` is **required** and it returns `None` (writes
  nothing) when all segments are silent — guard before muxing.
- A `Video` integer/float index returns a **frame** (numpy BGR); a slice returns
  a `Video`. `v[15]` is the frame at 15 *seconds* (or 15 *frames* under
  `time_unit="frames"`).
- `change_speed` and `replace_audio(match_duration=True)` change/retime audio too.
- Subtitle `start_time=True` (or `auto_detect_audio_start=True`) needs the video
  to detect audio onset; the moviepy fallback (`use_ffmpeg=False`) is much slower.
