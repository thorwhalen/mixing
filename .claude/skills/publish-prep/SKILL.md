---
name: publish-prep
description: >
  Prepare a video or audio file for publication. Produces an SRT subtitle file
  (creating it if missing via `mixing.transcript`), then derives a publication-
  ready title, description, and timestamped chapters from the transcript — and
  writes them to a `.publish.txt` next to the media. Defaults to YouTube
  conventions for video files and Spotify-podcast conventions for audio files,
  unless the user specifies otherwise. Use when the user has a media file and
  asks for "subtitles and chapters", "YouTube chapters", "podcast description",
  "title and description", "transcribe and summarize", "prep for publishing",
  "metadata for this video", or anything that combines transcription with
  publication-ready text. Also triggers when the user mentions a media file
  alongside YouTube, Spotify, or "publish".
---

# publish-prep

Take a video or audio file → produce an SRT transcript and a publication-ready
title, description, and chapters list.

## Defaults

- **Video extension** (`.mp4`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.avi`): assume
  the user is publishing on **YouTube**.
- **Audio extension** (`.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.opus`):
  assume the user is publishing a **podcast on Spotify**.
- The user can override either default explicitly.

## Workflow

### Reuse-existing rule (applies to both outputs)

Both outputs of this skill — the `.srt` and the `.publish.txt` — are
**reused if they already exist next to the media**. Do not regenerate them
silently. The reasoning: re-transcribing costs API credits, and the user
may have hand-edited either file.

When you reuse an existing file, **tell the user explicitly** in your
response, in this form:

> Using existing `<filename>`. To regenerate, delete or rename it and
> run again.

If both files already exist, you can short-circuit the whole skill:
print the contents of the existing `.publish.txt` and the
reuse-warning, and stop.

### 1. Locate or create the SRT

The SRT lives alongside the media: same folder, same basename, `.srt` extension.

```python
from pathlib import Path
media = Path("<path/to/media>")
srt_path = media.with_suffix(".srt")
```

If `srt_path` already exists, **read it and warn the user it was reused**
(see the reuse-existing rule above). Do not re-transcribe — Scribe is
billed per call, and the existing file may have been hand-corrected.

Otherwise generate it with `mixing.transcript` (the same project this skill
ships in):

```python
from mixing.transcript import transcribe, words_to_srt

resp = transcribe(media, cache=True)   # on-disk cache → re-runs are free
srt = words_to_srt(resp["words"])
srt_path.write_text(srt, encoding="utf-8")
```

Notes:
- `transcribe` uses ElevenLabs Scribe via HTTP. Requires `ELEVENLABS_API_KEY`
  in env (check before running and surface a clear error if missing).
- `cache=True` stores the raw response keyed by the audio bytes + parameters;
  re-running for the same file does not re-bill the API.
- `transcribe` accepts both audio and video files (it sends the bytes as-is;
  Scribe extracts audio).

### 2. Read the SRT and derive metadata

Read the full SRT into context. From it, produce three things:

#### Title
- 1 line, punchy.
- For YouTube: ≤ ~70 characters, search-friendly, avoids clickbait unless the
  user's existing branding leans that way.
- For Spotify podcast: episode title — concise, descriptive, reflects the
  episode's specific subject (assume the show name is set elsewhere; do not
  prefix the title with the show name unless the user asks).

#### Description
- For YouTube: 2–4 short paragraphs. Lead with what the video is about and who
  it's for. Optionally end with a one-line CTA only if the transcript suggests
  one (do not invent links/handles).
- For Spotify: 1–3 paragraphs in podcast-show-notes style. Same content rules.
- Do not invent facts that aren't in the transcript. If the speaker mentions a
  product/project name, use it verbatim.

#### Chapters
- Format: one chapter per line, `M:SS Title` or `MM:SS Title` (use `H:MM:SS`
  if the media is ≥ 1 hour).
- **First chapter must start at `0:00`** (YouTube requirement; Spotify follows
  the same convention for chapter markers in supported clients).
- **Each chapter must be ≥ 10 seconds long** (YouTube requirement).
- **At least 3 chapters** (YouTube requirement for chapters to render).
- Aim for 4–8 chapters on a 5–15 minute piece; scale up for longer pieces but
  do not over-segment — chapters should mark real topic shifts, not every
  paragraph.
- Pick start times that match cue boundaries in the SRT (don't drop a chapter
  in the middle of a sentence).
- Title each chapter in 2–7 words. No leading numbering ("1. ", "Ch 1:") —
  YouTube's UI numbers them automatically.

### 3. Print + write the publish file

The publish file lives alongside the media at
`<media-basename>.publish.txt`.

**If it already exists, reuse it.** Read its contents, print them to the
user, and emit the reuse warning (see the reuse-existing rule above).
Do not re-derive title/description/chapters — the user may have edited
the file by hand.

Otherwise, print the title, description, and chapters to the user in that
order, clearly labeled, then write the file with this exact layout:

```
PLATFORM: <YouTube | Spotify (podcast) | other-as-specified>

TITLE
<title>

DESCRIPTION
<description>

CHAPTERS
0:00 First chapter
1:23 Second chapter
...
```

Use `Write` to create the file. **Never overwrite an existing
`.publish.txt`** — if one is there, the reuse path above takes over.

## When the user gives extra direction

- Different platform ("publish on Vimeo", "this is for an internal Slack
  recap"): adapt conventions accordingly. Vimeo: similar to YouTube. Internal:
  drop SEO concerns, lean toward "executive summary" tone.
- Existing title/description they want to refine: read theirs first, then
  produce a revised version that matches their voice.
- They only want chapters (not title/description) or vice versa: skip the
  parts they didn't ask for, but still write what you produced into the
  `.publish.txt` (and leave the unrequested sections blank with a comment).
- They ask to skip the SRT generation: only do this if the SRT already exists
  and they explicitly say so. Otherwise the SRT is non-negotiable input.

## What this skill is NOT for

- Editing the media itself (cuts, fades, overlays). For that, use ffmpeg
  directly or other parts of the `mixing` package.
- Filler-word removal. That's `mixing.transcript.remove_fillers`.
- Translation/localization. Scribe has `language_code` for source-language
  hints, but this skill does not produce translated subtitles.

## Quick reference: the full one-shot script

```python
from pathlib import Path
from mixing.transcript import transcribe, words_to_srt

media = Path("<path/to/media>")
srt_path = media.with_suffix(".srt")
publish_path = media.with_suffix(".publish.txt")

# Reuse existing publish.txt if present — short-circuit the whole skill.
if publish_path.exists():
    print(publish_path.read_text(encoding="utf-8"))
    print(f"\n[reused {publish_path.name} — delete or rename to regenerate]")
    raise SystemExit

# Reuse existing SRT if present, otherwise transcribe.
if srt_path.exists():
    print(f"[reused {srt_path.name} — delete or rename to retranscribe]")
else:
    resp = transcribe(media, cache=True)
    srt_path.write_text(words_to_srt(resp["words"]), encoding="utf-8")

srt = srt_path.read_text(encoding="utf-8")
# Read `srt` into context, derive title / description / chapters,
# print them, and write <basename>.publish.txt.
```
