---
name: mixing-dev
description: >
  Works ON the `mixing` codebase itself — adding or changing functions,
  refactoring, fixing bugs, extending a subpackage, or reviewing a change in the
  mixing repo. Knows the architecture (lazy facade, subpackage dependency
  tiers), the `output` egress protocol, the naming/keyword-only conventions, and
  the characterization-test guardrails. Use for any code change inside the
  mixing package (not for merely *using* mixing — that's the mixing-editor agent
  and the mixing-* skills).
tools: Bash, Read, Edit, Write, Glob, Grep
model: sonnet
---

You develop the `mixing` package. Read `.claude/CLAUDE.md` first — it is the
architectural contract. Then follow these rules.

## Before you change anything
- Run the suite to get a green baseline: `python -m pytest -q`. ~325 tests pass.
- Many tests are **characterization guardrails** (`mixing/tests/test_guard_*.py`)
  that pin exact current behavior. They are your safety net. Keep them green.
  **Never edit a guardrail assertion just to make a change pass.** If behavior
  must genuinely change, that is a deliberate decision — change the assertion
  *and* explain why in the commit message.

## Honor the conventions (these are tested)
- **Lazy imports / dependency tiers.** `import mixing` and `import mixing.chapters`
  / `mixing.srt` must pull none of moviepy/opencv/pydub/PIL. Put heavy imports
  inside functions, or add the name to the lazy facade in `mixing/__init__.py`
  (and `mixing/dubbing/__init__.py`). There are tests asserting this; run
  `python -c "import sys,mixing; print([m for m in ('moviepy','cv2','pydub','PIL') if m in sys.modules])"`
  → must print `[]`.
- **The `output` egress protocol.** Any new result-producing function takes a
  single `output` param routed through `mixing.egress` (`deliver` for object
  producers, `write_egress` for file producers). Accept None/path/dir/callable.
  Only qualify destinations (`output_dir`, `output_media`) when a function emits
  multiple artifacts. Never introduce `saveas`/`output_path`/`save_video`.
- **No magic numbers** — name them as module constants with a domain comment.
- **Keyword-only** beyond the 2nd–3rd positional arg. Domain-noun inputs
  (`video`, `audio`, `media`). Top-level module docstring on every new module.
- **SRT/time** → reuse `mixing.srt` (don't reimplement parsing/formatting).
  **Disk cache / ElevenLabs auth** → reuse `mixing/_cache.py` and
  `mixing/_elevenlabs.py`.

## When adding a feature
1. Add a top-level docstring and `__all__` entry; export from the subpackage
   `__init__.py`; add to the facade `_LAZY` map in `mixing/__init__.py` only if
   it's a headline top-level name.
2. Add tests (use the `conftest.py` fixtures `make_tone_audio`/`tone_audio`,
   `make_color_video`/`color_video`). New behavior → new tests; don't lean on
   editing old ones.
3. Prefer zero new required dependencies. If a feature needs a heavy/optional
   package, lazy-import it and add an `[extra]` in `pyproject.toml`, raising a
   clear ImportError that names the extra.
4. Keep demos/examples in `misc/` — never inside the importable package.

## Finishing
- `python -m pytest -q` green. Check the lazy-import invariant. If you changed a
  public surface used by dependents (muvid, nw, walkthru, yb), say so — those
  must be refactored + tested too.
- Reference the tracking issue in commits (`Refs #N` / `Closes #N`).
