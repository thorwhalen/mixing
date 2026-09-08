"""Guardrail: no library module uses ``print()`` for its own output.

Follow-up to #32 / #38 (the ``quiet=`` seam on ``save_frame``): the same
unconditional-print defect existed across the package, making library calls
unusable for a stdout-parsing consumer such as ``paces``. #39 replaces those
prints with a package-wide ``logging.getLogger(__name__)`` seam (silent by
default via the ``NullHandler`` installed in ``mixing/__init__.py``).

This test walks every ``.py`` file under the ``mixing`` package (excluding
tests) with ``ast`` and asserts no ``print(...)`` call remains, other than an
explicit, reasoned allowlist. A call inside a docstring (e.g. a doctest
example) is just string content to the parser and is never flagged.
"""

import ast
from pathlib import Path

import mixing

PACKAGE_ROOT = Path(mixing.__file__).parent

# (relative path from the package root, line number): reason
ALLOWLIST = {
    ("video/video_ops.py", 551): (
        "Video.save_frame's copy-to-clipboard progress message, gated behind "
        "the quiet= keyword added in #38 (mixing#32) — opt-in interactive "
        "output, not unconditional noise."
    ),
    ("video/video_ops.py", 568): (
        "Video.save_frame's 'Saved frame to: ...' message, gated behind the "
        "same quiet= keyword as above (mixing#32/#38)."
    ),
}


def _print_calls(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_no_unconditional_print_in_library_code():
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT)
        if rel.parts[0] == "tests":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno in _print_calls(tree):
            if (str(rel), lineno) not in ALLOWLIST:
                violations.append(f"{rel}:{lineno}")

    assert not violations, (
        "Unconditional print() found in library code (use "
        "logging.getLogger(__name__) instead, or add a reasoned allowlist "
        f"entry): {violations}"
    )
