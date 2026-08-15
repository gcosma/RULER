"""The documentation's code must run.

Every ```python block in the manual and the tutorial is executed here, in
order, sharing one namespace per document -- exactly as a reader following
along would run them. A block whose first line is a ``# requires:`` comment
documents code that needs the reader's own model or training loop and is
skipped.

This keeps the docs from rotting: an API change that breaks an example breaks
the test suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"

FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def python_blocks(path: Path) -> list[str]:
    return FENCE.findall(path.read_text())


def run_document(path: Path) -> int:
    """Execute a document's runnable blocks sequentially; return how many ran."""
    namespace: dict = {"__name__": "__main__"}
    ran = 0
    for block in python_blocks(path):
        if block.lstrip().startswith("# requires:"):
            continue
        exec(compile(block, f"{path.name}", "exec"), namespace)
        ran += 1
    return ran


def test_manual_examples_run():
    ran = run_document(DOCS / "manual.md")
    assert ran >= 8, f"expected the manual's example blocks to run, got {ran}"


def test_tutorial_runs_start_to_finish():
    torch = pytest.importorskip("torch")  # noqa: F841  (parts 2-7 need it)
    ran = run_document(DOCS / "tutorial.md")
    assert ran >= 8, f"expected the tutorial's blocks to run, got {ran}"


def test_choosing_inputs_examples_are_runnable_or_marked():
    """Every block either runs or carries a ``# requires:`` marker.

    An unmarked block that needs the reader's own model or data would raise
    a NameError here, so this catches illustrative fragments that pose as
    runnable code.
    """
    ran = run_document(DOCS / "choosing-inputs.md")
    assert ran >= 0


def test_requires_blocks_are_marked_consistently():
    """Skipped blocks must say what they need, so a reader isn't misled."""
    for name in ("manual.md", "tutorial.md", "choosing-inputs.md"):
        for block in python_blocks(DOCS / name):
            first = block.lstrip().splitlines()[0]
            if first.startswith("# requires:"):
                assert len(first) > len("# requires: "), name
