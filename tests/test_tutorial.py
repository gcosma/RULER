"""The tutorial's code must run, and its printed outputs must be real.

Every ```python block in tutorial.md is executed here in order, sharing one
namespace — exactly as a reader following along would run them. Whenever a
block is followed by a ```text block of output, the captured stdout must
match it byte for byte, so the outputs shown in the tutorial cannot drift
from what the code actually prints.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).resolve().parent.parent / "tutorial" / "README.md"

FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def test_tutorial_runs_and_outputs_match():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    namespace: dict = {"__name__": "__main__"}
    previous_output = None
    ran = verified = 0
    for language, body in FENCE.findall(TUTORIAL.read_text()):
        if language == "python":
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(body, "tutorial/README.md", "exec"), namespace)
            previous_output = buffer.getvalue()
            ran += 1
        elif language == "text" and previous_output and previous_output.strip():
            assert previous_output.rstrip("\n") == body.rstrip("\n"), (
                "tutorial output drift:\n--- shown ---\n"
                f"{body}\n--- actual ---\n{previous_output}"
            )
            verified += 1
            previous_output = None
        else:
            previous_output = None

    assert ran >= 9, f"expected the tutorial's blocks to run, got {ran}"
    assert verified >= 9, f"expected outputs to be verified, got {verified}"


def test_demo_script_matches_the_tutorial():
    """tutorial/demo.py must print exactly the outputs the tutorial shows."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import runpy

    expected = [
        body.rstrip("\n")
        for language, body in FENCE.findall(TUTORIAL.read_text())
        if language == "text" and "pip install" not in body
    ]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(TUTORIAL.parent / "demo.py"), run_name="__main__")
    printed = buffer.getvalue()
    position = 0
    for chunk in expected:
        index = printed.find(chunk, position)
        assert index >= 0, f"demo output missing tutorial chunk:\n{chunk}"
        position = index + len(chunk)


def test_reproduction_notebook_executes():
    """tutorial/reproduce_breast_cancer.ipynb runs top to bottom and reproduces
    the paper's Breast Cancer results across 10 seeds and all four unlearning
    methods."""
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor

    root = Path(__file__).resolve().parent.parent
    path = root / "tutorial" / "reproduce_breast_cancer.ipynb"
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=600).preprocess(nb, {"metadata": {"path": str(root)}})

    printed = "".join(
        o.get("text", "")
        for cell in nb.cells
        if cell.cell_type == "code"
        for o in cell.get("outputs", [])
        if o.get("output_type") == "stream"
    )
    for method in ("Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB"):
        assert method in printed, f"{method} missing from the results table"

    # The pre-unlearning M4 averaged over 10 seeds must reproduce the paper's
    # Breast Cancer value (~0.60).
    pre = float(re.search(r"mean over 10 seeds = ([\d.]+)", printed).group(1))
    assert 0.57 <= pre <= 0.64, f"aggregate pre-unlearning M4 {pre} outside the paper's ~0.60"

    # The paper's headline for this dataset: every method leaves a negative M2
    # (residual memorisation) and keeps M4 above the 0.50 null.
    for method in ("Gradient Ascent", "NegGrad+", "Fine-Tuning", "SCRUB"):
        row = re.search(rf"{re.escape(method)}\s+([-+][\d.]+)\s+([-+][\d.]+)\s+"
                        rf"([-+][\d.]+)\s+([-+][\d.]+)", printed)
        assert row, f"could not parse the {method} row from:\n{printed}"
        m2, m4 = float(row.group(2)), float(row.group(4))
        assert m2 < 0, f"{method}: M2 {m2} is not negative (paper: residual memorisation)"
        assert m4 > 0.55, f"{method}: M4 {m4} not above the null (paper: still memorised)"
