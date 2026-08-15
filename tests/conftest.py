"""Shared fixtures.

Tests that need real weights use the checkpoints shipped with the repository
and skip cleanly when they are absent, so the suite still runs in a fresh clone
that has not fetched them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def checkpoints_dir() -> Path:
    return REPO_ROOT / "checkpoints"


@pytest.fixture(scope="session")
def has_checkpoints(checkpoints_dir: Path) -> bool:
    return checkpoints_dir.is_dir() and any(checkpoints_dir.glob("*_orig.pt"))
