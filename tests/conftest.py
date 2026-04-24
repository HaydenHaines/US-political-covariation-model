"""Shared pytest helpers for WetherVane tests."""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def skip_if_missing(*rel_paths: str) -> pytest.MarkDecorator:
    """Skip a test/class when any of the required data files are missing.

    ``data/raw/`` and ``data/assembled/`` are gitignored, so CI never has them.
    Tests that exercise those paths are decorated with this marker so CI
    reports them as skipped rather than failing with ``FileNotFoundError``.
    Locally (where the files are populated) the tests still run.
    """
    missing = [p for p in rel_paths if not (PROJECT_ROOT / p).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"requires gitignored data file(s): {missing}",
    )
