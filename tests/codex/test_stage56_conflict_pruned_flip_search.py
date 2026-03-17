from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_conflict_pruned_flip_search import (  # noqa: E402
    generate_remove_sets,
    remove_indices,
)


def test_generate_remove_sets_covers_sizes() -> None:
    rows = generate_remove_sets([1, 2, 3], max_remove=2)
    assert (1,) in rows
    assert (1, 2) in rows
    assert (2, 3) in rows


def test_generate_remove_sets_respects_min_size() -> None:
    rows = generate_remove_sets([1, 2, 3, 4], max_remove=3, min_remove=2)
    assert (1,) not in rows
    assert (1, 2) in rows
    assert (1, 2, 3) in rows


def test_remove_indices_prunes_requested_values() -> None:
    remaining = remove_indices([1, 2, 3, 4], [2, 4])
    assert remaining == [1, 3]
