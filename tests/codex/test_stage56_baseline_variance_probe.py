from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_baseline_variance_probe import (  # noqa: E402
    build_probe_sets,
    summarize_trials,
)


def test_build_probe_sets_includes_union_and_kernels() -> None:
    robust_summary = {
        "kernel_rows": [
            {"kernel": [1, 2]},
            {"kernel": [3, 4, 5]},
        ]
    }
    rows = build_probe_sets([9, 8], robust_summary)
    assert rows[0] == {"probe_name": "original_union", "indices": [9, 8]}
    assert rows[1]["probe_name"] == "kernel_1,2"
    assert rows[2]["indices"] == [3, 4, 5]


def test_summarize_trials_handles_basic_stats() -> None:
    stats = summarize_trials([1.0, 3.0, 5.0])
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
