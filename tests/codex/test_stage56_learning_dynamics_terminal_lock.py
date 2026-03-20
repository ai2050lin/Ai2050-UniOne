from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_lock import build_learning_dynamics_terminal_lock_summary


def test_learning_dynamics_terminal_lock_is_monotone() -> None:
    summary = build_learning_dynamics_terminal_lock_summary()
    hm = summary["headline_metrics"]
    assert hm["locked_seed"] > 0.0
    assert hm["locked_feature"] > hm["locked_seed"]
    assert hm["locked_structure"] > hm["locked_feature"]
    assert hm["locked_global"] > hm["locked_structure"]
