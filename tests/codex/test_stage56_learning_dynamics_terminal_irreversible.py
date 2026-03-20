from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_irreversible import build_learning_dynamics_terminal_irreversible_summary


def test_learning_dynamics_terminal_irreversible_is_monotone() -> None:
    summary = build_learning_dynamics_terminal_irreversible_summary()
    hm = summary["headline_metrics"]
    assert hm["irreversible_seed"] > 0.0
    assert hm["irreversible_feature"] > hm["irreversible_seed"]
    assert hm["irreversible_structure"] > hm["irreversible_feature"]
    assert hm["irreversible_global"] > hm["irreversible_structure"]
