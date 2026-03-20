from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_closure import build_learning_dynamics_terminal_closure_summary


def test_learning_dynamics_terminal_closure_is_monotone() -> None:
    summary = build_learning_dynamics_terminal_closure_summary()
    hm = summary["headline_metrics"]
    assert hm["closure_seed"] > 0.0
    assert hm["closure_feature"] > hm["closure_seed"]
    assert hm["closure_structure"] > hm["closure_feature"]
    assert hm["closure_global"] > hm["closure_structure"]
