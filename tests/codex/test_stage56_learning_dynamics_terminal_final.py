from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_final import build_learning_dynamics_terminal_final_summary


def test_learning_dynamics_terminal_final_grows_monotonically() -> None:
    summary = build_learning_dynamics_terminal_final_summary()
    hm = summary["headline_metrics"]
    assert hm["final_seed"] > 0.0
    assert hm["final_feature"] > hm["final_seed"]
    assert hm["final_structure"] > hm["final_feature"]
    assert hm["final_global"] > hm["final_structure"]
