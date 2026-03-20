from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_final_closure import build_learning_dynamics_terminal_final_closure_summary


def test_learning_dynamics_terminal_final_closure_is_monotone() -> None:
    summary = build_learning_dynamics_terminal_final_closure_summary()
    hm = summary["headline_metrics"]
    assert hm["closure_seed_v2"] > 0.0
    assert hm["closure_feature_v2"] > hm["closure_seed_v2"]
    assert hm["closure_structure_v2"] > hm["closure_feature_v2"]
    assert hm["closure_global_v2"] > hm["closure_structure_v2"]
