from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_dynamics_terminal_canonical_closure import build_learning_dynamics_terminal_canonical_closure_summary


def test_learning_dynamics_terminal_canonical_closure_is_monotone() -> None:
    summary = build_learning_dynamics_terminal_canonical_closure_summary()
    hm = summary["headline_metrics"]
    assert hm["canonical_seed"] > 0.0
    assert hm["canonical_feature"] > hm["canonical_seed"]
    assert hm["canonical_structure"] > hm["canonical_feature"]
    assert hm["canonical_global"] > hm["canonical_structure"]
