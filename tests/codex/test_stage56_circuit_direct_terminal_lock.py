from __future__ import annotations

import sys
from pathlib import Path


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_circuit_direct_terminal_lock import build_circuit_direct_terminal_lock_summary


def test_circuit_direct_terminal_lock_margin_is_positive() -> None:
    summary = build_circuit_direct_terminal_lock_summary()
    hm = summary["headline_metrics"]
    assert hm["direct_margin_v5"] > hm["direct_attractor_v5"]
    assert hm["direct_gate_v5"] < hm["direct_binding_v5"]
