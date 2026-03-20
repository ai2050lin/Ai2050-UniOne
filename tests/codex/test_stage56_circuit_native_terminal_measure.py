from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_circuit_native_terminal_measure import build_circuit_native_terminal_measure_summary


def test_circuit_native_terminal_measure_margin_is_positive() -> None:
    summary = build_circuit_native_terminal_measure_summary()
    hm = summary["headline_metrics"]
    assert hm["direct_binding_v3"] > hm["direct_gate_v3"]
    assert hm["direct_attractor_v3"] > hm["direct_gate_v3"]
    assert hm["direct_margin_v3"] > 0.0
