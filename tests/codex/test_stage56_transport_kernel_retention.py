from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_transport_kernel_retention import build_transport_kernel_retention_summary


def test_transport_kernel_retention_positive() -> None:
    summary = build_transport_kernel_retention_summary()
    hm = summary["headline_metrics"]

    assert hm["readout_retention"] > hm["update_retention"]
    assert hm["stage_retention"] > 0.0
    assert hm["successor_retention"] > 0.0
    assert hm["bridge_retention"] > 0.0
    assert hm["transport_kernel_stability"] > 0.0
    assert hm["retention_margin"] > 0.0
