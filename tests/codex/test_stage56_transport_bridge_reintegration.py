from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tests" / "codex"))

from stage56_transport_bridge_reintegration import build_transport_bridge_reintegration_summary


def test_transport_bridge_reintegration_positive() -> None:
    summary = build_transport_bridge_reintegration_summary()
    hm = summary["headline_metrics"]

    assert 0.0 < hm["restricted_readout_gain"] < 1.0
    assert 0.0 < hm["admissible_update_strength"] <= 1.0
    assert 0.0 < hm["stage_transport_strength"] < 1.0
    assert 0.0 < hm["successor_alignment_strength"] <= 1.0
    assert 0.0 < hm["protocol_bridge_strength"] <= 1.0
    assert hm["protocol_bridge_term"] > 0.0
