from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_training_terminal_bridge_v36 import build_training_terminal_bridge_v36_summary


def test_stage56_training_terminal_bridge_v36() -> None:
    summary = build_training_terminal_bridge_v36_summary()
    hm = summary["headline_metrics"]

    for key in [
        "plasticity_rule_alignment_v36",
        "structure_rule_alignment_v36",
        "topology_training_readiness_v36",
        "topology_training_gap_v36",
        "systemic_low_risk_broadening_guard_v36",
    ]:
        assert 0.0 <= hm[key] <= 1.0

    assert abs(hm["topology_training_gap_v36"] - (1.0 - hm["topology_training_readiness_v36"])) < 1e-9

    out_path = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v36_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["topology_training_readiness_v36"] == hm["topology_training_readiness_v36"]
