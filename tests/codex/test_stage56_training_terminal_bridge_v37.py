from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_training_terminal_bridge_v37 import build_training_terminal_bridge_v37_summary


def test_stage56_training_terminal_bridge_v37() -> None:
    summary = build_training_terminal_bridge_v37_summary()
    hm = summary["headline_metrics"]

    for key in [
        "plasticity_rule_alignment_v37",
        "structure_rule_alignment_v37",
        "topology_training_readiness_v37",
        "topology_training_gap_v37",
        "systemic_low_risk_band_guard_v37",
    ]:
        assert 0.0 <= hm[key] <= 1.0

    assert abs(hm["topology_training_gap_v37"] - (1.0 - hm["topology_training_readiness_v37"])) < 1e-9

    out_path = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v37_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["topology_training_readiness_v37"] == hm["topology_training_readiness_v37"]
