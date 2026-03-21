from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_training_terminal_bridge_v33 import build_training_terminal_bridge_v33_summary


def test_stage56_training_terminal_bridge_v33() -> None:
    summary = build_training_terminal_bridge_v33_summary()
    hm = summary["headline_metrics"]

    for key in [
        "plasticity_rule_alignment_v33",
        "structure_rule_alignment_v33",
        "topology_training_readiness_v33",
        "topology_training_gap_v33",
        "low_risk_expansion_guard_v33",
    ]:
        assert 0.0 <= hm[key] <= 1.0

    assert abs(hm["topology_training_gap_v33"] - (1.0 - hm["topology_training_readiness_v33"])) < 1e-9

    out_path = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v33_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["topology_training_readiness_v33"] == hm["topology_training_readiness_v33"]
