from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_failure_boundary_trigger import build_failure_boundary_trigger_summary


def test_stage57_failure_boundary_trigger() -> None:
    summary = build_failure_boundary_trigger_summary()
    hm = summary["headline_metrics"]

    assert hm["live_boundary_pass_rate"] == 1.0
    assert hm["triggerability_score"] == 1.0
    assert hm["counterexample_activation_score"] > 0.55
    assert hm["boundary_system_readiness"] > 0.70
    assert all(summary["live_checks"].values()) is False
    assert all(summary["synthetic_stress"].values())

    out_path = ROOT / "tests" / "codex_temp" / "stage57_failure_boundary_trigger_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["boundary_system_readiness"] == hm["boundary_system_readiness"]
