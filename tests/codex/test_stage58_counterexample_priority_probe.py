from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_counterexample_priority_probe import build_counterexample_priority_probe_summary


def test_stage58_counterexample_priority_probe() -> None:
    summary = build_counterexample_priority_probe_summary()
    hm = summary["headline_metrics"]

    assert hm["top_priority_name"] == "long_horizon_coupled_scale_stress"
    assert hm["top_priority_triggered"] is True
    assert hm["probe_coverage"] == 1.0
    assert hm["closure_risk_index"] > 0.45

    out_path = ROOT / "tests" / "codex_temp" / "stage58_counterexample_priority_probe_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["top_priority_name"] == hm["top_priority_name"]
