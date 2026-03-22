from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage70_direct_stability_counterexample_probe import build_direct_stability_counterexample_probe_summary


def test_stage70_direct_stability_counterexample_probe() -> None:
    summary = build_direct_stability_counterexample_probe_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["adversarial_stability_support"] > 0.72
    assert hm["counterexample_pressure"] < 0.24
    assert hm["survives_counterexample"] is True
    assert status["status_short"] == "counterexample_survived"

    out_path = ROOT / "tests" / "codex_temp" / "stage70_direct_stability_counterexample_probe_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
