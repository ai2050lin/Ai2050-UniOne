from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage86_optimal_law_robustness_scan import build_optimal_law_robustness_scan_summary


def test_stage86_optimal_law_robustness_scan() -> None:
    summary = build_optimal_law_robustness_scan_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["baseline_best_law_name"] in {"sqrt", "log", "rational"}
    assert hm["order_invariance"] is True
    assert hm["best_law_flip_rate"] <= 0.25
    assert hm["best_law_mean_margin"] >= 0.003
    assert hm["scenario_replacement_stability"] >= 0.50
    assert hm["optimal_law_robustness_score"] >= 0.70
    assert len(summary["scan_records"]) == 8
    assert status["status_short"] in {
        "optimal_law_robustness_ready",
        "optimal_law_robustness_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage86_optimal_law_robustness_scan_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
