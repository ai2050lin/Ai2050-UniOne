from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage72_language_projection_covariance import build_language_projection_covariance_summary


def test_stage72_language_projection_covariance() -> None:
    summary = build_language_projection_covariance_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["context_covariance_stability"] > 0.79
    assert hm["bias_gate_transport"] > 0.85
    assert hm["route_conditioned_projection"] > 0.90
    assert hm["context_shift_resilience"] > 0.94
    assert hm["projection_counterexample_resistance"] > 0.79
    assert hm["projection_gap"] < 0.10
    assert hm["language_projection_repair_score"] > 0.88
    assert len(summary["scenario_records"]) == 4
    assert status["status_short"] in {
        "language_projection_covariance_transition",
        "language_projection_covariance_repaired",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage72_language_projection_covariance_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
