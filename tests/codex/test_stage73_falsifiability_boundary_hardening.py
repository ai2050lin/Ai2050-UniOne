from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage73_falsifiability_boundary_hardening import build_falsifiability_boundary_hardening_summary


def test_stage73_falsifiability_boundary_hardening() -> None:
    summary = build_falsifiability_boundary_hardening_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["executable_boundary_coverage"] > 0.94
    assert hm["task_counterexample_activation"] > 0.69
    assert hm["shared_state_rejection_power"] > 0.80
    assert hm["boundary_counterexample_discrimination"] > 0.88
    assert hm["falsifiability_boundary_hardening_score"] > 0.82
    assert hm["weakest_failure_mode_name"] in {
        "context_covariance",
        "fiber_emergence",
        "learning_stability",
        "shared_state",
    }
    assert hm["weakest_failure_mode_score"] > 0.68
    assert len(summary["failure_mode_map"]) == 4
    assert summary["boundary_bridge"]["task_triggered"] is True
    assert status["status_short"] in {
        "falsifiability_boundary_transition",
        "falsifiability_boundary_hardened",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage73_falsifiability_boundary_hardening_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
