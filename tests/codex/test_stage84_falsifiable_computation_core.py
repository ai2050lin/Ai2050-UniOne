from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage84_falsifiable_computation_core import build_falsifiable_computation_core_summary


def test_stage84_falsifiable_computation_core() -> None:
    summary = build_falsifiable_computation_core_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["executable_theorem_discrimination"] > 0.84
    assert hm["counterexample_contact_strength"] > 0.79
    assert hm["route_projection_failure_traceability"] > 0.88
    assert hm["shared_state_refutation_power"] > 0.80
    assert hm["hardest_counterexample_name"] in {
        "context_covariance_break",
        "route_conflict_overflow",
        "novelty_bounded_break",
        "shared_state_decoupling",
    }
    assert hm["hardest_counterexample_intensity"] < 0.45
    assert hm["falsifiable_computation_core_score"] > 0.78
    assert len(summary["scenario_records"]) == 4
    assert status["status_short"] in {
        "falsifiable_computation_core_ready",
        "falsifiable_computation_core_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage84_falsifiable_computation_core_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
