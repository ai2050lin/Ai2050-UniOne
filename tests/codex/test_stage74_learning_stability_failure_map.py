from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage74_learning_stability_failure_map import build_learning_stability_failure_map_summary


def test_stage74_learning_stability_failure_map() -> None:
    summary = build_learning_stability_failure_map_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["learning_failure_surface_coverage"] > 0.58
    assert hm["average_guarded_update_score"] > 0.60
    assert hm["average_recovery_buffer"] > 0.62
    assert hm["bounded_learning_window_score"] > 0.60
    assert hm["worst_case_failure_name"] in {
        "semantic_patch_insert",
        "route_rebind_insert",
        "context_switch_write",
        "compositional_binding_write",
        "long_horizon_refresh",
    }
    assert hm["worst_case_failure_intensity"] < 0.58
    assert hm["stability_repair_priority"] > 0.43
    assert hm["learning_stability_failure_map_score"] > 0.61
    assert len(summary["scenario_records"]) == 5
    assert status["status_short"] in {
        "learning_stability_failure_map_transition",
        "learning_stability_failure_map_ready",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage74_learning_stability_failure_map_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
