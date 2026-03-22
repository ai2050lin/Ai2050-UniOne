from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary


def test_stage76_sqrt_repair_generalization() -> None:
    summary = build_sqrt_repair_generalization_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["best_law_name"] == "sqrt"
    assert hm["generalized_repair_coverage"] > 0.74
    assert hm["repaired_average_failure_intensity"] < 0.36
    assert hm["repaired_average_guarded_update"] > 0.80
    assert hm["repaired_bounded_learning_window"] > 0.75
    assert hm["route_rebind_support"] > 0.73
    assert hm["context_switch_support"] > 0.73
    assert hm["repaired_worst_case_name"] == "compositional_binding_write"
    assert hm["repaired_worst_case_failure_intensity"] < 0.42
    assert hm["repair_generalization_score"] > 0.76
    assert len(summary["repaired_records"]) == 5
    assert status["status_short"] in {
        "sqrt_repair_generalized",
        "sqrt_repair_generalization_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage76_sqrt_repair_generalization_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
