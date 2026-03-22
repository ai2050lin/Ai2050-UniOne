from __future__ import annotations

import json
from pathlib import Path

from stage112_world_task_boundary_bridge import build_world_task_boundary_bridge_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage112_world_task_boundary_bridge() -> None:
    summary = build_world_task_boundary_bridge_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["bridge_family_coverage"] >= 0.95
    assert hm["hardest_family_name"] in {
        "style_dialogue_family",
        "logic_negation_family",
        "syntax_rewrite_family",
        "bilingual_alias_family",
        "macro_abstract_family",
    }
    assert hm["hardest_family_pressure"] >= 0.45
    assert hm["weakest_native_under_task_name"] in {
        "anchor_recurrence_family",
        "minimal_transport_efficiency_quantity",
    }
    assert hm["weakest_native_under_task_score"] >= 0.52
    assert hm["task_boundary_closure_gain"] >= 0.45
    assert hm["world_task_boundary_bridge_score"] >= 0.55
    assert len(summary["task_family_records"]) == 5
    assert status["status_short"] in {
        "world_task_boundary_bridge_ready",
        "world_task_boundary_bridge_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage112_world_task_boundary_bridge_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
