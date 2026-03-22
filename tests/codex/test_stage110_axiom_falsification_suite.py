from __future__ import annotations

import json
from pathlib import Path

from stage110_axiom_falsification_suite import build_axiom_falsification_suite_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage110_axiom_falsification_suite() -> None:
    summary = build_axiom_falsification_suite_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["attack_coverage"] >= 0.95
    assert hm["strongest_attack_name"] in {
        "style_context_flip_attack",
        "logic_negation_attack",
        "syntax_voice_shift_attack",
        "bilingual_alias_attack",
        "macro_abstract_bridge_attack",
        "anchor_overlap_attack",
    }
    assert hm["strongest_attack_intensity"] >= 0.45
    assert hm["weakest_axiom_after_attack_name"] in {
        "projection_covariance_axiom",
        "distributed_routing_axiom",
        "bounded_repair_axiom",
        "anchor_separability_axiom",
        "falsifiable_boundary_axiom",
    }
    assert hm["weakest_axiom_after_attack_score"] >= 0.15
    assert hm["task_bridge_retest_pressure"] >= 0.45
    assert hm["falsification_survival_score"] >= 0.40
    assert hm["axiom_falsification_suite_score"] >= 0.52
    assert len(summary["attack_records"]) == 6
    assert len(summary["axiom_attack_records"]) == 5
    assert status["status_short"] in {
        "axiom_falsification_suite_ready",
        "axiom_falsification_suite_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage110_axiom_falsification_suite_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
