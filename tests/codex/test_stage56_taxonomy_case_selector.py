from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stage56_taxonomy_case_selector import select_representative_cases  # noqa: E402


def make_row(model_id: str, category: str, case_role: str, best_mixed: float, union: float, synergy: float) -> dict:
    return {
        "group_label": f"{model_id}_{category}",
        "model_id": model_id,
        "category": category,
        "prototype_term": f"{category}_proto",
        "instance_term": f"{category}_inst",
        "case_role": case_role,
        "best_mixed": {"metrics": {"joint_adv_mean": best_mixed}},
        "stage6_reference": {
            "union_joint_adv": union,
            "union_synergy_joint": synergy,
        },
    }


def test_select_representative_cases_picks_one_per_model_category() -> None:
    rows = [
        make_row("m1", "fruit", "weak_drag_or_conflict", 0.3, 0.2, 0.1),
        make_row("m1", "fruit", "weak_bridge_positive", 0.2, 0.1, 0.0),
        make_row("m2", "fruit", "weak_bridge_positive", 0.1, 0.3, 0.2),
    ]
    selected = select_representative_cases(rows)
    assert len(selected) == 2
    first = [row for row in selected if row["model_id"] == "m1"][0]
    assert first["case_role"] == "weak_bridge_positive"


def test_select_representative_cases_breaks_ties_by_best_mixed_then_stage6() -> None:
    rows = [
        make_row("m1", "tech", "weak_drag_or_conflict", 0.2, 0.1, 0.1),
        make_row("m1", "tech", "weak_drag_or_conflict", 0.3, 0.05, 0.0),
        make_row("m1", "tech", "weak_drag_or_conflict", 0.3, 0.2, -0.1),
    ]
    selected = select_representative_cases(rows)
    assert len(selected) == 1
    assert selected[0]["stage6_reference"]["union_joint_adv"] == 0.2
