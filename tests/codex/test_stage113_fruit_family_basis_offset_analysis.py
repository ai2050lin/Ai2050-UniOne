from __future__ import annotations

import json
from pathlib import Path

from stage113_fruit_family_basis_offset_analysis import (
    build_fruit_family_basis_offset_analysis_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage113_fruit_family_basis_offset_analysis() -> None:
    summary = build_fruit_family_basis_offset_analysis_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["qwen_family_lock_strength"] >= 0.75
    assert hm["qwen_offset_expressivity"] >= 0.65
    assert hm["deepseek_basis_support"] >= 0.55
    assert hm["deepseek_offset_split_strength"] >= 0.55
    assert hm["hierarchical_ordering_strength"] >= 0.45
    assert hm["attribute_transfer_support"] >= 0.65
    assert hm["conflict_gate_necessity"] >= 0.52
    assert hm["strongest_mechanism_name"] in {
        "fruit_family_basis",
        "apple_instance_offset",
        "attribute_fiber_bundle",
        "conflict_repair_gate",
        "basis_before_offset_order",
    }
    assert hm["weakest_mechanism_name"] in {
        "fruit_family_basis",
        "apple_instance_offset",
        "attribute_fiber_bundle",
        "conflict_repair_gate",
        "basis_before_offset_order",
    }
    assert hm["theory_foundation_score"] >= 0.62
    assert len(summary["fruit_basis_records"]) == 5
    assert len(summary["why_records"]) == 3
    assert status["status_short"] in {
        "fruit_family_basis_offset_ready",
        "fruit_family_basis_offset_transition",
    }

    out_path = (
        ROOT
        / "tests"
        / "codex_temp"
        / "stage113_fruit_family_basis_offset_analysis_20260323"
        / "summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["theory_foundation_score"] == hm["theory_foundation_score"]
