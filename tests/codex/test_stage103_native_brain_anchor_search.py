from __future__ import annotations

import json
from pathlib import Path

from stage103_native_brain_anchor_search import build_native_brain_anchor_search_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage103_native_brain_anchor_search() -> None:
    summary = build_native_brain_anchor_search_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["generic_seed_recurrence_strength"] >= 0.70
    assert hm["dimension_specific_anchor_strength"] >= 0.55
    assert hm["layer_anchor_stability"] >= 0.50
    assert hm["anchor_ambiguity_penalty"] <= 0.75
    assert hm["closure_bridge_support"] >= 0.65
    assert hm["weakest_anchor_mode_name"] in {
        "generic_recurrence_gap",
        "specific_anchor_gap",
        "layer_stability_gap",
        "anchor_ambiguity_gap",
        "closure_bridge_gap",
    }
    assert hm["native_brain_anchor_search_score"] >= 0.58
    assert len(summary["shared_anchor_candidates"]) >= 5
    assert len(summary["specific_anchor_candidates"]) >= 5
    assert status["status_short"] in {
        "native_brain_anchor_search_ready",
        "native_brain_anchor_search_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage103_native_brain_anchor_search_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
