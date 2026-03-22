from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary


def test_stage92_brain_grounding_counterexample_pack() -> None:
    summary = build_brain_grounding_counterexample_pack_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["brain_counterexample_coverage"] >= 1.0
    assert hm["multi_axis_grounding_break_rate"] >= 0.66
    assert hm["hardest_counterexample_name"] in {
        "neuron_anchor_collapse",
        "bundle_desynchronization",
        "field_proxy_blind_spot",
        "plasticity_pressure_inversion",
        "cross_modal_grounding_gap",
        "distributed_field_fragmentation",
    }
    assert hm["hardest_counterexample_intensity"] >= 0.60
    assert hm["weakest_component_name"] in {
        "neuron_anchor",
        "bundle_sync",
        "distributed_field",
        "field_observability",
        "repair_grounding",
    }
    assert hm["weakest_component_floor"] <= 0.52
    assert hm["brain_grounding_counterexample_score"] >= 0.74
    assert len(summary["scenario_records"]) == 6
    assert status["status_short"] in {
        "brain_grounding_counterexample_pack_ready",
        "brain_grounding_counterexample_pack_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage92_brain_grounding_counterexample_pack_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
