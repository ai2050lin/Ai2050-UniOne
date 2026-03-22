from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage77_brain_grounded_route_scaling import build_brain_grounded_route_scaling_summary


def test_stage77_brain_grounded_route_scaling() -> None:
    summary = build_brain_grounded_route_scaling_summary()
    hm = summary["headline_metrics"]
    profile = summary["route_scale_profile"]
    status = summary["status"]

    assert hm["neuron_level_support"] > 0.70
    assert hm["mesoscopic_bundle_support"] > 0.74
    assert hm["distributed_network_support"] > 0.78
    assert hm["route_scale_grounding_score"] > 0.78
    assert hm["brain_constrained_repair_score"] > 0.76
    assert hm["distributed_network_support"] > hm["neuron_level_support"]
    assert profile["dominant_scale_name"] == "distributed_network"
    assert profile["single_neuron_is_sufficient"] is False
    assert status["status_short"] in {
        "brain_grounded_route_scaling_ready",
        "brain_grounded_route_scaling_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage77_brain_grounded_route_scaling_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["route_scale_profile"]["dominant_scale_name"] == profile["dominant_scale_name"]
