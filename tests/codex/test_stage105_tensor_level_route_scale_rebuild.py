from __future__ import annotations

import json
from pathlib import Path

from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage105_tensor_level_route_scale_rebuild() -> None:
    base = build_tensor_level_language_projection_rebuild_summary()
    assert base["headline_metrics"]["raw_language_projection_score"] >= 0.66

    summary = build_tensor_level_route_scale_rebuild_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["local_anchor_support"] >= 0.60
    assert hm["mesoscopic_bundle_support"] >= 0.65
    assert hm["distributed_network_support"] >= 0.68
    assert hm["route_structure_coupling_strength"] >= 0.68
    assert hm["degradation_tolerance"] >= 0.65
    assert hm["dominant_scale_name"] == "distributed_network"
    assert hm["reconstructed_route_scale_score"] >= 0.70
    assert status["status_short"] in {
        "tensor_level_route_scale_rebuild_ready",
        "tensor_level_route_scale_rebuild_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage105_tensor_level_route_scale_rebuild_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
