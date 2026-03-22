from __future__ import annotations

import json
from pathlib import Path

from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage106_forward_backward_trace_rebuild() -> None:
    assert build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]["raw_language_projection_score"] >= 0.66
    assert build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]["reconstructed_route_scale_score"] >= 0.70

    summary = build_forward_backward_trace_rebuild_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["loss_drop_ratio"] >= 0.45
    assert hm["frontier_drop_ratio"] >= 0.45
    assert hm["boundary_drop_ratio"] >= 0.35
    assert hm["loss_monotonicity"] >= 1.0
    assert hm["frontier_boundary_coupling"] >= 0.65
    assert hm["raw_forward_selectivity"] >= 0.60
    assert hm["raw_backward_fidelity"] >= 0.65
    assert hm["raw_novelty_binding_capacity"] >= 0.60
    assert hm["raw_forward_backward_rebuild_score"] >= 0.68
    assert len(summary["step_records"]) == 6
    assert status["status_short"] in {
        "forward_backward_trace_rebuild_ready",
        "forward_backward_trace_rebuild_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage106_forward_backward_trace_rebuild_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
