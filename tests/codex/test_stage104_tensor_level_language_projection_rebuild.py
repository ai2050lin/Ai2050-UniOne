from __future__ import annotations

import json
from pathlib import Path

from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary


ROOT = Path(__file__).resolve().parents[2]


def test_stage104_tensor_level_language_projection_rebuild() -> None:
    summary = build_tensor_level_language_projection_rebuild_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["reconstructed_context_gate_coherence"] >= 0.60
    assert hm["reconstructed_bias_transport"] >= 0.60
    assert hm["reconstructed_route_projection"] >= 0.55
    assert hm["cross_dimension_projection_stability"] >= 0.80
    assert hm["cross_dimension_separation"] >= 0.62
    assert hm["raw_language_projection_score"] >= 0.66
    assert len(summary["dimension_records"]) == 3
    assert status["status_short"] in {
        "tensor_level_language_projection_rebuild_ready",
        "tensor_level_language_projection_rebuild_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage104_tensor_level_language_projection_rebuild_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
