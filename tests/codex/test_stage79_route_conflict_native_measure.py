from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary


def test_stage79_route_conflict_native_measure() -> None:
    summary = build_route_conflict_native_measure_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["attention_like_selection"] > 0.84
    assert hm["gradient_like_correction"] > 0.80
    assert hm["route_conflict_mass"] < 0.40
    assert hm["conflict_resolution_readiness"] > 0.82
    assert hm["inference_route_coherence"] > 0.83
    assert hm["training_route_alignment"] > 0.79
    assert hm["route_computation_closure_score"] > 0.81
    assert len(summary["scenario_records"]) == 3
    assert status["status_short"] in {
        "route_conflict_native_measure_ready",
        "route_conflict_native_measure_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage79_route_conflict_native_measure_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
