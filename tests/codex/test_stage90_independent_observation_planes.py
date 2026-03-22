from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage90_independent_observation_planes import build_independent_observation_planes_summary


def test_stage90_independent_observation_planes() -> None:
    summary = build_independent_observation_planes_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["plane_signal_mean"] >= 0.78
    assert hm["surface_anchor_independence"] >= 0.95
    assert hm["source_plane_separation"] >= 0.95
    assert hm["exclusive_anchor_ratio"] >= 0.95
    assert hm["variable_coupling_overlap"] <= 0.55
    assert hm["backfeed_risk_after_split"] <= 0.60
    assert hm["independent_observation_planes_score"] >= 0.74
    assert len(summary["plane_records"]) == 4
    assert len(summary["overlap_matrix"]) == 6
    assert status["status_short"] in {
        "independent_observation_planes_ready",
        "independent_observation_planes_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage90_independent_observation_planes_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
