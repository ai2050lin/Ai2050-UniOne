from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage94_cross_plane_failure_coupling_map import build_cross_plane_failure_coupling_map_summary


def test_stage94_cross_plane_failure_coupling_map() -> None:
    summary = build_cross_plane_failure_coupling_map_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["propagation_coverage"] >= 0.95
    assert hm["hardest_coupling_path"] in {
        "language_plane->brain_plane",
        "language_plane->intelligence_plane",
        "language_plane->falsification_plane",
        "brain_plane->language_plane",
        "brain_plane->intelligence_plane",
        "brain_plane->falsification_plane",
        "intelligence_plane->language_plane",
        "intelligence_plane->brain_plane",
        "intelligence_plane->falsification_plane",
        "falsification_plane->language_plane",
        "falsification_plane->brain_plane",
        "falsification_plane->intelligence_plane",
    }
    assert hm["hardest_path_intensity"] >= 0.50
    assert hm["weakest_receiver_plane"] in {
        "language_plane",
        "brain_plane",
        "intelligence_plane",
        "falsification_plane",
    }
    assert hm["weakest_receiver_floor"] <= 0.45
    assert hm["cross_plane_load_mean"] >= 0.30
    assert hm["theorem_spillover_pressure"] >= 0.55
    assert hm["cross_plane_failure_coupling_score"] >= 0.60
    assert len(summary["edge_weights"]) == 12
    assert len(summary["coupling_paths"]) == 12
    assert len(summary["propagation_records"]) >= 100
    assert status["status_short"] in {
        "cross_plane_failure_coupling_map_ready",
        "cross_plane_failure_coupling_map_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage94_cross_plane_failure_coupling_map_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
