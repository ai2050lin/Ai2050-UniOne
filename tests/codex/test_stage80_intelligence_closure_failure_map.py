from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary


def test_stage80_intelligence_closure_failure_map() -> None:
    summary = build_intelligence_closure_failure_map_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["closure_failure_surface_coverage"] > 0.34
    assert hm["average_recovery_coherence"] > 0.76
    assert hm["abstraction_bridge_strength"] > 0.72
    assert hm["worst_case_name"] in {
        "multi_hop_composition",
        "context_transfer",
        "novelty_generalization",
        "conflict_recovery",
        "abstraction_compression",
    }
    assert hm["worst_case_failure_intensity"] < 0.46
    assert hm["intelligence_closure_failure_map_score"] > 0.72
    assert len(summary["scenario_records"]) == 5
    assert status["status_short"] in {
        "intelligence_closure_failure_map_ready",
        "intelligence_closure_failure_map_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage80_intelligence_closure_failure_map_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
