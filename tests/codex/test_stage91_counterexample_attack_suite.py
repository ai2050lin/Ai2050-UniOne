from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage91_counterexample_attack_suite import build_counterexample_attack_suite_summary


def test_stage91_counterexample_attack_suite() -> None:
    summary = build_counterexample_attack_suite_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["attack_suite_coverage"] >= 1.0
    assert hm["multi_plane_breach_rate"] >= 0.50
    assert hm["hardest_attack_name"] in {
        "parameter_perturbation",
        "order_shuffle_attack",
        "scenario_replacement_attack",
        "boundary_overload_attack",
        "cross_plane_coupling_resonance",
        "brain_grounding_shock",
    }
    assert hm["hardest_attack_intensity"] >= 0.58
    assert hm["weakest_plane_name"] in {
        "language_plane",
        "brain_plane",
        "intelligence_plane",
        "falsification_plane",
    }
    assert hm["weakest_plane_attack_floor"] <= 0.72
    assert hm["counterexample_attack_suite_score"] >= 0.70
    assert len(summary["attack_records"]) == 6
    assert status["status_short"] in {
        "counterexample_attack_suite_ready",
        "counterexample_attack_suite_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage91_counterexample_attack_suite_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
