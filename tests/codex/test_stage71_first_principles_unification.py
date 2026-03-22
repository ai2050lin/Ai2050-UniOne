from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage71_first_principles_unification import build_first_principles_unification_summary


def test_stage71_first_principles_unification() -> None:
    summary = build_first_principles_unification_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["unified_state_readiness"] > 0.74
    assert hm["language_projection_coherence"] > 0.74
    assert hm["brain_encoding_groundedness"] > 0.75
    assert hm["intelligence_functional_closure"] > 0.75
    assert hm["local_generation_closure"] > 0.68
    assert hm["falsifiability_boundary_strength"] > 0.76
    assert hm["first_principles_unification_score"] > 0.74
    assert hm["weakest_axis_name"] in {
        "language_projection",
        "brain_grounding",
        "intelligence_closure",
        "local_generation",
        "falsifiability_boundary",
    }
    assert len(summary["unified_state_variables"]) == 10
    assert len(summary["falsification_boundaries"]) == 4
    assert status["status_short"] in {
        "first_principles_unification_transition",
        "first_principles_unification_frontier",
    }
    axis_scores = {
        "language_projection": hm["language_projection_coherence"],
        "brain_grounding": hm["brain_encoding_groundedness"],
        "intelligence_closure": hm["intelligence_functional_closure"],
        "local_generation": hm["local_generation_closure"],
        "falsifiability_boundary": hm["falsifiability_boundary_strength"],
    }
    expected_weakest = min(axis_scores, key=axis_scores.get)
    assert hm["weakest_axis_name"] == expected_weakest
    if status["status_short"] == "first_principles_unification_frontier":
        assert "前沿区" in status["status_label"]
    else:
        assert "过渡区" in status["status_label"]

    out_path = ROOT / "tests" / "codex_temp" / "stage71_first_principles_unification_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
