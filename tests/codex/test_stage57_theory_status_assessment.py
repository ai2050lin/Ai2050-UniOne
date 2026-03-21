from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_theory_status_assessment import build_theory_status_assessment_summary


def test_stage57_theory_status_assessment() -> None:
    summary = build_theory_status_assessment_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["phenomenology_strength"] > 0.65
    assert hm["first_principles_support"] > 0.65
    assert 0.50 <= hm["first_principles_closure"] < 0.75
    assert hm["falsifiability_strength"] > 0.65
    assert hm["repair_dependency_penalty"] >= 0.90
    assert status["status_short"] == "phenomenological_model"
    assert "fiber_reuse" in status["necessary_components"]
    assert "context_grounding" in status["necessary_components"]

    out_path = ROOT / "tests" / "codex_temp" / "stage57_theory_status_assessment_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
