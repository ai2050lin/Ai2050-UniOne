from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_kernel_feedback_reintegration import build_kernel_feedback_reintegration_summary


def test_stage57_kernel_feedback_reintegration() -> None:
    summary = build_kernel_feedback_reintegration_summary()
    hm = summary["headline_metrics"]
    reintegrated = summary["reintegrated_candidates"]

    assert set(reintegrated.keys()) == {"sqrt", "log"}
    assert hm["best_reintegrated_candidate_name"] in reintegrated
    assert hm["best_reintegrated_overall_readiness"] > 0.75
    assert hm["best_reintegrated_structure_anchor"] > 0.70
    assert hm["best_reintegrated_local_compatibility"] > 0.75
    assert hm["best_feedback_gain"] > 0.60
    assert hm["reintegrated_margin"] >= 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage57_kernel_feedback_reintegration_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_reintegrated_candidate_name"] == hm["best_reintegrated_candidate_name"]
