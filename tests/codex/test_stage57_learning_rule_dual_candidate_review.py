from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_learning_rule_dual_candidate_review import build_learning_rule_dual_candidate_review_summary


def test_stage57_learning_rule_dual_candidate_review() -> None:
    summary = build_learning_rule_dual_candidate_review_summary()
    hm = summary["headline_metrics"]
    review = summary["candidate_review"]

    assert set(review.keys()) == {"sqrt", "log"}
    assert hm["best_candidate_name"] in review
    assert hm["best_candidate_overall_readiness"] >= 0.0
    assert hm["best_candidate_domination_penalty"] < 0.35
    assert hm["best_candidate_structure_anchor_score"] > 0.70
    assert hm["best_candidate_local_law_compatibility"] > 0.65
    assert hm["readiness_margin"] >= 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage57_learning_rule_dual_candidate_review_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_candidate_name"] == hm["best_candidate_name"]
