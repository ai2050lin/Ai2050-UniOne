from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_rule_bounded_law_comparison import build_learning_rule_bounded_law_comparison_summary


def test_stage56_learning_rule_bounded_law_comparison() -> None:
    summary = build_learning_rule_bounded_law_comparison_summary()
    hm = summary["headline_metrics"]
    candidates = summary["candidate_laws"]

    assert set(candidates.keys()) == {"log", "sqrt", "rational"}
    assert hm["best_law_name"] in candidates
    assert 0.0 <= hm["best_law_readiness"] <= 1.0
    assert hm["best_law_domination_penalty"] < 0.5
    assert hm["law_readiness_gap"] >= 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage56_learning_rule_bounded_law_comparison_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_law_name"] == hm["best_law_name"]
