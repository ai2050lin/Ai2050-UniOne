from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_learning_term_boundedization import build_learning_term_boundedization_summary


def test_stage56_learning_term_boundedization() -> None:
    summary = build_learning_term_boundedization_summary()
    hm = summary["headline_metrics"]

    assert hm["raw_ratio_v100_v90"] > 100.0
    assert hm["bounded_ratio_v101_v100"] < hm["raw_ratio_v101_v100"]
    assert hm["bounded_domination_penalty"] < hm["raw_domination_penalty"]
    assert 0.0 <= hm["bounded_readiness"] <= 1.0

    out_path = ROOT / "tests" / "codex_temp" / "stage56_learning_term_boundedization_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["bounded_readiness"] == hm["bounded_readiness"]
