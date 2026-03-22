from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage68_nested_vs_direct_comparison import build_nested_vs_direct_comparison_summary


def test_stage68_nested_vs_direct_comparison() -> None:
    summary = build_nested_vs_direct_comparison_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["closure_gap"] < 0.03
    assert hm["falsifiability_gap"] < 0.05
    assert hm["dependency_gap"] < 0.05
    assert hm["readiness_gap"] < 0.05
    assert hm["direct_consistency_score"] > 0.95
    assert hm["interpretability_gain"] > 0.96
    assert status["status_short"] == "direct_chain_preferred"

    out_path = ROOT / "tests" / "codex_temp" / "stage68_nested_vs_direct_comparison_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
