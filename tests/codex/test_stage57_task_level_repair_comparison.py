from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary


def test_stage57_task_level_repair_comparison() -> None:
    summary = build_task_level_repair_comparison_summary()
    hm = summary["headline_metrics"]
    candidates = summary["candidate_repairs"]

    assert set(candidates.keys()) == {"sqrt", "log"}
    assert hm["best_repair_candidate_name"] == "sqrt"
    assert hm["best_repair_task_count"] == 2
    assert hm["best_repair_readiness"] > 0.75
    assert hm["best_language_trigger_after_repair"] is False
    assert hm["best_brain_trigger_after_repair"] is False
    assert candidates["log"]["language_triggered_after_repair"] is True
    assert candidates["log"]["brain_triggered_after_repair"] is True
    assert hm["repair_readiness_margin"] > 0.015

    out_path = ROOT / "tests" / "codex_temp" / "stage57_task_level_repair_comparison_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_repair_candidate_name"] == hm["best_repair_candidate_name"]
