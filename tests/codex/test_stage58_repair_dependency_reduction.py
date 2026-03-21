from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_repair_dependency_reduction import build_repair_dependency_reduction_summary


def test_stage58_repair_dependency_reduction() -> None:
    summary = build_repair_dependency_reduction_summary()
    hm = summary["headline_metrics"]
    strategies = summary["strategy_results"]

    assert hm["best_strategy_name"] == "joint_nativeization"
    assert hm["best_safe_task_count"] == 2
    assert hm["reduced_dependency_penalty"] < 0.70
    assert hm["dependency_reduction_gain"] > 0.25
    assert strategies["fiber_nativeization"]["safe_task_count"] == 1
    assert strategies["context_nativeization"]["safe_task_count"] == 1
    assert strategies["joint_nativeization"]["safe_task_count"] == 2

    out_path = ROOT / "tests" / "codex_temp" / "stage58_repair_dependency_reduction_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_strategy_name"] == hm["best_strategy_name"]
