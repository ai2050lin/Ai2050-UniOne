from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_minimal_repair_set_ablation import build_minimal_repair_set_ablation_summary


def test_stage57_minimal_repair_set_ablation() -> None:
    summary = build_minimal_repair_set_ablation_summary()
    hm = summary["headline_metrics"]
    necessary = set(summary["necessary_components"])
    cases = summary["ablation_cases"]

    assert hm["full_repair_safe_task_count"] == 2
    assert hm["drop_fiber_safe_task_count"] < hm["full_repair_safe_task_count"]
    assert hm["drop_context_safe_task_count"] < hm["full_repair_safe_task_count"]
    assert hm["drop_both_safe_task_count"] == 0
    assert hm["minimum_joint_repair_required"] is True
    assert necessary == {"fiber_reuse", "context_grounding"}
    assert cases["drop_fiber"]["language_triggered"] is True
    assert cases["drop_context"]["brain_triggered"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage57_minimal_repair_set_ablation_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["minimum_joint_repair_required"] is True
