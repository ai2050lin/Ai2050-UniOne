from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_language_task_boundary_trigger import build_language_task_boundary_trigger_summary


def test_stage57_language_task_boundary_trigger() -> None:
    summary = build_language_task_boundary_trigger_summary()
    hm = summary["headline_metrics"]

    assert hm["stressed_long_forgetting"] > 0.20
    assert hm["stressed_base_perplexity_delta"] > 1000.0
    assert hm["stressed_novel_accuracy_after"] < 0.90
    assert hm["task_boundary_readiness"] < 0.60
    assert summary["task_trigger"]["triggered"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage57_language_task_boundary_trigger_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["task_trigger"]["triggered"] is True
