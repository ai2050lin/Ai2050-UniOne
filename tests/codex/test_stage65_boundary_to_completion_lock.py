from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage65_boundary_to_completion_lock import build_boundary_to_completion_lock_summary


def test_stage65_boundary_to_completion_lock() -> None:
    summary = build_boundary_to_completion_lock_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["completion_lock_score"] > 0.72
    assert hm["completion_lock_confidence"] > 0.70
    assert hm["remaining_locked_boundary_count"] == 1
    assert hm["remaining_final_step_count"] == 1
    assert status["status_short"] == "boundary_locked_completion_pending"

    out_path = ROOT / "tests" / "codex_temp" / "stage65_boundary_to_completion_lock_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
