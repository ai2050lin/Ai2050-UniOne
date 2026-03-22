from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_final_boundary_clearance import build_final_boundary_clearance_summary


def test_stage67_final_boundary_clearance() -> None:
    summary = build_final_boundary_clearance_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["final_boundary_clearance"] > 0.75
    assert hm["boundary_lock_confidence"] > 0.78
    assert hm["remaining_boundary_count"] == 1
    assert status["status_short"] == "final_boundary_not_cleared"

    out_path = ROOT / "tests" / "codex_temp" / "stage67_final_boundary_clearance_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
