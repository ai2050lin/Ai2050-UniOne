from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage67_context_fiber_primitive_repair import build_context_fiber_primitive_repair_summary


def test_stage67_context_fiber_primitive_repair() -> None:
    summary = build_context_fiber_primitive_repair_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["upgraded_context_score"] > 0.72
    assert hm["upgraded_fiber_score"] > 0.73
    assert hm["repaired_primitive_closure"] > 0.72
    assert hm["repaired_reconstruction_error"] < 0.24
    assert status["status_short"] == "context_fiber_repair_active"

    out_path = ROOT / "tests" / "codex_temp" / "stage67_context_fiber_primitive_repair_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
