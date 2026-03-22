from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_uniqueness_hardening import build_uniqueness_hardening_summary


def test_stage62_uniqueness_hardening() -> None:
    summary = build_uniqueness_hardening_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["hardened_uniqueness_score"] > 0.82
    assert hm["residual_uniqueness_gap"] < 0.18
    assert hm["cross_task_lock_score"] > 0.81
    assert status["status_short"] == "uniqueness_strengthened_not_closed"

    out_path = ROOT / "tests" / "codex_temp" / "stage62_uniqueness_hardening_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
