from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage63_first_principles_completion_possibility import build_first_principles_completion_possibility_summary


def test_stage63_first_principles_completion_possibility() -> None:
    summary = build_first_principles_completion_possibility_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["theoretical_possibility_score"] > 0.76
    assert hm["completion_blocker_penalty"] > 0.60
    assert 0.50 < hm["current_completion_readiness"] < 0.60
    assert status["status_short"] == "high_possibility_not_completed"

    out_path = ROOT / "tests" / "codex_temp" / "stage63_first_principles_completion_possibility_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
