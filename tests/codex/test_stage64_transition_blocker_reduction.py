from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage64_transition_blocker_reduction import build_transition_blocker_reduction_summary


def test_stage64_transition_blocker_reduction() -> None:
    summary = build_transition_blocker_reduction_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["blocker_reduction_gain"] > 0.67
    assert hm["reduced_completion_blocker"] < 0.51
    assert hm["updated_completion_readiness"] > 0.62
    assert hm["updated_completion_gap"] < 0.38
    assert status["status_short"] == "blocker_reduced_not_resolved"

    out_path = ROOT / "tests" / "codex_temp" / "stage64_transition_blocker_reduction_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
