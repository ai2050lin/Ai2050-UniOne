from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_transition_stability_retest import build_transition_stability_retest_summary


def test_stage62_transition_stability_retest() -> None:
    summary = build_transition_stability_retest_summary()
    hm = summary["headline_metrics"]
    cases = summary["case_results"]

    assert hm["stable_case_count"] == 1
    assert hm["stability_pass_rate"] == 0.25
    assert hm["transition_still_holds"] is False
    assert hm["transition_stability_score"] > 0.45
    assert cases["long_replay_drag"]["passes_transition"] is False

    out_path = ROOT / "tests" / "codex_temp" / "stage62_transition_stability_retest_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["stable_case_count"] == hm["stable_case_count"]
