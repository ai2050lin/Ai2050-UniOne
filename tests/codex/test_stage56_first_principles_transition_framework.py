from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_first_principles_transition_framework import build_first_principles_transition_framework_summary


def test_stage56_first_principles_transition_framework() -> None:
    summary = build_first_principles_transition_framework_summary()
    hm = summary["headline_metrics"]

    assert 0.0 <= hm["primitive_transition_readiness"] <= 1.0
    assert 0.0 <= hm["local_law_closure"] <= 1.0
    assert 0.0 <= hm["falsifiability_upgrade"] <= 1.0
    assert 0.0 <= hm["first_principles_transition_score"] <= 1.0

    out_path = ROOT / "tests" / "codex_temp" / "stage56_first_principles_transition_framework_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["first_principles_transition_score"] == hm["first_principles_transition_score"]
