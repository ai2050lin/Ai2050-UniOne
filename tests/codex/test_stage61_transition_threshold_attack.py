from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_transition_threshold_attack import build_transition_threshold_attack_summary


def test_stage61_transition_threshold_attack() -> None:
    summary = build_transition_threshold_attack_summary()
    hm = summary["headline_metrics"]

    assert hm["attacked_closure"] > 0.58
    assert hm["attacked_falsifiability"] > 0.72
    assert hm["attacked_dependency_penalty"] < 0.62
    assert hm["crossed_transition"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage61_transition_threshold_attack_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["crossed_transition"] == hm["crossed_transition"]
