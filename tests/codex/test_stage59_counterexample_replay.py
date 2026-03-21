from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_counterexample_replay import build_counterexample_replay_summary


def test_stage59_counterexample_replay() -> None:
    summary = build_counterexample_replay_summary()
    hm = summary["headline_metrics"]

    assert hm["scenario_name"] == "long_horizon_coupled_scale_stress"
    assert hm["replay_reproducibility"] == 1.0
    assert hm["replay_before_triggered"] is True
    assert hm["replay_after_triggered"] is False
    assert hm["replay_margin_gain"] > 0.05

    out_path = ROOT / "tests" / "codex_temp" / "stage59_counterexample_replay_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["scenario_name"] == hm["scenario_name"]
