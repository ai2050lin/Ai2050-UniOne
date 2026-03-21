from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary


def test_stage57_fiber_reuse_reinforcement() -> None:
    summary = build_fiber_reuse_reinforcement_summary()
    hm = summary["headline_metrics"]

    assert hm["fiber_reuse"] > 0.45
    assert hm["cross_region_share_stability"] > 0.70
    assert hm["route_fiber_coupling_balance"] > 0.70
    assert hm["pressure_under_reuse"] > 0.55
    assert hm["reinforcement_readiness"] > 0.55

    out_path = ROOT / "tests" / "codex_temp" / "stage57_fiber_reuse_reinforcement_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["fiber_reuse"] == hm["fiber_reuse"]
