from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_dependency_below_floor_probe import build_dependency_below_floor_probe_summary


def test_stage60_dependency_below_floor_probe() -> None:
    summary = build_dependency_below_floor_probe_summary()
    hm = summary["headline_metrics"]

    assert hm["safe_point_count"] >= 2
    assert hm["new_dependency_floor_explicit_share"] == 0.39
    assert hm["new_dependency_floor_penalty"] < 0.62
    assert hm["new_floor_coupled_margin"] > 0.618

    out_path = ROOT / "tests" / "codex_temp" / "stage60_dependency_below_floor_probe_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["new_dependency_floor_explicit_share"] == hm["new_dependency_floor_explicit_share"]
