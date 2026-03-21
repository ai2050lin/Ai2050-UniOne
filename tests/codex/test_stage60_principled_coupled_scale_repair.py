from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_principled_coupled_scale_repair import build_principled_coupled_scale_repair_summary


def test_stage60_principled_coupled_scale_repair() -> None:
    summary = build_principled_coupled_scale_repair_summary()
    hm = summary["headline_metrics"]

    assert hm["best_principled_bundle_name"] == "principled_coupled_bundle"
    assert hm["best_principled_success"] is True
    assert hm["best_principled_dependency_penalty"] < 0.68
    assert hm["best_principled_combined_margin"] > 0.621

    out_path = ROOT / "tests" / "codex_temp" / "stage60_principled_coupled_scale_repair_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_principled_bundle_name"] == hm["best_principled_bundle_name"]
