from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_coupled_scale_repair import build_coupled_scale_repair_summary


def test_stage59_coupled_scale_repair() -> None:
    summary = build_coupled_scale_repair_summary()
    hm = summary["headline_metrics"]
    bundles = summary["bundle_results"]

    assert hm["best_bundle_name"] == "coupled_scale_bundle"
    assert hm["best_repair_success"] is True
    assert hm["best_repaired_combined_margin"] > 0.61
    assert bundles["pressure_only"]["repair_success"] is False
    assert bundles["coupled_scale_bundle"]["repair_success"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage59_coupled_scale_repair_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["best_bundle_name"] == hm["best_bundle_name"]
