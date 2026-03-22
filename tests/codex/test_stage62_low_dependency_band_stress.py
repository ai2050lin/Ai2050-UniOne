from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage62_low_dependency_band_stress import build_low_dependency_band_stress_summary


def test_stage62_low_dependency_band_stress() -> None:
    summary = build_low_dependency_band_stress_summary()
    hm = summary["headline_metrics"]

    assert hm["stressed_safe_point_count"] == 2
    assert hm["stressed_band_upper"] == 0.41
    assert hm["stressed_band_lower"] == 0.39
    assert hm["stressed_band_width"] > 0.019
    assert hm["band_resilience_score"] > 0.54

    out_path = ROOT / "tests" / "codex_temp" / "stage62_low_dependency_band_stress_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["stressed_band_lower"] == hm["stressed_band_lower"]
