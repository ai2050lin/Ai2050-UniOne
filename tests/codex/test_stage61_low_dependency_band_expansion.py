from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_low_dependency_band_expansion import build_low_dependency_band_expansion_summary


def test_stage61_low_dependency_band_expansion() -> None:
    summary = build_low_dependency_band_expansion_summary()
    hm = summary["headline_metrics"]

    assert hm["safe_point_count"] >= 4
    assert hm["band_upper"] == 0.41
    assert hm["band_lower"] == 0.33
    assert hm["band_width"] >= 0.06

    out_path = ROOT / "tests" / "codex_temp" / "stage61_low_dependency_band_expansion_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["band_lower"] == hm["band_lower"]
