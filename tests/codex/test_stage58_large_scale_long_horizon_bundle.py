from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_large_scale_long_horizon_bundle import build_large_scale_long_horizon_bundle_summary


def test_stage58_large_scale_long_horizon_bundle() -> None:
    summary = build_large_scale_long_horizon_bundle_summary()
    hm = summary["headline_metrics"]
    cases = summary["case_results"]

    assert hm["best_strategy_name"] == "joint_nativeization"
    assert hm["validated_case_count"] == 3
    assert hm["survival_rate"] == 0.75
    assert 0.65 < hm["large_scale_long_horizon_readiness"] < 0.80
    assert hm["worst_case_name"] == "coupled_scale_stress"
    assert cases["coupled_scale_stress"]["triggered"] is True

    out_path = ROOT / "tests" / "codex_temp" / "stage58_large_scale_long_horizon_bundle_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["worst_case_name"] == hm["worst_case_name"]
