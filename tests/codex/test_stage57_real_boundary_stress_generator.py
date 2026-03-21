from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_real_boundary_stress_generator import build_real_boundary_stress_generator_summary


def test_stage57_real_boundary_stress_generator() -> None:
    summary = build_real_boundary_stress_generator_summary()
    hm = summary["headline_metrics"]
    scenarios = summary["scenario_results"]

    assert hm["real_trigger_rate"] >= 0.75
    assert hm["triggered_case_count"] >= 3
    assert hm["scale_bridge_factor"] > 0.60
    assert hm["stress_generator_readiness"] > 0.70
    assert set(scenarios.keys()) == {
        "context_overload",
        "fiber_congestion_wave",
        "kernel_domination_rebound",
        "coupled_patch_erosion",
    }
    assert any(item["triggered"] for item in scenarios.values())

    out_path = ROOT / "tests" / "codex_temp" / "stage57_real_boundary_stress_generator_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["real_trigger_rate"] == hm["real_trigger_rate"]
