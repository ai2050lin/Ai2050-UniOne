from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_falsifiable_kernel_minimum import build_falsifiable_kernel_minimum_summary


def test_stage57_falsifiable_kernel_minimum() -> None:
    summary = build_falsifiable_kernel_minimum_summary()
    hm = summary["headline_metrics"]
    boundaries = summary["failure_boundaries"]

    assert hm["falsifiability_coverage"] > 0.55
    assert hm["boundary_sharpness"] > 0.05
    assert hm["counterexample_readiness"] > 0.55
    assert hm["kernel_minimum_viability"] > 0.45
    assert set(boundaries.keys()) == {
        "patch_failure_rule",
        "fiber_failure_rule",
        "route_failure_rule",
        "kernel_failure_rule",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage57_falsifiable_kernel_minimum_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["kernel_minimum_viability"] == hm["kernel_minimum_viability"]
