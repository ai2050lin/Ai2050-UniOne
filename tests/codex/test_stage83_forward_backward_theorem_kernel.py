from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage83_forward_backward_theorem_kernel import build_forward_backward_theorem_kernel_summary


def test_stage83_forward_backward_theorem_kernel() -> None:
    summary = build_forward_backward_theorem_kernel_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["projection_error_bound"] < 0.08
    assert hm["repair_contraction_ratio"] < 0.23
    assert hm["bounded_novelty_margin"] > 0.70
    assert hm["cross_projection_consistency"] > 0.87
    assert hm["theorem_premise_satisfaction"] > 0.86
    assert hm["theorem_conclusion_strength"] > 0.82
    assert hm["forward_backward_theorem_kernel_score"] > 0.80
    assert len(summary["scenario_records"]) == 3
    assert status["status_short"] in {
        "forward_backward_theorem_kernel_ready",
        "forward_backward_theorem_kernel_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage83_forward_backward_theorem_kernel_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
