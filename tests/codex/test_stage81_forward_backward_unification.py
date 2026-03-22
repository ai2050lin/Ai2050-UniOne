from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage81_forward_backward_unification import build_forward_backward_unification_summary


def test_stage81_forward_backward_unification() -> None:
    summary = build_forward_backward_unification_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["forward_selectivity"] > 0.87
    assert hm["backward_fidelity"] > 0.80
    assert hm["novelty_binding_alignment"] > 0.70
    assert hm["loop_stability_gain"] > 0.81
    assert hm["forward_backward_unification_score"] > 0.81
    assert len(summary["scenario_records"]) == 3
    assert status["status_short"] in {
        "forward_backward_unification_ready",
        "forward_backward_unification_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage81_forward_backward_unification_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
