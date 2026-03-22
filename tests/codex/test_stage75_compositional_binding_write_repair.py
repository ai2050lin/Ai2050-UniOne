from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage75_compositional_binding_write_repair import build_compositional_binding_write_repair_summary


def test_stage75_compositional_binding_write_repair() -> None:
    summary = build_compositional_binding_write_repair_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["worst_case_failure_name"] == "compositional_binding_write"
    assert hm["raw_drive"] > 0.70
    assert hm["best_law_name"] in {"log", "sqrt", "rational"}
    assert hm["best_repair_gain"] > 0.08
    assert hm["best_failure_intensity_after"] < 0.46
    assert hm["best_stability_window_gain"] > 0.62
    assert hm["best_repaired_learning_stability_score"] > 0.71
    assert set(summary["law_results"].keys()) == {"log", "sqrt", "rational"}
    assert status["status_short"] in {
        "compositional_binding_repair_transition",
        "compositional_binding_repair_ready",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage75_compositional_binding_write_repair_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
