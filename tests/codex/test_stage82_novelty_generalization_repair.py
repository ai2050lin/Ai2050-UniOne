from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary


def test_stage82_novelty_generalization_repair() -> None:
    summary = build_novelty_generalization_repair_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["worst_case_name"] == "novelty_generalization"
    assert hm["best_law_name"] in {"sqrt", "log", "rational"}
    assert hm["best_failure_after"] < 0.40
    assert hm["best_repair_gain"] > 0.05
    assert hm["best_coupling_after"] > 0.79
    assert hm["best_repaired_novelty_score"] > 0.79
    assert status["status_short"] in {
        "novelty_generalization_repair_ready",
        "novelty_generalization_repair_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage82_novelty_generalization_repair_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
