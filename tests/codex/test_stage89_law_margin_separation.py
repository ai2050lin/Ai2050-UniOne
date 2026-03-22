from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage89_law_margin_separation import build_law_margin_separation_summary


def test_stage89_law_margin_separation() -> None:
    summary = build_law_margin_separation_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["separated_best_law_name"] in {"sqrt", "log", "rational"}
    assert hm["family_win_rate"] >= 0.90
    assert hm["mean_pairwise_margin"] >= 0.60
    assert hm["minimum_pairwise_margin"] >= 0.55
    assert hm["dominance_axis_coverage"] >= 0.90
    assert hm["robustness_anchor"] >= 0.90
    assert hm["law_margin_separation_score"] >= 0.75
    assert len(summary["family_records"]) == 10
    assert status["status_short"] in {
        "law_margin_separation_ready",
        "law_margin_separation_transition",
    }

    out_path = ROOT / "tests" / "codex_temp" / "stage89_law_margin_separation_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
