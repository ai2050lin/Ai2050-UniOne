from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary


def test_stage60_symbolic_coefficient_grounding() -> None:
    summary = build_symbolic_coefficient_grounding_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["coefficient_grounding_coverage"] == 1.0
    assert hm["native_coefficient_score"] > 0.75
    assert 0.20 < hm["residual_grounding_gap"] < 0.30
    assert status["status_short"] == "coefficients_partially_grounded"

    out_path = ROOT / "tests" / "codex_temp" / "stage60_symbolic_coefficient_grounding_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
