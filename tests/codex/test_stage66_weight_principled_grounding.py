from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage66_weight_principled_grounding import build_weight_principled_grounding_summary


def test_stage66_weight_principled_grounding() -> None:
    summary = build_weight_principled_grounding_summary()
    hm = summary["headline_metrics"]
    status = summary["status"]

    assert hm["structural_weight_grounding"] > 0.76
    assert hm["selector_weight_consistency"] > 0.79
    assert hm["principled_weight_score"] > 0.77
    assert hm["weight_subjectivity_penalty"] < 0.23
    assert status["status_short"] == "weight_grounding_strengthened"

    out_path = ROOT / "tests" / "codex_temp" / "stage66_weight_principled_grounding_20260322" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"]["status_short"] == status["status_short"]
