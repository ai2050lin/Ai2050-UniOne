from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_encoding_mechanism_closed_form_v98 import build_encoding_mechanism_closed_form_v98_summary


def test_stage56_encoding_mechanism_closed_form_v98() -> None:
    summary = build_encoding_mechanism_closed_form_v98_summary()
    hm = summary["headline_metrics"]

    assert hm["feature_term_v98"] > 0.0
    assert hm["structure_term_v98"] > 0.0
    assert hm["learning_term_v98"] > 0.0
    assert hm["pressure_term_v98"] >= 0.0
    assert hm["encoding_margin_v98"] > hm["feature_term_v98"]
    assert hm["encoding_margin_v98"] > hm["structure_term_v98"]

    out_path = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v98_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["encoding_margin_v98"] == hm["encoding_margin_v98"]
