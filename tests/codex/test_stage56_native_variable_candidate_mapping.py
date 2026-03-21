from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary


def test_stage56_native_variable_candidate_mapping() -> None:
    summary = build_native_variable_candidate_mapping_summary()
    hm = summary["headline_metrics"]
    mapping = summary["candidate_mapping"]

    assert 0.0 <= hm["primitive_set_readiness"] <= 1.0
    assert 0.0 <= hm["native_mapping_completeness"] <= 1.0
    assert hm["weakest_link_name"] in mapping
    assert len(mapping) == 6

    out_path = ROOT / "tests" / "codex_temp" / "stage56_native_variable_candidate_mapping_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["weakest_link_name"] == hm["weakest_link_name"]
