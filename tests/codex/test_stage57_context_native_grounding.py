from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary


def test_stage57_context_native_grounding() -> None:
    summary = build_context_native_grounding_summary()
    hm = summary["headline_metrics"]

    assert hm["context_native_readiness"] > 0.72
    assert hm["conditional_gate_stability"] > 0.70
    assert hm["context_bias_compressibility"] > 0.80
    assert hm["context_route_alignment"] > 0.68
    assert hm["context_upgrade_gain"] > 0.0

    out_path = ROOT / "tests" / "codex_temp" / "stage57_context_native_grounding_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["context_upgrade_gain"] == hm["context_upgrade_gain"]
