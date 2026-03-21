from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_brain_encoding_direct_refinement_v31 import build_brain_encoding_direct_refinement_v31_summary


def test_stage56_brain_encoding_direct_refinement_v31() -> None:
    summary = build_brain_encoding_direct_refinement_v31_summary()
    hm = summary["headline_metrics"]

    for key in [
        "direct_origin_measure_v31",
        "direct_feature_measure_v31",
        "direct_structure_measure_v31",
        "direct_route_measure_v31",
        "direct_brain_measure_v31",
        "direct_brain_gap_v31",
        "direct_systemic_band_alignment_v31",
    ]:
        assert 0.0 <= hm[key] <= 1.0

    assert abs(hm["direct_brain_gap_v31"] - (1.0 - hm["direct_brain_measure_v31"])) < 1e-9

    out_path = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v31_20260321" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["headline_metrics"]["direct_brain_measure_v31"] == hm["direct_brain_measure_v31"]
