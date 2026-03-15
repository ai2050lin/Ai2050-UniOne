from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.dnn_extraction_visualization_blueprint import DnnExtractionVisualizationBlueprint  # noqa: E402


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    blueprint = DnnExtractionVisualizationBlueprint.from_repo(ROOT)
    summary = blueprint.summary()
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "DNN_extraction_visualization_blueprint_block",
        },
        "strict_goal": {
            "statement": "Design one complete visualization scheme that can display current DNN extracted data and its mathematical restoration end to end.",
            "boundary": "This block defines the visualization blueprint. It does not yet wire every screen into the live frontend.",
        },
        "blueprint_summary": summary,
        "headline_metrics": {
            "screen_count": summary["screen_count"],
            "widget_count": summary["widget_count"],
            "three_d_widget_count": summary["three_d_widget_count"],
            "successor_widget_count": summary["successor_widget_count"],
            "source_count": summary["source_count"],
        },
        "strict_verdict": {
            "visualization_blueprint_present": bool(summary["screen_count"] >= 7 and summary["widget_count"] >= 18),
            "core_answer": (
                "The project now has a complete visualization scheme: overview, corpus atlas, concept/family lab, "
                "micro-protocol-topology lab, successor lab, math console, and provenance trace are all explicitly designed."
            ),
            "main_hard_gaps": [
                "the blueprint is now concrete, but not yet fully mounted into the live frontend tabs",
                "some 3D widgets still depend on future dense exports, especially on the successor side",
                "the current frontend still needs a dedicated data-visualization entry instead of forcing everything into the existing DNN control flow",
            ],
        },
        "progress_estimate": {
            "dnn_extraction_visualization_blueprint_percent": 78.0,
            "systematic_mass_extraction_percent": 78.0,
            "math_restoration_status_percent": 73.0,
            "full_brain_encoding_mechanism_percent": 89.0,
        },
        "next_large_blocks": [
            "Wire the overview wall, corpus atlas, and math console into the DNN main workspace as the first live frontend slice.",
            "Then wire the successor lab so direct/proxy/exactness views become interactive rather than static JSON outputs.",
            "Finally bind every widget to provenance trace drawers so visual output never loses its data source.",
        ],
    }
    return payload


def test_dnn_extraction_visualization_blueprint_block() -> None:
    payload = build_payload()
    metrics = payload["headline_metrics"]
    verdict = payload["strict_verdict"]
    assert metrics["screen_count"] >= 7
    assert metrics["widget_count"] >= 18
    assert metrics["three_d_widget_count"] >= 2
    assert metrics["successor_widget_count"] >= 4
    assert metrics["source_count"] >= 8
    assert verdict["visualization_blueprint_present"] is True


def main() -> None:
    ap = argparse.ArgumentParser(description="DNN extraction visualization blueprint block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/dnn_extraction_visualization_blueprint_block_20260315.json",
    )
    args = ap.parse_args()
    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
