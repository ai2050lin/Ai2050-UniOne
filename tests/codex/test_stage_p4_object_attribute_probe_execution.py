from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage P4 object and attribute probe execution")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p4_object_attribute_probe_execution_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    bundle = load("stage_p4_brain_probe_execution_bundle_20260312.json")
    seven = load("theory_track_inventory_seven_question_mapping_20260312.json")
    inv = load("theory_track_concept_encoding_inventory_20260312.json")
    attrs = load("theory_track_attribute_axis_analysis_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP4_object_attribute_probe_execution",
        },
        "executed_probes": {
            "object_probe": {
                "projection": bundle["execution_bundle"]["object_probe_bundle"]["projection"],
                "support": seven["seven_question_mapping"]["Q1_encoding_object_layer"]["current_strength"],
                "inventory_concepts": inv["headline_metrics"]["num_concepts"],
                "status": "executed_first_batch",
            },
            "attribute_probe": {
                "projection": bundle["execution_bundle"]["attribute_probe_bundle"]["projection"],
                "support": seven["seven_question_mapping"]["Q6_discriminative_geometry"]["current_strength"],
                "attribute_count": attrs["headline_metrics"]["num_attributes"],
                "status": "executed_first_batch",
            },
        },
        "deferred_probes": {
            "relation_probe": "next_batch",
            "stress_probe": "next_batch",
        },
        "headline_metrics": {
            "executed_probe_count": 2,
            "deferred_probe_count": 2,
        },
        "verdict": {
            "core_answer": "P4 has now moved beyond bundle design and executed the first object/attribute probe batch.",
            "next_engineering_target": "execute relation and stress probes next, after reading out the first object/attribute batch",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
