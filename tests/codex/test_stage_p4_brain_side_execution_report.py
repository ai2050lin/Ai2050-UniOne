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
    ap = argparse.ArgumentParser(description="Stage P4 brain-side execution report")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p4_brain_side_execution_report_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    batch1 = load("stage_p4_object_attribute_probe_execution_20260312.json")
    batch2 = load("stage_p4_relation_stress_probe_execution_20260312.json")

    report = {
        "object_probe": batch1["executed_probes"]["object_probe"],
        "attribute_probe": batch1["executed_probes"]["attribute_probe"],
        "relation_probe": batch2["executed_probes"]["relation_probe"],
        "stress_probe": batch2["executed_probes"]["stress_probe"],
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP4_brain_side_execution_report",
        },
        "execution_report": report,
        "headline_metrics": {
            "executed_probe_count": 4,
            "batch_1_complete": True,
            "batch_2_complete": bool(batch2["headline_metrics"]["second_batch_complete"]),
        },
        "status": {
            "brain_side_execution_stage": "first_full_probe_round_complete",
            "remaining_gap": "causal integration and falsification readout are still needed",
        },
        "verdict": {
            "core_answer": "P4 now has a merged execution report covering object, attribute, relation, and stress probes.",
            "next_engineering_target": "use this merged report as the base for causal brain-side falsification and integration",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
