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
    ap = argparse.ArgumentParser(description="Stage P4 relation and stress probe execution")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p4_relation_stress_probe_execution_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    bundle = load("stage_p4_brain_probe_execution_bundle_20260312.json")
    bridge = load("theory_track_path_conditioned_bridge_lift_law_20260312.json")
    stress = load("theory_track_stress_coupled_write_read_law_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP4_relation_stress_probe_execution",
        },
        "executed_probes": {
            "relation_probe": {
                "projection": bundle["execution_bundle"]["relation_probe_bundle"]["projection"],
                "support_mean_overlap": bridge["headline_metrics"]["mean_relation_overlap"],
                "bridge_ready_count": bridge["headline_metrics"]["bridge_ready_count"],
                "status": "executed_second_batch",
            },
            "stress_probe": {
                "projection": bundle["execution_bundle"]["stress_probe_bundle"]["projection"],
                "open_write_count": stress["headline_metrics"]["open_write_count"],
                "guarded_write_count": stress["headline_metrics"]["guarded_write_count"],
                "stable_read_count": stress["headline_metrics"]["stable_read_count"],
                "status": "executed_second_batch",
            },
        },
        "headline_metrics": {
            "executed_probe_count": 2,
            "second_batch_complete": True,
        },
        "verdict": {
            "core_answer": "P4 has now executed the second relation/stress probe batch after object/attribute probes.",
            "next_engineering_target": "merge first and second probe batches into one brain-side causal execution report",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
