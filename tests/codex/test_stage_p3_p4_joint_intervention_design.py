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
    ap = argparse.ArgumentParser(description="Stage P3-P4 joint intervention design")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_p4_joint_intervention_design_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p3_joint = load("stage_p3_reasoning_slice_joint_filtered_benchmark_20260312.json")
    p4_bundle = load("stage_p4_causal_falsification_bundle_20260312.json")
    theorem_binding = load("theory_track_icspb_theorem_to_p4_binding_20260312.json")

    interventions = [
        {
            "name": "scaffolded_readout_vs_baseline_intervention",
            "target_gap": "object_to_readout_compatibility",
            "p3_operator": p3_joint["winner_operator"],
            "contrast": "baseline_filtered_readout",
            "brain_side_block": "object_family_patch_falsification",
            "expected_effect": "family patch separation should be preserved while readout score rises under scaffolded transport.",
        },
        {
            "name": "reasoning_slice_transport_intervention",
            "target_gap": "reasoning_slice_engineering_integration",
            "p3_operator": p3_joint["winner_operator"],
            "contrast": "dual_overlap_transport_operator",
            "brain_side_block": "reasoning_slice_falsification",
            "expected_effect": "joint object-to-readout and reasoning-slice transport should outperform readout-only contrast without collapsing into a global loop.",
        },
        {
            "name": "stress_guard_intervention",
            "target_gap": "stress_bound_dynamic_update_closure",
            "p3_operator": p3_joint["winner_operator"],
            "contrast": "unguarded_transport_variant",
            "brain_side_block": "stress_asymmetry_falsification",
            "expected_effect": "write should narrow before concept collapse, and stable-read should remain comparatively preserved.",
        },
        {
            "name": "anchored_relation_lift_intervention",
            "target_gap": "bridge_role_dense_coupling",
            "p3_operator": p3_joint["winner_operator"],
            "contrast": "free_symbolic_role_layer",
            "brain_side_block": "relation_anchor_falsification",
            "expected_effect": "family-anchored relation lift should retain relation coherence without losing object anchoring.",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_P4_joint_intervention_design",
        },
        "selected_family": p3_joint["selected_family"],
        "winner_operator": p3_joint["winner_operator"],
        "winner_joint_score": p3_joint["scores"]["joint_reasoning_filtered_score"],
        "falsification_block_count": len(p4_bundle["falsification_blocks"]),
        "theorem_binding_count": theorem_binding["theorem_binding_count"],
        "interventions": interventions,
        "verdict": {
            "core_answer": "P3 and P4 can now be advanced together through a small set of intervention designs that directly target the remaining encoding gaps.",
            "next_engineering_target": "execute these interventions in priority order rather than running separate benchmark and probe pipelines.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
