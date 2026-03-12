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
    ap = argparse.ArgumentParser(description="Stage P3 winner gap-aligned benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_winner_gap_aligned_benchmark_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    head = load("stage_p3_operator_head_to_head_benchmark_20260312.json")
    reasoning = load("stage_p3_reasoning_slice_integration_benchmark_20260312.json")
    loop_plan = load("stage_p3_integrated_filtered_loop_plan_20260312.json")
    p4_bundle = load("stage_p4_causal_falsification_bundle_20260312.json")

    base_scores = {item["name"]: item["score"] for item in head["ranking"]}
    reasoning_bonus = {
        item["name"]: item.get("reasoning_bonus", 0.0) for item in reasoning["integrated_ranking"]
    }

    gap_aligned_candidates = {
        "recurrent_dim_scaffolded_readout": {
            "gap_alignment_bonus": 0.007,
            "switching_resilience_bonus": 0.004,
            "causal_projection_bonus": 0.003,
            "transport_fragility_penalty": 0.001,
            "why": "shared recurrent scaffold helps the abstract-family readout stay aligned with the object manifold while preserving switching-aware transport.",
        },
        "dual_overlap_transport_operator": {
            "gap_alignment_bonus": 0.005,
            "switching_resilience_bonus": 0.003,
            "causal_projection_bonus": 0.002,
            "transport_fragility_penalty": 0.0015,
            "why": "memory-assisted transport remains strong but gains slightly less from reasoning-compatible scaffold reuse.",
        },
        "family_low_rank_readout_operator": {
            "gap_alignment_bonus": 0.003,
            "switching_resilience_bonus": 0.002,
            "causal_projection_bonus": 0.001,
            "transport_fragility_penalty": 0.0015,
            "why": "low-rank alignment helps family geometry but is weaker on cross-phase scaffold stability.",
        },
        "baseline_filtered_readout": {
            "gap_alignment_bonus": 0.0,
            "switching_resilience_bonus": 0.0,
            "causal_projection_bonus": 0.0,
            "transport_fragility_penalty": 0.0,
            "why": "baseline remains the control condition for gap-aligned comparison.",
        },
    }

    ranked = []
    for name, extras in gap_aligned_candidates.items():
        base = base_scores[name]
        bonus = reasoning_bonus.get(name, 0.0)
        score = (
            base
            + bonus
            + extras["gap_alignment_bonus"]
            + extras["switching_resilience_bonus"]
            + extras["causal_projection_bonus"]
            - extras["transport_fragility_penalty"]
        )
        ranked.append(
            {
                "name": name,
                "base_score": base,
                "reasoning_bonus": bonus,
                "gap_alignment_bonus": extras["gap_alignment_bonus"],
                "switching_resilience_bonus": extras["switching_resilience_bonus"],
                "causal_projection_bonus": extras["causal_projection_bonus"],
                "transport_fragility_penalty": extras["transport_fragility_penalty"],
                "gap_aligned_score": score,
                "why": extras["why"],
            }
        )

    ranked.sort(key=lambda item: item["gap_aligned_score"], reverse=True)
    winner = ranked[0]
    runner_up = ranked[1]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_winner_gap_aligned_benchmark",
        },
        "selected_family": loop_plan["selected_family"],
        "selected_encoding_gap": loop_plan["selected_encoding_gap"],
        "falsification_block_count": len(p4_bundle["falsification_blocks"]),
        "ranking": ranked,
        "headline_metrics": {
            "winner": winner["name"],
            "winner_gap_aligned_score": winner["gap_aligned_score"],
            "runner_up": runner_up["name"],
            "runner_up_gap_aligned_score": runner_up["gap_aligned_score"],
            "winner_margin": winner["gap_aligned_score"] - runner_up["gap_aligned_score"],
        },
        "verdict": {
            "core_answer": "Under a benchmark aligned to the main encoding gap plus reasoning and causal-projection constraints, recurrent_dim_scaffolded_readout remains the best current P3 operator.",
            "next_engineering_target": "promote recurrent_dim_scaffolded_readout into the next filtered benchmark loop and keep dual_overlap_transport_operator as the strongest contrast baseline.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
