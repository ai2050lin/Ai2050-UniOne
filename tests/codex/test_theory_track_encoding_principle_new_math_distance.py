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
    ap = argparse.ArgumentParser(description="Theory track encoding principle and new-math distance")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_principle_new_math_distance_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    phase_map = load("theory_track_phase_p1_p4_current_map_20260312.json")
    gaps = load("theory_track_encoding_mechanism_core_gaps_20260312.json")
    axioms = load("theory_track_icspb_axiom_layer_20260312.json")
    operators = load("theory_track_icspb_operator_generation_20260312.json")
    predictions = load("theory_track_icspb_falsifiable_predictions_20260312.json")
    reasoning_law = load("theory_track_modality_unified_reasoning_law_20260312.json")
    benchmark = load("stage_p3_winner_gap_aligned_benchmark_20260312.json")

    severity_penalty = {"highest": 0.10, "high": 0.06, "medium": 0.03}
    total_penalty = sum(severity_penalty[item["severity"]] for item in gaps["gaps"])
    phase_scores = [
        phase_map["phases"]["P1"]["current_score"],
        phase_map["phases"]["P2"]["current_score"],
        phase_map["phases"]["P3"]["current_score"],
        phase_map["phases"]["P4"]["current_score"],
    ]
    mean_phase_score = sum(phase_scores) / len(phase_scores)

    encoding_principle_readiness = max(0.0, min(1.0, mean_phase_score - 0.25 * total_penalty))
    new_math_readiness = max(
        0.0,
        min(
            1.0,
            0.35
            + 0.04 * axioms["axiom_count"]
            + 0.03 * operators["generated_operator_count"]
            + 0.02 * predictions["prediction_count"]
            + 0.04 * sum(
                1
                for key in reasoning_law["components"]
                if "W_reason" in key or "Tau_reason" in key
            )
            - 0.15 * total_penalty,
        ),
    )

    roadmap = [
        {
            "route": "inventory_to_operator",
            "what_it_means": "继续从 encoding inventory 里抽 family patch、低秩轴、recurrent dims、stress profile，直接生成 operator-form 候选。",
            "current_strength": "high",
            "distance_reduction": 0.03,
        },
        {
            "route": "operator_to_gap_closure",
            "what_it_means": "围绕 object_to_readout_compatibility 主硬伤，持续做 filtered benchmark，不再 broad search。",
            "current_strength": "medium_high",
            "distance_reduction": 0.025,
        },
        {
            "route": "reasoning_slice_unification",
            "what_it_means": "把意识统一处理这条线索并入编码主理论，验证 modality-conditioned entry 和 family-conditioned shared reasoning slice。",
            "current_strength": "medium",
            "distance_reduction": 0.015,
        },
        {
            "route": "brain_side_falsification",
            "what_it_means": "把四类 probe 和十条预测推进到 intervention / causal falsification，避免理论只在内部自洽。",
            "current_strength": "medium",
            "distance_reduction": 0.03,
        },
        {
            "route": "special_math_formalization",
            "what_it_means": "继续把 ICSPB 从公理层推进到 theorem / exclusion / transport law，形成真正新数学体系。",
            "current_strength": "medium",
            "distance_reduction": 0.04,
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_principle_new_math_distance",
        },
        "encoding_principle_readiness": encoding_principle_readiness,
        "new_math_readiness": new_math_readiness,
        "remaining_distance": {
            "encoding_principle_gap": 1.0 - encoding_principle_readiness,
            "new_math_gap": 1.0 - new_math_readiness,
        },
        "current_assets": {
            "phase_mean_score": mean_phase_score,
            "axiom_count": axioms["axiom_count"],
            "generated_operator_count": operators["generated_operator_count"],
            "prediction_count": predictions["prediction_count"],
            "reasoning_slice_signal_count": sum(
                1
                for key in reasoning_law["components"]
                if "W_reason" in key or "Tau_reason" in key
            ),
            "current_best_p3_operator": benchmark["headline_metrics"]["winner"],
        },
        "roadmap": roadmap,
        "verdict": {
            "core_answer": "当前拼图已经足够支撑对编码原理和新数学体系的系统逼近，但距离最终闭合仍主要卡在 readout compatibility、brain-side causal closure 和 theorem-level formalization。",
            "distance_reading": "离编码原理的系统闭合已经进入中后段，离完整新数学体系的严格建立还在中段偏后，需要继续把 ICSPB 从对象层推到定理层和因果判伪层。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
