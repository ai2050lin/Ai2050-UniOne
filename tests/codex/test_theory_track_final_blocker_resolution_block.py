from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def gap(x: float) -> float:
    return clamp01(1.0 - x)


def main() -> None:
    start = time.time()
    unclosed = load_json(TEMP / "theory_track_unclosed_problem_map_block_20260314.json")
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    quotient = load_json(TEMP / "gauge_quotient_canonicalization_block.json")
    inverse_lift = load_json(TEMP / "guit_ugmt_inverse_lift_strengthened_block.json")
    theta = load_json(TEMP / "theory_track_unique_theta_star_generation_theorem_block_20260314.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")
    final_stack = load_json(TEMP / "complete_intelligence_math_final_assessment.json")

    replay_score = float(replay["headline_metrics"]["assessment_score"])
    quotient_score = float(quotient["headline_metrics"]["strict_candidate_score"])
    inverse_lift_score = float(inverse_lift["headline_metrics"]["strict_inverse_lift_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])
    stack_score = float(final_stack["headline_metrics"]["assessment_score"])

    canonical_witness_gap = gap(quotient_score)
    inverse_lift_gap = gap(inverse_lift_score)
    replay_gap = gap(replay_score)
    theta_gap = gap(theta_score)
    external_gap = gap(external_score)

    dependency_pressure = clamp01(
        0.28 * canonical_witness_gap
        + 0.20 * inverse_lift_gap
        + 0.18 * theta_gap
        + 0.18 * replay_gap
        + 0.16 * external_gap
    )

    one_shot_route_readiness = clamp01(
        0.26 * quotient_score
        + 0.22 * inverse_lift_score
        + 0.18 * theta_score
        + 0.16 * replay_score
        + 0.08 * external_score
        + 0.10 * stack_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Final_Blocker_Resolution_Block",
        },
        "headline_metrics": {
            "replay_score": replay_score,
            "canonical_witness_score": quotient_score,
            "inverse_lift_score": inverse_lift_score,
            "theta_score": theta_score,
            "external_score": external_score,
            "full_stack_score": stack_score,
            "dependency_pressure": dependency_pressure,
            "one_shot_route_readiness": one_shot_route_readiness,
        },
        "why_not_closed": [
            "canonical witness 还没形成，所以 gauge quotient 还只是 strictification-path viable，不是 strong canonical witness。",
            "strict inverse lift 依赖 canonical quotient stronger evidence，因此 GUIT -> UGMT 还没有 strict 提升。",
            "unique theta* 还没有成为 canonical witness，所以完整参数层仍是窄 basin，而不是唯一见证。",
            "strict replay 虽然 gate-level 已闭合，但结构恢复强度还没跨进 strict 带，导致 canonical write regime 仍不够硬。",
            "always-on external validation 仍是块级闭合，不是长期自然外部流下的常驻证明。",
        ],
        "one_shot_resolution_route": [
            "先把 replay structural recovery ratio 推过 strict 带，形成 canonical write regime 的运行底座。",
            "在此基础上强化 gauge quotient canonicalization，把窄 basin 收缩成 strong canonical witness candidate。",
            "再利用 stronger quotient evidence 推高 GUIT -> UGMT inverse lift，形成 strict bridge 前的最后结构支撑。",
            "随后把 unique theta* 从 readiness 推成 canonical witness，使参数层从 constrained solve 走到 unique witness。",
            "最后把这四层一起接到 true always-on external validation，拿到长期外部证明。",
        ],
        "proof_obligations": {
            "O1_replay_strict_recovery": replay_score,
            "O2_canonical_witness": quotient_score,
            "O3_strict_inverse_lift": inverse_lift_score,
            "O4_unique_theta_witness": theta_score,
            "O5_always_on_external_proof": external_score,
        },
        "verdict": {
            "overall_pass": one_shot_route_readiness >= 0.90,
            "strict_final_pass": False,
            "core_answer": (
                "The project is not blocked by missing high-level ideas anymore. It is blocked by a final dependency chain: "
                "strict replay recovery -> canonical witness -> strict inverse lift -> unique theta witness -> always-on external proof."
            ),
        },
    }

    out_file = TEMP / "final_blocker_resolution_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
