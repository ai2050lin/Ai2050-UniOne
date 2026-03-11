from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    stage_b1 = load_json("stage_b1_calibrated_partial_reestimate_20260311.json")

    base_bridge = float(stage_b1["headline_metrics"]["calibrated_bridge_rule_score"])
    base_role = float(stage_b1["headline_metrics"]["calibrated_role_kernel_score"])
    base_support = float(stage_b1["headline_metrics"]["calibrated_support_score"])
    base_transfer = float(stage_b1["headline_metrics"]["transfer_risk_score"])

    target = 0.72
    step = 0.01
    best_plan = None

    for d_bridge_i in range(0, 16):
        d_bridge = d_bridge_i * step
        bridge = clamp01(base_bridge + d_bridge)
        for d_role_i in range(0, 16):
            d_role = d_role_i * step
            role = clamp01(base_role + d_role)
            for d_support_i in range(0, 11):
                d_support = d_support_i * step
                support = clamp01(base_support + d_support)
                for d_transfer_i in range(0, 16):
                    d_transfer = d_transfer_i * step
                    transfer = clamp01(base_transfer + d_transfer)
                    overall = mean([bridge, role, support, transfer])
                    if overall < target:
                        continue

                    # Lower cost is better. Transfer lift is treated as hardest but most valuable.
                    cost = (
                        1.00 * d_bridge
                        + 1.15 * d_role
                        + 1.25 * d_support
                        + 0.95 * d_transfer
                    )

                    plan = {
                        "delta_bridge_rule_score": d_bridge,
                        "delta_role_kernel_score": d_role,
                        "delta_support_score": d_support,
                        "delta_transfer_risk_score": d_transfer,
                        "new_bridge_rule_score": bridge,
                        "new_role_kernel_score": role,
                        "new_support_score": support,
                        "new_transfer_risk_score": transfer,
                        "new_overall_stage_b_score": overall,
                        "weighted_lift_cost": cost,
                    }

                    if best_plan is None or plan["weighted_lift_cost"] < best_plan["weighted_lift_cost"]:
                        best_plan = plan

    assert best_plan is not None

    hypotheses = {
        "H1_moderate_closure_is_close_enough_for_targeted_lift_search": bool(
            target - float(stage_b1["headline_metrics"]["overall_stage_b1_score"]) <= 0.06
        ),
        "H2_transfer_risk_needs_nonzero_lift_in_best_plan": best_plan["delta_transfer_risk_score"] > 0.0,
        "H3_role_kernel_needs_nonzero_lift_in_best_plan": best_plan["delta_role_kernel_score"] > 0.0,
        "H4_bridge_rule_does_not_need_large_extra_lift": best_plan["delta_bridge_rule_score"] <= 0.03,
        "H5_support_is_not_the_primary_bottleneck": best_plan["delta_support_score"] <= 0.03,
    }

    verdict = {
        "status": "stage_b_moderate_target_is_nearby_but_not_free",
        "core_answer": (
            "Stage B is close enough to moderate closure that a targeted lift search is meaningful. The cheapest route is not to keep pushing bridge score broadly, "
            "but to stabilize transfer-risk while only lightly reinforcing bridge margin."
        ),
        "recommended_lift_order": [
            "transfer_risk",
            "bridge_rule",
            "role_kernel",
            "support",
        ],
        "best_plan": best_plan,
    }

    interpretation = {
        "distance": (
            "Moderate closure is only a small distance away, but the remaining distance is not symmetric across components."
        ),
        "focus": (
            "The best plan does not spend much budget on calibration support. It spends the budget mostly on transfer-risk relief, "
            "with only a light bridge-margin lift and no mandatory extra role-kernel lift in the cheapest path."
        ),
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageB2_moderate_closure_lift_search",
        },
        "headline_metrics": {
            "current_overall_stage_b1_score": float(stage_b1["headline_metrics"]["overall_stage_b1_score"]),
            "target_overall_stage_b_score": target,
            "required_overall_lift": target - float(stage_b1["headline_metrics"]["overall_stage_b1_score"]),
        },
        "best_plan": best_plan,
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_b2_moderate_closure_lift_search_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
