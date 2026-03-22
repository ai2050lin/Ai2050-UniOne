from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage83_forward_backward_theorem_kernel_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage72_language_projection_covariance import build_language_projection_covariance_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary
from stage81_forward_backward_unification import build_forward_backward_unification_summary
from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_forward_backward_theorem_kernel_summary() -> dict:
    projection = build_language_projection_covariance_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    closure = build_intelligence_closure_failure_map_summary()["headline_metrics"]
    loop = build_forward_backward_unification_summary()["headline_metrics"]
    novelty = build_novelty_generalization_repair_summary()["headline_metrics"]

    projection_error_bound = _clip01(
        1.0
        - (
            0.34 * (1.0 - projection["projection_gap"])
            + 0.28 * projection["context_covariance_stability"]
            + 0.22 * projection["context_shift_resilience"]
            + 0.16 * (1.0 - route["route_conflict_mass"])
        )
    )
    repair_contraction_ratio = _clip01(
        0.36 * (1.0 - loop["backward_fidelity"])
        + 0.24 * route["route_conflict_mass"]
        + 0.22 * (1.0 - novelty["best_coupling_after"])
        + 0.18 * closure["closure_repair_priority"]
    )
    bounded_novelty_margin = _clip01(
        0.30 * (1.0 - novelty["best_failure_after"])
        + 0.24 * novelty["best_repaired_novelty_score"]
        + 0.24 * loop["novelty_binding_alignment"]
        + 0.22 * (1.0 - closure["worst_case_failure_intensity"])
    )
    cross_projection_consistency = _clip01(
        0.28 * loop["forward_selectivity"]
        + 0.24 * loop["backward_fidelity"]
        + 0.18 * projection["route_conditioned_projection"]
        + 0.16 * route["inference_route_coherence"]
        + 0.14 * (1.0 - projection_error_bound)
    )
    theorem_premise_satisfaction = _clip01(
        0.26 * projection["context_covariance_stability"]
        + 0.24 * (1.0 - route["route_conflict_mass"])
        + 0.24 * loop["forward_backward_unification_score"]
        + 0.26 * novelty["best_repaired_novelty_score"]
    )
    theorem_conclusion_strength = _clip01(
        0.28 * cross_projection_consistency
        + 0.24 * bounded_novelty_margin
        + 0.24 * (1.0 - repair_contraction_ratio)
        + 0.24 * (1.0 - projection_error_bound)
    )
    forward_backward_theorem_kernel_score = _clip01(
        0.18 * theorem_premise_satisfaction
        + 0.22 * theorem_conclusion_strength
        + 0.20 * cross_projection_consistency
        + 0.20 * bounded_novelty_margin
        + 0.20 * (1.0 - repair_contraction_ratio)
    )

    scenario_records = [
        {
            "name": "stable_context_projection",
            "premise_margin": _clip01(
                0.46 * projection["context_covariance_stability"]
                + 0.30 * projection["context_shift_resilience"]
                + 0.24 * (1.0 - projection["projection_gap"])
            ),
            "conclusion_margin": _clip01(
                0.44 * (1.0 - projection_error_bound)
                + 0.32 * cross_projection_consistency
                + 0.24 * loop["forward_selectivity"]
            ),
        },
        {
            "name": "conflict_bounded_repair",
            "premise_margin": _clip01(
                0.40 * (1.0 - route["route_conflict_mass"])
                + 0.34 * route["conflict_resolution_readiness"]
                + 0.26 * loop["backward_fidelity"]
            ),
            "conclusion_margin": _clip01(
                0.40 * (1.0 - repair_contraction_ratio)
                + 0.34 * route["training_route_alignment"]
                + 0.26 * loop["loop_stability_gain"]
            ),
        },
        {
            "name": "novelty_bounded_closure",
            "premise_margin": _clip01(
                0.42 * novelty["best_repaired_novelty_score"]
                + 0.30 * novelty["best_coupling_after"]
                + 0.28 * loop["novelty_binding_alignment"]
            ),
            "conclusion_margin": _clip01(
                0.40 * bounded_novelty_margin
                + 0.34 * (1.0 - closure["worst_case_failure_intensity"])
                + 0.26 * theorem_conclusion_strength
            ),
        },
    ]

    return {
        "headline_metrics": {
            "projection_error_bound": projection_error_bound,
            "repair_contraction_ratio": repair_contraction_ratio,
            "bounded_novelty_margin": bounded_novelty_margin,
            "cross_projection_consistency": cross_projection_consistency,
            "theorem_premise_satisfaction": theorem_premise_satisfaction,
            "theorem_conclusion_strength": theorem_conclusion_strength,
            "forward_backward_theorem_kernel_score": forward_backward_theorem_kernel_score,
        },
        "scenario_records": scenario_records,
        "theorem_kernel_equations": {
            "language_projection_bound": "||Y_lang(t+1)-Y_lang*(t+1)|| <= E_proj(t+1), E_proj(t+1) < projection_error_bound",
            "repair_contraction": "E_repair(t+1) <= lambda_repair * E_repair(t) + mu_conflict * route_conflict_mass, lambda_repair < 1",
            "novelty_boundedness": "E_novelty(t+1) <= sqrt(raw_drive) * (1 - p_eff) + route_conflict_mass",
            "closure_theorem": "if q/b/g covariance and bounded repair both hold, then closure_error(t+1) < closure_error(t) + epsilon",
        },
        "status": {
            "status_short": (
                "forward_backward_theorem_kernel_ready"
                if forward_backward_theorem_kernel_score >= 0.84
                and repair_contraction_ratio <= 0.23
                and projection_error_bound <= 0.12
                else "forward_backward_theorem_kernel_transition"
            ),
            "status_label": "前向路由、反向修复和语言投影已经首次压进同一组定理约束，但收缩率和误差上界还没有完全收紧到强定理区。",
        },
        "project_readout": {
            "summary": "这一轮不再只做经验摘要，而是把语言投影误差、路由冲突残量、反向修复收缩和新颖绑定有界性压成同一组 theorem kernel 约束。",
            "next_question": "下一步要把这些定理约束推进成可判伪反例块，检查哪一种反例能最直接打穿当前 theorem kernel。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage83 Forward Backward Theorem Kernel",
        "",
        f"- projection_error_bound: {hm['projection_error_bound']:.6f}",
        f"- repair_contraction_ratio: {hm['repair_contraction_ratio']:.6f}",
        f"- bounded_novelty_margin: {hm['bounded_novelty_margin']:.6f}",
        f"- cross_projection_consistency: {hm['cross_projection_consistency']:.6f}",
        f"- theorem_premise_satisfaction: {hm['theorem_premise_satisfaction']:.6f}",
        f"- theorem_conclusion_strength: {hm['theorem_conclusion_strength']:.6f}",
        f"- forward_backward_theorem_kernel_score: {hm['forward_backward_theorem_kernel_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_forward_backward_theorem_kernel_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
