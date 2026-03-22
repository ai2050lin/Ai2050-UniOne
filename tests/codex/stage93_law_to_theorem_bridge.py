from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage93_law_to_theorem_bridge_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage82_novelty_generalization_repair import build_novelty_generalization_repair_summary
from stage83_forward_backward_theorem_kernel import build_forward_backward_theorem_kernel_summary
from stage89_law_margin_separation import build_law_margin_separation_summary
from stage91_counterexample_attack_suite import build_counterexample_attack_suite_summary
from stage92_brain_grounding_counterexample_pack import build_brain_grounding_counterexample_pack_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def build_law_to_theorem_bridge_summary() -> dict:
    novelty = build_novelty_generalization_repair_summary()["headline_metrics"]
    theorem = build_forward_backward_theorem_kernel_summary()
    theorem_hm = theorem["headline_metrics"]
    margin = build_law_margin_separation_summary()["headline_metrics"]
    attacks = build_counterexample_attack_suite_summary()["headline_metrics"]
    brain = build_brain_grounding_counterexample_pack_summary()["headline_metrics"]

    premise_clause_strength = _clip01(
        0.34 * theorem_hm["theorem_premise_satisfaction"]
        + 0.24 * margin["law_margin_separation_score"]
        + 0.22 * margin["robustness_anchor"]
        + 0.20 * margin["family_win_rate"]
    )
    boundary_clause_strength = _clip01(
        0.30 * (1.0 - theorem_hm["projection_error_bound"])
        + 0.28 * (1.0 - theorem_hm["repair_contraction_ratio"])
        + 0.22 * margin["minimum_pairwise_margin"]
        + 0.20 * theorem_hm["bounded_novelty_margin"]
    )
    failure_clause_explicitness = _clip01(
        0.24 * attacks["attack_suite_coverage"]
        + 0.24 * attacks["multi_plane_breach_rate"]
        + 0.22 * brain["brain_counterexample_coverage"]
        + 0.14 * brain["multi_axis_grounding_break_rate"]
        + 0.16 * attacks["counterexample_attack_suite_score"]
    )
    brain_compatibility_clause = _clip01(
        0.30 * brain["brain_grounding_residual"]
        + 0.24 * (1.0 - brain["hardest_counterexample_intensity"])
        + 0.18 * brain["weakest_component_floor"]
        + 0.16 * (1.0 - attacks["hardest_attack_intensity"])
        + 0.12 * (1.0 - attacks["weakest_plane_attack_floor"])
    )
    theorem_ready_gap = _clip01(
        1.0
        - min(
            premise_clause_strength,
            boundary_clause_strength,
            failure_clause_explicitness,
            brain_compatibility_clause,
        )
    )
    law_to_theorem_bridge_score = _clip01(
        0.24 * premise_clause_strength
        + 0.24 * boundary_clause_strength
        + 0.20 * failure_clause_explicitness
        + 0.18 * brain_compatibility_clause
        + 0.14 * (1.0 - theorem_ready_gap)
    )

    clause_records = [
        {
            "name": "bounded_drive_clause",
            "premise_support": _clip01(
                0.40 * novelty["best_repaired_novelty_score"]
                + 0.34 * margin["minimum_pairwise_margin"]
                + 0.26 * margin["robustness_anchor"]
            ),
            "boundary_support": _clip01(
                0.42 * (1.0 - novelty["best_failure_after"])
                + 0.30 * theorem_hm["bounded_novelty_margin"]
                + 0.28 * margin["mean_pairwise_margin"]
            ),
        },
        {
            "name": "repair_contraction_clause",
            "premise_support": _clip01(
                0.42 * theorem_hm["theorem_premise_satisfaction"]
                + 0.32 * (1.0 - theorem_hm["repair_contraction_ratio"])
                + 0.26 * theorem_hm["cross_projection_consistency"]
            ),
            "boundary_support": _clip01(
                0.40 * (1.0 - theorem_hm["repair_contraction_ratio"])
                + 0.34 * theorem_hm["theorem_conclusion_strength"]
                + 0.26 * margin["minimum_pairwise_margin"]
            ),
        },
        {
            "name": "failure_visibility_clause",
            "premise_support": _clip01(
                0.42 * attacks["counterexample_attack_suite_score"]
                + 0.30 * attacks["multi_plane_breach_rate"]
                + 0.28 * brain["brain_counterexample_coverage"]
            ),
            "boundary_support": _clip01(
                0.38 * failure_clause_explicitness
                + 0.32 * (1.0 - attacks["system_attack_survival_score"])
                + 0.30 * brain["multi_axis_grounding_break_rate"]
            ),
        },
        {
            "name": "brain_compatibility_clause",
            "premise_support": _clip01(
                0.40 * brain["brain_grounding_residual"]
                + 0.30 * (1.0 - brain["hardest_counterexample_intensity"])
                + 0.30 * brain["weakest_component_floor"]
            ),
            "boundary_support": _clip01(
                0.38 * brain_compatibility_clause
                + 0.34 * (1.0 - attacks["hardest_attack_intensity"])
                + 0.28 * (1.0 - attacks["weakest_plane_attack_floor"])
            ),
        },
    ]

    return {
        "headline_metrics": {
            "premise_clause_strength": premise_clause_strength,
            "boundary_clause_strength": boundary_clause_strength,
            "failure_clause_explicitness": failure_clause_explicitness,
            "brain_compatibility_clause": brain_compatibility_clause,
            "theorem_ready_gap": theorem_ready_gap,
            "law_to_theorem_bridge_score": law_to_theorem_bridge_score,
        },
        "clause_records": clause_records,
        "bridge_equations": {
            "law_premise": "if family_win_rate(sqrt) ~ 1 and robustness_anchor(sqrt) ~ 1, then admissible_law_premise holds",
            "law_boundary": "E_novelty(t+1) <= sqrt(raw_drive) * (1 - p_eff) + route_conflict_mass + epsilon_boundary",
            "failure_clause": "if attack_suite breaches >= 2 planes or brain_counterexample breaches >= 3 components, theorem_closure fails",
            "brain_clause": "field_observability, bundle_sync, distributed_field must remain above grounding thresholds for theorem transfer to survive",
        },
        "status": {
            "status_short": (
                "law_to_theorem_bridge_ready"
                if law_to_theorem_bridge_score >= 0.82 and theorem_ready_gap <= 0.46
                else "law_to_theorem_bridge_transition"
            ),
            "status_label": "候选律到定理桥接块已经把前提、边界、失败条件和脑编码兼容条款写出来，但脑编码兼容项仍然明显偏弱。",
        },
        "project_readout": {
            "summary": "这一轮把 sqrt 的多轴持续占优推进成更接近定理的四件套：前提条款、边界条款、失败条款、脑编码兼容条款。",
            "next_question": "下一步要把脑编码失配如何跨语言、智能、可判伪三面扩散，写成正式的跨平面失效传播图谱。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage93 Law To Theorem Bridge",
        "",
        f"- premise_clause_strength: {hm['premise_clause_strength']:.6f}",
        f"- boundary_clause_strength: {hm['boundary_clause_strength']:.6f}",
        f"- failure_clause_explicitness: {hm['failure_clause_explicitness']:.6f}",
        f"- brain_compatibility_clause: {hm['brain_compatibility_clause']:.6f}",
        f"- theorem_ready_gap: {hm['theorem_ready_gap']:.6f}",
        f"- law_to_theorem_bridge_score: {hm['law_to_theorem_bridge_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_law_to_theorem_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
