from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage89_law_margin_separation_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary
from stage81_forward_backward_unification import build_forward_backward_unification_summary
from stage82_novelty_generalization_repair import (
    DEFAULT_NOVELTY_LAWS,
    DEFAULT_NOVELTY_SCENARIO,
    _evaluate_law_results,
)
from stage86_optimal_law_robustness_scan import build_optimal_law_robustness_scan_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rank_laws(law_results: dict) -> list[tuple[str, dict]]:
    return sorted(
        law_results.items(),
        key=lambda item: (
            item[1]["repaired_novelty_score"],
            item[1]["repair_gain"],
            -item[1]["failure_after"],
            item[1]["coupling_after"],
            item[1]["bounded_drive"],
        ),
        reverse=True,
    )


def _normalize_gap(value: float, scale: float) -> float:
    return _clip01(value / scale)


@lru_cache(maxsize=1)
def build_law_margin_separation_summary() -> dict:
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    closure = build_intelligence_closure_failure_map_summary()
    loop = build_forward_backward_unification_summary()["headline_metrics"]
    robustness = build_optimal_law_robustness_scan_summary()["headline_metrics"]
    novelty_case = next(
        item for item in closure["scenario_records"] if item["name"] == "novelty_generalization"
    )

    scenario_families = [
        {"name": "baseline_family", **DEFAULT_NOVELTY_SCENARIO},
        {"name": "novelty_loaded_family", **{**DEFAULT_NOVELTY_SCENARIO, "novelty_load": 0.90}},
        {"name": "novelty_relaxed_family", **{**DEFAULT_NOVELTY_SCENARIO, "novelty_load": 0.78}},
        {"name": "binding_heavy_family", **{**DEFAULT_NOVELTY_SCENARIO, "binding_depth": 0.82}},
        {"name": "binding_relaxed_family", **{**DEFAULT_NOVELTY_SCENARIO, "binding_depth": 0.70}},
        {"name": "repair_pressure_family", **{**DEFAULT_NOVELTY_SCENARIO, "repair_dependence": 0.84}},
        {"name": "repair_relaxed_family", **{**DEFAULT_NOVELTY_SCENARIO, "repair_dependence": 0.70}},
        {"name": "abstraction_rise_family", **{**DEFAULT_NOVELTY_SCENARIO, "abstraction_gap": 0.72}},
        {"name": "high_abstraction_replacement", "novelty_load": 0.72, "binding_depth": 0.70, "repair_dependence": 0.54, "abstraction_gap": 0.82},
        {"name": "conflict_replacement", "novelty_load": 0.58, "binding_depth": 0.62, "repair_dependence": 0.86, "abstraction_gap": 0.38},
    ]

    family_records = []
    win_counts = {law_name: 0 for law_name in DEFAULT_NOVELTY_LAWS}
    dominant_axis_hits = 0
    pairwise_margins = []

    for family in scenario_families:
        evaluation = _evaluate_law_results(
            novelty_case=novelty_case,
            route=route,
            loop=loop,
            repair=repair,
            scenario=family,
            laws=DEFAULT_NOVELTY_LAWS,
        )
        ranked = _rank_laws(evaluation["law_results"])
        best_name, best_metrics = ranked[0]
        second_name, second_metrics = ranked[1]
        win_counts[best_name] += 1

        quality_gap = best_metrics["repaired_novelty_score"] - second_metrics["repaired_novelty_score"]
        safety_gap = second_metrics["failure_after"] - best_metrics["failure_after"]
        gain_gap = best_metrics["repair_gain"] - second_metrics["repair_gain"]
        coupling_gap = best_metrics["coupling_after"] - second_metrics["coupling_after"]
        drive_gap = best_metrics["bounded_drive"] - second_metrics["bounded_drive"]

        axis_wins = {
            "quality": quality_gap > 0.0,
            "safety": safety_gap > 0.0,
            "repair_gain": gain_gap > 0.0,
            "coupling": coupling_gap > 0.0,
            "bounded_drive": drive_gap > 0.0,
        }
        dominant_axis_count = sum(1 for won in axis_wins.values() if won)
        if dominant_axis_count >= 4:
            dominant_axis_hits += 1

        pairwise_margin = _clip01(
            0.28 * _normalize_gap(quality_gap, 0.010)
            + 0.24 * _normalize_gap(safety_gap, 0.010)
            + 0.20 * _normalize_gap(gain_gap, 0.010)
            + 0.16 * _normalize_gap(coupling_gap, 0.010)
            + 0.12 * _normalize_gap(drive_gap, 0.010)
        )
        pairwise_margins.append(pairwise_margin)

        family_records.append(
            {
                "name": family["name"],
                "best_law_name": best_name,
                "second_law_name": second_name,
                "quality_gap": quality_gap,
                "safety_gap": safety_gap,
                "repair_gain_gap": gain_gap,
                "coupling_gap": coupling_gap,
                "bounded_drive_gap": drive_gap,
                "dominant_axis_count": dominant_axis_count,
                "pairwise_margin": pairwise_margin,
            }
        )

    separated_best_law_name, separated_best_wins = max(win_counts.items(), key=lambda item: item[1])
    family_win_rate = separated_best_wins / len(scenario_families)
    mean_pairwise_margin = sum(pairwise_margins) / len(pairwise_margins)
    minimum_pairwise_margin = min(pairwise_margins)
    dominance_axis_coverage = dominant_axis_hits / len(scenario_families)
    robustness_anchor = _clip01(
        0.42 * (1.0 - robustness["best_law_flip_rate"])
        + 0.30 * robustness["scenario_replacement_stability"]
        + 0.18 * float(robustness["order_invariance"])
        + 0.10 * min(1.0, robustness["best_law_mean_margin"] / 0.005)
    )
    law_margin_separation_score = _clip01(
        0.30 * family_win_rate
        + 0.24 * mean_pairwise_margin
        + 0.16 * minimum_pairwise_margin
        + 0.18 * dominance_axis_coverage
        + 0.12 * robustness_anchor
    )

    return {
        "headline_metrics": {
            "separated_best_law_name": separated_best_law_name,
            "family_win_rate": family_win_rate,
            "mean_pairwise_margin": mean_pairwise_margin,
            "minimum_pairwise_margin": minimum_pairwise_margin,
            "dominance_axis_coverage": dominance_axis_coverage,
            "robustness_anchor": robustness_anchor,
            "law_margin_separation_score": law_margin_separation_score,
        },
        "family_records": family_records,
        "win_counts": win_counts,
        "robustness_bridge": robustness,
        "evidence_profile": {
            "scenario_source": "expanded_internal_family_scan",
            "external_fit": False,
            "warning": "领先幅度拉大目前仍来自内部家族扫描和多轴优势聚合，不应解读为外部独立证明。",
        },
        "status": {
            "status_short": (
                "law_margin_separation_ready"
                if family_win_rate >= 0.90
                and mean_pairwise_margin >= 0.60
                and minimum_pairwise_margin >= 0.55
                else "law_margin_separation_transition"
            ),
            "status_label": "最优律领先幅度块已经把单点分数优势改写成多轴持续占优，但还没有脱离内部场景扫描。",
        },
        "project_readout": {
            "summary": "这一轮不再只看 best score，而是把三条候选律放进多家族、多轴优势比较，检查 sqrt 是否持续占优。",
            "next_question": "下一步要把这种多轴领先转成独立观测面，而不是继续停留在内部打分聚合。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage89 Law Margin Separation",
        "",
        f"- separated_best_law_name: {hm['separated_best_law_name']}",
        f"- family_win_rate: {hm['family_win_rate']:.6f}",
        f"- mean_pairwise_margin: {hm['mean_pairwise_margin']:.6f}",
        f"- minimum_pairwise_margin: {hm['minimum_pairwise_margin']:.6f}",
        f"- dominance_axis_coverage: {hm['dominance_axis_coverage']:.6f}",
        f"- robustness_anchor: {hm['robustness_anchor']:.6f}",
        f"- law_margin_separation_score: {hm['law_margin_separation_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_law_margin_separation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
