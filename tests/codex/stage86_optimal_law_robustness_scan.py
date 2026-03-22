from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage86_optimal_law_robustness_scan_20260322"
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


def _rank_laws(law_results: dict) -> list[tuple[str, dict]]:
    return sorted(
        law_results.items(),
        key=lambda item: item[1]["repaired_novelty_score"],
        reverse=True,
    )


@lru_cache(maxsize=1)
def build_optimal_law_robustness_scan_summary() -> dict:
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    closure = build_intelligence_closure_failure_map_summary()
    loop = build_forward_backward_unification_summary()["headline_metrics"]
    novelty_case = next(
        item for item in closure["scenario_records"] if item["name"] == "novelty_generalization"
    )

    scenario_variants = [
        {"name": "baseline", **DEFAULT_NOVELTY_SCENARIO},
        {"name": "novelty_up", **{**DEFAULT_NOVELTY_SCENARIO, "novelty_load": DEFAULT_NOVELTY_SCENARIO["novelty_load"] + 0.04}},
        {"name": "novelty_down", **{**DEFAULT_NOVELTY_SCENARIO, "novelty_load": DEFAULT_NOVELTY_SCENARIO["novelty_load"] - 0.04}},
        {"name": "binding_up", **{**DEFAULT_NOVELTY_SCENARIO, "binding_depth": DEFAULT_NOVELTY_SCENARIO["binding_depth"] + 0.04}},
        {"name": "repair_down", **{**DEFAULT_NOVELTY_SCENARIO, "repair_dependence": DEFAULT_NOVELTY_SCENARIO["repair_dependence"] - 0.04}},
        {"name": "abstraction_up", **{**DEFAULT_NOVELTY_SCENARIO, "abstraction_gap": DEFAULT_NOVELTY_SCENARIO["abstraction_gap"] + 0.04}},
        {"name": "scenario_replacement_high_abstraction", "novelty_load": 0.72, "binding_depth": 0.70, "repair_dependence": 0.54, "abstraction_gap": 0.82},
        {"name": "scenario_replacement_conflict", "novelty_load": 0.58, "binding_depth": 0.62, "repair_dependence": 0.86, "abstraction_gap": 0.38},
    ]

    baseline_eval = _evaluate_law_results(
        novelty_case=novelty_case,
        route=route,
        loop=loop,
        repair=repair,
        scenario=DEFAULT_NOVELTY_SCENARIO,
        laws=DEFAULT_NOVELTY_LAWS,
    )
    baseline_rank = _rank_laws(baseline_eval["law_results"])
    baseline_best = baseline_rank[0][0]

    reversed_laws = dict(reversed(list(DEFAULT_NOVELTY_LAWS.items())))
    reversed_eval = _evaluate_law_results(
        novelty_case=novelty_case,
        route=route,
        loop=loop,
        repair=repair,
        scenario=DEFAULT_NOVELTY_SCENARIO,
        laws=reversed_laws,
    )
    reversed_best = _rank_laws(reversed_eval["law_results"])[0][0]

    scan_records = []
    flip_count = 0
    for variant in scenario_variants:
        variant_eval = _evaluate_law_results(
            novelty_case=novelty_case,
            route=route,
            loop=loop,
            repair=repair,
            scenario=variant,
            laws=DEFAULT_NOVELTY_LAWS,
        )
        ranked = _rank_laws(variant_eval["law_results"])
        best_name = ranked[0][0]
        margin = ranked[0][1]["repaired_novelty_score"] - ranked[1][1]["repaired_novelty_score"]
        if best_name != baseline_best:
            flip_count += 1
        scan_records.append(
            {
                "name": variant["name"],
                "best_law_name": best_name,
                "best_margin": margin,
                "scenario_kind": (
                    "replacement"
                    if "replacement" in variant["name"]
                    else "perturbation"
                ),
            }
        )

    best_law_flip_rate = flip_count / len(scan_records)
    best_law_mean_margin = sum(record["best_margin"] for record in scan_records) / len(scan_records)
    order_invariance = baseline_best == reversed_best
    scenario_replacement_stability = sum(
        1 for record in scan_records if record["scenario_kind"] == "replacement" and record["best_law_name"] == baseline_best
    ) / sum(1 for record in scan_records if record["scenario_kind"] == "replacement")

    optimal_law_robustness_score = min(
        1.0,
        0.32 * (1.0 - best_law_flip_rate)
        + 0.24 * min(1.0, best_law_mean_margin / 0.01)
        + 0.22 * float(order_invariance)
        + 0.22 * scenario_replacement_stability,
    )

    return {
        "headline_metrics": {
            "baseline_best_law_name": baseline_best,
            "order_invariance": order_invariance,
            "best_law_flip_rate": best_law_flip_rate,
            "best_law_mean_margin": best_law_mean_margin,
            "scenario_replacement_stability": scenario_replacement_stability,
            "optimal_law_robustness_score": optimal_law_robustness_score,
        },
        "scan_records": scan_records,
        "status": {
            "status_short": (
                "optimal_law_robustness_ready"
                if best_law_flip_rate <= 0.25 and order_invariance and best_law_mean_margin >= 0.003
                else "optimal_law_robustness_transition"
            ),
            "status_label": "最优律扰动稳健性已经开始覆盖参数扰动、顺序打乱和场景替换，但仍是内部扫描，不是外部泛化验证。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage86 Optimal Law Robustness Scan",
        "",
        f"- baseline_best_law_name: {hm['baseline_best_law_name']}",
        f"- order_invariance: {hm['order_invariance']}",
        f"- best_law_flip_rate: {hm['best_law_flip_rate']:.6f}",
        f"- best_law_mean_margin: {hm['best_law_mean_margin']:.6f}",
        f"- scenario_replacement_stability: {hm['scenario_replacement_stability']:.6f}",
        f"- optimal_law_robustness_score: {hm['optimal_law_robustness_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_optimal_law_robustness_scan_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
