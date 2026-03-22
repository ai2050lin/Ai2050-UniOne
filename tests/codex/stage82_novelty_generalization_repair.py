from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage82_novelty_generalization_repair_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage76_sqrt_repair_generalization import build_sqrt_repair_generalization_summary
from stage79_route_conflict_native_measure import build_route_conflict_native_measure_summary
from stage80_intelligence_closure_failure_map import build_intelligence_closure_failure_map_summary
from stage81_forward_backward_unification import build_forward_backward_unification_summary


DEFAULT_NOVELTY_SCENARIO = {
    "novelty_load": 0.84,
    "binding_depth": 0.76,
    "repair_dependence": 0.78,
    "abstraction_gap": 0.61,
}

DEFAULT_NOVELTY_LAWS = {
    "sqrt": {
        "drift_gain": 0.16,
        "recovery_gain": 0.17,
        "bridge_gain": 0.14,
        "coupling_gain": 0.16,
        "interpretability": 0.87,
    },
    "log": {
        "drift_gain": 0.14,
        "recovery_gain": 0.15,
        "bridge_gain": 0.13,
        "coupling_gain": 0.14,
        "interpretability": 0.91,
    },
    "rational": {
        "drift_gain": 0.15,
        "recovery_gain": 0.16,
        "bridge_gain": 0.13,
        "coupling_gain": 0.15,
        "interpretability": 0.89,
    },
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded_novelty_drive(raw_drive: float, law_name: str) -> float:
    if law_name == "sqrt":
        return math.sqrt(raw_drive)
    if law_name == "log":
        return math.log1p(3.0 * raw_drive) / math.log(4.0)
    if law_name == "rational":
        return (2.0 * raw_drive) / (1.0 + raw_drive)
    raise ValueError(f"unknown law_name={law_name}")


def _evaluate_law_results(
    *,
    novelty_case: dict,
    route: dict,
    loop: dict,
    repair: dict,
    scenario: dict,
    laws: dict,
) -> dict:
    raw_drive = (
        0.32 * scenario["novelty_load"]
        + 0.24 * scenario["binding_depth"]
        + 0.24 * scenario["repair_dependence"]
        + 0.20 * scenario["abstraction_gap"]
    )

    law_results = {}
    for law_name, params in laws.items():
        bounded_drive = _bounded_novelty_drive(raw_drive, law_name)
        alignment_anchor = _clip01(
            0.34 * route["training_route_alignment"]
            + 0.28 * loop["novelty_binding_alignment"]
            + 0.22 * repair["repaired_bounded_learning_window"]
            + 0.16 * (1.0 - route["route_conflict_mass"])
        )

        closure_drift_after = _clip01(
            novelty_case["closure_drift"]
            - params["drift_gain"] * bounded_drive
            - 0.10 * alignment_anchor
            + 0.03 * scenario["abstraction_gap"]
        )
        recovery_after = _clip01(
            novelty_case["recovery_coherence"]
            + params["recovery_gain"] * bounded_drive
            + 0.08 * alignment_anchor
            - 0.03 * route["route_conflict_mass"]
        )
        abstraction_after = _clip01(
            novelty_case["abstraction_bridge"]
            + params["bridge_gain"] * bounded_drive
            + 0.08 * alignment_anchor
            - 0.02 * scenario["abstraction_gap"]
        )
        coupling_after = _clip01(
            0.34 * recovery_after
            + 0.30 * abstraction_after
            + 0.20 * loop["forward_backward_unification_score"]
            + 0.16 * params["coupling_gain"]
        )
        failure_after = _clip01(
            0.34 * closure_drift_after
            + 0.22 * (1.0 - recovery_after)
            + 0.20 * (1.0 - abstraction_after)
            + 0.14 * scenario["repair_dependence"]
            + 0.10 * scenario["abstraction_gap"]
        )
        repair_gain = novelty_case["failure_intensity"] - failure_after
        repaired_novelty_score = _clip01(
            0.24 * recovery_after
            + 0.22 * abstraction_after
            + 0.18 * coupling_after
            + 0.18 * (1.0 - failure_after)
            + 0.18 * params["interpretability"]
        )

        law_results[law_name] = {
            "bounded_drive": bounded_drive,
            "closure_drift_after": closure_drift_after,
            "recovery_after": recovery_after,
            "abstraction_after": abstraction_after,
            "coupling_after": coupling_after,
            "failure_after": failure_after,
            "repair_gain": repair_gain,
            "repaired_novelty_score": repaired_novelty_score,
        }
    return {"raw_drive": raw_drive, "law_results": law_results}


@lru_cache(maxsize=1)
def build_novelty_generalization_repair_summary() -> dict:
    repair = build_sqrt_repair_generalization_summary()["headline_metrics"]
    route = build_route_conflict_native_measure_summary()["headline_metrics"]
    closure = build_intelligence_closure_failure_map_summary()
    loop = build_forward_backward_unification_summary()["headline_metrics"]

    novelty_case = next(
        item for item in closure["scenario_records"] if item["name"] == "novelty_generalization"
    )

    scenario = DEFAULT_NOVELTY_SCENARIO
    laws = DEFAULT_NOVELTY_LAWS

    evaluation = _evaluate_law_results(
        novelty_case=novelty_case,
        route=route,
        loop=loop,
        repair=repair,
        scenario=scenario,
        laws=laws,
    )
    raw_drive = evaluation["raw_drive"]
    law_results = evaluation["law_results"]

    best_law_name, best_law = max(
        law_results.items(),
        key=lambda item: (
            item[1]["repaired_novelty_score"],
            item[1]["repair_gain"],
            -item[1]["failure_after"],
        ),
    )
    sorted_laws = sorted(
        law_results.items(),
        key=lambda item: item[1]["repaired_novelty_score"],
        reverse=True,
    )
    best_law_margin = sorted_laws[0][1]["repaired_novelty_score"] - sorted_laws[1][1]["repaired_novelty_score"]

    sensitivity_scenarios = [
        {"name": "novelty_up", **{**scenario, "novelty_load": scenario["novelty_load"] + 0.04}},
        {"name": "novelty_down", **{**scenario, "novelty_load": scenario["novelty_load"] - 0.04}},
        {"name": "binding_up", **{**scenario, "binding_depth": scenario["binding_depth"] + 0.04}},
        {"name": "repair_up", **{**scenario, "repair_dependence": scenario["repair_dependence"] + 0.04}},
        {"name": "abstraction_up", **{**scenario, "abstraction_gap": scenario["abstraction_gap"] + 0.04}},
    ]
    sensitivity_records = []
    for variant in sensitivity_scenarios:
        variant_eval = _evaluate_law_results(
            novelty_case=novelty_case,
            route=route,
            loop=loop,
            repair=repair,
            scenario=variant,
            laws=laws,
        )
        variant_sorted = sorted(
            variant_eval["law_results"].items(),
            key=lambda item: item[1]["repaired_novelty_score"],
            reverse=True,
        )
        sensitivity_records.append(
            {
                "name": variant["name"],
                "best_law_name": variant_sorted[0][0],
                "best_law_margin": (
                    variant_sorted[0][1]["repaired_novelty_score"]
                    - variant_sorted[1][1]["repaired_novelty_score"]
                ),
            }
        )
    best_law_robust_fraction = sum(
        1 for record in sensitivity_records if record["best_law_name"] == best_law_name
    ) / len(sensitivity_records)

    return {
        "headline_metrics": {
            "worst_case_name": novelty_case["name"],
            "raw_drive": raw_drive,
            "best_law_name": best_law_name,
            "best_failure_after": best_law["failure_after"],
            "best_repair_gain": best_law["repair_gain"],
            "best_coupling_after": best_law["coupling_after"],
            "best_repaired_novelty_score": best_law["repaired_novelty_score"],
            "best_law_margin": best_law_margin,
            "best_law_robust_fraction": best_law_robust_fraction,
        },
        "worst_case_before": novelty_case,
        "law_results": law_results,
        "sensitivity_records": sensitivity_records,
        "evidence_profile": {
            "scenario_source": "handcrafted_internal_case",
            "external_fit": False,
            "warning": "最优律判断当前来自内部设定场景与局部敏感性分析，不应被解读为外部独立证明。",
        },
        "status": {
            "status_short": (
                "novelty_generalization_repair_ready"
                if best_law["failure_after"] <= 0.39 and best_law["repaired_novelty_score"] >= 0.79
                else "novelty_generalization_repair_transition"
            ),
            "status_label": "新颖泛化修复已经出现首个可执行候选，但还没有证明它能在更高抽象压力下稳定成立。",
        },
        "project_readout": {
            "summary": "这一轮直接围绕 novelty_generalization 设计修复律，专门压低新颖结构并入旧结构时的闭合裂缝。",
            "next_question": "下一步要把这条修复律并回前向与反向统一块，检查新颖泛化环的耦合是否真正增强。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage82 Novelty Generalization Repair",
        "",
        f"- worst_case_name: {hm['worst_case_name']}",
        f"- raw_drive: {hm['raw_drive']:.6f}",
        f"- best_law_name: {hm['best_law_name']}",
        f"- best_failure_after: {hm['best_failure_after']:.6f}",
        f"- best_repair_gain: {hm['best_repair_gain']:.6f}",
        f"- best_coupling_after: {hm['best_coupling_after']:.6f}",
        f"- best_repaired_novelty_score: {hm['best_repaired_novelty_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_novelty_generalization_repair_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
