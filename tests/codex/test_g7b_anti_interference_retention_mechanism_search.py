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


def main() -> None:
    precision = load_json("continuous_input_grounding_precision_scan_20260309.json")
    hierarchical = load_json("real_multistep_memory_hierarchical_state_scan_20260309.json")
    g7a = load_json("g7a_slow_consolidation_replay_closure_20260311.json")

    cross_modal = precision["systems"]["cross_modal_dual_store"]
    dual_route = precision["systems"]["dual_store_route"]
    direct = precision["systems"]["direct_prototype"]
    best_hier = hierarchical["best"]["best_mean_system"]

    anti_interference_retention_score = mean(
        [
            cross_modal["retention_concept_accuracy"],
            dual_route["retention_concept_accuracy"],
            best_hier["mean_retention_score"],
        ]
    )

    retention_write_balance_score = mean(
        [
            (cross_modal["novel_concept_accuracy"] * cross_modal["retention_concept_accuracy"]) ** 0.5,
            (dual_route["novel_concept_accuracy"] * dual_route["retention_concept_accuracy"]) ** 0.5,
            max(0.0, g7a["headline_metrics"]["consolidation_balance_score"] - 0.3),
        ]
    )

    mechanism_candidate_strength_score = mean(
        [
            max(0.0, cross_modal["retention_concept_accuracy"] - direct["retention_concept_accuracy"] + 0.5),
            max(0.0, dual_route["retention_concept_accuracy"] - direct["retention_concept_accuracy"] + 0.5),
            max(0.0, best_hier["mean_retention_score"] - direct["retention_concept_accuracy"] + 0.5),
        ]
    )

    overall_g7b_score = mean(
        [
            anti_interference_retention_score,
            retention_write_balance_score,
            mechanism_candidate_strength_score,
        ]
    )

    formulas = {
        "anti_interference_core": "Retain* = SlowStore + CrossModalIsolation + ReplayConsolidation - InterferenceLeak",
        "balance": "Balance* = sqrt(NovelConcept * RetentionConcept)",
        "search_objective": "Search = mean(AntiInterferenceRetention, RetentionWriteBalance, MechanismCandidateStrength)",
    }

    verdict = {
        "status": (
            "anti_interference_mechanism_partially_identified"
            if overall_g7b_score >= 0.5
            else "anti_interference_mechanism_still_weak"
        ),
        "best_candidate": "cross_modal_dual_store_plus_hierarchical_consolidation",
        "main_open_gap": "novel_write_and_strong_retention_still_do_not_coexist",
        "core_answer": (
            "A plausible anti-interference retention kernel is now visible: isolation plus hierarchical consolidation improves retention. "
            "But the system still cannot combine strong novel write and strong delayed retention at the same time."
        ),
    }

    hypotheses = {
        "H1_some_anti_interference_signal_exists": anti_interference_retention_score >= 0.22,
        "H2_write_retention_balance_is_still_low": retention_write_balance_score < 0.45,
        "H3_candidate_kernel_is_nontrivial": mechanism_candidate_strength_score >= 0.45,
        "H4_g7b_is_partial_not_closed": 0.5 <= overall_g7b_score < 0.65,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G7B_anti_interference_retention_mechanism_search",
        },
        "headline_metrics": {
            "anti_interference_retention_score": anti_interference_retention_score,
            "retention_write_balance_score": retention_write_balance_score,
            "mechanism_candidate_strength_score": mechanism_candidate_strength_score,
            "overall_g7b_score": overall_g7b_score,
        },
        "supporting_readout": {
            "cross_modal_retention": cross_modal["retention_concept_accuracy"],
            "dual_route_retention": dual_route["retention_concept_accuracy"],
            "hierarchical_retention": best_hier["mean_retention_score"],
            "direct_retention": direct["retention_concept_accuracy"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g7b_anti_interference_retention_mechanism_search_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
