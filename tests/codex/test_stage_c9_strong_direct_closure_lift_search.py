from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import test_stage_c8_retention_compatible_direct_consensus_search as c8


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C9 strong direct closure lift search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.22)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_c9_strong_direct_closure_lift_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_c8 = json.loads((TEMP_DIR / "stage_c8_retention_compatible_direct_consensus_search_20260311.json").read_text(encoding="utf-8"))
    direct = stage_c8["baselines"]["direct_multimodal"]
    base = stage_c8["best_retention_compatible_candidate"]["config"]
    strong_multimodal_target = float(stage_c8["headline_metrics"]["strong_multimodal_target"])

    best_objective = None
    best_consistency = None
    best_strong_compatible = None

    for direct_weight in [0.5, 0.65]:
        for modality_weight in [0.35, 0.5]:
            for family_weight in [0.1, 0.2]:
                for replay_steps in [0, 1, 3]:
                    for replay_after_novel_steps in [1, 2]:
                        for replay_boost in [0.0, 0.1]:
                            rows = []
                            for offset in range(int(args.num_seeds)):
                                rows.append(
                                    c8.run_system(
                                        seed=int(args.seed) + offset,
                                        noise=float(args.noise),
                                        dropout_p=float(args.dropout_p),
                                        missing_modality_p=float(args.missing_modality_p),
                                        direct_weight=direct_weight,
                                        modality_weight=modality_weight,
                                        canonical_weight=float(base["canonical_weight"]),
                                        trace_weight=float(base["trace_weight"]),
                                        family_weight=family_weight,
                                        trace_pull=float(base["trace_pull"]),
                                        replay_steps=replay_steps,
                                        replay_boost=replay_boost,
                                        replay_after_novel_steps=replay_after_novel_steps,
                                    )
                                )
                            summary = c8.summarize(rows)
                            candidate = {
                                "config": {
                                    "direct_weight": direct_weight,
                                    "modality_weight": modality_weight,
                                    "canonical_weight": float(base["canonical_weight"]),
                                    "trace_weight": float(base["trace_weight"]),
                                    "family_weight": family_weight,
                                    "trace_pull": float(base["trace_pull"]),
                                    "replay_steps": replay_steps,
                                    "replay_boost": replay_boost,
                                    "replay_after_novel_steps": replay_after_novel_steps,
                                },
                                "summary": summary,
                                "objective": c8.objective(summary),
                            }
                            if best_objective is None or candidate["objective"] > best_objective["objective"]:
                                best_objective = candidate
                            if best_consistency is None or summary["crossmodal_consistency"] > best_consistency["summary"]["crossmodal_consistency"]:
                                best_consistency = candidate
                            if (
                                summary["crossmodal_consistency"] >= float(stage_c8["headline_metrics"]["baseline_consistency_ceiling"])
                                and summary["retention_concept_accuracy"] >= float(direct["retention_concept_accuracy"])
                                and summary["overall_concept_accuracy"] >= float(direct["overall_concept_accuracy"])
                            ):
                                if best_strong_compatible is None or summary["crossmodal_consistency"] > best_strong_compatible["summary"]["crossmodal_consistency"]:
                                    best_strong_compatible = candidate

    assert best_objective is not None
    assert best_consistency is not None

    if best_strong_compatible is None:
        status = "stage_c9_no_strong_compatible_family_found"
        core_answer = (
            "No searched family reached strong direct closure while preserving retention-compatible direct consensus."
        )
        main_open_gap = "strong_target_gap_remains_after_local_lift_search"
        best_compatible_consistency = 0.0
    else:
        best_compatible_consistency = float(best_strong_compatible["summary"]["crossmodal_consistency"])
        status = (
            "stage_c9_strong_direct_closure_found"
            if best_compatible_consistency >= strong_multimodal_target
            else "stage_c9_partial_direct_closure_ceiling_estimated"
        )
        core_answer = (
            "Local lift search improves the retention-compatible direct family, but it still does not reach the strong direct closure target."
            if best_compatible_consistency < strong_multimodal_target
            else "A retention-compatible family reaches the strong direct closure target."
        )
        main_open_gap = (
            "needs_new_mechanism_beyond_local_lift"
            if best_compatible_consistency < strong_multimodal_target
            else "strong_target_reached"
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageC9_strong_direct_closure_lift_search",
        },
        "headline_metrics": {
            "strong_multimodal_target": strong_multimodal_target,
            "best_objective_consistency": float(best_objective["summary"]["crossmodal_consistency"]),
            "best_consistency_value": float(best_consistency["summary"]["crossmodal_consistency"]),
            "best_compatible_consistency": best_compatible_consistency,
            "strong_target_gap_after_best_compatible": float(max(0.0, strong_multimodal_target - best_compatible_consistency)),
        },
        "best_objective_candidate": best_objective,
        "best_consistency_candidate": best_consistency,
        "best_strong_compatible_candidate": best_strong_compatible,
        "hypotheses": {
            "H1_local_lift_improves_retention_compatible_consistency": bool(
                best_strong_compatible is not None
                and best_compatible_consistency > float(stage_c8["best_retention_compatible_candidate"]["summary"]["crossmodal_consistency"])
            ),
            "H2_local_lift_reaches_strong_target": bool(best_compatible_consistency >= strong_multimodal_target),
        },
        "verdict": {
            "status": status,
            "core_answer": core_answer,
            "main_open_gap": main_open_gap,
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
