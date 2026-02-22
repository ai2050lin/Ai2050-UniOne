import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h3_category_adaptive_search import eval_one_config
from scripts.h3_holdout_validation import build_holdout_tasks


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _cfgs(layers: List[int], topks: List[int], alphas: List[float], t: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for l in layers:
        for k in topks:
            for a in alphas:
                out.append({"layer_idx": int(l), "top_k": int(k), "alpha": float(a), "t": float(t)})
    return out


def _rank_key(item: Dict[str, Any]) -> tuple:
    # Prefer: no falsify -> better mean uplift -> better win rate
    return (
        1 if item.get("falsify_count_total", 0) == 0 else 0,
        float(item.get("mean_uplift", 0.0)),
        float(item.get("mean_win_rate", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search constructive config for residual math_add strict failure.")
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--task-profile", choices=["standard", "expanded"], default="expanded")
    parser.add_argument("--max-per-category", type=int, default=48)
    parser.add_argument("--search-seeds", default="20260311,20260312")
    parser.add_argument("--validate-seeds", default="20260313")
    parser.add_argument("--layers", default="4,5,6")
    parser.add_argument("--topks", default="1,2,4,8")
    parser.add_argument("--alphas", default="0.0,0.005,0.01")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk-validate", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="tempdata/h3_math_residual_config_search_20260221.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    search_seeds = _parse_int_list(args.search_seeds)
    validate_seeds = _parse_int_list(args.validate_seeds)
    layers = _parse_int_list(args.layers)
    topks = _parse_int_list(args.topks)
    alphas = _parse_float_list(args.alphas)
    candidates = _cfgs(layers, topks, alphas, args.temperature)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    ref_cache: Dict[int, torch.Tensor] = {}

    # phase 1: search seeds
    search_rows: List[Dict[str, Any]] = []
    for cfg in candidates:
        per_seed = []
        falsify_total = 0
        for seed in search_seeds:
            tasks = build_holdout_tasks(seed=seed, max_per_category=args.max_per_category, task_profile=args.task_profile)["math_add_holdout"]
            r = eval_one_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                tasks=tasks,
                cfg=cfg,
                seed=seed,
                reference_cache=ref_cache,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "task_count": r["task_count"],
                    "uplift_logprob": r["uplift_logprob"],
                    "win_rate": r["win_rate"],
                    "p_value_sign_test": r["p_value_sign_test"],
                    "bootstrap_ci95": r["bootstrap_ci95"],
                    "verdict": r["verdict"],
                }
            )
            if r["verdict"] == "falsify":
                falsify_total += 1
        row = {
            "config": cfg,
            "search": per_seed,
            "falsify_count_total": falsify_total,
            "mean_uplift": round(float(np.mean([x["uplift_logprob"] for x in per_seed])), 8),
            "mean_win_rate": round(float(np.mean([x["win_rate"] for x in per_seed])), 4),
        }
        search_rows.append(row)

    search_rows.sort(key=_rank_key, reverse=True)
    shortlist = search_rows[: max(1, args.topk_validate)]

    # phase 2: validate seeds
    validated: List[Dict[str, Any]] = []
    for row in shortlist:
        cfg = row["config"]
        per_seed = []
        falsify_total = 0
        for seed in validate_seeds:
            tasks = build_holdout_tasks(seed=seed, max_per_category=args.max_per_category, task_profile=args.task_profile)["math_add_holdout"]
            r = eval_one_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                tasks=tasks,
                cfg=cfg,
                seed=seed + 999,
                reference_cache=ref_cache,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "task_count": r["task_count"],
                    "uplift_logprob": r["uplift_logprob"],
                    "win_rate": r["win_rate"],
                    "p_value_sign_test": r["p_value_sign_test"],
                    "bootstrap_ci95": r["bootstrap_ci95"],
                    "verdict": r["verdict"],
                }
            )
            if r["verdict"] == "falsify":
                falsify_total += 1
        validated.append(
            {
                **row,
                "validate": per_seed,
                "validate_falsify_count_total": falsify_total,
                "validate_mean_uplift": round(float(np.mean([x["uplift_logprob"] for x in per_seed])), 8),
                "validate_mean_win_rate": round(float(np.mean([x["win_rate"] for x in per_seed])), 4),
            }
        )

    validated.sort(
        key=lambda x: (
            1 if x.get("falsify_count_total", 0) == 0 and x.get("validate_falsify_count_total", 0) == 0 else 0,
            float(x.get("mean_uplift", 0.0) + x.get("validate_mean_uplift", 0.0)),
            float(x.get("mean_win_rate", 0.0) + x.get("validate_mean_win_rate", 0.0)),
        ),
        reverse=True,
    )

    best = validated[0] if validated else None
    status = "candidate_found" if best is not None else "no_candidate"

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_math_residual_config_search",
        "config": {
            "model": args.model,
            "task_profile": args.task_profile,
            "max_per_category": args.max_per_category,
            "search_seeds": search_seeds,
            "validate_seeds": validate_seeds,
            "layers": layers,
            "topks": topks,
            "alphas": alphas,
            "temperature": args.temperature,
            "topk_validate": args.topk_validate,
            "device": device,
        },
        "candidate_count": len(candidates),
        "status": status,
        "best_candidate": best,
        "shortlist_validated": validated,
        "top_search_rows": search_rows[:10],
        "conclusion": (
            "Residual math_add search completed; use best_candidate as constructive override trial."
            if best is not None
            else "No candidate found."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "status": status, "candidate_count": len(candidates)}, ensure_ascii=False, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
