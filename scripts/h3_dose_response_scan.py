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

from scripts.h3_category_adaptive_search import build_category_tasks, eval_one_config
from scripts.task_level_causal_eval import _resolve_layer_stack


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _layer_grid(num_layers: int, user_layers: List[int]) -> List[int]:
    if user_layers:
        out = [max(0, min(num_layers - 1, int(v))) for v in user_layers]
        return sorted(set(out))
    # default scan points across depth
    candidates = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    return sorted(set(max(0, min(num_layers - 1, int(round((num_layers - 1) * r)))) for r in candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 dose-response scan on layer/top_k/alpha.")
    parser.add_argument("--models", default="gpt2,distilgpt2,gpt2-medium")
    parser.add_argument("--layers", default="", help="Comma-separated absolute layer indices.")
    parser.add_argument("--top-ks", default="8,16,32")
    parser.add_argument("--alphas", default="0.1,0.2,0.35")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--max-per-category", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260224)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--support-categories-min", type=int, default=2)
    parser.add_argument("--falsify-categories-max", type=int, default=0)
    parser.add_argument("--output", default="tempdata/h3_dose_response_scan_20260224.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    top_ks = _parse_int_list(args.top_ks)
    alphas = _parse_float_list(args.alphas)
    user_layers = _parse_int_list(args.layers) if args.layers else []
    category_tasks = build_category_tasks(seed=args.seed, max_per_category=args.max_per_category)

    model_runs: List[Dict[str, Any]] = []
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        layers_obj, _ = _resolve_layer_stack(model)
        n_layers = len(layers_obj)
        layers = _layer_grid(num_layers=n_layers, user_layers=user_layers)
        ref_cache: Dict[int, torch.Tensor] = {}
        points: List[Dict[str, Any]] = []

        for layer_idx in layers:
            for top_k in top_ks:
                for alpha in alphas:
                    cfg = {
                        "layer_idx": int(layer_idx),
                        "top_k": int(top_k),
                        "alpha": float(alpha),
                        "t": float(args.t),
                    }
                    category_results: Dict[str, Dict[str, Any]] = {}
                    for category, tasks in category_tasks.items():
                        r = eval_one_config(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            tasks=tasks,
                            cfg=cfg,
                            seed=args.seed + 17,
                            reference_cache=ref_cache,
                        )
                        category_results[category] = r

                    support_categories = sum(
                        1 for v in category_results.values() if v.get("verdict") == "support"
                    )
                    falsify_categories = sum(
                        1 for v in category_results.values() if v.get("verdict") == "falsify"
                    )
                    avg_uplift = float(np.mean([float(v.get("uplift_logprob", 0.0)) for v in category_results.values()]))
                    avg_win_rate = float(np.mean([float(v.get("win_rate", 0.0)) for v in category_results.values()]))
                    stable = (
                        support_categories >= args.support_categories_min
                        and falsify_categories <= args.falsify_categories_max
                    )
                    points.append(
                        {
                            "layer_idx": int(layer_idx),
                            "layer_ratio": round(float(layer_idx / max(1, n_layers - 1)), 4),
                            "top_k": int(top_k),
                            "alpha": float(alpha),
                            "avg_uplift_logprob": round(avg_uplift, 8),
                            "avg_win_rate": round(avg_win_rate, 4),
                            "support_categories": int(support_categories),
                            "falsify_categories": int(falsify_categories),
                            "stable": bool(stable),
                            "category_results": category_results,
                        }
                    )

        stable_points = [p for p in points if p["stable"]]
        best_point = (
            sorted(stable_points, key=lambda x: (x["avg_uplift_logprob"], x["avg_win_rate"]), reverse=True)[0]
            if stable_points
            else sorted(points, key=lambda x: (x["avg_uplift_logprob"], x["avg_win_rate"]), reverse=True)[0]
        )
        model_runs.append(
            {
                "model": model_name,
                "num_layers": int(n_layers),
                "scan_points": points,
                "stable_count": len(stable_points),
                "best_point": best_point,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Extract common stable intervals across models.
    per_model_alpha_sets = []
    per_model_topk_sets = []
    per_model_layer_bins = []
    for run in model_runs:
        stable = [p for p in run["scan_points"] if p["stable"]]
        per_model_alpha_sets.append(set(round(float(p["alpha"]), 4) for p in stable))
        per_model_topk_sets.append(set(int(p["top_k"]) for p in stable))
        per_model_layer_bins.append(set(round(float(p["layer_ratio"]), 1) for p in stable))

    common_alphas = sorted(set.intersection(*per_model_alpha_sets)) if per_model_alpha_sets else []
    common_top_ks = sorted(set.intersection(*per_model_topk_sets)) if per_model_topk_sets else []
    common_layer_ratio_bins = sorted(set.intersection(*per_model_layer_bins)) if per_model_layer_bins else []

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_dose_response_scan",
        "config": {
            "models": models,
            "layers": user_layers,
            "top_ks": top_ks,
            "alphas": alphas,
            "t": args.t,
            "max_per_category": args.max_per_category,
            "sampling_strategy": "balanced_equal_per_category",
            "category_task_counts": {k: len(v) for k, v in category_tasks.items()},
            "support_categories_min": args.support_categories_min,
            "falsify_categories_max": args.falsify_categories_max,
            "device": device,
            "seed": args.seed,
        },
        "runs": model_runs,
        "common_stable_interval": {
            "alpha_values": common_alphas,
            "top_k_values": common_top_ks,
            "layer_ratio_bins": common_layer_ratio_bins,
        },
        "conclusion": (
            "Dose-response scan completed; use common stable interval as next H3 locked profile search space."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "common_stable_interval": result["common_stable_interval"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
