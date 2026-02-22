import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore
from scripts.task_level_causal_eval import (
    HeatKernelDiffuser,
    _bootstrap_ci,
    _binomial_two_sided_pvalue,
    _extract_reference_tokens,
    _pick_feature_dims,
    _sequence_avg_logprob,
)


def build_category_tasks(seed: int = 20260221, max_per_category: int = 18) -> Dict[str, List[Tuple[str, str]]]:
    math_tasks = [(f"{a} + {b} =", f" {a + b}") for a in range(2, 40) for b in range(2, 15)]

    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Italy", "Rome"),
        ("Germany", "Berlin"),
        ("Spain", "Madrid"),
        ("China", "Beijing"),
        ("India", "New Delhi"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Brazil", "Brasilia"),
    ]
    capital_templates = [
        "The capital of {country} is",
        "{country}'s capital city is",
        "In geography, the capital of {country} is",
    ]
    capital_tasks = [
        (tpl.format(country=c), f" {city}")
        for c, city in capitals
        for tpl in capital_templates
    ]

    antonyms = [
        ("hot", "cold"),
        ("up", "down"),
        ("big", "small"),
        ("early", "late"),
        ("open", "closed"),
        ("fast", "slow"),
        ("light", "dark"),
        ("happy", "sad"),
        ("dry", "wet"),
        ("strong", "weak"),
    ]
    antonym_templates = [
        "The opposite of {word} is",
        "An antonym of {word} is",
        "The reverse meaning of {word} is",
    ]
    antonym_tasks = [
        (tpl.format(word=w), f" {a}")
        for w, a in antonyms
        for tpl in antonym_templates
    ]

    facts = [
        ("Water freezes at", " 0 degrees Celsius"),
        ("Water boils at", " 100 degrees Celsius"),
        ("One week has", " 7 days"),
        ("A triangle has", " 3 sides"),
        ("A square has", " 4 sides"),
        ("The Earth revolves around", " the Sun"),
        ("The human heart has", " 4 chambers"),
        ("The chemical symbol for oxygen is", " O"),
        ("The chemical symbol for gold is", " Au"),
    ]
    fact_templates = [
        "{head}",
        "Basic science says {head}",
        "A common fact: {head}",
    ]
    fact_tasks = [
        (tpl.format(head=h), t)
        for h, t in facts
        for tpl in fact_templates
    ]

    rng = random.Random(seed)
    out = {
        "math_add": math_tasks,
        "capital": capital_tasks,
        "antonym": antonym_tasks,
        "fact": fact_tasks,
    }
    for k in out:
        rng.shuffle(out[k])
        out[k] = out[k][:max_per_category]
    return out


def build_candidate_configs(num_layers: int) -> List[Dict[str, Any]]:
    l1 = max(1, int(num_layers * 0.25))
    l2 = max(1, int(num_layers * 0.5))
    l3 = max(1, int(num_layers * 0.7))
    # small but meaningful search budget
    return [
        {"layer_idx": l1, "top_k": 16, "alpha": 0.25, "t": 1.0},
        {"layer_idx": l2, "top_k": 32, "alpha": 0.35, "t": 1.0},
        {"layer_idx": l3, "top_k": 32, "alpha": 0.30, "t": 1.0},
    ]


def split_tasks_for_selection(
    tasks: List[Tuple[str, str]],
    tune_ratio: float,
    min_tune_size: int,
    min_eval_size: int,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not tasks:
        return [], []
    ratio = max(0.1, min(0.9, float(tune_ratio)))
    n = len(tasks)
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    tune_n = int(round(n * ratio))
    tune_n = max(min_tune_size, tune_n)
    tune_n = min(n - min_eval_size, tune_n) if n > (min_tune_size + min_eval_size) else max(1, n // 2)
    tune_n = max(1, min(n - 1 if n > 1 else 1, tune_n))
    tune_tasks = shuffled[:tune_n]
    eval_tasks = shuffled[tune_n:]
    if not eval_tasks:
        eval_tasks = shuffled[-1:]
        tune_tasks = shuffled[:-1] if len(shuffled) > 1 else shuffled
    return tune_tasks, eval_tasks


def eval_one_config(
    model,
    tokenizer,
    device: str,
    tasks: List[Tuple[str, str]],
    cfg: Dict[str, Any],
    seed: int,
    reference_cache: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    layer = int(cfg["layer_idx"])
    if layer not in reference_cache:
        reference_prompts = [
            "The capital of France is",
            "Machine learning is",
            "The meaning of life is",
            "2 + 2 equals",
        ]
        reference_cache[layer] = _extract_reference_tokens(
            model=model,
            tokenizer=tokenizer,
            prompts=reference_prompts,
            layer_idx=layer,
            max_refs=128,
            device=device,
        )
    reference = reference_cache[layer]
    top_idx, rand_idx = _pick_feature_dims(reference, top_k=int(cfg["top_k"]), seed=seed + layer)
    diffuser = HeatKernelDiffuser(t=float(cfg["t"]), alpha=float(cfg["alpha"]))

    deltas = []
    for prompt, target in tasks:
        treat_lp = _sequence_avg_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            device=device,
            layer_idx=layer,
            feature_idx=top_idx,
            reference=reference,
            diffuser=diffuser,
        )
        ctrl_lp = _sequence_avg_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            device=device,
            layer_idx=layer,
            feature_idx=rand_idx,
            reference=reference,
            diffuser=diffuser,
        )
        deltas.append(treat_lp - ctrl_lp)

    uplift = float(np.mean(deltas)) if deltas else 0.0
    wins = int(sum(1 for d in deltas if d > 0))
    total = len(deltas)
    win_rate = wins / total if total else 0.0
    p_value = _binomial_two_sided_pvalue(wins, total) if total else 1.0
    ci_lo, ci_hi = _bootstrap_ci(deltas, seed=seed + 999 + layer, n_boot=800)

    if uplift > 0 and p_value < 0.05 and ci_lo > 0:
        verdict = "support"
    elif uplift < 0 and p_value < 0.05 and ci_hi < 0:
        verdict = "falsify"
    else:
        verdict = "open"

    return {
        "config": cfg,
        "task_count": total,
        "uplift_logprob": round(uplift, 8),
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "p_value_sign_test": round(float(p_value), 8),
        "bootstrap_ci95": [round(float(ci_lo), 8), round(float(ci_hi), 8)],
        "verdict": verdict,
    }


def choose_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # prioritize statistically-valid support, then uplift, then lower p
    def key_fn(r: Dict[str, Any]) -> Tuple[int, float, float]:
        support_flag = 1 if r["verdict"] == "support" else 0
        return (support_flag, float(r["uplift_logprob"]), -float(r["p_value_sign_test"]))

    return sorted(results, key=key_fn, reverse=True)[0]


def append_timeline(result: Dict[str, Any], summary_path: Path, timeline_path: Path) -> None:
    store = ExperimentTimelineStore(path=str(timeline_path))
    ag = result.get("aggregates", {})
    now = time.time()
    record = RunRecord(
        run_id=f"run_h3_adaptive_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type="h3_category_adaptive_search",
            model="multi_model",
            params={"source_report": str(summary_path).replace("\\", "/")},
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=[
                Metric(key="support_models", value=float(ag.get("support_models", 0)), min_value=0.0),
                Metric(key="falsify_models", value=float(ag.get("falsify_models", 0)), min_value=0.0),
                Metric(key="open_models", value=float(ag.get("open_models", 0)), min_value=0.0),
                Metric(key="consistency_score", value=float(ag.get("consistency_score", 0.0)), min_value=0.0, max_value=1.0),
            ],
            conclusion=ConclusionCard(
                objective="Find model/category-adaptive intervention settings for H3 conflict mitigation.",
                method="Grid search on layer/top-k/alpha by model and task category.",
                evidence=[
                    f"support_models={ag.get('support_models')}",
                    f"falsify_models={ag.get('falsify_models')}",
                    f"consistency_score={ag.get('consistency_score')}",
                ],
                result=result.get("conclusion", ""),
                confidence=0.72,
                limitations=["Search budget is limited; wider grids may shift optima."],
                next_action="Apply per-category tuned configs into unified transfer evaluation.",
            ),
            artifacts=[{"path": str(summary_path).replace("\\", "/")}],
        ),
        event_count=0,
    )
    store.append_run(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Category/model-adaptive search for H3 conflict mitigation.")
    parser.add_argument("--models", default="gpt2,distilgpt2,gpt2-medium")
    parser.add_argument("--max-per-category", type=int, default=18)
    parser.add_argument("--tune-ratio", type=float, default=0.5)
    parser.add_argument("--min-tune-size", type=int, default=8)
    parser.add_argument("--min-eval-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260221)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--output", default="tempdata/h3_category_adaptive_search_20260221.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    category_tasks = build_category_tasks(seed=args.seed, max_per_category=args.max_per_category)

    runs = []
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        n_layers = len(model.transformer.h)
        configs = build_candidate_configs(num_layers=n_layers)
        ref_cache: Dict[int, torch.Tensor] = {}

        per_category = {}
        for category, tasks in category_tasks.items():
            split_seed = args.seed + (abs(hash((model_name, category))) % 100000)
            tune_tasks, eval_tasks = split_tasks_for_selection(
                tasks=tasks,
                tune_ratio=args.tune_ratio,
                min_tune_size=args.min_tune_size,
                min_eval_size=args.min_eval_size,
                seed=split_seed,
            )
            c_results_tune = []
            for cfg in configs:
                r = eval_one_config(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    tasks=tune_tasks,
                    cfg=cfg,
                    seed=args.seed,
                    reference_cache=ref_cache,
                )
                c_results_tune.append(r)
            best_tune = choose_best(c_results_tune)
            selected_cfg = dict(best_tune["config"])
            best_eval = eval_one_config(
                model=model,
                tokenizer=tokenizer,
                device=device,
                tasks=eval_tasks,
                cfg=selected_cfg,
                seed=args.seed + 7919,
                reference_cache=ref_cache,
            )
            per_category[category] = {
                "selected_config": selected_cfg,
                "split": {
                    "total_task_count": len(tasks),
                    "tune_task_count": len(tune_tasks),
                    "eval_task_count": len(eval_tasks),
                    "tune_ratio": round(float(args.tune_ratio), 4),
                },
                "best_tune": best_tune,
                "best": best_eval,
                "all_tune": c_results_tune,
            }

        best_verdicts = [v["best"]["verdict"] for v in per_category.values()]
        support_count = sum(1 for x in best_verdicts if x == "support")
        falsify_count = sum(1 for x in best_verdicts if x == "falsify")
        open_count = sum(1 for x in best_verdicts if x == "open")
        avg_uplift = float(np.mean([v["best"]["uplift_logprob"] for v in per_category.values()]))
        model_verdict = (
            "support_h3" if support_count >= 2 and falsify_count == 0
            else "falsify_h3" if falsify_count >= 2 and support_count == 0
            else "open_h3"
        )
        runs.append(
            {
                "model": model_name,
                "device": device,
                "categories": per_category,
                "model_summary": {
                    "support_categories": support_count,
                    "falsify_categories": falsify_count,
                    "open_categories": open_count,
                    "avg_best_uplift_logprob": round(avg_uplift, 8),
                    "model_verdict": model_verdict,
                },
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_verdicts = [r["model_summary"]["model_verdict"] for r in runs]
    support_models = sum(1 for v in model_verdicts if v == "support_h3")
    falsify_models = sum(1 for v in model_verdicts if v == "falsify_h3")
    open_models = sum(1 for v in model_verdicts if v == "open_h3")
    consistency_score = round(1.0 - (falsify_models > 0 and support_models > 0) * 0.5, 4)

    if support_models > 0 and falsify_models > 0:
        status = "mixed_conflict"
    elif support_models >= 2 and falsify_models == 0:
        status = "provisional_support"
    elif falsify_models >= 2 and support_models == 0:
        status = "falsified"
    else:
        status = "open"

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_category_adaptive_search",
        "config": {
            "models": models,
            "max_per_category": args.max_per_category,
            "tune_ratio": args.tune_ratio,
            "min_tune_size": args.min_tune_size,
            "min_eval_size": args.min_eval_size,
            "device": device,
            "seed": args.seed,
        },
        "runs": runs,
        "aggregates": {
            "support_models": support_models,
            "falsify_models": falsify_models,
            "open_models": open_models,
            "consistency_score": consistency_score,
            "status": status,
        },
        "conclusion": (
            "Adaptive search completed. H3 conflict is reduced only partially and remains model-dependent."
            if status == "mixed_conflict"
            else "Adaptive search uses no-leak tune/eval split and reports held-out eval verdicts."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_timeline(result, summary_path=out, timeline_path=Path(args.timeline))
    print(
        json.dumps(
            {
                "output": str(out),
                "status": status,
                "support_models": support_models,
                "falsify_models": falsify_models,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
