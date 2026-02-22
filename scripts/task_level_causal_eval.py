"""
Task-level causal evaluation for H3.

Upgraded version:
1. 100+ prompt-target tasks (programmatic suite)
2. Multi-token sequence scoring (average token log-probability)
3. Statistical summary (paired sign test p-value + bootstrap CI)
"""

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HeatKernelDiffuser:
    def __init__(self, t: float = 1.0, alpha: float = 0.35):
        self.t = t
        self.alpha = alpha

    def diffuse(self, activations: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        flat = activations.reshape(-1, activations.size(-1))
        distances = torch.cdist(flat, reference)
        weights = torch.exp(-(distances ** 2) / (4 * self.t))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        diffused = torch.matmul(weights, reference).reshape_as(activations)
        return (1 - self.alpha) * activations + self.alpha * diffused


def _build_task_suite(seed: int = 20260220, max_tasks: int = 200) -> List[Tuple[str, str, str]]:
    math_tasks = []
    for a in range(2, 32):
        for b in range(2, 13):
            math_tasks.append(("math_add", f"{a} + {b} =", f" {a + b}"))

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
        ("Russia", "Moscow"),
        ("Egypt", "Cairo"),
        ("Mexico", "Mexico City"),
        ("Argentina", "Buenos Aires"),
        ("South Korea", "Seoul"),
        ("Thailand", "Bangkok"),
    ]
    capital_tasks = [("capital", f"The capital of {c} is", f" {city}") for c, city in capitals]

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
    antonym_tasks = [("antonym", f"The opposite of {w} is", f" {a}") for w, a in antonyms]

    facts = [
        ("fact", "Water freezes at", " 0 degrees Celsius"),
        ("fact", "Water boils at", " 100 degrees Celsius"),
        ("fact", "One week has", " 7 days"),
        ("fact", "A triangle has", " 3 sides"),
        ("fact", "A square has", " 4 sides"),
        ("fact", "The Earth revolves around", " the Sun"),
        ("fact", "The largest ocean is", " the Pacific Ocean"),
        ("fact", "The human heart has", " 4 chambers"),
        ("fact", "The chemical symbol for oxygen is", " O"),
        ("fact", "The chemical symbol for gold is", " Au"),
    ]

    tasks = math_tasks + capital_tasks + antonym_tasks + facts
    rnd = random.Random(seed)
    rnd.shuffle(tasks)
    return tasks[:max_tasks]


def _extract_reference_tokens(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    max_refs: int,
    device: str,
) -> torch.Tensor:
    captures: List[torch.Tensor] = []
    storage = {"value": None}

    def _hook(_module, _input, output):
        storage["value"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = model.transformer.h[layer_idx].register_forward_hook(_hook)
    try:
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inp)
            val = storage["value"]
            if val is None:
                continue
            captures.append(val.reshape(-1, val.size(-1)))
    finally:
        handle.remove()

    if not captures:
        raise RuntimeError("No reference activations captured.")
    merged = torch.cat(captures, dim=0)
    return merged[: min(max_refs, merged.size(0))]


def _pick_feature_dims(reference: torch.Tensor, top_k: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(top_k, reference.size(-1))
    var = reference.var(dim=0)
    top_idx = torch.topk(var, k=k, largest=True).indices
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(seed + 999)
    rand_idx = torch.randperm(reference.size(-1), generator=gen, device=reference.device)[:k]
    return top_idx, rand_idx


def _sequence_avg_logprob(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str,
    layer_idx: int = -1,
    feature_idx: torch.Tensor = None,
    reference: torch.Tensor = None,
    diffuser: HeatKernelDiffuser = None,
) -> float:
    """Average log-probability for target tokens conditioned on prompt."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("-inf")

    full_ids = prompt_ids + target_ids
    if len(full_ids) <= 1:
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    hook_handle = None
    if layer_idx >= 0 and feature_idx is not None and reference is not None and diffuser is not None:
        applied = {"done": False}

        def _hook(_module, _input, output):
            if applied["done"]:
                return output
            applied["done"] = True
            if isinstance(output, tuple):
                act = output[0]
                mixed = diffuser.diffuse(act, reference)
                out = act.clone()
                out[..., feature_idx] = mixed[..., feature_idx]
                return (out,) + output[1:]
            mixed = diffuser.diffuse(output, reference)
            out = output.clone()
            out[..., feature_idx] = mixed[..., feature_idx]
            return out

        hook_handle = model.transformer.h[layer_idx].register_forward_hook(_hook)

    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # [seq, vocab]
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    start_pos = len(prompt_ids)
    total = 0.0
    count = 0
    for i, tok in enumerate(target_ids):
        pos = start_pos + i
        if pos <= 0 or pos >= logits.size(0):
            continue
        pred_logits = logits[pos - 1]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        total += float(log_probs[int(tok)].item())
        count += 1
    if count == 0:
        return float("-inf")
    return total / count


def _binomial_two_sided_pvalue(wins: int, total: int) -> float:
    if total <= 0:
        return 1.0
    p = 0.5
    # Exact binomial tail (two-sided by doubling min-tail)
    lower_tail = 0.0
    for k in range(0, wins + 1):
        lower_tail += math.comb(total, k) * (p ** k) * ((1 - p) ** (total - k))
    upper_tail = 0.0
    for k in range(wins, total + 1):
        upper_tail += math.comb(total, k) * (p ** k) * ((1 - p) ** (total - k))
    pval = min(1.0, 2 * min(lower_tail, upper_tail))
    return float(pval)


def _bootstrap_ci(values: List[float], seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return (float(lo), float(hi))


def run_eval(
    model_name: str = "gpt2",
    layer_idx: int = 3,
    top_k: int = 32,
    alpha: float = 0.35,
    t: float = 1.0,
    seed: int = 20260220,
    max_refs: int = 128,
    max_tasks: int = 200,
    bootstrap_samples: int = 2000,
    device: str = "auto",
    output_path: str = "",
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    diffuser = HeatKernelDiffuser(t=t, alpha=alpha)

    reference_prompts = [
        "The capital of France is",
        "Machine learning is",
        "The meaning of life is",
        "2 + 2 equals",
    ]
    reference = _extract_reference_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=reference_prompts,
        layer_idx=layer_idx,
        max_refs=max_refs,
        device=device,
    )
    top_idx, rand_idx = _pick_feature_dims(reference, top_k=top_k, seed=seed)

    tasks = _build_task_suite(seed=seed, max_tasks=max_tasks)
    rows = []
    deltas = []
    category_to_deltas: Dict[str, List[float]] = {}
    for category, prompt, target in tasks:
        base_lp = _sequence_avg_logprob(
            model=model, tokenizer=tokenizer, prompt=prompt, target=target, device=device
        )
        treat_lp = _sequence_avg_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            device=device,
            layer_idx=layer_idx,
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
            layer_idx=layer_idx,
            feature_idx=rand_idx,
            reference=reference,
            diffuser=diffuser,
        )
        delta = treat_lp - ctrl_lp
        deltas.append(delta)
        category_to_deltas.setdefault(category, []).append(delta)
        rows.append(
            {
                "category": category,
                "prompt": prompt,
                "target": target,
                "baseline_avg_logprob": round(base_lp, 8),
                "treatment_avg_logprob": round(treat_lp, 8),
                "control_avg_logprob": round(ctrl_lp, 8),
                "delta_treat_vs_control": round(delta, 8),
            }
        )

    avg_base = sum(r["baseline_avg_logprob"] for r in rows) / len(rows)
    avg_treat = sum(r["treatment_avg_logprob"] for r in rows) / len(rows)
    avg_ctrl = sum(r["control_avg_logprob"] for r in rows) / len(rows)
    uplift = avg_treat - avg_ctrl
    win_rate = sum(1 for r in rows if r["delta_treat_vs_control"] > 0) / len(rows)
    wins = sum(1 for d in deltas if d > 0)
    p_value = _binomial_two_sided_pvalue(wins=wins, total=len(deltas))
    ci_lo, ci_hi = _bootstrap_ci(deltas, seed=seed + 77, n_boot=bootstrap_samples)

    if uplift > 0 and p_value < 0.05 and ci_lo > 0:
        verdict = "support_h3"
    elif uplift < 0 and p_value < 0.05 and ci_hi < 0:
        verdict = "falsify_h3"
    else:
        verdict = "open_h3"

    category_summary = {}
    for cat, cat_deltas in category_to_deltas.items():
        cat_wins = sum(1 for d in cat_deltas if d > 0)
        cat_total = len(cat_deltas)
        cat_uplift = sum(cat_deltas) / cat_total if cat_total else 0.0
        cat_p = _binomial_two_sided_pvalue(cat_wins, cat_total) if cat_total else 1.0
        cat_ci_lo, cat_ci_hi = _bootstrap_ci(cat_deltas, seed=seed + len(cat) * 13, n_boot=min(1500, bootstrap_samples))
        category_summary[cat] = {
            "count": cat_total,
            "uplift_logprob": round(cat_uplift, 8),
            "win_rate": round(cat_wins / cat_total if cat_total else 0.0, 4),
            "p_value_sign_test": round(cat_p, 8),
            "bootstrap_ci95": [round(cat_ci_lo, 8), round(cat_ci_hi, 8)],
        }

    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "open_falsification_task_eval",
        "model": model_name,
        "intervention": {
            "layer_idx": layer_idx,
            "top_k": int(top_idx.numel()),
            "alpha": alpha,
            "t": t,
        },
        "summary": {
            "task_count": len(rows),
            "device": device,
            "avg_baseline_logprob": round(avg_base, 8),
            "avg_treatment_logprob": round(avg_treat, 8),
            "avg_control_logprob": round(avg_ctrl, 8),
            "task_score_uplift_logprob": round(uplift, 8),
            "win_rate": round(win_rate, 4),
            "wins": wins,
            "p_value_sign_test": round(p_value, 8),
            "bootstrap_ci95": [round(ci_lo, 8), round(ci_hi, 8)],
            "verdict": verdict,
            "category_summary": category_summary,
        },
        "details": rows,
    }

    out = output_path.strip() or f"tempdata/task_level_causal_eval_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{model_name.replace('/','_')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out}")
    print(
        f"[SUMMARY] model={model_name}, uplift={uplift:.8f}, win_rate={win_rate:.4f}, "
        f"p={p_value:.8f}, ci95=({ci_lo:.8f},{ci_hi:.8f}), verdict={verdict}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-level causal evaluation for H3.")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--max-refs", type=int, default=128)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    run_eval(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        top_k=args.top_k,
        alpha=args.alpha,
        t=args.t,
        seed=args.seed,
        max_refs=args.max_refs,
        max_tasks=args.max_tasks,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
