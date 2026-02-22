"""
Batch geometric intervention validation for S1 causal-necessity evidence.

Goals:
1. Run a consistent geometric intervention over many prompts.
2. Compare with random/shuffled controls.
3. Save fixed-schema JSON evidence for timeline and UI ingestion.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HeatKernelDiffuser:
    def __init__(self, t: float = 1.0, alpha: float = 0.1):
        self.t = t
        self.alpha = alpha

    def diffuse(self, activations: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        flat = activations.reshape(-1, activations.size(-1))
        distances = torch.cdist(flat, reference)
        weights = torch.exp(-(distances ** 2) / (4 * self.t))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        diffused = torch.matmul(weights, reference).reshape_as(activations)
        return (1 - self.alpha) * activations + self.alpha * diffused


def _extract_reference_tokens(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    max_refs: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    acts: List[torch.Tensor] = []
    storage = {"value": None}

    def _hook(_module, _input, output):
        storage["value"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = model.transformer.h[layer_idx].register_forward_hook(_hook)
    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            value = storage["value"]
            if value is None:
                continue
            acts.append(value.reshape(-1, value.size(-1)))
    finally:
        handle.remove()

    if not acts:
        raise RuntimeError("No reference activations captured.")
    refs = torch.cat(acts, dim=0)
    return refs[: min(max_refs, refs.size(0))]


def _build_test_prompts(seed: int = 20260220) -> List[str]:
    stems = [
        "The future of",
        "The purpose of",
        "The reason for",
        "A good scientific theory should",
        "The best way to learn",
        "An intelligent system can",
        "The key challenge in",
        "A robust model should",
        "The main cause of",
        "A useful explanation is",
        "The value of",
        "The impact of",
        "A reliable method for",
        "The hidden structure of",
        "The practical use of",
        "The geometric view of",
        "A principled approach to",
        "A scalable strategy for",
    ]
    topics = [
        "artificial intelligence",
        "education",
        "sleep",
        "human memory",
        "scientific reasoning",
        "language modeling",
        "vision-language learning",
        "causal inference",
        "optimization",
        "neural geometry",
        "model interpretability",
        "long-context reasoning",
        "transfer learning",
        "symbolic logic",
        "generalization",
    ]
    endings = [
        "is",
        "depends on",
        "can be described by",
    ]

    prompts = []
    for stem in stems:
        for topic in topics:
            for ending in endings:
                prompts.append(f"{stem} {topic} {ending}")

    # Dedup + deterministic shuffle
    unique = list(dict.fromkeys(prompts))
    rnd = random.Random(seed)
    rnd.shuffle(unique)
    return unique


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _two_proportion_pvalue(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    if total_a <= 0 or total_b <= 0:
        return 1.0
    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    if variance <= 1e-12:
        return 1.0
    z = (p1 - p2) / math.sqrt(variance)
    # Two-sided p-value from normal CDF approximation.
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = max(0.0, min(1.0, 2 * (1 - cdf)))
    return round(p_value, 6)


def _compact_examples(records: List[Dict], limit: int = 12) -> Dict[str, List[Dict]]:
    changed = [r for r in records if r.get("changed")]
    unchanged = [r for r in records if not r.get("changed")]
    return {
        "changed_examples": changed[:limit],
        "unchanged_examples": unchanged[: min(6, limit)],
    }


def run_batch_test(
    model_name: str = "gpt2",
    layer_idx: int = 6,
    prompt_limit: int = 60,
    alpha: float = 0.15,
    t: float = 1.0,
    max_new_tokens: int = 12,
    max_refs: int = 128,
    seed: int = 20260220,
    include_details: bool = False,
    output_path: str = "",
) -> Dict:
    start = time.time()
    os.makedirs("tempdata", exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)

    device = "cpu"
    diffuser = HeatKernelDiffuser(t=t, alpha=alpha)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if layer_idx < 0 or layer_idx >= len(model.transformer.h):
        raise ValueError(
            f"layer_idx={layer_idx} out of range, total layers={len(model.transformer.h)}"
        )

    reference_prompts = [
        "The capital of France is",
        "Machine learning is",
        "The meaning of life is",
        "2 + 2 equals",
        "The best explanation for gravity is",
    ]
    test_prompts = _build_test_prompts(seed=seed)
    if prompt_limit > 0:
        test_prompts = test_prompts[: min(prompt_limit, len(test_prompts))]

    reference = _extract_reference_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=reference_prompts,
        layer_idx=layer_idx,
        max_refs=max_refs,
        device=device,
    )

    baseline_outputs = {
        prompt: _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for prompt in test_prompts
    }

    def run_condition(ref_tensor: torch.Tensor, condition_name: str) -> Dict:
        per_prompt = []
        changed = 0
        total = len(test_prompts)

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            original_text = baseline_outputs[prompt]
            intervention_applied = {"done": False}

            def _hook(_module, _input, output):
                if intervention_applied["done"]:
                    return output
                intervention_applied["done"] = True
                if isinstance(output, tuple):
                    act = output[0]
                    mod = diffuser.diffuse(act, ref_tensor)
                    return (mod,) + output[1:]
                return diffuser.diffuse(output, ref_tensor)

            handle = model.transformer.h[layer_idx].register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    out2 = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                handle.remove()

            intervened_text = tokenizer.decode(out2[0], skip_special_tokens=True)
            is_changed = intervened_text != original_text
            changed += int(is_changed)
            per_prompt.append(
                {
                    "prompt": prompt,
                    "original_output": original_text,
                    "intervened_output": intervened_text,
                    "changed": is_changed,
                }
            )

        return {
            "name": condition_name,
            "total_prompts": total,
            "changed_outputs": changed,
            "changed_rate": round(changed / total if total else 0.0, 4),
            "details": per_prompt,
        }

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    random_reference = (
        torch.randn(reference.shape, dtype=reference.dtype, device=device, generator=gen)
        * (reference.std() + 1e-6)
        + reference.mean()
    )
    shuffled_reference = reference[torch.randperm(reference.size(0), generator=gen)]

    treatment = run_condition(reference, "geometric_reference")
    control_random = run_condition(random_reference, "random_reference")
    control_shuffled = run_condition(shuffled_reference, "shuffled_reference")
    elapsed = time.time() - start

    uplift_random = round(treatment["changed_rate"] - control_random["changed_rate"], 4)
    uplift_shuffled = round(treatment["changed_rate"] - control_shuffled["changed_rate"], 4)
    pvalue_random = _two_proportion_pvalue(
        treatment["changed_outputs"],
        treatment["total_prompts"],
        control_random["changed_outputs"],
        control_random["total_prompts"],
    )
    pvalue_shuffled = _two_proportion_pvalue(
        treatment["changed_outputs"],
        treatment["total_prompts"],
        control_shuffled["changed_outputs"],
        control_shuffled["total_prompts"],
    )

    if uplift_random >= 0.05 and pvalue_random < 0.05:
        verdict = "positive_signal"
        conclusion_text = (
            "Geometric-reference intervention shows a statistically meaningful uplift "
            "vs random control in this configuration."
        )
    elif uplift_random > 0:
        verdict = "weak_signal"
        conclusion_text = (
            "A weak directional uplift exists but does not yet meet robust statistical criteria."
        )
    else:
        verdict = "no_signal"
        conclusion_text = (
            "No reliable uplift over controls in this configuration; causal evidence remains insufficient."
        )

    result = {
        "schema_version": "1.1",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "causal_intervention_batch",
        "model": model_name,
        "device": device,
        "intervention": {
            "type": "heat_kernel",
            "layer_idx": layer_idx,
            "t": diffuser.t,
            "alpha": diffuser.alpha,
            "reference_tokens": int(reference.size(0)),
        },
        "dataset": {
            "prompt_source": "programmatic_prompt_grid",
            "total_prompts": len(test_prompts),
            "seed": seed,
        },
        "summary": {
            "total_prompts": treatment["total_prompts"],
            "treatment_changed_outputs": treatment["changed_outputs"],
            "treatment_changed_rate": treatment["changed_rate"],
            "control_random_changed_outputs": control_random["changed_outputs"],
            "control_random_changed_rate": control_random["changed_rate"],
            "control_shuffled_changed_outputs": control_shuffled["changed_outputs"],
            "control_shuffled_changed_rate": control_shuffled["changed_rate"],
            "causal_uplift_vs_random": uplift_random,
            "causal_uplift_vs_shuffled": uplift_shuffled,
            "pvalue_vs_random": pvalue_random,
            "pvalue_vs_shuffled": pvalue_shuffled,
            "elapsed_seconds": round(elapsed, 2),
            "verdict": verdict,
        },
        "conclusion": conclusion_text,
    }

    if include_details:
        result["details"] = {
            "treatment": treatment["details"],
            "control_random": control_random["details"],
            "control_shuffled": control_shuffled["details"],
        }
    else:
        result["details"] = {
            "treatment": _compact_examples(treatment["details"]),
            "control_random": _compact_examples(control_random["details"]),
            "control_shuffled": _compact_examples(control_shuffled["details"]),
        }

    default_name = (
        f"tempdata/geometric_intervention_batch_results_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_"
        f"{model_name.replace('/', '_')}_n{len(test_prompts)}_l{layer_idx}_a{alpha:.2f}.json"
    )
    out_path = output_path.strip() or default_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_path}")
    print(
        f"[SUMMARY] model={model_name}, n={len(test_prompts)}, layer={layer_idx}, alpha={alpha:.2f}, "
        f"treatment={treatment['changed_rate']:.4f}, random={control_random['changed_rate']:.4f}, "
        f"shuffled={control_shuffled['changed_rate']:.4f}, uplift_r={uplift_random:.4f}, "
        f"p_r={pvalue_random:.6f}, elapsed={elapsed:.2f}s"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch geometric intervention with control groups.")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--layer-idx", type=int, default=6)
    parser.add_argument("--prompt-limit", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-refs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()
    run_batch_test(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        prompt_limit=args.prompt_limit,
        alpha=args.alpha,
        t=args.t,
        max_new_tokens=args.max_new_tokens,
        max_refs=args.max_refs,
        seed=args.seed,
        include_details=args.include_details,
        output_path=args.output_path,
    )
