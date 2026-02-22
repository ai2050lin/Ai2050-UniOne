"""
Feature-selective causal intervention probe.

Compared with layer-wide smoothing, this script intervenes only on top-k
feature dimensions and reports forward-pass metrics:
- top1 token change rate
- KL divergence at the final prompt position
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HeatKernelDiffuser:
    def __init__(self, t: float = 1.0, alpha: float = 0.2):
        self.t = t
        self.alpha = alpha

    def diffuse(self, activations: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        flat = activations.reshape(-1, activations.size(-1))
        distances = torch.cdist(flat, reference)
        weights = torch.exp(-(distances ** 2) / (4 * self.t))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        diffused = torch.matmul(weights, reference).reshape_as(activations)
        return (1 - self.alpha) * activations + self.alpha * diffused


def _build_prompts(seed: int = 20260220) -> List[str]:
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
    ]
    topics = [
        "artificial intelligence",
        "education",
        "sleep",
        "human memory",
        "scientific reasoning",
        "language modeling",
        "causal inference",
        "optimization",
        "neural geometry",
        "model interpretability",
    ]
    endings = ["is", "depends on", "can be explained by", "requires"]
    prompts = [f"{s} {t} {e}" for s in stems for t in topics for e in endings]
    rnd = random.Random(seed)
    rnd.shuffle(prompts)
    return prompts


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
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            value = storage["value"]
            if value is None:
                continue
            captures.append(value.reshape(-1, value.size(-1)))
    finally:
        handle.remove()

    if not captures:
        raise RuntimeError("No reference activations captured.")
    merged = torch.cat(captures, dim=0)
    return merged[: min(max_refs, merged.size(0))]


def _select_feature_dims(reference: torch.Tensor, top_k: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = reference.size(-1)
    k = min(top_k, hidden_dim)
    var_scores = reference.var(dim=0)
    top_idx = torch.topk(var_scores, k=k, largest=True).indices
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(seed + 404)
    random_idx = torch.randperm(hidden_dim, generator=gen, device=reference.device)[:k]
    return top_idx, random_idx


def _run_probe(
    *,
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    reference: torch.Tensor,
    selected_idx: torch.Tensor,
    random_idx: torch.Tensor,
    diffuser: HeatKernelDiffuser,
    device: str,
) -> Dict[str, float]:
    top1_changed_treatment = 0
    top1_changed_control = 0
    kls_treatment: List[float] = []
    kls_control: List[float] = []
    total = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline = model(**inputs).logits[:, -1, :]
        baseline_prob = F.softmax(baseline, dim=-1)
        baseline_top1 = baseline.argmax(dim=-1)

        def _run_with_indices(feature_idx: torch.Tensor) -> torch.Tensor:
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

            handle = model.transformer.h[layer_idx].register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :]
            finally:
                handle.remove()
            return logits

        treated_logits = _run_with_indices(selected_idx)
        control_logits = _run_with_indices(random_idx)

        treated_prob = F.softmax(treated_logits, dim=-1)
        control_prob = F.softmax(control_logits, dim=-1)
        treated_top1 = treated_logits.argmax(dim=-1)
        control_top1 = control_logits.argmax(dim=-1)

        top1_changed_treatment += int((treated_top1 != baseline_top1).item())
        top1_changed_control += int((control_top1 != baseline_top1).item())

        kl_t = F.kl_div((treated_prob + 1e-12).log(), baseline_prob, reduction="batchmean")
        kl_c = F.kl_div((control_prob + 1e-12).log(), baseline_prob, reduction="batchmean")
        kls_treatment.append(float(kl_t.item()))
        kls_control.append(float(kl_c.item()))
        total += 1

    return {
        "total_prompts": total,
        "top1_change_rate_treatment": round(top1_changed_treatment / total if total else 0.0, 4),
        "top1_change_rate_control": round(top1_changed_control / total if total else 0.0, 4),
        "top1_uplift": round(
            (top1_changed_treatment - top1_changed_control) / total if total else 0.0, 4
        ),
        "mean_kl_treatment": round(sum(kls_treatment) / len(kls_treatment) if kls_treatment else 0.0, 6),
        "mean_kl_control": round(sum(kls_control) / len(kls_control) if kls_control else 0.0, 6),
        "kl_uplift": round(
            (sum(kls_treatment) / len(kls_treatment) - sum(kls_control) / len(kls_control))
            if kls_treatment and kls_control
            else 0.0,
            6,
        ),
    }


def run_feature_probe(
    *,
    model_name: str = "gpt2",
    layer_idx: int = 3,
    prompt_limit: int = 240,
    top_k: int = 32,
    alpha: float = 0.35,
    t: float = 1.0,
    max_refs: int = 128,
    seed: int = 20260220,
    output_path: str = "",
) -> Dict:
    os.makedirs("tempdata", exist_ok=True)
    start = time.time()

    torch.manual_seed(seed)
    random.seed(seed)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if layer_idx < 0 or layer_idx >= len(model.transformer.h):
        raise ValueError(f"layer_idx out of range: {layer_idx}")

    reference_prompts = [
        "The capital of France is",
        "Machine learning is",
        "The meaning of life is",
        "2 + 2 equals",
        "A strong scientific explanation is",
    ]
    prompts = _build_prompts(seed=seed)
    if prompt_limit > 0:
        prompts = prompts[: min(prompt_limit, len(prompts))]

    reference = _extract_reference_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=reference_prompts,
        layer_idx=layer_idx,
        max_refs=max_refs,
        device=device,
    )
    selected_idx, random_idx = _select_feature_dims(reference, top_k=top_k, seed=seed)
    diffuser = HeatKernelDiffuser(t=t, alpha=alpha)
    metrics = _run_probe(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=layer_idx,
        reference=reference,
        selected_idx=selected_idx,
        random_idx=random_idx,
        diffuser=diffuser,
        device=device,
    )

    elapsed = round(time.time() - start, 2)
    verdict = "weak_signal" if metrics["top1_uplift"] > 0 or metrics["kl_uplift"] > 0 else "no_signal"
    result = {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "causal_intervention_feature_probe",
        "model": model_name,
        "intervention": {
            "layer_idx": layer_idx,
            "alpha": alpha,
            "t": t,
            "top_k_features": int(selected_idx.numel()),
            "feature_selector": "reference_variance_topk",
        },
        "dataset": {
            "prompt_source": "programmatic_prompt_grid",
            "total_prompts": len(prompts),
            "seed": seed,
        },
        "summary": {
            **metrics,
            "elapsed_seconds": elapsed,
            "verdict": verdict,
        },
        "artifacts": [],
    }

    default_name = (
        f"tempdata/feature_selective_probe_{datetime.now(timezone.utc).strftime('%Y%m%d')}_"
        f"{model_name.replace('/', '_')}_n{len(prompts)}_l{layer_idx}_k{int(selected_idx.numel())}_a{alpha:.2f}.json"
    )
    out = output_path.strip() or default_name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out}")
    print(
        f"[SUMMARY] model={model_name}, n={len(prompts)}, layer={layer_idx}, k={int(selected_idx.numel())}, "
        f"top1_uplift={metrics['top1_uplift']:.4f}, kl_uplift={metrics['kl_uplift']:.6f}, elapsed={elapsed:.2f}s"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-selective causal intervention probe.")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--prompt-limit", type=int, default=240)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--max-refs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    run_feature_probe(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        prompt_limit=args.prompt_limit,
        top_k=args.top_k,
        alpha=args.alpha,
        t=args.t,
        max_refs=args.max_refs,
        seed=args.seed,
        output_path=args.output_path,
    )
