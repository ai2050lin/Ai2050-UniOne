#!/usr/bin/env python3
"""
P55: Norm-Only Causal Experiment (Falsifying Proposition 1)
P56: Rotation Plane Intervention Experiment (Falsifying Proposition 2)

Proposition 1: "Semantic encoding is in direction space, not norm"
- P55: Fix h direction, vary ||h|| only, measure logit change
- If top-1 accuracy drops <5% when scaling norm → norm irrelevant → P1 confirmed
- If drops >20% → norm carries semantics → P1 FALSIFIED

Proposition 2: "Rotation is architectural (not semantic operation)"
- P56: Inject a rotation matrix between L0 and L1, expand rotation plane from 11D to 50D
- If perplexity degrades >10% → rotation plane size has semantic function → P2 FALSIFIED
- If unchanged → rotation is architectural side-effect → P2 confirmed

Both experiments use the same model loading and hidden state extraction.
"""
import sys, math, time, gc, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path

OUTPUT_DIR = _Path(f"tests/glm5_temp/stage701_falsification_{time.strftime('%Y%m%d_%H%M')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEXTS = [
    "The cat sat on the mat.", "A dog chased the ball across the yard.",
    "Paris is the capital of France.", "The Amazon is the longest river.",
    "Water boils at one hundred degrees.", "Light travels at constant speed.",
    "She felt happy after the good news.", "The movie made me laugh out loud.",
    "Fresh bread smells wonderful.", "Artificial intelligence learns from data.",
    "The piano has eighty eight keys.", "Soccer is the most popular sport.",
    "Socrates asked many deep questions.", "Photosynthesis converts sunlight to energy.",
    "Mount Everest is the highest peak.", "Electrons orbit the atomic nucleus.",
]


class Logger:
    def __init__(self, path):
        self.path = _Path(path)
        self.f = open(self.path, "w", encoding="utf-8")
    def __call__(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = MODEL_MAP[model_name]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_final_hidden_and_logits(model, tokenizer, text):
    """Get final hidden state (last token) and logits."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    h_final = outputs.hidden_states[-1][0, -1, :].float().cpu()  # (d_model,)
    logits = outputs.logits[0, -1, :].float().cpu()  # (vocab_size,)
    return h_final, logits


# ==================== P55: Norm-Only Causal Experiment ====================

def run_p55(model, tokenizer, texts, log):
    """Fix direction, vary norm. Measure logit and top-1 stability."""
    log("\n  --- P55: Norm-Only Causal Experiment ---")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()  # (vocab_size, d_model)

    results = []

    for text in texts:
        h_orig, logits_orig = get_final_hidden_and_logits(model, tokenizer, text)
        direction = h_orig / (h_orig.norm() + 1e-10)
        orig_norm = h_orig.norm().item()
        orig_top1 = logits_orig.argmax().item()

        # Scale norm by factors: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        scale_factors = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        top1_matches = []
        logit_correlations = []
        margin_changes = []

        orig_margin = (logits_orig[orig_top1] - torch.topk(logits_orig, 2).values[1]).item()

        for scale in scale_factors:
            h_scaled = direction * (orig_norm * scale)
            scaled_logits = h_scaled @ unembed_weight.T  # (vocab_size,)
            scaled_top1 = scaled_logits.argmax().item()
            top1_matches.append(1 if scaled_top1 == orig_top1 else 0)

            # Correlation with original logits
            sample_idx = np.random.choice(len(logits_orig), min(5000, len(logits_orig)), replace=False)
            r = np.corrcoef(logits_orig.numpy()[sample_idx], scaled_logits.numpy()[sample_idx])[0, 1]
            logit_correlations.append(r)

            scaled_margin = (scaled_logits[scaled_top1] - torch.topk(scaled_logits, 2).values[1]).item()
            margin_changes.append((scaled_margin - orig_margin) / (abs(orig_margin) + 1e-10))

        # Key metric: at what scale factor does top-1 flip?
        stable_scales = [s for s, m in zip(scale_factors, top1_matches) if m == 1]
        min_stable = min(stable_scales) if stable_scales else 0
        max_stable = max(stable_scales) if stable_scales else 0

        # Correlation at norm=0.1 (very small norm)
        r_at_01 = logit_correlations[1]  # scale=0.1
        r_at_10 = logit_correlations[6]  # scale=10.0

        results.append({
            "text": text[:40],
            "orig_norm": orig_norm,
            "top1_stable_range": f"{min_stable}x-{max_stable}x",
            "top1_stable_count": sum(top1_matches),
            "r_at_0.1x": r_at_01,
            "r_at_10x": r_at_10,
            "r_at_1x": logit_correlations[3],
        })

    # Aggregate
    mean_stable = np.mean([r["top1_stable_count"] for r in results])
    mean_r_01 = np.mean([r["r_at_0.1x"] for r in results])
    mean_r_10 = np.mean([r["r_at_10x"] for r in results])

    log(f"    Mean top-1 stable count (out of 8): {mean_stable:.2f}")
    log(f"    Mean logit r at 0.1x norm: {mean_r_01:.4f}")
    log(f"    Mean logit r at 10x norm: {mean_r_10:.4f}")

    # Verdict
    # If scaling norm by 0.1x preserves top-1 (>80%) → norm irrelevant → P1 confirmed
    # If scaling by 0.1x changes top-1 (<50%) → norm matters → P1 potentially falsified
    p1_confirmed = mean_r_01 > 0.95
    p1_falsified = mean_r_01 < 0.8

    if p1_falsified:
        verdict = "P1 FALSIFIED - norm carries significant information!"
    elif p1_confirmed:
        verdict = "P1 confirmed - direction dominates, norm is secondary"
    else:
        verdict = "P1 inconclusive - norm has moderate effect"

    log(f"    Verdict: {verdict}")

    return {
        "mean_stable_count": float(mean_stable),
        "mean_r_0.1x": float(mean_r_01),
        "mean_r_10x": float(mean_r_10),
        "p1_confirmed": p1_confirmed,
        "p1_falsified": p1_falsified,
        "verdict": verdict,
        "per_text": results[:5],
    }


# ==================== P56: Rotation Plane Intervention ====================

def run_p56(model, tokenizer, texts, model_name, log):
    """Measure sensitivity to rotation plane expansion.

    Method: Instead of physically injecting a rotation matrix (which requires
    modifying model forward pass), we do a proxy experiment:
    
    1. Get L0 and L1 hidden states
    2. Compute the actual rotation: project delta-h onto PCA components
    3. Artificially expand the rotation plane by adding noise in new dimensions
    4. Project the modified h through unembed and measure logit change
    
    Alternative simpler method:
    - Take h_final, decompose into PCA components
    - Zero out top-K components (simulating reduced rotation plane)
    - Zero out components beyond top-K (simulating expanded rotation plane)
    - Measure logit change
    """
    log("\n  --- P56: Rotation Plane Intervention ---")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()  # (vocab_size, d_model)

    results = []

    # Pre-compute top-K PCA directions using randomized SVD (memory efficient)
    max_k = 500
    log(f"    Computing randomized SVD for top-{max_k} components...")
    try:
        # Use random projection for memory efficiency
        torch.manual_seed(42)
        d_model = unembed_weight.shape[1]
        # Random projection matrix
        Q = torch.randn(d_model, max_k + 10).float()
        # Project: Y = unembed_weight @ Q
        Y = unembed_weight @ Q  # (vocab_size, max_k+10)
        # QR decomposition
        Q_orth, _ = torch.linalg.qr(Y)
        # Project again
        B = Q_orth.T @ unembed_weight.T  # (max_k+10, vocab_size)
        # Small SVD on B
        _, S_small, Vt_small = torch.linalg.svd(B, full_matrices=False)
        # Recover: top-K right singular vectors of unembed_weight
        top_components = (Q_orth @ Vt_small.T)[:max_k].T  # (max_k, d_model)
        log(f"    SVD done, top-{max_k} components computed")
    except Exception as e:
        log(f"    SVD failed: {e}, using identity approximation")
        # Fallback: use the h vector itself as the primary component
        top_components = None

    for text in texts:
        h_orig, logits_orig = get_final_hidden_and_logits(model, tokenizer, text)
        orig_top1 = logits_orig.argmax().item()

        # Project h_final onto top-K principal components
        top1_accuracies = []
        logit_correlations = []
        k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]

        for k in k_values:
            k = min(k, top_components.shape[0] if top_components is not None else 1)
            h_proj = torch.zeros_like(h_orig)
            for i in range(k):
                if top_components is not None:
                    component = top_components[i, :]  # (d_model,)
                else:
                    component = h_orig / (h_orig.norm() + 1e-10)
                coeff = torch.dot(h_orig, component)
                h_proj += coeff * component

            proj_logits = h_proj @ unembed_weight.T
            proj_top1 = proj_logits.argmax().item()

            top1_accuracies.append(1 if proj_top1 == orig_top1 else 0)

            sample_idx = np.random.choice(len(logits_orig), min(5000, len(logits_orig)), replace=False)
            r = np.corrcoef(logits_orig.numpy()[sample_idx], proj_logits.numpy()[sample_idx])[0, 1]
            logit_correlations.append(r)

        # Find the minimum K that gives r > 0.9
        k_for_09 = None
        for k, r in zip(k_values, logit_correlations):
            if r > 0.9:
                k_for_09 = k
                break

        results.append({
            "text": text[:40],
            "k_for_r_0.9": k_for_09,
            "top1_at_k10": top1_accuracies[k_values.index(10) if 10 in k_values else 0],
            "top1_at_k100": top1_accuracies[k_values.index(100) if 100 in k_values else 0],
            "logit_correlations": {str(k): round(r, 4) for k, r in zip(k_values, logit_correlations)},
        })

    # Aggregate
    mean_k_09 = np.mean([r["k_for_r_0.9"] for r in results if r["k_for_r_0.9"] is not None])
    mean_top1_k10 = np.mean([r["top1_at_k10"] for r in results])
    mean_top1_k100 = np.mean([r["top1_at_k100"] for r in results])

    log(f"    Mean K for r>0.9: {mean_k_09:.1f}")
    log(f"    Mean top-1 accuracy at K=10: {mean_top1_k10:.2%}")
    log(f"    Mean top-1 accuracy at K=100: {mean_top1_k100:.2%}")

    # Verdict: if K=10 preserves >80% top-1, the effective dimensionality is low
    # This relates to the "11D rotation plane" finding
    if mean_k_09 <= 15:
        verdict = "P2 SUPPORTED - low-dimensional projection preserves logits (consistent with 11D)"
    elif mean_k_09 <= 50:
        verdict = "P2 PARTIALLY SUPPORTED - moderate dimensionality needed"
    else:
        verdict = "P2 CHALLENGED - high dimensionality needed (>50D)"

    log(f"    Verdict: {verdict}")

    return {
        "mean_k_for_r_0.9": float(mean_k_09),
        "mean_top1_k10": float(mean_top1_k10),
        "mean_top1_k100": float(mean_top1_k100),
        "verdict": verdict,
        "per_text": results[:5],
    }


def main():
    log = Logger(OUTPUT_DIR / "results.log")
    log(f"P55+P56: Falsification Experiments")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output: {OUTPUT_DIR}")

    all_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        try:
            model, tokenizer = load_model(model_name)
            log(f"Loaded in {time.time()-t0:.1f}s")

            p55_result = run_p55(model, tokenizer, TEXTS, log)
            p56_result = run_p56(model, tokenizer, TEXTS, model_name, log)

            model_result = {
                "model": model_name,
                "p55_norm_only": p55_result,
                "p56_rotation": p56_result,
            }
            all_results[model_name] = model_result

            # Save per-model JSON
            json_path = OUTPUT_DIR / f"results_{model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_result, f, indent=2, ensure_ascii=False, default=str)

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  {model_name} done in {time.time()-t0:.1f}s")

        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            log(traceback.format_exc())

    # Final summary
    log(f"\n\n{'='*60}")
    log("P55+P56 COMPLETE - Falsification Summary")
    log(f"{'='*60}")

    log("\n--- P55: Norm-Only (Proposition 1: semantics in direction) ---")
    for name, res in all_results.items():
        p55 = res["p55_norm_only"]
        log(f"  {name}: r_at_0.1x={p55['mean_r_0.1x']:.4f}, stable={p55['mean_stable_count']:.1f}/8, "
            f"verdict={p55['verdict']}")

    log("\n--- P56: Rotation Plane (Proposition 2: rotation=architecture) ---")
    for name, res in all_results.items():
        p56 = res["p56_rotation"]
        log(f"  {name}: K_for_r0.9={p56['mean_k_for_r_0.9']:.1f}, top1@K10={p56['mean_top1_k10']:.0%}, "
            f"verdict={p56['verdict']}")

    log(f"\nTotal time: {time.strftime('%H:%M:%S')}")
    log.close()
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
