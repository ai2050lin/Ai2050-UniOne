"""
P61+P63: Information Capacity Analysis + Minimal Sufficient Primitives
Phase III - Primitives (基元化)

P61: Information capacity analysis for each primitive
  - A: d-dim direction -> log2(d) bits per dimension
  - B: k-dim subspace -> k * log2(d/k) bits
  - C: L-layer trajectory -> continuous
  - D: N x N attention -> N^2 * log2(N) bits
  - E: M gates -> M bits

P63: Minimal sufficient primitive
  - Progressive dimension reduction: what % of dimensions needed for >95% top-1?
  - Sparse coding analysis: how sparse are the delta-h vectors?

Models: Qwen3-4B, DeepSeek-7B, Gemma4-4B-it, GLM4-9B-chat (sequential)
"""

import torch
import numpy as np
from pathlib import Path as _Path
import time
import os

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEST_TEXTS = [
    "The cat sat on the mat.",
    "In quantum mechanics, particles exhibit wave-particle duality.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris.",
    "Attention is all you need for transformer models.",
    "E = mc squared represents mass-energy equivalence.",
    "She walked slowly through the dark forest.",
    "The stock market crashed in 1929.",
    "Neural networks learn by backpropagation of errors.",
    "The quick brown fox jumps over the lazy dog.",
]


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg):
        safe = msg.encode('utf-8', errors='replace').decode('utf-8')
        print(safe)
        self.f.write(safe + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load_model(model_name):
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    logits = outputs.logits
    last_hs = [hs[0, -1, :].float().cpu() for hs in hidden_states]
    last_logits = logits[0, -1, :].float().cpu()
    return last_hs, last_logits


# ============================================================
# P61: Information Capacity Analysis
# ============================================================

def analyze_information_capacity(model_name, model, tokenizer, log):
    """Analyze information capacity of each encoding primitive."""
    log(f"\n{'='*60}")
    log(f"P61 Information Capacity: {model_name}")
    log(f"{'='*60}")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()

    # Collect data
    all_delta_h = []  # per-layer delta_h for each text
    all_final_h = []
    n_layers = None

    for text in TEST_TEXTS:
        hs_list, logits = get_hidden_states(model, tokenizer, text)
        n_layers = len(hs_list)
        all_final_h.append(hs_list[-1])

        for i in range(1, n_layers):
            delta = hs_list[i] - hs_list[i - 1]
            all_delta_h.append(delta)

    d_model = all_final_h[0].shape[0]
    seq_len = max(len(tokenizer(t)["input_ids"]) for t in TEST_TEXTS)

    # ---- A: Direction encoding capacity ----
    log(f"\n  --- A: Direction encoding ---")
    # Stack all final h vectors
    H = torch.stack(all_final_h)  # (n_texts, d_model)
    # PCA to find effective dimensionality
    mean_h = H.mean(0)
    centered = H - mean_h
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)

    # Effective rank: number of singular values > 1% of max
    effective_rank = (S > 0.01 * S[0]).sum().item()
    # Cumulative variance explained
    total_var = (S ** 2).sum().item()
    cumvar = torch.cumsum(S ** 2, 0) / total_var

    # Bits: effective_rank * log2(d_model)
    bits_A = effective_rank * np.log2(d_model)
    log(f"    d_model = {d_model}")
    log(f"    Effective rank (SV > 1% max) = {effective_rank}/{d_model}")
    log(f"    95% variance in top {torch.searchsorted(cumvar, 0.95).item() + 1} components")
    log(f"    99% variance in top {torch.searchsorted(cumvar, 0.99).item() + 1} components")
    log(f"    Information capacity: ~{bits_A:.0f} bits")

    # ---- B: Subspace encoding ----
    log(f"\n  --- B: Subspace encoding ---")
    # Find minimum k such that k-dim projection preserves >90% variance
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        k = torch.searchsorted(cumvar, threshold).item() + 1
        bits_B = k * np.log2(d_model / max(k, 1))
        log(f"    {threshold:.0%} variance: k={k} dims, ~{bits_B:.0f} bits")

    # ---- C: Trajectory encoding ----
    log(f"\n  --- C: Layer trajectory encoding ---")
    # Stack all delta_h vectors: (n_texts * (n_layers-1), d_model)
    DH = torch.stack(all_delta_h)
    delta_mean = DH.mean(0)
    delta_centered = DH - delta_mean
    _, S_delta, Vt_delta = torch.linalg.svd(delta_centered, full_matrices=False)

    delta_rank = (S_delta > 0.01 * S_delta[0]).sum().item()
    total_delta_var = (S_delta ** 2).sum().item()
    delta_cumvar = torch.cumsum(S_delta ** 2, 0) / total_delta_var

    # Trajectory bits = effective_rank * n_layers * bits_per_dim
    bits_C = delta_rank * n_layers * np.log2(d_model)
    log(f"    n_layers = {n_layers}")
    log(f"    Delta-h effective rank = {delta_rank}/{d_model}")
    log(f"    95% variance in top {torch.searchsorted(delta_cumvar, 0.95).item() + 1} components")
    log(f"    Information capacity: ~{bits_C:.0f} bits")

    # ---- D: Attention pattern encoding ----
    log(f"\n  --- D: Attention pattern encoding ---")
    # N x N attention matrix -> N^2 * log2(N) bits per layer
    bits_D_per_layer = seq_len ** 2 * np.log2(seq_len)
    bits_D_total = bits_D_per_layer * n_layers
    log(f"    seq_len = {seq_len}")
    log(f"    Per layer: {seq_len}^2 x log2({seq_len}) = ~{bits_D_per_layer:.0f} bits")
    log(f"    Total ({n_layers} layers): ~{bits_D_total:.0f} bits")

    # ---- E: Gate/modulation encoding ----
    log(f"\n  --- E: Gate/modulation encoding ---")
    # Sparsity analysis of delta-h: what fraction of dimensions have significant values?
    sparsities = []
    for dh in all_delta_h:
        # Fraction of dimensions with |value| > 10% of max
        threshold = 0.1 * dh.abs().max()
        sparse = (dh.abs() > threshold).float().mean().item()
        sparsities.append(sparse)

    avg_sparsity = np.mean(sparsities)
    # M active gates = sparsity * d_model
    M = int(avg_sparsity * d_model)
    bits_E = M  # binary gates
    log(f"    Average sparsity (|v| > 10% max): {avg_sparsity:.1%}")
    log(f"    Active gates M = {M}/{d_model}")
    log(f"    Binary gate capacity: ~{bits_E} bits")

    # Summary
    log(f"\n  --- Information Capacity Summary ---")
    log(f"  {'Primitive':<12} {'Capacity (bits)':>16} {'Normalized':>12}")
    log(f"  {'-'*42}")
    max_bits = max(bits_A, bits_B, bits_C, bits_E)
    log(f"  {'A_direction':<12} {bits_A:>15.0f} {bits_A/max_bits:>11.2f}")
    log(f"  {'B_subspace':<12} {bits_B:>15.0f} {bits_B/max_bits:>11.2f}")
    log(f"  {'C_trajectory':<12} {bits_C:>15.0f} {bits_C/max_bits:>11.2f}")
    log(f"  {'D_attention':<12} {bits_D_total:>15.0f} {bits_D_total/max_bits:>11.2f}")
    log(f"  {'E_gate':<12} {bits_E:>15.0f} {bits_E/max_bits:>11.2f}")

    return {
        "d_model": d_model, "n_layers": n_layers, "seq_len": seq_len,
        "effective_rank": effective_rank, "delta_rank": delta_rank,
        "sparsity": avg_sparsity, "active_gates": M,
        "bits": {"A": bits_A, "B": bits_B, "C": bits_C, "D": bits_D_total, "E": bits_E},
    }


# ============================================================
# P63: Minimal Sufficient Primitive
# ============================================================

def analyze_minimal_sufficient(model_name, model, tokenizer, log):
    """Find the minimum dimensionality needed to preserve >95% top-1 accuracy."""
    log(f"\n{'='*60}")
    log(f"P63 Minimal Sufficient Primitive: {model_name}")
    log(f"{'='*60}")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()

    d_model = unembed_weight.shape[1]

    # Collect final hidden states
    all_h = []
    all_logits = []
    for text in TEST_TEXTS:
        hs_list, logits = get_hidden_states(model, tokenizer, text)
        all_h.append(hs_list[-1])
        all_logits.append(logits)

    # PCA on final hidden states
    H = torch.stack(all_h)
    mean_h = H.mean(0)
    centered = H - mean_h
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    components = Vt  # (d_model, d_model)

    # Progressive dimension reduction
    test_fracs = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    log(f"\n  Progressive PCA Dimension Reduction:")
    log(f"  {'Dims':>8} {'% of d':>8} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Pearson r':>10} {'Cos sim':>8}")
    log(f"  {'-'*62}")

    results = []
    for frac in test_fracs:
        k = max(1, int(frac * d_model))
        k = min(k, d_model)

        # Project all h onto top-k PCA components
        top1_matches = 0
        top3_matches = 0
        top5_matches = 0
        all_r = []
        all_cos = []

        for i, (h, logits) in enumerate(zip(all_h, all_logits)):
            # Project
            h_centered = h - mean_h
            coords = h_centered @ components[:k].T  # (k,)
            h_proj = mean_h + components[:k].T @ coords  # (d_model,)

            proj_logits = h_proj @ unembed_weight.T
            orig_top1 = logits.argmax().item()
            proj_top1 = proj_logits.argmax().item()

            # Top-k accuracy
            orig_topk = logits.topk(5).indices.tolist()
            proj_topk = proj_logits.topk(5).indices.tolist()

            if proj_top1 == orig_top1:
                top1_matches += 1
            if proj_top1 in orig_topk[:3]:
                top3_matches += 1
            if proj_top1 in orig_topk[:5]:
                top5_matches += 1

            # Pearson r (sample)
            n = len(logits)
            idx = np.random.choice(n, min(3000, n), replace=False)
            r = np.corrcoef(logits.numpy()[idx], proj_logits.numpy()[idx])[0, 1]
            all_r.append(r)

            cos = torch.nn.functional.cosine_similarity(
                logits.unsqueeze(0), proj_logits.unsqueeze(0)
            ).item()
            all_cos.append(cos)

        avg_top1 = top1_matches / len(TEST_TEXTS)
        avg_top3 = top3_matches / len(TEST_TEXTS)
        avg_top5 = top5_matches / len(TEST_TEXTS)
        avg_r = np.mean(all_r)
        avg_cos = np.mean(all_cos)

        pct = frac * 100
        log(f"  {k:>7} {pct:>7.0f}% {avg_top1:>7.0%} {avg_top3:>7.0%} {avg_top5:>7.0%} {avg_r:>10.4f} {avg_cos:>8.4f}")
        results.append((k, frac, avg_top1, avg_r, avg_cos))

    # Find the minimum fraction for >95% top-1
    log(f"\n  --- Key thresholds ---")
    for threshold_top1 in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
        for k, frac, top1, r, cos in results:
            if top1 >= threshold_top1:
                log(f"    {threshold_top1:.0%} top-1 at {frac:.0%} dims (k={k}), r={r:.4f}")
                break
        else:
            log(f"    {threshold_top1:.0%} top-1 NOT reached at any fraction")

    # ---- Sparse coding analysis ----
    log(f"\n  --- Sparse Coding Analysis ---")
    # How sparse are the delta-h vectors?
    all_delta_h = []
    n_layers_check = None
    for text in TEST_TEXTS[:3]:
        hs_list, _ = get_hidden_states(model, tokenizer, text)
        n_layers_check = len(hs_list)
        for i in range(1, n_layers_check):
            delta = hs_list[i] - hs_list[i - 1]
            all_delta_h.append(delta)

    # For each threshold, compute sparsity and measure impact
    for sparsity_threshold in [0.05, 0.10, 0.20, 0.30, 0.50]:
        avg_sparse_pct = 0
        for dh in all_delta_h:
            max_val = dh.abs().max()
            kept = (dh.abs() >= sparsity_threshold * max_val).float().mean().item()
            avg_sparse_pct += kept
        avg_sparse_pct /= len(all_delta_h)
        log(f"    Threshold {sparsity_threshold:.0%}: avg {avg_sparse_pct:.1%} dims retained")

    # ---- Per-layer minimal sufficient dimensions ----
    log(f"\n  --- Per-Layer Delta-h PCA ---")
    for text in TEST_TEXTS[:1]:
        hs_list, orig_logits = get_hidden_states(model, tokenizer, text)
        log(f"    Text: '{text[:50]}...'")

        for layer_idx in [1, n_layers_check // 3, 2 * n_layers_check // 3, n_layers_check - 1]:
            if layer_idx >= len(hs_list):
                continue
            delta = hs_list[layer_idx] - hs_list[layer_idx - 1]

            # How many dimensions explain 90/95/99% of delta's energy?
            delta_sorted = delta.abs().sort(descending=True).values
            total_energy = (delta ** 2).sum().item()
            cum_energy = torch.cumsum(delta_sorted ** 2, 0) / total_energy

            d90 = (cum_energy < 0.90).sum().item() + 1
            d95 = (cum_energy < 0.95).sum().item() + 1
            d99 = (cum_energy < 0.99).sum().item() + 1

            log(f"    L{layer_idx:>2}: d90={d90}, d95={d95}, d99={d99} / {d_model}")

    return results


# ============================================================
# Cross-model comparison for P61+P63
# ============================================================

def run_all_models(log):
    """Run P61+P63 on all 4 models sequentially."""
    all_p61 = {}
    all_p63 = {}

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        t0 = time.time()
        try:
            log(f"\n\n>>> Loading {model_name}...")
            model, tokenizer = load_model(model_name)

            p61_results = analyze_information_capacity(model_name, model, tokenizer, log)
            all_p61[model_name] = p61_results

            p63_results = analyze_minimal_sufficient(model_name, model, tokenizer, log)
            all_p63[model_name] = p63_results

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  ERROR with {model_name}: {e}")
            import traceback
            log(traceback.format_exc())

        elapsed = time.time() - t0
        log(f"\n  {model_name} done in {elapsed:.1f}s")

    # Cross-model summary
    log(f"\n\n{'='*60}")
    log("CROSS-MODEL SUMMARY")
    log(f"{'='*60}")

    # P61 comparison
    log(f"\n  --- P61: Information Capacity (bits) ---")
    log(f"  {'Model':<12} {'A_dir':>10} {'B_sub':>10} {'C_traj':>10} {'D_att':>12} {'E_gate':>10} {'d_model':>8} {'eff_rank':>9}")
    log(f"  {'-'*82}")
    for mn, d in all_p61.items():
        b = d["bits"]
        log(f"  {mn:<12} {b['A']:>9.0f} {b['B']:>9.0f} {b['C']:>9.0f} {b['D']:>11.0f} {b['E']:>9.0f} {d['d_model']:>8} {d['effective_rank']:>9}")

    # P63 comparison: minimum dims for 80% top-1
    log(f"\n  --- P63: Minimum Dims for Accuracy ---")
    log(f"  {'Model':<12} {'80% top1':>10} {'90% top1':>10} {'95% top1':>10} {'100% top1':>11}")
    log(f"  {'-'*55}")
    for mn, res_list in all_p63.items():
        t80 = next((f"{k} ({f:.0%})" for k, f, t, r, c in res_list if t >= 0.8), "N/A")
        t90 = next((f"{k} ({f:.0%})" for k, f, t, r, c in res_list if t >= 0.9), "N/A")
        t95 = next((f"{k} ({f:.0%})" for k, f, t, r, c in res_list if t >= 0.95), "N/A")
        t100 = next((f"{k} ({f:.0%})" for k, f, t, r, c in res_list if t >= 1.0), "N/A")
        log(f"  {mn:<12} {t80:>10} {t90:>10} {t95:>10} {t100:>11}")

    # Final conclusion
    log(f"\n{'='*60}")
    log("CONCLUSION:")
    for mn in all_p61:
        d = all_p61[mn]
        log(f"  {mn}: effective_rank={d['effective_rank']}, "
            f"sparsity={d['sparsity']:.1%}, active_gates={d['active_gates']}")
    log(f"  The primitive with the most appropriate information capacity")
    log(f"  (sufficient for discrimination but not wastefully high)")
    log(f"  is likely the true encoding primitive.")
    log(f"{'='*60}")

    return all_p61, all_p63


def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_dir = _Path(f"tests/glm5_temp/stage704_info_capacity_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.log"
    log = Logger(str(log_path))

    log("=" * 60)
    log("P61+P63: Information Capacity + Minimal Sufficient Primitive")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    run_all_models(log)
    log.close()
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
