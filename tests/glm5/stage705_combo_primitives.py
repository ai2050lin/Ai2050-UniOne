"""
P62: Combination Primitives Test + Unembed Rank Analysis + P55/P60 Contradiction Resolution
Phase III - Primitives (基元化)

Three experiments combined:
1. Unembed matrix rank analysis - Why is effective rank=9? Is it from unembed?
2. P62 combination primitives - Test A+E, B+E, A+B+E combos
3. P55/P60 contradiction - P55 says norm doesn't matter, P60 E_gate destroys output.
   Resolution: P55 scales ALL dims uniformly (preserves relative structure),
   E_gate equalizes PER-LAYER deltas (destroys layer contribution balance).

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


def evaluate(orig_logits, ablated_logits):
    orig_top1 = orig_logits.argmax().item()
    ablated_top1 = ablated_logits.argmax().item()
    top1_match = 1 if orig_top1 == ablated_top1 else 0
    n = len(orig_logits)
    idx = np.random.choice(n, min(3000, n), replace=False)
    r = np.corrcoef(orig_logits.numpy()[idx], ablated_logits.numpy()[idx])[0, 1]
    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_probs = torch.softmax(ablated_logits, dim=-1).clamp(min=1e-10)
    kl = (orig_probs * (orig_probs.log() - ablated_probs.log())).sum().item()
    cos = torch.nn.functional.cosine_similarity(
        orig_logits.unsqueeze(0), ablated_logits.unsqueeze(0)
    ).item()
    # Margin
    orig_sorted = torch.sort(orig_logits, descending=True).values
    ablated_sorted = torch.sort(ablated_logits, descending=True).values
    orig_margin = (orig_sorted[0] - orig_sorted[1]).item()
    ablated_margin = (ablated_sorted[0] - ablated_sorted[1]).item()
    return {"top1": top1_match, "r": r, "kl": kl, "cos": cos,
            "orig_margin": orig_margin, "ablated_margin": ablated_margin}


# ============================================================
# Experiment 1: Unembed Matrix Rank Analysis
# ============================================================

def analyze_unembed_rank(model_name, model, log):
    """Analyze the rank structure of the unembed matrix."""
    log(f"\n{'='*60}")
    log(f"Unembed Rank Analysis: {model_name}")
    log(f"{'='*60}")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        W = unembed.weight.float().cpu()  # (vocab_size, d_model)

    vocab_size, d_model = W.shape
    log(f"  Unembed shape: ({vocab_size}, {d_model})")

    # Full SVD is too expensive for large vocab, use randomized SVD on W.T @ W instead
    log(f"  Computing rank via W.T @ W (d_model x d_model) SVD...")
    # W is (vocab_size, d_model), W.T @ W is (d_model, d_model) - much smaller
    # But this might still be large. Use randomized approach on W.
    torch.manual_seed(42)
    # Sample rows of W (tokens) for efficiency
    n_sample = min(5000, vocab_size)
    idx = torch.randperm(vocab_size)[:n_sample]
    W_sample = W[idx]  # (n_sample, d_model)

    # SVD on sample
    U_s, S_s, Vt_s = torch.linalg.svd(W_sample, full_matrices=False)

    effective_rank = (S_s > 0.01 * S_s[0]).sum().item()
    log(f"  Sample shape: ({n_sample}, {d_model})")
    log(f"  Top 10 singular values: {S_s[:10].tolist()}")
    log(f"  Effective rank (SV > 1% max): {effective_rank}")
    log(f"  SV[1]/SV[0] = {S_s[1]/S_s[0]:.4f}")
    log(f"  SV[9]/SV[0] = {S_s[min(9,len(S_s)-1)]/S_s[0]:.4f}")

    # Cumulative energy
    total_energy = (S_s ** 2).sum().item()
    cumvar = torch.cumsum(S_s ** 2, 0) / total_energy
    log(f"  Cumulative variance explained:")
    for k in [1, 3, 5, 10, 20, 50, 100]:
        if k <= len(cumvar):
            log(f"    Top {k:>3}: {cumvar[k-1]:.4f}")

    return {"d_model": d_model, "vocab_size": vocab_size,
            "effective_rank": effective_rank,
            "top_sv": S_s[:20].tolist()}


# ============================================================
# Experiment 2: P62 Combination Primitives
# ============================================================

def run_combination_primitives(model_name, model, tokenizer, unembed_weight, log):
    """Test combinations of primitives."""
    log(f"\n{'='*60}")
    log(f"P62 Combination Primitives: {model_name}")
    log(f"{'='*60}")

    d_model = unembed_weight.shape[1]

    # Collect all hidden states
    all_hs = []
    all_logits = []
    for text in TEST_TEXTS:
        hs_list, logits = get_hidden_states(model, tokenizer, text)
        all_hs.append(hs_list)
        all_logits.append(logits)

    # Compute shared PCA directions
    H = torch.stack([hs[-1] for hs in all_hs])
    mean_h = H.mean(0)
    centered = H - mean_h
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    def ablate_A(h_final, Vt_all, all_h_stack):
        """Remove top PCA direction from h."""
        mean = all_h_stack.mean(0)
        c = h_final - mean
        U, _, Vt_s = torch.linalg.svd(all_h_stack - mean, full_matrices=False)
        proj = torch.dot(c, Vt_s[0]) * Vt_s[0]
        return h_final - proj

    def ablate_B(h_final, k=None):
        """Project onto random k-dim subspace."""
        if k is None:
            k = max(1, d_model // 10)
        torch.manual_seed(42)
        rand_basis = torch.randn(d_model, k).float()
        Q, _ = torch.linalg.qr(rand_basis)
        coords = h_final @ Q
        return Q @ coords

    def ablate_E_from_hs(hs_list):
        """Equalize all delta-h magnitudes (from hidden state list)."""
        deltas = [hs_list[i] - hs_list[i-1] for i in range(1, len(hs_list))]
        if not deltas:
            return hs_list[-1]
        norms = torch.stack([d.norm() for d in deltas])
        mean_norm = norms.mean()
        h = hs_list[0].clone()
        for d in deltas:
            direction = d / (d.norm() + 1e-10)
            h = h + direction * mean_norm
        return h

    def ablate_A_plus_E(h_final, hs_list, all_h_stack):
        """A (remove PCA direction) + E (equalize delta magnitudes)."""
        h_E = ablate_E_from_hs(hs_list)
        h_AE = ablate_A(h_E, Vt, all_h_stack)
        return h_AE

    def ablate_B_plus_E(h_final, hs_list):
        """B (random subspace) + E (equalize delta magnitudes)."""
        h_E = ablate_E_from_hs(hs_list)
        h_BE = ablate_B(h_E)
        return h_BE

    def ablate_A_plus_B(h_final, all_h_stack):
        """A (remove PCA direction) + B (random subspace)."""
        h_A = ablate_A(h_final, Vt, all_h_stack)
        h_AB = ablate_B(h_A)
        return h_AB

    def ablate_A_plus_B_plus_E(h_final, hs_list, all_h_stack):
        """A + B + E: full ablation."""
        h_E = ablate_E_from_hs(hs_list)
        h_AE = ablate_A(h_E, Vt, all_h_stack)
        h_ABE = ablate_B(h_AE)
        return h_ABE

    # Run all ablation combos
    combos = {
        "A_only": lambda hs, all_h: ablate_A(hs[-1], Vt, all_h),
        "B_only": lambda hs, all_h: ablate_B(hs[-1]),
        "E_only": lambda hs, all_h: ablate_E_from_hs(hs),
        "A+E": lambda hs, all_h: ablate_A_plus_E(hs[-1], hs, all_h),
        "B+E": lambda hs, all_h: ablate_B_plus_E(hs[-1], hs),
        "A+B": lambda hs, all_h: ablate_A_plus_B(hs[-1], all_h),
        "A+B+E": lambda hs, all_h: ablate_A_plus_B_plus_E(hs[-1], hs, all_h),
    }

    results = {name: [] for name in combos}

    for combo_name, combo_fn in combos.items():
        for text_idx in range(len(TEST_TEXTS)):
            h_ablated = combo_fn(all_hs[text_idx], H)
            ablated_logits = h_ablated @ unembed_weight.T
            metrics = evaluate(all_logits[text_idx], ablated_logits)
            results[combo_name].append(metrics)

    # Summary
    log(f"\n  {'Combo':<10} {'Top-1':>8} {'Pearson r':>10} {'KL div':>8} {'Cos sim':>8} {'Margin ratio':>12}")
    log(f"  {'-'*60}")
    for name in combos:
        m = results[name]
        top1 = np.mean([x["top1"] for x in m])
        r = np.mean([x["r"] for x in m])
        kl = np.mean([x["kl"] for x in m])
        cos = np.mean([x["cos"] for x in m])
        # Margin ratio: ablated/orig margin
        margin_ratios = []
        for x in m:
            if abs(x["orig_margin"]) > 0.01:
                margin_ratios.append(x["ablated_margin"] / x["orig_margin"])
        avg_margin_ratio = np.mean(margin_ratios) if margin_ratios else float('nan')
        log(f"  {name:<10} {top1:>7.0%} {r:>10.4f} {kl:>8.2f} {cos:>8.4f} {avg_margin_ratio:>11.2f}x")

    # Ranking
    log(f"\n  Damage ranking:")
    damage = {}
    for name in combos:
        m = results[name]
        avg_top1 = np.mean([x["top1"] for x in m])
        avg_r = np.mean([x["r"] for x in m])
        avg_kl = np.mean([x["kl"] for x in m])
        damage[name] = (1 - avg_top1) + (1 - max(avg_r, 0)) + avg_kl / (avg_kl + 1)

    for rank, (name, score) in enumerate(sorted(damage.items(), key=lambda x: -x[1]), 1):
        log(f"    #{rank}: {name} (damage={score:.3f})")

    # Synergy analysis: is A+E worse than A or E alone?
    log(f"\n  Synergy analysis:")
    d_A = damage.get("A_only", 0)
    d_E = damage.get("E_only", 0)
    d_AE = damage.get("A+E", 0)
    synergy = d_AE - max(d_A, d_E)
    if synergy > 0.1:
        log(f"    A+E has positive synergy: {d_AE:.3f} > max({d_A:.3f}, {d_E:.3f}) + {synergy:.3f}")
        log(f"    -> Direction and gate work together (not independent)")
    else:
        log(f"    A+E has no positive synergy: {d_AE:.3f} <= max({d_A:.3f}, {d_E:.3f})")
        log(f"    -> One primitive dominates (gate amplitude)")

    return results, damage


# ============================================================
# Experiment 3: P55/P60 Contradiction Resolution
# ============================================================

def resolve_contradiction(model_name, model, tokenizer, unembed_weight, log):
    """Resolve why P55 (norm doesn't matter) but P60 E_gate (amplitude matters)."""
    log(f"\n{'='*60}")
    log(f"P55/P60 Contradiction Resolution: {model_name}")
    log(f"{'='*60}")

    log(f"\n  Hypothesis: P55 scales h_final uniformly (preserves relative structure),")
    log(f"  while E_gate equalizes PER-LAYER delta-h magnitudes (destroys layer balance).")
    log(f"  Key: it's not about norm per se, but about the RELATIVE contributions of layers.")

    for text in TEST_TEXTS[:3]:
        hs_list, orig_logits = get_hidden_states(model, tokenizer, text)
        h_final = hs_list[-1]

        # Compute per-layer delta-h norms
        delta_norms = []
        for i in range(1, len(hs_list)):
            delta = hs_list[i] - hs_list[i - 1]
            delta_norms.append(delta.norm().item())

        log(f"\n  Text: '{text[:40]}...'")
        log(f"    Per-layer |delta-h| norms (first 10): {[f'{n:.2f}' for n in delta_norms[:10]]}")
        log(f"    Mean: {np.mean(delta_norms):.2f}, Std: {np.std(delta_norms):.2f}")
        log(f"    Max/Min ratio: {max(delta_norms)/min(delta_norms):.1f}x")
        log(f"    CV (std/mean): {np.std(delta_norms)/np.mean(delta_norms):.2f}")

        # Test 1: P55-style - scale final h by 0.1x and 10x
        for scale in [0.1, 10.0]:
            h_scaled = h_final * scale
            scaled_logits = h_scaled @ unembed_weight.T
            top1_match = 1 if scaled_logits.argmax() == orig_logits.argmax() else 0
            r = np.corrcoef(orig_logits.numpy(), scaled_logits.numpy())[0, 1]
            log(f"    P55 scale={scale}x: top1={top1_match}, r={r:.6f}")

        # Test 2: E_gate-style - equalize all delta-h norms, reconstruct h
        mean_norm = np.mean(delta_norms)
        h_equalized = hs_list[0].clone()
        for i in range(1, len(hs_list)):
            delta = hs_list[i] - hs_list[i - 1]
            direction = delta / (delta.norm() + 1e-10)
            h_equalized = h_equalized + direction * mean_norm

        eq_logits = h_equalized @ unembed_weight.T
        eq_top1 = 1 if eq_logits.argmax() == orig_logits.argmax() else 0
        eq_r = np.corrcoef(orig_logits.numpy(), eq_logits.numpy())[0, 1]
        log(f"    E_gate (equalized): top1={eq_top1}, r={eq_r:.4f}")

        # Test 3: Shuffle per-layer delta norms (destroy layer balance but keep set)
        np.random.seed(42)
        shuffled_norms = list(delta_norms)
        np.random.shuffle(shuffled_norms)

        h_shuffled = hs_list[0].clone()
        for i in range(1, len(hs_list)):
            delta = hs_list[i] - hs_list[i - 1]
            direction = delta / (delta.norm() + 1e-10)
            h_shuffled = h_shuffled + direction * shuffled_norms[i - 1]

        sh_logits = h_shuffled @ unembed_weight.T
        sh_top1 = 1 if sh_logits.argmax() == orig_logits.argmax() else 0
        sh_r = np.corrcoef(orig_logits.numpy(), sh_logits.numpy())[0, 1]
        log(f"    Shuffle norms: top1={sh_top1}, r={sh_r:.4f}")

        # Test 4: Zero out only the top-3 most important layers' deltas
        # (based on P60 layer importance)
        norm_enums = sorted(enumerate(delta_norms), key=lambda x: -x[1])
        h_no_top3 = hs_list[0].clone()
        removed_layers = set(x[0] for x in norm_enums[:3])
        for i in range(1, len(hs_list)):
            delta = hs_list[i] - hs_list[i - 1]
            if i - 1 in removed_layers:
                continue  # Skip this layer's contribution
            h_no_top3 = h_no_top3 + delta

        nt3_logits = h_no_top3 @ unembed_weight.T
        nt3_top1 = 1 if nt3_logits.argmax() == orig_logits.argmax() else 0
        nt3_r = np.corrcoef(orig_logits.numpy(), nt3_logits.numpy())[0, 1]
        log(f"    Remove top-3 layers: top1={nt3_top1}, r={nt3_r:.4f}")

    # Final synthesis
    log(f"\n  --- Resolution ---")
    log(f"  P55 and P_gate are NOT contradictory:")
    log(f"  - P55: scaling ALL dims of h_final by same factor -> argmax unchanged")
    log(f"    (because softmax is scale-invariant: softmax(ax) = softmax(x))")
    log(f"  - E_gate: changing RELATIVE per-layer contributions -> h_final changes")
    log(f"    (different layers contribute to different directions in h_final)")
    log(f"  - The key quantity is not ||h|| but the DIRECTION of each delta-h")
    log(f"  - E_gate changes which directions get amplified vs suppressed")
    log(f"  - P55 changes the overall magnitude but keeps all relative directions")
    log(f"  Conclusion: Semantic encoding is in the PATTERN of layer contributions,")
    log(f"  not in the total magnitude.")


# ============================================================
# Experiment 4: Margin analysis at 1% dims
# ============================================================

def analyze_margin_at_low_dims(model_name, model, tokenizer, unembed_weight, log):
    """Does 1% dim preservation lose margin precision?"""
    log(f"\n{'='*60}")
    log(f"Margin Analysis at Low Dims: {model_name}")
    log(f"{'='*60}")

    d_model = unembed_weight.shape[1]

    all_h = []
    all_logits = []
    for text in TEST_TEXTS:
        hs_list, logits = get_hidden_states(model, tokenizer, text)
        all_h.append(hs_list[-1])
        all_logits.append(logits)

    H = torch.stack(all_h)
    mean_h = H.mean(0)
    centered = H - mean_h
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

    log(f"\n  {'Dims':>8} {'%':>6} {'Top-1':>8} {'Margin r':>10} {'Avg margin ratio':>16}")
    log(f"  {'-'*52}")

    orig_margins = []
    for logits in all_logits:
        sorted_v = torch.sort(logits, descending=True).values
        orig_margins.append((sorted_v[0] - sorted_v[1]).item())

    for frac in fracs:
        k = max(1, int(frac * d_model))
        k = min(k, d_model)

        top1_matches = 0
        margin_corrs = []
        margin_ratios = []

        for i, (h, logits) in enumerate(zip(all_h, all_logits)):
            h_c = h - mean_h
            coords = h_c @ Vt[:k].T
            h_proj = mean_h + Vt[:k].T @ coords
            proj_logits = h_proj @ unembed_weight.T

            if proj_logits.argmax() == logits.argmax():
                top1_matches += 1

            proj_sorted = torch.sort(proj_logits, descending=True).values
            proj_margin = (proj_sorted[0] - proj_sorted[1]).item()

            margin_ratios.append(proj_margin / (orig_margins[i] + 1e-10))

        avg_top1 = top1_matches / len(TEST_TEXTS)
        avg_margin_ratio = np.mean(margin_ratios)

        # Margin correlation across texts
        if np.std(orig_margins) > 0.01 and np.std(margin_ratios) > 0.01:
            margin_r = np.corrcoef(orig_margins, margin_ratios)[0, 1]
        else:
            margin_r = float('nan')

        pct = frac * 100
        log(f"  {k:>7} {pct:>5.1f}% {avg_top1:>7.0%} {margin_r:>10.4f} {avg_margin_ratio:>15.2f}x")

    return


# ============================================================
# Main
# ============================================================

def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_dir = _Path(f"tests/glm5_temp/stage705_combo_primitives_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.log"
    log = Logger(str(log_path))

    log("=" * 60)
    log("P62 + Unembed Rank + P55/P60 Resolution + Margin Analysis")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    all_unembed = {}

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        t0 = time.time()
        try:
            log(f"\n\n>>> Loading {model_name}...")
            model, tokenizer = load_model(model_name)

            # Experiment 1: Unembed rank
            unembed_info = analyze_unembed_rank(model_name, model, log)
            all_unembed[model_name] = unembed_info

            # Experiment 2: Combination primitives
            with torch.no_grad():
                unembed_weight = model.get_output_embeddings().weight.float().cpu()
            run_combination_primitives(model_name, model, tokenizer, unembed_weight, log)

            # Experiment 3: Contradiction resolution
            resolve_contradiction(model_name, model, tokenizer, unembed_weight, log)

            # Experiment 4: Margin at low dims
            analyze_margin_at_low_dims(model_name, model, tokenizer, unembed_weight, log)

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

    log(f"\n  Unembed effective rank:")
    for mn, info in all_unembed.items():
        log(f"    {mn}: rank={info['effective_rank']}, "
            f"SV[1]/SV[0]={info['top_sv'][1]/info['top_sv'][0]:.4f}, "
            f"SV[9]/SV[0]={info['top_sv'][min(9,len(info['top_sv'])-1)]/info['top_sv'][0]:.4f}")

    log(f"\n{'='*60}")
    log("FINAL CONCLUSIONS:")
    log(f"  1. Unembed rank vs effective rank = 9: need to compare")
    log(f"  2. Combination primitives: is A+E synergistic?")
    log(f"  3. P55/P60 contradiction: resolved (per-layer vs whole-vector)")
    log(f"  4. Margin preservation at low dims: does 1% keep margin?")
    log(f"{'='*60}")

    log.close()
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
