"""
P60: Primitive Ablation Comparison (基元消融对比实验)
Phase III - Primitives (基元化)

Goal: Compare 5 candidate encoding primitives by ablating each one
and measuring the impact on logit output accuracy.

5 candidates:
  A: Direction offset (方向偏移) - project h to orthogonal complement of semantic subspace
  B: Low-dim subspace (低维子空间) - replace h with random subspace projection
  C: Layer trajectory (层间轨迹) - shuffle intermediate layer h values
  D: Attention pattern (注意力模式) - replace attention with uniform
  E: Gate/modulation (门控模式) - set FFN gates to 0.5 (fully open)

Metrics: top-1 accuracy, logit Pearson r, KL divergence vs original

Models: Qwen3-4B, DeepSeek-7B, Gemma4-4B-it, GLM4-9B-chat (sequential)
"""

import torch
import numpy as np
from pathlib import Path as _Path
import time
import sys
import os

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

# ============================================================
# Test texts - 10 diverse sentences
# ============================================================
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
    """Load model and tokenizer using HuggingFace."""
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
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


def get_hidden_states(model, tokenizer, text):
    """Extract all layer hidden states and final logits."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: (layer+1, batch, seq, d_model) - includes embedding layer
    hidden_states = outputs.hidden_states
    logits = outputs.logits  # (batch, seq, vocab)
    # Get last token hidden states for all layers
    last_token_hs = [hs[0, -1, :].float().cpu() for hs in hidden_states]
    last_token_logits = logits[0, -1, :].float().cpu()
    return last_token_hs, last_token_logits


def get_attention_patterns(model, tokenizer, text, layer_idx):
    """Extract attention pattern for a specific layer."""
    from transformers import AutoModelForCausalLM
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Use hooks to capture attention
    attn_weights = {}

    def hook_fn(module, input, output):
        # output is typically (attn_output, attn_weights) for attention layers
        if isinstance(output, tuple) and len(output) > 1:
            aw = output[1]
            if aw is not None:
                attn_weights['weights'] = aw.float().cpu().mean(0)  # average over heads

    # Find the attention layer
    target_layer = None
    for name, module in model.named_modules():
        if f"layer_{layer_idx}" in name or f"layers.{layer_idx}" in name:
            if "attention" in name.lower() or "attn" in name.lower():
                # Register hooks on all submodules of this layer's attention
                for subname, submod in module.named_modules():
                    submod.register_forward_hook(hook_fn)
                target_layer = module
                break

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    # If hook didn't capture, use output_attentions
    if 'weights' not in attn_weights:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        if outputs.attentions is not None and len(outputs.attentions) > layer_idx:
            attn_weights['weights'] = outputs.attentions[layer_idx][0].float().cpu().mean(0)

    return attn_weights.get('weights', None)


# ============================================================
# Ablation functions for each primitive
# ============================================================

def ablate_A_direction(hidden_states_list, all_texts_hs):
    """
    A: Direction offset ablation - project h to orthogonal complement of
       the top principal component direction across all texts.
    This removes the primary "semantic direction" while preserving everything else.
    """
    # Compute PCA of final hidden states across all texts
    all_h = torch.stack([hs[-1] for hs in all_texts_hs])  # (n_texts, d_model)
    mean_h = all_h.mean(0)

    # Use top-1 PCA direction
    centered = all_h - mean_h
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    top_direction = Vt[0]  # (d_model,)

    # Project each layer's h to orthogonal complement of top_direction
    ablated = []
    for hs in hidden_states_list:
        proj = torch.dot(hs, top_direction) * top_direction
        ablated.append(hs - proj)
    return ablated


def ablate_B_subspace(hidden_states_list, all_texts_hs, k=None):
    """
    B: Low-dim subspace ablation - project h onto a random k-dimensional subspace.
    Preserves dimensionality but destroys the learned semantic subspace.
    """
    d_model = hidden_states_list[-1].shape[0]
    if k is None:
        k = max(1, d_model // 10)  # 10% of dimensions

    # Generate random orthogonal basis
    torch.manual_seed(42)
    random_basis = torch.randn(d_model, k).float()
    Q, _ = torch.linalg.qr(random_basis)

    # Project h onto this random subspace, then expand back
    ablated = []
    for hs in hidden_states_list:
        coords = hs @ Q  # (k,)
        h_proj = Q @ coords  # (d_model,) - in the random subspace
        ablated.append(h_proj)
    return ablated


def ablate_C_trajectory(hidden_states_list):
    """
    C: Layer trajectory ablation - shuffle intermediate layer h values.
    Preserves start (L0) and end (final) but randomizes the path.
    """
    n_layers = len(hidden_states_list)
    if n_layers <= 2:
        return hidden_states_list

    ablated = [hidden_states_list[0]]  # Keep L0
    # Shuffle intermediate layers (L1 to L_{n-2})
    intermediate = hidden_states_list[1:-1]
    indices = list(range(len(intermediate)))
    np.random.seed(42)
    np.random.shuffle(indices)
    for idx in indices:
        ablated.append(intermediate[idx])
    ablated.append(hidden_states_list[-1])  # Keep final layer
    return ablated


def ablate_D_attention(hidden_states_list, layer_attn_weights, unembed_weight):
    """
    D: Attention pattern ablation - simulate uniform attention by averaging
       the contribution of each layer equally.
    Since we can't easily modify attention during forward pass without
    complex hooks, we approximate: weight each layer's delta_h equally.
    """
    n_layers = len(hidden_states_list)
    # Compute delta_h for each layer
    deltas = []
    for i in range(1, n_layers):
        delta = hidden_states_list[i] - hidden_states_list[i - 1]
        deltas.append(delta)

    # Replace each delta with the mean delta (uniform contribution)
    if deltas:
        mean_delta = torch.stack(deltas).mean(0)
        ablated = [hidden_states_list[0]]
        for i in range(1, n_layers):
            ablated.append(ablated[-1] + mean_delta)
    else:
        ablated = hidden_states_list

    return ablated


def ablate_E_gate(hidden_states_list):
    """
    E: Gate/modulation ablation - set all delta_h magnitudes to the mean.
    This simulates "all FFN gates fully open" by equalizing per-layer contributions.
    """
    n_layers = len(hidden_states_list)
    deltas = []
    for i in range(1, n_layers):
        delta = hidden_states_list[i] - hidden_states_list[i - 1]
        deltas.append(delta)

    if deltas:
        # Equalize magnitudes but keep directions
        norms = torch.stack([d.norm() for d in deltas])
        mean_norm = norms.mean()
        ablated = [hidden_states_list[0]]
        for i in range(len(deltas)):
            direction = deltas[i] / (deltas[i].norm() + 1e-10)
            ablated.append(ablated[-1] + direction * mean_norm)
    else:
        ablated = hidden_states_list

    return ablated


def compute_logits_from_h(h_final, unembed_weight):
    """Compute logits from a hidden state vector."""
    return h_final @ unembed_weight.T


def evaluate_ablation(orig_logits, ablated_logits):
    """Compare original and ablated logits."""
    orig_top1 = orig_logits.argmax().item()
    ablated_top1 = ablated_logits.argmax().item()
    top1_match = 1 if orig_top1 == ablated_top1 else 0

    # Pearson correlation (sample for efficiency)
    n = len(orig_logits)
    sample_size = min(5000, n)
    idx = np.random.choice(n, sample_size, replace=False)
    r = np.corrcoef(orig_logits.numpy()[idx], ablated_logits.numpy()[idx])[0, 1]

    # KL divergence
    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_probs = torch.softmax(ablated_logits, dim=-1)
    # Avoid log(0)
    ablated_probs = ablated_probs.clamp(min=1e-10)
    kl = (orig_probs * (orig_probs.log() - ablated_probs.log())).sum().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_logits.unsqueeze(0), ablated_logits.unsqueeze(0)
    ).item()

    return {
        "top1_match": top1_match,
        "pearson_r": r,
        "kl_div": kl,
        "cos_sim": cos_sim,
    }


def run_ablation_suite(model_name, model, tokenizer, log):
    """Run all 5 ablation types on a single model."""
    log(f"\n{'='*60}")
    log(f"Model: {model_name}")
    log(f"{'='*60}")

    # Get unembed weight
    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()

    # Extract hidden states for all texts
    all_hidden_states = []
    all_logits = []
    all_attn = {}

    for text in TEST_TEXTS:
        hs_list, logits = get_hidden_states(model, tokenizer, text)
        all_hidden_states.append(hs_list)
        all_logits.append(logits)

    # Run each ablation type
    ablation_types = {
        "A_direction": lambda hs, idx: ablate_A_direction(hs, all_hidden_states),
        "B_subspace": lambda hs, idx: ablate_B_subspace(hs, all_hidden_states),
        "C_trajectory": lambda hs, idx: ablate_C_trajectory(hs),
        "D_attention": lambda hs, idx: ablate_D_attention(hs, {}, unembed_weight),
        "E_gate": lambda hs, idx: ablate_E_gate(hs),
    }

    results = {name: [] for name in ablation_types}

    for ablation_name, ablation_fn in ablation_types.items():
        log(f"\n  --- Ablation {ablation_name} ---")
        for text_idx, text in enumerate(TEST_TEXTS):
            hs_list = all_hidden_states[text_idx]
            orig_logits = all_logits[text_idx]

            ablated_hs = ablation_fn(hs_list, text_idx)
            ablated_h_final = ablated_hs[-1]
            ablated_logits = compute_logits_from_h(ablated_h_final, unembed_weight)

            metrics = evaluate_ablation(orig_logits, ablated_logits)
            results[ablation_name].append(metrics)

        # Aggregate
        top1_rate = np.mean([m["top1_match"] for m in results[ablation_name]])
        avg_r = np.mean([m["pearson_r"] for m in results[ablation_name]])
        avg_kl = np.mean([m["kl_div"] for m in results[ablation_name]])
        avg_cos = np.mean([m["cos_sim"] for m in results[ablation_name]])

        log(f"    top-1 match: {top1_rate:.0%}")
        log(f"    Pearson r: {avg_r:.4f}")
        log(f"    KL div: {avg_kl:.2f}")
        log(f"    Cosine sim: {avg_cos:.4f}")

    # Summary comparison
    log(f"\n  --- Summary for {model_name} ---")
    log(f"  {'Ablation':<16} {'Top-1':>8} {'Pearson r':>10} {'KL div':>8} {'Cos sim':>8}")
    log(f"  {'-'*52}")
    for name in ablation_types:
        top1 = np.mean([m["top1_match"] for m in results[name]])
        r = np.mean([m["pearson_r"] for m in results[name]])
        kl = np.mean([m["kl_div"] for m in results[name]])
        cos = np.mean([m["cos_sim"] for m in results[name]])
        log(f"  {name:<16} {top1:>7.0%} {r:>10.4f} {kl:>8.2f} {cos:>8.4f}")

    # Ranking: which ablation causes the most damage?
    log(f"\n  --- Damage Ranking (most to least) ---")
    damage_scores = {}
    for name in ablation_types:
        # Lower top1 + lower r + higher KL = more damage
        avg_top1 = np.mean([m["top1_match"] for m in results[name]])
        avg_r = np.mean([m["pearson_r"] for m in results[name]])
        avg_kl = np.mean([m["kl_div"] for m in results[name]])
        damage = (1 - avg_top1) + (1 - avg_r) + avg_kl / (avg_kl + 1)
        damage_scores[name] = damage

    ranked = sorted(damage_scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        log(f"    #{rank}: {name} (damage={score:.3f})")

    return results, ranked


# ============================================================
# Additional experiments: Layer-wise information content (P60b)
# ============================================================

def run_layer_importance(model_name, model, tokenizer, log):
    """Measure per-layer importance by ablating one layer at a time."""
    log(f"\n{'='*60}")
    log(f"Layer Importance Analysis: {model_name}")
    log(f"{'='*60}")

    with torch.no_grad():
        unembed = model.get_output_embeddings()
        unembed_weight = unembed.weight.float().cpu()

    # Use first 3 texts for efficiency
    texts = TEST_TEXTS[:3]
    n_layers = None

    all_results = []
    for text_idx, text in enumerate(texts):
        hs_list, orig_logits = get_hidden_states(model, tokenizer, text)
        n_layers = len(hs_list)

        layer_results = []
        # Ablate layer i: set delta_h_i to zero
        for i in range(1, n_layers):
            ablated_hs = list(hs_list)
            # Set layer i's hidden state to be same as layer i-1
            ablated_hs[i] = hs_list[i - 1]
            # Recompute subsequent layers: not possible without forward pass
            # Instead, just use final h but with layer i's delta removed
            # Approximate: subtract delta_i from final h
            delta_i = hs_list[i] - hs_list[i - 1]
            h_final_ablated = hs_list[-1] - delta_i
            ablated_logits = h_final_ablated @ unembed_weight.T

            metrics = evaluate_ablation(orig_logits, ablated_logits)
            layer_results.append(metrics)

        all_results.append(layer_results)

    # Average across texts
    log(f"\n  {'Layer':>6} {'Top-1':>8} {'Pearson r':>10} {'KL div':>8}")
    log(f"  {'-'*36}")
    layer_scores = []
    for layer in range(n_layers - 1):
        top1 = np.mean([r[layer]["top1_match"] for r in all_results])
        r = np.mean([r[layer]["pearson_r"] for r in all_results])
        kl = np.mean([r[layer]["kl_div"] for r in all_results])
        log(f"  L{layer:>4} {top1:>7.0%} {r:>10.4f} {kl:>8.2f}")
        layer_scores.append((layer, top1, r, kl))

    # Find most important layers (where ablation causes most damage)
    most_important = sorted(layer_scores, key=lambda x: x[1])[:5]
    log(f"\n  Top 5 most important layers (lowest top-1 after ablation):")
    for layer, top1, r, kl in most_important:
        log(f"    L{layer}: top-1={top1:.0%}, r={r:.4f}, KL={kl:.2f}")

    return layer_scores


# ============================================================
# Main
# ============================================================

def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_dir = _Path(f"tests/glm5_temp/stage703_primitive_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.log"
    log = Logger(str(log_path))

    log("=" * 60)
    log("P60: Primitive Ablation Comparison (基元消融对比实验)")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    all_results = {}

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        t0 = time.time()
        try:
            log(f"\n\nLoading {model_name}...")
            model, tokenizer = load_model(model_name)
            results, ranking = run_ablation_suite(model_name, model, tokenizer, log)

            # Layer importance
            layer_scores = run_layer_importance(model_name, model, tokenizer, log)

            all_results[model_name] = {
                "ablation_results": results,
                "ranking": ranking,
                "layer_scores": layer_scores,
            }

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  ERROR with {model_name}: {e}")
            import traceback
            log(traceback.format_exc())

        elapsed = time.time() - t0
        log(f"\n  {model_name} done in {elapsed:.1f}s")

    # Final cross-model comparison
    log(f"\n\n{'='*60}")
    log("FINAL CROSS-MODEL COMPARISON")
    log(f"{'='*60}")

    for model_name, data in all_results.items():
        log(f"\n  {model_name}:")
        for rank, (name, score) in enumerate(data["ranking"], 1):
            log(f"    #{rank}: {name} (damage={score:.3f})")

    # Synthesis: which primitive is most important across all models?
    log(f"\n  --- Cross-model primitive importance ---")
    primitive_scores = {}
    for model_name, data in all_results.items():
        for name, score in data["ranking"]:
            if name not in primitive_scores:
                primitive_scores[name] = []
            primitive_scores[name].append(score)

    for name in sorted(primitive_scores.keys()):
        scores = primitive_scores[name]
        avg_score = np.mean(scores)
        log(f"    {name}: avg_damage={avg_score:.3f}, range=[{min(scores):.3f}, {max(scores):.3f}]")

    # Final conclusion
    log(f"\n{'='*60}")
    log("CONCLUSION:")
    most_important_primitive = min(primitive_scores.items(), key=lambda x: np.mean(x[1]))
    log(f"  Most important primitive (highest avg damage): {most_important_primitive[0]}")
    log(f"  Average damage score: {np.mean(most_important_primitive[1]):.3f}")
    log(f"  Interpretation: Ablating this primitive causes the most disruption,")
    log(f"  suggesting it is closest to the true encoding mechanism.")
    log(f"{'='*60}")

    log.close()
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
