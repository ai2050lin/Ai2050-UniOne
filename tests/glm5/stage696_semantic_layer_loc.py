#!/usr/bin/env python3
"""
P50: Fine-tuning Layer Semantic Localization (Stage696)

Core question: After the ~90deg rotation completes at L0->L1,
do subsequent layers' delta-h directions carry semantic information?

Method:
1. For each text, extract h at every layer (L1 to Ln)
2. Compute delta-h_l = h_l - h_{l-1} for each layer
3. Normalize delta-h_l directions
4. Test: does delta-h direction vary by semantic category (ANOVA)?
   - If ANOVA F > 3.0 for some layer → that layer participates in semantic encoding
   - If all F < 2.0 → semantic encoding is NOT in direction deltas

Key comparison with P48:
- P48 showed: rotation angle (L0→final) is NOT category-dependent
- P50 tests: fine-tuning deltas (L1→L2, L2→L3, ...) ARE category-dependent?
- This is the critical test for whether "fine-tuning layers carry semantics"

9 categories, 4 tokens per category, 4 models.
"""
import sys, math, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from sklearn.decomposition import PCA
from scipy import stats

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

# 9 categories x 4 texts = 36 texts
TEXTS = [
    # concrete (0-3)
    "The cat sat on the mat.", "The dog chased the ball.",
    "Birds fly south in winter.", "The seal swam in the cold ocean.",
    # factual (4-7)
    "Paris is the capital of France.", "Tokyo is a large city.",
    "The Amazon is a long river.", "Gravity causes objects to fall.",
    # aesthetic (8-11)
    "She carefully folded the origami crane.", "The orchestra played beautifully.",
    "His writing was elegant and precise.", "The painting was incredibly detailed.",
    # logical (12-15)
    "If it rains then the ground gets wet.", "She studied hard because she wanted to pass.",
    "The boy who was running fell down.", "Although tired she continued working.",
    # temporal (16-19)
    "Yesterday it rained heavily all day.", "She will finish the report by Friday.",
    "The project was completed last month.", "He arrived before the ceremony began.",
    # math (20-23)
    "Two plus two equals four exactly.", "The derivative of x squared is two x.",
    "The equation can be solved step by step.", "The experiment yielded consistent results.",
    # science (24-27)
    "DNA contains genetic instructions for life.", "The neural network learned patterns.",
    "The hypothesis was supported by data.", "The match was exciting to watch.",
    # ambiguous (28-31)
    "A red apple is a fruit.", "The bank by the river was flooded.",
    "A fair decision was made by the judge.", "Spring flowers bloom in March.",
    # general (32-35)
    "The quick brown fox jumps over the lazy dog.", "She has been working on this project.",
    "They went to the market when it started.", "The report was submitted on time.",
]

CATEGORIES = ["concrete", "factual", "aesthetic", "logical", "temporal",
              "math", "science", "ambiguous", "general"]
CAT_MAP = {i: CATEGORIES[i // 4] for i in range(36)}

TIMESTAMP = time.strftime("%Y%m%d_%H%M")
OUTPUT_DIR = _Path(f"d:\\develop\\TransformerLens-main\\tests\\glm5_temp\\stage696_semantic_layer_loc_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, filepath):
        self.f = open(filepath, "w", encoding="utf-8")
    def __call__(self, msg="", end="\n"):
        try:
            safe_msg = msg.encode('gbk', errors='replace').decode('gbk')
        except:
            safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg, end=end)
        self.f.write(msg + end)
        if end == "\n":
            self.f.flush()
    def close(self):
        self.f.close()


def load_model(model_name):
    """Load model and tokenizer using HuggingFace."""
    model_path = MODEL_MAP[model_name]
    log = Logger(OUTPUT_DIR / f"_load_{model_name}.log")
    log(f"Loading {model_name}...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    log(f"Loaded via HuggingFace in {time.time()-t0:.1f}s")
    log.close()
    return model, tokenizer


def get_hidden_states(model, tokenizer, text):
    """Extract per-layer hidden states using HuggingFace model."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    last_token_states = []
    for hs in hidden_states:
        last_token_states.append(hs[0, -1, :].float().cpu())
    return torch.stack(last_token_states)  # (num_layers+1, d_model)


def compute_layer_deltas(hidden_states):
    """Compute delta-h for each layer transition (L1-L0, L2-L1, ..., Ln-L(n-1))."""
    # hidden_states: (num_layers+1, d_model)
    deltas = hidden_states[1:] - hidden_states[:-1]  # (num_layers, d_model)
    return deltas


def anova_on_directions(directions, cat_labels, n_cats=9):
    """ANOVA: do direction components vary by category?

    For each principal component of the direction vectors,
    test if category explains variance.

    Returns: F_statistic, p_value
    """
    # Project directions onto their top PCs
    if len(directions) < n_cats:
        return 0.0, 1.0

    # Use top-5 PCs
    pca = PCA(n_components=min(5, directions.shape[0]-1, directions.shape[1]))
    projected = pca.fit_transform(directions.numpy())  # (n_texts, n_components)

    # For each PC, run ANOVA
    f_values = []
    p_values = []
    for i in range(projected.shape[1]):
        groups = [projected[np.array(cat_labels) == c, i] for c in range(n_cats)]
        # Only keep groups with >1 element
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            f_values.append(0.0)
            p_values.append(1.0)
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        f_values.append(f_stat)
        p_values.append(p_val)

    # Return max F across PCs (most significant direction)
    max_f = max(f_values) if f_values else 0.0
    min_p = min(p_values) if p_values else 1.0
    return max_f, min_p


def main():
    log_path = OUTPUT_DIR / "results.log"
    log = Logger(log_path)

    log("=" * 80)
    log("P50: Fine-tuning Layer Semantic Localization")
    log(f"Timestamp: {TIMESTAMP}")
    log("=" * 80)

    all_model_results = {}

    for model_name in MODEL_MAP:
        log(f"\n{'='*60}")
        log(f"Processing model: {model_name}")
        log(f"{'='*60}")
        t0 = time.time()

        # Load model
        model, tokenizer = load_model(model_name)

        # Extract hidden states for all texts
        all_hidden_states = []  # list of (num_layers+1, d_model) tensors
        for text in TEXTS:
            try:
                hs = get_hidden_states(model, tokenizer, text)
                all_hidden_states.append(hs)
            except Exception as e:
                log(f"  Error for text: {e}")
                all_hidden_states.append(None)

        # Filter successful extractions
        valid_indices = [i for i, hs in enumerate(all_hidden_states) if hs is not None]
        log(f"Valid texts: {len(valid_indices)}/{len(TEXTS)}")

        if len(valid_indices) < 18:
            log(f"  Too few valid texts, skipping {model_name}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Get number of layers
        num_layers = all_hidden_states[valid_indices[0]].shape[0] - 1
        d_model = all_hidden_states[valid_indices[0]].shape[1]
        log(f"Layers: {num_layers}, d_model: {d_model}")

        # Compute layer deltas for each text
        cat_labels = []
        all_deltas = []  # list of (num_layers, d_model) tensors
        for idx in valid_indices:
            hs = all_hidden_states[idx]
            deltas = compute_layer_deltas(hs)
            all_deltas.append(deltas)
            cat_labels.append(CAT_MAP[idx])

        cat_labels = np.array([CATEGORIES.index(c) for c in cat_labels])

        # For each layer, compute delta direction and run ANOVA
        layer_results = []
        semantic_layers = []

        for layer_idx in range(num_layers):
            # Collect delta directions for this layer
            layer_deltas = torch.stack([d[layer_idx] for d in all_deltas])  # (n_texts, d_model)

            # Compute norms and directions
            norms = layer_deltas.norm(dim=1)
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()

            # Normalize to get directions
            directions = layer_deltas / (norms.unsqueeze(1) + 1e-8)

            # ANOVA on direction components
            f_stat, p_val = anova_on_directions(directions, cat_labels)

            # Also compute pairwise direction consistency within categories
            intra_cat_cos = []
            inter_cat_cos = []
            for ci in range(9):
                ci_indices = np.where(cat_labels == ci)[0]
                for cj in range(ci+1, 9):
                    cj_indices = np.where(cat_labels == cj)[0]
                    for ii in ci_indices:
                        for jj in cj_indices:
                            cos_val = F.cosine_similarity(
                                directions[ii].unsqueeze(0),
                                directions[jj].unsqueeze(0)
                            ).item()
                            inter_cat_cos.append(cos_val)
                for ii in ci_indices:
                    for jj in ci_indices:
                        if ii < jj:
                            cos_val = F.cosine_similarity(
                                directions[ii].unsqueeze(0),
                                directions[jj].unsqueeze(0)
                            ).item()
                            intra_cat_cos.append(cos_val)

            mean_intra = np.mean(intra_cat_cos) if intra_cat_cos else 0
            mean_inter = np.mean(inter_cat_cos) if inter_cat_cos else 0
            discriminability = mean_intra - mean_inter  # higher = more category-specific

            # Also compute delta direction consistency across all texts
            all_dirs = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
            pairwise_cos = []
            for i in range(len(all_dirs)):
                for j in range(i+1, len(all_dirs)):
                    c = F.cosine_similarity(all_dirs[i].unsqueeze(0), all_dirs[j].unsqueeze(0)).item()
                    pairwise_cos.append(c)
            mean_pairwise = np.mean(pairwise_cos) if pairwise_cos else 0

            result = {
                "layer": layer_idx,
                "mean_delta_norm": mean_norm,
                "std_delta_norm": std_norm,
                "anova_f": f_stat,
                "anova_p": p_val,
                "intra_cat_cos": mean_intra,
                "inter_cat_cos": mean_inter,
                "discriminability": discriminability,
                "pairwise_cos": mean_pairwise,
                "is_semantic": f_stat > 3.0,
            }
            layer_results.append(result)

            if f_stat > 3.0:
                semantic_layers.append(layer_idx)

            # Print progress every 5 layers
            if (layer_idx + 1) % 5 == 0 or layer_idx < 3:
                log(f"  L{layer_idx:2d}: norm={mean_norm:.2f}+/-{std_norm:.2f}  "
                    f"F={f_stat:.2f}  p={p_val:.3f}  disc={discriminability:.4f}  "
                    f"pairwise={mean_pairwise:.3f}  {'***SEMANTIC***' if f_stat > 3.0 else ''}")

        # Find layers with highest ANOVA F
        layer_results.sort(key=lambda x: x["anova_f"], reverse=True)
        top5 = layer_results[:5]

        log(f"\n  === Summary for {model_name} ===")
        log(f"  Total layers: {num_layers}")
        log(f"  Semantic layers (F>3.0): {len(semantic_layers)}")
        if semantic_layers:
            log(f"  Semantic layer indices: {semantic_layers}")
        else:
            log(f"  WARNING: No layer has F>3.0! Fine-tuning deltas may not carry semantics.")

        log(f"\n  Top-5 layers by ANOVA F:")
        for r in top5:
            log(f"    L{r['layer']:2d}: F={r['anova_f']:.2f}  disc={r['discriminability']:.4f}  "
                f"norm={r['mean_delta_norm']:.2f}  pairwise={r['pairwise_cos']:.3f}")

        # Layer-wise summary of discriminability
        log(f"\n  Layer-wise discriminability (intra - inter category cos):")
        sorted_by_layer = sorted(layer_results, key=lambda x: x["layer"])
        disc_values = [r["discriminability"] for r in sorted_by_layer]
        log(f"    Mean disc: {np.mean(disc_values):.4f} +/- {np.std(disc_values):.4f}")
        log(f"    Max disc: {max(disc_values):.4f} at L{sorted_by_layer[np.argmax(disc_values)]['layer']}")
        log(f"    Min disc: {min(disc_values):.4f} at L{sorted_by_layer[np.argmin(disc_values)]['layer']}")

        # Overall statistics
        f_values = [r["anova_f"] for r in layer_results]
        log(f"\n  ANOVA F statistics:")
        log(f"    Mean F: {np.mean(f_values):.2f}")
        log(f"    Max F: {max(f_values):.2f}")
        log(f"    Layers with F>2.0: {sum(1 for f in f_values if f > 2.0)}")
        log(f"    Layers with F>3.0: {sum(1 for f in f_values if f > 3.0)}")

        # Save results
        model_result = {
            "model": model_name,
            "num_layers": num_layers,
            "d_model": d_model,
            "n_valid_texts": len(valid_indices),
            "semantic_layers": semantic_layers,
            "n_semantic_layers": len(semantic_layers),
            "layer_results": layer_results,
            "mean_anova_f": np.mean(f_values),
            "max_anova_f": max(f_values),
        }
        all_model_results[model_name] = model_result

        # Save JSON (safe serialization)
        import json
        json_path = OUTPUT_DIR / f"results_{model_name}.json"
        safe_result = {
            "model": model_name,
            "num_layers": num_layers,
            "d_model": d_model,
            "n_valid_texts": len(valid_indices),
            "semantic_layers": semantic_layers,
            "n_semantic_layers": len(semantic_layers),
            "mean_anova_f": float(np.mean([r["anova_f"] for r in layer_results])),
            "max_anova_f": float(max(r["anova_f"] for r in layer_results)),
            "layer_details": [
                {
                    "layer": r["layer"],
                    "mean_delta_norm": float(r["mean_delta_norm"]),
                    "std_delta_norm": float(r["std_delta_norm"]),
                    "anova_f": float(r["anova_f"]),
                    "anova_p": float(r["anova_p"]),
                    "intra_cat_cos": float(r["intra_cat_cos"]),
                    "inter_cat_cos": float(r["inter_cat_cos"]),
                    "discriminability": float(r["discriminability"]),
                    "pairwise_cos": float(r["pairwise_cos"]),
                    "is_semantic": bool(r["is_semantic"]),
                }
                for r in sorted(layer_results, key=lambda x: x["layer"])
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(safe_result, f, indent=2)

        elapsed = time.time() - t0
        log(f"  Completed in {elapsed:.1f}s")

        # Cleanup
        del model, all_hidden_states, all_deltas
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison
    log(f"\n{'='*80}")
    log("CROSS-MODEL COMPARISON")
    log(f"{'='*80}")

    log(f"\n  A: Semantic Layer Count (F > 3.0)")
    log(f"  {'Model':>12s}: {'Semantic Layers':>14s}  {'Max F':>6s}  {'Mean F':>6s}  {'Mean Disc':>8s}")
    for m, r in all_model_results.items():
        disc_vals = [lr["discriminability"] for lr in r["layer_results"]]
        log(f"  {m:>12s}: {r['n_semantic_layers']:>3d} / {r['num_layers']:>3d} layers  "
            f"{r['max_anova_f']:>6.2f}  {r['mean_anova_f']:>6.2f}  "
            f"{np.mean(disc_vals):>8.4f}")

    log(f"\n  B: Top-3 Semantic Layers per Model")
    for m, r in all_model_results.items():
        top3 = sorted(r["layer_results"], key=lambda x: x["anova_f"], reverse=True)[:3]
        layers_str = ", ".join([f"L{lr['layer']}(F={lr['anova_f']:.1f})" for lr in top3])
        log(f"  {m:>12s}: {layers_str}")

    log(f"\n  C: Key Question - Does ANY layer carry semantics?")
    any_semantic = any(r["n_semantic_layers"] > 0 for r in all_model_results.values())
    if any_semantic:
        log(f"  YES: At least one model has semantic layers (F>3.0)")
        for m, r in all_model_results.items():
            if r["n_semantic_layers"] > 0:
                log(f"    {m}: layers {r['semantic_layers']}")
    else:
        log(f"  NO: No model has any layer with F>3.0")
        log(f"  >> IMPLICATION: Fine-tuning direction deltas do NOT carry semantic category info")
        log(f"  >> The encoding primitive may NOT be 'direction delta' - need to search elsewhere")

    log(f"\n  D: Delta norm profile (magnitude of fine-tuning)")
    log(f"  {'Model':>12s}: {'L1 norm':>8s}  {'Mid norm':>8s}  {'Last norm':>8s}  {'Ratio last/first':>14s}")
    for m, r in all_model_results.items():
        sorted_lr = sorted(r["layer_results"], key=lambda x: x["layer"])
        l1_norm = sorted_lr[0]["mean_delta_norm"]
        mid_idx = len(sorted_lr) // 2
        mid_norm = sorted_lr[mid_idx]["mean_delta_norm"]
        last_norm = sorted_lr[-1]["mean_delta_norm"]
        ratio = last_norm / (l1_norm + 1e-8)
        log(f"  {m:>12s}: {l1_norm:>8.2f}  {mid_norm:>8.2f}  {last_norm:>8.2f}  {ratio:>14.2f}")

    # Save combined results
    import json
    combined_path = OUTPUT_DIR / "combined_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in all_model_results.items()}, f, indent=2)

    log(f"\nResults saved to {OUTPUT_DIR}")
    log(f"\n{'='*80}")
    log("P50 COMPLETE")
    log(f"{'='*80}")

    log.close()


if __name__ == "__main__":
    main()
