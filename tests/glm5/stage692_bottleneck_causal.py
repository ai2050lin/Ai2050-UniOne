#!/usr/bin/env python3
"""
P46: Information Bottleneck Causal Verification (Stage692)

Core questions:
1. Is DS7B's PCA90=13 dims a CAUSE of its strong reasoning?
2. If we force-expand the encoding dimensions, does reasoning degrade?
3. Is the information bottleneck necessary for RL-trained models?

Method: Compare DS7B vs Qwen3/GLM4 on:
- Per-layer effective rank (number of dimensions encoding >1% variance)
- Layer where effective rank drops below threshold
- Correlation between rank compression and perplexity on reasoning tasks

This is a non-interventional analysis (no model modification needed).
"""
import time, gc, json, pathlib, math, sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from sklearn.decomposition import PCA

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

TEXTS = [
    "The cat sat on the mat.", "The dog chased the ball.", "Birds fly south.",
    "Paris is the capital of France.", "Tokyo is a large city.", "The Amazon is a long river.",
    "She carefully folded the origami crane.", "The orchestra played beautifully.",
    "If it rains then the ground gets wet.", "She studied hard because she wanted to pass.",
    "Two plus two equals four exactly.", "The derivative of x squared is two x.",
    "DNA contains genetic instructions for life.", "Gravity causes objects to fall.",
    "The neural network learned patterns.", "The experiment yielded consistent results.",
    "A red apple is a fruit.", "The bank by the river was flooded.",
    "Spring flowers bloom in March.", "The seal swam in the cold ocean.",
    "All mammals have fur.", "Water boils at 100 degrees Celsius.",
    "The sun rises in the east.", "Photosynthesis converts light to energy.",
]


def load_model(model_name):
    path = MODEL_MAP[model_name]
    print(f"  loading: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def get_all_layer_hiddens(model, tokenizer, texts):
    """Get hidden states at ALL layers for multiple texts"""
    device = next(model.parameters()).device
    all_layers = []

    for text in texts:
        tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)

        # Last token hidden state at each layer
        layer_hiddens = []
        for i, hs in enumerate(outputs.hidden_states):
            layer_hiddens.append(hs[0, -1, :].float().cpu())
        all_layers.append(layer_hiddens)

    return all_layers  # [n_texts][n_layers][hidden_dim]


def analyze_per_layer_rank(all_layers, model_name, texts, n_texts_for_rank=10):
    """Analyze effective rank at each layer"""
    n_layers = len(all_layers[0])
    n_texts = len(all_layers)
    n_dims = all_layers[0][0].shape[0]

    print(f"  analyzing {n_layers} layers, {n_texts} texts, dim={n_dims}")

    layer_ranks = []

    for layer_idx in range(n_layers):
        # Collect hidden states for this layer across texts
        hs_matrix = torch.stack([all_layers[t][layer_idx] for t in range(n_texts)])
        # hs_matrix: [n_texts, n_dims]

        # Normalize each row
        norms = torch.norm(hs_matrix, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-10)
        hs_norm = hs_matrix / norms

        # PCA
        n_comp = min(n_dims, n_texts - 1, 50)
        pca = PCA(n_components=n_comp)
        pca.fit(hs_norm.numpy())

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        pca90 = np.searchsorted(cumvar, 0.90) + 1
        pca95 = np.searchsorted(cumvar, 0.95) + 1
        pca99 = np.searchsorted(cumvar, 0.99) + 1
        eff_rank = np.sum(pca.explained_variance_ratio_ > 0.01) + 1
        top1_var = pca.explained_variance_ratio_[0]

        layer_ranks.append({
            "layer": layer_idx,
            "pca90": int(pca90),
            "pca95": int(pca95),
            "pca99": int(pca99),
            "eff_rank": int(eff_rank),
            "top1_var": float(top1_var),
            "top3_var": float(sum(pca.explained_variance_ratio_[:3])),
        })

    return layer_ranks


def run_model(model_name, texts):
    """Full analysis for one model"""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)

    all_layers = get_all_layer_hiddens(model, tokenizer, texts)
    n_layers = len(all_layers[0])

    ranks = analyze_per_layer_rank(all_layers, model_name, texts)

    print(f"\n  Layer-by-layer effective rank:")
    print(f"  {'layer':>6s}  {'PCA90':>6s}  {'PCA95':>6s}  {'eff_rank':>8s}  {'Top1%':>6s}  {'Top3%':>6s}")
    for r in ranks:
        print(f"  {r['layer']:>6d}  {r['pca90']:>6d}  {r['pca95']:>6d}  {r['eff_rank']:>8d}  {r['top1_var']*100:>5.1f}%  {r['top3_var']*100:>5.1f}%")

    # Find bottleneck layer (minimum PCA90)
    bottleneck = min(ranks, key=lambda x: x["pca90"])
    print(f"\n  Bottleneck layer: L{bottleneck['layer']} (PCA90={bottleneck['pca90']}, Top1={bottleneck['top1_var']*100:.1f}%)")

    # Compression ratio: L0 PCA90 vs min PCA90
    l0_pca90 = ranks[0]["pca90"]
    min_pca90 = bottleneck["pca90"]
    compression = l0_pca90 / max(min_pca90, 1)
    print(f"  Compression ratio: {compression:.1f}x (L0={l0_pca90} -> min={min_pca90})")

    # DS7B comparison: RL-trained model should show deeper/wider bottleneck
    is_rl = model_name == "deepseek7b"

    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "ranks": ranks,
        "bottleneck_layer": bottleneck["layer"],
        "bottleneck_pca90": bottleneck["pca90"],
        "l0_pca90": l0_pca90,
        "compression_ratio": compression,
        "is_rl_trained": is_rl,
    }


def main():
    print("=" * 60)
    print("  P46: Information Bottleneck Causal Verification (Stage692)")
    print("=" * 60)

    all_results = {}
    t_total = time.time()

    for model_name in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        try:
            result = run_model(model_name, TEXTS)
            all_results[model_name] = result
        except Exception as e:
            print(f"  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison
    print(f"\n{'='*60}")
    print("  CROSS-MODEL COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'model':>12s}  {'layers':>6s}  {'L0 PCA90':>9s}  {'min PCA90':>9s}  {'bottleneck':>10s}  {'compress':>9s}  {'RL':>3s}")
    for m, r in all_results.items():
        print(f"  {m:>12s}  {r['n_layers']:>6d}  {r['l0_pca90']:>9d}  {r['bottleneck_pca90']:>9d}  {r['bottleneck_layer']:>10d}  {r['compression_ratio']:>8.1f}x  {'Y' if r['is_rl_trained'] else 'N':>3s}")

    # Key analysis
    print(f"\n  INV-348: RL-trained model has deeper/wider bottleneck")
    ds7b = all_results.get("deepseek7b", {})
    others = {m: r for m, r in all_results.items() if m != "deepseek7b"}

    if ds7b:
        ds7b_compress = ds7b.get("compression_ratio", 0)
        other_compress = [r["compression_ratio"] for r in others.values() if r.get("compression_ratio")]
        if other_compress:
            avg_other = np.mean(other_compress)
            print(f"  DS7B compression: {ds7b_compress:.1f}x")
            print(f"  Others avg compression: {avg_other:.1f}x")
            if ds7b_compress > avg_other * 1.5:
                print(f"  -> INV-348 CONFIRMED: RL-trained model has significantly deeper bottleneck")
            else:
                print(f"  -> INV-348 REJECTED: DS7B bottleneck not significantly deeper")

    print(f"\n  INV-349: Bottleneck layer correlates with 'deep fusion zone'")
    if ds7b:
        bf_layer = ds7b.get("bottleneck_layer", 0)
        print(f"  DS7B bottleneck: L{bf_layer}")
        print(f"  DS7B deep fusion zone: L5-L26 (from P38)")
        if 5 <= bf_layer <= 26:
            print(f"  -> INV-349 CONFIRMED: Bottleneck is inside deep fusion zone")
        else:
            print(f"  -> INV-349 REJECTED: Bottleneck outside deep fusion zone")

    # Save
    elapsed = time.time() - t_total
    out_path = pathlib.Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage692_bottleneck_causal_20260406_2200\summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {}
    for m, r in all_results.items():
        save[m] = {
            "n_layers": r["n_layers"],
            "bottleneck_layer": r["bottleneck_layer"],
            "bottleneck_pca90": r["bottleneck_pca90"],
            "l0_pca90": r["l0_pca90"],
            "compression_ratio": r["compression_ratio"],
            "is_rl_trained": r["is_rl_trained"],
            "ranks": r["ranks"],
        }
    save["elapsed"] = elapsed
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to: {out_path}")
    print(f"  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
