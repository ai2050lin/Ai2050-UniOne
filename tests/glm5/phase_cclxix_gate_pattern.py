"""
Phase CCLXIX: 门控选通模式分析 (Gate Activation Pattern Analysis)
================================================
核心问题: FFN的门控选通σ(W_gate @ h)有什么数学结构?

背景:
  INV-44: FFN概念差异>92%来自非线性
  INV-45: 线性近似cos_linear<0(指向反方向)
  INV-46: 非线性部分cos_nonlinear>0(驱动概念信号)
  
  关键假说: 门控σ(W_gate @ h)对概念A和概念B选择不同的中间神经元,
  这种"选择性切换"就是FFN纠正力的数学本质.

验证:
  Exp1: 门控稀疏度 — σ(W_gate @ h)有多少神经元被激活?
        -> 是稀疏激活(<<50%)还是稠密(>50%)?
        -> 稀疏度是否随深度变化?
  Exp2: 门控模式与概念类别 — 不同概念的门控重叠度
        -> 同类概念(如dog vs cat)的门控重叠 > 异类概念(如dog vs apple)?
        -> 门控模式是否能区分概念类别?
  Exp3: 门控模式的几何结构 — W_gate行向量在d_model空间中的结构
        -> W_gate行是否有聚类结构?
        -> 概念向量h与W_gate行的点积模式

用法:
  python phase_cclxix_gate_pattern.py --model qwen3 --exp 1
  python phase_cclxix_gate_pattern.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, LayerWeights,
)

OUTPUT_DIR = Path("results/causal_fiber")

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake",
                "lion", "bear", "whale", "dolphin", "rabbit", "deer"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango",
                "grape", "banana", "pasta", "pizza", "cookie", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel",
                "pliers", "ruler", "level", "clamp", "file", "shovel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano",
                "canyon", "glacier", "meadow", "island", "valley", "cliff"],
}

TEMPLATES = [
    "The {} is",
]


def json_serialize(obj):
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return obj
    elif obj is None:
        return None
    return obj


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================
# Exp1: 门控稀疏度
# ============================================================
def exp1_gate_sparsity(model_name):
    """Measure sparsity of gate activation σ(W_gate @ h).
    
    Key measures:
      - Fraction of neurons with σ() > 0.5 (activated)
      - Fraction of neurons with σ() > 0.1 (weakly activated)
      - Entropy of gate distribution
      - How does sparsity vary with depth?
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Gate Activation Sparsity")
    print(f"  Model: {model_name}")
    print(f"  Key test: Is gate activation sparse or dense?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    
    # Collect concept sentences
    all_sentences = []
    all_labels = []
    for cat, words in CONCEPTS.items():
        for word in words:
            for tmpl in TEMPLATES:
                all_sentences.append(tmpl.format(word))
                all_labels.append(cat)
    
    # Sample
    rng = np.random.RandomState(42)
    sample_indices = []
    for cat in CONCEPTS:
        cat_indices = [i for i, l in enumerate(all_labels) if l == cat]
        sample_indices.extend(rng.choice(cat_indices, min(6, len(cat_indices)), replace=False))
    
    print(f"  Total sentences: {len(sample_indices)}")
    
    # Results
    gate_frac_05 = defaultdict(list)  # layer -> [fraction σ>0.5]
    gate_frac_01 = defaultdict(list)  # layer -> [fraction σ>0.1]
    gate_entropy = defaultdict(list)   # layer -> [entropy]
    gate_mean = defaultdict(list)      # layer -> [mean σ]
    gate_top10_sum = defaultdict(list) # layer -> [top-10 neurons' sum / total]
    n_intermediate = 0
    
    for sent_idx in sample_indices:
        sent = all_sentences[sent_idx]
        input_ids = tokenizer(sent, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        # Capture post-LN residual (FFN input) and gate activations
        ln_out = {}
        gate_act = {}
        
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            
            def make_ffn_pre(key):
                def hook(module, args):
                    if isinstance(args, tuple):
                        ln_out[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        ln_out[key] = args[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            # For Qwen2/Qwen3: hook after gate_proj to get pre-activation
            # But we can't easily get the intermediate activation after gate_proj + sigmoid
            # Instead, compute gate activation manually from LN output and W_gate
            # We'll capture LN output and compute σ(W_gate @ h) ourselves
            
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for h in hooks:
            h.remove()
        
        # Compute gate activations from LN output + W_gate weights
        for li in range(n_layers):
            key = f"L{li}"
            if key not in ln_out:
                continue
            
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            
            h_ln = ln_out[key]  # [d_model]
            W_gate = lw.W_gate  # [intermediate, d_model]
            
            if n_intermediate == 0:
                n_intermediate = W_gate.shape[0]
            
            # Gate pre-activation: z = W_gate @ h_ln
            z = W_gate @ h_ln  # [intermediate]
            
            # Sigmoid activation: σ(z)
            # Clip to avoid overflow
            z_clipped = np.clip(z, -500, 500)
            sigma_z = 1.0 / (1.0 + np.exp(-z_clipped))
            
            # Sparsity measures
            frac_05 = float(np.mean(sigma_z > 0.5))
            frac_01 = float(np.mean(sigma_z > 0.1))
            
            # Entropy
            p = sigma_z / max(np.sum(sigma_z), 1e-10)
            p = np.clip(p, 1e-10, 1.0)
            ent = float(-np.sum(p * np.log(p)))
            
            # Mean
            mean_sigma = float(np.mean(sigma_z))
            
            # Top-10 concentration
            sorted_sigma = np.sort(sigma_z)[::-1]
            total_sigma = np.sum(sigma_z)
            if total_sigma > 1e-10:
                top10_frac = float(np.sum(sorted_sigma[:10]) / total_sigma)
            else:
                top10_frac = 0.0
            
            gate_frac_05[li].append(frac_05)
            gate_frac_01[li].append(frac_01)
            gate_entropy[li].append(ent)
            gate_mean[li].append(mean_sigma)
            gate_top10_sum[li].append(top10_frac)
    
    # Print results
    print(f"\n  Gate sparsity (n_intermediate={n_intermediate}):")
    print(f"  {'Layer':>6} | {'σ>0.5%':>7} | {'σ>0.1%':>7} | {'mean':>6} | {'entropy':>8} | {'top10%':>7}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
    
    layer_summary = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in range(n_layers):
        if li not in gate_frac_05:
            continue
        avg_05 = np.mean(gate_frac_05[li])
        avg_01 = np.mean(gate_frac_01[li])
        avg_mean = np.mean(gate_mean[li])
        avg_ent = np.mean(gate_entropy[li])
        avg_top10 = np.mean(gate_top10_sum[li])
        
        layer_summary.append({
            "layer": li,
            "frac_activated_05": float(avg_05),
            "frac_activated_01": float(avg_01),
            "mean_sigma": float(avg_mean),
            "entropy": float(avg_ent),
            "top10_concentration": float(avg_top10),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_05:>7.4f} | {avg_01:>7.4f} | {avg_mean:>6.4f} | {avg_ent:>8.2f} | {avg_top10:>7.4f}")
    
    # Analysis
    shallow_05 = np.mean([ls["frac_activated_05"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    mid_05 = np.mean([ls["frac_activated_05"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    deep_05 = np.mean([ls["frac_activated_05"] for ls in layer_summary if ls["layer"] >= n_layers * 0.7])
    
    print(f"\n  Fraction σ>0.5 by depth:")
    print(f"    Shallow: {shallow_05:.4f}")
    print(f"    Mid:     {mid_05:.4f}")
    print(f"    Deep:    {deep_05:.4f}")
    
    if mid_05 < 0.3:
        print(f"  >>> Gate activation is SPARSE (<30% activated)")
    elif mid_05 < 0.5:
        print(f"  >>> Gate activation is MODERATELY SPARSE (30-50%)")
    else:
        print(f"  >>> Gate activation is DENSE (>50%)")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_gate_sparsity.json"
    
    summary = {
        "experiment": "exp1_gate_sparsity",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_intermediate": n_intermediate,
        "layer_summary": layer_summary,
        "shallow_frac_05": float(shallow_05),
        "mid_frac_05": float(mid_05),
        "deep_frac_05": float(deep_05),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: 门控模式与概念类别
# ============================================================
def exp2_gate_concept_overlap(model_name):
    """Measure overlap of gate activation patterns across concepts.
    
    Key measures:
      - Jaccard overlap: |A ∩ B| / |A ∪ B| where A,B are top-k activated neurons
      - Pearson correlation of gate activation vectors
      - Within-category overlap vs across-category overlap
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Gate Pattern Concept Overlap")
    print(f"  Model: {model_name}")
    print(f"  Key test: Do different concepts activate different gate patterns?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    
    template = "The {} is"
    
    # Sample words per category
    rng = np.random.RandomState(42)
    words_per_cat = {}
    for cat, wlist in CONCEPTS.items():
        words_per_cat[cat] = rng.choice(wlist, min(8, len(wlist)), replace=False).tolist()
    
    # Run all words, capture gate activations
    all_gate_acts = {}  # word -> {layer: sigma_z}
    
    for cat, words in words_per_cat.items():
        for word in words:
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1
            
            # Capture post-LN residual
            ln_out = {}
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                if hasattr(layer, 'mlp'):
                    def make_ffn_pre(key):
                        def hook(module, args):
                            if isinstance(args, tuple):
                                ln_out[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                ln_out[key] = args[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
            
            with torch.no_grad():
                _ = model(input_ids)
            
            for h in hooks:
                h.remove()
            
            # Compute gate activations
            word_gate = {}
            for li in range(n_layers):
                key = f"L{li}"
                if key not in ln_out:
                    continue
                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None:
                    continue
                z = lw.W_gate @ ln_out[key]
                z_clipped = np.clip(z, -500, 500)
                sigma_z = 1.0 / (1.0 + np.exp(-z_clipped))
                word_gate[li] = sigma_z
            
            all_gate_acts[word] = word_gate
            print(f"  {word}: done")
    
    # Compute overlaps
    all_words = []
    all_cats = []
    for cat, words in words_per_cat.items():
        for w in words:
            all_words.append(w)
            all_cats.append(cat)
    
    # Pairs: within-category and across-category
    within_pairs = []
    across_pairs = []
    for i in range(len(all_words)):
        for j in range(i+1, len(all_words)):
            if all_cats[i] == all_cats[j]:
                within_pairs.append((all_words[i], all_words[j]))
            else:
                across_pairs.append((all_words[i], all_words[j]))
    
    # Limit across pairs for balance
    if len(across_pairs) > len(within_pairs) * 2:
        rng_sample = np.random.RandomState(123)
        indices = rng_sample.choice(len(across_pairs), len(within_pairs) * 2, replace=False)
        across_pairs = [across_pairs[i] for i in indices]
    
    print(f"\n  Within pairs: {len(within_pairs)}, Across pairs: {len(across_pairs)}")
    
    # Top-k for Jaccard
    k_top = 100
    
    # Compute overlaps per layer
    within_jaccard = defaultdict(list)
    across_jaccard = defaultdict(list)
    within_pearson = defaultdict(list)
    across_pearson = defaultdict(list)
    within_cos_gate = defaultdict(list)
    across_cos_gate = defaultdict(list)
    
    for w1, w2 in within_pairs:
        for li in range(n_layers):
            if li not in all_gate_acts[w1] or li not in all_gate_acts[w2]:
                continue
            g1 = all_gate_acts[w1][li]
            g2 = all_gate_acts[w2][li]
            
            # Jaccard on top-k
            top1 = set(np.argsort(g1)[-k_top:])
            top2 = set(np.argsort(g2)[-k_top:])
            jacc = len(top1 & top2) / max(len(top1 | top2), 1)
            within_jaccard[li].append(jacc)
            
            # Pearson correlation
            cos_g = proper_cos(g1, g2)
            within_cos_gate[li].append(cos_g)
            
            # Binary overlap (both >0.5)
            both_active = np.sum((g1 > 0.5) & (g2 > 0.5))
            either_active = np.sum((g1 > 0.5) | (g2 > 0.5))
            bin_jacc = both_active / max(either_active, 1)
            within_pearson[li].append(bin_jacc)
    
    for w1, w2 in across_pairs:
        for li in range(n_layers):
            if li not in all_gate_acts[w1] or li not in all_gate_acts[w2]:
                continue
            g1 = all_gate_acts[w1][li]
            g2 = all_gate_acts[w2][li]
            
            top1 = set(np.argsort(g1)[-k_top:])
            top2 = set(np.argsort(g2)[-k_top:])
            jacc = len(top1 & top2) / max(len(top1 | top2), 1)
            across_jaccard[li].append(jacc)
            
            cos_g = proper_cos(g1, g2)
            across_cos_gate[li].append(cos_g)
            
            both_active = np.sum((g1 > 0.5) & (g2 > 0.5))
            either_active = np.sum((g1 > 0.5) | (g2 > 0.5))
            bin_jacc = both_active / max(either_active, 1)
            across_pearson[li].append(bin_jacc)
    
    # Print results
    print(f"\n  Gate pattern overlap (top-{k_top} Jaccard):")
    print(f"  {'Layer':>6} | {'W_jacc':>7} | {'X_jacc':>7} | {'W/X':>6} | {'W_cos':>7} | {'X_cos':>7} | {'W_bin':>7} | {'X_bin':>7}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    
    layer_summary = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in range(n_layers):
        wj = np.mean(within_jaccard[li]) if within_jaccard[li] else 0
        xj = np.mean(across_jaccard[li]) if across_jaccard[li] else 0
        wc = np.mean(within_cos_gate[li]) if within_cos_gate[li] else 0
        xc = np.mean(across_cos_gate[li]) if across_cos_gate[li] else 0
        wb = np.mean(within_pearson[li]) if within_pearson[li] else 0
        xb = np.mean(across_pearson[li]) if across_pearson[li] else 0
        ratio = wj / max(xj, 1e-6)
        
        layer_summary.append({
            "layer": li,
            "within_jaccard": float(wj),
            "across_jaccard": float(xj),
            "within_across_ratio": float(ratio),
            "within_cos": float(wc),
            "across_cos": float(xc),
            "within_bin_jaccard": float(wb),
            "across_bin_jaccard": float(xb),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {wj:>7.4f} | {xj:>7.4f} | {ratio:>6.2f} | {wc:>7.4f} | {xc:>7.4f} | {wb:>7.4f} | {xb:>7.4f}")
    
    # Analysis
    mid_wj = np.mean([ls["within_jaccard"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    mid_xj = np.mean([ls["across_jaccard"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    mid_wc = np.mean([ls["within_cos"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    mid_xc = np.mean([ls["across_cos"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    
    print(f"\n  Mid-layer gate overlap:")
    print(f"    Within Jaccard: {mid_wj:.4f}")
    print(f"    Across Jaccard: {mid_xj:.4f}")
    print(f"    W/X ratio: {mid_wj/max(mid_xj, 1e-6):.2f}")
    print(f"    Within cos: {mid_wc:.4f}")
    print(f"    Across cos: {mid_xc:.4f}")
    
    if mid_wj > mid_xj + 0.05:
        print(f"  >>> Within-category gate patterns are MORE similar than across-category")
    else:
        print(f"  >>> Gate patterns are NOT category-specific (within ≈ across)")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_gate_concept_overlap.json"
    
    summary = {
        "experiment": "exp2_gate_concept_overlap",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_top": k_top,
        "n_within_pairs": len(within_pairs),
        "n_across_pairs": len(across_pairs),
        "layer_summary": layer_summary,
        "mid_within_jaccard": float(mid_wj),
        "mid_across_jaccard": float(mid_xj),
        "mid_within_cos": float(mid_wc),
        "mid_across_cos": float(mid_xc),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: 门控差异与概念差异的因果关系
# ============================================================
def exp3_gate_causal_analysis(model_name):
    """Test whether gate pattern differences CAUSE concept signal differences.
    
    Key idea:
      If gate switching is the mechanism, then:
      1. Forcing concept A's gate pattern on concept B's input should
         shift the FFN output toward A
      2. The "gate switching" between A and B should predict the
         FFN output difference
    
    Method:
      1. Get gate activations for A and B: σ_A, σ_B
      2. Compute "gate switch": which neurons switch on/off between A and B
      3. Measure: does the gate switch vector correlate with the FFN output difference?
    
    Simpler approach:
      - FFN_B(h_A) = W_down @ [σ_B ⊙ (W_up @ h̃_A)]  (B's gate on A's input)
      - Compare with FFN_A(h_A) = W_down @ [σ_A ⊙ (W_up @ h̃_A)]  (A's gate on A's input)
      - The difference is purely due to gate switching
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Gate Switching and Concept Signal")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does gate switching CAUSE the FFN output difference?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    
    template = "The {} is"
    
    # Concept pairs
    pairs = [
        ("dog", "cat"),     # within animals
        ("dog", "apple"),   # across categories
        ("hammer", "knife"), # within tools
        ("mountain", "river"), # within nature
    ]
    
    pair_results = []
    
    for wordA, wordB in pairs:
        print(f"\n  Pair: {wordA} -> {wordB}")
        
        # Run both and capture LN output
        def run_get_ln(text):
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1
            ln_out = {}
            ffn_out = {}
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                if hasattr(layer, 'mlp'):
                    def make_ffn_pre(key):
                        def hook(module, args):
                            if isinstance(args, tuple):
                                ln_out[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                ln_out[key] = args[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    def make_ffn_post(key):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                ffn_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                ffn_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
                    hooks.append(layer.mlp.register_forward_hook(make_ffn_post(f"L{li}")))
            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()
            return ln_out, ffn_out
        
        lnA, ffnA = run_get_ln(template.format(wordA))
        lnB, ffnB = run_get_ln(template.format(wordB))
        
        # For each layer, compute:
        # 1. Actual FFN difference: ffnB - ffnA
        # 2. Gate-switching contribution:
        #    FFN_A(h_A) = W_down @ [σ_A ⊙ (W_up @ h̃_A)]
        #    FFN_B(h_A) = W_down @ [σ_B ⊙ (W_up @ h̃_A)]  (B's gate, A's input)
        #    gate_switch_diff = FFN_B(h_A) - FFN_A(h_A)
        # 3. Input contribution:
        #    FFN_A(h_B) = W_down @ [σ_A ⊙ (W_up @ h̃_B)]  (A's gate, B's input)
        #    input_diff = FFN_A(h_B) - FFN_A(h_A)
        
        layer_data = []
        for li in range(n_layers):
            key = f"L{li}"
            if key not in lnA or key not in lnB:
                continue
            if key not in ffnA or key not in ffnB:
                continue
            
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None or lw.W_up is None or lw.W_down is None:
                continue
            
            h_A = lnA[key]  # [d_model]
            h_B = lnB[key]
            
            W_gate = lw.W_gate  # [inter, d_model]
            W_up = lw.W_up      # [inter, d_model]
            W_down = lw.W_down  # [d_model, inter]
            
            # Gate activations
            z_A = W_gate @ h_A
            z_A_clipped = np.clip(z_A, -500, 500)
            sigma_A = 1.0 / (1.0 + np.exp(-z_A_clipped))
            
            z_B = W_gate @ h_B
            z_B_clipped = np.clip(z_B, -500, 500)
            sigma_B = 1.0 / (1.0 + np.exp(-z_B_clipped))
            
            # Up projections
            up_A = W_up @ h_A   # [inter]
            up_B = W_up @ h_B   # [inter]
            
            # FFN outputs
            # FFN_A(h_A) = W_down @ (σ_A ⊙ up_A)
            ffn_A_computed = W_down @ (sigma_A * up_A)
            # FFN_B(h_A) = W_down @ (σ_B ⊙ up_A) — B's gate on A's input
            ffn_B_gate_on_A = W_down @ (sigma_B * up_A)
            # FFN_A(h_B) = W_down @ (σ_A ⊙ up_B) — A's gate on B's input
            ffn_A_gate_on_B = W_down @ (sigma_A * up_B)
            # FFN_B(h_B) = W_down @ (σ_B ⊙ up_B)
            ffn_B_computed = W_down @ (sigma_B * up_B)
            
            # Actual FFN difference
            delta_ffn_actual = ffnB[key] - ffnA[key]
            
            # Gate-switching contribution (same input, different gate)
            delta_gate_switch = ffn_B_gate_on_A - ffn_A_computed
            
            # Input contribution (same gate, different input)
            delta_input = ffn_A_gate_on_B - ffn_A_computed
            
            # Full interaction
            delta_full = ffn_B_computed - ffn_A_computed
            delta_interaction = delta_full - delta_gate_switch - delta_input
            
            # How much does gate-switching explain the actual FFN difference?
            cos_gate_actual = proper_cos(delta_gate_switch, delta_ffn_actual)
            cos_input_actual = proper_cos(delta_input, delta_ffn_actual)
            cos_full_actual = proper_cos(delta_full, delta_ffn_actual)
            
            # Norm ratios
            gate_norm = np.linalg.norm(delta_gate_switch)
            input_norm = np.linalg.norm(delta_input)
            actual_norm = np.linalg.norm(delta_ffn_actual)
            full_norm = np.linalg.norm(delta_full)
            
            gate_frac = gate_norm / max(gate_norm + input_norm, 1e-10)
            
            # Gate switching statistics
            n_switch_on = int(np.sum((sigma_B > 0.5) & (sigma_A <= 0.5)))   # A off -> B on
            n_switch_off = int(np.sum((sigma_A > 0.5) & (sigma_B <= 0.5)))  # A on -> B off
            n_both_on = int(np.sum((sigma_A > 0.5) & (sigma_B > 0.5)))
            n_both_off = int(np.sum((sigma_A <= 0.5) & (sigma_B <= 0.5)))
            
            layer_data.append({
                "layer": li,
                "cos_gate_actual": float(cos_gate_actual),
                "cos_input_actual": float(cos_input_actual),
                "cos_full_actual": float(cos_full_actual),
                "gate_norm": float(gate_norm),
                "input_norm": float(input_norm),
                "actual_norm": float(actual_norm),
                "gate_frac": float(gate_frac),
                "n_switch_on": n_switch_on,
                "n_switch_off": n_switch_off,
                "n_both_on": n_both_on,
                "n_both_off": n_both_off,
                "switch_ratio": float((n_switch_on + n_switch_off) / max(n_both_on + n_switch_on + n_switch_off, 1)),
            })
        
        pair_results.append({
            "pair": f"{wordA}->{wordB}",
            "layer_data": layer_data,
        })
    
    # Print
    for pr in pair_results:
        print(f"\n  Pair: {pr['pair']}")
        print(f"  {'Layer':>6} | {'cos_gate':>9} | {'cos_input':>10} | {'gate%':>6} | {'switch%':>8} | {'n_on':>5} | {'n_off':>6}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}")
        
        sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        
        for ld in pr['layer_data']:
            if ld['layer'] in sample_layers or ld['layer'] == n_layers - 1:
                print(f"  {ld['layer']:>6} | {ld['cos_gate_actual']:>+9.4f} | {ld['cos_input_actual']:>+10.4f} | {ld['gate_frac']:>6.3f} | {ld['switch_ratio']:>8.4f} | {ld['n_switch_on']:>5} | {ld['n_switch_off']:>6}")
    
    # Aggregate
    all_cos_gate = defaultdict(list)
    all_cos_input = defaultdict(list)
    all_gate_frac = defaultdict(list)
    all_switch_ratio = defaultdict(list)
    
    for pr in pair_results:
        for ld in pr['layer_data']:
            li = ld['layer']
            all_cos_gate[li].append(ld['cos_gate_actual'])
            all_cos_input[li].append(ld['cos_input_actual'])
            all_gate_frac[li].append(ld['gate_frac'])
            all_switch_ratio[li].append(ld['switch_ratio'])
    
    print(f"\n  Aggregate (mean across pairs):")
    print(f"  {'Layer':>6} | {'cos_gate':>9} | {'cos_input':>10} | {'gate%':>6} | {'switch%':>8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")
    
    agg_summary = []
    for li in range(n_layers):
        if li not in all_cos_gate:
            continue
        avg_cg = np.mean(all_cos_gate[li])
        avg_ci = np.mean(all_cos_input[li])
        avg_gf = np.mean(all_gate_frac[li])
        avg_sr = np.mean(all_switch_ratio[li])
        
        agg_summary.append({
            "layer": li,
            "cos_gate_actual": float(avg_cg),
            "cos_input_actual": float(avg_ci),
            "gate_frac": float(avg_gf),
            "switch_ratio": float(avg_sr),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_cg:>+9.4f} | {avg_ci:>+10.4f} | {avg_gf:>6.3f} | {avg_sr:>8.4f}")
    
    # Analysis
    mid_cg = np.mean([a["cos_gate_actual"] for a in agg_summary if n_layers * 0.3 <= a["layer"] < n_layers * 0.7])
    mid_ci = np.mean([a["cos_input_actual"] for a in agg_summary if n_layers * 0.3 <= a["layer"] < n_layers * 0.7])
    mid_gf = np.mean([a["gate_frac"] for a in agg_summary if n_layers * 0.3 <= a["layer"] < n_layers * 0.7])
    
    print(f"\n  Mid-layer summary:")
    print(f"    Gate-switching cos with actual: {mid_cg:+.4f}")
    print(f"    Input-difference cos with actual: {mid_ci:+.4f}")
    print(f"    Gate fraction of total: {mid_gf:.4f}")
    
    if mid_cg > mid_ci + 0.05:
        print(f"  >>> Gate switching is the DOMINANT source of FFN output difference")
    elif mid_ci > mid_cg + 0.05:
        print(f"  >>> Input difference is the DOMINANT source")
    else:
        print(f"  >>> Both gate switching and input contribute roughly equally")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_gate_causal.json"
    
    summary = {
        "experiment": "exp3_gate_causal",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "pair_results": pair_results,
        "aggregate_summary": agg_summary,
        "mid_cos_gate": float(mid_cg),
        "mid_cos_input": float(mid_ci),
        "mid_gate_frac": float(mid_gf),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    
    if args.exp in ["1", "all"]:
        exp1_gate_sparsity(args.model)
    
    if args.exp in ["2", "all"]:
        exp2_gate_concept_overlap(args.model)
    
    if args.exp in ["3", "all"]:
        exp3_gate_causal_analysis(args.model)
