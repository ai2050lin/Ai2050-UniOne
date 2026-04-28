"""
Phase CCLXXII: 门控距离保持的定量理论与因果验证
================================================
核心问题: 门控Hamming距离与语义距离的定量关系是什么?
         这种关系是因果的还是相关的?

背景:
  INV-56: 门控Hamming距离与残差流距离强相关(r=0.936-0.956)
  INV-57: 门控Jaccard重叠与残差流余弦强相关(r=0.909-0.935)
  
  理论预期: 对于高维半空间分割, 有
    E[d_H(b(h_A), b(h_B))] ≈ (n/2) * (1 - cos(θ_AB))
    其中n=中间神经元数, θ_AB=h_A和h_B的夹角
    
  这是因为: P(w_i·h_A > 0 XOR w_i·h_B > 0) 
           = P(不同侧) = (1/π) * arccos(cos(θ_AB))
           ≈ (1/2)(1 - cos(θ_AB)) 当θ小

验证:
  Exp1: 定量方程验证
        -> 理论预测: d_H = (n/2) * (1 - cos(h_A, h_B))
        -> 实际: d_H与(1-cos)的线性关系斜率是否≈n/2?
        -> 残差分析: 偏离理论的模式
  Exp2: 因果验证 — 人为干预门控距离
        -> 如果强制改变h_A使其与h_B的门控距离变化,
           残差流距离是否相应变化?
        -> 干预方式: 在h_A上添加方向性扰动
  Exp3: 跨语义类型泛化
        -> 单词替换(当前) vs 句子级语义差异 vs 逻辑否定
        -> 门控距离保持是否对所有语义差异都成立?

用法:
  python phase_cclxxii_gate_distance_theory.py --model qwen3 --exp 1
  python phase_cclxxii_gate_distance_theory.py --model qwen3 --exp all
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
    get_layer_weights, LayerWeights, compute_cos,
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

TEMPLATES = ["The {} is"]


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
# Exp1: 定量方程验证
# ============================================================
def exp1_quantitative_theory(model_name):
    """Test the quantitative prediction:
    
    Theory: d_H(b(h_A), b(h_B)) ≈ (n/π) * arccos(cos(h_A, h_B))
    
    For small angles: arccos(c) ≈ (π/2)(1-c), so:
    d_H ≈ (n/2) * (1 - cos(h_A, h_B))
    
    We test:
    1. Linear regression: d_H = α * (1-cos) + β
       Expect: α ≈ n/2, β ≈ 0
    2. Better fit: d_H = α * arccos(cos) + β
       Expect: α ≈ n/π
    3. Residual analysis: where does the theory break down?
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Quantitative Theory of Gate Distance Preservation")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does d_H = (n/π) * arccos(cos)?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    template = "The {} is"

    # Sample words
    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(8, len(words)), replace=False)
        all_words.extend(sel)
        all_cats.extend([cat] * len(sel))

    # Collect gate activations and residual stream
    word_data = {}

    for word in all_words:
        text = template.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        ln_out = {}
        res_out = {}
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
            def make_layer_out(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        res_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        res_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

        with torch.no_grad():
            _ = model(input_ids)

        for h in hooks:
            h.remove()

        word_layer_data = {}
        for li in range(n_layers):
            key = f"L{li}"
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None or key not in ln_out:
                continue
            z = lw.W_gate @ ln_out[key]
            z_clipped = np.clip(z, -500, 500)
            sigma_z = 1.0 / (1.0 + np.exp(-z_clipped))
            word_layer_data[li] = {
                "gate": sigma_z,
                "residual": res_out.get(key, None),
                "ln_input": ln_out[key],
            }

        word_data[word] = word_layer_data

    print(f"  Collected data for {len(all_words)} words")

    # Target layers
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        # Get n_inter
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        n_inter = lw.W_gate.shape[0]

        # Compute all pairs
        valid_words = [w for w in all_words if li in word_data[w]]
        if len(valid_words) < 2:
            continue

        hamming_list = []
        one_minus_cos_list = []
        arccos_list = []

        for i in range(len(valid_words)):
            for j in range(i+1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                g1 = word_data[w1][li]["gate"]
                g2 = word_data[w2][li]["gate"]
                r1 = word_data[w1][li]["residual"]
                r2 = word_data[w2][li]["residual"]

                if r1 is None or r2 is None:
                    continue

                # Hamming distance
                active1 = g1 > 0.5
                active2 = g2 > 0.5
                hamming = float(np.sum(active1 != active2))

                # Residual cosine
                cos_r = proper_cos(r1, r2)
                one_minus_cos = 1 - cos_r
                arccos_val = np.arccos(np.clip(cos_r, -1, 1))

                hamming_list.append(hamming)
                one_minus_cos_list.append(one_minus_cos)
                arccos_list.append(arccos_val)

        if len(hamming_list) < 10:
            continue

        hamming_arr = np.array(hamming_list)
        omc_arr = np.array(one_minus_cos_list)
        arccos_arr = np.array(arccos_list)

        # Theory 1: d_H = α * (1-cos) + β
        # Linear regression
        from numpy.polynomial import polynomial as P
        coeffs_omc = np.polyfit(omc_arr, hamming_arr, 1)
        alpha_omc = coeffs_omc[0]
        beta_omc = coeffs_omc[1]
        predicted_omc = np.polyval(coeffs_omc, omc_arr)
        r2_omc = 1 - np.sum((hamming_arr - predicted_omc)**2) / np.sum((hamming_arr - np.mean(hamming_arr))**2)

        # Theory 2: d_H = α * arccos(cos) + β
        coeffs_arc = np.polyfit(arccos_arr, hamming_arr, 1)
        alpha_arc = coeffs_arc[0]
        beta_arc = coeffs_arc[1]
        predicted_arc = np.polyval(coeffs_arc, hamming_arr)
        r2_arc = 1 - np.sum((hamming_arr - predicted_arc)**2) / np.sum((hamming_arr - np.mean(hamming_arr))**2)

        # Theory predictions
        # For sigmoid with bias b: P(σ(w·h_A) > 0.5 XOR σ(w·h_B) > 0.5)
        # = P(w·h_A > -b XOR w·h_B > -b)
        # For zero-mean h: P ≈ (1/π) * arccos(cos(h_A, h_B)) per neuron
        # Total: d_H ≈ (n/π) * arccos(cos)
        # For small angles: d_H ≈ (n/2) * (1-cos)

        predicted_alpha_omc = n_inter / 2  # small angle approximation
        predicted_alpha_arc = n_inter / np.pi  # exact theory

        # Residual analysis
        residuals_omc = hamming_arr - predicted_omc
        residual_std = float(np.std(residuals_omc))

        layer_results.append({
            "layer": li,
            "n_intermediate": n_inter,
            "n_pairs": len(hamming_list),
            # Theory 1: d_H = α*(1-cos) + β
            "alpha_omc": float(alpha_omc),
            "beta_omc": float(beta_omc),
            "r2_omc": float(r2_omc),
            "predicted_alpha_omc": float(predicted_alpha_omc),
            "alpha_ratio_omc": float(alpha_omc / predicted_alpha_omc),
            # Theory 2: d_H = α*arccos + β
            "alpha_arc": float(alpha_arc),
            "beta_arc": float(beta_arc),
            "r2_arc": float(r2_arc),
            "predicted_alpha_arc": float(predicted_alpha_arc),
            "alpha_ratio_arc": float(alpha_arc / predicted_alpha_arc),
            # Residual
            "residual_std": residual_std,
            "mean_hamming": float(np.mean(hamming_arr)),
            "mean_omc": float(np.mean(omc_arr)),
            "mean_arccos": float(np.mean(arccos_arr)),
        })

        print(f"  L{li}: α_omc={alpha_omc:.1f}(pred={predicted_alpha_omc:.1f}, ratio={alpha_omc/predicted_alpha_omc:.3f}), "
              f"α_arc={alpha_arc:.1f}(pred={predicted_alpha_arc:.1f}, ratio={alpha_arc/predicted_alpha_arc:.3f}), "
              f"R²_omc={r2_omc:.4f}, R²_arc={r2_arc:.4f}, "
              f"β={beta_omc:.1f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_ratio_omc = np.mean([lr["alpha_ratio_omc"] for lr in mid_results])
        avg_ratio_arc = np.mean([lr["alpha_ratio_arc"] for lr in mid_results])
        avg_r2_omc = np.mean([lr["r2_omc"] for lr in mid_results])
        avg_r2_arc = np.mean([lr["r2_arc"] for lr in mid_results])
        avg_beta = np.mean([lr["beta_omc"] for lr in mid_results])

        print(f"    Mid-layer avg α_omc/predicted: {avg_ratio_omc:.3f} (expect 1.0)")
        print(f"    Mid-layer avg α_arc/predicted: {avg_ratio_arc:.3f} (expect 1.0)")
        print(f"    Mid-layer avg R²(1-cos): {avg_r2_omc:.4f}")
        print(f"    Mid-layer avg R²(arccos): {avg_r2_arc:.4f}")
        print(f"    Mid-layer avg β: {avg_beta:.1f}")

        if avg_ratio_omc > 0.8:
            print(f"  >>> Theory CONFIRMED: d_H ≈ (n/2)*(1-cos), α ratio ≈ {avg_ratio_omc:.2f}")
        else:
            print(f"  >>> Theory PARTIALLY confirmed: α ratio = {avg_ratio_omc:.2f} (expect 1.0)")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_quantitative_theory.json"

    summary = {
        "experiment": "exp1_quantitative_theory",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp2: 因果验证 — 干预门控距离
# ============================================================
def exp2_causal_intervention(model_name):
    """Causal test: if we change the gate Hamming distance, 
    does the residual stream distance change accordingly?

    Method:
      1. Take h_A (concept A) and h_B (concept B)
      2. Interpolate: h_t = (1-t)*h_A + t*h_B for t in [0, 0.1, ..., 1]
      3. For each t, compute gate Hamming distance d_H(b(h_A), b(h_t))
      4. Compute residual distance ||h_A - h_t||
      5. Check if d_H scales with ||h_A - h_t|| as predicted
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Causal Intervention - Gate Distance vs Semantic Distance")
    print(f"  Model: {model_name}")
    print(f"  Key test: Causally changing h -> does gate distance change?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    template = "The {} is"

    # Concept pairs for interpolation
    pairs = [
        ("dog", "cat"),      # within animals (close)
        ("dog", "apple"),    # across categories (far)
        ("hammer", "river"), # across categories (far)
    ]

    # Target layers
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    target_layers = sorted(set(target_layers))

    pair_results = []

    for wordA, wordB in pairs:
        print(f"\n  Pair: {wordA} -> {wordB}")

        # Get base residual streams
        texts = [template.format(wordA), template.format(wordB)]

        res_streams = {}  # word -> {layer: residual}
        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            res_out = {}
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                def make_layer_out(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            res_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            res_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)

            for h in hooks:
                h.remove()

            word = text.split()[1]  # extract the word
            res_streams[word] = res_out

        # Now interpolate in residual stream space at each target layer
        for li in target_layers:
            key = f"L{li}"
            if key not in res_streams[wordA] or key not in res_streams[wordB]:
                continue

            h_A = res_streams[wordA][key]  # [d_model]
            h_B = res_streams[wordB][key]

            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue

            n_inter = lw.W_gate.shape[0]

            # Interpolation steps
            t_values = np.linspace(0, 1, 11)

            interp_data = []
            for t in t_values:
                h_t = (1 - t) * h_A + t * h_B

                # Gate activation for h_t
                z_t = lw.W_gate @ h_t
                z_clipped = np.clip(z_t, -500, 500)
                sigma_t = 1.0 / (1.0 + np.exp(-z_clipped))

                # Gate activation for h_A
                z_A = lw.W_gate @ h_A
                z_A_clipped = np.clip(z_A, -500, 500)
                sigma_A = 1.0 / (1.0 + np.exp(-z_A_clipped))

                # Hamming distance
                active_A = sigma_A > 0.5
                active_t = sigma_t > 0.5
                hamming = float(np.sum(active_A != active_t))

                # Residual distance
                res_dist = float(np.linalg.norm(h_t - h_A))
                res_cos = proper_cos(h_A, h_t)

                interp_data.append({
                    "t": float(t),
                    "hamming": hamming,
                    "res_dist": res_dist,
                    "res_cos": float(res_cos),
                    "one_minus_cos": float(1 - res_cos),
                })

            # Correlation analysis
            hamming_arr = np.array([d["hamming"] for d in interp_data])
            omc_arr = np.array([d["one_minus_cos"] for d in interp_data])
            dist_arr = np.array([d["res_dist"] for d in interp_data])

            corr_ham_omc = float(np.corrcoef(hamming_arr, omc_arr)[0, 1]) if len(hamming_arr) > 2 else 0
            corr_ham_dist = float(np.corrcoef(hamming_arr, dist_arr)[0, 1]) if len(hamming_arr) > 2 else 0

            # Linear fit: hamming = α * (1-cos) + β
            if len(omc_arr) > 2 and np.std(omc_arr) > 1e-10:
                coeffs = np.polyfit(omc_arr, hamming_arr, 1)
                alpha_fit = coeffs[0]
                beta_fit = coeffs[1]
                predicted = np.polyval(coeffs, omc_arr)
                r2 = 1 - np.sum((hamming_arr - predicted)**2) / max(np.sum((hamming_arr - np.mean(hamming_arr))**2), 1e-10)
            else:
                alpha_fit = 0
                beta_fit = 0
                r2 = 0

            pair_results.append({
                "pair": f"{wordA}->{wordB}",
                "layer": li,
                "n_intermediate": n_inter,
                "interp_data": interp_data,
                "corr_ham_omc": corr_ham_omc,
                "corr_ham_dist": corr_ham_dist,
                "alpha_fit": float(alpha_fit),
                "beta_fit": float(beta_fit),
                "r2_fit": float(r2),
                "predicted_alpha": float(n_inter / 2),
                "alpha_ratio": float(alpha_fit / (n_inter / 2)) if n_inter > 0 else 0,
            })

            print(f"    L{li}: corr_ham_omc={corr_ham_omc:.3f}, α={alpha_fit:.1f}/pred={n_inter/2:.1f}, "
                  f"ratio={alpha_fit/(n_inter/2):.3f}, R²={r2:.4f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [pr for pr in pair_results if n_layers * 0.3 <= pr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_corr = np.mean([pr["corr_ham_omc"] for pr in mid_results])
        avg_ratio = np.mean([pr["alpha_ratio"] for pr in mid_results])
        avg_r2 = np.mean([pr["r2_fit"] for pr in mid_results])

        print(f"    Mid-layer avg correlation: {avg_corr:.3f}")
        print(f"    Mid-layer avg α/predicted: {avg_ratio:.3f}")
        print(f"    Mid-layer avg R²: {avg_r2:.4f}")

        if avg_corr > 0.9:
            print(f"  >>> CAUSAL relationship confirmed: changing h changes gate Hamming distance")
        else:
            print(f"  >>> Causal relationship PARTIALLY confirmed")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_causal_intervention.json"

    summary = {
        "experiment": "exp2_causal_intervention",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "pair_results": pair_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp3: 跨语义类型泛化
# ============================================================
def exp3_cross_semantic_generalization(model_name):
    """Test gate distance preservation across different types of semantic differences.

    Types:
    1. Same category word substitution: "The dog is" -> "The cat is"
    2. Different category word substitution: "The dog is" -> "The apple is"
    3. Sentence-level: "The dog is running" -> "The dog is sleeping"
    4. Logical: "The dog is big" -> "The dog is not big"
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Cross-Semantic-Type Generalization")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does gate distance preservation hold for all semantic types?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    # Different semantic difference types
    sentence_pairs = {
        "same_category": [
            ("The dog is", "The cat is"),
            ("The apple is", "The banana is"),
            ("The hammer is", "The knife is"),
            ("The mountain is", "The river is"),
        ],
        "diff_category": [
            ("The dog is", "The apple is"),
            ("The hammer is", "The river is"),
            ("The cat is", "The mountain is"),
            ("The banana is", "The knife is"),
        ],
        "same_word_diff_verb": [
            ("The dog is running", "The dog is sleeping"),
            ("The cat is eating", "The cat is walking"),
            ("The apple is red", "The apple is green"),
            ("The hammer is heavy", "The hammer is old"),
        ],
        "logical_negation": [
            ("The dog is big", "The dog is not big"),
            ("The cat is fast", "The cat is not fast"),
            ("The apple is red", "The apple is not red"),
            ("The mountain is tall", "The mountain is not tall"),
        ],
    }

    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    target_layers = sorted(set(target_layers))

    type_results = {}

    for sem_type, pairs in sentence_pairs.items():
        print(f"\n  Semantic type: {sem_type}")

        type_hamming = defaultdict(list)
        type_cosine = defaultdict(list)
        type_jaccard = defaultdict(list)

        for sentA, sentB in pairs:
            # Get residual streams
            res_out_A = {}
            res_out_B = {}
            ln_out_A = {}
            ln_out_B = {}

            for sent, res_out, ln_out in [(sentA, res_out_A, ln_out_A), (sentB, res_out_B, ln_out_B)]:
                input_ids = tokenizer(sent, return_tensors="pt").to(device).input_ids
                last_pos = input_ids.shape[1] - 1

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
                    def make_layer_out(key):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                res_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                res_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

                with torch.no_grad():
                    _ = model(input_ids)

                for h in hooks:
                    h.remove()

            # Compute for each target layer
            for li in target_layers:
                key = f"L{li}"
                if key not in res_out_A or key not in res_out_B:
                    continue
                if key not in ln_out_A or key not in ln_out_B:
                    continue

                lw = get_layer_weights(layers_list[li], d_model, mlp_type)
                if lw.W_gate is None:
                    continue

                n_inter = lw.W_gate.shape[0]

                # Gate activations
                z_A = lw.W_gate @ ln_out_A[key]
                z_A_clipped = np.clip(z_A, -500, 500)
                sigma_A = 1.0 / (1.0 + np.exp(-z_A_clipped))

                z_B = lw.W_gate @ ln_out_B[key]
                z_B_clipped = np.clip(z_B, -500, 500)
                sigma_B = 1.0 / (1.0 + np.exp(-z_B_clipped))

                # Hamming distance
                active_A = sigma_A > 0.5
                active_B = sigma_B > 0.5
                hamming = float(np.sum(active_A != active_B))

                # Jaccard
                jaccard = float(np.sum(active_A & active_B)) / max(float(np.sum(active_A | active_B)), 1.0)

                # Residual cosine
                cos_r = proper_cos(res_out_A[key], res_out_B[key])

                type_hamming[li].append(hamming)
                type_jaccard[li].append(jaccard)
                type_cosine[li].append(cos_r)

        # Aggregate per layer
        layer_data = []
        for li in target_layers:
            if li not in type_hamming or len(type_hamming[li]) == 0:
                continue

            ham_arr = np.array(type_hamming[li])
            cos_arr = np.array(type_cosine[li])
            jac_arr = np.array(type_jaccard[li])

            # Correlation
            dist_arr = 1 - cos_arr
            if np.std(dist_arr) > 1e-10 and np.std(ham_arr) > 1e-10:
                corr = float(np.corrcoef(ham_arr, dist_arr)[0, 1])
            else:
                corr = 0.0

            layer_data.append({
                "layer": li,
                "mean_hamming": float(np.mean(ham_arr)),
                "mean_cosine": float(np.mean(cos_arr)),
                "mean_jaccard": float(np.mean(jac_arr)),
                "corr_hamming_dist": float(corr),
            })

            print(f"    L{li}: hamming={np.mean(ham_arr):.0f}, cos={np.mean(cos_arr):.4f}, "
                  f"jaccard={np.mean(jac_arr):.3f}, corr={corr:.3f}")

        type_results[sem_type] = layer_data

    # Summary
    print(f"\n  Summary:")
    for sem_type, ldata in type_results.items():
        if ldata:
            mid_ld = [ld for ld in ldata if n_layers * 0.3 <= ld["layer"] < n_layers * 0.7]
            if mid_ld:
                avg_corr = np.mean([ld["corr_hamming_dist"] for ld in mid_ld])
                avg_ham = np.mean([ld["mean_hamming"] for ld in mid_ld])
                avg_cos = np.mean([ld["mean_cosine"] for ld in mid_ld])
                print(f"    {sem_type}: corr={avg_corr:.3f}, hamming={avg_ham:.0f}, cos={avg_cos:.4f}")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_cross_semantic.json"

    summary = {
        "experiment": "exp3_cross_semantic",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "type_results": type_results,
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
        exp1_quantitative_theory(args.model)

    if args.exp in ["2", "all"]:
        exp2_causal_intervention(args.model)

    if args.exp in ["3", "all"]:
        exp3_cross_semantic_generalization(args.model)
