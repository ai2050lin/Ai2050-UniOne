"""
Phase CCLXXIII: W_down的"解码"机制 — 如何从门控模式恢复连续语义信号
================================================================
核心问题: 门控模式是高维二值化的(9728-18944维), 但FFN输出是连续的d_model维向量。
W_down如何从二值门控模式中"读出"连续语义信号?

背景:
  INV-56: 门控Hamming距离与语义距离强相关(r=0.94-0.96)
  INV-59: 定量方程 d_H ≈ (n/2)(1-cos)
  INV-44: FFN概念差异>92%来自非线性
  INV-50: 门控切换是FFN差异的主导来源(cos_gate=+0.47)
  INV-55: W_down判别性列不指向概念W_U方向(cos_own≈0)
  
  FFN计算: out = W_down @ (σ(W_gate @ h) ⊙ (W_up @ h))
                      = W_down @ (g ⊙ u)
  
  两个概念A, B的差异:
  Δout = W_down @ (g_A ⊙ u_A - g_B ⊙ u_B)
       = W_down @ (Δg ⊙ ū + ḡ ⊙ Δu + Δg ⊙ Δu)
  
  其中 Δg = g_A - g_B, ḡ = (g_A+g_B)/2, Δu = u_A-u_B, ū = (u_A+u_B)/2

实验:
  Exp1: Δout的分解 — gate项vs up项vs交叉项
        -> Δout ≈ W_down @ (Δg ⊙ ū) + W_down @ (ḡ ⊙ Δu) + W_down @ (Δg ⊙ Δu)
        -> 哪一项占主导? 方向如何?
        -> gate项的方向是否与概念方向对齐?

  Exp2: "有效W_down"分析
        -> 概念A的有效W_down: W_down_eff_A = W_down @ diag(g_A)
        -> 是否 W_down_eff_A @ h_A 的方向对齐概念A?
        -> 不同概念的有效W_down是否不同?

  Exp3: 门控差异Δg的有效维度与信息瓶颈
        -> Δg有多少非零分量?
        -> Δg的PCA有效维度是多少?
        -> 是否存在低维子空间编码概念差异?
        -> W_down在Δg子空间中的行为

用法:
  python phase_cclxxiii_down_decoding.py --model qwen3 --exp 1
  python phase_cclxxiii_down_decoding.py --model qwen3 --exp all
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
# Exp1: Δout的分解 — gate项vs up项vs交叉项
# ============================================================
def exp1_delta_decomposition(model_name):
    """Decompose FFN output difference into gate, up, and cross terms.
    
    Δout = W_down @ (g_A ⊙ u_A - g_B ⊙ u_B)
         = W_down @ (Δg ⊙ ū) + W_down @ (ḡ ⊙ Δu) + W_down @ (Δg ⊙ Δu)
    
    Key questions:
    1. Which term has the largest norm?
    2. Which term's direction aligns with the concept direction?
    3. Is the gate term (Δg ⊙ ū) the dominant source of concept direction?
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Δout Decomposition — Gate vs Up vs Cross")
    print(f"  Model: {model_name}")
    print(f"  Key test: Which term drives the concept direction?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    template = "The {} is"

    # Sample words from each category
    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(6, len(words)), replace=False)
        all_words.extend(sel)
        all_cats.extend([cat] * len(sel))

    # Collect full FFN intermediate activations
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

            h_input = ln_out[key]

            # Gate: g = σ(W_gate @ h)
            z = lw.W_gate @ h_input
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))

            # Up: u = W_up @ h
            u = lw.W_up @ h_input

            # FFN output components
            a = g * u  # activated intermediate [n_inter]
            ffn_out = lw.W_down @ a  # [d_model]

            word_layer_data[li] = {
                "gate": g,
                "up": u,
                "activated": a,
                "ffn_out": ffn_out,
                "residual": res_out.get(key, None),
                "ln_input": h_input,
            }

        word_data[word] = word_layer_data

    print(f"  Collected data for {len(all_words)} words")

    # Target layers
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    # Concept direction: average residual difference between categories
    # For each pair of categories, compute the "concept direction"
    cat_names = list(CONCEPTS.keys())
    
    layer_results = []

    for li in target_layers:
        # Get W_down
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]

        # Compute concept directions from residuals
        cat_means = {}
        for cat in cat_names:
            cat_words = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_data[w]]
            if len(cat_words) < 2:
                continue
            res_vecs = [word_data[w][li]["residual"] for w in cat_words if word_data[w][li]["residual"] is not None]
            if len(res_vecs) < 2:
                continue
            cat_means[cat] = np.mean(res_vecs, axis=0)

        if len(cat_means) < 2:
            continue

        # Test all within-category and cross-category pairs
        same_cat_gate_norms = []
        same_cat_up_norms = []
        same_cat_cross_norms = []
        same_cat_total_norms = []
        same_cat_gate_cos = []
        same_cat_up_cos = []
        same_cat_cross_cos = []

        diff_cat_gate_norms = []
        diff_cat_up_norms = []
        diff_cat_cross_norms = []
        diff_cat_total_norms = []
        diff_cat_gate_cos = []
        diff_cat_up_cos = []
        diff_cat_cross_cos = []

        for i, w1 in enumerate(all_words):
            for j, w2 in enumerate(all_words):
                if i >= j:
                    continue
                if li not in word_data[w1] or li not in word_data[w2]:
                    continue

                c1, c2 = all_cats[i], all_cats[j]

                g1, u1 = word_data[w1][li]["gate"], word_data[w1][li]["up"]
                g2, u2 = word_data[w2][li]["gate"], word_data[w2][li]["up"]
                r1, r2 = word_data[w1][li]["residual"], word_data[w2][li]["residual"]

                if r1 is None or r2 is None:
                    continue

                # Decomposition
                dg = g1 - g2  # Δg
                du = u1 - u2  # Δu
                g_avg = (g1 + g2) / 2  # ḡ
                u_avg = (u1 + u2) / 2  # ū

                # Three terms
                gate_term = W_down @ (dg * u_avg)  # W_down @ (Δg ⊙ ū)
                up_term = W_down @ (g_avg * du)      # W_down @ (ḡ ⊙ Δu)
                cross_term = W_down @ (dg * du)       # W_down @ (Δg ⊙ Δu)
                total_delta = gate_term + up_term + cross_term

                # Concept direction (from residual)
                concept_dir = r1 - r2
                concept_norm = np.linalg.norm(concept_dir)
                if concept_norm < 1e-10:
                    continue

                # Norms of each term
                gate_norm = np.linalg.norm(gate_term)
                up_norm = np.linalg.norm(up_term)
                cross_norm = np.linalg.norm(cross_term)
                total_norm = np.linalg.norm(total_delta)

                # Cosine alignment with concept direction
                gate_cos = proper_cos(gate_term, concept_dir)
                up_cos = proper_cos(up_term, concept_dir)
                cross_cos = proper_cos(cross_term, concept_dir)

                if c1 == c2:
                    same_cat_gate_norms.append(gate_norm)
                    same_cat_up_norms.append(up_norm)
                    same_cat_cross_norms.append(cross_norm)
                    same_cat_total_norms.append(total_norm)
                    same_cat_gate_cos.append(gate_cos)
                    same_cat_up_cos.append(up_cos)
                    same_cat_cross_cos.append(cross_cos)
                else:
                    diff_cat_gate_norms.append(gate_norm)
                    diff_cat_up_norms.append(up_norm)
                    diff_cat_cross_norms.append(cross_norm)
                    diff_cat_total_norms.append(total_norm)
                    diff_cat_gate_cos.append(gate_cos)
                    diff_cat_up_cos.append(up_cos)
                    diff_cat_cross_cos.append(cross_cos)

        # Aggregate
        all_gate_norms = same_cat_gate_norms + diff_cat_gate_norms
        all_up_norms = same_cat_up_norms + diff_cat_up_norms
        all_cross_norms = same_cat_cross_norms + diff_cat_cross_norms
        all_total_norms = same_cat_total_norms + diff_cat_total_norms
        all_gate_cos = same_cat_gate_cos + diff_cat_gate_cos
        all_up_cos = same_cat_up_cos + diff_cat_up_cos
        all_cross_cos = same_cat_cross_cos + diff_cat_cross_cos

        if len(all_gate_norms) < 5:
            continue

        # Fraction of total norm
        total_norm_arr = np.array(all_total_norms)
        gate_frac = np.mean(np.array(all_gate_norms) / np.maximum(total_norm_arr, 1e-10))
        up_frac = np.mean(np.array(all_up_norms) / np.maximum(total_norm_arr, 1e-10))
        cross_frac = np.mean(np.array(all_cross_norms) / np.maximum(total_norm_arr, 1e-10))

        layer_results.append({
            "layer": li,
            "n_pairs": len(all_gate_norms),
            # Fraction of total norm
            "gate_frac": float(gate_frac),
            "up_frac": float(up_frac),
            "cross_frac": float(cross_frac),
            # Norms
            "mean_gate_norm": float(np.mean(all_gate_norms)),
            "mean_up_norm": float(np.mean(all_up_norms)),
            "mean_cross_norm": float(np.mean(all_cross_norms)),
            "mean_total_norm": float(np.mean(all_total_norms)),
            # Cosine alignment with concept direction
            "mean_gate_cos": float(np.mean(all_gate_cos)),
            "mean_up_cos": float(np.mean(all_up_cos)),
            "mean_cross_cos": float(np.mean(all_cross_cos)),
            # Cross-category specific
            "diff_gate_cos": float(np.mean(diff_cat_gate_cos)) if diff_cat_gate_cos else 0,
            "diff_up_cos": float(np.mean(diff_cat_up_cos)) if diff_cat_up_cos else 0,
            "diff_cross_cos": float(np.mean(diff_cat_cross_cos)) if diff_cat_cross_cos else 0,
            # Same-category specific
            "same_gate_cos": float(np.mean(same_cat_gate_cos)) if same_cat_gate_cos else 0,
            "same_up_cos": float(np.mean(same_cat_up_cos)) if same_cat_up_cos else 0,
            "same_cross_cos": float(np.mean(same_cat_cross_cos)) if same_cat_cross_cos else 0,
        })

        lr = layer_results[-1]
        print(f"  L{li}: gate_frac={lr['gate_frac']:.3f}, up_frac={lr['up_frac']:.3f}, "
              f"cross_frac={lr['cross_frac']:.3f}")
        print(f"         gate_cos={lr['mean_gate_cos']:+.3f}, up_cos={lr['mean_up_cos']:+.3f}, "
              f"cross_cos={lr['mean_cross_cos']:+.3f}")
        print(f"         diff_cat: gate_cos={lr['diff_gate_cos']:+.3f}, up_cos={lr['diff_up_cos']:+.3f}")
        print(f"         same_cat: gate_cos={lr['same_gate_cos']:+.3f}, up_cos={lr['same_up_cos']:+.3f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_gate_frac = np.mean([lr["gate_frac"] for lr in mid_results])
        avg_up_frac = np.mean([lr["up_frac"] for lr in mid_results])
        avg_cross_frac = np.mean([lr["cross_frac"] for lr in mid_results])
        avg_gate_cos = np.mean([lr["mean_gate_cos"] for lr in mid_results])
        avg_up_cos = np.mean([lr["mean_up_cos"] for lr in mid_results])
        avg_cross_cos = np.mean([lr["mean_cross_cos"] for lr in mid_results])
        avg_diff_gate_cos = np.mean([lr["diff_gate_cos"] for lr in mid_results])
        avg_diff_up_cos = np.mean([lr["diff_up_cos"] for lr in mid_results])

        print(f"    Mid-layer gate_frac: {avg_gate_frac:.3f}")
        print(f"    Mid-layer up_frac: {avg_up_frac:.3f}")
        print(f"    Mid-layer cross_frac: {avg_cross_frac:.3f}")
        print(f"    Mid-layer gate_cos: {avg_gate_cos:+.3f}")
        print(f"    Mid-layer up_cos: {avg_up_cos:+.3f}")
        print(f"    Mid-layer cross_cos: {avg_cross_cos:+.3f}")
        print(f"    Mid-layer diff_cat gate_cos: {avg_diff_gate_cos:+.3f}")
        print(f"    Mid-layer diff_cat up_cos: {avg_diff_up_cos:+.3f}")

        if avg_gate_frac > avg_up_frac and avg_gate_cos > avg_up_cos:
            print(f"  >>> GATE TERM dominates: larger fraction AND better alignment")
        elif avg_gate_frac > avg_up_frac:
            print(f"  >>> GATE TERM larger fraction, but up term may have better alignment")
        else:
            print(f"  >>> UP TERM dominates or mixed")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_delta_decomposition.json"

    summary = {
        "experiment": "exp1_delta_decomposition",
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
# Exp2: "有效W_down"分析
# ============================================================
def exp2_effective_wdown(model_name):
    """Analyze the "effective W_down" for each concept.
    
    For concept A with gate g_A:
      FFN_out_A = W_down @ (g_A ⊙ u_A) = (W_down @ diag(g_A)) @ u_A = W_down_eff_A @ u_A
    
    Key questions:
    1. Does W_down_eff_A project u_A onto concept A's direction?
    2. Are W_down_eff_A and W_down_eff_B different for different concepts?
    3. What is the rank/effective dimension of W_down_eff?
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Effective W_down Analysis")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does W_down @ diag(g_A) decode concept A?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    template = "The {} is"

    # Sample words
    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(6, len(words)), replace=False)
        all_words.extend(sel)
        all_cats.extend([cat] * len(sel))

    # Collect activations
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

            h_input = ln_out[key]
            z = lw.W_gate @ h_input
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))
            u = lw.W_up @ h_input
            ffn_out = lw.W_down @ (g * u)

            word_layer_data[li] = {
                "gate": g,
                "up": u,
                "ffn_out": ffn_out,
                "residual": res_out.get(key, None),
            }

        word_data[word] = word_layer_data

    print(f"  Collected data for {len(all_words)} words")

    # Target layers
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    cat_names = list(CONCEPTS.keys())
    layer_results = []

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]
        n_inter = W_down.shape[1]

        # Per-category: average gate, average FFN output, average residual
        cat_gates = {}
        cat_ffn_outs = {}
        cat_residuals = {}
        for cat in cat_names:
            cat_words = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_data[w]]
            if len(cat_words) < 2:
                continue
            gates = [word_data[w][li]["gate"] for w in cat_words]
            ffn_outs = [word_data[w][li]["ffn_out"] for w in cat_words]
            residuals = [word_data[w][li]["residual"] for w in cat_words if word_data[w][li]["residual"] is not None]
            if len(residuals) < 2:
                continue
            cat_gates[cat] = np.mean(gates, axis=0)
            cat_ffn_outs[cat] = np.mean(ffn_outs, axis=0)
            cat_residuals[cat] = np.mean(residuals, axis=0)

        if len(cat_gates) < 2:
            continue

        # Test 1: Effective W_down for each category
        # W_down_eff_A = W_down @ diag(g_A)
        # Does W_down_eff_A @ u_A ≈ ffn_out_A? (should be exact)
        # Does W_down_eff_A project u_B differently?

        # Test 2: Cross-category projection
        # For each pair (A, B): does W_down @ diag(g_A) @ u_B differ from W_down @ diag(g_B) @ u_B?
        # The key question: does switching g_A -> g_B change the FFN output direction?

        cross_proj_data = []
        for catA in cat_names:
            if catA not in cat_gates:
                continue
            g_A = cat_gates[catA]
            u_A = None  # We'll use average u for this category
            # Get average u for catA
            catA_words = [w for w, c in zip(all_words, all_cats) if c == catA and li in word_data[w]]
            if len(catA_words) < 2:
                continue
            u_A_avg = np.mean([word_data[w][li]["up"] for w in catA_words], axis=0)
            res_A = cat_residuals[catA]

            for catB in cat_names:
                if catB not in cat_gates:
                    continue
                if catA == catB:
                    continue
                g_B = cat_gates[catB]
                res_B = cat_residuals[catB]

                # Concept direction (from residual)
                concept_dir = res_A - res_B
                concept_norm = np.linalg.norm(concept_dir)
                if concept_norm < 1e-10:
                    continue

                # FFN output using g_A with u_A: W_down @ (g_A ⊙ u_A)
                out_gate_A = W_down @ (g_A * u_A_avg)

                # FFN output using g_B with u_A: W_down @ (g_B ⊙ u_A)
                out_gate_B_on_A = W_down @ (g_B * u_A_avg)

                # The difference from switching gates
                gate_switch_diff = out_gate_A - out_gate_B_on_A

                # Does this difference align with concept direction?
                gate_switch_cos = proper_cos(gate_switch_diff, concept_dir)

                # Also: W_down @ (g_A ⊙ u_A) vs W_down @ (g_A ⊙ u_A_avg)
                # (just sanity check: should be similar)

                cross_proj_data.append({
                    "pair": f"{catA}->{catB}",
                    "gate_switch_cos": gate_switch_cos,
                    "gate_switch_norm": float(np.linalg.norm(gate_switch_diff)),
                    "out_A_norm": float(np.linalg.norm(out_gate_A)),
                    "out_B_on_A_norm": float(np.linalg.norm(out_gate_B_on_A)),
                })

        # Test 3: Rank of effective W_down
        # Average gate across all words
        all_gates = [word_data[w][li]["gate"] for w in all_words if li in word_data[w]]
        if len(all_gates) < 2:
            continue
        avg_gate = np.mean(all_gates, axis=0)

        # W_down_eff = W_down @ diag(avg_gate)
        W_down_eff = W_down * avg_gate[np.newaxis, :]  # [d_model, n_inter]

        # SVD for rank estimation
        from scipy.sparse.linalg import svds
        k_svd = min(100, min(W_down_eff.shape) - 1)
        try:
            U_eff, s_eff, Vt_eff = svds(W_down_eff.astype(np.float32), k=k_svd)
            s_eff = np.sort(s_eff)[::-1]
            total_energy = np.sum(s_eff ** 2)
            cum_energy = np.cumsum(s_eff ** 2) / total_energy
            n90 = int(np.searchsorted(cum_energy, 0.90)) + 1
            n95 = int(np.searchsorted(cum_energy, 0.95)) + 1
            top5_var = float(np.sum(s_eff[:5] ** 2) / total_energy)
            top10_var = float(np.sum(s_eff[:10] ** 2) / total_energy)
        except:
            n90, n95, top5_var, top10_var = -1, -1, -1, -1

        # Test 4: Are effective W_down columns different per category?
        # Compute W_down_eff for each category and compare
        cat_W_effs = {}
        for cat in cat_names:
            if cat not in cat_gates:
                continue
            cat_W_effs[cat] = W_down * cat_gates[cat][np.newaxis, :]

        # Pairwise cosine between category-specific W_down_eff column spaces
        cat_pair_cos = {}
        for catA in cat_names:
            for catB in cat_names:
                if catA >= catB:
                    continue
                if catA not in cat_W_effs or catB not in cat_W_effs:
                    continue
                # Compare column space: use Frobenius inner product / norm
                WA = cat_W_effs[catA]
                WB = cat_W_effs[catB]
                # Subspace similarity: Tr(WA^T @ WB @ WB^T @ WA) / (||WA||_F * ||WB||_F)
                # Simplified: cos between flattened matrices
                cos_mat = proper_cos(WA.flatten(), WB.flatten())
                cat_pair_cos[f"{catA}-{catB}"] = cos_mat

        # Aggregate cross-projection results
        if cross_proj_data:
            avg_gate_switch_cos = np.mean([d["gate_switch_cos"] for d in cross_proj_data])
            avg_gate_switch_norm = np.mean([d["gate_switch_norm"] for d in cross_proj_data])
        else:
            avg_gate_switch_cos = 0
            avg_gate_switch_norm = 0

        layer_results.append({
            "layer": li,
            "n_inter": n_inter,
            # Effective W_down rank
            "n90_eff": n90,
            "n95_eff": n95,
            "top5_var_eff": top5_var,
            "top10_var_eff": top10_var,
            # Cross-projection
            "avg_gate_switch_cos": float(avg_gate_switch_cos),
            "avg_gate_switch_norm": float(avg_gate_switch_norm),
            "cross_proj_data": cross_proj_data[:8],  # save a few examples
            # Category pair cosines
            "cat_pair_cos": cat_pair_cos,
        })

        lr = layer_results[-1]
        print(f"  L{li}: n90_eff={lr['n90_eff']}, top5_var_eff={lr['top5_var_eff']:.4f}, "
              f"gate_switch_cos={lr['avg_gate_switch_cos']:+.3f}, "
              f"gate_switch_norm={lr['avg_gate_switch_norm']:.1f}")
        print(f"         cat_pair_cos: {cat_pair_cos}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_n90 = np.mean([lr["n90_eff"] for lr in mid_results if lr["n90_eff"] > 0])
        avg_top5 = np.mean([lr["top5_var_eff"] for lr in mid_results if lr["top5_var_eff"] > 0])
        avg_gsc = np.mean([lr["avg_gate_switch_cos"] for lr in mid_results])
        avg_gsn = np.mean([lr["avg_gate_switch_norm"] for lr in mid_results])

        print(f"    Mid-layer avg n90_eff: {avg_n90:.0f}")
        print(f"    Mid-layer avg top5_var_eff: {avg_top5:.4f}")
        print(f"    Mid-layer avg gate_switch_cos: {avg_gsc:+.3f}")
        print(f"    Mid-layer avg gate_switch_norm: {avg_gsn:.1f}")

        if avg_gsc > 0.3:
            print(f"  >>> Switching gate pattern changes FFN output direction toward concept direction!")
        else:
            print(f"  >>> Gate switching effect on output direction is WEAK")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_effective_wdown.json"

    summary = {
        "experiment": "exp2_effective_wdown",
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
# Exp3: 门控差异Δg的有效维度与信息瓶颈
# ============================================================
def exp3_gate_delta_dimensionality(model_name):
    """Analyze the effective dimensionality of gate difference Δg.
    
    Key questions:
    1. How many components of Δg are non-zero (on average)?
    2. What is the PCA effective dimension of Δg across all concept pairs?
    3. Does Δg have low-dimensional structure?
    4. How does W_down project Δg into d_model space?
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Gate Difference Δg Dimensionality & Information Bottleneck")
    print(f"  Model: {model_name}")
    print(f"  Key test: Is Δg low-dimensional? How does W_down decode it?")
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

    # Collect gate activations
    word_gates = {}

    for word in all_words:
        text = template.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

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

        word_gates[word] = {}
        for li in range(n_layers):
            key = f"L{li}"
            if key not in ln_out:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            z = lw.W_gate @ ln_out[key]
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))
            word_gates[word][li] = g

    print(f"  Collected gates for {len(all_words)} words")

    # Target layers
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]
        n_inter = W_down.shape[1]

        # Compute all Δg pairs
        valid_words = [w for w in all_words if li in word_gates[w]]
        if len(valid_words) < 4:
            continue

        delta_g_list = []
        delta_g_same_cat = []
        delta_g_diff_cat = []

        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                c1 = all_cats[all_words.index(w1)]
                c2 = all_cats[all_words.index(w2)]

                dg = word_gates[w1][li] - word_gates[w2][li]
                delta_g_list.append(dg)

                if c1 == c2:
                    delta_g_same_cat.append(dg)
                else:
                    delta_g_diff_cat.append(dg)

        if len(delta_g_list) < 10:
            continue

        delta_g_arr = np.array(delta_g_list)  # [n_pairs, n_inter]

        # 1. Sparsity of Δg
        nonzero_per_pair = np.sum(np.abs(delta_g_arr) > 1e-6, axis=1)
        significant_per_pair = np.sum(np.abs(delta_g_arr) > 0.01, axis=1)
        strong_per_pair = np.sum(np.abs(delta_g_arr) > 0.1, axis=1)

        # 2. PCA of Δg
        from scipy.sparse.linalg import svds
        n_pca = min(100, delta_g_arr.shape[0] - 1, delta_g_arr.shape[1] - 1)
        if n_pca < 2:
            continue

        # Center the data
        delta_g_centered = delta_g_arr - delta_g_arr.mean(axis=0)

        # Covariance matrix: too large for full, use randomized SVD
        # Use delta_g_centered @ delta_g_centered.T / n_pairs for [n_pairs, n_pairs] matrix
        # Then eigendecompose to get principal directions in n_inter space

        # More efficient: SVD of delta_g_centered
        try:
            # Use truncated SVD on the centered data
            U_dg, s_dg, Vt_dg = svds(delta_g_centered.astype(np.float32), k=n_pca)
            s_dg = np.sort(s_dg)[::-1]
            Vt_dg = Vt_dg[np.argsort(-svds(delta_g_centered.astype(np.float32), k=n_pca)[1])]

            total_var = np.sum(s_dg ** 2)
            cum_var = np.cumsum(s_dg ** 2) / total_var
            n90_dg = int(np.searchsorted(cum_var, 0.90)) + 1
            n95_dg = int(np.searchsorted(cum_var, 0.95)) + 1
            top5_var_dg = float(np.sum(s_dg[:5] ** 2) / total_var)
            top10_var_dg = float(np.sum(s_dg[:10] ** 2) / total_var)
        except:
            n90_dg, n95_dg, top5_var_dg, top10_var_dg = -1, -1, -1, -1

        # 3. W_down projection of Δg
        # Project Δg through W_down: W_down @ Δg^T = [d_model, n_pairs]
        # This tells us what direction in d_model space each Δg maps to
        wdown_delta = (W_down @ delta_g_arr.T).T  # [n_pairs, d_model]

        # Compare: Δg through W_down vs Δg's actual variance structure
        wdown_delta_norm = np.linalg.norm(wdown_delta, axis=1)
        delta_g_norm = np.linalg.norm(delta_g_arr, axis=1)

        # Correlation: ||W_down @ Δg|| vs ||Δg||
        if np.std(wdown_delta_norm) > 1e-10 and np.std(delta_g_norm) > 1e-10:
            norm_corr = float(np.corrcoef(wdown_delta_norm, delta_g_norm)[0, 1])
        else:
            norm_corr = 0

        # 4. Δg's "active subspace" in W_down
        # Which W_down columns (intermediate neurons) does Δg most activate?
        # Average |Δg| across all pairs
        avg_abs_dg = np.mean(np.abs(delta_g_arr), axis=0)  # [n_inter]

        # Correlate with W_down column norms
        wdown_col_norms = np.linalg.norm(W_down, axis=0)  # [n_inter]
        if np.std(avg_abs_dg) > 1e-10 and np.std(wdown_col_norms) > 1e-10:
            col_corr = float(np.corrcoef(avg_abs_dg, wdown_col_norms)[0, 1])
        else:
            col_corr = 0

        # 5. Same vs diff category Δg
        same_nonzero = float(np.mean([np.sum(np.abs(dg) > 0.01) for dg in delta_g_same_cat])) if delta_g_same_cat else 0
        diff_nonzero = float(np.mean([np.sum(np.abs(dg) > 0.01) for dg in delta_g_diff_cat])) if delta_g_diff_cat else 0

        layer_results.append({
            "layer": li,
            "n_inter": n_inter,
            "n_pairs": len(delta_g_list),
            # Sparsity
            "mean_nonzero_per_pair": float(np.mean(nonzero_per_pair)),
            "mean_significant_per_pair": float(np.mean(significant_per_pair)),
            "mean_strong_per_pair": float(np.mean(strong_per_pair)),
            "frac_nonzero": float(np.mean(nonzero_per_pair) / n_inter),
            "frac_significant": float(np.mean(significant_per_pair) / n_inter),
            "frac_strong": float(np.mean(strong_per_pair) / n_inter),
            # PCA
            "n90_dg": n90_dg,
            "n95_dg": n95_dg,
            "top5_var_dg": top5_var_dg,
            "top10_var_dg": top10_var_dg,
            # W_down projection
            "norm_corr_wdown_dg": norm_corr,
            "mean_wdown_delta_norm": float(np.mean(wdown_delta_norm)),
            "mean_delta_g_norm": float(np.mean(delta_g_norm)),
            # Column correlation
            "col_corr_dg_wdown": col_corr,
            # Same vs diff
            "same_nonzero": same_nonzero,
            "diff_nonzero": diff_nonzero,
            "diff_same_ratio": float(diff_nonzero / max(same_nonzero, 1)),
        })

        lr = layer_results[-1]
        print(f"  L{li}: frac_sig={lr['frac_significant']:.3f}, frac_strong={lr['frac_strong']:.3f}, "
              f"n90_dg={lr['n90_dg']}, top5_var_dg={lr['top5_var_dg']:.4f}")
        print(f"         norm_corr={lr['norm_corr_wdown_dg']:.3f}, "
              f"col_corr={lr['col_corr_dg_wdown']:.3f}, "
              f"diff/same={lr['diff_same_ratio']:.2f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        avg_frac_sig = np.mean([lr["frac_significant"] for lr in mid_results])
        avg_frac_strong = np.mean([lr["frac_strong"] for lr in mid_results])
        avg_n90 = np.mean([lr["n90_dg"] for lr in mid_results if lr["n90_dg"] > 0])
        avg_top5 = np.mean([lr["top5_var_dg"] for lr in mid_results if lr["top5_var_dg"] > 0])
        avg_norm_corr = np.mean([lr["norm_corr_wdown_dg"] for lr in mid_results])
        avg_col_corr = np.mean([lr["col_corr_dg_wdown"] for lr in mid_results])

        print(f"    Mid-layer avg frac_significant: {avg_frac_sig:.3f}")
        print(f"    Mid-layer avg frac_strong: {avg_frac_strong:.3f}")
        print(f"    Mid-layer avg n90(Δg): {avg_n90:.0f}")
        print(f"    Mid-layer avg top5_var(Δg): {avg_top5:.4f}")
        print(f"    Mid-layer avg ||W_down·Δg|| corr: {avg_norm_corr:.3f}")
        print(f"    Mid-layer avg |Δg|-W_down col corr: {avg_col_corr:.3f}")

        if avg_top5 > 0.3:
            print(f"  >>> Δg is LOW-DIMENSIONAL: top5_var={avg_top5:.3f}")
        else:
            print(f"  >>> Δg is HIGH-DIMENSIONAL: top5_var={avg_top5:.3f}")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_gate_delta_dimensionality.json"

    summary = {
        "experiment": "exp3_gate_delta_dimensionality",
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
        exp1_delta_decomposition(args.model)

    if args.exp in ["2", "all"]:
        exp2_effective_wdown(args.model)

    if args.exp in ["3", "all"]:
        exp3_gate_delta_dimensionality(args.model)
