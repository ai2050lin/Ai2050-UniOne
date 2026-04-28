"""
Phase CCLXXI: 门控多面体几何与W_down读出机制
================================================
核心问题: W_down如何从高维门控模式中"读出"语义方向?

背景:
  INV-51: W_gate行高维无结构(top5_var=1-1.5%, 无聚类)
  INV-52: 概念特异性门控神经元存在(sel/rnd=42-98x)
  INV-53: W_gate行与W_U行空间无对齐(ratio≈1.0x)
  INV-54: 最后层弱门控-输出对齐(top_cos=0.28-0.43)
  INV-50: 门控切换cos_gate=+0.47主导FFN差异
  INV-47: W_down列空间不与W_U对齐

核心悖论: 
  - W_gate行无低维结构, W_down也不与W_U对齐
  - 但FFN输出确实指向概念方向(cos_nonlinear=+0.59~+0.94)
  - 这意味着"读出"机制必须来自门控模式与W_down的组合效应

验证:
  Exp1: W_down的"读出几何"
        -> W_down行在概念特异性神经元子空间中的投影结构
        -> W_down行是否"知道"哪些神经元对哪个概念敏感?
        -> 概念特异性W_down分量: W_down[:, disc_neurons_A] vs W_down[:, disc_neurons_B]
  Exp2: 门控多面体与概念距离
        -> 两个概念共享多少激活神经元?
        -> 共享神经元 vs 切换神经元 → 语义距离?
        -> 门控模式的Hamming距离与cos(h_A, h_B)的关系
  Exp3: 最后一层的特殊结构
        -> 为什么最后层门控-输出对齐增强?
        -> 最后层的W_down是否有特殊结构?
        -> 概念特异性神经元在最后层的分布

用法:
  python phase_cclxxi_readout_geometry.py --model qwen3 --exp 1
  python phase_cclxxi_readout_geometry.py --model qwen3 --exp all
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
# Exp1: W_down的读出几何
# ============================================================
def exp1_wdown_readout(model_name):
    """Analyze how W_down reads out concept signals from gate patterns.

    Key idea:
      FFN_output = W_down @ (σ(W_gate @ h) ⊙ (W_up @ h))
      
      If neurons {i1, i2, ...} are specific to concept A (high σ for A, low for B),
      then W_down[:, i1], W_down[:, i2], ... should collectively point toward A's direction.
      
    Measures:
      1. For each category, compute W_down_sub = W_down[:, disc_neurons_cat]
         Project this onto W_U direction of category words
      2. Compare: W_down_sub_A · w_U_A vs W_down_sub_A · w_U_B
         (should be: A's disc neurons → A's output, not B's)
      3. Random baseline: same number of random neurons
    """
    print(f"\n{'='*70}")
    print(f"Exp1: W_down Readout Geometry")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does W_down 'know' which neurons are concept-specific?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    W_U = get_W_U(model)  # [vocab_size, d_model]

    # Get token IDs for concept words
    concept_token_ids = {}
    for cat, words in CONCEPTS.items():
        for word in words:
            tok_ids = tokenizer.encode(word, add_special_tokens=False)
            if tok_ids:
                concept_token_ids[(cat, word)] = tok_ids[0]

    # Compute category W_U directions (mean of word embeddings, normalized)
    cat_wu_dirs = {}
    for cat, words in CONCEPTS.items():
        vecs = []
        for word in words:
            key = (cat, word)
            if key in concept_token_ids:
                vecs.append(W_U[concept_token_ids[key]])
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-10:
                cat_wu_dirs[cat] = mean_vec / norm

    template = "The {} is"

    # Collect gate activations
    rng = np.random.RandomState(42)
    words_per_cat = {}
    for cat, wlist in CONCEPTS.items():
        words_per_cat[cat] = rng.choice(wlist, min(8, len(wlist)), replace=False).tolist()

    all_gate_acts = {}  # (cat, word) -> {layer: sigma_z}

    for cat, words in words_per_cat.items():
        for word in words:
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

            all_gate_acts[(cat, word)] = word_gate

    print(f"  Collected gate activations for {len(all_gate_acts)} words")

    # Find discriminative neurons per category per layer
    categories = list(CONCEPTS.keys())
    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        # Compute mean gate activation per category
        cat_means = {}
        for cat in categories:
            acts = [all_gate_acts[(cat, w)][li] for (c, w) in all_gate_acts if c == cat and li in all_gate_acts[(c, w)]]
            if acts:
                cat_means[cat] = np.mean(acts, axis=0)

        if len(cat_means) < 2:
            continue

        n_inter = list(cat_means.values())[0].shape[0]

        # Get W_down for this layer
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_down is None:
            continue
        W_down = lw.W_down  # [d_model, n_inter]

        # Find discriminative neurons for each category
        k_disc = 200  # top-200 discriminative neurons
        cat_disc_neurons = {}
        for cat in categories:
            if cat not in cat_means:
                continue
            mean_this = cat_means[cat]
            other_means = [cat_means[c] for c in categories if c != cat and c in cat_means]
            if not other_means:
                continue
            max_other = np.maximum.reduce(other_means)
            selectivity = mean_this - max_other
            top_idx = np.argsort(selectivity)[::-1][:k_disc]
            cat_disc_neurons[cat] = top_idx.tolist()

        # Key measure: W_down restricted to disc neurons → projection onto W_U
        # For each category A, compute:
        #   readout_A = W_down[:, disc_neurons_A] @ mean_σ(disc_neurons_A)
        # This simulates the "readout" when only disc neurons A are active

        cat_readout_results = {}
        for cat in categories:
            if cat not in cat_disc_neurons or cat not in cat_wu_dirs:
                continue
            disc_idx = cat_disc_neurons[cat]
            w_u_dir = cat_wu_dirs[cat]  # [d_model]

            # Mean gate activation of disc neurons for this category
            mean_gate = cat_means[cat]
            gate_disc = mean_gate[disc_idx]  # [k_disc]

            # W_down restricted to disc neurons: [d_model, k_disc]
            W_down_disc = W_down[:, disc_idx]

            # Readout: W_down_disc @ gate_disc = [d_model]
            readout_vec = W_down_disc @ gate_disc

            # Cosine with own W_U direction vs other W_U directions
            cos_own = proper_cos(readout_vec, w_u_dir)
            cos_others = []
            for other_cat in categories:
                if other_cat != cat and other_cat in cat_wu_dirs:
                    cos_others.append(proper_cos(readout_vec, cat_wu_dirs[other_cat]))
            mean_cos_other = float(np.mean(cos_others)) if cos_others else 0.0

            # Random baseline: same number of random neurons
            rng_rd = np.random.RandomState(42 + li)
            rand_idx = rng_rd.choice(n_inter, k_disc, replace=False).tolist()
            W_down_rand = W_down[:, rand_idx]
            gate_rand = mean_gate[rand_idx]
            readout_rand = W_down_rand @ gate_rand
            cos_rand_own = proper_cos(readout_rand, w_u_dir)
            cos_rand_other = float(np.mean([proper_cos(readout_rand, cat_wu_dirs[oc]) 
                                            for oc in categories if oc != cat and oc in cat_wu_dirs] or [0]))

            # W_down disc columns' direction: mean column direction
            disc_col_norms = np.linalg.norm(W_down_disc, axis=0)
            valid = disc_col_norms > 1e-10
            if np.sum(valid) > 0:
                disc_col_mean = np.mean(W_down_disc[:, valid] / disc_col_norms[valid], axis=1)
                cos_disc_mean_own = proper_cos(disc_col_mean, w_u_dir)
            else:
                cos_disc_mean_own = 0.0

            cat_readout_results[cat] = {
                "cos_own": float(cos_own),
                "mean_cos_other": mean_cos_other,
                "own_vs_other": float(cos_own - mean_cos_other),
                "cos_rand_own": float(cos_rand_own),
                "cos_rand_other": float(cos_rand_other),
                "rand_own_vs_other": float(cos_rand_own - cos_rand_other),
                "cos_disc_col_mean_own": float(cos_disc_mean_own),
            }

        # Aggregate
        avg_cos_own = np.mean([r["cos_own"] for r in cat_readout_results.values()])
        avg_own_vs_other = np.mean([r["own_vs_other"] for r in cat_readout_results.values()])
        avg_rand_own = np.mean([r["cos_rand_own"] for r in cat_readout_results.values()])
        avg_rand_ovs = np.mean([r["rand_own_vs_other"] for r in cat_readout_results.values()])
        avg_disc_col = np.mean([r["cos_disc_col_mean_own"] for r in cat_readout_results.values()])

        layer_results.append({
            "layer": li,
            "n_intermediate": n_inter,
            "cat_readout": cat_readout_results,
            "avg_cos_own": float(avg_cos_own),
            "avg_own_vs_other": float(avg_own_vs_other),
            "avg_rand_own": float(avg_rand_own),
            "avg_rand_own_vs_other": float(avg_rand_ovs),
            "avg_disc_col_own": float(avg_disc_col),
        })

        print(f"  L{li}: cos_own={avg_cos_own:+.4f}, own-other={avg_own_vs_other:+.4f}, "
              f"rand_own={avg_rand_own:+.4f}, disc_col_own={avg_disc_col:+.4f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    deep_results = [lr for lr in layer_results if lr["layer"] >= n_layers * 0.7]
    
    if mid_results:
        m_cos_own = np.mean([lr["avg_cos_own"] for lr in mid_results])
        m_ovs = np.mean([lr["avg_own_vs_other"] for lr in mid_results])
        m_rand = np.mean([lr["avg_rand_own"] for lr in mid_results])
        m_rovs = np.mean([lr["avg_rand_own_vs_other"] for lr in mid_results])
        print(f"    Mid-layer avg cos_own: {m_cos_own:+.4f}")
        print(f"    Mid-layer avg own-other: {m_ovs:+.4f}")
        print(f"    Mid-layer avg rand_own: {m_rand:+.4f}")
        print(f"    Mid-layer avg rand_own-other: {m_rovs:+.4f}")

        if m_ovs > m_rovs + 0.02:
            print(f"  >>> W_down discriminative columns SELECTIVELY point to concept W_U (own>other)")
        else:
            print(f"  >>> W_down disc columns do NOT selectively point to concept W_U")

    if deep_results:
        d_cos_own = np.mean([lr["avg_cos_own"] for lr in deep_results])
        d_ovs = np.mean([lr["avg_own_vs_other"] for lr in deep_results])
        print(f"    Deep-layer avg cos_own: {d_cos_own:+.4f}")
        print(f"    Deep-layer avg own-other: {d_ovs:+.4f}")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_wdown_readout.json"

    summary = {
        "experiment": "exp1_wdown_readout",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "k_disc": k_disc,
        "layer_results": layer_results,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return summary


# ============================================================
# Exp2: 门控多面体与概念距离
# ============================================================
def exp2_gate_polytope_distance(model_name):
    """Analyze gate pattern overlaps and their relation to concept distances.

    Key measures:
      - Hamming distance of gate patterns: how many neurons switch between concepts?
      - Shared active neurons vs exclusive neurons
      - Correlation between gate Hamming distance and residual stream cosine distance
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Gate Polytope and Concept Distance")
    print(f"  Model: {model_name}")
    print(f"  Key test: How does gate pattern distance relate to concept distance?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    template = "The {} is"

    # All words
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        for word in words:
            all_words.append(word)
            all_cats.append(cat)

    # Sample
    rng = np.random.RandomState(42)
    sample_per_cat = 6
    sampled = []
    sampled_cats = []
    for cat in CONCEPTS:
        cat_words = [w for w, c in zip(all_words, all_cats) if c == cat]
        sel = rng.choice(cat_words, min(sample_per_cat, len(cat_words)), replace=False)
        sampled.extend(sel)
        sampled_cats.extend([cat] * len(sel))

    print(f"  Sampled {len(sampled)} words from {len(CONCEPTS)} categories")

    # Run all words and capture gate activations + residual stream
    word_data = {}  # word -> {layer: {"gate": sigma_z, "residual": h}}

    for word, cat in zip(sampled, sampled_cats):
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
            }

        word_data[word] = word_layer_data

    # Pairs
    within_pairs = []
    across_pairs = []
    for i in range(len(sampled)):
        for j in range(i+1, len(sampled)):
            if sampled_cats[i] == sampled_cats[j]:
                within_pairs.append((sampled[i], sampled[j], sampled_cats[i]))
            else:
                across_pairs.append((sampled[i], sampled[j], f"{sampled_cats[i]}-{sampled_cats[j]}"))

    # Limit across pairs
    if len(across_pairs) > len(within_pairs) * 2:
        indices = rng.choice(len(across_pairs), len(within_pairs) * 2, replace=False)
        across_pairs = [across_pairs[i] for i in indices]

    print(f"  Within pairs: {len(within_pairs)}, Across pairs: {len(across_pairs)}")

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    layer_results = []

    for li in target_layers:
        # Compute gate and residual distances for all pairs
        gate_hamming_W = []  # within
        gate_hamming_X = []  # across
        gate_jaccard_W = []
        gate_jaccard_X = []
        res_cos_W = []
        res_cos_X = []

        all_hamming = []
        all_jaccard = []
        all_res_cos = []
        all_pair_type = []  # 0=within, 1=across

        for w1, w2, ptype in within_pairs:
            if li not in word_data[w1] or li not in word_data[w2]:
                continue
            g1 = word_data[w1][li]["gate"]
            g2 = word_data[w2][li]["gate"]
            r1 = word_data[w1][li]["residual"]
            r2 = word_data[w2][li]["residual"]

            # Gate Hamming distance (σ>0.5)
            active1 = g1 > 0.5
            active2 = g2 > 0.5
            hamming = float(np.sum(active1 != active2))
            jaccard = float(np.sum(active1 & active2)) / max(float(np.sum(active1 | active2)), 1.0)

            gate_hamming_W.append(hamming)
            gate_jaccard_W.append(jaccard)

            if r1 is not None and r2 is not None:
                rc = proper_cos(r1, r2)
                res_cos_W.append(rc)
                all_res_cos.append(rc)
            else:
                all_res_cos.append(0.0)

            all_hamming.append(hamming)
            all_jaccard.append(jaccard)
            all_pair_type.append(0)

        for w1, w2, ptype in across_pairs:
            if li not in word_data[w1] or li not in word_data[w2]:
                continue
            g1 = word_data[w1][li]["gate"]
            g2 = word_data[w2][li]["gate"]
            r1 = word_data[w1][li]["residual"]
            r2 = word_data[w2][li]["residual"]

            active1 = g1 > 0.5
            active2 = g2 > 0.5
            hamming = float(np.sum(active1 != active2))
            jaccard = float(np.sum(active1 & active2)) / max(float(np.sum(active1 | active2)), 1.0)

            gate_hamming_X.append(hamming)
            gate_jaccard_X.append(jaccard)

            if r1 is not None and r2 is not None:
                rc = proper_cos(r1, r2)
                res_cos_X.append(rc)
                all_res_cos.append(rc)
            else:
                all_res_cos.append(0.0)

            all_hamming.append(hamming)
            all_jaccard.append(jaccard)
            all_pair_type.append(1)

        # Correlation: gate distance vs residual distance
        if len(all_hamming) > 10:
            hamming_arr = np.array(all_hamming)
            res_arr = np.array(all_res_cos)
            jaccard_arr = np.array(all_jaccard)

            # Hamming vs 1-cos (both are "distances")
            dist_arr = 1 - res_arr
            try:
                corr_hamming = float(np.corrcoef(hamming_arr, dist_arr)[0, 1])
            except:
                corr_hamming = 0.0
            try:
                corr_jaccard = float(np.corrcoef(jaccard_arr, res_arr)[0, 1])
            except:
                corr_jaccard = 0.0
        else:
            corr_hamming = 0.0
            corr_jaccard = 0.0

        avg_ham_W = float(np.mean(gate_hamming_W)) if gate_hamming_W else 0
        avg_ham_X = float(np.mean(gate_hamming_X)) if gate_hamming_X else 0
        avg_jac_W = float(np.mean(gate_jaccard_W)) if gate_jaccard_W else 0
        avg_jac_X = float(np.mean(gate_jaccard_X)) if gate_jaccard_X else 0
        avg_cos_W = float(np.mean(res_cos_W)) if res_cos_W else 0
        avg_cos_X = float(np.mean(res_cos_X)) if res_cos_X else 0

        layer_results.append({
            "layer": li,
            "within_hamming": avg_ham_W,
            "across_hamming": avg_ham_X,
            "within_jaccard": avg_jac_W,
            "across_jaccard": avg_jac_X,
            "within_res_cos": avg_cos_W,
            "across_res_cos": avg_cos_X,
            "corr_hamming_resdist": float(corr_hamming),
            "corr_jaccard_rescos": float(corr_jaccard),
        })

        print(f"  L{li}: hamming(W={avg_ham_W:.0f}, X={avg_ham_X:.0f}), "
              f"jaccard(W={avg_jac_W:.3f}, X={avg_jac_X:.3f}), "
              f"res_cos(W={avg_cos_W:.4f}, X={avg_cos_X:.4f}), "
              f"corr_ham={corr_hamming:.3f}, corr_jac={corr_jaccard:.3f}")

    # Summary
    print(f"\n  Summary:")
    mid_results = [lr for lr in layer_results if n_layers * 0.3 <= lr["layer"] < n_layers * 0.7]
    if mid_results:
        m_corr_ham = np.mean([lr["corr_hamming_resdist"] for lr in mid_results])
        m_corr_jac = np.mean([lr["corr_jaccard_rescos"] for lr in mid_results])
        m_ham_W = np.mean([lr["within_hamming"] for lr in mid_results])
        m_ham_X = np.mean([lr["across_hamming"] for lr in mid_results])
        m_jac_W = np.mean([lr["within_jaccard"] for lr in mid_results])
        m_jac_X = np.mean([lr["across_jaccard"] for lr in mid_results])

        print(f"    Mid-layer avg Hamming(W={m_ham_W:.0f}, X={m_ham_X:.0f}), ratio={m_ham_X/max(m_ham_W,1):.2f}")
        print(f"    Mid-layer avg Jaccard(W={m_jac_W:.3f}, X={m_jac_X:.3f})")
        print(f"    Mid-layer avg corr(hamming, res_dist): {m_corr_ham:.3f}")
        print(f"    Mid-layer avg corr(jaccard, res_cos): {m_corr_jac:.3f}")

        if m_corr_ham > 0.3:
            print(f"  >>> Gate Hamming distance MODERATELY correlates with concept distance")
        elif m_corr_ham > 0.1:
            print(f"  >>> Gate Hamming distance WEAKLY correlates with concept distance")
        else:
            print(f"  >>> Gate Hamming distance does NOT correlate with concept distance")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_gate_polytope.json"

    summary = {
        "experiment": "exp2_gate_polytope",
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
# Exp3: 最后一层的特殊结构
# ============================================================
def exp3_last_layer_structure(model_name):
    """Analyze the special structure of the last layer.

    Background:
      INV-54: Last layer has stronger gate-output alignment (top_cos=0.28-0.43)
      GLM4 L39: sil=0.91 extreme clustering

    Measures:
      1. Last layer W_gate row structure (detailed)
      2. Last layer W_down structure
      3. Comparison: last-1 vs last layer
      4. How does the last layer differ from mid layers?
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Last Layer Special Structure")
    print(f"  Model: {model_name}")
    print(f"  Key test: Why does the last layer have stronger gate-output alignment?")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    W_U = get_W_U(model)

    # Compare last layer, last-1, and mid layer
    compare_layers = [n_layers - 1, n_layers - 2, n_layers // 2]
    compare_labels = ["last", "last-1", "mid"]

    results = []

    for li, label in zip(compare_layers, compare_labels):
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None or lw.W_down is None:
            continue

        W_gate = lw.W_gate  # [n_inter, d_model]
        W_down = lw.W_down  # [d_model, n_inter]
        n_inter = W_gate.shape[0]

        # 1. W_gate row norms
        gate_row_norms = np.linalg.norm(W_gate, axis=1)
        gate_norm_mean = float(np.mean(gate_row_norms))
        gate_norm_std = float(np.std(gate_row_norms))
        gate_norm_max = float(np.max(gate_row_norms))
        gate_norm_top1pct = float(np.percentile(gate_row_norms, 99))

        # 2. W_down column norms
        down_col_norms = np.linalg.norm(W_down, axis=0)
        down_norm_mean = float(np.mean(down_col_norms))
        down_norm_std = float(np.std(down_col_norms))
        down_norm_max = float(np.max(down_col_norms))
        down_norm_top1pct = float(np.percentile(down_col_norms, 99))

        # 3. W_down rows projection onto W_U row space
        from scipy.sparse.linalg import svds
        k_svd = min(200, min(W_U.shape) - 2)
        U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_svd)
        U_wu = np.asarray(U_wu, dtype=np.float64)

        # Project W_down rows onto W_U space
        coeffs = U_wu.T @ W_down  # [k_svd, n_inter]
        W_down_proj = U_wu @ coeffs  # [d_model, n_inter]
        orig_norms_sq = np.sum(W_down ** 2, axis=1)  # [d_model]
        proj_norms_sq = np.sum(W_down_proj ** 2, axis=1)  # [d_model]
        proj_frac = float(np.mean(proj_norms_sq / np.maximum(orig_norms_sq, 1e-10)))

        # Random baseline
        rng = np.random.RandomState(42)
        rand_mat = rng.randn(d_model, n_inter) * np.mean(np.abs(W_down))
        rand_coeffs = U_wu.T @ rand_mat
        rand_proj = U_wu @ rand_coeffs
        rand_orig_sq = np.sum(rand_mat ** 2, axis=1)
        rand_proj_sq = np.sum(rand_proj ** 2, axis=1)
        rand_frac = float(np.mean(rand_proj_sq / np.maximum(rand_orig_sq, 1e-10)))

        # 4. SVD of W_down: how many singular values?
        try:
            U_d, s_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
            total_sv = np.sum(s_d ** 2)
            cum_sv = np.cumsum(s_d ** 2) / total_sv
            n80 = int(np.searchsorted(cum_sv, 0.80) + 1)
            n90 = int(np.searchsorted(cum_sv, 0.90) + 1)
            n95 = int(np.searchsorted(cum_sv, 0.95) + 1)
            top5_sv = float(cum_sv[4]) if len(cum_sv) >= 5 else 1.0
            top10_sv = float(cum_sv[9]) if len(cum_sv) >= 10 else 1.0
        except:
            n80 = n90 = n95 = 0
            top5_sv = top10_sv = 0.0

        # 5. W_down column alignment with W_U rows
        # For each W_down column, find the W_U row it's most aligned with
        # Use subsample for efficiency
        n_sample_cols = min(1000, n_inter)
        idx_cols = rng.choice(n_inter, n_sample_cols, replace=False)
        W_down_sub = W_down[:, idx_cols].T  # [n_sample, d_model]
        W_down_norms = np.linalg.norm(W_down_sub, axis=1, keepdims=True)
        valid = (W_down_norms > 1e-10).flatten()
        W_down_normalized = W_down_sub[valid] / W_down_norms[valid]

        # Sample W_U rows
        n_wu_sample = min(5000, W_U.shape[0])
        idx_wu = rng.choice(W_U.shape[0], n_wu_sample, replace=False)
        W_U_sub = W_U[idx_wu]
        W_U_norms = np.linalg.norm(W_U_sub, axis=1, keepdims=True)
        valid_u = (W_U_norms > 1e-10).flatten()
        W_U_normalized = W_U_sub[valid_u] / W_U_norms[valid_u]

        # Max cosine per W_down column with any W_U row
        cos_matrix = W_down_normalized @ W_U_normalized.T  # [n_down, n_wu]
        max_cos_per_col = np.max(cos_matrix, axis=1)
        mean_max_cos = float(np.mean(max_cos_per_col))
        frac_above_03 = float(np.mean(max_cos_per_col > 0.3))
        frac_above_05 = float(np.mean(max_cos_per_col > 0.5))

        results.append({
            "layer": li,
            "label": label,
            "n_intermediate": n_inter,
            "gate_row_norm_mean": gate_norm_mean,
            "gate_row_norm_top1pct": gate_norm_top1pct,
            "down_col_norm_mean": down_norm_mean,
            "down_col_norm_top1pct": down_norm_top1pct,
            "w_down_proj_in_WU": proj_frac,
            "random_proj_in_WU": rand_frac,
            "proj_ratio": proj_frac / max(rand_frac, 1e-6),
            "w_down_svd_n80": n80,
            "w_down_svd_n90": n90,
            "w_down_svd_n95": n95,
            "w_down_top5_sv": top5_sv,
            "w_down_top10_sv": top10_sv,
            "max_cos_wdown_wu_mean": mean_max_cos,
            "max_cos_above_03_frac": frac_above_03,
            "max_cos_above_05_frac": frac_above_05,
        })

        print(f"  {label} (L{li}): W_down proj_WU={proj_frac:.4f}(rand={rand_frac:.4f}, "
              f"ratio={proj_frac/max(rand_frac,1e-6):.2f}), "
              f"SVD(n90={n90}), max_cos={mean_max_cos:.3f}, "
              f"frac>0.3={frac_above_03:.3f}")

    # Compare
    if len(results) >= 2:
        last = [r for r in results if r["label"] == "last"]
        mid = [r for r in results if r["label"] == "mid"]
        if last and mid:
            print(f"\n  Last vs Mid comparison:")
            print(f"    W_down proj_WU ratio: last={last[0]['proj_ratio']:.2f}x vs mid={mid[0]['proj_ratio']:.2f}x")
            print(f"    W_down SVD(n90): last={last[0]['w_down_svd_n90']} vs mid={mid[0]['w_down_svd_n90']}")
            print(f"    Max cos(W_down, W_U): last={last[0]['max_cos_wdown_wu_mean']:.3f} vs mid={mid[0]['max_cos_wdown_wu_mean']:.3f}")
            print(f"    Frac W_down cols with cos>0.3: last={last[0]['max_cos_above_03_frac']:.3f} vs mid={mid[0]['max_cos_above_03_frac']:.3f}")

            if last[0]["proj_ratio"] > mid[0]["proj_ratio"] + 0.3:
                print(f"  >>> Last layer W_down is MORE aligned with W_U than mid layer")
            if last[0]["max_cos_above_03_frac"] > mid[0]["max_cos_above_03_frac"] + 0.1:
                print(f"  >>> Last layer has MORE W_down columns aligned with W_U rows")

    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_last_layer.json"

    summary = {
        "experiment": "exp3_last_layer",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "results": results,
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
        exp1_wdown_readout(args.model)

    if args.exp in ["2", "all"]:
        exp2_gate_polytope_distance(args.model)

    if args.exp in ["3", "all"]:
        exp3_last_layer_structure(args.model)
