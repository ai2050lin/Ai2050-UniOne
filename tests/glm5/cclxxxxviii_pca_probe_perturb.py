"""
CCLXXXXVIII(298): 补充验证 — PCA降维后的线性probe + 系统perturb
=========================================================
CCLXXXXVII发现:
  - Δg→WU ratio极低(0.05-0.08) → 线性readout不可见
  - 非线性probe=0.84-1.00 → 非线性可分
  - 但线性probe失败是方法问题(特征>>样本)

本实验:
  Exp1: PCA降维后线性probe — 真正比较Δg vs Δu的线性可分性
  Exp2: 系统perturb — 多词/多层/多PC方向的行为级验证
  Exp3: 邻域swap — 同类词之间swap g, 看logit变化

用法:
  python cclxxxxviii_pca_probe_perturb.py --model qwen3
  python cclxxxxviii_pca_probe_perturb.py --model glm4
  python cclxxxxviii_pca_probe_perturb.py --model deepseek7b
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
    get_layer_weights, LayerWeights, compute_cos, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "cclxxxxviii_log.txt"

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

TEMPLATE = "The {} is"


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_all_activations(model_name, n_words_per_cat=8):
    """Collect gate, up, residual activations."""
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)

    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(n_words_per_cat, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * len(sel))

    word_data = {}
    for wi, word in enumerate(all_words):
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        mlp_input = {}
        layer_out = {}
        hooks = []

        for li in range(n_layers):
            layer = layers_list[li]
            if hasattr(layer, 'mlp'):
                def make_mlp_pre(key):
                    def hook(module, args):
                        if isinstance(args, tuple):
                            mlp_input[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            mlp_input[key] = args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(f"L{li}")))

            def make_layer_out(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        layer_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        layer_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, last_pos].detach().float().cpu().numpy()

        for h in hooks:
            h.remove()

        word_data[word] = {
            "cat": all_cats[wi],
            "gates": {},
            "ups": {},
            "layer_out": {},
            "logits": logits,
        }

        for li in range(n_layers):
            key = f"L{li}"
            if key not in mlp_input:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue

            h_tilde = mlp_input[key]
            z = lw.W_gate @ h_tilde
            z_clipped = np.clip(z, -500, 500)
            g = 1.0 / (1.0 + np.exp(-z_clipped))
            u = lw.W_up @ h_tilde

            word_data[word]["gates"][li] = g
            word_data[word]["ups"][li] = u
            if key in layer_out:
                word_data[word]["layer_out"][li] = layer_out[key]

        print(f"  Word {wi+1}/{len(all_words)}: {word} ({all_cats[wi]})")

    print(f"  Collected data for {len(all_words)} words")
    return word_data, all_words, all_cats, model_info, layers_list, W_U, model, tokenizer, device


# ============================================================
# Exp1: PCA降维后线性probe — 真正比较Δg vs Δu的线性可分性
# ============================================================
def exp1_pca_linear_probe(word_data, all_words, all_cats, model_info, layers_list, log_f):
    """PCA降维到n_features<<n_samples后做线性probe.
    
    CCLXXXXVII线性probe失败是因为n_inter=9728-18944 >> n=24.
    现在先PCA降到5-20维, 再做线性分类.
    """
    log_f("\n" + "="*70)
    log_f("Exp1: PCA降维后线性Probe — Δg vs Δu")
    log_f("="*70)

    n_layers = model_info.n_layers
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse.linalg import svds

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    results = {}

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        if len(valid_words) < 8:
            continue

        X_g = np.array([word_data[w]["gates"][li] for w in valid_words], dtype=np.float32)
        X_u = np.array([word_data[w]["ups"][li] for w in valid_words], dtype=np.float32)
        X_h = np.array([word_data[w]["layer_out"][li] for w in valid_words 
                        if li in word_data[w]["layer_out"]], dtype=np.float32)
        y = np.array([word_data[w]["cat"] for w in valid_words])
        
        # For h, may have fewer valid words
        valid_h_words = [w for w in valid_words if li in word_data[w]["layer_out"]]
        y_h = np.array([word_data[w]["cat"] for w in valid_h_words])

        n_cats = len(set(y))
        chance = 1.0 / n_cats

        # PCA of gate and up spaces
        def pca_reduce(X, n_comp):
            Xc = X - X.mean(axis=0)
            k = min(n_comp, Xc.shape[0] - 1, Xc.shape[1] - 1)
            if k < 2:
                return None
            U, s, Vt = svds(Xc.astype(np.float32), k=k)
            sort_idx = np.argsort(-s)
            return U[:, sort_idx][:, :n_comp], s[sort_idx][:n_comp]

        for n_comp in [3, 5, 10, 20]:
            pca_g = pca_reduce(X_g, n_comp)
            pca_u = pca_reduce(X_u, n_comp)
            pca_h = pca_reduce(X_h, n_comp) if len(X_h) >= n_comp + 1 else None

            results[f"L{li}_n{n_comp}"] = {}

            # Gate PCA classification
            if pca_g is not None:
                X_g_pca = pca_g[0]
                scaler = StandardScaler().fit(X_g_pca)
                X_g_scaled = scaler.transform(X_g_pca)
                try:
                    lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
                    cv = min(5, len(valid_words) // n_cats)
                    if cv >= 2:
                        acc = float(np.mean(cross_val_score(lr, X_g_scaled, y, cv=cv)))
                    else:
                        acc = -1
                except:
                    acc = -1
                results[f"L{li}_n{n_comp}"]["gate_pca_lin"] = acc

            # Up PCA classification
            if pca_u is not None:
                X_u_pca = pca_u[0]
                scaler = StandardScaler().fit(X_u_pca)
                X_u_scaled = scaler.transform(X_u_pca)
                try:
                    lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
                    cv = min(5, len(valid_words) // n_cats)
                    if cv >= 2:
                        acc = float(np.mean(cross_val_score(lr, X_u_scaled, y, cv=cv)))
                    else:
                        acc = -1
                except:
                    acc = -1
                results[f"L{li}_n{n_comp}"]["up_pca_lin"] = acc

            # Residual stream PCA classification (baseline)
            if pca_h is not None:
                X_h_pca = pca_h[0]
                scaler = StandardScaler().fit(X_h_pca)
                X_h_scaled = scaler.transform(X_h_pca)
                try:
                    lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
                    cv = min(5, len(valid_h_words) // n_cats)
                    if cv >= 2:
                        acc = float(np.mean(cross_val_score(lr, X_h_scaled, y_h, cv=cv)))
                    else:
                        acc = -1
                except:
                    acc = -1
                results[f"L{li}_n{n_comp}"]["resid_pca_lin"] = acc

        # Print for this layer
        log_f(f"  L{li} (chance={chance:.3f}):")
        for n_comp in [3, 5, 10, 20]:
            key = f"L{li}_n{n_comp}"
            if key in results:
                r = results[key]
                g_acc = r.get("gate_pca_lin", -1)
                u_acc = r.get("up_pca_lin", -1)
                h_acc = r.get("resid_pca_lin", -1)
                log_f(f"    PCA{n_comp}: gate={g_acc:.3f} up={u_acc:.3f} resid={h_acc:.3f}")

    # Summary
    log_f(f"\n  === Exp1 Summary ===")
    mid_n = 10  # Use 10 PCA components
    mid_layers = [li for li in target_layers if n_layers * 0.3 <= li < n_layers * 0.7]
    
    gate_accs = []
    up_accs = []
    resid_accs = []
    for li in mid_layers:
        key = f"L{li}_n{mid_n}"
        if key in results:
            if results[key].get("gate_pca_lin", -1) > 0:
                gate_accs.append(results[key]["gate_pca_lin"])
            if results[key].get("up_pca_lin", -1) > 0:
                up_accs.append(results[key]["up_pca_lin"])
            if results[key].get("resid_pca_lin", -1) > 0:
                resid_accs.append(results[key]["resid_pca_lin"])

    chance = 1.0 / len(set(all_cats))
    if gate_accs:
        log_f(f"  Mid-layer PCA-10 linear probe:")
        log_f(f"    Gate: {np.mean(gate_accs):.3f} ± {np.std(gate_accs):.3f}")
        log_f(f"    Up:   {np.mean(up_accs):.3f} ± {np.std(up_accs):.3f}" if up_accs else "    Up: N/A")
        log_f(f"    Resid:{np.mean(resid_accs):.3f} ± {np.std(resid_accs):.3f}" if resid_accs else "    Resid: N/A")
        log_f(f"    Chance: {chance:.3f}")

        if np.mean(gate_accs) > chance * 3:
            log_f(f"  ★★★ Gate PCA linear probe > 3x chance → Δg has LINEAR category info!")
        elif np.mean(gate_accs) > chance * 1.5:
            log_f(f"  ★★ Gate PCA linear probe > 1.5x chance → Δg has WEAK linear category info")
        else:
            log_f(f"  ★ Gate PCA linear probe ≈ chance → Δg has NO linear category info")

    return results


# ============================================================
# Exp2: 系统perturb — 多词/多层/多PC方向
# ============================================================
def exp2_systematic_perturb(word_data, all_words, all_cats, model_info,
                            layers_list, W_U, model, tokenizer, device, log_f):
    """Systematic perturbation: multiple words, layers, PC directions.
    
    Key question: Which PC directions of Δg cause systematic logit changes?
    """
    log_f("\n" + "="*70)
    log_f("Exp2: Systematic Perturb — Multi-word/Layer/PC")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    from scipy.sparse.linalg import svds

    # Test 3 layers: early, middle, late
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    # Test 2 words per category
    rng = np.random.RandomState(42)
    test_words = []
    test_cats_list = []
    for cat in CONCEPTS:
        sel = rng.choice(CONCEPTS[cat], 2, replace=False).tolist()
        test_words.extend(sel)
        test_cats_list.extend([cat] * 2)

    results = {}

    for li in test_layers:
        lw = get_layer_weights(layers_list[li], d_model, model_info.mlp_type)
        W_down = lw.W_down

        # Compute gate PCA
        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        if len(valid_words) < 4:
            continue

        gate_arr = np.array([word_data[w]["gates"][li] for w in valid_words], dtype=np.float32)
        gate_centered = gate_arr - gate_arr.mean(axis=0)

        n_pca = min(10, gate_centered.shape[0] - 1, gate_centered.shape[1] - 1)
        if n_pca < 2:
            continue

        U_gate, s_gate, Vt_gate = svds(gate_centered, k=n_pca)
        sort_idx = np.argsort(-s_gate)
        s_gate = s_gate[sort_idx]
        Vt_gate = Vt_gate[sort_idx, :]  # PC directions [n_pca, n_inter]

        log_f(f"\n  Layer L{li}: top-5 singular values = {s_gate[:5].round(2)}")

        # For each test word, perturb along top-5 PCs
        for word in test_words[:6]:  # Limit to 6 words for speed
            if word not in word_data or li not in word_data[word]["gates"]:
                continue
            cat = word_data[word]["cat"]

            # Get baseline logits
            text = TEMPLATE.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids

            with torch.no_grad():
                base_logits = model(input_ids).logits[0, -1].detach().float().cpu().numpy()

            base_top1 = np.argmax(base_logits)
            base_top1_token = tokenizer.decode([base_top1])

            # Get word's gate and up
            g_word = word_data[word]["gates"][li]
            u_word = word_data[word]["ups"][li]
            mean_g = gate_arr.mean(axis=0)

            for pc_idx in range(min(5, n_pca)):
                pc_dir = Vt_gate[pc_idx]
                eps = 1.0

                # Compute Δout from perturbation
                dg = eps * pc_dir
                g_perturbed = np.clip(mean_g + dg, 0.001, 0.999)
                delta_out = W_down @ ((g_perturbed - mean_g) * u_word)

                # Compute Δlogits
                delta_logits = W_U @ delta_out  # [vocab_size]
                perturbed_logits = base_logits + delta_logits
                perturbed_top1 = np.argmax(perturbed_logits)
                perturbed_top1_token = tokenizer.decode([perturbed_top1])

                # Top-1 change?
                top1_changed = int(base_top1 != perturbed_top1)

                # Logit margin change (for the word's category)
                word_tok_ids = tokenizer.encode(word, add_special_tokens=False)
                word_tok_id = word_tok_ids[0] if word_tok_ids else None

                # Concept direction alignment
                cat_mean_logits = {}
                for c in set(all_cats):
                    c_words = [w for w in valid_words if word_data[w]["cat"] == c]
                    if len(c_words) >= 2:
                        cat_mean_logits[c] = np.mean([word_data[w]["logits"] for w in c_words], axis=0)

                best_cat_cos = 0
                best_cat_name = ""
                for c, cl in cat_mean_logits.items():
                    cos = proper_cos(delta_logits, cl)
                    if abs(cos) > abs(best_cat_cos):
                        best_cat_cos = cos
                        best_cat_name = c

                key = f"L{li}_{word}_PC{pc_idx}"
                results[key] = {
                    "layer": li,
                    "word": word,
                    "cat": cat,
                    "pc": pc_idx,
                    "top1_changed": top1_changed,
                    "delta_logits_norm": float(np.linalg.norm(delta_logits)),
                    "best_cat": best_cat_name,
                    "best_cat_cos": float(best_cat_cos),
                    "same_cat": int(best_cat_name == cat),
                }

                log_f(f"    {word}({cat}) PC{pc_idx}: top1_changed={top1_changed}, "
                      f"||Δlogits||={results[key]['delta_logits_norm']:.2f}, "
                      f"best_cat={best_cat_name}(cos={best_cat_cos:.3f}), "
                      f"same_cat={results[key]['same_cat']}")

    # Summary
    log_f(f"\n  === Exp2 Summary ===")
    top1_change_rate = np.mean([r["top1_changed"] for r in results.values()])
    same_cat_rate = np.mean([r["same_cat"] for r in results.values()])
    avg_delta_logits = np.mean([r["delta_logits_norm"] for r in results.values()])
    avg_best_cos = np.mean([abs(r["best_cat_cos"]) for r in results.values()])

    log_f(f"  Top-1 change rate: {top1_change_rate:.3f}")
    log_f(f"  Same-category alignment rate: {same_cat_rate:.3f}")
    log_f(f"  Avg ||Δlogits||: {avg_delta_logits:.2f}")
    log_f(f"  Avg |best concept cos|: {avg_best_cos:.3f}")

    return results


# ============================================================
# Exp3: 邻域swap — 同类词之间swap g
# ============================================================
def exp3_neighborhood_swap(word_data, all_words, all_cats, model_info,
                           layers_list, W_U, model, tokenizer, device, log_f):
    """Swap g between words in same category and different categories.
    
    Key test: 
    - Same-cat swap: g_cat1_word1 ⊙ u_cat1_word2 → should have MINIMAL effect
    - Diff-cat swap: g_cat1_word1 ⊙ u_cat2_word2 → should have LARGER effect
    - If same-cat swap also causes large changes → g is not "just gain modulation"
    """
    log_f("\n" + "="*70)
    log_f("Exp3: Neighborhood Swap — Same-cat vs Diff-cat")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Middle layer
    target_layer = n_layers // 2
    lw = get_layer_weights(layers_list[target_layer], d_model, model_info.mlp_type)
    W_down = lw.W_down

    valid_words = [w for w in all_words if target_layer in word_data[w]["gates"]]
    if len(valid_words) < 8:
        log_f("  Not enough valid words")
        return {}

    log_f(f"  Target layer: L{target_layer}")

    # Get all words' gates and ups
    word_g = {w: word_data[w]["gates"][target_layer] for w in valid_words}
    word_u = {w: word_data[w]["ups"][target_layer] for w in valid_words}
    word_logits = {w: word_data[w]["logits"] for w in valid_words}

    # Compute original FFN outputs
    word_ffn = {}
    for w in valid_words:
        ffn_out = W_down @ (word_g[w] * word_u[w])
        word_ffn[w] = ffn_out

    # Same-cat pairs and diff-cat pairs
    same_cat_pairs = []
    diff_cat_pairs = []
    for i, w1 in enumerate(valid_words):
        for j, w2 in enumerate(valid_words):
            if i >= j:
                continue
            c1, c2 = word_data[w1]["cat"], word_data[w2]["cat"]
            if c1 == c2:
                same_cat_pairs.append((w1, w2, c1, c2))
            else:
                diff_cat_pairs.append((w1, w2, c1, c2))

    log_f(f"  Same-cat pairs: {len(same_cat_pairs)}, Diff-cat pairs: {len(diff_cat_pairs)}")

    # For each pair, compute swap effect
    def compute_swap_effect(w1, w2):
        """Swap g between w1 and w2, compute Δlogits."""
        # Original: W_down @ (g1 ⊙ u1) and W_down @ (g2 ⊙ u2)
        # Swapped: W_down @ (g2 ⊙ u1) and W_down @ (g1 ⊙ u2)
        
        # Swap for w1: use g2 instead of g1
        delta_ffn_w1 = W_down @ ((word_g[w2] - word_g[w1]) * word_u[w1])
        # Swap for w2: use g1 instead of g2
        delta_ffn_w2 = W_down @ ((word_g[w1] - word_g[w2]) * word_u[w2])

        # Δlogits
        delta_logits_w1 = W_U @ delta_ffn_w1
        delta_logits_w2 = W_U @ delta_ffn_w2

        return {
            "w1": w1, "w2": w2,
            "delta_logits_norm_w1": float(np.linalg.norm(delta_logits_w1)),
            "delta_logits_norm_w2": float(np.linalg.norm(delta_logits_w2)),
            "delta_ffn_norm_w1": float(np.linalg.norm(delta_ffn_w1)),
            "delta_ffn_norm_w2": float(np.linalg.norm(delta_ffn_w2)),
            # Cos with concept directions
            "w1_best_cat": max(word_data[w1]["cat"], word_data[w2]["cat"],
                              key=lambda c: abs(proper_cos(delta_logits_w1, 
                                  np.mean([word_logits[ww] for ww in valid_words 
                                           if word_data[ww]["cat"] == c], axis=0)))),
            "w1_best_cos": max([abs(proper_cos(delta_logits_w1, 
                             np.mean([word_logits[ww] for ww in valid_words 
                                      if word_data[ww]["cat"] == c], axis=0)))
                               for c in set(all_cats)]),
            "w2_best_cos": max([abs(proper_cos(delta_logits_w2, 
                             np.mean([word_logits[ww] for ww in valid_words 
                                      if word_data[ww]["cat"] == c], axis=0)))
                               for c in set(all_cats)]),
        }

    same_results = []
    diff_results = []

    # Sample pairs (limit to 20 each for speed)
    rng = np.random.RandomState(42)
    if len(same_cat_pairs) > 20:
        same_cat_sample = rng.choice(len(same_cat_pairs), 20, replace=False)
        same_cat_pairs = [same_cat_pairs[i] for i in same_cat_sample]
    if len(diff_cat_pairs) > 20:
        diff_cat_sample = rng.choice(len(diff_cat_pairs), 20, replace=False)
        diff_cat_pairs = [diff_cat_pairs[i] for i in diff_cat_sample]

    for w1, w2, c1, c2 in same_cat_pairs:
        r = compute_swap_effect(w1, w2)
        same_results.append(r)

    for w1, w2, c1, c2 in diff_cat_pairs:
        r = compute_swap_effect(w1, w2)
        diff_results.append(r)

    # Summary
    if same_results and diff_results:
        same_dlogits = np.mean([r["delta_logits_norm_w1"] for r in same_results])
        diff_dlogits = np.mean([r["delta_logits_norm_w1"] for r in diff_results])
        same_dffn = np.mean([r["delta_ffn_norm_w1"] for r in same_results])
        diff_dffn = np.mean([r["delta_ffn_norm_w1"] for r in diff_results])
        same_cos = np.mean([r["w1_best_cos"] for r in same_results])
        diff_cos = np.mean([r["w1_best_cos"] for r in diff_results])

        ratio = diff_dlogits / max(same_dlogits, 1e-10)

        log_f(f"\n  === Exp3 Summary ===")
        log_f(f"  Same-cat swap: ||Δlogits||={same_dlogits:.4f}, ||Δffn||={same_dffn:.4f}, cos={same_cos:.3f}")
        log_f(f"  Diff-cat swap: ||Δlogits||={diff_dlogits:.4f}, ||Δffn||={diff_dffn:.4f}, cos={diff_cos:.3f}")
        log_f(f"  Ratio (diff/same): {ratio:.2f}x")

        if ratio > 3:
            log_f(f"  ★★★ Diff-cat >> Same-cat → Δg encodes category info → NOT just noise")
        elif ratio > 1.5:
            log_f(f"  ★★ Diff-cat > Same-cat → Δg partially encodes category info")
        else:
            log_f(f"  ★ Diff-cat ≈ Same-cat → Δg does NOT encode category-specific info")

    return {"same": same_results, "diff": diff_results}


# ============================================================
# Main
# ============================================================
def run_all_experiments(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCLXXXXVIII(298): PCA Probe + Systematic Perturb + Swap")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    # Collect data
    log_f("\n--- Collecting activations ---")
    (word_data, all_words, all_cats, model_info, layers_list,
     W_U, model, tokenizer, device) = collect_all_activations(model_name)

    n_layers = model_info.n_layers
    log_f(f"\nModel: {model_info.model_class}, {n_layers} layers, d_model={model_info.d_model}")

    # Run experiments
    exp1_results = exp1_pca_linear_probe(
        word_data, all_words, all_cats, model_info, layers_list, log_f)

    exp2_results = exp2_systematic_perturb(
        word_data, all_words, all_cats, model_info, layers_list,
        W_U, model, tokenizer, device, log_f)

    exp3_results = exp3_neighborhood_swap(
        word_data, all_words, all_cats, model_info, layers_list,
        W_U, model, tokenizer, device, log_f)

    # Release model
    release_model(model)

    # Save log
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    log_f(f"\n  Results saved to {LOG_FILE}")
    return log_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_all_experiments(args.model)
