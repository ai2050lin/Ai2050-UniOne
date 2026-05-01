"""
CCLXXXXVII(297): 生死实验 — Δg在多readout下的可见性
=========================================================
核心问题: W_down@Δg≈0是因为Δg真的"不可见"(控制变量), 
         还是仅仅因为W_down碰巧投影掉了它?

三种场景:
  场景A: Δg在所有readout下都不可见 → "控制空间"成立 → 框架成立
  场景B: Δg在logits下可见 → "控制空间"推翻 → 框架需要重构
  场景C: 线性不可见但非线性可见 → 需要更精细的理论

实验:
  Exp1: W_down@Δg vs W_U@Δg — Δg在logit空间是否有方向?
  Exp2: Δg的线性probe分类器 — 能否从Δg预测类别?
  Exp3: Δg在W_O(attention输出)空间下的投影
  Exp4: Perturb Δg沿PC方向 → 行为级验证(logits变化)
  Exp5: Δg在residual stream差(Δh)中的可见性

用法:
  python cclxxxxvii_multi_readout.py --model qwen3
  python cclxxxxvii_multi_readout.py --model glm4
  python cclxxxxvii_multi_readout.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "cclxxxxvii_log.txt"

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


def collect_all_activations(model_name, n_words_per_cat=6):
    """Collect gate, up, residual, and logit activations for all words.
    Returns comprehensive data dict.
    """
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    W_U = get_W_U(model)  # [vocab_size, d_model]

    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(n_words_per_cat, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * len(sel))

    # Precompute W_U row space basis (top-200 components)
    from scipy.sparse.linalg import svds
    k_wu = min(200, W_U.shape[0] - 2, W_U.shape[1] - 2)
    U_wut, s_wut, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    sort_idx = np.argsort(-s_wut)
    U_wut = U_wut[:, sort_idx]  # [d_model, k_wu] — W_U行空间基
    s_wut = s_wut[sort_idx]

    # Collect per-word data
    word_data = {}
    for wi, word in enumerate(all_words):
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        # Hooks to capture: MLP input (after layernorm), layer output, MLP output
        mlp_input = {}
        layer_out = {}
        mlp_out = {}
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

                def make_mlp_out(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            mlp_out[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            mlp_out[key] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_hook(make_mlp_out(f"L{li}")))

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

        # Compute gate and up for each layer
        word_data[word] = {
            "cat": all_cats[wi],
            "gates": {},
            "ups": {},
            "mlp_input": {},
            "layer_out": {},
            "mlp_out": {},
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
            word_data[word]["mlp_input"][li] = h_tilde
            if key in layer_out:
                word_data[word]["layer_out"][li] = layer_out[key]
            if key in mlp_out:
                word_data[word]["mlp_out"][li] = mlp_out[key]

        print(f"  Word {wi+1}/{len(all_words)}: {word} ({all_cats[wi]}) — {len(word_data[word]['gates'])} layers")

    print(f"  Collected data for {len(all_words)} words")

    return word_data, all_words, all_cats, model_info, layers_list, W_U, U_wut, s_wut, model, tokenizer, device


# ============================================================
# Exp1: W_down@Δg vs W_U@Δg — 核心生死测试
# ============================================================
def exp1_wdown_vs_wu(word_data, all_words, all_cats, model_info, layers_list, W_U, U_wut, s_wut, log_f):
    """Compare Δg visibility under W_down vs W_U readout.

    This is THE critical test:
    - If W_U@Δg ≈ 0 (like W_down@Δg ≈ 0): "control space" hypothesis supported
    - If W_U@Δg ≠ 0: "control space" hypothesis overturned
    """
    log_f("\n" + "="*70)
    log_f("Exp1: W_down@Δg vs W_U@Δg — THE LIFE-OR-DEATH TEST")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Sample layers
    target_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    results = {}

    for li in target_layers:
        # Get W_down and W_U for this layer
        lw = get_layer_weights(layers_list[li], d_model, model_info.mlp_type)
        W_down = lw.W_down  # [d_model, n_inter]

        # Compute Δg, Δu for all pairs
        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        if len(valid_words) < 4:
            continue

        # Category-level mean gates
        cat_gates = defaultdict(list)
        cat_ups = defaultdict(list)
        for w in valid_words:
            cat = word_data[w]["cat"]
            cat_gates[cat].append(word_data[w]["gates"][li])
            cat_ups[cat].append(word_data[w]["ups"][li])

        cat_mean_g = {c: np.mean(gs, axis=0) for c, gs in cat_gates.items()}
        cat_mean_u = {c: np.mean(us, axis=0) for c, us in cat_ups.items()}

        # Concept direction (in residual stream)
        cat_mean_h = {}
        for w in valid_words:
            cat = word_data[w]["cat"]
            if li in word_data[w]["layer_out"]:
                if cat not in cat_mean_h:
                    cat_mean_h[cat] = []
                cat_mean_h[cat].append(word_data[w]["layer_out"][li])
        cat_mean_h = {c: np.mean(hs, axis=0) for c, hs in cat_mean_h.items()}

        cat_names = sorted(cat_mean_g.keys())
        if len(cat_names) < 2:
            continue

        # Compute all pairwise Δg, Δu, Δh
        delta_g_wdown_cos = []
        delta_g_wu_cos = []
        delta_g_wu_norm = []
        delta_u_wdown_cos = []
        delta_u_wu_cos = []
        delta_h_wu_cos = []

        for i, cA in enumerate(cat_names):
            for j, cB in enumerate(cat_names):
                if i >= j:
                    continue

                dg = cat_mean_g[cA] - cat_mean_g[cB]
                du = cat_mean_u[cA] - cat_mean_u[cB]

                # Δg is in intermediate space [n_inter], must go through W_down to d_model
                Wdown_dg = W_down @ dg  # [d_model]
                Wdown_du = W_down @ du  # [d_model]

                Wdown_dg_norm = np.linalg.norm(Wdown_dg)
                Wdown_du_norm = np.linalg.norm(Wdown_du)

                # ★★★ KEY TEST: Does W_down@Δg project onto W_U row space? ★★★
                # If yes → Δg affects logits through W_down → "control space" weakened
                # If no → Δg is invisible even through the logit path → "control space" supported
                if Wdown_dg_norm > 1e-10:
                    proj_coeffs = U_wut.T @ Wdown_dg  # [k]
                    proj_energy = np.sum(proj_coeffs ** 2)
                    wu_ratio = proj_energy / max(Wdown_dg_norm ** 2, 1e-20)
                else:
                    wu_ratio = 0.0

                if Wdown_du_norm > 1e-10:
                    proj_coeffs_u = U_wut.T @ Wdown_du
                    proj_energy_u = np.sum(proj_coeffs_u ** 2)
                    wu_ratio_u = proj_energy_u / max(Wdown_du_norm ** 2, 1e-20)
                else:
                    wu_ratio_u = 0.0

                # Δh (residual stream difference) and its W_U projection
                if cA in cat_mean_h and cB in cat_mean_h:
                    dh = cat_mean_h[cA] - cat_mean_h[cB]
                    dh_norm = np.linalg.norm(dh)
                    if dh_norm > 1e-10:
                        proj_coeffs_h = U_wut.T @ dh
                        proj_energy_h = np.sum(proj_coeffs_h ** 2)
                        wu_ratio_h = proj_energy_h / max(dh_norm ** 2, 1e-20)
                    else:
                        wu_ratio_h = 0.0
                else:
                    wu_ratio_h = -1.0

                delta_g_wdown_cos.append(Wdown_dg_norm)
                delta_g_wu_norm.append(wu_ratio)
                delta_u_wdown_cos.append(Wdown_du_norm)
                delta_u_wu_cos.append(wu_ratio_u)
                delta_h_wu_cos.append(wu_ratio_h)

        n_pairs = len(delta_g_wdown_cos)
        if n_pairs == 0:
            continue

        results[li] = {
            "n_pairs": n_pairs,
            "Wdown_dg_norm_mean": float(np.mean(delta_g_wdown_cos)),
            "Wdown_dg_norm_std": float(np.std(delta_g_wdown_cos)),
            "Wdown_du_norm_mean": float(np.mean(delta_u_wdown_cos)),
            "Wdown_du_norm_std": float(np.std(delta_u_wdown_cos)),
            # ★★★★★ THE KEY METRICS ★★★★★
            "dg_wu_row_ratio_mean": float(np.mean(delta_g_wu_norm)),  # Δg → W_down → W_U row ratio
            "dg_wu_row_ratio_std": float(np.std(delta_g_wu_norm)),
            "du_wu_row_ratio_mean": float(np.mean(delta_u_wu_cos)),
            "du_wu_row_ratio_std": float(np.std(delta_u_wu_cos)),
            "dh_wu_row_ratio_mean": float(np.mean([v for v in delta_h_wu_cos if v >= 0])) if any(v >= 0 for v in delta_h_wu_cos) else -1,
        }

        log_f(f"  L{li}: Wdown||Δg||={results[li]['Wdown_dg_norm_mean']:.6f} "
              f"Wdown||Δu||={results[li]['Wdown_du_norm_mean']:.6f}")
        log_f(f"        dg→WU_row_ratio={results[li]['dg_wu_row_ratio_mean']:.4f} "
              f"du→WU_row_ratio={results[li]['du_wu_row_ratio_mean']:.4f} "
              f"dh→WU_row_ratio={results[li]['dh_wu_row_ratio_mean']:.4f}")

    # Summary
    log_f(f"\n  === Exp1 Summary ===")
    for li in sorted(results.keys()):
        r = results[li]
        log_f(f"  L{li}: dg_wu_ratio={r['dg_wu_row_ratio_mean']:.4f}  "
              f"du_wu_ratio={r['du_wu_row_ratio_mean']:.4f}  "
              f"dh_wu_ratio={r['dh_wu_row_ratio_mean']:.4f}  "
              f"Wdown||dg||={r['Wdown_dg_norm_mean']:.6f}  "
              f"Wdown||du||={r['Wdown_du_norm_mean']:.6f}")

    # Critical judgment
    mid_layers = [li for li in results if n_layers * 0.3 <= li < n_layers * 0.7]
    if mid_layers:
        avg_dg_ratio = np.mean([results[li]["dg_wu_row_ratio_mean"] for li in mid_layers])
        avg_du_ratio = np.mean([results[li]["du_wu_row_ratio_mean"] for li in mid_layers])
        log_f(f"\n  ★★★ CRITICAL JUDGMENT (mid-layers) ★★★")
        log_f(f"  Δg→W_U row ratio: {avg_dg_ratio:.4f}")
        log_f(f"  Δu→W_U row ratio: {avg_du_ratio:.4f}")
        if avg_dg_ratio < 0.1:
            log_f(f"  → SCENARIO A: Δg invisible in all readouts — 'control space' SUPPORTED")
        elif avg_dg_ratio > 0.5:
            log_f(f"  → SCENARIO B: Δg visible in logits — 'control space' OVERTURNED")
        else:
            log_f(f"  → SCENARIO C: Δg partially visible — needs refined theory")

    return results


# ============================================================
# Exp2: Linear/Nonlinear probe on Δg — category classification
# ============================================================
def exp2_probe_classifier(word_data, all_words, all_cats, model_info, layers_list, log_f):
    """Can we classify categories from Δg? And from Δu for comparison?

    If linear probe on Δg achieves high accuracy:
      → Δg carries category information (even if W_down@Δg≈0)
      → "Control space" needs revision

    If only nonlinear probe works:
      → Δg has nonlinear category structure
      → Scenario C
    """
    log_f("\n" + "="*70)
    log_f("Exp2: Probe Classifier on Δg vs Δu")
    log_f("="*70)

    n_layers = model_info.n_layers

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    results = {}

    for li in target_layers:
        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        if len(valid_words) < 8:
            continue

        # Build feature matrices: gate vectors and up vectors
        X_g = np.array([word_data[w]["gates"][li] for w in valid_words], dtype=np.float32)
        X_u = np.array([word_data[w]["ups"][li] for w in valid_words], dtype=np.float32)
        y = np.array([word_data[w]["cat"] for w in valid_words])
        cat_names_unique = sorted(set(y))
        if len(cat_names_unique) < 2:
            continue

        # Subsample features for tractability
        n_inter = X_g.shape[1]
        if n_inter > 500:
            # Use top-200 variance dimensions
            var_g = np.var(X_g, axis=0)
            top_dims = np.argsort(-var_g)[:200]
            X_g_sub = X_g[:, top_dims]
            var_u = np.var(X_u, axis=0)
            top_dims_u = np.argsort(-var_u)[:200]
            X_u_sub = X_u[:, top_dims_u]
        else:
            X_g_sub = X_g
            X_u_sub = X_u

        # Random baseline
        X_rand = np.random.RandomState(li).randn(*X_g_sub.shape).astype(np.float32)

        # Scale
        scaler_g = StandardScaler().fit(X_g_sub)
        X_g_scaled = scaler_g.transform(X_g_sub)
        scaler_u = StandardScaler().fit(X_u_sub)
        X_u_scaled = scaler_u.transform(X_u_sub)
        scaler_r = StandardScaler().fit(X_rand)
        X_rand_scaled = scaler_r.transform(X_rand)

        cv = min(5, len(valid_words) // len(cat_names_unique))
        if cv < 2:
            continue

        # Linear probe
        try:
            lr_g = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
            acc_g_lin = float(np.mean(cross_val_score(lr_g, X_g_scaled, y, cv=cv)))
        except:
            acc_g_lin = -1

        try:
            lr_u = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
            acc_u_lin = float(np.mean(cross_val_score(lr_u, X_u_scaled, y, cv=cv)))
        except:
            acc_u_lin = -1

        try:
            lr_r = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr')
            acc_r_lin = float(np.mean(cross_val_score(lr_r, X_rand_scaled, y, cv=cv)))
        except:
            acc_r_lin = -1

        # Nonlinear probe (small MLP)
        try:
            mlp_g = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
            acc_g_nl = float(np.mean(cross_val_score(mlp_g, X_g_scaled, y, cv=cv)))
        except:
            acc_g_nl = -1

        try:
            mlp_u = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
            acc_u_nl = float(np.mean(cross_val_score(mlp_u, X_u_scaled, y, cv=cv)))
        except:
            acc_u_nl = -1

        results[li] = {
            "n_words": len(valid_words),
            "n_cats": len(cat_names_unique),
            "n_inter": n_inter,
            "acc_g_linear": acc_g_lin,
            "acc_u_linear": acc_u_lin,
            "acc_rand_linear": acc_r_lin,
            "acc_g_nonlinear": acc_g_nl,
            "acc_u_nonlinear": acc_u_nl,
        }

        log_f(f"  L{li}: n_inter={n_inter}, n_cats={len(cat_names_unique)}")
        log_f(f"    Gate:  linear={acc_g_lin:.3f}, nonlinear={acc_g_nl:.3f}")
        log_f(f"    Up:    linear={acc_u_lin:.3f}, nonlinear={acc_u_nl:.3f}")
        log_f(f"    Random: linear={acc_r_lin:.3f}")

    # Summary
    log_f(f"\n  === Exp2 Summary ===")
    for li in sorted(results.keys()):
        r = results[li]
        log_f(f"  L{li}: g_lin={r['acc_g_linear']:.3f} g_nl={r['acc_g_nonlinear']:.3f}  "
              f"u_lin={r['acc_u_linear']:.3f} u_nl={r['acc_u_nonlinear']:.3f}  "
              f"rand={r['acc_rand_linear']:.3f}")

    # Critical judgment
    mid_layers = [li for li in results if n_layers * 0.3 <= li < n_layers * 0.7]
    if mid_layers:
        avg_g_lin = np.mean([results[li]["acc_g_linear"] for li in mid_layers])
        avg_g_nl = np.mean([results[li]["acc_g_nonlinear"] for li in mid_layers])
        avg_u_lin = np.mean([results[li]["acc_u_linear"] for li in mid_layers])
        chance = 1.0 / len(set(all_cats))

        log_f(f"\n  ★★★ CRITICAL JUDGMENT (mid-layers) ★★★")
        log_f(f"  Chance level: {chance:.3f}")
        log_f(f"  Δg linear probe: {avg_g_lin:.3f}")
        log_f(f"  Δg nonlinear probe: {avg_g_nl:.3f}")
        log_f(f"  Δu linear probe: {avg_u_lin:.3f}")
        if avg_g_lin < chance * 1.5:
            log_f(f"  → Δg carries NO linear category info — consistent with 'control space'")
        elif avg_g_lin > chance * 3:
            log_f(f"  → Δg carries STRONG linear category info — 'control space' OVERTURNED")
        else:
            log_f(f"  → Δg carries weak linear category info — partially supports 'control space'")

        if avg_g_nl > avg_g_lin * 1.5 and avg_g_nl > chance * 3:
            log_f(f"  → Nonlinear >> Linear: Δg has nonlinear category structure (Scenario C)")

    return results


# ============================================================
# Exp3: Δg in W_O (attention output) space
# ============================================================
def exp3_attention_readout(word_data, all_words, all_cats, model_info, layers_list, log_f):
    """Does Δg project onto attention output (W_O) space?

    If Δg is visible through attention value output:
      → Δg affects not just MLP but also attention mechanism
      → Broader impact than "just MLP control"
    """
    log_f("\n" + "="*70)
    log_f("Exp3: Δg in Attention Output (W_O) Space")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    results = {}

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, model_info.mlp_type)
        W_down = lw.W_down
        W_O = lw.W_o  # [d_model, d_model] or [d_model, d_kv*heads]

        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        if len(valid_words) < 4:
            continue

        # Category-level means
        cat_gates = defaultdict(list)
        for w in valid_words:
            cat_gates[word_data[w]["cat"]].append(word_data[w]["gates"][li])
        cat_mean_g = {c: np.mean(gs, axis=0) for c, gs in cat_gates.items()}

        cat_names = sorted(cat_mean_g.keys())
        if len(cat_names) < 2:
            continue

        # W_O row space basis — handle GQA where W_O may not be square
        # W_O: [d_model, d_kv*heads], compute SVD of W_O^T
        from scipy.sparse.linalg import svds
        W_O_T = W_O.T.astype(np.float32)  # [d_kv*heads, d_model]
        k_wo = min(50, W_O_T.shape[0] - 2, W_O_T.shape[1] - 2)
        if k_wo < 2:
            continue
        U_wo, s_wo, _ = svds(W_O_T, k=k_wo)
        sort_idx = np.argsort(-s_wo)
        U_wo = U_wo[:, sort_idx]  # [d_kv*heads, k_wo] or [d_model, k_wo]
        s_wo = s_wo[sort_idx]

        # W_O maps from attention hidden dim → d_model
        # Its row space is in d_model space
        # Actually, W_O: [d_model, d_attn_hidden] maps attn_hidden → d_model
        # So W_O's rows are in d_model space, columns are in attn_hidden space
        # The "row space of W_O" in d_model space = span of W_O's rows
        # We want to check if Wdown_dg (in d_model) projects onto this space
        # Use SVD of W_O: W_O = U_svd @ diag(s) @ Vt_svd
        # W_O rows' span = rows of U_svd (in d_model space)
        U_wo_full, s_wo_full, Vt_wo_full = svds(W_O.astype(np.float32), k=k_wo)
        sort_idx2 = np.argsort(-s_wo_full)
        U_wo_full = U_wo_full[:, sort_idx2]  # [d_model, k_wo] — row space basis in d_model

        # For each category pair, check Δg visibility
        dg_wdown_cos_list = []
        dg_wo_cos_list = []
        dg_wo_ratio_list = []

        for i, cA in enumerate(cat_names):
            for j, cB in enumerate(cat_names):
                if i >= j:
                    continue

                dg = cat_mean_g[cA] - cat_mean_g[cB]

                # Through W_down
                Wdown_dg = W_down @ dg
                wdown_norm = np.linalg.norm(Wdown_dg)

                # Through W_O (indirectly: does W_down@Δg project onto W_O row space?)
                if wdown_norm > 1e-10:
                    proj_coeffs = U_wo_full.T @ Wdown_dg  # [k_wo]
                    proj_energy = np.sum(proj_coeffs ** 2)
                    wo_ratio = proj_energy / max(wdown_norm ** 2, 1e-20)
                else:
                    wo_ratio = 0.0

                dg_wdown_cos_list.append(wdown_norm)
                dg_wo_cos_list.append(wo_ratio)

        results[li] = {
            "n_pairs": len(dg_wdown_cos_list),
            "Wdown_dg_norm_mean": float(np.mean(dg_wdown_cos_list)),
            "dg_WO_ratio_mean": float(np.mean(dg_wo_cos_list)),
            "dg_WO_ratio_std": float(np.std(dg_wo_cos_list)),
        }

        log_f(f"  L{li}: Wdown||dg||={results[li]['Wdown_dg_norm_mean']:.6f}, "
              f"dg→WO_ratio={results[li]['dg_WO_ratio_mean']:.4f}")

    log_f(f"\n  === Exp3 Summary ===")
    for li in sorted(results.keys()):
        r = results[li]
        log_f(f"  L{li}: dg→WO_ratio={r['dg_WO_ratio_mean']:.4f} ± {r['dg_WO_ratio_std']:.4f}")

    return results


# ============================================================
# Exp4: Perturb Δg along PC direction → behavioral verification
# ============================================================
def exp4_perturb_behavioral(word_data, all_words, all_cats, model_info,
                            layers_list, W_U, model, tokenizer, device, log_f):
    """Perturb Δg along its PC directions and measure logit changes.

    This is Experiment 1 from the user's proposal:
    - Keep u fixed, only modify g along Δg's PC directions
    - Measure: logit changes, top-1 prediction, stability
    """
    log_f("\n" + "="*70)
    log_f("Exp4: Perturb Δg along PC → Behavioral Verification")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Pick a middle layer
    target_layer = n_layers // 2

    lw = get_layer_weights(layers_list[target_layer], d_model, model_info.mlp_type)
    W_down = lw.W_down
    W_gate = lw.W_gate

    # Compute Δg PCA
    valid_words = [w for w in all_words if target_layer in word_data[w]["gates"]]
    if len(valid_words) < 4:
        log_f("  Not enough valid words for Exp4")
        return {}

    gate_arr = np.array([word_data[w]["gates"][target_layer] for w in valid_words], dtype=np.float32)
    gate_centered = gate_arr - gate_arr.mean(axis=0)

    from scipy.sparse.linalg import svds
    n_pca = min(20, gate_centered.shape[0] - 1, gate_centered.shape[1] - 1)
    if n_pca < 2:
        log_f("  Not enough PCA components")
        return {}

    U_gate, s_gate, Vt_gate = svds(gate_centered, k=n_pca)
    sort_idx = np.argsort(-s_gate)
    s_gate = s_gate[sort_idx]
    Vt_gate = Vt_gate[sort_idx, :]  # PC directions in gate space [n_pca, n_inter]

    log_f(f"  Target layer: L{target_layer}, n_pca={n_pca}")
    log_f(f"  Top-5 singular values: {s_gate[:5]}")

    # Pick test word
    test_word = valid_words[0]
    test_cat = word_data[test_word]["cat"]
    text = TEMPLATE.format(test_word)

    log_f(f"  Test word: '{test_word}' ({test_cat})")

    # Get baseline logits
    input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
    last_pos = input_ids.shape[1] - 1

    with torch.no_grad():
        base_outputs = model(input_ids)
        base_logits = base_outputs.logits[0, last_pos].detach().float().cpu().numpy()

    # Get the word's token ID for measuring probability
    test_tok_ids = tokenizer.encode(test_word, add_special_tokens=False)
    test_tok_id = test_tok_ids[0] if test_tok_ids else None

    base_top5 = np.argsort(-base_logits)[:5]
    base_top5_tokens = [tokenizer.decode([t]) for t in base_top5]
    log_f(f"  Baseline top-5: {list(zip(base_top5_tokens, base_logits[base_top5].round(3)))}")

    # Perturb along each PC direction
    epsilon_values = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for pc_idx in range(min(5, n_pca)):
        pc_dir = Vt_gate[pc_idx]  # [n_inter]
        pc_var = s_gate[pc_idx] ** 2
        results[f"PC{pc_idx}"] = {"var": float(pc_var)}

        for eps in epsilon_values:
            # Modify gate: g → g + eps * pc_dir
            # We need to do this through hook intervention
            gate_delta = eps * pc_dir

            # Strategy: Hook into MLP to modify the gate output
            captured_logits = {}

            def make_gate_perturb_hook(delta, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    # output shape: [1, seq_len, n_inter]
                    # Apply perturbation to last position
                    out_perturbed = out.clone()
                    out_perturbed[0, last_pos, :] += torch.tensor(
                        delta, dtype=out.dtype, device=out.device
                    )
                    if isinstance(output, tuple):
                        return (out_perturbed,) + output[1:]
                    return out_perturbed
                return hook

            # We need to hook into the gate_proj output (after sigmoid)
            # But gate_proj output is before sigmoid, and we want to modify post-sigmoid
            # Alternative: Modify the h̃ (MLP input) to achieve desired gate change
            # gate = σ(W_gate @ h̃), so Δg ≈ σ'(z) * W_gate @ Δh̃
            # This is too complex. Simpler approach: modify residual stream at this layer

            # Actually, the cleanest approach: modify the MLP output directly
            # Δout = W_down @ (Δg ⊙ ū), where Δg = eps * pc_dir
            # So Δout = W_down @ (eps * pc_dir ⊙ ū)

            # Get ū (mean gate value)
            mean_g = gate_arr.mean(axis=0)
            g_perturbed = mean_g + gate_delta

            # Clamp to valid sigmoid range
            g_perturbed = np.clip(g_perturbed, 0.001, 0.999)

            # Compute MLP output change
            u_test = word_data[test_word]["ups"][target_layer]
            delta_out = W_down @ ((g_perturbed - mean_g) * u_test)

            # Check W_down and W_U visibility of this perturbation
            delta_out_norm = np.linalg.norm(delta_out)

            if delta_out_norm > 1e-10:
                # Project onto W_U row space
                from scipy.sparse.linalg import svds as svds2
                k = min(50, W_U.shape[0] - 2, W_U.shape[1] - 2)
                # Use cached U_wut if available (we don't have it here, use simple dot)
                # W_U @ delta_out gives logit changes
                delta_logits = W_U @ delta_out  # [vocab_size]
                delta_logits_norm = np.linalg.norm(delta_logits)

                # Which tokens are most affected?
                top_affected = np.argsort(-np.abs(delta_logits))[:5]
                top_affected_tokens = [tokenizer.decode([t]) for t in top_affected]
                top_affected_vals = delta_logits[top_affected]

                # Does it affect the test word's logit?
                if test_tok_id is not None:
                    test_logit_change = float(delta_logits[test_tok_id])
                else:
                    test_logit_change = 0.0

                # Cos with concept directions
                cat_concept_dirs = {}
                for c in set(all_cats):
                    cat_words = [w for w in valid_words if word_data[w]["cat"] == c]
                    if len(cat_words) >= 2:
                        cat_logits = np.mean([word_data[w]["logits"] for w in cat_words], axis=0)
                        cat_norm = np.linalg.norm(cat_logits)
                        if cat_norm > 1e-10:
                            cat_concept_dirs[c] = cat_logits / cat_norm

                # Cos between delta_logits and concept directions
                concept_cos = {}
                for c, cdir in cat_concept_dirs.items():
                    concept_cos[c] = proper_cos(delta_logits, cdir)

            else:
                delta_logits_norm = 0
                top_affected_tokens = []
                top_affected_vals = []
                test_logit_change = 0.0
                concept_cos = {}

            results[f"PC{pc_idx}"][f"eps_{eps}"] = {
                "delta_out_norm": float(delta_out_norm),
                "delta_logits_norm": float(delta_logits_norm),
                "test_word_logit_change": float(test_logit_change),
                "top_affected": list(zip(top_affected_tokens,
                                         [float(v) for v in top_affected_vals])),
                "concept_cos": concept_cos,
            }

            log_f(f"  PC{pc_idx} eps={eps}: ||Δout||={delta_out_norm:.6f}, "
                  f"||Δlogits||={delta_logits_norm:.6f}, "
                  f"test_logit_Δ={test_logit_change:.4f}")
            if concept_cos:
                best_cat = max(concept_cos, key=concept_cos.get)
                log_f(f"    Best concept cos: {best_cat}={concept_cos[best_cat]:.4f}")

    return results


# ============================================================
# Exp5: Δg contribution to residual stream Δh
# ============================================================
def exp5_dg_in_residual(word_data, all_words, all_cats, model_info, layers_list, log_f):
    """Decompose residual stream change Δh into gate and up contributions.

    FFN output = W_down @ (g ⊙ u)
    Δh = Δ(FFN) = W_down @ (Δg ⊙ ū) + W_down @ (ḡ ⊙ Δu) + W_down @ (Δg ⊙ Δu)

    Compare: ||W_down@(Δg⊙ū)|| vs ||W_down@(ḡ⊙Δu)|| vs ||Δh||
    """
    log_f("\n" + "="*70)
    log_f("Exp5: Δg Contribution to Residual Stream Δh")
    log_f("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model

    target_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in target_layers:
        target_layers.append(n_layers - 1)
    target_layers = sorted(set(target_layers))

    results = {}

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, model_info.mlp_type)
        W_down = lw.W_down

        valid_words = [w for w in all_words if li in word_data[w]["gates"]
                       and li in word_data[w]["layer_out"]]
        if len(valid_words) < 4:
            continue

        # Compute mean g and u
        all_g = np.array([word_data[w]["gates"][li] for w in valid_words])
        all_u = np.array([word_data[w]["ups"][li] for w in valid_words])
        mean_g = all_g.mean(axis=0)
        mean_u = all_u.mean(axis=0)

        # Category means
        cat_gates = defaultdict(list)
        cat_ups = defaultdict(list)
        cat_h = defaultdict(list)
        for w in valid_words:
            c = word_data[w]["cat"]
            cat_gates[c].append(word_data[w]["gates"][li])
            cat_ups[c].append(word_data[w]["ups"][li])
            cat_h[c].append(word_data[w]["layer_out"][li])

        cat_mean_g = {c: np.mean(gs, axis=0) for c, gs in cat_gates.items()}
        cat_mean_u = {c: np.mean(us, axis=0) for c, us in cat_ups.items()}
        cat_mean_h = {c: np.mean(hs, axis=0) for c, hs in cat_h.items()}

        cat_names = sorted(cat_mean_g.keys())
        if len(cat_names) < 2:
            continue

        # Decompose Δh for each category pair
        gate_frac_list = []
        up_frac_list = []
        cross_frac_list = []

        for i, cA in enumerate(cat_names):
            for j, cB in enumerate(cat_names):
                if i >= j:
                    continue

                dg = cat_mean_g[cA] - cat_mean_g[cB]
                du = cat_mean_u[cA] - cat_mean_u[cB]
                dh = cat_mean_h[cA] - cat_mean_h[cB]

                # FFN decomposition
                gate_term = W_down @ (dg * mean_u)
                up_term = W_down @ (mean_g * du)
                cross_term = W_down @ (dg * du)

                gate_norm = np.linalg.norm(gate_term)
                up_norm = np.linalg.norm(up_term)
                cross_norm = np.linalg.norm(cross_term)
                dh_norm = np.linalg.norm(dh)
                total_ffn = gate_norm + up_norm + cross_norm

                if total_ffn > 1e-10:
                    gate_frac_list.append(gate_norm / total_ffn)
                    up_frac_list.append(up_norm / total_ffn)
                    cross_frac_list.append(cross_norm / total_ffn)

        n_pairs = len(gate_frac_list)
        if n_pairs == 0:
            continue

        results[li] = {
            "n_pairs": n_pairs,
            "gate_frac_mean": float(np.mean(gate_frac_list)),
            "gate_frac_std": float(np.std(gate_frac_list)),
            "up_frac_mean": float(np.mean(up_frac_list)),
            "up_frac_std": float(np.std(up_frac_list)),
            "cross_frac_mean": float(np.mean(cross_frac_list)),
            "cross_frac_std": float(np.std(cross_frac_list)),
        }

        log_f(f"  L{li}: gate_frac={results[li]['gate_frac_mean']:.4f} "
              f"up_frac={results[li]['up_frac_mean']:.4f} "
              f"cross_frac={results[li]['cross_frac_mean']:.4f}")

    log_f(f"\n  === Exp5 Summary ===")
    for li in sorted(results.keys()):
        r = results[li]
        log_f(f"  L{li}: gate={r['gate_frac_mean']:.4f} up={r['up_frac_mean']:.4f} "
              f"cross={r['cross_frac_mean']:.4f}")

    return results


# ============================================================
# Main
# ============================================================
def run_all_experiments(model_name):
    """Run all experiments for a single model."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCLXXXXVII(297): Multi-Readout Life-or-Death Test")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    # Collect data
    log_f("\n--- Collecting activations ---")
    (word_data, all_words, all_cats, model_info, layers_list,
     W_U, U_wut, s_wut, model, tokenizer, device) = collect_all_activations(model_name)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    n_inter = model_info.intermediate_size

    log_f(f"\nModel: {model_info.model_class}, {n_layers} layers, d_model={d_model}, n_inter={n_inter}")

    # Run experiments
    exp1_results = exp1_wdown_vs_wu(
        word_data, all_words, all_cats, model_info, layers_list, W_U, U_wut, s_wut, log_f)

    exp2_results = exp2_probe_classifier(
        word_data, all_words, all_cats, model_info, layers_list, log_f)

    exp3_results = exp3_attention_readout(
        word_data, all_words, all_cats, model_info, layers_list, log_f)

    exp4_results = exp4_perturb_behavioral(
        word_data, all_words, all_cats, model_info, layers_list,
        W_U, model, tokenizer, device, log_f)

    exp5_results = exp5_dg_in_residual(
        word_data, all_words, all_cats, model_info, layers_list, log_f)

    # Final verdict
    log_f(f"\n{'='*70}")
    log_f(f"FINAL VERDICT for {model_name}")
    log_f(f"{'='*70}")

    # Compile evidence
    mid_layers = [li for li in exp1_results if n_layers * 0.3 <= li < n_layers * 0.7]
    if mid_layers:
        avg_dg_wu = np.mean([exp1_results[li]["dg_wu_row_ratio_mean"] for li in mid_layers])
        avg_du_wu = np.mean([exp1_results[li]["du_wu_row_ratio_mean"] for li in mid_layers])
        log_f(f"  Exp1 (W_U readout): Δg→WU_ratio={avg_dg_wu:.4f}, Δu→WU_ratio={avg_du_wu:.4f}")

    if mid_layers:
        avg_g_lin = np.mean([exp2_results.get(li, {}).get("acc_g_linear", 0) for li in mid_layers
                             if li in exp2_results])
        avg_g_nl = np.mean([exp2_results.get(li, {}).get("acc_g_nonlinear", 0) for li in mid_layers
                            if li in exp2_results])
        chance = 1.0 / len(set(all_cats))
        log_f(f"  Exp2 (Probe): Δg_linear={avg_g_lin:.3f}, Δg_nonlinear={avg_g_nl:.3f}, chance={chance:.3f}")

    if mid_layers:
        avg_wo = np.mean([exp3_results.get(li, {}).get("dg_WO_ratio_mean", 0) for li in mid_layers
                          if li in exp3_results])
        log_f(f"  Exp3 (Attention): Δg→WO_ratio={avg_wo:.4f}")

    if mid_layers:
        avg_gf = np.mean([exp5_results.get(li, {}).get("gate_frac_mean", 0) for li in mid_layers
                          if li in exp5_results])
        avg_uf = np.mean([exp5_results.get(li, {}).get("up_frac_mean", 0) for li in mid_layers
                          if li in exp5_results])
        log_f(f"  Exp5 (Δh decomposition): gate_frac={avg_gf:.4f}, up_frac={avg_uf:.4f}")

    # Overall judgment
    log_f(f"\n  ★★★★★ OVERALL JUDGMENT ★★★★★")
    if mid_layers:
        evidence_for_A = 0
        evidence_for_B = 0
        evidence_for_C = 0

        # Exp1 evidence
        if avg_dg_wu < 0.1:
            evidence_for_A += 2
            log_f(f"  Exp1: Δg→WU low ({avg_dg_wu:.4f}) → supports Scenario A")
        elif avg_dg_wu > 0.5:
            evidence_for_B += 2
            log_f(f"  Exp1: Δg→WU high ({avg_dg_wu:.4f}) → supports Scenario B")
        else:
            evidence_for_C += 1
            log_f(f"  Exp1: Δg→WU moderate ({avg_dg_wu:.4f}) → partial support for C")

        # Exp2 evidence
        if avg_g_lin < chance * 1.5:
            evidence_for_A += 2
            log_f(f"  Exp2: Δg probe low ({avg_g_lin:.3f}) → supports Scenario A")
        elif avg_g_lin > chance * 3:
            evidence_for_B += 2
            log_f(f"  Exp2: Δg probe high ({avg_g_lin:.3f}) → supports Scenario B")
        else:
            evidence_for_C += 1
            log_f(f"  Exp2: Δg probe moderate ({avg_g_lin:.3f}) → partial support for C")

        if avg_g_nl > avg_g_lin * 1.5:
            evidence_for_C += 1
            log_f(f"  Exp2: Nonlinear >> Linear → supports Scenario C")

        # Verdict
        log_f(f"\n  Evidence: A={evidence_for_A}, B={evidence_for_B}, C={evidence_for_C}")
        if evidence_for_A >= 3:
            log_f(f"  ★ VERDICT: Scenario A — Δg is a CONTROL VARIABLE (framework holds)")
        elif evidence_for_B >= 3:
            log_f(f"  ★ VERDICT: Scenario B — Δg is a PROJECTION ARTIFACT (framework needs rebuild)")
        else:
            log_f(f"  ★ VERDICT: Scenario C — Mixed evidence (needs refined theory)")

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
