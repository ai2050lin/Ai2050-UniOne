"""
CCLXXXXIX(299): 关键对照 — Perturb Δu vs Perturb Δg
=========================================================
CCLXXXXVIII发现perturb Δg不改变top-1预测。
关键对照: perturb Δu是否也不改变top-1?

如果Δu也不改变top-1 → 中间层perturb本身不改变预测(残差连接稀释)
如果Δu改变top-1 → Δg确实是"调节"而非"选择"!

同时测试更大的eps(5.0, 10.0)看是否因为perturb太弱。

用法:
  python cclxxxxix_du_vs_dg_perturb.py --model qwen3
"""
import argparse, os, sys, time, gc
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
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "cclxxxxix_log.txt"

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


def run_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCLXXXXIX(299): Perturb Δu vs Perturb Δg — THE ULTIMATE TEST")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

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
        sel = rng.choice(words, 8, replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * 8)

    from scipy.sparse.linalg import svds

    # Collect activations
    log_f("\n--- Collecting activations ---")
    word_data = {}
    for wi, word in enumerate(all_words):
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        mlp_input = {}
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

        with torch.no_grad():
            outputs = model(input_ids)
            base_logits = outputs.logits[0, last_pos].detach().float().cpu().numpy()

        for h in hooks:
            h.remove()

        word_data[word] = {"cat": all_cats[wi], "gates": {}, "ups": {}, "logits": base_logits}

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

    log_f(f"  Collected {len(all_words)} words")

    # ===== THE KEY EXPERIMENT =====
    # For each target layer, compare perturb Δg vs perturb Δu
    
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    test_words = ["dog", "pizza", "hammer", "mountain"]  # One per category
    eps_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    log_f(f"\n{'='*70}")
    log_f(f"THE ULTIMATE TEST: Perturb Δu vs Perturb Δg")
    log_f(f"Target layers: {target_layers}")
    log_f(f"Test words: {test_words}")
    log_f(f"Epsilon values: {eps_values}")
    log_f(f"{'='*70}")

    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down = lw.W_down

        # Compute PCA of gates and ups
        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        gate_arr = np.array([word_data[w]["gates"][li] for w in valid_words], dtype=np.float32)
        up_arr = np.array([word_data[w]["ups"][li] for w in valid_words], dtype=np.float32)

        gate_centered = gate_arr - gate_arr.mean(axis=0)
        up_centered = up_arr - up_arr.mean(axis=0)

        n_pca = min(10, gate_centered.shape[0] - 1, gate_centered.shape[1] - 1)
        U_g, s_g, Vt_g = svds(gate_centered, k=n_pca)
        sort_g = np.argsort(-s_g); Vt_g = Vt_g[sort_g]; s_g = s_g[sort_g]

        U_u, s_u, Vt_u = svds(up_centered, k=n_pca)
        sort_u = np.argsort(-s_u); Vt_u = Vt_u[sort_u]; s_u = s_u[sort_u]

        mean_g = gate_arr.mean(axis=0)
        mean_u = up_arr.mean(axis=0)

        log_f(f"\n  --- Layer L{li} ---")
        log_f(f"  Gate PCA top-5: {s_g[:5].round(2)}")
        log_f(f"  Up PCA top-5: {s_u[:5].round(2)}")

        for word in test_words:
            if word not in word_data or li not in word_data[word]["gates"]:
                continue
            cat = word_data[word]["cat"]
            g_w = word_data[word]["gates"][li]
            u_w = word_data[word]["ups"][li]
            base_logits = word_data[word]["logits"]
            base_top1 = np.argmax(base_logits)
            base_top1_tok = tokenizer.decode([base_top1])

            log_f(f"\n  Word: '{word}' ({cat}), base_top1='{base_top1_tok}'")

            for pc_idx in [0, 1]:  # Top 2 PCs only
                for eps in eps_values:
                    # ===== PERTURB Δg =====
                    dg = eps * Vt_g[pc_idx]
                    g_perturbed = np.clip(mean_g + dg, 0.001, 0.999)
                    delta_out_g = W_down @ ((g_perturbed - mean_g) * u_w)
                    delta_logits_g = W_U @ delta_out_g
                    perturbed_logits_g = base_logits + delta_logits_g
                    top1_g = np.argmax(perturbed_logits_g)
                    top1_changed_g = int(base_top1 != top1_g)
                    top1_tok_g = tokenizer.decode([top1_g])

                    # ===== PERTURB Δu =====
                    du = eps * Vt_u[pc_idx]
                    u_perturbed = mean_u + du
                    delta_out_u = W_down @ (g_w * (u_perturbed - mean_u))
                    delta_logits_u = W_U @ delta_out_u
                    perturbed_logits_u = base_logits + delta_logits_u
                    top1_u = np.argmax(perturbed_logits_u)
                    top1_changed_u = int(base_top1 != top1_u)
                    top1_tok_u = tokenizer.decode([top1_u])

                    # Logit margin change
                    word_tok_ids = tokenizer.encode(word, add_special_tokens=False)
                    word_tok_id = word_tok_ids[0] if word_tok_ids else -1
                    if word_tok_id >= 0:
                        margin_g = perturbed_logits_g[word_tok_id] - np.max(
                            np.delete(perturbed_logits_g, word_tok_id))
                        margin_u = perturbed_logits_u[word_tok_id] - np.max(
                            np.delete(perturbed_logits_u, word_tok_id))
                        base_margin = base_logits[word_tok_id] - np.max(
                            np.delete(base_logits, word_tok_id))
                    else:
                        margin_g = margin_u = base_margin = 0

                    log_f(f"    PC{pc_idx} eps={eps:5.1f}: "
                          f"Δg: ||Δlog||={np.linalg.norm(delta_logits_g):7.2f}, "
                          f"top1_change={top1_changed_g}({top1_tok_g[:8]:8s}), "
                          f"margin={margin_g:+.3f}  |  "
                          f"Δu: ||Δlog||={np.linalg.norm(delta_logits_u):7.2f}, "
                          f"top1_change={top1_changed_u}({top1_tok_u[:8]:8s}), "
                          f"margin={margin_u:+.3f}")

    # ===== SUMMARY =====
    log_f(f"\n{'='*70}")
    log_f(f"SUMMARY: Δg vs Δu Perturb Comparison")
    log_f(f"{'='*70}")

    # Collect all results across layers/words/PCs
    all_results = []
    for li in target_layers:
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down = lw.W_down

        valid_words = [w for w in all_words if li in word_data[w]["gates"]]
        gate_arr = np.array([word_data[w]["gates"][li] for w in valid_words], dtype=np.float32)
        up_arr = np.array([word_data[w]["ups"][li] for w in valid_words], dtype=np.float32)
        gate_centered = gate_arr - gate_arr.mean(axis=0)
        up_centered = up_arr - up_arr.mean(axis=0)
        n_pca = min(10, gate_centered.shape[0] - 1, gate_centered.shape[1] - 1)
        _, _, Vt_g = svds(gate_centered, k=n_pca)
        _, _, Vt_u = svds(up_centered, k=n_pca)
        mean_g = gate_arr.mean(axis=0)
        mean_u = up_arr.mean(axis=0)

        for word in test_words:
            if word not in word_data or li not in word_data[word]["gates"]:
                continue
            g_w = word_data[word]["gates"][li]
            u_w = word_data[word]["ups"][li]
            base_logits = word_data[word]["logits"]
            base_top1 = np.argmax(base_logits)

            for pc_idx in range(min(5, n_pca)):
                for eps in [1.0, 5.0, 10.0]:
                    # Δg perturb
                    dg = eps * Vt_g[pc_idx]
                    g_perturbed = np.clip(mean_g + dg, 0.001, 0.999)
                    delta_out_g = W_down @ ((g_perturbed - mean_g) * u_w)
                    delta_logits_g = W_U @ delta_out_g
                    top1_g = np.argmax(base_logits + delta_logits_g)
                    top1_changed_g = int(base_top1 != top1_g)

                    # Δu perturb
                    du = eps * Vt_u[pc_idx]
                    u_perturbed = mean_u + du
                    delta_out_u = W_down @ (g_w * (u_perturbed - mean_u))
                    delta_logits_u = W_U @ delta_out_u
                    top1_u = np.argmax(base_logits + delta_logits_u)
                    top1_changed_u = int(base_top1 != top1_u)

                    all_results.append({
                        "layer": li, "word": word, "pc": pc_idx, "eps": eps,
                        "dg_top1_changed": top1_changed_g,
                        "du_top1_changed": top1_changed_u,
                        "dg_delta_logits_norm": float(np.linalg.norm(delta_logits_g)),
                        "du_delta_logits_norm": float(np.linalg.norm(delta_logits_u)),
                    })

    # Aggregate
    dg_top1_rate = np.mean([r["dg_top1_changed"] for r in all_results])
    du_top1_rate = np.mean([r["du_top1_changed"] for r in all_results])
    dg_dlog = np.mean([r["dg_delta_logits_norm"] for r in all_results])
    du_dlog = np.mean([r["du_delta_logits_norm"] for r in all_results])

    # By eps
    for eps in [1.0, 5.0, 10.0]:
        subset = [r for r in all_results if r["eps"] == eps]
        dg_rate = np.mean([r["dg_top1_changed"] for r in subset])
        du_rate = np.mean([r["du_top1_changed"] for r in subset])
        dg_log = np.mean([r["dg_delta_logits_norm"] for r in subset])
        du_log = np.mean([r["du_delta_logits_norm"] for r in subset])
        log_f(f"  eps={eps:5.1f}: Δg top1_rate={dg_rate:.3f} (||Δlog||={dg_log:.1f}), "
              f"Δu top1_rate={du_rate:.3f} (||Δlog||={du_log:.1f})")

    log_f(f"\n  Overall: Δg top1_rate={dg_top1_rate:.3f}, Δu top1_rate={du_top1_rate:.3f}")
    log_f(f"  Overall: Δg ||Δlogits||={dg_dlog:.1f}, Δu ||Δlogits||={du_dlog:.1f}")

    log_f(f"\n  ★★★ CRITICAL JUDGMENT ★★★")
    if du_top1_rate > dg_top1_rate * 2:
        log_f(f"  → Δu changes top-1 much more than Δg → Δg is MODULATION, Δu is SELECTION")
    elif du_top1_rate > dg_top1_rate * 1.3:
        log_f(f"  → Δu changes top-1 somewhat more than Δg → Weak evidence for modulation")
    else:
        log_f(f"  → Δu and Δg both don't change top-1 → Middle-layer perturb doesn't affect prediction")
        log_f(f"  → Need to test with hook-based real perturbation (not additive)")

    release_model(model)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    log_f(f"\n  Results saved to {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_experiment(args.model)
