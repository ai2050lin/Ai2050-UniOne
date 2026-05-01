"""
CCC(300): Hook-based真正的因果干预 — 冻结u, 只动g
=========================================================
CCLXXXXIX发现加性perturb不改变top-1(残差连接稀释)。
现在用hook在模型forward中修改gate输出, 让后续层处理修改后的表示。

核心实验:
  Exp1: Hook修改gate → 真正的因果干预
    - 正常forward: 获取baseline预测
    - Hook修改: 在目标层将g替换为另一词的g
    - 测量: top-1是否改变? logit margin是否改变?

  Exp2: Hook修改gate vs Hook修改up
    - 分别用hook修改g和u
    - 比较哪个对最终预测影响更大

用法:
  python ccc_hook_causal.py --model qwen3
  python ccc_hook_causal.py --model glm4
  python ccc_hook_causal.py --model deepseek7b
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
LOG_FILE = TEMP_DIR / "ccc_hook_causal_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert"],
}

TEMPLATE = "The {} is"


def run_hook_experiment(model_name):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log_f(msg=""):
        print(msg)
        log_lines.append(msg)

    log_f(f"\n{'#'*70}")
    log_f(f"CCC(300): Hook-based Causal Intervention")
    log_f(f"Model: {model_name}")
    log_f(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_f(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)

    # ===== Step 1: Collect gate and up values for all words =====
    log_f("\n--- Step 1: Collecting gate/up activations ---")

    rng = np.random.RandomState(42)
    all_words = []
    all_cats = []
    for cat, words in CONCEPTS.items():
        sel = rng.choice(words, min(4, len(words)), replace=False)
        all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
        all_cats.extend([cat] * 4)

    word_gates = {}  # word -> {layer_idx: gate_array}
    word_ups = {}    # word -> {layer_idx: up_array}

    for word in all_words:
        text = TEMPLATE.format(word)
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        mlp_input = {}
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            if hasattr(layer, 'mlp'):
                def make_hook(key):
                    def hook(module, args):
                        if isinstance(args, tuple):
                            mlp_input[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            mlp_input[key] = args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.mlp.register_forward_pre_hook(make_hook(f"L{li}")))

        with torch.no_grad():
            _ = model(input_ids)

        for h in hooks:
            h.remove()

        word_gates[word] = {}
        word_ups[word] = {}
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
            word_gates[word][li] = torch.tensor(g, dtype=torch.bfloat16)
            word_ups[word][li] = torch.tensor(u, dtype=torch.bfloat16)

    log_f(f"  Collected gates/ups for {len(all_words)} words")

    # ===== Step 2: Hook-based causal intervention =====
    log_f(f"\n{'='*70}")
    log_f(f"Step 2: Hook-based Causal Intervention")
    log_f(f"{'='*70}")

    target_layer = n_layers // 2

    # Test word pairs: same-cat and diff-cat
    test_pairs_same = []  # (word_A, word_B) same category
    test_pairs_diff = []  # (word_A, word_B) diff category

    cat_words = defaultdict(list)
    for w, c in zip(all_words, all_cats):
        cat_words[c].append(w)

    # Same-cat pairs
    for cat, words in cat_words.items():
        for i in range(min(2, len(words))):
            for j in range(i+1, min(3, len(words))):
                test_pairs_same.append((words[i], words[j], cat, cat))

    # Diff-cat pairs
    cats = list(cat_words.keys())
    for i in range(min(2, len(cats))):
        for j in range(i+1, min(4, len(cats))):
            w1 = cat_words[cats[i]][0]
            w2 = cat_words[cats[j]][0]
            test_pairs_diff.append((w1, w2, cats[i], cats[j]))

    log_f(f"  Target layer: L{target_layer}")
    log_f(f"  Same-cat pairs: {len(test_pairs_same)}")
    log_f(f"  Diff-cat pairs: {len(test_pairs_diff)}")

    # ===== Intervention function =====
    def run_intervention(word_A, word_B, intervention_type, target_li):
        """Run hook-based intervention.
        
        intervention_type:
          'swap_g': Replace g_A with g_B at target_li
          'swap_u': Replace u_A with u_B at target_li  
          'swap_both': Replace both g and u
          'zero_g': Set g to 0.5 (neutral) at target_li
          'zero_u': Set u to 0 at target_li
        """
        text_A = TEMPLATE.format(word_A)
        input_ids = tokenizer(text_A, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1

        # Get replacement values
        g_B = word_gates[word_B][target_li].to(device)
        u_B = word_ups[word_B][target_li].to(device)
        g_A = word_gates[word_A][target_li].to(device)
        u_A = word_ups[word_A][target_li].to(device)

        intervention_done = [False]

        def make_intervention_hook():
            def hook(module, input, output):
                if intervention_done[0]:
                    return output
                intervention_done[0] = True

                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output

                # output is the MLP output: W_down @ (g * u)
                # We need to modify g or u BEFORE the element-wise multiply
                # But the hook is on the MLP (post-computation)
                # So we need to recompute the output with modified g/u

                # Actually, we should hook into the activation function (silu) output
                # Or we can hook into the MLP pre-forward to modify the input
                # But that's the layernorm output, not g/u

                # Strategy: Hook at MLP level, recompute output
                # We need W_down from the layer
                lw = get_layer_weights(layers_list[target_li], d_model, mlp_type)
                W_down_t = torch.tensor(lw.W_down, dtype=torch.bfloat16, device=device)

                if intervention_type == 'swap_g':
                    new_g = g_B
                    new_u = u_A
                elif intervention_type == 'swap_u':
                    new_g = g_A
                    new_u = u_B
                elif intervention_type == 'swap_both':
                    new_g = g_B
                    new_u = u_B
                elif intervention_type == 'zero_g':
                    new_g = torch.ones_like(g_A) * 0.5
                    new_u = u_A
                elif intervention_type == 'zero_u':
                    new_g = g_A
                    new_u = torch.zeros_like(u_A)
                else:
                    return output

                # Recompute MLP output: W_down @ (new_g * new_u)
                new_out = (W_down_t @ (new_g * new_u)).unsqueeze(0).unsqueeze(0)
                new_out = new_out.expand_as(out)

                if isinstance(output, tuple):
                    return (new_out,) + output[1:]
                return new_out
            return hook

        # Register hook on target layer's MLP
        target_mlp = layers_list[target_li].mlp
        hook_handle = target_mlp.register_forward_hook(make_intervention_hook())

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, last_pos].detach().float().cpu().numpy()

        hook_handle.remove()

        return logits

    # ===== Run interventions =====
    results = []

    for pair_type, pairs in [("same_cat", test_pairs_same), ("diff_cat", test_pairs_diff)]:
        for word_A, word_B, cat_A, cat_B in pairs:
            # Baseline: no intervention
            text_A = TEMPLATE.format(word_A)
            input_ids = tokenizer(text_A, return_tensors="pt").to(device).input_ids
            with torch.no_grad():
                base_logits = model(input_ids).logits[0, -1].detach().float().cpu().numpy()
            base_top1 = np.argmax(base_logits)
            base_top1_tok = tokenizer.decode([base_top1])

            # Interventions
            for itype in ['swap_g', 'swap_u', 'swap_both']:
                try:
                    interv_logits = run_intervention(word_A, word_B, itype, target_layer)
                    interv_top1 = np.argmax(interv_logits)
                    interv_top1_tok = tokenizer.decode([interv_top1])
                    top1_changed = int(base_top1 != interv_top1)

                    # Logit margin
                    word_tok_ids = tokenizer.encode(word_A, add_special_tokens=False)
                    word_tok_id = word_tok_ids[0] if word_tok_ids else -1
                    if word_tok_id >= 0:
                        base_margin = base_logits[word_tok_id] - np.max(np.delete(base_logits, word_tok_id))
                        interv_margin = interv_logits[word_tok_id] - np.max(np.delete(interv_logits, word_tok_id))
                    else:
                        base_margin = interv_margin = 0

                    # Cos similarity of logits
                    logit_cos = float(np.dot(base_logits, interv_logits) / 
                                     (np.linalg.norm(base_logits) * np.linalg.norm(interv_logits) + 1e-10))

                    results.append({
                        "pair_type": pair_type,
                        "word_A": word_A, "word_B": word_B,
                        "cat_A": cat_A, "cat_B": cat_B,
                        "intervention": itype,
                        "base_top1": base_top1_tok,
                        "interv_top1": interv_top1_tok,
                        "top1_changed": top1_changed,
                        "base_margin": float(base_margin),
                        "interv_margin": float(interv_margin),
                        "margin_change": float(interv_margin - base_margin),
                        "logit_cos": logit_cos,
                    })

                    log_f(f"  {pair_type} {word_A}→{word_B} {itype}: "
                          f"top1={base_top1_tok}→{interv_top1_tok} (changed={top1_changed}), "
                          f"margin={base_margin:.2f}→{interv_margin:.2f} (Δ={interv_margin-base_margin:.2f}), "
                          f"logit_cos={logit_cos:.4f}")
                except Exception as e:
                    log_f(f"  {pair_type} {word_A}→{word_B} {itype}: FAILED ({e})")

    # ===== Summary =====
    log_f(f"\n{'='*70}")
    log_f(f"SUMMARY")
    log_f(f"{'='*70}")

    for itype in ['swap_g', 'swap_u', 'swap_both']:
        subset = [r for r in results if r["intervention"] == itype]
        if not subset:
            continue

        top1_rate = np.mean([r["top1_changed"] for r in subset])
        avg_margin_change = np.mean([r["margin_change"] for r in subset])
        avg_logit_cos = np.mean([r["logit_cos"] for r in subset])

        # Same vs diff
        same = [r for r in subset if r["pair_type"] == "same_cat"]
        diff = [r for r in subset if r["pair_type"] == "diff_cat"]

        same_top1 = np.mean([r["top1_changed"] for r in same]) if same else -1
        diff_top1 = np.mean([r["top1_changed"] for r in diff]) if diff else -1
        same_margin = np.mean([r["margin_change"] for r in same]) if same else 0
        diff_margin = np.mean([r["margin_change"] for r in diff]) if diff else 0

        log_f(f"\n  {itype}:")
        log_f(f"    Overall: top1_change_rate={top1_rate:.3f}, avg_margin_Δ={avg_margin_change:.3f}, logit_cos={avg_logit_cos:.4f}")
        log_f(f"    Same-cat: top1_rate={same_top1:.3f}, margin_Δ={same_margin:.3f}")
        log_f(f"    Diff-cat: top1_rate={diff_top1:.3f}, margin_Δ={diff_margin:.3f}")

    # Critical judgment
    swap_g_results = [r for r in results if r["intervention"] == "swap_g"]
    swap_u_results = [r for r in results if r["intervention"] == "swap_u"]

    if swap_g_results and swap_u_results:
        g_top1 = np.mean([r["top1_changed"] for r in swap_g_results])
        u_top1 = np.mean([r["top1_changed"] for r in swap_u_results])
        g_margin = np.mean([r["margin_change"] for r in swap_g_results])
        u_margin = np.mean([r["margin_change"] for r in swap_u_results])

        log_f(f"\n  ★★★ CRITICAL JUDGMENT ★★★")
        log_f(f"  swap_g: top1_rate={g_top1:.3f}, margin_Δ={g_margin:.3f}")
        log_f(f"  swap_u: top1_rate={u_top1:.3f}, margin_Δ={u_margin:.3f}")

        if u_top1 > g_top1 * 1.5:
            log_f(f"  → Δu changes top-1 MORE than Δg → Δu=SELECTION, Δg=MODULATION")
        elif g_top1 > u_top1 * 1.5:
            log_f(f"  → Δg changes top-1 MORE than Δu → UNEXPECTED! Need investigation")
        else:
            log_f(f"  → Δu and Δg have similar top-1 change rates → Both contribute to prediction")

        if abs(g_margin) > abs(u_margin) * 2:
            log_f(f"  → Δg affects confidence MORE → Δg=CONFIDENCE MODULATION")
        elif abs(u_margin) > abs(g_margin) * 2:
            log_f(f"  → Δu affects confidence MORE → Δu=DIRECTION, Δg=GAIN")
        else:
            log_f(f"  → Both affect confidence similarly")

    release_model(model)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    log_f(f"\n  Results saved to {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_hook_experiment(args.model)
