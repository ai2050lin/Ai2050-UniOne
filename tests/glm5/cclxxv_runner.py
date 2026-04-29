"""Minimal CCLXXV runner - writes results directly, no stdout dependency."""
import os, sys, json, time, gc, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxv_runner_log.txt"
RESULTS_DIR = r"d:\Ai2050\TransformerLens-Project\results\causal_fiber"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()
    print(msg, flush=True)

sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
import numpy as np
import torch
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

CONCEPTS = {
    "animal": ["dog", "cat", "horse", "bird"],
    "food": ["apple", "bread", "cheese", "rice"],
    "tool": ["hammer", "knife", "scissors", "saw"],
    "vehicle": ["car", "bus", "train", "plane"],
}

def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def run_model(model_name):
    log(f"=== Starting {model_name} ===")
    try:
        log("Loading model...")
        model, tokenizer, device = load_model(model_name)
        model_info = get_model_info(model, model_name)
        n_layers = model_info.n_layers
        d_model = model_info.d_model
        mlp_type = model_info.mlp_type
        layers_list = get_layers(model)
        W_U = get_W_U(model)
        log(f"Model loaded: {n_layers} layers, d_model={d_model}, mlp_type={mlp_type}")

        # Collect data
        template = "The {} is"
        rng = np.random.RandomState(42)
        all_words, all_cats = [], []
        for cat, words in CONCEPTS.items():
            sel = rng.choice(words, min(4, len(words)), replace=False)
            all_words.extend(sel.tolist() if hasattr(sel, 'tolist') else list(sel))
            all_cats.extend([cat] * len(sel))
        cat_names = list(CONCEPTS.keys())

        word_gates, word_ups, word_residuals = {}, {}, {}
        t0 = time.time()
        for wi, word in enumerate(all_words):
            text = template.format(word)
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1

            ln_out, res_out = {}, {}
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                if hasattr(layer, 'mlp'):
                    def make_ffn_pre(key):
                        def hook(module, args):
                            a = args[0] if not isinstance(args, tuple) else args[0]
                            ln_out[key] = a[0, last_pos].detach().float().cpu().numpy()
                        return hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
                def make_layer_out(key):
                    def hook(module, input, output):
                        o = output[0] if isinstance(output, tuple) else output
                        res_out[key] = o[0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.register_forward_hook(make_layer_out(f"L{li}")))

            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks:
                h.remove()

            g_dict, u_dict, r_dict = {}, {}, {}
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
                g_dict[li] = g
                u_dict[li] = u
                r_dict[li] = res_out.get(key, None)

            word_gates[word] = g_dict
            word_ups[word] = u_dict
            word_residuals[word] = r_dict
            log(f"  Word {wi+1}/{len(all_words)}: '{word}' ({time.time()-t0:.0f}s)")

        log(f"Data collection done ({time.time()-t0:.0f}s)")

        # Target layers
        tl = list(range(0, n_layers, max(1, n_layers // 6)))
        if n_layers - 1 not in tl:
            tl.append(n_layers - 1)
        tl = sorted(set(tl))
        log(f"Target layers: {tl}")

        # ============== Exp1: Δu Direction Analysis ==============
        log("=== Exp1: Δu Direction ===")
        exp1_results = []
        for li in tl:
            valid_words = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(valid_words) < 4:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            W_down = lw.W_down

            cat_means = {}
            for cat in cat_names:
                cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
                rv = [word_residuals[w][li] for w in cw if word_residuals[w].get(li) is not None]
                if len(rv) >= 2:
                    cat_means[cat] = np.mean(rv, axis=0)
            if len(cat_means) < 2:
                continue

            gc_list, uc_list, tc_list, dgu_list, dg_list = [], [], [], [], []
            gn_list, un_list = [], []
            cl = sorted(cat_means.keys())
            for i, cA in enumerate(cl):
                for j, cB in enumerate(cl):
                    if i >= j:
                        continue
                    cdir = cat_means[cA] - cat_means[cB]
                    if np.linalg.norm(cdir) < 1e-8:
                        continue
                    wA = [w for w, c in zip(all_words, all_cats) if c == cA and li in word_gates[w]]
                    wB = [w for w, c in zip(all_words, all_cats) if c == cB and li in word_gates[w]]
                    gA = np.mean([word_gates[w][li] for w in wA], axis=0)
                    gB = np.mean([word_gates[w][li] for w in wB], axis=0)
                    uA = np.mean([word_ups[w][li] for w in wA], axis=0)
                    uB = np.mean([word_ups[w][li] for w in wB], axis=0)
                    Dg, Du = gA - gB, uA - uB
                    g_bar, u_bar = (gA + gB) / 2, (uA + uB) / 2

                    gv = W_down @ (Dg * u_bar)
                    uv = W_down @ (g_bar * Du)
                    tv = gv + uv + W_down @ (Dg * Du)
                    gc_list.append(proper_cos(gv, cdir))
                    uc_list.append(proper_cos(uv, cdir))
                    tc_list.append(proper_cos(tv, cdir))
                    dgu_list.append(proper_cos(W_down @ Du, cdir))
                    dg_list.append(proper_cos(W_down @ Dg, cdir))
                    gn_list.append(float(np.linalg.norm(gv)))
                    un_list.append(float(np.linalg.norm(uv)))

            if not gc_list:
                continue
            tn = max(sum(gn_list) + sum(un_list), 1e-10)
            lr = {
                "layer": li, "n_pairs": len(gc_list),
                "gate_cos": float(np.mean(gc_list)), "up_cos": float(np.mean(uc_list)),
                "total_cos": float(np.mean(tc_list)), "W_down_Du_cos": float(np.mean(dgu_list)),
                "W_down_Dg_cos": float(np.mean(dg_list)),
                "gate_norm_frac": float(sum(gn_list)/tn), "up_norm_frac": float(sum(un_list)/tn),
            }
            exp1_results.append(lr)
            log(f"  L{li}: gate_cos={lr['gate_cos']:+.3f}, up_cos={lr['up_cos']:+.3f}, W_down@Δu={lr['W_down_Du_cos']:+.3f}")

        # ============== Exp2: ū Category Encoding ==============
        log("=== Exp2: ū Category ===")
        exp2_results = []
        for li in tl:
            valid_words = [w for w in all_words if li in word_ups[w]]
            if len(valid_words) < 4:
                continue
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            if lw.W_gate is None:
                continue
            W_down = lw.W_down
            wup = {w: W_down @ word_ups[w][li] for w in valid_words}

            scc, dcc = [], []
            for i in range(len(valid_words)):
                for j in range(i+1, len(valid_words)):
                    c = proper_cos(wup[valid_words[i]], wup[valid_words[j]])
                    c1 = all_cats[all_words.index(valid_words[i])]
                    c2 = all_cats[all_words.index(valid_words[j])]
                    (scc if c1==c2 else dcc).append(c)

            # Δu PCA
            du_list = [word_ups[valid_words[i]][li] - word_ups[valid_words[j]][li]
                       for i in range(len(valid_words)) for j in range(i+1, len(valid_words))]
            n90_du, top5_du = -1, -1
            if len(du_list) >= 10:
                du_arr = np.array(du_list, dtype=np.float32)
                du_c = du_arr - du_arr.mean(axis=0)
                from scipy.sparse.linalg import svds
                n_pca = min(50, du_c.shape[0]-1, du_c.shape[1]-1)
                if n_pca >= 2:
                    _, s, _ = svds(du_c.astype(np.float32), k=n_pca)
                    s = s[np.argsort(-s)]
                    tv = np.sum(s**2)
                    n90_du = int(np.searchsorted(np.cumsum(s**2)/tv, 0.90)) + 1
                    top5_du = float(np.sum(s[:5]**2)/tv)

            lr = {
                "layer": li,
                "same_cos": float(np.mean(scc)) if scc else 0,
                "diff_cos": float(np.mean(dcc)) if dcc else 0,
                "cos_sep": float(np.mean(scc)-np.mean(dcc)) if scc and dcc else 0,
                "n90_du": n90_du, "top5_var_du": top5_du,
            }
            exp2_results.append(lr)
            log(f"  L{li}: sep={lr['cos_sep']:+.3f}, n90(Δu)={lr['n90_du']}")

        # ============== Exp3: Δu Subspace ==============
        log("=== Exp3: Δu Subspace ===")
        tl8 = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in tl8: tl8.append(n_layers - 1)
        tl8 = sorted(set(tl8))

        from scipy.sparse.linalg import svds
        layer_pcas = {}
        for li in tl8:
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(vw) < 4: continue
            dg_l = [word_gates[vw[i]][li] - word_gates[vw[j]][li] for i in range(len(vw)) for j in range(i+1,len(vw))]
            du_l = [word_ups[vw[i]][li] - word_ups[vw[j]][li] for i in range(len(vw)) for j in range(i+1,len(vw))]
            if len(dg_l) < 10: continue
            dg_a = np.array(dg_l, dtype=np.float32)
            du_a = np.array(du_l, dtype=np.float32)
            np_dg = min(30, len(dg_l)-1, dg_a.shape[1]-1)
            np_du = min(30, len(du_l)-1, du_a.shape[1]-1)
            if np_dg < 2 or np_du < 2: continue
            dg_c = dg_a - dg_a.mean(axis=0)
            _, s_dg, Vt_dg = svds(dg_c.astype(np.float32), k=np_dg)
            idx = np.argsort(-s_dg); s_dg = s_dg[idx]; Vt_dg = Vt_dg[idx]
            du_c = du_a - du_a.mean(axis=0)
            _, s_du, Vt_du = svds(du_c.astype(np.float32), k=np_du)
            idx2 = np.argsort(-s_du); s_du = s_du[idx2]; Vt_du = Vt_du[idx2]
            ns = min(20, len(s_dg), len(s_du))
            layer_pcas[li] = {"Vt_dg": Vt_dg[:ns], "Vt_du": Vt_du[:ns]}

        du_rotation, same_layer = [], []
        ll = sorted(layer_pcas.keys())
        for i, l1 in enumerate(ll):
            for j, l2 in enumerate(ll):
                if i >= j: continue
                _, ca, _ = np.linalg.svd(layer_pcas[l1]["Vt_du"] @ layer_pcas[l2]["Vt_du"].T)
                du_rotation.append({"l1": l1, "l2": l2, "mean_cos": float(np.mean(ca))})
        for li in ll:
            _, ca, _ = np.linalg.svd(layer_pcas[li]["Vt_dg"] @ layer_pcas[li]["Vt_du"].T)
            same_layer.append({"layer": li, "dg_du_cos": float(np.mean(ca))})
            log(f"  L{li}: Δg↔Δu cos={float(np.mean(ca)):.3f}")

        # ============== Exp4: Δg-Δu Coordination ==============
        log("=== Exp4: Δg-Δu Coord ===")
        exp4_results = []
        for li in tl:
            vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
            if len(vw) < 4: continue
            dg_l, du_l, pc = [], [], []
            for i in range(len(vw)):
                for j in range(i+1, len(vw)):
                    dg_l.append(word_gates[vw[i]][li] - word_gates[vw[j]][li])
                    du_l.append(word_ups[vw[i]][li] - word_ups[vw[j]][li])
                    pc.append((all_cats[all_words.index(vw[i])], all_cats[all_words.index(vw[j])]))
            if len(dg_l) < 10: continue
            dg_a = np.array(dg_l, dtype=np.float32)
            du_a = np.array(du_l, dtype=np.float32)

            pcorrs = []
            for k in range(len(dg_l)):
                adg, adu = np.abs(dg_a[k]), np.abs(du_a[k])
                if np.std(adg) < 1e-10 or np.std(adu) < 1e-10: continue
                pcorrs.append(float(np.corrcoef(adg, adu)[0,1]))

            # Overlap
            kv = 100
            ovl = []
            for k in range(len(dg_l)):
                td = set(np.argsort(np.abs(dg_a[k]))[-kv:])
                tu = set(np.argsort(np.abs(du_a[k]))[-kv:])
                ovl.append(len(td & tu) / kv)
            n_inter = dg_a.shape[1]
            rovl = kv / n_inter

            # Weighted
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            W_down = lw.W_down
            wcn = np.linalg.norm(W_down, axis=0)
            wcorrs = []
            for k in range(len(dg_l)):
                adgw = np.abs(dg_a[k]) * wcn
                aduw = np.abs(du_a[k]) * wcn
                if np.std(adgw) < 1e-10 or np.std(aduw) < 1e-10: continue
                wcorrs.append(float(np.corrcoef(adgw, aduw)[0,1]))

            lr = {
                "layer": li, "n_inter": n_inter,
                "neuron_corr": float(np.mean(pcorrs)) if pcorrs else 0,
                "weighted_corr": float(np.mean(wcorrs)) if wcorrs else 0,
                "overlap_100": float(np.mean(ovl)), "random_overlap": float(rovl),
                "overlap_ratio": float(np.mean(ovl)/rovl) if rovl > 0 else 0,
            }
            exp4_results.append(lr)
            log(f"  L{li}: corr={lr['neuron_corr']:.3f}, weighted={lr['weighted_corr']:.3f}, ratio={lr['overlap_ratio']:.2f}x")

        # Save all results
        out_dir = os.path.join(RESULTS_DIR, f"{model_name}_cclxxv")
        os.makedirs(out_dir, exist_ok=True)

        def js(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            if isinstance(obj, dict): return {k: js(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [js(x) for x in obj]
            return obj

        for name, data in [("exp1", exp1_results), ("exp2", exp2_results), ("exp4", exp4_results)]:
            with open(os.path.join(out_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(js({"experiment": name, "model": model_name, "timestamp": datetime.now().isoformat(), "layer_results": data}), f, indent=2)

        with open(os.path.join(out_dir, "exp3.json"), 'w', encoding='utf-8') as f:
            json.dump(js({"experiment": "exp3", "model": model_name, "timestamp": datetime.now().isoformat(),
                          "du_cross_layer": du_rotation, "dg_du_same_layer": same_layer}), f, indent=2)

        log(f"All results saved to {out_dir}")

        # Print summary
        mid1 = [r for r in exp1_results if n_layers*0.3 <= r["layer"] < n_layers*0.7]
        if mid1:
            log(f"Exp1 mid: gate_cos={np.mean([r['gate_cos'] for r in mid1]):+.3f}, "
                f"up_cos={np.mean([r['up_cos'] for r in mid1]):+.3f}, "
                f"W_down@Δu={np.mean([r['W_down_Du_cos'] for r in mid1]):+.3f}")
        mid4 = [r for r in exp4_results if n_layers*0.3 <= r["layer"] < n_layers*0.7]
        if mid4:
            log(f"Exp4 mid: corr={np.mean([r['neuron_corr'] for r in mid4]):.3f}, "
                f"overlap_ratio={np.mean([r['overlap_ratio'] for r in mid4]):.2f}x")

        release_model(model)
        log(f"=== {model_name} COMPLETE ===")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())

from datetime import datetime

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_model(args.model)
