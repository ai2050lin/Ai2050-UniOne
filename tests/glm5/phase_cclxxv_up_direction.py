"""
Phase CCLXXV: Up向量ū的方向编码分析 — 完全基于CCLXXIV成功脚本模板
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

OUTPUT_DIR = Path("results/causal_fiber")

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


def run_all_experiments(model_name):
    print(f"\n{'='*60}")
    print(f"CCLXXV: Running {model_name}")
    print(f"{'='*60}")
    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    print(f"  Model: {n_layers}L, d={d_model}, mlp={mlp_type}")

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
        print(f"  Word {wi+1}/{len(all_words)}: '{word}' ({time.time()-t0:.0f}s)")

    print(f"  Data collection done ({time.time()-t0:.0f}s)")

    # Target layers
    tl = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in tl: tl.append(n_layers - 1)
    tl = sorted(set(tl))
    print(f"  Target layers: {tl}")

    # ============== Exp1: Δu Direction ==============
    print("  === Exp1: Δu Direction ===")
    exp1 = []
    for li in tl:
        vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(vw) < 4: continue
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None: continue
        W_down = lw.W_down
        cm = {}
        for cat in cat_names:
            cw = [w for w, c in zip(all_words, all_cats) if c == cat and li in word_residuals[w]]
            rv = [word_residuals[w][li] for w in cw if word_residuals[w].get(li) is not None]
            if len(rv) >= 2: cm[cat] = np.mean(rv, axis=0)
        if len(cm) < 2: continue
        gc_l, uc_l, xc_l, tc_l, dgu_l, dg_l = [], [], [], [], [], []
        gn_l, un_l, xn_l = [], [], []
        cl = sorted(cm.keys())
        for i, cA in enumerate(cl):
            for j, cB in enumerate(cl):
                if i >= j: continue
                cdir = cm[cA] - cm[cB]
                if np.linalg.norm(cdir) < 1e-8: continue
                wA = [w for w, c in zip(all_words, all_cats) if c == cA and li in word_gates[w]]
                wB = [w for w, c in zip(all_words, all_cats) if c == cB and li in word_gates[w]]
                gA = np.mean([word_gates[w][li] for w in wA], axis=0)
                gB = np.mean([word_gates[w][li] for w in wB], axis=0)
                uA = np.mean([word_ups[w][li] for w in wA], axis=0)
                uB = np.mean([word_ups[w][li] for w in wB], axis=0)
                Dg, Du = gA - gB, uA - uB
                gb, ub = (gA + gB) / 2, (uA + uB) / 2
                gv = W_down @ (Dg * ub)
                uv = W_down @ (gb * Du)
                xv = W_down @ (Dg * Du)
                tv = gv + uv + xv
                gc_l.append(proper_cos(gv, cdir))
                uc_l.append(proper_cos(uv, cdir))
                xc_l.append(proper_cos(xv, cdir))
                tc_l.append(proper_cos(tv, cdir))
                dgu_l.append(proper_cos(W_down @ Du, cdir))
                dg_l.append(proper_cos(W_down @ Dg, cdir))
                gn_l.append(float(np.linalg.norm(gv)))
                un_l.append(float(np.linalg.norm(uv)))
                xn_l.append(float(np.linalg.norm(xv)))
        if not gc_l: continue
        tn = max(sum(gn_l) + sum(un_l) + sum(xn_l), 1e-10)
        lr = {"layer": li, "n_pairs": len(gc_l),
              "gate_cos": float(np.mean(gc_l)), "up_cos": float(np.mean(uc_l)),
              "cross_cos": float(np.mean(xc_l)), "total_cos": float(np.mean(tc_l)),
              "W_down_Du_cos": float(np.mean(dgu_l)), "W_down_Dg_cos": float(np.mean(dg_l)),
              "gate_norm_frac": float(sum(gn_l)/tn), "up_norm_frac": float(sum(un_l)/tn)}
        exp1.append(lr)
        print(f"    L{li}: gate={lr['gate_cos']:+.3f} up={lr['up_cos']:+.3f} cross={lr['cross_cos']:+.3f} Du={lr['W_down_Du_cos']:+.3f} Dg={lr['W_down_Dg_cos']:+.3f}")

    # ============== Exp2: ū Category ==============
    print("  === Exp2: ū Category ===")
    exp2 = []
    for li in tl:
        vw = [w for w in all_words if li in word_ups[w]]
        if len(vw) < 4: continue
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        if lw.W_gate is None: continue
        W_down = lw.W_down
        wup = {w: W_down @ word_ups[w][li] for w in vw}
        scc, dcc = [], []
        for i in range(len(vw)):
            for j in range(i+1, len(vw)):
                c = proper_cos(wup[vw[i]], wup[vw[j]])
                c1 = all_cats[all_words.index(vw[i])]
                c2 = all_cats[all_words.index(vw[j])]
                (scc if c1==c2 else dcc).append(c)
        du_l = [word_ups[vw[i]][li] - word_ups[vw[j]][li]
                for i in range(len(vw)) for j in range(i+1, len(vw))]
        n90_du, top5_du, u_sep = -1, -1, -1
        if len(du_l) >= 10:
            du_a = np.array(du_l, dtype=np.float32)
            du_c = du_a - du_a.mean(axis=0)
            from scipy.sparse.linalg import svds
            np_ = min(50, du_c.shape[0]-1, du_c.shape[1]-1)
            if np_ >= 2:
                _, s, _ = svds(du_c.astype(np.float32), k=np_)
                s = s[np.argsort(-s)]
                tv_ = np.sum(s**2)
                n90_du = int(np.searchsorted(np.cumsum(s**2)/tv_, 0.90)) + 1
                top5_du = float(np.sum(s[:5]**2)/tv_)
        if scc and dcc:
            same_ang = np.mean([np.arccos(np.clip(c, -1, 1)) for c in scc])
            diff_ang = np.mean([np.arccos(np.clip(c, -1, 1)) for c in dcc])
            u_sep = float((1 - np.cos(same_ang)) / max(1 - np.cos(diff_ang), 1e-10))
        lr = {"layer": li, "same_cat_cos": float(np.mean(scc)) if scc else 0,
              "diff_cat_cos": float(np.mean(dcc)) if dcc else 0,
              "cos_sep": float(np.mean(scc)-np.mean(dcc)) if scc and dcc else 0,
              "u_sep_ratio": u_sep, "n90_du": n90_du, "top5_var_du": top5_du}
        exp2.append(lr)
        print(f"    L{li}: sep={lr['cos_sep']:+.3f} n90={lr['n90_du']} u_sep={lr['u_sep_ratio']:.2f}")

    # ============== Exp3: Δu Subspace ==============
    print("  === Exp3: Δu Subspace ===")
    tl8 = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in tl8: tl8.append(n_layers - 1)
    tl8 = sorted(set(tl8))
    from scipy.sparse.linalg import svds
    lpcas = {}
    for li in tl8:
        vw = [w for w in all_words if li in word_gates[w] and li in word_ups[w]]
        if len(vw) < 4: continue
        dg_l = [word_gates[vw[i]][li] - word_gates[vw[j]][li] for i in range(len(vw)) for j in range(i+1,len(vw))]
        du_l = [word_ups[vw[i]][li] - word_ups[vw[j]][li] for i in range(len(vw)) for j in range(i+1,len(vw))]
        if len(dg_l) < 10: continue
        dg_a = np.array(dg_l, dtype=np.float32); du_a = np.array(du_l, dtype=np.float32)
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
        lpcas[li] = {"Vt_dg": Vt_dg[:ns], "Vt_du": Vt_du[:ns]}
    du_rot, same_l = [], []
    ll = sorted(lpcas.keys())
    for i, l1 in enumerate(ll):
        for j, l2 in enumerate(ll):
            if i >= j: continue
            _, ca, _ = np.linalg.svd(lpcas[l1]["Vt_du"] @ lpcas[l2]["Vt_du"].T)
            mean_cos = float(np.mean(ca))
            du_rot.append({"layer1": l1, "layer2": l2, "mean_cos": mean_cos,
                           "mean_angle": float(np.degrees(np.arccos(np.clip(mean_cos, -1, 1))))})
    for li in ll:
        _, ca, _ = np.linalg.svd(lpcas[li]["Vt_dg"] @ lpcas[li]["Vt_du"].T)
        mc = float(np.mean(ca))
        same_l.append({"layer": li, "dg_du_cos": mc,
                       "dg_du_angle": float(np.degrees(np.arccos(np.clip(mc, -1, 1))))})
        print(f"    L{li}: dg_du_cos={mc:.4f} angle={np.degrees(np.arccos(np.clip(mc, -1, 1))):.1f}°")

    # ============== Exp4: Δg-Δu Coordination ==============
    print("  === Exp4: Δg-Δu Coord ===")
    exp4 = []
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
        dg_a = np.array(dg_l, dtype=np.float32); du_a = np.array(du_l, dtype=np.float32)
        pcorrs, sc_corrs, dc_corrs = [], [], []
        for k in range(len(dg_l)):
            adg, adu = np.abs(dg_a[k]), np.abs(du_a[k])
            if np.std(adg) < 1e-10 or np.std(adu) < 1e-10: continue
            r = float(np.corrcoef(adg, adu)[0,1])
            pcorrs.append(r)
            if pc[k][0] == pc[k][1]: sc_corrs.append(r)
            else: dc_corrs.append(r)
        kv = 100
        ovl = []
        for k in range(len(dg_l)):
            td = set(np.argsort(np.abs(dg_a[k]))[-kv:])
            tu = set(np.argsort(np.abs(du_a[k]))[-kv:])
            ovl.append(len(td & tu) / kv)
        ni = dg_a.shape[1]; rovl = kv / ni
        lw_ = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down = lw_.W_down
        wcn = np.linalg.norm(W_down, axis=0)
        wcorrs = []
        for k in range(len(dg_l)):
            adgw = np.abs(dg_a[k]) * wcn; aduw = np.abs(du_a[k]) * wcn
            if np.std(adgw) < 1e-10 or np.std(aduw) < 1e-10: continue
            wcorrs.append(float(np.corrcoef(adgw, aduw)[0,1]))
        lr = {"layer": li, "n_inter": ni,
              "neuron_corr": float(np.mean(pcorrs)) if pcorrs else 0,
              "weighted_corr": float(np.mean(wcorrs)) if wcorrs else 0,
              "overlap_100": float(np.mean(ovl)), "random_overlap": float(rovl),
              "overlap_ratio": float(np.mean(ovl)/rovl) if rovl > 0 else 0,
              "same_cat_corr": float(np.mean(sc_corrs)) if sc_corrs else 0,
              "diff_cat_corr": float(np.mean(dc_corrs)) if dc_corrs else 0}
        exp4.append(lr)
        print(f"    L{li}: corr={lr['neuron_corr']:.3f} wr={lr['weighted_corr']:.3f} ratio={lr['overlap_ratio']:.2f}x same={lr['same_cat_corr']:.3f} diff={lr['diff_cat_corr']:.3f}")

    # Save all results
    out_dir = OUTPUT_DIR / f"{model_name}_cclxxv"
    out_dir.mkdir(parents=True, exist_ok=True)
    def js(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, dict): return {k: js(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [js(x) for x in obj]
        return obj
    ts = datetime.now().isoformat()
    for name, fname, data in [
        ("exp1", "exp1_delta_u_direction.json", exp1),
        ("exp2", "exp2_u_bar_category.json", exp2),
        ("exp4", "exp4_dg_du_coordination.json", exp4),
    ]:
        with open(out_dir / fname, 'w', encoding='utf-8') as f:
            json.dump(js({"experiment": name, "model": model_name, "timestamp": ts, "layer_results": data}), f, indent=2)
    with open(out_dir / "exp3_delta_u_subspace.json", 'w', encoding='utf-8') as f:
        json.dump(js({"experiment": "exp3", "model": model_name, "timestamp": ts,
                      "du_cross_layer": du_rot, "dg_du_same_layer": same_l}), f, indent=2)
    print(f"  Saved to {out_dir}")

    # Summary
    mid1 = [r for r in exp1 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
    if mid1:
        print(f"  EXP1 mid: gate={np.mean([r['gate_cos'] for r in mid1]):+.3f} up={np.mean([r['up_cos'] for r in mid1]):+.3f} Du={np.mean([r['W_down_Du_cos'] for r in mid1]):+.3f}")
    mid4 = [r for r in exp4 if n_layers*0.3 <= r["layer"] < n_layers*0.7]
    if mid4:
        print(f"  EXP4 mid: corr={np.mean([r['neuron_corr'] for r in mid4]):.3f} ratio={np.mean([r['overlap_ratio'] for r in mid4]):.2f}x")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {model_name} COMPLETE ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_all_experiments(args.model)
