"""
Phase CLXXX: 随机基线对照实验 — 极简版
=======================================
仅做最关键的P801(Q/K/V对齐度随机基线)
用1个采样层, 200次随机试验
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, numpy as np, torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from model_utils import load_model, get_model_info, get_layers, get_layer_weights, release_model

def to_numpy(x):
    if isinstance(x, np.ndarray): return x.astype(np.float32)
    return x.detach().cpu().float().numpy().astype(np.float32)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

PAIRS = {
    'syntax': [
        ("The cat sits on the mat", "The cats sit on the mat"),
        ("She walks to school", "She walked to school"),
        ("The dog chased the cat", "The cat was chased by the dog"),
        ("He is running fast", "Is he running fast?"),
        ("A bird flies in the sky", "Birds fly in the sky"),
        ("The man reads a book", "The men read books"),
        ("She has eaten dinner", "She had eaten dinner"),
        ("They will go home", "They went home"),
        ("I can see the mountain", "Can I see the mountain?"),
        ("The children play outside", "The child plays outside"),
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),
        ("She walked to school", "She drove to school"),
        ("The king ruled the kingdom", "The queen ruled the kingdom"),
        ("He ate an apple", "He ate an orange"),
        ("The sun is bright", "The moon is bright"),
        ("The man is tall", "The woman is tall"),
        ("I love music", "I hate music"),
        ("The car is fast", "The bicycle is fast"),
        ("She bought a house", "She rented a house"),
        ("The doctor cured patients", "The teacher taught students"),
    ],
    'style': [
        ("The cat sat on the mat", "The feline rested upon the rug"),
        ("She walked to school", "She proceeded to the educational institution"),
        ("He is very happy", "He is exceedingly joyful"),
        ("The food was good", "The cuisine was delectable"),
        ("I think this is right", "In my humble opinion, this appears correct"),
    ],
    'tense': [
        ("I walk to school", "I walked to school"),
        ("She reads a book", "She read a book"),
        ("They play outside", "They played outside"),
        ("He runs every day", "He ran every day"),
        ("We eat dinner together", "We ate dinner together"),
    ],
    'polarity': [
        ("She is happy", "She is not happy"),
        ("The movie was good", "The movie was not good"),
        ("He can swim", "He cannot swim"),
        ("I like this song", "I do not like this song"),
        ("The test was easy", "The test was not easy"),
    ],
}

def get_residuals(model, tokenizer, device, text, layer_idx):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx]
    return to_numpy(hs[0].mean(dim=0))

def extract_func_dirs(model, tokenizer, device, model_name, pairs_dict, layer=0):
    d_model = get_model_info(model, model_name).d_model
    directions = {}
    for dim_name, pairs in pairs_dict.items():
        diffs = []
        for s1, s2 in pairs:
            r1 = get_residuals(model, tokenizer, device, s1, layer)
            r2 = get_residuals(model, tokenizer, device, s2, layer)
            diff = r2 - r1
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                diffs.append(diff / norm)
        if diffs:
            avg = np.mean(diffs, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-8:
                directions[dim_name] = avg / norm

    dim_order = list(directions.keys())
    ortho_dirs, ortho_labels = [], []
    for dn in dim_order:
        v = directions[dn].copy()
        for u in ortho_dirs:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 0.01:
            ortho_dirs.append(v / norm)
            ortho_labels.append(dn)

    if not ortho_dirs:
        return None, None, None
    return np.array(ortho_dirs), len(ortho_dirs), ortho_labels

def random_ortho(n_dirs, d, rng):
    A = rng.standard_normal((n_dirs, d))
    Q, R = np.linalg.qr(A.T)
    V = Q[:, :n_dirs].T
    for i in range(n_dirs):
        V[i] /= np.linalg.norm(V[i])
    return V

def alignment_ratio(W, V_sub, d_model):
    if W.shape[1] == d_model:
        sub_e = np.sum((W @ V_sub.T) ** 2)
    elif W.shape[0] == d_model:
        sub_e = np.sum((V_sub @ W) ** 2)
    else:
        return 0.0
    return float(sub_e / (np.sum(W ** 2) + 1e-30))

def wu_energy(V_sub, W_U):
    proj = W_U @ V_sub.T
    return float(np.sum(proj ** 2) / (np.sum(W_U ** 2) + 1e-30))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["glm4", "qwen3", "deepseek7b"])
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"# Phase CLXXX: Random Baseline ({args.model})", flush=True)
    print(f"{'#'*60}", flush=True)

    # 加载模型
    print("Loading model...", flush=True)
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    layers = get_layers(model)
    d_model = info.d_model
    print(f"  d_model={d_model}, n_layers={info.n_layers}", flush=True)

    # 提取功能方向
    print("Extracting functional directions...", flush=True)
    V_func, n_func, labels = extract_func_dirs(model, tokenizer, device, args.model, PAIRS, layer=0)
    if V_func is None:
        print("ERROR: no functional directions")
        release_model(model)
        sys.exit(1)
    print(f"  n_func={n_func}, labels={labels}", flush=True)

    # 功能方向间余弦
    for i in range(n_func):
        for j in range(i+1, n_func):
            c = abs(float(np.dot(V_func[i], V_func[j])))
            print(f"  |cos({labels[i]},{labels[j]})|={c:.4f}")

    # === P801: Q/K/V对齐度 ===
    print(f"\n--- P801: Q/K/V alignment (n_random={args.n_random}) ---", flush=True)

    # 用中间层
    target_layer = info.n_layers // 2
    lw = get_layer_weights(layers[target_layer], d_model, info.mlp_type)
    weights = {'W_Q': lw.W_q, 'W_K': lw.W_k, 'W_V': lw.W_v, 'W_O': lw.W_o}

    # 功能方向对齐度
    func_ratios = {}
    for wn, W in weights.items():
        func_ratios[wn] = alignment_ratio(W, V_func, d_model)
        print(f"  Functional {wn}: {func_ratios[wn]:.6f}", flush=True)

    # 随机方向对齐度
    rng = np.random.default_rng(args.seed)
    rand_ratios = defaultdict(list)

    for trial in range(args.n_random):
        if trial % 50 == 0:
            print(f"  Random trial {trial}/{args.n_random}...", flush=True)
        V_r = random_ortho(n_func, d_model, rng)
        for wn, W in weights.items():
            rand_ratios[wn].append(alignment_ratio(W, V_r, d_model))

    print(f"  Random trial {args.n_random}/{args.n_random} done!", flush=True)

    # 统计
    print(f"\n--- P801 Results ---", flush=True)
    p801 = {}
    for wn in ['W_Q', 'W_K', 'W_V', 'W_O']:
        rv = np.array(rand_ratios[wn])
        fv = func_ratios[wn]
        rm, rs = float(np.mean(rv)), float(np.std(rv))
        rp95 = float(np.percentile(rv, 95))
        rp99 = float(np.percentile(rv, 99))
        z = (fv - rm) / (rs + 1e-30)
        p = float(np.mean(rv >= fv))
        p801[wn] = {'func': fv, 'rand_mean': rm, 'rand_std': rs,
                     'p95': rp95, 'p99': rp99, 'z': z, 'p': p}
        print(f"  {wn}: func={fv:.6f}, rand={rm:.6f}±{rs:.6f}, "
              f"P95={rp95:.6f}, P99={rp99:.6f}, z={z:.1f}, p={p:.4f}", flush=True)

    # === P802: W_U能量 ===
    print(f"\n--- P802: W_U energy (n_random={args.n_random}) ---", flush=True)

    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    func_wu = wu_energy(V_func, W_U)
    print(f"  Functional W_U energy: {func_wu:.6f} ({func_wu*100:.4f}%)", flush=True)

    rng2 = np.random.default_rng(args.seed)
    rand_wu = []
    for trial in range(args.n_random):
        if trial % 50 == 0:
            print(f"  W_U random trial {trial}/{args.n_random}...", flush=True)
        V_r = random_ortho(n_func, d_model, rng2)
        rand_wu.append(wu_energy(V_r, W_U))

    rand_wu = np.array(rand_wu)
    wm, ws = float(np.mean(rand_wu)), float(np.std(rand_wu))
    wp95 = float(np.percentile(rand_wu, 95))
    wp99 = float(np.percentile(rand_wu, 99))
    wz = (func_wu - wm) / (ws + 1e-30)
    wp = float(np.mean(rand_wu >= func_wu))
    theory = n_func / d_model

    print(f"  Rand W_U: {wm:.6f}±{ws:.6f}, P95={wp95:.6f}, P99={wp99:.6f}", flush=True)
    print(f"  Theory: {theory:.6f}, z={wz:.1f}, p={wp:.4f}", flush=True)

    p802 = {'func': func_wu, 'rand_mean': wm, 'rand_std': ws,
            'p95': wp95, 'p99': wp99, 'z': wz, 'p': wp, 'theory': theory}

    # === P804: 不同维度基线 ===
    print(f"\n--- P804: Null distribution curve ---", flush=True)
    rng3 = np.random.default_rng(args.seed)
    p804 = {}
    for nd in [5, 10, 20, 50, 100]:
        ratios = []
        for _ in range(100):
            V_r = random_ortho(nd, d_model, rng3)
            ratios.append(alignment_ratio(weights['W_Q'], V_r, d_model))
        ratios = np.array(ratios)
        p804[nd] = {'mean': float(np.mean(ratios)), 'std': float(np.std(ratios)),
                     'theory': nd/d_model}
        print(f"  n={nd}: W_Q alignment={np.mean(ratios):.6f}±{np.std(ratios):.6f}, "
              f"theory={nd/d_model:.6f}", flush=True)

    # === 综合判断 ===
    print(f"\n{'#'*60}", flush=True)
    print("# VERDICT", flush=True)
    print(f"{'#'*60}", flush=True)

    all_fake = True
    for wn in ['W_Q', 'W_K', 'W_V', 'W_O']:
        if p801[wn]['p'] < 0.05:
            all_fake = False
            break

    if all_fake and wp > 0.05:
        verdict = "STATISTICAL ARTIFACT - functional-content separation is NOT real!"
    elif all_fake:
        verdict = "QKV alignment is artifact, but W_U energy is real"
    elif wp > 0.05:
        verdict = "W_U energy is artifact, but QKV alignment is real"
    else:
        verdict = "REAL STRUCTURE - functional-content separation confirmed!"

    print(f"\n  QKV verdict: {'artifact' if all_fake else 'real'}")
    print(f"  W_U verdict: {'artifact' if wp > 0.05 else 'real'}")
    print(f"\n  >>> {verdict}", flush=True)

    # 保存
    all_results = {
        'model': args.model,
        'd_model': d_model,
        'n_layers': info.n_layers,
        'timestamp': datetime.now().isoformat(),
        'config': {'n_random': args.n_random, 'seed': args.seed, 'n_func': n_func, 'labels': labels},
        'p801': p801,
        'p802': p802,
        'p804': p804,
        'verdict': verdict,
    }

    out_dir = Path("results/phase_clxxx")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_results.json"

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {out_path}", flush=True)
    release_model(model)
    print("Phase CLXXX done!", flush=True)
