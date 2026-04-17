"""
Phase CLXXXII: W_U逆向工程 — 从输出矩阵反推功能结构
====================================================
CLXXX-CLXXXI发现: 5维功能方向在线性对齐度和因果干预中与随机不可区分
唯一例外: Qwen3的W_U能量z=127 (功能方向1.49% >> 随机0.195%)

核心问题:
  1. W_U的SVD结构是什么? top奇异向量承载什么信息?
  2. 功能方向对齐了W_U的哪些奇异向量?
  3. Qwen3的W_U为什么特殊? GLM4/DS7B为什么不特殊?
  4. W_U的能量集中是"功能选择"还是"W_U结构"?

实验设计:
  P821: W_U SVD分析 — 奇异值谱、能量分布
  P822: 功能方向在W_U奇异空间中的分布 — 对齐了哪些奇异向量?
  P823: 随机方向在W_U奇异空间中的分布 — 对照组
  P824: W_U top奇异向量的语义分析 — 这些方向承载什么?
  P825: 扩展功能维度 — 用50-200维PCA，测W_U对齐度vs随机
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, numpy as np, torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from model_utils import load_model, get_model_info, release_model

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


def extract_func_dirs(model, tokenizer, device, model_name, pairs_dict, layer=0):
    d_model = get_model_info(model, model_name).d_model
    directions = {}
    for dim_name, pairs in pairs_dict.items():
        diffs = []
        for s1, s2 in pairs:
            tokens1 = tokenizer(s1, return_tensors="pt").to(device)
            tokens2 = tokenizer(s2, return_tensors="pt").to(device)
            with torch.no_grad():
                h1 = model(**tokens1, output_hidden_states=True).hidden_states[layer][0].mean(0)
                h2 = model(**tokens2, output_hidden_states=True).hidden_states[layer][0].mean(0)
            diff = to_numpy(h2 - h1)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["glm4", "qwen3", "deepseek7b"])
    parser.add_argument("--n-random", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'#'*60}", flush=True)
    print(f"# Phase CLXXXII: W_U Reverse Engineering ({args.model})", flush=True)
    print(f"{'#'*60}", flush=True)

    # 加载模型
    print("Loading model...", flush=True)
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
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

    # 获取W_U
    print("Extracting W_U...", flush=True)
    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    vocab_size = W_U.shape[0]
    print(f"  W_U shape: {W_U.shape}", flush=True)

    # === P821: W_U SVD分析 ===
    print(f"\n--- P821: W_U SVD analysis ---", flush=True)
    
    # W_U太大,用截断SVD
    from sklearn.decomposition import TruncatedSVD
    n_svd = min(200, d_model, W_U.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd.fit(W_U)
    
    s_wu = svd.singular_values_  # [n_svd]
    Vt_wu = svd.components_  # [n_svd, d_model]
    # U_wu可以通过transform获取: W_U @ Vt_wu.T
    U_wu = svd.transform(W_U)  # [vocab, n_svd] = W_U @ Vt_wu.T
    
    total_energy = np.sum(W_U ** 2)
    cum_energy = np.cumsum(s_wu ** 2) / total_energy
    
    print(f"  Top singular values: {s_wu[:10]}", flush=True)
    print(f"  Energy in top-1: {s_wu[0]**2/total_energy*100:.2f}%", flush=True)
    print(f"  Energy in top-5: {cum_energy[min(4,n_svd-1)]*100:.2f}%", flush=True)
    print(f"  Energy in top-10: {cum_energy[min(9,n_svd-1)]*100:.2f}%", flush=True)
    print(f"  Energy in top-50: {cum_energy[min(49,n_svd-1)]*100:.2f}%", flush=True)
    print(f"  Energy in top-100: {cum_energy[min(99,n_svd-1)]*100:.2f}%", flush=True)
    eff90 = int(np.searchsorted(cum_energy, 0.9)) if np.any(cum_energy >= 0.9) else n_svd
    eff95 = int(np.searchsorted(cum_energy, 0.95)) if np.any(cum_energy >= 0.95) else n_svd
    eff99 = int(np.searchsorted(cum_energy, 0.99)) if np.any(cum_energy >= 0.99) else n_svd
    print(f"  Effective rank (90% energy): {eff90}", flush=True)
    print(f"  Effective rank (95% energy): {eff95}", flush=True)
    print(f"  Effective rank (99% energy): {eff99}", flush=True)
    
    p821 = {
        'top_sv': s_wu[:20].tolist(),
        'energy_top1': float(s_wu[0]**2/total_energy),
        'energy_top5': float(cum_energy[min(4,n_svd-1)]),
        'energy_top10': float(cum_energy[min(9,n_svd-1)]),
        'energy_top50': float(cum_energy[min(49,n_svd-1)]),
        'energy_top100': float(cum_energy[min(99,n_svd-1)]),
        'eff_rank_90': eff90,
        'eff_rank_95': eff95,
        'eff_rank_99': eff99,
    }

    # === P822: 功能方向在W_U奇异空间中的分布 ===
    print(f"\n--- P822: Functional directions in W_U singular space ---", flush=True)
    
    # Vt_wu的每一行是一个d_model维的奇异向量
    # 功能方向V_func [n_func, d_model] 在Vt_wu中的投影
    func_sing_proj = V_func @ Vt_wu.T  # [n_func, rank] - 每个功能方向在各奇异向量上的投影
    
    # 每个功能方向在前k个奇异向量上的能量
    for i, dim_name in enumerate(labels):
        proj = func_sing_proj[i]  # [rank]
        proj_energy = proj ** 2  # 每个奇异向量上的能量
        total_func_proj = np.sum(proj_energy)
        
        # 各奇异向量的贡献
        top_indices = np.argsort(proj_energy)[::-1][:10]
        print(f"  {dim_name}: total_proj_energy={total_func_proj:.6f}", flush=True)
        print(f"    Top aligned singular vectors: {top_indices[:5]}", flush=True)
        for idx in top_indices[:5]:
            print(f"      SV[{idx}] energy={proj_energy[idx]:.6f} "
                  f"({proj_energy[idx]/total_func_proj*100:.2f}% of func projection)", flush=True)
    
    # 功能方向的W_U能量分解: 按奇异向量
    # W_U @ V_func.T = U_wu @ diag(s_wu) @ Vt_wu @ V_func.T
    # 即: W_U @ V_func.T[:, i] = sum_j s_j * (V_func[i] @ Vt_wu[j]) * U_wu[j]
    func_wu_proj = W_U @ V_func.T  # [vocab, n_func]
    func_wu_energy_per_dim = np.sum(func_wu_proj ** 2, axis=0)  # [n_func]
    total_func_wu = np.sum(func_wu_energy_per_dim)
    
    # 按奇异值分解
    # W_U @ V_func.T = U @ S @ Vt @ V_func.T
    # 每个奇异值j对功能方向i的贡献 = s_j^2 * (V_func[i] @ Vt_wu[j])^2
    contrib_matrix = np.zeros((n_func, len(s_wu)))
    for j in range(len(s_wu)):
        for i in range(n_func):
            contrib_matrix[i, j] = s_wu[j]**2 * (func_sing_proj[i, j])**2
    
    # 每个功能维度的能量来源(top奇异向量)
    p822 = {}
    for i, dim_name in enumerate(labels):
        dim_energy = func_wu_energy_per_dim[i]
        top_sv_idx = np.argsort(contrib_matrix[i])[::-1][:10]
        top_contrib = contrib_matrix[i][top_sv_idx]
        p822[dim_name] = {
            'total_energy': float(dim_energy),
            'energy_pct': float(dim_energy / np.sum(W_U ** 2) * 100),
            'top_sv_contributors': top_sv_idx.tolist(),
            'top_sv_contributions': top_contrib.tolist(),
            'top_sv_pct': [float(c/dim_energy*100) for c in top_contrib],
        }
        print(f"  {dim_name}: W_U energy={dim_energy:.4f} ({dim_energy/np.sum(W_U**2)*100:.4f}%)", flush=True)
        print(f"    Top SV contributors: {top_sv_idx[:5]}", flush=True)
        print(f"    Top SV contributions: {top_contrib[:5].tolist()}", flush=True)
        print(f"    Top SV % of dim: {[f'{c/dim_energy*100:.1f}%' for c in top_contrib[:5]]}", flush=True)

    # === P823: 随机方向在W_U奇异空间中的分布 ===
    print(f"\n--- P823: Random directions in W_U singular space ---", flush=True)
    
    rng = np.random.default_rng(args.seed)
    rand_wu_energies = []
    rand_sv_distributions = defaultdict(list)
    
    for trial in range(args.n_random):
        V_r = random_ortho(n_func, d_model, rng)
        
        # 随机方向的W_U能量
        rand_proj = W_U @ V_r.T  # [vocab, n_func]
        rand_energy = np.sum(rand_proj ** 2)
        rand_wu_energies.append(rand_energy / np.sum(W_U ** 2))
        
        # 随机方向在奇异空间中的投影
        rand_sing_proj = V_r @ Vt_wu.T
        for i in range(n_func):
            proj_e = rand_sing_proj[i] ** 2
            rand_sv_distributions[i].append(proj_e)
    
    rand_wu_energies = np.array(rand_wu_energies)
    
    # 随机方向的奇异向量对齐分布
    rand_sv_alignment = {}
    for i in range(n_func):
        all_proj = np.array(rand_sv_distributions[i])  # [n_random, rank]
        mean_proj = np.mean(all_proj, axis=0)  # [rank]
        # top奇异向量的对齐
        top_indices = np.argsort(mean_proj)[::-1][:10]
        rand_sv_alignment[f'rand_dim_{i}'] = {
            'mean_alignment_top': top_indices[:5].tolist(),
            'mean_alignment_vals': mean_proj[top_indices[:5]].tolist(),
        }
    
    print(f"  Random W_U energy: {np.mean(rand_wu_energies):.6f} +/- {np.std(rand_wu_energies):.6f}", flush=True)
    print(f"  Theory: {n_func/d_model:.6f}", flush=True)
    print(f"  Func W_U energy: {total_func_wu/np.sum(W_U**2):.6f}", flush=True)
    print(f"  z-score: {(total_func_wu/np.sum(W_U**2) - np.mean(rand_wu_energies))/(np.std(rand_wu_energies)+1e-30):.1f}", flush=True)

    p823 = {
        'rand_wu_mean': float(np.mean(rand_wu_energies)),
        'rand_wu_std': float(np.std(rand_wu_energies)),
        'func_wu': float(total_func_wu / np.sum(W_U**2)),
        'z': float((total_func_wu/np.sum(W_U**2) - np.mean(rand_wu_energies))/(np.std(rand_wu_energies)+1e-30)),
    }

    # === P824: W_U top奇异向量的语义分析 ===
    print(f"\n--- P824: W_U top singular vectors - semantic analysis ---", flush=True)
    
    # 对top奇异向量，找出W_U中哪些词的行最对齐
    # Vt_wu[j]是第j个d_model维奇异向量
    # W_U的第i行(词i)在该方向的投影 = U_wu[i,j] * s_wu[j]
    
    p824 = {}
    for sv_idx in [0, 1, 2, 3, 4, 9, 19, 49]:
        if sv_idx >= len(s_wu):
            continue
        # 找对齐该奇异向量最对的词
        word_scores = U_wu[:, sv_idx] * s_wu[sv_idx]  # [vocab]
        top_word_idx = np.argsort(np.abs(word_scores))[::-1][:20]
        top_words = []
        for idx in top_word_idx:
            try:
                w = tokenizer.decode([idx])
                top_words.append({'token_id': int(idx), 'word': w, 'score': float(word_scores[idx])})
            except:
                top_words.append({'token_id': int(idx), 'word': f'<{idx}>', 'score': float(word_scores[idx])})
        
        # 该奇异向量与功能方向的对齐
        sv_vec = Vt_wu[sv_idx]  # [d_model]
        func_align = {}
        for i, dim_name in enumerate(labels):
            cos = float(np.abs(np.dot(V_func[i], sv_vec)))
            func_align[dim_name] = cos
        
        p824[f'sv_{sv_idx}'] = {
            'singular_value': float(s_wu[sv_idx]),
            'energy_pct': float(s_wu[sv_idx]**2 / total_energy * 100),
            'top_words': top_words[:10],
            'func_alignment': func_align,
            'max_func_align': max(func_align.values()),
        }
        
        print(f"  SV[{sv_idx}]: s={s_wu[sv_idx]:.2f}, energy={s_wu[sv_idx]**2/total_energy*100:.2f}%", flush=True)
        # 安全打印top words (避免编码问题)
        safe_words = []
        for w in top_words[:5]:
            word_str = w['word'].encode('ascii', 'replace').decode('ascii')
            safe_words.append(word_str)
        print(f"    Top words: {safe_words}", flush=True)
        print(f"    Func alignment: {func_align}", flush=True)
        print(f"    Max func align: {max(func_align.values()):.4f}", flush=True)

    # === P825: 扩展功能维度 — PCA前k维的W_U对齐 ===
    print(f"\n--- P825: Extended functional dimensions (PCA) ---", flush=True)
    
    # 用功能对的差值做PCA，找前k个主方向
    all_diffs = []
    for dim_name, pairs in PAIRS.items():
        for s1, s2 in pairs:
            tokens1 = tokenizer(s1, return_tensors="pt").to(device)
            tokens2 = tokenizer(s2, return_tensors="pt").to(device)
            with torch.no_grad():
                h1 = model(**tokens1, output_hidden_states=True).hidden_states[0][0].mean(0)
                h2 = model(**tokens2, output_hidden_states=True).hidden_states[0][0].mean(0)
            diff = to_numpy(h2 - h1)
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                all_diffs.append(diff / norm)
    
    all_diffs = np.array(all_diffs)  # [n_diffs, d_model]
    print(f"  Total diffs: {all_diffs.shape[0]}", flush=True)
    
    # PCA
    from sklearn.decomposition import PCA
    n_components = min(50, all_diffs.shape[0] - 1, d_model)
    pca = PCA(n_components=n_components)
    pca.fit(all_diffs)
    
    pca_dirs = pca.components_  # [n_components, d_model]
    pca_var = pca.explained_variance_ratio_
    
    print(f"  PCA components: {n_components}", flush=True)
    print(f"  Top-5 variance: {pca_var[:5]}", flush=True)
    print(f"  Top-10 cumulative: {np.cumsum(pca_var)[:10]}", flush=True)
    
    # 不同维度数的W_U对齐度
    p825 = {}
    rng2 = np.random.default_rng(args.seed + 2)
    
    for k in [5, 10, 20, 30, 50]:
        if k > n_components:
            continue
        
        V_pca = pca_dirs[:k]  # [k, d_model]
        
        # PCA方向的W_U能量
        pca_wu_proj = W_U @ V_pca.T  # [vocab, k]
        pca_wu_energy = float(np.sum(pca_wu_proj ** 2) / np.sum(W_U ** 2))
        
        # 随机基线
        rand_wu_k = []
        for _ in range(30):
            V_r = random_ortho(k, d_model, rng2)
            rp = W_U @ V_r.T
            rand_wu_k.append(float(np.sum(rp ** 2) / np.sum(W_U ** 2)))
        rand_wu_k = np.array(rand_wu_k)
        
        z = (pca_wu_energy - np.mean(rand_wu_k)) / (np.std(rand_wu_k) + 1e-30)
        p = float(np.mean(rand_wu_k >= pca_wu_energy))
        theory = k / d_model
        
        p825[f'k_{k}'] = {
            'pca_wu': pca_wu_energy,
            'rand_mean': float(np.mean(rand_wu_k)),
            'rand_std': float(np.std(rand_wu_k)),
            'theory': theory,
            'z': float(z),
            'p': p,
            'pca_var_explained': float(np.sum(pca_var[:k])),
        }
        
        print(f"  k={k}: PCA_WU={pca_wu_energy:.6f}, rand={np.mean(rand_wu_k):.6f}±{np.std(rand_wu_k):.6f}, "
              f"theory={theory:.6f}, z={z:.1f}, p={p:.4f}, var={np.sum(pca_var[:k]):.4f}", flush=True)

    # === 综合判断 ===
    print(f"\n{'#'*60}", flush=True)
    print("# W_U REVERSE ENGINEERING VERDICT", flush=True)
    print(f"{'#'*60}", flush=True)
    
    # W_U是否低秩?
    eff_rank = int(np.searchsorted(cum_energy, 0.9))
    is_low_rank = eff_rank < d_model * 0.5
    
    print(f"  W_U effective rank (90%): {eff_rank}/{d_model}", flush=True)
    print(f"  Is low rank: {is_low_rank}", flush=True)
    
    # 功能方向是否对齐W_U的top奇异向量?
    func_max_sv_align = 0
    for dim_name in labels:
        top_sv = p822[dim_name]['top_sv_contributors'][0]
        if top_sv < 50:  # 对齐前50个奇异向量
            pct = p822[dim_name]['top_sv_pct'][0]
            func_max_sv_align = max(func_max_sv_align, pct)
    
    print(f"  Max func-SV alignment: {func_max_sv_align:.2f}%", flush=True)
    print(f"  W_U z-score (CLXXX): {p823['z']:.1f}", flush=True)
    
    if p823['z'] > 10:
        wu_verdict = "W_U pathway is REAL - training created special structure"
    elif p823['z'] > 2:
        wu_verdict = "W_U pathway is WEAKLY REAL"
    else:
        wu_verdict = "W_U pathway is ARTIFACT"
    
    print(f"\n  >>> {wu_verdict}", flush=True)

    # 保存
    all_results = {
        'model': args.model,
        'd_model': d_model,
        'vocab_size': vocab_size,
        'n_layers': info.n_layers,
        'timestamp': datetime.now().isoformat(),
        'config': {'n_random': args.n_random, 'seed': args.seed, 'n_func': n_func, 'labels': labels},
        'p821': p821,
        'p822': p822,
        'p823': p823,
        'p824': p824,
        'p825': p825,
        'verdict': wu_verdict,
        'eff_rank_90': eff_rank,
    }

    out_dir = Path("results/phase_clxxxii")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_results.json"

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {out_path}", flush=True)
    release_model(model)
    print("Phase CLXXXII done!", flush=True)
