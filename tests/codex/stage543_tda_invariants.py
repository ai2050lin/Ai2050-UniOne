"""
Stage 543: 拓扑数据分析不变量 (Topological Data Analysis Invariants)
=====================================================================
核心思路：用拓扑数据分析（TDA, Persistent Homology）分析编码空间的形状。

数学原理：
1. 持久同调 (Persistent Homology):
   - 对一组点云（名词编码），逐渐增大"邻域半径"ε
   - 在每个ε处，记录点云的拓扑特征：连通分量(H0)、环形(H1)、空腔(H2)
   - 特征的"出生时间"到"死亡时间"的区间叫"持久条"(persistence barcode)
   - 持久时间长的条代表"真实的拓扑结构"，短的代表噪声

2. 关键不变量：
   - Betti数 (Betti numbers): β0=连通分量数, β1=环数
   - 持久熵 (Persistent Entropy): 拓扑复杂度度量
   - 持久景观 (Persistence Landscape): 跨模型可比的函数表示
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import (
    load_qwen3_model,
    discover_layers,
)
from multimodel_language_shared import (
    encode_to_device,
    evenly_spaced_layers,
    free_model,
)

NOUN_FAMILIES = {
    "fruit": {"members": ["apple", "banana", "cherry"], "label": "水果"},
    "animal": {"members": ["cat", "dog", "horse"], "label": "动物"},
    "tool": {"members": ["hammer", "knife", "screwdriver"], "label": "工具"},
    "org": {"members": ["university", "company", "hospital"], "label": "组织"},
    "celestial": {"members": ["sun", "moon", "mars"], "label": "天体"},
    "abstract": {"members": ["freedom", "justice", "truth"], "label": "抽象"},
}
ALL_WORDS = []
for fam in NOUN_FAMILIES.values():
    ALL_WORDS.extend(fam["members"])


def get_hidden_states_at_layer(model, tokenizer, word, layer_idx):
    encoded = encode_to_device(model, tokenizer, word)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    return outputs.hidden_states[layer_idx + 1].squeeze(0).mean(dim=0)


def pairwise_distances(vectors):
    n = len(vectors)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            cos_sim = np.dot(vectors[i], vectors[j]) / max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-10)
            d = 1 - cos_sim
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def compute_persistent_homology_h0(dist_matrix, max_epsilon=None, n_points=None):
    """H0持久同调 - Union-Find算法"""
    if n_points is None:
        n_points = dist_matrix.shape[0]
    if max_epsilon is None:
        max_epsilon = float(dist_matrix.max())

    edges = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            edges.append((dist_matrix[i, j], i, j))
    edges.sort()

    parent = list(range(n_points))
    rank_uf = [0] * n_points

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank_uf[rx] < rank_uf[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank_uf[rx] == rank_uf[ry]:
            rank_uf[rx] += 1
        return True

    barcode = [(0.0, None)] * n_points

    for dist, i, j in edges:
        if dist > max_epsilon:
            break
        if union(i, j):
            for k in range(n_points):
                if barcode[k][1] is None:
                    barcode[k] = (0.0, float(dist))
                    break

    for k in range(n_points):
        if barcode[k][1] is None:
            barcode[k] = (0.0, max_epsilon)

    meaningful = [(b, d) for b, d in barcode if d is not None and (d - b) > 0.001]
    meaningful.sort(key=lambda x: x[1] - x[0], reverse=True)
    return meaningful


def compute_persistent_homology_h1(dist_matrix, max_epsilon=None):
    """H1持久同调 - 简化版（基于图论）"""
    n = dist_matrix.shape[0]
    if max_epsilon is None:
        max_epsilon = float(dist_matrix.max() * 0.8)

    eps_values = np.linspace(0, max_epsilon, 40)
    h1_betti = []

    for eps in eps_values:
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] <= eps:
                    adj[i, j] = adj[j, i] = True

        visited = set()
        n_components = 0
        for start in range(n):
            if start in visited:
                continue
            n_components += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in range(n):
                    if adj[node, neighbor] and neighbor not in visited:
                        stack.append(neighbor)

        n_edges = int(np.sum(adj)) // 2
        n_triangles = 0
        for i in range(n):
            neighbors_i = set(np.where(adj[i])[0])
            for j in neighbors_i:
                if j > i:
                    common = neighbors_i & set(np.where(adj[j])[0])
                    n_triangles += len([k for k in common if k > j])

        h1 = max(0, n_edges - n + n_components - 2 * n_triangles)
        h1_betti.append(h1)

    h1_barcode = []
    prev_h1 = 0
    birth_eps = None
    for idx, (eps, h1) in enumerate(zip(eps_values, h1_betti)):
        if h1 > prev_h1 and birth_eps is None:
            birth_eps = eps
        if h1 < prev_h1 and birth_eps is not None:
            h1_barcode.append((birth_eps, eps))
            birth_eps = None
        prev_h1 = h1
    if birth_eps is not None:
        h1_barcode.append((birth_eps, max_epsilon))
    h1_barcode.sort(key=lambda x: x[1] - x[0], reverse=True)

    return h1_barcode, h1_betti, eps_values.tolist()


def persistence_entropy(barcode, max_epsilon):
    if not barcode:
        return 0.0
    lifetimes = [d - b for b, d in barcode if d is not None]
    if not lifetimes or sum(lifetimes) < 1e-10:
        return 0.0
    total = sum(lifetimes)
    probs = [l / total for l in lifetimes]
    return -sum(p * np.log(p + 1e-10) for p in probs)


def betti_curve(barcode, n_points=30, max_epsilon=None):
    if max_epsilon is None:
        max_e = max(d for _, d in barcode if d is not None)
    else:
        max_e = max_epsilon
    eps_values = np.linspace(0, max_e, n_points).tolist()
    betti_numbers = np.zeros(n_points)
    for b, d in barcode:
        if d is None:
            continue
        for i, eps in enumerate(eps_values):
            if b <= eps < d:
                betti_numbers[i] += 1
    return eps_values, betti_numbers.tolist()


def persistence_landscape(barcode, resolution=30, max_epsilon=None, k=2):
    if max_epsilon is None:
        max_e = max(d for _, d in barcode if d is not None)
    else:
        max_e = max_epsilon
    t_values = np.linspace(0, max_e, resolution).tolist()
    landscape = np.zeros((k, resolution))
    for idx, t in enumerate(t_values):
        values = []
        for b, d in barcode:
            if d is None:
                continue
            values.append(min(t - b, d - t))
        values.sort(reverse=True)
        for ki in range(min(k, len(values))):
            landscape[ki, idx] = max(0, values[ki])
    return t_values, landscape.tolist()


def topological_invariant_summary(all_hidden_states, layer_idx, all_words, noun_families):
    vectors = [all_hidden_states[w][layer_idx].float().cpu().numpy() for w in all_words]
    n = len(vectors)
    dist_matrix = pairwise_distances(vectors)

    h0_barcode = compute_persistent_homology_h0(dist_matrix)
    h0_entropy = persistence_entropy(h0_barcode, float(dist_matrix.max()))

    h1_barcode, h1_betti, h1_eps = compute_persistent_homology_h1(dist_matrix)
    h1_entropy = persistence_entropy(h1_barcode, float(dist_matrix.max()))

    betti_eps, betti_curve_vals = betti_curve(h0_barcode, n_points=30, max_epsilon=float(dist_matrix.max()))
    landscape_t, landscape_vals = persistence_landscape(h0_barcode, resolution=30, max_epsilon=float(dist_matrix.max()), k=2)

    intra_topo = {}
    for fam_key, fam in noun_families.items():
        fam_vectors = [all_hidden_states[w][layer_idx].float().cpu().numpy() for w in fam["members"]]
        if len(fam_vectors) >= 2:
            fam_dist = pairwise_distances(fam_vectors)
            fam_h0 = compute_persistent_homology_h0(fam_dist)
            fam_entropy = persistence_entropy(fam_h0, float(fam_dist.max()))
            intra_topo[fam_key] = {
                "entropy": round(fam_entropy, 6),
                "n_bars": len(fam_h0),
                "max_persistence": round(max((d-b) for b, d in fam_h0 if d), 6) if fam_h0 else 0,
            }

    top_bars = sorted(h0_barcode, key=lambda x: (x[1] or 0) - x[0], reverse=True)

    return {
        "h0_n_bars": len(h0_barcode),
        "h0_top5_persistence": [round((d-b), 6) for b, d in top_bars[:5]],
        "h0_entropy": round(h0_entropy, 6),
        "h1_n_bars": len(h1_barcode),
        "h1_max_betti": int(max(h1_betti)) if h1_betti else 0,
        "h1_entropy": round(h1_entropy, 6),
        "betti_curve": {"eps": betti_eps, "values": betti_curve_vals},
        "landscape": {"t": landscape_t, "values": landscape_vals},
        "intra_family_topology": intra_topo,
        "max_dist": round(float(dist_matrix.max()), 6),
        "mean_dist": round(float(dist_matrix[np.triu_indices(n, k=1)].mean()), 6),
    }


def main():
    print("=" * 70)
    print("Stage 543: 拓扑数据分析不变量 (TDA Invariants)")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'codex_temp',
        f"stage543_tda_invariants_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] 加载 Qwen3 模型...")
    model, tokenizer = load_qwen3_model()
    device = next(model.parameters()).device
    print(f"    设备: {device}")

    all_layers = discover_layers(model)
    n_layers = len(all_layers)
    sample_layers = evenly_spaced_layers(model, count=7)
    print(f"    总层数: {n_layers}, 采样层: {sample_layers}")

    print(f"\n[2] 获取 {len(ALL_WORDS)} 个名词的多层编码...")
    all_hidden_states = {}
    for word in ALL_WORDS:
        hs = {}
        for layer_idx in sample_layers:
            h = get_hidden_states_at_layer(model, tokenizer, word, layer_idx)
            hs[layer_idx] = h
        all_hidden_states[word] = hs

    print(f"    hidden_dim: {list(all_hidden_states.values())[0][sample_layers[0]].shape[0]}")

    print("\n[3] 逐层计算拓扑不变量...")
    layer_topo = {}

    for layer_idx in sample_layers:
        topo = topological_invariant_summary(all_hidden_states, layer_idx, ALL_WORDS, NOUN_FAMILIES)
        layer_topo[layer_idx] = topo

        print(f"\n  Layer {layer_idx}:")
        print(f"    H0: {topo['h0_n_bars']}个条, 熵={topo['h0_entropy']:.4f}")
        print(f"    H0 Top5持久: {topo['h0_top5_persistence']}")
        print(f"    H1: {topo['h1_n_bars']}个条, 最大Betti={topo['h1_max_betti']}")
        print(f"    距离范围: [{topo['mean_dist']:.4f}, {topo['max_dist']:.4f}]")
        for fam_key, ft in topo['intra_family_topology'].items():
            print(f"      {fam_key}: 熵={ft['entropy']:.4f}, 持久={ft['max_persistence']:.4f}")

    # [4] 跨层拓扑演化
    print("\n[4] 跨层拓扑演化...")
    evolution = {}
    layer_list = sorted(sample_layers)
    for metric in ['h0_entropy', 'h1_entropy', 'h0_n_bars', 'max_dist']:
        values = {l: layer_topo[l][metric] for l in layer_list}
        evolution[metric] = values
        deltas = [abs(values[layer_list[i]] - values[layer_list[i-1]]) for i in range(1, len(layer_list))]
        max_idx = int(np.argmax(deltas)) if deltas else 0
        print(f"    {metric}: 最大变化 L{layer_list[max_idx]}→L{layer_list[max_idx+1]} (Δ={deltas[max_idx]:.6f})")

    # [5] 跨层Wasserstein距离
    print("\n[5] 跨层Wasserstein距离...")
    cross_layer = {}
    for i, l1 in enumerate(layer_list):
        for j, l2 in enumerate(layer_list):
            if i < j:
                bc1 = compute_persistent_homology_h0(
                    pairwise_distances([all_hidden_states[w][l1].float().cpu().numpy() for w in ALL_WORDS])
                )
                bc2 = compute_persistent_homology_h0(
                    pairwise_distances([all_hidden_states[w][l2].float().cpu().numpy() for w in ALL_WORDS])
                )
                # 简单的L1距离比较
                n_min = min(len(bc1), len(bc2))
                pers1 = sorted([(d-b) for b, d in bc1], reverse=True)
                pers2 = sorted([(d-b) for b, d in bc2], reverse=True)
                l1_dist = sum(abs(pers1[k] - pers2[k]) for k in range(min(n_min, 10))) / max(min(n_min, 10), 1)
                cross_layer[f"L{l1}_L{l2}"] = round(l1_dist, 6)
    for k, v in sorted(cross_layer.items()):
        print(f"    {k}: {v}")

    results = {
        "model": "Qwen3-4B",
        "timestamp": timestamp,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "n_words": len(ALL_WORDS),
        "layer_topology": {str(k): {kk: vv for kk, vv in v.items() if kk not in ['betti_curve', 'landscape']}
                          for k, v in layer_topo.items()},
        "betti_curves": {str(k): v['betti_curve'] for k, v in layer_topo.items()},
        "landscapes": {str(k): v['landscape'] for k, v in layer_topo.items()},
        "topology_evolution": {k: {str(lk): lv for lk, lv in v.items()} for k, v in evolution.items()},
        "cross_layer_distances": cross_layer,
    }

    output_path = os.path.join(output_dir, "tda_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[6] 结果已保存: {output_path}")

    free_model(model)
    print("\n模型已释放。")
    return output_path


if __name__ == "__main__":
    main()
