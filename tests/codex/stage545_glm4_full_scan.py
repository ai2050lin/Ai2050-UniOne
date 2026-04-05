"""
Stage 545: GLM4 四维不变量全量扫描
==================================
在一个脚本中完成GLM4的全部不变量数据采集：
1. 编码距离矩阵（INV-1/INV-4）
2. 信息几何（INV-7有效维度/条件数）
3. 拓扑分析（INV-8拓扑熵/INV-9晚期家族分离）
4. 场控制杆验证（INV-6）
5. 绑定效率排名（INV-2）
6. 词性层带分布（INV-5）

每个模型加载一次，全部采集完再释放。
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import (
    load_glm4_model,
    discover_layers,
)
from multimodel_language_shared import (
    encode_to_device,
    evenly_spaced_layers,
    free_model,
)

# ========== 测试词汇 ==========
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

WORDCLASS_WORDS = {
    "noun": ["apple", "cat", "hammer", "university", "sun", "freedom"],
    "adj": ["red", "big", "fast", "cold", "old", "new"],
    "verb": ["run", "eat", "think", "build", "move", "grow"],
    "adv": ["quickly", "slowly", "often", "never", "always", "very"],
    "pron": ["he", "she", "it", "they", "we", "you"],
    "prep": ["in", "on", "at", "from", "with", "for"],
}

BINDING_PAIRS = {
    "attribute": [
        ("the red apple", "the apple"),
        ("the big dog", "the dog"),
        ("the sharp knife", "the knife"),
        ("the cold water", "the water"),
        ("the old house", "the house"),
    ],
    "relation": [
        ("apple is a fruit", "apple is"),
        ("cat is an animal", "cat is"),
        ("sun is a star", "sun is"),
        ("water is a liquid", "water is"),
        ("hammer is a tool", "hammer is"),
    ],
    "syntax": [
        ("The apple fell", "apple fell"),
        ("cats chase mice", "cats mice"),
        ("She runs fast", "runs fast"),
        ("He reads books", "reads books"),
        ("They sing songs", "sing songs"),
    ],
    "association": [
        ("apple pie", "apple"),
        ("rain forest", "rain"),
        ("ocean wave", "ocean"),
        ("mountain peak", "mountain"),
        ("river bank", "river"),
    ],
}


def get_encoding(model, tokenizer, text, layer_idx):
    encoded = encode_to_device(model, tokenizer, text)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    return outputs.hidden_states[layer_idx + 1].squeeze(0).mean(dim=0)


# ===== 分析1：编码距离矩阵 =====
def compute_encoding_distances(all_encodings, layers, all_words):
    """INV-1/INV-4: 编码距离矩阵"""
    results = {}
    for layer_idx in layers:
        vectors = {w: all_encodings[w][layer_idx].float().cpu() for w in all_words}
        dist_matrix = {}
        intra_dists = []
        inter_dists = []
        fam_names = list(NOUN_FAMILIES.keys())
        
        for fi, fk1 in enumerate(fam_names):
            m1 = NOUN_FAMILIES[fk1]["members"]
            for i, w1 in enumerate(m1):
                for j, w2 in enumerate(m1):
                    if j > i:
                        d = 1 - F.cosine_similarity(vectors[w1].unsqueeze(0), vectors[w2].unsqueeze(0)).item()
                        dist_matrix[f"{w1}_{w2}"] = round(d, 6)
                        intra_dists.append(d)
            for fj, fk2 in enumerate(fam_names):
                if fj <= fi:
                    continue
                m2 = NOUN_FAMILIES[fk2]["members"]
                for w1 in m1:
                    for w2 in m2:
                        d = 1 - F.cosine_similarity(vectors[w1].unsqueeze(0), vectors[w2].unsqueeze(0)).item()
                        dist_matrix[f"{w1}_{w2}"] = round(d, 6)
                        inter_dists.append(d)
        
        results[layer_idx] = {
            "intra_mean": round(float(np.mean(intra_dists)), 6) if intra_dists else 0,
            "inter_mean": round(float(np.mean(inter_dists)), 6) if inter_dists else 0,
            "intra_inter_ratio": round(float(np.mean(intra_dists)) / max(float(np.mean(inter_dists)), 1e-10), 6),
            "dist_matrix": dist_matrix,
        }
    return results


# ===== 分析2：信息几何 =====
def compute_info_geometry(all_encodings, layers, all_words):
    """INV-7: 有效维度、条件数、本征值谱"""
    results = {}
    for layer_idx in layers:
        matrix = torch.stack([all_encodings[w][layer_idx].float() for w in all_words])
        n = matrix.shape[0]
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        
        eigenvalues, _ = torch.linalg.eigh(cov)
        top_eigs = eigenvalues[eigenvalues > 1e-10].flip(0).cpu().numpy()
        cumsum = np.cumsum(top_eigs) / max(top_eigs.sum(), 1e-10)
        
        nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
        cond_num = float(nonzero_eigs[-1] / nonzero_eigs[0]) if len(nonzero_eigs) > 1 else float('inf')
        
        results[layer_idx] = {
            "effective_dim_90": int(np.searchsorted(cumsum, 0.90) + 1) if len(cumsum) > 0 else 0,
            "effective_dim_95": int(np.searchsorted(cumsum, 0.95) + 1) if len(cumsum) > 0 else 0,
            "effective_dim_99": int(np.searchsorted(cumsum, 0.99) + 1) if len(cumsum) > 0 else 0,
            "condition_number": round(cond_num, 2),
            "n_nonzero_eigs": int(len(nonzero_eigs)),
            "top5_eigenvalues": [round(float(v), 6) for v in top_eigs[:5]],
            "total_variance": round(float(eigenvalues.sum().item()), 6),
        }
    return results


# ===== 分析3：拓扑分析 =====
def compute_tda(all_encodings, layers, all_words, noun_families):
    """INV-8/INV-9: 拓扑熵、拓扑结构"""
    results = {}
    for layer_idx in layers:
        vectors = [all_encodings[w][layer_idx].float().cpu().numpy() for w in all_words]
        n = len(vectors)
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                cos_sim = np.dot(vectors[i], vectors[j]) / max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-10)
                dist_matrix[i, j] = 1 - cos_sim
                dist_matrix[j, i] = dist_matrix[i, j]
        
        max_dist = float(dist_matrix.max())
        mean_dist = float(dist_matrix[np.triu_indices(n, k=1)].mean())
        
        # 拓扑熵（距离直方图的信息熵）
        if max_dist > 1e-6:
            hist, _ = np.histogram(dist_matrix[np.triu_indices(n, k=1)], bins=20)
            hist = hist / max(hist.sum(), 1)
            hist = hist[hist > 0]
            entropy = -sum(p * np.log(p) for p in hist)
        else:
            entropy = 0.0
        
        # H0持久同调（简化：持久条数和最大持久时间）
        triu_dists = dist_matrix[np.triu_indices(n, k=1)]
        sorted_dists = np.sort(triu_dists)
        if len(sorted_dists) > 0:
            max_persistence = float(sorted_dists[-1] - sorted_dists[0])
        else:
            max_persistence = 0.0
        
        # 家族内/间距离
        intra_dists = []
        inter_dists = []
        fam_names = list(noun_families.keys())
        for fi, fk1 in enumerate(fam_names):
            m1 = noun_families[fk1]["members"]
            for i, w1 in enumerate(m1):
                idx1 = all_words.index(w1)
                for j, w2 in enumerate(m1):
                    if j > i:
                        idx2 = all_words.index(w2)
                        intra_dists.append(dist_matrix[idx1, idx2])
            for fj, fk2 in enumerate(fam_names):
                if fj <= fi:
                    continue
                m2 = noun_families[fk2]["members"]
                for w1 in m1:
                    idx1 = all_words.index(w1)
                    for w2 in m2:
                        idx2 = all_words.index(w2)
                        inter_dists.append(dist_matrix[idx1, idx2])
        
        results[layer_idx] = {
            "topo_entropy": round(entropy, 6),
            "max_dist": round(max_dist, 6),
            "mean_dist": round(mean_dist, 6),
            "h0_max_persistence": round(max_persistence, 6),
            "intra_mean": round(float(np.mean(intra_dists)), 6) if intra_dists else 0,
            "inter_mean": round(float(np.mean(inter_dists)), 6) if inter_dists else 0,
            "intra_inter_ratio": round(float(np.mean(intra_dists)) / max(float(np.mean(inter_dists)), 1e-10), 6),
        }
    return results


# ===== 分析4：场控制杆 =====
def compute_field_control(model, tokenizer, layers):
    """INV-6: 场控制杆vs点控制杆"""
    results = {}
    for btype, pairs in BINDING_PAIRS.items():
        for bound_text, unbound_text in pairs:
            # 获取差异向量
            b_enc = {}
            u_enc = {}
            for li in range(len(layers)):
                le = layers[li]
                b_h = get_encoding(model, tokenizer, bound_text, le)
                u_h = get_encoding(model, tokenizer, unbound_text, le)
                b_enc[li] = b_h.float().cpu()
                u_enc[li] = u_h.float().cpu()
            
            # 找瓶颈层（差异最大的层）
            max_diff_layer = 0
            max_diff = 0
            for li in range(len(layers)):
                diff = float(torch.norm(b_enc[li] - u_enc[li]))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_layer = li
            
            # 在瓶颈层计算场vs点
            delta = b_enc[max_diff_layer] - u_enc[max_diff_layer]
            abs_delta = delta.abs().sort(descending=True).values
            
            total_energy = abs_delta.sum().item()
            top1 = (abs_delta[0] / max(total_energy, 1e-10)).item()
            top10 = (abs_delta[:10].sum() / max(total_energy, 1e-10)).item()
            top100 = (abs_delta[:100].sum() / max(total_energy, 1e-10)).item()
            
            # 统计量变化
            delta_mean = float(delta.mean())
            delta_std = float(delta.std())
            
            # 判定
            is_field = top100 < 0.50
            
            key = f"{btype}_{bound_text[:15]}"
            results[key] = {
                "binding_type": btype,
                "bottleneck_layer_idx": max_diff_layer,
                "delta_norm": round(max_diff, 6),
                "top1_concentration": round(top1, 6),
                "top10_concentration": round(top10, 6),
                "top100_concentration": round(top100, 6),
                "delta_mean": round(delta_mean, 6),
                "delta_std": round(delta_std, 6),
                "field_vs_point": "FIELD" if is_field else "POINT",
            }
    return results


# ===== 分析5：绑定效率排名 =====
def compute_binding_efficiency(model, tokenizer, layers):
    """INV-2: 四类绑定效率排名"""
    type_scores = {}
    for btype, pairs in BINDING_PAIRS.items():
        layer_binding = []
        for bound_text, unbound_text in pairs:
            for li in range(len(layers)):
                le = layers[li]
                b_h = get_encoding(model, tokenizer, bound_text, le)
                u_h = get_encoding(model, tokenizer, unbound_text, le)
                diff = float(torch.norm(b_h.float() - u_h.float()))
                layer_binding.append(diff)
        
        # 按层取平均
        n_pairs = len(pairs)
        avg_per_layer = []
        for li in range(len(layers)):
            vals = [layer_binding[pi * len(layers) + li] for pi in range(n_pairs)]
            avg_per_layer.append(float(np.mean(vals)))
        
        bottleneck_val = max(avg_per_layer)
        avg_val = float(np.mean(avg_per_layer))
        efficiency = bottleneck_val / max(avg_val, 1e-10)
        bottleneck_layer_idx = int(np.argmax(avg_per_layer))
        
        type_scores[btype] = {
            "efficiency_ratio": round(efficiency, 6),
            "bottleneck_layer_idx": bottleneck_layer_idx,
            "bottleneck_binding": round(bottleneck_val, 6),
            "avg_binding": round(avg_val, 6),
        }
    
    # 排名
    ranking = sorted(type_scores.keys(), key=lambda k: type_scores[k]["efficiency_ratio"], reverse=True)
    
    return {
        "type_scores": type_scores,
        "ranking": ranking,
    }


# ===== 分析6：词性层带分布 =====
def compute_wordclass_distribution(all_encodings, layers, wordclass_words):
    """INV-5: 六类词性的层带激活分布"""
    results = {}
    for wclass, words in wordclass_words.items():
        layer_norms = {}
        for layer_idx in layers:
            norms = [float(torch.norm(all_encodings[w][layer_idx])) for w in words if w in all_encodings]
            if norms:
                layer_norms[layer_idx] = {
                    "mean_norm": round(float(np.mean(norms)), 6),
                    "std_norm": round(float(np.std(norms)), 6),
                }
        results[wclass] = layer_norms
    return results


def main():
    print("=" * 70)
    print("Stage 545: GLM4 四维不变量全量扫描")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'codex_temp',
        f"stage545_glm4_full_scan_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1] 加载 GLM4 模型...")
    model, tokenizer = load_glm4_model()
    print(f"    设备: {next(model.parameters()).device}")
    
    n_layers = len(discover_layers(model))
    sample_layers = evenly_spaced_layers(model, count=7)
    print(f"    总层数: {n_layers}, 采样层: {sample_layers}")
    
    # 获取所有编码
    print(f"\n[2] 获取编码（名词+词性+绑定）...")
    noun_encodings = {}
    for word in ALL_WORDS:
        encs = {}
        for layer_idx in sample_layers:
            encs[layer_idx] = get_encoding(model, tokenizer, word, layer_idx)
        noun_encodings[word] = encs
    
    wc_encodings = {}
    for wclass, words in WORDCLASS_WORDS.items():
        for word in words:
            if word not in noun_encodings:
                encs = {}
                for layer_idx in sample_layers:
                    encs[layer_idx] = get_encoding(model, tokenizer, word, layer_idx)
                wc_encodings[word] = encs
    
    all_encodings = {**noun_encodings, **wc_encodings}
    hidden_dim = list(noun_encodings.values())[0][sample_layers[0]].shape[0]
    print(f"    hidden_dim: {hidden_dim}")
    print(f"    名词编码: {len(noun_encodings)} 词")
    print(f"    总编码: {len(all_encodings)} 词")
    
    # ===== 运行全部分析 =====
    print("\n[3] 分析1: 编码距离矩阵...")
    dist_results = compute_encoding_distances(noun_encodings, sample_layers, ALL_WORDS)
    last_layer = sample_layers[-1]
    lr = dist_results[last_layer]
    print(f"    最后一层: intra={lr['intra_mean']:.4f}, inter={lr['inter_mean']:.4f}, ratio={lr['intra_inter_ratio']:.4f}")
    
    print("\n[4] 分析2: 信息几何...")
    geo_results = compute_info_geometry(noun_encodings, sample_layers, ALL_WORDS)
    for li in sample_layers:
        gr = geo_results[li]
        print(f"    L{li}: eff_dim90={gr['effective_dim_90']}, cond={gr['condition_number']:.1e}, nonzero={gr['n_nonzero_eigs']}")
    
    print("\n[5] 分析3: 拓扑分析...")
    tda_results = compute_tda(noun_encodings, sample_layers, ALL_WORDS, NOUN_FAMILIES)
    for li in sample_layers:
        tr = tda_results[li]
        print(f"    L{li}: topo_entropy={tr['topo_entropy']:.4f}, max_dist={tr['max_dist']:.4f}, intra/inter={tr['intra_inter_ratio']:.4f}")
    
    print("\n[6] 分析4: 场控制杆...")
    field_results = compute_field_control(model, tokenizer, sample_layers)
    field_count = sum(1 for v in field_results.values() if v["field_vs_point"] == "FIELD")
    point_count = len(field_results) - field_count
    avg_top100 = float(np.mean([v["top100_concentration"] for v in field_results.values()]))
    print(f"    FIELD: {field_count}, POINT: {point_count}, avg_top100={avg_top100:.4f}")
    
    print("\n[7] 分析5: 绑定效率排名...")
    binding_eff = compute_binding_efficiency(model, tokenizer, sample_layers)
    print(f"    排名: {binding_eff['ranking']}")
    for bt, bs in binding_eff["type_scores"].items():
        print(f"      {bt}: 效率比={bs['efficiency_ratio']:.4f}, 瓶颈层idx={bs['bottleneck_layer_idx']}")
    
    print("\n[8] 分析6: 词性层带分布...")
    wc_dist = compute_wordclass_distribution(all_encodings, sample_layers, WORDCLASS_WORDS)
    for wclass in ["noun", "adj", "verb", "adv", "pron", "prep"]:
        if wclass in wc_dist:
            layer_norms = wc_dist[wclass]
            max_layer = max(layer_norms.keys(), key=lambda l: layer_norms[l]["mean_norm"])
            print(f"    {wclass}: 最大激活层={max_layer}, norm={layer_norms[max_layer]['mean_norm']:.4f}")
    
    # ===== 寻找维度坍缩层 =====
    print("\n[9] 维度坍缩分析...")
    dim_per_layer = [(li, geo_results[li]["effective_dim_90"]) for li in sample_layers]
    collapse_norm = None
    for i, (li, dim) in enumerate(dim_per_layer):
        if dim <= 1 and (i == 0 or dim_per_layer[i-1][1] > 1):
            collapse_norm = li / max(n_layers - 1, 1)
            print(f"    维度坍缩发生在 L{li} (归一化: {collapse_norm:.4f})")
            break
    
    if collapse_norm is None:
        if dim_per_layer[0][1] <= 1:
            collapse_norm = 0.0
            print("    维度从一开始就坍缩")
        else:
            print("    未检测到维度坍缩")
    
    # ===== 拓扑三层结构 =====
    topo_entropy_per_layer = [(li, tda_results[li]["topo_entropy"]) for li in sample_layers]
    early_topo = topo_entropy_per_layer[0][1]
    mid_topo = float(np.mean([v for _, v in topo_entropy_per_layer[1:-1]]))
    late_topo = topo_entropy_per_layer[-1][1]
    three_layer = "early_high_mid_low_late_mid" if early_topo > mid_topo and late_topo > mid_topo else "other"
    print(f"    拓扑三层: early={early_topo:.4f}, mid={mid_topo:.4f}, late={late_topo:.4f} → {three_layer}")
    
    # ===== 保存 =====
    results = {
        "model": "GLM4",
        "timestamp": timestamp,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "hidden_dim": hidden_dim,
        "encoding_distances": {str(k): {kk: vv for kk, vv in v.items() if kk != "dist_matrix"} for k, v in dist_results.items()},
        "info_geometry": {str(k): v for k, v in geo_results.items()},
        "tda": {str(k): v for k, v in tda_results.items()},
        "field_control": {k: v for k, v in field_results.items()},
        "binding_efficiency": binding_eff,
        "wordclass_distribution": {k: {str(lk): lv for lk, lv in v.items()} for k, v in wc_dist.items()},
        "summary": {
            "dim_collapse_normalized": collapse_norm,
            "topo_three_layer": three_layer,
            "field_control_ratio": f"{field_count}/{len(field_results)}",
            "binding_ranking": binding_eff["ranking"],
            "last_layer_intra_inter": dist_results[last_layer]["intra_inter_ratio"],
        },
    }
    
    output_path = os.path.join(output_dir, "glm4_full_scan.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[10] 结果已保存: {output_path}")
    
    free_model(model)
    print("模型已释放。")
    return output_path


if __name__ == "__main__":
    main()
