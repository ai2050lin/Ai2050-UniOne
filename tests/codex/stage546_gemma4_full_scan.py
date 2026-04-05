"""
Stage 546: Gemma4 四维不变量全量扫描
====================================
与stage545完全对等的设计，使用Gemma4模型。
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import (
    load_gemma4_model,
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


def compute_encoding_distances(all_encodings, layers, all_words):
    results = {}
    for layer_idx in layers:
        vectors = {w: all_encodings[w][layer_idx].float().cpu() for w in all_words}
        intra_dists = []
        inter_dists = []
        fam_names = list(NOUN_FAMILIES.keys())
        for fi, fk1 in enumerate(fam_names):
            m1 = NOUN_FAMILIES[fk1]["members"]
            for i, w1 in enumerate(m1):
                for j, w2 in enumerate(m1):
                    if j > i:
                        d = 1 - F.cosine_similarity(vectors[w1].unsqueeze(0), vectors[w2].unsqueeze(0)).item()
                        intra_dists.append(d)
            for fj, fk2 in enumerate(fam_names):
                if fj <= fi:
                    continue
                m2 = NOUN_FAMILIES[fk2]["members"]
                for w1 in m1:
                    for w2 in m2:
                        d = 1 - F.cosine_similarity(vectors[w1].unsqueeze(0), vectors[w2].unsqueeze(0)).item()
                        inter_dists.append(d)
        results[layer_idx] = {
            "intra_mean": round(float(np.mean(intra_dists)), 6) if intra_dists else 0,
            "inter_mean": round(float(np.mean(inter_dists)), 6) if inter_dists else 0,
            "intra_inter_ratio": round(float(np.mean(intra_dists)) / max(float(np.mean(inter_dists)), 1e-10), 6),
        }
    return results


def compute_info_geometry(all_encodings, layers, all_words):
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
        }
    return results


def compute_tda(all_encodings, layers, all_words, noun_families):
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
        if max_dist > 1e-6:
            hist, _ = np.histogram(dist_matrix[np.triu_indices(n, k=1)], bins=20)
            hist = hist / max(hist.sum(), 1)
            hist = hist[hist > 0]
            entropy = -sum(p * np.log(p) for p in hist)
        else:
            entropy = 0.0
        intra_dists, inter_dists = [], []
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
            "intra_mean": round(float(np.mean(intra_dists)), 6) if intra_dists else 0,
            "inter_mean": round(float(np.mean(inter_dists)), 6) if inter_dists else 0,
            "intra_inter_ratio": round(float(np.mean(intra_dists)) / max(float(np.mean(inter_dists)), 1e-10), 6),
        }
    return results


def compute_field_control(model, tokenizer, layers):
    results = {}
    for btype, pairs in BINDING_PAIRS.items():
        for bound_text, unbound_text in pairs:
            max_diff_layer = 0
            max_diff = 0
            b_enc_last = None
            u_enc_last = None
            for li in range(len(layers)):
                le = layers[li]
                b_h = get_encoding(model, tokenizer, bound_text, le).float().cpu()
                u_h = get_encoding(model, tokenizer, unbound_text, le).float().cpu()
                diff = float(torch.norm(b_h - u_h))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_layer = li
                    b_enc_last = b_h
                    u_enc_last = u_h
            delta = b_enc_last - u_enc_last
            abs_delta = delta.abs().sort(descending=True).values
            total_energy = abs_delta.sum().item()
            top100 = (abs_delta[:100].sum() / max(total_energy, 1e-10)).item()
            top10 = (abs_delta[:10].sum() / max(total_energy, 1e-10)).item()
            key = f"{btype}_{bound_text[:15]}"
            results[key] = {
                "binding_type": btype,
                "bottleneck_layer_idx": max_diff_layer,
                "top10_concentration": round(top10, 6),
                "top100_concentration": round(top100, 6),
                "field_vs_point": "FIELD" if top100 < 0.50 else "POINT",
            }
    return results


def compute_binding_efficiency(model, tokenizer, layers):
    type_scores = {}
    for btype, pairs in BINDING_PAIRS.items():
        n_pairs = len(pairs)
        layer_binding = []
        for bound_text, unbound_text in pairs:
            for li in range(len(layers)):
                le = layers[li]
                b_h = get_encoding(model, tokenizer, bound_text, le).float()
                u_h = get_encoding(model, tokenizer, unbound_text, le).float()
                diff = float(torch.norm(b_h - u_h))
                layer_binding.append(diff)
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
        }
    ranking = sorted(type_scores.keys(), key=lambda k: type_scores[k]["efficiency_ratio"], reverse=True)
    return {"type_scores": type_scores, "ranking": ranking}


def main():
    print("=" * 70)
    print("Stage 546: Gemma4 四维不变量全量扫描")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'codex_temp',
        f"stage546_gemma4_full_scan_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1] 加载 Gemma4 模型...")
    model, tokenizer = load_gemma4_model()
    print(f"    设备: {next(model.parameters()).device}")
    
    n_layers = len(discover_layers(model))
    sample_layers = evenly_spaced_layers(model, count=7)
    print(f"    总层数: {n_layers}, 采样层: {sample_layers}")
    
    print(f"\n[2] 获取编码...")
    all_encodings = {}
    for word in ALL_WORDS:
        encs = {}
        for layer_idx in sample_layers:
            encs[layer_idx] = get_encoding(model, tokenizer, word, layer_idx)
        all_encodings[word] = encs
    
    for wclass, words in WORDCLASS_WORDS.items():
        for word in words:
            if word not in all_encodings:
                encs = {}
                for layer_idx in sample_layers:
                    encs[layer_idx] = get_encoding(model, tokenizer, word, layer_idx)
                all_encodings[word] = encs
    
    hidden_dim = list(all_encodings.values())[0][sample_layers[0]].shape[0]
    print(f"    hidden_dim: {hidden_dim}, 总词数: {len(all_encodings)}")
    
    print("\n[3] 编码距离矩阵...")
    dist_results = compute_encoding_distances(all_encodings, sample_layers, ALL_WORDS)
    lr = dist_results[sample_layers[-1]]
    print(f"    最后一层: intra={lr['intra_mean']:.4f}, inter={lr['inter_mean']:.4f}, ratio={lr['intra_inter_ratio']:.4f}")
    
    print("\n[4] 信息几何...")
    geo_results = compute_info_geometry(all_encodings, sample_layers, ALL_WORDS)
    for li in sample_layers:
        gr = geo_results[li]
        print(f"    L{li}: eff_dim90={gr['effective_dim_90']}, cond={gr['condition_number']:.1e}, nonzero={gr['n_nonzero_eigs']}")
    
    print("\n[5] 拓扑分析...")
    tda_results = compute_tda(all_encodings, sample_layers, ALL_WORDS, NOUN_FAMILIES)
    for li in sample_layers:
        tr = tda_results[li]
        print(f"    L{li}: topo_entropy={tr['topo_entropy']:.4f}, max_dist={tr['max_dist']:.4f}, i/i={tr['intra_inter_ratio']:.4f}")
    
    print("\n[6] 场控制杆...")
    field_results = compute_field_control(model, tokenizer, sample_layers)
    field_count = sum(1 for v in field_results.values() if v["field_vs_point"] == "FIELD")
    print(f"    FIELD: {field_count}/{len(field_results)}")
    
    print("\n[7] 绑定效率排名...")
    binding_eff = compute_binding_efficiency(model, tokenizer, sample_layers)
    print(f"    排名: {binding_eff['ranking']}")
    
    # 维度坍缩
    print("\n[8] 维度坍缩分析...")
    dim_per_layer = [(li, geo_results[li]["effective_dim_90"]) for li in sample_layers]
    collapse_norm = None
    for i, (li, dim) in enumerate(dim_per_layer):
        if dim <= 1 and (i == 0 or dim_per_layer[i-1][1] > 1):
            collapse_norm = li / max(n_layers - 1, 1)
            print(f"    维度坍缩: L{li} (归一化: {collapse_norm:.4f})")
            break
    if collapse_norm is None:
        print(f"    未检测到维度坍缩 (L0 dim={dim_per_layer[0][1]})")
    
    # 拓扑三层
    topo_per_layer = [(li, tda_results[li]["topo_entropy"]) for li in sample_layers]
    early_t = topo_per_layer[0][1]
    mid_t = float(np.mean([v for _, v in topo_per_layer[1:-1]]))
    late_t = topo_per_layer[-1][1]
    three_layer = "yes" if early_t > mid_t and late_t > mid_t else "no"
    print(f"    拓扑三层: early={early_t:.4f}, mid={mid_t:.4f}, late={late_t:.4f} → {three_layer}")
    
    results = {
        "model": "Gemma4",
        "timestamp": timestamp,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "hidden_dim": hidden_dim,
        "encoding_distances": {str(k): v for k, v in dist_results.items()},
        "info_geometry": {str(k): v for k, v in geo_results.items()},
        "tda": {str(k): v for k, v in tda_results.items()},
        "field_control": field_results,
        "binding_efficiency": binding_eff,
        "summary": {
            "dim_collapse_normalized": collapse_norm,
            "topo_three_layer": three_layer,
            "field_control_ratio": f"{field_count}/{len(field_results)}",
            "binding_ranking": binding_eff["ranking"],
            "last_layer_intra_inter": dist_results[sample_layers[-1]]["intra_inter_ratio"],
        },
    }
    
    output_path = os.path.join(output_dir, "gemma4_full_scan.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[9] 结果已保存: {output_path}")
    
    free_model(model)
    print("模型已释放。")
    return output_path


if __name__ == "__main__":
    main()
