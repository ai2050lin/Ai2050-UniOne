# -*- coding: utf-8 -*-
"""
Qwen3 概念编码结构相似性与异构性深度分析 (Structural Similarity & Heterogeneity)
=============================================================================
目标：通过前面抽取的海量高能神经基底指纹 (Top 30 Indices)，从数学距离空间严格测定类别内和类别间的表征重合度。
算法：Jaccard Similarity (交并比)
- 相似性 (Homogeneity): 同分类内不同概念的平均 Jaccard 指数。用于衡量同类概念的特征抱团/同化强度。
- 异构性退化 (Heterogeneity limit): 不同分类概念的平均 Jaccard 指数。如果这个值在深层飙升，说明大模型在深层的表征极度坍缩，跨领域的特征被迫使用了同一频道。
"""

import os
import json
import numpy as np
from itertools import combinations

def calculate_jaccard(setA, setB):
    intersect = len(setA.intersection(setB))
    union = len(setA.union(setB))
    if union == 0:
        return 0.0
    return intersect / union

def run_similarity_analysis():
    print("\\n🧬 启动概念表征结构的相似度与异构性距离解算...")
    
    input_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tempdata', 'glm5_emergence', 'qwen3_massive_trajectory_reconstructed.json')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    trajectories = data["raw_trajectories"]
    total_layers = data["metadata"]["layers"]
    
    print(f"[*] 成功装载 {len(trajectories)} 条跨域微空间迹线，探测矩阵层厚度 {total_layers} 层。")

    layer_stats = []

    for L in range(total_layers):
        intra_sims = [] # 类内交并比
        inter_sims = [] # 类间交并比
        
        # 为了高效计算，首先提取这一层的所有集合对象
        # format: list of dict {"concept": xxx, "category": cat, "set": set(top_indices)}
        layer_concepts = []
        for t in trajectories:
            layer_trace = t["layer_traces"][L]
            layer_concepts.append({
                "concept": t["concept"],
                "category": t["category"],
                "set": set(layer_trace["top_indices"])
            })
            
        # O(N^2) 两两配对比对特征宇宙的重合引力
        for i in range(len(layer_concepts)):
            for j in range(i + 1, len(layer_concepts)):
                c1 = layer_concepts[i]
                c2 = layer_concepts[j]
                
                sim = calculate_jaccard(c1["set"], c2["set"])
                
                if c1["category"] == c2["category"]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)
                    
        # 核算本深度的时空宏观均值
        mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
        mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
        
        # 为了更直观的对比，计算 相似同化倍率 (Intra/Inter Ratio)
        # 代表本层保持类间特异性距离的能力
        ratio = mean_intra / mean_inter if mean_inter > 0 else 0.0
        
        layer_stats.append({
            "layer": L,
            "intra_similarity_pct": round(mean_intra * 100, 2),
            "inter_similarity_pct": round(mean_inter * 100, 2),
            "specificity_ratio": round(ratio, 2)
        })
        
        if L % 5 == 0 or L == total_layers - 1:
            print(f"   ► [Layer {L:02d}] 同类相似抱团率: {mean_intra*100:5.1f}% | 跨界异构排斥率(越小越排斥): {mean_inter*100:5.1f}% | 领域隔离度倍数: {ratio:.1f}x")

    # 成卷落盘
    output_dir = os.path.dirname(input_file)
    out_file = os.path.join(output_dir, "qwen3_structural_similarity.json")
    
    final_output = {
        "analysis_target": "Jaccard Structural Overlap",
        "layer_evolution": layer_stats
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\\n✅ 全态类内/类间距离演化矩阵测定完毕，已定盘至: {out_file}")

if __name__ == '__main__':
    run_similarity_analysis()
