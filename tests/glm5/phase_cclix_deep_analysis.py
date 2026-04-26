"""
Phase CCLIX 深度分析: 概念编码的机制性分解
=============================================
目的: 从初步统计发现→机制性理解

核心问题:
  1. 残差流被"句式模板"信号主导 → 需要提取概念特异性成分
  2. 类别信号弱(cos gap=0.05) → 是真的弱还是被模板信号掩盖?
  3. 抽象链方向cos≈-0.2 → 归一化后是否不同?

分析方法:
  A. 概念特异性分解: h_concept - h_mean(所有概念) → 去模板信号
  B. 逐层线性探针: 用概念特异性成分分类类别/抽象级别
  C. 归一化抽象链分析: 方向cos在归一化空间中
  D. 信息论度量: 概念编码的互信息

输出: 填充 PUZZLE_FRAMEWORK.md 的 KN-1a/KN-1b/KN-3a/KN-3b 格子

用法:
  python phase_cclix_deep_analysis.py --model qwen3
"""
import argparse, os, sys, json, time, gc
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
    collect_layer_outputs,
)

OUTPUT_DIR = Path("results/causal_fiber")

# 概念定义 (与phase_cclix相同)
CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano"],
}
ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
for cat, items in CONCEPTS.items():
    for item in items:
        ALL_CONCEPTS.append(item)
        CONCEPT_CATEGORIES[item] = cat

ABSTRACTION_CHAINS = [
    ["apple", "fruit", "food", "substance", "object"],
    ["dog", "animal", "organism", "being", "entity"],
    ["hammer", "tool", "instrument", "artifact", "object"],
    ["mountain", "terrain", "geography", "nature", "object"],
    ["rice", "grain", "food", "substance", "object"],
]

CONTEXT_TEMPLATES = [
    "The {concept} is",
    "I think about the {concept}",
    "The {concept} has",
]


def proper_cos(v1, v2):
    """正确的余弦相似度"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_all_residuals(model, tokenizer, device, model_info):
    """收集所有概念在所有模板下的残差流"""
    n_layers = model_info.n_layers
    
    # 收集: {concept: {template_idx: {layer: vec}}}
    all_residuals = {}
    for t_idx, template in enumerate(CONTEXT_TEMPLATES):
        for concept in ALL_CONCEPTS:
            prompt = template.format(concept=concept)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).detach().clone()
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            layer_outputs = collect_layer_outputs(
                model, inputs_embeds, position_ids, n_layers, target_type="layer"
            )
            
            if concept not in all_residuals:
                all_residuals[concept] = {}
            
            for li in range(n_layers):
                key = f"L{li}"
                if key in layer_outputs:
                    vec = layer_outputs[key][0, -1, :].numpy().astype(np.float32)
                    if li not in all_residuals[concept]:
                        all_residuals[concept][li] = []
                    all_residuals[concept][li].append(vec)
        
        print(f"  Template {t_idx} done")
    
    # 跨模板平均
    averaged = {}
    for concept in ALL_CONCEPTS:
        averaged[concept] = {}
        for li in range(n_layers):
            if li in all_residuals[concept]:
                averaged[concept][li] = np.mean(all_residuals[concept][li], axis=0)
    
    return averaged


def analysis_A_concept_specific_decomposition(averaged_residuals, model_info):
    """
    A. 概念特异性分解
    
    h_specific = h_concept - h_mean(所有概念)
    
    这去除了所有概念共享的"句式模板"信号,
    只留下概念之间的差异信号。
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    results = {}
    
    for li in range(n_layers):
        # 收集该层所有概念的残差
        concept_vecs = {}
        for concept in ALL_CONCEPTS:
            if li in averaged_residuals.get(concept, {}):
                concept_vecs[concept] = averaged_residuals[concept][li]
        
        if len(concept_vecs) < 3:
            continue
        
        # 全局均值 (句式模板信号)
        mean_vec = np.mean(list(concept_vecs.values()), axis=0)
        
        # 概念特异性成分
        specific_vecs = {}
        for concept, vec in concept_vecs.items():
            specific_vecs[concept] = vec - mean_vec
        
        # 1. 概念特异性的范数 (概念信号强度)
        specific_norms = {c: float(np.linalg.norm(v)) for c, v in specific_vecs.items()}
        
        # 2. 原始cos vs 特异性cos
        raw_same_cat_cos = []
        raw_diff_cat_cos = []
        spec_same_cat_cos = []
        spec_diff_cat_cos = []
        
        concept_list = list(concept_vecs.keys())
        for i in range(len(concept_list)):
            for j in range(i+1, len(concept_list)):
                c1, c2 = concept_list[i], concept_list[j]
                raw_cos = proper_cos(concept_vecs[c1], concept_vecs[c2])
                spec_cos = proper_cos(specific_vecs[c1], specific_vecs[c2])
                
                if CONCEPT_CATEGORIES[c1] == CONCEPT_CATEGORIES[c2]:
                    raw_same_cat_cos.append(raw_cos)
                    spec_same_cat_cos.append(spec_cos)
                else:
                    raw_diff_cat_cos.append(raw_cos)
                    spec_diff_cat_cos.append(spec_cos)
        
        # 3. 概念特异性成分的PCA有效维度
        spec_mat = np.stack(list(specific_vecs.values()))
        U, S, Vt = np.linalg.svd(spec_mat, full_matrices=False)
        total_var = (S**2).sum()
        if total_var > 1e-10:
            cumvar = np.cumsum(S**2) / total_var
            eff_dim_90 = int(np.searchsorted(cumvar, 0.9) + 1)
            eff_dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        else:
            eff_dim_90 = 0
            eff_dim_95 = 0
        
        # 4. 模板信号 vs 概念信号的能量比
        template_energy = float(np.linalg.norm(mean_vec)**2)
        concept_energy = float(np.mean([n**2 for n in specific_norms.values()]))
        signal_ratio = concept_energy / max(template_energy, 1e-10)
        
        # 5. 类别均值向量的cos (类别方向是否一致)
        category_means = {}
        for cat in CONCEPTS:
            cat_vecs = [specific_vecs[c] for c in CONCEPTS[cat] if c in specific_vecs]
            if cat_vecs:
                category_means[cat] = np.mean(cat_vecs, axis=0)
        
        cat_cos_pairs = []
        cat_names = list(category_means.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                cos = proper_cos(category_means[cat_names[i]], category_means[cat_names[j]])
                cat_cos_pairs.append(cos)
        
        results[li] = {
            "template_energy": template_energy,
            "concept_energy_mean": concept_energy,
            "signal_ratio": signal_ratio,
            "specific_norms_mean": float(np.mean(list(specific_norms.values()))),
            "raw_same_cat_cos": float(np.mean(raw_same_cat_cos)) if raw_same_cat_cos else 0,
            "raw_diff_cat_cos": float(np.mean(raw_diff_cat_cos)) if raw_diff_cat_cos else 0,
            "raw_cos_gap": float(np.mean(raw_same_cat_cos) - np.mean(raw_diff_cat_cos)) if (raw_same_cat_cos and raw_diff_cat_cos) else 0,
            "spec_same_cat_cos": float(np.mean(spec_same_cat_cos)) if spec_same_cat_cos else 0,
            "spec_diff_cat_cos": float(np.mean(spec_diff_cat_cos)) if spec_diff_cat_cos else 0,
            "spec_cos_gap": float(np.mean(spec_same_cat_cos) - np.mean(spec_diff_cat_cos)) if (spec_same_cat_cos and spec_diff_cat_cos) else 0,
            "eff_dim_90": eff_dim_90,
            "eff_dim_95": eff_dim_95,
            "cat_mean_cos": float(np.mean(cat_cos_pairs)) if cat_cos_pairs else 0,
        }
    
    return results


def analysis_B_linear_probe(averaged_residuals, model_info):
    """
    B. 逐层线性探针
    
    用概念特异性成分训练简单的线性分类器,
    看哪一层最能区分类别。
    
    由于样本少(24个概念), 用leave-one-out交叉验证。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    n_layers = model_info.n_layers
    results = {}
    
    # 准备标签
    category_labels = [CONCEPT_CATEGORIES[c] for c in ALL_CONCEPTS]
    categories = list(CONCEPTS.keys())
    label_to_idx = {cat: i for i, cat in enumerate(categories)}
    y = np.array([label_to_idx[CONCEPT_CATEGORIES[c]] for c in ALL_CONCEPTS])
    
    for li in range(n_layers):
        # 收集该层所有概念的残差
        concept_vecs = []
        valid_indices = []
        for idx, concept in enumerate(ALL_CONCEPTS):
            if li in averaged_residuals.get(concept, {}):
                concept_vecs.append(averaged_residuals[concept][li])
                valid_indices.append(idx)
        
        if len(concept_vecs) < 10:
            continue
        
        X = np.stack(concept_vecs)
        y_valid = y[valid_indices]
        
        # 1. 原始残差的探针准确率
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Leave-one-out CV
        correct_raw = 0
        for i in range(len(X_scaled)):
            X_train = np.delete(X_scaled, i, axis=0)
            y_train = np.delete(y_valid, i)
            X_test = X_scaled[i:i+1]
            
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            if pred == y_valid[i]:
                correct_raw += 1
        
        acc_raw = correct_raw / len(X_scaled)
        
        # 2. 概念特异性成分的探针准确率
        mean_vec = X.mean(axis=0)
        X_specific = X - mean_vec
        X_spec_scaled = scaler.fit_transform(X_specific)
        
        correct_spec = 0
        for i in range(len(X_spec_scaled)):
            X_train = np.delete(X_spec_scaled, i, axis=0)
            y_train = np.delete(y_valid, i)
            X_test = X_spec_scaled[i:i+1]
            
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            if pred == y_valid[i]:
                correct_spec += 1
        
        acc_spec = correct_spec / len(X_spec_scaled)
        
        # 3. 随机基线 (打乱标签)
        rng = np.random.RandomState(42)
        n_perm = 100
        perm_accs = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y_valid)
            correct_perm = 0
            for i in range(len(X_spec_scaled)):
                X_train = np.delete(X_spec_scaled, i, axis=0)
                y_train = np.delete(y_perm, i)
                X_test = X_spec_scaled[i:i+1]
                
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)[0]
                if pred == y_perm[i]:
                    correct_perm += 1
            perm_accs.append(correct_perm / len(X_spec_scaled))
        
        random_acc = float(np.mean(perm_accs))
        random_std = float(np.std(perm_accs))
        
        results[li] = {
            "acc_raw": acc_raw,
            "acc_specific": acc_spec,
            "acc_random_mean": random_acc,
            "acc_random_std": random_std,
            "acc_above_random": acc_spec > random_acc + 2 * random_std,
            "n_samples": len(X_scaled),
        }
        
        if li % 5 == 0 or li == n_layers - 1:
            print(f"  L{li}: raw={acc_raw:.3f}, specific={acc_spec:.3f}, "
                  f"random={random_acc:.3f}±{random_std:.3f}")
    
    return results


def analysis_C_normalized_abstraction(averaged_residuals, model_info):
    """
    C. 归一化抽象链分析
    
    在归一化空间中分析抽象链:
    1. 将每个概念的残差归一化为单位向量
    2. 计算归一化后的方向差异
    3. 检查嵌套/正交模式是否在归一化空间中不同
    """
    n_layers = model_info.n_layers
    results = {}
    
    for chain_idx, chain in enumerate(ABSTRACTION_CHAINS):
        # 收集该链中所有概念的残差
        chain_vecs = {}
        for concept in chain:
            if concept in averaged_residuals:
                chain_vecs[concept] = averaged_residuals[concept]
        
        if len(chain_vecs) < 3:
            continue
        
        # 逐层分析
        layer_results = {}
        for li in range(n_layers):
            # 检查所有概念在该层都有数据
            level_vecs = {}
            for level, concept in enumerate(chain):
                if concept in chain_vecs and li in chain_vecs[concept]:
                    level_vecs[level] = chain_vecs[concept][li]
            
            if len(level_vecs) < 3:
                continue
            
            # 归一化
            normed_vecs = {}
            for level, vec in level_vecs.items():
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    normed_vecs[level] = vec / norm
            
            if len(normed_vecs) < 3:
                continue
            
            # 1. 归一化空间中的cos
            normed_cos_pairs = []
            levels = sorted(normed_vecs.keys())
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    cos = proper_cos(normed_vecs[levels[i]], normed_vecs[levels[j]])
                    normed_cos_pairs.append(cos)
            
            # 2. 归一化差异方向
            normed_directions = {}
            for i in range(len(levels) - 1):
                d = normed_vecs[levels[i+1]] - normed_vecs[levels[i]]
                norm = np.linalg.norm(d)
                if norm > 1e-10:
                    normed_directions[i] = d / norm
            
            # 3. 归一化差异方向的cos (嵌套检验)
            dir_keys = sorted(normed_directions.keys())
            dir_cos_pairs = {}
            for i in dir_keys:
                for j in dir_keys:
                    if i < j:
                        cos = proper_cos(normed_directions[i], normed_directions[j])
                        dir_cos_pairs[f"d{i}_d{j}"] = cos
            
            # 4. 抽象级别距离 (归一化空间)
            level_dists = {}
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    dist = float(np.linalg.norm(normed_vecs[levels[i]] - normed_vecs[levels[j]]))
                    level_dists[f"L{levels[i]}_L{levels[j]}"] = dist
            
            # 5. 角度累积: 具体→抽象是否沿一致的弧线?
            if len(normed_directions) >= 2:
                cumulative_angle = 0
                for i in sorted(normed_directions.keys())[:-1]:
                    cos = proper_cos(normed_directions[i], normed_directions[i+1])
                    angle = np.arccos(np.clip(cos, -1, 1))
                    cumulative_angle += angle
            else:
                cumulative_angle = 0
            
            layer_results[li] = {
                "normed_cos_mean": float(np.mean(normed_cos_pairs)) if normed_cos_pairs else 0,
                "normed_dir_cos_mean": float(np.mean(list(dir_cos_pairs.values()))) if dir_cos_pairs else 0,
                "normed_dir_cos_pairs": dir_cos_pairs,
                "cumulative_angle_rad": float(cumulative_angle),
                "level_dists": level_dists,
            }
        
        results[f"chain_{chain_idx}"] = {
            "chain": chain,
            "layer_results": layer_results,
        }
    
    # 全局统计
    all_dir_cos = []
    all_normed_cos = []
    for chain_key, chain_data in results.items():
        for li, lr in chain_data["layer_results"].items():
            all_dir_cos.append(lr["normed_dir_cos_mean"])
            all_normed_cos.append(lr["normed_cos_mean"])
    
    results["global"] = {
        "normed_dir_cos_mean": float(np.mean(all_dir_cos)) if all_dir_cos else 0,
        "normed_dir_cos_std": float(np.std(all_dir_cos)) if all_dir_cos else 0,
        "normed_cos_mean": float(np.mean(all_normed_cos)) if all_normed_cos else 0,
        "interpretation": "",
    }
    
    mean_cos = results["global"]["normed_dir_cos_mean"]
    if mean_cos > 0.3:
        results["global"]["interpretation"] = "归一化空间: 倾向嵌套(cos>0.3)"
    elif mean_cos < -0.3:
        results["global"]["interpretation"] = "归一化空间: 倾向正交/反向(cos<-0.3)"
    else:
        results["global"]["interpretation"] = f"归一化空间: 无明确嵌套模式(cos={mean_cos:.3f})"
    
    return results


def analysis_D_concept_information(averaged_residuals, model_info):
    """
    D. 概念编码的信息论度量
    
    计算概念在残差流中编码了多少"类别信息"和"概念特异性信息"
    """
    n_layers = model_info.n_layers
    results = {}
    
    for li in range(n_layers):
        # 收集该层所有概念的残差
        concept_vecs = {}
        for concept in ALL_CONCEPTS:
            if li in averaged_residuals.get(concept, {}):
                concept_vecs[concept] = averaged_residuals[concept][li]
        
        if len(concept_vecs) < 3:
            continue
        
        mat = np.stack(list(concept_vecs.values()))
        mean_vec = mat.mean(axis=0)
        
        # 总方差
        total_var = float(np.trace((mat - mean_vec) @ (mat - mean_vec).T) / len(mat))
        
        # 组内方差 (类别内)
        within_var = 0
        for cat in CONCEPTS:
            cat_vecs = [concept_vecs[c] for c in CONCEPTS[cat] if c in concept_vecs]
            if cat_vecs:
                cat_mean = np.mean(cat_vecs, axis=0)
                within_var += sum(np.linalg.norm(v - cat_mean)**2 for v in cat_vecs)
        within_var /= len(mat)
        
        # 组间方差 (类别间)
        between_var = total_var - within_var
        
        # 类别可分性: between / total
        separability = between_var / max(total_var, 1e-10)
        
        # 概念特异性: 每个概念偏离类别中心的程度
        concept_spec = {}
        for cat in CONCEPTS:
            cat_vecs = [concept_vecs[c] for c in CONCEPTS[cat] if c in concept_vecs]
            if cat_vecs:
                cat_mean = np.mean(cat_vecs, axis=0)
                for c, v in zip([c for c in CONCEPTS[cat] if c in concept_vecs], cat_vecs):
                    concept_spec[c] = float(np.linalg.norm(v - cat_mean))
        
        results[li] = {
            "total_var": total_var,
            "within_var": within_var,
            "between_var": between_var,
            "separability": separability,
            "concept_specificity_mean": float(np.mean(list(concept_spec.values()))) if concept_spec else 0,
            "concept_specificity_std": float(np.std(list(concept_spec.values()))) if concept_spec else 0,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase CCLIX Deep Analysis")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    print("="*60)
    print(f"Phase CCLIX Deep Analysis")
    print(f"Model: {args.model}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.name}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    # 收集残差
    print("\n  收集残差流...")
    averaged_residuals = collect_all_residuals(model, tokenizer, device, model_info)
    
    # A. 概念特异性分解
    print("\n  A. 概念特异性分解...")
    decomp_results = analysis_A_concept_specific_decomposition(averaged_residuals, model_info)
    
    # 找信号最强的层
    best_signal_layer = max(decomp_results.items(), key=lambda x: x[1]["spec_cos_gap"])
    best_ratio_layer = max(decomp_results.items(), key=lambda x: x[1]["signal_ratio"])
    
    print(f"\n  === A. 概念特异性分解 ===")
    print(f"  最佳类别区分层: L{best_signal_layer[0]} (spec_cos_gap={best_signal_layer[1]['spec_cos_gap']:.3f})")
    print(f"  最佳信号比层: L{best_ratio_layer[0]} (ratio={best_ratio_layer[1]['signal_ratio']:.4f})")
    
    # 检查: 去模板后cos_gap是否增大?
    raw_gaps = [v["raw_cos_gap"] for v in decomp_results.values()]
    spec_gaps = [v["spec_cos_gap"] for v in decomp_results.values()]
    print(f"  原始cos_gap: {np.mean(raw_gaps):.3f} → 特异性cos_gap: {np.mean(spec_gaps):.3f}")
    print(f"  放大倍数: {np.mean(spec_gaps)/max(np.mean(raw_gaps), 1e-10):.1f}x")
    
    # B. 逐层线性探针
    print("\n  B. 逐层线性探针 (类别分类)...")
    probe_results = analysis_B_linear_probe(averaged_residuals, model_info)
    
    best_probe_layer = max(probe_results.items(), key=lambda x: x[1]["acc_specific"])
    print(f"\n  === B. 线性探针 ===")
    print(f"  最佳分类层: L{best_probe_layer[0]} (acc={best_probe_layer[1]['acc_specific']:.3f})")
    print(f"  随机基线: {best_probe_layer[1]['acc_random_mean']:.3f}±{best_probe_layer[1]['acc_random_std']:.3f}")
    
    # 统计显著超过随机的层数
    sig_layers = sum(1 for v in probe_results.values() if v["acc_above_random"])
    print(f"  显著超过随机的层数: {sig_layers}/{len(probe_results)}")
    
    # C. 归一化抽象链分析
    print("\n  C. 归一化抽象链分析...")
    # 需要收集抽象链概念的残差
    chain_concepts = set()
    for chain in ABSTRACTION_CHAINS:
        chain_concepts.update(chain)
    
    # 收集抽象链概念的残差
    chain_residuals = {}
    for concept in chain_concepts:
        if concept in averaged_residuals:
            chain_residuals[concept] = averaged_residuals[concept]
        else:
            # 需要额外收集
            prompt = f"The {concept} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).detach().clone()
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            layer_outputs = collect_layer_outputs(
                model, inputs_embeds, position_ids, model_info.n_layers, target_type="layer"
            )
            
            chain_residuals[concept] = {}
            for li in range(model_info.n_layers):
                key = f"L{li}"
                if key in layer_outputs:
                    chain_residuals[concept][li] = layer_outputs[key][0, -1, :].numpy().astype(np.float32)
    
    normed_abstraction = analysis_C_normalized_abstraction(chain_residuals, model_info)
    print(f"\n  === C. 归一化抽象链 ===")
    print(f"  归一化方向cos: {normed_abstraction['global']['normed_dir_cos_mean']:.3f}")
    print(f"  解读: {normed_abstraction['global']['interpretation']}")
    
    # D. 信息论度量
    print("\n  D. 概念编码信息论度量...")
    info_results = analysis_D_concept_information(averaged_residuals, model_info)
    
    best_sep_layer = max(info_results.items(), key=lambda x: x[1]["separability"])
    print(f"\n  === D. 信息论度量 ===")
    print(f"  最佳可分性层: L{best_sep_layer[0]} (separability={best_sep_layer[1]['separability']:.4f})")
    print(f"  平均可分性: {np.mean([v['separability'] for v in info_results.values()]):.4f}")
    
    # 保存结果
    all_results = {
        "phase": "CCLIX_deep",
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
        },
        "A_decomposition": {
            str(li): data for li, data in decomp_results.items()
        },
        "B_probe": {
            str(li): data for li, data in probe_results.items()
        },
        "C_normalized_abstraction": normed_abstraction,
        "D_information": {
            str(li): data for li, data in info_results.items()
        },
        "summary": {
            "best_category_layer": best_signal_layer[0],
            "spec_cos_gap_at_best": best_signal_layer[1]["spec_cos_gap"],
            "raw_cos_gap_mean": float(np.mean(raw_gaps)),
            "spec_cos_gap_mean": float(np.mean(spec_gaps)),
            "gap_amplification": float(np.mean(spec_gaps) / max(np.mean(raw_gaps), 1e-10)),
            "best_probe_layer": best_probe_layer[0],
            "best_probe_acc": best_probe_layer[1]["acc_specific"],
            "random_baseline_acc": best_probe_layer[1]["acc_random_mean"],
            "significant_probe_layers": sig_layers,
            "normed_abstraction_dir_cos": normed_abstraction["global"]["normed_dir_cos_mean"],
            "best_separability_layer": best_sep_layer[0],
            "best_separability": best_sep_layer[1]["separability"],
        },
    }
    
    out_path = OUTPUT_DIR / f"{args.model}_cclix" / "deep_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {out_path}")
    
    release_model(model)
    
    # 打印最终总结
    print("\n" + "="*60)
    print("Phase CCLIX 深度分析总结")
    print("="*60)
    s = all_results["summary"]
    print(f"  KN-1a: 概念编码位置")
    print(f"    - 去模板后类别cos_gap: {s['spec_cos_gap_mean']:.3f} (原始: {s['raw_cos_gap_mean']:.3f}, 放大{s['gap_amplification']:.1f}x)")
    print(f"    - 最佳类别区分层: L{s['best_category_layer']}")
    print(f"  KN-1b: 维度共享")
    print(f"    - 线性探针最佳层: L{s['best_probe_layer']} (acc={s['best_probe_acc']:.3f}, random={s['random_baseline_acc']:.3f})")
    print(f"    - 显著层数: {s['significant_probe_layers']}")
    print(f"  KN-3a: 抽象层级编码")
    print(f"    - 归一化方向cos: {s['normed_abstraction_dir_cos']:.3f}")
    print(f"  KN-3b: 嵌套vs正交")
    print(f"    - 可分性最佳层: L{s['best_separability_layer']} (sep={s['best_separability']:.4f})")


if __name__ == "__main__":
    main()
