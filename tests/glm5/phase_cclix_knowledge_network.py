"""
Phase CCLIX: Knowledge Network Mapping (知识网络映射)
=====================================================
目标格子: KN-1a★, KN-1b, KN-3a★, KN-3b (次要: WE-1c)

范式转换: 从"测统计量"→"选语言任务→追踪概念如何被编码"

实验设计:
  Exp1: 概念编码位置 (KN-1a)
    - 20+概念, 逐层残差流收集
    - PCA分析每个概念的有效维度
    - 与随机基线对比
    - 哪些层编码什么概念?
  
  Exp2: 概念间维度共享 (KN-1b)
    - 概念对之间的编码维度重叠度
    - 同类/异类概念对比
    - 层级特异性的共享模式
  
  Exp3: 抽象层级编码 (KN-3a)
    - 抽象链: 苹果→水果→食物→物体
    - 逐层追踪抽象级别的编码差异
    - 层级与抽象程度的对应关系
  
  Exp4: 抽象层级嵌套vs正交 (KN-3b)
    - 子集关系的编码方式
    - 高层概念方向是否能泛化低层概念?

铁律遵守:
  - 随机基线对比
  - 跨模型验证 (Qwen3 + GLM4 + DS7B)
  - eps扫描在线性区验证
  - 统计vs因果分离

用法:
  python phase_cclix_knowledge_network.py --model qwen3 --exp 1
  python phase_cclix_knowledge_network.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    collect_layer_outputs,
)

OUTPUT_DIR = Path("results/causal_fiber")
TEST_DIR = Path("tests/glm5_temp")


def proper_cos(v1, v2):
    """正确的余弦相似度: dot(v1,v2) / (||v1|| * ||v2||)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ===== 概念体系定义 =====

# 20+具体概念, 分4个类别
CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano"],
}

# 展平为列表
ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
for cat, items in CONCEPTS.items():
    for item in items:
        ALL_CONCEPTS.append(item)
        CONCEPT_CATEGORIES[item] = cat

# 抽象链 (KN-3a): 具体→抽象
ABSTRACTION_CHAINS = [
    # 链1: 苹果→水果→食物→物质→物体
    ["apple", "fruit", "food", "substance", "object"],
    # 链2: 狗→动物→生物→有机体→实体
    ["dog", "animal", "organism", "being", "entity"],
    # 链3: 锤子→工具→器具→人造物→物体
    ["hammer", "tool", "instrument", "artifact", "object"],
    # 链4: 山→地形→地理→自然→物体
    ["mountain", "terrain", "geography", "nature", "object"],
    # 链5: 米→谷物→食物→物质→物体
    ["rice", "grain", "food", "substance", "object"],
]

# 同义/近义概念对 (用于验证编码特异性)
SYNONYM_PAIRS = [
    ("dog", "puppy"),    # 幼体
    ("knife", "blade"),  # 近义
    ("ocean", "sea"),    # 近义
    ("mountain", "hill"), # 近义但不同量级
]

# 上下文模板 — 让模型"思考"概念
CONTEXT_TEMPLATES = [
    "The {concept} is",              # 基础
    "I think about the {concept}",   # 主动思考
    "The {concept} has",             # 属性思考
]


def collect_concept_residuals(model, tokenizer, device, model_info, 
                              concepts, template="The {concept} is"):
    """
    收集每个概念在每层的残差流表示
    
    Returns:
        dict: {concept: {layer_idx: residual_vector}}
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    results = {}
    
    for concept in concepts:
        prompt = template.format(concept=concept)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        
        # 收集每层输出
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone()
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        layer_outputs = collect_layer_outputs(
            model, inputs_embeds, position_ids, n_layers, target_type="layer"
        )
        
        # 取最后一个token的残差流
        concept_residuals = {}
        for li in range(n_layers):
            key = f"L{li}"
            if key in layer_outputs:
                # last token position
                vec = layer_outputs[key][0, -1, :].numpy().astype(np.float32)
                concept_residuals[li] = vec
        
        results[concept] = concept_residuals
        print(f"  [{concept}] collected {len(concept_residuals)} layers")
    
    return results


def compute_effective_dim(vectors, threshold=0.9):
    """
    PCA分析一组向量的有效维度
    
    Args:
        vectors: list of numpy arrays [d_model] or [n_samples, d_model]
        threshold: 累计方差解释比阈值
    
    Returns:
        dict: {eff_dim, top_variance_ratio, singular_values}
    """
    if len(vectors.shape) == 1:
        return {"eff_dim": 0, "top_variance_ratio": 0, "singular_values": []}
    
    # 中心化
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    
    # SVD
    if centered.shape[0] < centered.shape[1]:
        # 样本数 < 维度数: 用X@X^T的特征值分解更高效
        gram = centered @ centered.T  # [n, n]
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)  # 数值稳定
    else:
        _, eigenvalues, _ = np.linalg.svd(centered, full_matrices=False)
    
    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return {"eff_dim": 0, "top_variance_ratio": 0, "singular_values": eigenvalues.tolist()[:20]}
    
    cumvar = np.cumsum(eigenvalues) / total_var
    eff_dim = int(np.searchsorted(cumvar, threshold) + 1)
    top_variance_ratio = float(cumvar[min(9, len(cumvar)-1)])  # top-10方差
    
    return {
        "eff_dim": eff_dim,
        "top_variance_ratio": top_variance_ratio,
        "singular_values": eigenvalues.tolist()[:20],
    }


def compute_concept_pca_per_layer(residuals_dict, concepts, model_info):
    """
    对每层, 跨概念的残差流做PCA
    
    KN-1a核心问题: 一个概念在残差流中编码在哪些维度?
    
    方法: 在每层, 对所有概念的残差做PCA, 看每个概念在主成分上的投影
    
    Returns:
        dict: layer_idx -> {概念在PC上的投影, 有效维度等}
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layer_analysis = {}
    
    for li in range(n_layers):
        # 收集该层所有概念的残差
        layer_vecs = []
        valid_concepts = []
        for concept in concepts:
            if li in residuals_dict.get(concept, {}):
                layer_vecs.append(residuals_dict[concept][li])
                valid_concepts.append(concept)
        
        if len(layer_vecs) < 3:
            continue
        
        mat = np.stack(layer_vecs)  # [n_concepts, d_model]
        
        # 中心化
        mean_vec = mat.mean(axis=0)
        centered = mat - mean_vec
        
        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # U: [n_concepts, n_concepts], S: [min], Vt: [min, d_model]
        
        # 每个概念在主成分上的投影系数
        projections = U * S  # [n_concepts, n_components]
        
        # 方差解释
        total_var = (S**2).sum()
        if total_var < 1e-10:
            continue
        
        explained_ratio = (S**2) / total_var
        cum_explained = np.cumsum(explained_ratio)
        
        # 找有效维度数(90%方差)
        eff_dim = int(np.searchsorted(cum_explained, 0.9) + 1)
        
        # 每个概念的主导PC
        concept_top_pcs = {}
        for i, concept in enumerate(valid_concepts):
            abs_proj = np.abs(projections[i])
            top_pc = int(np.argmax(abs_proj))
            concept_top_pcs[concept] = {
                "top_pc": top_pc,
                "top_proj": float(abs_proj[top_pc]),
                "top5_pcs": [int(x) for x in np.argsort(abs_proj)[-5:][::-1]],
            }
        
        layer_analysis[li] = {
            "n_concepts": len(valid_concepts),
            "eff_dim_90": eff_dim,
            "explained_ratio_top5": explained_ratio[:5].tolist(),
            "cum_explained_top10": cum_explained[:10].tolist(),
            "concept_top_pcs": concept_top_pcs,
            "principal_directions_norm": [float(np.linalg.norm(Vt[j])) for j in range(min(5, Vt.shape[0]))],
        }
    
    return layer_analysis


def compute_dimension_overlap(residuals_dict, concepts, model_info, top_k=20):
    """
    KN-1b: 计算概念对之间的编码维度重叠度
    
    方法: 对每个概念, 找到其"编码方向"(残差流跨模板的主成分),
    然后计算不同概念编码方向之间的重叠
    
    Returns:
        dict: overlap statistics
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 对每层, 计算概念对之间的余弦相似度
    same_category_cos = []
    diff_category_cos = []
    
    layer_overlap = {}
    
    for li in range(n_layers):
        layer_vecs = {}
        for concept in concepts:
            if li in residuals_dict.get(concept, {}):
                layer_vecs[concept] = residuals_dict[concept][li]
        
        if len(layer_vecs) < 3:
            continue
        
        # 概念对之间的cos
        concept_names = list(layer_vecs.keys())
        pair_cos = {}
        
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                c1, c2 = concept_names[i], concept_names[j]
                v1, v2 = layer_vecs[c1], layer_vecs[c2]
                cos = proper_cos(v1, v2)
                pair_cos[f"{c1}-{c2}"] = cos
                
                cat1 = CONCEPT_CATEGORIES.get(c1, "unknown")
                cat2 = CONCEPT_CATEGORIES.get(c2, "unknown")
                if cat1 == cat2:
                    same_category_cos.append(cos)
                else:
                    diff_category_cos.append(cos)
        
        # 同类概念之间的残差差分 → 概念特异性方向
        # 用差分向量找概念特异性维度
        concept_specific_dirs = {}
        for concept in concept_names:
            # 该概念 vs 同类其他概念的平均
            cat = CONCEPT_CATEGORIES.get(concept, "unknown")
            same_cat = [c for c in concept_names if c != concept and CONCEPT_CATEGORIES.get(c) == cat]
            if same_cat:
                other_mean = np.mean([layer_vecs[c] for c in same_cat], axis=0)
                diff = layer_vecs[concept] - other_mean
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    concept_specific_dirs[concept] = diff / norm
        
        # 概念特异性方向之间的cos
        specific_cos = {}
        dir_names = list(concept_specific_dirs.keys())
        for i in range(len(dir_names)):
            for j in range(i+1, len(dir_names)):
                c1, c2 = dir_names[i], dir_names[j]
                cos = abs(proper_cos(concept_specific_dirs[c1], concept_specific_dirs[c2]))
                specific_cos[f"{c1}-{c2}"] = cos
        
        # 维度重叠: 两个概念特异性方向的top-k维度重叠
        dim_overlaps = []
        for i in range(len(dir_names)):
            for j in range(i+1, len(dir_names)):
                c1, c2 = dir_names[i], dir_names[j]
                d1, d2 = concept_specific_dirs[c1], concept_specific_dirs[c2]
                top_dims_1 = set(np.argsort(np.abs(d1))[-top_k:])
                top_dims_2 = set(np.argsort(np.abs(d2))[-top_k:])
                overlap = len(top_dims_1 & top_dims_2) / top_k
                dim_overlaps.append(overlap)
        
        layer_overlap[li] = {
            "pair_cos_mean": float(np.mean(list(pair_cos.values()))) if pair_cos else 0,
            "same_cat_cos_mean": float(np.mean(same_category_cos[-len(concept_names):])) if same_category_cos else 0,
            "diff_cat_cos_mean": float(np.mean(diff_category_cos[-len(concept_names):])) if diff_category_cos else 0,
            "specific_cos_mean": float(np.mean(list(specific_cos.values()))) if specific_cos else 0,
            "dim_overlap_mean": float(np.mean(dim_overlaps)) if dim_overlaps else 0,
            "dim_overlap_std": float(np.std(dim_overlaps)) if dim_overlaps else 0,
            "n_specific_dirs": len(concept_specific_dirs),
        }
    
    # 全局统计
    global_stats = {
        "same_category_cos_mean": float(np.mean(same_category_cos)) if same_category_cos else 0,
        "diff_category_cos_mean": float(np.mean(diff_category_cos)) if diff_category_cos else 0,
        "cos_gap": float(np.mean(same_category_cos) - np.mean(diff_category_cos)) if (same_category_cos and diff_category_cos) else 0,
    }
    
    return {"global": global_stats, "per_layer": layer_overlap}


def analyze_abstraction_chains(model, tokenizer, device, model_info, chains=None):
    """
    KN-3a: 抽象链编码分析
    
    分析: apple→fruit→food→substance→object
    在残差流中如何表示?
    
    方法:
    1. 收集每层残差
    2. 计算相邻抽象级别之间的差异方向
    3. 分析差异方向的维度和大小
    4. 检查高层概念方向是否能泛化低层概念
    """
    if chains is None:
        chains = ABSTRACTION_CHAINS
    
    n_layers = model_info.n_layers
    results = {}
    
    for chain_idx, chain in enumerate(chains):
        print(f"  Chain {chain_idx}: {' → '.join(chain)}")
        
        # 收集每个抽象级别的残差
        chain_residuals = {}
        for level, concept in enumerate(chain):
            prompt = f"The {concept} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).detach().clone()
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            layer_outputs = collect_layer_outputs(
                model, inputs_embeds, position_ids, n_layers, target_type="layer"
            )
            
            level_residuals = {}
            for li in range(n_layers):
                key = f"L{li}"
                if key in layer_outputs:
                    level_residuals[li] = layer_outputs[key][0, -1, :].numpy().astype(np.float32)
            
            chain_residuals[level] = level_residuals
        
        # 分析相邻级别的差异方向
        level_deltas = {}  # level -> {layer -> delta_vector}
        for level in range(len(chain) - 1):
            deltas = {}
            for li in range(n_layers):
                if li in chain_residuals[level] and li in chain_residuals[level+1]:
                    delta = chain_residuals[level+1][li] - chain_residuals[level][li]
                    deltas[li] = delta
            level_deltas[level] = deltas
        
        # 分析1: 差异方向的大小随层变化
        delta_norms = {}
        for level in range(len(chain) - 1):
            norms = {}
            for li, delta in level_deltas[level].items():
                norms[li] = float(np.linalg.norm(delta))
            delta_norms[f"level{level}_to_level{level+1}"] = norms
        
        # 分析2: 差异方向之间的cos (不同抽象跳跃是否沿相同方向?)
        delta_cos_matrix = {}
        for li in range(n_layers):
            deltas_at_layer = []
            for level in range(len(chain) - 1):
                if li in level_deltas[level]:
                    deltas_at_layer.append(level_deltas[level][li])
            
            if len(deltas_at_layer) >= 2:
                cos_matrix = np.zeros((len(deltas_at_layer), len(deltas_at_layer)))
                for i in range(len(deltas_at_layer)):
                    for j in range(len(deltas_at_layer)):
                        cos_matrix[i, j] = proper_cos(deltas_at_layer[i], deltas_at_layer[j])
                delta_cos_matrix[li] = cos_matrix.tolist()
        
        # 分析3: 概念向量在残差流中的分离度
        # 不同抽象级别的概念在同一层有多远?
        level_distances = {}
        for li in range(n_layers):
            dists = {}
            level_vecs = {}
            for level in range(len(chain)):
                if li in chain_residuals[level]:
                    level_vecs[level] = chain_residuals[level][li]
            
            for i in range(len(chain)):
                for j in range(i+1, len(chain)):
                    if i in level_vecs and j in level_vecs:
                        d = float(np.linalg.norm(level_vecs[i] - level_vecs[j]))
                        dists[f"L{i}_L{j}"] = d
            if dists:
                level_distances[li] = dists
        
        # 分析4: 抽象层级与层的对应关系
        # 哪些层最能区分不同抽象级别?
        layer_discrimination = {}
        for li in range(n_layers):
            level_vecs = {}
            for level in range(len(chain)):
                if li in chain_residuals[level]:
                    level_vecs[level] = chain_residuals[level][li]
            
            if len(level_vecs) >= 3:
                mat = np.stack(list(level_vecs.values()))
                mean = mat.mean(axis=0)
                centered = mat - mean
                total_var = np.trace(centered @ centered.T)
                between_var = 0
                for v in level_vecs.values():
                    between_var += np.linalg.norm(v - mean)**2
                # 简单指标: 不同级别之间的方差
                layer_discrimination[li] = float(between_var / max(total_var, 1e-10))
        
        results[f"chain_{chain_idx}"] = {
            "chain": chain,
            "delta_norms": delta_norms,
            "delta_cos_sample_layers": {str(k): v for k, v in list(delta_cos_matrix.items())[::5]},
            "level_distances_sample": {str(k): v for k, v in list(level_distances.items())[::5]},
            "layer_discrimination": layer_discrimination,
        }
    
    return results


def analyze_abstraction_nesting(residuals_dict, model_info):
    """
    KN-3b: 抽象层级是嵌套的(苹果⊂水果⊂食物)还是正交的?
    
    方法:
    1. 对于抽象链 apple→fruit→food→substance→object
    2. 计算 apple→fruit 的方向 d1, fruit→food 的方向 d2
    3. 如果嵌套: d1 和 d2 应该同向(cos>0)
    4. 如果正交: d1 和 d2 正交(cos≈0)
    5. 如果独立编码: d1 和 d2 无特定关系
    """
    n_layers = model_info.n_layers
    results = {}
    
    for chain_idx, chain in enumerate(ABSTRACTION_CHAINS):
        # 检查是否所有概念都有残差
        all_present = all(c in residuals_dict for c in chain)
        if not all_present:
            # 用CCLIX Exp1收集的数据
            # 但抽象链的某些概念可能不在CONCEPTS列表中
            # 这里只分析存在的部分
            present_chain = [c for c in chain if c in residuals_dict]
            if len(present_chain) < 3:
                continue
            chain = present_chain
        
        nesting_analysis = {}
        
        for li in range(n_layers):
            # 检查所有级别在该层都有残差
            level_vecs = {}
            for level, concept in enumerate(chain):
                if concept in residuals_dict and li in residuals_dict[concept]:
                    level_vecs[level] = residuals_dict[concept][li]
            
            if len(level_vecs) < 3:
                continue
            
            # 计算相邻级别的方向
            directions = {}
            for level in range(len(chain) - 1):
                if level in level_vecs and level+1 in level_vecs:
                    d = level_vecs[level+1] - level_vecs[level]
                    norm = np.linalg.norm(d)
                    if norm > 1e-10:
                        directions[level] = d / norm
            
            # 方向之间的cos
            dir_keys = sorted(directions.keys())
            cos_matrix = {}
            for i in dir_keys:
                for j in dir_keys:
                    if i < j:
                        cos = proper_cos(directions[i], directions[j])
                        cos_matrix[f"d{i}_vs_d{j}"] = cos
            
            # 投影检验: d1投影到d2上的比例
            projections = {}
            for i in dir_keys:
                for j in dir_keys:
                    if i != j and i in directions and j in directions:
                        proj = float(np.dot(directions[i], directions[j]))
                        projections[f"d{i}_proj_d{j}"] = proj
            
            nesting_analysis[li] = {
                "direction_cos": cos_matrix,
                "direction_projections": projections,
                "n_directions": len(directions),
            }
        
        results[f"chain_{chain_idx}"] = {
            "chain": chain,
            "nesting_analysis": nesting_analysis,
        }
    
    # 全局统计: 平均cos值
    all_cos = []
    for chain_key, chain_data in results.items():
        for li, analysis in chain_data["nesting_analysis"].items():
            for pair, cos in analysis["direction_cos"].items():
                all_cos.append(cos)
    
    global_nesting = {
        "mean_direction_cos": float(np.mean(all_cos)) if all_cos else 0,
        "std_direction_cos": float(np.std(all_cos)) if all_cos else 0,
        "n_positive_cos": int(sum(1 for c in all_cos if c > 0)),
        "n_negative_cos": int(sum(1 for c in all_cos if c < 0)),
        "n_direction_pairs": len(all_cos),
        "interpretation": "",
    }
    
    # 解读
    if global_nesting["mean_direction_cos"] > 0.3:
        global_nesting["interpretation"] = "倾向嵌套: 相邻抽象跳跃沿相似方向(cos>0.3)"
    elif global_nesting["mean_direction_cos"] < -0.3:
        global_nesting["interpretation"] = "倾向正交/反向: 相邻跳跃方向不一致(cos<-0.3)"
    else:
        global_nesting["interpretation"] = "无明确模式: cos≈0, 可能是独立编码"
    
    results["global_nesting"] = global_nesting
    return results


def random_baseline_analysis(model_info, n_concepts=24, n_templates=3, seed=42):
    """
    随机基线: 用随机向量模拟概念残差流
    
    目的: 验证我们在真实数据上发现的结构不是高维随机噪声的假象
    """
    rng = np.random.RandomState(seed)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 模拟: 每个概念每层一个随机向量
    fake_residuals = {}
    for i in range(n_concepts):
        concept = f"random_{i}"
        fake_residuals[concept] = {}
        for li in range(n_layers):
            fake_residuals[concept][li] = rng.randn(d_model).astype(np.float32)
    
    # 对fake数据做同样的分析
    # PCA有效维度
    layer_eff_dims = []
    for li in range(n_layers):
        mat = np.stack([fake_residuals[f"random_{i}"][li] for i in range(n_concepts)])
        mean = mat.mean(axis=0)
        centered = mat - mean
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        total = (S**2).sum()
        if total > 1e-10:
            cumvar = np.cumsum(S**2) / total
            eff_dim = int(np.searchsorted(cumvar, 0.9) + 1)
            layer_eff_dims.append(eff_dim)
    
    # 概念对cos
    pair_cos = []
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            cos = proper_cos(
                fake_residuals[f"random_{i}"][0],
                fake_residuals[f"random_{j}"][0]
            )
            pair_cos.append(cos)
    
    # 维度重叠
    dim_overlaps = []
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            v1 = fake_residuals[f"random_{i}"][0]
            v2 = fake_residuals[f"random_{j}"][0]
            top1 = set(np.argsort(np.abs(v1))[-20:])
            top2 = set(np.argsort(np.abs(v2))[-20:])
            overlap = len(top1 & top2) / 20
            dim_overlaps.append(overlap)
    
    return {
        "eff_dim_mean": float(np.mean(layer_eff_dims)) if layer_eff_dims else 0,
        "eff_dim_std": float(np.std(layer_eff_dims)) if layer_eff_dims else 0,
        "pair_cos_mean": float(np.mean(pair_cos)),
        "pair_cos_std": float(np.std(pair_cos)),
        "dim_overlap_mean": float(np.mean(dim_overlaps)),
        "dim_overlap_expected": 20.0 / model_info.d_model,  # 随机期望
    }


def run_exp1_concept_encoding(model_name):
    """
    Exp1: 概念编码位置 (KN-1a★)
    
    核心问题: 一个具体概念在残差流中编码在哪些维度?
    """
    print("\n" + "="*60)
    print("Exp1: 概念编码位置 (KN-1a★)")
    print("="*60)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.name}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    # 1. 收集所有概念的残差流 (使用多个模板增加鲁棒性)
    all_residuals = {}  # {concept: {template_idx: {layer: vec}}}
    
    for t_idx, template in enumerate(CONTEXT_TEMPLATES):
        print(f"\n  Template {t_idx}: {template}")
        residuals = collect_concept_residuals(
            model, tokenizer, device, model_info,
            ALL_CONCEPTS, template=template
        )
        for concept, layer_vecs in residuals.items():
            if concept not in all_residuals:
                all_residuals[concept] = {}
            all_residuals[concept][t_idx] = layer_vecs
    
    # 2. 跨模板平均残差 (更鲁棒的概念表示)
    averaged_residuals = {}
    for concept in ALL_CONCEPTS:
        averaged_residuals[concept] = {}
        for li in range(model_info.n_layers):
            vecs = []
            for t_idx in range(len(CONTEXT_TEMPLATES)):
                if t_idx in all_residuals.get(concept, {}) and li in all_residuals[concept][t_idx]:
                    vecs.append(all_residuals[concept][t_idx][li])
            if vecs:
                averaged_residuals[concept][li] = np.mean(vecs, axis=0)
    
    # 3. 逐层PCA分析
    print("\n  逐层PCA分析...")
    layer_pca = compute_concept_pca_per_layer(averaged_residuals, ALL_CONCEPTS, model_info)
    
    # 4. 每个概念的逐层范数和特异性
    concept_norms = {}
    concept_specificity = {}  # 概念偏离类别中心的程度
    
    for concept in ALL_CONCEPTS:
        norms = {}
        for li in range(model_info.n_layers):
            if li in averaged_residuals.get(concept, {}):
                norms[li] = float(np.linalg.norm(averaged_residuals[concept][li]))
        concept_norms[concept] = norms
        
        # 类别特异性: 该概念偏离同类概念平均的程度
        cat = CONCEPT_CATEGORIES[concept]
        same_cat = [c for c in ALL_CONCEPTS if c != concept and CONCEPT_CATEGORIES[c] == cat]
        specificity = {}
        for li in range(model_info.n_layers):
            if li in averaged_residuals.get(concept, {}):
                if same_cat:
                    same_cat_mean = np.mean(
                        [averaged_residuals[c][li] for c in same_cat if li in averaged_residuals.get(c, {})],
                        axis=0
                    )
                    spec = float(np.linalg.norm(averaged_residuals[concept][li] - same_cat_mean))
                    specificity[li] = spec
        concept_specificity[concept] = specificity
    
    # 5. 随机基线
    print("\n  随机基线...")
    random_results = random_baseline_analysis(model_info)
    
    # 6. 汇总
    exp1_results = {
        "model": model_name,
        "model_info": {
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "model_class": model_info.model_class,
        },
        "n_concepts": len(ALL_CONCEPTS),
        "n_templates": len(CONTEXT_TEMPLATES),
        "layer_pca_summary": {
            str(li): {
                "eff_dim_90": data["eff_dim_90"],
                "explained_top5": data["explained_ratio_top5"],
            }
            for li, data in layer_pca.items()
        },
        "concept_norms_sample": {
            concept: {str(li): norm for li, norm in list(norms.items())[::5]}
            for concept, norms in list(concept_norms.items())[::3]
        },
        "concept_specificity_sample": {
            concept: {str(li): spec for li, spec in list(specs.items())[::5]}
            for concept, specs in list(concept_specificity.items())[::3]
        },
        "random_baseline": random_results,
        "key_findings": "",
    }
    
    # 关键发现摘要
    eff_dims = [data["eff_dim_90"] for data in layer_pca.values()]
    mean_eff_dim = float(np.mean(eff_dims)) if eff_dims else 0
    
    # 找概念编码最集中的层
    min_eff_dim_layer = min(layer_pca.items(), key=lambda x: x[1]["eff_dim_90"]) if layer_pca else (0, {"eff_dim_90": 0})
    
    findings = [
        f"跨层平均有效维度(PCA 90%): {mean_eff_dim:.1f} (随机基线: {random_results['eff_dim_mean']:.1f})",
        f"概念编码最集中的层: L{min_eff_dim_layer[0]} (eff_dim={min_eff_dim_layer[1]['eff_dim_90']})",
        f"概念对平均cos(随机): {random_results['pair_cos_mean']:.3f}",
    ]
    
    # 检查同类vs异类概念的cos差异
    same_cat_cos = []
    diff_cat_cos = []
    for li in range(min(10, model_info.n_layers)):
        for i, c1 in enumerate(ALL_CONCEPTS):
            for j, c2 in enumerate(ALL_CONCEPTS):
                if i < j and li in averaged_residuals.get(c1, {}) and li in averaged_residuals.get(c2, {}):
                    cos = proper_cos(averaged_residuals[c1][li], averaged_residuals[c2][li])
                    if CONCEPT_CATEGORIES[c1] == CONCEPT_CATEGORIES[c2]:
                        same_cat_cos.append(cos)
                    else:
                        diff_cat_cos.append(cos)
    
    if same_cat_cos and diff_cat_cos:
        findings.append(
            f"同类概念cos: {np.mean(same_cat_cos):.3f}, 异类概念cos: {np.mean(diff_cat_cos):.3f}, "
            f"差距: {np.mean(same_cat_cos)-np.mean(diff_cat_cos):.3f}"
        )
    
    exp1_results["key_findings"] = "\n".join(findings)
    print("\n  === Exp1 关键发现 ===")
    for f in findings:
        print(f"  - {f}")
    
    release_model(model)
    return exp1_results, averaged_residuals


def run_exp2_dimension_sharing(model_name, averaged_residuals=None):
    """
    Exp2: 概念间维度共享 (KN-1b)
    
    核心问题: 不同概念是否共享编码维度?
    """
    print("\n" + "="*60)
    print("Exp2: 概念间维度共享 (KN-1b)")
    print("="*60)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 如果没有预计算的残差, 重新收集
    if averaged_residuals is None:
        averaged_residuals = {}
        for template in CONTEXT_TEMPLATES:
            residuals = collect_concept_residuals(
                model, tokenizer, device, model_info,
                ALL_CONCEPTS, template=template
            )
            for concept, layer_vecs in residuals.items():
                for li, vec in layer_vecs.items():
                    if concept not in averaged_residuals:
                        averaged_residuals[concept] = {}
                    if li not in averaged_residuals[concept]:
                        averaged_residuals[concept][li] = []
                    averaged_residuals[concept][li].append(vec)
        
        # 平均
        for concept in averaged_residuals:
            for li in averaged_residuals[concept]:
                averaged_residuals[concept][li] = np.mean(averaged_residuals[concept][li], axis=0)
    
    # 维度共享分析
    overlap_results = compute_dimension_overlap(averaged_residuals, ALL_CONCEPTS, model_info)
    
    # 同义/近义对分析
    synonym_analysis = {}
    for w1, w2 in SYNONYM_PAIRS:
        pair_result = {}
        for li in range(model_info.n_layers):
            if li in averaged_residuals.get(w1, {}) and li in averaged_residuals.get(w2, {}):
                cos = proper_cos(averaged_residuals[w1][li], averaged_residuals[w2][li])
                pair_result[li] = cos
        synonym_analysis[f"{w1}-{w2}"] = {
            "mean_cos": float(np.mean(list(pair_result.values()))) if pair_result else 0,
            "layer_cos_sample": {str(li): cos for li, cos in list(pair_result.items())[::5]},
        }
    
    exp2_results = {
        "model": model_name,
        "global_stats": overlap_results["global"],
        "per_layer_summary": {
            str(li): data for li, data in list(overlap_results["per_layer"].items())[::5]
        },
        "synonym_analysis": synonym_analysis,
        "key_findings": "",
    }
    
    findings = [
        f"同类概念cos: {overlap_results['global']['same_category_cos_mean']:.3f}",
        f"异类概念cos: {overlap_results['global']['diff_category_cos_mean']:.3f}",
        f"cos差距(同类-异类): {overlap_results['global']['cos_gap']:.3f}",
    ]
    
    for pair, data in synonym_analysis.items():
        findings.append(f"近义对 {pair} cos: {data['mean_cos']:.3f}")
    
    exp2_results["key_findings"] = "\n".join(findings)
    print("\n  === Exp2 关键发现 ===")
    for f in findings:
        print(f"  - {f}")
    
    release_model(model)
    return exp2_results


def run_exp3_abstraction_chains(model_name):
    """
    Exp3: 抽象层级编码 (KN-3a★)
    
    核心问题: 抽象链 apple→fruit→food→substance→object 在参数中如何表示?
    """
    print("\n" + "="*60)
    print("Exp3: 抽象层级编码 (KN-3a★)")
    print("="*60)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 分析抽象链
    chain_results = analyze_abstraction_chains(model, tokenizer, device, model_info)
    
    # 汇总关键发现
    findings = []
    
    # 1. 差异方向大小随层变化
    for chain_key, chain_data in chain_results.items():
        if chain_key.startswith("chain_"):
            chain = chain_data["chain"]
            delta_norms = chain_data["delta_norms"]
            
            # 找差异最大的层
            for level_key, norms in delta_norms.items():
                if norms:
                    max_layer = max(norms.items(), key=lambda x: x[1])
                    findings.append(
                        f"{chain_key} {level_key}: 最大差异层L{max_layer[0]}, norm={max_layer[1]:.2f}"
                    )
    
    # 2. 层区分能力
    for chain_key, chain_data in chain_results.items():
        if chain_key.startswith("chain_"):
            disc = chain_data["layer_discrimination"]
            if disc:
                best_layer = max(disc.items(), key=lambda x: x[1])
                findings.append(
                    f"{chain_key}: 最能区分抽象级别的层L{best_layer[0]}, score={best_layer[1]:.4f}"
                )
    
    exp3_results = {
        "model": model_name,
        "n_chains": len(ABSTRACTION_CHAINS),
        "chain_results": chain_results,
        "key_findings": "\n".join(findings),
    }
    
    print("\n  === Exp3 关键发现 ===")
    for f in findings:
        print(f"  - {f}")
    
    release_model(model)
    return exp3_results, chain_results


def run_exp4_abstraction_nesting(model_name):
    """
    Exp4: 抽象层级嵌套vs正交 (KN-3b)
    
    核心问题: 抽象层级是嵌套的(苹果⊂水果)还是正交的?
    """
    print("\n" + "="*60)
    print("Exp4: 抽象层级嵌套vs正交 (KN-3b)")
    print("="*60)
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 收集抽象链中所有概念的残差
    chain_concepts = set()
    for chain in ABSTRACTION_CHAINS:
        chain_concepts.update(chain)
    
    print(f"  收集 {len(chain_concepts)} 个抽象链概念的残差...")
    chain_residuals = collect_concept_residuals(
        model, tokenizer, device, model_info,
        list(chain_concepts), template="The {concept} is"
    )
    
    # 嵌套分析
    nesting_results = analyze_abstraction_nesting(chain_residuals, model_info)
    
    exp4_results = {
        "model": model_name,
        "n_chains": len(ABSTRACTION_CHAINS),
        "nesting_results": nesting_results,
        "key_findings": "",
    }
    
    findings = []
    global_data = nesting_results.get("global_nesting", {})
    findings.append(f"相邻抽象跳跃方向的平均cos: {global_data.get('mean_direction_cos', 0):.3f}")
    findings.append(f"cos标准差: {global_data.get('std_direction_cos', 0):.3f}")
    findings.append(f"正cos对数: {global_data.get('n_positive_cos', 0)}/{global_data.get('n_direction_pairs', 0)}")
    findings.append(f"解读: {global_data.get('interpretation', 'N/A')}")
    
    exp4_results["key_findings"] = "\n".join(findings)
    print("\n  === Exp4 关键发现 ===")
    for f in findings:
        print(f"  - {f}")
    
    release_model(model)
    return exp4_results


def main():
    parser = argparse.ArgumentParser(description="Phase CCLIX: Knowledge Network Mapping")
    parser.add_argument("--model", type=str, default="qwen3", 
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=str, default="all",
                       choices=["1", "2", "3", "4", "all"],
                       help="实验编号 (1=KN-1a, 2=KN-1b, 3=KN-3a, 4=KN-3b)")
    args = parser.parse_args()
    
    print("="*60)
    print(f"Phase CCLIX: Knowledge Network Mapping")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.exp}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_results = {
        "phase": "CCLIX",
        "date": datetime.now().isoformat(),
        "model": args.model,
        "target_cells": ["KN-1a", "KN-1b", "KN-3a", "KN-3b"],
    }
    
    averaged_residuals = None
    
    if args.exp in ["1", "all"]:
        exp1_results, averaged_residuals = run_exp1_concept_encoding(args.model)
        all_results["exp1_KN1a"] = exp1_results
        
        # 保存中间结果
        out_path = OUTPUT_DIR / f"{args.model}_cclix" / "exp1_concept_encoding.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp1_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Exp1 results saved to {out_path}")
    
    if args.exp in ["2", "all"]:
        exp2_results = run_exp2_dimension_sharing(args.model, averaged_residuals)
        all_results["exp2_KN1b"] = exp2_results
        
        out_path = OUTPUT_DIR / f"{args.model}_cclix" / "exp2_dimension_sharing.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp2_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Exp2 results saved to {out_path}")
    
    if args.exp in ["3", "all"]:
        exp3_results, chain_results = run_exp3_abstraction_chains(args.model)
        all_results["exp3_KN3a"] = exp3_results
        
        out_path = OUTPUT_DIR / f"{args.model}_cclix" / "exp3_abstraction_chains.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp3_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Exp3 results saved to {out_path}")
    
    if args.exp in ["4", "all"]:
        exp4_results = run_exp4_abstraction_nesting(args.model)
        all_results["exp4_KN3b"] = exp4_results
        
        out_path = OUTPUT_DIR / f"{args.model}_cclix" / "exp4_abstraction_nesting.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(exp4_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Exp4 results saved to {out_path}")
    
    # 保存完整结果
    full_out_path = OUTPUT_DIR / f"{args.model}_cclix" / "full_results.json"
    with open(full_out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full results saved to {full_out_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Phase CCLIX 总结")
    print("="*60)
    print(f"  目标格子: KN-1a★, KN-1b, KN-3a★, KN-3b")
    print(f"  模型: {args.model}")
    
    for exp_key in all_results:
        if exp_key.startswith("exp"):
            findings = all_results[exp_key].get("key_findings", "")
            if findings:
                print(f"\n  {exp_key}:")
                for line in findings.split("\n"):
                    print(f"    {line}")


if __name__ == "__main__":
    main()
