# -*- coding: utf-8 -*-
"""
Qwen3 编码结构四维度提取器
=========================
从 Qwen3-4B 中提取编码，验证四个关键数学特性：
  1. 高维抽象 — 语义收敛能力
  2. 低维精确 — 细粒度区分能力
  3. 特异性 — 概念子空间正交性
  4. 系统性 — 类比关系一致性

输出: tempdata/qwen3_structure_report.json + 4 张可视化图
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # 无头模式，兼容服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 第零部分：模型加载（复用已验证的 import_trace.py 逻辑）
# ============================================================

SNAPSHOT_PATH = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"

# 环境变量
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


def load_qwen3():
    """加载 Qwen3-4B 为 HookedTransformer"""
    import transformers.configuration_utils as config_utils
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 加载 Qwen3-4B，设备: {device}")
    print(f"    路径: {SNAPSHOT_PATH}")

    t0 = time.time()

    # 步骤 1: 在 CPU 上加载 HF 模型 (HookedTransformer 会自行处理设备迁移)
    hf_model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT_PATH, local_files_only=True, add_bos_token=False
    )

    # 修复1: Qwen3 tokenizer 缺少 bos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
        print(f"    [fix] 设置 bos_token = eos_token ({tokenizer.bos_token})")

    # 修复2: Monkey-patch PretrainedConfig 以修复 rope_theta
    _orig_getattr = config_utils.PretrainedConfig.__getattribute__

    def _patched_getattr(self, key):
        if key == "rope_theta":
            try:
                return _orig_getattr(self, key)
            except AttributeError:
                try:
                    rs = _orig_getattr(self, "rope_scaling")
                    if isinstance(rs, dict) and "rope_theta" in rs:
                        return rs["rope_theta"]
                except (AttributeError, TypeError):
                    pass
                return 1000000
        return _orig_getattr(self, key)

    config_utils.PretrainedConfig.__getattribute__ = _patched_getattr

    # 修复3: Monkey-patch get_tokenizer_with_bos 避免重新加载 tokenizer
    import transformer_lens.utils as tl_utils
    _orig_get_tok_bos = tl_utils.get_tokenizer_with_bos

    def _patched_get_tok_bos(tok):
        # 直接返回已修复的 tokenizer，避免重新 from_pretrained
        return tok

    tl_utils.get_tokenizer_with_bos = _patched_get_tok_bos
    print("    [fix] 已 monkey-patch rope_theta + get_tokenizer_with_bos")

    try:
        model = HookedTransformer.from_pretrained(
            "Qwen/Qwen3-4B", hf_model=hf_model, device=device, tokenizer=tokenizer,
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            dtype=torch.float16, default_prepend_bos=False
        )
    finally:
        config_utils.PretrainedConfig.__getattribute__ = _orig_getattr
        tl_utils.get_tokenizer_with_bos = _orig_get_tok_bos
        print("    [fix] 已恢复所有 monkey-patch")

    model.eval()
    print(f"[+] 模型加载完成 ({time.time() - t0:.1f}s)")
    print(f"    层数: {model.cfg.n_layers}, 维度: {model.cfg.d_model}")
    return model

# ----------------------------------------------------
# 锚点与对比矩阵构建
# ----------------------------------------------------
ANCHOR = "apple"
COMPARISON_TARGETS = {
    "CloseRelative": "banana",
    "DistantRelative": "rabbit",
    "Irrelevant": "sun"
}

# 随机宇宙噪声：为了保证统计平滑，选取数百个涵盖所有领域的单词
UNIVERSE_SAMPLES = [
    "dog", "car", "happy", "physics", "USA", "hammer", "Hydrogen", "second", 
    "tiger", "bus", "sad", "chemistry", "China", "screwdriver", "Helium", "minute",
    "bear", "train", "angry", "biology", "Japan", "wrench", "Lithium", "hour",
    "elephant", "airplane", "fearful", "mathematics", "Germany", "saw", "Carbon", "day",
    "monkey", "boat", "surprised", "medicine", "France", "drill", "Oxygen", "year",
    "whale", "submarine", "proud", "engineering", "Russia", "clamp", "Sodium", "century",
    "computer", "internet", "democracy", "freedom", "war", "peace", "love", "hate",
    "guitar", "piano", "music", "art", "painting", "sculpture", "mountain", "river",
    "ocean", "desert", "forest", "rain", "snow", "wind", "storm", "earthquake",
    "king", "queen", "president", "doctor", "teacher", "student", "police", "soldier",
    "pizza", "burger", "sushi", "pasta", "bread", "cheese", "wine", "coffee"
]

TOP_K_CAPTURE = 30

def calculate_jaccard(setA, setB):
    intersect = len(setA.intersection(setB))
    union = len(setA.union(setB))
    return intersect / union if union > 0 else 0.0

def run_anchor_topology():
    print(f"\n🍎 启动锚点万物测度仪 (Anchor: <{ANCHOR}>)...")
    model = load_qwen3()
    total_layers = model.cfg.n_layers
    
    # 构建探测池
    probe_concepts = [ANCHOR] + list(COMPARISON_TARGETS.values()) + UNIVERSE_SAMPLES
    # 去重
    probe_concepts = list(dict.fromkeys(probe_concepts))
    print(f"[*] 共计铺设 {len(probe_concepts)} 根测度探针，正横切 {total_layers} 层隐矩阵...")
    
    # 捕获所有探测词的特征序列
    concept_traces = {}
    with torch.no_grad():
        t0 = time.time()
        for i, concept in enumerate(probe_concepts):
            if i > 0 and i % 25 == 0:
                print(f"    --> 已深入采样 {i} 个特征波群 ({time.time()-t0:.1f}s)")
            
            prompt = f"The semantic concept of '{concept}' is"
            _, cache = model.run_with_cache(prompt)
            
            traces = []
            for L in range(total_layers):
                vector = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().float()
                _, top_indices = torch.topk(vector, TOP_K_CAPTURE)
                traces.append(set(top_indices.tolist()))
            
            concept_traces[concept] = traces

    print("\n🍎 开始计算各层针对锚点的 Jaccard 碰撞拓扑...")
    
    topology_evolution = []
    
    for L in range(total_layers):
        anchor_set = concept_traces[ANCHOR][L]
        
        # 精确对比指标
        sim_banana = calculate_jaccard(anchor_set, concept_traces[COMPARISON_TARGETS["CloseRelative"]][L])
        sim_rabbit = calculate_jaccard(anchor_set, concept_traces[COMPARISON_TARGETS["DistantRelative"]][L])
        sim_sun = calculate_jaccard(anchor_set, concept_traces[COMPARISON_TARGETS["Irrelevant"]][L])
        
        # 计算大宇宙背景均值
        universe_sims = []
        for word in UNIVERSE_SAMPLES:
            if word != ANCHOR:
                universe_sims.append(calculate_jaccard(anchor_set, concept_traces[word][L]))
        
        sim_universe = float(np.mean(universe_sims)) if universe_sims else 0.0
        
        stat = {
            "layer": L,
            "anchor": ANCHOR,
            "vs_banana": round(sim_banana * 100, 2),
            "vs_rabbit": round(sim_rabbit * 100, 2),
            "vs_sun": round(sim_sun * 100, 2),
            "vs_random_universe": round(sim_universe * 100, 2)
        }
        topology_evolution.append(stat)
        
        if L % 5 == 0 or L == total_layers - 1:
            print(f"   [Layer {L:02d}] 🍎 vs 🍌香蕉({stat['vs_banana']:4.1f}%) | 🐇兔子({stat['vs_rabbit']:4.1f}%) | ☀️太阳({stat['vs_sun']:4.1f}%) | 🌌大宇宙背景排斥({stat['vs_random_universe']:4.1f}%)")

    # 封存拓扑曲线
    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_anchor_topology.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_apple_universe_topology.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(topology_evolution, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Apple 万物引力测度归档完毕: {out_file}")

if __name__ == '__main__':
    run_anchor_topology()
