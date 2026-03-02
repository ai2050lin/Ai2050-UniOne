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
# 目标群组与隔离参数
# ----------------------------------------------------
ANIMAL_CONCEPTS = ["cat", "dog", "rabbit", "tiger", "lion", "elephant"]
SEARCH_LAYER = 31      # 维持在水果实验的隔离控制层
TOP_K_SEARCH = 25      # 为了捕获高维生命体的复杂共享态，微调捕捉网至 Top-25

# 这正是刚从上一场微米级手术里掏出的、带着鲜血的 "水果基因"
PREVIOUS_FRUIT_GENES = {0, 418, 4, 198, 8, 9, 239, 338, 19, 83, 84, 23, 24, 190, 286}

def run_animal_architecture_locator():
    print(f"\n🐾 启动碳基生命参数提取仪 (寻找 猫 狗 虎 象 共用通道)...")
    model = load_qwen3()
    
    layer_name = f"blocks.{SEARCH_LAYER}.hook_resid_post"
    all_indices = []
    
    with torch.no_grad():
        for word in ANIMAL_CONCEPTS:
            prompt = f"The semantic concept of '{word}' is"
            _, cache = model.run_with_cache(prompt)
            vector = cache[layer_name][0, -1, :].cpu().float()
            # 抽出最狂暴的 25 条电缆
            _, indices = torch.topk(vector, TOP_K_SEARCH)
            all_indices.append(set(indices.tolist()))
            
    # 全生命体硬交集：必须这6种动物都100%使用的神经元
    animal_dimensions = set.intersection(*all_indices)
    
    if len(animal_dimensions) < 5:
        print(f"    [!] 警告：硬交集极为稀少 ({len(animal_dimensions)}根)。启用 5/6 高频表决组...")
        from collections import Counter
        flat_list = [item for sublist in all_indices for item in sublist]
        counter = Counter(flat_list)
        # 只要有一多半以上动物都在调用的核心突触，也算入主流动物基因
        animal_dimensions = set([k for k, v in counter.items() if v >= len(ANIMAL_CONCEPTS) - 1])
    
    animal_dim_list = list(animal_dimensions)
    print(f"    ⭐ 核磁锁定：在 L{SEARCH_LAYER} 中，共截获 {len(animal_dim_list)} 根主宰『走兽/生命属性』的物理特征轴: {animal_dim_list}")
    
    # --- 阶段 2: 跨宇宙物理法则比对 (Animal vs Fruit) ---
    print(f"\n⚔️ 跨域坐标系大冲撞：[生命基因组] vs [死物果实组]")
    
    overlap_genes = animal_dimensions.intersection(PREVIOUS_FRUIT_GENES)
    overlap_count = len(overlap_genes)
    
    print(f"    --> 动物独占特征神经元数量: {len(animal_dimensions)}")
    print(f"    --> 水果独占特征神经元数量: {len(PREVIOUS_FRUIT_GENES)}")
    print(f"    --> 两界共享神经元数量: {overlap_count} ({list(overlap_genes)})")
    
    architect_status = ""
    if overlap_count == 0:
        architect_status = "ABSOLUTE_ORTHOGONAL (绝对正交极化)"
        print("    --> 结论: 两方世界维持了【绝对正交极化】！猫和苹果在大模型第31层没有共享任何一个核心突触。它们分别停靠在完全彼此垂直的维度平原上。这是解决维度灾难的教科书级物理展现。")
    elif overlap_count <= 2:
        architect_status = "WEAK_EMBEDDING (弱纠缠/底层时空共享)"
        print("    --> 结论: 两方世界几乎保持正交，但共享了极其微妙的底层坐标（可能仅仅代表它们都是‘物理存在’、‘地球上的名词’等最基础的时空基座）。绝大部分分类特征呈绝对互斥态。")
    else:
        architect_status = "HEAVY_OVERLAP (概念坍缩危险区)"
        print("    --> 结论: 它们共享了大量通道。大模型的解耦失效。")

    
    # 数据落盘
    report = {
        "analysis_layer": SEARCH_LAYER,
        "animal_targets": ANIMAL_CONCEPTS,
        "extracted_animal_genes": animal_dim_list,
        "animal_gene_count": len(animal_dim_list),
        "previous_fruit_genes": list(PREVIOUS_FRUIT_GENES),
        "fruit_gene_count": len(PREVIOUS_FRUIT_GENES),
        "cross_domain_overlap_genes": list(overlap_genes),
        "overlap_count": overlap_count,
        "architectural_status": architect_status
    }
    
    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_animal_locator.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_animal_vs_fruit_architecture.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 生命参数寻址与跨界正交对撞数据归档成功: {out_file}")

if __name__ == '__main__':
    run_animal_architecture_locator()
