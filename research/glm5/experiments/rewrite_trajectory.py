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
# 测试样例宇宙：几组具有代表性的词汇
# ----------------------------------------------------
TEST_CONCEPTS = [
    {"name": "cat", "group": "animal"},
    {"name": "quantum mechanics", "group": "science"},
    {"name": "China", "group": "country"},
    {"name": "sadness", "group": "emotion"},
    {"name": "addition", "group": "math"}
]

TOP_K_FEATURES = 30  # 每层我们只保留霸占流形绝对能量的前 30 根孤立神经轴

def run_trajectory_capture():
    print("\n🚀 启动 Qwen3 全层域特征指纹轨迹搜寻 (Concept Trajectory Map)...")
    model = load_qwen3()
    
    total_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    print(f"[*] 准备深入打穿所有 {total_layers} 个特征地层。目标维度宽 {d_model}...")
    
    trajectory_database = []
    
    with torch.no_grad():
        for item in TEST_CONCEPTS:
            concept = item["name"]
            group = item["group"]
            print(f"\n   [--->] 正在下潜剖析概念实体: <{concept}> | 归属: {group}")
            
            prompt = f"The semantic concept of '{concept}' is"
            
            # 完整捕获从底层直至顶层的缓存海
            _, cache = model.run_with_cache(prompt)
            
            concept_trace = {
                "concept": concept,
                "category": group,
                "top_k_limit": TOP_K_FEATURES,
                "layer_traces": []
            }
            
            for L in range(total_layers):
                # 提取概念 token 最后的残差流向量 [1, 2560]
                vector = cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().float()
                
                # 寻找最高能放电的 Top-K 神经元索引及其尖锐值
                top_values, top_indices = torch.topk(vector, TOP_K_FEATURES)
                
                layer_data = {
                    "layer": L,
                    "top_indices": top_indices.tolist(),
                    "top_values": torch.round(top_values * 10000) / 10000.0  # 保留4位精度
                }
                layer_data["top_values"] = layer_data["top_values"].tolist()
                
                concept_trace["layer_traces"].append(layer_data)
                
                if L % 10 == 0 or L == total_layers - 1:
                    print(f"      - 层级(L{L:02d}) 最高能指纹核址: {layer_data['top_indices'][:5]} (峰值: {layer_data['top_values'][0]:.2f})")
                    
            trajectory_database.append(concept_trace)

    # 落盘极其精要的指纹序列文件
    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_trajectory.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "qwen3_concept_trajectory_test.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory_database, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 全态极化特征指纹矩阵打包完毕: {out_file}")

if __name__ == '__main__':
    run_trajectory_capture()
