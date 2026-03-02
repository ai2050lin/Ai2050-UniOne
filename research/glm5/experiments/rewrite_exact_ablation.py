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

# --------------------------------------------------------------------------
# 精确特征寻址：通过“同义重叠”寻找确切的物理子空间维度
# --------------------------------------------------------------------------
TARGET_CONCEPT = "apple"
COMPARISON_CONCEPTS = ["banana", "strawberry", "cherry", "orange"] # 都是典型的水果

# 我们在倒数极其靠后的层级（决定最终输出前）去寻找这些决定性基因
SEARCH_LAYER = 31  # 30~33 是隔离特征最清晰的正交层
TOP_K_SEARCH = 20

def run_exact_decoder():
    print(f"\n🧬 启动微观属性降维解码器...")
    model = load_qwen3()
    
    # --- 阶段 1: 精确定位 "水果 (Fruit)" 的确切核神经元 ---
    print(f"\n[1/3] 正在对齐 {TARGET_CONCEPT} 与其类别近亲的所有激发电位...")
    
    layer_name = f"blocks.{SEARCH_LAYER}.hook_resid_post"
    
    all_indices = []
    with torch.no_grad():
        for word in [TARGET_CONCEPT] + COMPARISON_CONCEPTS:
            prompt = f"The semantic concept of '{word}' is"
            _, cache = model.run_with_cache(prompt)
            vector = cache[layer_name][0, -1, :].cpu().float()
            # 抽出每个水果词汇最核心的 20 根电位
            _, indices = torch.topk(vector, TOP_K_SEARCH)
            all_indices.append(set(indices.tolist()))
            
    # 求交集找到共享的那一个/几个突触
    fruit_dimensions = set.intersection(*all_indices)
    
    if not fruit_dimensions:
        print("    [!] 警告：在这 20 维的极值空间中未找到五种水果完全共享的神经元，退而求其次寻找多数共享。")
        from collections import Counter
        flat_list = [item for sublist in all_indices for item in sublist]
        counter = Counter(flat_list)
        # 取出现最频繁的 3 根神经元作为“水果性”的化身
        fruit_dimensions = set([k for k, v in counter.most_common(3)])
        
    fruit_dim_list = list(fruit_dimensions)
    print(f"    ⭐ 破译成功：在第 {SEARCH_LAYER} 隐层空间中，主宰『水果类别属性』的绝对物理特征维度坐标为: {fruit_dim_list}")
    
    # --- 阶段 2: 原始状态下的输出测试 ---
    print(f"\n[2/3] 正在测定未经干预前，主控探针词 <{TARGET_CONCEPT}> 的自然输出流向...")
    test_prompt = f"An apple is a kind of" # 诱导模型输出 fruit 相关的词汇
    
    # 正常推断
    with torch.no_grad():
        original_logits, _ = model.run_with_cache(test_prompt)
        last_token_logits = original_logits[0, -1, :]
        top_probs, top_tokens = torch.topk(torch.softmax(last_token_logits, dim=-1), 5)
        
        original_preds = [model.to_string(t) for t in top_tokens]
        print(f"    --> 正常补全概率极大值点: {original_preds}")


    # --- 阶段 3: 外科手术级特征消融 (Targeted Feature Ablation) ---
    print(f"\n[3/3] 🔴 正在施行特征流形消融手术 (Ablating Dimensions: {fruit_dim_list})...")
    
    # 定义 Hook 函数：当计算流到达目标层时，将这些“代表水果”的张量维度强制清零
    def ablate_fruit_hook(resid_post, hook):
        # resid_post.shape: [batch, seq_len, d_model]
        # 把代表性神经元干掉
        for dim in fruit_dim_list:
            resid_post[:, -1, dim] = 0.0
        return resid_post
        
    with torch.no_grad():
        ablation_logits = model.run_with_hooks(
            test_prompt,
            fwd_hooks=[(layer_name, ablate_fruit_hook)]
        )
        last_ablation_logits = ablation_logits[0, -1, :]
        top_ablation_probs, top_ablation_tokens = torch.topk(torch.softmax(last_ablation_logits, dim=-1), 5)
        
        ablation_preds = [model.to_string(t) for t in top_ablation_tokens]
        print(f"    --> 💔 失去『水果基因』后的概念漂移异变体: {ablation_preds}")

    # 数据落盘保存
    report = {
        "target": TARGET_CONCEPT,
        "ablation_layer": SEARCH_LAYER,
        "discovered_fruit_genes": fruit_dim_list,
        "test_prompt": test_prompt,
        "original_prediction": original_preds,
        "ablated_prediction": ablation_preds
    }
    
    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_ablation.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_exact_subspace_ablation.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 精确几何特征消融与灾难解构数据定稿: {out_file}")

if __name__ == '__main__':
    run_exact_decoder()
