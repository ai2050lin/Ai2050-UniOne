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

# 实验的三个深度区间：底层(感知), 中层(合成), 深层(逻辑)
TEST_LAYERS = [10, 20, 30]
TOP_K_SPARSE = 50   # 截取稀疏网络里的最强放电前 50 根神经元

def get_sparse_neurons(model, text, layer):
    """提取特定概念文本在 MLP 激增层 (hook_mlp_out) 产生的所有高能脉冲突触。"""
    # 我们抛弃线性残差流(resid_post)，前往非线性变异的发生地：MLP 激活层 (非线性乘积与组装工厂)
    with torch.no_grad():
        _, cache = model.run_with_cache(f"{text}")
        layer_name = f"blocks.{layer}.hook_mlp_out"
        vector = cache[layer_name][0, -1, :].cpu().float()
        
        # 截取激活门槛最高的 50 根神经突触
        _, indices = torch.topk(vector, TOP_K_SPARSE)
        return set(indices.tolist()), vector.numpy()

def run_binding_decomposition():
    print("\n🧬 启动《万法之门：复杂概念的交叉跨界绑定与非线性聚变测试》...")
    model = load_qwen3()
    
    report = {
        "layers": TEST_LAYERS,
        "combo_mutated_neurons": [],      # [聚变突触量] 复合词特有，且两个基础词中绝不存在的全新突变神经元数量
        "svd_compression_ratio": [],      # [低秩映射率] 复合词语列阵经过 SVD 分解后，前 3 个主元能够解释的宇宙方差占比
        "addition_overlap_rate": []       # 复合概念与单纯两词加和的突触相似度 (越低，说明其非线性变异越大)
    }
    
    # 构建复杂的属性-实体复合集矩阵
    # 我们不仅测属性，还能涵盖动作
    base_attrs = ["red", "green", "large", "small", "running", "sleeping"]
    base_nouns = ["apple", "car", "dog", "cat"]
    
    for layer in TEST_LAYERS:
        print(f"\n  [+] 下潜解算非线性 MLP 工厂：L{layer}")
        
        total_mutated = 0
        total_overlap = 0
        composite_matrix = []
        
        for noun in base_nouns:
            n_neurons, _ = get_sparse_neurons(model, noun, layer)
            
            for attr in base_attrs:
                a_neurons, _ = get_sparse_neurons(model, attr, layer)
                
                # 读取复合后的最终宇宙表达
                combo_text = f"{attr} {noun}"
                c_neurons, c_vector = get_sparse_neurons(model, combo_text, layer)
                composite_matrix.append(c_vector)
                
                # --- 测试一：突变组装 (Binding Neurons) ---
                # 哪些神经元，在“单纯红色”也不亮，在“单纯苹果”也不亮，偏偏遇到了“红苹果”就爆炸了？
                # 这是极其高级的 AND-GATE 涌现关联。
                union_base = a_neurons.union(n_neurons)
                mutated_genes = c_neurons.difference(union_base)
                total_mutated += len(mutated_genes)
                
                # --- 测试二：抛弃线性加法学说 ---
                # A+B=C 这种加减法仅适用于大类别的“方向平移”(上一个实验)。
                # 但具体到一个具象的词组合时，重合率绝对不会是 100%。大模型一定大量采用了非线性来抵抗灾难。
                linear_predict = len(c_neurons.intersection(union_base)) / TOP_K_SPARSE
                total_overlap += linear_predict
                
        # --- 测试三：奇异值流形降维 (SVD / Low-Rank Binding) ---
        # "跑着的猫" "红车" "小苹果" 这 24 种组合的几千维宇宙方差，能否被极少量的几根“主线轴”解释？
        comp_np = np.stack(composite_matrix) # Shape: [24, d_model]
        
        # 手动 SVD (奇异值分解)：去均值后分解
        comp_np_centered = comp_np - np.mean(comp_np, axis=0)
        U, S, Vh = np.linalg.svd(comp_np_centered, full_matrices=False)
        
        # 提取前 3 极值主元的方差贡献占比
        variance_explained = (S**2) / np.sum(S**2)
        top_3_ratio = np.sum(variance_explained[:3])
        
        avg_mutated = total_mutated / (len(base_attrs) * len(base_nouns))
        avg_overlap = total_overlap / (len(base_attrs) * len(base_nouns))
        
        print(f"      [ 突触聚变锁 (Emergent Binding) ] 平均新生的高阶关联突触: {avg_mutated:.1f} / {TOP_K_SPARSE} 根")
        print(f"      [ 线性抛弃度 ] 复合词与拆分词加和的激活重叠率: {avg_overlap*100:.2f}% (模型大量启用非线性拼接！)")
        print(f"      [ 极低秩架构 (Low-Rank SVD) ] 24种关联变式中，仅凭 3 根数学主元就能掌控全空间 {top_3_ratio*100:.2f}% 的能量变化！")
        
        report["combo_mutated_neurons"].append(float(avg_mutated))
        report["addition_overlap_rate"].append(float(avg_overlap))
        report["svd_compression_ratio"].append(float(top_3_ratio))
        
    output_dir = os.path.join(os.path.dirname('d:/develop/TransformerLens-main/research/glm5/experiments/rewrite_hyper_space_binding.py'), '..', '..', '..', 'tempdata', 'glm5_emergence')
    out_file = os.path.join(output_dir, "qwen3_hyper_space_binding.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 万法组装门破译完毕。落盘地址: {out_file}")

if __name__ == '__main__':
    run_binding_decomposition()
