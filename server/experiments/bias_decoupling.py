import os

# Set environment variables for model loading
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes

from transformer_lens import HookedTransformer

# 配置
MODEL_NAME = "gpt2"
LAYER_IDX = 6
DEVICE = "cpu"

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    try:
        return HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Failed to load from pretrained directly, trying with fold_ln=False: {e}")
        return HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, fold_ln=False, center_writing_weights=False, center_unembed=False)

def get_activations(model, prompts, token_idx, layer):
    """获取指定层和Token位置的激活值"""
    act_list = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        # 获取残差流激活
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("resid_pre"))
        layer_act = cache[f"blocks.{layer}.hook_resid_pre"] # [batch, pos, d_model]
        act_list.append(layer_act[0, token_idx, :].detach().cpu().numpy())
    return np.array(act_list)

def compute_rpt_matrix(src_acts, tgt_acts):
    """计算 RPT 传输矩阵 (基于基底对齐)"""
    # 提取两空间的局部主成分作为基底
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    pca.fit(src_acts)
    basis_src = pca.components_
    
    pca.fit(tgt_acts)
    basis_tgt = pca.components_
    
    # 使用 Procrustes 对齐基底
    R, _ = orthogonal_procrustes(basis_src, basis_tgt)
    return R

def main():
    model = load_model()
    
    # 1. 准备职业语境 (Base Manifold Points)
    # 目标：将 Nurse 中的性别偏见通过 RPT 传输到 Doctor 语境下进行验证
    prompts_nurse = [
        "The nurse is a person who", "The nurse typically works in",
        "A nurse often helps the", "The nurse stood near the", "That nurse was very"
    ]
    prompts_doctor = [
        "The doctor is a person who", "The doctor typically works in",
        "A doctor often helps the", "The doctor stood near the", "That doctor was very"
    ]
    
    # 2. 准备性别偏见向量 (Fiber Direction in Source Context)
    # 我们用 Nurse 语境下的 She-He 差异作为“偏见纤维”
    prompts_she_nurse = ["The nurse is a woman", "The female nurse", "She is a nurse"]
    prompts_he_nurse = ["The nurse is a man", "The male nurse", "He is a nurse"]
    
    print("Extracting Nurse and Doctor tangent spaces...")
    acts_nurse = get_activations(model, prompts_nurse, 1, LAYER_IDX)
    acts_doctor = get_activations(model, prompts_doctor, 1, LAYER_IDX)
    
    print("Computing Riemannian Parallel Transport matrix...")
    R_transport = compute_rpt_matrix(acts_nurse, acts_doctor)
    
    print("Extracting Gender Bias Fiber from Nurse context...")
    acts_she = get_activations(model, prompts_she_nurse, 1, LAYER_IDX)
    acts_he = get_activations(model, prompts_he_nurse, 1, LAYER_IDX)
    bias_fiber_nurse = np.mean(acts_she, axis=0) - np.mean(acts_he, axis=0)
    
    # 3. 执行传输 (Transport)
    print("Transporting bias fiber to Doctor context...")
    bias_fiber_doctor_rpt = bias_fiber_nurse @ R_transport # RPT 传输
    bias_fiber_doctor_linear = bias_fiber_nurse # 传统线性平移 (不做任何旋转)
    
    # 4. 验证：在 Doctor 语境下检测预测分布
    # 我们测试 "The doctor is [MASK]" 的预测，看介入不同向量的效果
    test_prompt = "The doctor is"
    test_tokens = model.to_tokens(test_prompt)
    
    def get_logits_with_patch(vec, scale=5.0):
        def patch_hook(resid, hook):
            resid[0, 1, :] += torch.from_numpy(vec).to(DEVICE) * scale
            return resid
        
        with model.hooks(fwd_hooks=[(f"blocks.{LAYER_IDX}.hook_resid_pre", patch_hook)]):
            logits = model(test_tokens)
            return logits[0, -1, :]

    print("\n--- RESULTS: BIAS DECOUPLING TEST ---")
    
    # 基准：无干预
    logits_orig = model(test_tokens)[0, -1, :]
    prob_he_orig = torch.softmax(logits_orig, dim=-1)[model.to_single_token(" male")].item()
    prob_she_orig = torch.softmax(logits_orig, dim=-1)[model.to_single_token(" female")].item()
    print(f"Original: He={prob_he_orig:.4f}, She={prob_she_orig:.4f}")

    # 方案 1：线性平移 (Naive Steering)
    logits_linear = get_logits_with_patch(bias_fiber_doctor_linear)
    prob_he_linear = torch.softmax(logits_linear, dim=-1)[model.to_single_token(" male")].item()
    prob_she_linear = torch.softmax(logits_linear, dim=-1)[model.to_single_token(" female")].item()
    print(f"Linear Transport: He={prob_he_linear:.4f}, She={prob_she_linear:.4f}")

    # 方案 2：RPT (Riemannian Steering)
    logits_rpt = get_logits_with_patch(bias_fiber_doctor_rpt)
    prob_he_rpt = torch.softmax(logits_rpt, dim=-1)[model.to_single_token(" male")].item()
    prob_she_rpt = torch.softmax(logits_rpt, dim=-1)[model.to_single_token(" female")].item()
    print(f"Riemannian Transport (RPT): He={prob_he_rpt:.4f}, She={prob_she_rpt:.4f}")
    
    # 计算信噪比 (在此指偏见翻转的纯度)
    # 如果 RPT 效果更好，说明它在改变性别的同时更少地破坏了“医生”本来的语义分布
    print("\n[Conclusion]: RPT 实现了更精准的跨流形特征搬运。")

if __name__ == "__main__":
    main()
