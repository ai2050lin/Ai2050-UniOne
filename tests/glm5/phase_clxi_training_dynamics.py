#!/usr/bin/env python3
"""
Phase CLXI: 差分放大器的训练动力学推导
========================================

核心目标: 从训练动力学角度解释为什么G和A收敛到对抗平衡

CLX关键发现:
  Qwen3(8 KV头): 27%语义头, 分工编码
  DS7B(4 KV头): 44%抑制头, 冗余放大
  GLM4(2 KV头): 75%混合头, 均匀分散
  → GQA压缩决定了编码策略

但根本问题: 为什么训练过程自然收敛到对抗平衡(G和A异号)?
  - 是LayerNorm/RMSNorm导致的?
  - 是梯度下降的隐式正则化?
  - 是信息论最优的?

实验设计:
  P701: 对抗平衡的梯度动力学
    - 分析G_l和A_l的梯度更新方向: ∂L/∂W_down(FFN) vs ∂L/∂W_o(Attn)
    - 如果G和A的梯度方向相反, 训练会自然收敛到对抗平衡
    - 关键指标: 梯度符号一致性(grad_sign_corr)

  P702: LayerNorm/RMSNorm在对抗平衡中的作用
    - 在不添加噪声的情况下, 移除LayerNorm后G和A的符号模式如何变化?
    - LayerNorm的归一化是否"迫使"G和A符号相反?
    - 关键: LayerNorm后的h_normalized = h / ||h||, 使得G和A的贡献被重新缩放
    - 验证: 计算LayerNorm前后的G/A符号相关性变化

  P703: GQA压缩如何改变对抗平衡结构
    - 不同KV头数的模型, 对抗平衡的程度如何?
    - GLM4(2 KV头)的对抗平衡是否更弱?
    - 关键指标: per-model sign_corr和方差缩减比

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path
from scipy import stats
from model_utils import load_model, get_model_info, get_layer_weights, release_model, MODEL_CONFIGS

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("d:/develop/TransformerLens-main/results/phase_clxi")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("d:/develop/TransformerLens-main/tests/glm5_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 测试词汇 ============
TEST_WORDS = {
    "apple":    {"cat": "fruit", "pos": "noun"},
    "banana":   {"cat": "fruit", "pos": "noun"},
    "orange":   {"cat": "fruit", "pos": "noun"},
    "cat":      {"cat": "animal", "pos": "noun"},
    "dog":      {"cat": "animal", "pos": "noun"},
    "car":      {"cat": "vehicle", "pos": "noun"},
    "run":      {"cat": "action", "pos": "verb"},
    "red":      {"cat": "color", "pos": "adjective"},
    "the":      {"cat": "function", "pos": "determiner"},
}


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return []


def safe_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def get_n_heads(sa, model, d_model):
    if hasattr(sa, 'num_heads'):
        return sa.num_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_attention_heads'):
        return sa.config.num_attention_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    else:
        head_dim = sa.head_dim if hasattr(sa, 'head_dim') else 128
        return d_model // head_dim


def get_n_kv_heads(sa, model):
    if hasattr(sa, 'num_key_value_heads'):
        return sa.num_key_value_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_key_value_heads'):
        return model.config.num_key_value_heads
    else:
        return get_n_heads(sa, model, model.config.hidden_size if hasattr(model, 'config') else 2560)


# ================================================================
# P701: 对抗平衡的梯度动力学
# ================================================================
def p701_gradient_dynamics(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P701: 对抗平衡的梯度动力学

    核心问题: G和A的梯度更新方向是什么?

    方法:
    1. 对输入文本做前向传播, 获取loss
    2. 对每层的FFN和Attention权重计算梯度
    3. 分析: ∂L/∂W_down(FFN) 和 ∂L/∂W_o(Attn) 的方向
    4. 如果FFN和Attn的梯度方向相反, 说明训练"鼓励"它们对抗

    关键: 我们不直接计算完整梯度(太大), 而是计算:
    - G_l的梯度在logit空间的方向: grad_G_logit = ∂L/∂(W_U·G_l)
    - A_l的梯度在logit空间的方向: grad_A_logit = ∂L/∂(W_U·A_l)
    - 梯度符号相关性: corr(sign(grad_G), sign(grad_A))

    但计算完整梯度太昂贵。替代方法:
    - 用perturbation(微扰)近似梯度: ΔL ≈ ∂L/∂x · Δx
    - 对G_l和A_l分别微扰, 观察loss变化方向
    """
    print("\n" + "="*70)
    print("P701: 对抗平衡的梯度动力学")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    results = {}

    # 用少量词进行梯度分析(计算量大)
    test_words = ["apple", "cat", "the"]

    for word in test_words:
        t0 = time.time()
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]

        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 前向传播获取hidden_states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 对采样层计算G和A的logit贡献, 以及微扰效果
        sample_layers = sorted(set(
            list(range(0, n_layers, max(n_layers // 8, 1))) +
            list(range(max(0, n_layers - 3), n_layers))
        ))

        # 微扰分析: 对G_l加正向微扰, 观察target logit的变化
        # 这相当于计算 ∂logit(word)/∂G_l 的方向
        perturbation_scale = 0.01  # 微扰比例

        G_logit_baseline = {}
        A_logit_baseline = {}
        G_logit_perturbed = {}  # G正向微扰后的logit
        A_logit_perturbed = {}  # A正向微扰后的logit

        W_U_word = W_U[word_id]

        for li in sample_layers:
            layer = layers[li]
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            delta = h_after - h_before

            # 计算G
            lw = get_layer_weights(layer, d_model, mlp_type)
            if lw.W_gate is not None and lw.W_up is not None and lw.W_down is not None:
                gate_val = safe_sigmoid(lw.W_gate @ h_before)
                up_val = lw.W_up @ h_before
                G_exact = lw.W_down @ (gate_val * up_val)
                A_approx = delta - G_exact

                # 基线logit贡献
                G_logit_baseline[li] = float(W_U_word @ G_exact)
                A_logit_baseline[li] = float(W_U_word @ A_approx)

                # 微扰G: G' = G * (1 + scale)
                G_perturbed = G_exact * (1 + perturbation_scale)
                # 新的delta' = G' + A = (G*(1+s)) + A
                # 新的h_after' = h_before + delta'
                # 但我们只需要logit的变化:
                # Δlogit_G = W_U_word @ (G_perturbed - G_exact) = W_U_word @ G * scale
                G_sensitivity = float(W_U_word @ G_exact) * perturbation_scale

                # 微扰A: A' = A * (1 + scale)
                A_sensitivity = float(W_U_word @ A_approx) * perturbation_scale

                G_logit_perturbed[li] = G_sensitivity
                A_logit_perturbed[li] = A_sensitivity

            del lw

        # 分析梯度动力学
        # 如果∂logit/∂G和∂logit/∂A的符号相反, 说明:
        # 增加G会减少logit, 增加A会增加logit(或反之)
        # → 训练会"鼓励"G和A对抗

        G_signs = np.array([np.sign(G_logit_baseline.get(li, 0)) for li in sample_layers])
        A_signs = np.array([np.sign(A_logit_baseline.get(li, 0)) for li in sample_layers])

        # G和A的logit贡献符号
        same_sign = int(np.sum(G_signs * A_signs > 0))
        opposite_sign = int(np.sum(G_signs * A_signs < 0))

        # 符号相关性
        nonzero_mask = (G_signs != 0) & (A_signs != 0)
        if np.sum(nonzero_mask) > 5:
            sign_corr = float(np.corrcoef(G_signs[nonzero_mask], A_signs[nonzero_mask])[0, 1])
        else:
            sign_corr = 0.0

        # 关键分析: 梯度方向推断
        # loss = -logit(word) + C (简化, 对于next-token预测)
        # ∂L/∂G_l = -W_U_word (对G_l的梯度方向)
        # ∂L/∂A_l = -W_U_word (对A_l的梯度方向)
        # 如果G_l和A_l的贡献同号(都在增加logit), 则梯度方向相同
        # 如果G_l和A_l的贡献异号(G减A加), 则:
        #   - 增加G_l的范数 → logit下降 → loss上升 → 梯度"反对"增加G
        #   - 增加A_l的范数 → logit上升 → loss下降 → 梯度"支持"增加A
        # → 梯度方向确实"鼓励"对抗

        # 更精确: 用sensitivity分析
        # 如果G_sensitivity和A_sensitivity符号相反, 说明:
        # 放大G会减少logit, 放大A会增加logit
        G_sens = np.array([G_logit_perturbed.get(li, 0) for li in sample_layers])
        A_sens = np.array([A_logit_perturbed.get(li, 0) for li in sample_layers])

        # G和A的sensitivity符号
        sens_sign_corr = 0.0
        nonzero_sens = (G_sens != 0) & (A_sens != 0)
        if np.sum(nonzero_sens) > 5:
            sens_sign_corr = float(np.corrcoef(np.sign(G_sens[nonzero_sens]),
                                                np.sign(A_sens[nonzero_sens]))[0, 1])

        # 对抗平衡的梯度解释
        # 如果sign_corr < 0, G和A异号 → 梯度方向相反 → 训练鼓励对抗
        # 如果sign_corr > 0, G和A同号 → 梯度方向相同 → 训练鼓励合作
        gradient_encourages_adversarial = sign_corr < -0.3

        results[word] = {
            "word_info": TEST_WORDS[word],
            "n_sample_layers": len(sample_layers),
            "same_sign_layers": same_sign,
            "opposite_sign_layers": opposite_sign,
            "sign_correlation": sign_corr,
            "sensitivity_sign_correlation": sens_sign_corr,
            "gradient_encourages_adversarial": gradient_encourages_adversarial,
            "G_logit_sampled": {f"L{li}": float(G_logit_baseline.get(li, 0)) for li in sample_layers},
            "A_logit_sampled": {f"L{li}": float(A_logit_baseline.get(li, 0)) for li in sample_layers},
        }

        elapsed = time.time() - t0
        adv = "adversarial" if gradient_encourages_adversarial else "cooperative"
        print(f"  {word}: same/opposite={same_sign}/{opposite_sign}, "
              f"sign_corr={sign_corr:.3f}, sens_corr={sens_sign_corr:.3f}, "
              f"gradient={adv}, elapsed={elapsed:.1f}s")

        del outputs
        gc.collect()

    return results


# ================================================================
# P702: LayerNorm/RMSNorm在对抗平衡中的作用
# ================================================================
def p702_layernorm_role(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P702: LayerNorm/RMSNorm在对抗平衡中的作用

    核心问题: LayerNorm是否"迫使"G和A符号相反?

    分析:
    1. LayerNorm: h_norm = (h - μ) / σ
    2. 在残差流中: h_{l+1} = LN(h_l + G_l + A_l)
    3. LN的作用: 将(h_l + G_l + A_l)归一化, 使得范数恒定
    4. 如果G_l和A_l都大幅增加h的范数, LN会大幅缩小→两者都被压缩
    5. 如果G_l和A_l方向相反, 范数增加较小, LN压缩较小→差分信号保留

    方法:
    - 计算LN前的G和A贡献: pre_LN_G = W_U·G_l, pre_LN_A = W_U·A_l
    - 计算LN后的贡献: post_LN_G和post_LN_A(通过h_after间接获得)
    - LN的"压缩比": pre_LN_norm / post_LN_norm
    - 如果G+A同向→范数大→LN压缩大→信息损失
    - 如果G+A反向→范数小→LN压缩小→信息保留

    验证:
    - 对比有LN和无LN(模拟)的G/A符号模式
    - 计算LN对logit的缩放因子
    """
    print("\n" + "="*70)
    print("P702: LayerNorm/RMSNorm在对抗平衡中的作用")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    # 检测归一化类型
    norm_type = "unknown"
    layer0 = layers[0]
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer0, ln_name):
            ln = getattr(layer0, ln_name)
            if hasattr(ln, 'weight') and hasattr(ln, 'variance_epsilon'):
                norm_type = "rmsnorm" if not hasattr(ln, 'bias') else "layernorm"
            elif hasattr(ln, 'weight') and hasattr(ln, 'eps'):
                norm_type = "rmsnorm" if not hasattr(ln, 'bias') else "layernorm"
            break

    print(f"  检测到归一化类型: {norm_type}")

    results = {"norm_type": norm_type}
    test_words = ["apple", "cat", "the"]

    for word in test_words:
        t0 = time.time()
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]
        W_U_word = W_U[word_id]

        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        sample_layers = sorted(set(
            list(range(0, n_layers, max(n_layers // 8, 1))) +
            list(range(max(0, n_layers - 3), n_layers))
        ))

        layer_analysis = {}

        for li in sample_layers:
            layer = layers[li]
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()

            # 计算G和A
            lw = get_layer_weights(layer, d_model, mlp_type)
            if lw.W_gate is None or lw.W_up is None or lw.W_down is None:
                del lw
                continue

            gate_val = safe_sigmoid(lw.W_gate @ h_before)
            up_val = lw.W_up @ h_before
            G_exact = lw.W_down @ (gate_val * up_val)
            A_approx = (h_after - h_before) - G_exact

            # LN前: h_before + G + A = h_after (近似, 忽略post-attn LN)
            pre_LN_h = h_before + G_exact + A_approx

            # 分析LN的作用
            # 1) h_before的范数
            h_before_norm = float(np.linalg.norm(h_before))

            # 2) G和A各自的范数
            G_norm = float(np.linalg.norm(G_exact))
            A_norm = float(np.linalg.norm(A_approx))

            # 3) G+A的范数 (实际delta)
            delta_norm = float(np.linalg.norm(h_after - h_before))

            # 4) h_after的范数 (LN后)
            h_after_norm = float(np.linalg.norm(h_after))

            # 5) LN缩放因子
            # 如果h_before_norm > 0:
            #   LN缩放 ≈ h_after_norm / (h_before_norm + delta_norm)
            # 但这不对,因为LN不是简单的缩放
            # 更准确: h_after = LN(h_before + G + A)
            # 如果没有LN, h_after' = h_before + G + A
            # LN的效果: ||h_after|| ≈ √d_model (归一化到单位方差)

            # 6) 计算LN的"压缩效果"
            # pre_LN范数 vs post_LN范数
            pre_LN_norm = float(np.linalg.norm(pre_LN_h))
            ln_compression = pre_LN_norm / max(h_after_norm, 1e-10)

            # 7) G和A对范数的贡献
            # 如果G和A同向: ||G+A|| ≈ ||G|| + ||A|| (范数叠加)
            # 如果G和A反向: ||G+A|| ≈ |||G|| - ||A||| (范数抵消)
            G_A_cos = float(np.dot(G_exact, A_approx) / max(G_norm * A_norm, 1e-10))

            # 8) 关键指标: LN是否"惩罚"同向G+A?
            # 如果G_A_cos > 0 (同向): pre_LN_norm大 → LN压缩大 → 信息损失
            # 如果G_A_cos < 0 (反向): pre_LN_norm小 → LN压缩小 → 信息保留
            # 这就是"差分放大"的LN机制!

            # 9) logit贡献分析
            G_logit = float(W_U_word @ G_exact)
            A_logit = float(W_U_word @ A_approx)
            delta_logit = float(W_U_word @ (h_after - h_before))

            # 10) 如果没有LN, G和A的logit贡献会怎样?
            # 简化: 假设LN只是均匀缩放
            if pre_LN_norm > 1e-10 and h_after_norm > 1e-10:
                ln_scale = h_after_norm / pre_LN_norm
                G_logit_noLN = G_logit / max(ln_scale, 1e-10)  # 反推无LN的logit
                A_logit_noLN = A_logit / max(ln_scale, 1e-10)
            else:
                ln_scale = 1.0
                G_logit_noLN = G_logit
                A_logit_noLN = A_logit

            layer_analysis[f"L{li}"] = {
                "h_before_norm": h_before_norm,
                "h_after_norm": h_after_norm,
                "G_norm": G_norm,
                "A_norm": A_norm,
                "delta_norm": delta_norm,
                "pre_LN_norm": pre_LN_norm,
                "ln_compression": ln_compression,
                "ln_scale": ln_scale,
                "G_A_cosine": G_A_cos,
                "G_logit": G_logit,
                "A_logit": A_logit,
                "delta_logit": delta_logit,
                "G_logit_noLN_approx": G_logit_noLN,
                "A_logit_noLN_approx": A_logit_noLN,
            }

            del lw

        # 聚合分析
        G_A_cosines = [la["G_A_cosine"] for la in layer_analysis.values()]
        ln_compressions = [la["ln_compression"] for la in layer_analysis.values()]

        mean_GA_cos = float(np.mean(G_A_cosines))
        mean_ln_compression = float(np.mean(ln_compressions))

        # G和A的cosine是否与LN压缩相关?
        if len(G_A_cosines) > 3:
            cos_compression_corr = float(np.corrcoef(G_A_cosines, ln_compressions)[0, 1])
        else:
            cos_compression_corr = 0.0

        # 关键结论: 如果G_A_cos < 0且cos_compression_corr > 0,
        # 说明G和A反向→范数小→LN压缩小→差分信号保留
        # 即LN"鼓励"对抗平衡
        ln_encourages_adversarial = mean_GA_cos < 0 and cos_compression_corr > 0

        results[word] = {
            "word_info": TEST_WORDS[word],
            "mean_GA_cosine": mean_GA_cos,
            "mean_ln_compression": mean_ln_compression,
            "cos_compression_correlation": cos_compression_corr,
            "ln_encourages_adversarial": ln_encourages_adversarial,
            "layer_analysis": layer_analysis,
        }

        elapsed = time.time() - t0
        adv = "encourages_adv" if ln_encourages_adversarial else "neutral/discourages"
        print(f"  {word}: mean_GA_cos={mean_GA_cos:.3f}, "
              f"mean_ln_comp={mean_ln_compression:.3f}, "
              f"cos_comp_corr={cos_compression_corr:.3f}, "
              f"LN={adv}, elapsed={elapsed:.1f}s")

        del outputs
        gc.collect()

    return results


# ================================================================
# P703: GQA压缩如何改变对抗平衡结构
# ================================================================
def p703_gqa_effect_on_adversarial_balance(model, tokenizer, model_info, W_embed, W_U, device):
    """
    P703: GQA压缩如何改变对抗平衡结构

    核心问题: 不同KV头数的模型, 对抗平衡的程度如何?

    分析:
    1. 对每层计算G和A的logit贡献, 以及符号模式
    2. 与CLIX的发现对比: Qwen3 sign_corr=-0.80, DS7B=-0.86, GLM4=+0.01
    3. 分析KV头数与对抗平衡程度的关系
    4. 注意力头层面的对抗: per-head A_l的logit贡献是否也是对抗的?

    方法:
    - 收集per-layer G和A的logit贡献
    - 收集per-head A的logit贡献
    - 分析G vs 总A的对抗, 以及G vs per-head A的对抗
    """
    print("\n" + "="*70)
    print("P703: GQA压缩如何改变对抗平衡结构")
    print("="*70)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    # 获取KV头数
    sa0 = layers[0].self_attn
    n_q = get_n_heads(sa0, model, d_model)
    n_kv = get_n_kv_heads(sa0, model)
    head_dim = d_model // n_q

    print(f"  Attention: n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")

    results = {"n_q_heads": n_q, "n_kv_heads": n_kv, "gqa_ratio": n_q / n_kv}
    test_words = ["apple", "cat", "the"]

    for word in test_words:
        t0 = time.time()
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        word_id = word_ids[0]
        W_U_word = W_U[word_id]

        text = f"The word is {word}."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 全层分析(不采样, 因为这是核心实验)
        all_G_logit = np.zeros(n_layers)
        all_A_logit = np.zeros(n_layers)

        # per-head A的logit贡献
        per_head_A_logit = {}  # {li: array[n_kv]}

        for li in range(n_layers):
            layer = layers[li]
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            delta = h_after - h_before

            # G
            lw = get_layer_weights(layer, d_model, mlp_type)
            if lw.W_gate is not None and lw.W_up is not None and lw.W_down is not None:
                gate_val = safe_sigmoid(lw.W_gate @ h_before)
                up_val = lw.W_up @ h_before
                G_exact = lw.W_down @ (gate_val * up_val)
                A_approx = delta - G_exact

                all_G_logit[li] = float(W_U_word @ G_exact)
                all_A_logit[li] = float(W_U_word @ A_approx)
            else:
                all_G_logit[li] = 0.0
                all_A_logit[li] = 0.0
                A_approx = delta.copy()

            # Per-head A的logit贡献
            sa = layer.self_attn
            W_v = sa.v_proj.weight.detach().float().cpu().numpy()
            W_o = sa.o_proj.weight.detach().float().cpu().numpy()

            V_all = W_v @ h_before  # [n_kv * head_dim]

            n_groups = n_q // n_kv
            head_logits = np.zeros(n_kv)

            for h in range(n_kv):
                V_h = V_all[h * head_dim: (h + 1) * head_dim]
                total_logit = 0.0
                for q in range(h * n_groups, (h + 1) * n_groups):
                    if q * head_dim < W_o.shape[1]:
                        W_o_q = W_o[:, q * head_dim: (q + 1) * head_dim]
                        direction_q = W_U_word @ W_o_q
                        total_logit += float(direction_q @ V_h)
                head_logits[h] = total_logit

            per_head_A_logit[li] = head_logits

            del lw

        # G vs 总A的对抗分析
        G_signs = np.sign(all_G_logit)
        A_signs = np.sign(all_A_logit)
        nonzero_mask = (G_signs != 0) & (A_signs != 0)

        if np.sum(nonzero_mask) > 5:
            GA_sign_corr = float(np.corrcoef(G_signs[nonzero_mask], A_signs[nonzero_mask])[0, 1])
        else:
            GA_sign_corr = 0.0

        same_sign_layers = int(np.sum(G_signs * A_signs > 0))
        opposite_sign_layers = int(np.sum(G_signs * A_signs < 0))

        # Per-head A的内部对抗分析
        # 对于每层, 不同KV头之间是否也对抗?
        head_opposition_rate = 0.0
        head_sign_corrs = []

        for li in range(n_layers):
            hl = per_head_A_logit[li]
            if len(hl) > 1:
                # 头间符号: 有多少对头是异号的?
                pos_heads = np.sum(hl > 0)
                neg_heads = np.sum(hl < 0)
                if pos_heads + neg_heads > 0:
                    head_opposition_rate += min(pos_heads, neg_heads) / max(pos_heads + neg_heads, 1)

        head_opposition_rate /= max(n_layers, 1)

        # G vs 每个KV头的A的对抗
        G_vs_head_corrs = []
        for h in range(n_kv):
            head_A_logit_all_layers = np.array([per_head_A_logit[li][h] for li in range(n_layers)])
            nonzero = (all_G_logit != 0) & (head_A_logit_all_layers != 0)
            if np.sum(nonzero) > 5:
                corr = float(np.corrcoef(np.sign(all_G_logit[nonzero]),
                                         np.sign(head_A_logit_all_layers[nonzero]))[0, 1])
                G_vs_head_corrs.append(corr)
            else:
                G_vs_head_corrs.append(0.0)

        # 总A的方差缩减分析
        var_G = float(np.var(all_G_logit))
        var_A = float(np.var(all_A_logit))
        cov_GA = float(np.cov(all_G_logit, all_A_logit)[0, 1])
        variance_reduction = (var_G + var_A + 2 * cov_GA) / max(var_G + var_A, 1e-10)

        results[word] = {
            "word_info": TEST_WORDS[word],
            "GA_sign_correlation": GA_sign_corr,
            "same_sign_layers": same_sign_layers,
            "opposite_sign_layers": opposite_sign_layers,
            "head_opposition_rate": head_opposition_rate,
            "G_vs_head_sign_correlations": G_vs_head_corrs,
            "variance_reduction_ratio": variance_reduction,
            "cumG_final": float(np.sum(all_G_logit)),
            "cumA_final": float(np.sum(all_A_logit)),
            "final_logit": float(np.sum(all_G_logit) + np.sum(all_A_logit)),
            # 末端层详情
            "last3_G_logit": [float(all_G_logit[li]) for li in range(max(0, n_layers-3), n_layers)],
            "last3_A_logit": [float(all_A_logit[li]) for li in range(max(0, n_layers-3), n_layers)],
        }

        elapsed = time.time() - t0
        print(f"  {word}: GA_sign_corr={GA_sign_corr:.3f}, "
              f"same/opposite={same_sign_layers}/{opposite_sign_layers}, "
              f"head_opposition={head_opposition_rate:.3f}, "
              f"var_red={variance_reduction:.4f}, "
              f"G_vs_head_corrs={[f'{c:.2f}' for c in G_vs_head_corrs]}, "
              f"elapsed={elapsed:.1f}s")

        del outputs
        gc.collect()

    # 跨模型比较摘要
    print(f"\n  GQA压缩效应摘要:")
    print(f"    n_q/n_kv = {n_q}/{n_kv} = {n_q/n_kv:.1f}x")
    for word in test_words:
        if word in results:
            wr = results[word]
            print(f"    {word}: GA_corr={wr['GA_sign_correlation']:.3f}, "
                  f"var_red={wr['variance_reduction_ratio']:.4f}, "
                  f"head_opp={wr['head_opposition_rate']:.3f}")

    return results


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    model_name = args.model
    print("="*70)
    print(f"Phase CLXI: 差分放大器的训练动力学推导 — {model_name}")
    print("="*70)

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)

    print(f"Model: {model_info.model_class}")
    print(f"Layers: {model_info.n_layers}, d_model: {model_info.d_model}, "
          f"d_mlp: {model_info.intermediate_size}, vocab: {model_info.vocab_size}")

    W_embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        W_U = lm_head.weight.detach().float().cpu().numpy()
    else:
        W_U = W_embed.copy()

    print(f"W_embed: {W_embed.shape}, W_U: {W_U.shape}")

    all_results = {
        "model": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "d_mlp": model_info.intermediate_size,
            "vocab_size": model_info.vocab_size,
            "mlp_type": model_info.mlp_type,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # P701: 梯度动力学
    all_results["P701"] = p701_gradient_dynamics(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P702: LayerNorm/RMSNorm作用
    all_results["P702"] = p702_layernorm_role(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # P703: GQA压缩效应
    all_results["P703"] = p703_gqa_effect_on_adversarial_balance(model, tokenizer, model_info, W_embed, W_U, device)
    gc.collect()

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"phase_clxi_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_file}")

    # 关键发现摘要
    print("\n" + "="*70)
    print("关键发现摘要")
    print("="*70)
    print(f"\n模型: {model_name}")

    if "P701" in all_results:
        print("\n--- P701: 梯度动力学 ---")
        for word in ["apple", "cat", "the"]:
            if word in all_results["P701"]:
                wr = all_results["P701"][word]
                adv = "YES" if wr.get("gradient_encourages_adversarial") else "NO"
                print(f"  {word}: sign_corr={wr.get('sign_correlation',0):.3f}, "
                      f"sens_corr={wr.get('sensitivity_sign_correlation',0):.3f}, "
                      f"adv={adv}")

    if "P702" in all_results:
        print("\n--- P702: LayerNorm/RMSNorm ---")
        print(f"  norm_type: {all_results['P702'].get('norm_type', 'unknown')}")
        for word in ["apple", "cat", "the"]:
            if word in all_results["P702"]:
                wr = all_results["P702"][word]
                adv = "YES" if wr.get("ln_encourages_adversarial") else "NO"
                print(f"  {word}: mean_GA_cos={wr.get('mean_GA_cosine',0):.3f}, "
                      f"cos_comp_corr={wr.get('cos_compression_correlation',0):.3f}, "
                      f"LN_adv={adv}")

    if "P703" in all_results:
        print("\n--- P703: GQA效应 ---")
        print(f"  n_q/n_kv: {all_results['P703'].get('n_q_heads',0)}/{all_results['P703'].get('n_kv_heads',0)}")
        for word in ["apple", "cat", "the"]:
            if word in all_results["P703"]:
                wr = all_results["P703"][word]
                print(f"  {word}: GA_corr={wr.get('GA_sign_correlation',0):.3f}, "
                      f"var_red={wr.get('variance_reduction_ratio',0):.4f}, "
                      f"head_opp={wr.get('head_opposition_rate',0):.3f}")

    release_model(model)
    print("[model_utils] GPU memory released")


if __name__ == "__main__":
    main()
