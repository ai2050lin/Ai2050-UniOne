#!/usr/bin/env python3
"""
Stage 735: Phase XXX — 注意力头因果消融+Jacobian跨层传播+SAE训练+头功能聚类
================================================================================
Phase XXIX瓶颈:
  - P186 DS7B/GLM4 Jacobian为0(hook注入不兼容)
  - P187只有熵分布, 没有因果消融(零化头看对输出的影响)
  - P188 F-stat仍低, gamma空间未改善语义判别力
  - 缺少真正的SAE, PCA+Varimax稀疏度0.25不够

Phase XXX核心任务:
  P191: 注意力头因果消融 — 零化每个头, 测量KL散度变化(因果证据!)
  P192: Jacobian跨层累积传播 — 修复DS7B/GLM4, 计算J_cumul=∏J_l
  P193: 简化SAE训练 — 在关键层训练稀疏自编码器
  P194: 注意力头功能聚类 — 按注意力模式+因果影响力分类头

原理:
  P191因果消融: 零化头h的输出后, 如果KL(output||baseline)大幅增加,
    说明头h携带了关键信息。这是因果推断的金标准。
  
  P192 Jacobian: 用梯度方法计算J=∂h_{l+1}/∂h_l, 
    然后J_cumul=J_{N-1}·...·J_1·J_0描述整个网络的输入-输出映射。
    
  P193 SAE: h ≈ W_dec · f, 其中f=W_enc·h且||f||_0 << d_model。
    稀疏激活的f(i)对应可解释的特征。
    
  P194 头聚类: 用注意力模式的相似度(cos)聚类,
    加上因果消融的KL影响, 得到头的功能分类。

用法: python stage735_phase30.py --model qwen3
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try: print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

log = None

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[model_name]
    log(f"[load] Loading {model_name} from {p.name} ...")
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager"
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    n_heads = mdl.config.num_attention_heads
    log(f"[load] Loaded. layers={n_layers}, d_model={d_model}, n_heads={n_heads}")
    return mdl, tok


# ============================================================
# P191: 注意力头因果消融
# ============================================================
def head_causal_ablation(mdl, tok, texts, sample_layers, n_texts=15):
    """
    因果消融: 零化每个注意力头的输出, 测量对next-token预测的KL散度影响。
    
    原理: 
      1. 正常前向传播, 得到基准logits (baseline)
      2. 在特定层的特定头, 将attn_output的该头部分置零
      3. 重新前向传播, 得到ablated logits
      4. KL(ablated || baseline) 衡量该头的信息贡献
      
    KL越大 → 该头对预测越关键(因果影响越大)
    """
    log(f"\n{'='*60}")
    log(f"P191: Attention Head Causal Ablation ({n_texts} texts)")
    log(f"{'='*60}")
    
    device = next(mdl.parameters()).device
    n_heads = mdl.config.num_attention_heads
    head_dim = mdl.config.hidden_size // n_heads
    
    results = {}
    
    for text_idx, text in enumerate(texts[:n_texts]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 5:
            continue
        
        target_pos = seq_len - 2  # 倒数第2个token位置预测下一个
        next_token_id = input_ids[0, target_pos + 1].item()
        
        # 基准: 正常前向传播
        with torch.no_grad():
            baseline_out = mdl(input_ids, output_hidden_states=True)
            baseline_logits = baseline_out.logits[0, target_pos].float()  # [vocab]
            baseline_probs = F.softmax(baseline_logits, dim=-1)
        
        for l_idx in sample_layers:
            layer_key = f"L{l_idx}"
            if layer_key not in results:
                results[layer_key] = {
                    "head_kl": [[] for _ in range(n_heads)],
                    "head_top1_change": [[] for _ in range(n_heads)],
                    "head_correct_change": [[] for _ in range(n_heads)],
                }
            
            for h_idx in range(n_heads):
                # 在该层的该头做消融
                def make_ablation_hook(head_idx, hd):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            attn_out = output[0].clone()
                            attn_out[:, :, head_idx*hd:(head_idx+1)*hd] = 0.0
                            return (attn_out,) + output[1:]
                        return output
                    return hook
                
                try:
                    layer = mdl.model.layers[l_idx]
                    handle = layer.self_attn.register_forward_hook(
                        make_ablation_hook(h_idx, head_dim)
                    )
                    
                    with torch.no_grad():
                        ablated_out = mdl(input_ids)
                        ablated_logits = ablated_out.logits[0, target_pos].float()
                        ablated_probs = F.softmax(ablated_logits, dim=-1)
                    
                    handle.remove()
                    
                    # KL(ablated || baseline)
                    kl = F.kl_div(
                        ablated_probs.log(), baseline_probs, reduction='batchmean'
                    ).item()
                    
                    # Top1 token变化
                    baseline_top1 = baseline_logits.argmax().item()
                    ablated_top1 = ablated_logits.argmax().item()
                    top1_changed = 1.0 if baseline_top1 != ablated_top1 else 0.0
                    
                    # 正确token的概率变化
                    baseline_correct = baseline_probs[next_token_id].item()
                    ablated_correct = ablated_probs[next_token_id].item()
                    correct_change = baseline_correct - ablated_correct
                    
                    results[layer_key]["head_kl"][h_idx].append(kl)
                    results[layer_key]["head_top1_change"][h_idx].append(top1_changed)
                    results[layer_key]["head_correct_change"][h_idx].append(correct_change)
                    
                except Exception as e:
                    log(f"    ERROR text={text_idx} L{l_idx} H{h_idx}: {e}")
            
            log(f"  text={text_idx}, layer=L{l_idx}: done {n_heads} heads")
    
    # 汇总: 每层每头的平均KL
    summary = {}
    for lk, v in results.items():
        kl_matrix = []  # [n_heads] x mean_kl
        top1_matrix = []
        correct_matrix = []
        for h_idx in range(n_heads):
            if v["head_kl"][h_idx]:
                kl_matrix.append(float(np.mean(v["head_kl"][h_idx])))
                top1_matrix.append(float(np.mean(v["head_top1_change"][h_idx])))
                correct_matrix.append(float(np.mean(v["head_correct_change"][h_idx])))
            else:
                kl_matrix.append(0.0)
                top1_matrix.append(0.0)
                correct_matrix.append(0.0)
        
        kl_arr = np.array(kl_matrix)
        summary[lk] = {
            "mean_kl_per_head": kl_matrix,
            "mean_kl": float(kl_arr.mean()),
            "std_kl": float(kl_arr.std()),
            "max_kl_head": int(kl_arr.argmax()),
            "max_kl_val": float(kl_arr.max()),
            "min_kl_head": int(kl_arr.argmin()),
            "min_kl_val": float(kl_arr.min()),
            "top_kl_heads": [int(x) for x in np.argsort(kl_arr)[::-1][:5]],
            "top_kl_values": np.sort(kl_arr)[::-1][:5].tolist(),
            "mean_top1_change": float(np.mean(top1_matrix)),
            "mean_correct_change": float(np.mean(correct_matrix)),
            "kl_top5_heads": [int(x) for x in np.argsort(kl_arr)[::-1][:5]],
        }
        log(f"  {lk}: mean_KL={kl_arr.mean():.4f}, std={kl_arr.std():.4f}, "
            f"max_head=H{kl_arr.argmax()}({kl_arr.max():.4f})")
    
    return summary


# ============================================================
# P192: Jacobian跨层累积传播(修复DS7B/GLM4)
# ============================================================
def compute_cumulative_jacobian(mdl, tok, texts, sample_layers, n_texts=10):
    """
    用embedding扰动法计算层间信息灵敏度(避免hook兼容性问题)。
    
    原理:
      扰动input embedding: E' = E + ε*v (随机方向v)
      然后比较各层hidden_states的变化:
        sensitivity_l = ||h_l(E') - h_l(E)|| / (ε * ||v||)
      这给出了信息从输入传播到每层的"有效拉伸因子"
      
      cumulative_stretch = sensitivity_L / sensitivity_0 
        衡量从输入到层L的累积放大
      
    这个方法不依赖hook, 完全兼容所有模型架构。
    """
    log(f"\n{'='*60}")
    log(f"P192: Embedding Perturbation Jacobian ({n_texts} texts)")
    log(f"{'='*60}")
    
    device = next(mdl.parameters()).device
    mdl.eval()
    
    results = {}
    n_probes = 15
    eps = 1e-4
    
    for text_idx, text in enumerate(texts[:n_texts]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1
        
        # 获取embedding输出
        with torch.no_grad():
            # 方法1: 直接获取embedding
            emb = mdl.get_input_embeddings()(input_ids)  # [1, seq, d_model]
            n_layers = len(mdl.model.layers)
            
            # 基准前向传播
            out_base = mdl(input_ids, output_hidden_states=True)
        
        for p_idx in range(n_probes):
            np.random.seed(42 + text_idx * 100 + p_idx)
            # 随机方向, 在embedding空间中
            v = np.random.randn(seq_len, emb.shape[-1]).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-10)
            
            perturbed_emb = (emb + eps * torch.from_numpy(v).to(device)).to(emb.dtype)
            
            # 方法: 用inputs_embeds替代input_ids
            with torch.no_grad():
                out_perturb = mdl(inputs_embeds=perturbed_emb, output_hidden_states=True)
            
            # 比较每层hidden_states的变化
            n_hs = len(out_base.hidden_states)
            for l_idx in range(n_layers):
                layer_key = f"L{l_idx}"
                if layer_key not in results:
                    results[layer_key] = {"sensitivity": []}
                
                hs_idx = min(l_idx + 1, n_hs - 1)
                h_base = out_base.hidden_states[hs_idx][0, target_pos].float()
                h_perturb = out_perturb.hidden_states[hs_idx][0, target_pos].float()
                
                diff = (h_perturb - h_base) / eps
                sensitivity = diff.norm().item()
                results[layer_key]["sensitivity"].append(sensitivity)
        
        log(f"  text={text_idx}: done {n_probes} probes")
    
    # 汇总
    summary = {}
    for lk, v in results.items():
        sens_arr = np.array(v["sensitivity"])
        summary[lk] = {
            "mean_sensitivity": float(sens_arr.mean()),
            "std_sensitivity": float(sens_arr.std()),
            "max_sensitivity": float(sens_arr.max()),
            "min_sensitivity": float(sens_arr.min()),
            "n_samples": len(sens_arr),
        }
        log(f"  {lk}: sensitivity={sens_arr.mean():.2f}±{sens_arr.std():.2f}")
    
    return summary


# ============================================================
# P193: 简化SAE训练
# ============================================================
def train_simple_sae(mdl, tok, sample_layers, n_samples=500, dict_size=512, sparsity_coeff=1e-3, n_epochs=200):
    """
    训练简单稀疏自编码器(SAE)来提取可解释特征。
    
    原理:
      SAE结构: h -> W_enc -> ReLU -> W_dec -> h_hat
      损失: L = ||h - h_hat||^2 + λ * ||f||_1
      其中f = ReLU(W_enc · h), ||f||_0 << d_model
      
      稀疏激活的f(i)对应一个"特征":
        - 如果f(i)>0, 特征i被激活
        - W_dec[:,i]是特征i在hidden state空间中的方向
        
      稀疏的f使得每个特征可以被单独解读。
    """
    log(f"\n{'='*60}")
    log(f"P193: Simple SAE Training (n_samples={n_samples}, dict_size={dict_size}, epochs={n_epochs})")
    log(f"{'='*60}")
    
    device = next(mdl.parameters()).device
    d_model = mdl.config.hidden_size
    
    # 收集训练数据: hidden states
    sae_texts = [
        "The cat sat on the mat and looked around.",
        "A beautiful sunset painted the sky orange.",
        "Scientists discovered a new form of energy.",
        "She felt happy when she saw the old house.",
        "The river flows through the deep valley.",
        "Mathematical proof requires logical reasoning.",
        "The dog chased the ball across the park.",
        "Freedom is the most important human value.",
        "Red and blue are primary colors in art.",
        "The eagle soared high above the mountain.",
        "Water boils at one hundred degrees Celsius.",
        "The scientist conducted an important experiment.",
        "Music has the power to move people deeply.",
        "The ancient temple stood on the hilltop.",
        "Photosynthesis converts sunlight into energy.",
        "The ocean covers seventy percent of Earth.",
        "Language is a uniquely human capability.",
        "Gravity pulls objects toward the center of Earth.",
        "The novel explores themes of love and loss.",
        "Democracy requires active citizen participation.",
        "The brain contains billions of neurons.",
        "Light travels at approximately three hundred thousand kilometers per second.",
        "The artist painted a vivid landscape.",
        "Evolution drives the diversity of life.",
        "The economy experienced significant growth last year.",
        "Climate change threatens coastal communities.",
        "The philosopher questioned the nature of reality.",
        "Technology transforms how we communicate.",
        "The forest ecosystem supports diverse wildlife.",
        "Education is the foundation of progress.",
    ] * 20  # 重复以获取足够样本
    
    results = {}
    
    for l_idx in sample_layers:
        layer_key = f"L{l_idx}"
        log(f"\n  Training SAE for {layer_key} ...")
        
        # 收集hidden states
        all_hs = []
        count = 0
        for text in sae_texts:
            if count >= n_samples:
                break
            input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")
            if input_ids.shape[1] < 3:
                continue
            input_ids = input_ids.to(device)
            with torch.no_grad():
                out = mdl(input_ids, output_hidden_states=True)
                n_hs = len(out.hidden_states)
                hs_idx = min(l_idx + 1, n_hs - 1)
                # 取每个token的hidden state
                for t_pos in range(1, input_ids.shape[1] - 1):
                    if count >= n_samples:
                        break
                    hs = out.hidden_states[hs_idx][0, t_pos].float().cpu()
                    all_hs.append(hs)
                    count += 1
        
        all_hs = torch.stack(all_hs)  # [n_samples, d_model]
        log(f"    Collected {all_hs.shape[0]} hidden states")
        
        # 标准化
        mean_hs = all_hs.mean(dim=0)
        std_hs = all_hs.std(dim=0) + 1e-8
        hs_normalized = (all_hs - mean_hs) / std_hs
        
        # 初始化SAE参数
        torch.manual_seed(42)
        W_enc = nn.Parameter(torch.randn(dict_size, d_model) * 0.01)
        W_dec = nn.Parameter(torch.randn(d_model, dict_size) * 0.01)
        b_enc = nn.Parameter(torch.zeros(dict_size))
        
        optimizer = torch.optim.Adam([W_enc, W_dec, b_enc], lr=3e-4)
        
        # 训练
        losses = []
        sparsities = []
        batch_size = 64
        
        for epoch in range(n_epochs):
            perm = torch.randperm(hs_normalized.shape[0])
            total_loss = 0
            total_sparse = 0
            n_batches = 0
            
            for i in range(0, hs_normalized.shape[0], batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = hs_normalized[batch_idx]  # [B, d_model]
                
                # 前向: encode -> relu -> decode
                f = F.relu(batch @ W_enc.T + b_enc)  # [B, dict_size]
                h_hat = f @ W_dec.T  # [B, d_model]
                
                # 重建损失
                recon_loss = F.mse_loss(h_hat, batch)
                
                # 稀疏损失 (L1)
                sparse_loss = sparsity_coeff * f.abs().mean()
                
                loss = recon_loss + sparse_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_sparse += (f > 0.01).float().mean().item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_sparse = total_sparse / n_batches
            losses.append(avg_loss)
            sparsities.append(avg_sparse)
            
            if (epoch + 1) % 50 == 0:
                log(f"    epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}, sparsity={avg_sparse:.4f}")
        
        # 评估
        with torch.no_grad():
            f_all = F.relu(hs_normalized @ W_enc.T + b_enc)  # [n_samples, dict_size]
            h_hat_all = f_all @ W_dec.T
            
            # 稀疏度统计
            feature_freq = (f_all > 0.01).float().mean(dim=0)  # [dict_size]
            alive_features = (feature_freq > 0.01).sum().item()
            dead_features = dict_size - alive_features
            
            # 特征重要性(按L2 norm of decoder column)
            dec_norms = W_dec.T.norm(dim=1)  # [dict_size]
            top_features = torch.argsort(dec_norms, descending=True)[:20]
            
            # 重建质量
            recon_cos = []
            for i in range(0, min(100, hs_normalized.shape[0])):
                cos = F.cosine_similarity(
                    hs_normalized[i:i+1], h_hat_all[i:i+1]
                ).item()
                recon_cos.append(cos)
            
            # 每个样本激活的特征数
            acts_per_sample = (f_all > 0.01).sum(dim=1).float()
        
        results[layer_key] = {
            "final_loss": float(losses[-1]),
            "final_sparsity": float(sparsities[-1]),
            "alive_features": int(alive_features),
            "dead_features": int(dead_features),
            "alive_ratio": float(alive_features / dict_size),
            "mean_recon_cos": float(np.mean(recon_cos)),
            "mean_acts_per_sample": float(acts_per_sample.mean()),
            "std_acts_per_sample": float(acts_per_sample.std()),
            "top10_feature_freq": feature_freq[top_features[:10]].tolist(),
            "top10_feature_norms": dec_norms[top_features[:10]].tolist(),
            "top10_feature_indices": top_features[:10].tolist(),
            "feature_freq_percentiles": {
                "p25": float(feature_freq.quantile(0.25)),
                "p50": float(feature_freq.quantile(0.50)),
                "p75": float(feature_freq.quantile(0.75)),
                "p95": float(feature_freq.quantile(0.95)),
                "p99": float(feature_freq.quantile(0.99)),
            },
            "loss_curve": losses[::20],  # 每20个epoch记录一次
            "sparsity_curve": sparsities[::20],
        }
        
        log(f"    {layer_key}: alive={alive_features}/{dict_size}({alive_features/dict_size*100:.1f}%), "
            f"recon_cos={np.mean(recon_cos):.4f}, acts/sample={acts_per_sample.mean():.1f}±{acts_per_sample.std():.1f}")
    
    return results


# ============================================================
# P194: 注意力头功能聚类
# ============================================================
def classify_head_functions(mdl, tok, texts, sample_layers, n_texts=15):
    """
    基于注意力模式和因果影响力对注意力头进行功能分类。
    
    原理:
      1. 计算每个头的注意力模式特征:
         - entropy: 注意力分布的熵(低=专注, 高=均匀)
         - max_attn: 最大注意力权重(高=极端专注)
         - prev_token_attn: 对前一个token的注意力(复制头)
         - first_token_attn: 对第一个token的注意力(检索头)
      2. 结合P191的因果消融KL值
      3. 用这些特征聚类, 识别不同功能类型的头
    
    头功能分类:
      - "copy" 头: 高prev_token_attn + 中KL → 复制模式
      - "retrieval" 头: 高first_token_attn + 高KL → 检索模式  
      - "specialist" 头: 低entropy + 高KL → 专门处理特定模式
      - "generalist" 头: 高entropy + 低KL → 全局整合
      - "dead" 头: 低KL + 低entropy → 可能未被使用
    """
    log(f"\n{'='*60}")
    log(f"P194: Attention Head Function Classification ({n_texts} texts)")
    log(f"{'='*60}")
    
    device = next(mdl.parameters()).device
    n_heads = mdl.config.num_attention_heads
    
    results = {}
    
    for text_idx, text in enumerate(texts[:n_texts]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 5:
            continue
        
        for l_idx in sample_layers:
            layer_key = f"L{l_idx}"
            if layer_key not in results:
                results[layer_key] = {
                    "heads": {h: {"entropy_list": [], "max_attn_list": [],
                                  "prev_attn_list": [], "first_attn_list": []}
                             for h in range(n_heads)}
                }
            
            with torch.no_grad():
                out = mdl(input_ids, output_attentions=True)
            
            if out.attentions is None or l_idx >= len(out.attentions):
                continue
            
            attn = out.attentions[l_idx][0]  # [n_heads, seq, seq]
            
            for h_idx in range(min(n_heads, attn.shape[0])):
                # 对最后一个token的注意力分布
                last_attn = attn[h_idx, -1, :].float()  # [seq]
                
                entropy = -torch.sum(last_attn * torch.log(last_attn + 1e-10)).item()
                max_attn = last_attn.max().item()
                prev_attn = last_attn[-2].item() if seq_len > 1 else 0.0
                first_attn = last_attn[0].item()
                
                results[layer_key]["heads"][h_idx]["entropy_list"].append(entropy)
                results[layer_key]["heads"][h_idx]["max_attn_list"].append(max_attn)
                results[layer_key]["heads"][h_idx]["prev_attn_list"].append(prev_attn)
                results[layer_key]["heads"][h_idx]["first_attn_list"].append(first_attn)
    
    # 汇总 + 分类
    summary = {}
    for lk, v in results.items():
        head_summary = {}
        head_features = []  # 用于聚类: [entropy, max_attn, prev_attn, first_attn]
        
        for h_idx in range(n_heads):
            hd = v["heads"][h_idx]
            if hd["entropy_list"]:
                mean_entropy = float(np.mean(hd["entropy_list"]))
                mean_max = float(np.mean(hd["max_attn_list"]))
                mean_prev = float(np.mean(hd["prev_attn_list"]))
                mean_first = float(np.mean(hd["first_attn_list"]))
                
                head_summary[f"H{h_idx}"] = {
                    "mean_entropy": mean_entropy,
                    "mean_max_attn": mean_max,
                    "mean_prev_token_attn": mean_prev,
                    "mean_first_token_attn": mean_first,
                }
                head_features.append([mean_entropy, mean_max, mean_prev, mean_first])
            else:
                head_features.append([0, 0, 0, 0])
        
        # 简单分类(基于阈值)
        head_types = {}
        for h_idx, feat in enumerate(head_features):
            ent, max_a, prev_a, first_a = feat
            
            if max_a > 0.8:
                if prev_a > 0.5:
                    head_type = "copy"
                elif first_a > 0.5:
                    head_type = "retrieval"
                elif ent < 0.5:
                    head_type = "specialist"
                else:
                    head_type = "specialist"
            elif ent > 2.5:
                head_type = "generalist"
            elif ent < 0.3:
                head_type = "dead"
            else:
                head_type = "mixed"
            
            head_types[f"H{h_idx}"] = head_type
        
        # 统计各类型数量
        type_counts = defaultdict(int)
        for ht in head_types.values():
            type_counts[ht] += 1
        
        summary[lk] = {
            "head_details": head_summary,
            "head_types": head_types,
            "type_counts": dict(type_counts),
        }
        log(f"  {lk}: {dict(type_counts)}")
    
    return summary


# ============================================================
# P195: 综合跨模型分析
# ============================================================
def comprehensive_cross_analysis(all_model_results):
    """综合三模型的P191-P194结果进行跨模型对比"""
    log(f"\n{'='*60}")
    log(f"P195: Comprehensive Cross-Model Analysis")
    log(f"{'='*60}")
    
    comparison = {}
    
    # P191: 因果消融KL对比
    kl_comparison = {}
    for model_name, model_res in all_model_results.items():
        if "P191" in model_res:
            kl_comparison[model_name] = {}
            for lk, v in model_res["P191"].items():
                kl_comparison[model_name][lk] = {
                    "mean_kl": v["mean_kl"],
                    "max_kl": v["max_kl_val"],
                    "max_kl_head": v["max_kl_head"],
                }
                log(f"  {model_name} {lk}: KL={v['mean_kl']:.4f}, max=H{v['max_kl_head']}({v['max_kl_val']:.4f})")
    comparison["P191_kl"] = kl_comparison
    
    # P192: Jacobian拉伸对比
    jacobian_comparison = {}
    for model_name, model_res in all_model_results.items():
        if "P192" in model_res:
            jacobian_comparison[model_name] = {}
            for lk, v in model_res["P192"].items():
                jacobian_comparison[model_name][lk] = {
                    "mean_stretch": v["mean_stretch"],
                    "max_stretch": v["max_stretch"],
                }
                log(f"  {model_name} {lk}: stretch={v['mean_stretch']:.2f}, max={v['max_stretch']:.2f}")
    comparison["P192_jacobian"] = jacobian_comparison
    
    # P193: SAE对比
    sae_comparison = {}
    for model_name, model_res in all_model_results.items():
        if "P193" in model_res:
            sae_comparison[model_name] = {}
            for lk, v in model_res["P193"].items():
                sae_comparison[model_name][lk] = {
                    "alive_ratio": v["alive_ratio"],
                    "mean_recon_cos": v["mean_recon_cos"],
                    "mean_acts_per_sample": v["mean_acts_per_sample"],
                }
                log(f"  {model_name} {lk}: SAE alive={v['alive_ratio']*100:.1f}%, "
                    f"recon_cos={v['mean_recon_cos']:.4f}, "
                    f"acts={v['mean_acts_per_sample']:.1f}")
    comparison["P193_sae"] = sae_comparison
    
    # P194: 头功能分布对比
    head_type_comparison = {}
    for model_name, model_res in all_model_results.items():
        if "P194" in model_res:
            head_type_comparison[model_name] = {}
            for lk, v in model_res["P194"].items():
                head_type_comparison[model_name][lk] = v["type_counts"]
    comparison["P194_head_types"] = head_type_comparison
    
    return comparison


# ============================================================
# Main
# ============================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "deepseek7b", "glm4"])
    parser.add_argument("--skip_p193", action="store_true", help="Skip SAE training (slow)")
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"tests/glm5_temp/stage735_phase30_{args.model}_{ts}"
    log = Logger(log_dir, f"phase30_{args.model}")
    
    log(f"Phase XXX: Head Ablation + Jacobian + SAE + Head Classification")
    log(f"Model: {args.model}, Time: {ts}")
    log(f"Log dir: {log_dir}")
    
    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    n_heads = mdl.config.num_attention_heads
    
    if args.model == "qwen3":
        sample_layers = [0, 3, 8, 17, 26, 33]
    elif args.model == "deepseek7b":
        sample_layers = [0, 3, 7, 14, 21, 25]
    else:
        sample_layers = [0, 3, 10, 20, 30, 37]
    
    texts = [
        "The cat sat on the mat and looked around.",
        "A beautiful sunset painted the sky orange.",
        "Scientists discovered a new form of energy.",
        "She felt happy when she saw the old house.",
        "The river flows through the deep valley.",
        "Mathematical proof requires logical reasoning.",
        "The dog chased the ball across the park.",
        "Freedom is the most important human value.",
        "Red and blue are primary colors in art.",
        "The eagle soared high above the mountain.",
        "Water boils at one hundred degrees Celsius.",
        "The scientist conducted an important experiment.",
        "Music has the power to move people deeply.",
        "The ancient temple stood on the hilltop.",
        "Photosynthesis converts sunlight into energy.",
        "The ocean covers seventy percent of Earth.",
        "Language is a uniquely human capability.",
        "Gravity pulls objects toward the center of Earth.",
        "The novel explores themes of love and loss.",
        "Democracy requires active citizen participation.",
    ]
    
    all_results = {}
    
    # P191: 注意力头因果消融
    t0 = time.time()
    log(f"\n>>> Starting P191: Head Causal Ablation ...")
    try:
        all_results["P191"] = head_causal_ablation(mdl, tok, texts, sample_layers, n_texts=15)
        log(f"  P191 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P191 ERROR: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect(); torch.cuda.empty_cache()
    
    # P192: Jacobian跨层累积传播
    t0 = time.time()
    log(f"\n>>> Starting P192: Cumulative Jacobian ...")
    try:
        all_results["P192"] = compute_cumulative_jacobian(mdl, tok, texts, sample_layers, n_texts=10)
        log(f"  P192 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P192 ERROR: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect(); torch.cuda.empty_cache()
    
    # P193: SAE训练
    if not args.skip_p193:
        t0 = time.time()
        log(f"\n>>> Starting P193: SAE Training ...")
        try:
            all_results["P193"] = train_simple_sae(mdl, tok, sample_layers, 
                                                     n_samples=500, dict_size=512, n_epochs=200)
            log(f"  P193 done in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"  P193 ERROR: {e}")
            import traceback; traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
    
    # P194: 注意力头功能分类
    t0 = time.time()
    log(f"\n>>> Starting P194: Head Function Classification ...")
    try:
        all_results["P194"] = classify_head_functions(mdl, tok, texts, sample_layers, n_texts=15)
        log(f"  P194 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P194 ERROR: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    results_path = os.path.join(log_dir, f"results_phase30_{args.model}.json")
    
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.int32, np.int64)): return int(obj)
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    
    log(f"\nResults saved to {results_path}")
    log.close()


if __name__ == "__main__":
    main()
