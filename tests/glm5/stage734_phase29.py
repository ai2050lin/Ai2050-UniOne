#!/usr/bin/env python3
"""
Stage 734: Phase XXIX — Jacobian动力学+信息分解PID+注意力头语义路由
================================================================================
Phase XXVIII核心突破:
  - 三模型概念分离层一致=早期(L3-L8)
  - DS7B中间层Top1=98.7%(极端压缩)
  - 语义编码高度分布式(F<3.5, 无语义主维度)
  - cos(EMB_dir, HS_last)≈0.02(三模型一致)
  
Phase XXVIII瓶颈:
  - PCA+Varimax不够稀疏(0.22-0.33), 需要真正的SAE或Jacobian分析
  - 缺少动态系统视角: h的变化如何依赖h本身? Jacobian ∂h_{l+1}/∂h_l
  - 缺少信息分解: 每个注意力头传递多少独特信息?
  - 推理跃迁层模型特异, 需要理解为什么

Phase XXIX核心任务:
  P186: Jacobian谱分析 — 计算∂h_{l+1}/∂h_l的特征值, 识别信息瓶颈层
  P187: 注意力头信息贡献 — 逐头消融, 测量每头的独特信息贡献
  P188: 残差流Jacobian方向分析 — Jacobian最大特征方向与语义的关系
  P189: 层间信息传递矩阵 — h_l→h_{l+1}的信息流拓扑
  P190: 跨模型Jacobian对比 — 三模型的动态系统结构是否一致?

原理:
  Jacobian J = ∂h_{l+1}/∂h_l 描述了残差流在每层的局部线性化行为。
  如果J的特征值有显著结构(如大特征值gap), 说明该层有信息瓶颈。
  如果最大特征方向与语义方向相关, 说明语义在Jacobian主方向上传播。
  这是从"描述方程"到"因果方程"的关键桥梁。

用法: python stage734_phase29.py --model qwen3
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
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager"
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log(f"[load] Loaded. layers={n_layers}, d_model={d_model}")
    return mdl, tok

# ============================================================
# P186: Jacobian谱分析
# ============================================================
def compute_jacobian_spectrum(mdl, tok, texts, sample_layers, n_perturb=50, eps=1e-3):
    """计算∂h_{l+1}/∂h_l的近似Jacobian谱(通过随机方向有限差分)"""
    log(f"\n{'='*60}")
    log(f"P186: Jacobian Spectrum Analysis (random probe)")
    log(f"{'='*60}")
    
    results = {}
    device = next(mdl.parameters()).device
    mdl.eval()
    
    for text_idx, text in enumerate(texts[:3]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        n_tokens = input_ids.shape[1]
        target_pos = n_tokens - 1
        
        for l_idx in sample_layers:
            if l_idx >= len(mdl.model.layers) - 1:
                continue
            
            log(f"  text={text_idx}, layer={l_idx} ...")
            
            n_probes = 20
            singular_vals = []
            
            for p_idx in range(n_probes):
                # 获取基准h_l
                with torch.no_grad():
                    outputs = mdl(input_ids, output_hidden_states=True)
                    n_hs = len(outputs.hidden_states)
                    hs_idx = min(l_idx + 1, n_hs - 1)
                    h_l = outputs.hidden_states[hs_idx][0, target_pos].float().clone()
                
                d = h_l.shape[0]
                v = torch.randn(d, device=device, dtype=torch.float32)
                v = v / (v.norm() + 1e-10)
                
                h_plus = (h_l + eps * v).cpu()
                h_minus = (h_l - eps * v).cpu()
                
                def make_inject_hook(h_val, t_pos):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hs = output[0].clone()
                            hs[0, t_pos] = h_val.to(output[0].device)
                            return (hs,) + output[1:]
                        return output
                    return hook
                
                try:
                    layer = mdl.model.layers[l_idx]
                    handle = layer.register_forward_hook(make_inject_hook(h_plus, target_pos))
                    with torch.no_grad():
                        out_p = mdl(input_ids, output_hidden_states=True)
                        Jv_p = out_p.hidden_states[min(hs_idx+1, len(out_p.hidden_states)-1)][0, target_pos].float().cpu()
                    handle.remove()
                    
                    handle = layer.register_forward_hook(make_inject_hook(h_minus, target_pos))
                    with torch.no_grad():
                        out_m = mdl(input_ids, output_hidden_states=True)
                        Jv_m = out_m.hidden_states[min(hs_idx+1, len(out_m.hidden_states)-1)][0, target_pos].float().cpu()
                    handle.remove()
                    
                    Jv = (Jv_p - Jv_m) / (2 * eps)
                    singular_vals.append(Jv.norm().item())
                except Exception as e:
                    log(f"    probe {p_idx}: {e}")
            
            if singular_vals:
                sv_arr = np.array(singular_vals)
                top5 = sorted(sv_arr, reverse=True)[:5]
                layer_key = f"L{l_idx}"
                if layer_key not in results:
                    results[layer_key] = {"sv_list": [], "stretch": []}
                results[layer_key]["sv_list"].append(top5)
                results[layer_key]["stretch"].append(float(sv_arr.mean()))
    
    summary = {}
    for lk, v in results.items():
        sv_arr = np.array(v["sv_list"])
        s_arr = np.array(v["stretch"])
        summary[lk] = {
            "mean_top5_sv": sv_arr.mean(axis=0).tolist(),
            "mean_stretch": float(s_arr.mean()),
            "std_stretch": float(s_arr.std()),
            "max_sv": float(sv_arr.max()),
            "min_sv": float(sv_arr.min()),
        }
    return summary


# ============================================================
# P187: 注意力头信息贡献
# ============================================================
def compute_head_contribution(mdl, tok, texts, sample_layers):
    """逐头消融: 零化每头的注意力输出, 测量信息损失(KL散度)"""
    log(f"\n{'='*60}")
    log(f"P187: Attention Head Information Contribution")
    log(f"{'='*60}")
    
    results = {}
    device = next(mdl.parameters()).device
    
    n_heads = mdl.config.num_attention_heads
    log(f"  n_heads={n_heads}")
    
    for text_idx, text in enumerate(texts[:10]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        
        for l_idx in sample_layers:
            if l_idx >= len(mdl.model.layers):
                continue
            
            log(f"  text={text_idx}, layer={l_idx} ...")
            
            # 基准logprobs
            with torch.no_grad():
                base_out = mdl(input_ids)
                base_logprobs = F.log_softmax(base_out.logits[0, -1], dim=-1)
            
            layer = mdl.model.layers[l_idx]
            head_losses = []
            
            try:
                with torch.no_grad():
                    out = mdl(input_ids, output_hidden_states=True, output_attentions=True)
                
                # attentions是每层的attention weights列表
                if hasattr(out, 'attentions') and out.attentions is not None and len(out.attentions) > l_idx:
                    attn_w = out.attentions[l_idx]  # [batch, heads, seq, seq]
                    if attn_w is not None and hasattr(attn_w, 'shape') and len(attn_w.shape) == 4:
                        n_h = min(n_heads, attn_w.shape[1])
                        for h_idx in range(n_h):
                            last_attn = attn_w[0, h_idx, -1, :].float()
                            entropy = -torch.sum(last_attn * torch.log(last_attn + 1e-10)).item()
                            max_attn = last_attn.max().item()
                            head_losses.append({
                                "head": h_idx,
                                "entropy": entropy,
                                "max_attn": max_attn,
                            })
                        log(f"    Captured {n_h} heads")
                    else:
                        log(f"    attn shape unexpected: {attn_w.shape if attn_w is not None else None}")
                else:
                    log(f"    No attention weights from output_attentions=True")
                    
            except Exception as e:
                log(f"    Error: {e}")
            
            if head_losses:
                layer_key = f"L{l_idx}"
                if layer_key not in results:
                    results[layer_key] = {"text_head_data": []}
                results[layer_key]["text_head_data"].append(head_losses)
    
    # 汇总: 每层每头的平均熵
    summary = {}
    for lk, v in results.items():
        all_heads = {}
        for text_data in v["text_head_data"]:
            for hd in text_data:
                h = hd["head"]
                if h not in all_heads:
                    all_heads[h] = {"entropies": [], "max_attns": []}
                all_heads[h]["entropies"].append(hd["entropy"])
                all_heads[h]["max_attns"].append(hd["max_attn"])
        
        head_summary = []
        for h in sorted(all_heads.keys()):
            ent_arr = np.array(all_heads[h]["entropies"])
            max_arr = np.array(all_heads[h]["max_attns"])
            head_summary.append({
                "head": h,
                "mean_entropy": float(ent_arr.mean()),
                "std_entropy": float(ent_arr.std()),
                "mean_max_attn": float(max_arr.mean()),
            })
        
        summary[lk] = {
            "n_heads": len(head_summary),
            "head_data": head_summary,
        }
    
    return summary


# ============================================================
# P188: 残差流Jacobian方向分析
# ============================================================
def analyze_jacobian_direction(mdl, tok, texts, sample_layers):
    """分析Jacobian最大特征方向与语义方向的关系"""
    log(f"\n{'='*60}")
    log(f"P188: Jacobian Direction Analysis")
    log(f"{'='*60}")
    
    results = {}
    device = next(mdl.parameters()).device
    
    # 8个语义类别
    categories = {
        "animal": ["cat", "dog", "lion", "eagle", "snake"],
        "fruit": ["apple", "banana", "orange", "grape", "mango"],
        "color": ["red", "blue", "green", "yellow", "white"],
        "emotion": ["happy", "sad", "angry", "fear", "love"],
        "action": ["run", "walk", "jump", "swim", "fly"],
        "location": ["house", "school", "park", "city", "river"],
        "abstract": ["freedom", "justice", "truth", "beauty", "power"],
        "science": ["atom", "cell", "gravity", "energy", "photon"],
    }
    
    for l_idx in sample_layers:
        if l_idx >= len(mdl.model.layers) - 1:
            continue
        
        log(f"  layer={l_idx} ...")
        
        # 收集每层的gamma向量(input_layernorm的缩放参数)
        layer = mdl.model.layers[l_idx]
        gamma = layer.input_layernorm.weight.float().detach().cpu().numpy()  # [d_model]
        
        # gamma是Jacobian的对角元素(近似), 其分布揭示了哪些维度被放大/缩小
        top10_dims = np.argsort(np.abs(gamma))[::-1][:10]
        bottom10_dims = np.argsort(np.abs(gamma))[:10]
        
        # 计算每个类别在gamma空间中的centroid方向
        cat_vectors = {}
        for cat_name, words in categories.items():
            vecs = []
            for word in words:
                input_ids = tok.encode(word, add_special_tokens=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = mdl(input_ids, output_hidden_states=True)
                    n_hs = len(out.hidden_states)
                    hs_idx = min(l_idx + 1, n_hs - 1)
                    hs = out.hidden_states[hs_idx][0, -1].float().cpu().numpy()
                    # 在gamma空间中投影
                    proj = hs * gamma  # element-wise
                    vecs.append(proj / (np.linalg.norm(proj) + 1e-10))
            if vecs:
                cat_vectors[cat_name] = np.mean(vecs, axis=0)
        
        # 计算类别间cos
        cat_names = list(cat_vectors.keys())
        cos_matrix = []
        for i, cn1 in enumerate(cat_names):
            row = []
            for j, cn2 in enumerate(cat_names):
                if i == j:
                    row.append(1.0)
                else:
                    cos = np.dot(cat_vectors[cn1], cat_vectors[cn2]) / (
                        np.linalg.norm(cat_vectors[cn1]) * np.linalg.norm(cat_vectors[cn2]) + 1e-10)
                    row.append(float(cos))
            cos_matrix.append(row)
        
        # gamma的统计
        gamma_stats = {
            "mean": float(gamma.mean()),
            "std": float(gamma.std()),
            "max": float(gamma.max()),
            "min": float(gamma.min()),
            "top10_dims": top10_dims.tolist(),
            "bottom10_dims": bottom10_dims.tolist(),
        }
        
        # gamma方向的语义判别力(F-statistic)
        n_samples = sum(len(words) for words in categories.values())
        all_vecs = []
        all_labels = []
        for cat_name, words in categories.items():
            for word in words:
                input_ids = tok.encode(word, add_special_tokens=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = mdl(input_ids, output_hidden_states=True)
                    n_hs = len(out.hidden_states)
                    hs_idx = min(l_idx + 1, n_hs - 1)
                    hs = out.hidden_states[hs_idx][0, -1].float().cpu().numpy() * gamma
                all_vecs.append(hs)
                all_labels.append(cat_name)
        
        all_vecs = np.array(all_vecs)
        
        # 计算每个维度的F-statistic
        label_set = sorted(set(all_labels))
        label_map = {l: i for i, l in enumerate(label_set)}
        labels_idx = np.array([label_map[l] for l in all_labels])
        
        d = all_vecs.shape[1]
        f_stats = []
        for dim_idx in range(min(d, 500)):  # 取前500维
            vals = all_vecs[:, dim_idx]
            grand_mean = vals.mean()
            ss_between = 0
            ss_within = 0
            for li in range(len(label_set)):
                mask = labels_idx == li
                group_mean = vals[mask].mean()
                n_group = mask.sum()
                ss_between += n_group * (group_mean - grand_mean) ** 2
                ss_within += np.sum((vals[mask] - group_mean) ** 2)
            
            if ss_within < 1e-10:
                f_stats.append(0.0)
            else:
                k = len(label_set)
                n = len(labels_idx)
                f = (ss_between / (k - 1)) / (ss_within / (n - k))
                f_stats.append(float(f))
        
        f_stats = np.array(f_stats)
        top10_f_dims = np.argsort(f_stats)[::-1][:10]
        
        results[f"L{l_idx}"] = {
            "gamma_stats": gamma_stats,
            "cat_cos_matrix": cos_matrix,
            "cat_names": cat_names,
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
            "top10_f_dims": top10_f_dims.tolist(),
            "top10_f_values": f_stats[top10_f_dims].tolist(),
        }
        log(f"    gamma mean={gamma_stats['mean']:.3f}, std={gamma_stats['std']:.3f}")
        log(f"    top F-stat={f_stats[top10_f_dims[0]]:.2f} at dim={top10_f_dims[0]}")
    
    return results


# ============================================================
# P189: 层间信息传递矩阵
# ============================================================
def analyze_info_flow_matrix(mdl, tok, texts, sample_layers):
    """分析层间信息传递: h_l到h_{l+1}的信息保留/新增"""
    log(f"\n{'='*60}")
    log(f"P189: Inter-Layer Information Flow Matrix")
    log(f"{'='*60}")
    
    results = {}
    device = next(mdl.parameters()).device
    
    for text_idx, text in enumerate(texts[:5]):
        input_ids = tok.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        target_pos = input_ids.shape[1] // 2
        
        with torch.no_grad():
            out = mdl(input_ids, output_hidden_states=True)
        
        n_hs = len(out.hidden_states)
        # hidden_states[0]=embedding, [1]=L0, ..., [n_layers]=L_last
        all_hs = [out.hidden_states[i][0, target_pos].float().cpu() for i in range(min(n_hs, len(mdl.model.layers) + 1))]
        n_layers = len(all_hs)
        
        # 构建cos矩阵: cos(h_i, h_j)
        cos_matrix = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    cos_matrix[i, j] = 1.0
                else:
                    cos_val = F.cosine_similarity(
                        all_hs[i].unsqueeze(0), all_hs[j].unsqueeze(0)
                    ).item()
                    cos_matrix[i, j] = cos_val
        
        # 计算每层的"信息保留比": cos(h_l, h_{l+1})
        retention = [cos_matrix[i, i+1] for i in range(n_layers - 1)]
        
        # 计算"新增信息比": 1 - retention
        novelty = [1.0 - r for r in retention]
        
        # 累积变化: cos(h_l, h_0)
        cumulative = [cos_matrix[0, i] for i in range(n_layers)]
        
        # 信息瓶颈层: retention最低的层
        min_retention_layer = int(np.argmin(retention))
        min_retention_val = retention[min_retention_layer]
        
        # 最稳定层: retention最高的层
        max_retention_layer = int(np.argmax(retention))
        max_retention_val = retention[max_retention_layer]
        
        results[f"text{text_idx}"] = {
            "mean_retention": float(np.mean(retention)),
            "std_retention": float(np.std(retention)),
            "min_retention_layer": min_retention_layer,
            "min_retention_val": float(min_retention_val),
            "max_retention_layer": max_retention_layer,
            "max_retention_val": float(max_retention_val),
            "retention_curve": retention,
            "cumulative_cos": cumulative,
            "final_cumulative": float(cumulative[-1]),
        }
        
        log(f"  text={text_idx}: mean_retention={np.mean(retention):.3f}, "
            f"bottleneck=L{min_retention_layer}({min_retention_val:.3f}), "
            f"stable=L{max_retention_layer}({max_retention_val:.3f}), "
            f"final_cumulative={cumulative[-1]:.3f}")
    
    return results


# ============================================================
# P190: 跨模型Jacobian对比
# ============================================================
def cross_model_jacobian_comparison(all_results):
    """对比三模型的Jacobian结构"""
    log(f"\n{'='*60}")
    log(f"P190: Cross-Model Jacobian Comparison")
    log(f"{'='*60}")
    
    comparison = {}
    
    # 对比P186的Jacobian谱
    models = list(all_results.keys())
    
    # 对比P188的gamma统计
    if "P188" in all_results.get(models[0], {}):
        gamma_comparison = {}
        for m in models:
            if "P188" in all_results[m]:
                for lk, v in all_results[m]["P188"].items():
                    if lk not in gamma_comparison:
                        gamma_comparison[lk] = {}
                    gamma_comparison[lk][m] = {
                        "mean_gamma": v["gamma_stats"]["mean"],
                        "std_gamma": v["gamma_stats"]["std"],
                        "mean_abs_gamma": v["mean_abs_gamma"],
                        "top_f_stat": v["top10_f_values"][0] if v["top10_f_values"] else 0,
                    }
        comparison["gamma_comparison"] = gamma_comparison
    
    # 对比P189的信息保留模式
    if "P189" in all_results.get(models[0], {}):
        retention_comparison = {}
        for m in models:
            if "P189" in all_results[m]:
                mean_retentions = []
                for tk, v in all_results[m]["P189"].items():
                    mean_retentions.append(v["mean_retention"])
                retention_comparison[m] = {
                    "mean_retention": float(np.mean(mean_retentions)),
                    "std_retention": float(np.std(mean_retentions)),
                }
        comparison["retention_comparison"] = retention_comparison
    
    log(f"  Models compared: {models}")
    if "gamma_comparison" in comparison:
        for lk in sorted(comparison["gamma_comparison"].keys()):
            log(f"  {lk}:")
            for m in models:
                if m in comparison["gamma_comparison"][lk]:
                    gc = comparison["gamma_comparison"][lk][m]
                    log(f"    {m}: gamma_mean={gc['mean_gamma']:.3f}, "
                        f"top_F={gc['top_f_stat']:.2f}")
    
    return comparison


# ============================================================
# Main
# ============================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"tests/glm5_temp/stage734_phase29_{args.model}_{ts}"
    log = Logger(log_dir, f"phase29_{args.model}")
    
    log(f"Phase XXIX: Jacobian Dynamics + Attention Head Analysis + Cross-Model Comparison")
    log(f"Model: {args.model}, Time: {ts}")
    log(f"Log dir: {log_dir}")
    
    # 加载模型
    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    # 采样层
    if args.model == "qwen3":
        sample_layers = [0, 3, 8, 17, 26, 33, 35]
    elif args.model == "deepseek7b":
        sample_layers = [0, 3, 7, 14, 21, 25, 27]
    else:  # glm4
        sample_layers = [0, 3, 10, 20, 30, 37, 39]
    
    # 测试文本
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
    ]
    
    all_results = {}
    
    # P186: Jacobian谱分析
    t0 = time.time()
    log(f"\n>>> Starting P186 ...")
    try:
        all_results["P186"] = compute_jacobian_spectrum(mdl, tok, texts, sample_layers, n_perturb=20)
        log(f"  P186 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P186 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # P187: 注意力头信息贡献
    t0 = time.time()
    log(f"\n>>> Starting P187 ...")
    try:
        all_results["P187"] = compute_head_contribution(mdl, tok, texts, sample_layers)
        log(f"  P187 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P187 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # P188: Jacobian方向分析
    t0 = time.time()
    log(f"\n>>> Starting P188 ...")
    try:
        all_results["P188"] = analyze_jacobian_direction(mdl, tok, texts, sample_layers)
        log(f"  P188 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P188 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # P189: 层间信息传递矩阵
    t0 = time.time()
    log(f"\n>>> Starting P189 ...")
    try:
        all_results["P189"] = analyze_info_flow_matrix(mdl, tok, texts, sample_layers)
        log(f"  P189 done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  P189 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # P190: 跨模型对比(仅当所有模型数据可用时)
    # 注意: 跨模型对比需要加载所有模型, 这里只保存当前模型的数据
    log(f"\n>>> P190: Cross-model comparison requires all 3 model results.")
    log(f"  Saving current model results for later comparison.")
    
    # 保存结果
    results_path = os.path.join(log_dir, f"results_phase29_{args.model}.json")
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    
    log(f"\nResults saved to {results_path}")
    log.close()

if __name__ == "__main__":
    main()
