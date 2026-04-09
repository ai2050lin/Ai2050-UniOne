#!/usr/bin/env python3
"""
Stage 733: Phase XXVIII — 神经元级编码机制+跨模型全层追踪+SAE分解
================================================================================
Phase XXVII核心突破:
  - float32修复了KL=0 (P176)
  - 概念分离关键层=L3 (P177, Qwen3 only)
  - 推理跃迁层=L34 (P178)
  - 类别分离仅0.041 (P180)
  
Phase XXVII瓶颈:
  - DS7B/GLM4的float32+CPU-offload过慢, P177-P180未完成
  - 只在Qwen3上做了全层追踪, 缺少跨模型验证
  - 缺少神经元级分析(SAE/稀疏分解)

Phase XXVIII核心任务:
  P181: 跨模型全层EMB→HS变换追踪(bfloat16加速版)
  P182: 跨模型推理跃迁层验证(改进检测: top-5包含)
  P183: SAE稀疏分解 — 用ICA/PCA提取hidden state的可解释特征原子
  P184: 单维度语义分析 — h的每个维度与语义标签的相关性
  P185: 跨模型概念分离层对比 — 哪些规律是语言通用的?

策略: 所有forward pass用bfloat16(速度10x), 只在计算KL时转float32
用法: python stage733_phase28.py --model qwen3
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
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log(f"[load] Loaded. layers={n_layers}, d_model={d_model}")
    return mdl, tok

def find_token_positions(tok, text, target_word):
    input_ids = tok.encode(text, add_special_tokens=True)
    tokens = tok.convert_ids_to_tokens(input_ids)
    target_ids = tok.encode(target_word, add_special_tokens=False)
    target_tokens = tok.convert_ids_to_tokens(target_ids)
    positions = []
    for i in range(len(tokens)):
        if tokens[i] == target_tokens[0]:
            if len(target_tokens) == 1 or (i + len(target_tokens) <= len(tokens)
                and tokens[i:i+len(target_tokens)] == target_tokens):
                positions.append(i)
        elif tokens[i].lstrip("\u0120\u2581 ") == target_tokens[0].lstrip("\u0120\u2581 "):
            if len(target_tokens) == 1:
                positions.append(i)
    return positions, input_ids

def compute_kl(p, q, eps=1e-10):
    p = p.float()
    q = q.float()
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    kl = (p * (log_p - log_q)).sum(-1).mean().item()
    return kl

def get_all_hidden_states(model, tok, text, target_pos=None):
    """bfloat16前向传播, 获取所有层的hidden states"""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states  # list of [1, seq_len, d_model]
    logits = out.logits
    if target_pos is not None:
        last_logits = logits[:, target_pos, :]
    else:
        last_logits = logits[:, -1, :]
    return hs, last_logits

def get_last_probs(model, tok, text):
    """获取最后一个token的输出概率"""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp)
    probs = F.softmax(out.logits[:, -1, :].float(), dim=-1)
    return probs

def cos_sim(a, b):
    """计算两个向量的cosine相似度"""
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def inter_concept_cos(hs_dict, concepts):
    """计算所有概念对之间的cosine相似度"""
    cos_vals = []
    names = list(hs_dict.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = cos_sim(hs_dict[names[i]], hs_dict[names[j]])
            cos_vals.append(c)
    return np.mean(cos_vals), np.std(cos_vals), cos_vals

# ============================================================
# P181: 跨模型全层EMB→HS变换追踪
# ============================================================

def run_p181(model_name):
    """全层EMB→HS变换追踪: bfloat16加速版, 覆盖更多概念"""
    log("\n" + "="*70)
    log("P181: 全层EMB→HS变换追踪 (跨模型, bfloat16加速)")
    log("="*70)

    mdl, tok = load_model(model_name)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    # 10个核心概念 × 4个模板 = 40个文本
    concepts = ["cat", "dog", "apple", "tree", "water", "stone", "book", "car", "sun", "house"]
    templates = [
        "The {c} is on the table.",
        "A small {c} sat quietly.",
        "She saw a {c} yesterday.",
        "The {c} moved slowly.",
    ]
    
    # 选取代表性层: 0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 中间层, 最后3层
    all_layers = list(range(n_layers))
    sample_layers = sorted(set([0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, n_layers//2, n_layers-3, n_layers-2, n_layers-1]))
    # 确保不超过实际层数
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    log(f"  Sampling layers: {sample_layers}")
    log(f"  Concepts: {concepts} ({len(concepts)})")
    log(f"  Templates: {len(templates)}")
    
    # 对每个概念收集各层的hidden states(平均跨模板)
    layer_concept_hs = {l: {} for l in sample_layers}
    
    for concept in concepts:
        concept_hs_per_layer = {l: [] for l in sample_layers}
        
        for tmpl in templates:
            text = tmpl.format(c=concept)
            pos_list, _ = find_token_positions(tok, text, concept)
            if not pos_list:
                continue
            target_pos = pos_list[-1]
            
            hs, _ = get_all_hidden_states(mdl, tok, text, target_pos)
            
            for l in sample_layers:
                h = hs[l][0, target_pos, :]  # [d_model]
                concept_hs_per_layer[l].append(h)
        
        # 平均跨模板
        for l in sample_layers:
            if concept_hs_per_layer[l]:
                avg_h = torch.stack(concept_hs_per_layer[l]).mean(dim=0)
                layer_concept_hs[l][concept] = avg_h
    
    # 计算每层的inter-concept cos
    layer_results = []
    
    # 也计算embedding层
    emb_hs = {}
    for concept in concepts:
        ids = tok.encode(concept, add_special_tokens=False)
        emb = mdl.get_input_embeddings()(torch.tensor(ids).to(mdl.device))[0]
        emb_hs[concept] = emb
    
    mean_emb_cos, std_emb_cos, _ = inter_concept_cos(emb_hs, concepts)
    layer_results.append({
        "layer": "EMB", "mean_cos": round(mean_emb_cos, 4), "std_cos": round(std_emb_cos, 4),
        "n_concepts": len(emb_hs)
    })
    log(f"  EMB: cos={mean_emb_cos:.4f} +/- {std_emb_cos:.4f}")
    
    for l in sample_layers:
        if len(layer_concept_hs[l]) < 3:
            continue
        mean_c, std_c, _ = inter_concept_cos(layer_concept_hs[l], concepts)
        layer_results.append({
            "layer": l, "mean_cos": round(mean_c, 4), "std_cos": round(std_c, 4),
            "n_concepts": len(layer_concept_hs[l])
        })
        log(f"  L{l:2d}: cos={mean_c:.4f} +/- {std_c:.4f}")
    
    # 找最佳/最差分离层
    non_emb = [r for r in layer_results if r["layer"] != "EMB"]
    best = min(non_emb, key=lambda x: x["mean_cos"])
    worst = max(non_emb, key=lambda x: x["mean_cos"])
    
    log(f"\n  Best separation: L{best['layer']} (cos={best['mean_cos']:.4f})")
    log(f"  Worst separation: L{worst['layer']} (cos={worst['mean_cos']:.4f})")
    
    # EMB→HS方向追踪: cos(EMB_vector, HS_last_vector) per concept
    last_layer = n_layers - 1
    dir_results = {}
    for concept in concepts:
        if concept in emb_hs and concept in layer_concept_hs.get(last_layer, {}):
            c = cos_sim(emb_hs[concept], layer_concept_hs[last_layer][concept])
            dir_results[concept] = round(c, 4)
    
    log(f"\n  cos(EMB_dir, HS_{last_layer}_dir):")
    for c, v in dir_results.items():
        log(f"    {c}: {v:.4f}")
    if dir_results:
        log(f"    mean: {np.mean(list(dir_results.values())):.4f}")
    
    # 归一化方向(cos of unit vectors)
    norm_dir_results = {}
    for concept in concepts:
        if concept in emb_hs and concept in layer_concept_hs.get(last_layer, {}):
            e = emb_hs[concept].float()
            h = layer_concept_hs[last_layer][concept].float()
            e_unit = e / (e.norm() + 1e-10)
            h_unit = h / (h.norm() + 1e-10)
            c = F.cosine_similarity(e_unit.unsqueeze(0), h_unit.unsqueeze(0)).item()
            norm_dir_results[concept] = round(c, 4)
    
    log(f"\n  cos(normalized_EMB_dir, normalized_HS_{last_layer}_dir):")
    for c, v in norm_dir_results.items():
        log(f"    {c}: {v:.4f}")
    if norm_dir_results:
        log(f"    mean: {np.mean(list(norm_dir_results.values())):.4f}")
    
    # Early/Mid/Late分区
    mid = n_layers // 2
    early_non_emb = [r for r in non_emb if isinstance(r["layer"], int) and r["layer"] < mid // 2]
    mid_non_emb = [r for r in non_emb if isinstance(r["layer"], int) and mid // 2 <= r["layer"] < mid + mid // 2]
    late_non_emb = [r for r in non_emb if isinstance(r["layer"], int) and r["layer"] >= mid + mid // 2]
    
    early_cos = np.mean([r["mean_cos"] for r in early_non_emb]) if early_non_emb else 0
    mid_cos = np.mean([r["mean_cos"] for r in mid_non_emb]) if mid_non_emb else 0
    late_cos = np.mean([r["mean_cos"] for r in late_non_emb]) if late_non_emb else 0
    
    log(f"\n  Early/Mid/Late: {early_cos:.3f} / {mid_cos:.3f} / {late_cos:.3f}")
    
    return {
        "layer_results": layer_results,
        "best_layer": best,
        "worst_layer": worst,
        "dir_results": dir_results,
        "norm_dir_results": norm_dir_results,
        "early_mid_late": [round(early_cos, 3), round(mid_cos, 3), round(late_cos, 3)],
    }


# ============================================================
# P182: 跨模型推理跃迁层验证(改进检测)
# ============================================================

def run_p182(model_name):
    """推理跃迁层: 改进检测方式(层间累积变化+推理vs非推理分化层), 跨模型验证"""
    log("\n" + "="*70)
    log("P182: 推理跃迁层验证 (改进检测: 累积变化+分化分析)")
    log("="*70)
    
    mdl, tok = load_model(model_name)
    n_layers = len(mdl.model.layers)
    
    # 推理文本(35个, 大规模)
    reasoning_texts = [
        # 三段论 (8)
        {"text": "All cats are animals. Whiskers is a cat. Therefore, Whiskers is", 
         "expected": ["an", "animal"], "type": "syllogism"},
        {"text": "All mammals are warm-blooded. Dolphins are mammals. Therefore, dolphins are",
         "expected": ["warm", "blood"], "type": "syllogism"},
        {"text": "All roses are flowers. This is a rose. Therefore, this is", 
         "expected": ["a", "flower"], "type": "syllogism"},
        {"text": "All birds have feathers. Penguins are birds. Therefore, penguins have",
         "expected": ["feathers"], "type": "syllogism"},
        {"text": "All metals conduct electricity. Copper is a metal. Therefore, copper",
         "expected": ["conducts", "can"], "type": "syllogism"},
        {"text": "All fish live in water. Salmon are fish. Therefore, salmon live",
         "expected": ["in", "water"], "type": "syllogism"},
        {"text": "All trees need sunlight. Oaks are trees. Therefore, oaks need",
         "expected": ["sunlight"], "type": "syllogism"},
        {"text": "All insects have six legs. Ants are insects. Therefore, ants have",
         "expected": ["six"], "type": "syllogism"},
        
        # 条件推理 (6)
        {"text": "If it rains, the ground gets wet. It is raining. Therefore, the ground is",
         "expected": ["wet"], "type": "conditional"},
        {"text": "If you study hard, you pass the exam. She studied hard. Therefore, she will",
         "expected": ["pass"], "type": "conditional"},
        {"text": "If the temperature drops below zero, water freezes. The temperature is below zero. Therefore, water",
         "expected": ["freezes", "will"], "type": "conditional"},
        {"text": "If a number is divisible by 4, it is even. 16 is divisible by 4. Therefore, 16 is",
         "expected": ["even"], "type": "conditional"},
        {"text": "If you heat ice, it melts. The ice is heated. Therefore, the ice",
         "expected": ["melts", "will"], "type": "conditional"},
        {"text": "If A is true and B is true, then C is true. A is true. B is true. Therefore, C is",
         "expected": ["true"], "type": "conditional"},
        
        # 传递性 (6)
        {"text": "Alice is taller than Bob. Bob is taller than Carol. Therefore, Alice is taller than",
         "expected": ["Carol"], "type": "transitivity"},
        {"text": "A is greater than B. B is greater than C. Therefore, A is greater than",
         "expected": ["C"], "type": "transitivity"},
        {"text": "Red is warmer than blue. Blue is warmer than green. Therefore, red is warmer than",
         "expected": ["green"], "type": "transitivity"},
        {"text": "Paris is north of Lyon. Lyon is north of Marseille. Therefore, Paris is north of",
         "expected": ["Marseille"], "type": "transitivity"},
        {"text": "Gold is more valuable than silver. Silver is more valuable than bronze. Therefore, gold is more valuable than",
         "expected": ["bronze", "both"], "type": "transitivity"},
        {"text": "X is heavier than Y. Y is heavier than Z. Therefore, X is heavier than",
         "expected": ["Z"], "type": "transitivity"},
        
        # 量词推理 (5)
        {"text": "Some dogs are brown. Max is a dog. Max is",
         "expected": ["brown", "possibly"], "type": "quantifier"},
        {"text": "All students passed the test. John is a student. John",
         "expected": ["passed"], "type": "quantifier"},
        {"text": "No birds can swim underwater. Penguins are birds. Therefore, penguins",
         "expected": ["cannot", "can", "not"], "type": "quantifier"},
        {"text": "Every car needs fuel. This is a car. This",
         "expected": ["needs", "car"], "type": "quantifier"},
        {"text": "Most people like chocolate. Sarah is a person. Sarah likely",
         "expected": ["likes"], "type": "quantifier"},
        
        # 否定推理 (5)
        {"text": "The cat is not black. The cat is not white. The cat is probably",
         "expected": ["not", "gray", "grey", "brown", "orange"], "type": "negation"},
        {"text": "It is not Monday today. It is not Tuesday today. It is not Wednesday today. Today might be",
         "expected": ["Thursday", "Friday", "Saturday", "Sunday"], "type": "negation"},
        {"text": "Tom does not like apples. Tom does not like bananas. Tom might like",
         "expected": ["oranges", "grapes", "pears", "cherries"], "type": "negation"},
        {"text": "The opposite of hot is not warm. The opposite of hot is",
         "expected": ["cold"], "type": "negation"},
        {"text": "If not A, then B. Not A is true. Therefore,",
         "expected": ["B", "b"], "type": "negation"},
        
        # 反事实 (5)
        {"text": "If the sun were blue, the sky would appear",
         "expected": ["blue", "different"], "type": "counterfactual"},
        {"text": "If humans could fly, commuting would be",
         "expected": ["faster", "different", "easier"], "type": "counterfactual"},
        {"text": "If water were solid at room temperature, fish would",
         "expected": ["not", "die", "suffocate"], "type": "counterfactual"},
        {"text": "If gravity did not exist, objects would",
         "expected": ["float", "drift", "fly"], "type": "counterfactual"},
        {"text": "If cats could talk, they would probably say",
         "expected": ["feed", "food", "me"], "type": "counterfactual"},
    ]
    
    # 非推理(普通描述)文本作为对照组
    control_texts = [
        "The cat sat on the warm mat.", "A dog barked at the stranger.",
        "She ate a sweet apple.", "The sun was bright and warm.",
        "He read an interesting book.", "The tree grew tall and strong.",
        "Water is essential for life.", "The stone was heavy and rough.",
        "She drove her car to work.", "A bird flew across the sky.",
        "The red flower bloomed.", "He built a house of wood.",
        "The river flowed downstream.", "She painted a beautiful picture.",
        "The children played outside.", "A gentle wind blew through.",
    ]
    
    log(f"  Reasoning texts: {len(reasoning_texts)}, Control texts: {len(control_texts)}")
    
    # === Part 1: 推理准确率测试 ===
    results = []
    
    for rt in reasoning_texts:
        text = rt["text"]
        expected = rt["expected"]
        rtype = rt["type"]
        
        hs, last_logits = get_all_hidden_states(mdl, tok, text)
        probs = F.softmax(last_logits.float(), dim=-1)
        top5 = torch.topk(probs, 5)
        top5_tokens = tok.convert_ids_to_tokens(top5.indices[0].tolist()) if top5.indices.dim() > 1 else tok.convert_ids_to_tokens(top5.indices.tolist())
        top5_probs = top5.values[0].tolist() if top5.values.dim() > 1 else top5.values.tolist()
        
        # 改进检测: top-5中是否包含任何expected token
        matched = False
        matched_token = None
        for exp in expected:
            for t in top5_tokens:
                if exp.lower() in t.lower().replace("\u0120", "").replace("\u2581", ""):
                    matched = True
                    matched_token = t
                    break
            if matched:
                break
        
        results.append({
            "type": rtype, "matched": matched, "matched_token": matched_token,
            "top5": top5_tokens[:3],
            "top5_probs": [round(p, 4) for p in top5_probs[:3]],
            "expected": expected,
        })
        
        status = "OK" if matched else "MISS"
        log(f"  [{status}] {rtype:12s} | top5={top5_tokens[:3]} | exp={expected}")
    
    # === Part 2: 层间累积变化(推理文本vs控制文本) ===
    log(f"\n  --- Layer divergence analysis ---")
    
    # 收集推理文本和控制文本在各层的hidden states
    reasoning_hs_per_layer = defaultdict(list)
    control_hs_per_layer = defaultdict(list)
    
    sample_layers = sorted(set([0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 
                                  n_layers-5, n_layers-4, n_layers-3, n_layers-2, n_layers-1]))
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    # 采样一部分推理文本(避免太慢)
    sample_reasoning = reasoning_texts[:15]
    
    for rt in sample_reasoning:
        hs, _ = get_all_hidden_states(mdl, tok, rt["text"])
        for l in sample_layers:
            h = hs[l][0, -1, :].float().cpu()
            reasoning_hs_per_layer[l].append(h)
    
    for ct in control_texts:
        hs, _ = get_all_hidden_states(mdl, tok, ct)
        for l in sample_layers:
            h = hs[l][0, -1, :].float().cpu()
            control_hs_per_layer[l].append(h)
    
    # 计算每层: 推理文本内部的平均cos, 控制文本内部的平均cos, 推理vs控制的cos
    divergence_results = []
    
    for l in sample_layers:
        r_hs = torch.stack(reasoning_hs_per_layer[l])  # [n_r, d]
        c_hs = torch.stack(control_hs_per_layer[l])     # [n_c, d]
        
        r_mean = r_hs.mean(dim=0)
        c_mean = c_hs.mean(dim=0)
        
        # 推理内部cos
        r_pairs = []
        for i in range(min(10, len(r_hs))):
            for j in range(i+1, min(10, len(r_hs))):
                r_pairs.append(cos_sim(r_hs[i], r_hs[j]))
        r_intra = np.mean(r_pairs) if r_pairs else 0
        
        # 控制内部cos
        c_pairs = []
        for i in range(min(10, len(c_hs))):
            for j in range(i+1, min(10, len(c_hs))):
                c_pairs.append(cos_sim(c_hs[i], c_hs[j]))
        c_intra = np.mean(c_pairs) if c_pairs else 0
        
        # 推理vs控制
        cross_pairs = []
        for i in range(min(10, len(r_hs))):
            for j in range(min(10, len(c_hs))):
                cross_pairs.append(cos_sim(r_hs[i], c_hs[j]))
        cross_cos = np.mean(cross_pairs) if cross_pairs else 0
        
        # 分化度: 推理vs控制的差异
        divergence = cross_cos - max(r_intra, c_intra)  # 正=不分化, 负=分化
        
        divergence_results.append({
            "layer": l, "r_intra": round(r_intra, 4),
            "c_intra": round(c_intra, 4), "cross": round(cross_cos, 4),
            "divergence": round(divergence, 4),
        })
        
        log(f"  L{l:2d}: r_intra={r_intra:.3f} c_intra={c_intra:.3f} cross={cross_cos:.3f} div={divergence:.3f}")
    
    # 找最大分化层(divergence最负=推理vs控制最不同)
    max_div_layer = min(divergence_results, key=lambda x: x["divergence"])
    min_div_layer = max(divergence_results, key=lambda x: x["divergence"])
    
    log(f"\n  Max divergence (reasoning vs control differ most): L{max_div_layer['layer']} (div={max_div_layer['divergence']:.3f})")
    log(f"  Min divergence (reasoning vs control similar): L{min_div_layer['layer']} (div={min_div_layer['divergence']:.3f})")
    
    # === Part 3: 累积变化检测(相对L0的变化) ===
    log(f"\n  --- Cumulative change from L0 ---")
    
    cum_change = []
    for l in sample_layers:
        r_mean = torch.stack(reasoning_hs_per_layer[l]).mean(dim=0)
        r_l0 = torch.stack(reasoning_hs_per_layer[0]).mean(dim=0)
        cos_l0 = cos_sim(r_mean, r_l0)
        cum_change.append({"layer": l, "cos_from_l0": round(cos_l0, 4)})
        log(f"  L{l:2d}: cos(from_L0)={cos_l0:.4f}")
    
    # 找变化最大层(相对L0的cos最低)
    min_cos_layer = min(cum_change, key=lambda x: x["cos_from_l0"])
    log(f"\n  Max cumulative change (furthest from L0): L{min_cos_layer['layer']} (cos={min_cos_layer['cos_from_l0']:.4f})")
    
    # 统计推理准确率
    total = len(results)
    matched_count = sum(1 for r in results if r["matched"])
    match_rate = matched_count / total if total > 0 else 0
    
    type_stats = defaultdict(lambda: {"total": 0, "matched": 0})
    for r in results:
        type_stats[r["type"]]["total"] += 1
        if r["matched"]:
            type_stats[r["type"]]["matched"] += 1
    
    log(f"\n  Total: {matched_count}/{total} = {match_rate:.1%}")
    log(f"\n  By type:")
    for t, s in type_stats.items():
        rate = s["matched"]/s["total"] if s["total"] > 0 else 0
        log(f"    {t:14s}: {s['matched']}/{s['total']} = {rate:.0%}")
    
    return {
        "total": total, "matched": matched_count, "match_rate": round(match_rate, 4),
        "type_stats": {t: {"total": s["total"], "matched": s["matched"],
                           "rate": round(s["matched"]/s["total"], 4) if s["total"] > 0 else 0}
                      for t, s in type_stats.items()},
        "divergence": divergence_results,
        "max_divergence_layer": max_div_layer,
        "min_divergence_layer": min_div_layer,
        "cumulative_change": cum_change,
        "max_change_layer": min_cos_layer,
    }


# ============================================================
# P183: SAE稀疏分解 — ICA/PCA提取可解释特征
# ============================================================

def run_p183(model_name):
    """SAE稀疏分解: 用PCA+varimax旋转提取hidden state的可解释特征"""
    log("\n" + "="*70)
    log("P183: SAE稀疏分解 — PCA+Varimax提取可解释特征原子")
    log("="*70)
    
    mdl, tok = load_model(model_name)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    # 收集大量hidden states作为"字典学习"的训练数据
    # 100个不同文本 × 最后token → 100个d_model维向量
    collection_texts = [
        # 语义类别 (动物、水果、颜色、动作、情感、地点)
        "The cat sat on the warm mat.",
        "A dog barked loudly at the stranger.",
        "The bird flew across the blue sky.",
        "Fish swim in the deep ocean.",
        "The lion roared in the savanna.",
        "A rabbit hopped through the green garden.",
        "The horse galloped across the open field.",
        "A bear slept in the dark cave.",
        "The eagle soared above the mountains.",
        "A dolphin jumped out of the water.",
        # 水果/食物
        "She ate a sweet red apple.",
        "The orange was juicy and fresh.",
        "He bought a green watermelon.",
        "The banana was yellow and ripe.",
        "She baked a chocolate cake.",
        "The bread smelled delicious.",
        "He cooked a spicy curry.",
        "The soup was hot and tasty.",
        "A fresh salad with vegetables.",
        "The ice cream melted quickly.",
        # 颜色/属性
        "The red car drove fast.",
        "A blue bird sang beautifully.",
        "The green forest was peaceful.",
        "She wore a yellow dress.",
        "The white snow covered everything.",
        "The black cat was mysterious.",
        "A golden sunset painted the sky.",
        "The purple flower bloomed.",
        # 动作/事件
        "She ran quickly to catch the bus.",
        "He wrote a long letter home.",
        "The children played in the park.",
        "She sang a beautiful song.",
        "He built a tall tower.",
        "The teacher explained the lesson.",
        "She danced gracefully on stage.",
        "He read an interesting book.",
        # 情感/状态
        "She felt happy and excited.",
        "He was sad and lonely.",
        "The news made everyone worried.",
        "She was surprised by the gift.",
        "He felt angry about the decision.",
        "The atmosphere was calm and peaceful.",
        # 推理
        "All cats are animals. This is a cat. Therefore, this is an animal.",
        "If it rains, the ground gets wet. It is raining.",
        "A is greater than B. B is greater than C. Therefore, A is greater than C.",
        "No fish can live on land. Sharks are fish.",
        "Every student must study. She is a student.",
        # 科学
        "The Earth orbits around the Sun.",
        "Water boils at 100 degrees Celsius.",
        "Light travels faster than sound.",
        "Photosynthesis converts sunlight to energy.",
        "Gravity pulls objects toward the Earth.",
        # 地点
        "Paris is the capital of France.",
        "The Great Wall is in China.",
        "New York is a big city.",
        "The Amazon is a vast river.",
        "Mount Everest is the tallest mountain.",
        # 抽象
        "Knowledge is power and wisdom.",
        "Freedom requires responsibility.",
        "Time waits for no one.",
        "Beauty is in the eye of the beholder.",
        "Truth is often stranger than fiction.",
        "Love conquers all obstacles.",
        "Hope springs eternal in the human heart.",
        "Justice must be served fairly.",
        # 更多语义变体
        "A small kitten played with yarn.",
        "The old man walked slowly home.",
        "Bright stars filled the night sky.",
        "The river flowed peacefully downstream.",
        "A fierce storm approached the coast.",
        "The baby laughed at the funny clown.",
        "Soft music played in the background.",
        "The candle flickered in the dark room.",
        "Fresh coffee aroma filled the kitchen.",
        "The autumn leaves turned red and gold.",
        "A gentle breeze cooled the warm afternoon.",
        "The old book had yellowed pages.",
        "She carefully opened the wooden door.",
        "The clock struck midnight silently.",
        "Raindrops fell on the tin roof.",
        "The bridge stretched across the wide river.",
        "A single cloud drifted in the blue sky.",
        "The garden was full of blooming flowers.",
        "He whispered a secret in her ear.",
        "The fire crackled in the fireplace.",
        # 对比文本(反向语义)
        "The cat hated the cold water.",
        "Darkness filled the empty room.",
        "Silence followed the loud explosion.",
        "The hot sun made everyone thirsty.",
        "The frozen lake reflected the moon.",
        "She quickly forgot the sad memory.",
        "The tiny ant lifted the heavy crumb.",
        "The wise fool made a clever mistake.",
        "The soft stone broke the hard hammer.",
        "The fast turtle won the slow race.",
    ]
    
    log(f"  Collecting hidden states from {len(collection_texts)} texts...")
    
    # 收集多个层的hidden states
    target_layers = [0, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    all_hs_data = {}  # layer -> tensor [n_texts, d_model]
    
    for text in collection_texts:
        hs, _ = get_all_hidden_states(mdl, tok, text)
        for l in target_layers:
            h = hs[l][0, -1, :].float().cpu()  # 最后token的hidden state
            if l not in all_hs_data:
                all_hs_data[l] = []
            all_hs_data[l].append(h)
    
    results = {}
    
    for l in target_layers:
        hs_matrix = torch.stack(all_hs_data[l])  # [n_texts, d_model]
        n_samples, n_features = hs_matrix.shape
        log(f"\n  Layer L{l}: {n_samples} samples, {n_features} dims")
        
        # 1. PCA降维
        # 中心化
        mean_h = hs_matrix.mean(dim=0)
        centered = hs_matrix - mean_h
        
        # 清理nan/inf
        centered = torch.nan_to_num(centered, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # SVD (等价于PCA)
        try:
            U, S, Vt = torch.linalg.svd(centered.float(), full_matrices=False)
        except Exception as e:
            log(f"    SVD failed: {e}, trying numpy...")
            U_np, S_np, Vt_np = np.linalg.svd(centered.float().numpy(), full_matrices=False)
            S = torch.tensor(S_np)
            Vt = torch.tensor(Vt_np)
        
        # 检查nan
        if torch.isnan(S).any():
            log(f"    WARNING: S contains NaN, replacing with zeros")
            S = torch.nan_to_num(S, nan=0.0)
        
        # 方差解释比
        total_var = (S ** 2).sum()
        if total_var < 1e-10:
            log(f"    WARNING: total variance ~0, skipping PCA")
            results[f"L{l}"] = {"error": "zero_variance"}
            continue
        explained_var = (S ** 2) / total_var
        
        # 取前50个主成分(或更少如果样本不够)
        n_components = min(50, n_samples - 1, n_features)
        
        log(f"    Top-{n_components} PCs explain: {explained_var[:n_components].sum().item():.4f} of variance")
        log(f"    Top-5: {explained_var[:5].tolist()}")
        
        # 2. Varimax旋转(使载荷更稀疏, 更可解释)
        # 对前n_components个PC做varimax
        loadings = Vt[:n_components, :].T  # [d_model, n_components]
        
        # 简化的varimax旋转(kaiser normalization)
        loadings_np = loadings.numpy()
        rotated, rotation_matrix = varimax(loadings_np)
        
        # 3. 稀疏度分析
        # 每个feature(旋转后的成分)的稀疏度
        sparsities = []
        feature_top_dims = []
        for i in range(min(20, n_components)):
            feat_weights = np.abs(rotated[:, i])
            # Hoyer稀疏度: (sqrt(n) - L1/L2) / (sqrt(n) - 1)
            l1 = np.sum(feat_weights)
            l2 = np.sqrt(np.sum(feat_weights**2))
            n = len(feat_weights)
            hoyer = (np.sqrt(n) - l1/l2) / (np.sqrt(n) - 1) if n > 1 else 0
            sparsities.append(round(hoyer, 4))
            
            # 找top-5贡献维度
            top_idx = np.argsort(feat_weights)[-5:][::-1]
            feature_top_dims.append(top_idx.tolist())
        
        mean_sparsity = np.mean(sparsities)
        log(f"    Mean sparsity (Hoyer, top-20): {mean_sparsity:.4f}")
        log(f"    Per-feature sparsity: {[f'{s:.3f}' for s in sparsities[:10]]}")
        
        # 4. 将每个hidden state投影到旋转后的空间, 分析哪些feature被激活
        # rotated shape: [d_model, n_components], centered shape: [n_samples, d_model]
        rot_tensor = torch.tensor(rotated[:, :n_components].astype(np.float32))  # [d_model, n_components]
        projected = centered.float() @ rot_tensor  # [n_samples, n_components]
        
        # 每个文本在每个feature上的激活值
        activation_stats = []
        for i in range(min(10, n_components)):
            act_vals = projected[:, i].numpy()
            activation_stats.append({
                "feature": i,
                "mean": round(float(np.mean(act_vals)), 4),
                "std": round(float(np.std(act_vals)), 4),
                "max": round(float(np.max(np.abs(act_vals))), 4),
                "n_active": int(np.sum(np.abs(act_vals) > np.std(act_vals))),  # 激活数
            })
        
        # 5. 概念localization: 哪些维度对特定语义最敏感?
        # 对比"动物"文本 vs "非动物"文本的hidden state差异
        animal_indices = list(range(10))  # 前10个是动物文本
        non_animal_indices = list(range(10, 30))  # 水果+颜色
        
        if len(animal_indices) > 0 and len(non_animal_indices) > 0:
            animal_mean = hs_matrix[animal_indices].mean(dim=0)
            non_animal_mean = hs_matrix[non_animal_indices].mean(dim=0)
            diff = animal_mean - non_animal_mean
            diff_norm = diff / (diff.norm() + 1e-10)
            
            # 投影到旋转空间看哪些feature区分动物
            diff_projected = (diff.unsqueeze(0).float() @ torch.tensor(rotated[:, :n_components].astype(np.float32)))[0]
            top_animal_features = torch.topk(diff_projected.abs(), 5)
            log(f"    Animal-distinguishing features: {top_animal_features.indices.tolist()}")
            log(f"    Feature activations: {[round(v, 3) for v in top_animal_features.values.tolist()]}")
        
        results[f"L{l}"] = {
            "n_components": n_components,
            "explained_variance_top5": [round(v, 4) for v in explained_var[:5].tolist()],
            "explained_variance_total": round(explained_var[:n_components].sum().item(), 4),
            "mean_sparsity": round(mean_sparsity, 4),
            "sparsities": sparsities[:20],
            "feature_top_dims": feature_top_dims[:10],
            "activation_stats": activation_stats[:10],
        }
    
    return results


def varimax(loadings, max_iter=100, tol=1e-6):
    """Varimax旋转: 使载荷矩阵更稀疏, 更可解释"""
    n_vars, n_factors = loadings.shape
    
    # Kaiser normalization
    col_norms = np.sqrt((loadings**2).sum(axis=0))
    col_norms[col_norms < 1e-10] = 1.0
    loadings_norm = loadings / col_norms
    
    # Varimax算法(不依赖scipy.ortho_group)
    rotation = np.eye(n_factors)
    
    for _ in range(max_iter):
        old_rotation = rotation.copy()
        
        x = loadings_norm @ rotation
        u, s, vt = np.linalg.svd(loadings_norm.T @ (x**3 - x @ np.diag((x**2).sum(axis=0)) / n_vars))
        rotation = u @ vt
        
        if np.max(np.abs(rotation - old_rotation)) < tol:
            break
    
    rotated = loadings_norm @ rotation
    rotated = rotated * col_norms  # 恢复normalization
    
    return rotated, rotation


# ============================================================
# P184: 单维度语义分析
# ============================================================

def run_p184(model_name):
    """单维度语义分析: h的每个维度与语义类别的相关性"""
    log("\n" + "="*70)
    log("P184: 单维度语义分析 — h的维度与语义类别的关联")
    log("="*70)
    
    mdl, tok = load_model(model_name)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    # 语义类别及其文本
    semantic_categories = {
        "animal": [
            "The cat sat on the mat.", "A dog barked loudly.",
            "The bird flew in the sky.", "Fish swim in the ocean.",
            "The lion roared.", "A rabbit hopped.",
            "The horse galloped.", "A bear slept.",
            "The eagle soared.", "A dolphin jumped.",
        ],
        "fruit": [
            "She ate a sweet apple.", "The orange was juicy.",
            "A ripe banana.", "Fresh strawberries.",
            "The watermelon was big.", "Grapes grew on the vine.",
            "A peach fell from the tree.", "The lemon was sour.",
            "Cherries are red and small.", "A mango tasted delicious.",
        ],
        "color": [
            "The sky is blue.", "Her dress was red.",
            "Green leaves on the tree.", "A yellow sunflower.",
            "The white snow fell.", "Black clouds gathered.",
            "Purple flowers bloomed.", "A golden ring.",
            "The pink sunset glowed.", "Silver moonlight.",
        ],
        "emotion": [
            "She felt very happy.", "He was deeply sad.",
            "The news made her angry.", "They were excited.",
            "He felt worried and anxious.", "She was surprised.",
            "The movie was terrifying.", "He felt lonely.",
            "She was grateful.", "The child was curious.",
        ],
        "action": [
            "She ran quickly.", "He jumped high.",
            "They sang together.", "She wrote carefully.",
            "He built a house.", "She danced gracefully.",
            "The baby crawled.", "He swam across.",
            "She cooked dinner.", "They laughed loudly.",
        ],
        "location": [
            "The city was busy.", "Mountains were tall.",
            "The beach was sandy.", "The forest was dark.",
            "The desert was hot.", "The river was wide.",
            "A small village.", "The island was remote.",
            "The valley was green.", "The cave was deep.",
        ],
        "abstract": [
            "Knowledge is important.", "Freedom matters greatly.",
            "Time passes quickly.", "Beauty exists everywhere.",
            "Truth reveals itself.", "Love is powerful.",
            "Hope never dies.", "Justice must prevail.",
            "Wisdom comes with age.", "Peace is precious.",
        ],
        "science": [
            "The Earth orbits the Sun.", "Water boils at 100 degrees.",
            "Light travels fast.", "Gravity pulls downward.",
            "Atoms are small.", "DNA carries information.",
            "The universe expands.", "Energy cannot be destroyed.",
            "Evolution drives change.", "Chemical reactions occur.",
        ],
    }
    
    # 收集每层的hidden states
    target_layers = [0, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers and l >= 3]))  # 跳过L0-L2(embedding相似性导致F=0)
    
    results = {}
    
    for l in target_layers:
        log(f"\n  --- Layer L{l} ---")
        
        # 收集每个类别的hidden states
        category_vectors = {}
        all_vectors = []
        all_labels = []
        
        for cat_name, texts in semantic_categories.items():
            cat_hs = []
            for text in texts:
                hs, _ = get_all_hidden_states(mdl, tok, text)
                h = hs[l][0, -1, :].float().cpu()
                cat_hs.append(h)
                all_vectors.append(h)
                all_labels.append(cat_name)
            category_vectors[cat_name] = torch.stack(cat_hs)  # [n_texts, d_model]
        
        all_matrix = torch.stack(all_vectors)  # [total_texts, d_model]
        n_total, n_dims = all_matrix.shape
        
        # 1. 对每个维度, 计算ANOVA F-statistic (类别间差异/类别内差异)
        # 更简单: 对每个维度, 计算类内均值差的方差
        dim_discrimination = np.zeros(n_dims)
        
        for d in range(n_dims):
            # 每个类别在该维度的均值
            cat_means = []
            cat_vars = []
            for cat_name, cat_hs in category_vectors.items():
                vals = cat_hs[:, d].numpy()
                cat_means.append(np.mean(vals))
                cat_vars.append(np.var(vals))
            
            cat_means = np.array(cat_means)
            cat_vars = np.array(cat_vars)
            
            # F-statistic: (类间方差) / (类内方差)
            between_var = np.var(cat_means)
            within_var = np.mean(cat_vars) + 1e-10
            dim_discrimination[d] = between_var / within_var
        
        # Top-20最区分维度
        top_disc_dims = np.argsort(dim_discrimination)[-20:][::-1]
        log(f"    Top-20 discriminating dims: {top_disc_dims.tolist()}")
        log(f"    Top-5 F-stats: {[round(dim_discrimination[d], 2) for d in top_disc_dims[:5]]}")
        
        # 2. 对top维度, 分析它在各类别上的分布
        dim_profiles = {}
        for d in top_disc_dims[:10]:
            profile = {}
            for cat_name, cat_hs in category_vectors.items():
                vals = cat_hs[:, d].numpy()
                profile[cat_name] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }
            dim_profiles[f"dim_{d}"] = profile
            
            # 显示该维度在各类别的均值
            means = [profile[c]["mean"] for c in semantic_categories.keys()]
            log(f"    dim_{d}: {means}")
        
        # 3. 每个类别的"签名方向": 各类别均值之间的差异
        cat_mean_vectors = {}
        for cat_name, cat_hs in category_vectors.items():
            cat_mean_vectors[cat_name] = cat_hs.mean(dim=0)
        
        # 4. 维度的选择性分析: 是否存在"动物维度"?
        # 用one-vs-rest t-test找每个类别的特征维度
        category_feature_dims = {}
        for target_cat in semantic_categories.keys():
            target_vals = category_vectors[target_cat]  # [n, d_model]
            other_vals = torch.cat([category_vectors[c] for c in semantic_categories.keys() if c != target_cat])
            
            dim_tstats = np.zeros(n_dims)
            for d in range(n_dims):
                t_vals = target_vals[:, d].float().numpy()
                o_vals = other_vals[:, d].float().numpy()
                # Welch t-test
                t_mean_diff = np.mean(t_vals) - np.mean(o_vals)
                t_se = np.sqrt(np.var(t_vals)/len(t_vals) + np.var(o_vals)/len(o_vals) + 1e-10)
                dim_tstats[d] = t_mean_diff / t_se
            
            top_cat_dims = np.argsort(np.abs(dim_tstats))[-5:][::-1].tolist()
            category_feature_dims[target_cat] = {
                "top_dims": top_cat_dims,
                "top_tstats": [round(float(dim_tstats[d]), 2) for d in top_cat_dims],
            }
            log(f"    {target_cat:10s} top dims: {top_cat_dims} t={category_feature_dims[target_cat]['top_tstats']}")
        
        # 5. 共享维度分析: 多个类别是否共用相同的高区分维度?
        all_top_dims = []
        for cat, info in category_feature_dims.items():
            all_top_dims.extend(info["top_dims"])
        from collections import Counter
        dim_freq = Counter(all_top_dims)
        shared_dims = dim_freq.most_common(5)
        log(f"    Shared discriminating dims (freq): {[(d, f) for d, f in shared_dims]}")
        
        # 6. 维度独立性: top-20维度之间是否独立?
        if len(top_disc_dims) >= 5:
            idx = sorted(top_disc_dims[:10])  # 排序以避免负步长
            top_dim_vectors = all_matrix[:, idx].contiguous().numpy().copy()
            corr = np.corrcoef(top_dim_vectors.T)
            mean_abs_corr = np.mean(np.abs(corr - np.eye(corr.shape[0])))
            log(f"    Top-10 dim inter-correlation (mean|corr|): {mean_abs_corr:.4f}")
        
        results[f"L{l}"] = {
            "top_disc_dims": top_disc_dims.tolist()[:20],
            "top_fstats": [round(float(dim_discrimination[d]), 4) for d in top_disc_dims[:20]],
            "dim_profiles": dim_profiles,
            "category_features": category_feature_dims,
            "shared_dims": shared_dims,
            "mean_abs_corr": round(float(mean_abs_corr), 4) if len(top_disc_dims) >= 5 else None,
        }
    
    return results


# ============================================================
# P185: 跨模型概念分离层对比
# ============================================================

def run_p185(model_name):
    """跨模型对比: 概念分离层是否跨模型一致?"""
    log("\n" + "="*70)
    log("P185: 跨模型概念分离层对比")
    log("="*70)
    
    mdl, tok = load_model(model_name)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    # 标准化的概念集和模板(所有模型使用相同文本)
    concepts = ["cat", "dog", "apple", "tree", "water", "stone", "book", "car", "sun", "house",
                "bird", "fish", "rose", "mountain", "river", "knife", "ring", "lamp", "bread", "ship"]
    templates = [
        "The {c} is on the table.",
        "A small {c} sat quietly.",
        "She saw a {c} yesterday.",
    ]
    
    # 标准化采样层(用比例而非绝对层号, 以适配不同深度的模型)
    layer_ratios = [0.0, 0.03, 0.06, 0.1, 0.17, 0.25, 0.33, 0.5, 0.67, 0.75, 0.83, 0.9, 0.94, 0.97, 1.0]
    sample_layers = sorted(set([max(0, min(n_layers-1, int(r * (n_layers-1)))) for r in layer_ratios]))
    
    log(f"  Model: {model_name}, layers={n_layers}, d_model={d_model}")
    log(f"  Concepts: {len(concepts)}, Templates: {len(templates)}")
    log(f"  Sample layers (by ratio): {sample_layers}")
    
    # 收集数据
    layer_concept_hs = {l: {} for l in sample_layers}
    
    for concept in concepts:
        for tmpl in templates:
            text = tmpl.format(c=concept)
            pos_list, _ = find_token_positions(tok, text, concept)
            if not pos_list:
                continue
            target_pos = pos_list[-1]
            hs, _ = get_all_hidden_states(mdl, tok, text, target_pos)
            for l in sample_layers:
                h = hs[l][0, target_pos, :].float()
                if concept not in layer_concept_hs[l]:
                    layer_concept_hs[l][concept] = []
                layer_concept_hs[l][concept].append(h)
    
    # 计算每层cos
    layer_results = []
    for l in sample_layers:
        if len(layer_concept_hs[l]) < 3:
            continue
        # 平均跨模板
        avg_hs = {}
        for c, h_list in layer_concept_hs[l].items():
            avg_hs[c] = torch.stack(h_list).mean(dim=0)
        
        mean_cos, std_cos, _ = inter_concept_cos(avg_hs, concepts)
        layer_ratio = l / max(n_layers - 1, 1)
        layer_results.append({
            "layer": l,
            "ratio": round(layer_ratio, 4),
            "mean_cos": round(mean_cos, 4),
            "std_cos": round(std_cos, 4),
            "n_concepts": len(avg_hs),
        })
        log(f"  L{l:2d} (r={layer_ratio:.2f}): cos={mean_cos:.4f} +/- {std_cos:.4f}")
    
    # 找最佳/最差分离层(用ratio)
    non_emb = [r for r in layer_results if r["ratio"] > 0]
    best = min(non_emb, key=lambda x: x["mean_cos"])
    worst = max(non_emb, key=lambda x: x["mean_cos"])
    
    log(f"\n  Best separation: L{best['layer']} (ratio={best['ratio']:.2f}, cos={best['mean_cos']:.4f})")
    log(f"  Worst separation: L{worst['layer']} (ratio={worst['ratio']:.2f}, cos={worst['mean_cos']:.4f})")
    
    # 类别分析(最后层)
    categories = {
        "animal": ["cat", "dog", "bird", "fish"],
        "plant": ["tree", "rose"],
        "natural": ["water", "stone", "mountain", "river", "sun"],
        "artifact": ["house", "knife", "ring", "lamp", "car", "ship", "book", "bread"],
    }
    
    last_layer = n_layers - 1
    if last_layer in layer_concept_hs:
        last_avg = {}
        for c, h_list in layer_concept_hs[last_layer].items():
            last_avg[c] = torch.stack(h_list).mean(dim=0)
        
        intra_cos = []
        inter_cos = []
        
        for cat_name, cat_concepts in categories.items():
            cat_available = [c for c in cat_concepts if c in last_avg]
            if len(cat_available) >= 2:
                for i in range(len(cat_available)):
                    for j in range(i+1, len(cat_available)):
                        c = cos_sim(last_avg[cat_available[i]], last_avg[cat_available[j]])
                        intra_cos.append(c)
        
        cat_names = list(categories.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                for ci in categories[cat_names[i]]:
                    for cj in categories[cat_names[j]]:
                        if ci in last_avg and cj in last_avg:
                            c = cos_sim(last_avg[ci], last_avg[cj])
                            inter_cos.append(c)
        
        mean_intra = np.mean(intra_cos) if intra_cos else 0
        mean_inter = np.mean(inter_cos) if inter_cos else 0
        sep = mean_inter - mean_intra  # 正=好分离
        
        log(f"\n  Last layer category analysis:")
        log(f"    Intra-category cos: {mean_intra:.4f} (n={len(intra_cos)})")
        log(f"    Inter-category cos: {mean_inter:.4f} (n={len(inter_cos)})")
        log(f"    Separation: {sep:.4f}")
    
    return {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "layer_results": layer_results,
        "best_layer": best,
        "worst_layer": worst,
        "last_layer": {
            "intra_cos": round(float(mean_intra), 4) if intra_cos else None,
            "inter_cos": round(float(mean_inter), 4) if inter_cos else None,
            "separation": round(float(sep), 4) if intra_cos and inter_cos else None,
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    global log
    parser = argparse.ArgumentParser(description="Phase XXVIII")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--skip", type=str, default="", help="Skip experiments, e.g. 'P181,P182'")
    args = parser.parse_args()
    
    model_name = args.model
    skip_set = set(args.skip.split(",")) if args.skip else set()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = _Path(f"d:/develop/TransformerLens-main/tests/glm5_temp/stage733_phase28_{model_name}_{timestamp}")
    log = Logger(str(log_dir), f"phase28_{model_name}")
    
    log(f"\n{'='*70}")
    log(f"Phase XXVIII: Stage 733 — {model_name}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log dir: {log_dir}")
    log(f"Skip: {skip_set}")
    log(f"{'='*70}")
    
    all_results = {"model": model_name, "timestamp": timestamp}
    
    try:
        # P181: 跨模型全层EMB→HS变换
        if "P181" not in skip_set:
            t0 = time.time()
            r = run_p181(model_name)
            all_results["P181"] = r
            log(f"\n  P181 done in {time.time()-t0:.1f}s")
            del r; gc.collect(); torch.cuda.empty_cache()
        
        # P182: 跨模型推理跃迁层
        if "P182" not in skip_set:
            t0 = time.time()
            r = run_p182(model_name)
            all_results["P182"] = r
            log(f"\n  P182 done in {time.time()-t0:.1f}s")
            del r; gc.collect(); torch.cuda.empty_cache()
        
        # P183: SAE稀疏分解
        if "P183" not in skip_set:
            t0 = time.time()
            r = run_p183(model_name)
            all_results["P183"] = r
            log(f"\n  P183 done in {time.time()-t0:.1f}s")
            del r; gc.collect(); torch.cuda.empty_cache()
        
        # P184: 单维度语义分析
        if "P184" not in skip_set:
            t0 = time.time()
            r = run_p184(model_name)
            all_results["P184"] = r
            log(f"\n  P184 done in {time.time()-t0:.1f}s")
            del r; gc.collect(); torch.cuda.empty_cache()
        
        # P185: 跨模型概念分离层对比
        if "P185" not in skip_set:
            t0 = time.time()
            r = run_p185(model_name)
            all_results["P185"] = r
            log(f"\n  P185 done in {time.time()-t0:.1f}s")
            del r; gc.collect(); torch.cuda.empty_cache()
        
        # 保存结果
        results_path = log_dir / f"results_phase28_{model_name}.json"
        # 简化: 移除大矩阵数据
        def simplify(obj, max_depth=3):
            if max_depth <= 0: return str(obj)
            if isinstance(obj, dict):
                return {k: simplify(v, max_depth-1) for k, v in list(obj.items())[:50]}
            if isinstance(obj, (list, tuple)):
                return [simplify(v, max_depth-1) for v in obj[:50]]
            if isinstance(obj, np.ndarray):
                return obj.tolist()[:20]
            if isinstance(obj, torch.Tensor):
                return obj.tolist()[:20]
            return obj
        
        simple_results = simplify(all_results)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(simple_results, f, indent=2, ensure_ascii=False)
        log(f"\n  Results saved to {results_path}")
    
    except Exception as e:
        log(f"\n  ERROR: {e}")
        import traceback
        log(traceback.format_exc())
    
    log("\n" + "="*70)
    log(f"Phase XXVIII COMPLETE — {model_name} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)

if __name__ == "__main__":
    main()
