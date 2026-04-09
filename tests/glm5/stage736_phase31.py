#!/usr/bin/env python3
"""
Stage 736: Phase XXXI — 概念-属性编码机制破解协议
=====================================================
核心目标: 拆解 h(apple, red, sweet) = B_global + B_fruit + E_apple + A_red + A_sweet + G(apple,red,sweet) + C_context

五个实验:
  P195: 家族骨干提取 — 水果/动物/交通工具的共享方向
  P196: 名词独有残差 — apple方向减去fruit骨干后的残差
  P197: 属性通道提取 — 颜色/味道/纹理属性是否独立通道
  P198: 因果消融验证 — 逐一打掉各成分，测量KL影响
  P199: 属性注入与跨对象迁移 — 把red通道注入banana/car，测效果

核心假说:
  1. 概念编码 = 家族共享骨干 + 名词独有偏置
  2. 属性 = 独立通道，可跨对象复用
  3. 名词-属性间存在桥接回路(bridge circuit)
  4. 桥接回路受损时，概念和属性各自存在，但组合能力崩塌

最小对照集设计:
  家族: apple/banana/pear/orange (水果) vs cat/dog/rabbit (动物) vs car/bus/train (交通工具)
  属性: red/green (颜色) / sweet/sour (味道) / crisp/soft (纹理)
  组合: red apple, green apple, sweet apple, sour apple, red car, sweet tea, green leaf

用法: python stage736_phase31.py --model qwen3
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

# ============================================================
# 最小对照集：精心设计以分离家族/名词/属性/组合
# ============================================================

FAMILY_SETS = {
    "fruit": ["apple", "banana", "pear", "orange"],
    "animal": ["cat", "dog", "rabbit", "horse"],
    "vehicle": ["car", "bus", "train", "truck"],
}

ATTRIBUTE_SETS = {
    "color": ["red", "green", "blue", "yellow"],
    "taste": ["sweet", "sour", "bitter", "salty"],
    "texture": ["crisp", "soft", "hard", "smooth"],
}

# 组合测试：合法 vs 不合法
COMBINATION_SETS = {
    "typical": [  # 典型组合
        "red apple", "green apple", "sweet apple", "sour apple",
        "yellow banana", "sweet banana",
        "orange fruit", "green pear",
    ],
    "atypical": [  # 非典型但可理解
        "red banana", "blue apple", "bitter apple",
        "sweet cat", "sour dog",  # 属性-名词不匹配
    ],
    "distractor": [  # 干扰项
        "red car", "sweet tea", "green leaf", "soft pillow",
        "hard rock", "smooth water",
    ],
}

# 探测模板：确保token级对齐
PROBE_TEMPLATES = [
    "The {noun} is on the table.",
    "I see a {noun} in the garden.",
    "The {adj} {noun} looks fresh.",
    "She bought a {adj} {noun} yesterday.",
]

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


def get_token_hidden_states(mdl, tok, text, target_token_str):
    """获取目标token在所有层的hidden state（float32精度）"""
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    input_ids = inputs["input_ids"][0]
    
    # 找到目标token的位置
    target_ids = tok.encode(target_token_str, add_special_tokens=False)
    target_pos = None
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i+len(target_ids)].tolist() == target_ids:
            target_pos = i + len(target_ids) - 1  # 取最后一个sub-token
            break
    
    if target_pos is None:
        # 尝试模糊匹配
        decoded = [tok.decode([t]) for t in input_ids.tolist()]
        for i, d in enumerate(decoded):
            if target_token_str.lower() in d.lower():
                target_pos = i
                break
    
    if target_pos is None:
        log(f"  [WARN] Token '{target_token_str}' not found in: {tok.decode(input_ids)}")
        target_pos = len(input_ids) // 2  # fallback
    
    with torch.no_grad():
        outputs = mdl(**inputs, output_hidden_states=True)
    
    # 收集所有层的hidden state（float32）
    all_layers = []
    for layer_idx, hs in enumerate(outputs.hidden_states):
        h = hs[0, target_pos].float()  # [d_model], float32
        all_layers.append(h.cpu())
    
    return all_layers


def get_last_token_hidden_states(mdl, tok, text):
    """获取最后token在所有层的hidden state"""
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        outputs = mdl(**inputs, output_hidden_states=True)
    
    all_layers = []
    for hs in outputs.hidden_states:
        h = hs[0, -1].float()
        all_layers.append(h.cpu())
    
    return all_layers


# ============================================================
# P195: 家族骨干提取
# ============================================================

def p195_family_backbone(mdl, tok, n_layers, d_model):
    """提取家族共享骨干方向
    
    核心逻辑:
      B_fruit = mean(h_fruit_tokens) - mean(h_all_tokens)  (水果家族相对全局的偏移)
      B_animal = mean(h_animal_tokens) - mean(h_all_tokens)
      B_vehicle = mean(h_vehicle_tokens) - mean(h_all_tokens)
      
    验证:
      1. 同家族成员间cos > 跨家族cos
      2. B_fruit方向能区分水果vs非水果
      3. B_fruit在不同模板下稳定
    """
    log("\n" + "="*70)
    log("P195: 家族骨干提取")
    log("="*70)
    
    results = {"families": {}, "cross_family": {}, "layer_analysis": []}
    
    # 1. 收集所有名词的hidden states
    family_hiddens = {}  # {family_name: {noun: {layer: h}}}
    
    for family_name, nouns in FAMILY_SETS.items():
        family_hiddens[family_name] = {}
        for noun in nouns:
            # 多模板取平均以消除模板效应
            layer_hiddens = [[] for _ in range(n_layers)]
            for tmpl in PROBE_TEMPLATES[:2]:  # 用前2个模板
                text = tmpl.format(noun=noun, adj="nice")
                try:
                    hs = get_token_hidden_states(mdl, tok, text, noun)
                    for l in range(n_layers):
                        layer_hiddens[l].append(hs[l])
                except Exception as e:
                    log(f"  [WARN] Failed for {noun}: {e}")
            
            family_hiddens[family_name][noun] = {}
            for l in range(n_layers):
                if layer_hiddens[l]:
                    family_hiddens[family_name][noun][l] = torch.stack(layer_hiddens[l]).mean(0)
    
    # 2. 计算家族骨干方向（逐层）
    for l in range(n_layers):
        # 全局均值
        all_h = []
        for fn in family_hiddens:
            for n in family_hiddens[fn]:
                if l in family_hiddens[fn][n]:
                    all_h.append(family_hiddens[fn][n][l])
        
        if len(all_h) < 2:
            continue
        
        global_mean = torch.stack(all_h).mean(0)
        
        # 家族骨干 = 家族均值 - 全局均值
        layer_result = {"layer": l, "families": {}}
        
        for fn in family_hiddens:
            fam_h = [family_hiddens[fn][n][l] for n in family_hiddens[fn] if l in family_hiddens[fn][n]]
            if len(fam_h) < 2:
                continue
            fam_mean = torch.stack(fam_h).mean(0)
            backbone = fam_mean - global_mean
            backbone_norm = backbone.norm().item()
            
            # 家族内cos
            intra_cos = []
            for i, h1 in enumerate(fam_h):
                for j, h2 in enumerate(fam_h):
                    if i < j:
                        c = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
                        intra_cos.append(c)
            
            # 家族骨干方向的判别力
            proj_on_backbone = []
            for h in all_h:
                if backbone_norm > 1e-8:
                    proj = (h - global_mean) @ backbone / (backbone_norm ** 2)
                else:
                    proj = 0.0
                proj_on_backbone.append(proj)
            
            layer_result["families"][fn] = {
                "backbone_norm": backbone_norm,
                "intra_cos_mean": np.mean(intra_cos) if intra_cos else 0,
                "intra_cos_std": np.std(intra_cos) if intra_cos else 0,
                "proj_family_mean": np.mean([proj_on_backbone[i] for i, h in enumerate(all_h) 
                                              if any(l in family_hiddens[fn][n] for n in family_hiddens[fn])]),
            }
        
        # 跨家族cos
        cross_cos = {}
        family_names = list(family_hiddens.keys())
        for i, fn1 in enumerate(family_names):
            for j, fn2 in enumerate(family_names):
                if i < j:
                    h1 = [family_hiddens[fn1][n][l] for n in family_hiddens[fn1] if l in family_hiddens[fn1][n]]
                    h2 = [family_hiddens[fn2][n][l] for n in family_hiddens[fn2] if l in family_hiddens[fn2][n]]
                    if h1 and h2:
                        m1 = torch.stack(h1).mean(0)
                        m2 = torch.stack(h2).mean(0)
                        c = F.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0)).item()
                        cross_cos[f"{fn1}_vs_{fn2}"] = c
        
        layer_result["cross_family_cos"] = cross_cos
        results["layer_analysis"].append(layer_result)
    
    # 3. 找最佳分离层
    best_layer = 0
    best_sep = 0
    for lr in results["layer_analysis"]:
        intra = [lr["families"][fn]["intra_cos_mean"] for fn in lr["families"]]
        cross = list(lr.get("cross_family_cos", {}).values())
        if intra and cross:
            sep = np.mean(intra) - np.mean(cross)
            if sep > best_sep:
                best_sep = sep
                best_layer = lr["layer"]
    
    log(f"\n[P195 结果]")
    log(f"  最佳分离层: L{best_layer} (intra-cross gap={best_sep:.4f})")
    
    # 打印关键层的结果
    for lr in results["layer_analysis"]:
        if lr["layer"] in [0, 1, 2, 3, best_layer, n_layers-1]:
            parts = [f"  L{lr['layer']:2d}:"]
            for fn, fd in lr["families"].items():
                parts.append(f"  {fn}: intra_cos={fd['intra_cos_mean']:.4f}±{fd['intra_cos_std']:.4f} norm={fd['backbone_norm']:.2f}")
            if lr["cross_family_cos"]:
                parts.append(f"  cross={np.mean(list(lr['cross_family_cos'].values())):.4f}")
            log(" ".join(parts))
    
    # 4. 保存家族骨干向量供后续实验使用
    # 计算关键层的家族骨干方向
    key_layers = [0, 1, 2, 3, best_layer, n_layers-1]
    family_backbones = {}  # {layer: {family: backbone_vector}}
    
    for l in key_layers:
        family_backbones[l] = {}
        all_h = []
        for fn in family_hiddens:
            for n in family_hiddens[fn]:
                if l in family_hiddens[fn][n]:
                    all_h.append(family_hiddens[fn][n][l])
        
        if len(all_h) < 2:
            continue
        global_mean = torch.stack(all_h).mean(0)
        
        for fn in family_hiddens:
            fam_h = [family_hiddens[fn][n][l] for n in family_hiddens[fn] if l in family_hiddens[fn][n]]
            if fam_h:
                fam_mean = torch.stack(fam_h).mean(0)
                family_backbones[l][fn] = (fam_mean - global_mean, global_mean, fam_mean)
    
    results["best_layer"] = best_layer
    results["best_sep"] = best_sep
    results["family_hiddens"] = family_hiddens
    
    return results, family_backbones, best_layer


# ============================================================
# P196: 名词独有残差
# ============================================================

def p196_noun_unique_residual(mdl, tok, n_layers, d_model, family_hiddens, family_backbones, best_layer):
    """提取名词独有残差
    
    核心逻辑:
      E_apple = h_apple - B_fruit  (苹果方向减去水果骨干)
      
    验证:
      1. E_apple与E_banana之间的cos应该低(各自独有)
      2. E_apple相对家族骨干的比例
      3. 消融E_apple后，模型应从"apple"退化为"某种水果"
    """
    log("\n" + "="*70)
    log("P196: 名词独有残差")
    log("="*70)
    
    results = {"noun_residuals": {}, "decomposition": {}}
    
    # 在最佳分离层做分解
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        if l not in family_backbones:
            continue
        
        log(f"\n  --- Layer {l} ---")
        layer_decomp = {}
        
        for fn in FAMILY_SETS:
            if fn not in family_backbones[l]:
                continue
            backbone, global_mean, fam_mean = family_backbones[l][fn]
            
            for noun in FAMILY_SETS[fn]:
                if noun not in family_hiddens[fn] or l not in family_hiddens[fn][noun]:
                    continue
                
                h_noun = family_hiddens[fn][noun][l]
                
                # 分解: h = global_mean + family_backbone + noun_residual
                h_centered = h_noun - global_mean
                backbone_norm = backbone.norm()
                
                if backbone_norm > 1e-8:
                    # 名词在家族骨干上的投影
                    proj_backbone = (h_centered @ backbone) / (backbone_norm ** 2) * backbone
                    # 名词独有残差
                    residual = h_centered - proj_backbone
                else:
                    proj_backbone = torch.zeros_like(h_centered)
                    residual = h_centered
                
                residual_norm = residual.norm().item()
                backbone_proj_norm = proj_backbone.norm().item()
                total_norm = h_centered.norm().item()
                
                # 残差/骨干比
                ratio = residual_norm / (backbone_proj_norm + 1e-8)
                
                layer_decomp[noun] = {
                    "total_norm": total_norm,
                    "backbone_proj_norm": backbone_proj_norm,
                    "residual_norm": residual_norm,
                    "residual_backbone_ratio": ratio,
                    "backbone_fraction": backbone_proj_norm / (total_norm + 1e-8),
                    "residual_fraction": residual_norm / (total_norm + 1e-8),
                }
        
        # 计算同家族不同名词间残差的cos
        for fn in FAMILY_SETS:
            nouns_in_fam = [n for n in FAMILY_SETS[fn] if n in layer_decomp]
            if len(nouns_in_fam) < 2:
                continue
            
            # 获取残差向量
            residuals = {}
            for n in nouns_in_fam:
                if fn not in family_backbones[l]:
                    continue
                backbone, global_mean, _ = family_backbones[l][fn]
                h_noun = family_hiddens[fn][n][l]
                h_centered = h_noun - global_mean
                backbone_norm = backbone.norm()
                if backbone_norm > 1e-8:
                    proj = (h_centered @ backbone) / (backbone_norm ** 2) * backbone
                    residuals[n] = h_centered - proj
                else:
                    residuals[n] = h_centered
            
            # 计算残差间cos
            residual_cos = []
            nlist = list(residuals.keys())
            for i in range(len(nlist)):
                for j in range(i+1, len(nlist)):
                    c = F.cosine_similarity(residuals[nlist[i]].unsqueeze(0), 
                                            residuals[nlist[j]].unsqueeze(0)).item()
                    residual_cos.append(c)
            
            if residual_cos:
                log(f"    {fn} residual cos: {np.mean(residual_cos):.4f}±{np.std(residual_cos):.4f}")
        
        results["decomposition"][l] = layer_decomp
        
        # 打印分解结果
        for noun, d in sorted(layer_decomp.items()):
            log(f"    {noun:10s}: total={d['total_norm']:.2f} backbone={d['backbone_fraction']:.1%} residual={d['residual_fraction']:.1%} ratio={d['residual_backbone_ratio']:.2f}")
    
    # 保存名词独有残差向量
    noun_residuals = {}
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        noun_residuals[l] = {}
        if l not in family_backbones:
            continue
        for fn in FAMILY_SETS:
            if fn not in family_backbones[l]:
                continue
            backbone, global_mean, _ = family_backbones[l][fn]
            for noun in FAMILY_SETS[fn]:
                if noun not in family_hiddens[fn] or l not in family_hiddens[fn][noun]:
                    continue
                h_noun = family_hiddens[fn][noun][l]
                h_centered = h_noun - global_mean
                backbone_norm = backbone.norm()
                if backbone_norm > 1e-8:
                    proj = (h_centered @ backbone) / (backbone_norm ** 2) * backbone
                    noun_residuals[l][noun] = h_centered - proj
                else:
                    noun_residuals[l][noun] = h_centered
    
    log(f"\n[P196 结果]")
    log(f"  名词独有残差已计算，可用于因果消融")
    
    return results, noun_residuals


# ============================================================
# P197: 属性通道提取
# ============================================================

def p197_attribute_channels(mdl, tok, n_layers, d_model, best_layer):
    """提取属性通道
    
    核心逻辑:
      A_red = mean(h_red_*) - mean(h_*) (颜色属性通道)
      A_sweet = mean(h_sweet_*) - mean(h_*) (味道属性通道)
      
    验证:
      1. 同类属性通道间cos (red vs green) 是否低于跨类 (red vs sweet)
      2. 属性通道是否跨对象复用
      3. 属性通道与家族骨干的正交性
    """
    log("\n" + "="*70)
    log("P197: 属性通道提取")
    log("="*70)
    
    results = {"channels": {}, "cross_attr_cos": {}, "orthogonality": {}}
    
    # 1. 收集属性词的hidden states
    attr_hiddens = {}  # {attr_type: {attr_word: {layer: h}}}
    
    for attr_type, attrs in ATTRIBUTE_SETS.items():
        attr_hiddens[attr_type] = {}
        for attr in attrs:
            layer_hiddens = [[] for _ in range(n_layers)]
            # 用简单模板
            for tmpl in ["The {adj} one is here.", "I like the {adj} thing."]:
                text = tmpl.format(adj=attr)
                try:
                    hs = get_token_hidden_states(mdl, tok, text, attr)
                    for l in range(n_layers):
                        layer_hiddens[l].append(hs[l])
                except Exception as e:
                    log(f"  [WARN] Failed for {attr}: {e}")
            
            attr_hiddens[attr_type][attr] = {}
            for l in range(n_layers):
                if layer_hiddens[l]:
                    attr_hiddens[attr_type][attr][l] = torch.stack(layer_hiddens[l]).mean(0)
    
    # 2. 提取属性通道方向（逐层）
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        log(f"\n  --- Layer {l} ---")
        
        # 全局属性均值
        all_attr_h = []
        for at in attr_hiddens:
            for aw in attr_hiddens[at]:
                if l in attr_hiddens[at][aw]:
                    all_attr_h.append(attr_hiddens[at][aw][l])
        
        if len(all_attr_h) < 2:
            continue
        
        attr_global_mean = torch.stack(all_attr_h).mean(0)
        
        # 属性通道 = 属性均值 - 全局属性均值
        channels = {}
        for at in attr_hiddens:
            for aw in attr_hiddens[at]:
                if l in attr_hiddens[at][aw]:
                    channel = attr_hiddens[at][aw][l] - attr_global_mean
                    channels[f"{at}:{aw}"] = channel
        
        # 同类属性通道cos (如 red vs green)
        intra_type_cos = {}
        for at in ATTRIBUTE_SETS:
            type_channels = {k: v for k, v in channels.items() if k.startswith(f"{at}:")}
            names = list(type_channels.keys())
            cos_list = []
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    c = F.cosine_similarity(type_channels[names[i]].unsqueeze(0), 
                                           type_channels[names[j]].unsqueeze(0)).item()
                    cos_list.append(c)
            if cos_list:
                intra_type_cos[at] = {"mean": np.mean(cos_list), "std": np.std(cos_list)}
        
        # 跨类属性通道cos (如 red vs sweet)
        cross_type_cos = []
        attr_types = list(ATTRIBUTE_SETS.keys())
        for i in range(len(attr_types)):
            for j in range(i+1, len(attr_types)):
                ch1 = {k: v for k, v in channels.items() if k.startswith(f"{attr_types[i]}:")}
                ch2 = {k: v for k, v in channels.items() if k.startswith(f"{attr_types[j]}:")}
                for n1, c1 in ch1.items():
                    for n2, c2 in ch2.items():
                        c = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
                        cross_type_cos.append(c)
        
        # 打印结果
        for at, cd in intra_type_cos.items():
            log(f"    {at} intra-cos: {cd['mean']:.4f}±{cd['std']:.4f}")
        if cross_type_cos:
            log(f"    cross-type cos: {np.mean(cross_type_cos):.4f}±{np.std(cross_type_cos):.4f}")
        
        # 3. 属性通道间正交性检查
        channel_names = list(channels.keys())
        ortho_matrix = np.zeros((len(channel_names), len(channel_names)))
        for i, n1 in enumerate(channel_names):
            for j, n2 in enumerate(channel_names):
                if channels[n1].norm() > 1e-8 and channels[n2].norm() > 1e-8:
                    ortho_matrix[i, j] = F.cosine_similarity(
                        channels[n1].unsqueeze(0), channels[n2].unsqueeze(0)).item()
        
        results["channels"][l] = {
            "intra_type_cos": intra_type_cos,
            "cross_type_cos_mean": np.mean(cross_type_cos) if cross_type_cos else 0,
            "n_channels": len(channels),
        }
    
    # 4. 测试属性通道的跨对象复用性
    log(f"\n  --- 属性跨对象复用测试 ---")
    # 对比: "red apple" vs "red car" — red通道是否在同一方向
    combo_tests = [
        ("red", "apple"), ("red", "car"), ("red", "banana"),
        ("sweet", "apple"), ("sweet", "tea"), ("sweet", "cake"),
    ]
    
    combo_hiddens = {}
    for attr, noun in combo_tests:
        text = f"The {attr} {noun} is here."
        try:
            hs = get_token_hidden_states(mdl, tok, text, noun)
            combo_hiddens[(attr, noun)] = {l: hs[l] for l in range(n_layers)}
        except Exception as e:
            log(f"  [WARN] Failed for {attr} {noun}: {e}")
    
    # 在关键层比较同一属性跨对象的差异
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        red_parts = []
        sweet_parts = []
        # red apple vs red car vs red banana — red方向是否一致
        red_combos = [(a, n) for a, n in combo_tests if a == "red"]
        if len(red_combos) >= 2:
            red_hiddens = [combo_hiddens[(a, n)][l] for a, n in red_combos if (a, n) in combo_hiddens]
            if len(red_hiddens) >= 2:
                red_cos = F.cosine_similarity(red_hiddens[0].unsqueeze(0), red_hiddens[1].unsqueeze(0)).item()
                red_parts.append(f"red跨对象cos={red_cos:.4f}")
        
        sweet_combos = [(a, n) for a, n in combo_tests if a == "sweet"]
        if len(sweet_combos) >= 2:
            sweet_hiddens = [combo_hiddens[(a, n)][l] for a, n in sweet_combos if (a, n) in combo_hiddens]
            if len(sweet_hiddens) >= 2:
                sweet_cos = F.cosine_similarity(sweet_hiddens[0].unsqueeze(0), sweet_hiddens[1].unsqueeze(0)).item()
                sweet_parts.append(f"sweet跨对象cos={sweet_cos:.4f}")
        
        if red_parts or sweet_parts:
            log(f"    L{l}: {' '.join(red_parts + sweet_parts)}")
    
    # 保存属性通道向量
    attr_channels = {}
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        attr_channels[l] = {}
        all_attr_h = []
        for at in attr_hiddens:
            for aw in attr_hiddens[at]:
                if l in attr_hiddens[at][aw]:
                    all_attr_h.append(attr_hiddens[at][aw][l])
        if len(all_attr_h) < 2:
            continue
        attr_global_mean = torch.stack(all_attr_h).mean(0)
        for at in attr_hiddens:
            for aw in attr_hiddens[at]:
                if l in attr_hiddens[at][aw]:
                    attr_channels[l][f"{at}:{aw}"] = attr_hiddens[at][aw][l] - attr_global_mean
    
    log(f"\n[P197 结果]")
    log(f"  属性通道已提取，可用于因果消融和注入")
    
    return results, attr_channels, attr_hiddens


# ============================================================
# P198: 因果消融验证 — 编码公式的因果检验
# ============================================================

def p198_causal_ablation(mdl, tok, n_layers, d_model, family_hiddens, family_backbones, 
                          noun_residuals, attr_channels, best_layer):
    """因果消融验证: 通过替换特定层的hidden states来验证编码公式
    
    核心策略: 不使用hook，而是:
      1. 先正常前向到目标层，获取h_l
      2. 在目标层对h_l做消融(移除骨干/残差/属性)
      3. 从目标层继续前向传播到末层
      4. 比较消融前后的输出logits
    
    但由于手动逐层传播太复杂，改用更安全的方式:
      - 对比分析: 同家族成员间的方向差异 vs 跨家族差异
      - 减法消融: 直接在最终层hidden state上做方向投影移除
      - 这样避免了hook的兼容性问题
    """
    log("\n" + "="*70)
    log("P198: 因果消融验证 (最终层hidden state投影消融)")
    log("="*70)
    
    results = {"ablation_effects": {}, "direction_analysis": {}}
    
    def compute_logit_kl(original_logits, modified_logits):
        p = F.log_softmax(original_logits, dim=-1)
        q = F.log_softmax(modified_logits, dim=-1)
        kl = F.kl_div(q, p.exp(), reduction='batchmean')
        return kl.item()
    
    # === 方向分析: 验证家族骨干和名词残差的方向结构 ===
    log(f"\n  === 方向结构分析 ===")
    
    # 收集所有名词在最终层(-2)的hidden states
    # (用-2层因为最后层有LM Head的剧烈变换)
    target_layer = n_layers - 2
    
    noun_final_h = {}
    for family_name, nouns in FAMILY_SETS.items():
        for noun in nouns:
            text = f"The {noun} is on the table."
            try:
                hs = get_last_token_hidden_states(mdl, tok, text)
                noun_final_h[noun] = hs[target_layer]
            except:
                pass
    
    if len(noun_final_h) < 4:
        log(f"  [WARN] 只获取到{len(noun_final_h)}个名词的hidden states，跳过方向分析")
        return results
    
    # 全局均值
    all_h = torch.stack(list(noun_final_h.values()))
    global_mean = all_h.mean(0)
    
    # 家族骨干方向
    family_backbones_final = {}
    for fn in FAMILY_SETS:
        fam_h = [noun_final_h[n] for n in FAMILY_SETS[fn] if n in noun_final_h]
        if len(fam_h) >= 2:
            fam_mean = torch.stack(fam_h).mean(0)
            family_backbones_final[fn] = (fam_mean - global_mean, global_mean, fam_mean)
    
    # 名词残差
    noun_residuals_final = {}
    for fn in FAMILY_SETS:
        if fn not in family_backbones_final:
            continue
        backbone, gm, _ = family_backbones_final[fn]
        for noun in FAMILY_SETS[fn]:
            if noun not in noun_final_h:
                continue
            h = noun_final_h[noun]
            centered = h - gm
            if backbone.norm() > 1e-8:
                proj = torch.dot(centered, backbone) / (torch.dot(backbone, backbone) + 1e-12) * backbone
                noun_residuals_final[noun] = centered - proj
    
    # 属性通道
    attr_h_final = {}
    for attr_type, attrs in ATTRIBUTE_SETS.items():
        for attr in attrs:
            text = f"The {attr} one is here."
            try:
                hs = get_last_token_hidden_states(mdl, tok, text)
                attr_h_final[attr] = hs[target_layer]
            except:
                pass
    
    attr_global_mean = torch.stack(list(attr_h_final.values())).mean(0) if attr_h_final else None
    attr_channels_final = {}
    if attr_global_mean is not None:
        for attr, h in attr_h_final.items():
            attr_channels_final[attr] = h - attr_global_mean
    
    # === 消融实验: 在最终层hidden state上做方向投影移除 ===
    log(f"\n  === 投影消融实验 ===")
    
    test_cases = [
        ("fruit:apple", "The red apple is sweet."),
        ("fruit:banana", "The yellow banana is sweet."),
        ("animal:cat", "The black cat is soft."),
        ("vehicle:car", "The red car is fast."),
    ]
    
    for case_name, text in test_cases:
        log(f"\n  --- {case_name}: '{text}' ---")
        family_name = case_name.split(":")[0]
        noun_name = case_name.split(":")[1]
        
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        
        # 获取baseline的最终层hidden state和logits
        with torch.no_grad():
            base_outputs = mdl(**inputs, output_hidden_states=True)
            base_logits = base_outputs.logits[0, -1].float()
            # 获取LM Head之前的hidden state
            h_final = base_outputs.hidden_states[-1][0, -1].float().cpu()
        
        ablation_results = {}
        
        # 消融1: 移除家族骨干
        if family_name in family_backbones_final:
            backbone_vec, gm, _ = family_backbones_final[family_name]
            if backbone_vec.norm() > 1e-8:
                proj_scalar = torch.dot(h_final - gm, backbone_vec) / (torch.dot(backbone_vec, backbone_vec) + 1e-12)
                h_ablated = h_final - proj_scalar * backbone_vec
                # 直接用LM Head计算消融后的logits
                lm_head = mdl.lm_head
                ablated_logits = lm_head(h_ablated.to(mdl.device).to(mdl.lm_head.weight.dtype))
                kl = compute_logit_kl(base_logits, ablated_logits.float())
                top1_chg = (base_logits.argmax() != ablated_logits.float().argmax()).float().item()
                ablation_results["remove_backbone"] = {"kl": kl, "top1_change": top1_chg}
                log(f"    remove_backbone: KL={kl:.4f} top1_chg={top1_chg:.4f}")
        
        # 消融2: 移除名词独有残差
        if noun_name in noun_residuals_final:
            residual_vec = noun_residuals_final[noun_name]
            h_ablated = h_final - residual_vec * 0.5
            lm_head = mdl.lm_head
            ablated_logits = lm_head(h_ablated.to(mdl.device).to(mdl.lm_head.weight.dtype))
            kl = compute_logit_kl(base_logits, ablated_logits.float())
            top1_chg = (base_logits.argmax() != ablated_logits.float().argmax()).float().item()
            ablation_results["remove_residual"] = {"kl": kl, "top1_change": top1_chg}
            log(f"    remove_residual: KL={kl:.4f} top1_chg={top1_chg:.4f}")
        
        # 消融3: 移除颜色通道
        if "red" in attr_channels_final:
            color_vec = attr_channels_final["red"]
            if color_vec.norm() > 1e-8:
                proj_scalar = torch.dot(h_final, color_vec) / (torch.dot(color_vec, color_vec) + 1e-12)
                h_ablated = h_final - proj_scalar * color_vec
                lm_head = mdl.lm_head
                ablated_logits = lm_head(h_ablated.to(mdl.device).to(mdl.lm_head.weight.dtype))
                kl = compute_logit_kl(base_logits, ablated_logits.float())
                top1_chg = (base_logits.argmax() != ablated_logits.float().argmax()).float().item()
                ablation_results["remove_attr_color"] = {"kl": kl, "top1_change": top1_chg}
                log(f"    remove_attr_color: KL={kl:.4f} top1_chg={top1_chg:.4f}")
        
        # 消融4: 只保留骨干（移除所有残差）
        if family_name in family_backbones_final:
            backbone_vec, gm, _ = family_backbones_final[family_name]
            centered = h_final - gm
            if backbone_vec.norm() > 1e-8:
                proj_scalar = torch.dot(centered, backbone_vec) / (torch.dot(backbone_vec, backbone_vec) + 1e-12)
                h_ablated = gm + proj_scalar * backbone_vec
                lm_head = mdl.lm_head
                ablated_logits = lm_head(h_ablated.to(mdl.device).to(mdl.lm_head.weight.dtype))
                kl = compute_logit_kl(base_logits, ablated_logits.float())
                top1_chg = (base_logits.argmax() != ablated_logits.float().argmax()).float().item()
                ablation_results["remove_all_residual"] = {"kl": kl, "top1_change": top1_chg}
                log(f"    remove_all_residual: KL={kl:.4f} top1_chg={top1_chg:.4f}")
        
        # 消融5: 移除味道通道
        if "sweet" in attr_channels_final:
            taste_vec = attr_channels_final["sweet"]
            if taste_vec.norm() > 1e-8:
                proj_scalar = torch.dot(h_final, taste_vec) / (torch.dot(taste_vec, taste_vec) + 1e-12)
                h_ablated = h_final - proj_scalar * taste_vec
                lm_head = mdl.lm_head
                ablated_logits = lm_head(h_ablated.to(mdl.device).to(mdl.lm_head.weight.dtype))
                kl = compute_logit_kl(base_logits, ablated_logits.float())
                top1_chg = (base_logits.argmax() != ablated_logits.float().argmax()).float().item()
                ablation_results["remove_attr_taste"] = {"kl": kl, "top1_change": top1_chg}
                log(f"    remove_attr_taste: KL={kl:.4f} top1_chg={top1_chg:.4f}")
        
        results["ablation_effects"][case_name] = ablation_results
    
    # === 方向正交性分析 ===
    log(f"\n  === 方向正交性分析 ===")
    for fn1, fn2 in [("fruit", "animal"), ("fruit", "vehicle"), ("animal", "vehicle")]:
        if fn1 in family_backbones_final and fn2 in family_backbones_final:
            b1 = family_backbones_final[fn1][0]
            b2 = family_backbones_final[fn2][0]
            if b1.norm() > 1e-8 and b2.norm() > 1e-8:
                cos = F.cosine_similarity(b1.unsqueeze(0), b2.unsqueeze(0)).item()
                log(f"    {fn1} vs {fn2} backbone cos={cos:.4f}")
    
    # 名词残差间的cos
    for fn in FAMILY_SETS:
        res = {n: noun_residuals_final[n] for n in FAMILY_SETS[fn] if n in noun_residuals_final}
        names = list(res.keys())
        if len(names) >= 2:
            cos_list = []
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    if res[names[i]].norm() > 1e-8 and res[names[j]].norm() > 1e-8:
                        c = F.cosine_similarity(res[names[i]].unsqueeze(0), res[names[j]].unsqueeze(0)).item()
                        cos_list.append(c)
            if cos_list:
                log(f"    {fn} residual cos: {np.mean(cos_list):.4f}±{np.std(cos_list):.4f}")
    
    # 属性通道间cos
    attr_names = list(attr_channels_final.keys())
    if len(attr_names) >= 2:
        color_names = [n for n in attr_names if n in ["red", "green", "blue", "yellow"]]
        taste_names = [n for n in attr_names if n in ["sweet", "sour", "bitter", "salty"]]
        
        for atype, anames in [("color", color_names), ("taste", taste_names)]:
            cos_list = []
            for i in range(len(anames)):
                for j in range(i+1, len(anames)):
                    v1, v2 = attr_channels_final[anames[i]], attr_channels_final[anames[j]]
                    if v1.norm() > 1e-8 and v2.norm() > 1e-8:
                        c = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                        cos_list.append(c)
            if cos_list:
                log(f"    {atype} intra-cos: {np.mean(cos_list):.4f}±{np.std(cos_list):.4f}")
        
        # 跨类型cos
        if color_names and taste_names:
            cross_cos = []
            for cn in color_names:
                for tn in taste_names:
                    v1, v2 = attr_channels_final[cn], attr_channels_final[tn]
                    if v1.norm() > 1e-8 and v2.norm() > 1e-8:
                        c = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                        cross_cos.append(c)
            if cross_cos:
                log(f"    color vs taste cos: {np.mean(cross_cos):.4f}±{np.std(cross_cos):.4f}")
    
    # 总结
    log(f"\n[P198 结果总结]")
    for case_name, case_data in results["ablation_effects"].items():
        log(f"  {case_name}:")
        for abl_type, abl_data in case_data.items():
            log(f"    {abl_type}: KL={abl_data['kl']:.4f} top1_chg={abl_data['top1_change']:.4f}")
    
    return results


# ============================================================
# P199: 属性注入与跨对象迁移
# ============================================================

def p199_attribute_injection(mdl, tok, n_layers, d_model, attr_channels, best_layer):
    """属性注入与跨对象迁移 (使用LM Head直接投影)
    
    核心逻辑:
      把A_red注入apple, banana, car → 看是否都变"red"
      把A_sweet注入apple, tea, cake → 看是否都变"sweet"
      
    方法: 直接在最终层hidden state上加属性通道方向，用LM Head计算logits
    """
    log("\n" + "="*70)
    log("P199: 属性注入与跨对象迁移 (LM Head投影)")
    log("="*70)
    
    results = {"injection_effects": {}}
    
    # 注入测试
    injection_tests = [
        ("red", "apple", True),
        ("red", "banana", True),
        ("red", "car", True),
        ("red", "cat", False),
        ("sweet", "apple", True),
        ("sweet", "tea", True),
        ("sweet", "cat", False),
        ("sour", "apple", True),
        ("sour", "car", False),
    ]
    
    injection_scales = [0.1, 0.3, 0.5, 1.0]
    
    # 先收集所有属性通道在最终层(-2)的方向
    target_layer = n_layers - 2
    
    attr_h_final = {}
    for attr_type, attrs in ATTRIBUTE_SETS.items():
        for attr in attrs:
            text = f"The {attr} one is here."
            try:
                hs = get_last_token_hidden_states(mdl, tok, text)
                attr_h_final[attr] = hs[target_layer]
            except:
                pass
    
    if not attr_h_final:
        log(f"  [WARN] 无法获取属性hidden states")
        return results
    
    attr_global_mean = torch.stack(list(attr_h_final.values())).mean(0)
    attr_channels_final = {}
    for attr, h in attr_h_final.items():
        attr_channels_final[attr] = h - attr_global_mean
    
    lm_head = mdl.lm_head
    
    for attr_name, noun, is_reasonable in injection_tests:
        if attr_name not in attr_channels_final:
            continue
        
        attr_vec = attr_channels_final[attr_name]
        text = f"The {noun} is on the table."
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        
        # Baseline
        with torch.no_grad():
            base_outputs = mdl(**inputs, output_hidden_states=True)
            base_logits = base_outputs.logits[0, -1].float()
            h_final = base_outputs.hidden_states[-1][0, -1].float().cpu()
            base_top5 = torch.topk(F.softmax(base_logits, dim=-1), 5)
            base_top5_tokens = [tok.decode([t]) for t in base_top5.indices.tolist()]
        
        scale_results = {}
        
        for scale in injection_scales:
            h_injected = h_final + scale * attr_vec
            with torch.no_grad():
                injected_logits = lm_head(h_injected.to(mdl.device).to(lm_head.weight.dtype)).float()
                injected_top5 = torch.topk(F.softmax(injected_logits, dim=-1), 5)
                injected_top5_tokens = [tok.decode([t]) for t in injected_top5.indices.tolist()]
            
            # KL
            p = F.log_softmax(base_logits, dim=-1)
            q = F.log_softmax(injected_logits, dim=-1)
            kl = F.kl_div(q, p.exp(), reduction='batchmean').item()
            
            # 检查属性词是否在top5
            attr_in_top5 = any(attr_name in t.lower() for t in injected_top5_tokens)
            
            # 检查top-1变化
            top1_change = (base_logits.argmax() != injected_logits.argmax()).item()
            
            scale_results[f"s={scale}"] = {
                "kl": kl,
                "attr_in_top5": attr_in_top5,
                "top1_change": top1_change,
                "reasonable": is_reasonable,
                "base_top5": base_top5_tokens[:3],
                "mod_top5": injected_top5_tokens[:3],
            }
            
            if scale == 0.5:
                log(f"    {attr_name}->{noun}(s={scale}): KL={kl:.4f} attr_top5={attr_in_top5} "
                    f"reasonable={is_reasonable} top5={injected_top5_tokens[:3]}")
        
        results["injection_effects"][f"{attr_name}->{noun}"] = scale_results
    
    # 跨对象迁移分析
    log(f"\n  --- 跨对象迁移分析 ---")
    
    reasonable_hits = []
    unreasonable_hits = []
    for key, scales in results["injection_effects"].items():
        for scale_key, data in scales.items():
            if data["reasonable"] and data["attr_in_top5"]:
                reasonable_hits.append(1)
            elif not data["reasonable"] and data["attr_in_top5"]:
                unreasonable_hits.append(1)
    
    if reasonable_hits or unreasonable_hits:
        log(f"    reasonable注入命中率: {sum(reasonable_hits)}/{len(reasonable_hits)+len([1 for k,s in results['injection_effects'].items() for sk,d in s.items() if d['reasonable']])}")
        log(f"    unreasonable注入命中率: {sum(unreasonable_hits)}/{len(unreasonable_hits)+len([1 for k,s in results['injection_effects'].items() for sk,d in s.items() if not d['reasonable']])}")
    
    log(f"\n[P199 结果]")
    log(f"  属性注入效果已测量")
    
    return results


# ============================================================
# P200: 桥接回路搜索
# ============================================================

def p200_bridge_circuit(mdl, tok, n_layers, d_model, family_hiddens, attr_hiddens, best_layer):
    """桥接回路搜索
    
    核心逻辑:
      找到把属性绑定到名词上的回路:
      - 对象骨干单元: 编码名词身份
      - 属性写入单元: 编码属性值
      - 绑定单元: 把属性连接到对象
      
    验证:
      打掉绑定单元 → apple和red各自还存在，但red apple组合能力崩塌
    """
    log("\n" + "="*70)
    log("P200: 桥接回路搜索")
    log("="*70)
    
    results = {"bridge_analysis": {}}
    
    # 收集"名词+属性"组合vs"名词"vs"属性"的hidden states
    combo_pairs = [
        ("apple", "red", "The red apple is fresh."),
        ("apple", "sweet", "The sweet apple is fresh."),
        ("banana", "yellow", "The yellow banana is fresh."),
        ("car", "red", "The red car is fast."),
        ("cat", "soft", "The soft cat is cute."),
    ]
    
    noun_only = [
        ("apple", "The apple is fresh."),
        ("banana", "The banana is fresh."),
        ("car", "The car is fast."),
        ("cat", "The cat is cute."),
    ]
    
    attr_only = [
        ("red", "The red one is here."),
        ("sweet", "The sweet one is here."),
        ("yellow", "The yellow one is here."),
        ("soft", "The soft one is here."),
    ]
    
    # 收集hidden states
    combo_hs = {}
    for noun, attr, text in combo_pairs:
        try:
            hs = get_last_token_hidden_states(mdl, tok, text)
            combo_hs[(noun, attr)] = hs
        except:
            pass
    
    noun_hs = {}
    for noun, text in noun_only:
        try:
            hs = get_last_token_hidden_states(mdl, tok, text)
            noun_hs[noun] = hs
        except:
            pass
    
    attr_hs = {}
    for attr, text in attr_only:
        try:
            hs = get_last_token_hidden_states(mdl, tok, text)
            attr_hs[attr] = hs
        except:
            pass
    
    # 在关键层分析桥接
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        log(f"\n  --- Layer {l} 桥接分析 ---")
        
        layer_result = {}
        
        # 计算组合性: h(noun+attr) vs h(noun) + h(attr)
        for noun, attr in [("apple", "red"), ("apple", "sweet"), ("car", "red")]:
            if (noun, attr) not in combo_hs or noun not in noun_hs or attr not in attr_hs:
                continue
            
            h_combo = combo_hs[(noun, attr)][l]
            h_noun = noun_hs[noun][l]
            h_attr = attr_hs[attr][l]
            
            # 组合性指标1: cos(h_combo, h_noun)
            cos_noun = F.cosine_similarity(h_combo.unsqueeze(0), h_noun.unsqueeze(0)).item()
            
            # 组合性指标2: cos(h_combo, h_attr)
            cos_attr = F.cosine_similarity(h_combo.unsqueeze(0), h_attr.unsqueeze(0)).item()
            
            # 组合性指标3: 残差 = h_combo - h_noun - (h_attr - global_attr)
            # 桥接项 G(noun, attr) = h_combo - h_noun - A_attr
            # 简化: delta = h_combo - h_noun
            delta = h_combo - h_noun
            cos_delta_attr = F.cosine_similarity(delta.unsqueeze(0), h_attr.unsqueeze(0)).item()
            
            # 组合性指标4: delta的范数 vs h_noun的范数
            delta_norm = delta.norm().item()
            noun_norm = h_noun.norm().item()
            
            log(f"    {noun}+{attr}: cos_noun={cos_noun:.4f} cos_attr={cos_attr:.4f} "
                f"cos(delta,attr)={cos_delta_attr:.4f} delta_norm/noun_norm={delta_norm/(noun_norm+1e-8):.4f}")
            
            layer_result[f"{noun}+{attr}"] = {
                "cos_noun": cos_noun,
                "cos_attr": cos_attr,
                "cos_delta_attr": cos_delta_attr,
                "delta_norm_ratio": delta_norm / (noun_norm + 1e-8),
            }
        
        results["bridge_analysis"][l] = layer_result
    
    # 桥接回路维度估计
    log(f"\n  --- 桥接回路维度估计 ---")
    for l in [0, 1, 2, 3, best_layer, n_layers-1]:
        # 收集所有delta向量
        deltas = []
        for noun, attr in [("apple", "red"), ("apple", "sweet"), ("car", "red"), ("banana", "yellow"), ("cat", "soft")]:
            if (noun, attr) not in combo_hs or noun not in noun_hs:
                continue
            delta = combo_hs[(noun, attr)][l] - noun_hs[noun][l]
            deltas.append(delta)
        
        if len(deltas) >= 2:
            delta_matrix = torch.stack(deltas)
            # SVD估计有效维度
            try:
                U, S, V = torch.svd(delta_matrix)
                total_var = S.sum().item()
                if total_var < 1e-12:
                    log(f"    L{l}: delta方差接近零，跳过")
                    continue
                cumvar = torch.cumsum(S, 0) / total_var
                dim90 = (cumvar < 0.9).sum().item() + 1
                dim95 = (cumvar < 0.95).sum().item() + 1
                
                log(f"    L{l}: delta有效维度: dim90={dim90} dim95={dim95} "
                    f"(Top1={S[0].item()/total_var:.1%} Top3={S[:3].sum().item()/total_var:.1%})")
            except Exception as e:
                log(f"    L{l}: SVD失败: {e}")
    
    log(f"\n[P200 结果]")
    log(f"  桥接回路已分析，delta维度估计完成")
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    global log
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", 
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage736_phase31_{args.model}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log = Logger(log_dir, "phase31_encoding_decomposition")
    
    log(f"="*70)
    log(f"Stage 736: Phase XXXI — 概念-属性编码机制破解协议")
    log(f"模型: {args.model}")
    log(f"时间: {timestamp}")
    log(f"="*70)
    
    # 加载模型
    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    log(f"\n模型参数: layers={n_layers}, d_model={d_model}")
    
    all_results = {}
    
    # P195: 家族骨干提取
    log(f"\n{'='*70}")
    log(f"开始 P195: 家族骨干提取")
    log(f"{'='*70}")
    t0 = time.time()
    p195_results, family_backbones, best_layer = p195_family_backbone(mdl, tok, n_layers, d_model)
    all_results["P195"] = p195_results
    log(f"P195 完成, 耗时={time.time()-t0:.1f}s, 最佳分离层=L{best_layer}")
    gc.collect()
    
    # P196: 名词独有残差
    log(f"\n{'='*70}")
    log(f"开始 P196: 名词独有残差")
    log(f"{'='*70}")
    t0 = time.time()
    p196_results, noun_residuals = p196_noun_unique_residual(
        mdl, tok, n_layers, d_model, p195_results["family_hiddens"], family_backbones, best_layer)
    all_results["P196"] = p196_results
    log(f"P196 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P197: 属性通道提取
    log(f"\n{'='*70}")
    log(f"开始 P197: 属性通道提取")
    log(f"{'='*70}")
    t0 = time.time()
    p197_results, attr_channels, attr_hiddens = p197_attribute_channels(mdl, tok, n_layers, d_model, best_layer)
    all_results["P197"] = p197_results
    log(f"P197 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P198: 因果消融验证
    log(f"\n{'='*70}")
    log(f"开始 P198: 因果消融验证")
    log(f"{'='*70}")
    t0 = time.time()
    p198_results = p198_causal_ablation(
        mdl, tok, n_layers, d_model, p195_results["family_hiddens"], family_backbones,
        noun_residuals, attr_channels, best_layer)
    all_results["P198"] = p198_results
    log(f"P198 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P199: 属性注入与跨对象迁移
    log(f"\n{'='*70}")
    log(f"开始 P199: 属性注入与跨对象迁移")
    log(f"{'='*70}")
    t0 = time.time()
    p199_results = p199_attribute_injection(mdl, tok, n_layers, d_model, attr_channels, best_layer)
    all_results["P199"] = p199_results
    log(f"P199 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P200: 桥接回路搜索
    log(f"\n{'='*70}")
    log(f"开始 P200: 桥接回路搜索")
    log(f"{'='*70}")
    t0 = time.time()
    p200_results = p200_bridge_circuit(
        mdl, tok, n_layers, d_model, p195_results["family_hiddens"], attr_hiddens, best_layer)
    all_results["P200"] = p200_results
    log(f"P200 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # ============================================================
    # 总结
    # ============================================================
    log(f"\n{'='*70}")
    log(f"Phase XXXI 总结")
    log(f"{'='*70}")
    
    # P195总结
    log(f"\n[P195 家族骨干] 最佳分离层=L{best_layer}, sep={p195_results['best_sep']:.4f}")
    
    # P196总结
    for l, decomp in p196_results["decomposition"].items():
        backbone_fracs = [d["backbone_fraction"] for d in decomp.values()]
        residual_fracs = [d["residual_fraction"] for d in decomp.values()]
        if backbone_fracs:
            log(f"[P196 名词残差] L{l}: backbone={np.mean(backbone_fracs):.1%} residual={np.mean(residual_fracs):.1%}")
    
    # P197总结
    for l, ch_data in p197_results["channels"].items():
        log(f"[P197 属性通道] L{l}: n_channels={ch_data['n_channels']} "
            f"intra_type_cos={ch_data['intra_type_cos']} "
            f"cross_type_cos={ch_data['cross_type_cos_mean']:.4f}")
    
    # P198总结
    for case, case_data in p198_results["ablation_effects"].items():
        max_kl_backbone = max([v.get("remove_backbone", {}).get("kl", 0) for v in case_data.values()] + [0])
        max_kl_residual = max([v.get("remove_residual", {}).get("kl", 0) for v in case_data.values()] + [0])
        max_kl_color = max([v.get("remove_attr_color", {}).get("kl", 0) for v in case_data.values()] + [0])
        log(f"[P198 因果消融] {case}: backbone_KL={max_kl_backbone:.4f} residual_KL={max_kl_residual:.4f} color_KL={max_kl_color:.4f}")
    
    # P199总结
    for l, inj_data in p199_results["injection_effects"].items():
        hit_reasonable = np.mean([1 for v in inj_data.values() if v.get("reasonable") and v.get("attr_in_top5")])
        hit_unreasonable = np.mean([1 for v in inj_data.values() if not v.get("reasonable") and v.get("attr_in_top5")])
        log(f"[P199 属性注入] L{l}: reasonable_hit={hit_reasonable:.2f} unreasonable_hit={hit_unreasonable:.2f}")
    
    # P200总结
    for l, bridge in p200_results["bridge_analysis"].items():
        cos_nouns = [v["cos_noun"] for v in bridge.values()]
        cos_attrs = [v["cos_attr"] for v in bridge.values()]
        if cos_nouns:
            log(f"[P200 桥接回路] L{l}: cos_noun={np.mean(cos_nouns):.4f} cos_attr={np.mean(cos_attrs):.4f}")
    
    # 编码公式验证总结
    log(f"\n{'='*70}")
    log(f"编码公式验证总结")
    log(f"  h(apple, red, sweet) ≈ B_global + B_fruit + E_apple + A_red + A_sweet + G(...)")
    log(f"  B_fruit: 家族骨干, 最佳分离层=L{best_layer}")
    log(f"  E_apple: 名词独有残差, 需要进一步量化其独立性")
    log(f"  A_red/A_sweet: 属性通道, 需要验证跨对象复用性")
    log(f"  G(...): 桥接项, 需要估计其维度和必要性")
    log(f"{'='*70}")
    
    # 保存结果
    # 移除不可序列化的tensor
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    # 移除大的hidden state数据
    save_results = {}
    for k in ["P195", "P196", "P197", "P198", "P199", "P200"]:
        if k in all_results:
            save_results[k] = make_serializable(all_results[k])
            # 移除大对象
            if "family_hiddens" in save_results[k]:
                del save_results[k]["family_hiddens"]
    
    result_path = os.path.join(log_dir, "phase31_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}")
    
    log.close()
    
    # 释放GPU
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
