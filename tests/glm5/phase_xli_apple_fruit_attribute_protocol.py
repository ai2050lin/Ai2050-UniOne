"""
Phase XLI-P253: 苹果-水果-属性分层拆解协议 v2 (增强版)
======================================================

核心目标: 破解编码机制在神经元和参数级别是怎么形成的
具体问题: apple(苹果)里到底有多少是fruit(水果)家族共享骨干?

工作假说:
  h(apple, red, sweet) ≈ B_global + B_fruit + E_apple + A_red + A_sweet + G(apple, red, sweet) + C_context

五步协议:
  Step1: 最小对照集 - 严格配平数据拆开四类东西 (v2: 扩展到5个家族, 3类属性)
  Step2: 群体签名 - 多prompt下取稳定激活, 抽出5类签名 (v2: 增加PCA分析)
  Step3: 定向因果消融 - 分别打掉四类候选结构 (v2: 多层消融, KL散度)
  Step4: 属性注入和跨对象迁移 - 验证属性通道独立性 (v2: 更多迁移对)
  Step5: 桥接回路搜索 - 找名词-属性绑定的神经元级回路 (v2: 逐层全扫描)

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import functools
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

# ===================== 配置 =====================
OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Model path not found: {p}")
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(p_abs, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    # 先加载到CPU再移到GPU (避免device_map="auto"在Windows下的CUDA崩溃)
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cpu"
    )
    mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

# ===================== Step1: 最小对照集 (v2增强版) =====================
# v2: 扩展到5个家族, 3类属性, 更多组合, 更多干扰项

STIMULI = {
    # 家族对照: 5个语义家族, 每个家族8个成员
    "fruit_family": [
        "apple", "banana", "pear", "orange", "grape", "mango",
        "strawberry", "watermelon",
    ],
    "animal_family": [
        "cat", "dog", "rabbit", "horse", "lion", "eagle",
        "elephant", "dolphin",
    ],
    "vehicle_family": [
        "car", "bus", "train", "plane", "boat", "bicycle",
        "truck", "helicopter",
    ],
    "furniture_family": [
        "chair", "table", "desk", "sofa", "bed", "cabinet",
        "shelf", "bench",
    ],
    "weather_family": [
        "rain", "snow", "wind", "storm", "fog", "sun",
        "cloud", "hail",
    ],
    
    # 属性词: 3类, 每类8个
    "color_attrs": [
        "red", "green", "yellow", "orange", "brown", "white",
        "blue", "black",
    ],
    "taste_attrs": [
        "sweet", "sour", "bitter", "salty", "crisp", "soft",
        "spicy", "fresh",
    ],
    "texture_attrs": [
        "hard", "soft", "smooth", "rough", "juicy", "dry",
        "wet", "sharp",
    ],
    
    # 组合: 名词+属性 (v2: 大幅增加)
    "fruit_color_combos": [
        "red apple", "green apple", "yellow banana", "orange orange",
        "green pear", "red grape", "yellow mango",
        "red strawberry", "green watermelon",
    ],
    "fruit_taste_combos": [
        "sweet apple", "sour apple", "sweet banana", "sour orange",
        "sweet pear", "bitter grape", "sweet mango",
        "sweet strawberry", "fresh watermelon",
    ],
    "fruit_texture_combos": [
        "soft apple", "hard apple", "smooth banana", "rough orange",
        "juicy pear", "dry grape", "soft mango",
    ],
    "animal_color_combos": [
        "brown cat", "white dog", "brown rabbit", "black horse",
        "golden eagle", "white cat", "black dog",
        "gray elephant", "blue dolphin",
    ],
    "vehicle_color_combos": [
        "red car", "green bus", "blue train", "white plane",
        "yellow boat", "black bicycle", "silver car",
        "white truck", "green helicopter",
    ],
    "furniture_color_combos": [
        "brown chair", "white table", "black desk", "red sofa",
        "white bed", "brown cabinet",
    ],
    "weather_color_combos": [
        "gray rain", "white snow", "dark storm", "bright sun",
        "white cloud",
    ],
    
    # 干扰: 属性+不相关名词 (v2: 更多)
    "color_distractors": [
        "red car", "green leaf", "yellow sun", "white snow",
        "brown table", "orange wall", "blue sky", "black night",
    ],
    "taste_distractors": [
        "sweet tea", "sour candy", "bitter coffee", "salty soup",
        "spicy food", "fresh bread", "sweet cake", "sour lemon",
    ],
    "texture_distractors": [
        "hard rock", "soft pillow", "smooth glass", "rough sand",
        "wet ground", "sharp knife", "dry desert",
    ],
}

# 多prompt模板 - 消除prompt偏差 (v2: 增加到7个)
PROMPT_TEMPLATES = [
    "The {word} is",
    "A {word} can be",
    "This {word} has",
    "I see a {word}",
    "The {word} looks",
    "That {word} seems",
    "Every {word} feels",
]

def get_all_single_words():
    """获取所有单字刺激"""
    words = set()
    for key in ["fruit_family", "animal_family", "vehicle_family",
                "furniture_family", "weather_family",
                "color_attrs", "taste_attrs", "texture_attrs"]:
        words.update(STIMULI[key])
    return sorted(words)

def get_all_combo_words():
    """获取所有组合刺激"""
    combos = []
    for key in ["fruit_color_combos", "fruit_taste_combos", "fruit_texture_combos",
                "animal_color_combos", "vehicle_color_combos",
                "furniture_color_combos", "weather_color_combos",
                "color_distractors", "taste_distractors", "texture_distractors"]:
        combos.extend(STIMULI[key])
    return combos


# ===================== Step2: 群体签名提取 (v2增强版) =====================

def collect_hidden_states(mdl, tok, device, words, prompt_template="The {word} is"):
    """收集多prompt下的全层hidden states"""
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    
    results = {}
    for word in words:
        prompt = prompt_template.replace("{word}", word)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True, output_attentions=True)
        
        # 全层hidden states: [n_layers+1, d_model]
        hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        # 注意力: [n_layers, n_heads, seq_len, seq_len]
        attn = None
        if out.attentions is not None:
            attn = torch.stack([a[0].float().cpu() for a in out.attentions])
        
        results[word] = {
            "hidden_states": hs,  # [n_layers+1, d_model]
            "attentions": attn,   # [n_layers, n_heads, seq_len, seq_len]
            "input_ids": inputs["input_ids"][0].cpu(),
        }
        del out
    gc.collect()
    return results


def compute_group_signatures(all_results, n_layers):
    """
    计算五类群体签名:
    1. 苹果群体签名
    2. 水果家族群体签名
    3. 苹果相对水果的独有签名
    4. 颜色通道签名
    5. 味道通道签名
    v2: 增加5个家族签名
    """
    signatures = {}
    
    family_keys = ["fruit_family", "animal_family", "vehicle_family",
                   "furniture_family", "weather_family"]
    attr_keys = ["color_attrs", "taste_attrs", "texture_attrs"]
    
    for layer_idx in range(n_layers + 1):
        layer_sigs = {}
        
        # 计算各家族签名
        family_centers = {}
        for fkey in family_keys:
            vecs = []
            for word in STIMULI[fkey]:
                if word in all_results:
                    vecs.append(all_results[word]["hidden_states"][layer_idx])
            if vecs:
                center = torch.stack(vecs).mean(0)
                center = center / (center.norm() + 1e-8)
                family_centers[fkey] = center
        
        # 全局名词骨干 = 所有名词的均值
        all_noun_vecs = []
        for fkey in family_keys:
            for word in STIMULI[fkey]:
                if word in all_results:
                    all_noun_vecs.append(all_results[word]["hidden_states"][layer_idx])
        if all_noun_vecs:
            global_center = torch.stack(all_noun_vecs).mean(0)
            global_center = global_center / (global_center.norm() + 1e-8)
        else:
            global_center = None
        
        # 苹果签名
        apple_vec = None
        if "apple" in all_results:
            apple_vec = all_results["apple"]["hidden_states"][layer_idx]
            apple_vec_normed = apple_vec / (apple_vec.norm() + 1e-8)
        
        # 苹果相对水果的独有签名 = apple - fruit_center投影
        apple_unique = None
        if apple_vec is not None and "fruit_family" in family_centers:
            fruit_center = family_centers["fruit_family"]
            proj_on_fruit = (apple_vec @ fruit_center) * fruit_center
            apple_unique = apple_vec - proj_on_fruit
            apple_unique = apple_unique / (apple_unique.norm() + 1e-8)
        
        # 属性通道签名
        attr_centers = {}
        for akey in attr_keys:
            vecs = []
            for word in STIMULI[akey]:
                if word in all_results:
                    vecs.append(all_results[word]["hidden_states"][layer_idx])
            if vecs:
                center = torch.stack(vecs).mean(0)
                center = center / (center.norm() + 1e-8)
                attr_centers[akey] = center
        
        layer_sigs = {
            "global_center": global_center,
            "family_centers": family_centers,
            "apple_vec": apple_vec,
            "apple_unique": apple_unique,
            "attr_centers": attr_centers,
        }
        # 兼容旧接口
        layer_sigs["fruit_center"] = family_centers.get("fruit_family")
        layer_sigs["animal_center"] = family_centers.get("animal_family")
        layer_sigs["vehicle_center"] = family_centers.get("vehicle_family")
        layer_sigs["color_center"] = attr_centers.get("color_attrs")
        layer_sigs["taste_center"] = attr_centers.get("taste_attrs")
        signatures[layer_idx] = layer_sigs
    
    return signatures


def analyze_signatures(signatures, all_results, n_layers):
    """
    分析签名的关键性质 (v2: 增加PCA分析和家族间距离矩阵)
    """
    analysis = {"layer_data": [], "pca_analysis": []}
    
    family_keys = ["fruit_family", "animal_family", "vehicle_family",
                   "furniture_family", "weather_family"]
    attr_keys = ["color_attrs", "taste_attrs", "texture_attrs"]
    
    for layer_idx in range(n_layers + 1):
        sig = signatures[layer_idx]
        row = {"layer": layer_idx}
        
        # 1. 苹果 vs 各家族的cos相似度
        if sig["apple_vec"] is not None:
            for fkey in family_keys:
                fc = sig["family_centers"].get(fkey)
                if fc is not None:
                    cos_val = F.cosine_similarity(
                        sig["apple_vec"].unsqueeze(0), fc.unsqueeze(0)
                    ).item()
                    row[f"cos_apple_{fkey}"] = cos_val
        
        # 兼容旧key名
        if "cos_apple_fruit_family" in row:
            row["cos_apple_fruit"] = row["cos_apple_fruit_family"]
        if "cos_apple_animal_family" in row:
            row["cos_apple_animal"] = row["cos_apple_animal_family"]
        if "cos_apple_vehicle_family" in row:
            row["cos_apple_vehicle"] = row["cos_apple_vehicle_family"]
        
        # 2. 苹果独有签名 vs 水果骨干的正交性
        if sig["apple_unique"] is not None and sig["fruit_center"] is not None:
            cos_unique_fruit = F.cosine_similarity(
                sig["apple_unique"].unsqueeze(0), sig["fruit_center"].unsqueeze(0)
            ).item()
            row["cos_apple_unique_vs_fruit"] = cos_unique_fruit
        
        # 3. 属性通道之间的正交性矩阵
        for i, ak1 in enumerate(attr_keys):
            for j, ak2 in enumerate(attr_keys):
                if i < j:
                    c1 = sig["attr_centers"].get(ak1)
                    c2 = sig["attr_centers"].get(ak2)
                    if c1 is not None and c2 is not None:
                        cos_val = F.cosine_similarity(
                            c1.unsqueeze(0), c2.unsqueeze(0)
                        ).item()
                        row[f"cos_{ak1}_vs_{ak2}"] = cos_val
        
        # 兼容旧key名
        if "cos_color_attrs_vs_taste_attrs" in row:
            row["cos_color_vs_taste"] = row["cos_color_attrs_vs_taste_attrs"]
        if "cos_color_attrs_vs_texture_attrs" in row:
            row["cos_color_vs_texture"] = row["cos_color_attrs_vs_texture_attrs"]
        if "cos_taste_attrs_vs_texture_attrs" in row:
            row["cos_taste_vs_texture"] = row["cos_taste_attrs_vs_texture_attrs"]
        
        # 4. 属性通道 vs 水果骨干的正交性
        for akey in attr_keys:
            ac = sig["attr_centers"].get(akey)
            if ac is not None and sig["fruit_center"] is not None:
                cos_val = F.cosine_similarity(
                    ac.unsqueeze(0), sig["fruit_center"].unsqueeze(0)
                ).item()
                row[f"cos_{akey}_vs_fruit"] = cos_val
        
        # 兼容
        if "cos_color_attrs_vs_fruit" in row:
            row["cos_color_vs_fruit"] = row["cos_color_attrs_vs_fruit"]
        if "cos_taste_attrs_vs_fruit" in row:
            row["cos_taste_vs_fruit"] = row["cos_taste_attrs_vs_fruit"]
        
        # 5. 苹果独有签名 vs 属性通道
        if sig["apple_unique"] is not None:
            for akey in attr_keys:
                ac = sig["attr_centers"].get(akey)
                if ac is not None:
                    cos_val = F.cosine_similarity(
                        sig["apple_unique"].unsqueeze(0), ac.unsqueeze(0)
                    ).item()
                    row[f"cos_apple_unique_vs_{akey}"] = cos_val
        
        # 兼容
        if "cos_apple_unique_vs_color_attrs" in row:
            row["cos_apple_unique_vs_color"] = row["cos_apple_unique_vs_color_attrs"]
        if "cos_apple_unique_vs_taste_attrs" in row:
            row["cos_apple_unique_vs_taste"] = row["cos_apple_unique_vs_taste_attrs"]
        
        # 6. 家族内部一致性 (pairwise cos)
        for fkey in family_keys:
            family_words = [w for w in STIMULI[fkey] if w in all_results]
            if len(family_words) >= 2:
                pairwise = []
                for i in range(len(family_words)):
                    for j in range(i+1, len(family_words)):
                        vi = all_results[family_words[i]]["hidden_states"][layer_idx]
                        vj = all_results[family_words[j]]["hidden_states"][layer_idx]
                        c = F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
                        pairwise.append(c)
                row[f"{fkey}_internal_cos"] = np.mean(pairwise)
        
        # 兼容
        if "fruit_family_internal_cos" in row:
            row["fruit_family_internal_cos_old"] = row["fruit_family_internal_cos"]
        if "animal_family_internal_cos" in row:
            row["animal_family_internal_cos_old"] = row["animal_family_internal_cos"]
        
        # 7. 跨家族距离矩阵
        for i, fk1 in enumerate(family_keys):
            for j, fk2 in enumerate(family_keys):
                if i < j:
                    c1 = sig["family_centers"].get(fk1)
                    c2 = sig["family_centers"].get(fk2)
                    if c1 is not None and c2 is not None:
                        cos_val = F.cosine_similarity(
                            c1.unsqueeze(0), c2.unsqueeze(0)
                        ).item()
                        row[f"cross_{fk1}_{fk2}"] = cos_val
        
        # 兼容
        if "cross_fruit_family_animal_family" in row:
            row["cross_family_cos_fruit_animal"] = row["cross_fruit_family_animal_family"]
        
        analysis["layer_data"].append(row)
    
    # PCA分析: 在关键层分析家族和属性的子空间结构
    key_layers_pca = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    key_layers_pca = sorted(set([l for l in key_layers_pca if 0 <= l <= n_layers]))
    
    for layer_idx in key_layers_pca:
        pca_row = {"layer": layer_idx}
        
        # 收集所有名词向量
        all_noun_vecs = []
        all_noun_labels = []
        for fkey in family_keys:
            for word in STIMULI[fkey]:
                if word in all_results:
                    all_noun_vecs.append(all_results[word]["hidden_states"][layer_idx].numpy())
                    all_noun_labels.append(fkey)
        
        if len(all_noun_vecs) >= 5:
            noun_matrix = np.stack(all_noun_vecs)
            # PCA降维
            pca = PCA(n_components=min(20, noun_matrix.shape[0]-1, noun_matrix.shape[1]))
            pca.fit(noun_matrix)
            
            # 计算各家族在主成分上的分离度
            family_separations = {}
            for fkey in family_keys:
                indices = [i for i, l in enumerate(all_noun_labels) if l == fkey]
                if len(indices) >= 2:
                    family_vecs = noun_matrix[indices]
                    family_center = family_vecs.mean(0)
                    family_spread = np.mean(np.linalg.norm(family_vecs - family_center, axis=1))
                    family_separations[fkey] = round(family_spread, 6)
            
            pca_row["explained_variance_ratio"] = [round(v, 6) for v in pca.explained_variance_ratio_[:10]]
            pca_row["cumulative_variance_10d"] = round(sum(pca.explained_variance_ratio_[:10]), 6)
            pca_row["family_separations"] = family_separations
            
            # 计算家族间在PC空间的可分性
            if len(family_separations) >= 2:
                # Fisher判别比: 类间/类内
                all_centers = []
                for fkey in family_keys:
                    indices = [i for i, l in enumerate(all_noun_labels) if l == fkey]
                    if indices:
                        all_centers.append(noun_matrix[indices].mean(0))
                
                if len(all_centers) >= 2:
                    centers_arr = np.stack(all_centers)
                    global_center = centers_arr.mean(0)
                    between_var = np.mean(np.linalg.norm(centers_arr - global_center, axis=1))
                    within_var = np.mean(list(family_separations.values()))
                    fisher_ratio = between_var / (within_var + 1e-8)
                    pca_row["fisher_ratio"] = round(fisher_ratio, 6)
        
        # 收集所有属性向量
        all_attr_vecs = []
        all_attr_labels = []
        for akey in attr_keys:
            for word in STIMULI[akey]:
                if word in all_results:
                    all_attr_vecs.append(all_results[word]["hidden_states"][layer_idx].numpy())
                    all_attr_labels.append(akey)
        
        if len(all_attr_vecs) >= 5:
            attr_matrix = np.stack(all_attr_vecs)
            pca_attr = PCA(n_components=min(10, attr_matrix.shape[0]-1, attr_matrix.shape[1]))
            pca_attr.fit(attr_matrix)
            pca_row["attr_explained_variance_ratio"] = [round(v, 6) for v in pca_attr.explained_variance_ratio_[:5]]
            pca_row["attr_cumulative_variance_5d"] = round(sum(pca_attr.explained_variance_ratio_[:5]), 6)
        
        analysis["pca_analysis"].append(pca_row)
    
    return analysis


# ===================== Step3: 定向因果消融 (v2增强版) =====================

def directional_ablation(mdl, tok, device, model_name, signatures, n_layers):
    """
    定向因果消融 v2: 多层消融, KL散度, 更详细的logit分析
    """
    print(f"\n  Step3: 定向因果消融 ({model_name})")
    
    d_model = mdl.config.hidden_size
    results = {"ablation_types": []}
    
    # 测试词: 扩大范围
    test_words = [
        "apple", "banana", "pear", "cat", "dog", "car",
        "red", "green", "sweet", "sour", "hard", "soft",
    ]
    
    # 选择消融层: 早期/中期/晚期各选一层
    early_layer = max(1, n_layers // 6)
    mid_layer = n_layers // 2
    late_layer = min(n_layers - 2, 3 * n_layers // 4)
    
    ablation_layers = [early_layer, mid_layer, late_layer]
    print(f"    消融层: {ablation_layers} (早/中/晚期)")
    
    # 四类消融方向
    ablation_types = ["fruit_backbone", "apple_unique", "color_channel", "taste_channel", "texture_channel"]
    
    for abl_type in ablation_types:
        abl_results = {"ablation_type": abl_type, "effects": []}
        print(f"\n    消融类型: {abl_type}")
        
        for test_word in test_words:
            prompt = f"The {test_word}"
            inputs = tok(prompt, return_tensors="pt").to(device)
            
            # 基线: 正常forward
            with torch.no_grad():
                base_out = mdl(**inputs, output_hidden_states=True)
            base_logits = base_out.logits[0, -1].float().cpu()
            base_probs = F.softmax(base_logits, dim=-1)
            base_top10 = torch.topk(base_logits, 10)
            base_top10_tokens = [tok.decode([t], errors='replace') for t in base_top10.indices.tolist()]
            
            for layer_idx in ablation_layers:
                sig = signatures.get(layer_idx)
                if sig is None:
                    continue
                
                # 获取要消融的方向
                abl_dir = None
                if abl_type == "fruit_backbone":
                    abl_dir = sig.get("fruit_center")
                elif abl_type == "apple_unique":
                    abl_dir = sig.get("apple_unique")
                elif abl_type == "color_channel":
                    abl_dir = sig.get("color_center")
                elif abl_type == "taste_channel":
                    abl_dir = sig.get("taste_center")
                elif abl_type == "texture_channel":
                    abl_dir = sig.get("attr_centers", {}).get("texture_attrs")
                
                if abl_dir is None:
                    continue
                
                abl_dir = abl_dir.to(device)
                
                # Hook函数: 从hidden state中移除abl_dir方向的投影
                def make_ablation_hook(direction):
                    def hook_fn(module, input, output):
                        # output可能是tuple或其他类型
                        if isinstance(output, tuple):
                            hs = output[0]  # [batch, seq, d_model]
                            dir_t = direction.to(hs.dtype)
                            proj = (hs @ dir_t / (dir_t.norm() + 1e-8))
                            hs_abl = hs - proj.unsqueeze(-1) * dir_t.unsqueeze(0).unsqueeze(0)
                            return (hs_abl,) + output[1:]
                        else:
                            # 单个tensor
                            hs = output
                            dir_t = direction.to(hs.dtype)
                            proj = (hs @ dir_t / (dir_t.norm() + 1e-8))
                            hs_abl = hs - proj.unsqueeze(-1) * dir_t.unsqueeze(0).unsqueeze(0)
                            return hs_abl
                    return hook_fn
                
                # 注册hook
                target_layer = mdl.model.layers[layer_idx]
                handle = target_layer.register_forward_hook(make_ablation_hook(abl_dir))
                
                # 干预forward
                with torch.no_grad():
                    abl_out = mdl(**inputs, output_hidden_states=True)
                abl_logits = abl_out.logits[0, -1].float().cpu()
                abl_probs = F.softmax(abl_logits, dim=-1)
                abl_top10 = torch.topk(abl_logits, 10)
                abl_top10_tokens = [tok.decode([t], errors='replace') for t in abl_top10.indices.tolist()]
                
                # 效果度量 v2: 增加KL散度
                logit_shift = (abl_logits - base_logits).norm().item()
                kl_div = F.kl_div(
                    abl_probs.log(), base_probs, reduction='sum'
                ).item()
                top1_change = 1 if base_top10.indices[0] != abl_top10.indices[0] else 0
                top5_overlap = len(set(base_top10_tokens[:5]) & set(abl_top10_tokens[:5]))
                top10_overlap = len(set(base_top10_tokens) & set(abl_top10_tokens))
                
                # 目标词logit变化
                target_id = inputs["input_ids"][0, -1].item()
                target_logit_change = (abl_logits[target_id] - base_logits[target_id]).item()
                
                effect = {
                    "test_word": test_word,
                    "layer": layer_idx,
                    "logit_shift": round(logit_shift, 4),
                    "kl_divergence": round(kl_div, 4),
                    "top1_changed": top1_change,
                    "top5_overlap": top5_overlap,
                    "top10_overlap": top10_overlap,
                    "target_logit_change": round(target_logit_change, 4),
                    "base_top1": base_top10_tokens[0],
                    "abl_top1": abl_top10_tokens[0],
                }
                abl_results["effects"].append(effect)
                
                print(f"      L{layer_idx} {test_word}: logit_shift={logit_shift:.4f}, "
                      f"KL={kl_div:.4f}, top1_change={top1_change}, "
                      f"base_top1='{base_top10_tokens[0]}', abl_top1='{abl_top10_tokens[0]}'")
                
                # 移除hook
                handle.remove()
                del abl_out
            del base_out
        gc.collect()
        
        results["ablation_types"].append(abl_results)
    
    return results


# ===================== Step4: 属性注入和跨对象迁移 (v2增强版) =====================

def attribute_injection(mdl, tok, device, model_name, signatures, n_layers):
    """
    属性注入 v2: 更多迁移对, 多层注入, 更精细的alpha扫描
    """
    print(f"\n  Step4: 属性注入和跨对象迁移 ({model_name})")
    
    d_model = mdl.config.hidden_size
    results = {"injections": []}
    
    # 注入计划 v2: 扩大到跨家族
    injection_plan = [
        # 同家族内迁移
        ("red", "apple"), ("red", "banana"), ("red", "pear"),
        ("sweet", "apple"), ("sweet", "banana"), ("sweet", "pear"),
        ("green", "apple"), ("green", "grape"),
        # 跨家族迁移
        ("red", "car"), ("red", "cat"), ("red", "chair"),
        ("sweet", "tea"), ("sweet", "cake"), ("sweet", "soup"),
        ("green", "leaf"), ("green", "car"),
        # 质感迁移
        ("hard", "apple"), ("hard", "rock"), ("soft", "apple"),
        ("juicy", "orange"), ("juicy", "steak"),  # 预期: 前者自然, 后者冲突
    ]
    
    # 注入层: 早/中/晚
    early_layer = max(1, n_layers // 6)
    mid_layer = n_layers // 2
    late_layer = min(n_layers - 2, 3 * n_layers // 4)
    injection_layers = [early_layer, mid_layer, late_layer]
    print(f"    注入层: {injection_layers}")
    
    for attr_word, target_noun in injection_plan:
        inj_result = {"attr": attr_word, "target": target_noun, "layer_effects": []}
        
        prompt_attr = f"The {attr_word}"
        prompt_target = f"The {target_noun}"
        prompt_combo = f"The {attr_word} {target_noun}"
        
        inputs_target = tok(prompt_target, return_tensors="pt").to(device)
        inputs_attr = tok(prompt_attr, return_tensors="pt").to(device)
        inputs_combo = tok(prompt_combo, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_target = mdl(**inputs_target, output_hidden_states=True)
            out_attr = mdl(**inputs_attr, output_hidden_states=True)
            out_combo = mdl(**inputs_combo, output_hidden_states=True)
        
        for layer_idx in injection_layers:
            if layer_idx >= len(out_target.hidden_states):
                continue
            h_target = out_target.hidden_states[layer_idx][0, -1].float()
            h_attr = out_attr.hidden_states[layer_idx][0, -1].float()
            h_combo_next = out_combo.hidden_states[min(layer_idx + 1, len(out_combo.hidden_states)-1)][0, -1].float()
            
            # 属性delta = 属性词的hidden - 名词的hidden
            attr_delta = h_attr - h_target
            attr_delta_norm = attr_delta.norm().item()
            
            # 注入: target + alpha * attr_delta, v2: 更精细的alpha
            for alpha in [0.25, 0.5, 1.0, 1.5, 2.0]:
                injected_h = h_target + alpha * attr_delta
                
                cos_with_combo = F.cosine_similarity(
                    injected_h.unsqueeze(0), h_combo_next.unsqueeze(0)
                ).item()
                cos_with_target = F.cosine_similarity(
                    injected_h.unsqueeze(0), h_target.unsqueeze(0)
                ).item()
                
                layer_effect = {
                    "layer": layer_idx,
                    "alpha": alpha,
                    "attr_delta_norm": round(attr_delta_norm, 4),
                    "cos_injected_vs_combo": round(cos_with_combo, 4),
                    "cos_injected_vs_target": round(cos_with_target, 4),
                }
                inj_result["layer_effects"].append(layer_effect)
        
        # combo的logit分析
        base_logits = out_target.logits[0, -1].float().cpu()
        combo_logits = out_combo.logits[0, -1].float().cpu()
        
        target_token_id = inputs_target["input_ids"][0, -1].item()
        inj_result["base_target_logit"] = round(base_logits[target_token_id].item(), 4)
        inj_result["combo_target_logit"] = round(combo_logits[target_token_id].item(), 4)
        
        # 找combo中属性词的logit
        attr_token_id = inputs_attr["input_ids"][0, -1].item()
        inj_result["base_attr_logit"] = round(base_logits[attr_token_id].item(), 4)
        inj_result["combo_attr_logit"] = round(combo_logits[attr_token_id].item(), 4)
        
        results["injections"].append(inj_result)
        
        # 找最佳注入效果的alpha和层
        best_effect = max(inj_result["layer_effects"], key=lambda x: x["cos_injected_vs_combo"])
        print(f"    {attr_word} -> {target_noun}: "
              f"best_alpha={best_effect['alpha']}, "
              f"cos_combo={best_effect['cos_injected_vs_combo']:.4f}, "
              f"delta_norm={best_effect['attr_delta_norm']:.4f}")
        
        del out_target, out_attr, out_combo
    gc.collect()
    
    return results


# ===================== Step5: 桥接回路搜索 (v2增强版) =====================

def bridge_circuit_search(mdl, tok, device, model_name, signatures, n_layers):
    """
    桥接回路搜索 v2: 逐层全扫描, 更多测试三元组, 增加非线性G分析
    """
    print(f"\n  Step5: 桥接回路搜索 ({model_name})")
    
    d_model = mdl.config.hidden_size
    results = {"bridge_layers": [], "bridge_summary": []}
    
    # 核心测试三元组: 扩大
    test_triples = [
        ("apple", "red", "red apple"),
        ("apple", "sweet", "sweet apple"),
        ("apple", "green", "green apple"),
        ("banana", "yellow", "yellow banana"),
        ("banana", "sweet", "sweet banana"),
        ("car", "red", "red car"),
        ("cat", "brown", "brown cat"),
        ("car", "fast", "fast car"),       # 抽象属性
        ("apple", "fresh", "fresh apple"),  # 抽象属性
    ]
    
    for noun, attr, combo in test_triples:
        prompt_noun = f"The {noun}"
        prompt_attr = f"The {attr}"
        prompt_combo = f"The {combo}"
        
        inputs_noun = tok(prompt_noun, return_tensors="pt").to(device)
        inputs_attr = tok(prompt_attr, return_tensors="pt").to(device)
        inputs_combo = tok(prompt_combo, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_noun = mdl(**inputs_noun, output_hidden_states=True)
            out_attr = mdl(**inputs_attr, output_hidden_states=True)
            out_combo = mdl(**inputs_combo, output_hidden_states=True)
        
        bridge_data = {"noun": noun, "attr": attr, "combo": combo, "layers": []}
        
        for layer_idx in range(n_layers + 1):
            h_noun = out_noun.hidden_states[layer_idx][0, -1].float()
            h_attr = out_attr.hidden_states[layer_idx][0, -1].float()
            h_combo = out_combo.hidden_states[layer_idx][0, -1].float()
            
            # 桥接项G = h_combo - h_noun
            G_approx = h_combo - h_noun
            
            # 属性偏移 = h_attr - h_noun
            attr_delta = h_attr - h_noun
            
            # G与属性偏移的关系
            cos_G_vs_attr = F.cosine_similarity(
                G_approx.unsqueeze(0), attr_delta.unsqueeze(0)
            ).item()
            G_norm = G_approx.norm().item()
            
            # 非线性残差: G - proj(G on attr_delta)
            # 如果G完全等于attr_delta方向, 则残差=0
            proj_G_on_attr = (G_approx @ attr_delta / (attr_delta.norm()**2 + 1e-8)) * attr_delta
            G_residual = G_approx - proj_G_on_attr
            G_residual_norm = G_residual.norm().item()
            G_residual_ratio = G_residual_norm / (G_norm + 1e-8)
            
            # combo中noun方向和attr方向各占多少
            noun_dir = h_noun / (h_noun.norm() + 1e-8)
            attr_dir = h_attr / (h_attr.norm() + 1e-8)
            
            proj_on_noun = (h_combo @ noun_dir).item()
            proj_on_attr = (h_combo @ attr_dir).item()
            
            # 加法预测误差: |h_combo - h_noun - attr_delta| vs |G|
            add_predict = h_noun + attr_delta
            add_error = (h_combo - add_predict).norm().item()
            add_cos = F.cosine_similarity(
                h_combo.unsqueeze(0), add_predict.unsqueeze(0)
            ).item()
            
            layer_data = {
                "layer": layer_idx,
                "cos_bridge_vs_attr": round(cos_G_vs_attr, 4),
                "bridge_norm": round(G_norm, 4),
                "G_residual_ratio": round(G_residual_ratio, 4),
                "proj_on_noun": round(proj_on_noun, 4),
                "proj_on_attr": round(proj_on_attr, 4),
                "add_cos": round(add_cos, 4),
                "add_error": round(add_error, 4),
            }
            bridge_data["layers"].append(layer_data)
        
        results["bridge_layers"].append(bridge_data)
        
        # 摘要: 桥接最强层, 加法最优层
        best_bridge_layer = max(
            bridge_data["layers"], key=lambda x: abs(x["cos_bridge_vs_attr"])
        )
        best_add_layer = max(
            bridge_data["layers"], key=lambda x: x["add_cos"]
        )
        summary = {
            "combo": combo,
            "best_bridge_layer": best_bridge_layer["layer"],
            "best_bridge_cos": best_bridge_layer["cos_bridge_vs_attr"],
            "best_add_layer": best_add_layer["layer"],
            "best_add_cos": best_add_layer["add_cos"],
        }
        results["bridge_summary"].append(summary)
        
        print(f"    {combo}: 桥接最强L{best_bridge_layer['layer']}(cos={best_bridge_layer['cos_bridge_vs_attr']:.4f}), "
              f"加法最优L{best_add_layer['layer']}(cos={best_add_layer['add_cos']:.4f}), "
              f"G残差比={best_bridge_layer['G_residual_ratio']:.4f}")
        
        del out_noun, out_attr, out_combo
    gc.collect()
    
    return results


# ===================== 主函数 =====================

def run_model(model_name):
    """运行完整协议"""
    print(f"\n{'='*70}")
    print(f"Phase XLI-P253 v2: 苹果-水果-属性分层拆解协议 ({model_name})")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    t0 = time.time()
    mdl, tok, device = load_model(model_name)
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    print(f"  模型: {model_name}, d_model={d_model}, n_layers={n_layers}, device={device}")
    
    all_results = {}
    
    # Step1+2: 收集所有单字和组合的hidden states
    print(f"\n  Step1+2: 收集群体签名 (v2: 7个prompt模板)...")
    
    single_words = get_all_single_words()
    combo_words = get_all_combo_words()
    all_words = list(set(single_words + combo_words))
    print(f"    单字刺激: {len(single_words)}个, 组合刺激: {len(combo_words)}个")
    
    # 多prompt取平均
    multi_prompt_results = {}
    for t_idx, template in enumerate(PROMPT_TEMPLATES):
        print(f"    模板 [{t_idx+1}/{len(PROMPT_TEMPLATES)}]: '{template}'")
        results_t = collect_hidden_states(mdl, tok, device, single_words, template)
        for word, data in results_t.items():
            if word not in multi_prompt_results:
                multi_prompt_results[word] = {
                    "hidden_states_sum": data["hidden_states"],
                    "count": 1,
                }
            else:
                multi_prompt_results[word]["hidden_states_sum"] += data["hidden_states"]
                multi_prompt_results[word]["count"] += 1
        del results_t
        gc.collect()
    
    # 取平均
    for word in multi_prompt_results:
        cnt = multi_prompt_results[word]["count"]
        multi_prompt_results[word]["hidden_states"] = (
            multi_prompt_results[word]["hidden_states_sum"] / cnt
        )
    
    # 组合词只用默认模板
    print(f"    收集组合词hidden states...")
    combo_results = collect_hidden_states(mdl, tok, device, combo_words)
    
    # 合并
    all_results = {}
    for word in multi_prompt_results:
        all_results[word] = {"hidden_states": multi_prompt_results[word]["hidden_states"]}
    for word, data in combo_results.items():
        all_results[word] = data
    del multi_prompt_results, combo_results
    gc.collect()
    
    # 计算群体签名
    print(f"  计算群体签名...")
    signatures = compute_group_signatures(all_results, n_layers)
    
    # 分析签名
    print(f"  分析签名 (含PCA)...")
    analysis = analyze_signatures(signatures, all_results, n_layers)
    
    # 找关键层
    key_layers = {}
    for ld in analysis["layer_data"]:
        layer = ld["layer"]
        if "cos_apple_fruit" in ld and "cos_apple_animal" in ld:
            separation = ld["cos_apple_fruit"] - ld["cos_apple_animal"]
            key_layers[layer] = key_layers.get(layer, 0) + separation
    
    if key_layers:
        best_sep_layer = max(key_layers, key=key_layers.get)
        print(f"  水果分离最强层: L{best_sep_layer} (separation={key_layers[best_sep_layer]:.4f})")
    
    # PCA摘要
    for pca_row in analysis.get("pca_analysis", []):
        print(f"  PCA L{pca_row['layer']}: "
              f"cumvar10d={pca_row.get('cumulative_variance_10d', 'N/A')}, "
              f"fisher={pca_row.get('fisher_ratio', 'N/A')}")
    
    # Step3: 定向因果消融
    ablation_results = directional_ablation(
        mdl, tok, device, model_name, signatures, n_layers
    )
    
    # Step4: 属性注入
    injection_results = attribute_injection(
        mdl, tok, device, model_name, signatures, n_layers
    )
    
    # Step5: 桥接回路搜索
    bridge_results = bridge_circuit_search(
        mdl, tok, device, model_name, signatures, n_layers
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_xli_apple_fruit_v2_{model_name}_{timestamp}.json"
    
    # 转换为可序列化格式
    def sanitize(obj):
        """递归转换为JSON可序列化类型"""
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        elif isinstance(obj, float):
            return round(obj, 6)
        elif isinstance(obj, (torch.Tensor,)):
            return sanitize(obj.tolist())
        else:
            return obj
    
    serializable_analysis = {"layer_data": [], "pca_analysis": []}
    for ld in analysis["layer_data"]:
        row = sanitize(ld)
        serializable_analysis["layer_data"].append(row)
    for pr in analysis.get("pca_analysis", []):
        serializable_analysis["pca_analysis"].append(sanitize(pr))
    
    elapsed = time.time() - t0
    
    output = {
        "schema_version": "agi_research_result.v2",
        "experiment_id": "phase_xli_p253_apple_fruit_attribute_protocol_v2",
        "title": "苹果-水果-属性分层拆解协议 v2",
        "model_name": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "working_hypothesis": (
            "h(apple, red, sweet) ≈ B_global + B_fruit + E_apple + A_red + A_sweet + G(apple, red, sweet) + C_context"
        ),
        "step2_signatures": serializable_analysis,
        "step3_ablation": ablation_results,
        "step4_injection": injection_results,
        "step5_bridge": bridge_results,
        "core_findings": {},
    }
    
    # 提取核心发现
    findings = []
    
    # 发现1: 苹果-水果共享度 (取中期层)
    mid_layer_data = [ld for ld in serializable_analysis["layer_data"] if ld["layer"] == n_layers // 2]
    if mid_layer_data:
        mld = mid_layer_data[0]
        if "cos_apple_fruit" in mld and "cos_apple_animal" in mld:
            findings.append(
                f"[家族归属] 苹果-水果cos={mld['cos_apple_fruit']:.4f}, "
                f"苹果-动物cos={mld['cos_apple_animal']:.4f}, "
                f"水果优势={mld['cos_apple_fruit'] - mld['cos_apple_animal']:.4f}"
            )
    
    # 发现2: 属性通道正交性
    if mid_layer_data:
        mld = mid_layer_data[0]
        if "cos_color_vs_taste" in mld:
            orth = "正交" if abs(mld["cos_color_vs_taste"]) < 0.3 else "相关"
            findings.append(
                f"[通道正交] 颜色-味道cos={mld['cos_color_vs_taste']:.4f}({orth}), "
                f"颜色-质感cos={mld.get('cos_color_vs_texture', 'N/A')}, "
                f"味道-质感cos={mld.get('cos_taste_vs_texture', 'N/A')}"
            )
    
    # 发现3: 苹果独有签名
    if mid_layer_data:
        mld = mid_layer_data[0]
        if "cos_apple_unique_vs_fruit" in mld:
            findings.append(
                f"[苹果独有] 独有签名-水果cos={mld['cos_apple_unique_vs_fruit']:.4f}, "
                f"独有-颜色cos={mld.get('cos_apple_unique_vs_color', 'N/A')}, "
                f"独有-味道cos={mld.get('cos_apple_unique_vs_taste', 'N/A')}"
            )
    
    # 发现4: PCA子空间结构
    for pca_row in serializable_analysis.get("pca_analysis", []):
        if pca_row["layer"] == n_layers // 2:
            findings.append(
                f"[PCA子空间] L{pca_row['layer']}: "
                f"10d累积方差={pca_row.get('cumulative_variance_10d', 'N/A')}, "
                f"Fisher比={pca_row.get('fisher_ratio', 'N/A')}"
            )
    
    # 发现5: 消融效果摘要
    for abl in ablation_results.get("ablation_types", []):
        effects = abl.get("effects", [])
        if effects:
            avg_kl = np.mean([e["kl_divergence"] for e in effects])
            top1_rate = np.mean([e["top1_changed"] for e in effects])
            findings.append(
                f"[消融] {abl['ablation_type']}: avg_KL={avg_kl:.4f}, "
                f"top1变化率={top1_rate:.2%}"
            )
    
    # 发现6: 桥接回路摘要
    for bs in bridge_results.get("bridge_summary", []):
        findings.append(
            f"[桥接] {bs['combo']}: 桥接L{bs['best_bridge_layer']}(cos={bs['best_bridge_cos']:.4f}), "
            f"加法L{bs['best_add_layer']}(cos={bs['best_add_cos']:.4f})"
        )
    
    output["core_findings"] = findings
    
    output = sanitize(output)
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存: {out_file}")
    
    # 打印核心发现
    print(f"\n  ===== 核心发现 =====")
    for f in findings:
        print(f"    {f}")
    
    print(f"\n  总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    
    # 释放模型
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Phase XLI-P253 v2: 苹果-水果-属性分层拆解协议")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "deepseek7b", "glm4"],
                       help="模型名称 (默认qwen3)")
    args = parser.parse_args()
    
    run_model(args.model)


if __name__ == "__main__":
    main()
