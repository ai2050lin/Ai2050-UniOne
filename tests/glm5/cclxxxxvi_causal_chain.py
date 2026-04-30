"""
CCLXXXXVI(296): 断裂机制的精确因果链分析
基于CCLXXXXV发现: Qwen3=post_attn LN放大触发断裂, GLM4=温和重排

核心问题:
1. Qwen3: LN weight在哪些维度最大? 这些维度与什么语义相关?
   - 如果LN weight的"热点维度"与断裂层variance暴增的维度一致 → LN是"因"
   - 如果不一致 → LN只是放大了Attention/MLP已经产生的变化

2. GLM4: 既然没有norm爆炸, Attention如何产生断裂?
   - 分析GLM4断裂层的attention pattern变化
   - 哪些token之间的attention weight变化最大?

3. 三模型: 断裂层前后的inter-category cosine具体变化
   - 在LN归一化后的空间中计算cos
   - 在原始空间中计算cos
   - 两者之差揭示LN的贡献

Exp1: Qwen3 LN Weight分析
  - 提取断裂层input LN和post_attn LN的weight
  - 哪些维度的weight最大?
  - 这些维度与CCLXXXXV中variance暴增的维度(Dim 4, 396)是否一致?

Exp2: Qwen3 因果验证 — LN放大是"因"还是"果"?
  - 在断裂层, 如果把LN weight设为全1(消除放大效应), cos是否还下降?
  - 对比: 正常前向 vs 修改LN weight前向

Exp3: GLM4 Attention Pattern分析
  - 断裂层的attention weight矩阵
  - 哪些head的attention pattern变化最大?
  - 是否有head突然"聚焦"到特定category token?

Exp4: 三模型断裂层前后cos分解
  - LN前的cos vs LN后的cos
  - Attention的cos贡献 vs MLP的cos贡献
  - 在"归一化空间"中的cos变化
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxvi_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXVI Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn.functional as F

log("Importing model_utils...")
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

CATEGORIES_13 = {
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "donkey", "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "elephant", "giraffe", "zebra", "monkey", "camel"],
    "bird": ["eagle", "hawk", "owl", "crow", "swan", "goose", "duck", "penguin", "parrot", "robin", "sparrow", "pigeon", "seagull", "falcon", "vulture", "crane", "stork", "heron", "peacock", "flamingo"],
    "fish": ["shark", "whale", "dolphin", "salmon", "trout", "tuna", "cod", "bass", "carp", "catfish", "perch", "pike", "eel", "herring", "sardine", "anchovy", "flounder", "sole", "mackerel", "swordfish"],
    "insect": ["ant", "bee", "spider", "butterfly", "mosquito", "fly", "wasp", "beetle", "cockroach", "grasshopper", "cricket", "dragonfly", "ladybug", "moth", "flea", "tick", "mantis", "caterpillar", "worm", "snail"],
    "plant": ["tree", "flower", "grass", "bush", "shrub", "vine", "fern", "moss", "algae", "weed", "oak", "pine", "maple", "birch", "willow", "cactus", "bamboo", "palm", "rose", "lily"],
    "fruit": ["apple", "orange", "banana", "grape", "pear", "peach", "cherry", "plum", "mango", "lemon", "lime", "melon", "berry", "strawberry", "blueberry", "raspberry", "fig", "date", "coconut", "pineapple"],
    "vegetable": ["carrot", "potato", "tomato", "onion", "garlic", "cabbage", "lettuce", "spinach", "celery", "pea", "bean", "corn", "mushroom", "pepper", "cucumber", "pumpkin", "squash", "radish", "turnip", "broccoli"],
    "body_part": ["hand", "foot", "head", "heart", "brain", "eye", "ear", "nose", "mouth", "tooth", "neck", "shoulder", "arm", "finger", "knee", "chest", "back", "hip", "ankle", "wrist"],
    "tool": ["hammer", "knife", "scissors", "saw", "drill", "wrench", "screwdriver", "plier", "axe", "chisel", "ruler", "file", "clamp", "level", "shovel", "rake", "hoe", "trowel", "spade", "mallet"],
    "vehicle": ["car", "bus", "truck", "train", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "ship", "submarine", "rocket", "tractor", "van", "taxi", "ambulance", "sled", "canoe", "wagon", "cart"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove", "jacket", "sweater", "vest", "skirt", "pants", "jeans", "sock", "boot", "sandal", "tie", "uniform", "cape"],
    "weapon": ["sword", "spear", "bow", "arrow", "shield", "axe_w", "dagger", "mace", "pike_w", "lance", "crossbow", "catapult", "pistol", "rifle", "cannon", "grenade", "dynamite", "knife_w", "club", "whip"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "couch", "shelf", "cabinet", "drawer", "wardrobe", "dresser", "bench", "stool", "armchair", "bookcase", "mirror", "lamp", "rug", "curtain", "pillow"],
}

SUPERCLASS_MAP = {
    "animal": "animate", "bird": "animate", "fish": "animate", "insect": "animate",
    "plant": "plant", "fruit": "plant", "vegetable": "plant",
    "body_part": "body",
    "tool": "artifact", "vehicle": "artifact", "clothing": "artifact",
    "weapon": "artifact", "furniture": "artifact",
}

FRACTURE_LAYERS = {"qwen3": 6, "glm4": 2, "deepseek7b": 7}


def get_ln_weights(model, model_name, frac_layer):
    """Extract LayerNorm weights at key layers"""
    layers_list = get_layers(model)
    result = {}
    
    for l in [max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1]:
        if l >= len(layers_list):
            continue
        layer = layers_list[l]
        
        # Input LN
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, 'weight'):
                    w = ln.weight.detach().cpu().float().numpy()
                    result[f"L{l}_input_ln"] = w
                break
        
        # Post-attn LN
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, 'weight'):
                    w = ln.weight.detach().cpu().float().numpy()
                    result[f"L{l}_post_attn_ln"] = w
                break
    
    return result


def run_exp1_qwen3_ln_weight(model, tokenizer, device, model_name):
    """
    Exp1: Qwen3 LN Weight分析
    """
    log(f"\n{'='*60}")
    log(f"Exp1: LN Weight Analysis ({model_name})")
    log(f"{'='*60}")
    
    frac_layer = FRACTURE_LAYERS[model_name]
    ln_weights = get_ln_weights(model, model_name, frac_layer)
    
    # 分析每个LN weight
    for key, w in sorted(ln_weights.items()):
        log(f"\n--- {key} ---")
        log(f"  Shape: {w.shape}, Mean: {w.mean():.4f}, Std: {w.std():.4f}")
        log(f"  Min: {w.min():.4f}, Max: {w.max():.4f}")
        
        # Top-20 dimensions by weight
        top_dims = np.argsort(w)[-20:][::-1]
        bottom_dims = np.argsort(w)[:20]
        
        log(f"  Top-20 dims by weight:")
        for dim in top_dims:
            log(f"    Dim {dim:>5}: weight={w[dim]:.4f}")
        
        log(f"  Bottom-20 dims by weight:")
        for dim in bottom_dims:
            log(f"    Dim {dim:>5}: weight={w[dim]:.4f}")
        
        # Weight distribution
        log(f"  Weight > 2.0: {(w > 2.0).sum()} dims ({(w > 2.0).mean()*100:.1f}%)")
        log(f"  Weight > 1.5: {(w > 1.5).sum()} dims ({(w > 1.5).mean()*100:.1f}%)")
        log(f"  Weight > 1.0: {(w > 1.0).sum()} dims ({(w > 1.0).mean()*100:.1f}%)")
        log(f"  Weight < 0.5: {(w < 0.5).sum()} dims ({(w < 0.5).mean()*100:.1f}%)")
        log(f"  Weight < 0.1: {(w < 0.1).sum()} dims ({(w < 0.1).mean()*100:.1f}%)")
    
    # 关键对比: 断裂层 vs 前一层的post_attn LN weight
    frac_key = f"L{frac_layer}_post_attn_ln"
    pre_key = f"L{frac_layer-1}_post_attn_ln"
    
    if frac_key in ln_weights and pre_key in ln_weights:
        log(f"\n--- Post-attn LN weight comparison: Pre-fracture vs Fracture ---")
        w_pre = ln_weights[pre_key]
        w_frac = ln_weights[frac_key]
        
        diff = w_frac - w_pre
        log(f"  Weight diff: mean={diff.mean():.6f}, std={diff.std():.6f}")
        log(f"  Correlation: {np.corrcoef(w_pre, w_frac)[0,1]:.6f}")
        
        # Top-20 dims with largest weight increase
        top_increase = np.argsort(diff)[-20:][::-1]
        log(f"  Top-20 dims with LARGEST weight increase:")
        for dim in top_increase:
            log(f"    Dim {dim:>5}: pre={w_pre[dim]:.4f}, frac={w_frac[dim]:.4f}, diff={diff[dim]:+.4f}")
        
        # 对比CCLXXXXV中variance暴增的维度
        # Dim 4 和 Dim 396 是CCLXXXXV中variance暴增最多的
        cclxxxxv_hot_dims = [4, 396, 0, 100, 19, 34, 49, 9, 14, 130, 18, 22]
        log(f"\n  CCLXXXXV hot dims (variance暴增) in fracture LN:")
        for dim in cclxxxxv_hot_dims:
            log(f"    Dim {dim:>5}: pre_weight={w_pre[dim]:.4f}, frac_weight={w_frac[dim]:.4f}, diff={diff[dim]:+.4f}")
    
    return ln_weights


def run_exp2_causal_test(model, tokenizer, device, model_name):
    """
    Exp2: Qwen3因果验证 — LN放大是"因"还是"果"?
    修改断裂层的LN weight为全1, 看cos是否还下降
    """
    log(f"\n{'='*60}")
    log(f"Exp2: Causal Test - LN Amplification ({model_name})")
    log(f"{'='*60}")
    
    frac_layer = FRACTURE_LAYERS[model_name]
    layers_list = get_layers(model)
    d_model = model.config.hidden_size
    
    cat_names = sorted(CATEGORIES_13.keys())
    
    # 辅助函数: 收集指定层的residual stream
    def collect_residuals(target_layers, modify_ln=False):
        """收集指定层的residual stream, 可选修改LN weight"""
        all_rs = {l: defaultdict(list) for l in target_layers}
        residual_cache = {}
        hook_handles = []
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    residual_cache[layer_idx] = output[0].detach().clone()
                else:
                    residual_cache[layer_idx] = output.detach().clone()
            return hook_fn
        
        for l in target_layers:
            h = layers_list[l].register_forward_hook(make_hook(l))
            hook_handles.append(h)
        
        # 可选: 修改LN weight
        original_weights = {}
        if modify_ln:
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(layers_list[frac_layer], ln_name):
                    ln = getattr(layers_list[frac_layer], ln_name)
                    if hasattr(ln, 'weight'):
                        original_weights[ln_name] = ln.weight.data.clone()
                        ln.weight.data.fill_(1.0)  # 设为全1
                        log(f"  Modified {ln_name} weight to all-1s")
                    break
        
        for cat in cat_names:
            for word in CATEGORIES_13[cat]:
                inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    residual_cache.clear()
                    try:
                        model(input_ids)
                    except:
                        continue
                for l in target_layers:
                    if l in residual_cache:
                        rs = residual_cache[l][0, -1, :].cpu().float().numpy()
                        all_rs[l][cat].append(rs)
        
        # 恢复LN weight
        for ln_name, w in original_weights.items():
            ln = getattr(layers_list[frac_layer], ln_name)
            ln.weight.data = w
        
        for h in hook_handles:
            h.remove()
        
        return all_rs
    
    # 计算inter-category cosine
    def compute_avg_cos(residuals_dict, layer):
        cat_means = {}
        for cat in sorted(residuals_dict[layer].keys()):
            vecs = np.array(residuals_dict[layer][cat])
            cat_means[cat] = vecs.mean(axis=0)
        
        coses = []
        for c1, c2 in combinations(sorted(cat_means.keys()), 2):
            v1, v2 = cat_means[c1], cat_means[c2]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            coses.append(cos)
        return np.mean(coses)
    
    target_layers = [frac_layer - 1, frac_layer, frac_layer + 1]
    
    # 1. 正常前向
    log(f"\n  Normal forward pass...")
    normal_rs = collect_residuals(target_layers, modify_ln=False)
    
    for l in target_layers:
        avg_cos = compute_avg_cos(normal_rs, l)
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"    L{l} ({marker}): avg_cos={avg_cos:.4f}")
    
    # 2. 修改LN weight为全1
    log(f"\n  Modified LN weight forward pass (all-1s)...")
    modified_rs = collect_residuals(target_layers, modify_ln=True)
    
    for l in target_layers:
        avg_cos = compute_avg_cos(modified_rs, l)
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"    L{l} ({marker}): avg_cos={avg_cos:.4f}")
    
    # 3. 对比
    log(f"\n  --- Causal comparison ---")
    for l in target_layers:
        normal_cos = compute_avg_cos(normal_rs, l)
        modified_cos = compute_avg_cos(modified_rs, l)
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"    L{l} ({marker}): normal_cos={normal_cos:.4f}, modified_cos={modified_cos:.4f}, "
            f"diff={modified_cos - normal_cos:+.4f}")
    
    # 4. Per-category analysis at fracture layer
    log(f"\n  --- Per-category norms at fracture layer ---")
    for cat in cat_names:
        normal_norms = [np.linalg.norm(v) for v in normal_rs[frac_layer][cat]]
        modified_norms = [np.linalg.norm(v) for v in modified_rs[frac_layer][cat]]
        log(f"    {cat:>12}: normal_norm={np.mean(normal_norms):.2f}, "
            f"modified_norm={np.mean(modified_norms):.2f}, "
            f"ratio={np.mean(normal_norms)/max(np.mean(modified_norms), 1e-10):.2f}")


def run_exp3_glm4_attention(model, tokenizer, device, model_name):
    """
    Exp3: GLM4 Attention Pattern分析
    """
    log(f"\n{'='*60}")
    log(f"Exp3: GLM4 Attention Pattern Analysis ({model_name})")
    log(f"{'='*60}")
    
    frac_layer = FRACTURE_LAYERS[model_name]
    layers_list = get_layers(model)
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads
    
    cat_names = sorted(CATEGORIES_13.keys())
    
    # Hook attention weights and outputs
    key_layers = [max(0, frac_layer-1), frac_layer, min(len(layers_list)-1, frac_layer+1)]
    
    attn_data = {l: defaultdict(list) for l in key_layers}
    attn_weights_data = {l: defaultdict(list) for l in key_layers}
    
    for l in key_layers:
        layer = layers_list[l]
        
        # Hook attention weights
        hook_handles = []
        
        def make_attn_hook(layer_idx):
            def hook_fn(module, input, output):
                # output from self_attn: (hidden_states, attn_weights, past_key_value)
                # But different models return different things
                # Let's just collect the attention output
                if isinstance(output, tuple) and len(output) >= 2:
                    # Try to get attention weights
                    if output[1] is not None:
                        attn_weights_data[layer_idx]['weights'].append(
                            output[1].detach().cpu().float().numpy()
                        )
                # Get hidden states output
                if isinstance(output, tuple):
                    attn_data[layer_idx]['output'].append(
                        output[0].detach().cpu().float().numpy()
                    )
            return hook_fn
        
        h = layer.self_attn.register_forward_hook(make_attn_hook(l))
        hook_handles.append(h)
        
        # Run forward for each category
        for cat in cat_names:
            for word in CATEGORIES_13[cat]:
                inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    try:
                        model(input_ids)
                    except:
                        continue
        
        for h in hook_handles:
            h.remove()
    
    # Analyze attention outputs
    log(f"\n  Attention output statistics:")
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        
        if 'output' not in attn_data[l]:
            log(f"    L{l} ({marker}): No attention output data")
            continue
        
        outputs = attn_data[l]['output']
        if len(outputs) == 0:
            log(f"    L{l} ({marker}): Empty attention output")
            continue
        
        # Compute per-category norms
        log(f"    L{l} ({marker}): {len(outputs)} outputs collected")
        
        # Compute inter-category cosine of attention outputs
        # Group by category (assuming same order)
        cat_attn_means = {}
        idx = 0
        for cat in cat_names:
            cat_outputs = []
            for word in CATEGORIES_13[cat]:
                if idx < len(outputs):
                    # Get last token
                    out = outputs[idx]
                    if out.ndim == 3:
                        cat_outputs.append(out[0, -1, :])
                    elif out.ndim == 2:
                        cat_outputs.append(out[-1, :])
                    elif out.ndim == 1:
                        cat_outputs.append(out)
                    idx += 1
            if cat_outputs:
                cat_attn_means[cat] = np.mean(cat_outputs, axis=0)
        
        if len(cat_attn_means) >= 2:
            coses = []
            for c1, c2 in combinations(sorted(cat_attn_means.keys()), 2):
                v1, v2 = cat_attn_means[c1], cat_attn_means[c2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                coses.append(cos)
            log(f"      avg_cos(attn_output)={np.mean(coses):.4f}")
            
            # Per-superclass
            sup_cos = defaultdict(list)
            for c1, c2 in combinations(sorted(cat_attn_means.keys()), 2):
                v1, v2 = cat_attn_means[c1], cat_attn_means[c2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                s1, s2 = SUPERCLASS_MAP[c1], SUPERCLASS_MAP[c2]
                if s1 == s2:
                    sup_cos[f"within_{s1}"].append(cos)
                else:
                    sup_cos["cross"].append(cos)
            
            for key in sorted(sup_cos.keys()):
                log(f"      {key}: avg_cos={np.mean(sup_cos[key]):.4f}")


def run_exp4_cos_decomposition(model, tokenizer, device, model_name):
    """
    Exp4: 三模型断裂层cos分解
    - LN前的cos vs LN后的cos
    - 在"归一化空间"中计算cos
    """
    log(f"\n{'='*60}")
    log(f"Exp4: Cos Decomposition at Fracture ({model_name})")
    log(f"{'='*60}")
    
    frac_layer = FRACTURE_LAYERS[model_name]
    layers_list = get_layers(model)
    d_model = model.config.hidden_size
    cat_names = sorted(CATEGORIES_13.keys())
    
    # Hook layernorm input/output AND layer output
    key_layers = [max(0, frac_layer-1), frac_layer, min(len(layers_list)-1, frac_layer+1)]
    
    cache = defaultdict(lambda: defaultdict(list))
    hook_handles = []
    
    def make_hook(layer_idx, hook_type):
        def hook_fn(module, input, output):
            key = f"L{layer_idx}_{hook_type}"
            if hook_type in ['input_ln_in', 'input_ln_out', 'post_attn_ln_in', 'post_attn_ln_out', 'layer_out']:
                val = None
                if hook_type.endswith('_in'):
                    if isinstance(input, tuple) and len(input) > 0:
                        val = input[0].detach().clone()
                elif hook_type.endswith('_out') or hook_type == 'layer_out':
                    if isinstance(output, tuple):
                        val = output[0].detach().clone()
                    else:
                        val = output.detach().clone()
                
                if val is not None:
                    cache[key][hook_type].append(val[0, -1, :].cpu().float().numpy())
        return hook_fn
    
    # Register hooks
    for l in key_layers:
        layer = layers_list[l]
        
        # Layer output hook
        h = layer.register_forward_hook(lambda m, i, o, l=l: 
            cache[f"L{l}_layer_out"]['layer_out'].append(
                (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
            ))
        hook_handles.append(h)
        
        # Input LN hooks
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(lambda m, i, o, l=l: 
                    cache[f"L{l}_input_ln_out"]['input_ln_out'].append(
                        (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
                    ))
                hook_handles.append(h)
                break
        
        # Post-attn LN hooks
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(lambda m, i, o, l=l:
                    cache[f"L{l}_post_attn_ln_out"]['post_attn_ln_out'].append(
                        (o[0] if isinstance(o, tuple) else o).detach().clone()[0, -1, :].cpu().float().numpy()
                    ))
                hook_handles.append(h)
                break
    
    # Forward pass
    cat_idx = 0
    for cat in cat_names:
        for word in CATEGORIES_13[cat]:
            inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                try:
                    model(input_ids)
                except:
                    continue
            cat_idx += 1
    
    for h in hook_handles:
        h.remove()
    
    # Compute cos for each representation
    log(f"\n  Inter-category cosine in different representations:")
    
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"\n  --- L{l} ({marker}) ---")
        
        for rep_type in ['layer_out', 'input_ln_out', 'post_attn_ln_out']:
            key = f"L{l}_{rep_type}"
            if key not in cache or rep_type not in cache[key] or len(cache[key][rep_type]) == 0:
                log(f"    {rep_type}: No data")
                continue
            
            vecs = cache[key][rep_type]
            # Group by category (13 cats x 20 words = 260 total)
            cat_means = {}
            idx = 0
            for cat in cat_names:
                cat_vecs = []
                for word in CATEGORIES_13[cat]:
                    if idx < len(vecs):
                        cat_vecs.append(vecs[idx])
                        idx += 1
                if cat_vecs:
                    cat_means[cat] = np.mean(cat_vecs, axis=0)
            
            if len(cat_means) < 2:
                log(f"    {rep_type}: Not enough categories")
                continue
            
            coses = []
            for c1, c2 in combinations(sorted(cat_means.keys()), 2):
                v1, v2 = cat_means[c1], cat_means[c2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = np.dot(v1, v2) / (n1 * n2)
                    coses.append(cos)
            
            if coses:
                log(f"    {rep_type}: avg_cos={np.mean(coses):.4f}, min={np.min(coses):.4f}, max={np.max(coses):.4f}")
            
            # Per-superclass
            sup_cos = defaultdict(list)
            for c1, c2 in combinations(sorted(cat_means.keys()), 2):
                v1, v2 = cat_means[c1], cat_means[c2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = np.dot(v1, v2) / (n1 * n2)
                    s1, s2 = SUPERCLASS_MAP[c1], SUPERCLASS_MAP[c2]
                    if s1 == s2:
                        sup_cos[f"within_{s1}"].append(cos)
                    else:
                        sup_cos["cross"].append(cos)
            
            for key_s in sorted(sup_cos.keys()):
                if sup_cos[key_s]:
                    log(f"      {key_s}: avg_cos={np.mean(sup_cos[key_s]):.4f}")


def run_model(model_name):
    log(f"=== Starting {model_name} ===")
    try:
        model, tokenizer, device = load_model(model_name)
        info = get_model_info(model, model_name)
        log(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
        
        # Exp1: LN Weight analysis (all models)
        ln_weights = run_exp1_qwen3_ln_weight(model, tokenizer, device, model_name)
        
        # Exp2: Causal test (only Qwen3 - the model with LN amplification)
        if model_name == "qwen3":
            run_exp2_causal_test(model, tokenizer, device, model_name)
        else:
            log(f"\n  Skipping Exp2 (causal test) for {model_name} - no LN amplification")
        
        # Exp3: GLM4 Attention pattern (only GLM4)
        if model_name == "glm4":
            run_exp3_glm4_attention(model, tokenizer, device, model_name)
        else:
            log(f"\n  Skipping Exp3 (GLM4 attention) for {model_name}")
        
        # Exp4: Cos decomposition (all models)
        run_exp4_cos_decomposition(model, tokenizer, device, model_name)
        
        release_model(model)
        
        # Save LN weights
        result_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxxvi")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        save_data = {}
        for key, w in ln_weights.items():
            save_data[key] = {
                'top20_dims': np.argsort(w)[-20:][::-1].tolist(),
                'top20_weights': w[np.argsort(w)[-20:][::-1]].tolist(),
                'bottom20_dims': np.argsort(w)[:20].tolist(),
                'bottom20_weights': w[np.argsort(w)[:20]].tolist(),
                'mean': float(w.mean()),
                'std': float(w.std()),
            }
        
        with open(result_dir / "ln_weights.json", 'w') as f:
            json.dump(save_data, f, indent=2)
        
        log(f"Results saved to {result_dir}")
        log(f"=== Finished {model_name} ===\n")
        
    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")
    
    log("CCLXXXXVI: Fracture Mechanism Causal Chain Analysis")
    log("=" * 60)
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        run_model(model_name)
        log(f"Waiting 10s before next model...")
        time.sleep(10)
    
    log("=== All models completed ===")
