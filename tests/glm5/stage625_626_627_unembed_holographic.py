#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage625-626-627: Unembed全息读出方程 + 旋转速度因果实验 + logit空间传播

Stage625: Unembed全息读出的精确方程（P0）
  原理：已知消歧方向旋转到与原始方向几乎正交(survival<5%)，但unembed仍能读出。
  需要分析unembed矩阵如何从"编码转换"后的表示中提取消歧信息。
  - 对每层hidden state做unembed投影：logits_l = h_l @ W_unembed
  - 测量歧义词两个释义token的logit差（disambiguation logit margin）
  - 分析消歧信息何时进入logit空间
  - 对unembed矩阵做SVD，分析哪些奇异值/方向对消歧贡献最大
  - 测量unembed对每层hidden state的"读出效率"：从h_l能否完美恢复消歧信息

Stage626: 旋转速度→消歧度的因果关系（P0）
  原理：Stage622发现消歧方向每层旋转19-54°，Gemma4最快(54°)→消歧最差(15%)。
  旋转速度是否直接决定消歧度？
  - baseline: 原始模型的旋转速度和消歧度
  - intervention 1: 增大MLP权重（增大旋转）→消歧度变化
  - intervention 2: 缩小MLP权重（减小旋转）→消歧度变化
  - 比较旋转速度变化量与消歧度变化量的关系

Stage627: 消歧信息在logit空间的逐层演化
  原理：综合Stage625和626，追踪消歧信息如何在logit空间逐层累积。
  - 对每个歧义词，提取每层最后一个token的hidden state
  - 投影到logit空间，计算两个释义token的logit margin
  - 追踪margin的逐层变化：出现层、增长曲线、饱和层
  - 比较四模型的logit margin演化模式

用法: python stage625_626_627_unembed_holographic.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os, copy
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, encode_to_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def safe_get_device(model):
    for attr in [None, 'model', 'model.model']:
        try:
            obj = model
            if attr:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
            return next(obj.parameters()).device
        except (StopIteration, AttributeError):
            continue
    return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_disamb_tokens(tokenizer, word):
    """获取歧义词的两个释义token ID"""
    pairs = {
        "bank": (tokenizer.encode("bank", add_special_tokens=False)[-1],
                 tokenizer.encode("river", add_special_tokens=False)[-1]),
        "apple": (tokenizer.encode("apple", add_special_tokens=False)[-1],
                  tokenizer.encode("Apple", add_special_tokens=False)[-1]),
        "plant": (tokenizer.encode("plant", add_special_tokens=False)[-1],
                  tokenizer.encode("factory", add_special_tokens=False)[-1]),
        "bank2": (tokenizer.encode("bank", add_special_tokens=False)[-1],
                  tokenizer.encode("money", add_special_tokens=False)[-1]),
        "spring": (tokenizer.encode("spring", add_special_tokens=False)[-1],
                   tokenizer.encode("season", add_special_tokens=False)[-1]),
    }
    return pairs.get(word, (None, None))


DISAMB_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank",
     "bank", "financial"),
    ("She ate a red apple.", "Apple released the iPhone.", "apple",
     "apple", "Apple"),
    ("The factory plant employs workers.", "She watered the plant.", "plant",
     "plant", "factory"),
    ("He went to the bank to deposit money.", "The river bank was steep.", "bank2",
     "bank", "river"),
    ("The spring water was cold.", "The metal spring bounced back.", "spring",
     "spring", "season"),
]


def extract_all_layer_hidden(model, tokenizer, sentence, layers):
    """提取每层最后一个token的hidden state，layers是discover_layers返回的模块列表"""
    device = safe_get_device(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    inputs = move_to_device(inputs, model)

    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() >= 2:
                hidden_states[layer_idx] = h[0, -1, :].float().detach().cpu()
        return hook_fn

    for li, layer_module in enumerate(layers):
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    except Exception as e:
        print(f"    Forward failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    if len(hidden_states) == 0:
        return None
    return hidden_states


def get_unembed_matrix(model):
    """获取unembedding矩阵"""
    # Try different architectures
    for attr_path in ['lm_head', 'embed_out', 'output']:
        try:
            w = getattr(model, attr_path, None)
            if w is not None and hasattr(w, 'weight'):
                return w.weight.float().detach().cpu()
        except:
            pass
    
    # Check model.model
    try:
        inner = getattr(model, 'model', model)
        for attr_path in ['lm_head', 'embed_out', 'output']:
            try:
                w = getattr(inner, attr_path, None)
                if w is not None and hasattr(w, 'weight'):
                    return w.weight.float().detach().cpu()
            except:
                pass
    except:
        pass
    
    return None


def get_layer_modules(model, layer_idx):
    """获取指定层的MLP模块"""
    model_for_layers = getattr(model, 'model', model)
    layer_list = None
    for attr_name in ['layers', 'decoder_layers', 'block', 'h', 'encoder', 'transformer']:
        candidate = getattr(model_for_layers, attr_name, None)
        if candidate is not None and hasattr(candidate, '__len__') and len(candidate) > 0:
            layer_list = candidate
            break
    
    if layer_list is None:
        for child in model_for_layers.children():
            if hasattr(child, 'self_attn') or hasattr(child, 'attention'):
                layer_list = [c for c in model_for_layers.children() 
                              if hasattr(c, 'self_attn') or hasattr(c, 'attention')]
                break
    
    if layer_list and layer_idx < len(layer_list):
        return layer_list[layer_idx]
    return None


# ============ Stage625: Unembed全息读出的精确方程 ============

def run_stage625(model, tokenizer, model_key):
    """分析unembed矩阵如何从各层hidden state中读出消歧信息"""
    print("\n" + "="*70)
    print("Stage625: Unembed全息读出的精确方程")
    print("="*70)
    
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}
    
    print(f"  Unembed矩阵形状: {W_unembed.shape}")
    
    # SVD of unembed matrix
    U, S, Vh = torch.linalg.svd(W_unembed.float(), full_matrices=False)
    print(f"  Unembed SVD前5个奇异值: {S[:5].tolist()}")
    print(f"  Unembed SVD前20个奇异值: {S[:20].tolist()}")
    
    # Effective rank of unembed (90% energy)
    cumsum = torch.cumsum(S**2, dim=0)
    eff_rank_90 = (cumsum < 0.9 * cumsum[-1]).sum().item() + 1
    eff_rank_99 = (cumsum < 0.99 * cumsum[-1]).sum().item() + 1
    print(f"  Unembed有效秩(90%能量): {eff_rank_90}")
    print(f"  Unembed有效秩(99%能量): {eff_rank_99}")
    
    # For each disamb pair, measure logit margin at each layer
    layers_to_test = discover_layers(model)
    sample_layers = layers_to_test[::max(1, len(layers_to_test)//10)][:10]
    
    results = {}
    
    for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
        tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
        tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
        
        h_a_layers = extract_all_layer_hidden(model, tokenizer, ctx_a, layers_to_test)
        h_b_layers = extract_all_layer_hidden(model, tokenizer, ctx_b, layers_to_test)
        
        if not h_a_layers or not h_b_layers:
            continue
        
        pair_results = {
            "logit_margin_per_layer": {},
            "hidden_cos_per_layer": {},
            "unembed_direction_contribution": {},
        }
        
        for li in sorted(h_a_layers.keys()):
            h_a = h_a_layers[li].float().squeeze(0)  # [hidden_dim]
            h_b = h_b_layers[li].float().squeeze(0)
            
            if h_a.shape != h_b.shape:
                continue
            
            # Logit margin: difference in logit for correct tokens
            logits_a = (h_a @ W_unembed.T)  # [vocab_size]
            logits_b = (h_b @ W_unembed.T)
            
            # Disambiguation signal: for context A, token_a should be higher
            margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
            margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
            
            # Average disambiguation logit margin
            disamb_margin = (margin_a + margin_b) / 2
            
            pair_results["logit_margin_per_layer"][str(li)] = round(disamb_margin, 4)
            
            # Hidden space cosine distance
            cos_dist = 1 - cos_sim(h_a, h_b)
            pair_results["hidden_cos_per_layer"][str(li)] = round(cos_dist, 4)
            
            # Direction analysis: how much does the difference direction project onto unembed
            delta = h_a - h_b  # [hidden_dim]
            delta_norm = torch.norm(delta).item()
            if delta_norm > 1e-8:
                delta_unit = delta / delta_norm
                
                # Project delta onto each unembed row
                # For the two target tokens
                proj_a = cos_sim(delta, W_unembed[tok_a])
                proj_b = cos_sim(delta, W_unembed[tok_b])
                
                pair_results["unembed_direction_contribution"][str(li)] = {
                    "delta_norm": round(delta_norm, 4),
                    "cos_to_tok_a": round(proj_a, 4),
                    "cos_to_tok_b": round(proj_b, 4),
                }
        
        results[word] = pair_results
    
    # Summary statistics
    summary = {
        "unembed_shape": list(W_unembed.shape),
        "unembed_top5_sv": S[:5].tolist(),
        "unembed_top20_sv": S[:20].tolist(),
        "eff_rank_90": eff_rank_90,
        "eff_rank_99": eff_rank_99,
    }
    
    # Average logit margin across all pairs and layers
    all_margins = []
    for word, pr in results.items():
        for li, m in pr.get("logit_margin_per_layer", {}).items():
            all_margins.append(abs(m))
    
    if all_margins:
        summary["mean_abs_logit_margin"] = round(float(np.mean(all_margins)), 4)
        summary["max_abs_logit_margin"] = round(float(np.max(all_margins)), 4)
    
    # Check at which layer disambiguation logit margin first appears
    first_appear_layers = []
    for word, pr in results.items():
        margins = pr.get("logit_margin_per_layer", {})
        for li_str, m in sorted(margins.items(), key=lambda x: int(x[0])):
            if abs(m) > 0.1:  # threshold
                first_appear_layers.append(int(li_str))
                break
    
    if first_appear_layers:
        summary["mean_first_appear_layer"] = round(float(np.mean(first_appear_layers)), 1)
        summary["min_first_appear_layer"] = min(first_appear_layers)
    
    # Check correlation between hidden cos_dist and logit margin
    hidden_dists = []
    logit_margins = []
    for word, pr in results.items():
        for li in pr.get("hidden_cos_per_layer", {}):
            if li in pr.get("logit_margin_per_layer", {}):
                hidden_dists.append(pr["hidden_cos_per_layer"][li])
                logit_margins.append(abs(pr["logit_margin_per_layer"][li]))
    
    if len(hidden_dists) > 5:
        corr = float(np.corrcoef(hidden_dists, logit_margins)[0, 1])
        summary["hidden_logit_correlation"] = round(corr, 4)
    
    print(f"\n  --- Stage625 摘要 ---")
    print(f"  Unembed有效秩(90%能量): {eff_rank_90}")
    print(f"  Unembed有效秩(99%能量): {eff_rank_99}")
    print(f"  平均logit margin: {summary.get('mean_abs_logit_margin', 'N/A')}")
    print(f"  最大logit margin: {summary.get('max_abs_logit_margin', 'N/A')}")
    print(f"  消歧logit首次出现层: {summary.get('mean_first_appear_layer', 'N/A')}")
    print(f"  Hidden-logit相关性: {summary.get('hidden_logit_correlation', 'N/A')}")
    
    # Per-pair detail
    for word, pr in results.items():
        margins = pr.get("logit_margin_per_layer", {})
        if margins:
            layers_sorted = sorted(margins.items(), key=lambda x: int(x[0]))
            first_l, first_m = layers_sorted[0]
            last_l, last_m = layers_sorted[-1]
            print(f"  [{word}] L{first_l} margin={first_m}, L{last_l} margin={last_m}")
    
    summary["per_pair"] = results
    return summary


# ============ Stage626: 旋转速度→消歧度的因果关系 ============

def get_mlp_module(layer_module):
    for name, child in layer_module.named_children():
        nl = name.lower()
        if 'mlp' in nl or 'feed_forward' in nl or 'ffn' in nl:
            return child
    return None


def run_stage626(model, tokenizer, model_key):
    """通过修改MLP权重改变旋转速度，测试对消歧度的影响"""
    print("\n" + "="*70)
    print("Stage626: 旋转速度→消歧度的因果关系")
    print("="*70)
    
    layers_to_test = discover_layers(model)
    
    # Baseline: measure disambiguation at each layer
    def measure_disamb_at_layers(model_ref):
        """测量每层的消歧度(logit margin)"""
        W_unembed = get_unembed_matrix(model_ref)
        if W_unembed is None:
            return {}
        
        disamb_results = {}
        
        for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
            tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
            tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
            
            h_a_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_a, layers_to_test)
            h_b_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_b, layers_to_test)
            
            if not h_a_layers or not h_b_layers:
                continue
            
            for li in sorted(h_a_layers.keys()):
                h_a = h_a_layers[li].float().squeeze(0)
                h_b = h_b_layers[li].float().squeeze(0)
                
                if h_a.shape != h_b.shape:
                    continue
                
                logits_a = (h_a @ W_unembed.T)
                logits_b = (h_b @ W_unembed.T)
                
                margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
                margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
                disamb_margin = (margin_a + margin_b) / 2
                
                if li not in disamb_results:
                    disamb_results[li] = []
                disamb_results[li].append(abs(disamb_margin))
        
        # Average across pairs
        avg_disamb = {}
        for li, margins in disamb_results.items():
            avg_disamb[li] = float(np.mean(margins))
        
        return avg_disamb
    
    def measure_rotation_speed(model_ref):
        """测量消歧方向每层的平均旋转角度"""
        rotations = []
        
        for ctx_a, ctx_b, word, _, _ in DISAMB_PAIRS[:3]:  # Use first 3 for speed
            h_a_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_a, layers_to_test)
            h_b_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_b, layers_to_test)
            
            if not h_a_layers or not h_b_layers:
                continue
            
            prev_dir = None
            for li in sorted(h_a_layers.keys()):
                h_a = h_a_layers[li].float().squeeze(0)
                h_b = h_b_layers[li].float().squeeze(0)
                delta = h_a - h_b
                delta_norm = torch.norm(delta)
                if delta_norm < 1e-8:
                    continue
                curr_dir = delta / delta_norm
                
                if prev_dir is not None:
                    cos_val = cos_sim(prev_dir, curr_dir)
                    cos_val = max(-1.0, min(1.0, cos_val))
                    angle = np.degrees(np.arccos(cos_val))
                    rotations.append(angle)
                
                prev_dir = curr_dir
        
        return float(np.mean(rotations)) if rotations else 0.0
    
    # Save original MLP weights from discover_layers modules
    mlp_modules = []
    mlp_original_params = []
    for li in range(min(len(layers_to_test), 10)):  # First 10 layers
        layer = layers_to_test[li]
        mlp = get_mlp_module(layer)
        if mlp is not None:
            params = {}
            for name, param in mlp.named_parameters():
                if param.dim() >= 2:
                    params[name] = param.data.clone()
            if params:
                mlp_modules.append((li, mlp))
                mlp_original_params.append(params)
    
    if not mlp_modules:
        print("  [SKIP] 无法找到MLP模块")
        return {}
    
    print(f"  找到 {len(mlp_modules)} 个MLP模块用于干预")
    
    # Baseline measurements
    print("  测量baseline...")
    baseline_disamb = measure_disamb_at_layers(model)
    baseline_rotation = measure_rotation_speed(model)
    n_layers = len(layers_to_test)
    baseline_final_disamb = baseline_disamb.get(n_layers - 1, 0) if baseline_disamb else 0
    
    print(f"  Baseline旋转速度: {baseline_rotation:.1f}°/层")
    print(f"  Baseline末层消歧度(logit margin): {baseline_final_disamb:.4f}")
    
    # Intervention 1: Scale MLP weights by 0.5 (reduce rotation)
    print("\n  干预1: MLP权重×0.5 (减小旋转)...")
    for (li, mlp), orig_params in zip(mlp_modules, mlp_original_params):
        for name, param in mlp.named_parameters():
            if name in orig_params and param.dim() >= 2:
                param.data.copy_(orig_params[name] * 0.5)
    
    torch.cuda.empty_cache()
    reduced_disamb = measure_disamb_at_layers(model)
    reduced_rotation = measure_rotation_speed(model)
    reduced_final_disamb = reduced_disamb.get(n_layers - 1, 0) if reduced_disamb else 0
    
    print(f"  缩小后旋转速度: {reduced_rotation:.1f}°/层")
    print(f"  缩小后末层消歧度: {reduced_final_disamb:.4f}")
    
    # Restore original weights
    for (li, mlp), orig_params in zip(mlp_modules, mlp_original_params):
        for name, param in mlp.named_parameters():
            if name in orig_params and param.dim() >= 2:
                param.data.copy_(orig_params[name])
    torch.cuda.empty_cache()
    
    # Intervention 2: Scale MLP weights by 2.0 (increase rotation)
    print("\n  干预2: MLP权重×2.0 (增大旋转)...")
    for (li, mlp), orig_params in zip(mlp_modules, mlp_original_params):
        for name, param in mlp.named_parameters():
            if name in orig_params and param.dim() >= 2:
                param.data.copy_(orig_params[name] * 2.0)
    
    torch.cuda.empty_cache()
    increased_disamb = measure_disamb_at_layers(model)
    increased_rotation = measure_rotation_speed(model)
    increased_final_disamb = increased_disamb.get(n_layers - 1, 0) if increased_disamb else 0
    
    print(f"  增大后旋转速度: {increased_rotation:.1f}°/层")
    print(f"  增大后末层消歧度: {increased_final_disamb:.4f}")
    
    # Restore original weights
    for (li, mlp), orig_params in zip(mlp_modules, mlp_original_params):
        for name, param in mlp.named_parameters():
            if name in orig_params and param.dim() >= 2:
                param.data.copy_(orig_params[name])
    torch.cuda.empty_cache()
    
    # Summary
    rotation_change_reduced = reduced_rotation - baseline_rotation
    rotation_change_increased = increased_rotation - baseline_rotation
    disamb_change_reduced = reduced_final_disamb - baseline_final_disamb
    disamb_change_increased = increased_final_disamb - baseline_final_disamb
    
    summary = {
        "baseline_rotation_speed": round(baseline_rotation, 1),
        "baseline_final_disamb": round(baseline_final_disamb, 4),
        "reduced_rotation_speed": round(reduced_rotation, 1),
        "reduced_final_disamb": round(reduced_final_disamb, 4),
        "increased_rotation_speed": round(increased_rotation, 1),
        "increased_final_disamb": round(increased_final_disamb, 4),
        "rotation_change_reduced": round(rotation_change_reduced, 1),
        "disamb_change_reduced": round(disamb_change_reduced, 4),
        "rotation_change_increased": round(rotation_change_increased, 1),
        "disamb_change_increased": round(disamb_change_increased, 4),
        "causal_direction": "",
        "num_layers_modified": len(mlp_modules),
    }
    
    # Determine causal direction
    if rotation_change_reduced < 0 and disamb_change_reduced > 0:
        summary["causal_direction"] = "NEGATIVE: 减小旋转→增大消歧"
    elif rotation_change_reduced < 0 and disamb_change_reduced < 0:
        summary["causal_direction"] = "POSITIVE: 减小旋转→减小消歧"
    elif rotation_change_increased > 0 and disamb_change_increased < 0:
        summary["causal_direction"] = "NEGATIVE: 增大旋转→减小消歧"
    elif rotation_change_increased > 0 and disamb_change_increased > 0:
        summary["causal_direction"] = "POSITIVE: 增大旋转→增大消歧"
    else:
        summary["causal_direction"] = "UNCLEAR"
    
    print(f"\n  --- Stage626 摘要 ---")
    print(f"  Baseline: 旋转={baseline_rotation:.1f}°, 消歧={baseline_final_disamb:.4f}")
    print(f"  MLP×0.5: 旋转={reduced_rotation:.1f}°(Δ{rotation_change_reduced:+.1f}°), 消歧={reduced_final_disamb:.4f}(Δ{disamb_change_reduced:+.4f})")
    print(f"  MLP×2.0: 旋转={increased_rotation:.1f}°(Δ{rotation_change_increased:+.1f}°), 消歧={increased_final_disamb:.4f}(Δ{disamb_change_increased:+.4f})")
    print(f"  因果方向: {summary['causal_direction']}")
    
    # Per-layer comparison
    per_layer = {}
    for li in layers_to_test:
        if li in baseline_disamb:
            per_layer[str(li)] = {
                "baseline": round(baseline_disamb[li], 4),
                "reduced": round(reduced_disamb.get(li, 0), 4),
                "increased": round(increased_disamb.get(li, 0), 4),
            }
    summary["per_layer"] = per_layer
    
    return summary


# ============ Stage627: 消歧信息在logit空间的逐层演化 ============

def run_stage627(model, tokenizer, model_key):
    """追踪消歧信息在logit空间的逐层演化"""
    print("\n" + "="*70)
    print("Stage627: 消歧信息在logit空间的逐层演化")
    print("="*70)
    
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}
    
    layers_to_test = discover_layers(model)
    
    results = {}
    
    for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
        tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
        tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
        
        h_a_layers = extract_all_layer_hidden(model, tokenizer, ctx_a, layers_to_test)
        h_b_layers = extract_all_layer_hidden(model, tokenizer, ctx_b, layers_to_test)
        
        if not h_a_layers or not h_b_layers:
            continue
        
        pair_data = {
            "layers": [],
            "logit_margin": [],
            "logit_correct_a": [],
            "logit_correct_b": [],
            "hidden_cos_dist": [],
            "logit_entropy_a": [],
            "logit_entropy_b": [],
        }
        
        for li in sorted(h_a_layers.keys()):
            h_a = h_a_layers[li].float().squeeze(0)
            h_b = h_b_layers[li].float().squeeze(0)
            
            if h_a.shape != h_b.shape:
                continue
            
            logits_a = (h_a @ W_unembed.T)
            logits_b = (h_b @ W_unembed.T)
            
            # Logit margin
            margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
            margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
            disamb_margin = (margin_a + margin_b) / 2
            
            # Hidden cos dist
            cos_dist = 1 - cos_sim(h_a, h_b)
            
            # Correct token logits
            correct_a = logits_a[tok_a].item()
            correct_b = logits_b[tok_b].item()
            
            # Entropy of logit distribution (top-100)
            topk_a = torch.topk(logits_a, 100)
            topk_b = torch.topk(logits_b, 100)
            probs_a = F.softmax(topk_a.values, dim=0)
            probs_b = F.softmax(topk_b.values, dim=0)
            entropy_a = -(probs_a * torch.log(probs_a + 1e-10)).sum().item()
            entropy_b = -(probs_b * torch.log(probs_b + 1e-10)).sum().item()
            
            pair_data["layers"].append(li)
            pair_data["logit_margin"].append(round(disamb_margin, 4))
            pair_data["logit_correct_a"].append(round(correct_a, 4))
            pair_data["logit_correct_b"].append(round(correct_b, 4))
            pair_data["hidden_cos_dist"].append(round(cos_dist, 4))
            pair_data["logit_entropy_a"].append(round(entropy_a, 4))
            pair_data["logit_entropy_b"].append(round(entropy_b, 4))
        
        results[word] = pair_data
        
        if pair_data["logit_margin"]:
            max_margin_idx = np.argmax(np.abs(pair_data["logit_margin"]))
            print(f"  [{word}] 最大margin在L{pair_data['layers'][max_margin_idx]}={pair_data['logit_margin'][max_margin_idx]}, "
                  f"末层L{pair_data['layers'][-1]}={pair_data['logit_margin'][-1]}")
    
    # Summary: averaged across pairs
    summary = {}
    
    # Find common layers
    all_layer_lists = [r["layers"] for r in results.values() if r["layers"]]
    if not all_layer_lists:
        return summary
    
    # Find layers present in all pairs
    common_layers = set(all_layer_lists[0])
    for ll in all_layer_lists[1:]:
        common_layers &= set(ll)
    common_layers = sorted(common_layers)
    
    if not common_layers:
        return summary
    
    avg_margin = []
    avg_cos_dist = []
    avg_entropy_a = []
    avg_entropy_b = []
    
    for li in common_layers:
        margins = []
        cos_dists = []
        ent_a = []
        ent_b = []
        
        for word, r in results.items():
            if li in r["layers"]:
                idx = r["layers"].index(li)
                margins.append(r["logit_margin"][idx])
                cos_dists.append(r["hidden_cos_dist"][idx])
                ent_a.append(r["logit_entropy_a"][idx])
                ent_b.append(r["logit_entropy_b"][idx])
        
        avg_margin.append(float(np.mean(margins)))
        avg_cos_dist.append(float(np.mean(cos_dists)))
        avg_entropy_a.append(float(np.mean(ent_a)))
        avg_entropy_b.append(float(np.mean(ent_b)))
    
    summary["layers"] = common_layers
    summary["avg_abs_logit_margin"] = [round(abs(m), 4) for m in avg_margin]
    summary["avg_logit_margin_signed"] = [round(m, 4) for m in avg_margin]
    summary["avg_hidden_cos_dist"] = [round(c, 4) for c in avg_cos_dist]
    summary["avg_entropy_a"] = [round(e, 4) for e in avg_entropy_a]
    summary["avg_entropy_b"] = [round(e, 4) for e in avg_entropy_b]
    
    # Key metrics
    abs_margins = [abs(m) for m in avg_margin]
    max_margin_idx = np.argmax(abs_margins)
    summary["max_margin_layer"] = common_layers[max_margin_idx]
    summary["max_margin_value"] = abs_margins[max_margin_idx]
    
    # First layer where margin > threshold
    first_significant = None
    for i, (li, m) in enumerate(zip(common_layers, abs_margins)):
        if m > 0.5:
            first_significant = li
            break
    summary["first_significant_margin_layer"] = first_significant
    summary["first_significant_margin_value"] = round(abs_margins[common_layers.index(first_significant)], 4) if first_significant is not None else None
    
    # Margin growth rate (per layer in second half)
    n = len(common_layers)
    mid = n // 2
    if mid < n - 1:
        growth_rate = (abs_margins[-1] - abs_margins[mid]) / (n - 1 - mid) if abs_margins[mid] > 0.01 else 0
        summary["margin_growth_rate_late"] = round(growth_rate, 4)
    
    # Final margin
    summary["final_abs_margin"] = abs_margins[-1]
    summary["final_signed_margin"] = avg_margin[-1]
    
    # Monotonicity of margin (what fraction of steps increase)
    increases = sum(1 for i in range(1, len(abs_margins)) if abs_margins[i] > abs_margins[i-1])
    summary["margin_monotonicity"] = round(increases / max(1, len(abs_margins) - 1), 3)
    
    print(f"\n  --- Stage627 摘要 ---")
    print(f"  最大margin层: L{summary['max_margin_layer']}, 值={summary['max_margin_value']:.4f}")
    print(f"  首次显著margin层: L{summary.get('first_significant_margin_layer', 'N/A')}")
    print(f"  末层margin: {summary['final_abs_margin']:.4f} (signed={summary['final_signed_margin']:.4f})")
    print(f"  Margin单调性: {summary['margin_monotonicity']:.3f}")
    print(f"  后期增长率: {summary.get('margin_growth_rate_late', 'N/A')}")
    
    # Evolution curve summary
    print(f"  Margin演化 (每5层):")
    step = max(1, len(common_layers) // 10)
    for i in range(0, len(common_layers), step):
        li = common_layers[i]
        m = abs_margins[i]
        cd = avg_cos_dist[i]
        print(f"    L{li}: margin={m:.4f}, cos_dist={cd:.4f}")
    
    summary["per_pair"] = results
    return summary


# ============ Main ============

def main():
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    valid_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    if model_key not in valid_keys:
        print(f"用法: python {Path(__file__).name} [{'|'.join(valid_keys)}]")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Stage625-626-627: Unembed全息读出 + 旋转因果 + logit演化")
    print(f"模型: {model_key}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    bundle = load_model_bundle(model_key)
    if isinstance(bundle, tuple):
        model, tokenizer = bundle[0], bundle[1]
    else:
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
    model.eval()
    
    try:
        # Stage625
        t1 = time.time()
        r625 = run_stage625(model, tokenizer, model_key)
        print(f"\n  Stage625完成 ({time.time()-t1:.0f}s)")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stage626
        t2 = time.time()
        r626 = run_stage626(model, tokenizer, model_key)
        print(f"\n  Stage626完成 ({time.time()-t2:.0f}s)")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stage627
        t3 = time.time()
        r627 = run_stage627(model, tokenizer, model_key)
        print(f"\n  Stage627完成 ({time.time()-t3:.0f}s)")
        
    finally:
        free_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    
    total_time = time.time() - t0
    
    # Combine results
    combined = {
        "model": model_key,
        "timestamp": TIMESTAMP,
        "total_time": round(total_time, 1),
        "stage625": r625,
        "stage626": r626,
        "stage627": r627,
    }
    
    out_path = OUTPUT_DIR / f"stage625_626_627_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {total_time:.0f}s")


if __name__ == "__main__":
    main()
