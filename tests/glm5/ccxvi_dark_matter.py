"""
CCXVI(366): 跨层概念漂移与"暗物质"分析
=========================================

CCXVB发现: 概念方向在层间几乎不变(cos=0.82-0.97), 
86-92%的概念信号不在W_U行空间中("暗物质")。

本实验回答:
1. 跨多层后概念方向偏离多少? cos(Δ_l, Δ_{l+k}) for k=3,6,12
2. "暗物质"是什么? 不在W_U行空间中的86%分量在哪里?
3. 概念方向的范数如何变化? 是增长还是衰减?
4. Attn/MLP对概念方向的逐层修改是什么模式?

三个实验:
  Exp1: 跨层概念漂移 — cos(Δ_l, Δ_{l+k}) for k=1,3,6,12,18
  Exp2: "暗物质"分析 — Δ在W_E, W_U, Attn/MLP输出空间中的投影
  Exp3: Attn/MLP贡献分解 — 每层的Attn输出和MLP输出对Δ的贡献

用法:
  python ccxvi_dark_matter.py --model qwen3 --exp 1
  python ccxvi_dark_matter.py --model qwen3 --exp 2
  python ccxvi_dark_matter.py --model qwen3 --exp 3
  python ccxvi_dark_matter.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANS_TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

TEMP = Path("tests/glm5_temp")

CONCEPTS = {
    "apple": {
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
    },
}

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


def collect_attn_mlp_outputs(model, tokenizer, device, text, target_layer):
    """
    收集目标层的Attn输出和MLP输出
    
    Transformer层结构:
    h_post_ln1 = LN1(h)
    attn_out = Attn(h_post_ln1)  → 形状 [seq_len, d_model]
    h_post_attn = h + attn_out    → Attn后的残差
    
    h_post_ln2 = LN2(h_post_attn)
    mlp_out = MLP(h_post_ln2)     → 形状 [seq_len, d_model]
    h_post_mlp = h_post_attn + mlp_out  → 层输出
    
    我们需要: attn_out 和 mlp_out
    """
    all_layers = get_layers(model)
    if target_layer >= len(all_layers):
        return None, None
    
    layer = all_layers[target_layer]
    captured = {}
    
    # Hook on self_attn output
    def make_attn_hook():
        def hook(module, inp, output):
            if isinstance(output, tuple):
                # Attn output: (hidden_states, attn_weights, past_key_value)
                captured['attn_out'] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured['attn_out'] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    
    # Hook on MLP output
    def make_mlp_hook():
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured['mlp_out'] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured['mlp_out'] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    
    hooks = []
    if hasattr(layer, 'self_attn'):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook()))
    if hasattr(layer, 'mlp'):
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook()))
    
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return None, None
    
    for h in hooks: h.remove()
    
    attn_out = captured.get('attn_out')
    mlp_out = captured.get('mlp_out')
    gc.collect()
    return attn_out, mlp_out


# ================================================================
# Exp1: 跨层概念漂移
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts, baseline_states):
    """
    cos(Δ_l, Δ_{l+k}) for k=1,3,6,12,18
    
    这直接衡量概念方向在多层传播后的偏离程度。
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Cross-Layer Concept Drift")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # 收集所有层的概念delta
    all_layers_capture = list(range(n_layers))
    
    print(f"\n  Collecting baseline at all {n_layers} layers...")
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_layers_capture)
    
    all_deltas_full = {}
    for cname, cdata in concepts.items():
        concept_states_list = {l: [] for l in all_layers_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_layers_capture)
            for l in all_layers_capture:
                if l in states:
                    concept_states_list[l].append(states[l])
        
        deltas = {}
        for l in all_layers_capture:
            if concept_states_list[l] and l in bl_all:
                mean_state = np.mean(concept_states_list[l], axis=0)
                deltas[l] = mean_state - bl_all[l]
        all_deltas_full[cname] = deltas
        del concept_states_list
        gc.collect()
    
    # 计算跨层漂移
    skip_layers = [1, 3, 6, 12, 18]
    ref_layers = [6, 12, 18, 24, 30]  # 参考层
    
    results = {}
    
    for ref_l in ref_layers:
        if ref_l >= n_layers:
            continue
        
        layer_results = {}
        
        for cname, deltas in all_deltas_full.items():
            if ref_l not in deltas:
                continue
            
            delta_ref = deltas[ref_l]
            n_ref = np.linalg.norm(delta_ref)
            if n_ref < 1e-8:
                continue
            
            drift_data = {}
            for k in skip_layers:
                target_l = ref_l + k
                if target_l >= n_layers:
                    continue
                if target_l not in deltas:
                    continue
                
                delta_target = deltas[target_l]
                n_target = np.linalg.norm(delta_target)
                if n_target < 1e-8:
                    continue
                
                cos = float(np.dot(delta_ref, delta_target) / (n_ref * n_target))
                norm_ratio = n_target / n_ref
                
                drift_data[str(k)] = {
                    "cosine": cos,
                    "norm_ratio": norm_ratio,
                    "target_layer": target_l,
                }
            
            layer_results[cname] = drift_data
        
        results[str(ref_l)] = layer_results
    
    # 打印结果
    print(f"\n  Cross-layer concept drift:")
    for ref_l, layer_data in results.items():
        print(f"  Ref L{ref_l}:")
        for cname, drift in layer_data.items():
            drift_str = ", ".join([f"k={k}: cos={v['cosine']:.3f}" for k, v in sorted(drift.items())])
            print(f"    {cname}: {drift_str}")
    
    return results


# ================================================================
# Exp2: "暗物质"分析
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts, baseline_states):
    """
    Δ不在W_U行空间中的86%分量是什么?
    
    分析Δ在以下空间中的投影:
    1. W_U行空间 (lm_head权重)
    2. W_E列空间 (输入嵌入)
    3. 各层Attn输出空间
    4. 各层MLP输出空间
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Dark Matter Analysis")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]
    
    # 收集所有层的概念delta
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    all_deltas_full = {}
    for cname, cdata in concepts.items():
        concept_states_list = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states:
                    concept_states_list[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states_list[l] and l in bl_all:
                mean_state = np.mean(concept_states_list[l], axis=0)
                deltas[l] = mean_state - bl_all[l]
        all_deltas_full[cname] = deltas
        del concept_states_list
        gc.collect()
    
    # W_U行空间基
    print(f"\n  Computing W_U row space basis...")
    W_U = get_W_U(model)  # [vocab_size, d_model]
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    
    # W_E列空间基
    print(f"  Computing W_E row space basis...")
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()  # [vocab_size, d_model]
    k_we = min(200, min(W_E.shape) - 2)
    # W_E行空间在R^d_model中, 由svds(W_E^T)的U张成
    U_we, s_we, _ = svds(W_E.T.astype(np.float32), k=k_we)
    U_we = np.asarray(U_we, dtype=np.float64)  # [d_model, k_we]
    del W_E
    gc.collect()
    
    # 收集各层的Attn/MLP输出(对baseline)
    print(f"  Collecting Attn/MLP outputs for baseline...")
    baseline_attn_outputs = {}
    baseline_mlp_outputs = {}
    for l in key_layers:
        attn_out, mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, BASELINE_TEXT, l)
        if attn_out is not None:
            baseline_attn_outputs[l] = attn_out
        if mlp_out is not None:
            baseline_mlp_outputs[l] = mlp_out
    
    # 收集各概念的Attn/MLP输出
    concept_attn_deltas = {}  # {concept: {layer: attn_delta}}
    concept_mlp_deltas = {}
    for cname, cdata in concepts.items():
        concept_attn_deltas[cname] = {}
        concept_mlp_deltas[cname] = {}
        # 用第一个template作为代表
        for template in cdata["templates"][:1]:
            for l in key_layers:
                attn_out, mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, template, l)
                if attn_out is not None and l in baseline_attn_outputs:
                    concept_attn_deltas[cname][l] = attn_out - baseline_attn_outputs[l]
                if mlp_out is not None and l in baseline_mlp_outputs:
                    concept_mlp_deltas[cname][l] = mlp_out - baseline_mlp_outputs[l]
        gc.collect()
    
    # 分析
    results = {}
    
    for l in key_layers:
        print(f"\n  --- Layer {l} ---")
        layer_results = {}
        
        for cname, deltas in all_deltas_full.items():
            if l not in deltas:
                continue
            
            delta = deltas[l]
            delta_norm_sq = np.linalg.norm(delta) ** 2
            if delta_norm_sq < 1e-16:
                continue
            
            # 1. W_U行空间投影
            proj_wu = U_wu.T @ delta
            ratio_wu = np.sum(proj_wu ** 2) / delta_norm_sq
            
            # 2. W_E行空间投影
            proj_we = U_we.T @ delta
            ratio_we = np.sum(proj_we ** 2) / delta_norm_sq
            
            # 3. W_U和W_E的并集投影
            U_combined = np.hstack([U_wu, U_we])
            # 正交化
            Q, R = np.linalg.qr(U_combined, mode='reduced')
            proj_combined = Q.T @ delta
            ratio_combined = np.sum(proj_combined ** 2) / delta_norm_sq
            
            # 4. Attn/MLP输出的方向
            cos_attn = 0.0
            cos_mlp = 0.0
            if cname in concept_attn_deltas and l in concept_attn_deltas[cname]:
                attn_delta = concept_attn_deltas[cname][l]
                n_attn = np.linalg.norm(attn_delta)
                if n_attn > 1e-8:
                    cos_attn = float(np.dot(delta, attn_delta) / (np.sqrt(delta_norm_sq) * n_attn))
            
            if cname in concept_mlp_deltas and l in concept_mlp_deltas[cname]:
                mlp_delta = concept_mlp_deltas[cname][l]
                n_mlp = np.linalg.norm(mlp_delta)
                if n_mlp > 1e-8:
                    cos_mlp = float(np.dot(delta, mlp_delta) / (np.sqrt(delta_norm_sq) * n_mlp))
            
            # 5. Attn/MLP输出在delta上的投影比
            proj_attn_ratio = 0.0
            proj_mlp_ratio = 0.0
            if cname in concept_attn_deltas and l in concept_attn_deltas[cname]:
                attn_delta = concept_attn_deltas[cname][l]
                proj_attn = np.dot(delta, attn_delta) / delta_norm_sq * attn_delta if np.linalg.norm(attn_delta) > 1e-8 else np.zeros_like(delta)
                proj_attn_ratio = np.sum(proj_attn ** 2) / delta_norm_sq
            if cname in concept_mlp_deltas and l in concept_mlp_deltas[cname]:
                mlp_delta = concept_mlp_deltas[cname][l]
                proj_mlp = np.dot(delta, mlp_delta) / delta_norm_sq * mlp_delta if np.linalg.norm(mlp_delta) > 1e-8 else np.zeros_like(delta)
                proj_mlp_ratio = np.sum(proj_mlp ** 2) / delta_norm_sq
            
            # 6. 残余 = delta - proj_to_WU - proj_to_WE
            residual = delta - Q @ (Q.T @ delta)
            residual_ratio = np.sum(residual ** 2) / delta_norm_sq
            
            layer_results[cname] = {
                "ratio_wu": float(ratio_wu),
                "ratio_we": float(ratio_we),
                "ratio_combined": float(ratio_combined),
                "cos_attn": float(cos_attn),
                "cos_mlp": float(cos_mlp),
                "proj_attn_ratio": float(proj_attn_ratio),
                "proj_mlp_ratio": float(proj_mlp_ratio),
                "residual_ratio": float(residual_ratio),
                "delta_norm": float(np.sqrt(delta_norm_sq)),
            }
            
            print(f"    {cname}: WU={ratio_wu:.3f}, WE={ratio_we:.3f}, "
                  f"combined={ratio_combined:.3f}, residual={residual_ratio:.3f}, "
                  f"cos_attn={cos_attn:.3f}, cos_mlp={cos_mlp:.3f}")
        
        results[str(l)] = layer_results
    
    # 随机基线
    np.random.seed(999)
    n_random = 20
    random_ratios = {"wu": [], "we": [], "combined": [], "residual": []}
    for _ in range(n_random):
        r = np.random.randn(d_model)
        r_norm_sq = np.linalg.norm(r) ** 2
        if r_norm_sq < 1e-16:
            continue
        proj_wu = U_wu.T @ r
        random_ratios["wu"].append(np.sum(proj_wu ** 2) / r_norm_sq)
        proj_we = U_we.T @ r
        random_ratios["we"].append(np.sum(proj_we ** 2) / r_norm_sq)
        U_combined = np.hstack([U_wu, U_we])
        Q, R = np.linalg.qr(U_combined, mode='reduced')
        proj_c = Q.T @ r
        random_ratios["combined"].append(np.sum(proj_c ** 2) / r_norm_sq)
        random_ratios["residual"].append(1.0 - np.sum(proj_c ** 2) / r_norm_sq)
    
    results["random_baseline"] = {
        "wu_mean": float(np.mean(random_ratios["wu"])),
        "wu_std": float(np.std(random_ratios["wu"])),
        "we_mean": float(np.mean(random_ratios["we"])),
        "we_std": float(np.std(random_ratios["we"])),
        "combined_mean": float(np.mean(random_ratios["combined"])),
        "combined_std": float(np.std(random_ratios["combined"])),
        "residual_mean": float(np.mean(random_ratios["residual"])),
        "residual_std": float(np.std(random_ratios["residual"])),
    }
    
    print(f"\n  Random baseline: WU={np.mean(random_ratios['wu']):.4f}, "
          f"WE={np.mean(random_ratios['we']):.4f}, "
          f"combined={np.mean(random_ratios['combined']):.4f}, "
          f"residual={np.mean(random_ratios['residual']):.4f}")
    
    return results


# ================================================================
# Exp3: Attn/MLP贡献分解
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts, baseline_states):
    """
    每层的Attn和MLP对概念方向的具体贡献
    
    Δ_{l+1} = Δ_l + Δ_attn_l + Δ_mlp_l
    
    其中 Δ_attn_l = Attn(h_concept) - Attn(h_baseline)
         Δ_mlp_l = MLP(h_post_attn_concept) - MLP(h_post_attn_baseline)
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: Attn/MLP Contribution Decomposition")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]
    
    # 收集所有层的delta
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    all_deltas_full = {}
    for cname, cdata in concepts.items():
        concept_states_list = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states:
                    concept_states_list[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states_list[l] and l in bl_all:
                mean_state = np.mean(concept_states_list[l], axis=0)
                deltas[l] = mean_state - bl_all[l]
        all_deltas_full[cname] = deltas
        del concept_states_list
        gc.collect()
    
    # 收集Attn/MLP输出
    print(f"  Collecting Attn/MLP outputs...")
    
    baseline_attn = {}
    baseline_mlp = {}
    for l in range(n_layers):
        attn_out, mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, BASELINE_TEXT, l)
        if attn_out is not None:
            baseline_attn[l] = attn_out
        if mlp_out is not None:
            baseline_mlp[l] = mlp_out
    
    concept_attn = {cname: {} for cname in concepts}
    concept_mlp = {cname: {} for cname in concepts}
    
    for cname, cdata in concepts.items():
        for template in cdata["templates"]:
            for l in range(n_layers):
                attn_out, mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, template, l)
                if attn_out is not None and l in baseline_attn:
                    if l not in concept_attn[cname]:
                        concept_attn[cname][l] = []
                    concept_attn[cname][l].append(attn_out)
                if mlp_out is not None and l in baseline_mlp:
                    if l not in concept_mlp[cname]:
                        concept_mlp[cname][l] = []
                    concept_mlp[cname][l].append(mlp_out)
            gc.collect()
    
    # 计算Attn/MLP的delta
    results = {}
    
    for l in key_layers:
        print(f"\n  --- Layer {l} ---")
        layer_results = {}
        
        for cname, deltas in all_deltas_full.items():
            if l not in deltas:
                continue
            
            delta_l = deltas[l]
            delta_l_norm = np.linalg.norm(delta_l)
            if delta_l_norm < 1e-8:
                continue
            
            # Attn delta
            attn_delta = None
            if cname in concept_attn and l in concept_attn[cname] and l in baseline_attn:
                attn_mean = np.mean(concept_attn[cname][l], axis=0)
                attn_delta = attn_mean - baseline_attn[l]
            
            # MLP delta
            mlp_delta = None
            if cname in concept_mlp and l in concept_mlp[cname] and l in baseline_mlp:
                mlp_mean = np.mean(concept_mlp[cname][l], axis=0)
                mlp_delta = mlp_mean - baseline_mlp[l]
            
            # 分析
            n_attn = np.linalg.norm(attn_delta) if attn_delta is not None else 0
            n_mlp = np.linalg.norm(mlp_delta) if mlp_delta is not None else 0
            
            attn_to_delta = float(np.dot(attn_delta, delta_l) / (n_attn * delta_l_norm)) if n_attn > 1e-8 else 0
            mlp_to_delta = float(np.dot(mlp_delta, delta_l) / (n_mlp * delta_l_norm)) if n_mlp > 1e-8 else 0
            
            # Attn/MLP delta相对于delta_l的大小
            attn_ratio = n_attn / delta_l_norm
            mlp_ratio = n_mlp / delta_l_norm
            
            # 验证: delta_{l+1} ≈ delta_l + attn_delta + mlp_delta?
            if l + 1 in deltas:
                delta_next = deltas[l + 1]
                predicted_next = delta_l
                if attn_delta is not None:
                    predicted_next = predicted_next + attn_delta
                if mlp_delta is not None:
                    predicted_next = predicted_next + mlp_delta
                
                n_pred = np.linalg.norm(predicted_next)
                n_next = np.linalg.norm(delta_next)
                
                cos_pred_actual = float(np.dot(predicted_next, delta_next) / (n_pred * n_next)) if n_pred > 1e-8 and n_next > 1e-8 else 0
            else:
                cos_pred_actual = None
            
            layer_results[cname] = {
                "attn_norm_ratio": float(attn_ratio),
                "mlp_norm_ratio": float(mlp_ratio),
                "cos_attn_delta": float(attn_to_delta),
                "cos_mlp_delta": float(mlp_to_delta),
                "cos_pred_actual": float(cos_pred_actual) if cos_pred_actual is not None else None,
            }
            
            print(f"    {cname}: attn_ratio={attn_ratio:.3f}, mlp_ratio={mlp_ratio:.3f}, "
                  f"cos_attn={attn_to_delta:.3f}, cos_mlp={mlp_to_delta:.3f}")
        
        results[str(l)] = layer_results
    
    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXVI: Cross-Layer Drift & Dark Matter — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    # 收集baseline
    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]
    all_capture = set(key_layers)
    for l in key_layers:
        all_capture.add(l + 1)
    all_capture = sorted([l for l in all_capture if l < n_layers])
    
    baseline_states, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    all_results = {}

    if args.exp in ["1", "all"]:
        exp1_results = run_exp1(model, tokenizer, device, model_info, CONCEPTS, baseline_states)
        all_results["exp1"] = exp1_results

    if args.exp in ["2", "all"]:
        exp2_results = run_exp2(model, tokenizer, device, model_info, CONCEPTS, baseline_states)
        all_results["exp2"] = exp2_results

    if args.exp in ["3", "all"]:
        exp3_results = run_exp3(model, tokenizer, device, model_info, CONCEPTS, baseline_states)
        all_results["exp3"] = exp3_results

    # 保存
    output_path = TEMP / f"ccxvi_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    release_model(model)
    print(f"\nCCXVI {model_name} 完成!")


if __name__ == "__main__":
    main()
