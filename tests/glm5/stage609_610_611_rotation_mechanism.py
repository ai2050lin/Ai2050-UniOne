#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage609-610-611: 旋转机制深度分解 + 旋转轴子空间分析 + Gemma4消歧修复

Stage609: Attention vs MLP旋转贡献的定量分离
  - 逐层分别零化MLP和Attention，测量各自的旋转角度贡献
  - 验证"Attention负责初始重组，MLP负责持续旋转"假说

Stage610: 旋转轴子空间分析
  - 对消歧方向逐层变化做SVD，确定旋转是否发生在固定子空间内
  - 如果旋转轴子空间维度很低（<<hidden_dim），则旋转是结构化的

Stage611: Gemma4消歧失败修复实验
  - 选择性零化Gemma4的某些高旋转MLP层，降低旋转速度
  - 验证消歧度是否恢复，确认旋转速度是消歧失败的根因

用法: python stage609_610_611_rotation_mechanism.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers,
    encode_to_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


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
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def find_mlp_and_attn_components(layer_module):
    """找出transformer层中的MLP和Attention子模块"""
    attn_module = None
    mlp_module = None

    for name, child in layer_module.named_children():
        name_lower = name.lower()
        if 'attn' in name_lower or 'attention' in name_lower or 'self_attn' in name_lower:
            attn_module = child
        if 'mlp' in name_lower or 'feed_forward' in name_lower or 'ffn' in name_lower:
            mlp_module = child

    return attn_module, mlp_module


DISAMB_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank"),
    ("She ate a red apple.", "Apple released the iPhone.", "apple"),
    ("The factory plant employs workers.", "She watered the plant.", "plant"),
    ("The hot spring resort.", "Spring is beautiful.", "spring"),
    ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
]


def get_all_hidden_states(model, tokenizer, sentences):
    """获取所有句子的所有层hidden states"""
    states = {}
    for s in sentences:
        h_list = []
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for h in out.hidden_states:
            h_list.append(h[0, -1, :].float().cpu())
        states[s] = h_list
    return states


# ============ Stage609: Attention vs MLP旋转贡献定量分离 ============

def run_stage609(model, tokenizer, model_key):
    """
    逐层分别零化MLP和Attention的输出，定量分离两者的旋转贡献。

    算法原理（使用hook拦截输出）：
    - 正常：output_i = x + Attn(x) + MLP(x + Attn(x))
    - 零化MLP输出：MLP输出被替换为零 → 只有Attention贡献
    - 零化Attn输出：Attn输出被替换为零 → 只有MLP贡献
    
    对于消歧方向 d_i = h_i(ctx1) - h_i(ctx2)：
    - 旋转角度 = arccos(d_{i-1} · d_i / |d_{i-1}| |d_i|)
    - 分别测量只保留Attn、只保留MLP时的旋转角度
    """
    print(f"\n  --- Stage609: Attention vs MLP旋转贡献定量分离 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    def make_zero_hook(return_tuple=False):
        """创建一个将输出替换为零的hook"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                x = output[0]
                zeros = torch.zeros_like(x)
                return (zeros,) + output[1:]
            else:
                return torch.zeros_like(output)
        return hook_fn

    def run_with_zeroing(target_layer_idx, zero_attn=False, zero_mlp=False, sentences=None):
        """用hook零化指定层的attn/mlp，返回hidden states"""
        layer = layers[target_layer_idx]
        attn_mod, mlp_mod = find_mlp_and_attn_components(layer)
        
        hooks = []
        if zero_attn and attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_zero_hook(return_tuple=True)))
        if zero_mlp and mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(make_zero_hook(return_tuple=False)))

        try:
            all_states = {}
            for s in sentences:
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
                enc = move_to_device(enc, model)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                states = [h[0, -1, :].float().cpu() for h in out.hidden_states]
                all_states[s] = states
        finally:
            for h in hooks:
                h.remove()
        
        return all_states

    results = {}

    for s1, s2, word in DISAMB_PAIRS[:3]:
        print(f"\n    Processing '{word}'...")

        # 1. Normal pass: get all hidden states
        normal_states = run_with_zeroing(0, zero_attn=False, zero_mlp=False, sentences=[s1, s2])
        h1_normal = normal_states[s1]
        h2_normal = normal_states[s2]

        # Find peak layer
        peak_l = 0
        max_d = 0
        for li in range(n_layers):
            d = 1 - cos_sim(h1_normal[li], h2_normal[li])
            if d > max_d:
                max_d = d
                peak_l = li

        # 2. For each layer, measure rotation under different ablations
        layer_contributions = []
        target_layers = list(range(1, min(n_layers, peak_l + 4)))

        for li in target_layers:
            layer_info = {"layer": li}

            for abl_name, z_attn, z_mlp in [
                ("normal", False, False),
                ("zero_attn", True, False),
                ("zero_mlp", False, True),
            ]:
                abl_states = run_with_zeroing(li, zero_attn=z_attn, zero_mlp=z_mlp, sentences=[s1, s2])
                h1_a = abl_states[s1]
                h2_a = abl_states[s2]

                if li < len(h1_a) and li > 0:
                    d_prev = h1_a[li - 1] - h2_a[li - 1]
                    d_curr = h1_a[li] - h2_a[li]
                    d_prev_n = torch.norm(d_prev).item()
                    d_curr_n = torch.norm(d_curr).item()
                    if d_prev_n > 1e-10 and d_curr_n > 1e-10:
                        dc = cos_sim(d_prev, d_curr)
                        rot = np.degrees(np.arccos(np.clip(dc, -1, 1)))
                        disamb = 1 - cos_sim(h1_a[li], h2_a[li])
                    else:
                        rot = 0.0
                        disamb = 0.0
                else:
                    rot = 0.0
                    disamb = 0.0
                layer_info[f"rot_{abl_name}"] = round(rot, 2)
                layer_info[f"disamb_{abl_name}"] = round(disamb, 4)

            # Compute contributions
            rot_n = layer_info["rot_normal"]
            rot_za = layer_info["rot_zero_attn"]
            rot_zm = layer_info["rot_zero_mlp"]

            # If zeroing attn reduces rotation → attn contributes to rotation
            # If zeroing mlp reduces rotation → mlp contributes to rotation
            attn_contrib = rot_n - rot_za
            mlp_contrib = rot_n - rot_zm

            layer_info["attn_contribution_deg"] = round(attn_contrib, 2)
            layer_info["mlp_contribution_deg"] = round(mlp_contrib, 2)
            layer_info["attn_frac"] = round(attn_contrib / max(rot_n, 0.01), 3) if rot_n > 0.01 else 0
            layer_info["mlp_frac"] = round(mlp_contrib / max(rot_n, 0.01), 3) if rot_n > 0.01 else 0

            layer_contributions.append(layer_info)

        # Summary
        if layer_contributions:
            avg_attn = np.mean([lc["attn_contribution_deg"] for lc in layer_contributions])
            avg_mlp = np.mean([lc["mlp_contribution_deg"] for lc in layer_contributions])
            # Compare early layers (L1-L3) vs late layers (after peak-2)
            early = [lc for lc in layer_contributions if lc["layer"] <= 3]
            late = [lc for lc in layer_contributions if lc["layer"] > peak_l - 2]
            early_attn = np.mean([lc["attn_frac"] for lc in early]) if early else 0
            late_mlp = np.mean([lc["mlp_frac"] for lc in late]) if late else 0

            results[word] = {
                "peak_layer": peak_l,
                "peak_disamb": round(max_d, 4),
                "avg_attn_contribution_deg": round(float(avg_attn), 2),
                "avg_mlp_contribution_deg": round(float(avg_mlp), 2),
                "early_layers_attn_frac": round(float(early_attn), 3),
                "late_layers_mlp_frac": round(float(late_mlp), 3),
                "layer_data": layer_contributions,
            }
            print(f"    {word}: peak=L{peak_l}, "
                  f"avg_attn_contrib={float(avg_attn):.1f}deg, "
                  f"avg_mlp_contrib={float(avg_mlp):.1f}deg")

    elapsed = time.time() - t0
    print(f"  Stage609 done in {elapsed:.1f}s")
    return {"attn_mlp_rotation_separation": results}


# ============ Stage610: 旋转轴子空间分析 ============

def run_stage610(model, tokenizer, model_key):
    """
    对消歧方向逐层变化做PCA/SVD分析，确定旋转是否发生在固定子空间内。

    算法原理：
    - 收集所有歧义词在所有层之间的差异向量变化 delta_d_l = d_{l+1} - d_l
    - 将这些delta_d堆叠成矩阵 D = [delta_d_1, delta_d_2, ..., delta_d_L]
    - 对 D 做SVD，检查奇异值的分布
    - 如果前几个奇异值占据了大部分能量，说明旋转发生在低维子空间内
    - 这个"旋转子空间"可能是所有消歧操作共享的"处理通道"
    """
    print(f"\n  --- Stage610: 旋转轴子空间分析 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    results = {}

    # Collect delta vectors for each word
    all_deltas = []  # list of delta_d vectors

    word_delta_info = {}

    for s1, s2, word in DISAMB_PAIRS:
        h1s, h2s = [], []
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for h in out.hidden_states:
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    h1s.append(hv)
                else:
                    h2s.append(hv)

        # Compute delta vectors
        deltas = []
        for li in range(1, n_layers):
            d_prev = h1s[li - 1] - h2s[li - 1]
            d_curr = h1s[li] - h2s[li]
            delta = d_curr - d_prev  # how the disambiguation direction changed
            deltas.append(delta)
            all_deltas.append(delta)

        word_delta_info[word] = {
            "n_deltas": len(deltas),
            "avg_delta_norm": round(float(np.mean([torch.norm(d).item() for d in deltas])), 4),
            "max_delta_norm": round(float(np.max([torch.norm(d).item() for d in deltas])), 4),
        }

    # Stack all deltas into a matrix
    delta_matrix = torch.stack(all_deltas, dim=0)  # [n_deltas, hidden_dim]
    n_deltas, hidden_dim = delta_matrix.shape

    # SVD of the delta matrix
    U, S, Vt = torch.linalg.svd(delta_matrix.float(), full_matrices=False)
    # U: [n_deltas, min(n_deltas, hidden_dim)]
    # S: singular values
    # Vt: [min(n_deltas, hidden_dim), hidden_dim]

    # Energy concentration analysis
    total_energy = (S ** 2).sum().item()
    cum_energy = torch.cumsum(S ** 2, dim=0)
    cum_ratio = cum_energy / total_energy

    # Find how many components needed for 50%, 80%, 95% energy
    thresholds = [0.5, 0.8, 0.95, 0.99]
    dims_for_threshold = {}
    for thr in thresholds:
        n_comp = int((cum_ratio < thr).sum().item()) + 1
        dims_for_threshold[f"{int(thr*100)}%"] = min(n_comp, len(S))

    # Top-10 singular values
    top_sv = S[:min(10, len(S))].tolist()
    top_sv_ratio = [(float(s**2) / total_energy) for s in S[:min(10, len(S))]]

    # Check: do all words' deltas live in the same subspace?
    # For each word, compute its "subspace overlap" with the global rotation subspace
    word_subspace_overlap = {}
    idx = 0
    for word in word_delta_info:
        n_d = word_delta_info[word]["n_deltas"]
        word_deltas = all_deltas[idx:idx + n_d]
        idx += n_d

        if not word_deltas:
            continue

        word_delta_matrix = torch.stack(word_deltas, dim=0)
        # Project onto top-k right singular vectors
        for k in [1, 3, 5, 10]:
            if k > len(S):
                continue
            Vk = Vt[:k, :]  # [k, hidden_dim] - top-k rotation directions
            # Project each delta onto this subspace
            projections = word_delta_matrix.float() @ Vk.T  # [n_d, k]
            original_norms = torch.norm(word_delta_matrix.float(), dim=1)  # [n_d]
            proj_norms = torch.norm(projections, dim=1)  # [n_d]
            overlap = float(torch.mean(proj_norms / (original_norms + 1e-10)))
            word_subspace_overlap[f"{word}_top{k}_overlap"] = round(overlap, 4)

    # Also check: for individual words, what's their own subspace dimensionality?
    word_own_dim = {}
    idx = 0
    for word in word_delta_info:
        n_d = word_delta_info[word]["n_deltas"]
        word_deltas = all_deltas[idx:idx + n_d]
        idx += n_d

        if len(word_deltas) < 2:
            continue

        word_delta_matrix = torch.stack(word_deltas, dim=0)
        _, wS, _ = torch.linalg.svd(word_delta_matrix.float(), full_matrices=False)
        w_total = (wS ** 2).sum().item()
        w_cum = torch.cumsum(wS ** 2, dim=0) / w_total
        n95 = int((w_cum < 0.95).sum().item()) + 1
        word_own_dim[word] = {
            "n_components_95pct": min(n95, len(wS)),
            "n_components_total": min(len(wS), hidden_dim),
            "top1_ratio": round(float(wS[0]**2 / w_total), 4) if len(wS) > 0 else 0,
        }

    results = {
        "hidden_dim": hidden_dim,
        "total_n_deltas": n_deltas,
        "n_words": len(DISAMB_PAIRS),
        "singular_values_top10": [round(float(s), 4) for s in top_sv],
        "singular_value_ratios_top10": [round(r, 6) for r in top_sv_ratio],
        "dims_for_energy_threshold": dims_for_threshold,
        "word_delta_stats": word_delta_info,
        "word_subspace_overlap": word_subspace_overlap,
        "word_own_dimensionality": word_own_dim,
    }

    print(f"    hidden_dim={hidden_dim}, n_deltas={n_deltas}")
    print(f"    50% energy in {dims_for_threshold['50%']} dims, "
          f"95% in {dims_for_threshold['95%']} dims, "
          f"99% in {dims_for_threshold['99%']} dims")
    print(f"    Top-3 SV ratios: {top_sv_ratio[:3]}")

    elapsed = time.time() - t0
    print(f"  Stage610 done in {elapsed:.1f}s")
    return {"rotation_subspace_analysis": results}


# ============ Stage611: Gemma4消歧修复实验 ============

def run_stage611(model, tokenizer, model_key):
    """
    对所有模型测试，但主要针对Gemma4：
    选择性零化某些层的MLP，降低旋转速度，验证消歧度是否恢复。

    算法原理：
    - 从stage606数据知道哪些层旋转速度最高
    - 策略1：零化后半部分层的MLP（保留前半部分让消歧形成）
    - 策略2：零化peak层之后的所有MLP
    - 策略3：每隔一层零化MLP（降低旋转频率）
    - 测量每种策略下末层消歧度是否变化
    """
    print(f"\n  --- Stage611: 选择性MLP零化消歧修复 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    results = {}

    for s1, s2, word in DISAMB_PAIRS[:3]:
        print(f"\n    Processing '{word}'...")

        # 1. Normal pass - find peak layer and baseline disambiguation
        h1_n, h2_n = [], []
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for h in out.hidden_states:
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    h1_n.append(hv)
                else:
                    h2_n.append(hv)

        peak_l = 0
        max_d = 0
        for li in range(n_layers):
            d = 1 - cos_sim(h1_n[li], h2_n[li])
            if d > max_d:
                max_d = d
                peak_l = li

        baseline_last_disamb = 1 - cos_sim(h1_n[-1], h2_n[-1])

        # 2. Different MLP zeroing strategies
        strategies = {
            "normal": [],
            "zero_half_post_peak": list(range(peak_l + 1, n_layers)),  # zero MLP after peak
            "zero_all_post_peak": list(range(peak_l, n_layers)),        # zero from peak onwards
            "zero_alternate_post_peak": list(range(peak_l, n_layers, 2)),  # every other after peak
            "zero_last_quarter": list(range(3 * n_layers // 4, n_layers)),  # last 25% layers
            "zero_last_half": list(range(n_layers // 2, n_layers)),  # last 50% layers
        }

        strategy_results = {}

        for strat_name, zero_layers in strategies.items():
            if strat_name == "normal":
                strategy_results[strat_name] = {
                    "last_disamb": round(baseline_last_disamb, 6),
                    "peak_disamb": round(max_d, 4),
                    "peak_layer": peak_l,
                    "n_zeroed_layers": 0,
                }
                continue

            # Zero specified MLP layers using hooks
            hooks = []
            for li in zero_layers:
                if li < n_layers:
                    layer = layers[li]
                    _, mlp_mod = find_mlp_and_attn_components(layer)
                    if mlp_mod is not None:
                        def make_zero_mlp_hook():
                            def hook_fn(module, input, output):
                                if isinstance(output, tuple):
                                    x = output[0]
                                    zeros = torch.zeros_like(x)
                                    return (zeros,) + output[1:]
                                else:
                                    return torch.zeros_like(output)
                            return hook_fn
                        hooks.append(mlp_mod.register_forward_hook(make_zero_mlp_hook()))

            try:
                h1_s, h2_s = [], []
                for s in [s1, s2]:
                    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
                    enc = move_to_device(enc, model)
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                    for h in out.hidden_states:
                        hv = h[0, -1, :].float().cpu()
                        if s == s1:
                            h1_s.append(hv)
                        else:
                            h2_s.append(hv)

                last_disamb = 1 - cos_sim(h1_s[-1], h2_s[-1])
                if peak_l < len(h1_s):
                    peak_disamb = 1 - cos_sim(h1_s[peak_l], h2_s[peak_l])
                else:
                    peak_disamb = 0

                strategy_results[strat_name] = {
                    "last_disamb": round(last_disamb, 6),
                    "peak_disamb": round(peak_disamb, 4),
                    "n_zeroed_layers": len(zero_layers),
                    "zeroed_layer_indices": zero_layers,
                    "improvement_vs_normal": round(last_disamb - baseline_last_disamb, 6),
                }
            finally:
                for h in hooks:
                    h.remove()

        # 3. Find best strategy
        best_strat = "normal"
        best_disamb = baseline_last_disamb
        for sn, sr in strategy_results.items():
            if sr["last_disamb"] > best_disamb:
                best_disamb = sr["last_disamb"]
                best_strat = sn

        results[word] = {
            "baseline_last_disamb": round(baseline_last_disamb, 6),
            "peak_layer": peak_l,
            "peak_disamb": round(max_d, 4),
            "best_strategy": best_strat,
            "best_strategy_disamb": round(best_disamb, 6),
            "strategies": strategy_results,
        }

        print(f"    {word}: baseline_last={baseline_last_disamb:.4f}, "
              f"peak=L{peak_l}({max_d:.4f}), "
              f"best='{best_strat}'({best_disamb:.4f})")

    elapsed = time.time() - t0
    print(f"  Stage611 done in {elapsed:.1f}s")
    return {"gemma4_repair_experiment": results}


# ============ Main ============

MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]


def run_single_model(mk):
    print(f"\n{'='*60}")
    print(f"  Loading {mk}...")
    print(f"{'='*60}")
    t0 = time.time()
    model, tokenizer = load_model_bundle(mk)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    try:
        s609 = run_stage609(model, tokenizer, mk)
        s610 = run_stage610(model, tokenizer, mk)
        s611 = run_stage611(model, tokenizer, mk)
        result = {"stage609": s609, "stage610": s610, "stage611": s611}
    except Exception as e:
        import traceback
        print(f"  ERROR in {mk}: {e}")
        traceback.print_exc()
        result = {"error": str(e)}
    finally:
        free_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(3)
    return result


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target not in MODEL_KEYS:
            print(f"Unknown model: {target}. Use one of: {MODEL_KEYS}")
            return
        models_to_run = [target]
    else:
        models_to_run = MODEL_KEYS

    combined_path = OUTPUT_DIR / f"stage609_610_611_combined_{TIMESTAMP}.json"
    combined = {"timestamp": TIMESTAMP, "models": {}}

    existing_files = sorted(OUTPUT_DIR.glob("stage609_610_611_combined_*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if existing_files and len(sys.argv) == 1:
        try:
            with open(existing_files[0], "r", encoding="utf-8") as f:
                prev = json.load(f)
            combined = prev
            combined_path = existing_files[0]
            print(f"Resuming from {existing_files[0].name}")
        except:
            pass

    for mk in models_to_run:
        if mk in combined["models"] and "error" not in combined["models"][mk]:
            print(f"\n  Skipping {mk} (already completed)")
            continue

        result = run_single_model(mk)
        combined["models"][mk] = result

        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {combined_path}")

    print(f"\nAll done. Results: {combined_path}")


if __name__ == "__main__":
    main()
