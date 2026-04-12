"""
Phase LXXVII-P390/391/392: 从信号传播到语言能力的因果链
=========================================================

阶段B核心任务 - 从"描述性规律"到"因果性理论"的跨越:

P390: cos衰减 → logit变化 → token概率 的完整因果链 ★★★最关键★★★
  - 之前已知: cos(h_int, h_base) 随层指数衰减
  - 之前已知: W_up谱范数决定衰减速度
  - 但不知道: cos衰减如何转化为"输出概率分布的变化"？
  - 方法:
    1. 在每层注入维度方向, 捕获该层cos衰减量
    2. 同时捕获最终层的logit变化量
    3. 建立 cos_drop → Δlogit → Δprob 的完整映射
    4. 关键问题: cos每降0.1, logit变化多少? token概率变化多少?
    5. 验证: 是否存在"临界cos值", 低于此值干预完全失效?

P391: 中间层ICA计算基 — 寻找真正的正交计算基底
  - P387发现3D子空间有旋转+混合, 但真正的计算可能有几十个维度
  - 方法:
    1. 对中间层hidden state做ICA(独立成分分析)
    2. 在多组prompt上收集中间层激活
    3. 找到统计独立的成分数
    4. 检查style/logic/grammar维度在这些独立成分上的投影
  - 目标: 中间层真正有多少个独立计算维度?

P392: 语言的"可操纵维度"完整地图 — 不止style/logic/grammar
  - 之前只测了3个维度, 但语言远不止3维度
  - 新增维度:
    - 情感(sentiment): positive/negative
    - 时态(tense): past/present
    - 语态(voice): active/passive (已有grammar)
    - 数量(number): singular/plural (已有grammar)
    - 礼貌(politeness): formal/informal (已有style)
    - 确定性(certainty): certain/uncertain
    - 复杂度(complexity): simple/complex
  - 方法:
    1. 定义8-10个维度词对
    2. 计算所有维度之间的cos相似度(在W_lm空间)
    3. 对所有维度做层次聚类
    4. 分析维度空间的维度(有效秩)
  - 目标: 语言在W_lm空间到底有多少个独立维度?

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 扩展维度词对
DIM_PAIRS = {
    "style": [("formal", "informal"), ("polite", "rude")],
    "logic": [("true", "false"), ("correct", "wrong")],
    "grammar": [("active", "passive"), ("singular", "plural")],
    "sentiment": [("happy", "sad"), ("good", "bad")],
    "tense": [("was", "is"), ("went", "goes")],
    "certainty": [("definitely", "maybe"), ("always", "sometimes")],
    "complexity": [("simple", "complicated"), ("easy", "difficult")],
    "quantity": [("many", "few"), ("all", "none")],
}

PROMPTS = [
    "The apple is",
    "The weather today is",
    "She said that",
    "In the future, we will",
]

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True,
        "use_fast": False,
    },
}


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"Loading {model_name}...")
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu",
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = mdl.device
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, use_fast=cfg["use_fast"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok, device


def get_w_lm_normed(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm_normed(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm_normed(model, tokenizer, word_neg)
    diff = w_pos - w_neg
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        return w_pos, 0.0
    return diff / norm, norm


def get_layers(model, max_layers=None):
    if hasattr(model.model, "layers"):
        all_layers = list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        all_layers = list(model.model.encoder.layers)
    else:
        raise ValueError("Cannot find layers")
    if max_layers is None:
        return all_layers
    return all_layers[:max_layers]


def get_mlp(model, layer):
    if hasattr(layer, "mlp"):
        return layer.mlp
    elif hasattr(layer, "feed_forward"):
        return layer.feed_forward
    raise ValueError("Cannot find MLP")


# ========== P390: cos衰减 → logit变化 → token概率 的完整因果链 ★★★ ==========

def run_p390(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P390: cos->logit->prob causal chain - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, min(n_layers_total, 36), 2))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    dim_directions = {}
    for name in test_dims:
        if name in DIM_PAIRS:
            pos, neg = DIM_PAIRS[name][0]
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            dim_directions[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线输出
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
        probs_base = F.softmax(logits_base, dim=-1)

    # ===== Part 1: 在L0注入, 跟踪各层cos衰减 + 最终logit变化 =====
    print(f"\n  === Part 1: L0 injection - track cos decay across layers ===")
    l0_decay = {}

    for dim_name in dim_directions:
        direction = dim_directions[dim_name]["direction"]
        pos_word = dim_directions[dim_name]["pos"]
        neg_word = dim_directions[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        # 在L0注入 (修改inputs_embeds)
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        # 捕获注入后各层hidden state
        captured_int = {}
        def make_hook(storage, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage[key] = output[0].detach().float()
                else:
                    storage[key] = output.detach().float()
            return hook

        handles = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
            probs_int = F.softmax(logits_int, dim=-1)
        for h in handles:
            h.remove()

        # 同时捕获基线hidden states (也需要hook)
        captured_base_l0 = {}
        handles_base = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_base.append(layer.register_forward_hook(make_hook(captured_base_l0, f"L{i}")))
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
        for h in handles_base:
            h.remove()

        # 计算每层cos衰减
        cos_per_layer = {}
        delta_h_norm_per_layer = {}
        for l in scan_layers:
            key = f"L{l}"
            if key in captured_int and key in captured_base_l0:
                h_int = captured_int[key][0, -1, :].cpu().numpy()
                h_base = captured_base_l0[key][0, -1, :].cpu().numpy()
                norm_int = np.linalg.norm(h_int)
                norm_base = np.linalg.norm(h_base)
                if norm_int > 1e-8 and norm_base > 1e-8:
                    cos_val = float(np.dot(h_int, h_base) / (norm_int * norm_base))
                else:
                    cos_val = 0.0
                cos_per_layer[l] = cos_val
                delta_h_norm_per_layer[l] = float(np.linalg.norm(h_int - h_base))

        # 最终logit变化
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        delta_logit = delta_pos - delta_neg

        prob_pos_base = float(probs_base[pos_id].cpu())
        prob_pos_int = float(probs_int[pos_id].cpu())
        prob_neg_base = float(probs_base[neg_id].cpu())
        prob_neg_int = float(probs_int[neg_id].cpu())
        delta_prob = (prob_pos_int - prob_pos_base) - (prob_neg_int - prob_neg_base)

        eps = 1e-10
        kl_div = float(torch.sum(probs_int * torch.log((probs_int + eps) / (probs_base + eps))))

        l0_decay[dim_name] = {
            "cos_per_layer": cos_per_layer,
            "delta_h_norm_per_layer": delta_h_norm_per_layer,
            "final_delta_logit": delta_logit,
            "final_delta_prob": delta_prob,
            "final_kl_divergence": kl_div,
        }

        cos_strs = [f"L{l}:{cos_per_layer.get(l, 'N/A'):.4f}" for l in scan_layers[:5]]
        print(f"  {dim_name}: cos_decay=[{', '.join(cos_strs)}...], "
              f"dlogit={delta_logit:.3f}, dprob={delta_prob:.4f}, KL={kl_div:.4f}")

    # ===== Part 2: 在不同层注入, 测量干预效率 =====
    print(f"\n  === Part 2: Per-layer injection - intervention efficiency ===")
    per_layer_intervention = {}

    for dim_name in dim_directions:
        direction = dim_directions[dim_name]["direction"]
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        pos_word = dim_directions[dim_name]["pos"]
        neg_word = dim_directions[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        print(f"\n  --- Dimension: {dim_name} ---")
        layer_chain = []

        for inj_layer in scan_layers:
            # 在inj_layer注入
            def inject_hook(module, input, output, il=inj_layer, wt=w_tensor):
                if isinstance(output, tuple):
                    h = output[0].detach().float()
                else:
                    h = output.detach().float()
                h_modified = h.clone()
                h_modified[0, -1, :] += wt.to(h.dtype)
                if isinstance(output, tuple):
                    return (h_modified.to(output[0].dtype),) + output[1:]
                return h_modified.to(output.dtype)

            handle_inj = layers[inj_layer].register_forward_hook(inject_hook)

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
                probs_int = F.softmax(logits_int, dim=-1)

            handle_inj.remove()

            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            delta_logit = delta_pos - delta_neg

            prob_pos_base = float(probs_base[pos_id].cpu())
            prob_pos_int = float(probs_int[pos_id].cpu())
            prob_neg_base = float(probs_base[neg_id].cpu())
            prob_neg_int = float(probs_int[neg_id].cpu())
            delta_prob = (prob_pos_int - prob_pos_base) - (prob_neg_int - prob_neg_base)

            eps = 1e-10
            kl_div = float(torch.sum(probs_int * torch.log((probs_int + eps) / (probs_base + eps))))

            top1_base = int(logits_base.argmax().cpu())
            top1_int = int(logits_int.argmax().cpu())
            top1_changed = top1_base != top1_int

            layer_chain.append({
                "injection_layer": inj_layer,
                "delta_logit": delta_logit,
                "delta_prob": delta_prob,
                "kl_divergence": kl_div,
                "top1_changed": top1_changed,
            })

            if inj_layer % 10 == 0 or inj_layer == scan_layers[-1]:
                print(f"    L{inj_layer}: dlogit={delta_logit:.3f}, dprob={delta_prob:.4f}, "
                      f"KL={kl_div:.4f}, top1_change={top1_changed}")

        per_layer_intervention[dim_name] = layer_chain

    # ===== Part 3: 核心分析 - cos衰减与logit变化的因果链 =====
    print(f"\n  === Part 3: Causal chain regression ===")
    regression_results = {}

    for dim_name in dim_directions:
        # 从Part1: L0注入后的cos衰减 + 最终logit变化
        decay = l0_decay[dim_name]
        cos_arr = np.array([decay["cos_per_layer"].get(l, 0) for l in scan_layers])
        dh_norm_arr = np.array([decay["delta_h_norm_per_layer"].get(l, 0) for l in scan_layers])

        # 从Part2: 各层注入效率
        intervention = per_layer_intervention[dim_name]
        inj_layers = np.array([c["injection_layer"] for c in intervention])
        dlogit_arr = np.array([c["delta_logit"] for c in intervention])
        dprob_arr = np.array([c["delta_prob"] for c in intervention])
        kl_arr = np.array([c["kl_divergence"] for c in intervention])

        # 回归1: 注入层深度 vs delta_logit
        try:
            slope_layer_dlogit = np.dot(inj_layers - inj_layers.mean(), dlogit_arr - dlogit_arr.mean()) / max(np.sum((inj_layers - inj_layers.mean())**2), 1e-10)
            intercept = dlogit_arr.mean() - slope_layer_dlogit * inj_layers.mean()
            r2_layer_dlogit = 1 - np.sum((dlogit_arr - slope_layer_dlogit * inj_layers - intercept)**2) / max(np.sum((dlogit_arr - dlogit_arr.mean())**2), 1e-10)
        except:
            slope_layer_dlogit = 0
            r2_layer_dlogit = -999

        # 回归2: cos衰减 vs delta_h_norm
        try:
            # cos越低 = 信号偏离越大
            cos_drop = 1 - cos_arr
            corr_cos_drop_dh = np.corrcoef(cos_drop, dh_norm_arr)[0, 1]
        except:
            corr_cos_drop_dh = 0

        # 临界层深度: 注入效果 |delta_logit| < 1.0 的最深有效层
        effective = np.abs(dlogit_arr) > 1.0
        if np.any(effective):
            max_effective_layer = int(inj_layers[effective].max())
            min_effective_layer = int(inj_layers[effective].min())
        else:
            max_effective_layer = -1
            min_effective_layer = -1

        # "干预效率曲线"特征: 前期下降, 中期低谷, 后期回升?
        # 用二次拟合检测U型
        try:
            coeffs = np.polyfit(inj_layers, dlogit_arr, 2)
            is_u_shape = coeffs[0] > 0  # 二次项>0 = U型
            u_vertex = -coeffs[1] / (2 * coeffs[0]) if abs(coeffs[0]) > 1e-10 else -1
        except:
            is_u_shape = False
            u_vertex = -1
            coeffs = [0, 0, 0]

        regression_results[dim_name] = {
            "l0_cos_decay": {str(k): float(v) for k, v in decay["cos_per_layer"].items()},
            "l0_delta_h_norm": {str(k): float(v) for k, v in decay["delta_h_norm_per_layer"].items()},
            "l0_final_dlogit": float(decay["final_delta_logit"]),
            "l0_final_dprob": float(decay["final_delta_prob"]),
            "l0_final_kl": float(decay["final_kl_divergence"]),
            "layer_vs_dlogit_r2": float(r2_layer_dlogit),
            "layer_vs_dlogit_slope": float(slope_layer_dlogit),
            "max_effective_injection_layer": max_effective_layer,
            "min_effective_injection_layer": min_effective_layer,
            "is_u_shape_intervention": bool(is_u_shape),
            "u_vertex_layer": float(u_vertex),
            "quadratic_coeffs": [float(c) for c in coeffs],
        }

        print(f"  {dim_name}: L0_cos_decay={[f'{v:.3f}' for v in list(decay['cos_per_layer'].values())[:5]]}, "
              f"L0_dlogit={decay['final_delta_logit']:.3f}, "
              f"layer_vs_dlogit_R2={r2_layer_dlogit:.3f}, "
              f"U_shape={is_u_shape}(vertex@L{u_vertex:.1f}), "
              f"effective_range=[L{min_effective_layer}-L{max_effective_layer}]")

    results = {
        "l0_decay": l0_decay,
        "per_layer_intervention": per_layer_intervention,
        "regression_results": regression_results,
        "scan_layers": scan_layers,
        "beta": beta,
        "test_dims": list(dim_directions.keys()),
    }
    return results


# ========== P391: 中间层ICA计算基 ==========

def run_p391(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P391: ICA computation basis at middle layers - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    n_layers_total = len(get_layers(model))
    hidden_dim = embed.weight.shape[1]

    # 选中间层做分析
    mid_layer = n_layers_total // 2
    last_layer = n_layers_total - 1
    target_layers = [0, mid_layer // 2, mid_layer, mid_layer + mid_layer // 2, last_layer]
    target_layers = sorted(set(l for l in target_layers if l < n_layers_total))
    n_scan = max(target_layers) + 1
    layers = get_layers(model, n_scan)

    # 维度方向
    dim_directions = {}
    for name, pairs in DIM_PAIRS.items():
        pos, neg = pairs[0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}

    # 在多组prompt上收集激活
    print(f"\n  Collecting activations on {len(PROMPTS)} prompts...")

    all_activations = {l: [] for l in target_layers}

    for prompt in PROMPTS:
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        captured = {}

        def make_hook(storage, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage[key] = output[0].detach().float()
                else:
                    storage[key] = output.detach().float()
            return hook

        handles = []
        for i, layer in enumerate(layers):
            if i in target_layers:
                handles.append(layer.register_forward_hook(make_hook(captured, f"L{i}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        for h in handles:
            h.remove()

        for l in target_layers:
            key = f"L{l}"
            if key in captured:
                h = captured[key][0, -1, :].cpu().numpy()  # 最后token
                all_activations[l].append(h)

        # 也在注入后的激活上收集
        for dim_name in ["style", "logic", "grammar", "sentiment", "tense"]:
            if dim_name not in dim_directions:
                continue
            direction = dim_directions[dim_name]["direction"]
            w_tensor = torch.tensor(direction * 5.0, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

            captured_int = {}
            handles_int = []
            for i, layer in enumerate(layers):
                if i in target_layers:
                    handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
            for h in handles_int:
                h.remove()

            for l in target_layers:
                key = f"L{l}"
                if key in captured_int:
                    h = captured_int[key][0, -1, :].cpu().numpy()
                    all_activations[l].append(h)

    # ===== ICA分析 =====
    print(f"\n  === ICA Analysis ===")
    ica_results = {}

    for l in target_layers:
        activations = np.array(all_activations[l])  # [n_samples, d_model]
        n_samples = activations.shape[0]

        if n_samples < 10:
            print(f"  L{l}: insufficient samples ({n_samples})")
            continue

        # PCA降维到合理维度(避免ICA在高维空间不稳定)
        # 先用PCA找到覆盖95%方差的子空间
        from sklearn.decomposition import PCA, FastICA

        # Center
        mean_act = activations.mean(axis=0)
        act_centered = activations - mean_act

        # PCA: 保留95%方差
        n_pca = min(50, n_samples - 1, act_centered.shape[1])
        pca = PCA(n_components=n_pca)
        act_pca = pca.fit_transform(act_centered)

        # 保留95%方差的维度数
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n_95 = max(n_95, 3)  # 至少3维

        # 再用PCA降到n_95维
        pca_95 = PCA(n_components=n_95)
        act_pca95 = pca_95.fit_transform(act_centered)

        # ICA在PCA子空间上
        # 尝试不同成分数, 选最佳
        best_n_ica = 3
        best_kurtosis = -999

        for n_ica in [3, 5, 8, 10, min(15, n_95)]:
            try:
                ica = FastICA(n_components=n_ica, max_iter=500, random_state=42, tol=0.01)
                S = ica.fit_transform(act_pca95)  # [n_samples, n_ica]

                # 评估: 独立成分的平均超额峰度
                kurtosis_vals = []
                for j in range(n_ica):
                    s = S[:, j]
                    kurt = float(np.mean((s - s.mean())**4) / max(np.mean((s - s.mean())**2)**2, 1e-10) - 3.0)
                    kurtosis_vals.append(kurt)

                avg_kurtosis = np.mean(np.abs(kurtosis_vals))

                if avg_kurtosis > best_kurtosis:
                    best_kurtosis = avg_kurtosis
                    best_n_ica = n_ica
            except Exception as e:
                continue

        # 用最佳成分数做ICA
        try:
            ica = FastICA(n_components=best_n_ica, max_iter=500, random_state=42, tol=0.01)
            S = ica.fit_transform(act_pca95)
            A = ica.mixing_  # [n_95, n_ica]

            # 映射回原始空间: mixing_matrix = pca_components^T @ A
            mixing_full = pca_95.components_.T @ A  # [d_model, n_ica]

            # 计算每个维度方向在ICA成分上的投影
            dim_projections = {}
            for dim_name, dim_info in dim_directions.items():
                direction = dim_info["direction"]
                proj = mixing_full.T @ direction  # [n_ica]
                dim_projections[dim_name] = {
                    "projections": proj.tolist(),
                    "max_component": int(np.argmax(np.abs(proj))),
                    "max_projection": float(np.abs(proj).max()),
                    "norm": float(np.linalg.norm(proj)),
                }

            # ICA成分之间的统计独立性(互信息近似)
            from sklearn.metrics import mutual_info_score
            mi_matrix = np.zeros((best_n_ica, best_n_ica))
            for i in range(best_n_ica):
                for j in range(best_n_ica):
                    if i != j:
                        # 离散化
                        s_i = np.digitize(S[:, i], bins=np.linspace(S[:, i].min(), S[:, i].max(), 10))
                        s_j = np.digitize(S[:, j], bins=np.linspace(S[:, j].min(), S[:, j].max(), 10))
                        mi_matrix[i, j] = mutual_info_score(s_i, s_j)

            ica_results[f"L{l}"] = {
                "n_pca_95": n_95,
                "n_ica_components": best_n_ica,
                "best_kurtosis": float(best_kurtosis),
                "dim_projections": dim_projections,
                "mi_matrix_diag_zero_mean": float(mi_matrix[mi_matrix > 0].mean()) if np.any(mi_matrix > 0) else 0,
                "pca_explained_variance_top5": pca.explained_variance_ratio_[:5].tolist(),
            }

            print(f"  L{l}: PCA95={n_95}维, ICA={best_n_ica}成分, "
                  f"avg_kurtosis={best_kurtosis:.2f}, "
                  f"dim projections: " + ", ".join(
                      f"{n}→comp{dim_projections[n]['max_component']}({dim_projections[n]['max_projection']:.3f})"
                      for n in ["style", "logic", "grammar"] if n in dim_projections
                  ))
        except Exception as e:
            ica_results[f"L{l}"] = {"error": str(e)}
            print(f"  L{l}: ICA failed: {e}")

    results = {
        "ica_results": ica_results,
        "target_layers": target_layers,
        "dim_names": list(DIM_PAIRS.keys()),
    }
    return results


# ========== P392: 语言的"可操纵维度"完整地图 ==========

def run_p392(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P392: Complete manipulable dimension map - {model_name}")
    print(f"{'='*60}")

    # 所有维度
    all_dim_names = list(DIM_PAIRS.keys())
    all_dim_directions = {}

    for name in all_dim_names:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        all_dim_directions[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
        }
        print(f"  {name}: |W_lm diff|={norm:.4f} ({pos}/{neg})")

    # ===== 1. W_lm空间的维度间cos矩阵 =====
    print(f"\n  === Cosine Similarity Matrix in W_lm Space ===")
    cos_matrix = np.zeros((len(all_dim_names), len(all_dim_names)))

    for i, n1 in enumerate(all_dim_names):
        for j, n2 in enumerate(all_dim_names):
            cos_val = float(np.dot(all_dim_directions[n1]["direction"], all_dim_directions[n2]["direction"]))
            cos_matrix[i, j] = cos_val

    # 打印上三角
    print(f"  Dimensions: {all_dim_names}")
    for i, n1 in enumerate(all_dim_names):
        for j, n2 in enumerate(all_dim_names):
            if i < j:
                print(f"    cos({n1}, {n2}) = {cos_matrix[i,j]:.4f}")

    # ===== 2. SVD分析: 有效维度 =====
    print(f"\n  === SVD: Effective Dimensionality ===")
    D = np.stack([all_dim_directions[n]["direction"] for n in all_dim_names])  # [n_dims, d_model]
    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    total_energy = np.sum(S**2)
    energy_ratios = S**2 / total_energy
    cum_energy = np.cumsum(energy_ratios)

    n_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    n_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
    n_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
    eff_rank = np.sum(energy_ratios > 0.01)

    print(f"  Singular values: {[f'{s:.4f}' for s in S]}")
    print(f"  Energy ratios: {[f'{e:.4f}' for e in energy_ratios]}")
    print(f"  Cumulative energy: {[f'{c:.4f}' for c in cum_energy]}")
    print(f"  Effective rank (>1%): {eff_rank}")
    print(f"  90% variance: {n_90} components")
    print(f"  95% variance: {n_95} components")
    print(f"  99% variance: {n_99} components")

    # ===== 3. 层次聚类 =====
    print(f"\n  === Hierarchical Clustering ===")
    # 用1-|cos|作为距离
    dist_matrix = 1 - np.abs(cos_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # 对称化

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # 转换为压缩距离矩阵
        dist_condensed = squareform(dist_matrix)
        Z = linkage(dist_condensed, method='ward')

        # 切分为2-4类
        cluster_results = {}
        for n_clusters in [2, 3, 4]:
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            clusters = {}
            for i, label in enumerate(labels):
                cluster_name = f"cluster_{label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(all_dim_names[i])
            cluster_results[f"n_{n_clusters}"] = clusters
            print(f"  {n_clusters} clusters: {clusters}")

    except ImportError:
        cluster_results = {"error": "scipy not available for clustering"}
        print(f"  scipy not available, skipping clustering")

    # ===== 4. 多prompt维度稳定性 =====
    print(f"\n  === Multi-prompt Dimension Stability ===")
    embed = model.get_input_embeddings()

    # 在不同prompt下, 维度方向是否稳定?
    # 用注入+捕获中间层的方式验证
    mid_layer = len(get_layers(model)) // 2
    n_layers_total = len(get_layers(model))
    scan_layers_mid = [0, mid_layer, n_layers_total - 1]
    n_scan = max(scan_layers_mid) + 1
    layers = get_layers(model, n_scan)

    dim_stability = {}
    core_dims = ["style", "logic", "grammar", "sentiment", "tense"]

    for dim_name in core_dims:
        if dim_name not in all_dim_directions:
            continue
        direction = all_dim_directions[dim_name]["direction"]
        pos_word = all_dim_directions[dim_name]["pos"]
        neg_word = all_dim_directions[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        prompt_effects = []

        for prompt in PROMPTS:
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # 基线logits
            with torch.no_grad():
                logits_base = model(inputs_embeds=inputs_embeds, position_ids=position_ids).logits[0, -1, :].float()

            # 注入
            w_tensor = torch.tensor(direction * 8.0, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()

            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            effect = delta_pos - delta_neg
            prompt_effects.append(effect)

        # 效果稳定性: 变异系数
        mean_effect = np.mean(prompt_effects)
        std_effect = np.std(prompt_effects)
        cv = std_effect / max(abs(mean_effect), 1e-10)

        # 符号一致性: 效果方向是否跨prompt一致
        sign_consistency = sum(1 for e in prompt_effects if np.sign(e) == np.sign(mean_effect)) / len(prompt_effects)

        dim_stability[dim_name] = {
            "effects_per_prompt": prompt_effects,
            "mean_effect": float(mean_effect),
            "std_effect": float(std_effect),
            "cv": float(cv),
            "sign_consistency": float(sign_consistency),
        }

        print(f"  {dim_name}: mean_effect={mean_effect:.3f}, std={std_effect:.3f}, "
              f"CV={cv:.3f}, sign_consistency={sign_consistency:.2f}")

    # ===== 5. 维度间的"竞争"检测 =====
    print(f"\n  === Dimension Competition ===")
    competition_results = {}

    # 两两注入, 检测是否互相干扰
    for i, dim1 in enumerate(core_dims[:4]):
        for j, dim2 in enumerate(core_dims[:4]):
            if i >= j:
                continue

            d1 = all_dim_directions[dim1]["direction"]
            d2 = all_dim_directions[dim2]["direction"]
            p1_id = tokenizer.encode(all_dim_directions[dim1]["pos"], add_special_tokens=False)[0]
            n1_id = tokenizer.encode(all_dim_directions[dim1]["neg"], add_special_tokens=False)[0]
            p2_id = tokenizer.encode(all_dim_directions[dim2]["pos"], add_special_tokens=False)[0]
            n2_id = tokenizer.encode(all_dim_directions[dim2]["neg"], add_special_tokens=False)[0]

            prompt = PROMPTS[0]
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # 基线
            with torch.no_grad():
                logits_base = model(inputs_embeds=inputs_embeds, position_ids=position_ids).logits[0, -1, :].float()

            # 单独注入dim1
            w1 = torch.tensor(d1 * 8.0, dtype=torch.float32, device=device)
            ie1 = inputs_embeds.clone()
            ie1[0, -1, :] += w1.to(model.dtype)
            with torch.no_grad():
                l1 = model(inputs_embeds=ie1, position_ids=position_ids).logits[0, -1, :].float()

            # 单独注入dim2
            w2 = torch.tensor(d2 * 8.0, dtype=torch.float32, device=device)
            ie2 = inputs_embeds.clone()
            ie2[0, -1, :] += w2.to(model.dtype)
            with torch.no_grad():
                l2 = model(inputs_embeds=ie2, position_ids=position_ids).logits[0, -1, :].float()

            # 联合注入
            w12 = torch.tensor((d1 + d2) * 8.0, dtype=torch.float32, device=device)
            ie12 = inputs_embeds.clone()
            ie12[0, -1, :] += w12.to(model.dtype)
            with torch.no_grad():
                l12 = model(inputs_embeds=ie12, position_ids=position_ids).logits[0, -1, :].float()

            # 各维度效果
            eff1_single = float(l1[p1_id] - l1[n1_id] - logits_base[p1_id] + logits_base[n1_id])
            eff2_single = float(l2[p2_id] - l2[n2_id] - logits_base[p2_id] + logits_base[n2_id])
            eff1_joint = float(l12[p1_id] - l12[n1_id] - logits_base[p1_id] + logits_base[n1_id])
            eff2_joint = float(l12[p2_id] - l12[n2_id] - logits_base[p2_id] + logits_base[n2_id])

            # 干扰 = 联合效果 - 单独效果
            interference_1 = eff1_joint - eff1_single
            interference_2 = eff2_joint - eff2_single

            key = f"{dim1}_vs_{dim2}"
            competition_results[key] = {
                "eff1_single": eff1_single,
                "eff2_single": eff2_single,
                "eff1_joint": eff1_joint,
                "eff2_joint": eff2_joint,
                "interference_on_dim1": interference_1,
                "interference_on_dim2": interference_2,
                "cos_between_dims": float(np.dot(d1, d2)),
            }

            print(f"  {dim1} vs {dim2}: cos={np.dot(d1,d2):.4f}, "
                  f"interf_on_{dim1}={interference_1:.3f}, interf_on_{dim2}={interference_2:.3f}")

    results = {
        "cos_matrix": cos_matrix.tolist(),
        "dim_names": all_dim_names,
        "svd_results": {
            "singular_values": S.tolist(),
            "energy_ratios": energy_ratios.tolist(),
            "cumulative_energy": cum_energy.tolist(),
            "effective_rank": int(eff_rank),
            "n_90": n_90,
            "n_95": n_95,
            "n_99": n_99,
        },
        "cluster_results": cluster_results,
        "dim_stability": dim_stability,
        "competition_results": competition_results,
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Phase LXXVII: Causal chain from signal propagation to language ability")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p390", "p391", "p392", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    results = {}

    timestamp = time.strftime("%Y%m%d_%H%M")

    try:
        if args.exp in ["p390", "all"]:
            r = run_p390(model, tokenizer, device, args.model)
            results["p390"] = r
    except Exception as e:
        print(f"P390 failed: {e}")
        traceback.print_exc()
        results["p390_error"] = str(e)

    try:
        if args.exp in ["p391", "all"]:
            r = run_p391(model, tokenizer, device, args.model)
            results["p391"] = r
    except Exception as e:
        print(f"P391 failed: {e}")
        traceback.print_exc()
        results["p391_error"] = str(e)

    try:
        if args.exp in ["p392", "all"]:
            r = run_p392(model, tokenizer, device, args.model)
            results["p392"] = r
    except Exception as e:
        print(f"P392 failed: {e}")
        traceback.print_exc()
        results["p392_error"] = str(e)

    # 保存结果
    out_file = OUT_DIR / f"phase_lxxvii_p390_392_{args.model}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # 释放GPU
    del model
    torch.cuda.empty_cache()
    print(f"GPU memory released for {args.model}")


if __name__ == "__main__":
    main()
