"""
Phase LXXXII-P407/408/409/410: V_lang vs V_structure + W_lm Rank解释 + 暗能量机制
================================================================================

阶段E核心任务 - 从25维结构空间到7维功能空间的精确区分:

P407: V_lang vs V_structure区分
  - P405发现: 25个词对方向两两正交(结构正交)
  - 但P396-P402只验证了7个维度(style/logic/grammar/sentiment/tense/certainty/quantity)能通过干预影响输出
  - 核心问题: 剩余18个正交维度是否也能通过干预影响输出?
  - 方法: 对所有25维方向做L0注入, 测量Δlogit
  - 预期: 只有部分维度能产生显著Δlogit→"功能维度" vs "结构维度"

P408: W_lm Rank=257的解释
  - P403发现: 三模型W_lm Rank=257
  - 假设1: Rank=257与attention头数有关(num_heads * head_dim?)
  - 假设2: Rank=257与MLP中间维度有关(intermediate_size/某因子?)
  - 假设3: Rank=257是TruncatedSVD的截断效应(实际Rank可能更高)
  - 方法: 分析模型架构参数 + 更精确的SVD(更多分量)

P409: DS7B暗能量机制——逐层dh_norm分析
  - P406发现: DS7B的dh_norm=2177-2650, 远大于GLM4(213-249)
  - 核心问题: dh_norm在哪些层暴增?
  - 方法: 逐层捕获dh, 绘制dh_norm vs layer曲线
  - 关键: 找到dh_norm暴增的"拐点层"

P410: 信号分配效率的精确公式
  - P404发现: eff_L1理论下界与实际差异大
  - 目标: 从W_lm SVD推导eff_L1的精确公式
  - 关键量: ||W_diff||^2, ||W_lm||_F^2, dh_norm, cos(dh, W_diff)
  - 公式推导: eff_L1 = |cos(dh, w_diff)| * ||w_diff|| / (vocab_size * mean(||w_i||))

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
    "quantity": [("many", "few")],
}

EXTRA_DIM_PAIRS = {
    "complexity": [("complex", "simple")],
    "formality": [("professional", "casual")],
    "politeness": [("please", "now")],
    "specificity": [("specifically", "generally")],
    "speed": [("quickly", "slowly")],
    "size": [("large", "small")],
    "age": [("old", "new")],
    "distance": [("near", "far")],
    "brightness": [("bright", "dark")],
    "temperature": [("hot", "cold")],
    "weight": [("heavy", "light")],
    "sound": [("loud", "quiet")],
    "frequency": [("often", "rarely")],
    "importance": [("important", "trivial")],
    "position": [("above", "below")],
    "direction": [("forward", "backward")],
    "completeness": [("complete", "partial")],
    "value": [("valuable", "worthless")],
}

ALL_DIMS = {**DIM_PAIRS, **EXTRA_DIM_PAIRS}

PROMPTS = ["The apple is", "In the future, people will", "The scientist explained that", "When the rain stopped,"]

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True, "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True, "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True, "use_fast": False,
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


def get_w_lm(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm(model, tokenizer, word_neg)
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


# ========== P407: V_lang vs V_structure区分 ==========

def run_p407(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P407: V_lang vs V_structure distinction - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    layers = get_layers(model)
    beta = 8.0

    # 对所有25维方向做L0注入, 测量Δlogit
    results_per_dim = {}

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    for dim_name, pairs in ALL_DIMS.items():
        pos, neg = pairs[0]
        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm < 1e-8:
                print(f"  {dim_name}: SKIP (norm too small)")
                continue

            pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
            neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]

            # L0注入
            w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds_base.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()

            # Δlogit
            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            dlogit = delta_pos - delta_neg

            # 概率变化
            probs_base = torch.softmax(logits_base, dim=-1)
            probs_int = torch.softmax(logits_int, dim=-1)
            prob_pos_base = float(probs_base[pos_id].cpu())
            prob_pos_int = float(probs_int[pos_id].cpu())
            prob_neg_base = float(probs_base[neg_id].cpu())
            prob_neg_int = float(probs_int[neg_id].cpu())

            # Top-5 token变化
            delta_logits_all = (logits_int - logits_base).cpu().numpy()
            top5_idx = np.argsort(np.abs(delta_logits_all))[::-1][:5]
            top5_tokens = []
            for idx in top5_idx:
                tok_str = tokenizer.decode([int(idx)])[:15]
                top5_tokens.append({
                    "token": tok_str,
                    "delta_logit": float(delta_logits_all[idx]),
                })

            # 判断是否为功能维度
            is_functional = abs(dlogit) > 1.0  # Δlogit > 1.0 认为功能显著
            is_structural = abs(dlogit) < 0.5  # Δlogit < 0.5 认为仅结构正交

            results_per_dim[dim_name] = {
                "pos": pos, "neg": neg,
                "dlogit": dlogit,
                "delta_pos": delta_pos, "delta_neg": delta_neg,
                "prob_pos_base": prob_pos_base, "prob_pos_int": prob_pos_int,
                "prob_neg_base": prob_neg_base, "prob_neg_int": prob_neg_int,
                "is_functional": is_functional,
                "is_structural_only": is_structural,
                "top5_tokens": top5_tokens,
            }

            func_flag = " *** FUNCTIONAL ***" if is_functional else (" structural" if is_structural else " marginal")
            print(f"  {dim_name} ({pos}/{neg}): dlogit={dlogit:.3f}{func_flag}")

        except Exception as e:
            print(f"  {dim_name}: ERROR - {e}")

    # 统计
    n_functional = sum(1 for r in results_per_dim.values() if r["is_functional"])
    n_structural = sum(1 for r in results_per_dim.values() if r["is_structural_only"])
    n_marginal = len(results_per_dim) - n_functional - n_structural

    print(f"\n  === Summary ===")
    print(f"  Functional (|dlogit|>1.0): {n_functional}")
    print(f"  Structural (|dlogit|<0.5): {n_structural}")
    print(f"  Marginal (0.5<=|dlogit|<=1.0): {n_marginal}")

    # 按dlogit排序
    sorted_dims = sorted(results_per_dim.items(), key=lambda x: abs(x[1]["dlogit"]), reverse=True)
    print(f"\n  Ranked by |dlogit|:")
    for name, r in sorted_dims:
        func = "F" if r["is_functional"] else ("S" if r["is_structural_only"] else "M")
        print(f"    {func} {name}: |dlogit|={abs(r['dlogit']):.3f}")

    results = {
        "results_per_dim": results_per_dim,
        "n_functional": n_functional,
        "n_structural": n_structural,
        "n_marginal": n_marginal,
    }
    return results


# ========== P408: W_lm Rank=257的解释 ==========

def run_p408(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P408: W_lm Rank=257 explanation - {model_name}")
    print(f"{'='*60}")

    # 收集模型架构参数
    config = model.config
    arch_info = {}

    # 常见参数名
    param_names = [
        "hidden_size", "num_attention_heads", "num_hidden_layers",
        "intermediate_size", "num_key_value_heads", "head_dim",
        "vocab_size", "max_position_embeddings",
    ]
    for name in param_names:
        if hasattr(config, name):
            arch_info[name] = getattr(config, name)

    print(f"  Model architecture:")
    for k, v in arch_info.items():
        print(f"    {k}: {v}")

    # 计算关键比值
    hidden_size = arch_info.get("hidden_size", 0)
    num_heads = arch_info.get("num_attention_heads", 0)
    intermediate_size = arch_info.get("intermediate_size", 0)
    num_kv_heads = arch_info.get("num_key_value_heads", num_heads)

    if hidden_size > 0 and num_heads > 0:
        head_dim = hidden_size // num_heads
        print(f"\n  Computed head_dim = {hidden_size}/{num_heads} = {head_dim}")

    # W_lm SVD with more components
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape

    from sklearn.decomposition import TruncatedSVD

    # 用更多分量验证Rank
    for n_comp in [256, 512, min(hidden_dim, 768)]:
        if n_comp > hidden_dim:
            continue
        svd = TruncatedSVD(n_components=n_comp)
        svd.fit(W_lm)
        cumvar = np.cumsum(svd.explained_variance_ratio_)
        rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
        rank_999 = int(np.searchsorted(cumvar, 0.999) + 1)
        print(f"\n  TruncatedSVD(n={n_comp}): rank_95={rank_95}, rank_99={rank_99}, rank_999={rank_999}")

    # 检查W_lm是否与embedding共享权重
    embed_weight = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    # 用子集检查避免内存溢出
    is_tied = bool(np.allclose(W_lm[:100], embed_weight[:100], atol=1e-3))
    print(f"\n  W_lm tied with embedding: {is_tied}")

    if not is_tied:
        # 计算W_lm与embedding的关系
        # W_lm ≈ A @ embed_weight + B?
        from sklearn.decomposition import PCA
        # embed_weight的SVD
        svd_embed = TruncatedSVD(n_components=256)
        svd_embed.fit(embed_weight)
        embed_rank_95 = int(np.searchsorted(np.cumsum(svd_embed.explained_variance_ratio_), 0.95) + 1)
        print(f"  Embedding rank_95: {embed_rank_95}")

        # W_lm在embedding空间中的投影
        # W_lm @ embed_weight^T 的SVD → W_lm与embedding的共享结构
        cross = W_lm @ embed_weight.T  # [vocab_size, vocab_size] - 太大
        # 改用: embed_weight @ W_lm^T 的前几个分量
        # 或者: W_lm在embed_weight右奇异向量上的投影
        V_embed = svd_embed.components_  # [256, hidden_dim]
        proj_W_on_embed = W_lm @ V_embed.T  # [vocab_size, 256]
        proj_norm = np.sum(proj_W_on_embed ** 2, axis=1)
        total_norm = np.sum(W_lm ** 2, axis=1)
        embed_frac = np.mean(proj_norm / np.maximum(total_norm, 1e-10))
        print(f"  W_lm projection on embed space: {embed_frac:.4f}")

    # 关键假设: Rank=257 ≈ hidden_dim/10
    # 但这可能只是TruncatedSVD(256)的截断效应
    # 用完整SVD验证(对小子集)
    print(f"\n  === Full SVD on small subset ===")
    # 取W_lm的随机1000行做完整SVD
    np.random.seed(42)
    idx = np.random.choice(vocab_size, min(1000, vocab_size), replace=False)
    W_sub = W_lm[idx]
    s_full = np.linalg.svd(W_sub, compute_uv=False)
    rank_full_95 = int(np.sum(s_full > 0.05 * s_full[0]))
    rank_full_99 = int(np.sum(s_full > 0.01 * s_full[0]))
    print(f"  Full SVD on {len(idx)} rows: rank(5%)={rank_full_95}, rank(1%)={rank_full_99}")
    print(f"  Top 20 singular values: {[f'{s:.2f}' for s in s_full[:20]]}")

    # W_lm^T @ W_lm 的特征值 → 行空间的有效维度
    print(f"\n  === W_lm^T @ W_lm eigenvalue analysis ===")
    # 用随机投影加速
    n_proj = min(hidden_dim, 512)
    random_proj = np.random.randn(hidden_dim, n_proj) / np.sqrt(n_proj)
    W_proj = W_lm @ random_proj  # [vocab_size, n_proj]
    G = W_proj.T @ W_proj / vocab_size  # [n_proj, n_proj]
    eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
    # 这些是W_lm^T@W_lm/vocab_size的特征值的近似
    eig_rank_95 = int(np.sum(eigvals > 0.05 * eigvals[0]))
    print(f"  Approximate eigenvalues (top 20): {[f'{e:.4f}' for e in eigvals[:20]]}")
    print(f"  Approximate rank(5%): {eig_rank_95}")

    results = {
        "arch_info": arch_info,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "intermediate_size": intermediate_size,
        "num_kv_heads": num_kv_heads,
        "is_tied": is_tied,
        "W_lm_shape": [vocab_size, hidden_dim],
        "full_svd_rank_5pct": rank_full_95,
        "full_svd_rank_1pct": rank_full_99,
        "approx_eigen_rank": eig_rank_95,
    }
    return results


# ========== P409: DS7B暗能量机制——逐层dh_norm分析 ==========

def run_p409(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P409: Dark energy mechanism - layer-by-layer dh_norm - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()

    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0
    n_layers_total = len(get_layers(model))
    # 逐层扫描
    scan_layers = list(range(0, n_layers_total, 1))  # 每层都扫
    layers = get_layers(model)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw
        w_diff_norm = float(np.linalg.norm(w_diff))
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {
            "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线 - 逐层捕获
    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles = []
    for l in scan_layers:
        if l < len(layers):
            handles.append(layers[l].register_forward_hook(make_hook(captured_base, f"L{l}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles:
        h.remove()

    h_base_per_layer = {}
    for l in scan_layers:
        key = f"L{l}"
        if key in captured_base:
            h_base_per_layer[l] = captured_base[key][0, -1, :].cpu().numpy()

    # 对每个维度, 逐层追踪dh
    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handles = []
        for l in scan_layers:
            if l < len(layers):
                handles.append(layers[l].register_forward_hook(make_hook(captured_int, f"L{l}")))
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles:
            h.remove()

        # 逐层分析
        dh_norm_list = []
        cos_wdiff_list = []
        proj_wdiff_list = []
        dlogit_theory_list = []  # 用dh预测的Δlogit

        for l in scan_layers:
            key = f"L{l}"
            if key in captured_int and key in captured_base:
                dh = (captured_int[key][0, -1, :] - captured_base[key][0, -1, :]).cpu().numpy()
                dh_norm = float(np.linalg.norm(dh))
                cos_wd = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0.0
                proj_wd = float(np.dot(dh, w_diff))

                # 用该层dh预测Δlogit
                delta_logits_theory = W_lm @ dh
                dlogit_th = float(delta_logits_theory[pos_id] - delta_logits_theory[neg_id])

                dh_norm_list.append(dh_norm)
                cos_wdiff_list.append(cos_wd)
                proj_wdiff_list.append(proj_wd)
                dlogit_theory_list.append(dlogit_th)

        # 找dh_norm暴增的拐点
        dh_norm_arr = np.array(dh_norm_list)
        dh_norm_diff = np.diff(dh_norm_arr)
        max_increase_idx = int(np.argmax(dh_norm_diff))
        max_increase_val = float(dh_norm_diff[max_increase_idx])

        # 找cos暴降的拐点
        cos_arr = np.array(cos_wdiff_list)
        cos_diff = np.diff(cos_arr)
        max_cos_drop_idx = int(np.argmin(cos_diff))
        max_cos_drop_val = float(cos_diff[max_cos_drop_idx])

        print(f"\n  {dim_name}:")
        print(f"    dh_norm: L0={dh_norm_list[0]:.1f} -> L{n_layers_total-1}={dh_norm_list[-1]:.1f}")
        print(f"    Max dh_norm increase at L{scan_layers[max_increase_idx]}: +{max_increase_val:.1f}")
        print(f"    cos_wdiff: L0={cos_wdiff_list[0]:.4f} -> L{n_layers_total-1}={cos_wdiff_list[-1]:.4f}")
        print(f"    Max cos drop at L{scan_layers[max_cos_drop_idx]}: {max_cos_drop_val:.4f}")

        results_per_dim[dim_name] = {
            "dh_norm_per_layer": {str(scan_layers[i]): dh_norm_list[i] for i in range(len(dh_norm_list))},
            "cos_wdiff_per_layer": {str(scan_layers[i]): cos_wdiff_list[i] for i in range(len(cos_wdiff_list))},
            "proj_wdiff_per_layer": {str(scan_layers[i]): proj_wdiff_list[i] for i in range(len(proj_wdiff_list))},
            "dlogit_theory_per_layer": {str(scan_layers[i]): dlogit_theory_list[i] for i in range(len(dlogit_theory_list))},
            "max_dh_norm_increase_layer": scan_layers[max_increase_idx],
            "max_dh_norm_increase_val": max_increase_val,
            "max_cos_drop_layer": scan_layers[max_cos_drop_idx],
            "max_cos_drop_val": max_cos_drop_val,
        }

    results = {
        "n_layers_total": n_layers_total,
        "scan_layers": scan_layers,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P410: 信号分配效率的精确公式 ==========

def run_p410(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P410: Precise formula for signal allocation efficiency - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape

    # 预计算W_lm统计量
    W_lm_F_sq = float(np.sum(W_lm ** 2))
    row_norms = np.sqrt(np.sum(W_lm ** 2, axis=1))
    mean_row_norm = float(np.mean(row_norms))
    std_row_norm = float(np.std(row_norms))

    # W_lm的SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=min(hidden_dim, 256))
    svd.fit(W_lm)
    V_right = svd.components_  # [256, hidden_dim]

    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0
    layers = get_layers(model)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw
        w_diff_norm = float(np.linalg.norm(w_diff))
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {
            "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
            "w_pos_raw": w_pos_raw, "w_neg_raw": w_neg_raw,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handle = layers[-1].register_forward_hook(make_hook(captured_base, "last"))
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
    handle.remove()
    h_base = captured_base["last"][0, -1, :].cpu().numpy()

    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]
        w_pos_raw = dim_info[dim_name]["w_pos_raw"]
        w_neg_raw = dim_info[dim_name]["w_neg_raw"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handle = layers[-1].register_forward_hook(make_hook(captured_int, "last"))
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        handle.remove()

        h_int = captured_int["last"][0, -1, :].cpu().numpy()
        dh = h_int - h_base
        dh_norm = float(np.linalg.norm(dh))

        # 实际Δlogit
        actual_delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        actual_delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        actual_dlogit = actual_delta_pos - actual_delta_neg

        # 理论预测
        delta_logits = W_lm @ dh
        total_energy_L1 = float(np.sum(np.abs(delta_logits)))
        eff_L1_actual = abs(actual_dlogit) / max(total_energy_L1, 1e-10)

        # ★★★ 精确公式推导 ★★★
        # Δlogit_i = W_lm[i] · dh
        # |Δlogit_i| ≈ ||W_lm[i]|| * ||dh|| * |cos(W_lm[i], dh)|
        # 如果cos近似均匀分布在[-1,1]:
        #   E[|cos|] = 1/2 (不是1/sqrt(d), 因为是cos不是|cos|的平均)
        #   但实际上 |cos| 在高维空间中 E[|cos|] ≈ sqrt(2/(pi*d))
        # 
        # 总能量L1 = Σ|Δlogit_i| ≈ vocab_size * mean_||w_i|| * ||dh|| * sqrt(2/(pi*d))
        # 目标能量 = |Δlogit_target| = |W_diff · dh| = ||w_diff|| * ||dh|| * |cos(w_diff, dh)|
        # 
        # eff_L1 = 目标/总 ≈ ||w_diff|| * |cos(w_diff, dh)| / (vocab_size * mean_||w_i|| * sqrt(2/(pi*d)))
        
        cos_dh_wdiff = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0.0
        
        # 公式1: eff = ||w_diff|| * |cos| / (vocab_size * mean_||w_i|| * sqrt(2/(pi*d)))
        formula_1 = float(w_diff_norm * abs(cos_dh_wdiff) / (vocab_size * mean_row_norm * np.sqrt(2 / (np.pi * hidden_dim))))
        
        # 公式2: eff = |proj(dh, w_diff)| / (vocab_size * mean_||w_i|| * ||dh|| * sqrt(2/(pi*d)))
        proj_dh_wdiff = float(np.dot(dh, w_diff))
        formula_2 = float(abs(proj_dh_wdiff) / (vocab_size * mean_row_norm * dh_norm * np.sqrt(2 / (np.pi * hidden_dim))))
        
        # 公式3: eff = |Δlogit_actual| / (||W_lm||_F * ||dh|| * sqrt(1/d))
        formula_3 = float(abs(actual_dlogit) / (np.sqrt(W_lm_F_sq) * dh_norm * np.sqrt(1.0 / hidden_dim)))
        
        # 公式4: 从W_lm行空间的结构出发
        # 如果dh主要在V_lang方向(7维), 则:
        # total_energy ≈ vocab_size * mean_||w_i||^2 * ||dh||^2 / hidden_dim (随机投影)
        # 但实际total_energy_L1已测出
        # eff = |Δlogit| / total_energy_L1
        # 用高斯近似: total_energy_L1 ≈ vocab_size * E[|N(0, σ²)|] = vocab_size * σ * sqrt(2/pi)
        # 其中 σ² = Var[W_lm[i] · dh] ≈ ||w_i||² * ||dh||² / hidden_dim
        sigma_proj = float(np.std(delta_logits))
        total_energy_gaussian = vocab_size * sigma_proj * np.sqrt(2 / np.pi)
        eff_gaussian = abs(actual_dlogit) / max(total_energy_gaussian, 1e-10)
        
        # 公式5: 精确到SVD分量
        # dh在W_lm右奇异向量上的投影
        proj_on_sing = dh @ V_right.T  # [256]
        # Δlogit_i = U[i] @ diag(S) @ proj_on_sing
        # 总能量 ≈ Σ_k S_k² * proj_on_sing[k]² (近似, 忽略U的结构)
        # 但更精确: total_energy_L2 = ||W_lm @ dh||² = dh^T @ W_lm^T @ W_lm @ dh
        total_energy_L2 = float(np.sum(delta_logits ** 2))
        eff_L2_actual = actual_dlogit**2 / max(total_energy_L2, 1e-10)

        results_per_dim[dim_name] = {
            "actual_dlogit": actual_dlogit,
            "eff_L1_actual": eff_L1_actual,
            "eff_L2_actual": eff_L2_actual,
            "formula_1": formula_1,
            "formula_2": formula_2,
            "formula_3": formula_3,
            "eff_gaussian": eff_gaussian,
            "cos_dh_wdiff": cos_dh_wdiff,
            "dh_norm": dh_norm,
            "w_diff_norm": w_diff_norm,
            "proj_dh_wdiff": proj_dh_wdiff,
            "total_energy_L1": total_energy_L1,
            "total_energy_L2": total_energy_L2,
            "sigma_proj": sigma_proj,
        }

        print(f"\n  {dim_name}:")
        print(f"    eff_L1_actual={eff_L1_actual:.6e}")
        print(f"    formula_1(||w_diff||*cos/V*mean*sqrt)={formula_1:.6e}")
        print(f"    formula_2(proj/V*mean*dh*sqrt)={formula_2:.6e}")
        print(f"    formula_3(dlogit/||W||*dh*sqrt)={formula_3:.6e}")
        print(f"    eff_gaussian={eff_gaussian:.6e}")
        print(f"    Best formula match: ", end="")
        errors = {
            "f1": abs(formula_1 - eff_L1_actual) / max(eff_L1_actual, 1e-10),
            "f2": abs(formula_2 - eff_L1_actual) / max(eff_L1_actual, 1e-10),
            "f3": abs(formula_3 - eff_L1_actual) / max(eff_L1_actual, 1e-10),
            "gauss": abs(eff_gaussian - eff_L1_actual) / max(eff_L1_actual, 1e-10),
        }
        best = min(errors, key=errors.get)
        print(f"{best} (error={errors[best]:.2f})")

    # ★★★ 全局最优公式 ★★★
    print(f"\n  === Global best formula ===")
    all_errors = {"formula_1": [], "formula_2": [], "formula_3": [], "eff_gaussian": []}
    for dim_name, r in results_per_dim.items():
        for fname in all_errors:
            all_errors[fname].append(abs(r[fname] - r["eff_L1_actual"]) / max(r["eff_L1_actual"], 1e-10))
    
    for fname in all_errors:
        mean_err = np.mean(all_errors[fname])
        print(f"  {fname}: mean relative error = {mean_err:.2f}")

    results = {
        "W_lm_shape": [vocab_size, hidden_dim],
        "W_lm_F_sq": W_lm_F_sq,
        "mean_row_norm": mean_row_norm,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p407", "p408", "p409", "p410", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p407", "p408", "p409", "p410"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p407":
                results["p407"] = run_p407(model, tokenizer, device, args.model)
            elif exp == "p408":
                results["p408"] = run_p408(model, tokenizer, device, args.model)
            elif exp == "p409":
                results["p409"] = run_p409(model, tokenizer, device, args.model)
            elif exp == "p410":
                results["p410"] = run_p410(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    out_file = OUT_DIR / f"phase_lxxxii_p407_410_{args.model}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
