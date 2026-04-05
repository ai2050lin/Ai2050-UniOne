#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage592: 消歧策略统一理论 — 四模型对比
目标：
  1. 分析消歧路径选择与模型架构参数的关系（层数、head数、hidden维度、MLP维度）
  2. 分析消歧效率（消歧度/FLOPS）是否因策略不同而不同
  3. 分析消歧信息的编码位置分布（哪些维度带承载消歧）
  4. 寻找是否存在"最优消歧策略"或策略选择受什么约束
  5. 测试消歧路径与编码空间的几何关系（编码空间曲率、各向异性）
模型：Qwen3 / DeepSeek7B / GLM4 / Gemma4（依次运行，避免GPU溢出）
"""

from __future__ import annotations
import sys, json, time, gc, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, get_model_device,
    discover_layers, move_batch_to_model_device, MODEL_SPECS
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 20个歧义词
POLYSEMY_WORDS = [
    {"word": "bank",    "ctx1": "The river bank was muddy after the heavy rain.", "ctx2": "The bank approved the loan for the new business."},
    {"word": "plant",   "ctx1": "The factory plant employs over five hundred workers.", "ctx2": "She watered the plant in the garden every morning."},
    {"word": "spring",  "ctx1": "The hot spring resort attracts many tourists each year.", "ctx2": "Spring is the most beautiful season of the year."},
    {"word": "apple",   "ctx1": "She ate a sweet red apple from the orchard.", "ctx2": "Apple released the new iPhone at the conference."},
    {"word": "pool",    "ctx1": "They swam in the swimming pool all afternoon.", "ctx2": "The car pool arrangement saved everyone money."},
    {"word": "seal",    "ctx1": "The seal balanced a ball on its nose at the zoo.", "ctx2": "Please seal the envelope before mailing it."},
    {"word": "fair",    "ctx1": "The county fair had rides and games for everyone.", "ctx2": "The judge made a fair and impartial decision."},
    {"word": "match",   "ctx1": "He struck the match to light the candle.", "ctx2": "The football match was exciting to watch."},
    {"word": "light",   "ctx1": "Please turn on the light in the dark room.", "ctx2": "The package was very light and easy to carry."},
    {"word": "ring",    "ctx1": "She wore a diamond ring on her finger.", "ctx2": "Please ring the bell when you arrive."},
    {"word": "bark",    "ctx1": "The dog started to bark at the stranger.", "ctx2": "The tree bark was rough and dark."},
    {"word": "row",     "ctx1": "They had a huge row about the money.", "ctx2": "She sat in the front row of the theater."},
    {"word": "ruler",   "ctx1": "He used a wooden ruler to measure the desk.", "ctx2": "The ruler made an important decision today."},
    {"word": "nail",    "ctx1": "He hit the nail with a heavy hammer.", "ctx2": "She painted her fingernail a bright red color."},
    {"word": "square",  "ctx1": "The town square was filled with people.", "ctx2": "A square has four equal sides."},
    {"word": "mouse",   "ctx1": "The mouse ran across the kitchen floor.", "ctx2": "He clicked the mouse to select the file."},
    {"word": "bat",     "ctx1": "The bat flew out of the cave at dusk.", "ctx2": "He swung the baseball bat with great force."},
    {"word": "park",    "ctx1": "They walked through the park after lunch.", "ctx2": "Please park the car in the garage."},
    {"word": "saw",     "ctx1": "He cut the wood with a hand saw.", "ctx2": "She saw the movie last weekend."},
    {"word": "draft",   "ctx1": "The cold draft blew through the window.", "ctx2": "She wrote the first draft of her novel."},
]


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_hidden_state(model, tokenizer, sentence, layer_idx=None):
    """获取hidden state（指定层或末层）"""
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    if layer_idx is not None:
        return out.hidden_states[layer_idx][0, -1, :].float().cpu()
    return out.hidden_states[-1][0, -1, :].float().cpu()


def get_all_hidden_states(model, tokenizer, sentence):
    """获取所有层的hidden states"""
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return [h[0, -1, :].float().cpu() for h in out.hidden_states]


def get_config_params(model_key, model):
    """提取模型架构参数（兼容不同config格式）"""
    config = model.config
    params = {}
    layers = discover_layers(model)
    
    # 从config尝试获取，全部fallback到实际层结构
    n_layers = len(layers)
    params["n_layers"] = n_layers
    
    # hidden_dim: 从q_proj或config
    hidden_dim = getattr(config, "hidden_size", 0)
    if hidden_dim == 0:
        hidden_dim = getattr(config, "n_embd", 0)
    if hidden_dim == 0 and layers:
        if hasattr(layers[0].self_attn, 'q_proj'):
            hidden_dim = layers[0].self_attn.q_proj.in_features
        elif hasattr(layers[0].self_attn, 'qkv_proj'):
            hidden_dim = layers[0].self_attn.qkv_proj.in_features
    params["hidden_dim"] = hidden_dim
    
    # n_heads: 从config或从q_proj output / head_dim推算
    n_heads = getattr(config, "num_attention_heads", 0)
    if n_heads == 0:
        n_heads = getattr(config, "n_head", 0)
    if n_heads == 0 and layers and hasattr(layers[0].self_attn, 'q_proj'):
        q_out = layers[0].self_attn.q_proj.out_features
        # Gemma4典型head_dim=256, Gemma2=256, Gemma1=256
        # Qwen3典型head_dim=128, DS7B=128, GLM4=128
        for hd in [256, 128, 64]:
            if q_out % hd == 0 and q_out // hd >= 1:
                params["_head_dim_guess"] = hd
                n_heads = q_out // hd
                break
    params["n_heads"] = n_heads
    
    # head_dim
    if params["n_heads"] > 0 and params["hidden_dim"] > 0:
        params["head_dim"] = params["hidden_dim"] // params["n_heads"]
    elif n_layers > 0 and hasattr(layers[0].self_attn, 'q_proj'):
        params["head_dim"] = params.get("_head_dim_guess", 128)
    else:
        params["head_dim"] = 128  # 默认
    
    # n_kv_heads
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(config, "n_kv_head", None)
    if n_kv_heads is None or n_kv_heads == 0:
        n_kv_heads = params["n_heads"]  # 默认MHA
    params["n_kv_heads"] = n_kv_heads
    
    # vocab_size
    params["vocab_size"] = getattr(config, "vocab_size", 0)
    
    # MLP维度
    mlp_dim = 0
    if layers:
        layer = layers[0]
        if hasattr(layer.mlp, "gate_proj"):
            mlp_dim = layer.mlp.gate_proj.out_features
        elif hasattr(layer.mlp, "gate_up_proj"):
            mlp_dim = layer.mlp.gate_up_proj.out_features // 2
        elif hasattr(layer.mlp, "c_fc"):
            mlp_dim = layer.mlp.c_fc.out_features
    if mlp_dim == 0:
        mlp_dim = getattr(config, "intermediate_size",
                  getattr(config, "ffn_dim", 4 * params["hidden_dim"]))
    params["mlp_dim"] = mlp_dim
    
    # 参数量估计
    # attention: 4 * hidden * head_dim * n_heads + 4 * hidden (bias)
    attn_params = 4 * params["hidden_dim"] * params["head_dim"] * params["n_heads"]
    # MLP: 2 * hidden * mlp_dim + 2 * mlp_dim
    mlp_params = 2 * params["hidden_dim"] * params["mlp_dim"]
    # embedding
    emb_params = params["vocab_size"] * params["hidden_dim"]
    params["total_params_B"] = (attn_params * params["n_layers"] + mlp_params * params["n_layers"] + emb_params) / 1e9
    
    # 每层attention FLOPS比例
    params["attn_to_mlp_ratio"] = attn_params / mlp_params if mlp_params > 0 else 0
    
    # GQA (Grouped Query Attention)
    params["is_gqa"] = params["n_kv_heads"] != params["n_heads"]
    params["gqa_ratio"] = params["n_heads"] // params["n_kv_heads"]
    
    return params


def compute_disamb_metrics(model, tokenizer, pw):
    """计算消歧的详细指标"""
    hs1_all = get_all_hidden_states(model, tokenizer, pw["ctx1"])
    hs2_all = get_all_hidden_states(model, tokenizer, pw["ctx2"])
    n_layers = len(hs1_all) - 1  # 排除embedding
    
    results = {}
    
    # 1. 逐层消歧度（1-cos）
    layer_disamb = [1 - cos(h1, h2) for h1, h2 in zip(hs1_all, hs2_all)]
    results["layer_disamb"] = layer_disamb
    
    # 2. 峰值消歧度和峰值层
    peak_layer = max(range(1, len(layer_disamb)), key=lambda i: layer_disamb[i])
    results["peak_layer"] = peak_layer
    results["peak_disamb"] = layer_disamb[peak_layer]
    
    # 3. 最终消歧度
    results["final_disamb"] = layer_disamb[-1]
    
    # 4. 消歧效率：峰值消歧度 / (峰值层 / 总层数)
    # （越早达到峰值消歧度，效率越高）
    peak_pct = peak_layer / n_layers
    results["disamb_efficiency"] = results["peak_disamb"] / max(peak_pct, 0.01)
    
    # 5. 消歧增量曲线（每层贡献）
    deltas = [layer_disamb[i] - layer_disamb[i-1] for i in range(1, len(layer_disamb))]
    results["layer_deltas"] = deltas
    
    # 6. 最大单层增量
    if deltas:
        max_delta_idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
        results["max_delta_layer"] = max_delta_idx + 1
        results["max_delta_value"] = deltas[max_delta_idx]
    
    # 7. 消歧信息编码维度分析
    # 计算消歧方向：h1-h2在不同层的方向
    h1_final = hs1_all[-1]
    h2_final = hs2_all[-1]
    disamb_vec = h1_final - h2_final
    
    # SVD分析消歧方向的能量分布
    U, S, Vt = torch.linalg.svd(disamb_vec.unsqueeze(0), full_matrices=False)
    results["disamb_vec_rank1_energy"] = (S[0] ** 2 / (S ** 2).sum()).item()
    
    # 8. 编码空间几何分析
    # 随机采样一些词的hidden state，计算编码空间的各向异性
    random_words = ["the", "cat", "run", "happy", "big", "small", "fast", "slow",
                    "blue", "green", "house", "car", "book", "water", "fire"]
    random_hs = []
    for w in random_words:
        try:
            h = get_hidden_state(model, tokenizer, f"The word is {w}.")
            random_hs.append(h)
        except:
            pass
    
    if len(random_hs) >= 5:
        H = torch.stack(random_hs)
        # 奇异值谱
        _, S_all, _ = torch.linalg.svd(H - H.mean(0), full_matrices=False)
        S_norm = S_all / S_all[0]
        results["sv_spectrum_ratio_5th"] = S_norm[4].item() if len(S_norm) > 4 else 0
        results["sv_spectrum_ratio_10th"] = S_norm[9].item() if len(S_norm) > 9 else 0
        results["effective_rank"] = (S_all.sum() / S_all[0]).item()
    
    # 9. embedding层消歧（预编码贡献）
    emb_disamb = layer_disamb[0]
    results["embedding_disamb"] = emb_disamb
    results["network_disamb_gain"] = results["peak_disamb"] - emb_disamb
    
    # 10. MLP vs attention消歧贡献分解
    # 使用逐层消融估算每层的attn/mlp贡献
    layers = discover_layers(model)
    attn_contribs = []
    mlp_contribs = []
    
    for li in [0, peak_layer, n_layers - 1]:
        li = min(li, n_layers - 1)
        try:
            # Attention消融
            from torch.utils.hooks import RemovableHandle
            handles = []
            def zero_hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + out[1:]
                return torch.zeros_like(out)
            
            h_attn_abl = get_hidden_state_with_hook(
                model, tokenizer, pw["ctx1"], li, "attn", layers
            )
            h_baseline = hs1_all[li]
            attn_loss = 1 - cos(h_baseline, h_attn_abl)
            
            h_mlp_abl = get_hidden_state_with_hook(
                model, tokenizer, pw["ctx1"], li, "mlp", layers
            )
            mlp_loss = 1 - cos(h_baseline, h_mlp_abl)
            
            attn_contribs.append({"layer": li, "attn_loss": attn_loss, "mlp_loss": mlp_loss})
        except Exception as e:
            attn_contribs.append({"layer": li, "error": str(e)})
    
    results["attn_mlp_contribs"] = attn_contribs
    
    return results


def get_hidden_state_with_hook(model, tokenizer, sentence, layer_idx, component, layers):
    """消融指定层的指定组件后获取hidden state"""
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    
    def zero_hook(mod, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    
    target = layers[layer_idx].self_attn if component == "attn" else layers[layer_idx].mlp
    handle = target.register_forward_hook(zero_hook)
    
    try:
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
    finally:
        handle.remove()
    
    return out.hidden_states[layer_idx + 1][0, -1, :].float().cpu()


def run_model(model_key):
    """对单个模型运行完整分析"""
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — 消歧策略统一理论分析")
    print(f"{'='*60}")
    
    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if bundle is None:
        return {"error": f"Failed to load {model_key}"}
    model, tokenizer = bundle
    n_layers = len(discover_layers(model))
    
    # 1. 架构参数
    config_params = get_config_params(model_key, model)
    print(f"  架构: L={config_params['n_layers']}, H={config_params['hidden_dim']}, "
          f"A={config_params['n_heads']}, MLP={config_params['mlp_dim']}, "
          f"GQA={config_params['is_gqa']}, Params={config_params['total_params_B']:.2f}B")
    
    # 2. 逐词消歧指标
    word_results = []
    for i, pw in enumerate(POLYSEMY_WORDS):
        print(f"  [{i+1}/20] {pw['word']}...", end="", flush=True)
        try:
            metrics = compute_disamb_metrics(model, tokenizer, pw)
            metrics["word"] = pw["word"]
            word_results.append(metrics)
            print(f" peak=L{metrics['peak_layer']}({metrics['peak_layer']/n_layers*100:.0f}%), "
                  f"disamb={metrics['peak_disamb']:.4f}, eff={metrics['disamb_efficiency']:.2f}")
        except Exception as e:
            print(f" ERROR: {e}")
            word_results.append({"word": pw["word"], "error": str(e)})
    
    # 3. 汇总统计
    valid = [w for w in word_results if "error" not in w]
    summary = {}
    
    if valid:
        # 峰值层分布
        peak_layers = [w["peak_layer"] / n_layers for w in valid]
        summary["mean_peak_pct"] = float(np.mean(peak_layers))
        summary["std_peak_pct"] = float(np.std(peak_layers))
        
        # 消歧度
        peak_disambs = [w["peak_disamb"] for w in valid]
        summary["mean_peak_disamb"] = float(np.mean(peak_disambs))
        summary["std_peak_disamb"] = float(np.std(peak_disambs))
        
        # 消歧效率
        effs = [w["disamb_efficiency"] for w in valid]
        summary["mean_efficiency"] = float(np.mean(effs))
        summary["std_efficiency"] = float(np.std(effs))
        
        # embedding预编码贡献
        emb_disambs = [w["embedding_disamb"] for w in valid]
        summary["mean_emb_disamb"] = float(np.mean(emb_disambs))
        
        # 网络加工增益
        gains = [w["network_disamb_gain"] for w in valid]
        summary["mean_network_gain"] = float(np.mean(gains))
        summary["network_gain_ratio"] = float(np.mean(gains) / max(np.mean(peak_disambs), 1e-8))
        
        # 最大单层增量
        max_deltas = [abs(w.get("max_delta_value", 0)) for w in valid]
        summary["mean_max_delta"] = float(np.mean(max_deltas))
        max_delta_layers = [w.get("max_delta_layer", 0) / n_layers for w in valid]
        summary["mean_max_delta_layer_pct"] = float(np.mean(max_delta_layers))
        
        # 消歧方向rank-1能量
        rank1s = [w.get("disamb_vec_rank1_energy", 0) for w in valid]
        summary["mean_rank1_energy"] = float(np.mean(rank1s))
        
        # 编码空间各向异性
        sv5s = [w.get("sv_spectrum_ratio_5th", 0) for w in valid]
        sv10s = [w.get("sv_spectrum_ratio_10th", 0) for w in valid]
        eff_ranks = [w.get("effective_rank", 0) for w in valid]
        summary["mean_sv_ratio_5th"] = float(np.mean([s for s in sv5s if s > 0])) if any(s > 0 for s in sv5s) else 0
        summary["mean_sv_ratio_10th"] = float(np.mean([s for s in sv10s if s > 0])) if any(s > 0 for s in sv10s) else 0
        summary["mean_effective_rank"] = float(np.mean([r for r in eff_ranks if r > 0])) if any(r > 0 for r in eff_ranks) else 0
    
    elapsed = time.time() - t0
    summary["n_valid"] = len(valid)
    summary["elapsed_s"] = round(elapsed, 1)
    
    print(f"\n  汇总:")
    print(f"    有效词数: {len(valid)}/20")
    print(f"    峰值层: {summary.get('mean_peak_pct', 0)*100:.1f}% ± {summary.get('std_peak_pct', 0)*100:.1f}%")
    print(f"    峰值消歧度: {summary.get('mean_peak_disamb', 0):.4f} ± {summary.get('std_peak_disamb', 0):.4f}")
    print(f"    消歧效率: {summary.get('mean_efficiency', 0):.2f} ± {summary.get('std_efficiency', 0):.2f}")
    print(f"    embedding预编码: {summary.get('mean_emb_disamb', 0):.4f}")
    print(f"    网络加工增益: {summary.get('mean_network_gain', 0):.4f} (占比{summary.get('network_gain_ratio', 0)*100:.1f}%)")
    print(f"    最大单层增量: {summary.get('mean_max_delta', 0):.4f} @ {summary.get('mean_max_delta_layer_pct', 0)*100:.0f}%层")
    print(f"    消歧方向rank1能量: {summary.get('mean_rank1_energy', 0):.4f}")
    print(f"    编码空间有效秩: {summary.get('mean_effective_rank', 0):.2f}")
    print(f"    耗时: {elapsed:.1f}s")
    
    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "config_params": config_params,
        "word_results": word_results,
        "summary": summary,
    }


def main():
    print("=" * 60)
    print("  Stage592: 消歧策略统一理论 — 四模型对比")
    print("=" * 60)
    
    results = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        results[mk] = run_model(mk)
    
    # 跨模型对比分析
    print("\n" + "=" * 60)
    print("  跨模型对比分析")
    print("=" * 60)
    
    cross = {}
    for mk, r in results.items():
        if "error" in r:
            continue
        s = r["summary"]
        cp = r["config_params"]
        cross[mk] = {
            "label": MODEL_SPECS[mk]["label"] if mk in MODEL_SPECS else mk,
            "n_layers": cp["n_layers"],
            "hidden_dim": cp["hidden_dim"],
            "n_heads": cp["n_heads"],
            "n_kv_heads": cp["n_kv_heads"],
            "mlp_dim": cp["mlp_dim"],
            "total_params_B": round(cp["total_params_B"], 2),
            "is_gqa": cp["is_gqa"],
            "attn_to_mlp_ratio": round(cp["attn_to_mlp_ratio"], 3),
            "mean_peak_pct": round(s.get("mean_peak_pct", 0), 4),
            "mean_peak_disamb": round(s.get("mean_peak_disamb", 0), 4),
            "mean_efficiency": round(s.get("mean_efficiency", 0), 2),
            "mean_emb_disamb": round(s.get("mean_emb_disamb", 0), 4),
            "mean_network_gain": round(s.get("mean_network_gain", 0), 4),
            "network_gain_ratio": round(s.get("network_gain_ratio", 0), 4),
            "mean_max_delta": round(s.get("mean_max_delta", 0), 4),
            "mean_max_delta_layer_pct": round(s.get("mean_max_delta_layer_pct", 0), 4),
            "mean_rank1_energy": round(s.get("mean_rank1_energy", 0), 4),
            "mean_effective_rank": round(s.get("mean_effective_rank", 0), 2),
        }
    
    # 打印对比表
    print(f"\n  {'指标':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*65}")
    
    metrics_to_show = [
        ("层数", "n_layers"), ("隐藏维度", "hidden_dim"), ("注意力头数", "n_heads"),
        ("MLP维度", "mlp_dim"), ("参数量(B)", "total_params_B"),
        ("GQA", "is_gqa"), ("Attn/MLP比", "attn_to_mlp_ratio"),
        ("峰值层(%)", "mean_peak_pct"), ("峰值消歧度", "mean_peak_disamb"),
        ("消歧效率", "mean_efficiency"), ("Embedding消歧", "mean_emb_disamb"),
        ("网络增益", "mean_network_gain"), ("网络增益占比", "network_gain_ratio"),
        ("最大单层增量", "mean_max_delta"), ("增量层(%)", "mean_max_delta_layer_pct"),
        ("Rank1能量", "mean_rank1_energy"), ("编码有效秩", "mean_effective_rank"),
    ]
    
    for label, key in metrics_to_show:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = cross.get(mk, {}).get(key, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
            else:
                vals.append(str(v))
        print(f"  {label:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
    
    # 相关性分析
    print(f"\n  策略与架构的相关性:")
    # 收集跨模型数据
    model_data = {mk: v for mk, v in cross.items() if "error" not in v}
    if len(model_data) >= 3:
        mks = list(model_data.keys())
        numeric_keys = ["n_layers", "hidden_dim", "n_heads", "mlp_dim", "total_params_B",
                       "attn_to_mlp_ratio", "mean_peak_pct", "mean_peak_disamb",
                       "mean_efficiency", "mean_emb_disamb", "mean_network_gain",
                       "mean_max_delta", "mean_rank1_energy", "mean_effective_rank"]
        
        for k1 in numeric_keys:
            for k2 in numeric_keys:
                if k1 >= k2:
                    continue
                v1 = [model_data[mk].get(k1, 0) for mk in mks]
                v2 = [model_data[mk].get(k2, 0) for mk in mks]
                if all(v != 0 for v in v1) and all(v != 0 for v in v2):
                    corr = np.corrcoef(v1, v2)[0, 1]
                    if abs(corr) > 0.7:
                        print(f"    {k1} ↔ {k2}: r={corr:.3f}")
    
    # 保存结果
    output = {
        "timestamp": TIMESTAMP,
        "models": {mk: {k: v for k, v in r.items() if k != "word_results" or True 
                   for k, v in {"config_params": r.get("config_params"), 
                                "summary": r.get("summary"),
                                "word_results": r.get("word_results")}.items()}
                  for mk, r in results.items()},
        "cross_model_summary": cross,
    }
    
    # 精简word_results以减小文件大小
    for mk in output["models"]:
        if "word_results" in output["models"][mk]:
            output["models"][mk]["word_results"] = [
                {k: v for k, v in wr.items() 
                 if k not in ("layer_disamb", "layer_deltas", "attn_mlp_contribs")}
                for wr in output["models"][mk]["word_results"]
            ]
    
    out_path = OUTPUT_DIR / f"stage592_disamb_strategy_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果保存到: {out_path}")
    
    return output


if __name__ == "__main__":
    main()
