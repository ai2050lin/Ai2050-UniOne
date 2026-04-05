#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage589-590合并: 四模型家族预测泛化与跨家族迁移
目标：
  1. 留一法预测：用N-1个家族成员预测第N个的编码骨架（验证stage581的动物cos=0.99）
  2. 跨家族零样本预测：用一个家族的骨干预测另一个家族的新成员
  3. 预测精度分解：全局骨干贡献 vs 家族骨干贡献 vs 残差
  4. 跨模型对比哪些家族可预测、哪些不可预测
模型：Qwen3 / DeepSeek7B / GLM4 / Gemma4
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
    discover_layers, move_batch_to_model_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 六类概念家族（每类6-8个成员）
FAMILIES = {
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "lemon"],
    "animal": ["cat", "dog", "horse", "elephant", "tiger", "lion", "bear", "wolf"],
    "tool": ["hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "knife", "chisel"],
    "country": ["China", "France", "Japan", "Brazil", "Germany", "India", "Canada", "Italy"],
    "celestial": ["sun", "moon", "Mars", "Venus", "Jupiter", "Saturn", "Mercury", "Neptune"],
    "abstract": ["freedom", "justice", "truth", "beauty", "love", "hope", "peace", "courage"],
}


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_encoding(model, tokenizer, word, layer_frac=0.8):
    """获取指定词在指定层的hidden state"""
    enc = tokenizer(f"The {word} is very important.", return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    n_layers = len(discover_layers(model))
    target_layer = int(n_layers * layer_frac)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[target_layer][0, -1, :].float().cpu()


def get_all_layer_encoding(model, tokenizer, word):
    """获取指定词在所有层的hidden state"""
    enc = tokenizer(f"The {word} is very important.", return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return [h[0, -1, :].float().cpu() for h in out.hidden_states]


def leave_one_out_predict(encodings, word_list, target_idx):
    """留一法：用其他成员的均值预测目标成员的编码"""
    target = encodings[target_idx]
    others = [encodings[i] for i in range(len(encodings)) if i != target_idx]
    prediction = torch.stack(others).mean(dim=0)
    return cos(prediction, target)


def cross_family_predict(source_encodings, target_encoding):
    """跨家族零样本：用源家族的均值预测目标家族成员"""
    source_mean = torch.stack(source_encodings).mean(dim=0)
    return cos(source_mean, target_encoding)


def decompose_prediction(global_mean, family_mean, word_encoding):
    """分解预测精度：全局骨干 + 家族骨干 + 残差"""
    global_contrib = cos(global_mean, word_encoding)
    residual_after_global = word_encoding - global_mean * cos(global_mean, word_encoding)
    family_residual = family_mean - global_mean * cos(global_mean, family_mean)
    family_contrib = cos(family_residual, residual_after_global) if residual_after_global.norm() > 1e-8 else 0
    return {
        "global_bone_cos": round(global_contrib, 4),
        "family_bone_cos": round(max(0, family_contrib), 4),
        "residual_norm": round(residual_after_global.norm().item(), 4),
    }


def run_single_model(model_key):
    print(f"\n{'='*60}")
    print(f"Stage589-590 - 模型: {model_key}")
    print(f"{'='*60}")
    t0 = time.time()

    model, tokenizer = load_model_bundle(model_key)
    n_layers = len(discover_layers(model))
    print(f"  层数={n_layers}")

    # 获取所有词的编码（在80%层，即晚层写入区）
    family_encodings = {}
    for fam, words in FAMILIES.items():
        print(f"  编码 {fam}...", end="", flush=True)
        encs = []
        for w in words:
            h = get_encoding(model, tokenizer, w, layer_frac=0.8)
            encs.append(h)
        family_encodings[fam] = encs
        print(f" done")

    # 全局骨干（所有词的均值）
    all_encs = []
    for encs in family_encodings.values():
        all_encs.extend(encs)
    global_mean = torch.stack(all_encs).mean(dim=0)

    results = {}
    for fam, words in FAMILIES.items():
        encs = family_encodings[fam]
        family_mean = torch.stack(encs).mean(dim=0)

        # 留一法预测
        loo_scores = []
        for i in range(len(words)):
            sc = leave_one_out_predict(encs, words, i)
            loo_scores.append({"word": words[i], "cos": round(sc, 4)})
        mean_loo = np.mean([s["cos"] for s in loo_scores])

        # 家族内聚性
        intra_cos = []
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                intra_cos.append(cos(encs[i], encs[j]))
        mean_intra = np.mean(intra_cos) if intra_cos else 0

        # 跨家族距离
        inter_cos = []
        for other_fam, other_encs in family_encodings.items():
            if other_fam == fam:
                continue
            for oe in other_encs:
                inter_cos.append(cos(family_mean, oe))
        mean_inter = np.mean(inter_cos) if inter_cos else 0

        # 编码分解
        decomp = decompose_prediction(global_mean, family_mean, family_mean)

        results[fam] = {
            "n_members": len(words),
            "leave_one_out": loo_scores,
            "mean_loo_cos": round(mean_loo, 4),
            "mean_intra_cos": round(mean_intra, 4),
            "mean_inter_cos": round(mean_inter, 4),
            "cohesion_ratio": round(mean_intra / max(mean_inter, 1e-8), 4),
            "decomposition": decomp,
        }
        print(f"  {fam}: LOO={mean_loo:.4f}, intra={mean_intra:.4f}, "
              f"inter={mean_inter:.4f}, ratio={mean_intra/max(mean_inter,1e-8):.2f}")

    # 跨家族零样本预测
    print("\n  跨家族零样本预测:")
    cross_family = {}
    for src_fam in FAMILIES:
        cross_family[src_fam] = {}
        src_mean = torch.stack(family_encodings[src_fam]).mean(dim=0)
        for tgt_fam in FAMILIES:
            if tgt_fam == src_fam:
                continue
            tgt_scores = [cos(src_mean, e) for e in family_encodings[tgt_fam]]
            mean_cross = np.mean(tgt_scores)
            cross_family[src_fam][tgt_fam] = round(mean_cross, 4)

    # 打印跨家族矩阵
    fam_names = list(FAMILIES.keys())
    print("  " + " " * 12 + "  ".join(f"{n[:5]:>5}" for n in fam_names))
    for src in fam_names:
        row = f"  {src[:10]:10}"
        for tgt in fam_names:
            if src == tgt:
                row += "  --  "
            else:
                row += f" {cross_family[src][tgt]:.3f} "
        print(row)

    # 层动态分析（选3个家族，看LOO随层变化）
    print("\n  层动态分析(LOO vs layer):")
    layer_dynamics = {}
    sample_families = ["animal", "fruit", "abstract"]
    for fam in sample_families:
        words = FAMILIES[fam]
        all_layer_encs = {w: get_all_layer_encoding(model, tokenizer, w) for w in words}
        layer_loo = []
        for l in range(n_layers):
            layer_encs = [all_layer_encs[w][l] for w in words]
            loo = np.mean([leave_one_out_predict(layer_encs, words, i) for i in range(len(words))])
            layer_loo.append(round(loo, 4))
        layer_dynamics[fam] = layer_loo
        peak_l = layer_loo.index(max(layer_loo))
        print(f"    {fam}: peak=L{peak_l}({max(layer_loo):.3f}), "
              f"L0={layer_loo[0]:.3f}, L{peak_l}={max(layer_loo):.3f}, L{n_layers-1}={layer_loo[-1]:.3f}")

    total = time.time() - t0

    # 汇总
    summary = {
        "model": model_key,
        "n_layers": n_layers,
        "total_time_s": round(total, 1),
        "family_results": results,
        "cross_family": cross_family,
        "layer_dynamics": layer_dynamics,
        "ranked_loo": {fam: results[fam]["mean_loo_cos"] for fam in FAMILIES},
    }

    free_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)

    return summary


def main():
    print("=" * 60)
    print("Stage 589-590: 四模型家族预测泛化与跨家族迁移")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    all_results = {}

    for mk in model_keys:
        try:
            summary = run_single_model(mk)
            all_results[mk] = summary
        except Exception as e:
            print(f"\n  {mk} ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[mk] = {"error": str(e)}

    # 跨模型对比
    print("\n" + "=" * 60)
    print("跨模型家族可预测性排名")
    print("=" * 60)
    for mk, s in all_results.items():
        if "error" in s:
            continue
        ranked = sorted(s["ranked_loo"].items(), key=lambda x: -x[1])
        print(f"\n  {mk}:")
        for fam, loo in ranked:
            print(f"    {fam:12s}: LOO={loo:.4f}")

    # Spearman跨模型一致性
    print("\n  Spearman跨模型一致性:")
    model_names = [mk for mk in model_keys if "error" not in all_results.get(mk, {})]
    for i, mk1 in enumerate(model_names):
        for mk2 in model_names[i+1:]:
            r1 = all_results[mk1]["ranked_loo"]
            r2 = all_results[mk2]["ranked_loo"]
            common = [k for k in r1 if k in r2]
            if len(common) >= 3:
                from scipy.stats import spearmanr
                v1 = [r1[k] for k in common]
                v2 = [r2[k] for k in common]
                rho, p = spearmanr(v1, v2)
                print(f"    {mk1} vs {mk2}: rho={rho:.3f}, p={p:.4f}")

    final = {
        "timestamp": TIMESTAMP,
        "stage": "589-590",
        "title": "四模型家族预测泛化与跨家族迁移",
        "models": all_results,
    }

    out_path = OUTPUT_DIR / f"stage589_family_prediction_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
