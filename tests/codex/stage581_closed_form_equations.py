#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage581-584合并: 从工作变量到闭式方程
目标：参数化R_l(螺旋旋转)、探索f_l(信息写入)函数形式、编码预测、判伪预测
模型：Qwen3-4B
"""

from __future__ import annotations
import sys, time, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from qwen3_language_shared import (
    load_qwen3_model, discover_layers, qwen_hidden_dim,
    remove_hooks, move_batch_to_model_device
)


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def safe_tok(tokenizer, tid, maxlen=20):
    return repr(tokenizer.decode([tid]))[:maxlen].encode('ascii', errors='replace').decode('ascii')


def extract_all_layer_hidden(model, tokenizer, sentence):
    """提取所有层的hidden state"""
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return [h[0, -1, :].float().cpu() for h in out.hidden_states]


def fit_rotation(h_l, h_l1):
    """拟合从h_l到h_l1的旋转矩阵（最小二乘）"""
    # h_{l+1} = R @ h_l + f_l
    # 先消去范数差异，只拟合旋转部分
    h_l_n = h_l / h_l.norm()
    h_l1_n = h_l1 / h_l1.norm()

    # R = h_l1_n @ h_l_n^T (rank-1近似)
    # 更精确的做法：用多个hidden state来拟合
    # 但对于单个向量，旋转角度 = arccos(h_l · h_l1)
    cos_angle = torch.dot(h_l_n, h_l1_n).clamp(-1, 1)
    angle = torch.acos(cos_angle).item()

    # 残差（信息写入量）
    # f_l = h_{l+1} - R @ h_l
    # 对于单向量，简化为：
    delta = h_l1 - h_l
    delta_norm = delta.norm().item()

    return {
        "angle_deg": round(np.degrees(angle), 2),
        "cos_angle": round(cos_angle.item(), 6),
        "delta_norm": round(delta_norm, 4),
        "h_l_norm": round(h_l.norm().item(), 4),
        "h_l1_norm": round(h_l1.norm().item(), 4),
        "norm_growth": round(h_l1.norm().item() / max(h_l.norm().item(), 1e-8), 4),
    }


def batch_rotation_analysis(model, tokenizer, sentences):
    """对多个句子做旋转分析，拟合更稳健的旋转矩阵"""
    n_layers = len(discover_layers(model))

    results = {}
    for label, sent in sentences.items():
        hidden_states = extract_all_layer_hidden(model, tokenizer, sent)

        layer_data = []
        for l in range(n_layers - 1):
            r = fit_rotation(hidden_states[l], hidden_states[l + 1])
            r["layer"] = l
            layer_data.append(r)
        results[label] = layer_data

    return results


def predict_encoding(model, tokenizer, family_words, target_word):
    """
    用已知家族成员预测新词编码
    编码 = 全局名词骨干 + 家族骨干 + 独有残差
    """
    encodings = {}
    for w in family_words + [target_word]:
        enc = tokenizer(w, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        encodings[w] = out.hidden_states[-1][0, -1, :].float().cpu()

    # 家族骨干 = 已知成员的质心
    family_vectors = torch.stack([encodings[w] for w in family_words])
    family_centroid = family_vectors.mean(dim=0)

    # 预测编码 ≈ 家族骨干
    predicted = family_centroid
    actual = encodings[target_word]

    # 计算预测精度
    pred_cos = cos(predicted, actual)

    # 分析残差
    residual = actual - predicted
    residual_norm = residual.norm().item()
    actual_norm = actual.norm().item()

    return {
        "target_word": target_word,
        "pred_cos": round(pred_cos, 4),
        "residual_ratio": round(residual_norm / max(actual_norm, 1e-8), 4),
        "family_centroid_norm": round(family_centroid.norm().item(), 4),
    }


def generate_falsifiable_predictions(model, tokenizer):
    """生成可判伪的定量预测"""
    predictions = {}

    # 预测1：所有名词在L0的hidden state norm应该在相似范围内
    nouns = ["apple", "bank", "car", "dog", "tree", "river", "mountain", "city", "book", "music"]
    norms_l0 = []
    for w in nouns:
        enc = tokenizer(w, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        norms_l0.append(out.hidden_states[0][0, -1, :].float().cpu().norm().item())

    predictions["noun_l0_norm_cv"] = round(np.std(norms_l0) / max(np.mean(norms_l0), 1e-8), 4)
    predictions["noun_l0_norm_range"] = [round(min(norms_l0), 2), round(max(norms_l0), 2)]

    # 预测2：同家族名词的编码距离 < 跨家族
    fruits = ["apple", "banana", "orange", "grape", "mango"]
    animals = ["dog", "cat", "bird", "fish", "horse"]
    tools = ["hammer", "screwdriver", "wrench", "saw", "knife"]

    def avg_intra_cos(words):
        encs = []
        for w in words:
            enc = tokenizer(w, return_tensors="pt", truncation=True, max_length=64)
            enc = move_batch_to_model_device(model, enc)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            encs.append(out.hidden_states[-1][0, -1, :].float().cpu())
        cosines = []
        for i in range(len(encs)):
            for j in range(i+1, len(encs)):
                cosines.append(cos(encs[i], encs[j]))
        return np.mean(cosines)

    fruit_intra = avg_intra_cos(fruits)
    animal_intra = avg_intra_cos(animals)
    tool_intra = avg_intra_cos(tools)

    # 跨家族
    all_embs = []
    for w in fruits + animals + tools:
        enc = tokenizer(w, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        all_embs.append(out.hidden_states[-1][0, -1, :].float().cpu())

    cross_cos = []
    for i in range(len(fruits)):
        for j in range(len(fruits), len(fruits) + len(animals)):
            cross_cos.append(cos(all_embs[i], all_embs[j]))
        for j in range(len(fruits) + len(animals), len(all_embs)):
            cross_cos.append(cos(all_embs[i], all_embs[j]))
    cross_mean = np.mean(cross_cos)

    predictions["fruit_intra"] = round(fruit_intra, 4)
    predictions["animal_intra"] = round(animal_intra, 4)
    predictions["tool_intra"] = round(tool_intra, 4)
    predictions["cross_family"] = round(cross_mean, 4)
    predictions["intra_vs_cross_ratio"] = round(np.mean([fruit_intra, animal_intra, tool_intra]) / max(cross_mean, 1e-8), 4)

    # 预测3：旋转角度在不同句子间应该是稳定的（架构不变量）
    sentences = {
        "The cat": "The cat sat on the mat.",
        "The dog": "The dog ran in the park.",
        "She likes": "She likes to read books.",
        "He plays": "He plays guitar every day.",
    }
    angles = []
    for label, sent in sentences.items():
        hidden = extract_all_layer_hidden(model, tokenizer, sent)
        for l in range(len(hidden) - 1):
            r = fit_rotation(hidden[l], hidden[l + 1])
            angles.append(r["angle_deg"])
    predictions["rotation_angle_mean"] = round(np.mean(angles), 2)
    predictions["rotation_angle_std"] = round(np.std(angles), 2)

    return predictions


def main():
    print("=" * 70)
    print("stage581-584: 从工作变量到闭式方程")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    # ── 实验1: R_l旋转参数化 ─────────────────────
    print("\n[2] R_l螺旋旋转参数化（每层旋转角）:")
    sentences = {
        "apple": "She ate a red apple.",
        "bank": "The bank approved the loan.",
        "cat": "The cat sat on the mat.",
    }

    batch_results = batch_rotation_analysis(model, tokenizer, sentences)

    # 汇总统计
    all_angles = []
    all_norm_growths = []
    for label, data in batch_results.items():
        for d in data:
            all_angles.append(d["angle_deg"])
            all_norm_growths.append(d["norm_growth"])

    print(f"  旋转角统计: mean={np.mean(all_angles):.2f}°, std={np.std(all_angles):.2f}°, "
          f"min={min(all_angles):.2f}°, max={max(all_angles):.2f}°")
    print(f"  范数增长: mean={np.mean(all_norm_growths):.4f}, std={np.std(all_norm_growths):.4f}")

    print(f"\n  逐层旋转角(取apple数据):")
    for d in batch_results["apple"]:
        l = d["layer"]
        if l % max(1, n_layers // 12) == 0 or l == n_layers - 2:
            print(f"    L{l:2d}→L{l+1}: angle={d['angle_deg']:6.2f}°, "
                  f"cos={d['cos_angle']:.6f}, norm_growth={d['norm_growth']:.4f}")

    # U型分布检验
    early_angles = [d["angle_deg"] for d in batch_results["apple"][:n_layers//4]]
    mid_angles = [d["angle_deg"] for d in batch_results["apple"][n_layers//4:3*n_layers//4]]
    late_angles = [d["angle_deg"] for d in batch_results["apple"][3*n_layers//4:]]
    print(f"\n  U型分布检验:")
    print(f"    早层(0-25%): {np.mean(early_angles):.2f}°")
    print(f"    中层(25-75%): {np.mean(mid_angles):.2f}°")
    print(f"    晚层(75-100%): {np.mean(late_angles):.2f}°")

    # ── 实验2: f_l信息写入探索 ───────────────────
    print("\n[3] f_l信息写入量(delta_norm)分析:")
    for label, data in batch_results.items():
        deltas = [d["delta_norm"] for d in data]
        print(f"  {label}: mean_delta={np.mean(deltas):.4f}, "
              f"total_delta={sum(deltas):.2f}, ratio_to_final_norm={sum(deltas)/max(data[-1]['h_l1_norm'], 1):.4f}")

    # ── 实验3: 编码预测 ──────────────────────────
    print("\n[4] 编码预测(家族骨干预测新词编码):")
    families = {
        "水果": (["apple", "banana", "orange", "grape"], "mango"),
        "动物": (["dog", "cat", "bird", "fish"], "horse"),
        "工具": (["hammer", "screwdriver", "wrench", "saw"], "knife"),
        "天体": (["sun", "moon", "star", "planet"], "comet"),
        "国家": (["China", "France", "Japan", "Germany"], "Brazil"),
    }

    for family_name, (members, target) in families.items():
        r = predict_encoding(model, tokenizer, members, target)
        print(f"  {family_name}(用{len(members)}个成员预测'{target}'): "
              f"pred_cos={r['pred_cos']:.4f}, residual_ratio={r['residual_ratio']:.4f}")

    # ── 实验4: 可判伪预测 ────────────────────────
    print("\n[5] 可判伪预测:")
    preds = generate_falsifiable_predictions(model, tokenizer)
    for k, v in preds.items():
        print(f"  {k}: {v}")

    # 总验证
    print(f"\n  判伪总结:")
    if preds["intra_vs_cross_ratio"] > 1.0:
        print(f"  [验证通过] 同家族cos > 跨家族cos (ratio={preds['intra_vs_cross_ratio']:.2f})")
    else:
        print(f"  [验证失败] 跨家族cos > 同家族cos (ratio={preds['intra_vs_cross_ratio']:.2f})")

    if preds["rotation_angle_std"] < 5.0:
        print(f"  [验证通过] 旋转角度跨句子稳定(std={preds['rotation_angle_std']:.2f}°)")
    else:
        print(f"  [验证失败] 旋转角度跨句子不稳定(std={preds['rotation_angle_std']:.2f}°)")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
