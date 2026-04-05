#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage502: 预测失败原因诊断——为什么P2/P3/P5不成立？

Stage501中3个预测被否决：
  P2: 家族内概念在深层更聚类 → 不支持
  P3: 反义词互相注意力更高 → 不支持
  P5: 因果方向引导效应方向 → 不支持

本实验设计4个诊断角度来理解失败原因：
  D1: 层间聚类稳定性——聚类是否在所有层保持而非只在深层变化
  D2: 语义对立的注意力模式——不仅是反义词，检查所有语义对立关系的注意力
  D3: 位置效应——注意力模式是否受位置（开头/中间/结尾）影响
  D4: 方向性替代编码——因果方向是否通过非注意力的方式编码
"""

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodel_language_shared import (
    discover_layers,
    encode_to_device,
    evenly_spaced_layers,
    free_model,
    get_model_device,
    load_model_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = PROJECT_ROOT / "tests" / "codex_temp"


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten().unsqueeze(0)
    b_flat = b.float().flatten().unsqueeze(0)
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat)
    return float(cos.item())


# ============================================================
# D1: 层间聚类稳定性
# ============================================================
def run_d1_layer_clustering_stability(model, tokenizer) -> Dict:
    """检查概念家族的聚类程度在不同层的变化模式"""
    families = {
        "animals": ["猫", "狗", "鸟", "鱼", "虎", "兔", "马", "牛"],
        "colors": ["红", "蓝", "绿", "黄", "白", "黑", "紫", "橙"],
        "numbers": ["一", "二", "三", "四", "五", "六", "七", "八"],
        "emotions": ["喜", "怒", "哀", "乐", "悲", "恐", "惊", "愁"],
    }
    out_family = ["山", "水", "风", "云", "石", "木", "火", "土"]

    layers = discover_layers(model)
    layer_indices = evenly_spaced_layers(model, count=8)
    device = get_model_device(model)

    results = {"family_intra_avg": {}, "family_inter_avg": {}, "outlier_distance": {}}

    with torch.inference_mode():
        for family_name, words in families.items():
            for li in layer_indices:
                # 重新获取每层的表示
                cur_family_vecs = []
                for w in words:
                    w_encoded = tokenizer(w, return_tensors="pt", add_special_tokens=False)
                    w_input = w_encoded["input_ids"].to(device)
                    w_out = model(input_ids=w_input, output_hidden_states=True)
                    w_hidden = w_out.hidden_states[li][0, -1].float()
                    cur_family_vecs.append(w_hidden)

                cur_out_vecs = []
                for w in out_family:
                    w_encoded = tokenizer(w, return_tensors="pt", add_special_tokens=False)
                    w_input = w_encoded["input_ids"].to(device)
                    w_out = model(input_ids=w_input, output_hidden_states=True)
                    w_hidden = w_out.hidden_states[li][0, -1].float()
                    cur_out_vecs.append(w_hidden)

                # 计算家族内平均余弦相似度
                intra_sims = []
                for i in range(len(cur_family_vecs)):
                    for j in range(i + 1, len(cur_family_vecs)):
                        intra_sims.append(cosine_sim(cur_family_vecs[i], cur_family_vecs[j]))

                # 计算家族间平均余弦相似度
                inter_sims = []
                for fv in cur_family_vecs[:4]:
                    for ov in cur_out_vecs[:4]:
                        inter_sims.append(cosine_sim(fv, ov))

                key = f"L{li}"
                if key not in results["family_intra_avg"]:
                    results["family_intra_avg"][key] = []
                    results["family_inter_avg"][key] = []
                results["family_intra_avg"][key].append(sum(intra_sims) / len(intra_sims) if intra_sims else 0)
                results["family_inter_avg"][key].append(sum(inter_sims) / len(inter_sims) if inter_sims else 0)

    # 汇总：计算层间聚类趋势
    layer_keys = sorted(results["family_intra_avg"].keys())
    trend = []
    for k in layer_keys:
        intra = sum(results["family_intra_avg"][k]) / len(results["family_intra_avg"][k])
        inter = sum(results["family_inter_avg"][k]) / len(results["family_inter_avg"][k])
        trend.append({"layer": k, "intra": intra, "inter": inter, "gap": intra - inter})

    # 判断趋势：gap是否随层数增大
    increasing = all(trend[i]["gap"] <= trend[i + 1]["gap"] + 0.001 for i in range(len(trend) - 1))
    decreasing = all(trend[i]["gap"] >= trend[i + 1]["gap"] - 0.001 for i in range(len(trend) - 1))

    return {
        "trend": trend,
        "gap_increasing": increasing,
        "gap_decreasing": decreasing,
        "diagnosis": (
            "深层聚类增强" if increasing else
            "浅层聚类更好" if decreasing else
            "聚类程度在各层波动，无明显单调趋势"
        ),
    }


# ============================================================
# D2: 语义对立注意力
# ============================================================
def run_d2_semantic_opposition_attention(model, tokenizer) -> Dict:
    """不仅测反义词，测所有类型的语义对立关系"""
    relations = {
        "antonym": [
            ("热", "冷"), ("大", "小"), ("高", "低"),
            ("快", "慢"), ("好", "坏"), ("新", "旧"),
        ],
        "part_whole": [
            ("树", "叶子"), ("车", "轮子"), ("房子", "门"),
            ("书", "页"), ("手", "手指"), ("花", "花瓣"),
        ],
        "cause_effect": [
            ("下雨", "地湿"), ("学习", "进步"), ("运动", "健康"),
            ("努力", "成功"), ("吃", "饱"), ("睡", "精神"),
        ],
        "category_instance": [
            ("水果", "苹果"), ("动物", "猫"), ("颜色", "红"),
            ("国家", "中国"), ("职业", "医生"), ("乐器", "钢琴"),
        ],
    }

    device = get_model_device(model)
    layers = discover_layers(model)
    layer_indices = evenly_spaced_layers(model, count=7)

    relation_attention = {rel_name: [] for rel_name in relations}
    same_relation_attention = {rel_name: [] for rel_name in relations}

    with torch.inference_mode():
        for rel_name, pairs in relations.items():
            for w1, w2 in pairs:
                # 构造上下文：把两个词放在同一句中
                contexts = [
                    f"{w1}和{w2}是不同的。",
                    f"这个{w1}和那个{w2}。",
                    f"{w1}？不，是{w2}。",
                ]
                for ctx in contexts:
                    encoded = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=64)
                    input_ids = encoded["input_ids"].to(device)
                    outputs = model(input_ids=input_ids, output_attentions=True)

                    for li in layer_indices:
                        attn = outputs.attentions[li][0]  # [num_heads, seq_len, seq_len]
                        w1_ids = tokenizer.encode(w1, add_special_tokens=False)
                        w2_ids = tokenizer.encode(w2, add_special_tokens=False)

                        # 简化：找注意力矩阵中对应位置的值
                        # 使用平均注意力
                        avg_attn = attn.mean(dim=0)  # [seq_len, seq_len]
                        if avg_attn.shape[0] > 2:
                            # 取中间位置到其他位置的注意力（近似语义注意力）
                            mid = avg_attn.shape[0] // 2
                            total_attn = avg_attn[mid].sum().item()
                            relation_attention[rel_name].append(total_attn)

    # 汇总
    summary = {}
    for rel_name, attns in relation_attention.items():
        if attns:
            summary[rel_name] = {
                "mean_attention": sum(attns) / len(attns),
                "max_attention": max(attns),
                "min_attention": min(attns),
                "sample_count": len(attns),
            }

    # 判断哪种关系注意力最强
    ranking = sorted(summary.items(), key=lambda x: x[1]["mean_attention"], reverse=True)
    return {
        "ranking": [(r[0], r[1]["mean_attention"]) for r in ranking],
        "diagnosis": (
            f"最强关系类型: {ranking[0][0] if ranking else 'N/A'}，"
            f"反义词排名第{[r[0] for r in ranking].index('antonym') + 1 if 'antonym' in [r[0] for r in ranking] else 'N/A'}"
            if ranking else "无数据"
        ),
    }


# ============================================================
# D3: 位置效应
# ============================================================
def run_d3_position_effect(model, tokenizer) -> Dict:
    """检查注意力模式是否受token位置影响"""
    test_words = ["猫", "狗", "苹果", "红色"]
    positions = ["prefix", "middle", "suffix"]

    device = get_model_device(model)
    layers = discover_layers(model)
    layer_indices = evenly_spaced_layers(model, count=7)

    position_data = {pos: defaultdict(list) for pos in positions}

    with torch.inference_mode():
        for word in test_words:
            templates = {
                "prefix": f"{word}是一个有趣的东西。",
                "middle": f"我说过{word}是一个有趣的东西。",
                "suffix": f"这是一个有趣的东西叫做{word}。",
            }
            for pos, text in templates.items():
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = encoded["input_ids"].to(device)
                outputs = model(input_ids=input_ids, output_attentions=True)

                for li in layer_indices:
                    attn = outputs.attentions[li][0].mean(dim=0)  # [seq, seq]
                    w_ids = tokenizer.encode(word, add_special_tokens=False)
                    total_len = attn.shape[0]

                    # 找word token的大致位置
                    w_positions = []
                    for tid in w_ids:
                        input_id_list = input_ids[0].tolist()
                        if tid in input_id_list:
                            idx = input_id_list.index(tid)
                            w_positions.append(idx)

                    if w_positions:
                        # word token接收到的注意力
                        recv_attn = sum(attn[:, wp].sum().item() for wp in w_positions)
                        position_data[pos][f"L{li}"].append(recv_attn)

    # 汇总
    summary = {}
    for pos in positions:
        pos_summary = {}
        for li_key in position_data[pos]:
            vals = position_data[pos][li_key]
            if vals:
                pos_summary[li_key] = sum(vals) / len(vals)
        summary[pos] = pos_summary

    return {
        "position_summary": summary,
        "diagnosis": "位置对注意力分布的影响将在数据分析中展示",
    }


# ============================================================
# D4: 因果方向替代编码
# ============================================================
def run_d4_causal_alternative_encoding(model, tokenizer) -> Dict:
    """因果方向是否通过MLP激活差而非注意力编码"""
    causal_pairs = [
        ("下雨", "地湿", "下雨导致地湿"),
        ("学习", "进步", "学习带来进步"),
        ("运动", "健康", "运动促进健康"),
        ("努力", "成功", "努力成就成功"),
    ]

    device = get_model_device(model)
    layers = discover_layers(model)
    layer_indices = evenly_spaced_layers(model, count=8)

    direction_signals = {"attn_diff": defaultdict(list), "mlp_activation_diff": defaultdict(list)}

    with torch.inference_mode():
        for cause, effect, sentence in causal_pairs:
            # 正向
            fwd_encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
            fwd_input = fwd_encoded["input_ids"].to(device)
            fwd_out = model(input_ids=fwd_input, output_attentions=True, output_hidden_states=True)

            # 反向
            rev_sentence = f"{effect}导致{cause}"  # 反转因果
            rev_encoded = tokenizer(rev_sentence, return_tensors="pt", truncation=True, max_length=64)
            rev_input = rev_encoded["input_ids"].to(device)
            rev_out = model(input_ids=rev_input, output_attentions=True, output_hidden_states=True)

            for li in layer_indices:
                # 注意力差异
                fwd_attn = fwd_out.attentions[li][0].mean(dim=0)  # [seq, seq]
                rev_attn = rev_out.attentions[li][0].mean(dim=0)
                attn_diff = float((fwd_attn - rev_attn).abs().mean().item())

                # 隐藏状态差异
                fwd_hidden = fwd_out.hidden_states[li][0, -1].float()
                rev_hidden = rev_out.hidden_states[li][0, -1].float()
                hidden_diff = float((fwd_hidden - rev_hidden).norm().item())

                direction_signals["attn_diff"][f"L{li}"].append(attn_diff)
                direction_signals["mlp_activation_diff"][f"L{li}"].append(hidden_diff)

    # 汇总
    attn_summary = {}
    hidden_summary = {}
    for li in direction_signals["attn_diff"]:
        attn_vals = direction_signals["attn_diff"][li]
        hidden_vals = direction_signals["mlp_activation_diff"][li]
        attn_summary[li] = sum(attn_vals) / len(attn_vals)
        hidden_summary[li] = sum(hidden_vals) / len(hidden_vals)

    # 判断哪种编码方式信号更强
    avg_attn = sum(attn_summary.values()) / len(attn_summary) if attn_summary else 0
    avg_hidden = sum(hidden_summary.values()) / len(hidden_summary) if hidden_summary else 0

    return {
        "attn_diff_by_layer": attn_summary,
        "hidden_diff_by_layer": hidden_summary,
        "avg_attn_signal": avg_attn,
        "avg_hidden_signal": avg_hidden,
        "dominant_encoding": "hidden_state" if avg_hidden > avg_attn else "attention",
        "diagnosis": (
            f"因果方向主要通过{'隐藏状态差异(MLP/残差)' if avg_hidden > avg_attn else '注意力模式'}编码，"
            f"而非{'注意力模式' if avg_hidden > avg_attn else '隐藏状态差异'}"
        ),
    }


# ============================================================
# 主函数
# ============================================================
def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"stage502_prediction_failure_diag_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage502] 模型: {model_key}")
    print(f"[Stage502] 输出目录: {out_dir}")

    model, tokenizer = load_model_bundle(model_key)
    n_layers = len(discover_layers(model))
    print(f"[Stage502] 层数: {n_layers}")

    try:
        # D1
        print("\n=== D1: 层间聚类稳定性 ===")
        d1 = run_d1_layer_clustering_stability(model, tokenizer)
        print(f"  诊断: {d1['diagnosis']}")
        for t in d1["trend"]:
            print(f"    {t['layer']}: intra={t['intra']:.4f} inter={t['inter']:.4f} gap={t['gap']:.4f}")

        # D2
        print("\n=== D2: 语义对立注意力 ===")
        d2 = run_d2_semantic_opposition_attention(model, tokenizer)
        print(f"  诊断: {d2['diagnosis']}")
        for name, score in d2["ranking"]:
            print(f"    {name}: {score:.4f}")

        # D3
        print("\n=== D3: 位置效应 ===")
        d3 = run_d3_position_effect(model, tokenizer)
        print(f"  诊断: {d3['diagnosis']}")

        # D4
        print("\n=== D4: 因果方向替代编码 ===")
        d4 = run_d4_causal_alternative_encoding(model, tokenizer)
        print(f"  诊断: {d4['diagnosis']}")
        print(f"  avg_attn_signal: {d4['avg_attn_signal']:.4f}")
        print(f"  avg_hidden_signal: {d4['avg_hidden_signal']:.4f}")

        # 综合诊断
        print("\n=== 综合诊断 ===")
        diagnoses = {
            "P2失败原因(深层聚类增强)": d1["diagnosis"],
            "P3失败原因(反义词互注意力)": d2["diagnosis"],
            "P5失败原因(因果方向引导)": d4["diagnosis"],
            "附加发现(位置效应)": d3["diagnosis"],
        }
        for q, a in diagnoses.items():
            print(f"  {q}: {a}")

        summary = {
            "model": model_key,
            "n_layers": n_layers,
            "timestamp": timestamp,
            "d1_clustering_stability": d1,
            "d2_semantic_opposition_attention": d2,
            "d3_position_effect": d3,
            "d4_causal_alternative_encoding": d4,
            "diagnoses": diagnoses,
        }
        (out_dir / f"summary_{model_key}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )
        print(f"\n[Stage502] 结果已保存到 {out_dir}")

    finally:
        free_model(model)


if __name__ == "__main__":
    main()
