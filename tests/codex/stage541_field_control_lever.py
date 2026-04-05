#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage541: 场控制杆实验验证
=============================
核心假说：绑定信息编码在hidden state的统计量中（均值、方差、熵），
         而非单个神经元中。

实验设计：
1. 对每种绑定（属性/关系/语法/联想），测量bound vs unbound hidden state的
   - 统计量差异：Δmean, Δstd, Δentropy, Δsparsity
2. 与单神经元top-k差异对比：统计量差异 vs top-k神经元差异
3. 如果统计量差异 > top-k差异，则支持"场"假说
4. 如果统计量差异 < top-k差异，则支持"点"假说

注意：两个模型顺序测试，避免GPU溢出。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen3_language_shared import get_model_device, discover_layers
from multimodel_language_shared import (
    load_qwen3_model, load_deepseek_model,
    encode_to_device, evenly_spaced_layers,
    free_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage541_field_control_lever_20260404"

BINDING_PAIRS = {
    "attribute": [
        ("the red apple", "the apple"),
        ("the bright sun", "the sun"),
        ("the big cat", "the cat"),
    ],
    "relation": [
        ("apple is a fruit", "apple is"),
        ("cat is an animal", "cat is"),
        ("sun is a star", "sun is"),
    ],
    "grammar": [
        ("I ate an apple", "an apple was eaten"),
        ("the cat ran", "the cat was chased"),
    ],
    "association": [
        ("apple pie", "apple juice"),
        ("cat and dog", "cat and mouse"),
    ],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_hidden_state(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    encoded = encode_to_device(model, tokenizer, text)
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)
    return outputs.hidden_states[layer_idx + 1][0, -1, :].float()


def compute_statistics(h: torch.Tensor) -> Dict[str, float]:
    """计算hidden state的统计量"""
    mean = h.mean().item()
    std = h.std().item()
    # 熵 = -Σ p_i * log(p_i), 其中 p_i = softmax(|h_i| / ||h||)
    abs_h = h.abs()
    probs = abs_h / abs_h.sum().clamp(min=1e-10)
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum().item()
    # 稀疏度 = L0范数 / L1范数 (近似)
    l0 = (h.abs() > 0.01 * h.abs().max()).float().sum().item()
    sparsity = 1.0 - l0 / h.shape[0]
    # 能量 = ||h||^2
    energy = (h ** 2).sum().item()
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "entropy": round(entropy, 6),
        "sparsity": round(sparsity, 6),
        "energy": round(energy, 4),
    }


def compute_topk_delta(h_bound: torch.Tensor, h_unbound: torch.Tensor, k: int = 10) -> float:
    """top-k神经元的差异占总差异的比例"""
    delta = (h_bound - h_unbound).abs()
    total = delta.sum().item()
    if total < 1e-10:
        return 0.0
    topk = torch.topk(delta, min(k, len(delta))).values.sum().item()
    return topk / total


def compute_statistics_delta(
    stats_bound: Dict[str, float], stats_unbound: Dict[str, float]
) -> Dict[str, float]:
    return {
        f"delta_{k}": abs(stats_bound[k] - stats_unbound[k])
        for k in stats_bound
    }


def run_field_experiment(model, tokenizer, sample_layers, model_name):
    """运行场控制杆实验"""
    print(f"\n  === {model_name} 场控制杆实验 ===")

    results = {}
    for bind_type, pairs in BINDING_PAIRS.items():
        type_results = []
        for bound_text, unbound_text in pairs:
            pair_results = {}
            for li in sample_layers:
                h_bound = get_hidden_state(model, tokenizer, bound_text, li)
                h_unbound = get_hidden_state(model, tokenizer, unbound_text, li)

                stats_b = compute_statistics(h_bound)
                stats_u = compute_statistics(h_unbound)
                stat_delta = compute_statistics_delta(stats_b, stats_u)

                # top-k集中度
                topk1 = compute_topk_delta(h_bound, h_unbound, k=1)
                topk10 = compute_topk_delta(h_bound, h_unbound, k=10)
                topk100 = compute_topk_delta(h_bound, h_unbound, k=100)

                # L2总差异
                l2_total = torch.norm(h_bound - h_unbound).item()

                pair_results[str(li)] = {
                    "stat_delta": stat_delta,
                    "topk1_concentration": round(topk1, 6),
                    "topk10_concentration": round(topk10, 6),
                    "topk100_concentration": round(topk100, 6),
                    "l2_total": round(l2_total, 4),
                }

            type_results.append({
                "pair": (bound_text, unbound_text),
                "layer_results": pair_results,
            })
        results[bind_type] = type_results

    # 汇总：统计量变化 vs top-k集中度
    summary_data = {}
    for bind_type, type_results in results.items():
        stat_deltas = {"delta_mean": [], "delta_std": [], "delta_entropy": [], "delta_sparsity": []}
        topk_concs = {"topk1": [], "topk10": [], "topk100": []}

        for tr in type_results:
            for li_str, lr in tr["layer_results"].items():
                for k in stat_deltas:
                    stat_deltas[k].append(lr["stat_delta"][k])
                topk_concs["topk1"].append(lr["topk1_concentration"])
                topk_concs["topk10"].append(lr["topk10_concentration"])
                topk_concs["topk100"].append(lr["topk100_concentration"])

        avg_stat = {k: round(sum(v) / len(v), 6) for k, v in stat_deltas.items()}
        avg_topk = {k: round(sum(v) / len(v), 6) for k, v in topk_concs.items()}

        summary_data[bind_type] = {
            "avg_stat_delta": avg_stat,
            "avg_topk_concentration": avg_topk,
            "field_vs_point": (
                "FIELD" if avg_topk["topk100"] < 0.5 else "POINT"
            ),
        }

        print(f"    {bind_type}: "
              f"Δmean={avg_stat['delta_mean']:.4f} "
              f"Δstd={avg_stat['delta_std']:.4f} "
              f"Δentropy={avg_stat['delta_entropy']:.4f} | "
              f"top1={avg_topk['topk1']:.4f} "
              f"top10={avg_topk['topk10']:.4f} "
              f"top100={avg_topk['topk100']:.4f} "
              f"→ {summary_data[bind_type]['field_vs_point']}")

    return {"model": model_name, "summary": summary_data, "detailed": results}


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print("=" * 70)
    print("stage541: 场控制杆实验验证")
    print("=" * 70)
    started = time.time()

    # Qwen3
    print("\n[1/2] Qwen3...")
    model_q, tok_q = load_qwen3_model(prefer_cuda=True)
    sl_q = evenly_spaced_layers(model_q, count=7)
    result_q = run_field_experiment(model_q, tok_q, sl_q, "Qwen3-4B")
    free_model(model_q)

    # DeepSeek7B
    print("\n[2/2] DeepSeek7B...")
    model_d, tok_d = load_deepseek_model(prefer_cuda=True)
    sl_d = evenly_spaced_layers(model_d, count=7)
    result_d = run_field_experiment(model_d, tok_d, sl_d, "DeepSeek-R1-Distill-Qwen-7B")
    free_model(model_d)

    # 综合判定
    print("\n[综合判定]")
    field_count = 0
    point_count = 0
    for model_result in [result_q, result_d]:
        for bt, sd in model_result["summary"].items():
            if sd["field_vs_point"] == "FIELD":
                field_count += 1
            else:
                point_count += 1

    verdict = "FIELD（场控制杆）" if field_count > point_count else "POINT（点控制杆）"
    print(f"  场判定: {field_count}/8, 点判定: {point_count}/8 → {verdict}")

    elapsed = time.time() - started

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage541_field_control_lever",
        "title": "场控制杆实验验证",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 3),
        "qwen3_summary": result_q["summary"],
        "deepseek7b_summary": result_d["summary"],
        "verdict": verdict,
        "field_count": field_count,
        "point_count": point_count,
        "core_answer": (
            f"场控制杆验证结果：{field_count}/8组合判定为FIELD，{point_count}/8判定为POINT。\n"
            "判定标准：如果top-100神经元承载了<50%的绑定差异信息，则判定为场。\n"
            "关键数据：top-1/10/100神经元的集中度比例可以量化信息分布的集中程度。"
        ),
    }

    out_path = OUTPUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# stage541: 场控制杆实验验证\n",
        "## 判定\n",
        f"**{verdict}** ({field_count}场 vs {point_count}点)\n",
        "## 判定标准\n",
        "- top-100神经元承载<50%绑定差异 → FIELD\n",
        "- top-100神经元承载≥50%绑定差异 → POINT\n",
        "## 详细数据\n",
    ]
    for model_name in ["Qwen3-4B", "DeepSeek-R1-Distill-Qwen-7B"]:
        mr = result_q if "Qwen3" in model_name else result_d
        report.append(f"\n### {model_name}\n")
        report.append("| 类型 | Δmean | Δstd | Δentropy | top1 | top10 | top100 | 判定 |")
        report.append("|------|-------|------|----------|------|-------|--------|------|")
        for bt, sd in mr["summary"].items():
            s = sd["avg_stat_delta"]
            t = sd["avg_topk_concentration"]
            report.append(
                f"| {bt} | {s['delta_mean']:.4f} | {s['delta_std']:.4f} | "
                f"{s['delta_entropy']:.4f} | {t['topk1']:.4f} | {t['topk10']:.4f} | "
                f"{t['topk100']:.4f} | {sd['field_vs_point']} |"
            )

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n结果: {out_path}")
    print(f"总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
