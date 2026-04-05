#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage496: 跨任务控制变量稳定性实验
目标：用全新的语言任务验证4类控制变量(route_heads, write_neurons,
      mixed_binding_circuits, late_readout_amplifiers)是否重复出现
方法：
  1. 构造5个全新语言现象（反义词对、因果关系、条件推理、类比推理、否定句）
  2. 逐层消融attention head和MLP neuron
  3. 记录每层每个组件对每个任务的因果贡献
  4. 验证4类控制变量是否在新任务中稳定出现
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "codex"))
from qwen3_language_shared import (
    discover_layers,
    load_qwen3_model,
    load_qwen3_tokenizer,
    move_batch_to_model_device,
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
    remove_hooks,
)


DEEPSEEK_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)


def load_deepseek_model(*, prefer_cuda: bool = True):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    kwargs = {
        "pretrained_model_name_or_path": str(DEEPSEEK_MODEL_PATH),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cpu"
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(DEEPSEEK_MODEL_PATH), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ============================================================
# 5个全新语言现象
# ============================================================

NEW_PHENOMENA = [
    {
        "id": "antonym",
        "name": "反义词对",
        "description": "从正反义词中预测另一个",
        "prompts": [
            ("大", ["小", "巨", "多", "高"]),
            ("热", ["冷", "温", "暖", "凉"]),
            ("好", ["坏", "差", "优", "良"]),
            ("黑", ["白", "暗", "明", "灰"]),
            ("快", ["慢", "速", "迟", "急"]),
        ],
    },
    {
        "id": "causation",
        "name": "因果关系",
        "description": "从原因预测结果",
        "prompts": [
            ("因为下雨，所以地面", ["湿", "干", "滑", "潮"]),
            ("因为没吃饭，所以很", ["饿", "饱", "累", "困"]),
            ("用力推门，门就", ["开", "关", "坏", "动"]),
            ("太阳出来了，天变", ["亮", "暗", "晴", "热"]),
            ("下了雪，天气很", ["冷", "热", "暖", "凉"]),
        ],
    },
    {
        "id": "condition",
        "name": "条件推理",
        "description": "如果...就...",
        "prompts": [
            ("如果明天晴天，我就去", ["玩", "睡", "走", "跑"]),
            ("如果温度超过100度，水就会", ["开", "结", "冻", "沸"]),
            ("如果不吃饭，就会", ["饿", "饱", "瘦", "胖"]),
            ("如果是冬天，就会很", ["冷", "热", "暖", "凉"]),
            ("如果努力学习，就能考", ["好", "差", "高", "低"]),
        ],
    },
    {
        "id": "analogy",
        "name": "类比推理",
        "description": "A对B就像C对D",
        "prompts": [
            ("鸟会飞，鱼会", ["游", "飞", "走", "跑"]),
            ("猫吃鱼，狗吃", ["肉", "鱼", "草", "粮"]),
            ("医生治病，老师", ["教书", "治病", "开车", "做饭"]),
            ("树长在土里，鱼活在", ["水里", "土里", "天上", "山上"]),
            ("白天有太阳，晚上有", ["月亮", "太阳", "星星", "云"]),
        ],
    },
    {
        "id": "negation",
        "name": "否定句",
        "description": "不是A而是B",
        "prompts": [
            ("这不是猫，而是", ["狗", "猫", "鸟", "鱼"]),
            ("他不喜欢红色，喜欢", ["蓝色", "红色", "绿色", "白色"]),
            ("不是早上，而是", ["晚上", "早上", "中午", "下午"]),
            ("不是热的，而是", ["冷的", "热的", "温的", "凉的"]),
            ("不是大的，而是", ["小的", "大的", "高的", "远的"]),
        ],
    },
]


def get_target_prediction(
    model, tokenizer, prompt: str, candidates: List[str]
) -> Tuple[str, float]:
    """获取模型对prompt的预测及其概率"""
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1, :].float()

    # 获取每个候选词的概率
    probs = torch.softmax(logits, dim=-1)
    best_candidate = ""
    best_prob = 0.0
    for cand in candidates:
        token_ids = tokenizer.encode(cand, add_special_tokens=False)
        if token_ids:
            p = probs[token_ids[0]].item()
            if p > best_prob:
                best_prob = p
                best_candidate = cand
    return best_candidate, best_prob


class ZeroModule(torch.nn.Module):
    """零化模块：输出全零张量，兼容任意forward签名"""
    def __init__(self, return_tuple: bool = False):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, *args, **kwargs):
        hidden = None
        if args:
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        if hidden is not None:
            z = torch.zeros_like(hidden)
            return (z, None) if self.return_tuple else z
        return None


def ablate_layer_component(
    model, tokenizer, prompt: str, candidates: List[str],
    layer_idx: int, component: str  # "attn" or "mlp"
) -> Dict:
    """消融某一层的某个组件，测量对预测的影响"""
    layers = discover_layers(model)
    layer = layers[layer_idx]
    device = next(model.parameters()).device

    # 正常预测
    normal_pred, normal_prob = get_target_prediction(model, tokenizer, prompt, candidates)

    zero_mod = ZeroModule()

    # 消融后预测
    if component == "attn":
        original_attn = layer.self_attn
        layer.self_attn = ZeroModule(return_tuple=True)
    else:  # mlp
        original_mlp = layer.mlp
        layer.mlp = ZeroModule(return_tuple=False)

    try:
        ablated_pred, ablated_prob = get_target_prediction(model, tokenizer, prompt, candidates)
    finally:
        if component == "attn":
            layer.self_attn = original_attn
        else:
            layer.mlp = original_mlp

    impact = normal_prob - ablated_prob
    return {
        "layer": layer_idx,
        "component": component,
        "normal_pred": normal_pred,
        "normal_prob": round(normal_prob, 6),
        "ablated_pred": ablated_pred,
        "ablated_prob": round(ablated_prob, 6),
        "impact": round(impact, 6),
        "prediction_changed": normal_pred != ablated_pred,
    }


def classify_control_variable(results: List[Dict], total_layers: int) -> Dict:
    """
    根据消融结果分类控制变量类型
    返回：哪些层以head为主，哪些以neuron为主
    """
    head_impacts = []
    mlp_impacts = []
    for r in results:
        if r["component"] == "attn":
            head_impacts.append(abs(r["impact"]))
        else:
            mlp_impacts.append(abs(r["impact"]))

    if not head_impacts or not mlp_impacts:
        return {"type": "unclear", "reason": "无数据"}

    avg_head = np.mean(head_impacts)
    avg_mlp = np.mean(mlp_impacts)
    max_head_layer = results[max(range(len(head_impacts)), key=lambda i: head_impacts[i])]["layer"]
    max_mlp_layer = results[total_layers + max(range(len(mlp_impacts)), key=lambda i: mlp_impacts[i])]["layer"]

    # 判断是否晚层读出放大
    mid_layer = total_layers // 2
    late_layer_count = sum(1 for r in results if r["layer"] > mid_layer and abs(r["impact"]) > 0.01)

    # 判断控制变量类型
    if avg_head > avg_mlp * 2:
        var_type = "route_heads_dominant"
    elif avg_mlp > avg_head * 2:
        var_type = "write_neurons_dominant"
    elif late_layer_count > total_layers // 3:
        var_type = "late_readout_amplifiers"
    else:
        var_type = "mixed_binding_circuits"

    return {
        "type": var_type,
        "avg_head_impact": round(avg_head, 6),
        "avg_mlp_impact": round(avg_mlp, 6),
        "head_to_mlp_ratio": round(avg_head / max(avg_mlp, 1e-9), 4),
        "max_head_layer": max_head_layer,
        "max_mlp_layer": max_mlp_layer,
        "late_layer_active_count": late_layer_count,
    }


def run_experiment(model_name: str) -> Dict:
    """运行完整实验"""
    print(f"\n{'='*60}")
    print(f"Stage496: 跨任务控制变量稳定性实验 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    layers = discover_layers(model)
    total_layers = len(layers)
    print(f"层数: {total_layers}")

    # 为了效率，选取代表性层进行消融（early, mid, late各2层）
    layer_indices = sorted(set([
        0,
        max(1, total_layers // 6),
        max(2, total_layers // 3),
        total_layers // 2,
        total_layers * 2 // 3,
        total_layers * 5 // 6,
        total_layers - 1,
    ]))

    all_results = {}
    for phenomenon in NEW_PHENOMENA:
        print(f"\n--- {phenomenon['name']} ---")
        phenomenon_results = []

        for prompt, candidates in tqdm(phenomenon["prompts"], desc=phenomenon["name"]):
            normal_pred, normal_prob = get_target_prediction(model, tokenizer, prompt, candidates)
            prompt_results = []

            for layer_idx in layer_indices:
                for component in ["attn", "mlp"]:
                    result = ablate_layer_component(
                        model, tokenizer, prompt, candidates,
                        layer_idx, component
                    )
                    prompt_results.append(result)

            # 分类这个prompt的控制变量
            cv = classify_control_variable(prompt_results, len(layer_indices))
            prompt_summary = {
                "prompt": prompt,
                "candidates": candidates,
                "normal_pred": normal_pred,
                "normal_prob": round(normal_prob, 6),
                "control_variable_type": cv["type"],
                "control_variable_detail": cv,
                "layer_results": prompt_results,
            }
            phenomenon_results.append(prompt_summary)
            print(f"  '{prompt}' -> '{normal_pred}' ({normal_prob:.4f}) CV={cv['type']}")

        # 汇总这个现象的控制变量分布
        cv_counts = {}
        for pr in phenomenon_results:
            cv_type = pr["control_variable_type"]
            cv_counts[cv_type] = cv_counts.get(cv_type, 0) + 1

        dominant_cv = max(cv_counts, key=cv_counts.get) if cv_counts else "unclear"

        all_results[phenomenon["id"]] = {
            "name": phenomenon["name"],
            "description": phenomenon["description"],
            "prompt_summaries": phenomenon_results,
            "control_variable_distribution": cv_counts,
            "dominant_control_variable": dominant_cv,
        }
        print(f"  => 主导控制变量: {dominant_cv} (分布: {cv_counts})")

    # 全局统计
    global_cv_counts = {}
    for pheno_id, pheno_data in all_results.items():
        dcv = pheno_data["dominant_control_variable"]
        global_cv_counts[dcv] = global_cv_counts.get(dcv, 0) + 1

    # 验证4类控制变量是否都出现
    four_types = ["route_heads_dominant", "write_neurons_dominant",
                  "mixed_binding_circuits", "late_readout_amplifiers"]
    found_types = [t for t in four_types if t in global_cv_counts]
    missing_types = [t for t in four_types if t not in global_cv_counts]

    summary = {
        "stage": "stage496_cross_task_control_variable_stability",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_layers": total_layers,
        "layer_indices_tested": layer_indices,
        "phenomena_count": len(NEW_PHENOMENA),
        "results": all_results,
        "global_control_variable_distribution": global_cv_counts,
        "four_types_found": found_types,
        "four_types_missing": missing_types,
        "stability_score": len(found_types) / len(four_types),
        "core_answer": (
            f"在{model_name}上，{len(NEW_PHENOMENA)}个全新语言现象中，"
            f"发现了{len(found_types)}/4类控制变量：{found_types}。"
            f"{'缺失：' + str(missing_types) if missing_types else '四类全部出现，支持控制变量普遍性假说。'}"
        ),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage496: 跨任务控制变量稳定性实验")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage496_cross_task_cv_stability_{time.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = run_experiment(args.model)
    summary["elapsed_seconds"] = round(time.time() - start, 1)

    out_path = output_dir / f"summary_{args.model}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存到: {out_path}")
    print(f"\n核心结论: {summary['core_answer']}")


if __name__ == "__main__":
    main()
