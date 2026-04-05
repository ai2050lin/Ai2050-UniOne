#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage499: 跨token路由机制实验
目标：用必须跨越多个token搬运信息的任务，验证route_heads是否出现
设计原理：
  - Stage496的单token补全任务中route_heads从未成为主导
  - 原因：单token预测中attention头的作用被embedding层覆盖
  - 解决：设计需要"从句首搬信息到句尾"的任务
方法：
  1. 代词消解："张三喜欢苹果，___喜欢吃香蕉" → "他"
  2. 指代链："猫追老鼠，老鼠怕___" → "猫"  
  3. 属性传承："这朵花是红色的，颜色很___" → "鲜艳/深/好看"
  4. 数量推理："三只猫和两只狗，总共有___只动物" → "五"
  5. 逻辑推理："如果下雨就不出门，现在下雨了，所以___" → "不出门"
  对比实验：消融attention vs 消融MLP，看哪个对跨token搬运更关键
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
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
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
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
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
# 跨token搬运任务
# ============================================================

CROSS_TOKEN_TASKS = [
    {
        "id": "pronoun_resolve",
        "name": "代词消解",
        "description": "从句首主语搬运到句尾代词",
        "prompts": [
            {"text": "张三喜欢苹果，", "target": "他", "candidates": ["他", "她", "它", "我"]},
            {"text": "李四是个医生，", "target": "她", "candidates": ["她", "他", "它", "我"], "note": "李四可能男女皆可"},
            {"text": "小明有一只猫，", "target": "他", "candidates": ["他", "她", "它", "小"]},
            {"text": "中国是一个大国，", "target": "它", "candidates": ["它", "她", "他", "这"]},
            {"text": "妈妈做了好吃的，", "target": "她", "candidates": ["她", "他", "妈", "好"]},
        ],
    },
    {
        "id": "reference_chain",
        "name": "指代链追踪",
        "description": "跨实体关系追踪",
        "prompts": [
            {"text": "猫追老鼠，老鼠怕", "target": "猫", "candidates": ["猫", "狗", "鼠", "人"]},
            {"text": "老师教学生，学生尊敬", "target": "老师", "candidates": ["老师", "学生", "校长", "父母"]},
            {"text": "太阳照亮月亮，月亮反射", "target": "太阳", "candidates": ["太阳", "月亮", "星星", "光"]},
            {"text": "国王统治国家，人民服从", "target": "国王", "candidates": ["国王", "国家", "人民", "法律"]},
            {"text": "水蒸发变成云，云凝结变成", "target": "雨", "candidates": ["雨", "水", "雪", "冰"]},
        ],
    },
    {
        "id": "attribute_transfer",
        "name": "属性传承",
        "description": "从已知属性推导未知属性",
        "prompts": [
            {"text": "这朵花是红色的，颜色很", "target": "鲜艳", "candidates": ["鲜艳", "深", "好看", "淡"]},
            {"text": "这个苹果很甜，味道", "target": "不错", "candidates": ["不错", "很好", "很甜", "酸"]},
            {"text": "这条河很宽，水流", "target": "很急", "candidates": ["很急", "缓慢", "清澈", "浑浊"]},
            {"text": "这座山很高，山顶很", "target": "冷", "candidates": ["冷", "热", "美", "陡"]},
            {"text": "这辆车很快，速度", "target": "很快", "candidates": ["很快", "很慢", "一般", "适中"]},
        ],
    },
    {
        "id": "counting",
        "name": "数量推理",
        "description": "跨token数量计算",
        "prompts": [
            {"text": "三只猫和两只狗，总共有", "target": "五", "candidates": ["五", "三", "六", "两"]},
            {"text": "一个苹果加两个橘子，共有", "target": "三", "candidates": ["三", "两", "一", "四"]},
            {"text": "十减三等于", "target": "七", "candidates": ["七", "三", "十", "六"]},
            {"text": "五加五再减二，等于", "target": "八", "candidates": ["八", "五", "十", "七"]},
            {"text": "一年有十二个月，半年有", "target": "六", "candidates": ["六", "十二", "三", "四"]},
        ],
    },
    {
        "id": "logic_chain",
        "name": "逻辑推理链",
        "description": "多步逻辑推导",
        "prompts": [
            {"text": "如果下雨就不出门，现在下雨了，所以", "target": "不出门", "candidates": ["不出门", "出门", "下雨", "打伞"]},
            {"text": "所有鸟都会飞，企鹅是鸟，所以企鹅", "target": "会飞", "candidates": ["会飞", "不会飞", "会游", "会走"]},
            {"text": "高个子比矮个子高，小明比小红高，所以", "target": "小明", "candidates": ["小明", "小红", "高", "矮"]},
            {"text": "冰比水冷，水比空气冷，所以冰比空气", "target": "冷", "candidates": ["冷", "热", "暖", "凉"]},
            {"text": "猫吃鱼，鱼吃虾，虾吃水草，所以猫不吃", "target": "水草", "candidates": ["水草", "鱼", "虾", "肉"]},
        ],
    },
]


class ZeroModule(torch.nn.Module):
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


def get_target_prob(model, tokenizer, prompt: str, target: str) -> float:
    """获取模型对prompt后接target的概率"""
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1, :].float()
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return probs[target_ids[0]].item()


def ablate_and_measure(model, tokenizer, prompt: str, target: str,
                       layer_idx: int, component: str) -> Dict:
    """消融后测量目标概率变化"""
    layers = discover_layers(model)
    layer = layers[layer_idx]

    # 正常概率
    normal_prob = get_target_prob(model, tokenizer, prompt, target)

    if component == "attn":
        orig = layer.self_attn
        layer.self_attn = ZeroModule(return_tuple=True)
    else:
        orig = layer.mlp
        layer.mlp = ZeroModule(return_tuple=False)

    try:
        ablated_prob = get_target_prob(model, tokenizer, prompt, target)
    finally:
        if component == "attn":
            layer.self_attn = orig
        else:
            layer.mlp = orig

    return {
        "layer": layer_idx,
        "component": component,
        "normal_prob": round(normal_prob, 6),
        "ablated_prob": round(ablated_prob, 6),
        "impact": round(abs(normal_prob - ablated_prob), 6),
    }


def run_single_task(model, tokenizer, task: Dict) -> Dict:
    """对单个任务的所有prompt做层消融"""
    layers = discover_layers(model)
    total_layers = len(layers)

    # 选取关键层：early/mid/late
    sample_layers = sorted(set([
        0, max(1, total_layers // 6), max(2, total_layers // 3),
        total_layers // 2, total_layers * 2 // 3,
        total_layers * 5 // 6, total_layers - 1,
    ]))

    prompt_results = []
    for p in task["prompts"]:
        text = p["text"]
        target = p["target"]
        candidates = p["candidates"]

        # 获取正常预测
        normal_prob = get_target_prob(model, tokenizer, text, target)

        # 检查模型是否认为target合理
        all_probs = {}
        for c in candidates:
            all_probs[c] = get_target_prob(model, tokenizer, text, c)

        best_candidate = max(all_probs, key=all_probs.get)
        model_correct = (best_candidate == target)

        # 层消融
        layer_results = []
        for li in sample_layers:
            for comp in ["attn", "mlp"]:
                r = ablate_and_measure(model, tokenizer, text, target, li, comp)
                layer_results.append(r)

        prompt_results.append({
            "text": text,
            "target": target,
            "candidates": candidates,
            "model_correct": model_correct,
            "normal_target_prob": round(normal_prob, 6),
            "all_candidate_probs": {k: round(v, 6) for k, v in all_probs.items()},
            "layer_results": layer_results,
        })

    # 汇总分析
    # 对每个层，比较attn vs mlp的总impact
    layer_attn_impacts = {}
    layer_mlp_impacts = {}
    for li in sample_layers:
        attn_impacts = [pr["layer_results"][i]["impact"]
                        for i, pr in enumerate(prompt_results)
                        for lr in pr["layer_results"]
                        if lr["layer"] == li and lr["component"] == "attn"]
        mlp_impacts = [pr["layer_results"][i]["impact"]
                       for i, pr in enumerate(prompt_results)
                       for lr in pr["layer_results"]
                       if lr["layer"] == li and lr["component"] == "mlp"]
        layer_attn_impacts[li] = float(np.mean(attn_impacts)) if attn_impacts else 0
        layer_mlp_impacts[li] = float(np.mean(mlp_impacts)) if mlp_impacts else 0

    # 找出attn > mlp的层
    attn_dominant_layers = [li for li in sample_layers if layer_attn_impacts[li] > layer_mlp_impacts[li]]
    mlp_dominant_layers = [li for li in sample_layers if layer_mlp_impacts[li] > layer_attn_impacts[li]]

    # 整体判断
    avg_attn = float(np.mean(list(layer_attn_impacts.values())))
    avg_mlp = float(np.mean(list(layer_mlp_impacts.values())))

    if avg_attn > avg_mlp * 1.5:
        dominant = "route_heads"
    elif avg_mlp > avg_attn * 1.5:
        dominant = "write_neurons"
    else:
        dominant = "mixed"

    return {
        "task_id": task["id"],
        "task_name": task["name"],
        "description": task["description"],
        "prompt_count": len(task["prompts"]),
        "model_correct_count": sum(1 for pr in prompt_results if pr["model_correct"]),
        "layer_attn_impacts": {str(k): round(v, 6) for k, v in layer_attn_impacts.items()},
        "layer_mlp_impacts": {str(k): round(v, 6) for k, v in layer_mlp_impacts.items()},
        "attn_dominant_layers": attn_dominant_layers,
        "mlp_dominant_layers": mlp_dominant_layers,
        "avg_attn_impact": round(avg_attn, 6),
        "avg_mlp_impact": round(avg_mlp, 6),
        "attn_to_mlp_ratio": round(avg_attn / max(avg_mlp, 1e-9), 4),
        "dominant_mechanism": dominant,
        "prompt_results": prompt_results,
    }


def run_experiment(model_name: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Stage499: 跨token路由机制实验 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    layers = discover_layers(model)
    print(f"层数: {len(layers)}")

    all_results = {}
    for task in CROSS_TOKEN_TASKS:
        print(f"\n--- {task['name']} ---")
        result = run_single_task(model, tokenizer, task)
        all_results[task["id"]] = result
        print(f"  模型正确率: {result['model_correct_count']}/{result['prompt_count']}")
        print(f"  attn总impact: {result['avg_attn_impact']:.6f}, mlp总impact: {result['avg_mlp_impact']:.6f}")
        print(f"  attn:mlp比: {result['attn_to_mlp_ratio']:.2f}")
        print(f"  主导机制: {result['dominant_mechanism']}")
        if result['attn_dominant_layers']:
            print(f"  attn主导层: {result['attn_dominant_layers']}")

    # 全局分析
    route_heads_count = sum(1 for r in all_results.values() if r["dominant_mechanism"] == "route_heads")
    write_neurons_count = sum(1 for r in all_results.values() if r["dominant_mechanism"] == "write_neurons")
    mixed_count = sum(1 for r in all_results.values() if r["dominant_mechanism"] == "mixed")

    # 专门检查哪些层attn最关键
    attn_peak_layers = {}
    for tid, rd in all_results.items():
        if rd["layer_attn_impacts"]:
            peak_li = max(rd["layer_attn_impacts"], key=rd["layer_attn_impacts"].get)
            attn_peak_layers[tid] = int(peak_li)

    summary = {
        "stage": "stage499_cross_token_routing",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_layers": len(layers),
        "results": all_results,
        "aggregate": {
            "route_heads_dominant_count": route_heads_count,
            "write_neurons_dominant_count": write_neurons_count,
            "mixed_count": mixed_count,
            "total_tasks": len(CROSS_TOKEN_TASKS),
            "route_heads_ratio": route_heads_count / len(CROSS_TOKEN_TASKS),
            "avg_attn_to_mlp_ratio": round(float(np.mean([r["attn_to_mlp_ratio"] for r in all_results.values()])), 4),
            "attn_peak_layers_by_task": attn_peak_layers,
        },
        "core_answer": (
            f"在{model_name}上，{len(CROSS_TOKEN_TASKS)}个跨token任务中，"
            f"{route_heads_count}个以route_heads为主导，{write_neurons_count}个以write_neurons为主导，"
            f"{mixed_count}个为混合模式。"
            + (f"route_heads在跨token任务中显著出现（占比{route_heads_count/len(CROSS_TOKEN_TASKS):.0%}），"
               "验证了跨token信息搬运需要attention路由。" if route_heads_count > 0
               else "route_heads仍未成为主导，可能需要更长的距离搬运任务。")
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage499: 跨token路由机制实验")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage499_cross_token_routing_{time.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = run_experiment(args.model)
    summary["elapsed_seconds"] = round(time.time() - start, 1)

    out_path = output_dir / f"summary_{args.model}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果保存到: {out_path}")
    print(f"\n核心结论: {summary['core_answer']}")


if __name__ == "__main__":
    main()
