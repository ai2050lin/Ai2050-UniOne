#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage497: 编码写入-擦除因果链实验
目标：追踪信息从"写入层"到"读出层"的完整因果链
方法：
  1. 选择已知的编码写入层（如多层义切换的敏感层）
  2. 在写入层消融后，追踪信息在后续层如何恢复/丢失
  3. 在读出层消融后，检查早期写入是否仍然存在
  4. 构建"写入→传播→读出"的因果链图谱
核心问题：信息是通过残差流"忠实传递"还是被中间层"重写"？
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

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


# 测试用例：已知的语言现象对
TEST_PAIRS = [
    {
        "id": "apple_fruit",
        "name": "apple=水果",
        "context": "我昨天买了一个",
        "target": "苹果",
        "alt_context": "我昨天买了一台",
        "alt_target": "苹果",  # 但此时应为"电脑"
    },
    {
        "id": "python_snake",
        "name": "python=蛇",
        "context": "动物园里有一条",
        "target": "蟒蛇",
        "alt_context": "程序员用",
        "alt_target": "Python",
    },
    {
        "id": "king_queen",
        "name": "king-queen关系",
        "context": "国王的妻子是",
        "target": "王后",
        "alt_context": "女王",
        "alt_target": "国王",
    },
    {
        "id": "day_night",
        "name": "day-night反义",
        "context": "白天的反义词是",
        "target": "夜晚",
        "alt_context": "夜晚的反义词是",
        "alt_target": "白天",
    },
    {
        "id": "cause_effect",
        "name": "因果链",
        "context": "因为下雨，地面",
        "target": "湿了",
        "alt_context": "因为没下雨，地面",
        "alt_target": "干的",
    },
]


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


def get_logit_diff(model, tokenizer, context1: str, context2: str, target_word: str) -> float:
    """
    计算两个context下对同一target的logit差异
    差异越大，说明模型对context的区分越强
    """
    device = next(model.parameters()).device

    def get_top_logit(ctx, target):
        encoded = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=64)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            logits = outputs.logits[0, -1, :].float()
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return 0.0
        return logits[target_ids[0]].item()

    logit1 = get_top_logit(context1, target_word)
    logit2 = get_top_logit(context2, target_word)
    return logit1 - logit2


def capture_residual_at_layer(model, tokenizer, text: str, layer_idx: int) -> np.ndarray:
    """捕获指定层之后的残差流向量"""
    device = next(model.parameters()).device
    layers = discover_layers(model)

    captured = [None]

    def hook_fn(module, input, output):
        # output是tuple: (hidden_states, ...) 对于标准transformer
        if isinstance(output, tuple):
            captured[0] = output[0].detach().float().cpu()
        else:
            captured[0] = output.detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        model(**encoded)

    handle.remove()
    return captured[0][0, -1, :].numpy()  # 最后一个token


def cascading_ablation(model, tokenizer, pair: Dict) -> Dict:
    """
    级联消融实验：
    1. 逐层消融，测量对语义区分能力的影响
    2. 识别"写入层"（消融后语义区分急剧下降的层）
    3. 识别"读出层"（消融后预测改变的层）
    4. 分析中间层是否"忠实传播"还是"重写"
    """
    layers = discover_layers(model)
    total_layers = len(layers)
    device = next(model.parameters()).device

    context1 = pair["context"]
    context2 = pair["alt_context"]
    target = pair["target"]

    # 基线：正常模型的语义区分能力
    baseline_diff = get_logit_diff(model, tokenizer, context1, context2, target)
    baseline_pred1, _ = get_top_prediction(model, tokenizer, context1)
    baseline_pred2, _ = get_top_prediction(model, tokenizer, context2)

    layer_impacts = []
    residual_cosine_chain = []

    # 捕获逐层残差流
    residual1_list = []
    residual2_list = []
    for layer_idx in range(total_layers):
        r1 = capture_residual_at_layer(model, tokenizer, context1, layer_idx)
        r2 = capture_residual_at_layer(model, tokenizer, context2, layer_idx)
        residual1_list.append(r1)
        residual2_list.append(r2)

    # 计算相邻层间残差流的余弦相似度变化
    for i in range(1, len(residual1_list)):
        # 残差流差异在层间的传播
        diff_i = residual1_list[i] - residual2_list[i]
        diff_prev = residual1_list[i-1] - residual2_list[i-1]
        cos = np.dot(diff_i, diff_prev) / (np.linalg.norm(diff_i) * np.linalg.norm(diff_prev) + 1e-9)
        residual_cosine_chain.append({
            "from_layer": i - 1,
            "to_layer": i,
            "cosine_preservation": round(cos, 4),
            "diff_norm": round(np.linalg.norm(diff_i), 4),
        })

    # 逐层消融（选代表性层）
    sample_layers = list(range(0, total_layers, max(1, total_layers // 10)))
    if total_layers - 1 not in sample_layers:
        sample_layers.append(total_layers - 1)

    for layer_idx in sample_layers:
        layer = layers[layer_idx]

        # 消融attention
        orig_attn = layer.self_attn
        layer.self_attn = ZeroModule(return_tuple=True)
        try:
            attn_diff = get_logit_diff(model, tokenizer, context1, context2, target)
            pred1, _ = get_top_prediction(model, tokenizer, context1)
            pred2, _ = get_top_prediction(model, tokenizer, context2)
        finally:
            layer.self_attn = orig_attn

        # 消融MLP
        orig_mlp = layer.mlp
        layer.mlp = ZeroModule(return_tuple=False)
        try:
            mlp_diff = get_logit_diff(model, tokenizer, context1, context2, target)
            pred_mlp1, _ = get_top_prediction(model, tokenizer, context1)
            pred_mlp2, _ = get_top_prediction(model, tokenizer, context2)
        finally:
            layer.mlp = orig_mlp

        layer_impacts.append({
            "layer": layer_idx,
            "attn_ablation_diff": round(attn_diff, 4),
            "mlp_ablation_diff": round(mlp_diff, 4),
            "attn_impact": round(abs(baseline_diff - attn_diff), 4),
            "mlp_impact": round(abs(baseline_diff - mlp_diff), 4),
            "attn_preserves_distinction": abs(attn_diff) > abs(baseline_diff) * 0.5,
            "mlp_preserves_distinction": abs(mlp_diff) > abs(baseline_diff) * 0.5,
        })

    # 识别写入层和读出层
    attn_impacts = [li["attn_impact"] for li in layer_impacts]
    mlp_impacts = [li["mlp_impact"] for li in layer_impacts]

    write_layer_attn = layer_impacts[max(range(len(attn_impacts)), key=lambda i: attn_impacts[i])]["layer"]
    write_layer_mlp = layer_impacts[max(range(len(mlp_impacts)), key=lambda i: mlp_impacts[i])]["layer"]

    # 检查残差流的忠实传播
    cos_values = [r["cosine_preservation"] for r in residual_cosine_chain]
    avg_preservation = np.mean(cos_values)
    is_faithful = avg_preservation > 0.7  # 高余弦相似度 = 忠实传播

    return {
        "pair_id": pair["id"],
        "pair_name": pair["name"],
        "baseline_semantic_distinction": round(baseline_diff, 4),
        "baseline_pred_context1": baseline_pred1,
        "baseline_pred_context2": baseline_pred2,
        "layer_impacts": layer_impacts,
        "residual_cosine_chain": residual_cosine_chain,
        "analysis": {
            "write_layer_attn": write_layer_attn,
            "write_layer_mlp": write_layer_mlp,
            "avg_residual_cosine_preservation": round(avg_preservation, 4),
            "propagation_mode": "faithful" if is_faithful else "rewritten",
            "faithful_proposition": (
                "残差流在中间层忠实传播语义区分信息" if is_faithful
                else "残差流在中间层被重写，语义区分信息不忠实传播"
            ),
        },
    }


def get_top_prediction(model, tokenizer, prompt: str, top_k: int = 5) -> tuple:
    """获取模型的top-k预测"""
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    top_words = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    return top_words[0].strip(), float(top_probs[0].item())


def run_experiment(model_name: str) -> Dict:
    """运行完整实验"""
    print(f"\n{'='*60}")
    print(f"Stage497: 编码写入-擦除因果链实验 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    layers = discover_layers(model)
    print(f"层数: {len(layers)}")

    pair_results = []
    for pair in TEST_PAIRS:
        print(f"\n--- {pair['name']} ---")
        result = cascading_ablation(model, tokenizer, pair)
        pair_results.append(result)
        analysis = result["analysis"]
        print(f"  语义区分基线: {result['baseline_semantic_distinction']:.4f}")
        print(f"  写入层(attn): L{analysis['write_layer_attn']}, (mlp): L{analysis['write_layer_mlp']}")
        print(f"  传播模式: {analysis['propagation_mode']} (余弦保持: {analysis['avg_residual_cosine_preservation']:.4f})")

    # 汇总
    faithful_count = sum(1 for r in pair_results if r["analysis"]["propagation_mode"] == "faithful")
    rewritten_count = len(pair_results) - faithful_count

    write_layers_attn = [r["analysis"]["write_layer_attn"] for r in pair_results]
    write_layers_mlp = [r["analysis"]["write_layer_mlp"] for r in pair_results]

    summary = {
        "stage": "stage497_encoding_write_erase_causal_chain",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_layers": len(layers),
        "pair_results": pair_results,
        "aggregate": {
            "faithful_propagation_count": faithful_count,
            "rewritten_propagation_count": rewritten_count,
            "avg_cosine_preservation": round(np.mean([r["analysis"]["avg_residual_cosine_preservation"] for r in pair_results]), 4),
            "write_layer_attn_range": [min(write_layers_attn), max(write_layers_attn)],
            "write_layer_mlp_range": [min(write_layers_mlp), max(write_layers_mlp)],
            "dominant_propagation_mode": "faithful" if faithful_count > rewritten_count else "rewritten",
        },
        "core_answer": (
            f"在{model_name}上，{len(pair_results)}个语言对中，"
            f"{faithful_count}个显示忠实传播，{rewritten_count}个显示重写传播。"
            f"平均余弦保持={np.mean([r['analysis']['avg_residual_cosine_preservation'] for r in pair_results]):.4f}。"
            f"{'残差流主要忠实传播语义信息' if faithful_count > rewritten_count else '残差流在传播过程中被显著重写'}。"
        ),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage497: 编码写入-擦除因果链实验")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage497_causal_chain_{time.strftime('%Y%m%d')}"
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
