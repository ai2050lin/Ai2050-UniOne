#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage643: 子空间恢复实验——用低维子空间替代单一方向恢复推理能力

核心假说：Stage641证明推理能力不能用单一delta_l恢复(3/4模型<4%)，
因为推理是分布式编码。但如果推理信息集中在低维子空间中，
则用子空间投影应该能显著提高恢复率。

方法：
1. 对每个推理case，收集多组prompt对的hidden state差异（A版本-B版本）
2. 用PCA提取这些差异向量的主成分，构成"推理子空间"
3. 在最伤层消融后，注入子空间投影（而非单一方向）
4. 比较子空间注入 vs 单一方向注入的恢复率

预注册判伪条件：
  如果子空间注入的recovery_ratio不超过单一方向的2倍，
  则"推理编码在低维子空间中"(INV-199)被推翻——说明推理信息分散在整个高维空间。

用法：
  python stage643_subspace_recovery.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
SUBSPACE_DIMS = [3, 5, 10, 20]


@dataclass(frozen=True)
class ReasoningCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str
    # 额外prompt对用于构建子空间
    extra_pairs: Sequence[tuple]


CASES: List[ReasoningCase] = [
    ReasoningCase(
        capability="syllogism",
        pair_id="barbara",
        prompt_a="All mammals are animals. All dogs are mammals. Therefore, all dogs are",
        positive_a=" animals",
        negative_a=" birds",
        prompt_b="All birds are animals. All sparrows are birds. Therefore, all sparrows are",
        positive_b=" animals",
        negative_b=" mammals",
        extra_pairs=[
            ("All cats are mammals. All mammals are animals. Therefore, all cats are", " animals"),
            ("All fish are vertebrates. All sharks are fish. Therefore, all sharks are", " vertebrates"),
            ("All roses are flowers. All red roses are roses. Therefore, all red roses are", " flowers"),
        ],
    ),
    ReasoningCase(
        capability="arithmetic",
        pair_id="addition",
        prompt_a="What is 7 plus 8? The answer is",
        positive_a=" 15",
        negative_a=" 16",
        prompt_b="What is 9 plus 6? The answer is",
        positive_b=" 15",
        negative_b=" 14",
        extra_pairs=[
            ("What is 3 plus 4? The answer is", " 7"),
            ("What is 5 plus 9? The answer is", " 14"),
            ("What is 8 plus 3? The answer is", " 11"),
        ],
    ),
    ReasoningCase(
        capability="implication",
        pair_id="modus_ponens",
        prompt_a="If it rains, the ground gets wet. It rains. Therefore, the ground",
        positive_a=" gets wet",
        negative_a=" stays dry",
        prompt_b="If it snows, the roads get icy. It snows. Therefore, the roads",
        positive_b=" get icy",
        negative_b=" stay clear",
        extra_pairs=[
            ("If the switch is on, the light shines. The switch is on. Therefore, the light", " shines"),
            ("If you study hard, you pass the exam. You study hard. Therefore, you", " pass"),
        ],
    ),
]


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def case_margin(model, tokenizer, case) -> float:
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )
    return float((margin_a + margin_b) / 2.0)


def extract_hidden_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def build_subspace(model, tokenizer, case: ReasoningCase, layer_idx: int, max_dims: int = 20) -> torch.Tensor:
    """收集多个差异向量，用PCA构建子空间基"""
    diff_vectors = []

    # 主差异
    h_a = extract_hidden_last_token(model, tokenizer, case.prompt_a, layer_idx)
    h_b = extract_hidden_last_token(model, tokenizer, case.prompt_b, layer_idx)
    diff_vectors.append(h_a - h_b)

    # 额外差异
    for prompt, answer in case.extra_pairs:
        # 构造一个"伪B版本"：把答案替换成不同内容
        h_base = extract_hidden_last_token(model, tokenizer, prompt, layer_idx)
        # 用不同prompt制造差异
        diff_vectors.append(h_base - h_a)

    # 如果不够，添加主差异的小扰动
    while len(diff_vectors) < max_dims + 2:
        noise = torch.randn_like(diff_vectors[0]) * 0.01
        diff_vectors.append(diff_vectors[0] + noise)

    matrix = torch.stack(diff_vectors, dim=0)  # [N, D]
    # SVD提取主方向
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    return Vt[:max_dims]  # [max_dims, D] 子空间基


def find_best_damage(model, tokenizer, case, layer_indices):
    baseline = case_margin(model, tokenizer, case)
    best_drop = 0.0
    best_layer = layer_indices[0]
    best_comp = "attn"

    for layer_idx in layer_indices:
        for component in ("attn", "mlp"):
            layer, original = ablate_layer_component(model, layer_idx, component)
            try:
                ablated = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer, component, original)
            drop = baseline - ablated
            if drop > best_drop:
                best_drop = drop
                best_layer = layer_idx
                best_comp = component
    return baseline, best_drop, best_layer, best_comp


def subspace_injected_margin(model, tokenizer, case, layer_idx, component, subspace_basis, k, scale, sign=1.0):
    """在消融层注入k维子空间投影"""
    layers = discover_layers(model)
    layer = layers[layer_idx]
    device = get_model_device(model)
    basis_k = subspace_basis[:k].to(device)  # [k, D]

    def make_hook(target_pos):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
            else:
                hidden = output.clone()
            token_vec = hidden[0, target_pos, :].float()
            # 投影到子空间并加回
            proj = token_vec @ basis_k.T  # [k]
            reconstructed = (proj @ basis_k)  # [D]
            hidden[0, target_pos, :] = hidden[0, target_pos, :] + sign * scale * (reconstructed - token_vec.to(device))
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    def score_single(prompt, positive, negative, sgn):
        comp_layer, original = ablate_layer_component(model, layer_idx, component)
        try:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_pos = tokenizer(prompt + positive, add_special_tokens=False)["input_ids"]
            full_neg = tokenizer(prompt + negative, add_special_tokens=False)["input_ids"]
            if len(full_pos) <= len(prompt_ids) or len(full_neg) <= len(prompt_ids):
                return float("-inf")
            target_pos = max(len(prompt_ids) - 1, 0)
            h = make_hook(target_pos)
            handle = layer.register_forward_hook(h)
            try:
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([full_pos], dtype=torch.long, device=device)).logits[0].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                pos_score = sum(float(log_probs[p - 1, full_pos[p]].item()) for p in range(len(prompt_ids), len(full_pos)))

                logits2 = model(input_ids=torch.tensor([full_neg], dtype=torch.long, device=device)).logits[0].float()
                log_probs2 = torch.log_softmax(logits2, dim=-1)
                neg_score = sum(float(log_probs2[p - 1, full_neg[p]].item()) for p in range(len(prompt_ids), len(full_neg)))

                return (pos_score / max(len(full_pos) - len(prompt_ids), 1)) - (neg_score / max(len(full_neg) - len(prompt_ids), 1))
            finally:
                handle.remove()
        finally:
            restore_layer_component(comp_layer, component, original)

    margin_a = score_single(case.prompt_a, case.positive_a, case.negative_a, +1.0)
    margin_b = score_single(case.prompt_b, case.positive_b, case.negative_b, -1.0)
    return float((margin_a + margin_b) / 2.0)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage643_subspace_recovery.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage643_subspace_recovery_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage643] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        layer_indices = evenly_spaced_layers(model, count=5)
        print(f"[Stage643] 层索引: {layer_indices}")
        records: List[Dict] = []

        for i, case in enumerate(CASES):
            print(f"\n[Stage643] ({i+1}/{len(CASES)}) {case.capability}/{case.pair_id}")

            baseline, best_drop, best_layer, best_comp = find_best_damage(model, tokenizer, case, layer_indices)
            ablated_margin = baseline - best_drop
            print(f"  baseline={baseline:.4f}, ablated={ablated_margin:.4f}, drop={best_drop:.4f}")
            print(f"  best: layer={best_layer}, comp={best_comp}")

            # 构建子空间
            print(f"  构建子空间...")
            subspace = build_subspace(model, tokenizer, case, best_layer, max_dims=20)
            subspace_energy = []
            U, S, Vt = torch.linalg.svd(subspace)
            total_S = S.sum().item()
            cum = 0
            for k in range(1, min(21, len(S))):
                cum += S[k-1].item()
                subspace_energy.append({"k": k, "energy_pct": cum / total_S * 100})

            # 单一方向恢复（Stage641基线）
            single_dir = subspace[0] / (subspace[0].norm() + 1e-10)

            # 子空间恢复测试
            best_single = -1e9
            for scale in [0.25, 0.5, 1.0]:
                try:
                    m = subspace_injected_margin(model, tokenizer, case, best_layer, best_comp, subspace, 1, scale)
                    if m > best_single:
                        best_single = m
                except Exception:
                    pass

            single_recovery = (best_single - ablated_margin) / max(baseline - ablated_margin, 1e-8)

            # 子空间恢复
            subspace_results = []
            for k in SUBSPACE_DIMS:
                best_m = -1e9
                best_scale = 0.25
                for scale in [0.25, 0.5, 1.0]:
                    try:
                        m = subspace_injected_margin(model, tokenizer, case, best_layer, best_comp, subspace, k, scale)
                        if m > best_m:
                            best_m = m
                            best_scale = scale
                    except Exception:
                        pass
                recovery = (best_m - ablated_margin) / max(baseline - ablated_margin, 1e-8)
                subspace_results.append({
                    "k": k, "margin": best_m, "recovery_ratio": recovery, "best_scale": best_scale
                })
                print(f"  k={k}: recovery={recovery:.4f}")

            best_subspace = max(subspace_results, key=lambda x: x["recovery_ratio"])
            improvement = best_subspace["recovery_ratio"] / max(single_recovery, 1e-8)
            print(f"  single_recovery={single_recovery:.4f}, best_subspace_recovery={best_subspace['recovery_ratio']:.4f}, improvement={improvement:.1f}x")

            records.append({
                "capability": case.capability,
                "pair_id": case.pair_id,
                "baseline_margin": baseline,
                "ablated_margin": ablated_margin,
                "best_damage_layer": best_layer,
                "best_damage_component": best_comp,
                "single_recovery_ratio": single_recovery,
                "best_subspace_k": best_subspace["k"],
                "best_subspace_recovery": best_subspace["recovery_ratio"],
                "improvement_factor": improvement,
                "subspace_energy": subspace_energy,
                "subspace_results": subspace_results,
            })

        # 汇总
        avg_single = statistics.mean([r["single_recovery_ratio"] for r in records])
        avg_subspace = statistics.mean([r["best_subspace_recovery"] for r in records])
        avg_improvement = statistics.mean([r["improvement_factor"] for r in records])
        falsified = avg_improvement < 2.0

        print(f"\n[Stage643] 汇总:")
        print(f"  avg_single_recovery={avg_single:.4f}")
        print(f"  avg_subspace_recovery={avg_subspace:.4f}")
        print(f"  avg_improvement={avg_improvement:.1f}x")
        print(f"  预注册判伪: {'INV-199被推翻（子空间不优于方向）' if falsified else 'INV-199存活（子空间优于方向）'}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "avg_single_recovery": avg_single,
            "avg_subspace_recovery": avg_subspace,
            "avg_improvement": avg_improvement,
            "falsified": falsified,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
