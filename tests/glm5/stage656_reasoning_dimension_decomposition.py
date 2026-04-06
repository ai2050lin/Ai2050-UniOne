#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage656: P7 推理能力维度分解

P5发现：3/4模型的推理能力（syllogism）无法用单一方向完全恢复（direction injection恢复力<50%），
说明推理需要多个维度的协同。

本脚本在四模型上运行，分解推理能力的维度结构：
实验1：推理delta的SVD维度分析 — 推理能力编码在几个有效维度上？
实验2：逐层SVD维度追踪 — 推理维度的跨层演化（合并/分裂/稳定）
实验3：推理子空间 vs 消歧子空间 — 两种能力的子空间关系
实验4：方向注入恢复力 vs 维度数 — 注入多少个方向能达到X%恢复力？

预注册判伪条件：
INV-233: "推理能力编码在>2个有效维度上"
  如果90% variance只需要1个维度（rank-1），则INV-233被推翻
INV-234: "推理子空间与消歧子空间近似正交（mean|cos|<0.3）"
  如果子空间主方向cos>0.5，则INV-234被推翻
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    ablate_layer_component,
    discover_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


REASONING_CASES: List[TestCase] = [
    TestCase("syllogism",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects"),
    TestCase("syllogism_neg",
             "All fish are animals. No dogs are fish. Dogs are",
             " animals", " fish",
             "All roses are flowers. No trees are roses. Trees are",
             " flowers", " roses"),
    TestCase("chain_reasoning",
             "If A implies B and B implies C, then A implies",
             " C", " B",
             "If X implies Y and Y implies Z, then X implies",
             " Z", " Y"),
    TestCase("arithmetic_chain",
             "If x = 7, y = x + 3, and z = y + 5, then z =",
             " 15", " 13",
             "If a = 10, b = a + 2, and c = b + 3, then c =",
             " 15", " 13"),
]

DISAMBIG_CASES: List[TestCase] = [
    TestCase("bank_financial",
             "She went to the bank to deposit money. The bank",
             " teller", " river",
             "He walked along the river bank. The bank",
             " river", " teller"),
    TestCase("relation_capital",
             "Paris is the capital of France. The capital of France is",
             " Paris", " Berlin",
             "Berlin is the capital of Germany. The capital of Germany is",
             " Berlin", " Paris"),
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def extract_last_token_hidden(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    captured = {"value": None}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["value"] = output[0][:, -1, :].detach().cpu()
        else:
            captured["value"] = output[:, -1, :].detach().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"].squeeze(0)


def effective_rank_from_svd(S: torch.Tensor, threshold: float = 0.9) -> int:
    cumsum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    k = (cumsum >= threshold).nonzero()
    return k[0].item() + 1 if len(k) > 0 else len(S)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage656_reasoning_dimension_decomposition.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage656_reasoning_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage656] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage656] layers={num_layers}")

        # ========================================
        # 实验1: 推理delta的末层SVD维度分析
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Reasoning delta SVD dimensionality at final layer")
        print(f"{'='*60}")

        final_layer = num_layers - 1
        reasoning_deltas = {}
        for c in REASONING_CASES:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            reasoning_deltas[c.name] = ha - hb

        rnames = list(reasoning_deltas.keys())
        r_delta_matrix = torch.stack([reasoning_deltas[n] for n in rnames])
        U_r, S_r, Vt_r = torch.linalg.svd(r_delta_matrix.float(), full_matrices=False)

        r_eff_dims = []
        for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
            cumsum = torch.cumsum(S_r ** 2, dim=0) / (S_r ** 2).sum()
            k = (cumsum >= thresh).nonzero()
            k = k[0].item() + 1 if len(k) > 0 else len(S_r)
            r_eff_dims.append({"threshold": thresh, "dims": k})

        print(f"  Reasoning singular values: {[round(s.item(), 4) for s in S_r[:6]]}")
        for ed in r_eff_dims:
            print(f"    {ed['threshold']*100:.0f}% variance: {ed['dims']} dims")

        # 推理delta之间的cos矩阵
        r_cos_matrix = []
        for i, n1 in enumerate(rnames):
            row = []
            for j, n2 in enumerate(rnames):
                cos_val = torch.dot(reasoning_deltas[n1], reasoning_deltas[n2]) / \
                          (reasoning_deltas[n1].norm() * reasoning_deltas[n2].norm() + 1e-10)
                row.append(round(cos_val.item(), 4))
            r_cos_matrix.append(row)

        print("  Reasoning inter-case cos matrix:")
        for i, n1 in enumerate(rnames):
            print(f"    {n1}: {r_cos_matrix[i]}")

        off_diag_r = [r_cos_matrix[i][j] for i in range(len(rnames)) for j in range(len(rnames)) if i != j]
        mean_abs_cos_r = statistics.mean([abs(c) for c in off_diag_r]) if off_diag_r else 0
        print(f"  mean|cos| (inter-reasoning): {mean_abs_cos_r:.4f}")

        # INV-233判定
        rank90 = effective_rank_from_svd(S_r, 0.9)
        inv233_status = "CONFIRMED" if rank90 > 2 else ("PARTIAL" if rank90 > 1 else "FALSIFIED")
        print(f"  INV-233 (reasoning needs >2 dims): {inv233_status} (rank90={rank90})")

        # ========================================
        # 实验2: 逐层SVD维度追踪
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Cross-layer SVD dimension tracking")
        print(f"{'='*60}")

        # 采样6个均匀分布的层
        sample_count = min(6, num_layers)
        sample_layers = sorted(set([round(i * (num_layers - 1) / (sample_count - 1))
                                     for i in range(sample_count)]))

        layer_svd_info = []
        for li in sample_layers:
            ldeltas = []
            for c in REASONING_CASES:
                ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, li)
                hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, li)
                ldeltas.append((ha - hb).float())
            ldm = torch.stack(ldeltas)
            _, lS, _ = torch.linalg.svd(ldm, full_matrices=False)

            r90 = effective_rank_from_svd(lS, 0.9)
            r95 = effective_rank_from_svd(lS, 0.95)

            # 主方向稳定性（相邻层主方向的cos）
            layer_svd_info.append({
                "layer": li,
                "singular_values": [round(s, 4) for s in lS[:6].tolist()],
                "rank90": r90,
                "rank95": r95,
                "top1_fraction": round((lS[0] ** 2 / (lS ** 2).sum()).item(), 4),
            })
            print(f"  L{li}: rank90={r90}, rank95={r95}, "
                  f"top1_frac={layer_svd_info[-1]['top1_fraction']:.4f}, "
                  f"SV={[round(s.item(), 2) for s in lS[:4]]}")

        # 检查维度演化模式
        rank90_values = [lsi["rank90"] for lsi in layer_svd_info]
        rank_trend = "STABLE" if max(rank90_values) - min(rank90_values) <= 1 else \
                     "GROWING" if rank90_values[-1] > rank90_values[0] + 1 else "SHRINKING"
        print(f"  Dimension trend: {rank_trend} (rank90: {rank90_values})")

        # ========================================
        # 实验3: 推理子空间 vs 消歧子空间
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Reasoning subspace vs Disambiguation subspace")
        print(f"{'='*60}")

        disambig_deltas = {}
        for c in DISAMBIG_CASES:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            disambig_deltas[c.name] = ha - hb

        dnames = list(disambig_deltas.keys())
        d_delta_matrix = torch.stack([disambig_deltas[n] for n in dnames])
        U_d, S_d, Vt_d = torch.linalg.svd(d_delta_matrix.float(), full_matrices=False)

        # 子空间主方向对比
        reasoning_top1 = Vt_r[0]  # 推理第一主方向
        disambig_top1 = Vt_d[0]  # 消歧第一主方向

        sub_cos = torch.dot(reasoning_top1, disambig_top1).item()
        print(f"  Reasoning top1 dir norm: {reasoning_top1.norm().item():.4f}")
        print(f"  Disambig top1 dir norm: {disambig_top1.norm().item():.4f}")
        print(f"  cos(reasoning_top1, disambig_top1): {sub_cos:.4f}")

        # 扩展到前k个主方向
        k_compare = min(3, min(len(Vt_r), len(Vt_d)))
        cross_cos = []
        for i in range(k_compare):
            for j in range(k_compare):
                c = torch.dot(Vt_r[i], Vt_d[j]).item()
                cross_cos.append({"r_comp": i, "d_comp": j, "cos": round(c, 4)})
        cross_cos.sort(key=lambda x: abs(x["cos"]), reverse=True)

        print(f"  Top-{k_compare} cross-subspace cos:")
        for cc in cross_cos[:5]:
            print(f"    R{cc['r_comp']}<->D{cc['d_comp']}: cos={cc['cos']}")

        max_cross_cos = max(abs(cc["cos"]) for cc in cross_cos)
        mean_cross_cos = statistics.mean(abs(cc["cos"]) for cc in cross_cos)

        inv234_status = "SURVIVED" if mean_cross_cos < 0.3 else "FALSIFIED"
        print(f"  INV-234 (reasoning⊥disambig subspaces): {inv234_status}")
        print(f"  mean|cos|: {mean_cross_cos:.4f}, max|cos|: {max_cross_cos:.4f}")

        # ========================================
        # 实验4: 方向注入恢复力 vs 维度数
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Direction injection recovery vs dimension count")
        print(f"{'='*60}")

        # 方法：消融某一层MLP，然后注入前k个SVD主方向
        # 简化版本：直接测量消融后的恢复潜力
        # 用syllogism case在L_mid层消融MLP

        syl_case = REASONING_CASES[0]
        ablate_layer = num_layers // 2
        baseline_margin = case_margin(model, tokenizer, syl_case)

        # 消融L_mid MLP
        layer_obj, orig = ablate_layer_component(model, ablate_layer, "mlp")
        try:
            ablated_margin = case_margin(model, tokenizer, syl_case)
        finally:
            restore_layer_component(layer_obj, "mlp", orig)

        damage = baseline_margin - ablated_margin
        print(f"  baseline margin: {baseline_margin:.4f}")
        print(f"  ablated margin (L{ablate_layer} MLP): {ablated_margin:.4f}")
        print(f"  damage: {damage:.4f}")

        # 用多个推理case的delta构建"推理方向集"
        # 测量：注入前k个推理方向后，margin恢复多少
        # 简化方案：用final layer的推理delta矩阵的主方向做投影分析
        # 注入后margin = baseline + alpha * sum(cos(delta_case, injection_dir) * ||delta_case|| * ||injection_dir||)
        # 这里用隐藏态投影近似

        inject_results = []
        for k in [1, 2, 3, min(4, len(rnames))]:
            # 构造k维子空间
            V_k = Vt_r[:k].float()  # [k, hidden_dim]

            # 计算每个推理delta在k维子空间内的投影
            proj_fractions = []
            for rn in rnames:
                delta_vec = reasoning_deltas[rn].float()
                proj = sum(torch.dot(delta_vec, V_k[i]) * V_k[i] for i in range(k))
                frac = (proj.norm() / (delta_vec.norm() + 1e-10)).item()
                proj_fractions.append(round(frac, 4))

            mean_proj = statistics.mean(proj_fractions)
            inject_results.append({
                "k": k,
                "mean_projection_fraction": round(mean_proj, 4),
                "per_case_fractions": proj_fractions,
            })
            print(f"  k={k}: mean projection fraction={mean_proj:.4f} ({[f'{f:.3f}' for f in proj_fractions]})")

        # ========================================
        # 汇总
        # ========================================
        print(f"\n{'='*60}")
        print("Stage656 Summary:")
        print(f"{'='*60}")
        print(f"  Reasoning rank90: {rank90}")
        print(f"  Inter-reasoning mean|cos|: {mean_abs_cos_r:.4f}")
        print(f"  Dimension trend: {rank_trend}")
        print(f"  Cross-subspace mean|cos|: {mean_cross_cos:.4f}")
        print(f"  INV-233: {inv233_status}")
        print(f"  INV-234: {inv234_status}")
        print(f"  Ablation damage (L{ablate_layer}): {damage:.4f}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "exp1_reasoning_svd": {
                "cos_matrix": r_cos_matrix,
                "cos_matrix_names": rnames,
                "mean_abs_cos": round(mean_abs_cos_r, 4),
                "effective_dims": r_eff_dims,
                "singular_values": [round(s, 4) for s in S_r[:10].tolist()],
                "rank90": rank90,
            },
            "exp2_layer_tracking": {
                "layer_svd_info": layer_svd_info,
                "rank_trend": rank_trend,
                "rank90_values": rank90_values,
            },
            "exp3_cross_subspace": {
                "sub_cos_top1": round(sub_cos, 4),
                "cross_cos_top5": cross_cos[:5],
                "mean_cross_cos": round(mean_cross_cos, 4),
                "max_cross_cos": round(max_cross_cos, 4),
            },
            "exp4_injection_recovery": {
                "ablate_layer": ablate_layer,
                "baseline_margin": round(baseline_margin, 4),
                "ablated_margin": round(ablated_margin, 4),
                "damage": round(damage, 4),
                "inject_results": inject_results,
            },
            "invariants": {
                "inv233_status": inv233_status,
                "inv234_status": inv234_status,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
