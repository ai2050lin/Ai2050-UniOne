#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage655: P6 Gemma4专项深度解析 — GQA架构与方向耦合因果链

P5发现Gemma4三个异常：
1. 方向耦合：relation_capital与syntax_sv的cos=-0.930（几乎反平行）
2. 抑制最强：avg_inhibit_drop=-1.67（其他模型-0.34~-0.58）
3. Attn-MLP非等价：Δattn≠Δmlp（其余三模型完全等价）

本脚本在四模型上统一运行，验证：
实验1：GQA头数与方向耦合的关系 — GQA是否直接导致方向耦合？
实验2：跨能力方向空间分析 — 方向子空间的维度与结构
实验3：抑制层的精确位置与模式 — 哪些层在抑制，抑制幅度
实验4：Attn-MLP非等价性的量化 — Gemma4的协同增强 vs 其他模型

预注册判伪条件：
INV-231: "GQA模型(Gemma4)的方向耦合显著高于非GQA模型"
  如果DS7B(也是GQA)的方向耦合也高，则INV-231修正为"GQA+特定参数组合导致"
INV-232: "Attn-MLP非等价性与方向耦合正相关"
  如果非等价但方向正交，则INV-232被推翻
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


CASES: List[TestCase] = [
    TestCase("syllogism",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects"),
    TestCase("relation_capital",
             "Paris is the capital of France. The capital of France is",
             " Paris", " Berlin",
             "Berlin is the capital of Germany. The capital of Germany is",
             " Berlin", " Paris"),
    TestCase("arithmetic",
             "If x = 7 and y = 3, then x + y =",
             " 10", " 11",
             "If x = 15 and y = 8, then x + y =",
             " 23", " 22"),
    TestCase("syntax_sv",
             "The key to the cabinet", " is", " are",
             "The keys to the cabinet", " are", " is"),
    # 新增两个case增强方向空间分析
    TestCase("coreference",
             "Mary gave the book to John because she",
             " wanted", " wanted",
             "Tom gave the book to Mary because he",
             " wanted", " wanted",
             ),
    TestCase("style_formal",
             "The results demonstrate that", " therefore", " so",
             "The data show that", " thus", " so"),
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


def get_gqa_info(model, model_key: str) -> Dict:
    """获取GQA相关信息"""
    info = {
        "model_key": model_key,
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "is_gqa": False,
        "gqa_ratio": None,
    }
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return info
        num_attn = getattr(cfg, "num_attention_heads", None)
        num_kv = getattr(cfg, "num_key_value_heads", None)
        if num_attn is None:
            num_attn = getattr(cfg, "n_head", None)
        if num_kv is None:
            num_kv = getattr(cfg, "n_head", None)
        info["num_attention_heads"] = num_attn
        info["num_key_value_heads"] = num_kv
        if num_attn and num_kv and num_attn != num_kv:
            info["is_gqa"] = True
            info["gqa_ratio"] = num_attn / num_kv
    except Exception:
        pass
    return info


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage655_gemma4_deep_analysis.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage655_gemma4_deep_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage655] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage655] layers={num_layers}")

        # ========================================
        # 基础架构信息
        # ========================================
        gqa_info = get_gqa_info(model, model_key)
        print(f"\n{'='*60}")
        print(f"Architecture: {gqa_info}")
        print(f"{'='*60}")

        # ========================================
        # 实验1: GQA与方向耦合关系
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Cross-capability direction coupling analysis")
        print(f"{'='*60}")

        final_layer = num_layers - 1
        # 用4个核心case计算方向
        core_cases = CASES[:4]
        deltas = {}
        for c in core_cases:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            deltas[c.name] = ha - hb

        names = list(deltas.keys())
        cos_matrix = []
        cos_pairs = []
        for i, n1 in enumerate(names):
            row = []
            for j, n2 in enumerate(names):
                cos_val = torch.dot(deltas[n1], deltas[n2]) / (deltas[n1].norm() * deltas[n2].norm() + 1e-10)
                row.append(round(cos_val.item(), 4))
                if i < j:
                    cos_pairs.append({"pair": f"{n1}<->{n2}", "cos": round(cos_val.item(), 4)})
            cos_matrix.append(row)

        print("  cos matrix (4 core cases):")
        for i, n1 in enumerate(names):
            print(f"    {n1}: {cos_matrix[i]}")

        off_diag = [cos_matrix[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
        mean_abs_cos = statistics.mean([abs(c) for c in off_diag])
        max_abs_cos = max(abs(c) for c in off_diag)
        max_coupling_pair = max(cos_pairs, key=lambda x: abs(x["cos"]))

        # 分析方向子空间维度
        delta_matrix = torch.stack([deltas[n] for n in names])  # [4, hidden_dim]
        U, S, Vt = torch.linalg.svd(delta_matrix.float(), full_matrices=False)
        effective_dims = []
        for thresh in [0.9, 0.95, 0.99]:
            cumsum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            k = (cumsum >= thresh).nonzero()
            k = k[0].item() + 1 if len(k) > 0 else len(S)
            effective_dims.append({"threshold": thresh, "dims": k})

        print(f"\n  Direction subspace effective dims:")
        for ed in effective_dims:
            print(f"    {ed['threshold']*100:.0f}% variance: {ed['dims']} dims")
        print(f"  Singular values: {[round(s.item(), 4) for s in S[:6]]}")
        print(f"  mean|cos| (off-diag): {mean_abs_cos:.4f}")
        print(f"  max|cos| (off-diag): {max_abs_cos:.4f}")
        print(f"  Max coupling pair: {max_coupling_pair['pair']}, cos={max_coupling_pair['cos']}")

        # ========================================
        # 实验2: 扩展6-case方向空间
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Extended 6-case direction space")
        print(f"{'='*60}")

        deltas6 = {}
        for c in CASES:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            deltas6[c.name] = ha - hb

        names6 = list(deltas6.keys())
        delta_matrix6 = torch.stack([deltas6[n] for n in names6])
        U6, S6, Vt6 = torch.linalg.svd(delta_matrix6.float(), full_matrices=False)

        effective_dims6 = []
        for thresh in [0.9, 0.95, 0.99]:
            cumsum = torch.cumsum(S6 ** 2, dim=0) / (S6 ** 2).sum()
            k = (cumsum >= thresh).nonzero()
            k = k[0].item() + 1 if len(k) > 0 else len(S6)
            effective_dims6.append({"threshold": thresh, "dims": k})

        # 6-case cos matrix
        cos_matrix6 = []
        for i, n1 in enumerate(names6):
            row = []
            for j, n2 in enumerate(names6):
                cos_val = torch.dot(deltas6[n1], deltas6[n2]) / (deltas6[n1].norm() * deltas6[n2].norm() + 1e-10)
                row.append(round(cos_val.item(), 4))
            cos_matrix6.append(row)

        off_diag6 = [cos_matrix6[i][j] for i in range(len(names6)) for j in range(len(names6)) if i != j]
        mean_abs_cos6 = statistics.mean([abs(c) for c in off_diag6])
        max_abs_cos6 = max(abs(c) for c in off_diag6)

        print(f"  6-case effective dims:")
        for ed in effective_dims6:
            print(f"    {ed['threshold']*100:.0f}%: {ed['dims']} dims")
        print(f"  6-case singular values: {[round(s.item(), 4) for s in S6[:8]]}")
        print(f"  6-case mean|cos|: {mean_abs_cos6:.4f}")
        print(f"  6-case max|cos|: {max_abs_cos6:.4f}")

        # INV-231判定
        inv231_status = "CONFIRMED" if (max_abs_cos > 0.5 and gqa_info["is_gqa"]) else \
                        ("GQA_NOT_COUPLING" if (max_abs_cos < 0.3 and gqa_info["is_gqa"]) else \
                         "NON_GQA_HIGH_COUPLING" if (max_abs_cos > 0.5 and not gqa_info["is_gqa"]) else "LOW_COUPLING")
        print(f"  INV-231 (GQA→coupling): {inv231_status}")

        # ========================================
        # 实验3: 抑制层精确位置与模式
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Inhibition layer precise mapping")
        print(f"{'='*60}")

        # 对4个核心case逐层消融MLP
        test_layer_count = min(10, num_layers)
        test_layers = sorted(set([round(i * (num_layers - 1) / (test_layer_count - 1))
                                   for i in range(test_layer_count)]))

        inhibition_map = {}
        for c in core_cases:
            baseline = case_margin(model, tokenizer, c)
            layer_effects = []
            for li in test_layers:
                layer_obj, orig = ablate_layer_component(model, li, "mlp")
                try:
                    abl_margin = case_margin(model, tokenizer, c)
                finally:
                    restore_layer_component(layer_obj, "mlp", orig)
                drop = baseline - abl_margin
                layer_effects.append({"layer": li, "drop": round(drop, 4)})
            inhibition_map[c.name] = {
                "baseline": round(baseline, 4),
                "layer_effects": layer_effects,
                "inhibit_layers": [le["layer"] for le in layer_effects if le["drop"] < -0.05],
                "promote_layers": [le["layer"] for le in layer_effects if le["drop"] > 0.05],
            }
            inh_count = len(inhibition_map[c.name]["inhibit_layers"])
            print(f"  {c.name}: baseline={baseline:.4f}, inhibit_layers={inh_count}/{len(test_layers)}, "
                  f"layers={[le['layer'] for le in layer_effects if le['drop'] < -0.05]}")

        # 总体抑制统计
        total_test_points = len(core_cases) * len(test_layers)
        total_inhibit = sum(len(im["inhibit_layers"]) for im in inhibition_map.values())
        inhibit_rate = total_inhibit / total_test_points if total_test_points > 0 else 0

        # 找出最强的抑制层
        all_inhibitions = []
        for cname, im in inhibition_map.items():
            for le in im["layer_effects"]:
                if le["drop"] < -0.05:
                    all_inhibitions.append({"case": cname, "layer": le["layer"], "drop": le["drop"]})
        all_inhibitions.sort(key=lambda x: x["drop"])

        print(f"\n  Overall inhibition rate: {inhibit_rate:.2%} ({total_inhibit}/{total_test_points})")
        if all_inhibitions:
            print(f"  Strongest inhibitions:")
            for ai in all_inhibitions[:5]:
                print(f"    {ai['case']} L{ai['layer']}: drop={ai['drop']:.4f}")

        # ========================================
        # 实验4: Attn-MLP非等价性量化
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Attn-MLP non-equivalence quantification")
        print(f"{'='*60}")

        case = core_cases[0]  # syllogism
        baseline = case_margin(model, tokenizer, case)
        attn_mlp_equivalence = []

        # 选取3个均匀分布的层
        sample_layers = [0, num_layers // 2, num_layers - 1]
        if num_layers > 6:
            sample_layers = sorted(set([round(i * (num_layers - 1) / 2) for i in range(3)]))

        for li in sample_layers:
            # 零化Attn
            layer_obj_attn, orig_attn = ablate_layer_component(model, li, "attn")
            try:
                margin_no_attn = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer_obj_attn, "attn", orig_attn)

            # 零化MLP
            layer_obj_mlp, orig_mlp = ablate_layer_component(model, li, "mlp")
            try:
                margin_no_mlp = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer_obj_mlp, "mlp", orig_mlp)

            # 同时零化
            layer_obj_attn2, orig_attn2 = ablate_layer_component(model, li, "attn")
            layer_obj_mlp2, orig_mlp2 = ablate_layer_component(model, li, "mlp")
            try:
                margin_no_both = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer_obj_mlp2, "mlp", orig_mlp2)
                restore_layer_component(layer_obj_attn2, "attn", orig_attn2)

            delta_attn = baseline - margin_no_attn
            delta_mlp = baseline - margin_no_mlp
            delta_both = baseline - margin_no_both
            compensation = delta_attn + delta_mlp - delta_both
            are_equal = abs(delta_attn - delta_mlp) < 0.01

            attn_mlp_equivalence.append({
                "layer": li,
                "delta_attn": round(delta_attn, 4),
                "delta_mlp": round(delta_mlp, 4),
                "delta_both": round(delta_both, 4),
                "compensation": round(compensation, 4),
                "are_equal": are_equal,
            })
            print(f"  L{li}: Δattn={delta_attn:.4f}, Δmlp={delta_mlp:.4f}, "
                  f"Δboth={delta_both:.4f}, comp={compensation:.4f}, equal={are_equal}")

        equal_count = sum(1 for e in attn_mlp_equivalence if e["are_equal"])
        mean_comp = statistics.mean([e["compensation"] for e in attn_mlp_equivalence])
        max_comp = max(abs(e["compensation"]) for e in attn_mlp_equivalence)

        inv232_status = "CONFIRMED" if (abs(mean_comp) > 0.02 and max_abs_cos > 0.3) else \
                        "FALSIFIED" if (abs(mean_comp) > 0.02 and max_abs_cos < 0.3) else \
                        "WEAK"
        print(f"\n  Attn=MLP layers: {equal_count}/{len(attn_mlp_equivalence)}")
        print(f"  Mean compensation: {mean_comp:.4f}")
        print(f"  Max |compensation|: {max_comp:.4f}")
        print(f"  INV-232 (Attn-MLP non-equiv vs coupling): {inv232_status}")

        # ========================================
        # 汇总
        # ========================================
        print(f"\n{'='*60}")
        print("Stage655 Summary:")
        print(f"{'='*60}")
        print(f"  GQA: is_gqa={gqa_info['is_gqa']}, ratio={gqa_info['gqa_ratio']}")
        print(f"  4-case max|cos|: {max_abs_cos:.4f}")
        print(f"  6-case max|cos|: {max_abs_cos6:.4f}")
        print(f"  6-case subspace dims: {[ed['dims'] for ed in effective_dims6]}")
        print(f"  Inhibition rate: {inhibit_rate:.2%}")
        print(f"  Attn=MLP layers: {equal_count}/{len(attn_mlp_equivalence)}")
        print(f"  INV-231: {inv231_status}")
        print(f"  INV-232: {inv232_status}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "gqa_info": gqa_info,
            "exp1_4case": {
                "cos_matrix": cos_matrix,
                "cos_matrix_names": names,
                "mean_abs_cos": round(mean_abs_cos, 4),
                "max_abs_cos": round(max_abs_cos, 4),
                "max_coupling_pair": max_coupling_pair,
                "effective_dims": effective_dims,
                "singular_values": [round(s, 4) for s in S[:10].tolist()],
            },
            "exp2_6case": {
                "cos_matrix": cos_matrix6,
                "cos_matrix_names": names6,
                "mean_abs_cos": round(mean_abs_cos6, 4),
                "max_abs_cos": round(max_abs_cos6, 4),
                "effective_dims": effective_dims6,
                "singular_values": [round(s, 4) for s in S6[:10].tolist()],
            },
            "exp3_inhibition": {
                "inhibition_map": inhibition_map,
                "inhibit_rate": round(inhibit_rate, 4),
                "total_inhibit": total_inhibit,
                "total_test_points": total_test_points,
                "strongest_inhibitions": all_inhibitions[:5],
            },
            "exp4_attn_mlp": {
                "equivalence": attn_mlp_equivalence,
                "equal_count": equal_count,
                "mean_compensation": round(mean_comp, 4),
                "max_compensation": round(max_comp, 4),
            },
            "invariants": {
                "inv231_status": inv231_status,
                "inv232_status": inv232_status,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
