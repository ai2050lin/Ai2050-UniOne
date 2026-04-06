#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage669: P22 五类能力×三模型不变量矩阵

目标：构建5能力×3模型的不变量检验矩阵，系统化判伪

不变量列表（来自P15-P20）：
- INV-285: 信号效率>50%（P17确认）
- INV-286: 抵消有序（z-score>1.96）（P17推翻→有序）
- INV-293: 抵消率30-90%（P19确认）
- INV-294: 跨能力干扰<30%（P20确认）
- INV-124: 方向正交化（低cos between capabilities）
- INV-269: 低维编码（PCA rank90 < 20）
- INV-273: 旋转角≈90°

判伪标准：
INV-299: ">=70%的不变量组合成立 → 统一理论有坚实基础"
INV-300: "存在模型×能力组合系统性推翻>=2个不变量 → 架构依赖显著"
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[TestCase] = [
    # 消歧 (3)
    TestCase(capability="disamb", pair_id="bank",
             prompt_a="The river bank was muddy.", positive_a=" shore", negative_a=" finance",
             prompt_b="The bank approved the loan.", positive_b=" finance", negative_b=" shore"),
    TestCase(capability="disamb", pair_id="plant",
             prompt_a="The plant was green.", positive_a=" leaf", negative_a=" factory",
             prompt_b="The plant closed down.", positive_b=" factory", negative_b=" leaf"),
    TestCase(capability="disamb", pair_id="bat",
             prompt_a="The bat flew through the cave.", positive_a=" animal", negative_a=" sports",
             prompt_b="He swung the bat at the ball.", positive_b=" sports", negative_b=" animal"),
    # 语法 (3)
    TestCase(capability="syntax", pair_id="subj_verb",
             prompt_a="The key to the cabinet", positive_a=" is", negative_a=" are",
             prompt_b="The keys to the cabinet", positive_b=" are", negative_b=" is"),
    TestCase(capability="syntax", pair_id="clause_agree",
             prompt_a="The report that was written by the interns", positive_a=" was", negative_a=" were",
             prompt_b="The reports that were written by the intern", positive_b=" were", negative_b=" was"),
    TestCase(capability="syntax", pair_id="plural_pp",
             prompt_a="The label on the bottles", positive_a=" is", negative_a=" are",
             prompt_b="The labels on the bottle", positive_b=" are", negative_b=" is"),
    # 关系 (3)
    TestCase(capability="relation", pair_id="capital",
             prompt_a="Paris is the capital of France. The capital of France is",
             positive_a=" Paris", negative_a=" Berlin",
             prompt_b="Berlin is the capital of Germany. The capital of Germany is",
             positive_b=" Berlin", negative_b=" Paris"),
    TestCase(capability="relation", pair_id="author",
             prompt_a="Shakespeare wrote Hamlet. The author of Hamlet is",
             positive_a=" Shakespeare", negative_a=" Dickens",
             prompt_b="Tolstoy wrote War and Peace. The author of War and Peace is",
             positive_b=" Tolstoy", negative_b=" Shakespeare"),
    TestCase(capability="relation", pair_id="chemical",
             prompt_a="The chemical symbol for water is", positive_a=" H2O", negative_a=" CO2",
             prompt_b="The chemical symbol for carbon dioxide is", positive_b=" CO2", negative_b=" H2O"),
    # 指代 (3)
    TestCase(capability="coref", pair_id="winner",
             prompt_a="Alice thanked Mary because Alice had won. The winner was",
             positive_a=" Alice", negative_a=" Mary",
             prompt_b="Alice thanked Mary because Mary had won. The winner was",
             positive_b=" Mary", negative_b=" Alice"),
    TestCase(capability="coref", pair_id="apology",
             prompt_a="John apologized to David because John was late. The late person was",
             positive_a=" John", negative_a=" David",
             prompt_b="John apologized to David because David was late. The late person was",
             positive_b=" David", negative_b=" John"),
    TestCase(capability="coref", pair_id="help",
             prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
             positive_a=" Emma", negative_a=" Sara",
             prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
             positive_b=" Sara", negative_b=" Emma"),
    # 风格 (3)
    TestCase(capability="style", pair_id="formal_rewrite",
             prompt_a="Choose the more formal rewrite: I need your help.", positive_a=" assistance", negative_a=" help",
             prompt_b="Choose the more casual rewrite: I require your assistance.", positive_b=" help", negative_b=" assistance"),
    TestCase(capability="style", pair_id="formal_word",
             prompt_a="Choose the more formal next word: The ceremony will", positive_a=" commence", negative_a=" start",
             prompt_b="Choose the more casual next word: The game will", positive_b=" start", negative_b=" commence"),
    TestCase(capability="style", pair_id="formal_request",
             prompt_a="Choose the more formal request ending: Please review the file and", positive_a=" advise", negative_a=" tell",
             prompt_b="Choose the more casual request ending: Please look at the file and", positive_b=" tell", negative_b=" advise"),
]

CAPABILITIES = ["disamb", "syntax", "relation", "coref", "style"]
INVARIANTS = ["INV-285(sig_eff>50%)", "INV-286(z>1.96)", "INV-293(cancel 30-90%)",
              "INV-269(pca90<20)", "INV-273(rot~90deg)"]


def extract_all_layer_hidden(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = {}
    hooks = []
    layers = discover_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0][:, -1, :].detach().cpu().float()
            else:
                hidden_states[layer_idx] = output[:, -1, :].detach().cpu().float()
        return hook_fn

    for li in range(len(layers)):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return hidden_states


def compute_all_invariants(model, tokenizer, case: TestCase, num_layers: int,
                            hidden_dim: int) -> Dict[str, bool]:
    """对单个样本计算所有不变量"""
    hidden_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
    hidden_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)
    
    # 计算increment_diffs
    increment_diffs = {}
    prev_a, prev_b = None, None
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            continue
        if prev_a is None:
            increment_diffs[li] = h_a.flatten() - h_b.flatten()
        else:
            increment_diffs[li] = (h_a.flatten() - prev_a.flatten()) - (h_b.flatten() - prev_b.flatten())
        prev_a = h_a
        prev_b = h_b
    
    if not increment_diffs:
        return {inv: False for inv in INVARIANTS}
    
    d_final = hidden_a[num_layers - 1].flatten() - hidden_b[num_layers - 1].flatten()
    d_norm = d_final.norm()
    d_unit = d_final / (d_norm + 1e-10)
    
    # INV-285: 信号效率>50%
    signal_projs = [float(torch.dot(increment_diffs[li], d_unit)) for li in increment_diffs]
    net_signal = sum(signal_projs)
    total_signal_mag = sum(abs(s) for s in signal_projs)
    sig_eff = abs(net_signal) / (total_signal_mag + 1e-10)
    inv_285 = sig_eff > 0.5
    
    # INV-286: z-score>1.96
    diff_norms = [increment_diffs[li].norm().item() for li in increment_diffs]
    sum_diff_norms = sum(diff_norms)
    actual_cancel = 1 - d_norm / (sum_diff_norms + 1e-10)
    dim = d_final.shape[0]
    n_sim = 30
    random_cancels = []
    for _ in range(n_sim):
        rv = [torch.randn(dim) * n for n in diff_norms]
        rs = sum(rv)
        random_cancels.append(1 - float(rs.norm()) / (sum_diff_norms + 1e-10))
    theoretical_mean = statistics.mean(random_cancels)
    theoretical_std = statistics.stdev(random_cancels) if len(random_cancels) > 1 else 0
    z_score = (actual_cancel - theoretical_mean) / (theoretical_std + 1e-10)
    inv_286 = z_score > 1.96
    
    # INV-293: 抵消率30-90%
    inv_293 = 0.3 < actual_cancel < 0.9
    
    # INV-269: PCA rank90<20
    diff_list = [increment_diffs[li] for li in increment_diffs]
    stacked = torch.stack(diff_list)
    U_s, S_s, Vt_s = torch.linalg.svd(stacked, full_matrices=False)
    total_energy = (S_s ** 2).sum()
    cum_energy = torch.cumsum(S_s ** 2, dim=0) / total_energy
    rank90 = int((cum_energy >= 0.9).nonzero()[0].item() + 1) if (cum_energy >= 0.9).any() else len(S_s)
    inv_269 = rank90 < 20
    
    # INV-273: 平均旋转角≈90° (60-120度)
    rotation_angles = []
    sorted_keys = sorted(increment_diffs.keys())
    for i in range(len(sorted_keys) - 1):
        d1 = increment_diffs[sorted_keys[i]]
        d2 = increment_diffs[sorted_keys[i + 1]]
        cos_val = float(torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10))
        cos_val = max(-1.0, min(1.0, cos_val))
        angle = float(torch.acos(torch.tensor(cos_val))) * 180 / 3.14159
        rotation_angles.append(angle)
    
    avg_rot = statistics.mean(rotation_angles) if rotation_angles else 90
    inv_273 = 60 < avg_rot < 120
    
    return {
        INVARIANTS[0]: bool(inv_285),
        INVARIANTS[1]: bool(inv_286),
        INVARIANTS[2]: bool(inv_293),
        INVARIANTS[3]: bool(inv_269),
        INVARIANTS[4]: bool(inv_273),
        "details": {
            "sig_eff": float(sig_eff),
            "z_score": float(z_score),
            "cancel_rate": float(actual_cancel),
            "pca_rank90": int(rank90),
            "avg_rotation_deg": float(avg_rot),
        }
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage669: P22 5×3不变量矩阵 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    
    # 获取hidden_dim
    hidden_dim = None
    for case in CASES[:1]:
        ha = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        for h in ha.values():
            if h is not None:
                hidden_dim = h.shape[-1]
                break
        break
    
    print(f"  层数={num_layers}, 维度={hidden_dim}")
    
    # 构建 5能力 × 5不变量 矩阵
    cap_inv_matrix = {}  # (capability, invariant) -> list of bools
    
    for case in CASES:
        print(f"  [{case.capability}/{case.pair_id}] ", end="", flush=True)
        
        invs = compute_all_invariants(model, tokenizer, case, num_layers, hidden_dim)
        details = invs.pop("details")
        
        for inv_name, passed in invs.items():
            key = (case.capability, inv_name)
            if key not in cap_inv_matrix:
                cap_inv_matrix[key] = []
            cap_inv_matrix[key].append(passed)
        
        print(" | ".join(f"{k.split('(')[0]}:{'Y' if v else 'N'}" for k, v in invs.items()))
    
    # ========== 汇总矩阵 ==========
    print(f"\n{'='*70}")
    print(f"不变量矩阵 (确认率%)")
    print(f"{'='*70}")
    
    summary_matrix = {}
    for cap in CAPABILITIES:
        summary_matrix[cap] = {}
        for inv_name in INVARIANTS:
            key = (cap, inv_name)
            if key in cap_inv_matrix and cap_inv_matrix[key]:
                rate = statistics.mean(cap_inv_matrix[key]) * 100
                summary_matrix[cap][inv_name] = rate
    
    # 打印矩阵
    header = f"  {'能力':>10}"
    for inv in INVARIANTS:
        short = inv.split('(')[0]
        header += f"  {short:>12}"
    print(header)
    print("  " + "-" * (12 + 13 * len(INVARIANTS)))
    
    total_confirmed = 0
    total_cells = 0
    
    for cap in CAPABILITIES:
        row = f"  {cap:>10}"
        for inv in INVARIANTS:
            rate = summary_matrix.get(cap, {}).get(inv, -1)
            if rate >= 0:
                total_confirmed += int(rate > 50)
                total_cells += 1
                marker = "Y" if rate > 50 else "N"
                row += f"  {rate:>5.0f}%{marker}"
            else:
                row += f"  {'N/A':>12}"
        print(row)
    
    # 全局统计
    global_rate = total_confirmed / (total_cells + 1e-10) * 100
    print(f"\n  全局确认率: {total_confirmed}/{total_cells} = {global_rate:.1f}%")
    
    # 按不变量统计
    print(f"\n  按不变量统计:")
    for inv in INVARIANTS:
        rates = []
        for cap in CAPABILITIES:
            r = summary_matrix.get(cap, {}).get(inv, -1)
            if r >= 0:
                rates.append(r)
        if rates:
            print(f"    {inv}: mean={statistics.mean(rates):.1f}%, "
                  f"min={min(rates):.0f}%, max={max(rates):.0f}%")
    
    # INV-299/300判伪
    print(f"\n  INV-299: 全局确认率={global_rate:.1f}% → "
          f"{'>=70% 确认(统一理论有基础)' if global_rate >= 70 else '<70% 推翻'}")
    
    # 找系统性推翻
    failed_combos = []
    for cap in CAPABILITIES:
        n_failed = sum(1 for inv in INVARIANTS
                      if summary_matrix.get(cap, {}).get(inv, 0) <= 50)
        if n_failed >= 2:
            failed_combos.append((cap, n_failed))
    
    if failed_combos:
        print(f"  INV-300: 系统性推翻(>=2个不变量)的能力: "
              + ", ".join(f"{c}({n}个)" for c, n in failed_combos))
    else:
        print(f"  INV-300: 无能力系统性推翻>=2个不变量 → 架构依赖不显著")
    
    # 保存 - 转换tuple keys为string
    raw_serializable = {}
    for k, v in cap_inv_matrix.items():
        key_str = f"{k[0]}__{k[1]}"
        raw_serializable[key_str] = [bool(x) for x in v]
    
    out_path = OUTPUT_DIR / f"stage669_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary_matrix": summary_matrix, "raw": raw_serializable},
                  f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return summary_matrix


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
