#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage666: P19 五类能力全层精确方程验证

目标：验证方程(1)(2)(3)在五类语言能力上是否成立：
- 方程(1): h_l = h_0 + sum(increment_k)，误差=0
- 方程(2): d_l = d_0 + sum((increment_A - increment_B)_k)，误差=0
- 方程(3): 抵消率 = ||d_L|| / sum(||increment_diff_k||)，范围56-78%
- 额外：关键层带位置、信号效率、PCA低维性

能力类型：消歧(disamb)、语法(syntax)、关系(relation)、指代(coref)、风格(style)
每种能力选5个样本

判伪标准：
INV-291: ">=3类能力复现'低维+旋转+抵消'模式 → 统一理论有基础"
INV-292: "方程(1)(2)在所有能力上误差=0"
INV-293: "抵消率在所有能力上范围50-80%"
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
    # ===== 消歧 (5) =====
    TestCase(capability="disamb", pair_id="bank",
             prompt_a="The river bank was muddy.",
             positive_a=" shore", negative_a=" finance",
             prompt_b="The bank approved the loan.",
             positive_b=" finance", negative_b=" shore"),
    TestCase(capability="disamb", pair_id="plant",
             prompt_a="The plant was green.",
             positive_a=" leaf", negative_a=" factory",
             prompt_b="The plant closed down.",
             positive_b=" factory", negative_b=" leaf"),
    TestCase(capability="disamb", pair_id="bat",
             prompt_a="The bat flew through the cave.",
             positive_a=" animal", negative_a=" sports",
             prompt_b="He swung the bat at the ball.",
             positive_b=" sports", negative_b=" animal"),
    TestCase(capability="disamb", pair_id="watch",
             prompt_a="She looked at her watch before dinner.",
             positive_a=" clock", negative_a=" observe",
             prompt_b="Watch the road carefully.",
             positive_b=" observe", negative_b=" clock"),
    TestCase(capability="disamb", pair_id="light",
             prompt_a="The light filled the room.",
             positive_a=" lamp", negative_a=" weight",
             prompt_b="The bag felt light in my hand.",
             positive_b=" weight", negative_b=" lamp"),
    # ===== 语法 (5) =====
    TestCase(capability="syntax", pair_id="subj_verb",
             prompt_a="The key to the cabinet",
             positive_a=" is", negative_a=" are",
             prompt_b="The keys to the cabinet",
             positive_b=" are", negative_b=" is"),
    TestCase(capability="syntax", pair_id="clause_agree",
             prompt_a="The report that was written by the interns",
             positive_a=" was", negative_a=" were",
             prompt_b="The reports that were written by the intern",
             positive_b=" were", negative_b=" was"),
    TestCase(capability="syntax", pair_id="plural_pp",
             prompt_a="The label on the bottles",
             positive_a=" is", negative_a=" are",
             prompt_b="The labels on the bottle",
             positive_b=" are", negative_b=" is"),
    TestCase(capability="syntax", pair_id="dist_agree",
             prompt_a="The bouquet of roses",
             positive_a=" smells", negative_a=" smell",
             prompt_b="The roses in the bouquet",
             positive_b=" smell", negative_b=" smells"),
    TestCase(capability="syntax", pair_id="collective",
             prompt_a="The picture near the windows",
             positive_a=" was", negative_a=" were",
             prompt_b="The pictures near the window",
             positive_b=" were", negative_b=" was"),
    # ===== 关系 (5) =====
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
             prompt_a="The chemical symbol for water is",
             positive_a=" H2O", negative_a=" CO2",
             prompt_b="The chemical symbol for carbon dioxide is",
             positive_b=" CO2", negative_b=" H2O"),
    TestCase(capability="relation", pair_id="currency",
             prompt_a="The currency used in Japan is",
             positive_a=" yen", negative_a=" euro",
             prompt_b="The currency used in Britain is",
             positive_b=" pound", negative_b=" yen"),
    TestCase(capability="relation", pair_id="inventor",
             prompt_a="The telephone was invented by",
             positive_a=" Bell", negative_a=" Edison",
             prompt_b="The light bulb is associated with",
             positive_b=" Edison", negative_b=" Bell"),
    # ===== 指代 (5) =====
    TestCase(capability="coref", pair_id="winner",
             prompt_a="Alice thanked Mary because Alice had won the prize. The person who won was",
             positive_a=" Alice", negative_a=" Mary",
             prompt_b="Alice thanked Mary because Mary had won the prize. The person who won was",
             positive_b=" Mary", negative_b=" Alice"),
    TestCase(capability="coref", pair_id="apology",
             prompt_a="John apologized to David because John was late. The person who was late was",
             positive_a=" John", negative_a=" David",
             prompt_b="John apologized to David because David was late. The person who was late was",
             positive_b=" David", negative_b=" John"),
    TestCase(capability="coref", pair_id="help",
             prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
             positive_a=" Emma", negative_a=" Sara",
             prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
             positive_b=" Sara", negative_b=" Emma"),
    TestCase(capability="coref", pair_id="congrat",
             prompt_a="Liam congratulated Noah because Liam had succeeded. The one who succeeded was",
             positive_a=" Liam", negative_a=" Noah",
             prompt_b="Liam congratulated Noah because Noah had succeeded. The one who succeeded was",
             positive_b=" Noah", negative_b=" Liam"),
    TestCase(capability="coref", pair_id="blame",
             prompt_a="Olivia blamed Mia because Olivia was careless. The careless person was",
             positive_a=" Olivia", negative_a=" Mia",
             prompt_b="Olivia blamed Mia because Mia was careless. The careless person was",
             positive_b=" Mia", negative_b=" Olivia"),
    # ===== 风格 (5) =====
    TestCase(capability="style", pair_id="formal_rewrite",
             prompt_a="Choose the more formal rewrite: I need your help with this request.",
             positive_a=" assistance", negative_a=" help",
             prompt_b="Choose the more casual rewrite: I require your assistance with this request.",
             positive_b=" help", negative_b=" assistance"),
    TestCase(capability="style", pair_id="formal_word",
             prompt_a="Choose the more formal next word: The ceremony will",
             positive_a=" commence", negative_a=" start",
             prompt_b="Choose the more casual next word: The game will",
             positive_b=" start", negative_b=" commence"),
    TestCase(capability="style", pair_id="formal_request",
             prompt_a="Choose the more formal request ending: Please review the attached file and",
             positive_a=" advise", negative_a=" tell",
             prompt_b="Choose the more casual request ending: Please look at the file and",
             positive_b=" tell", negative_b=" advise"),
    TestCase(capability="style", pair_id="formal_apology",
             prompt_a="Choose the more formal apology word: We",
             positive_a=" regret", negative_a=" sorry",
             prompt_b="Choose the more casual apology word: I am",
             positive_b=" sorry", negative_b=" regret"),
    TestCase(capability="style", pair_id="formal_depart",
             prompt_a="Choose the more formal departure verb: The guests will",
             positive_a=" depart", negative_a=" leave",
             prompt_b="Choose the more casual departure verb: My friends will",
             positive_b=" leave", negative_b=" depart"),
]


def extract_all_layer_hidden(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    """提取所有层最后一个token的hidden state"""
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


def analyze_case(model, tokenizer, case: TestCase, num_layers: int) -> Dict:
    """对单个样本做方程(1)(2)(3)的全层验证"""
    
    # 提取hidden states
    hidden_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
    hidden_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)
    
    hidden_dim = None
    for h in hidden_a.values():
        if h is not None:
            hidden_dim = h.shape[-1]
            break
    
    # ========== 方程(1)验证: h_l = h_0 + sum(increment_k) ==========
    eq1_errors_a = []
    eq1_errors_b = []
    
    increment_a = {}
    increment_b = {}
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            continue
        if li == 0:
            increment_a[li] = h_a.flatten()
            increment_b[li] = h_b.flatten()
        else:
            increment_a[li] = (h_a - hidden_a[li - 1]).flatten()
            increment_b[li] = (h_b - hidden_b[li - 1]).flatten()
    
    # 验证: h_L = h_0 + sum
    h0_a = hidden_a[0].flatten()
    h0_b = hidden_b[0].flatten()
    sum_incr_a = sum(increment_a.values())
    sum_incr_b = sum(increment_b.values())
    
    if num_layers - 1 in hidden_a:
        eq1_err_a = float((hidden_a[num_layers - 1].flatten() - (h0_a + sum_incr_a)).norm())
        eq1_err_b = float((hidden_b[num_layers - 1].flatten() - (h0_b + sum_incr_b)).norm())
        eq1_errors_a.append(eq1_err_a)
        eq1_errors_b.append(eq1_err_b)
    
    # ========== 方程(2)验证: d_L = d_0 + sum((increment_A - increment_B)_k) ==========
    increment_diffs = {}
    for li in increment_a:
        increment_diffs[li] = increment_a[li] - increment_b[li]
    
    d_0 = h0_a - h0_b
    d_L = hidden_a[num_layers - 1].flatten() - hidden_b[num_layers - 1].flatten()
    sum_diff = sum(increment_diffs.values())
    
    eq2_error = float((d_L - (d_0 + sum_diff)).norm())
    eq2_error_rel = eq2_error / (d_L.norm() + 1e-10)
    
    # ========== 方程(3)验证: 抵消率 ==========
    diff_norms = [float(increment_diffs[li].norm()) for li in increment_diffs]
    sum_diff_norms = sum(diff_norms)
    d_final_norm = float(d_L.norm())
    cancellation_rate = 1 - d_final_norm / (sum_diff_norms + 1e-10)
    
    # ========== 信号效率 ==========
    d_unit = d_L / (d_final_norm + 1e-10)
    signal_projections = [float(torch.dot(increment_diffs[li], d_unit)) for li in increment_diffs]
    net_signal = sum(signal_projections)
    total_signal_mag = sum(abs(s) for s in signal_projections)
    signal_efficiency = abs(net_signal) / (total_signal_mag + 1e-10)
    
    # ========== PCA低维性 ==========
    diff_list = [increment_diffs[li] for li in increment_diffs]
    stacked = torch.stack(diff_list)
    U_s, S_s, Vt_s = torch.linalg.svd(stacked, full_matrices=False)
    total_energy = (S_s ** 2).sum()
    cumsum_s = torch.cumsum(S_s ** 2, dim=0) / total_energy
    
    # 找90%和95%能量的维度数
    rank90 = int((cumsum_s >= 0.9).nonzero()[0].item() + 1) if (cumsum_s >= 0.9).any() else len(S_s)
    rank95 = int((cumsum_s >= 0.95).nonzero()[0].item() + 1) if (cumsum_s >= 0.95).any() else len(S_s)
    
    # ========== 逐层旋转角 ==========
    rotation_angles = []
    for li in range(1, num_layers):
        if li in increment_diffs and li - 1 in increment_diffs:
            d1 = increment_diffs[li - 1]
            d2 = increment_diffs[li]
            cos_val = float(torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10))
            cos_val = max(-1.0, min(1.0, cos_val))
            angle = float(torch.acos(torch.tensor(cos_val)))
            rotation_angles.append(angle)
    
    avg_rotation = statistics.mean(rotation_angles) if rotation_angles else 0
    std_rotation = statistics.stdev(rotation_angles) if len(rotation_angles) > 1 else 0
    
    # ========== 关键层带(强度峰值层) ==========
    layer_norms = [(li, float(increment_diffs[li].norm())) for li in increment_diffs]
    if layer_norms:
        peak_layer = max(layer_norms, key=lambda x: x[1])[0]
        peak_norm = max(layer_norms, key=lambda x: x[1])[1]
        avg_norm = statistics.mean([n for _, n in layer_norms])
    else:
        peak_layer = 0
        peak_norm = 0
        avg_norm = 0
    
    # ========== 层对齐(cos with d_L) ==========
    layer_alignment = {}
    for li in increment_diffs:
        cos_val = float(torch.dot(increment_diffs[li], d_unit))
        layer_alignment[li] = cos_val / (increment_diffs[li].norm() + 1e-10)
    
    # ========== margin ==========
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
               score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
               score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    
    # ========== 抵消的有序性(z-score) ==========
    dim = d_L.shape[0]
    n_sim = 50
    random_cancel_rates = []
    for _ in range(n_sim):
        random_vecs = [torch.randn(dim) * n for _, n in layer_norms]
        random_sum = sum(random_vecs)
        random_cancel = 1 - float(random_sum.norm()) / (sum_diff_norms + 1e-10)
        random_cancel_rates.append(random_cancel)
    theoretical_mean = statistics.mean(random_cancel_rates)
    theoretical_std = statistics.stdev(random_cancel_rates) if len(random_cancel_rates) > 1 else 0
    z_score = (cancellation_rate - theoretical_mean) / (theoretical_std + 1e-10)
    
    return {
        "eq1_error_a": eq1_errors_a[0] if eq1_errors_a else -1,
        "eq1_error_b": eq1_errors_b[0] if eq1_errors_b else -1,
        "eq2_error": eq2_error,
        "eq2_error_rel": eq2_error_rel,
        "cancellation_rate": cancellation_rate,
        "sum_diff_norms": sum_diff_norms,
        "d_final_norm": d_final_norm,
        "signal_efficiency": signal_efficiency,
        "pca_rank90": rank90,
        "pca_rank95": rank95,
        "pca_top_sv": [float(x) for x in S_s[:5]],
        "avg_rotation_rad": avg_rotation,
        "avg_rotation_deg": avg_rotation * 180 / 3.14159,
        "std_rotation_deg": std_rotation * 180 / 3.14159,
        "peak_layer": peak_layer,
        "peak_norm": peak_norm,
        "avg_layer_norm": avg_norm,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "z_score": z_score,
        "theoretical_cancel_rate": theoretical_mean,
        "hidden_dim": hidden_dim,
        "n_layers_used": len(increment_diffs),
        "layer_alignment": {str(k): v for k, v in layer_alignment.items()},
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage666: P19 五类能力全层精确方程验证 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    
    # 按能力分组的结果
    cap_results = {}
    
    for case in CASES:
        cap = case.capability
        print(f"\n  [{cap}/{case.pair_id}] ", end="", flush=True)
        
        result = analyze_case(model, tokenizer, case, num_layers)
        
        eq1_str = f"eq1={result['eq1_error_a']:.2e},{result['eq1_error_b']:.2e}"
        eq2_str = f"eq2={result['eq2_error']:.2e}"
        cancel_str = f"cancel={result['cancellation_rate']*100:.1f}%"
        sig_str = f"sig_eff={result['signal_efficiency']*100:.1f}%"
        pca_str = f"pca90={result['pca_rank90']}"
        z_str = f"z={result['z_score']:.1f}"
        margin_str = f"mA={result['margin_a']:.3f},mB={result['margin_b']:.3f}"
        
        print(f"{eq1_str} | {eq2_str} | {cancel_str} | {sig_str} | {pca_str} | {z_str} | {margin_str}")
        
        if cap not in cap_results:
            cap_results[cap] = []
        cap_results[cap].append(result)
    
    # ========== 汇总：按能力类型 ==========
    print(f"\n{'='*70}")
    print(f"汇总：按能力类型")
    print(f"{'='*70}")
    
    summary = {}
    capabilities_consistent = 0  # INV-291计数器
    
    for cap in ["disamb", "syntax", "relation", "coref", "style"]:
        results = cap_results.get(cap, [])
        if not results:
            continue
        
        print(f"\n  --- {cap} ({len(results)} samples) ---")
        
        eq2_errors = [r['eq2_error'] for r in results]
        cancel_rates = [r['cancellation_rate'] for r in results]
        sig_effs = [r['signal_efficiency'] for r in results]
        pca90s = [r['pca_rank90'] for r in results]
        z_scores = [r['z_score'] for r in results]
        avg_rots = [r['avg_rotation_deg'] for r in results]
        margins_a = [r['margin_a'] for r in results]
        
        print(f"    方程(2)误差: mean={statistics.mean(eq2_errors):.2e}, max={max(eq2_errors):.2e}")
        print(f"    抵消率: mean={statistics.mean(cancel_rates)*100:.1f}%, "
              f"range=[{min(cancel_rates)*100:.1f}%, {max(cancel_rates)*100:.1f}%]")
        print(f"    信号效率: mean={statistics.mean(sig_effs)*100:.1f}%")
        print(f"    PCA90维度: mean={statistics.mean(pca90s):.1f}")
        print(f"    z-score: mean={statistics.mean(z_scores):.1f}")
        print(f"    平均旋转角: mean={statistics.mean(avg_rots):.1f}°")
        print(f"    Margin A: mean={statistics.mean(margins_a):.3f}")
        
        # 判断是否"一致"：低维+抵消+高信号效率
        is_consistent = (
            statistics.mean(eq2_errors) < 0.01 and
            0.3 < statistics.mean(cancel_rates) < 0.9 and
            statistics.mean(sig_effs) > 0.5
        )
        print(f"    → {'一致(复现核心模式)' if is_consistent else '不一致(偏离核心模式)'}")
        if is_consistent:
            capabilities_consistent += 1
        
        summary[cap] = {
            "n_samples": len(results),
            "eq2_error_mean": statistics.mean(eq2_errors),
            "eq2_error_max": max(eq2_errors),
            "cancellation_mean": statistics.mean(cancel_rates),
            "cancellation_min": min(cancel_rates),
            "cancellation_max": max(cancel_rates),
            "signal_efficiency_mean": statistics.mean(sig_effs),
            "pca90_mean": statistics.mean(pca90s),
            "z_score_mean": statistics.mean(z_scores),
            "rotation_mean_deg": statistics.mean(avg_rots),
            "margin_a_mean": statistics.mean(margins_a),
            "is_consistent": is_consistent,
        }
    
    # ========== 全局统计 ==========
    print(f"\n{'='*70}")
    print(f"全局统计")
    print(f"{'='*70}")
    
    all_eq2 = [r['eq2_error'] for rs in cap_results.values() for r in rs]
    all_cancel = [r['cancellation_rate'] for rs in cap_results.values() for r in rs]
    all_sig = [r['signal_efficiency'] for rs in cap_results.values() for r in rs]
    all_z = [r['z_score'] for rs in cap_results.values() for r in rs]
    
    print(f"  方程(2)误差: mean={statistics.mean(all_eq2):.2e}, max={max(all_eq2):.2e}")
    print(f"  抵消率全局: mean={statistics.mean(all_cancel)*100:.1f}%")
    print(f"  信号效率全局: mean={statistics.mean(all_sig)*100:.1f}%")
    print(f"  z-score全局: mean={statistics.mean(all_z):.1f}")
    
    # INV-291: >=3类能力一致
    print(f"\n  INV-291: {capabilities_consistent}/5类能力复现核心模式 "
          f"→ {'确认(>=3)' if capabilities_consistent >= 3 else '推翻(<3)'}")
    
    # INV-292: 方程(1)(2)误差=0
    all_eq2_ok = all(e < 0.01 for e in all_eq2)
    print(f"  INV-292: 方程(2)误差{'全部<0.01' if all_eq2_ok else '有>=0.01'} → "
          f"{'确认' if all_eq2_ok else '推翻'}")
    
    # INV-293: 抵消率50-80%
    all_cancel_ok = all(0.3 < c < 0.9 for c in all_cancel)
    print(f"  INV-293: 抵消率{'全部30-90%' if all_cancel_ok else '有超出30-90%'} → "
          f"{'确认' if all_cancel_ok else '推翻'}")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"stage666_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": cap_results}, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return summary


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
