#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage670: P23 Gemma4统一编码分析

问题：Gemma4之前被发现编码策略不同（UNIFIED vs SEPARATED）
这里用统一的P22框架验证六大不变量，并比较编码效率

分析维度：
1. 不变量矩阵 — 与Qwen3/DS7B/GLM4对比
2. 编码效率 — 每维信号承载力（margin/pca_dim）
3. 编码鲁棒性 — 单方向损坏对margin的影响
4. Pareto前沿 — 效率vs鲁棒性
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


def compute_full_analysis(model, tokenizer, case: TestCase, num_layers: int,
                          hidden_dim: int) -> Dict:
    """完整分析：不变量 + 编码效率 + 鲁棒性"""
    hidden_a = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
    hidden_b = extract_all_layer_hidden(model, tokenizer, case.prompt_b)

    # 计算margin
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
               score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
               score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)

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

    # === 六大不变量 ===
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

    # INV-273: 旋转角≈90°
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

    # === 编码效率 ===
    # 每维信号承载力 = |margin| / pca_dim
    efficiency = abs(margin_a) / (rank90 + 1e-10)

    # === 鲁棒性测试 ===
    # 在最后一层注入噪声，测量margin变化
    layers = discover_layers(model)
    robustness_drops = []
    torch.cuda.empty_cache()

    for noise_scale in [0.5, 1.0]:
        hooks = []
        def make_noise_hook(scale):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    noise = torch.randn_like(hidden) * scale
                    hidden = hidden + noise
                    return (hidden,) + output[1:]
                else:
                    noise = torch.randn_like(output) * scale
                    return output + noise
            return hook_fn

        if num_layers > 0:
            hooks.append(layers[-1].register_forward_hook(make_noise_hook(noise_scale)))
        try:
            torch.cuda.empty_cache()
            margin_a_noisy = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                             score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
            drop = abs(margin_a - margin_a_noisy) / (abs(margin_a) + 1e-10)
            robustness_drops.append(min(drop, 5.0))  # cap at 500%
        except:
            robustness_drops.append(-1)
        finally:
            for h in hooks:
                h.remove()
            torch.cuda.empty_cache()

    return {
        "invariants": {
            INVARIANTS[0]: bool(inv_285),
            INVARIANTS[1]: bool(inv_286),
            INVARIANTS[2]: bool(inv_293),
            INVARIANTS[3]: bool(inv_269),
            INVARIANTS[4]: bool(inv_273),
        },
        "details": {
            "sig_eff": float(sig_eff),
            "z_score": float(z_score),
            "cancel_rate": float(actual_cancel),
            "pca_rank90": int(rank90),
            "avg_rotation_deg": float(avg_rot),
            "margin_a": float(margin_a),
            "margin_b": float(margin_b),
            "d_norm": float(d_norm),
            "efficiency_per_dim": float(efficiency),
            "robustness_drop_s05": float(robustness_drops[0]) if robustness_drops else -1,
            "robustness_drop_s10": float(robustness_drops[1]) if len(robustness_drops) > 1 else -1,
        }
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage670: P23 Gemma4/通用统一编码分析 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))

    hidden_dim = None
    for case in CASES[:1]:
        ha = extract_all_layer_hidden(model, tokenizer, case.prompt_a)
        for h in ha.values():
            if h is not None:
                hidden_dim = h.shape[-1]
                break
        break

    print(f"  层数={num_layers}, 维度={hidden_dim}")

    # 收集数据
    cap_inv_matrix = {}
    cap_details = {}

    for case in CASES:
        print(f"  [{case.capability}/{case.pair_id}] ", end="", flush=True)
        result = compute_full_analysis(model, tokenizer, case, num_layers, hidden_dim)

        invs = result["invariants"]
        details = result["details"]

        for inv_name, passed in invs.items():
            key = (case.capability, inv_name)
            if key not in cap_inv_matrix:
                cap_inv_matrix[key] = []
            cap_inv_matrix[key].append(passed)

        det_key = case.capability
        if det_key not in cap_details:
            cap_details[det_key] = {"sig_eff": [], "z_score": [], "cancel_rate": [],
                                     "pca_rank90": [], "avg_rotation_deg": [],
                                     "margin": [], "efficiency": [], "robustness_s05": [], "robustness_s10": []}
        cap_details[det_key]["sig_eff"].append(details["sig_eff"])
        cap_details[det_key]["z_score"].append(details["z_score"])
        cap_details[det_key]["cancel_rate"].append(details["cancel_rate"])
        cap_details[det_key]["pca_rank90"].append(details["pca_rank90"])
        cap_details[det_key]["avg_rotation_deg"].append(details["avg_rotation_deg"])
        cap_details[det_key]["margin"].append(abs(details["margin_a"]))
        cap_details[det_key]["efficiency"].append(details["efficiency_per_dim"])
        cap_details[det_key]["robustness_s05"].append(details["robustness_drop_s05"])
        cap_details[det_key]["robustness_s10"].append(details["robustness_drop_s10"])

        inv_str = " ".join(f"{k.split('(')[0]}:{'Y' if v else 'N'}" for k, v in invs.items())
        print(f"{inv_str} | eff={details['efficiency_per_dim']:.4f} rob={details['robustness_drop_s10']:.2f}")

    # ========== 不变量矩阵 ==========
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

    global_rate = total_confirmed / (total_cells + 1e-10) * 100
    print(f"\n  全局确认率: {total_confirmed}/{total_cells} = {global_rate:.1f}%")

    # ========== 编码效率 vs 鲁棒性 ==========
    print(f"\n{'='*70}")
    print(f"编码效率 vs 鲁棒性分析")
    print(f"{'='*70}")

    print(f"\n  {'能力':>10}  {'AvgMargin':>10}  {'PCA90':>6}  {'效率/维':>10}  {'鲁棒性(s=0.5)':>14}  {'鲁棒性(s=1.0)':>14}")
    print(f"  {'-'*75}")

    efficiencies = []
    robustnesses = []

    for cap in CAPABILITIES:
        if cap not in cap_details:
            continue
        d = cap_details[cap]
        avg_margin = statistics.mean(d["margin"])
        avg_pca = statistics.mean(d["pca_rank90"])
        avg_eff = statistics.mean([e for e in d["efficiency"] if e >= 0] or [0])
        avg_rob_s05 = statistics.mean([r for r in d["robustness_s05"] if r >= 0] or [0])
        avg_rob_s10 = statistics.mean([r for r in d["robustness_s10"] if r >= 0] or [0])

        efficiencies.append(avg_eff)
        robustnesses.append(avg_rob_s10)

        print(f"  {cap:>10}  {avg_margin:>10.4f}  {avg_pca:>6.1f}  {avg_eff:>10.4f}  {avg_rob_s05:>14.2f}  {avg_rob_s10:>14.2f}")

    # Pareto分析
    print(f"\n  编码效率全局均值: {statistics.mean(efficiencies):.4f}")
    print(f"  鲁棒性(s=1.0)全局均值: {statistics.mean(robustnesses):.2f}")

    # 编码策略判断
    avg_eff_global = statistics.mean(efficiencies)
    avg_rob_global = statistics.mean(robustnesses)

    print(f"\n  编码策略判断:")
    if avg_eff_global > 0.05 and avg_rob_global < 0.5:
        strategy = "HIGH_EFF_LOW_ROB → UNIFIED(高效但脆弱)"
    elif avg_eff_global < 0.02 and avg_rob_global > 0.3:
        strategy = "LOW_EFF_HIGH_ROB → SEPARATED(冗余但鲁棒)"
    else:
        strategy = "BALANCED → 混合策略"

    print(f"    {model_key}: {strategy}")
    print(f"    效率={avg_eff_global:.4f}, 鲁棒性={avg_rob_global:.2f}")

    # INV-299/300
    print(f"\n  INV-299: 全局确认率={global_rate:.1f}% → "
          f"{'>=70% 确认' if global_rate >= 70 else '<70% 推翻'}")

    failed_combos = []
    for cap in CAPABILITIES:
        n_failed = sum(1 for inv in INVARIANTS
                      if summary_matrix.get(cap, {}).get(inv, 0) <= 50)
        if n_failed >= 2:
            failed_combos.append((cap, n_failed))

    if failed_combos:
        print(f"  INV-300: 系统性推翻: " + ", ".join(f"{c}({n}个)" for c, n in failed_combos))
    else:
        print(f"  INV-300: 无能力系统性推翻>=2个不变量")

    # 保存
    summary_serializable = {}
    for k, v in summary_matrix.items():
        summary_serializable[k] = {kk: float(vv) for kk, vv in v.items()}

    result_data = {
        "model": model_key,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "global_rate": global_rate,
        "strategy": strategy,
        "avg_efficiency": avg_eff_global,
        "avg_robustness": avg_rob_global,
        "summary_matrix": summary_serializable,
        "cap_details": {k: {kk: float(statistics.mean(vv)) if vv else 0
                            for kk, vv in v.items()} for k, v in cap_details.items()},
    }

    out_path = OUTPUT_DIR / f"stage670_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")

    free_model(model)
    return result_data


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "gemma4"
    run_experiment(model_key)
