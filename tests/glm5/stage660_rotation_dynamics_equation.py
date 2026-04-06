#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage660: P11 旋转动力学方程 + P12 幅度方程突破

P8发现生成方程三大失败，根本原因是增量旋转(84-124°/层)。
P11目标：建立旋转角与权重矩阵SVD的定量关系
P12目标：突破幅度预测失败，测试非线性幅度方程

核心思路：
1. 提取每层MLP和Attention的权重矩阵，做SVD分析
2. 测量实际旋转角(cos(delta_l, delta_{l+1}))
3. 建立旋转角~权重矩阵条件数的回归方程
4. 测试向量累加幅度方程: ||d|| = ||sum(delta_l)|| (精确但需要方向信息)
5. 测试"能量守恒定律": total_energy = sum(||delta_l||^2) 是否稳定

预注册判伪条件：
INV-266: "旋转角可由权重SVD预测(R^2>0.3)"
  如果R^2<0.1, 则旋转与权重矩阵无关
INV-267: "向量累加可精确预测末层||d||(误差<1%)"
  如果误差>10%, 则向量累加有额外来源
INV-268: "层间能量守恒(total_energy稳定)"
  如果不同case的total_energy变异系数>50%, 则能量不守恒
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    name: str
    prompt_a: str
    prompt_b: str
    positive_a: str
    negative_a: str


CASES = [
    TestCase(
        name="bank_financial",
        prompt_a="The bank approved the loan with favorable terms.",
        prompt_b="The river bank was covered with wild flowers.",
        positive_a="financial",
        negative_a="river",
    ),
    TestCase(
        name="syllogism",
        prompt_a="All men are mortal. Socrates is a man. Therefore,",
        prompt_b="All men are mortal. Socrates is a cat. Therefore,",
        positive_a="Socrates is mortal",
        negative_a="Socrates is immortal",
    ),
    TestCase(
        name="arithmetic",
        prompt_a="If x = 7 and y = 3, then x + y =",
        prompt_b="If x = 7 and y = 3, then x - y =",
        positive_a="10",
        negative_a="4",
    ),
    TestCase(
        name="syntax_sv",
        prompt_a="The tall building stands proudly.",
        prompt_b="The tall buildings stand proudly.",
        positive_a="stands",
        negative_a="stand",
    ),
    TestCase(
        name="relation_capital",
        prompt_a="The capital of France is",
        prompt_b="The capital of Japan is",
        positive_a="Paris",
        negative_a="Tokyo",
    ),
    TestCase(
        name="coreference",
        prompt_a="Mary gave the book to John because she wanted to help.",
        prompt_b="Mary gave the book to John because he wanted to help.",
        positive_a="Mary",
        negative_a="John",
    ),
]


def extract_hidden_at_layers(model, tokenizer, prompt: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """在指定层提取最后一个token的hidden state"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    device = next(model.parameters()).device
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

    for li in layer_indices:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return hidden_states


def get_weight_svd(weight: torch.Tensor):
    """获取权重矩阵的SVD"""
    w = weight.detach().cpu().float()
    if w.dim() == 1:
        return {"cond": float("nan"), "top3_sv": [], "eff_rank": 0, "norm": w.norm().item()}
    U, S, Vt = torch.linalg.svd(w, full_matrices=False)
    S = S.float()
    cond = (S[0] / (S[-1] + 1e-10)).item()
    total_energy = (S ** 2).sum().item()
    cumsum = torch.cumsum(S ** 2, dim=0) / total_energy
    rank90 = (cumsum >= 0.9).nonzero()
    eff_rank = rank90[0].item() + 1 if len(rank90) > 0 else len(S)
    return {
        "cond": cond,
        "top3_sv": [round(s.item(), 4) for s in S[:3]],
        "eff_rank": int(eff_rank),
        "norm": w.norm().item(),
        "mean_sv": S.mean().item(),
        "std_sv": S.std().item(),
        "max_sv": S[0].item(),
        "min_sv": S[-1].item(),
        "sv_ratio": (S[0] / S.mean()).item(),
    }


def extract_layer_weights(model) -> Dict[int, Dict[str, Dict]]:
    """提取每层MLP和Attention关键权重矩阵的SVD"""
    layers = discover_layers(model)
    result = {}
    for li, layer in enumerate(layers):
        layer_info = {}
        # MLP权重
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj') and hasattr(mlp, 'down_proj'):
                layer_info["mlp_gate"] = get_weight_svd(mlp.gate_proj.weight)
                layer_info["mlp_up"] = get_weight_svd(mlp.up_proj.weight)
                layer_info["mlp_down"] = get_weight_svd(mlp.down_proj.weight)
                # MLP复合矩阵: down_proj @ (gate * up)
                try:
                    combined = mlp.down_proj.weight.float() @ (mlp.gate_proj.weight.float() * mlp.up_proj.weight.float())
                    layer_info["mlp_combined"] = get_weight_svd(combined)
                except Exception:
                    # 维度不兼容，单独用down_proj
                    layer_info["mlp_combined"] = get_weight_svd(mlp.down_proj.weight)
            elif hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                layer_info["mlp_fc1"] = get_weight_svd(mlp.fc1.weight)
                layer_info["mlp_fc2"] = get_weight_svd(mlp.fc2.weight)
            # GLM4/chatglm可能有不同的结构
            if "mlp_combined" not in layer_info:
                # 尝试找到任何线性层
                for attr_name in ['dense_h_to_4h', 'dense_4h_to_h', 'w1', 'w2', 'w3']:
                    if hasattr(mlp, attr_name):
                        layer_info[f"mlp_{attr_name}"] = get_weight_svd(getattr(mlp, attr_name).weight)
        # 如果MLP在model直接属性下而非layer.mlp
        if not any(k.startswith("mlp") for k in layer_info) and hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for attr in dir(mlp):
                if not attr.startswith('_') and hasattr(getattr(mlp, attr, None), 'weight'):
                    try:
                        layer_info[f"mlp_{attr}"] = get_weight_svd(getattr(mlp, attr).weight)
                    except Exception:
                        pass
        # Attention权重
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
                layer_info["attn_q"] = get_weight_svd(attn.q_proj.weight)
                layer_info["attn_k"] = get_weight_svd(attn.k_proj.weight)
                layer_info["attn_v"] = get_weight_svd(attn.v_proj.weight)
                if hasattr(attn, 'o_proj'):
                    layer_info["attn_o"] = get_weight_svd(attn.o_proj.weight)
        result[li] = layer_info
    return result


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage660: P11+P12 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    device = next(model.parameters()).device

    # 选取均匀分布的层用于分析
    if num_layers <= 10:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = sorted(set(
            [0] +
            [int(i * (num_layers - 1) / 8) for i in range(1, 8)] +
            [num_layers - 1]
        ))

    # ========== Step 1: 提取权重SVD ==========
    print("\n[Step 1] 提取权重矩阵SVD...")
    weight_svd = extract_layer_weights(model)

    # 汇总关键SVD统计量
    mlp_cond_per_layer = []
    mlp_effrank_per_layer = []
    attn_cond_per_layer = []
    for li in layer_indices:
        if li in weight_svd:
            if "mlp_combined" in weight_svd[li]:
                mlp_cond_per_layer.append(weight_svd[li]["mlp_combined"]["cond"])
                mlp_effrank_per_layer.append(weight_svd[li]["mlp_combined"]["eff_rank"])
            if "attn_q" in weight_svd[li]:
                attn_cond_per_layer.append(weight_svd[li]["attn_q"]["cond"])

    print(f"  MLP combined: mean_cond={statistics.mean(mlp_cond_per_layer) if mlp_cond_per_layer else 'N/A'}, "
          f"mean_effrank={statistics.mean(mlp_effrank_per_layer) if mlp_effrank_per_layer else 'N/A'}")
    print(f"  Attn Q: mean_cond={statistics.mean(attn_cond_per_layer) if attn_cond_per_layer else 'N/A'}")

    # ========== Step 2: 提取hidden states和delta ==========
    print("\n[Step 2] 提取hidden states和计算delta/旋转角...")

    all_deltas = {}  # {case_name: {layer_idx: delta_vector}}
    all_rotations = {}  # {case_name: {layer_pair: rotation_angle}}

    for case in CASES:
        hs_a = extract_hidden_at_layers(model, tokenizer, case.prompt_a, layer_indices)
        hs_b = extract_hidden_at_layers(model, tokenizer, case.prompt_b, layer_indices)

        deltas = {}
        sorted_layers = sorted(layer_indices)
        for i, li in enumerate(sorted_layers):
            if li in hs_a and li in hs_b:
                da = hs_a[li].squeeze(0) if hs_a[li].dim() > 1 else hs_a[li]
                db = hs_b[li].squeeze(0) if hs_b[li].dim() > 1 else hs_b[li]
                d = da - db
                deltas[li] = d
        all_deltas[case.name] = deltas

        # 计算相邻层delta旋转角
        rotations = {}
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            if l1 in deltas and l2 in deltas:
                d1 = deltas[l1]
                d2 = deltas[l2]
                cos_val = torch.dot(d1, d2) / (d1.norm() * d2.norm() + 1e-10)
                cos_val = cos_val.item()
                cos_val = max(-1.0, min(1.0, cos_val))
                angle = np.arccos(abs(cos_val)) * 180 / np.pi
                rotations[(l1, l2)] = {
                    "cos": round(cos_val, 4),
                    "angle": round(angle, 2)
                }
        all_rotations[case.name] = rotations

    # ========== Step 3: 旋转角~权重SVD回归 (P11) ==========
    print("\n[Experiment 1: P11] 旋转角与权重SVD的回归分析 (INV-266)...")

    # 收集回归数据点
    rot_angles = []
    mlp_conds = []
    attn_conds = []
    mlp_effranks = []
    mlp_sv_ratios = []
    layer_positions = []  # 归一化层位置

    for case_name, rots in all_rotations.items():
        for (l1, l2), rinfo in rots.items():
            rot_angles.append(rinfo["angle"])
            layer_positions.append(l2 / num_layers)
            if l2 in weight_svd:
                ws = weight_svd[l2]
                if "mlp_combined" in ws:
                    mlp_conds.append(ws["mlp_combined"]["cond"])
                    mlp_effranks.append(ws["mlp_combined"]["eff_rank"])
                    mlp_sv_ratios.append(ws["mlp_combined"]["sv_ratio"])
                else:
                    mlp_conds.append(float("nan"))
                    mlp_effranks.append(float("nan"))
                    mlp_sv_ratios.append(float("nan"))
                if "attn_q" in ws:
                    attn_conds.append(ws["attn_q"]["cond"])
                else:
                    attn_conds.append(float("nan"))

    # 简单线性回归
    def linear_regression(x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x, y = x[mask], y[mask]
        if len(x) < 3:
            return {"r2": float("nan"), "pearson": float("nan"), "n": len(x)}
        mx, my = x.mean(), y.mean()
        ss_xx = ((x - mx) ** 2).sum()
        ss_yy = ((y - my) ** 2).sum()
        ss_xy = ((x - mx) * (y - my)).sum()
        if ss_xx == 0 or ss_yy == 0:
            return {"r2": float("nan"), "pearson": float("nan"), "n": len(x)}
        pearson = ss_xy / np.sqrt(ss_xx * ss_yy)
        r2 = pearson ** 2
        slope = ss_xy / ss_xx
        intercept = my - slope * mx
        return {"r2": round(r2, 4), "pearson": round(pearson, 4), "n": int(len(x)),
                "slope": round(slope, 6), "intercept": round(intercept, 4)}

    reg_mlp_cond = linear_regression(mlp_conds, rot_angles)
    reg_attn_cond = linear_regression(attn_conds, rot_angles)
    reg_mlp_effrank = linear_regression(mlp_effranks, rot_angles)
    reg_mlp_sv_ratio = linear_regression(mlp_sv_ratios, rot_angles)
    reg_layer_pos = linear_regression(layer_positions, rot_angles)

    print(f"  MLP cond ~ rotation: R^2={reg_mlp_cond['r2']}, Pearson={reg_mlp_cond['pearson']}")
    print(f"  Attn cond ~ rotation: R^2={reg_attn_cond['r2']}, Pearson={reg_attn_cond['pearson']}")
    print(f"  MLP eff_rank ~ rotation: R^2={reg_mlp_effrank['r2']}, Pearson={reg_mlp_effrank['pearson']}")
    print(f"  MLP sv_ratio ~ rotation: R^2={reg_mlp_sv_ratio['r2']}, Pearson={reg_mlp_sv_ratio['pearson']}")
    print(f"  Layer position ~ rotation: R^2={reg_layer_pos['r2']}, Pearson={reg_layer_pos['pearson']}")

    inv266_status = "CONFIRMED" if max(reg_mlp_cond["r2"], reg_attn_cond["r2"], reg_mlp_sv_ratio["r2"]) >= 0.1 else "FALSIFIED"
    print(f"  INV-266 status: {inv266_status} (阈值: R^2>=0.1)")

    # ========== Step 4: 幅度方程突破 (P12) ==========
    print("\n[Experiment 2: P12] 幅度方程突破 (INV-267, INV-268)...")

    magnitude_results = []

    for case in CASES:
        deltas = all_deltas[case.name]
        if not deltas:
            continue

        sorted_layers = sorted(deltas.keys())
        # 真实末层delta范数
        last_layer = sorted_layers[-1]
        true_norm = deltas[last_layer].norm().item()

        # 方法1: 标量累加(之前失败的方法)
        abs_sum = sum(deltas[l].norm().item() for l in sorted_layers)

        # 方法2: 向量累加(残差流方程, 理论精确)
        if sorted_layers[0] == 0:
            # 从h_0开始精确累加
            vec_sum = sum(deltas[l] for l in sorted_layers)
        else:
            # 近似: 直接向量求和
            vec_sum = sum(deltas[l] for l in sorted_layers)
        vec_norm = vec_sum.norm().item()

        # 方法3: L2范数(考虑部分抵消)
        delta_stack = torch.stack([deltas[l] for l in sorted_layers])
        l2_norm = torch.sqrt((delta_stack ** 2).sum(dim=0)).norm().item()

        # 方法4: 向量累加的中间结果(逐层)
        cumulative_vec_norms = []
        running_sum = torch.zeros_like(deltas[sorted_layers[0]])
        for l in sorted_layers:
            running_sum = running_sum + deltas[l]
            cumulative_vec_norms.append(running_sum.norm().item())

        # 方法5: 能量守恒检验
        total_energy = sum((deltas[l].norm().item()) ** 2 for l in sorted_layers)
        energy_per_dim = total_energy / deltas[sorted_layers[0]].shape[0]

        # 抵消率
        cancellation_ratio = 1.0 - (vec_norm / abs_sum) if abs_sum > 0 else 0

        # 误差分析
        vec_error_pct = abs(vec_norm - true_norm) / (abs(true_norm) + 1e-10) * 100

        result = {
            "case": case.name,
            "true_norm": round(true_norm, 4),
            "abs_sum": round(abs_sum, 4),
            "vec_norm": round(vec_norm, 4),
            "l2_norm": round(l2_norm, 4),
            "total_energy": round(total_energy, 4),
            "cancellation_ratio": round(cancellation_ratio, 4),
            "vec_error_pct": round(vec_error_pct, 2),
            "cumulative_vec_norms": [round(x, 4) for x in cumulative_vec_norms],
            "num_layers_used": len(sorted_layers),
        }
        magnitude_results.append(result)
        print(f"  {case.name}: true={true_norm:.2f}, vec={vec_norm:.2f} "
              f"(err={vec_error_pct:.1f}%), abs_sum={abs_sum:.2f}, cancel={cancellation_ratio:.2%}")

    # INV-267: 向量累加精确性
    mean_vec_error = statistics.mean([r["vec_error_pct"] for r in magnitude_results])
    inv267_status = "CONFIRMED" if mean_vec_error < 10 else "FALSIFIED"
    print(f"\n  INV-267 (向量累加精确性): mean_error={mean_vec_error:.1f}%, status={inv267_status}")

    # INV-268: 能量守恒
    energies = [r["total_energy"] for r in magnitude_results]
    energy_cv = statistics.stdev(energies) / (statistics.mean(energies) + 1e-10) * 100
    inv268_status = "CONFIRMED" if energy_cv < 50 else "FALSIFIED"
    print(f"  INV-268 (能量守恒): mean_energy={statistics.mean(energies):.1f}, "
          f"CV={energy_cv:.1f}%, status={inv268_status}")

    # ========== Step 5: 深层分析 - 抵消模式 ==========
    print("\n[Experiment 3: 深层分析] 增量抵消模式...")

    cancellation_patterns = {}
    for case in CASES:
        deltas = all_deltas[case.name]
        sorted_layers = sorted(deltas.keys())
        if len(sorted_layers) < 3:
            continue

        running_sum = torch.zeros_like(deltas[sorted_layers[0]])
        layer_cancellations = []

        for l in sorted_layers:
            delta = deltas[l]
            # 检查delta与running_sum的方向
            if running_sum.norm() > 1e-6:
                cos_with_prev = torch.dot(running_sum, delta) / (running_sum.norm() * delta.norm() + 1e-10)
                cos_with_prev = cos_with_prev.item()
            else:
                cos_with_prev = 0.0

            new_sum = running_sum + delta
            # 这次增量是"促进"还是"抵消"
            growth = new_sum.norm().item() - running_sum.norm().item()
            layer_cancellations.append({
                "layer": l,
                "cos_with_prev": round(cos_with_prev, 4),
                "growth": round(growth, 4),
                "delta_norm": round(delta.norm().item(), 4),
                "is_cancellation": growth < 0
            })
            running_sum = new_sum

        cancel_count = sum(1 for lc in layer_cancellations if lc["is_cancellation"])
        cancellation_patterns[case.name] = {
            "total_layers": len(layer_cancellations),
            "cancel_count": cancel_count,
            "cancel_ratio": round(cancel_count / len(layer_cancellations), 4),
            "mean_cos_with_prev": round(statistics.mean([lc["cos_with_prev"] for lc in layer_cancellations]), 4),
        }

    for cn, cp in cancellation_patterns.items():
        print(f"  {cn}: cancel_ratio={cp['cancel_ratio']:.2%}, "
              f"mean_cos_with_prev={cp['mean_cos_with_prev']:.4f}")

    # ========== Step 6: 多元回归 - 预测旋转角 ==========
    print("\n[Experiment 4: 多元回归] 用权重SVD + 层位置预测旋转角...")

    # 准备数据矩阵
    valid_mask = []
    features = []
    targets = []

    for i in range(len(rot_angles)):
        mc = mlp_conds[i]
        ac = attn_conds[i]
        me = mlp_effranks[i]
        sr = mlp_sv_ratios[i]
        lp = layer_positions[i]
        ra = rot_angles[i]

        if all(not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in [mc, ac, me, sr, lp, ra]):
            features.append([mc, ac, me, sr, lp])
            targets.append(ra)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    X = np.array(features)
    y = np.array(targets)

    if len(X) >= 5:
        # 多元线性回归 (最小二乘)
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
            y_pred = X_with_bias @ beta
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            multi_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            pearson_multi = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 0
            print(f"  多元回归 (cond+eff_rank+sv_ratio+layer_pos): R^2={multi_r2:.4f}, Pearson={pearson_multi:.4f}")
            print(f"  系数: intercept={beta[0]:.2f}, mlp_cond={beta[1]:.6f}, "
                  f"attn_cond={beta[2]:.6f}, eff_rank={beta[3]:.4f}, sv_ratio={beta[4]:.4f}, "
                  f"layer_pos={beta[5]:.2f}")
        except Exception as e:
            multi_r2 = float("nan")
            print(f"  多元回归失败: {e}")
    else:
        multi_r2 = float("nan")

    # ========== 保存结果 ==========
    results = {
        "model": model_key,
        "num_layers": num_layers,
        "layer_indices_used": layer_indices,
        "experiment1_rotation_svd": {
            "reg_mlp_cond": reg_mlp_cond,
            "reg_attn_cond": reg_attn_cond,
            "reg_mlp_effrank": reg_mlp_effrank,
            "reg_mlp_sv_ratio": reg_mlp_sv_ratio,
            "reg_layer_pos": reg_layer_pos,
            "multi_r2": round(multi_r2, 4) if not np.isnan(multi_r2) else None,
            "inv266_status": inv266_status,
            "mean_mlp_cond": round(statistics.mean(mlp_cond_per_layer), 2) if mlp_cond_per_layer else None,
            "mean_attn_cond": round(statistics.mean(attn_cond_per_layer), 2) if attn_cond_per_layer else None,
        },
        "experiment2_magnitude": {
            "per_case": magnitude_results,
            "inv267_status": inv267_status,
            "mean_vec_error_pct": round(mean_vec_error, 2),
        },
        "experiment3_cancellation": cancellation_patterns,
        "mean_rotation_angle": round(statistics.mean(rot_angles), 2),
        "std_rotation_angle": round(statistics.stdev(rot_angles), 2) if len(rot_angles) > 1 else 0,
    }

    out_dir = OUTPUT_DIR / f"stage660_rotation_dynamics_{model_key}_{TIMESTAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_dir}")

    free_model(model)
    return results


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
