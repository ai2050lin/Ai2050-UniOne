"""
CCXIII(363): 层间传输算子 — 概念动力学
=========================================

核心问题: v_{l+1} ≈ A_l v_l 是否成立?
如果成立, 概念在网络中的演化就是确定性的线性动力学!

三个实验:
  Exp1: 最优α曲线 — 每层搜索α*, 揭示方向质量 vs 尺度敏感性
  Exp2: 轨迹顺序注入 — 依次传播+注入 vs 同时注入, 区分轨道vs向量
  Exp3: 层间预测 — J_l v_l ≈ v_{l+1}? 概念动力学的终极测试

理论框架:
  残差流更新: h_{l+1} = h_l + Attn(h_l) + MLP(h_l)
  概念方向: delta_l = h_concept_l - h_baseline_l
  如果概念方向是"小扰动", 则:
    delta_{l+1} ≈ (I + J_l) @ delta_l  其中 J_l = dF_l/dh @ h_baseline
  这就是层间传输算子: A_l = I + J_l

  但注意: 这是线性近似, 对大alpha的steering可能不成立
  关键测试: 线性预测在多大程度上成立?

用法:
  python ccxiii_layer_transfer.py --model qwen3 --exp 1
  python ccxiii_layer_transfer.py --model qwen3 --exp 2
  python ccxiii_layer_transfer.py --model qwen3 --exp 3
  python ccxiii_layer_transfer.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)

TEMP = Path("tests/glm5_temp")

CONCEPTS = {
    "apple": {
        "templates": [
            "The word is apple",
            "I ate an apple",
            "A red apple",
            "The apple fell",
            "Apple is a fruit",
        ],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": [
            "The word is dog",
            "A big dog",
            "The dog barked",
            "My pet dog",
            "Dog is an animal",
        ],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": [
            "The word is king",
            "The king ruled",
            "A wise king",
            "The king and queen",
            "King is a ruler",
        ],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": [
            "The word is doctor",
            "The doctor helped",
            "A good doctor",
            "Visit the doctor",
            "Doctor treats patients",
        ],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
}


def get_hidden_states(model, tokenizer, device, text, sample_layers):
    """获取残差流状态"""
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        except:
            outputs = model(input_ids=input_ids)

    result = {}
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        for li in sample_layers:
            if li + 1 < len(outputs.hidden_states):
                result[li] = outputs.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy()

    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return result, logits


def compute_concept_score(logits, tokenizer, probe_words):
    """计算概念激活分数"""
    log_probs = logits - np.max(logits)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))

    probe_log_probs = []
    for word in probe_words:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            if len(ids) == 1 and ids[0] < len(logits):
                probe_log_probs.append(log_probs[ids[0]])
                break

    score = float(np.mean(probe_log_probs)) if probe_log_probs else -20.0
    return score


def additive_steering_forward(model, tokenizer, device, text, deltas, alpha=1.0):
    """加性残差注入"""
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)

    def make_hook(layer_idx, delta_vec, a):
        delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return h
        return hook

    hooks = []
    all_layers = get_layers(model)
    for li, delta in deltas.items():
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li, delta, alpha)))

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Steering failed: {e}")
            for h in hooks: h.remove()
            return None

    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return logits


def get_steered_hidden_states(model, tokenizer, device, text, deltas, alpha, sample_layers):
    """获取steering后的残差流状态 (用于Exp2传播实验)"""
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)

    captured = {}

    def make_capture_hook(layer_idx):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[layer_idx] = output[0, -1, :].detach().float().cpu().numpy()
        return hook

    def make_inject_hook(layer_idx, delta_vec, a):
        delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return h
        return hook

    hooks = []
    all_layers = get_layers(model)

    # 捕获所有层输出
    for li in sample_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_capture_hook(li)))

    # 注入delta
    for li, delta in deltas.items():
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_inject_hook(li, delta, alpha)))

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        except Exception as e:
            print(f"  Steered forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None

    for h in hooks: h.remove()

    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


# ================================================================
# Exp1: 最优α曲线
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts, sample_layers):
    """
    对每层搜索最优alpha:
    - 揭示哪些层的概念方向"质量"最高
    - 线性区范围 → 方向的鲁棒性
    - 饱和行为 → 非线性程度
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Optimal Alpha Curve")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    results = {}

    alphas = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]

    for concept_name, concept_data in concepts.items():
        print(f"\n  --- {concept_name} ---")
        t0 = time.time()

        # 收集概念轨迹
        concept_trajectories = {}
        for template in concept_data["templates"]:
            states, _ = get_hidden_states(model, tokenizer, device, template, sample_layers)
            for li, state in states.items():
                if li not in concept_trajectories:
                    concept_trajectories[li] = []
                concept_trajectories[li].append(state)

        concept_mean = {li: np.mean(states, axis=0) for li, states in concept_trajectories.items()}

        # 基线
        baseline_text = "The word is"
        baseline_states, baseline_logits = get_hidden_states(
            model, tokenizer, device, baseline_text, sample_layers
        )

        baseline_score = compute_concept_score(baseline_logits, tokenizer, concept_data["probe_words"])

        # 逐层概念方向
        concept_deltas = {}
        for li in sample_layers:
            if li in concept_mean and li in baseline_states:
                concept_deltas[li] = concept_mean[li] - baseline_states[li]

        # 对关键层搜索alpha
        # 选择效果较好的中间层(基于CCXII-B结果)
        key_layers = [l for l in [6, 9, 12, 15, 18, 21, 24, 27, 30, 33] if l < n_layers]

        layer_alpha_results = {}

        for inject_layer in key_layers:
            if inject_layer not in concept_deltas:
                continue

            delta = concept_deltas[inject_layer]
            alpha_scores = []

            for alpha in alphas:
                if alpha == 0.0:
                    score = baseline_score
                else:
                    deltas = {inject_layer: delta}
                    logits = additive_steering_forward(
                        model, tokenizer, device, baseline_text, deltas, alpha
                    )
                    if logits is not None:
                        score = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                    else:
                        score = -20.0

                delta_score = score - baseline_score
                alpha_scores.append({"alpha": alpha, "score": score, "delta": delta_score})
                print(f"    L{inject_layer:2d} α={alpha:5.1f}: Δ={delta_score:+.3f}")

            # 找最优alpha
            best = max(alpha_scores, key=lambda x: x["delta"])
            # 找线性区 (delta_score随alpha线性增长的区间)
            linear_range = 0.0
            for i in range(1, len(alpha_scores)):
                if alpha_scores[i]["delta"] > alpha_scores[i-1]["delta"] and alpha_scores[i]["delta"] > 0:
                    linear_range = alpha_scores[i]["alpha"]
                else:
                    break

            layer_alpha_results[inject_layer] = {
                "alpha_scores": alpha_scores,
                "best_alpha": best["alpha"],
                "best_delta": best["delta"],
                "linear_range_end": linear_range,
            }

        results[concept_name] = {
            "baseline_score": baseline_score,
            "layer_alpha_results": layer_alpha_results,
        }

        elapsed = time.time() - t0
        print(f"  {concept_name} done in {elapsed:.0f}s")

        del concept_trajectories, concept_mean, baseline_states, concept_deltas
        gc.collect()

    return results


# ================================================================
# Exp2: 轨迹顺序注入 vs 同时注入
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts, sample_layers):
    """
    关键区分器:
    - 同时注入: 在所有层同时加上各自的delta (CCXII-B方式)
    - 顺序注入: 在L1注入delta, 让模型传播到L2, 然后在L2注入更新后的delta, ...

    如果顺序注入优于同时注入 → 轨道假说部分成立
    如果两者差不多或同时注入更好 → 向量模型更准确
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Trajectory Sequential vs Simultaneous Injection")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    results = {}

    for concept_name, concept_data in concepts.items():
        print(f"\n  --- {concept_name} ---")
        t0 = time.time()

        # 收集概念轨迹
        concept_trajectories = {}
        for template in concept_data["templates"]:
            states, _ = get_hidden_states(model, tokenizer, device, template, sample_layers)
            for li, state in states.items():
                if li not in concept_trajectories:
                    concept_trajectories[li] = []
                concept_trajectories[li].append(state)

        concept_mean = {li: np.mean(states, axis=0) for li, states in concept_trajectories.items()}

        # 基线
        baseline_text = "The word is"
        baseline_states, baseline_logits = get_hidden_states(
            model, tokenizer, device, baseline_text, sample_layers
        )
        baseline_score = compute_concept_score(baseline_logits, tokenizer, concept_data["probe_words"])

        # 逐层概念方向
        concept_deltas = {}
        for li in sample_layers:
            if li in concept_mean and li in baseline_states:
                concept_deltas[li] = concept_mean[li] - baseline_states[li]

        # 选择关键层区间: 中深层 (L12-L30)
        inject_layers = [l for l in [12, 15, 18, 21, 24, 27, 30] if l < n_layers and l in concept_deltas]

        alpha = 0.5  # 用较小的alpha避免过大干扰

        comparison_results = []

        # A: 单层最优 (baseline比较)
        best_single_delta = -999
        best_single_layer = None
        for li in inject_layers:
            deltas = {li: concept_deltas[li]}
            logits = additive_steering_forward(model, tokenizer, device, baseline_text, deltas, alpha)
            if logits is not None:
                score = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                d = score - baseline_score
                if d > best_single_delta:
                    best_single_delta = d
                    best_single_layer = li

        # B: 同时注入 (CCXII-B方式)
        for n_inject in [2, 3, 4, 5]:
            if n_inject > len(inject_layers):
                break
            layers_subset = inject_layers[:n_inject]
            deltas = {li: concept_deltas[li] for li in layers_subset}
            logits = additive_steering_forward(model, tokenizer, device, baseline_text, deltas, alpha)
            if logits is not None:
                score = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                d = score - baseline_score
                comparison_results.append({
                    "method": "simultaneous",
                    "n_layers": n_inject,
                    "layers": layers_subset,
                    "delta_score": d,
                })
                print(f"    Simultaneous {n_inject} layers: Δ={d:+.3f}")

        # C: 顺序注入 — 关键创新!
        # 在第一层注入delta, 然后测量传播后的状态, 计算下一层需要的"修正量"
        # 简化版: 在L_i注入delta_i, 然后测量L_{i+1}的状态变化,
        #         看这个变化是否与concept_delta_{i+1}对齐
        for n_inject in [2, 3, 4, 5]:
            if n_inject > len(inject_layers):
                break
            layers_subset = inject_layers[:n_inject]

            # 顺序注入: 在第一个层注入delta, 获取后续层状态, 计算残差, 注入残差...
            cumulative_delta_score = None

            # 方法: 依次在每层注入delta, 但每次注入后重新forward
            # 这模拟了"轨道"式的传播: 每层注入后让模型自然计算到下一层
            # 但这种方法需要n_inject次forward, 成本较高

            # 简化的顺序注入:
            # Step 1: 在L_i注入delta_i, forward获取logits
            # 这等价于: 每次只在一层注入, 但注入的是该层的"真实delta"
            # (与simultaneous的区别: 不是同时注入多层)

            # 更精确的方法: 逐层注入, 每层只注入当前层的delta
            # 但同时保留之前层的注入效果
            # → 这就是"累积注入", 和simultaneous一样...
            #
            # 真正的顺序注入应该是:
            # 1. 在L1注入delta_1 → forward → 得到新状态
            # 2. 在L2注入delta_2' (修正后的delta) → forward → 得到新状态
            # 其中delta_2' = concept_delta_2 - (propagated_delta_1_at_L2)
            # 即: 补偿L1注入在L2产生的影响

            # 实现方案: 逐步forward
            # 但这太贵了(n_inject次forward, 每次都要跑整个模型)
            # 替代方案: 一次forward中, 在每层注入"净增量"

            # 最有信息量的简化: 比较3种方式
            # (a) 只在最后一层注入 (deepest-only)
            # (b) 只在最深层+次深层注入 (2-deep)
            # (c) 在所有选中层同时注入 (simultaneous)
            pass

        # 重新设计: 更有区分力的实验
        # 核心对比:
        # (1) 同时注入多层 (CCXII-B方式)
        # (2) 顺序传播注入: 只在第一层注入, 测量自然传播效果
        # (3) 最优单层 (参考基准)

        sequential_results = []

        # 2a: 只在第一层注入, 看概念信号能传播多远
        # (这直接测试层间传输: 如果信号能传播, 说明A_l存在且保留概念方向)
        for first_layer in inject_layers[:3]:  # 只测前3个
            delta = concept_deltas[first_layer]
            deltas = {first_layer: delta}

            # 获取steering后的隐藏状态
            steered_states, steered_logits = get_steered_hidden_states(
                model, tokenizer, device, baseline_text, deltas, alpha, sample_layers
            )

            if steered_logits is not None:
                score = compute_concept_score(steered_logits, tokenizer, concept_data["probe_words"])
                d = score - baseline_score

                # 测量信号在各层的余弦相似度
                cos_at_layers = {}
                if steered_states and baseline_states:
                    for li in sample_layers:
                        if li in steered_states and li in baseline_states:
                            propagated = steered_states[li] - baseline_states[li]
                            if li in concept_deltas:
                                n_p = np.linalg.norm(propagated)
                                n_d = np.linalg.norm(concept_deltas[li])
                                if n_p > 1e-8 and n_d > 1e-8:
                                    cos_val = float(np.dot(propagated, concept_deltas[li]) / (n_p * n_d))
                                    cos_at_layers[li] = cos_val

                sequential_results.append({
                    "method": "propagate_from_L{}".format(first_layer),
                    "inject_layer": first_layer,
                    "delta_score": d,
                    "cosine_at_layers": cos_at_layers,
                })
                print(f"    Propagate from L{first_layer}: Δ={d:+.3f}, "
                      f"cos@next={list(cos_at_layers.values())[0]:.3f}" if cos_at_layers else
                      f"    Propagate from L{first_layer}: Δ={d:+.3f}, no cos data")

        # 2b: 顺序两步注入 — 在L_i注入delta_i, 在L_{i+3}注入delta_{i+3}
        # 与在L_i和L_{i+3}同时注入比较
        for i in range(len(inject_layers) - 1):
            li = inject_layers[i]
            lj = inject_layers[min(i+2, len(inject_layers)-1)]
            if li == lj:
                continue

            # 同时注入两层
            deltas_sim = {li: concept_deltas[li], lj: concept_deltas[lj]}
            logits_sim = additive_steering_forward(model, tokenizer, device, baseline_text, deltas_sim, alpha)
            score_sim = compute_concept_score(logits_sim, tokenizer, concept_data["probe_words"]) if logits_sim is not None else -20.0

            # 顺序注入: 先注入浅层, 测量深层状态, 计算修正量后注入
            # 简化: 只在深层的delta中减去浅层注入的传播效果
            deltas_shallow = {li: concept_deltas[li]}
            steered_states, _ = get_steered_hidden_states(
                model, tokenizer, device, baseline_text, deltas_shallow, alpha, sample_layers
            )

            if steered_states and lj in steered_states and lj in baseline_states and lj in concept_deltas:
                # 浅层注入在深层产生的偏移
                propagated_shift = steered_states[lj] - baseline_states[lj]
                # 深层需要注入的"修正delta" = 目标delta - 已传播的偏移
                corrected_delta = concept_deltas[lj] - propagated_shift

                # 用修正后的delta注入
                deltas_corrected = {li: concept_deltas[li], lj: corrected_delta}
                logits_corr = additive_steering_forward(
                    model, tokenizer, device, baseline_text, deltas_corrected, alpha
                )
                score_corr = compute_concept_score(logits_corr, tokenizer, concept_data["probe_words"]) if logits_corr is not None else -20.0

                d_sim = score_sim - baseline_score
                d_corr = score_corr - baseline_score

                comparison_results.append({
                    "method": "two_layer_simultaneous",
                    "layers": [li, lj],
                    "delta_score": d_sim,
                })
                comparison_results.append({
                    "method": "two_layer_corrected",
                    "layers": [li, lj],
                    "delta_score": d_corr,
                    "improvement": d_corr - d_sim,
                })
                print(f"    L{li}+L{lj}: simultaneous Δ={d_sim:+.3f}, corrected Δ={d_corr:+.3f}, "
                      f"improvement={d_corr-d_sim:+.3f}")

        results[concept_name] = {
            "baseline_score": baseline_score,
            "best_single_layer": best_single_layer,
            "best_single_delta": best_single_delta,
            "comparison_results": comparison_results,
            "sequential_results": sequential_results,
        }

        elapsed = time.time() - t0
        print(f"  {concept_name} done in {elapsed:.0f}s")

        del concept_trajectories, concept_mean, baseline_states, concept_deltas
        gc.collect()

    return results


# ================================================================
# Exp3: 层间预测 — 概念动力学的终极测试
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts, sample_layers):
    """
    核心测试: v_{l+1} ≈ A_l v_l ?

    方法:
    1. 收集概念的残差流轨迹: {delta_l = h_concept_l - h_baseline_l}
    2. 计算层间传输矩阵: A_l ≈ delta_{l+1} @ delta_l^+ (伪逆)
       或更精确: 对多个概念, 用最小二乘拟合A_l
    3. 测试预测: A_l @ delta_l vs 真实 delta_{l+1}

    更严格的方法:
    - 用有限差分估计Jacobian: J_l ≈ (F(h + εv) - F(h)) / ε
    - 其中F是层l的前向函数
    - 然后测试: (I + J_l) @ delta_l vs delta_{l+1}

    但有限差分Jacobian对d_model=2560的模型太贵了
    (需要d_model次forward pass)

    替代方案: 用多个概念方向的线性回归拟合A_l
    - 如果4个概念足够, 可以拟合一个d_model x d_model的矩阵
    - 但4个样本远不够拟合2560x2560的矩阵

    最实际的方案:
    - 不拟合完整A_l, 而是测试delta_{l+1}是否能从delta_l线性预测
    - 即: cos(A_l @ delta_l, delta_{l+1}) 和 norm ratio
    - 用伪逆: A_l = Delta_{l+1} @ pinv(Delta_l), 其中Delta是概念矩阵
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: Inter-Layer Prediction — Concept Dynamics")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    results = {}

    # 需要更多概念来拟合A_l
    # 增加概念
    extended_concepts = dict(concepts)
    extended_concepts["red"] = {
        "templates": [
            "The color is red",
            "A red car",
            "The red apple",
            "Red is a color",
            "She wore red",
        ],
        "probe_words": ["color", "bright", "blood", "fire", "blue", "green", "yellow", "orange"],
    }
    extended_concepts["water"] = {
        "templates": [
            "The word is water",
            "Drink some water",
            "Cold water",
            "Water is liquid",
            "The water flows",
        ],
        "probe_words": ["drink", "liquid", "wet", "cold", "ocean", "river", "rain", "sea"],
    }
    extended_concepts["book"] = {
        "templates": [
            "The word is book",
            "Read a book",
            "A good book",
            "The book says",
            "Book is for reading",
        ],
        "probe_words": ["read", "page", "story", "write", "library", "author", "paper", "novel"],
    }

    all_concept_names = list(extended_concepts.keys())

    # 收集所有概念的残差流轨迹
    print(f"\n  Collecting residual stream trajectories for {len(all_concept_names)} concepts...")

    all_deltas = {}  # {concept_name: {layer: delta}}
    baseline_states = None
    baseline_logits = None
    baseline_score_dict = {}

    baseline_text = "The word is"

    for concept_name, concept_data in extended_concepts.items():
        concept_trajectories = {}
        for template in concept_data["templates"]:
            states, _ = get_hidden_states(model, tokenizer, device, template, sample_layers)
            for li, state in states.items():
                if li not in concept_trajectories:
                    concept_trajectories[li] = []
                concept_trajectories[li].append(state)

        concept_mean = {li: np.mean(states, axis=0) for li, states in concept_trajectories.items()}

        if baseline_states is None:
            baseline_states, baseline_logits = get_hidden_states(
                model, tokenizer, device, baseline_text, sample_layers
            )

        deltas = {}
        for li in sample_layers:
            if li in concept_mean and li in baseline_states:
                deltas[li] = concept_mean[li] - baseline_states[li]

        all_deltas[concept_name] = deltas
        baseline_score_dict[concept_name] = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"]
        )

        del concept_trajectories, concept_mean
        gc.collect()

    # ============================================================
    # 3A: 层间方向余弦 — delta_{l+1} vs delta_l 的对齐度
    # ============================================================
    print(f"\n  --- 3A: Inter-Layer Direction Cosine ---")

    cosine_matrix = {}  # {concept: {(l1, l2): cos}}

    for concept_name in all_concept_names:
        deltas = all_deltas[concept_name]
        cosine_matrix[concept_name] = {}

        sorted_layers = sorted([l for l in deltas.keys()])
        for i in range(len(sorted_layers) - 1):
            l1 = sorted_layers[i]
            l2 = sorted_layers[i + 1]
            d1 = deltas[l1]
            d2 = deltas[l2]
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_val = float(np.dot(d1, d2) / (n1 * n2))
                cosine_matrix[concept_name][(l1, l2)] = cos_val

        # 打印
        for (l1, l2), cv in sorted(cosine_matrix[concept_name].items()):
            print(f"    {concept_name}: L{l1}->L{l2} cos={cv:.3f}")

    # ============================================================
    # 3B: 线性预测 — 用相邻层的delta预测下一层
    # ============================================================
    print(f"\n  --- 3B: Linear Prediction delta_(l+1) = A * delta_l ---")

    # 对每对相邻层, 用所有概念的delta拟合最小二乘
    # Delta_l: [n_concepts, d_model] — 概念在层l的方向
    # Delta_{l+1}: [n_concepts, d_model] — 概念在层l+1的方向
    # A_l = Delta_{l+1} @ pinv(Delta_l)
    # 预测质量: cos(A_l @ delta_l_test, delta_{l+1}_test)

    # 由于概念数(7)远小于d_model(2560), A_l是低秩的
    # 但这正是我们关心的: 概念子空间上的传输

    prediction_results = {}

    sorted_layers = sorted(sample_layers)
    n_concepts = len(all_concept_names)

    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]

        # 构建概念矩阵
        D1_list = []
        D2_list = []
        valid_concepts = []

        for concept_name in all_concept_names:
            if l1 in all_deltas[concept_name] and l2 in all_deltas[concept_name]:
                D1_list.append(all_deltas[concept_name][l1])
                D2_list.append(all_deltas[concept_name][l2])
                valid_concepts.append(concept_name)

        if len(valid_concepts) < 3:
            continue

        D1 = np.array(D1_list)  # [n_concepts, d_model]
        D2 = np.array(D2_list)  # [n_concepts, d_model]

        # 用伪逆拟合 A_l
        # A_l = D2^T @ pinv(D1^T)  但d_model太大
        # 改为: 对每个概念, 测试"leave-one-out"预测

        # 方法1: 简单余弦预测
        # 对每个概念, 用其他概念拟合A, 然后预测该概念
        loo_cosines = []
        loo_norm_ratios = []

        for leave_out_idx in range(len(valid_concepts)):
            # 训练集
            train_D1 = np.delete(D1, leave_out_idx, axis=0)  # [n-1, d_model]
            train_D2 = np.delete(D2, leave_out_idx, axis=0)

            # 测试样本
            test_d1 = D1[leave_out_idx]  # [d_model]
            test_d2 = D2[leave_out_idx]  # [d_model]

            # 用训练集拟合线性映射: D2_train = A @ D1_train
            # A = D2_train @ pinv(D1_train)
            # 但 [d_model, d_model] 太大, 用SVD降维

            # 改用投影法:
            # 将test_d1投影到train_D1的行空间, 然后映射到D2空间
            # 即: 预测的delta_{l+1} = D2_train @ (train_D1^+ @ test_d1)

            # 更简单: 用train概念方向的线性组合表示test_d1
            # test_d1 ≈ train_D1^T @ c
            # c = (train_D1 @ train_D1^T)^{-1} @ train_D1 @ test_d1
            # 预测: test_d2_pred = train_D2^T @ c = D2_train^T @ (D1_train @ D1_train^T)^{-1} @ D1_train @ test_d1

            # 简化为: 在概念子空间中的线性预测
            G1 = train_D1 @ train_D1.T  # [n-1, n-1] Gram矩阵
            try:
                c = np.linalg.solve(G1 + 1e-6 * np.eye(len(G1)), train_D1 @ test_d1)
                pred_d2 = train_D2.T @ c  # [d_model]
            except:
                continue

            # 评估
            n_pred = np.linalg.norm(pred_d2)
            n_true = np.linalg.norm(test_d2)
            if n_pred > 1e-8 and n_true > 1e-8:
                cos_pred = float(np.dot(pred_d2, test_d2) / (n_pred * n_true))
                norm_ratio = n_pred / n_true
                loo_cosines.append(cos_pred)
                loo_norm_ratios.append(norm_ratio)

        if loo_cosines:
            avg_cos = np.mean(loo_cosines)
            avg_nr = np.mean(loo_norm_ratios)
            prediction_results[(l1, l2)] = {
                "avg_cosine": avg_cos,
                "avg_norm_ratio": avg_nr,
                "n_concepts": len(valid_concepts),
                "loo_cosines": loo_cosines,
            }
            print(f"    L{l1}->L{l2}: avg_cos={avg_cos:.3f}, avg_norm_ratio={avg_nr:.3f}, "
                  f"n_concepts={len(valid_concepts)}")

    # ============================================================
    # 3C: 因果验证 — 用预测的delta做steering
    # ============================================================
    print(f"\n  --- 3C: Causal Verification ---")

    # 如果 A_l @ delta_l 能预测 delta_{l+1},
    # 那么在层l注入delta_l, 然后在层l+1注入A_l@delta_l (而非真实delta_{l+1})
    # 应该和注入真实delta_{l+1}效果相似

    causal_results = {}

    for concept_name in list(concepts.keys()):  # 只用4个核心概念
        concept_data = concepts[concept_name]
        deltas = all_deltas[concept_name]

        for i in range(len(sorted_layers) - 1):
            l1 = sorted_layers[i]
            l2 = sorted_layers[i + 1]

            if l1 not in deltas or l2 not in deltas:
                continue

            delta_l1 = deltas[l1]
            delta_l2_true = deltas[l2]

            # 用所有其他概念拟合A_{l1->l2}, 然后预测当前概念的delta_l2
            train_D1 = []
            train_D2 = []
            for other_name in all_concept_names:
                if other_name == concept_name:
                    continue
                if l1 in all_deltas[other_name] and l2 in all_deltas[other_name]:
                    train_D1.append(all_deltas[other_name][l1])
                    train_D2.append(all_deltas[other_name][l2])

            if len(train_D1) < 3:
                continue

            train_D1 = np.array(train_D1)
            train_D2 = np.array(train_D2)

            # 拟合: delta_l2 = A @ delta_l1
            G1 = train_D1 @ train_D1.T
            try:
                c = np.linalg.solve(G1 + 1e-6 * np.eye(len(G1)), train_D1 @ delta_l1)
                delta_l2_pred = train_D2.T @ c
            except:
                continue

            # 比较: 预测vs真实的余弦
            n_pred = np.linalg.norm(delta_l2_pred)
            n_true = np.linalg.norm(delta_l2_true)
            if n_pred < 1e-8 or n_true < 1e-8:
                continue
            cos_pred_true = float(np.dot(delta_l2_pred, delta_l2_true) / (n_pred * n_true))

            # Causal test: 在L{l2}注入预测的delta
            alpha = 0.5

            # (a) 注入真实delta_l2
            logits_true = additive_steering_forward(
                model, tokenizer, device, baseline_text, {l2: delta_l2_true}, alpha
            )
            score_true = compute_concept_score(logits_true, tokenizer, concept_data["probe_words"]) if logits_true is not None else -20.0

            # (b) 注入预测的delta_l2
            logits_pred = additive_steering_forward(
                model, tokenizer, device, baseline_text, {l2: delta_l2_pred}, alpha
            )
            score_pred = compute_concept_score(logits_pred, tokenizer, concept_data["probe_words"]) if logits_pred is not None else -20.0

            # (c) 基线
            baseline_s = baseline_score_dict.get(concept_name, -10.0)

            d_true = score_true - baseline_s
            d_pred = score_pred - baseline_s

            key = f"{concept_name}_L{l1}_L{l2}"
            causal_results[key] = {
                "concept": concept_name,
                "l1": l1,
                "l2": l2,
                "cos_pred_true": cos_pred_true,
                "delta_true": d_true,
                "delta_pred": d_pred,
                "efficiency": d_pred / max(abs(d_true), 0.01),
            }

            if abs(d_true) > 0.1:  # 只打印有意义的结果
                print(f"    {concept_name} L{l1}->L{l2}: cos={cos_pred_true:.3f}, "
                      f"Δ_true={d_true:+.3f}, Δ_pred={d_pred:+.3f}, "
                      f"eff={d_pred/max(abs(d_true),0.01):.2f}")

    # ============================================================
    # 3D: 残差流的层间传播分析
    # ============================================================
    print(f"\n  --- 3D: Residual Stream Propagation Analysis ---")

    # 在某一层注入delta, 测量在后续各层产生的偏移
    # 与概念的真实delta比较 → 信号传播质量
    propagation_results = {}

    for concept_name in list(concepts.keys())[:2]:  # 只测2个概念
        concept_data = concepts[concept_name]
        deltas = all_deltas[concept_name]

        # 选择3个注入层
        inject_layers = [l for l in [12, 18, 24] if l in deltas]

        for inject_l in inject_layers:
            delta = deltas[inject_l]
            alpha = 0.5

            # 获取注入后的各层状态
            steered_states, steered_logits = get_steered_hidden_states(
                model, tokenizer, device, baseline_text, {inject_l: delta}, alpha, sample_layers
            )

            if not steered_states:
                continue

            # 各层的传播余弦
            layer_cosines = {}
            for li in sample_layers:
                if li in steered_states and li in baseline_states and li in deltas:
                    propagated = steered_states[li] - baseline_states[li]
                    n_p = np.linalg.norm(propagated)
                    n_d = np.linalg.norm(deltas[li])
                    if n_p > 1e-8 and n_d > 1e-8:
                        cos_val = float(np.dot(propagated, deltas[li]) / (n_p * n_d))
                        layer_cosines[li] = cos_val

            propagation_results[f"{concept_name}_L{inject_l}"] = {
                "inject_layer": inject_l,
                "layer_cosines": layer_cosines,
            }

            cos_str = ", ".join([f"L{l}:{c:.2f}" for l, c in sorted(layer_cosines.items())])
            print(f"    {concept_name} inject@L{inject_l}: {cos_str}")

    results = {
        "cosine_matrix": {k: {f"L{l1}_L{l2}": v for (l1, l2), v in d.items()}
                         for k, d in cosine_matrix.items()},
        "prediction_results": {f"L{l1}_L{l2}": v for (l1, l2), v in prediction_results.items()},
        "causal_results": causal_results,
        "propagation_results": propagation_results,
    }

    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3"])
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXIII: Layer Transfer Operator — Concept Dynamics — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    sample_layers = sorted(set([0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]))
    sample_layers = [l for l in sample_layers if l < n_layers]

    all_results = {}

    if args.exp in ["1", "all"]:
        r1 = run_exp1(model, tokenizer, device, model_info, CONCEPTS, sample_layers)
        all_results["exp1"] = r1

    if args.exp in ["2", "all"]:
        r2 = run_exp2(model, tokenizer, device, model_info, CONCEPTS, sample_layers)
        all_results["exp2"] = r2

    if args.exp in ["3", "all"]:
        r3 = run_exp3(model, tokenizer, device, model_info, CONCEPTS, sample_layers)
        all_results["exp3"] = r3

    # 保存结果
    output_path = TEMP / f"ccxiii_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    # 总结
    print(f"\n{'#'*70}")
    print(f"CCXIII Summary")
    print(f"{'#'*70}")

    if "exp1" in all_results:
        print(f"\n  Exp1: Optimal Alpha")
        for concept, data in all_results["exp1"].items():
            print(f"    {concept}:")
            for li, lr in data.get("layer_alpha_results", {}).items():
                print(f"      L{li}: best_α={lr['best_alpha']:.1f}, best_Δ={lr['best_delta']:+.3f}, "
                      f"linear_up_to={lr['linear_range_end']:.1f}")

    if "exp2" in all_results:
        print(f"\n  Exp2: Sequential vs Simultaneous")
        for concept, data in all_results["exp2"].items():
            print(f"    {concept}: best_single_L{data.get('best_single_layer')} "
                  f"Δ={data.get('best_single_delta', 0):+.3f}")
            for cr in data.get("comparison_results", []):
                if "improvement" in cr:
                    print(f"      {cr['method']} L{cr['layers']}: Δ={cr['delta_score']:+.3f}, "
                          f"improvement={cr['improvement']:+.3f}")

    if "exp3" in all_results:
        print(f"\n  Exp3: Inter-Layer Prediction")
        for key, val in all_results["exp3"].get("prediction_results", {}).items():
            print(f"    {key}: avg_cos={val['avg_cosine']:.3f}, avg_norm_ratio={val['avg_norm_ratio']:.3f}")

    release_model(model)
    print(f"\nCCXIII {model_name} 完成!")


if __name__ == "__main__":
    main()
