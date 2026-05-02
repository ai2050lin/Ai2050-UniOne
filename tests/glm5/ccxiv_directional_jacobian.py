"""
CCXIV(364): 方向性Jacobian — 精确概念传播动力学
===================================================

THE definitive test for concept dynamics in Transformer residual streams.

核心问题: J_{l+1} @ delta_l 是否精确等于 delta_{l+1}?

方法: 有限差分Jacobian-Vector Product (JVP)
  J @ v = (F(h + ε*v) - F(h)) / ε

与CCXIII LOO方法的根本区别:
- LOO: 从7个概念拟合A_l → 7维概念子空间投影, 因果效率<0.1
- JVP: 直接测量J@v → 2560维全空间精确结果, 无需拟合

三个实验:
  Exp1: 精确JVP — 概念方向的Jacobian-Vector Product
    对每个概念和关键层, 计算 J_{l+1} @ delta_l
    与真实 delta_{l+1} 比较 → 线性动力学的终极验证

  Exp2: 线性边界 — α从0.01到2.0, 线性近似何时崩溃?
    变化steering强度α, 测量有效Jacobian的偏差
    直接解释α*=0.5

  Exp3: 概念选择性 — Jacobian是否"偏爱"概念方向?
    比较 J@delta_concept vs J@r_random
    概念是否在Jacobian的"稳定子空间"中?

理论框架:
  残差流更新: h_{l+1} = F_l(h_l) = h_l + Attn_l(h_l) + MLP_l(h_l)
  概念方向: delta_l = h_concept_l - h_baseline_l
  线性近似: delta_{l+1} = F_l(h_baseline + delta_l) - F_l(h_baseline) ≈ J_l @ delta_l
  其中 J_l = dF_l/dh @ h_baseline

  如果 J_l @ delta_l ≈ delta_{l+1} (余弦>0.9), 则概念动力学是线性的
  如果 J_l @ delta_concept >> J_l @ r_random, 则Jacobian有概念选择性

用法:
  python ccxiv_directional_jacobian.py --model qwen3 --exp 1
  python ccxiv_directional_jacobian.py --model qwen3 --exp 2
  python ccxiv_directional_jacobian.py --model qwen3 --exp 3
  python ccxiv_directional_jacobian.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

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

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)

    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook

    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))

    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None

    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


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


def compute_jvp(model, tokenizer, device, baseline_text, inject_layer, direction,
                epsilon=0.01, capture_layers=None, baseline_states=None):
    """
    计算Jacobian-Vector Product: J_{inject_layer+1} @ direction

    方法: 有限差分
    J @ v ≈ (F(h + ε*v) - F(h)) / ε

    在inject_layer的输出注入ε*direction, 在capture_layers捕获结果

    Returns:
        jvp_dict: {layer: jvp_vector} 每个capture_layer的JVP
        perturbed_states: {layer: state} 扰动后的状态
    """
    if capture_layers is None:
        capture_layers = [inject_layer + 1]

    all_layers_list = get_layers(model)

    # 扰动hook: 在inject_layer的输出加上ε*direction
    def make_inject_hook(delta_vec, eps):
        delta_tensor = torch.tensor(eps * delta_vec, dtype=torch.float32, device=device)
        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += delta_tensor.to(h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += delta_tensor.to(h.device)
                return h
        return hook

    # 捕获hook
    captured = {}
    def make_capture_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook

    hooks = []
    if inject_layer < len(all_layers_list):
        hooks.append(all_layers_list[inject_layer].register_forward_hook(
            make_inject_hook(direction, epsilon)))
    for li in capture_layers:
        if li < len(all_layers_list):
            hooks.append(all_layers_list[li].register_forward_hook(make_capture_hook(li)))

    input_ids = tokenizer.encode(baseline_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  JVP forward failed: {e}")
            for h in hooks: h.remove()
            return {}, {}

    for h in hooks: h.remove()

    # 计算JVP: (perturbed - baseline) / epsilon
    jvp_dict = {}
    for li in capture_layers:
        if li in captured and baseline_states is not None and li in baseline_states:
            jvp_dict[li] = (captured[li] - baseline_states[li]) / epsilon
        elif li in captured:
            # 如果没有baseline, 无法计算JVP
            jvp_dict[li] = None

    gc.collect()
    return jvp_dict, captured


# ================================================================
# Exp1: 精确JVP — 概念方向的Jacobian-Vector Product
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts, key_layers):
    """
    对每个概念和关键层, 计算精确 J_{l+1} @ delta_l

    与CCXIII LOO方法的根本区别:
    - LOO: 从7概念拟合A_l (7维→2560维投影), 因果效率<0.1
    - JVP: 直接测量J@v (全空间精确), 无需拟合

    核心比较:
    1. cos(J@delta_l, delta_{l+1}) — 方向预测质量
    2. ||J@delta_l|| / ||delta_{l+1}|| — 幅度预测质量
    3. 因果效率: 用J@delta_l做steering vs 用delta_{l+1}做steering
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Exact Jacobian-Vector Product for Concept Directions")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    # 需要捕获的层: key_layers + key_layers+1 (JVP目标层)
    # 加上更深层的传播分析
    all_capture = set()
    for l in key_layers:
        all_capture.add(l)
        all_capture.add(l + 1)
        # 传播分析: 也捕获l+3和l+6
        if l + 3 < n_layers:
            all_capture.add(l + 3)
        if l + 6 < n_layers:
            all_capture.add(l + 6)
    all_capture = sorted([l for l in all_capture if l < n_layers])

    # Step 1: 收集baseline状态
    print(f"\n  Collecting baseline states at {len(all_capture)} layers...")
    baseline_states, baseline_logits = collect_states_at_layers(
        model, tokenizer, device, BASELINE_TEXT, all_capture
    )
    baseline_score_dict = {}
    for cname, cdata in concepts.items():
        baseline_score_dict[cname] = compute_concept_score(
            baseline_logits, tokenizer, cdata["probe_words"]
        )

    # Step 2: 收集概念状态
    print(f"  Collecting concept states...")
    all_deltas = {}  # {concept: {layer: delta}}
    concept_scores = {}  # {concept: {layer: steered_score}}

    for cname, cdata in concepts.items():
        concept_states_list = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(
                model, tokenizer, device, template, all_capture
            )
            for l in all_capture:
                if l in states:
                    concept_states_list[l].append(states[l])

        concept_mean = {}
        for l in all_capture:
            if concept_states_list[l]:
                concept_mean[l] = np.mean(concept_states_list[l], axis=0)

        deltas = {}
        for l in all_capture:
            if l in concept_mean and l in baseline_states:
                deltas[l] = concept_mean[l] - baseline_states[l]

        all_deltas[cname] = deltas
        del concept_states_list, concept_mean
        gc.collect()

    # Step 3: 计算JVP
    print(f"\n  Computing JVPs...")
    results = {}

    for cname, cdata in concepts.items():
        print(f"\n  --- {cname} ---")
        t0 = time.time()
        deltas = all_deltas[cname]
        layer_results = {}

        for inject_l in key_layers:
            if inject_l not in deltas:
                continue

            delta_l = deltas[inject_l]
            target_l = inject_l + 1
            if target_l >= n_layers:
                continue

            # 需要捕获的层: target_l + 传播分析层
            capture_for_this = [target_l]
            for dl in [3, 6]:
                if inject_l + dl < n_layers:
                    capture_for_this.append(inject_l + dl)
            capture_for_this = list(set(capture_for_this))

            # 计算JVP
            epsilon = 0.01
            jvp_dict, perturbed_states = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, delta_l, epsilon,
                capture_layers=capture_for_this,
                baseline_states=baseline_states
            )

            if target_l not in jvp_dict or jvp_dict[target_l] is None:
                print(f"    L{inject_l}→L{target_l}: JVP computation failed")
                continue

            jvp_at_target = jvp_dict[target_l]

            # 与真实delta_{l+1}比较
            if target_l in deltas:
                true_delta = deltas[target_l]
                n_jvp = np.linalg.norm(jvp_at_target)
                n_true = np.linalg.norm(true_delta)
                n_inject = np.linalg.norm(delta_l)

                cos_jvp_true = 0.0
                if n_jvp > 1e-8 and n_true > 1e-8:
                    cos_jvp_true = float(np.dot(jvp_at_target, true_delta) / (n_jvp * n_true))

                norm_ratio = n_jvp / n_true if n_true > 1e-8 else 0.0
                amplification = n_jvp / n_inject if n_inject > 1e-8 else 0.0

                # 残差: J@delta - delta_true
                residual = jvp_at_target - true_delta
                residual_norm = np.linalg.norm(residual) / n_true if n_true > 1e-8 else 0.0
            else:
                cos_jvp_true = None
                norm_ratio = None
                amplification = None
                residual_norm = None

            # 传播分析: JVP在更远层的预测质量
            propagation_cosines = {}
            for cl in capture_for_this:
                if cl != target_l and cl in jvp_dict and jvp_dict[cl] is not None:
                    if cl in deltas and cl in baseline_states:
                        # 有限差分直接测量了注入在更远层的效果
                        # 这个效果可以与概念的真实delta比较
                        propagated = perturbed_states.get(cl)
                        if propagated is not None and cl in baseline_states:
                            actual_shift = propagated - baseline_states[cl]
                            n_shift = np.linalg.norm(actual_shift)
                            n_delta = np.linalg.norm(deltas[cl])
                            if n_shift > 1e-8 and n_delta > 1e-8:
                                propagation_cosines[cl] = float(
                                    np.dot(actual_shift, deltas[cl]) / (n_shift * n_delta)
                                )

            # 因果验证: 用JVP结果做steering
            alpha = 0.5
            # (a) 用JVP结果(即J@delta_l)在target层注入
            if target_l < n_layers and target_l in deltas:
                from tests.glm5.ccxiii_layer_transfer import additive_steering_forward
                jvp_logits = additive_steering_forward(
                    model, tokenizer, device, BASELINE_TEXT,
                    {target_l: jvp_at_target}, alpha
                )
                jvp_score = compute_concept_score(
                    jvp_logits, tokenizer, cdata["probe_words"]
                ) if jvp_logits is not None else -20.0

                # (b) 用真实delta在target层注入
                true_logits = additive_steering_forward(
                    model, tokenizer, device, BASELINE_TEXT,
                    {target_l: deltas[target_l]}, alpha
                )
                true_score = compute_concept_score(
                    true_logits, tokenizer, cdata["probe_words"]
                ) if true_logits is not None else -20.0

                baseline_s = baseline_score_dict.get(cname, -10.0)
                d_jvp = jvp_score - baseline_s
                d_true = true_score - baseline_s
                causal_eff = d_jvp / max(abs(d_true), 0.01)
            else:
                d_jvp = None
                d_true = None
                causal_eff = None

            layer_results[inject_l] = {
                "cos_jvp_true": cos_jvp_true,
                "norm_ratio": norm_ratio,
                "amplification": amplification,
                "residual_norm": residual_norm,
                "propagation_cosines": propagation_cosines,
                "d_jvp": d_jvp,
                "d_true": d_true,
                "causal_eff": causal_eff,
            }

            cos_str = f"cos={cos_jvp_true:.3f}" if cos_jvp_true is not None else "cos=N/A"
            norm_str = f"norm_r={norm_ratio:.3f}" if norm_ratio is not None else "norm_r=N/A"
            amp_str = f"amp={amplification:.3f}" if amplification is not None else "amp=N/A"
            eff_str = f"eff={causal_eff:.3f}" if causal_eff is not None else "eff=N/A"
            print(f"    L{inject_l}→L{target_l}: {cos_str}, {norm_str}, {amp_str}, {eff_str}")

        results[cname] = layer_results
        elapsed = time.time() - t0
        print(f"  {cname} done in {elapsed:.0f}s")
        gc.collect()

    return results, all_deltas, baseline_states, baseline_score_dict


# ================================================================
# Exp2: 线性边界 — α从0.01到2.0, 线性近似何时崩溃?
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts, key_layers,
             all_deltas, baseline_states, baseline_score_dict):
    """
    关键实验: 线性近似的有效范围

    变化steering强度α:
    - 线性预测: delta_{l+1}(α) 应该 = α * J @ delta_l
    - 如果实际 delta_{l+1}(α) 偏离 α * J @ delta_l → 非线性效应

    测量: 对每个α, 做α*delta_l的steering, 捕获l+1的状态
    有效Jacobian: J_eff(α) = delta_{l+1}(α) / α
    与J(0)比较 → 线性边界在哪?
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Linearity Boundary — When Does Linear Approx Break?")
    print(f"{'='*60}")

    n_layers = model_info.n_layers

    # 先用ε=0.01计算精确J@delta_l (作为线性基准)
    # 选择2个概念和3个层来节省时间
    test_concepts = ["apple", "king"]
    test_layers = [12, 18, 24]

    # 收集精确JVP (线性基准)
    print(f"\n  Computing precise JVPs as linear baseline...")
    precise_jvps = {}

    for cname in test_concepts:
        if cname not in all_deltas:
            continue
        deltas = all_deltas[cname]
        precise_jvps[cname] = {}

        for inject_l in test_layers:
            if inject_l not in deltas:
                continue
            target_l = inject_l + 1
            if target_l >= n_layers:
                continue

            jvp_dict, _ = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, deltas[inject_l], epsilon=0.01,
                capture_layers=[target_l],
                baseline_states=baseline_states
            )

            if target_l in jvp_dict and jvp_dict[target_l] is not None:
                precise_jvps[cname][inject_l] = jvp_dict[target_l]
                print(f"    {cname} L{inject_l}: JVP computed, norm={np.linalg.norm(jvp_dict[target_l]):.3f}")

    # 线性边界测试
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = {}

    for cname in test_concepts:
        if cname not in all_deltas or cname not in precise_jvps:
            continue
        deltas = all_deltas[cname]
        cdata = concepts[cname]

        print(f"\n  --- {cname} ---")
        layer_results = {}

        for inject_l in test_layers:
            if inject_l not in deltas or inject_l not in precise_jvps[cname]:
                continue
            target_l = inject_l + 1
            if target_l >= n_layers:
                continue

            delta_l = deltas[inject_l]
            j_linear = precise_jvps[cname][inject_l]  # J @ delta_l (精确)

            alpha_results = []

            for alpha in alphas:
                # 用alpha*delta_l在inject_l做steering, 捕获target_l的状态
                capture_layers = [target_l]

                # 需要手动做steering + capture
                all_layers_list = get_layers(model)
                captured = {}

                def make_inject_hook(delta_vec, a):
                    delta_tensor = torch.tensor(a * delta_vec, dtype=torch.float32, device=device)
                    def hook(module, inp, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            h[:, -1, :] += delta_tensor.to(h.device)
                            return (h,) + output[1:]
                        else:
                            h = output.clone()
                            h[:, -1, :] += delta_tensor.to(h.device)
                            return h
                    return hook

                def make_capture_hook(li):
                    def hook(module, inp, output):
                        if isinstance(output, tuple):
                            captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            captured[li] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook

                hooks = []
                if inject_l < len(all_layers_list):
                    hooks.append(all_layers_list[inject_l].register_forward_hook(
                        make_inject_hook(delta_l, alpha)))
                if target_l < len(all_layers_list):
                    hooks.append(all_layers_list[target_l].register_forward_hook(
                        make_capture_hook(target_l)))

                input_ids = tokenizer.encode(BASELINE_TEXT, add_special_tokens=True,
                                            return_tensors="pt").to(device)
                with torch.no_grad():
                    try:
                        outputs = model(input_ids=input_ids)
                    except Exception as e:
                        print(f"    α={alpha} failed: {e}")
                        for h in hooks: h.remove()
                        continue

                for h in hooks: h.remove()

                if target_l in captured and target_l in baseline_states:
                    # 实际效果
                    actual_delta = captured[target_l] - baseline_states[target_l]

                    # 线性预测
                    predicted_delta = alpha * j_linear

                    # 比较
                    n_actual = np.linalg.norm(actual_delta)
                    n_predicted = np.linalg.norm(predicted_delta)

                    cos_actual_predicted = 0.0
                    if n_actual > 1e-8 and n_predicted > 1e-8:
                        cos_actual_predicted = float(
                            np.dot(actual_delta, predicted_delta) / (n_actual * n_predicted)
                        )

                    # 有效Jacobian (归一化到α=1)
                    j_eff = actual_delta / alpha if alpha > 0 else np.zeros_like(actual_delta)
                    n_j_eff = np.linalg.norm(j_eff)
                    n_j_linear = np.linalg.norm(j_linear)

                    cos_j_eff_j_linear = 0.0
                    if n_j_eff > 1e-8 and n_j_linear > 1e-8:
                        cos_j_eff_j_linear = float(
                            np.dot(j_eff, j_linear) / (n_j_eff * n_j_linear)
                        )

                    # 非线性度: ||actual - predicted|| / ||predicted||
                    nonlinear_err = np.linalg.norm(actual_delta - predicted_delta)
                    nonlinear_ratio = nonlinear_err / max(n_predicted, 1e-8)

                    # 因果效果
                    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
                    score = compute_concept_score(logits, tokenizer, cdata["probe_words"])
                    baseline_s = baseline_score_dict.get(cname, -10.0)
                    d_score = score - baseline_s

                    alpha_results.append({
                        "alpha": alpha,
                        "cos_actual_predicted": cos_actual_predicted,
                        "cos_j_eff_j_linear": cos_j_eff_j_linear,
                        "norm_ratio_actual_predicted": n_actual / max(n_predicted, 1e-8),
                        "nonlinear_ratio": nonlinear_ratio,
                        "delta_score": d_score,
                    })

                    print(f"    L{inject_l} α={alpha:5.2f}: "
                          f"cos={cos_actual_predicted:.3f}, "
                          f"j_eff_cos={cos_j_eff_j_linear:.3f}, "
                          f"nonlin={nonlinear_ratio:.3f}, "
                          f"Δ={d_score:+.3f}")

                gc.collect()

            layer_results[inject_l] = alpha_results

        results[cname] = layer_results

    return results


# ================================================================
# Exp3: 概念选择性 — Jacobian是否"偏爱"概念方向?
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts, key_layers,
             all_deltas, baseline_states, baseline_score_dict):
    """
    核心问题: Jacobian是否对概念方向有选择性?

    如果 ||J @ delta_concept|| / ||delta_concept|| >> ||J @ r_random|| / ||r_random||
    则Jacobian有"概念放大"特性

    如果 cos(J @ delta_concept, delta_{concept, next}) >> cos(J @ r_random, delta_{random, next})
    则Jacobian有"概念保持"特性

    这将揭示: 概念是否在Jacobian的"稳定子空间"中传播
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: Concept Selectivity — Does J Favor Concept Directions?")
    print(f"{'='*60}")

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    test_layers = [12, 18, 24]  # 中深层, 信号传播最好的区域
    n_random = 10  # 随机方向数量

    np.random.seed(42)

    results = {}

    for inject_l in test_layers:
        if inject_l + 1 >= n_layers:
            continue
        target_l = inject_l + 1

        print(f"\n  --- Layer {inject_l} → {target_l} ---")

        # A: 概念方向的JVP
        concept_jvps = {}
        concept_amplifications = {}
        concept_direction_preservations = {}

        for cname, deltas in all_deltas.items():
            if inject_l not in deltas or target_l not in deltas:
                continue

            delta_l = deltas[inject_l]
            delta_target = deltas[target_l]

            # 归一化方向 (公平比较)
            n_delta = np.linalg.norm(delta_l)
            if n_delta < 1e-8:
                continue
            delta_l_unit = delta_l / n_delta

            # JVP
            jvp_dict, _ = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, delta_l_unit, epsilon=0.01,
                capture_layers=[target_l],
                baseline_states=baseline_states
            )

            if target_l in jvp_dict and jvp_dict[target_l] is not None:
                jvp_result = jvp_dict[target_l]
                n_jvp = np.linalg.norm(jvp_result)

                # 放大倍数: ||J @ v|| / ||v|| (v是单位向量, ||v||=1)
                amplification = n_jvp

                # 方向保持: cos(J @ delta_unit, delta_target_unit)
                n_target = np.linalg.norm(delta_target)
                if n_target > 1e-8:
                    delta_target_unit = delta_target / n_target
                    direction_pres = float(
                        np.dot(jvp_result, delta_target_unit) / max(n_jvp, 1e-8)
                    )
                else:
                    direction_pres = 0.0

                concept_jvps[cname] = jvp_result
                concept_amplifications[cname] = amplification
                concept_direction_preservations[cname] = direction_pres

        # B: 随机方向的JVP
        random_amplifications = []
        random_direction_preservations = []

        for ri in range(n_random):
            r = np.random.randn(d_model)
            r = r / np.linalg.norm(r)  # 单位随机方向

            jvp_dict, _ = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, r, epsilon=0.01,
                capture_layers=[target_l],
                baseline_states=baseline_states
            )

            if target_l in jvp_dict and jvp_dict[target_l] is not None:
                jvp_result = jvp_dict[target_l]
                n_jvp = np.linalg.norm(jvp_result)

                random_amplifications.append(n_jvp)

                # 随机方向没有"正确的下一层方向", 所以用余弦的自守性
                # cos(J@r, r) — Jacobian是否保持输入方向?
                cos_pres = float(np.dot(jvp_result, r) / max(n_jvp, 1e-8))
                random_direction_preservations.append(cos_pres)

        # 汇总
        concept_amp_mean = np.mean(list(concept_amplifications.values())) if concept_amplifications else 0
        concept_amp_std = np.std(list(concept_amplifications.values())) if len(concept_amplifications) > 1 else 0
        random_amp_mean = np.mean(random_amplifications) if random_amplifications else 0
        random_amp_std = np.std(random_amplifications) if len(random_amplifications) > 1 else 0

        concept_dir_mean = np.mean(list(concept_direction_preservations.values())) if concept_direction_preservations else 0
        random_dir_mean = np.mean(random_direction_preservations) if random_direction_preservations else 0

        selectivity_amp = concept_amp_mean / max(random_amp_mean, 1e-8)
        selectivity_dir = concept_dir_mean / max(abs(random_dir_mean), 1e-8) if random_dir_mean != 0 else float('inf')

        results[inject_l] = {
            "concept_amplifications": concept_amplifications,
            "concept_direction_preservations": concept_direction_preservations,
            "random_amplifications": random_amplifications,
            "random_direction_preservations": random_direction_preservations,
            "concept_amp_mean": concept_amp_mean,
            "concept_amp_std": concept_amp_std,
            "random_amp_mean": random_amp_mean,
            "random_amp_std": random_amp_std,
            "concept_dir_mean": concept_dir_mean,
            "random_dir_mean": random_dir_mean,
            "selectivity_amp": selectivity_amp,
            "selectivity_dir": selectivity_dir,
        }

        print(f"    Concept: amp={concept_amp_mean:.3f}±{concept_amp_std:.3f}, "
              f"dir_pres={concept_dir_mean:.3f}")
        print(f"    Random:  amp={random_amp_mean:.3f}±{random_amp_std:.3f}, "
              f"dir_pres={random_dir_mean:.3f}")
        print(f"    Selectivity: amp_ratio={selectivity_amp:.3f}, "
              f"dir_ratio={selectivity_dir:.3f}")

        # 打印每个概念的细节
        for cname in concept_amplifications:
            print(f"      {cname}: amp={concept_amplifications[cname]:.3f}, "
                  f"dir_pres={concept_direction_preservations[cname]:.3f}")

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
    print(f"CCXIV: Directional Jacobian — Exact Concept Dynamics — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    key_layers = [l for l in [6, 12, 18, 24, 30] if l + 1 < n_layers]

    all_results = {}

    # Exp1需要先运行, 因为Exp2/3依赖其输出
    exp1_results = None
    all_deltas = None
    baseline_states = None
    baseline_score_dict = None

    if args.exp in ["1", "all"]:
        exp1_results, all_deltas, baseline_states, baseline_score_dict = run_exp1(
            model, tokenizer, device, model_info, CONCEPTS, key_layers
        )
        all_results["exp1"] = exp1_results

    # 为Exp2/3准备数据 (如果跳过了Exp1)
    if all_deltas is None:
        # 重新收集概念deltas
        all_capture = set()
        for l in key_layers:
            all_capture.add(l)
            all_capture.add(l + 1)
        all_capture = sorted([l for l in all_capture if l < n_layers])

        baseline_states, baseline_logits = collect_states_at_layers(
            model, tokenizer, device, BASELINE_TEXT, all_capture
        )
        baseline_score_dict = {}
        for cname, cdata in CONCEPTS.items():
            baseline_score_dict[cname] = compute_concept_score(
                baseline_logits, tokenizer, cdata["probe_words"]
            )

        all_deltas = {}
        for cname, cdata in CONCEPTS.items():
            concept_states_list = {l: [] for l in all_capture}
            for template in cdata["templates"]:
                states, _ = collect_states_at_layers(
                    model, tokenizer, device, template, all_capture
                )
                for l in all_capture:
                    if l in states:
                        concept_states_list[l].append(states[l])

            concept_mean = {}
            for l in all_capture:
                if concept_states_list[l]:
                    concept_mean[l] = np.mean(concept_states_list[l], axis=0)

            deltas = {}
            for l in all_capture:
                if l in concept_mean and l in baseline_states:
                    deltas[l] = concept_mean[l] - baseline_states[l]
            all_deltas[cname] = deltas

            del concept_states_list, concept_mean
            gc.collect()

    if args.exp in ["2", "all"]:
        exp2_results = run_exp2(
            model, tokenizer, device, model_info, CONCEPTS, key_layers,
            all_deltas, baseline_states, baseline_score_dict
        )
        all_results["exp2"] = exp2_results

    if args.exp in ["3", "all"]:
        exp3_results = run_exp3(
            model, tokenizer, device, model_info, CONCEPTS, key_layers,
            all_deltas, baseline_states, baseline_score_dict
        )
        all_results["exp3"] = exp3_results

    # 保存结果
    output_path = TEMP / f"ccxiv_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    # 总结
    print(f"\n{'#'*70}")
    print(f"CCXIV Summary")
    print(f"{'#'*70}")

    if "exp1" in all_results:
        print(f"\n  Exp1: Exact JVP")
        for cname, layer_data in all_results["exp1"].items():
            print(f"    {cname}:")
            for li, lr in layer_data.items():
                cos_str = f"cos={lr['cos_jvp_true']:.3f}" if lr.get('cos_jvp_true') is not None else "cos=N/A"
                norm_str = f"norm_r={lr['norm_ratio']:.3f}" if lr.get('norm_ratio') is not None else "norm_r=N/A"
                amp_str = f"amp={lr['amplification']:.3f}" if lr.get('amplification') is not None else "amp=N/A"
                eff_str = f"eff={lr['causal_eff']:.3f}" if lr.get('causal_eff') is not None else "eff=N/A"
                print(f"      L{li}: {cos_str}, {norm_str}, {amp_str}, {eff_str}")

    if "exp2" in all_results:
        print(f"\n  Exp2: Linearity Boundary")
        for cname, layer_data in all_results["exp2"].items():
            print(f"    {cname}:")
            for li, alpha_list in layer_data.items():
                for ar in alpha_list:
                    if ar["alpha"] in [0.01, 0.1, 0.5, 1.0, 2.0]:
                        print(f"      L{li} α={ar['alpha']:.2f}: "
                              f"cos={ar['cos_actual_predicted']:.3f}, "
                              f"nonlin={ar['nonlinear_ratio']:.3f}, "
                              f"Δ={ar['delta_score']:+.3f}")

    if "exp3" in all_results:
        print(f"\n  Exp3: Concept Selectivity")
        for li, lr in all_results["exp3"].items():
            print(f"    L{li}: concept_amp={lr['concept_amp_mean']:.3f}, "
                  f"random_amp={lr['random_amp_mean']:.3f}, "
                  f"selectivity={lr['selectivity_amp']:.3f}")

    release_model(model)
    print(f"\nCCXIV {model_name} 完成!")


if __name__ == "__main__":
    main()
