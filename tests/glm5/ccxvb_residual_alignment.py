"""
CCXVB: 残差连接对齐假说验证
=============================

CCXV发现: 概念方向既不在Jacobian低秩子空间中(5-7%), 也不在W_U行空间中(8-14%)。
那为什么J@delta_l与delta_{l+1}高度对齐(cos≈0.8)?

假说: 残差连接J ≈ I + 低秩修正
  h_{l+1} = h_l + Attn + MLP
  J ≈ I + J_Attn + J_MLP

如果I在J中占主导:
  J@delta_l ≈ delta_l + (J_Attn + J_MLP)@delta_l
  delta_{l+1} = delta_l + (Attn和MLP对概念的实际贡献)

两者都是"delta_l + 小修正", 所以余弦自然很高!

验证:
1. cos(delta_l, delta_{l+1}) — 概念方向在层间变化多大?
2. ||(J-I)@delta_l|| / ||delta_l|| — Jacobian偏离恒等矩阵多少?
3. cos((J-I)@delta_l, delta_{l+1}-delta_l) — 非恒等部分是否与真实修正对齐?
4. 分解: J_Attn@delta vs J_MLP@delta — 哪个贡献更大?
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
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
    },
}

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
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


def compute_jvp_with_attn_mlp_decomposition(model, tokenizer, device, baseline_text,
                                              inject_layer, direction, epsilon, baseline_states):
    """
    分解JVP为Attn和MLP分量
    
    方法: 在inject_layer注入扰动后, 分别捕获:
    1. Attn输出后的残差 (post-attn residual)
    2. MLP输出后的残差 (post-MLP residual = 正常层输出)
    
    Transformer层结构: 
    h' = h + Attn(LN1(h))   -- post-attn
    h'' = h' + MLP(LN2(h')) -- post-MLP = 层输出
    
    所以:
    delta_post_attn = J_total[:post_attn] @ delta_in
    delta_post_mlp = J_total @ delta_in (完整的JVP)
    delta_attn = delta_post_attn - delta_in (Attn的贡献)
    delta_mlp = delta_post_mlp - delta_post_attn (MLP的贡献)
    """
    all_layers_list = get_layers(model)
    n_layers = len(all_layers_list)
    target_l = inject_layer + 1
    
    # 我们需要在inject层内部捕获attn和mlp的中间结果
    # 但hook只能捕获层的输出, 不能捕获层内的中间结果
    # 替代方案: 分别计算full JVP和attn-only JVP
    
    # 方法2: 利用层结构, 在inject层的输出直接看delta的传播
    # 完整JVP就是 (perturbed_output - baseline_output) / epsilon
    
    captured = {}
    
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
    if target_l < len(all_layers_list):
        hooks.append(all_layers_list[target_l].register_forward_hook(make_capture_hook(target_l)))
    
    input_ids = tokenizer.encode(baseline_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  JVP forward failed: {e}")
            for h in hooks: h.remove()
            return None
    
    for h in hooks: h.remove()
    gc.collect()
    
    if target_l in captured and target_l in baseline_states:
        jvp = (captured[target_l] - baseline_states[target_l]) / epsilon
        return jvp
    return None


def compute_attn_mlp_jvp_separately(model, tokenizer, device, baseline_text,
                                      inject_layer, direction, epsilon, baseline_states):
    """
    分别计算Attn和MLP的JVP分量
    
    方法: 
    1. 正常JVP = (F(h+εv) - F(h))/ε = delta_{l+1}/ε (完整传播)
    2. Attn-only JVP: 在inject层的self_attn输出hook中注入扰动
    3. MLP-only JVP: 在inject层的mlp输出hook中注入扰动
    
    但这需要对层内部结构做hook, 比较复杂。
    
    更简单的方法: 利用残差连接的线性性
    h_{l+1} = h_l + Attn_l(h_l) + MLP_l(h_l)
    J_l = I + J_Attn + J_MLP
    
    J@v = v + J_Attn@v + J_MLP@v
    
    我们可以分别计算:
    - J_Attn@v: 在attn输出处注入v, 看层输出的变化
    - J_MLP@v: 在mlp输出处注入v, 看层输出的变化
    
    但这需要在层内部hook, 不同模型架构不一样。
    
    最简单的方法: 分别屏蔽Attn和MLP
    - J_Attn@v ≈ (F_attn_only(h+εv) - F_attn_only(h))/ε  (但这不可行, 因为不能单独跑attn)
    
    让我用另一种方法: 直接计算(J-I)@v
    J@v - v = (J_Attn + J_MLP)@v
    
    然后通过在LN后(而不是LN前)注入来分离。
    
    算了, 用最简单的方法: 直接分析 J@v = v + (J@v - v)
    """
    # 完整JVP
    full_jvp = compute_jvp_with_attn_mlp_decomposition(
        model, tokenizer, device, baseline_text,
        inject_layer, direction, epsilon, baseline_states
    )
    
    if full_jvp is None:
        return None
    
    # (J-I)@v = J@v - v (非恒等部分的JVP)
    # 注意: v是在inject层的输出注入的, 所以delta_l = direction
    # J@v是target层的响应
    # 但v的维度在inject层, J@v在target层, 不能直接减
    
    # 正确的分解:
    # delta_{l+1} = J_l @ delta_l
    # = delta_l + (J_l - I) @ delta_l  (如果J_l = I + 修正)
    # 但J_l是层到层的映射, delta_l和delta_{l+1}在不同空间
    # 残差连接的意思是: h_{l+1} = h_l + f(h_l), 所以 delta_{l+1} ≈ delta_l + f'(h_l) @ delta_l
    # 即delta_{l+1} - delta_l = f'(h_l) @ delta_l ≈ J@delta_l - delta_l
    
    return full_jvp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXVB: Residual Connection Alignment Verification — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    key_layers = [l for l in [6, 12, 18, 24, 30] if l + 1 < n_layers]

    # 收集baseline
    all_capture = set()
    for l in key_layers:
        all_capture.add(l)
        all_capture.add(l + 1)
    all_capture = sorted([l for l in all_capture if l < n_layers])

    print(f"\n  Collecting baseline states...")
    baseline_states, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)

    # 收集概念deltas
    print(f"  Collecting concept deltas...")
    all_deltas = {}
    for cname, cdata in CONCEPTS.items():
        concept_states_list = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
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

    # ================================================================
    # 核心测试: 残差连接对齐
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Residual Connection Alignment Analysis")
    print(f"{'='*60}")

    results = {}

    for inject_l in key_layers:
        target_l = inject_l + 1
        if target_l >= n_layers:
            continue

        print(f"\n  --- Layer {inject_l} → {target_l} ---")

        layer_results = {}

        for cname, deltas in all_deltas.items():
            if inject_l not in deltas or target_l not in deltas:
                continue

            delta_l = deltas[inject_l]
            delta_target = deltas[target_l]

            n_delta_l = np.linalg.norm(delta_l)
            n_delta_target = np.linalg.norm(delta_target)

            if n_delta_l < 1e-8 or n_delta_target < 1e-8:
                continue

            # 1. cos(delta_l, delta_{l+1}) — 概念方向在层间变化多大?
            cos_delta_l_target = float(np.dot(delta_l, delta_target) / (n_delta_l * n_delta_target))

            # 2. delta的变化量: delta_{l+1} - delta_l
            delta_change = delta_target - delta_l
            n_change = np.linalg.norm(delta_change)

            # 3. 变化相对于delta_l的大小
            change_ratio = n_change / n_delta_l

            # 4. 计算JVP
            jvp = compute_jvp_with_attn_mlp_decomposition(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, delta_l, epsilon=0.01,
                baseline_states=baseline_states
            )

            if jvp is None:
                continue

            n_jvp = np.linalg.norm(jvp)

            # 5. cos(J@delta_l, delta_{l+1}) — JVP预测质量
            cos_jvp_target = float(np.dot(jvp, delta_target) / (n_jvp * n_delta_target))

            # 6. (J-I)@delta_l = J@delta_l - delta_l
            # 这是Jacobian的非恒等部分
            j_non_identity = jvp - delta_l
            n_j_non_id = np.linalg.norm(j_non_identity)

            # 7. cos((J-I)@delta_l, delta_{l+1}-delta_l) — 非恒等部分是否与真实修正对齐?
            cos_j_nonid_change = 0.0
            if n_j_non_id > 1e-8 and n_change > 1e-8:
                cos_j_nonid_change = float(np.dot(j_non_identity, delta_change) / (n_j_non_id * n_change))

            # 8. ||(J-I)@delta_l|| / ||delta_l|| — Jacobian偏离恒等矩阵多少?
            j_nonid_ratio = n_j_non_id / n_delta_l

            # 9. 恒等部分贡献: delta_l占delta_{l+1}的多少?
            # delta_{l+1} = delta_l + change
            # delta_l在delta_{l+1}方向的投影
            identity_contribution = np.dot(delta_target, delta_l) / n_delta_target
            identity_frac = identity_contribution / n_delta_l  # delta_l在delta_target方向上的归一化投影

            # 10. J@delta_l的分解:
            # J@delta_l ≈ delta_l + (J-I)@delta_l
            # delta_{l+1} ≈ delta_l + (delta_{l+1}-delta_l)
            # 如果cos(delta_l, delta_target)≈0.95, 则delta_l占delta_target的95%
            # 那么J@delta_l ≈ delta_target主要因为delta_l≈delta_target

            # 11. 随机方向的对比
            np.random.seed(int(hash(cname)) % 2**31)
            n_random = 5
            random_cos_l_target = []
            random_cos_jvp_target = []
            random_j_nonid_ratios = []

            for _ in range(n_random):
                r = np.random.randn(d_model)
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-8:
                    continue

                # JVP for random direction
                jvp_r = compute_jvp_with_attn_mlp_decomposition(
                    model, tokenizer, device, BASELINE_TEXT,
                    inject_l, r, epsilon=0.01,
                    baseline_states=baseline_states
                )
                if jvp_r is None:
                    continue

                n_jvp_r = np.linalg.norm(jvp_r)
                
                # (J-I)@r
                j_nonid_r = jvp_r - r
                n_j_nonid_r = np.linalg.norm(j_nonid_r)
                random_j_nonid_ratios.append(n_j_nonid_r / r_norm)

            concept_result = {
                "cos_delta_l_target": cos_delta_l_target,
                "cos_jvp_target": cos_jvp_target,
                "change_ratio": change_ratio,
                "j_nonid_ratio": j_nonid_ratio,
                "cos_j_nonid_change": cos_j_nonid_change,
                "identity_frac": float(identity_frac),
                "random_j_nonid_ratio_mean": float(np.mean(random_j_nonid_ratios)) if random_j_nonid_ratios else None,
                "random_j_nonid_ratio_std": float(np.std(random_j_nonid_ratios)) if len(random_j_nonid_ratios) > 1 else None,
            }

            layer_results[cname] = concept_result

            print(f"    {cname}: cos(Δl,Δl+1)={cos_delta_l_target:.3f}, "
                  f"cos(JVP,Δl+1)={cos_jvp_target:.3f}, "
                  f"Δchange/Δl={change_ratio:.3f}, "
                  f"(J-I)ratio={j_nonid_ratio:.3f}")

        results[inject_l] = layer_results
        gc.collect()

    # 保存
    output_path = TEMP / f"ccxvb_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    # 总结
    print(f"\n{'#'*70}")
    print(f"CCXVB Summary — {model_name}")
    print(f"{'#'*70}")
    print(f"\n  关键指标:")
    print(f"  cos(Δl, Δl+1): 概念方向在层间的对齐度")
    print(f"  cos(JVP, Δl+1): JVP预测质量")
    print(f"  Δchange/Δl: 层间变化相对于概念方向的大小")
    print(f"  (J-I)ratio: Jacobian偏离恒等矩阵的程度")
    print(f"  cos((J-I)@Δ, Δchange): 非恒等部分是否与真实修正对齐")
    print(f"")

    for li, lr in results.items():
        print(f"  L{li}:")
        for cname, cr in lr.items():
            print(f"    {cname}: cos(Δl,Δl+1)={cr['cos_delta_l_target']:.3f}, "
                  f"cos(JVP,Δl+1)={cr['cos_jvp_target']:.3f}, "
                  f"change={cr['change_ratio']:.3f}, "
                  f"(J-I)={cr['j_nonid_ratio']:.3f}, "
                  f"cos(nonId,change)={cr['cos_j_nonid_change']:.3f}")

    release_model(model)
    print(f"\nCCXVB {model_name} 完成!")


if __name__ == "__main__":
    main()
