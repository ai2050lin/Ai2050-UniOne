"""
CCXV(365): Jacobian特征谱与概念通道
======================================

CCXIV发现了最关键的未解问题: 为什么J@delta_l指向delta_{l+1}?
本实验通过测量Jacobian的完整谱结构来回答这个问题。

三个实验:
  Exp1: 随机化SVD — Jacobian的top-k奇异值和输出方向
    方法: 生成n_probe个随机向量, 计算J@v_i, 用SVD提取主成分
    目标: Jacobian的谱结构(奇异值分布, 有效秩, 长尾)

  Exp2: 概念在Jacobian谱中的位置
    方法: 将概念delta投影到Jacobian的奇异向量空间
    目标: 概念是否在低秩"传播通道"中?

  Exp3: W_U行空间与Jacobian对齐
    方法: 计算delta_l和J@delta_l在W_U行空间中的投影比例
    目标: 概念方向是否主要在W_U行空间中? Jacobian是否保持这个空间?

理论框架:
  h_{l+1} = F_l(h_l), J_l = dF_l/dh @ h_baseline
  
  如果delta_l主要在W_U行空间(≈vocab语义空间)中,
  且J@delta_l也主要在W_U行空间中,
  则Jacobian的"概念保持"效应可以由W_U结构解释。

  如果概念方向集中在Jacobian的低秩子空间中,
  则概念的传播是低维动力学的结果。

用法:
  python ccxv_jacobian_eigenspectrum.py --model qwen3 --exp 1
  python ccxv_jacobian_eigenspectrum.py --model qwen3 --exp 2
  python ccxv_jacobian_eigenspectrum.py --model qwen3 --exp 3
  python ccxv_jacobian_eigenspectrum.py --model qwen3 --exp all
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
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
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
    "mountain": {
        "templates": [
            "The word is mountain",
            "A tall mountain",
            "The mountain peak",
            "Climb the mountain",
            "Mountain is high",
        ],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": [
            "The word is ocean",
            "The deep ocean",
            "Ocean waves",
            "Swim in the ocean",
            "Ocean is vast",
        ],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
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


def compute_jvp(model, tokenizer, device, baseline_text, inject_layer, direction,
                epsilon=0.01, capture_layers=None, baseline_states=None):
    """计算Jacobian-Vector Product: J @ v ≈ (F(h + ε*v) - F(h)) / ε"""
    if capture_layers is None:
        capture_layers = [inject_layer + 1]

    all_layers_list = get_layers(model)

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

    jvp_dict = {}
    for li in capture_layers:
        if li in captured and baseline_states is not None and li in baseline_states:
            jvp_dict[li] = (captured[li] - baseline_states[li]) / epsilon

    gc.collect()
    return jvp_dict, captured


# ================================================================
# Exp1: 随机化SVD — Jacobian的谱结构
# ================================================================
def run_exp1(model, tokenizer, device, model_info, key_layers, baseline_states):
    """
    随机化SVD: 用n_probe个随机向量探测Jacobian的谱结构
    
    方法:
    1. 生成n_probe个随机高斯向量 Omega = [v_1, ..., v_n_probe]
    2. 计算 Y = J @ Omega = [J@v_1, ..., J@v_n_probe] (n_probe次forward)
    3. SVD(Y) → 近似Jacobian的奇异值和左奇异向量
    
    这给出Jacobian输出空间的低秩近似。
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Randomized SVD — Jacobian Spectral Structure")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    n_probe = 80  # 探测向量数量 (需要>k来估计top-k)
    
    np.random.seed(42)
    results = {}

    for inject_l in key_layers:
        target_l = inject_l + 1
        if target_l >= n_layers:
            continue
        
        if inject_l not in baseline_states or target_l not in baseline_states:
            print(f"  L{inject_l}: missing baseline states, skipping")
            continue

        print(f"\n  --- Layer {inject_l} → {target_l} ---")
        print(f"  Computing {n_probe} JVPs...")
        t0 = time.time()

        # 生成随机探测向量
        Omega = np.random.randn(d_model, n_probe).astype(np.float32)
        # 归一化每列
        for i in range(n_probe):
            Omega[:, i] /= np.linalg.norm(Omega[:, i])

        # 计算 Y = J @ Omega
        Y = np.zeros((d_model, n_probe), dtype=np.float64)

        for i in range(n_probe):
            v = Omega[:, i]
            jvp_dict, _ = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, v, epsilon=0.01,
                capture_layers=[target_l],
                baseline_states=baseline_states
            )
            if target_l in jvp_dict and jvp_dict[target_l] is not None:
                Y[:, i] = jvp_dict[target_l]
            
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{n_probe} JVPs done ({elapsed:.0f}s)")

        elapsed = time.time() - t0
        print(f"  All {n_probe} JVPs done in {elapsed:.0f}s")

        # SVD of Y
        U_y, s_y, Vt_y = np.linalg.svd(Y, full_matrices=False)
        # U_y: [d_model, n_probe], s_y: [n_probe], Vt_y: [n_probe, n_probe]

        # 奇异值分析
        total_energy = np.sum(s_y ** 2)
        cum_energy = np.cumsum(s_y ** 2) / total_energy

        # 有效秩: energy >= 90% / 95% / 99% 需要多少分量
        rank_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
        rank_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
        rank_99 = int(np.searchsorted(cum_energy, 0.99)) + 1

        # Shannon熵有效秩
        p = s_y ** 2 / total_energy
        entropy = -np.sum(p * np.log(p + 1e-30))
        effective_rank = np.exp(entropy)

        # 奇异值分布统计
        top1 = s_y[0]
        top10_mean = np.mean(s_y[:10])
        top50_mean = np.mean(s_y[:50]) if len(s_y) >= 50 else np.mean(s_y)
        all_mean = np.mean(s_y)
        ratio_top1_top10 = top1 / top10_mean if top10_mean > 0 else 0
        ratio_top10_rest = top10_mean / np.mean(s_y[10:]) if len(s_y) > 10 else 0

        print(f"  Top-1 singular value: {top1:.3f}")
        print(f"  Top-10 mean: {top10_mean:.3f}, Top-50 mean: {top50_mean:.3f}")
        print(f"  Rank-90%: {rank_90}, Rank-95%: {rank_95}, Rank-99%: {rank_99}")
        print(f"  Effective rank (Shannon): {effective_rank:.1f}")
        print(f"  Ratio top1/top10: {ratio_top1_top10:.3f}, top10/rest: {ratio_top10_rest:.3f}")

        # 保存前50个奇异值和对应的左奇异向量
        n_save = min(50, len(s_y))
        results[inject_l] = {
            "singular_values": s_y[:n_save].tolist(),
            "rank_90": rank_90,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "effective_rank": float(effective_rank),
            "top1": float(top1),
            "top10_mean": float(top10_mean),
            "top50_mean": float(top50_mean),
            "ratio_top1_top10": float(ratio_top1_top10),
            "ratio_top10_rest": float(ratio_top10_rest),
            "U_y_top50": U_y[:, :n_save].tolist(),  # [d_model, n_save]
            "cum_energy_top50": cum_energy[:n_save].tolist(),
        }

        del Y, Omega
        gc.collect()

    return results


# ================================================================
# Exp2: 概念在Jacobian谱中的位置
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts, key_layers,
            baseline_states, all_deltas, exp1_results):
    """
    概念方向在Jacobian奇异向量空间中的投影
    
    核心问题: 概念delta是否在Jacobian的低秩"传播通道"中?
    
    如果delta_{l+1}主要在Jacobian的top-k奇异向量空间中,
    则概念传播是低维动力学的结果。
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Concept Position in Jacobian Spectrum")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    results = {}

    for inject_l in key_layers:
        target_l = inject_l + 1
        if target_l >= n_layers:
            continue
        if inject_l not in exp1_results:
            continue

        print(f"\n  --- Layer {inject_l} → {target_l} ---")

        # 加载Jacobian的奇异向量
        U_y = np.array(exp1_results[inject_l]["U_y_top50"])  # [d_model, n_save]
        s_y = np.array(exp1_results[inject_l]["singular_values"])
        n_sv = U_y.shape[1]

        concept_projections = {}
        random_baseline = {}

        # A: 概念方向的投影
        for cname, deltas in all_deltas.items():
            if target_l not in deltas:
                continue
            
            delta = deltas[target_l]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue

            # 投影到U_y的各奇异向量
            proj_coeffs = U_y.T @ delta  # [n_sv]
            proj_energies = proj_coeffs ** 2
            total_proj_energy = np.sum(proj_energies)
            ratio_in_topk = total_proj_energy / (delta_norm ** 2)

            # 在top-5, top-10, top-20, top-50中的能量比
            cum_proj = np.cumsum(proj_energies) / total_proj_energy
            in_top5 = float(np.sum(proj_energies[:5]) / (delta_norm ** 2))
            in_top10 = float(np.sum(proj_energies[:10]) / (delta_norm ** 2))
            in_top20 = float(np.sum(proj_energies[:20]) / (delta_norm ** 2))
            in_top50 = float(np.sum(proj_energies[:50]) / (delta_norm ** 2)) if n_sv >= 50 else float(total_proj_energy / (delta_norm ** 2))

            # 概念与各奇异向量的余弦
            top_cosines = []
            for k in range(min(10, n_sv)):
                cos_k = float(np.dot(delta, U_y[:, k]) / (delta_norm * np.linalg.norm(U_y[:, k])))
                top_cosines.append(cos_k)

            # 与delta_l的投影 (也在inject_l层的奇异空间中)
            inject_proj = {}
            if inject_l in deltas:
                delta_l = deltas[inject_l]
                delta_l_norm = np.linalg.norm(delta_l)
                if delta_l_norm > 1e-8:
                    # J@delta_l的计算
                    jvp_dict, _ = compute_jvp(
                        model, tokenizer, device, BASELINE_TEXT,
                        inject_l, delta_l, epsilon=0.01,
                        capture_layers=[target_l],
                        baseline_states=baseline_states
                    )
                    if target_l in jvp_dict and jvp_dict[target_l] is not None:
                        jvp_result = jvp_dict[target_l]
                        jvp_norm = np.linalg.norm(jvp_result)
                        
                        # J@delta_l在奇异向量空间中的投影
                        jvp_proj_coeffs = U_y.T @ jvp_result
                        jvp_proj_energy = np.sum(jvp_proj_coeffs ** 2)
                        jvp_ratio_in_topk = jvp_proj_energy / (jvp_norm ** 2) if jvp_norm > 1e-8 else 0
                        
                        inject_proj = {
                            "jvp_norm": float(jvp_norm),
                            "jvp_ratio_in_U": float(jvp_ratio_in_topk),
                            "jvp_in_top10": float(np.sum(jvp_proj_coeffs[:10] ** 2) / (jvp_norm ** 2)) if jvp_norm > 1e-8 else 0,
                            "jvp_in_top20": float(np.sum(jvp_proj_coeffs[:20] ** 2) / (jvp_norm ** 2)) if jvp_norm > 1e-8 else 0,
                        }

            concept_projections[cname] = {
                "delta_norm": float(delta_norm),
                "ratio_in_U": float(ratio_in_topk),
                "in_top5": in_top5,
                "in_top10": in_top10,
                "in_top20": in_top20,
                "in_top50": in_top50,
                "top_cosines": top_cosines,
                "cum_proj_top20": cum_proj[:20].tolist(),
                "jvp_analysis": inject_proj,
            }

            print(f"    {cname}: in_top10={in_top10:.3f}, in_top50={in_top50:.3f}, "
                  f"top1_cos={top_cosines[0]:.3f}")

        # B: 随机方向的基线投影
        np.random.seed(123)
        n_random = 20
        random_in_top10 = []
        random_in_top50 = []
        random_ratio_in_U = []

        for _ in range(n_random):
            r = np.random.randn(d_model)
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-8:
                continue
            r_proj = U_y.T @ r
            r_proj_energy = np.sum(r_proj ** 2)
            random_ratio_in_U.append(r_proj_energy / (r_norm ** 2))
            random_in_top10.append(np.sum(r_proj[:10] ** 2) / (r_norm ** 2))
            random_in_top50.append(np.sum(r_proj[:50] ** 2) / (r_norm ** 2)) if n_sv >= 50 else random_in_top50.append(r_proj_energy / (r_norm ** 2))

        random_baseline = {
            "ratio_in_U_mean": float(np.mean(random_ratio_in_U)),
            "ratio_in_U_std": float(np.std(random_ratio_in_U)),
            "in_top10_mean": float(np.mean(random_in_top10)),
            "in_top10_std": float(np.std(random_in_top10)),
            "in_top50_mean": float(np.mean(random_in_top50)),
            "in_top50_std": float(np.std(random_in_top50)),
        }

        print(f"    Random baseline: in_top10={random_baseline['in_top10_mean']:.4f}±{random_baseline['in_top10_std']:.4f}, "
              f"in_top50={random_baseline['in_top50_mean']:.4f}±{random_baseline['in_top50_std']:.4f}")

        results[inject_l] = {
            "concept_projections": concept_projections,
            "random_baseline": random_baseline,
            "n_singular_vectors": n_sv,
        }

    return results


# ================================================================
# Exp3: W_U行空间与Jacobian对齐
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts, key_layers,
            baseline_states, all_deltas):
    """
    W_U行空间分析 — 概念方向是否在vocab语义空间中?
    
    核心问题:
    1. delta_l在W_U行空间中的投影比例
    2. J@delta_l是否保持W_U行空间
    3. W_U行空间能否解释Jacobian的"概念保持"效应?
    
    W_U行空间 = lm_head权重的行张成的子空间
    这是"vocab语义空间": 任何能影响logit输出的方向必须在这个空间中
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: W_U Row Space & Jacobian Alignment")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers

    # 获取W_U并计算其行空间基
    print(f"\n  Computing W_U row space basis...")
    W_U = get_W_U(model)  # [vocab_size, d_model]
    print(f"  W_U shape: {W_U.shape}")

    # SVD of W_U^T: U_wu [d_model, k] 是W_U行空间的基
    from scipy.sparse.linalg import svds
    n_wu_components = 200  # 足够覆盖主要子空间
    k_wu = min(n_wu_components, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)  # [d_model, k_wu]

    # W_U行空间的累积能量
    total_wu_energy = np.sum(s_wu ** 2)
    cum_wu_energy = np.cumsum(s_wu ** 2) / total_wu_energy
    wu_rank_90 = int(np.searchsorted(cum_wu_energy, 0.90)) + 1
    wu_rank_95 = int(np.searchsorted(cum_wu_energy, 0.95)) + 1
    wu_rank_99 = int(np.searchsorted(cum_wu_energy, 0.99)) + 1

    print(f"  W_U row space: rank_90%={wu_rank_90}, rank_95%={wu_rank_95}, rank_99%={wu_rank_99}")
    print(f"  Top-10 singular values: {s_wu[:10].round(2)}")

    del W_U
    gc.collect()

    results = {
        "wu_row_space": {
            "n_components": k_wu,
            "rank_90": wu_rank_90,
            "rank_95": wu_rank_95,
            "rank_99": wu_rank_99,
            "singular_values_top20": s_wu[:20].tolist(),
            "cum_energy_top20": cum_wu_energy[:20].tolist(),
        }
    }

    # 对每层和每个概念分析
    layer_results = {}

    for inject_l in key_layers:
        target_l = inject_l + 1
        if target_l >= n_layers:
            continue

        print(f"\n  --- Layer {inject_l} → {target_l} ---")

        concept_wu = {}

        for cname, deltas in all_deltas.items():
            if inject_l not in deltas or target_l not in deltas:
                continue

            delta_l = deltas[inject_l]
            delta_target = deltas[target_l]
            
            # A: delta_l在W_U行空间中的投影
            proj_l = U_wu.T @ delta_l  # [k_wu]
            proj_l_energy = np.sum(proj_l ** 2)
            delta_l_norm_sq = np.linalg.norm(delta_l) ** 2
            ratio_l = proj_l_energy / delta_l_norm_sq if delta_l_norm_sq > 1e-16 else 0

            # 在top-5, top-10, top-50 W_U奇异向量中的投影
            in_wu_top5 = np.sum(proj_l[:5] ** 2) / delta_l_norm_sq if delta_l_norm_sq > 1e-16 else 0
            in_wu_top10 = np.sum(proj_l[:10] ** 2) / delta_l_norm_sq if delta_l_norm_sq > 1e-16 else 0
            in_wu_top50 = np.sum(proj_l[:50] ** 2) / delta_l_norm_sq if delta_l_norm_sq > 1e-16 else 0

            # B: delta_{l+1}在W_U行空间中的投影
            proj_target = U_wu.T @ delta_target
            proj_target_energy = np.sum(proj_target ** 2)
            delta_target_norm_sq = np.linalg.norm(delta_target) ** 2
            ratio_target = proj_target_energy / delta_target_norm_sq if delta_target_norm_sq > 1e-16 else 0

            in_wu_target_top10 = np.sum(proj_target[:10] ** 2) / delta_target_norm_sq if delta_target_norm_sq > 1e-16 else 0
            in_wu_target_top50 = np.sum(proj_target[:50] ** 2) / delta_target_norm_sq if delta_target_norm_sq > 1e-16 else 0

            # C: J@delta_l在W_U行空间中的投影
            jvp_dict, _ = compute_jvp(
                model, tokenizer, device, BASELINE_TEXT,
                inject_l, delta_l, epsilon=0.01,
                capture_layers=[target_l],
                baseline_states=baseline_states
            )

            jvp_wu_ratio = None
            jvp_preserves_wu = None
            if target_l in jvp_dict and jvp_dict[target_l] is not None:
                jvp_result = jvp_dict[target_l]
                jvp_norm_sq = np.linalg.norm(jvp_result) ** 2
                proj_jvp = U_wu.T @ jvp_result
                proj_jvp_energy = np.sum(proj_jvp ** 2)
                jvp_wu_ratio = proj_jvp_energy / jvp_norm_sq if jvp_norm_sq > 1e-16 else 0

                # W_U保持度: 如果delta_l和J@delta_l都在W_U行空间中, 则Jacobian保持W_U空间
                jvp_preserves_wu = jvp_wu_ratio / max(ratio_l, 1e-8)

            # D: delta的"残余"方向(不在W_U行空间中的分量)
            residual_l = delta_l - U_wu @ proj_l
            residual_l_ratio = np.linalg.norm(residual_l) ** 2 / delta_l_norm_sq if delta_l_norm_sq > 1e-16 else 0

            concept_wu[cname] = {
                "delta_l_wu_ratio": float(ratio_l),
                "delta_l_in_wu_top5": float(in_wu_top5),
                "delta_l_in_wu_top10": float(in_wu_top10),
                "delta_l_in_wu_top50": float(in_wu_top50),
                "delta_target_wu_ratio": float(ratio_target),
                "delta_target_in_wu_top10": float(in_wu_target_top10),
                "delta_target_in_wu_top50": float(in_wu_target_top50),
                "jvp_wu_ratio": float(jvp_wu_ratio) if jvp_wu_ratio is not None else None,
                "jvp_preserves_wu": float(jvp_preserves_wu) if jvp_preserves_wu is not None else None,
                "residual_l_ratio": float(residual_l_ratio),
            }

            print(f"    {cname}: delta_l→W_U={ratio_l:.3f}, delta_target→W_U={ratio_target:.3f}, "
                  f"JVP→W_U={jvp_wu_ratio:.3f}" if jvp_wu_ratio is not None else
                  f"    {cname}: delta_l→W_U={ratio_l:.3f}, delta_target→W_U={ratio_target:.3f}")

        # E: 随机方向的基线
        np.random.seed(456)
        n_random = 20
        random_wu_ratios = []
        for _ in range(n_random):
            r = np.random.randn(d_model)
            r_norm_sq = np.linalg.norm(r) ** 2
            if r_norm_sq < 1e-16:
                continue
            proj_r = U_wu.T @ r
            random_wu_ratios.append(np.sum(proj_r ** 2) / r_norm_sq)

        layer_results[inject_l] = {
            "concept_wu": concept_wu,
            "random_wu_ratio_mean": float(np.mean(random_wu_ratios)),
            "random_wu_ratio_std": float(np.std(random_wu_ratios)),
        }

        print(f"    Random→W_U: {np.mean(random_wu_ratios):.4f}±{np.std(random_wu_ratios):.4f}")

    results["layers"] = layer_results
    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXV: Jacobian Eigen-Spectrum & Concept Channel — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    key_layers = [l for l in [6, 12, 18, 24, 30] if l + 1 < n_layers]

    # 收集baseline状态
    all_capture = set()
    for l in key_layers:
        all_capture.add(l)
        all_capture.add(l + 1)
    all_capture = sorted([l for l in all_capture if l < n_layers])

    print(f"\n  Collecting baseline states at {len(all_capture)} layers...")
    baseline_states, baseline_logits = collect_states_at_layers(
        model, tokenizer, device, BASELINE_TEXT, all_capture
    )

    # 收集概念deltas
    print(f"  Collecting concept deltas...")
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

    all_results = {}
    exp1_results = None

    if args.exp in ["1", "all"]:
        exp1_results = run_exp1(model, tokenizer, device, model_info, key_layers, baseline_states)
        all_results["exp1"] = exp1_results
        # 保存中间结果 (exp1的数据量大, 防止丢失)
        exp1_save = {}
        for k, v in exp1_results.items():
            exp1_save[str(k)] = {kk: vv for kk, vv in v.items() if kk != "U_y_top50"}
        all_results["exp1_summary"] = exp1_save

    if args.exp in ["2", "all"]:
        if exp1_results is None:
            # 尝试加载之前的结果
            prev_path = TEMP / f"ccxv_{model_name}_results.json"
            if prev_path.exists():
                with open(prev_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if "exp1" in prev:
                    # 需要恢复U_y_top50
                    print("  Warning: Exp1 U_y_top50 not available from saved results. Re-running Exp1...")
                    exp1_results = run_exp1(model, tokenizer, device, model_info, key_layers, baseline_states)
            else:
                exp1_results = run_exp1(model, tokenizer, device, model_info, key_layers, baseline_states)

        exp2_results = run_exp2(
            model, tokenizer, device, model_info, CONCEPTS, key_layers,
            baseline_states, all_deltas, exp1_results
        )
        all_results["exp2"] = exp2_results

    if args.exp in ["3", "all"]:
        exp3_results = run_exp3(
            model, tokenizer, device, model_info, CONCEPTS, key_layers,
            baseline_states, all_deltas
        )
        all_results["exp3"] = exp3_results

    # 保存结果 (去掉大型矩阵)
    save_results = {}
    for exp_name, exp_data in all_results.items():
        if exp_name == "exp1":
            # 去掉U_y_top50 (太大), 保留其他
            save_results["exp1"] = {}
            for layer_key, layer_data in exp_data.items():
                save_results["exp1"][str(layer_key)] = {
                    k: v for k, v in layer_data.items() if k != "U_y_top50"
                }
        else:
            save_results[exp_name] = exp_data

    output_path = TEMP / f"ccxv_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    # 总结
    print(f"\n{'#'*70}")
    print(f"CCXV Summary — {model_name}")
    print(f"{'#'*70}")

    if "exp1" in all_results:
        print(f"\n  Exp1: Jacobian Spectral Structure")
        for li, lr in all_results["exp1"].items():
            print(f"    L{li}: top1={lr['top1']:.1f}, top10_mean={lr['top10_mean']:.1f}, "
                  f"rank_90={lr['rank_90']}, rank_95={lr['rank_95']}, "
                  f"eff_rank={lr['effective_rank']:.1f}")

    if "exp2" in all_results:
        print(f"\n  Exp2: Concept Position in Jacobian Spectrum")
        for li, lr in all_results["exp2"].items():
            for cname, cp in lr["concept_projections"].items():
                print(f"    L{li} {cname}: in_top10={cp['in_top10']:.3f}, "
                      f"in_top50={cp['in_top50']:.3f}")
            rb = lr["random_baseline"]
            print(f"    L{li} Random: in_top10={rb['in_top10_mean']:.4f}±{rb['in_top10_std']:.4f}")

    if "exp3" in all_results:
        print(f"\n  Exp3: W_U Row Space Alignment")
        wu_info = all_results["exp3"].get("wu_row_space", {})
        print(f"    W_U rank_90={wu_info.get('rank_90')}, rank_95={wu_info.get('rank_95')}")
        for li, lr in all_results["exp3"].get("layers", {}).items():
            for cname, cw in lr["concept_wu"].items():
                print(f"    L{li} {cname}: delta_l→WU={cw['delta_l_wu_ratio']:.3f}, "
                      f"delta_target→WU={cw['delta_target_wu_ratio']:.3f}, "
                      f"JVP→WU={cw['jvp_wu_ratio']:.3f}" if cw['jvp_wu_ratio'] is not None else
                      f"    L{li} {cname}: delta_l→WU={cw['delta_l_wu_ratio']:.3f}")
            print(f"    L{li} Random→WU={lr['random_wu_ratio_mean']:.4f}±{lr['random_wu_ratio_std']:.4f}")

    release_model(model)
    print(f"\nCCXV {model_name} 完成!")


if __name__ == "__main__":
    main()
