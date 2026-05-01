"""
CCXLVII(347): 统一归一化验证 + 面内流形方向分析
=================================================
★★★★★ CCXLV-CCXLVI发现:
  1. fit_r2从0.88-0.98降到0.57-0.59(归一化后)
  2. "边对齐"是低维投影伪影
  3. 层间变换几乎完全切向(tangential=0.88-1.00)

★★★★★ 本实验解决两大问题:
  问题1: fit_r2为什么从0.88降到0.57?
    - CCXXXIX用"去均值+Procrustes对齐"得到0.88
    - CCXLVI用"去均值+L2归一化+Procrustes对齐"得到0.57
    - 需要搞清楚哪个是正确的

  问题2: 切向变换在面上具体沿什么方向?
    - 之前只确认了"不在径向"=在切向
    - 但切向是一个(D-1)维子空间, 方向不确定
    - 需要分析面内的主切向方向

方法论 — 统一的fit_r2计算:
  Step 1: 收集N个类别的残差中心 {c_1, ..., c_N}
  Step 2: 去均值: c_i' = c_i - mean(c)
  Step 3: **不做L2归一化** — 因为归一化会扭曲距离关系!
  Step 4: SVD投影到N-1维子空间
  Step 5: 构造正则单纯形(与数据同尺度)
  Step 6: Procrustes对齐
  Step 7: fit_r2 = 1 - ||aligned - regular||² / ||regular||²

  ★★★ 关键: 不归一化, 保持原始距离关系!
  之前CCXXXIX的0.88可能就是正确值
  CCXLVI的0.57是因为归一化抹掉了尺度信息

面内方向分析:
  Step 1: 在最佳层计算N个类中心
  Step 2: 去均值+SVD投影到N-1维
  Step 3: 对每个类c, 计算切向子空间:
    - 径向方向: r_c = c_centered / ||c_centered||
    - 切向子空间: 与r_c正交的(D-2)维子空间(在N-1维空间内)
  Step 4: 收集所有强度轨迹, 投影到切向子空间
  Step 5: PCA分析切向投影的主方向
  Step 6: 与单纯形面的法线比较 → 轨迹是否沿面的特定方向?

附加: 随机基线
  - 对随机高斯聚类(同样N, 同样d)计算fit_r2
  - 确保我们的fit_r2显著高于随机

用法:
  python ccxlvii_unified_simplex_tangential.py --model qwen3
  python ccxlvii_unified_simplex_tangential.py --model glm4
  python ccxlvii_unified_simplex_tangential.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxlvii_unified_log.txt"

# ============================================================
# 语义类别定义 — 4个情感 + N=6测试
# ============================================================

EMOTION_4 = {
    "classes": {
        "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                  "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
        "sad":   ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                  "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
        "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                  "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
        "scared":["fear", "terror", "dread", "panic", "fright", "horror",
                  "anxiety", "apprehension", "trepidation", "phobia", "alarm", "consternation"],
    },
    "order": ["happy", "sad", "angry", "scared"],
    "prompt": "The person felt {word} about the",
}

EMOTION_6 = {
    "classes": {
        "happy":   ["joy", "delight", "bliss", "glee", "cheer", "elation",
                    "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
        "sad":     ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
        "angry":   ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                    "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
        "scared":  ["fear", "terror", "dread", "panic", "fright", "horror",
                    "anxiety", "apprehension", "trepidation", "phobia", "alarm", "consternation"],
        "surprise":["astonishment", "amazement", "wonder", "shock", "stunned", "staggered",
                    "bewilderment", "awe", "disbelief", "startlement", "astonished", "dumbfounded"],
        "disgust": ["revulsion", "repugnance", "loathing", "abhorrence", "nausea", "aversion",
                    "distaste", "repulsion", "contempt", "dislike", "antipathy", "abomination"],
    },
    "order": ["happy", "sad", "angry", "scared", "surprise", "disgust"],
    "prompt": "The person felt {word} about the",
}

INTENSITY_MODIFIERS = {
    "mild": ["slightly", "somewhat", "a bit", "mildly", "faintly"],
    "strong": ["very", "extremely", "intensely", "deeply", "profoundly"],
}

# ============================================================
# 核心算法
# ============================================================

def construct_regular_simplex(N, scale=1.0):
    """
    构造N个点在R^{N-1}中的正则单纯形, 中心在原点
    
    方法: Gram矩阵法
    正则单纯形的Gram矩阵 G[i,j] = a^2 * (1 + delta_{ij} * N) / N
    其中a是边长, 这里用单位球上的点(距中心等距)
    
    简化: 用标准的Householder反射法
    """
    if N <= 1:
        return np.array([[0.0]])
    
    if N == 2:
        v = np.array([[-1.0], [1.0]]) * scale
        return v - np.mean(v, axis=0)

    D = N - 1  # 子空间维度
    
    # 方法: 先构造Gram矩阵, 然后Cholesky分解
    # 正则单纯形的顶点满足:
    # <v_i, v_j> = r^2 * (cos(theta)) 对于i≠j
    # <v_i, v_i> = r^2
    # 其中cos(theta) = -1/N (正则单纯形的内积)
    
    # 构造Gram矩阵
    r = 1.0  # 半径
    G = np.full((N, N), -r**2 / N)
    np.fill_diagonal(G, r**2 * (N - 1) / N)
    
    # Cholesky分解得到顶点坐标
    try:
        L = np.linalg.cholesky(G)
        vertices = L  # [N, N]
    except np.linalg.LinAlgError:
        # 如果Cholesky失败, 用SVD
        U, S, Vt = np.linalg.svd(G)
        vertices = U @ np.diag(np.sqrt(np.maximum(S, 0)))[:, :D]
        vertices = vertices[:, :D]
    
    # 只取前D列(N-1维)
    vertices = vertices[:, :D]
    
    # 去均值(确保中心在原点)
    vertices = vertices - np.mean(vertices, axis=0)
    
    # 归一化到指定scale
    current_scale = np.linalg.norm(vertices[0])
    if current_scale > 1e-10:
        vertices = vertices * scale / current_scale
    
    return vertices


def compute_simplex_fit_unified(centers, class_order, normalize=False):
    """
    统一的单纯形拟合方法 — CCXLVII核心
    
    关键区别: 默认不做L2归一化(保持距离关系)
    
    Returns:
        dict with fit_r2, proj_centers, regular_vertices, procrustes_R, etc.
    """
    N = len(class_order)
    D = N - 1  # 子空间维度

    center_mat = np.array([centers[c] for c in class_order])

    # Step 1: 去均值
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center

    # Step 2: 可选归一化
    if normalize:
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        centered_for_fit = centered / norms
    else:
        centered_for_fit = centered

    # Step 3: SVD投影到N-1维
    U, S, Vt = svd(centered_for_fit, full_matrices=False)
    proj_matrix = Vt[:D]  # [D, d_model]
    proj_centers = centered_for_fit @ proj_matrix.T  # [N, D]

    # Step 4: 构造正则单纯形(与数据同尺度)
    data_scale = np.linalg.norm(proj_centers[0])
    regular = construct_regular_simplex(N, scale=data_scale)

    # Step 5: Procrustes对齐
    R, scale_procrustes = orthogonal_procrustes(proj_centers, regular)
    aligned = proj_centers @ R

    # Step 6: 计算fit_r2
    residual = aligned - regular
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((regular - np.mean(regular, axis=0)) ** 2)
    fit_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0

    return {
        "fit_r2": float(fit_r2),
        "proj_centers": proj_centers,
        "regular_vertices": regular,
        "aligned_vertices": aligned,
        "procrustes_R": R,
        "proj_matrix": proj_matrix,
        "singular_values": S,
        "global_center": global_center,
        "data_scale": float(data_scale),
        "scale_procrustes": float(scale_procrustes),
    }


def compute_simplex_metrics(centers, class_order, fit_result):
    """计算单纯形质量指标"""
    N = len(class_order)
    proj = fit_result["proj_centers"]

    # 理想角: arccos(-1/N)
    ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi

    # 实际角度
    center_pt = np.mean(proj, axis=0)
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            v_i = proj[i] - center_pt
            v_j = proj[j] - center_pt
            cos = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-10)
            cos = np.clip(cos, -1, 1)
            angles.append(np.arccos(cos) * 180 / np.pi)

    actual_angle = np.mean(angles)

    # 边长变异系数
    dists = pdist(proj)
    cv = np.std(dists) / (np.mean(dists) + 1e-10)

    # 半径均匀性
    radii = [np.linalg.norm(proj[i] - center_pt) for i in range(N)]
    radius_cv = np.std(radii) / (np.mean(radii) + 1e-10)

    # 等距比: min_dist / max_dist
    iso_ratio = np.min(dists) / (np.max(dists) + 1e-10)

    return {
        "ideal_angle": float(ideal_angle),
        "actual_angle": float(actual_angle),
        "angle_deviation": float(abs(actual_angle - ideal_angle)),
        "edge_cv": float(cv),
        "radius_cv": float(radius_cv),
        "isoperimetric_ratio": float(iso_ratio),
        "n_vertices": N,
        "dim": N - 1,
    }


def random_baseline_fit_r2(N, n_words_per_class, d_model, n_trials=1000):
    """
    随机基线: 如果N个类中心是从随机高斯分布采样的, fit_r2是多少?
    
    Args:
        N: 类别数
        n_words_per_class: 每类词数
        d_model: 模型维度
        n_trials: Monte Carlo次数
    
    Returns:
        dict with mean, std, p05, p50, p95
    """
    fit_r2s = []
    for _ in range(n_trials):
        # 模拟: 每类n_words个词, 从N(d_model)中采样
        # 然后取每类的均值作为类中心
        # 加一点类内相关(类中心偏移)
        class_offset = np.random.randn(N, d_model) * 2.0  # 类间差异
        within_noise = np.random.randn(N * n_words_per_class, d_model)  # 类内噪声

        centers = {}
        class_names = [f"c{i}" for i in range(N)]
        for i, cn in enumerate(class_names):
            centers[cn] = class_offset[i]  # 用类中心(简化)

        fit = compute_simplex_fit_unified(centers, class_names, normalize=False)
        fit_r2s.append(fit["fit_r2"])

    fit_r2s = np.array(fit_r2s)
    return {
        "mean": float(np.mean(fit_r2s)),
        "std": float(np.std(fit_r2s)),
        "p05": float(np.percentile(fit_r2s, 5)),
        "p50": float(np.percentile(fit_r2s, 50)),
        "p95": float(np.percentile(fit_r2s, 95)),
        "n_trials": n_trials,
    }


def compute_tangential_direction_analysis(centers, class_order, fit_result,
                                          intensity_data=None):
    """
    面内流形方向分析 — CCXLVII核心创新
    
    分析切向变换在面上的具体方向
    """
    N = len(class_order)
    D = N - 1
    proj = fit_result["proj_centers"]
    center_pt = np.mean(proj, axis=0)

    results = {}

    for i, cls in enumerate(class_order):
        # 径向方向
        radial = proj[i] - center_pt
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1e-10:
            continue
        radial_dir = radial / radial_norm

        # 切向子空间: 在D维空间中, 与radial_dir正交的(D-1)维子空间
        # 方法: 从D维标准基中减去径向分量, 然后QR
        basis = np.eye(D)
        # 移除径向分量
        basis_orth = basis - np.outer(basis @ radial_dir, radial_dir)
        # QR分解得到正交基
        Q, R = np.linalg.qr(basis_orth.T)  # [D, D]
        # 取rank=D-1的列
        rank = np.sum(np.abs(np.diag(R)) > 1e-10)
        tangential_basis = Q[:, :rank-1] if rank > 1 else Q[:, :1]  # [D, D-1]

        results[cls] = {
            "radial_dir": radial_dir,
            "tangential_basis": tangential_basis,
            "radial_norm": float(radial_norm),
        }

    return results


def compute_intensity_trajectory(centers_strong, centers_mild, class_order, fit_result):
    """
    计算强度轨迹方向并分析其在单纯形面上的方向
    """
    N = len(class_order)
    D = N - 1
    proj_matrix = fit_result["proj_matrix"]

    trajectories = {}
    for cls in class_order:
        if cls not in centers_strong or cls not in centers_mild:
            continue

        # 强度方向(原始空间)
        delta = centers_strong[cls] - centers_mild[cls]
        delta_norm = np.linalg.norm(delta)

        if delta_norm < 1e-10:
            continue

        # 投影到N-1维子空间
        proj_delta = delta @ proj_matrix.T  # [D]

        # 径向方向(投影空间中)
        cls_idx = class_order.index(cls)
        center_pt = np.mean(fit_result["proj_centers"], axis=0)
        radial = fit_result["proj_centers"][cls_idx] - center_pt
        radial_norm = np.linalg.norm(radial)

        if radial_norm < 1e-10:
            continue

        radial_dir = radial / radial_norm

        # 轨迹在投影空间的方向
        proj_delta_norm = np.linalg.norm(proj_delta)
        if proj_delta_norm < 1e-10:
            continue

        proj_delta_dir = proj_delta / proj_delta_norm

        # 径向对齐
        radial_align = abs(np.dot(proj_delta_dir, radial_dir))

        # 切向分量
        tangential_component = proj_delta_dir - np.dot(proj_delta_dir, radial_dir) * radial_dir
        tangential_norm = np.linalg.norm(tangential_component)

        # 切向方向(归一化)
        if tangential_norm > 1e-10:
            tangential_dir = tangential_component / tangential_norm
        else:
            tangential_dir = np.zeros(D)

        # 分析切向方向与哪些其他类的径向方向相关
        # 这告诉我们轨迹在面上朝哪个类移动
        other_class_aligns = {}
        for j, other_cls in enumerate(class_order):
            if other_cls == cls:
                continue
            other_radial = fit_result["proj_centers"][j] - center_pt
            other_radial_norm = np.linalg.norm(other_radial)
            if other_radial_norm < 1e-10:
                continue
            other_radial_dir = other_radial / other_radial_norm

            # 切向方向在面上, 与其他类径向方向的点积
            # 注意: 径向方向不在切向子空间中, 但可以分解
            # 我们计算: tangential_dir在other_radial_dir上的投影
            # = |tangential_dir| * |other_radial_dir| * cos(angle)
            # 但更精确: 我们应该把other_radial_dir也分解为径向+切向
            # 简化: 直接用tangential_dir与(其他类径向-自身径向分量)的对齐
            
            # 方法: 计算other类的方向在自身切向子空间中的分量
            # other_dir_tangential = other_radial_dir - <other_radial_dir, radial_dir> * radial_dir
            other_tang = other_radial_dir - np.dot(other_radial_dir, radial_dir) * radial_dir
            other_tang_norm = np.linalg.norm(other_tang)
            if other_tang_norm > 1e-10:
                other_tang_dir = other_tang / other_tang_norm
                align = np.dot(tangential_dir, other_tang_dir)
            else:
                align = 0.0
            other_class_aligns[other_cls] = float(align)

        # 找最接近的类
        if other_class_aligns:
            closest_cls = max(other_class_aligns, key=other_class_aligns.get)
            closest_align = other_class_aligns[closest_cls]
        else:
            closest_cls = "none"
            closest_align = 0

        trajectories[cls] = {
            "radial_alignment": float(radial_align),
            "tangential_norm": float(tangential_norm),
            "tangential_fraction": float(tangential_norm ** 2),
            "closest_other_class": closest_cls,
            "closest_align": float(closest_align),
            "other_class_aligns": other_class_aligns,
        }

    return trajectories


# ============================================================
# 数据收集
# ============================================================

def collect_residuals(model, tokenizer, device, words, prompt_template, layers, batch_size=8):
    """收集指定词在指定层的残差"""
    all_residuals = {l: [] for l in layers}

    for start in range(0, len(words), batch_size):
        batch_words = words[start:start + batch_size]
        texts = [prompt_template.format(word=w) for w in batch_words]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for l in layers:
            # 取最后一个非padding token的残差
            hs = hidden_states[l].detach().cpu().float().numpy()
            for b in range(len(batch_words)):
                # 找最后一个非padding位置
                attn_mask = inputs["attention_mask"][b].cpu().numpy()
                last_pos = np.where(attn_mask > 0)[0][-1]
                all_residuals[l].append(hs[b, last_pos])

        del outputs, hidden_states
        gc.collect()

    return all_residuals


def compute_class_centers(residuals, class_to_indices):
    """计算各类的中心"""
    centers = {}
    for cls, indices in class_to_indices.items():
        vecs = np.array([residuals[i] for i in indices])
        centers[cls] = np.mean(vecs, axis=0)
    return centers


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name, args):
    """运行单个模型的所有实验"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 选取层: 采样8-10层
    if n_layers <= 10:
        layers = list(range(n_layers))
    else:
        step = n_layers / 10
        layers = sorted(set(int(i * step) for i in range(10)))

    log(f"\n{'='*70}")
    log(f"模型: {model_name} ({info.model_class})")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  采样层: {layers}")

    all_results = {}

    # ============================================================
    # 实验1: 统一fit_r2验证 — N=4和N=6
    # ============================================================
    log(f"\n{'='*70}")
    log(f"实验1: 统一fit_r2验证 — 归一化 vs 非归一化对比")
    log(f"{'='*70}")

    for domain_name, domain in [("emotion4", EMOTION_4), ("emotion6", EMOTION_6)]:
        log(f"\n--- 领域: {domain_name} (N={len(domain['order'])}) ---")

        class_order = domain["order"]
        all_words = []
        class_to_indices = {}

        for cls in class_order:
            words = domain["classes"][cls]
            start_idx = len(all_words)
            class_to_indices[cls] = list(range(start_idx, start_idx + len(words)))
            all_words.extend(words)

        # 收集所有层的残差
        all_residuals = collect_residuals(
            model, tokenizer, device, all_words, domain["prompt"], layers
        )

        # 每层计算两种fit_r2
        layer_results = []
        for l in layers:
            centers = compute_class_centers(all_residuals[l], class_to_indices)

            # 方法A: 不归一化(保持距离关系) — 这应该是正确方法
            fit_raw = compute_simplex_fit_unified(centers, class_order, normalize=False)
            metrics_raw = compute_simplex_metrics(centers, class_order, fit_raw)

            # 方法B: L2归一化(之前CCXLVI的方法)
            fit_norm = compute_simplex_fit_unified(centers, class_order, normalize=True)
            metrics_norm = compute_simplex_metrics(centers, class_order, fit_norm)

            layer_results.append({
                "layer": l,
                "fit_r2_raw": fit_raw["fit_r2"],
                "fit_r2_norm": fit_norm["fit_r2"],
                "data_scale": fit_raw["data_scale"],
                "singular_values": fit_raw["singular_values"][:5].tolist(),
                "angle_dev_raw": metrics_raw["angle_deviation"],
                "angle_dev_norm": metrics_norm["angle_deviation"],
                "edge_cv_raw": metrics_raw["edge_cv"],
                "edge_cv_norm": metrics_norm["edge_cv"],
            })

            log(f"  L{l:2d}: fit_r2_raw={fit_raw['fit_r2']:.4f}  "
                f"fit_r2_norm={fit_norm['fit_r2']:.4f}  "
                f"scale={fit_raw['data_scale']:.1f}  "
                f"angle_dev_raw={metrics_raw['angle_deviation']:.1f}°")

        # 找最佳层
        best_raw = max(layer_results, key=lambda x: x["fit_r2_raw"])
        best_norm = max(layer_results, key=lambda x: x["fit_r2_norm"])

        log(f"\n  ★ 最佳层(不归一化): L{best_raw['layer']}  "
            f"fit_r2_raw={best_raw['fit_r2_raw']:.4f}")
        log(f"  ★ 最佳层(L2归一化): L{best_norm['layer']}  "
            f"fit_r2_norm={best_norm['fit_r2_norm']:.4f}")

        all_results[domain_name] = {
            "layer_results": layer_results,
            "best_raw": best_raw,
            "best_norm": best_norm,
        }

    # ============================================================
    # 实验2: 随机基线
    # ============================================================
    log(f"\n{'='*70}")
    log(f"实验2: 随机基线 — fit_r2在随机数据上的分布")
    log(f"{'='*70}")

    for N in [4, 6]:
        baseline = random_baseline_fit_r2(N, 12, d_model, n_trials=500)
        log(f"\n  N={N} (D={N-1}, d_model={d_model}):")
        log(f"    随机fit_r2: mean={baseline['mean']:.4f} ± {baseline['std']:.4f}")
        log(f"    P5={baseline['p05']:.4f}  P50={baseline['p50']:.4f}  P95={baseline['p95']:.4f}")

        domain_name = f"emotion{N}"
        if domain_name in all_results:
            actual_raw = all_results[domain_name]["best_raw"]["fit_r2_raw"]
            actual_norm = all_results[domain_name]["best_norm"]["fit_r2_norm"]
            z_raw = (actual_raw - baseline['mean']) / (baseline['std'] + 1e-10)
            z_norm = (actual_norm - baseline['mean']) / (baseline['std'] + 1e-10)
            log(f"    实际fit_r2_raw={actual_raw:.4f} → z={z_raw:.1f}σ")
            log(f"    实际fit_r2_norm={actual_norm:.4f} → z={z_norm:.1f}σ")

        all_results[f"baseline_N{N}"] = baseline

    # ============================================================
    # 实验3: 面内流形方向分析 — 用N=4情感+强度
    # ============================================================
    log(f"\n{'='*70}")
    log(f"实验3: 面内流形方向分析 — 强度轨迹在面上朝哪里走?")
    log(f"{'='*70}")

    domain = EMOTION_4
    class_order = domain["order"]
    N = len(class_order)
    D = N - 1

    # 收集mild和strong的残差
    mild_words, mild_class_to_idx = [], {}
    strong_words, strong_class_to_idx = [], {}
    base_words_list, base_class_to_idx = [], {}

    for cls in class_order:
        # mild: modifier + word
        m_words = []
        for mod in INTENSITY_MODIFIERS["mild"][:3]:
            for w in domain["classes"][cls][:4]:
                m_words.append(f"{mod} {w}")
        start = len(mild_words)
        mild_class_to_idx[cls] = list(range(start, start + len(m_words)))
        mild_words.extend(m_words)

        # strong: modifier + word
        s_words = []
        for mod in INTENSITY_MODIFIERS["strong"][:3]:
            for w in domain["classes"][cls][:4]:
                s_words.append(f"{mod} {w}")
        start = len(strong_words)
        strong_class_to_idx[cls] = list(range(start, start + len(s_words)))
        strong_words.extend(s_words)

        # base: 原始词
        b_words = domain["classes"][cls][:6]
        start = len(base_words_list)
        base_class_to_idx[cls] = list(range(start, start + len(b_words)))
        base_words_list.extend(b_words)

    # 收集所有需要的残差
    best_layer = all_results["emotion4"]["best_raw"]["layer"]
    needed_layers = [best_layer]

    log(f"  收集mild残差 ({len(mild_words)}词)...")
    mild_res = collect_residuals(
        model, tokenizer, device, mild_words,
        "The person felt {word} about the", needed_layers
    )
    log(f"  收集strong残差 ({len(strong_words)}词)...")
    strong_res = collect_residuals(
        model, tokenizer, device, strong_words,
        "The person felt {word} about the", needed_layers
    )
    log(f"  收集base残差 ({len(base_words_list)}词)...")
    base_res = collect_residuals(
        model, tokenizer, device, base_words_list,
        domain["prompt"], needed_layers
    )

    # 计算中心
    centers_mild = compute_class_centers(mild_res[best_layer], mild_class_to_idx)
    centers_strong = compute_class_centers(strong_res[best_layer], strong_class_to_idx)
    base_centers = compute_class_centers(base_res[best_layer], base_class_to_idx)

    # 单纯形拟合(用base词的中心)
    fit = compute_simplex_fit_unified(base_centers, class_order, normalize=False)

    log(f"\n  基础单纯形拟合: fit_r2={fit['fit_r2']:.4f}")
    log(f"  数据尺度: {fit['data_scale']:.1f}")

    # 计算强度轨迹
    traj = compute_intensity_trajectory(centers_strong, centers_mild, class_order, fit)

    log(f"\n  ★★★★★ 面内流形方向分析 ★★★★★")
    log(f"  (最佳层L{best_layer})")
    for cls, t in traj.items():
        log(f"  {cls}:")
        log(f"    径向对齐: {t['radial_alignment']:.3f}")
        log(f"    切向分量: {t['tangential_norm']:.3f} "
            f"(占比: {t['tangential_fraction']:.3f})")
        log(f"    最接近的类: {t['closest_other_class']} "
            f"(对齐={t['closest_align']:.3f})")

        # 打印与其他类的对齐
        sorted_others = sorted(t['other_class_aligns'].items(),
                               key=lambda x: abs(x[1]), reverse=True)
        for other_cls, align in sorted_others:
            log(f"      → {other_cls}: {align:+.3f}")

    all_results["tangential_analysis"] = {
        "best_layer": best_layer,
        "base_fit_r2": fit["fit_r2"],
        "trajectories": {cls: {
            "radial_alignment": t["radial_alignment"],
            "tangential_fraction": t["tangential_fraction"],
            "closest_other_class": t["closest_other_class"],
            "closest_align": t["closest_align"],
            "other_class_aligns": t["other_class_aligns"],
        } for cls, t in traj.items()},
    }

    # ============================================================
    # 实验4: 层间变换的切向方向演化
    # ============================================================
    log(f"\n{'='*70}")
    log(f"实验4: 层间变换的切向方向演化")
    log(f"{'='*70}")

    # 在多个层收集base+mild+strong的残差
    log(f"  在所有采样层收集强度残差...")
    mild_res_all = collect_residuals(
        model, tokenizer, device, mild_words,
        "The person felt {word} about the", layers
    )
    strong_res_all = collect_residuals(
        model, tokenizer, device, strong_words,
        "The person felt {word} about the", layers
    )
    base_res_all = collect_residuals(
        model, tokenizer, device, base_words_list,
        domain["prompt"], layers
    )

    layer_tangential = []
    for l in layers:
        try:
            base_c = compute_class_centers(base_res_all[l], base_class_to_idx)
            centers_m = compute_class_centers(mild_res_all[l], mild_class_to_idx)
            centers_s = compute_class_centers(strong_res_all[l], strong_class_to_idx)

            fit_l = compute_simplex_fit_unified(base_c, class_order, normalize=False)
            traj_l = compute_intensity_trajectory(centers_s, centers_m, class_order, fit_l)

            if not traj_l:
                continue

            avg_radial = np.mean([t["radial_alignment"] for t in traj_l.values()])
            avg_tang = np.mean([t["tangential_fraction"] for t in traj_l.values()])

            # 最近类的分布
            closest_classes = [t["closest_other_class"] for t in traj_l.values()]

            layer_tangential.append({
                "layer": l,
                "fit_r2": fit_l["fit_r2"],
                "avg_radial_align": float(avg_radial),
                "avg_tang_fraction": float(avg_tang),
                "closest_classes": closest_classes,
                "trajectories": {cls: {
                    "radial_alignment": t["radial_alignment"],
                    "tangential_fraction": t["tangential_fraction"],
                    "closest_other_class": t["closest_other_class"],
                } for cls, t in traj_l.items()},
            })
        except Exception as e:
            log(f"  L{l}: 跳过 (错误: {e})")

    log(f"\n  层间演化:")
    log(f"  {'层':>4s}  {'fit_r2':>8s}  {'径向对齐':>8s}  {'切向占比':>8s}  {'最近类'}")
    for lt in layer_tangential:
        log(f"  L{lt['layer']:3d}  {lt['fit_r2']:8.4f}  {lt['avg_radial_align']:8.3f}  "
            f"{lt['avg_tang_fraction']:8.3f}  {lt['closest_classes']}")

    all_results["layer_evolution"] = layer_tangential

    # ============================================================
    # 核心结论
    # ============================================================
    log(f"\n{'='*70}")
    log(f"★★★★★ 核心结论 ★★★★★")
    log(f"{'='*70}")

    # 1. fit_r2对比
    for domain_name in ["emotion4", "emotion6"]:
        if domain_name in all_results:
            r = all_results[domain_name]
            log(f"\n  {domain_name}:")
            log(f"    fit_r2(不归一化) = {r['best_raw']['fit_r2_raw']:.4f} (L{r['best_raw']['layer']})")
            log(f"    fit_r2(L2归一化) = {r['best_norm']['fit_r2_norm']:.4f} (L{r['best_norm']['layer']})")

    # 2. 面内方向
    if "tangential_analysis" in all_results:
        ta = all_results["tangential_analysis"]
        log(f"\n  面内方向(最佳层L{ta['best_layer']}):")
        for cls, t in ta["trajectories"].items():
            log(f"    {cls}: 径向={t['radial_alignment']:.3f} "
                f"切向占比={t['tangential_fraction']:.3f} "
                f"→ {t['closest_other_class']}({t['closest_align']:+.3f})")

    release_model(model)

    # 保存结果
    out_path = TEMP / f"ccxlvii_unified_{model_name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    log(f"\n结果保存到: {out_path}")

    return all_results


# ============================================================
# 入口
# ============================================================

_log_file = None

def log(msg):
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()


def main():
    global _log_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()

    TEMP.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG, 'a', encoding='utf-8')

    log(f"\n{'='*70}")
    log(f"CCXLVII(347): 统一归一化验证 + 面内流形方向分析")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"模型: {args.model}")
    log(f"{'='*70}")

    try:
        result = run_experiment(args.model, args)
        log(f"\n★★★★★ {args.model} 完成 ★★★★★")
    except Exception as e:
        import traceback
        log(f"\n!!! 错误: {e}")
        traceback.print_exc(file=_log_file)
    finally:
        _log_file.close()


if __name__ == "__main__":
    main()
