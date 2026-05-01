"""
CCXLIII(343): 类内几何分析 — 突破原型层瓶颈
===============================================
★★★★★ 核心问题: F_ratio≈0.15-1.23, 类内方差≥类间方差!
单纯形只是原型层, 类内结构占大部分方差但未被描述。

三大实验:
1. 类内子空间结构: 每个类别内部的PCA维度分布
   - 类内向量分布在多少维子空间?
   - 子空间与单纯形方向的对齐度?
   - 是否存在类内子单纯形?

2. 连续语义方向: 强度维度的几何性质
   - glad→happy→ecstatic 在空间中是径向还是切向?
   - 径向=远离中心(强度轴), 切向=沿单纯形面
   - 如果径向为主: 单纯形+径向噪声模型
   - 如果切向为主: 更复杂的结构

3. 混合模型验证: 正则单纯形+噪声能否完整描述?
   - isotropic噪声: 各向同性 → 高斯围绕中心
   - anisotropic噪声: 各向异性 → 沿特定方向延伸
   - 测量: 类内协方差矩阵的特征值谱

数据设计:
- emotion: happy/sad/angry/scared 各8个同义词 (类内分析)
- intensity: glad→happy→ecstatic, sad→gloomy→despairing (连续语义)
- occupation: doctor/teacher/engineer 各8个同义词

用法:
  python ccxliii_within_class_geometry.py --model qwen3
  python ccxliii_within_class_geometry.py --model glm4
  python ccxliii_within_class_geometry.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, eigh
from sklearn.decomposition import PCA

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxliii_within_class_log.txt"

# ===== 数据定义 =====

# 实验1: 类内子空间 — 每类8个同义词
WITHIN_CLASS_DOMAINS = {
    "emotion_wide": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                      "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                      "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
                       "anxiety", "apprehension", "trepidation", "phobia", "alarm", "dismay"],
        },
        "order": ["happy", "sad", "angry", "scared"],
        "prompt": "The person felt {word} about the",
    },
    "occupation_wide": {
        "classes": {
            "doctor": ["surgeon", "physician", "nurse", "therapist", "pediatrician",
                       "cardiologist", "dermatologist", "neurologist", "psychiatrist", "clinician",
                       "anesthesiologist", "oncologist"],
            "teacher": ["professor", "instructor", "educator", "tutor", "lecturer",
                        "mentor", "coach", "trainer", "academic", "counselor",
                        "principal", "supervisor"],
            "engineer": ["architect", "designer", "developer", "programmer", "mechanic",
                         "technician", "builder", "constructor", "inventor", "analyst",
                         "planner", "consultant"],
        },
        "order": ["doctor", "teacher", "engineer"],
        "prompt": "{Word} is a skilled professional who",
    },
}

# 实验2: 连续语义强度 — 从弱到强的强度梯度
INTENSITY_DOMAINS = {
    "happy_intensity": {
        "intensities": {
            "mild":   ["content", "pleased", "satisfied", "comfortable"],
            "medium": ["happy", "glad", "cheerful", "joyful"],
            "strong": ["ecstatic", "elated", "euphoric", "jubilant"],
        },
        "order": ["mild", "medium", "strong"],
        "base_class": "happy",
        "prompt": "The person felt {word} about the",
    },
    "sad_intensity": {
        "intensities": {
            "mild":   ["down", "unhappy", "disappointed", "blue"],
            "medium": ["sad", "sorrowful", "melancholy", "gloomy"],
            "strong": ["despairing", "devastated", "anguished", "heartbroken"],
        },
        "order": ["mild", "medium", "strong"],
        "base_class": "sad",
        "prompt": "The person felt {word} about the",
    },
    "angry_intensity": {
        "intensities": {
            "mild":   ["annoyed", "irritated", "bothered", "irked"],
            "medium": ["angry", "mad", "furious", "enraged"],
            "strong": ["livid", "infuriated", "incensed", "rage-filled"],
        },
        "order": ["mild", "medium", "strong"],
        "base_class": "angry",
        "prompt": "The person felt {word} about the",
    },
    "scared_intensity": {
        "intensities": {
            "mild":   ["uneasy", "wary", "cautious", "concerned"],
            "medium": ["scared", "afraid", "frightened", "alarmed"],
            "strong": ["terrified", "horrified", "panicked", "petrified"],
        },
        "order": ["mild", "medium", "strong"],
        "base_class": "scared",
        "prompt": "The person felt {word} about the",
    },
}

# 实验3: 完整的4类emotion数据 (用于混合模型)
FULL_EMOTION_DOMAINS = {
    "emotion_4": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                      "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                      "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
                       "anxiety", "apprehension", "trepidation", "phobia", "alarm", "dismay"],
        },
        "order": ["happy", "sad", "angry", "scared"],
        "prompt": "The person felt {word} about the",
    },
}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def collect_residuals_at_layer(model, tokenizer, layers, li, word_dict,
                                prompt_template, n_words=12, device="cuda"):
    """收集某层的残差 - word_dict: {class_name: [word_list]}"""
    class_resids = {}
    for cls, words in word_dict.items():
        word_list = words[:n_words]
        resids = []
        for word in word_list:
            prompt = prompt_template.format(word=word, Word=word.capitalize())
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            captured = {}
            
            def mk_hook(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured[k] = o[0, -1, :].detach().float().cpu().numpy()
                return hook
            
            hook = layers[li].register_forward_hook(mk_hook("L"))
            with torch.no_grad():
                try:
                    _ = model(**toks)
                except:
                    pass
            hook.remove()
            
            if "L" in captured:
                resids.append(captured["L"])
        
        if len(resids) >= 4:
            class_resids[cls] = resids
    
    return class_resids


# ===== 实验1: 类内子空间分析 =====

def analyze_within_class_subspace(class_resids):
    """
    分析每个类别内部的子空间结构
    返回: 每类的PCA维度分布, 子空间与全局方向的对齐度
    """
    results = {}
    
    all_vecs = []
    for cls, vecs in class_resids.items():
        all_vecs.extend(vecs)
    all_vecs = np.array(all_vecs)
    global_mean = np.mean(all_vecs, axis=0)
    
    # 全局PCA
    centered_global = all_vecs - global_mean
    U_global, S_global, _ = svd(centered_global, full_matrices=False)
    # 取前N-1个全局主成分 (N=类别数)
    N = len(class_resids)
    global_top = U_global[:, :N-1]  # (n_samples, N-1)
    
    for cls, vecs in class_resids.items():
        vecs = np.array(vecs)
        n_samples, d = vecs.shape
        cls_mean = np.mean(vecs, axis=0)
        centered = vecs - cls_mean
        
        # 类内PCA
        U_cls, S_cls, Vt_cls = svd(centered, full_matrices=False)
        
        # 累计方差解释比
        var_explained = S_cls ** 2 / np.sum(S_cls ** 2)
        cumvar = np.cumsum(var_explained)
        
        # 找90%和95%方差需要的维度
        dim_90 = np.searchsorted(cumvar, 0.90) + 1
        dim_95 = np.searchsorted(cumvar, 0.95) + 1
        
        # 前3个特征值的比例
        top3_ratio = np.sum(var_explained[:3]) if len(var_explained) >= 3 else np.sum(var_explained)
        
        # 子空间与全局方向的对齐度
        # 类内前k个主方向 vs 全局类间方向(类中心到全局中心的方向)
        cls_center_from_global = cls_mean - global_mean
        cls_center_dir = cls_center_from_global / (np.linalg.norm(cls_center_from_global) + 1e-10)
        
        # 类内前3个主方向
        within_dirs = Vt_cls[:3]  # (3, d)
        
        # 对齐度: 类内主方向与类中心方向的点积
        alignment = [abs(np.dot(within_dirs[i], cls_center_dir)) for i in range(min(3, len(within_dirs)))]
        
        # 径向 vs 切向比例
        # 径向 = 沿类中心方向, 切向 = 垂直于类中心方向
        radial_var = np.dot(centered, cls_center_dir) ** 2
        radial_ratio = np.sum(radial_var) / (np.sum(centered ** 2) + 1e-10)
        
        results[cls] = {
            "n_samples": n_samples,
            "dim_90": int(dim_90),
            "dim_95": int(dim_95),
            "top1_var": float(var_explained[0]) if len(var_explained) > 0 else 0,
            "top2_var": float(var_explained[1]) if len(var_explained) > 1 else 0,
            "top3_var": float(var_explained[2]) if len(var_explained) > 2 else 0,
            "top3_ratio": float(top3_ratio),
            "alignment_top1": float(alignment[0]) if len(alignment) > 0 else 0,
            "alignment_top2": float(alignment[1]) if len(alignment) > 1 else 0,
            "alignment_top3": float(alignment[2]) if len(alignment) > 2 else 0,
            "radial_ratio": float(radial_ratio),
        }
    
    return results


# ===== 实验2: 连续语义方向分析 =====

def analyze_intensity_direction(intensity_resids, base_class_resids=None):
    """
    分析强度梯度在空间中的方向
    - 径向: 远离/接近类中心
    - 切向: 沿单纯形面
    """
    results = {}
    
    # 所有强度词的向量
    all_vecs = {}
    for level, vecs in intensity_resids.items():
        all_vecs[level] = np.array(vecs)
    
    # 计算每个强度水平的中心
    centers = {}
    for level, vecs in all_vecs.items():
        centers[level] = np.mean(vecs, axis=0)
    
    # 全局中心
    all_points = np.concatenate(list(all_vecs.values()), axis=0)
    global_center = np.mean(all_points, axis=0)
    
    # 弱→中→强 的轨迹向量
    levels = sorted(all_vecs.keys())
    if len(levels) >= 2:
        trajectory = centers[levels[-1]] - centers[levels[0]]  # 弱→强
        traj_norm = np.linalg.norm(trajectory)
        if traj_norm > 1e-10:
            trajectory_dir = trajectory / traj_norm
        else:
            trajectory_dir = np.zeros_like(trajectory)
    else:
        trajectory = np.zeros_like(centers[levels[0]])
        trajectory_dir = np.zeros_like(trajectory)
    
    # 径向分析: 轨迹方向 vs 类中心方向
    if base_class_resids is not None:
        base_center = np.mean(base_class_resids, axis=0)
        radial_dir = base_center - global_center
        radial_norm = np.linalg.norm(radial_dir)
        if radial_norm > 1e-10:
            radial_dir = radial_dir / radial_norm
        else:
            radial_dir = np.zeros_like(radial_dir)
        
        # 轨迹与径向的对齐度
        radial_alignment = abs(np.dot(trajectory_dir, radial_dir))
    else:
        # 用弱强度中心作为类中心方向
        base_center = centers[levels[0]]
        radial_dir = base_center - global_center
        radial_norm = np.linalg.norm(radial_dir)
        if radial_norm > 1e-10:
            radial_dir = radial_dir / radial_norm
        radial_alignment = abs(np.dot(trajectory_dir, radial_dir))
    
    # 每个强度水平到类中心的距离
    dists_to_center = {}
    for level in levels:
        dists = [np.linalg.norm(v - base_center) for v in all_vecs[level]]
        dists_to_center[level] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
        }
    
    # 强度方向上各水平的投影
    if traj_norm > 1e-10:
        projections = {}
        for level in levels:
            proj = [np.dot(v - centers[levels[0]], trajectory_dir) for v in all_vecs[level]]
            projections[level] = {
                "mean": float(np.mean(proj)),
                "std": float(np.std(proj)),
            }
    else:
        projections = {level: {"mean": 0, "std": 0} for level in levels}
    
    # 弱→中→强距离变化趋势
    dist_trend = [dists_to_center[level]["mean"] for level in levels]
    
    results = {
        "levels": levels,
        "trajectory_norm": float(traj_norm),
        "radial_alignment": float(radial_alignment),
        "dists_to_center": dists_to_center,
        "projections": projections,
        "dist_trend": dist_trend,
        "interpretation": "",
    }
    
    # 判定: 径向 vs 切向
    if radial_alignment > 0.7:
        results["interpretation"] = "RADIAL (强度方向沿类中心方向)"
    elif radial_alignment < 0.3:
        results["interpretation"] = "TANGENTIAL (强度方向垂直于类中心方向)"
    else:
        results["interpretation"] = f"MIXED (对齐度={radial_alignment:.3f})"
    
    # 距离趋势判定
    if len(dist_trend) >= 2:
        if dist_trend[-1] > dist_trend[0] * 1.1:
            results["dist_interpretation"] = "RECEDING (强情感远离中心)"
        elif dist_trend[-1] < dist_trend[0] * 0.9:
            results["dist_interpretation"] = "APPROACHING (强情感接近中心)"
        else:
            results["dist_interpretation"] = "STABLE (距离基本不变)"
    
    return results


# ===== 实验3: 混合模型验证 =====

def analyze_mixed_model(class_resids):
    """
    验证"正则单纯形+噪声"模型能否完整描述
    - 类内协方差矩阵的特征值谱
    - 各向同性 vs 各向异性
    - 噪声子空间与单纯形方向的关系
    """
    results = {}
    
    all_vecs = []
    for cls, vecs in class_resids.items():
        all_vecs.extend(vecs)
    all_vecs = np.array(all_vecs)
    global_mean = np.mean(all_vecs, axis=0)
    
    # 类中心
    class_centers = {}
    for cls, vecs in class_resids.items():
        class_centers[cls] = np.mean(vecs, axis=0)
    
    N = len(class_resids)
    
    # ===== 3a. 类间子空间 vs 类内子空间 =====
    
    # 类间协方差
    center_matrix = np.array([class_centers[c] - global_mean for c in class_resids.keys()])
    between_cov = center_matrix.T @ center_matrix / N
    
    # 类内协方差 (所有类内偏差的汇总)
    within_covs = {}
    all_within_centered = []
    for cls, vecs in class_resids.items():
        centered = np.array(vecs) - class_centers[cls]
        all_within_centered.append(centered)
        within_covs[cls] = centered.T @ centered / len(centered)
    
    all_within_centered = np.concatenate(all_within_centered, axis=0)
    pooled_within_cov = all_within_centered.T @ all_within_centered / len(all_within_centered)
    
    # 类间子空间的特征值
    between_eigs = np.sort(np.linalg.eigvalsh(between_cov))[::-1]
    between_eigs = between_eigs[between_eigs > 1e-10]
    
    # 类内协方差的特征值谱
    within_eigs = np.sort(np.linalg.eigvalsh(pooled_within_cov))[::-1]
    within_eigs = within_eigs[within_eigs > 1e-10]
    
    # 类内前几个特征值的占比 (各向同性=均匀, 各向异性=集中)
    within_var_ratio = within_eigs ** 2 / np.sum(within_eigs ** 2)
    top1_within_ratio = float(within_var_ratio[0]) if len(within_var_ratio) > 0 else 0
    top3_within_ratio = float(np.sum(within_var_ratio[:3])) if len(within_var_ratio) >= 3 else float(np.sum(within_var_ratio))
    
    # ===== 3b. 类内主方向 vs 类间方向的对齐 =====
    
    # 类间主方向
    between_dirs = np.linalg.eigh(between_cov)[1][:, ::-1][:, :N-1]
    # 类内主方向
    within_dirs = np.linalg.eigh(pooled_within_cov)[1][:, ::-1][:, :N-1]
    
    # 子空间重叠度: 两个子空间的Principal Angles
    if between_dirs.shape[1] > 0 and within_dirs.shape[1] > 0:
        # 投影矩阵
        P_between = between_dirs @ between_dirs.T
        P_within = within_dirs @ within_dirs.T
        # 子空间重叠 = ||P_between @ P_within||_F / sqrt(||P_between|| * ||P_within||)
        overlap = np.linalg.norm(P_between @ P_within) / (
            np.linalg.norm(P_between) * np.linalg.norm(P_within) + 1e-10)
    else:
        overlap = 0.0
    
    # ===== 3c. 各向同性检验 =====
    # 如果类内是各向同性的, 特征值应该近似相等
    # 检验: 最大特征值/最小特征值(前n_sep个)
    n_sep = min(N-1, len(within_eigs))
    if n_sep > 0 and within_eigs[n_sep-1] > 1e-10:
        anisotropy_ratio = float(within_eigs[0] / within_eigs[n_sep-1])
    else:
        anisotropy_ratio = float('inf')
    
    # 完全各向同性: ratio=1, 强各向异性: ratio>>1
    if anisotropy_ratio < 2:
        isotropy_verdict = "ISOTROPIC (各向同性)"
    elif anisotropy_ratio < 5:
        isotropy_verdict = "WEAKLY ANISOTROPIC (弱各向异性)"
    elif anisotropy_ratio < 10:
        isotropy_verdict = "MODERATELY ANISOTROPIC (中等各向异性)"
    else:
        isotropy_verdict = "STRONGLY ANISOTROPIC (强各向异性)"
    
    # ===== 3d. 噪声模型拟合 =====
    # 测试: 类内分布是否符合高斯?
    # 方法: 到类中心的距离分布 vs 高斯期望
    dist_distribution = {}
    for cls, vecs in class_resids.items():
        dists = [np.linalg.norm(v - class_centers[cls]) for v in vecs]
        dist_distribution[cls] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "skew": float(np.mean(((dists - np.mean(dists)) / (np.std(dists) + 1e-10)) ** 3)) if np.std(dists) > 1e-10 else 0,
        }
    
    # 高斯分布的偏度=0, 如果偏度>>0说明有重尾
    
    results = {
        "N": N,
        "between_eigs_top5": [float(x) for x in between_eigs[:5]],
        "within_eigs_top5": [float(x) for x in within_eigs[:5]],
        "within_top1_ratio": top1_within_ratio,
        "within_top3_ratio": top3_within_ratio,
        "subspace_overlap": float(overlap),
        "anisotropy_ratio": anisotropy_ratio,
        "isotropy_verdict": isotropy_verdict,
        "dist_distribution": dist_distribution,
        "n_within_dims_90": int(np.searchsorted(np.cumsum(within_var_ratio), 0.90) + 1),
        "n_within_dims_95": int(np.searchsorted(np.cumsum(within_var_ratio), 0.95) + 1),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"\n{'='*80}")
    log(f"CCXLIII: 类内几何分析 — 突破原型层瓶颈 — {model_name}")
    log(f"{'='*80}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    
    log(f"  d_model={d_model}, n_layers={n_layers}")
    
    all_results = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "exp1_within_class": {},
        "exp2_intensity": {},
        "exp3_mixed_model": {},
    }
    
    # 选择最佳层 (从之前实验可知, 通常是中后层)
    # 采样4个层: 前1/4, 中间, 后1/4, 最后
    sample_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]
    sample_layers = sorted(set([max(0, l) for l in sample_layers]))
    log(f"  采样层: {sample_layers}")
    
    # ===== 实验1: 类内子空间结构 =====
    log(f"\n{'='*60}")
    log(f"实验1: 类内子空间结构分析")
    log(f"{'='*60}")
    
    for domain_name, domain_info in WITHIN_CLASS_DOMAINS.items():
        log(f"\n--- Domain: {domain_name} ---")
        
        # 在最佳层(通常是中间层)收集数据
        best_li = sample_layers[1]  # 中间层
        class_resids = collect_residuals_at_layer(
            model, tokenizer, layers, best_li,
            domain_info["classes"],
            domain_info["prompt"], n_words=12, device=device
        )
        
        if len(class_resids) < 2:
            log(f"  有效类不足, 跳过")
            continue
        
        log(f"  在L{best_li}收集, 有效类: {list(class_resids.keys())}")
        
        within_results = analyze_within_class_subspace(class_resids)
        
        for cls, wr in within_results.items():
            log(f"  {cls:>12}: dim_90={wr['dim_90']}, dim_95={wr['dim_95']}, "
                f"top3_ratio={wr['top3_ratio']:.3f}, radial_ratio={wr['radial_ratio']:.3f}, "
                f"alignment_top1={wr['alignment_top1']:.3f}")
        
        all_results["exp1_within_class"][domain_name] = {
            "layer": best_li,
            "results": within_results,
        }
    
    # ===== 实验2: 连续语义方向 =====
    log(f"\n{'='*60}")
    log(f"实验2: 连续语义方向 — 强度梯度分析")
    log(f"{'='*60}")
    
    # 先收集4个基类中心
    best_li = sample_layers[1]
    emotion_resids = collect_residuals_at_layer(
        model, tokenizer, layers, best_li,
        FULL_EMOTION_DOMAINS["emotion_4"]["classes"],
        FULL_EMOTION_DOMAINS["emotion_4"]["prompt"],
        n_words=12, device=device
    )
    
    for domain_name, domain_info in INTENSITY_DOMAINS.items():
        log(f"\n--- Intensity: {domain_name} ---")
        
        # 收集所有强度水平的残差
        all_int_resids = {}
        for level, words in domain_info["intensities"].items():
            level_resids = []
            for word in words:
                prompt = domain_info["prompt"].format(word=word, Word=word.capitalize())
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                captured = {}
                
                def mk_hook(k):
                    def hook(m, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        captured[k] = o[0, -1, :].detach().float().cpu().numpy()
                    return hook
                
                hook = layers[best_li].register_forward_hook(mk_hook("L"))
                with torch.no_grad():
                    try:
                        _ = model(**toks)
                    except:
                        pass
                hook.remove()
                
                if "L" in captured:
                    level_resids.append(captured["L"])
            
            if level_resids:
                all_int_resids[level] = level_resids
        
        if len(all_int_resids) < 2:
            log(f"  强度水平不足, 跳过")
            continue
        
        # 获取基类残差
        base_cls = domain_info["base_class"]
        base_resids = emotion_resids.get(base_cls)
        
        intensity_results = analyze_intensity_direction(all_int_resids, base_resids)
        
        log(f"  轨迹范数={intensity_results['trajectory_norm']:.4f}")
        log(f"  径向对齐度={intensity_results['radial_alignment']:.4f}")
        log(f"  判定: {intensity_results['interpretation']}")
        log(f"  距离趋势: {intensity_results['dist_trend']}")
        if 'dist_interpretation' in intensity_results:
            log(f"  距离判定: {intensity_results['dist_interpretation']}")
        
        for level in intensity_results['levels']:
            d2c = intensity_results['dists_to_center'].get(level, {})
            log(f"    {level:>8}: dist_to_center={d2c.get('mean',0):.4f}±{d2c.get('std',0):.4f}")
        
        all_results["exp2_intensity"][domain_name] = {
            "layer": best_li,
            "results": intensity_results,
        }
    
    # ===== 实验3: 混合模型验证 =====
    log(f"\n{'='*60}")
    log(f"实验3: 混合模型验证 — 正则单纯形+噪声")
    log(f"{'='*60}")
    
    # 在多个层上分析
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        
        class_resids = collect_residuals_at_layer(
            model, tokenizer, layers, li,
            FULL_EMOTION_DOMAINS["emotion_4"]["classes"],
            FULL_EMOTION_DOMAINS["emotion_4"]["prompt"],
            n_words=12, device=device
        )
        
        if len(class_resids) < 3:
            log(f"  有效类不足, 跳过")
            continue
        
        mixed_results = analyze_mixed_model(class_resids)
        
        log(f"  N={mixed_results['N']}")
        log(f"  类间特征值top5: {[f'{x:.2f}' for x in mixed_results['between_eigs_top5']]}")
        log(f"  类内特征值top5: {[f'{x:.2f}' for x in mixed_results['within_eigs_top5']]}")
        log(f"  类内top1占比={mixed_results['within_top1_ratio']:.4f}, top3占比={mixed_results['within_top3_ratio']:.4f}")
        log(f"  子空间重叠度={mixed_results['subspace_overlap']:.4f}")
        log(f"  各向异性比={mixed_results['anisotropy_ratio']:.2f}")
        log(f"  判定: {mixed_results['isotropy_verdict']}")
        log(f"  类内90%方差维度={mixed_results['n_within_dims_90']}, 95%维度={mixed_results['n_within_dims_95']}")
        
        for cls, dd in mixed_results['dist_distribution'].items():
            log(f"    {cls:>8}: dist={dd['mean']:.4f}±{dd['std']:.4f}, skew={dd['skew']:.3f}")
        
        all_results["exp3_mixed_model"][f"L{li}"] = mixed_results
    
    release_model(model)
    
    # ===== 保存 =====
    out_path = TEMP / f"ccxliii_within_class_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存: {out_path}")
    
    # ===== 汇总 =====
    log(f"\n{'='*80}")
    log(f"CCXLIII 汇总 — {model_name}")
    log(f"{'='*80}")
    
    # 实验1汇总
    log(f"\n--- 实验1: 类内子空间结构 ---")
    for domain_name, domain_data in all_results["exp1_within_class"].items():
        log(f"  {domain_name}:")
        for cls, wr in domain_data["results"].items():
            log(f"    {cls:>12}: dim_90={wr['dim_90']}, dim_95={wr['dim_95']}, "
                f"top3={wr['top3_ratio']:.3f}, radial={wr['radial_ratio']:.3f}, "
                f"align1={wr['alignment_top1']:.3f}")
    
    # 实验2汇总
    log(f"\n--- 实验2: 连续语义方向 ---")
    for domain_name, domain_data in all_results["exp2_intensity"].items():
        r = domain_data["results"]
        log(f"  {domain_name:>20}: radial_align={r['radial_alignment']:.4f}, "
            f"traj_norm={r['trajectory_norm']:.4f}, "
            f"{r['interpretation']}")
        if 'dist_interpretation' in r:
            log(f"    {r['dist_interpretation']}")
    
    # 实验3汇总
    log(f"\n--- 实验3: 混合模型 ---")
    for layer_name, mr in all_results["exp3_mixed_model"].items():
        log(f"  {layer_name}: anisotropy={mr['anisotropy_ratio']:.2f}, "
            f"overlap={mr['subspace_overlap']:.4f}, "
            f"within_dims_90={mr['n_within_dims_90']}, "
            f"{mr['isotropy_verdict']}")
    
    # 核心结论
    log(f"\n{'='*60}")
    log(f"核心结论")
    log(f"{'='*60}")
    
    # 径向vs切向判定
    radial_aligns = [d["results"]["radial_alignment"] 
                     for d in all_results["exp2_intensity"].values()]
    if radial_aligns:
        mean_ra = np.mean(radial_aligns)
        log(f"  平均径向对齐度: {mean_ra:.4f}")
        if mean_ra > 0.5:
            log(f"  → ★★★★ 强度方向以径向为主 — 支持\"单纯形+径向噪声\"模型")
        elif mean_ra > 0.3:
            log(f"  → ★★★ 强度方向为混合 — 需要更复杂的模型")
        else:
            log(f"  → ★★ 强度方向以切向为主 — 单纯形+噪声模型不充分")
    
    # 各向同性判定
    anisotropies = [mr["anisotropy_ratio"] 
                    for mr in all_results["exp3_mixed_model"].values()]
    if anisotropies:
        mean_aniso = np.mean(anisotropies)
        log(f"  平均各向异性比: {mean_aniso:.2f}")
        if mean_aniso < 3:
            log(f"  → ★★★★★ 类内噪声接近各向同性 — \"正则单纯形+高斯噪声\"可能是好模型!")
        elif mean_aniso < 10:
            log(f"  → ★★★★ 类内噪声弱各向异性 — 需要主方向修正")
        else:
            log(f"  → ★★ 类内噪声强各向异性 — 需要更复杂的类内模型")
    
    # 类内维度
    dims_90 = [mr["n_within_dims_90"] 
               for mr in all_results["exp3_mixed_model"].values()]
    if dims_90:
        log(f"  类内90%方差维度: {dims_90} (mean={np.mean(dims_90):.1f})")
        log(f"  → 类内方差集中在{np.mean(dims_90):.0f}维子空间, 远小于d_model={d_model}")
    
    log(f"\nDone! {model_name}")


if __name__ == "__main__":
    main()
