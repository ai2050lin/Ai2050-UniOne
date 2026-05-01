"""
CCXXXIX(339): 单纯形结构严格验证 — 解决三大过度推广
====================================================
★★★★★ 三大问题及解决策略:

问题1: 线性可分≠真实几何
  → 解决: 多维几何验证 + 统计显著性检验
  → 不只看n_separating_PCs, 而是验证:
    a) 角度分布是否匹配正则单纯形(理想角=arccos(-1/N))
    b) 边长均匀性(CV系数)
    c) 置换检验: 与随机高斯聚类对比, n_sep是否显著高于随机
    d) 正则单纯形拟合度: 顶点在正则单纯形上的投影R²

问题2: 局部结构≠全局结构
  → 解决: 全层谱图
  → 在每个层计算Simplex Quality Index (SQI)
  → 绘制SQI vs layer曲线, 看结构是稳定还是局部现象
  → 如果只在1-2个层出现→局部现象; 如果跨多层稳定→全局结构

问题3: 不要过于绝对分类
  → 解决: 连续的Simplex Quality Index (SQI)
  → SQI = w1*(n_sep/(N-1)) + w2*regularity + w3*(1-angle_dev/90) + w4*radius_uniformity
  → 所有领域在SQI上形成连续谱
  → 不再是"有/无"二分, 而是0到1的质量分数

核心指标:
  1. Simplex Quality Index (SQI) — 连续质量分数
  2. Permutation p-value — 统计显著性
  3. Regular Simplex Fit R² — 正则单纯形拟合度
  4. SQI Layer Profile — 全层谱图

用法:
  python ccxxxix_simplex_rigorous.py --model qwen3
  python ccxxxix_simplex_rigorous.py --model glm4
  python ccxxxix_simplex_rigorous.py --model deepseek7b
"""
import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxxix_simplex_rigorous_log.txt"

# ===== 语义类别定义 =====
DOMAINS = {
    "habitat": {
        "classes": {
            "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
                     "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra"],
            "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
                      "crab", "seal", "squid", "lobster", "jellyfish", "starfish"],
            "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
                    "falcon", "pigeon", "robin", "condor", "albatross"],
            "space": ["astronaut", "satellite", "comet", "meteor", "nebula",
                      "quasar", "asteroid", "rocket", "spaceship", "cosmos", "planet"],
            "microscopic": ["bacterium", "virus", "cell", "amoeba", "paramecium",
                            "euglena", "diatom", "plasmodium", "ribosome", "mitochondria"],
            "virtual": ["algorithm", "program", "software", "database", "network",
                        "protocol", "encryption", "firewall", "browser", "server"],
        },
        "order": ["land", "ocean", "sky", "space", "microscopic", "virtual"],
        "prompt": "The {word} lives in the",
    },
    "emotion": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation", "euphoria",
                      "contentment", "pleasure", "gladness", "merriment", "jubilation"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay", "woe",
                    "anguish", "heartache", "mourning", "dejection", "forlorn"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility", "indignation",
                      "animosity", "vexation", "exasperation", "resentment", "bitterness"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror", "alarm",
                       "anxiety", "apprehension", "trepidation", "phobia", "consternation"],
            "surprised": ["astonishment", "amazement", "wonder", "shock", "startle",
                          "stupefaction", "disbelief", "incredulity", "bewilderment", "flabbergasted"],
            "disgusted": ["revulsion", "repugnance", "nausea", "loathing", "abhorrence",
                          "aversion", "distaste", "antipathy", "repulsion", "contempt"],
        },
        "order": ["happy", "sad", "angry", "scared", "surprised", "disgusted"],
        "prompt": "The person felt {word} about the",
    },
    "occupation": {
        "classes": {
            "doctor": ["surgeon", "physician", "nurse", "therapist", "pediatrician",
                       "cardiologist", "dermatologist", "neurologist", "psychiatrist", "oncologist"],
            "teacher": ["professor", "instructor", "educator", "tutor", "lecturer", "mentor",
                        "coach", "trainer", "academic", "scholar"],
            "engineer": ["architect", "designer", "developer", "programmer", "mechanic",
                         "technician", "builder", "constructor", "inventor", "fabricator"],
            "artist": ["painter", "sculptor", "musician", "dancer", "actor", "singer",
                       "poet", "writer", "composer", "illustrator"],
            "lawyer": ["attorney", "advocate", "counsel", "barrister", "solicitor",
                       "prosecutor", "defender", "judge", "magistrate", "paralegal"],
            "chef": ["cook", "baker", "pastry", "butcher", "sous", "line",
                     "prep", "saucier", "caterer", "restaurateur"],
        },
        "order": ["doctor", "teacher", "engineer", "artist", "lawyer", "chef"],
        "prompt": "{Word} is a skilled professional who",
    },
    "color": {
        "classes": {
            "red": ["apple", "rose", "ruby", "cherry", "tomato", "flame", "crimson",
                    "scarlet", "brick", "garnet", "maroon", "blood"],
            "blue": ["ocean", "sky", "sapphire", "azure", "navy", "cobalt", "indigo",
                     "teal", "cyan", "turquoise", "aquamarine", "cerulean"],
            "green": ["grass", "emerald", "forest", "lime", "mint", "olive", "jade",
                      "sage", "moss", "fern", "clover", "basil"],
            "yellow": ["sun", "gold", "lemon", "banana", "honey", "mustard", "amber",
                       "canary", "daffodil", "butter", "saffron", "marigold"],
            "purple": ["violet", "lavender", "plum", "grape", "orchid", "amethyst", "magenta",
                       "lilac", "mauve", "mulberry", "eggplant", "wisteria"],
            "orange": ["tangerine", "carrot", "pumpkin", "apricot", "peach", "mango",
                       "cantaloupe", "salmon", "coral", "copper", "rust", "amber"],
        },
        "order": ["red", "blue", "green", "yellow", "purple", "orange"],
        "prompt": "The color {word} is very",
    },
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_n_separating_pcs(class_resids, all_vecs, n_pc_extra=3):
    """计算n_separating_PCs"""
    arr = np.array(all_vecs)
    centered = arr - arr.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    n_classes = len(class_resids)
    n_pc = min(n_classes + n_pc_extra, Vt.shape[0])
    
    class_proj = {}
    for cls in class_resids:
        cls_arr = np.array(class_resids[cls])
        cls_centered = cls_arr - arr.mean(axis=0)
        pc_proj = cls_centered @ Vt[:n_pc].T
        class_proj[cls] = pc_proj
    
    n_separating = 0
    separating_f_ratios = []
    for pc_i in range(n_pc):
        means = [np.mean(class_proj[c][:, pc_i]) for c in class_resids]
        within_vars = [np.var(class_proj[c][:, pc_i]) for c in class_resids]
        f_ratio = np.var(means) / max(np.mean(within_vars), 1e-10)
        separating_f_ratios.append(f_ratio)
        if f_ratio > 1.0:
            n_separating += 1
    
    return n_separating, separating_f_ratios, Vt[:n_pc]


def compute_detailed_geometry(class_centers_dict):
    """
    ★★★★★ 详细几何分析 — 解决问题1: 线性可分≠真实几何
    
    不只看n_sep, 而是全面验证单纯形几何:
    1. 边长统计: mean, std, CV (正则单纯形所有边长相等)
    2. 角度统计: mean, std, 理想角arccos(-1/N), 偏差
    3. 顶点-质心距离: mean, std, CV (正则单纯形所有半径相等)
    4. 正则单纯形拟合R²: 顶点在最佳正则单纯形上的投影解释方差
    """
    class_names = list(class_centers_dict.keys())
    centers = np.array([class_centers_dict[c] for c in class_names])
    N = len(class_names)
    
    if N < 3:
        return None
    
    # 1. 边长统计
    pairwise_dists = squareform(pdist(centers))
    upper_tri = pairwise_dists[np.triu_indices(N, k=1)]
    mean_edge = np.mean(upper_tri)
    std_edge = np.std(upper_tri)
    cv_edge = std_edge / max(mean_edge, 1e-10)  # CV: 越小越均匀
    edge_uniformity = 1.0 - cv_edge  # 正则单纯形=1.0
    
    # 2. 角度统计
    centroid = np.mean(centers, axis=0)
    vectors = centers - centroid  # [N, d]
    
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = vectors[i], vectors[j]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_a) * 180 / np.pi)
    
    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi if N > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    angle_cv = std_angle / max(mean_angle, 1e-10) if mean_angle > 0 else 999
    
    # 3. 顶点-质心距离
    radii = np.linalg.norm(vectors, axis=1)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)
    cv_radius = std_radius / max(mean_radius, 1e-10)
    radius_uniformity = 1.0 - cv_radius
    
    # 4. ★★★★★ 正则单纯形拟合R² — 关键新指标!
    # 构造理想正则N-1维单纯形的顶点
    # 方法: 在N维空间中构造正则单纯形, 然后旋转对齐到实际数据
    n_dim = centers.shape[1]
    if N - 1 > n_dim:
        # 实际维度不够容纳正则单纯形
        simplex_fit_r2 = 0.0
    else:
        # 构造正则单纯形: 在N维空间中, 顶点在(1,1,...,1)方向上等距
        # 标准构造: 从标准正交基出发
        # 正则单纯形顶点: v_i = e_i - (1/N)(1,1,...,1) + (1/sqrt(N))*(1,1,...,1)偏移
        # 更简单的方法: Householder反射
        
        # 构造正则N-1维单纯形在N维空间中
        ideal_vertices = np.zeros((N, N))
        for i in range(N):
            ideal_vertices[i, i] = 1.0
        # 减去均值, 使质心在原点
        ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
        # 归一化, 使所有边长相等
        ideal_dists = squareform(pdist(ideal_vertices))
        ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
        ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
        
        # 用Procrustes对齐: 找旋转矩阵R, 使 ||centers - R @ ideal_vertices||^2 最小
        # centers: [N, d_dim], ideal_vertices: [N, N]
        # 需要投影到同一维度
        # CCXL修正: 原条件n_dim>=N导致n_dim=N-1时直接fit_r2=0, 改为n_dim>=N-1
        if n_dim >= N - 1:
            # centers有足够维度, 用前N维
            U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
            # 投影到N-1维(单纯形维度)
            proj_dim = min(N - 1, n_dim)
            centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T  # [N, proj_dim]
            
            # ideal也投影到同样维度
            U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
            ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T  # [N, proj_dim]
            
            # Procrustes对齐
            H = ideal_proj.T @ centers_proj  # [proj_dim, proj_dim]
            U_h, S_h, Vt_h = np.linalg.svd(H)
            R = U_h @ Vt_h  # 旋转矩阵 (CCXL修正: 原为Vt_h.T@U_h.T=逆旋转, 现为U@Vt=正确旋转)
            
            aligned_ideal = ideal_proj @ R  # [N, proj_dim]
            
            # R²: 理想顶点解释了多少方差
            residual = centers_proj - aligned_ideal
            ss_res = np.sum(residual ** 2)
            ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
            simplex_fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            simplex_fit_r2 = max(0.0, simplex_fit_r2)
        else:
            simplex_fit_r2 = 0.0
    
    return {
        "N": N,
        "n_dim": centers.shape[1],
        # 边长
        "mean_edge_length": round(float(mean_edge), 4),
        "std_edge_length": round(float(std_edge), 4),
        "cv_edge": round(float(cv_edge), 4),
        "edge_uniformity": round(float(edge_uniformity), 4),
        # 角度
        "mean_angle": round(float(mean_angle), 2),
        "std_angle": round(float(std_angle), 2),
        "ideal_angle": round(float(ideal_angle), 2),
        "angle_deviation": round(float(angle_deviation), 2),
        "angle_cv": round(float(angle_cv), 4),
        # 半径
        "mean_radius": round(float(mean_radius), 4),
        "std_radius": round(float(std_radius), 4),
        "cv_radius": round(float(cv_radius), 4),
        "radius_uniformity": round(float(radius_uniformity), 4),
        # 拟合
        "simplex_fit_r2": round(float(simplex_fit_r2), 4),
    }


def compute_simplex_quality_index(geom, n_sep, N):
    """
    ★★★★★ Simplex Quality Index (SQI) — 解决问题3: 连续谱而非二元分类
    
    SQI = 加权组合:
      - n_sep_ratio: n_sep / (N-1), 理想=1.0
      - edge_uniformity: 边长均匀性, 理想=1.0
      - angle_quality: 1 - angle_deviation/90, 理想≈1.0
      - simplex_fit_r2: 正则单纯形拟合度, 理想=1.0
    
    权重: 均等, 因为每个维度同等重要
    """
    if geom is None or N < 3:
        return 0.0
    
    n_sep_ratio = n_sep / max(N - 1, 1)
    edge_uni = geom.get("edge_uniformity", 0)
    angle_dev = geom.get("angle_deviation", 90)
    angle_quality = max(0, 1 - angle_dev / 90)
    fit_r2 = geom.get("simplex_fit_r2", 0)
    
    # 加权平均 (权重: 0.25, 0.25, 0.25, 0.25)
    sqi = 0.25 * min(n_sep_ratio, 1.0) + 0.25 * edge_uni + 0.25 * angle_quality + 0.25 * fit_r2
    
    return round(float(sqi), 4)


def permutation_test(class_resids, n_permutations=200):
    """
    ★★★★★ 置换检验 — 解决问题1: 统计显著性
    
    原假设H0: 类别标签与残差无关(即随机聚类)
    方法: 随机打乱类别标签, 重新计算n_sep, 重复200次
    p-value = (perm_n_sep >= observed_n_sep) / n_permutations
    
    如果p<0.05 → n_sep显著高于随机 → 语义分离真实存在
    如果p>=0.05 → n_sep可能是偶然 → 语义分离不显著
    """
    # 观测n_sep
    all_vecs = []
    for cls in class_resids:
        all_vecs.extend(class_resids[cls])
    
    observed_n_sep, _, _ = compute_n_separating_pcs(class_resids, all_vecs)
    
    # 合并所有向量
    all_vecs_arr = np.array(all_vecs)
    n_total = len(all_vecs_arr)
    class_sizes = [len(class_resids[c]) for c in class_resids]
    class_names = list(class_resids.keys())
    
    # 置换
    count_ge = 0
    for _ in range(n_permutations):
        # 随机打乱
        perm_indices = np.random.permutation(n_total)
        perm_vecs = all_vecs_arr[perm_indices]
        
        # 重新分配类别
        perm_class_resids = {}
        idx = 0
        for i, cls in enumerate(class_names):
            perm_class_resids[cls] = perm_vecs[idx:idx + class_sizes[i]].tolist()
            idx += class_sizes[i]
        
        # 计算置换后的n_sep
        perm_n_sep, _, _ = compute_n_separating_pcs(perm_class_resids, perm_vecs.tolist())
        if perm_n_sep >= observed_n_sep:
            count_ge += 1
    
    p_value = count_ge / n_permutations
    return observed_n_sep, p_value


def collect_residuals_at_layer(model, tokenizer, layers, li, domain_dict, 
                                prompt_template, n_words=10, device="cuda"):
    """收集某层某领域的残差"""
    class_resids = {}
    for cls, words in domain_dict.items():
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
        
        if len(resids) >= 5:
            class_resids[cls] = resids
    
    return class_resids


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}")
    log(f"CCXXXIX(339): 单纯形结构严格验证 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # ====================================================================
    # ★★★★★ Part 1: 全层SQI谱图 — 解决问题2: 局部≠全局
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: ★★★★★ 全层SQI谱图 — 局部vs全局结构")
    log("  每个领域在每个层计算SQI, 绘制完整谱图")
    log("="*60)
    
    # 采样层: 每3层采一个 (平衡精度和速度)
    sample_step = max(1, n_layers // 12)
    sampled_layers = list(range(0, n_layers, sample_step))
    if n_layers - 1 not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    sampled_layers = sorted(set(sampled_layers))
    
    log(f"  采样层: {sampled_layers}")
    
    all_sqi_profiles = {}
    all_geometry = {}
    
    for domain_name, domain_data in DOMAINS.items():
        log(f"\n  --- Domain: {domain_name} ---")
        domain_dict = domain_data["classes"]
        domain_order = domain_data["order"]
        prompt_template = domain_data["prompt"]
        
        sqi_profile = {}
        geom_profile = {}
        
        for li in sampled_layers:
            class_resids = collect_residuals_at_layer(
                model, tokenizer, layers, li,
                {c: domain_dict[c] for c in domain_order if c in domain_dict},
                prompt_template, n_words=8, device=device
            )
            
            valid_classes = [c for c in domain_order if c in class_resids]
            if len(valid_classes) < 3:
                sqi_profile[f"L{li}"] = {"sqi": 0, "n_sep": 0, "N": len(valid_classes)}
                continue
            
            current_resids = {c: class_resids[c] for c in valid_classes}
            all_vecs = []
            for c in valid_classes:
                all_vecs.extend(current_resids[c])
            
            n_sep, f_ratios, Vt = compute_n_separating_pcs(current_resids, all_vecs)
            N = len(valid_classes)
            
            # 几何分析
            geom = None
            if n_sep >= 2:
                arr = np.array(all_vecs)
                centered = arr - arr.mean(axis=0)
                
                class_centers = {}
                for c in valid_classes:
                    cls_arr = np.array(current_resids[c])
                    cls_centered = cls_arr - arr.mean(axis=0)
                    pc_proj = cls_centered @ Vt[:n_sep].T
                    class_centers[c] = np.mean(pc_proj, axis=0)
                
                geom = compute_detailed_geometry(class_centers)
            
            # SQI
            sqi = compute_simplex_quality_index(geom, n_sep, N)
            
            sqi_profile[f"L{li}"] = {
                "sqi": sqi,
                "n_sep": n_sep,
                "N": N,
                "n_sep_ratio": round(n_sep / max(N - 1, 1), 4),
            }
            geom_profile[f"L{li}"] = geom
            
            geom_str = ""
            if geom:
                geom_str = (f", edge_uni={geom['edge_uniformity']:.3f}, "
                           f"angle_dev={geom['angle_deviation']:.1f}, "
                           f"fit_r2={geom['simplex_fit_r2']:.3f}")
            
            log(f"    L{li}: SQI={sqi:.4f}, n_sep={n_sep}/{N-1}{geom_str}")
        
        all_sqi_profiles[domain_name] = sqi_profile
        all_geometry[domain_name] = geom_profile
    
    results["sqi_profiles"] = all_sqi_profiles
    results["geometry"] = all_geometry
    
    # ====================================================================
    # ★★★★★ Part 2: 置换检验 — 解决问题1: 统计显著性
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: ★★★★★ 置换检验 — 统计显著性验证")
    log("  H0: 语义分离与随机聚类无差异")
    log("  p<0.05 → 拒绝H0 → 语义分离真实存在")
    log("="*60)
    
    # 只在SQI最高的层做置换检验
    permutation_results = {}
    
    for domain_name, domain_data in DOMAINS.items():
        log(f"\n  --- Domain: {domain_name} ---")
        domain_dict = domain_data["classes"]
        domain_order = domain_data["order"]
        prompt_template = domain_data["prompt"]
        
        # 找SQI最高的层
        sqi_prof = all_sqi_profiles.get(domain_name, {})
        if not sqi_prof:
            continue
        
        best_layer_key = max(sqi_prof.keys(), key=lambda k: sqi_prof[k]["sqi"])
        best_li = int(best_layer_key.replace("L", ""))
        best_sqi = sqi_prof[best_layer_key]["sqi"]
        
        log(f"    Best layer: {best_layer_key} (SQI={best_sqi:.4f})")
        
        # 也测2个对比层: 一个中等SQI, 一个低SQI
        all_sqi_vals = [(k, sqi_prof[k]["sqi"]) for k in sqi_prof]
        all_sqi_vals.sort(key=lambda x: x[1], reverse=True)
        
        test_layers_perm = [best_li]
        if len(all_sqi_vals) > 2:
            mid_idx = len(all_sqi_vals) // 2
            low_idx = -1
            mid_li = int(all_sqi_vals[mid_idx][0].replace("L", ""))
            low_li = int(all_sqi_vals[low_idx][0].replace("L", ""))
            if mid_li not in test_layers_perm:
                test_layers_perm.append(mid_li)
            if low_li not in test_layers_perm:
                test_layers_perm.append(low_li)
        
        for li in test_layers_perm:
            class_resids = collect_residuals_at_layer(
                model, tokenizer, layers, li,
                {c: domain_dict[c] for c in domain_order if c in domain_dict},
                prompt_template, n_words=8, device=device
            )
            
            valid_classes = [c for c in domain_order if c in class_resids]
            if len(valid_classes) < 3:
                continue
            
            current_resids = {c: class_resids[c] for c in valid_classes}
            
            log(f"    L{li}: Running permutation test (200 permutations)...")
            observed_n_sep, p_value = permutation_test(current_resids, n_permutations=200)
            
            sig = "★★★" if p_value < 0.01 else ("★★" if p_value < 0.05 else ("★" if p_value < 0.1 else ""))
            
            log(f"    L{li}: n_sep={observed_n_sep}, p={p_value:.4f} {sig}")
            
            key = f"{domain_name}_L{li}"
            permutation_results[key] = {
                "domain": domain_name,
                "layer": li,
                "n_sep": observed_n_sep,
                "p_value": round(float(p_value), 4),
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
            }
    
    results["permutation_tests"] = permutation_results
    
    # ====================================================================
    # ★★★★★ Part 3: 连续SQI谱 — 解决问题3: 不做绝对分类
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: ★★★★★ 连续SQI谱 — 领域质量排序")
    log("  不再是'有/无'二分, 而是0-1的连续分数")
    log("="*60)
    
    # 每个领域的最佳SQI和平均SQI
    domain_sqi_summary = {}
    for domain_name in DOMAINS:
        sqi_prof = all_sqi_profiles.get(domain_name, {})
        if not sqi_prof:
            continue
        
        sqi_vals = [v["sqi"] for v in sqi_prof.values()]
        n_sep_ratios = [v["n_sep_ratio"] for v in sqi_prof.values()]
        
        best_sqi = max(sqi_vals)
        mean_sqi = np.mean(sqi_vals)
        # 跨层稳定性: SQI的CV (越小越稳定)
        sqi_std = np.std(sqi_vals)
        sqi_cv = sqi_std / max(mean_sqi, 1e-10)
        stability = 1.0 - min(sqi_cv, 1.0)
        
        # 找最佳层的几何
        best_layer_key = max(sqi_prof.keys(), key=lambda k: sqi_prof[k]["sqi"])
        best_geom = all_geometry.get(domain_name, {}).get(best_layer_key, None)
        
        domain_sqi_summary[domain_name] = {
            "best_sqi": round(float(best_sqi), 4),
            "mean_sqi": round(float(mean_sqi), 4),
            "stability": round(float(stability), 4),
            "best_layer": best_layer_key,
            "best_n_sep_ratio": round(float(max(n_sep_ratios)), 4),
            "best_geometry": best_geom,
        }
        
        geom_str = ""
        if best_geom:
            geom_str = (f", edge_uni={best_geom['edge_uniformity']:.3f}, "
                       f"angle_dev={best_geom['angle_deviation']:.1f}, "
                       f"fit_r2={best_geom['simplex_fit_r2']:.3f}")
        
        log(f"  {domain_name:12s}: best_SQI={best_sqi:.4f}, mean_SQI={mean_sqi:.4f}, "
            f"stability={stability:.4f}, best_n_sep_ratio={max(n_sep_ratios):.4f}{geom_str}")
    
    # 排序
    sorted_domains = sorted(domain_sqi_summary.items(), key=lambda x: x[1]["best_sqi"], reverse=True)
    log(f"\n  ★ Domain Quality Ranking (by best SQI):")
    for rank, (dname, ddata) in enumerate(sorted_domains, 1):
        log(f"    {rank}. {dname:12s}: SQI={ddata['best_sqi']:.4f}")
    
    results["sqi_summary"] = domain_sqi_summary
    
    # ====================================================================
    # Part 4: ★★★★ 可加性严格验证 — 在最佳层 + 置换检验
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: ★★★★ 可加性严格验证 + 每步置换检验")
    log("  不只看Δn_sep=1, 还要验证每步的统计显著性")
    log("="*60)
    
    # 对SQI最高的两个领域做可加性验证
    top_domains = [d[0] for d in sorted_domains[:2]]
    log(f"  Top domains for additivity: {top_domains}")
    
    additivity_rigorous = {}
    
    for domain_name in top_domains:
        domain_data = DOMAINS[domain_name]
        domain_dict = domain_data["classes"]
        domain_order = domain_data["order"]
        prompt_template = domain_data["prompt"]
        
        # 找最佳层
        sqi_prof = all_sqi_profiles.get(domain_name, {})
        if not sqi_prof:
            continue
        best_layer_key = max(sqi_prof.keys(), key=lambda k: sqi_prof[k]["sqi"])
        best_li = int(best_layer_key.replace("L", ""))
        
        log(f"\n  --- {domain_name} at L{best_li} ---")
        
        # 收集所有类的残差
        full_resids = collect_residuals_at_layer(
            model, tokenizer, layers, best_li,
            {c: domain_dict[c] for c in domain_order if c in domain_dict},
            prompt_template, n_words=8, device=device
        )
        
        valid_classes = [c for c in domain_order if c in full_resids]
        if len(valid_classes) < 4:
            log(f"    Skip: only {len(valid_classes)} valid classes")
            continue
        
        layer_additivity = {}
        prev_n_sep = 0
        
        for n_classes in range(3, len(valid_classes) + 1):
            current_classes = valid_classes[:n_classes]
            current_resids = {c: full_resids[c] for c in current_classes}
            
            all_vecs = []
            for c in current_classes:
                all_vecs.extend(current_resids[c])
            
            n_sep, f_ratios, Vt = compute_n_separating_pcs(current_resids, all_vecs)
            delta_n_sep = n_sep - prev_n_sep
            
            # 几何
            geom = None
            sqi = 0
            if n_sep >= 2:
                arr = np.array(all_vecs)
                centered = arr - arr.mean(axis=0)
                
                class_centers = {}
                for c in current_classes:
                    cls_arr = np.array(current_resids[c])
                    cls_centered = cls_arr - arr.mean(axis=0)
                    pc_proj = cls_centered @ Vt[:n_sep].T
                    class_centers[c] = np.mean(pc_proj, axis=0)
                
                geom = compute_detailed_geometry(class_centers)
                sqi = compute_simplex_quality_index(geom, n_sep, n_classes)
            
            # 置换检验(只做50次以节省时间)
            _, p_value = permutation_test(current_resids, n_permutations=50)
            
            match_str = "Y" if n_sep == n_classes - 1 else "N"
            sig_str = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else "")
            
            log(f"    N={n_classes}: n_sep={n_sep}(exp={n_classes-1}), "
                f"Delta={delta_n_sep}, match={match_str}, "
                f"SQI={sqi:.4f}, p={p_value:.3f}{sig_str}"
                + (f", fit_r2={geom['simplex_fit_r2']:.3f}" if geom else ""))
            
            layer_additivity[f"N{n_classes}"] = {
                "n_classes": n_classes,
                "n_separating_PCs": n_sep,
                "expected_N_minus_1": n_classes - 1,
                "delta_n_sep": delta_n_sep,
                "match": n_sep == n_classes - 1,
                "sqi": sqi,
                "p_value": round(float(p_value), 4),
                "significant": p_value < 0.05,
                "geometry": geom,
            }
            
            prev_n_sep = n_sep
        
        additivity_rigorous[domain_name] = {
            "layer": best_li,
            "additivity": layer_additivity,
        }
    
    results["additivity_rigorous"] = additivity_rigorous
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    out_path = TEMP / f"ccxxxix_simplex_rigorous_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nResults saved to {out_path}")
    
    # ====================================================================
    # 最终汇总
    # ====================================================================
    log("\n" + "="*60)
    log("FINAL SUMMARY - CCXXXIX")
    log("="*60)
    
    # SQI排序
    log("\n--- Domain Quality Ranking (SQI) ---")
    for rank, (dname, ddata) in enumerate(sorted_domains, 1):
        log(f"  {rank}. {dname:12s}: best_SQI={ddata['best_sqi']:.4f}, "
            f"mean_SQI={ddata['mean_sqi']:.4f}, stability={ddata['stability']:.4f}")
    
    # 置换检验
    log("\n--- Permutation Tests ---")
    for key, pdata in sorted(permutation_results.items(), key=lambda x: x[1]["p_value"]):
        sig = "***" if pdata["significant_001"] else ("**" if pdata["significant_005"] else "")
        log(f"  {key}: n_sep={pdata['n_sep']}, p={pdata['p_value']:.4f} {sig}")
    
    # 可加性
    log("\n--- Rigorous Additivity ---")
    for domain_name, domain_data in additivity_rigorous.items():
        log(f"  {domain_name} (L{domain_data['layer']}):")
        for n_key, n_data in sorted(domain_data["additivity"].items()):
            sig = "***" if n_data.get("significant") else ""
            log(f"    {n_key}: n_sep={n_data['n_separating_PCs']}, "
                f"Delta={n_data['delta_n_sep']}, SQI={n_data['sqi']:.4f}, "
                f"p={n_data['p_value']:.3f}{sig}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"CCXXXIX Log - {args.model}\n")
    
    run(args.model)
