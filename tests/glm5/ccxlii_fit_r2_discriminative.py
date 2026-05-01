"""
CCXLII(342): 硬伤1核心验证 — 高维空间中fit_r2的判别力问题
=============================================================
★★★★★ 关键发现(CCXLI): 在d_model=2560维空间中, 随机数据fit_r2≈0.999!

原因: N个点在d>>N维空间中总是可以完美拟合正则单纯形
  → 维度远大于点数, 自由度太高
  → 任何N个点都可以通过旋转对齐到正则单纯形

正确做法:
  1. 在模型数据中, centers已被投影到n_sep维(分离子空间)
  2. 随机基线也应在n_sep维空间中生成
  3. 更严格: 在n_sep维空间中, 生成N个随机高斯聚类中心

关键对比:
  - 语义fit_r2(在n_sep维空间) vs 随机fit_r2(在n_sep维空间)
  - 如果语义fit_r2 >> 随机fit_r2 → 单纯形结构有意义
  - 如果语义fit_r2 ≈ 随机fit_r2 → 单纯形结构无意义

附加验证:
  - 边长均匀性(edge_uniformity)的判别力
  - 角度偏差(angle_deviation)的判别力
  - 组合判别: fit_r2 + edge_uni + angle_dev

用法:
  python ccxlii_fit_r2_discriminative.py --model qwen3
  python ccxlii_fit_r2_discriminative.py --model glm4
  python ccxlii_fit_r2_discriminative.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxlii_discriminative_log.txt"

# 类别定义
DOMAINS = {
    "habitat_6": {
        "classes": {
            "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
                     "fox", "deer", "bear", "wolf"],
            "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
                      "crab", "seal", "squid", "lobster", "jellyfish"],
            "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
                    "falcon", "pigeon", "robin"],
            "space": ["astronaut", "satellite", "comet", "meteor", "nebula",
                      "quasar", "asteroid", "rocket", "spaceship", "planet"],
            "microscopic": ["bacterium", "virus", "cell", "amoeba", "paramecium",
                            "euglena", "diatom", "plasmodium", "ribosome"],
            "virtual": ["algorithm", "program", "software", "database", "network",
                        "protocol", "encryption", "firewall", "browser"],
        },
        "order": ["land", "ocean", "sky", "space", "microscopic", "virtual"],
        "prompt": "The {word} lives in the",
    },
    "emotion_6": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                      "contentment", "pleasure", "gladness", "merriment"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                      "indignation", "animosity", "vexation", "exasperation"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
                       "anxiety", "apprehension", "trepidation", "phobia"],
            "surprised": ["astonishment", "amazement", "wonder", "shock", "startle",
                          "disbelief", "bewilderment", "incredulity"],
            "disgusted": ["revulsion", "repugnance", "nausea", "loathing", "abhorrence",
                          "aversion", "distaste", "antipathy"],
        },
        "order": ["happy", "sad", "angry", "scared", "surprised", "disgusted"],
        "prompt": "The person felt {word} about the",
    },
    "occupation_6": {
        "classes": {
            "doctor": ["surgeon", "physician", "nurse", "therapist", "pediatrician",
                       "cardiologist", "dermatologist", "neurologist", "psychiatrist"],
            "teacher": ["professor", "instructor", "educator", "tutor", "lecturer",
                        "mentor", "coach", "trainer", "academic"],
            "engineer": ["architect", "designer", "developer", "programmer", "mechanic",
                         "technician", "builder", "constructor", "inventor"],
            "artist": ["painter", "sculptor", "musician", "dancer", "actor",
                       "singer", "poet", "writer", "composer"],
            "lawyer": ["attorney", "advocate", "counsel", "barrister", "solicitor",
                       "prosecutor", "defender", "judge", "magistrate"],
            "chef": ["cook", "baker", "pastry", "butcher", "sous",
                     "prep", "saucier", "caterer", "restaurateur"],
        },
        "order": ["doctor", "teacher", "engineer", "artist", "lawyer", "chef"],
        "prompt": "{Word} is a skilled professional who",
    },
    "habitat_8": {
        "classes": {
            "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
                     "fox", "deer", "bear", "wolf"],
            "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
                      "crab", "seal", "squid", "lobster", "jellyfish"],
            "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
                    "falcon", "pigeon", "robin"],
            "space": ["astronaut", "satellite", "comet", "meteor", "nebula",
                      "quasar", "asteroid", "rocket", "spaceship", "planet"],
            "microscopic": ["bacterium", "virus", "cell", "amoeba", "paramecium",
                            "euglena", "diatom", "plasmodium", "ribosome"],
            "virtual": ["algorithm", "program", "software", "database", "network",
                        "protocol", "encryption", "firewall", "browser"],
            "underground": ["mole", "worm", "ant", "termite", "badger", "ferret",
                            "gopher", "meerkat", "shrew", "vole"],
            "freshwater": ["frog", "trout", "beaver", "otter", "heron", "duck",
                           "catfish", "newt", "salamander", "carp"],
        },
        "order": ["land", "ocean", "sky", "space", "microscopic", "virtual", "underground", "freshwater"],
        "prompt": "The {word} lives in the",
    },
    "emotion_8": {
        "classes": {
            "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                      "contentment", "pleasure", "gladness", "merriment"],
            "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning"],
            "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                      "indignation", "animosity", "vexation", "exasperation"],
            "scared": ["fear", "terror", "dread", "panic", "fright", "horror",
                       "anxiety", "apprehension", "trepidation", "phobia"],
            "surprised": ["astonishment", "amazement", "wonder", "shock", "startle",
                          "disbelief", "bewilderment", "incredulity"],
            "disgusted": ["revulsion", "repugnance", "nausea", "loathing", "abhorrence",
                          "aversion", "distaste", "antipathy"],
            "confused": ["baffled", "perplexed", "puzzled", "bewildered", "confounded",
                         "mystified", "stumped", "disoriented", "flustered", "uncertain"],
            "proud": ["pride", "dignity", "honor", "triumph", "accomplishment",
                      "satisfaction", "prestige", "esteem", "glory", "arrogance"],
        },
        "order": ["happy", "sad", "angry", "scared", "surprised", "disgusted", "confused", "proud"],
        "prompt": "The person felt {word} about the",
    },
    "occupation_8": {
        "classes": {
            "doctor": ["surgeon", "physician", "nurse", "therapist", "pediatrician",
                       "cardiologist", "dermatologist", "neurologist", "psychiatrist"],
            "teacher": ["professor", "instructor", "educator", "tutor", "lecturer",
                        "mentor", "coach", "trainer", "academic"],
            "engineer": ["architect", "designer", "developer", "programmer", "mechanic",
                         "technician", "builder", "constructor", "inventor"],
            "artist": ["painter", "sculptor", "musician", "dancer", "actor",
                       "singer", "poet", "writer", "composer"],
            "lawyer": ["attorney", "advocate", "counsel", "barrister", "solicitor",
                       "prosecutor", "defender", "judge", "magistrate"],
            "chef": ["cook", "baker", "pastry", "butcher", "sous",
                     "prep", "saucier", "caterer", "restaurateur"],
            "scientist": ["researcher", "physicist", "chemist", "biologist", "astronomer",
                          "geologist", "ecologist", "mathematician", "geneticist"],
            "farmer": ["agriculturist", "rancher", "grower", "planter", "harvester",
                       "breeder", "cultivator", "dairyman", "shepherd"],
        },
        "order": ["doctor", "teacher", "engineer", "artist", "lawyer", "chef", "scientist", "farmer"],
        "prompt": "{Word} is a skilled professional who",
    },
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_geometry_in_subspace(centers, n_sep=None):
    """
    在分离子空间中计算几何指标
    centers: [N, d_model] — 原始高维类中心
    n_sep: 分离维度数 (None=自动计算)
    
    返回: 在n_sep维子空间中的 fit_r2, edge_uniformity, angle_deviation, n_sep
    """
    N = centers.shape[0]
    
    # 投影到分离子空间
    centered = centers - centers.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    if n_sep is None:
        # 自动确定: 用方差解释>90%的维度
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        n_sep = max(N - 1, int(np.searchsorted(cumvar, 0.99)) + 1)
    
    # 投影到n_sep维
    proj = centered @ Vt[:n_sep].T  # [N, n_sep]
    
    # 在投影空间中计算几何
    # 边长
    pairwise_dists = squareform(pdist(proj))
    upper_tri = pairwise_dists[np.triu_indices(N, k=1)]
    mean_edge = np.mean(upper_tri)
    std_edge = np.std(upper_tri)
    cv_edge = std_edge / max(mean_edge, 1e-10)
    edge_uniformity = 1.0 - cv_edge
    
    # 角度
    centroid = np.mean(proj, axis=0)
    vectors = proj - centroid
    angles = []
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = vectors[i], vectors[j]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_a) * 180 / np.pi)
    
    mean_angle = np.mean(angles) if angles else 0
    ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi if N > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    
    # 正则单纯形拟合R² (在n_sep维空间中)
    if n_sep >= N - 1:
        # 构造理想正则单纯形
        ideal_vertices = np.zeros((N, N))
        for i in range(N):
            ideal_vertices[i, i] = 1.0
        ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
        ideal_dists = squareform(pdist(ideal_vertices))
        ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
        ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
        
        # 理想单纯形也投影到n_sep维
        U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
        ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:n_sep].T
        
        # Procrustes对齐
        H = ideal_proj.T @ proj
        U_h, S_h, Vt_h = np.linalg.svd(H)
        R = U_h @ Vt_h
        
        aligned_ideal = ideal_proj @ R
        residual = proj - aligned_ideal
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((proj - proj.mean(axis=0)) ** 2)
        fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        fit_r2 = max(0.0, fit_r2)
    else:
        fit_r2 = 0.0
    
    return fit_r2, edge_uniformity, angle_deviation, n_sep


def compute_random_baseline_in_subspace(N_values, n_trials=1000):
    """
    ★★★★★ 关键修正: 在n_sep维子空间中生成随机基线
    
    对于每个(N, n_sep)组合:
      1. 在n_sep维空间中生成N个随机高斯点
      2. 计算fit_r2, edge_uniformity, angle_deviation
      3. 统计分布
    
    这才是公平的比较: 语义数据也是在n_sep维子空间中
    """
    results = {}
    for N in N_values:
        # n_sep = N-1 (正则单纯形的维度)
        n_sep = N - 1
        fit_r2_list = []
        edge_uni_list = []
        angle_dev_list = []
        
        for trial in range(n_trials):
            # 在n_sep维空间中生成N个随机点
            centers = np.random.randn(N, n_sep)
            
            # 计算几何
            pairwise_dists = squareform(pdist(centers))
            upper_tri = pairwise_dists[np.triu_indices(N, k=1)]
            mean_edge = np.mean(upper_tri)
            std_edge = np.std(upper_tri)
            cv_edge = std_edge / max(mean_edge, 1e-10)
            eu = 1.0 - cv_edge
            
            centroid = np.mean(centers, axis=0)
            vectors = centers - centroid
            angles = []
            for i in range(N):
                for j in range(i + 1, N):
                    v1, v2 = vectors[i], vectors[j]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                        angles.append(np.arccos(cos_a) * 180 / np.pi)
            
            mean_angle = np.mean(angles) if angles else 0
            ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi
            ad = abs(mean_angle - ideal_angle)
            
            # fit_r2
            ideal_vertices = np.zeros((N, N))
            for i in range(N):
                ideal_vertices[i, i] = 1.0
            ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
            ideal_dists = squareform(pdist(ideal_vertices))
            ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
            ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
            
            U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
            ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:n_sep].T
            
            H = ideal_proj.T @ centers
            U_h, S_h, Vt_h = np.linalg.svd(H)
            R = U_h @ Vt_h
            aligned_ideal = ideal_proj @ R
            ss_res = np.sum((centers - aligned_ideal) ** 2)
            ss_tot = np.sum((centers - centers.mean(axis=0)) ** 2)
            fr2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-10))
            
            fit_r2_list.append(fr2)
            edge_uni_list.append(eu)
            angle_dev_list.append(ad)
        
        results[N] = {
            "n_sep": n_sep,
            "n_trials": n_trials,
            "fit_r2": {
                "mean": round(float(np.mean(fit_r2_list)), 4),
                "std": round(float(np.std(fit_r2_list)), 4),
                "p5": round(float(np.percentile(fit_r2_list, 5)), 4),
                "p95": round(float(np.percentile(fit_r2_list, 95)), 4),
                "max": round(float(np.max(fit_r2_list)), 4),
            },
            "edge_uniformity": {
                "mean": round(float(np.mean(edge_uni_list)), 4),
                "std": round(float(np.std(edge_uni_list)), 4),
                "p5": round(float(np.percentile(edge_uni_list, 5)), 4),
                "p95": round(float(np.percentile(edge_uni_list, 95)), 4),
            },
            "angle_deviation": {
                "mean": round(float(np.mean(angle_dev_list)), 2),
                "std": round(float(np.std(angle_dev_list)), 2),
                "p5": round(float(np.percentile(angle_dev_list, 5)), 2),
                "p95": round(float(np.percentile(angle_dev_list, 95)), 2),
            },
        }
        log(f"  随机基线 N={N}(n_sep={n_sep}): fit_r2 mean={results[N]['fit_r2']['mean']:.4f} "
            f"p95={results[N]['fit_r2']['p95']:.4f} max={results[N]['fit_r2']['max']:.4f} | "
            f"edge_uni mean={results[N]['edge_uniformity']['mean']:.4f} | "
            f"angle_dev mean={results[N]['angle_deviation']['mean']:.2f}°")
    
    return results


def compute_n_separating_pcs(class_resids, all_vecs):
    """计算分离维度数"""
    arr = np.array(all_vecs)
    centered = arr - arr.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    n_classes = len(class_resids)
    n_pc = min(n_classes + 3, Vt.shape[0])
    
    class_proj = {}
    for cls in class_resids:
        cls_arr = np.array(class_resids[cls])
        cls_centered = cls_arr - arr.mean(axis=0)
        pc_proj = cls_centered @ Vt[:n_pc].T
        class_proj[cls] = pc_proj
    
    n_separating = 0
    for pc_i in range(n_pc):
        means = [np.mean(class_proj[c][:, pc_i]) for c in class_resids]
        within_vars = [np.var(class_proj[c][:, pc_i]) for c in class_resids]
        f_ratio = np.var(means) / max(np.mean(within_vars), 1e-10)
        if f_ratio > 1.0:
            n_separating += 1
    
    return n_separating


def collect_residuals_at_layer(model, tokenizer, layers, li, domain_dict, 
                                prompt_template, n_words=8, device="cuda"):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"\n{'='*80}")
    log(f"CCXLII: fit_r2判别力核心验证 — {model_name}")
    log(f"{'='*80}")
    
    # ===== Part 1: 修正的随机基线 — 在n_sep维子空间中 =====
    log(f"\n{'='*60}")
    log(f"Part 1: 修正随机基线 — 在n_sep=N-1维子空间中 (1000 trials)")
    log(f"{'='*60}")
    
    random_baseline = compute_random_baseline_in_subspace(
        N_values=list(range(3, 13)),
        n_trials=1000,
    )
    
    # ===== Part 2: 模型测试 =====
    log(f"\n{'='*60}")
    log(f"Part 2: 模型测试 — 在分离子空间中计算几何")
    log(f"{'='*60}")
    
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
        "random_baseline_subspace": {str(k): v for k, v in random_baseline.items()},
        "model_results": {},
    }
    
    # 采样层
    sample_step = max(1, n_layers // 10)
    sampled_layers = list(range(0, n_layers, sample_step))
    if n_layers - 1 not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    sampled_layers = sorted(set(sampled_layers))
    if len(sampled_layers) > 12:
        sampled_layers = sampled_layers[:12]
    log(f"  采样层: {sampled_layers}")
    
    for domain_name, domain_info in DOMAINS.items():
        log(f"\n--- Domain: {domain_name} ---")
        classes = domain_info["classes"]
        order = domain_info["order"]
        prompt_template = domain_info["prompt"]
        N = len(order)
        
        domain_results = []
        
        for li in sampled_layers:
            class_resids = collect_residuals_at_layer(
                model, tokenizer, layers, li,
                {c: classes[c] for c in order},
                prompt_template, n_words=8, device=device
            )
            
            valid_classes = [c for c in order if c in class_resids]
            if len(valid_classes) < 3:
                continue
            
            current_resids = {c: class_resids[c] for c in valid_classes}
            all_vecs = []
            for c in valid_classes:
                all_vecs.extend(current_resids[c])
            
            class_centers = np.array([np.mean(current_resids[c], axis=0) for c in valid_classes])
            n_sep = compute_n_separating_pcs(current_resids, all_vecs)
            
            # ★★★★★ 在n_sep维子空间中计算几何
            fit_r2, edge_uni, angle_dev, used_n_sep = compute_geometry_in_subspace(
                class_centers, n_sep=n_sep
            )
            
            # 也计算在N-1维子空间中的几何 (与随机基线公平对比)
            fit_r2_Nm1, edge_uni_Nm1, angle_dev_Nm1, _ = compute_geometry_in_subspace(
                class_centers, n_sep=N-1
            )
            
            layer_result = {
                "layer": li,
                "N": len(valid_classes),
                "n_sep": n_sep,
                "used_n_sep": used_n_sep,
                # 在n_sep维子空间
                "fit_r2": round(float(fit_r2), 4),
                "edge_uniformity": round(float(edge_uni), 4),
                "angle_deviation": round(float(angle_dev), 2),
                # 在N-1维子空间 (与随机基线对比)
                "fit_r2_Nm1": round(float(fit_r2_Nm1), 4),
                "edge_uniformity_Nm1": round(float(edge_uni_Nm1), 4),
                "angle_deviation_Nm1": round(float(angle_dev_Nm1), 2),
            }
            domain_results.append(layer_result)
            log(f"  L{li}: N={len(valid_classes)}, n_sep={n_sep}, "
                f"fit_r2(n_sep={used_n_sep})={fit_r2:.4f}, fit_r2(N-1)={fit_r2_Nm1:.4f}, "
                f"edge_uni={edge_uni_Nm1:.4f}, angle_dev={angle_dev_Nm1:.2f}°")
        
        if not domain_results:
            continue
        
        best = max(domain_results, key=lambda x: x["fit_r2_Nm1"])
        log(f"  ★ Best L{best['layer']}: fit_r2(N-1)={best['fit_r2_Nm1']:.4f}, "
            f"n_sep={best['n_sep']}, edge_uni={best['edge_uniformity_Nm1']:.4f}")
        
        # 与随机基线对比
        rb = random_baseline.get(N, {})
        rb_fit = rb.get("fit_r2", {})
        rb_eu = rb.get("edge_uniformity", {})
        rb_ad = rb.get("angle_deviation", {})
        
        p_value_fit = sum(1 for _ in range(1000) if np.random.random() < 0.01) / 1000  # placeholder
        semantic_fit_r2 = best["fit_r2_Nm1"]
        random_mean = rb_fit.get("mean", 0)
        random_p95 = rb_fit.get("p95", 0)
        
        log(f"  vs 随机基线: semantic={semantic_fit_r2:.4f} vs random_mean={random_mean:.4f} "
            f"random_p95={random_p95:.4f} → {'>>>' if semantic_fit_r2 > random_p95 else '<<<'}")
        log(f"  edge_uni: semantic={best['edge_uniformity_Nm1']:.4f} vs random_mean={rb_eu.get('mean',0):.4f}")
        log(f"  angle_dev: semantic={best['angle_deviation_Nm1']:.2f}° vs random_mean={rb_ad.get('mean',0):.2f}°")
        
        all_results["model_results"][domain_name] = {
            "N": N,
            "best_layer": best["layer"],
            "best_fit_r2_Nm1": best["fit_r2_Nm1"],
            "best_edge_uni_Nm1": best["edge_uniformity_Nm1"],
            "best_angle_dev_Nm1": best["angle_deviation_Nm1"],
            "best_n_sep": best["n_sep"],
            "all_layers": domain_results,
        }
    
    release_model(model)
    
    # ===== 保存 =====
    out_path = TEMP / f"ccxlii_discriminative_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存: {out_path}")
    
    # ===== 汇总 =====
    log(f"\n{'='*80}")
    log(f"CCXLII 汇总 — {model_name}")
    log(f"{'='*80}")
    
    log(f"\n--- 修正随机基线 (在n_sep=N-1维子空间中) ---")
    log(f"{'N':>3} | {'n_sep':>5} | {'fit_r2 mean':>12} | {'fit_r2 p95':>12} | {'fit_r2 max':>12} | {'edge_uni mean':>14} | {'angle_dev mean':>16}")
    log(f"{'-'*90}")
    for N in range(3, 13):
        rb = random_baseline[N]
        log(f"{N:>3} | {rb['n_sep']:>5} | {rb['fit_r2']['mean']:>12.4f} | {rb['fit_r2']['p95']:>12.4f} | "
            f"{rb['fit_r2']['max']:>12.4f} | {rb['edge_uniformity']['mean']:>14.4f} | {rb['angle_deviation']['mean']:>16.2f}°")
    
    log(f"\n--- 语义数据 vs 随机基线 ---")
    log(f"{'Domain':>20} | {'N':>3} | {'Best L':>6} | {'fit_r2':>8} | {'edge_uni':>9} | {'angle_dev':>11} | "
        f"{'rand fit_r2':>12} | {'rand edge':>10} | {'rand angle':>11} | {'Verdict':>10}")
    log(f"{'-'*130}")
    for domain_name, dr in all_results["model_results"].items():
        N = dr["N"]
        rb = random_baseline.get(N, {})
        rb_fit = rb.get("fit_r2", {}).get("mean", 0)
        rb_eu = rb.get("edge_uniformity", {}).get("mean", 0)
        rb_ad = rb.get("angle_deviation", {}).get("mean", 0)
        
        # 判定: 语义是否显著高于随机
        fit_above = dr["best_fit_r2_Nm1"] > rb.get("fit_r2", {}).get("p95", 0)
        eu_above = dr["best_edge_uni_Nm1"] > rb.get("edge_uniformity", {}).get("p95", 0)
        ad_below = dr["best_angle_dev_Nm1"] < rb.get("angle_deviation", {}).get("p5", 0)
        
        verdict = ""
        if fit_above and eu_above and ad_below:
            verdict = "★★★ 强"
        elif fit_above and (eu_above or ad_below):
            verdict = "★★ 中"
        elif fit_above:
            verdict = "★ 弱"
        else:
            verdict = "✗ 无"
        
        log(f"{domain_name:>20} | {N:>3} | L{dr['best_layer']:>4} | {dr['best_fit_r2_Nm1']:>8.4f} | "
            f"{dr['best_edge_uni_Nm1']:>9.4f} | {dr['best_angle_dev_Nm1']:>9.2f}° | "
            f"{rb_fit:>12.4f} | {rb_eu:>10.4f} | {rb_ad:>9.2f}° | {verdict:>10}")
    
    # N=6 vs N=8
    log(f"\n--- N=6 vs N=8 对比 ---")
    for base in ["habitat", "emotion", "occupation"]:
        k6 = f"{base}_6"
        k8 = f"{base}_8"
        r6 = all_results["model_results"].get(k6, {})
        r8 = all_results["model_results"].get(k8, {})
        if r6 and r8:
            rb6 = random_baseline.get(6, {}).get("fit_r2", {}).get("p95", 0)
            rb8 = random_baseline.get(8, {}).get("fit_r2", {}).get("p95", 0)
            delta = r8["best_fit_r2_Nm1"] - r6["best_fit_r2_Nm1"]
            log(f"  {base}: N=6 fit_r2={r6['best_fit_r2_Nm1']:.4f} (rand_p95={rb6:.4f}), "
                f"N=8 fit_r2={r8['best_fit_r2_Nm1']:.4f} (rand_p95={rb8:.4f}), Δ={delta:+.4f}")
    
    log(f"\nDone! {model_name}")


if __name__ == "__main__":
    main()
