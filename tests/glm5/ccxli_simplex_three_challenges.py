"""
CCXLI(341): 单纯形三大硬伤验证
=================================
★★★★★ 三大硬伤及解决策略:

硬伤1: N较小时容易高估
  → 解决: 
    a) N=3-12的随机基线fit_r2分布, 建立判别力标准
    b) 扩展到N=8: 用更多类别词验证大N下单纯形是否仍然成立
    c) 计算每N下的判别力: 语义fit_r2 vs 随机fit_r2

硬伤2: 样本构造偏差
  → 解决:
    a) "脏数据"测试: 加入边界模糊的类别词
    b) 更自然的语义分布: 用行为词而非纯同义词
    c) 对比: 干净同义词 vs 自然词 vs 行为词

硬伤3: 单纯形可能只是"类别原型层"
  → 解决:
    a) 分析类别内结构: 方差, 维度分布
    b) 类间方差 vs 类内方差 F-ratio
    c) 类内数据是否有子结构

用法:
  python ccxli_simplex_three_challenges.py --model qwen3
  python ccxli_simplex_three_challenges.py --model glm4
  python ccxli_simplex_three_challenges.py --model deepseek7b
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
LOG = TEMP / "ccxli_three_challenges_log.txt"

# ===== 扩展类别定义 (N=8) =====
DOMAINS_EXTENDED = {
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
    # 硬伤2: "脏数据" — 边界模糊的类别
    "emotion_dirty": {
        "classes": {
            "happy": ["smile", "laugh", "cheer", "celebrate", "excited", "pleased",
                      "thrilled", "grateful", "hopeful", "optimistic"],
            "sad": ["cry", "tear", "mourn", "lament", "depressed", "lonely",
                    "heartbroken", "disappointed", "regretful", "miserable"],
            "angry": ["yell", "scream", "fight", "argue", "frustrated", "irritated",
                      "annoyed", "impatient", "hostile", "aggressive"],
            "scared": ["hide", "tremble", "shake", "run", "nervous", "worried",
                       "uneasy", "stressed", "panicked", "desperate"],
            "surprised": ["gasp", "stare", "freeze", "blink", "amazed", "curious",
                          "intrigued", "impressed", "startled", "awestruck"],
            "disgusted": ["cringe", "frown", "avoid", "reject", "repelled", "offended",
                          "insulted", "appalled", "nauseated", "sickened"],
        },
        "order": ["happy", "sad", "angry", "scared", "surprised", "disgusted"],
        "prompt": "When this happened, the person felt {word}",
    },
    # 硬伤2: 行为+情感混合 (最大模糊性)
    "emotion_behaviors": {
        "classes": {
            "happy": ["laughing", "smiling", "dancing", "singing", "clapping",
                      "cheering", "jumping", "hugging", "skipping", "beaming"],
            "sad": ["crying", "weeping", "sighing", "trembling", "slouching",
                    "sobbing", "whimpering", "collapsing", "withdrawing", "staring"],
            "angry": ["screaming", "punching", "kicking", "slamming", "gritting",
                      "yelling", "stomping", "clenching", "glaring", "shouting"],
            "scared": ["running", "hiding", "freezing", "shaking", "trembling",
                       "screaming", "flinching", "cowering", "retreating", "panicking"],
            "surprised": ["gasping", "staring", "jumping", "dropping", "freezing",
                          "blinking", "exclaiming", "stepping", "opening", "turning"],
            "disgusted": ["gagging", "turning", "spitting", "covering", "cringing",
                          "wrinkling", "pushing", "backing", "avoiding", "frowning"],
        },
        "order": ["happy", "sad", "angry", "scared", "surprised", "disgusted"],
        "prompt": "The person was {word} at the sight",
    },
}

# 原始6类定义
DOMAINS_ORIGINAL = {
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
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_simplex_fit_r2(centers):
    """计算正则单纯形拟合R² (CCXL修正版)"""
    N = centers.shape[0]
    n_dim = centers.shape[1]
    
    if N < 3:
        return 0.0, 0.0, 0.0, 0.0, 0
    
    # 边长统计
    pairwise_dists = squareform(pdist(centers))
    upper_tri = pairwise_dists[np.triu_indices(N, k=1)]
    mean_edge = np.mean(upper_tri)
    std_edge = np.std(upper_tri)
    cv_edge = std_edge / max(mean_edge, 1e-10)
    edge_uniformity = 1.0 - cv_edge
    
    # 角度统计
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
    ideal_angle = np.arccos(-1.0 / N) * 180 / np.pi if N > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    
    # 半径均匀性
    radii = np.linalg.norm(vectors, axis=1)
    cv_radius = np.std(radii) / max(np.mean(radii), 1e-10)
    radius_uniformity = 1.0 - cv_radius
    
    # 正则单纯形拟合R²
    if N - 1 > n_dim:
        simplex_fit_r2 = 0.0
    else:
        ideal_vertices = np.zeros((N, N))
        for i in range(N):
            ideal_vertices[i, i] = 1.0
        ideal_vertices = ideal_vertices - ideal_vertices.mean(axis=0)
        ideal_dists = squareform(pdist(ideal_vertices))
        ideal_mean_dist = np.mean(ideal_dists[np.triu_indices(N, k=1)])
        ideal_vertices = ideal_vertices / max(ideal_mean_dist, 1e-10) * mean_edge
        
        if n_dim >= N - 1:
            U_c, S_c, Vt_c = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
            proj_dim = min(N - 1, n_dim)
            centers_proj = (centers - centers.mean(axis=0)) @ Vt_c[:proj_dim].T
            
            U_i, S_i, Vt_i = np.linalg.svd(ideal_vertices - ideal_vertices.mean(axis=0), full_matrices=False)
            ideal_proj = (ideal_vertices - ideal_vertices.mean(axis=0)) @ Vt_i[:proj_dim].T
            
            H = ideal_proj.T @ centers_proj
            U_h, S_h, Vt_h = np.linalg.svd(H)
            R = U_h @ Vt_h  # CCXL修正
            
            aligned_ideal = ideal_proj @ R
            residual = centers_proj - aligned_ideal
            ss_res = np.sum(residual ** 2)
            ss_tot = np.sum((centers_proj - centers_proj.mean(axis=0)) ** 2)
            simplex_fit_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            simplex_fit_r2 = max(0.0, simplex_fit_r2)
        else:
            simplex_fit_r2 = 0.0
    
    return edge_uniformity, angle_deviation, radius_uniformity, simplex_fit_r2, len(angles)


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


# ===== 硬伤1: 随机基线 =====
def compute_random_baseline(N_values, n_words_per_class=8, n_trials=500, d_model=2560):
    """
    对每个N, 生成随机高斯聚类, 计算fit_r2分布
    模拟: N个聚类, 每类8个点, 类间中心随机+类内噪声
    """
    results = {}
    for N in N_values:
        fit_r2_list = []
        for trial in range(n_trials):
            # 生成N个随机聚类中心
            centers = np.random.randn(N, d_model) * 2.0
            # 每个中心周围加噪声
            all_vecs = []
            for i in range(N):
                cluster = centers[i:i+1] + np.random.randn(n_words_per_class, d_model) * 0.5
                all_vecs.append(cluster)
            # 计算类中心
            class_centers = np.array([c.mean(axis=0) for c in all_vecs])
            
            _, _, _, fit_r2, _ = compute_simplex_fit_r2(class_centers)
            fit_r2_list.append(fit_r2)
        
        arr = np.array(fit_r2_list)
        results[N] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p5": round(float(np.percentile(arr, 5)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(np.max(arr)), 4),
            "median": round(float(np.median(arr)), 4),
        }
        log(f"  随机基线 N={N}: mean={results[N]['mean']:.4f}, std={results[N]['std']:.4f}, "
            f"p95={results[N]['p95']:.4f}, max={results[N]['max']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"\n{'='*80}")
    log(f"CCXLI: 三大硬伤验证 — {model_name}")
    log(f"{'='*80}")
    
    # ===== Part 1: 随机基线 (不需要模型) =====
    log(f"\n{'='*60}")
    log(f"Part 1: 随机基线 — N=3-12的fit_r2分布 (500 trials)")
    log(f"{'='*60}")
    
    random_baseline = compute_random_baseline(
        N_values=list(range(3, 13)),
        n_words_per_class=8,
        n_trials=500,
        d_model=2560
    )
    
    # ===== Part 2: 模型测试 =====
    log(f"\n{'='*60}")
    log(f"Part 2: 模型测试 — 扩展类别(N=8) + 脏数据")
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
        "random_baseline": {str(k): v for k, v in random_baseline.items()},
        "model_results": {},
        "within_class": {},
    }
    
    # 合并所有领域
    all_domains = {}
    all_domains.update(DOMAINS_ORIGINAL)
    all_domains.update(DOMAINS_EXTENDED)
    
    # 采样层
    sample_step = max(1, n_layers // 12)
    sampled_layers = list(range(0, n_layers, sample_step))
    if n_layers - 1 not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    sampled_layers = sorted(set(sampled_layers))
    # 限制层数
    if len(sampled_layers) > 15:
        sampled_layers = sampled_layers[:15]
    log(f"  采样层: {sampled_layers}")
    
    for domain_name, domain_info in all_domains.items():
        log(f"\n--- Domain: {domain_name} ---")
        classes = domain_info["classes"]
        order = domain_info["order"]
        prompt_template = domain_info["prompt"]
        N = len(order)
        
        domain_results = []
        best_fit_r2 = -1
        best_layer_data = None
        
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
            
            class_centers = {c: np.mean(current_resids[c], axis=0) for c in valid_classes}
            centers_arr = np.array([class_centers[c] for c in valid_classes])
            
            n_sep = compute_n_separating_pcs(current_resids, all_vecs)
            edge_uni, angle_dev, radius_uni, fit_r2, n_angles = compute_simplex_fit_r2(centers_arr)
            
            layer_result = {
                "layer": li,
                "N": len(valid_classes),
                "n_sep": n_sep,
                "edge_uniformity": round(float(edge_uni), 4),
                "angle_deviation": round(float(angle_dev), 2),
                "radius_uniformity": round(float(radius_uni), 4),
                "fit_r2": round(float(fit_r2), 4),
            }
            domain_results.append(layer_result)
            log(f"  L{li}: N={len(valid_classes)}, n_sep={n_sep}, fit_r2={fit_r2:.4f}, "
                f"edge_uni={edge_uni:.4f}, angle_dev={angle_dev:.2f}°")
            
            if fit_r2 > best_fit_r2:
                best_fit_r2 = fit_r2
                best_layer_data = {
                    "layer": li,
                    "class_resids": current_resids,
                    "all_vecs": all_vecs,
                    "centers_arr": centers_arr,
                }
        
        if not domain_results:
            log(f"  ★ 无有效结果")
            continue
        
        best = max(domain_results, key=lambda x: x["fit_r2"])
        log(f"  ★ Best L{best['layer']}: fit_r2={best['fit_r2']:.4f}")
        
        # 硬伤3: 类别内结构分析 (在最佳层)
        within_info = {}
        if best_layer_data:
            cr = best_layer_data["class_resids"]
            av = np.array(best_layer_data["all_vecs"])
            overall_mean = np.mean(av, axis=0)
            
            class_centers_arr = best_layer_data["centers_arr"]
            between_var = np.mean(np.var(class_centers_arr, axis=0))
            within_vars = [np.mean(np.var(cr[c], axis=0)) for c in cr]
            avg_within_var = np.mean(within_vars)
            f_ratio = between_var / max(avg_within_var, 1e-10)
            
            for cls_name in cr:
                arr = np.array(cr[cls_name])
                center = np.mean(arr, axis=0)
                centered = arr - center
                _, S, _ = np.linalg.svd(centered, full_matrices=False)
                total_var = np.sum(S**2)
                cumvar = np.cumsum(S**2) / total_var
                n_90 = int(np.searchsorted(cumvar, 0.9)) + 1
                within_info[cls_name] = {
                    "n_words": len(cr[cls_name]),
                    "n_dims_90var": n_90,
                    "within_var": round(float(np.mean(np.var(arr, axis=0))), 6),
                }
            
            within_info["_summary"] = {
                "between_var": round(float(between_var), 6),
                "avg_within_var": round(float(avg_within_var), 6),
                "f_ratio": round(float(f_ratio), 2),
            }
            
            log(f"  类间方差={between_var:.6f}, 类内方差={avg_within_var:.6f}, F_ratio={f_ratio:.2f}")
        
        all_results["model_results"][domain_name] = {
            "N": N,
            "best_layer": best["layer"],
            "best_fit_r2": best["fit_r2"],
            "best_edge_uni": best["edge_uniformity"],
            "best_angle_dev": best["angle_deviation"],
            "best_n_sep": best["n_sep"],
            "all_layers": domain_results,
            "within_class": within_info,
        }
    
    release_model(model)
    
    # ===== 保存结果 =====
    out_path = TEMP / f"ccxli_three_challenges_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存: {out_path}")
    
    # ===== 打印汇总 =====
    log(f"\n{'='*80}")
    log(f"CCXLI 汇总 — {model_name}")
    log(f"{'='*80}")
    
    log(f"\n--- 硬伤1: 随机基线 vs 语义fit_r2 ---")
    log(f"{'N':>3} | {'随机mean':>10} | {'随机p95':>10} | {'随机max':>10}")
    log(f"{'-'*50}")
    for N in range(3, 13):
        rb = random_baseline[N]
        log(f"{N:>3} | {rb['mean']:>10.4f} | {rb['p95']:>10.4f} | {rb['max']:>10.4f}")
    
    log(f"\n--- 所有领域结果 ---")
    log(f"{'Domain':>20} | {'N':>3} | {'Best fit_r2':>12} | {'n_sep':>5} | {'edge_uni':>10} | {'angle_dev':>10}")
    log(f"{'-'*80}")
    for domain_name, dr in all_results["model_results"].items():
        log(f"{domain_name:>20} | {dr['N']:>3} | {dr['best_fit_r2']:>12.4f} | {dr['best_n_sep']:>5} | "
            f"{dr['best_edge_uni']:>10.4f} | {dr['best_angle_dev']:>8.2f}°")
    
    # N=6 vs N=8 对比
    log(f"\n--- N=6 vs N=8 对比 (硬伤1核心) ---")
    for base in ["habitat", "emotion", "occupation"]:
        k6 = f"{base}_6"
        k8 = f"{base}_8"
        r6 = all_results["model_results"].get(k6, {})
        r8 = all_results["model_results"].get(k8, {})
        if r6 and r8:
            delta = r8["best_fit_r2"] - r6["best_fit_r2"]
            # 检查N=8的fit_r2是否仍远高于随机基线
            rb8 = random_baseline.get(8, {})
            above_random = r8["best_fit_r2"] > rb8.get("p95", 0)
            log(f"  {base}: N=6 fit_r2={r6['best_fit_r2']:.4f}, N=8 fit_r2={r8['best_fit_r2']:.4f}, "
                f"Δ={delta:+.4f}, {'>>>' if above_random else '??'} 高于随机p95({rb8.get('p95',0):.4f})")
    
    # 脏数据 vs 干净数据对比
    log(f"\n--- 脏数据 vs 干净数据 (硬伤2核心) ---")
    clean = all_results["model_results"].get("emotion_6", {})
    dirty = all_results["model_results"].get("emotion_dirty", {})
    behaviors = all_results["model_results"].get("emotion_behaviors", {})
    if clean:
        log(f"  干净(synonyms):  fit_r2={clean['best_fit_r2']:.4f}, edge_uni={clean['best_edge_uni']:.4f}")
    if dirty:
        log(f"  脏(natural):     fit_r2={dirty['best_fit_r2']:.4f}, edge_uni={dirty['best_edge_uni']:.4f}")
    if behaviors:
        log(f"  行为(behaviors): fit_r2={behaviors['best_fit_r2']:.4f}, edge_uni={behaviors['best_edge_uni']:.4f}")
    
    # 硬伤3汇总
    log(f"\n--- 硬伤3: 类间vs类内 (单纯形=原型层?) ---")
    for domain_name, dr in all_results["model_results"].items():
        wi = dr.get("within_class", {})
        summary = wi.get("_summary", {})
        if summary:
            log(f"  {domain_name}: F_ratio={summary['f_ratio']:.2f} "
                f"(between={summary['between_var']:.6f}, within={summary['avg_within_var']:.6f})")
    
    log(f"\nDone! {model_name}")


if __name__ == "__main__":
    main()
