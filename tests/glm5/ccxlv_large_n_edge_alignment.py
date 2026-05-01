"""
CCXLV(345): 大N边对齐验证 — 排除N=4巧合
============================================
★★★★★ CCXLIV发现: N=4时4个情感强度轨迹沿单纯形边!
  - 平均边对齐度=0.90-0.94
  - 48/48层×情感组合全部EDGE-ALIGNED

★★★★ 硬伤: N=4只有6条边, 4个轨迹×6条边, 巧合概率高!
  - 随机方向与某条边对齐>0.7的概率 ≈ arcsin(0.7)/π ≈ 0.24
  - 4个轨迹至少一个>0.7的概率 ≈ 1-(1-0.24)^4 ≈ 0.67
  - 太高了! 不能排除巧合!

★★★★★ 解决方案: 用N=6-8测试
  - N=6: 15条边, N=8: 28条边
  - 如果边对齐仍然>0.7 → 确认是真实性质
  - 如果边对齐<0.7 → N=4是巧合, 需要重新审视

同时验证跨领域: occupation和color

附加分析:
1. 注意力头对单纯形结构的贡献 (骨架→流形的机制)
2. 跨层演化: 单纯形fit_r2和边对齐度随层变化

用法:
  python ccxlv_large_n_edge_alignment.py --model qwen3
  python ccxlv_large_n_edge_alignment.py --model glm4
  python ccxlv_large_n_edge_alignment.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxlv_large_n_log.txt"

# ============================================================
# 领域定义 — 大N测试
# ============================================================

# N=6 情感 (6个基本情感, 心理学Ekman模型)
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

# N=6 职业 (6种职业类别)
OCCUPATION_6 = {
    "classes": {
        "doctor":   ["physician", "surgeon", "medic", "clinician", "healer", "doctor",
                     "practitioner", "specialist", "internist", "pediatrician", "cardiologist", "neurologist"],
        "teacher":  ["instructor", "educator", "tutor", "professor", "lecturer", "coach",
                     "mentor", "trainer", "academic", "scholar", "pedagogue", "schoolteacher"],
        "artist":   ["painter", "sculptor", "illustrator", "musician", "dancer", "actor",
                     "poet", "writer", "singer", "photographer", "designer", "composer"],
        "engineer": ["architect", "builder", "constructor", "developer", "programmer", "technician",
                     "mechanic", "inventor", "designer", "fabricator", "planner", "analyst"],
        "chef":     ["cook", "baker", "culinary", "cuisinier", "pastry", "grill",
                     "saucier", "roaster", "barista", "bartender", "caterer", "restaurateur"],
        "farmer":   ["agriculturist", "grower", "rancher", "cultivator", "breeder", "planter",
                     "harvester", "plowman", "gardener", "horticulturist", "agrarian", "dairyman"],
    },
    "order": ["doctor", "teacher", "artist", "engineer", "chef", "farmer"],
    "prompt": "The person worked as a {word} in the",
}

# N=8 动物栖息地 (8种栖息地)
HABITAT_8 = {
    "classes": {
        "ocean":    ["sea", "marine", "pelagic", "abyssal", "coastal", "tidal",
                     "deep-sea", "reef", "shore", "nautical", "aquatic", "saltwater"],
        "forest":   ["woodland", "jungle", "timber", "grove", "copse", "thicket",
                     "bush", "canopy", "rainforest", "taiga", "woodland", "wildwood"],
        "desert":   ["arid", "sandy", "barren", "wasteland", "dune", "scrubland",
                     "dryland", "badlands", "sahara", "wilderness", "steppe", "mesa"],
        "mountain": ["alpine", "peak", "summit", "ridge", "highland", "elevation",
                     "cliff", "plateau", "crag", "bluff", "pinnacle", "summit"],
        "grassland":["prairie", "savanna", "meadow", "pasture", "plain", "steppe",
                     "veldt", "pampas", "tundra", "field", "rangeland", "lawn"],
        "wetland":  ["swamp", "marsh", "bog", "mire", "fen", "slough",
                     "estuary", "bayou", "morass", "quagmire", "tideland", "delta"],
        "arctic":   ["polar", "glacial", "frozen", "tundra", "icecap", "permafrost",
                     "icefield", "snowfield", "frost", "subzero", "cryosphere", "icebound"],
        "cave":     ["cavern", "grotto", "underground", "subterranean", "cavity", "hollow",
                     "lair", "den", "burrow", "tunnel", "mine", "pit"],
    },
    "order": ["ocean", "forest", "desert", "mountain", "grassland", "wetland", "arctic", "cave"],
    "prompt": "The animal lived in the {word} near the",
}

# 强度梯度 (6类情感)
INTENSITY_6 = {
    "happy": {
        "mild":   ["content", "pleased", "satisfied", "comfortable"],
        "medium": ["happy", "glad", "cheerful", "joyful"],
        "strong": ["ecstatic", "elated", "euphoric", "jubilant"],
    },
    "sad": {
        "mild":   ["down", "unhappy", "disappointed", "blue"],
        "medium": ["sad", "sorrowful", "melancholy", "gloomy"],
        "strong": ["despairing", "devastated", "anguished", "heartbroken"],
    },
    "angry": {
        "mild":   ["annoyed", "irritated", "bothered", "irked"],
        "medium": ["angry", "mad", "furious", "enraged"],
        "strong": ["livid", "infuriated", "incensed", "rage-filled"],
    },
    "scared": {
        "mild":   ["uneasy", "wary", "cautious", "concerned"],
        "medium": ["scared", "afraid", "frightened", "alarmed"],
        "strong": ["terrified", "horrified", "panicked", "petrified"],
    },
    "surprise": {
        "mild":   ["curious", "intrigued", "interested", "attentive"],
        "medium": ["surprised", "amazed", "startled", "taken-aback"],
        "strong": ["astonished", "dumbfounded", "flabbergasted", "thunderstruck"],
    },
    "disgust": {
        "mild":   ["displeased", "uncomfortable", "uneasy", "off-put"],
        "medium": ["disgusted", "repulsed", "nauseated", "sickened"],
        "strong": ["revolted", "appalled", "horrified", "loathing"],
    },
}

# 职业强度梯度 (资深程度)
OCCUPATION_INTENSITY = {
    "doctor": {
        "mild":   ["intern", "student", "trainee", "novice"],
        "medium": ["doctor", "physician", "practitioner", "clinician"],
        "strong": ["expert", "specialist", "authority", "master"],
    },
    "teacher": {
        "mild":   ["assistant", "tutor", "aide", "substitute"],
        "medium": ["teacher", "instructor", "educator", "professor"],
        "strong": ["principal", "dean", "director", "chancellor"],
    },
    "artist": {
        "mild":   ["amateur", "hobbyist", "dabbler", "beginner"],
        "medium": ["artist", "creator", "maker", "craftsman"],
        "strong": ["virtuoso", "maestro", "master", "genius"],
    },
    "engineer": {
        "mild":   ["junior", "apprentice", "assistant", "trainee"],
        "medium": ["engineer", "developer", "designer", "analyst"],
        "strong": ["architect", "lead", "chief", "director"],
    },
    "chef": {
        "mild":   ["prep-cook", "assistant", "commis", "dishwasher"],
        "medium": ["chef", "cook", "cuisinier", "grill-master"],
        "strong": ["head-chef", "executive", "master-chef", "michelin"],
    },
    "farmer": {
        "mild":   ["hand", "helper", "laborer", "ranch-hand"],
        "medium": ["farmer", "grower", "rancher", "cultivator"],
        "strong": ["landowner", "proprietor", "magnate", "baron"],
    },
}

# 栖息地极端程度
HABITAT_INTENSITY = {
    "ocean": {
        "mild":   ["shore", "coast", "beach", "shallows"],
        "medium": ["sea", "ocean", "marine", "pelagic"],
        "strong": ["abyss", "deep-sea", "trench", "depths"],
    },
    "forest": {
        "mild":   ["grove", "copse", "woodland", "clearing"],
        "medium": ["forest", "jungle", "timber", "wood"],
        "strong": ["rainforest", "old-growth", "primordial", "ancient-forest"],
    },
    "desert": {
        "mild":   ["dryland", "scrubland", "badlands", "semi-arid"],
        "medium": ["desert", "wasteland", "sandy", "barren"],
        "strong": ["death-valley", "sahara", "dune-sea", "wasteland-dead"],
    },
}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def collect_residuals(model, tokenizer, layers, li, words, prompt_template, device="cuda"):
    """收集一组词在某层的残差"""
    resids = []
    for word in words:
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
    
    return resids


def compute_regular_simplex(N):
    """生成N维正则单纯形的顶点 (在N-1维空间中)"""
    vertices = np.zeros((N, N - 1))
    for i in range(N - 1):
        vertices[i, i] = 1.0
    last = np.full(N - 1, (1.0 - np.sqrt(N)) / (N - 1))
    vertices[N - 1] = last
    
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    edge_len = np.linalg.norm(vertices[0] - vertices[1])
    vertices = vertices / edge_len
    
    return vertices


def compute_simplex_fit(centers, class_order):
    """计算单纯形拟合度"""
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    
    # PCA投影
    centered = center_mat - global_center
    U, S, Vt = svd(centered, full_matrices=False)
    proj = centered @ Vt[:N-1].T  # (N, N-1)
    
    # 正则单纯形
    reg_simplex = compute_regular_simplex(N)
    
    # Procrustes对齐
    R, scale = orthogonal_procrustes(proj, reg_simplex)
    aligned = proj @ R
    
    # fit_r2
    mean_simplex = np.mean(reg_simplex, axis=0)
    ss_tot = np.sum((reg_simplex - mean_simplex) ** 2)
    ss_res = np.sum((aligned - reg_simplex) ** 2)
    fit_r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    
    # 等距性
    dists = pdist(aligned)
    isoperimetric = np.std(dists) / (np.mean(dists) + 1e-10)
    
    return {
        "fit_r2": float(fit_r2),
        "isoperimetric": float(isoperimetric),
        "proj_matrix": Vt[:N-1],
        "rotation": R,
        "global_center": global_center,
        "aligned_centers": aligned,
        "reg_simplex": reg_simplex,
        "singular_values": S[:N-1].tolist(),
    }


def compute_edge_alignment(simplex_info, intensity_resids, class_order, class_centers):
    """计算强度轨迹与单纯形边的对齐度"""
    N = len(class_order)
    
    # 单纯形边方向
    reg_simplex = simplex_info["reg_simplex"]
    R = simplex_info["rotation"]
    proj_matrix = simplex_info["proj_matrix"]
    global_center = simplex_info["global_center"]
    
    simplex_edges = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_dir = reg_simplex[j] - reg_simplex[i]
            edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-10)
            simplex_edges[(i, j)] = edge_dir
    
    # 径向方向
    radial_dirs = {}
    for c in class_order:
        rd = class_centers[c] - global_center
        rn = np.linalg.norm(rd)
        radial_dirs[c] = rd / (rn + 1e-10) if rn > 1e-10 else rd
    
    results = {}
    
    for cls_name, cls_intensities in intensity_resids.items():
        if cls_name not in class_centers:
            continue
        
        cls_idx = class_order.index(cls_name)
        levels = sorted(cls_intensities.keys())
        
        if len(levels) < 2:
            continue
        
        # 每个强度的中心
        int_centers = {}
        for level in levels:
            vecs = cls_intensities[level]
            if len(vecs) > 0:
                int_centers[level] = np.mean(vecs, axis=0)
        
        if len(int_centers) < 2:
            continue
        
        # 轨迹方向
        trajectory = int_centers[levels[-1]] - int_centers[levels[0]]
        traj_norm = np.linalg.norm(trajectory)
        
        if traj_norm < 1e-10:
            results[cls_name] = {"error": "trajectory too small"}
            continue
        
        trajectory_dir = trajectory / traj_norm
        
        # 径向对齐
        radial_alignment = abs(np.dot(trajectory_dir, radial_dirs[cls_name]))
        
        # 投影轨迹到子空间
        proj_trajectory = (trajectory) @ proj_matrix.T @ R
        pt_norm = np.linalg.norm(proj_trajectory)
        proj_traj_dir = proj_trajectory / (pt_norm + 1e-10)
        
        # 与每条边的对齐度
        edge_alignments = {}
        for (i, j), edge_dir in simplex_edges.items():
            alignment = abs(np.dot(proj_traj_dir, edge_dir))
            edge_alignments[(class_order[i], class_order[j])] = float(alignment)
        
        # 最佳边
        best_edge = max(edge_alignments, key=edge_alignments.get)
        best_alignment = edge_alignments[best_edge]
        
        # 第二佳边
        sorted_edges = sorted(edge_alignments.items(), key=lambda x: -x[1])
        second_alignment = sorted_edges[1][1] if len(sorted_edges) > 1 else 0
        
        # 判定
        if best_alignment > 0.7:
            verdict = "EDGE-ALIGNED"
        elif best_alignment > 0.5:
            verdict = "WEAK-EDGE"
        elif radial_alignment > 0.5:
            verdict = "RADIAL"
        else:
            verdict = "RANDOM-TANGENTIAL"
        
        results[cls_name] = {
            "radial_alignment": float(radial_alignment),
            "best_edge": best_edge,
            "best_alignment": float(best_alignment),
            "second_alignment": float(second_alignment),
            "specificity": float(best_alignment - second_alignment),
            "all_edges": {f"{k[0]}-{k[1]}": v for k, v in edge_alignments.items()},
            "verdict": verdict,
        }
    
    return results


def compute_chance_probability(N, n_trajectories, threshold=0.7):
    """
    计算N类单纯形中n个轨迹随机对齐某条边的概率
    P(随机方向与某特定边对齐>threshold) = arcsin(threshold)/π ≈ threshold for small threshold
    P(与任意一条边对齐>threshold) ≈ C(N,2) * arcsin(threshold)/π
    P(至少一个轨迹对齐>threshold) = 1 - (1-p_single)^(n_trajectories)
    P(全部对齐>threshold) = p_single^n_trajectories
    """
    import math
    n_edges = N * (N - 1) // 2
    p_single_edge = math.asin(threshold) / math.pi  # 单条边
    p_any_edge = min(1.0, n_edges * p_single_edge)  # 任意一条边
    p_at_least_one = 1.0 - (1.0 - p_any_edge) ** n_trajectories
    p_all = p_any_edge ** n_trajectories
    
    return {
        "N": N,
        "n_edges": n_edges,
        "p_single_edge": float(p_single_edge),
        "p_any_edge": float(p_any_edge),
        "p_at_least_one": float(p_at_least_one),
        "p_all": float(p_all),
    }


def analyze_layer_evolution(model, tokenizer, layers, domain_data, intensity_data, device):
    """分析单纯形结构和边对齐随层演化"""
    class_order = domain_data["order"]
    prompt = domain_data["prompt"]
    N = len(class_order)
    n_layers = len(layers)
    
    layer_results = []
    
    for li in range(n_layers):
        # 收集类中心
        centers = {}
        for cls in class_order:
            words = domain_data["classes"][cls]
            resids = collect_residuals(model, tokenizer, layers, li, words, prompt, device)
            if len(resids) >= 6:
                centers[cls] = np.mean(resids, axis=0)
        
        if len(centers) < N:
            continue
        
        # 单纯形拟合
        simplex_info = compute_simplex_fit(centers, class_order)
        
        # 边对齐 (如果有强度数据)
        edge_results = {}
        if intensity_data:
            int_resids = {}
            for cls in class_order:
                if cls not in intensity_data:
                    continue
                cls_int = {}
                for level, words in intensity_data[cls].items():
                    resids = collect_residuals(model, tokenizer, layers, li, words, prompt, device)
                    if len(resids) >= 2:
                        cls_int[level] = resids
                if cls_int:
                    int_resids[cls] = cls_int
            
            if int_resids:
                edge_results = compute_edge_alignment(simplex_info, int_resids, class_order, centers)
        
        # 汇总
        n_edge_aligned = sum(1 for v in edge_results.values() if v.get("verdict") == "EDGE-ALIGNED")
        avg_best_align = np.mean([v.get("best_alignment", 0) for v in edge_results.values()]) if edge_results else 0
        
        layer_results.append({
            "layer": li,
            "fit_r2": simplex_info["fit_r2"],
            "isoperimetric": simplex_info["isoperimetric"],
            "n_edge_aligned": n_edge_aligned,
            "n_intensity_classes": len(edge_results),
            "avg_best_alignment": float(avg_best_align),
            "edge_details": edge_results,
        })
        
        if li % 5 == 0 or li == n_layers - 1:
            log(f"  Layer {li:2d}: fit_r2={simplex_info['fit_r2']:.3f} iso={simplex_info['isoperimetric']:.3f} "
                f"edge_aligned={n_edge_aligned}/{len(edge_results)} avg_align={avg_best_align:.3f}")
    
    return layer_results


def run_domain(model_name, model, tokenizer, layers, device, domain_name, domain_data, intensity_data=None):
    """运行单个领域的完整分析"""
    class_order = domain_data["order"]
    prompt = domain_data["prompt"]
    N = len(class_order)
    
    log(f"\n{'='*70}")
    log(f"领域: {domain_name} (N={N}, {N*(N-1)//2}条边)")
    log(f"{'='*70}")
    
    # 1. 巧合概率计算
    chance = compute_chance_probability(N, min(4, N), threshold=0.7)
    log(f"\n巧合概率分析 (N={N}):")
    log(f"  边数={chance['n_edges']}, P(随机与某边对齐>0.7)={chance['p_any_edge']:.4f}")
    log(f"  P(至少1个轨迹对齐>0.7)={chance['p_at_least_one']:.4f}")
    log(f"  P(全部4个对齐>0.7)={chance['p_all']:.6f}")
    
    # 2. 找最佳层
    n_layers = len(layers)
    # 候选层: 中间到后部
    if n_layers <= 20:
        test_layers = list(range(n_layers))
    else:
        test_layers = list(range(n_layers // 3, n_layers))
    
    # 快速扫描: 只收集类中心, 检查fit_r2
    best_li = n_layers * 2 // 3
    best_r2 = 0
    
    log(f"\n快速扫描层 (每3层)...")
    for li in range(0, n_layers, 3):
        centers = {}
        for cls in class_order:
            words = domain_data["classes"][cls][:6]  # 只用6个词快速扫描
            resids = collect_residuals(model, tokenizer, layers, li, words, prompt, device)
            if len(resids) >= 4:
                centers[cls] = np.mean(resids, axis=0)
        
        if len(centers) < N:
            continue
        
        simplex_info = compute_simplex_fit(centers, class_order)
        r2 = simplex_info["fit_r2"]
        
        if r2 > best_r2:
            best_r2 = r2
            best_li = li
    
    log(f"最佳层: L{best_li} (fit_r2={best_r2:.3f})")
    
    # 3. 在最佳层完整分析
    log(f"\n在L{best_li}完整分析...")
    centers = {}
    for cls in class_order:
        words = domain_data["classes"][cls]
        resids = collect_residuals(model, tokenizer, layers, best_li, words, prompt, device)
        if len(resids) >= 6:
            centers[cls] = np.mean(resids, axis=0)
        log(f"  {cls}: {len(resids)} vectors collected")
    
    if len(centers) < N:
        log(f"  ERROR: 只收集到{len(centers)}/{N}个类别中心")
        return None
    
    simplex_info = compute_simplex_fit(centers, class_order)
    log(f"\n单纯形拟合 (N={N}):")
    log(f"  fit_r2 = {simplex_info['fit_r2']:.4f}")
    log(f"  isoperimetric = {simplex_info['isoperimetric']:.4f}")
    log(f"  奇异值: {[f'{s:.1f}' for s in simplex_info['singular_values']]}")
    
    # 4. 边对齐分析
    edge_results = {}
    if intensity_data:
        log(f"\n强度轨迹 × 边对齐分析:")
        int_resids = {}
        for cls in class_order:
            if cls not in intensity_data:
                continue
            cls_int = {}
            for level, words in intensity_data[cls].items():
                resids = collect_residuals(model, tokenizer, layers, best_li, words, prompt, device)
                if len(resids) >= 2:
                    cls_int[level] = resids
            if cls_int:
                int_resids[cls] = cls_int
                log(f"  {cls}: levels={list(cls_int.keys())}")
        
        if int_resids:
            edge_results = compute_edge_alignment(simplex_info, int_resids, class_order, centers)
            
            log(f"\n★★★ 边对齐结果 (N={N}, {N*(N-1)//2}条边):")
            n_aligned = 0
            for cls, ar in edge_results.items():
                if "error" in ar:
                    log(f"  {cls}: ERROR - {ar['error']}")
                    continue
                
                verdict = ar["verdict"]
                best_edge = ar["best_edge"]
                best_al = ar["best_alignment"]
                radial = ar["radial_alignment"]
                spec = ar["specificity"]
                
                marker = "★" if verdict == "EDGE-ALIGNED" else " "
                log(f"  {marker} {cls:10s}: best_edge={best_edge[0]}-{best_edge[1]} "
                    f"align={best_al:.3f} radial={radial:.3f} spec={spec:.3f} → {verdict}")
                
                if verdict == "EDGE-ALIGNED":
                    n_aligned += 1
            
            log(f"\n  总计: {n_aligned}/{len(edge_results)} EDGE-ALIGNED")
            avg_align = np.mean([v.get("best_alignment", 0) for v in edge_results.values()])
            log(f"  平均最佳边对齐度: {avg_align:.3f}")
            
            # 关键判定
            p_chance = chance["p_all"]
            n_observed = n_aligned
            n_tested = len(edge_results)
            
            log(f"\n★★★★★ 关键判定:")
            log(f"  N={N}, 边数={N*(N-1)//2}")
            log(f"  巧合概率P(全部对齐>0.7) = {p_chance:.6f}")
            log(f"  实际: {n_observed}/{n_tested} 对齐")
            
            if n_observed == n_tested and p_chance < 0.01:
                log(f"  ★★★★★ 极强证据: 全部对齐且巧合概率<1%!")
            elif n_observed >= n_tested * 0.7 and p_chance < 0.05:
                log(f"  ★★★★ 强证据: 多数对齐且巧合概率<5%")
            elif n_observed >= n_tested * 0.5:
                log(f"  ★★★ 中等证据: 过半对齐")
            else:
                log(f"  ★★ 弱证据: 少数对齐, 可能是巧合")
            
            # 与N=4对比
            chance_4 = compute_chance_probability(4, 4, threshold=0.7)
            log(f"\n  对比N=4: P_all={chance_4['p_all']:.6f} vs N={N}: P_all={p_chance:.6f}")
            if p_chance < chance_4['p_all']:
                log(f"  ★★★★ 大N巧合概率更低! 边对齐更可信!")
            else:
                log(f"  注意: 大N的巧合概率并未显著降低")
    
    # 5. 跨层演化 (只扫描最佳层附近)
    log(f"\n跨层演化扫描 (L{max(0,best_li-8)}-L{min(n_layers-1,best_li+4)})...")
    evolution = []
    for li in range(max(0, best_li - 8), min(n_layers, best_li + 5)):
        centers = {}
        for cls in class_order:
            words = domain_data["classes"][cls][:6]
            resids = collect_residuals(model, tokenizer, layers, li, words, prompt, device)
            if len(resids) >= 4:
                centers[cls] = np.mean(resids, axis=0)
        
        if len(centers) < N:
            continue
        
        si = compute_simplex_fit(centers, class_order)
        
        # 边对齐
        e_results = {}
        if intensity_data:
            int_resids = {}
            for cls in class_order:
                if cls not in intensity_data:
                    continue
                cls_int = {}
                for level, words in intensity_data[cls].items():
                    resids = collect_residuals(model, tokenizer, layers, li, words, prompt, device)
                    if len(resids) >= 2:
                        cls_int[level] = resids
                if cls_int:
                    int_resids[cls] = cls_int
            
            if int_resids:
                e_results = compute_edge_alignment(si, int_resids, class_order, centers)
        
        n_al = sum(1 for v in e_results.values() if v.get("verdict") == "EDGE-ALIGNED")
        avg_al = np.mean([v.get("best_alignment", 0) for v in e_results.values()]) if e_results else 0
        
        evolution.append({
            "layer": li,
            "fit_r2": si["fit_r2"],
            "n_edge_aligned": n_al,
            "avg_best_alignment": float(avg_al),
        })
        
        log(f"  L{li:2d}: fit_r2={si['fit_r2']:.3f} edge_aligned={n_al}/{len(e_results)} avg_align={avg_al:.3f}")
    
    return {
        "domain": domain_name,
        "N": N,
        "n_edges": N * (N - 1) // 2,
        "best_layer": best_li,
        "fit_r2": simplex_info["fit_r2"],
        "chance_probability": chance,
        "edge_alignment": edge_results,
        "evolution": evolution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")
    
    log(f"CCXLV(345): 大N边对齐验证 — {args.model}")
    log(f"{'='*70}")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    log(f"\n加载模型 {args.model}...")
    model, tokenizer, device = load_model(args.model)
    layers = get_layers(model)
    info = get_model_info(model, args.model)
    log(f"  d_model={info.d_model}, n_layers={info.n_layers}")
    
    all_results = {}
    
    # === 领域1: N=6 情感 (Ekman基本情感) ===
    log(f"\n{'#'*70}")
    log(f"# 领域1: 情感 N=6 (Ekman基本情感)")
    log(f"{'#'*70}")
    r1 = run_domain(args.model, model, tokenizer, layers, device,
                    "emotion_6", EMOTION_6, INTENSITY_6)
    if r1:
        all_results["emotion_6"] = r1
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 领域2: N=6 职业 ===
    log(f"\n{'#'*70}")
    log(f"# 领域2: 职业 N=6")
    log(f"{'#'*70}")
    r2 = run_domain(args.model, model, tokenizer, layers, device,
                    "occupation_6", OCCUPATION_6, OCCUPATION_INTENSITY)
    if r2:
        all_results["occupation_6"] = r2
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 领域3: N=8 栖息地 ===
    log(f"\n{'#'*70}")
    log(f"# 领域3: 栖息地 N=8")
    log(f"{'#'*70}")
    r3 = run_domain(args.model, model, tokenizer, layers, device,
                    "habitat_8", HABITAT_8, HABITAT_INTENSITY)
    if r3:
        all_results["habitat_8"] = r3
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 汇总 ===
    log(f"\n{'='*70}")
    log(f"★★★★★ 跨领域汇总 ({args.model})")
    log(f"{'='*70}")
    
    for domain_name, result in all_results.items():
        N = result["N"]
        n_edges = result["n_edges"]
        fit_r2 = result["fit_r2"]
        edge_al = result.get("edge_alignment", {})
        n_aligned = sum(1 for v in edge_al.values() if v.get("verdict") == "EDGE-ALIGNED")
        n_tested = len(edge_al)
        avg_align = np.mean([v.get("best_alignment", 0) for v in edge_al.values()]) if edge_al else 0
        chance_p = result.get("chance_probability", {}).get("p_all", 0)
        
        log(f"\n{domain_name} (N={N}, {n_edges}边):")
        log(f"  fit_r2={fit_r2:.3f}")
        log(f"  边对齐: {n_aligned}/{n_tested} (平均={avg_align:.3f})")
        log(f"  巧合概率: P_all={chance_p:.6f}")
        
        if n_aligned == n_tested and chance_p < 0.01:
            log(f"  ★★★★★ 确认: 非巧合!")
        elif n_aligned >= n_tested * 0.5:
            log(f"  ★★★ 有信号但需更多验证")
        else:
            log(f"  ★★ 弱信号或无信号")
    
    # === 关键结论 ===
    log(f"\n{'='*70}")
    log(f"★★★★★ 核心结论")
    log(f"{'='*70}")
    
    # 收集所有N的边对齐率
    total_tested = 0
    total_aligned = 0
    for domain_name, result in all_results.items():
        edge_al = result.get("edge_alignment", {})
        total_tested += len(edge_al)
        total_aligned += sum(1 for v in edge_al.values() if v.get("verdict") == "EDGE-ALIGNED")
    
    overall_rate = total_aligned / total_tested if total_tested > 0 else 0
    
    log(f"\n总体边对齐率: {total_aligned}/{total_tested} = {overall_rate:.1%}")
    
    # 对比N=4 (CCXLIV结果)
    log(f"\n与N=4对比:")
    log(f"  N=4 (CCXLIV): 48/48 = 100% EDGE-ALIGNED")
    log(f"  N=6-8 (CCXLV): {total_aligned}/{total_tested} = {overall_rate:.1%} EDGE-ALIGNED")
    
    if overall_rate > 0.7:
        log(f"\n★★★★★ 结论: 大N下边对齐仍然成立!")
        log(f"  强度轨迹沿单纯形边移动是语言几何的基本性质")
        log(f"  排除了N=4巧合的解释")
    elif overall_rate > 0.4:
        log(f"\n★★★★ 结论: 大N下边对齐部分成立")
        log(f"  可能在某些领域成立, 某些不成立")
        log(f"  需要更细致的领域分析")
    else:
        log(f"\n★★★ 结论: 大N下边对齐减弱")
        log(f"  N=4的边对齐可能是巧合或N=4特殊性质")
        log(f"  需要重新审视语言几何模型")
    
    # 保存JSON
    json_path = TEMP / f"ccxlv_large_n_{args.model}.json"
    # 转换不可序列化的对象
    for domain_name in all_results:
        if "evolution" in all_results[domain_name]:
            pass  # evolution already serializable
        if "edge_alignment" in all_results[domain_name]:
            for cls, ar in all_results[domain_name]["edge_alignment"].items():
                if "all_edges" in ar:
                    ar["all_edges"] = {str(k): v for k, v in ar["all_edges"].items()}
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n结果已保存: {json_path}")
    
    # 释放模型
    release_model(model)
    log(f"\nDone! {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
