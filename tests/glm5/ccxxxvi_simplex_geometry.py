"""
CCXXXVI(336): N-1维语义单纯形精确几何 + 更大N验证 + 传播-读取精确定位
======================================================================
核心突破1: ★★★★★ N-1维单纯形的精确几何 — 是正则单纯形还是不规则?
  → 在5维空间中计算6类别顶点的两两距离矩阵
  → 正则单纯形: 所有顶点等距, 距离=√(2*(1+1/N))
  → 不规则: 某些类别对更远(如ocean vs virtual), 某些更近(如land vs sky)
  → 关键指标: regularity_score = 1 - std(pairwise_dists)/mean(pairwise_dists)

核心突破2: ★★★★ 更大N的单纯形验证 — N=7,8,10
  → 增加"underground"(地下), "arctic"(极地), "tropical"(热带), "freshwater"(淡水)
  → 如果N=10仍有9个separating PCs → N-1维单纯形是普遍规律
  → 如果N=10时<9 → 存在维度上限

核心突破3: ★★★★ 传播-读取的精确层定位
  → 在每层计算4个指标:
    1. propagation_strength: comp_pos diff与last_pos diff的相关性
    2. comp_patch_effect: patching comp_pos的change_ratio
    3. readout_readiness: last_pos diff在W_U行空间的比例
    4. output_determination: last_pos patching的change_ratio
  → 寻找: 传播完成层(propagation~1, comp_patch~0) vs 读取开始层(readout急剧增加)

实验设计:
  Part 1: N-1维单纯形精确几何 (★核心)
    在最佳语义分离层, 将6类别投影到5维PCA空间
    计算: 两两距离矩阵, 正则性分数, 顶点-质心距离, 顶点角度
    
  Part 2: N=7,8,10的单纯形验证
    增加underground/arctic/tropical/freshwater
    如果N=10→9维 → 确认普遍规律
    
  Part 3: 传播-读取4指标逐层追踪
    对3个比较对, 在每层计算4个指标
    精确定位"传播完成层"和"读取开始层"

用法:
  python ccxxxvi_simplex_geometry.py --model qwen3
  python ccxxxvi_simplex_geometry.py --model glm4
  python ccxxxvi_simplex_geometry.py --model deepseek7b
"""
import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxxvi_simplex_geometry_log.txt"

# ★★★ 扩展到10类habitat — 最大化语义距离
ALL_HABITATS = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
             "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
              "crab", "seal", "squid", "lobster", "jellyfish", "starfish",
              "seahorse", "eel", "manta"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
            "falcon", "pigeon", "robin", "condor", "albatross", "vulture",
            "hummingbird", "stork"],
    "space": ["astronaut", "satellite", "comet", "meteor", "nebula",
              "quasar", "asteroid", "rocket", "spaceship", "cosmos",
              "pulsar", "galaxy", "supernova", "star", "planet"],
    "microscopic": ["bacterium", "virus", "cell", "amoeba", "paramecium",
                    "euglena", "diatom", "plasmodium", "ribosome", "mitochondria",
                    "flagellum", "cilium", "nucleus", "chromosome", "spore"],
    "virtual": ["algorithm", "program", "software", "database", "network",
                "protocol", "encryption", "firewall", "browser", "server",
                "compiler", "function", "variable", "module", "interface"],
    "underground": ["mole", "worm", "ant", "termite", "badger",
                    "ferret", "rabbit", "prairie", "meerkat", "nematode",
                    "earthworm", "grub", "larva", "beetle", "centipede"],
    "arctic": ["polar", "penguin", "walrus", "narwhal", "arctic",
               "caribou", "musk", "snowy", "beluga", "puffin",
               "seal", "reindeer", "ptarmigan", "lemming", "husky"],
    "tropical": ["parrot", "toucan", "gorilla", "orangutan", "jaguar",
                 "anaconda", "piranha", "sloth", "iguana", "macaw",
                 "chimpanzee", "bamboo", "orchid", "mango", "papaya"],
    "freshwater": ["trout", "bass", "salmon", "catfish", "pike",
                   "perch", "carp", "minnow", "crayfish", "tadpole",
                   "frog", "newt", "turtle", "duck", "heron"],
}

# 比较词对
SIZE_PAIRS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def project_to_subspace(vec, U_basis):
    """将向量投影到U_basis的列空间"""
    coeffs = U_basis.T @ vec
    proj = U_basis @ coeffs
    return proj, coeffs


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXVI(336): N-1维单纯形精确几何 + 更大N + 传播-读取 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    W_U = get_W_U(model)
    embed_layer = model.get_input_embeddings()
    
    # W_U行空间基
    log("Computing W_U row space basis (500 components)...")
    W_U_T = W_U.T.astype(np.float32)
    k = min(500, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    U_wu, s_wu, _ = svds(W_U_T, k=k)
    sort_idx = np.argsort(s_wu)[::-1]
    U_wu = U_wu[:, sort_idx].astype(np.float64)
    
    # Token IDs
    bigger_id = tokenizer.encode("bigger", add_special_tokens=False)[0]
    smaller_id = tokenizer.encode("smaller", add_special_tokens=False)[0]
    
    # ====================================================================
    # Part 1: N-1维单纯形精确几何 (★核心)
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: N-1维单纯形精确几何分析")
    log("="*60)
    
    # 确定最佳语义分离层 (基于CCXXXV结果)
    best_layer_map = {"qwen3": 27, "glm4": 20, "deepseek7b": 14}
    best_layer = best_layer_map.get(model_name, n_layers // 2)
    if best_layer >= n_layers:
        best_layer = n_layers // 2
    
    # 也测试相邻层
    test_layers_geo = sorted(set([
        best_layer - 4, best_layer - 2, best_layer, best_layer + 2, best_layer + 4,
        n_layers // 3, n_layers // 2
    ]))
    test_layers_geo = [l for l in test_layers_geo if 0 <= l < n_layers]
    
    log(f"  测试层: {test_layers_geo}")
    
    geometry_results = {}
    
    # 首先确定6类在哪些层有5个separating PCs
    hab_list_6 = ["land", "ocean", "sky", "space", "microscopic", "virtual"]
    
    for li in test_layers_geo:
        # 收集6类残差
        hab_resids = {}
        for hab in hab_list_6:
            words = ALL_HABITATS.get(hab, [])[:12]
            resids = []
            for word in words:
                prompt = f"The {word} lives in the"
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
                hab_resids[hab] = resids
        
        if len(hab_resids) < 4:
            log(f"  L{li}: insufficient habitats ({len(hab_resids)})")
            continue
        
        # PCA
        all_vecs = []
        all_labels = []
        for hab in hab_list_6:
            if hab in hab_resids:
                all_vecs.extend(hab_resids[hab])
                all_labels.extend([hab] * len(hab_resids[hab]))
        
        arr = np.array(all_vecs)
        centered = arr - arr.mean(axis=0)
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        n_habs = len(hab_resids)
        n_pc = min(n_habs + 2, Vt.shape[0])
        
        # 先计算n_separating_PCs
        hab_proj = {}
        for hab in hab_resids:
            hab_arr = np.array(hab_resids[hab])
            hab_centered = hab_arr - arr.mean(axis=0)
            pc_proj = hab_centered @ Vt[:n_pc].T
            hab_proj[hab] = pc_proj
        
        n_separating = 0
        for pc_i in range(n_pc):
            means = [np.mean(hab_proj[h][:, pc_i]) for h in hab_resids]
            within_vars = [np.var(hab_proj[h][:, pc_i]) for h in hab_resids]
            f_ratio = np.var(means) / max(np.mean(within_vars), 1e-10)
            if f_ratio > 1.0:
                n_separating += 1
        
        # ★★★ 核心: 在N-1维空间中计算单纯形几何
        # 投影到前n_separating个PC
        if n_separating < 2:
            log(f"  L{li}: only {n_separating} separating PCs, skip geometry")
            continue
        
        geom_dim = n_separating
        hab_centers = {}
        for hab in hab_resids:
            hab_arr = np.array(hab_resids[hab])
            hab_centered = hab_arr - arr.mean(axis=0)
            pc_proj = hab_centered @ Vt[:geom_dim].T
            hab_centers[hab] = np.mean(pc_proj, axis=0)  # [geom_dim]
        
        # 两两距离矩阵
        hab_names = list(hab_centers.keys())
        centers_arr = np.array([hab_centers[h] for h in hab_names])
        
        pairwise_dists = squareform(pdist(centers_arr))
        upper_tri = pairwise_dists[np.triu_indices(len(hab_names), k=1)]
        
        # 正则性分数
        mean_dist = np.mean(upper_tri)
        std_dist = np.std(upper_tri)
        regularity_score = 1.0 - std_dist / max(mean_dist, 1e-10)
        
        # 理论正则单纯形距离
        # 正则单纯形的顶点距离 = sqrt(2 * (1 + 1/N)) * radius
        # 但我们的半径由数据决定, 所以用相对值
        
        # 顶点到质心的距离
        centroid = np.mean(centers_arr, axis=0)
        vertex_radii = [np.linalg.norm(centers_arr[i] - centroid) for i in range(len(hab_names))]
        mean_radius = np.mean(vertex_radii)
        std_radius = np.std(vertex_radii)
        radius_uniformity = 1.0 - std_radius / max(mean_radius, 1e-10)
        
        # 顶点之间的角度 (从质心出发)
        angles = []
        for i in range(len(hab_names)):
            for j in range(i + 1, len(hab_names)):
                v1 = centers_arr[i] - centroid
                v2 = centers_arr[j] - centroid
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.dot(v1, v2) / (n1 * n2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)
        
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        # 理论正则单纯形角度
        # 正则N-单纯形: 任意两顶点从质心看的角度 = arccos(-1/N)
        ideal_angle = np.arccos(-1.0 / n_habs) * 180 / np.pi
        
        angle_deviation = abs(mean_angle - ideal_angle)
        angle_uniformity = 1.0 - std_angle / max(mean_angle, 1e-10)
        
        log(f"\n  L{li} Geometry ({n_habs} classes, {geom_dim}D):")
        log(f"    n_separating_PCs = {n_separating} (expected N-1 = {n_habs - 1})")
        log(f"    Pairwise distances: mean={mean_dist:.4f}, std={std_dist:.4f}")
        log(f"    Regularity score = {regularity_score:.4f}")
        log(f"    Vertex radii: mean={mean_radius:.4f}, std={std_radius:.4f}")
        log(f"    Radius uniformity = {radius_uniformity:.4f}")
        log(f"    Angles: mean={mean_angle:.2f}°, std={std_angle:.2f}°")
        log(f"    Ideal regular simplex angle = {ideal_angle:.2f}°")
        log(f"    Angle deviation = {angle_deviation:.2f}°")
        log(f"    Angle uniformity = {angle_uniformity:.4f}")
        
        # 找最近/最远的类别对
        dist_pairs = []
        for i in range(len(hab_names)):
            for j in range(i + 1, len(hab_names)):
                dist_pairs.append((hab_names[i], hab_names[j], pairwise_dists[i, j]))
        dist_pairs.sort(key=lambda x: x[2])
        
        log(f"    Closest pairs: {dist_pairs[:3]}")
        log(f"    Farthest pairs: {dist_pairs[-3:]}")
        
        geometry_results[f"L{li}"] = {
            "n_classes": n_habs,
            "n_separating_PCs": n_separating,
            "expected_dim_N_minus_1": n_habs - 1,
            "regularity_score": round(float(regularity_score), 4),
            "mean_pairwise_dist": round(float(mean_dist), 4),
            "std_pairwise_dist": round(float(std_dist), 4),
            "radius_uniformity": round(float(radius_uniformity), 4),
            "mean_radius": round(float(mean_radius), 4),
            "mean_angle": round(float(mean_angle), 2),
            "std_angle": round(float(std_angle), 2),
            "ideal_angle": round(float(ideal_angle), 2),
            "angle_deviation": round(float(angle_deviation), 2),
            "angle_uniformity": round(float(angle_uniformity), 4),
            "closest_pairs": [(p[0], p[1], round(float(p[2]), 4)) for p in dist_pairs[:3]],
            "farthest_pairs": [(p[0], p[1], round(float(p[2]), 4)) for p in dist_pairs[-3:]],
            "habitat_centers": {h: [round(float(c), 4) for c in hab_centers[h]] for h in hab_names},
            "pairwise_dist_matrix": {h1: {h2: round(float(pairwise_dists[i, j]), 4) 
                                          for j, h2 in enumerate(hab_names)} 
                                     for i, h1 in enumerate(hab_names)},
        }
    
    results["simplex_geometry"] = geometry_results
    
    # ====================================================================
    # Part 2: N=7,8,10的单纯形验证
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: 更大N的单纯形验证 (N=7,8,10)")
    log("="*60)
    
    hab_configs_large = {
        "7class": ["land", "ocean", "sky", "space", "microscopic", "virtual", "underground"],
        "8class": ["land", "ocean", "sky", "space", "microscopic", "virtual", "underground", "arctic"],
        "10class": list(ALL_HABITATS.keys()),
    }
    
    # 只在最佳层测试
    large_n_layers = sorted(set([best_layer, n_layers // 2, n_layers - n_layers // 4]))
    large_n_layers = [l for l in large_n_layers if 0 <= l < n_layers]
    
    large_n_results = {}
    
    for config_name, hab_list in hab_configs_large.items():
        log(f"\n  --- {config_name}: {hab_list} ---")
        
        for li in large_n_layers:
            hab_data = {}
            valid_habs = []
            
            for hab in hab_list:
                words = ALL_HABITATS.get(hab, [])[:12]
                resids = []
                for word in words:
                    prompt = f"The {word} lives in the"
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
                    hab_data[hab] = resids
                    valid_habs.append(hab)
            
            if len(valid_habs) < 3:
                log(f"    L{li}: insufficient valid habitats ({len(valid_habs)})")
                continue
            
            # PCA
            all_vecs = []
            all_labels = []
            for hab in valid_habs:
                all_vecs.extend(hab_data[hab])
                all_labels.extend([hab] * len(hab_data[hab]))
            
            arr = np.array(all_vecs)
            centered = arr - arr.mean(axis=0)
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            
            n_classes = len(valid_habs)
            n_pc = min(n_classes + 2, Vt.shape[0])
            
            hab_proj = {}
            for hab in valid_habs:
                hab_arr = np.array(hab_data[hab])
                hab_centered = hab_arr - arr.mean(axis=0)
                pc_proj = hab_centered @ Vt[:n_pc].T
                hab_proj[hab] = pc_proj
            
            n_separating = 0
            for pc_i in range(n_pc):
                means = [np.mean(hab_proj[h][:, pc_i]) for h in valid_habs]
                within_vars = [np.var(hab_proj[h][:, pc_i]) for h in valid_habs]
                f_ratio = np.var(means) / max(np.mean(within_vars), 1e-10)
                if f_ratio > 1.0:
                    n_separating += 1
            
            key = f"{config_name}_L{li}"
            large_n_results[key] = {
                "n_classes": n_classes,
                "valid_habitats": valid_habs,
                "expected_dim_N_minus_1": n_classes - 1,
                "n_separating_PCs": n_separating,
                "simplex_match": n_separating == n_classes - 1,
            }
            
            log(f"    L{li}: {config_name} → n_classes={n_classes}, "
                f"expected(N-1)={n_classes-1}, n_separating={n_separating}, "
                f"match={n_separating == n_classes - 1}")
    
    results["large_n_simplex"] = large_n_results
    
    # ====================================================================
    # Part 3: 传播-读取4指标逐层追踪
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: 传播-读取4指标逐层追踪")
    log("="*60)
    
    # 测试3个比较对
    test_pairs = SIZE_PAIRS[:3]
    sample_layers_pr = sorted(set(
        list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    ))
    
    log(f"  测试对: {test_pairs}")
    log(f"  测试层: {sample_layers_pr}")
    
    propagation_results = {}
    
    for big, small in test_pairs:
        prompt_b = f"The {big} is bigger than the {small}"
        prompt_s = f"The {big} is smaller than the {small}"
        
        toks_b = tokenizer(prompt_b, return_tensors="pt").to(device)
        toks_s = tokenizer(prompt_s, return_tensors="pt").to(device)
        
        # 找比较词位置
        tokens_b_list = [tokenizer.decode([t]) for t in toks_b.input_ids[0].tolist()]
        tokens_s_list = [tokenizer.decode([t]) for t in toks_s.input_ids[0].tolist()]
        comp_pos = 0
        for i in range(min(len(tokens_b_list), len(tokens_s_list))):
            if tokens_b_list[i] != tokens_s_list[i]:
                comp_pos = i
                break
        
        log(f"\n  {big}/{small}: comp_pos={comp_pos}")
        
        # Baseline logits
        with torch.no_grad():
            out_b = model(**toks_b)
            logits_b = out_b.logits[0, -1]
            diff_baseline = (logits_b[bigger_id] - logits_b[smaller_id]).item()
            
            out_s = model(**toks_s)
            logits_s = out_s.logits[0, -1]
            diff_baseline_s = (logits_s[bigger_id] - logits_s[smaller_id]).item()
        
        pair_data = {}
        
        for li in sample_layers_pr:
            # 收集bigger和smaller句在该层的输出
            captured_b = {}
            captured_s = {}
            
            def mk_hook(key, storage):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    storage[key] = o.detach().clone()
                return hook
            
            h1 = layers[li].register_forward_hook(mk_hook("r", captured_b))
            with torch.no_grad():
                _ = model(**toks_b)
            h1.remove()
            
            h2 = layers[li].register_forward_hook(mk_hook("r", captured_s))
            with torch.no_grad():
                _ = model(**toks_s)
            h2.remove()
            
            if "r" not in captured_b or "r" not in captured_s:
                continue
            
            res_b = captured_b["r"]  # [1, seq_len, d_model]
            res_s = captured_s["r"]
            
            # 指标1: propagation_strength
            # comp_pos和last_pos的bigger-smaller差异的余弦相似度
            diff_comp = res_b[0, comp_pos, :].float().cpu().numpy() - res_s[0, comp_pos, :].float().cpu().numpy()
            diff_last = res_b[0, -1, :].float().cpu().numpy() - res_s[0, -1, :].float().cpu().numpy()
            
            n_comp = np.linalg.norm(diff_comp)
            n_last = np.linalg.norm(diff_last)
            
            if n_comp > 1e-10 and n_last > 1e-10:
                propagation_cos = float(np.dot(diff_comp, diff_last) / (n_comp * n_last))
            else:
                propagation_cos = 0
            
            # diff强度
            comp_diff_strength = float(n_comp)
            last_diff_strength = float(n_last)
            
            # 指标2: comp_patch_effect (简化: 用change_ratio近似)
            # 这里我们不做完整patching, 只用差异强度变化来估计
            # comp_patch_effect ≈ comp_diff在深层的效果 / 浅层效果
            # 简化: 直接用comp_diff_strength / L0的comp_diff_strength
            
            # 指标3: readout_readiness
            # last_pos diff在W_U行空间的比例
            if n_last > 1e-10:
                proj_last, _ = project_to_subspace(diff_last / n_last, U_wu)
                readout_readiness = float(np.linalg.norm(proj_last)**2)
            else:
                readout_readiness = 0
            
            # 指标4: comp_pos diff在W_U行空间的比例
            if n_comp > 1e-10:
                proj_comp, _ = project_to_subspace(diff_comp / n_comp, U_wu)
                comp_in_row = float(np.linalg.norm(proj_comp)**2)
            else:
                comp_in_row = 0
            
            pair_data[f"L{li}"] = {
                "comp_diff_strength": round(comp_diff_strength, 4),
                "last_diff_strength": round(last_diff_strength, 4),
                "propagation_cos": round(propagation_cos, 4),
                "readout_readiness": round(readout_readiness, 4),
                "comp_in_WU_row": round(comp_in_row, 4),
            }
            
            log(f"    L{li}: comp={comp_diff_strength:.2f}(row={comp_in_row:.4f}), "
                f"last={last_diff_strength:.2f}(row={readout_readiness:.4f}), "
                f"prop_cos={propagation_cos:.4f}")
            
            del captured_b, captured_s
            gc.collect()
        
        propagation_results[f"{big}_{small}"] = {
            "comp_pos": comp_pos,
            "baseline_diff": round(diff_baseline, 4),
            "baseline_smaller_diff": round(diff_baseline_s, 4),
            "layer_data": pair_data,
        }
    
    # 汇总传播-读取指标
    log("\n  === Propagation-Readout Summary ===")
    avg_prop_cos = {}
    avg_readout = {}
    avg_comp_row = {}
    for pair_key, pair_data in propagation_results.items():
        for layer_key, layer_data in pair_data["layer_data"].items():
            li = int(layer_key[1:])
            avg_prop_cos.setdefault(li, []).append(layer_data["propagation_cos"])
            avg_readout.setdefault(li, []).append(layer_data["readout_readiness"])
            avg_comp_row.setdefault(li, []).append(layer_data["comp_in_WU_row"])
    
    prop_readout_summary = []
    for li in sorted(avg_prop_cos.keys()):
        mean_prop = np.mean(avg_prop_cos[li])
        mean_read = np.mean(avg_readout[li])
        mean_comp = np.mean(avg_comp_row[li])
        prop_readout_summary.append({
            "layer": li,
            "mean_propagation_cos": round(float(mean_prop), 4),
            "mean_readout_readiness": round(float(mean_read), 4),
            "mean_comp_in_WU_row": round(float(mean_comp), 4),
        })
        log(f"  L{li}: prop_cos={mean_prop:.4f}, readout={mean_read:.4f}, comp_row={mean_comp:.4f}")
    
    # ★★★ 寻找关键转变点
    # 传播完成层: propagation_cos达到峰值后开始下降
    # 读取开始层: readout_readiness急剧增加
    if len(prop_readout_summary) >= 3:
        # 读取开始层: readout_readiness > 之前层的2倍
        readout_start = None
        for i in range(1, len(prop_readout_summary)):
            prev = prop_readout_summary[i-1]["mean_readout_readiness"]
            curr = prop_readout_summary[i]["mean_readout_readiness"]
            if prev > 0.01 and curr > 2 * prev:
                readout_start = prop_readout_summary[i]["layer"]
                break
        
        # 传播完成层: propagation_cos > 0.5的第一个层
        propagation_complete = None
        for entry in prop_readout_summary:
            if entry["mean_propagation_cos"] > 0.5:
                propagation_complete = entry["layer"]
                break
        
        log(f"\n  ★ 传播完成层(prop_cos>0.5): L{propagation_complete}")
        log(f"  ★ 读取开始层(readout 2x增长): L{readout_start}")
        
        results["propagation_readout"] = {
            "per_pair": propagation_results,
            "summary": prop_readout_summary,
            "propagation_complete_layer": propagation_complete,
            "readout_start_layer": readout_start,
        }
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output_path = TEMP / f"ccxxxvi_simplex_geometry_{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    log(f"\nResults saved to {output_path}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    run(args.model)
