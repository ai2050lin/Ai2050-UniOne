"""
CCVI(356): 大N单领域Procrustes旋转分析 — N=50, 49个PC, 高分辨率旋转分析
=====================================================================
★★★★★ CCV核心发现:
  - 旋转角度>90°是Vehicle β_emb负的根本原因
  - 所有领域都显著>90°(t-test p<0.02)
  - 但N=10(9个PC)限制了旋转矩阵的分析精度

★★★★★ 本实验目标:
  1. 大N(50)animal/vehicle领域: 49个PC, 充分的旋转分析分辨率
  2. 高维旋转矩阵的完整特征值分解
  3. 2D旋转平面角度谱: 哪些子空间被旋转了多少度?
  4. 与Vehicle的对比: N=50时Vehicle是否也有异常?
  5. 旋转矩阵的"频谱密度": 是否有主旋转平面?

用法:
  python ccvi_large_n_rotation.py --model qwen3
  python ccvi_large_n_rotation.py --model glm4
  python ccvi_large_n_rotation.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 大N领域定义
# ============================================================

ANIMAL50 = {
    "dog":          ["dog", "puppy", "hound", "canine", "pooch"],
    "cat":          ["cat", "kitten", "feline", "tomcat", "kitty"],
    "wolf":         ["wolf", "werewolf", "lupine", "coyote", "jackal"],
    "lion":         ["lion", "tiger", "leopard", "cheetah", "panther"],
    "bird":         ["bird", "sparrow", "robin", "finch", "wren"],
    "eagle":        ["eagle", "hawk", "falcon", "vulture", "osprey"],
    "fish":         ["fish", "trout", "salmon", "bass", "perch"],
    "shark":        ["shark", "whale", "dolphin", "porpoise", "orca"],
    "snake":        ["snake", "serpent", "viper", "cobra", "python"],
    "lizard":       ["lizard", "gecko", "iguana", "chameleon", "salamander"],
    "horse":        ["horse", "stallion", "mare", "pony", "colt"],
    "cow":          ["cow", "cattle", "bull", "ox", "heifer"],
    "pig":          ["pig", "swine", "hog", "boar", "sow"],
    "sheep":        ["sheep", "lamb", "ram", "ewe", "fleece"],
    "goat":         ["goat", "billy", "kid", "nanny", "caprine"],
    "chicken":      ["chicken", "hen", "rooster", "cock", "poultry"],
    "duck":         ["duck", "drake", "goose", "swan", "mallard"],
    "rabbit":       ["rabbit", "bunny", "hare", "lagomorph", "cottontail"],
    "mouse":        ["mouse", "rat", "rodent", "vole", "hamster"],
    "bear":         ["bear", "grizzly", "polar", "cub", "ursine"],
    "elephant":     ["elephant", "pachyderm", "tusk", "mammoth", "trunk"],
    "giraffe":      ["giraffe", "ruminant", "hoof", "neck", "savanna"],
    "zebra":        ["zebra", "stripes", "equine", "mustang", "striped"],
    "monkey":       ["monkey", "ape", "primate", "chimp", "baboon"],
    "gorilla":      ["gorilla", "silverback", "simian", "primate", "ape"],
    "penguin":      ["penguin", "emperor", "frost", "ice", "flightless"],
    "owl":          ["owl", "nocturnal", "hoot", "raptor", "bird"],
    "parrot":       ["parrot", "macaw", "cockatoo", "tropical", "feather"],
    "turtle":       ["turtle", "tortoise", "shell", "reptile", "terrapin"],
    "crocodile":    ["crocodile", "alligator", "caiman", "reptile", "scale"],
    "frog":         ["frog", "toad", "amphibian", "tadpole", "ribbit"],
    "butterfly":    ["butterfly", "moth", "insect", "caterpillar", "cocoon"],
    "bee":          ["bee", "honey", "wasp", "hive", "pollinator"],
    "ant":          ["ant", "colony", "insect", "worker", "formic"],
    "spider":       ["spider", "arachnid", "web", "silk", "tarantula"],
    "crab":         ["crab", "lobster", "crustacean", "claw", "shellfish"],
    "octopus":      ["octopus", "squid", "tentacle", "cephalopod", "ink"],
    "jellyfish":    ["jellyfish", "tentacle", "sting", "medusa", "plankton"],
    "deer":         ["deer", "stag", "doe", "fawn", "buck"],
    "fox":          ["fox", "vixen", "canine", "crafty", "pelt"],
    "squirrel":     ["squirrel", "chipmunk", "rodent", "acorn", "bushy"],
    "beaver":       ["beaver", "dam", "rodent", "lodger", "gnaw"],
    "otter":        ["otter", "river", "mustelid", "playful", "swim"],
    "kangaroo":     ["kangaroo", "marsupial", "joey", "outback", "hop"],
    "koala":        ["koala", "marsupial", "eucalyptus", "bear", "pouch"],
    "panda":        ["panda", "bamboo", "bear", "china", "black"],
    "rhino":        ["rhino", "rhinoceros", "horn", "pachyderm", "charge"],
    "hippo":        ["hippo", "hippopotamus", "river", "africa", "water"],
    "camel":        ["camel", "dromedary", "hump", "desert", "arabian"],
    "bat":          ["bat", "nocturnal", "wing", "cave", "echolocation"],
    "seal":         ["seal", "sea", "flipper", "pinniped", "arctic"],
}

VEHICLE50 = {
    "car":          ["car", "automobile", "sedan", "vehicle", "coupe"],
    "truck":        ["truck", "lorry", "pickup", "freight", "haul"],
    "bus":          ["bus", "coach", "shuttle", "transit", "minibus"],
    "train":        ["train", "locomotive", "railway", "express", "metro"],
    "plane":        ["plane", "aircraft", "jet", "airplane", "liner"],
    "boat":         ["boat", "ship", "vessel", "yacht", "ferry"],
    "bike":         ["bike", "bicycle", "cycle", "motorcycle", "scooter"],
    "helicopter":   ["helicopter", "chopper", "rotorcraft", "copter", "whirlybird"],
    "tank":         ["tank", "armor", "panzer", "military", "armored"],
    "rocket":       ["rocket", "spaceship", "shuttle", "missile", "spacecraft"],
    "van":          ["van", "minivan", "cargo", "delivery", "transit"],
    "suv":          ["suv", "sport", "utility", "crossover", "offroad"],
    "taxi":         ["taxi", "cab", "hack", "livery", "minicab"],
    "ambulance":    ["ambulance", "emergency", "medic", "rescue", "siren"],
    "firetruck":    ["firetruck", "fire", "engine", "ladder", "rescue"],
    "police":       ["police", "cruiser", "patrol", "squad", "enforcement"],
    "tram":         ["tram", "trolley", "streetcar", "light", "rail"],
    "subway":       ["subway", "metro", "underground", "tube", "rapid"],
    "cablecar":     ["cablecar", "gondola", "lift", "ski", "tramway"],
    "ferryboat":    ["ferryboat", "boat", "water", "crossing", "transport"],
    "canoe":        ["canoe", "kayak", "paddle", "water", "rowing"],
    "sailboat":     ["sailboat", "yacht", "sail", "mast", "wind"],
    "submarine":    ["submarine", "sub", "underwater", "navy", "torpedo"],
    "hovercraft":   ["hovercraft", "air", "cushion", "hover", "surface"],
    "jetski":       ["jetski", "personal", "water", "craft", "motor"],
    "glider":       ["glider", "sailplane", "soar", "unpowered", "wing"],
    "balloon":      ["balloon", "hot", "air", "inflate", "basket"],
    "drone":        ["drone", "uav", "unmanned", "aerial", "quadcopter"],
    "spaceship":    ["spaceship", "spacecraft", "orbital", "launch", "station"],
    "satellite":    ["satellite", "orbit", "space", "relay", "geostationary"],
    "tractor":      ["tractor", "farm", "plow", "agricultural", "harvester"],
    "bulldozer":    ["bulldozer", "construction", "blade", "earth", "crawler"],
    "crane":        ["crane", "lift", "hoist", "construction", "tower"],
    "forklift":     ["forklift", "lift", "warehouse", "pallet", "industrial"],
    "excavator":    ["excavator", "dig", "bucket", "construction", "track"],
    "rv":           ["rv", "motorhome", "camper", "recreational", "caravan"],
    "golfcart":     ["golfcart", "golf", "cart", "electric", "course"],
    "skateboard":   ["skateboard", "deck", "wheel", "board", "trick"],
    "scooter":      ["scooter", "moped", "vespa", "kick", "electric"],
    "segway":       ["segway", "hoverboard", "balance", "electric", "stand"],
    "wheelchair":   ["wheelchair", "chair", "mobility", "access", "disabled"],
    "stroller":     ["stroller", "pram", "baby", "carriage", "infant"],
    "cart":         ["cart", "wagon", "trolley", "push", "pull"],
    "carriage":     ["carriage", "horse", "buggy", "coach", "wagon"],
    "rickshaw":     ["rickshaw", "pedicab", "cycle", "cart", "pull"],
    "snowmobile":   ["snowmobile", "sled", "snow", "winter", "track"],
    "atv":          ["atv", "quad", "offroad", "terrain", "four"],
    "bicycle":      ["bicycle", "bike", "pedal", "two", "wheel"],
    "unicycle":     ["unicycle", "one", "wheel", "balance", "pedal"],
    "tricycle":     ["tricycle", "trike", "three", "wheel", "pedal"],
}

DOMAINS_LARGE = {
    "animal50": ANIMAL50,
    "vehicle50": VEHICLE50,
}


# ============================================================
# 核心函数
# ============================================================

def get_category_centers_residual(model, tokenizer, device, categories, layer_idx):
    """在指定层收集残差中心 — 批量处理提高效率"""
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        residuals = []
        for word in words[:3]:  # 只用前3个同义词节省时间
            prompt = f"The word is {word}"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            
            with torch.no_grad():
                inputs_embeds = embed_layer(input_ids)
                
                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu().numpy()
                        else:
                            captured[key] = output.detach().float().cpu().numpy()
                    return hook
                
                hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
                _ = model(inputs_embeds=inputs_embeds)
                hook.remove()
                
                if f"L{layer_idx}" in captured:
                    res = captured[f"L{layer_idx}"][0, -1, :]
                    residuals.append(res)
        
        if len(residuals) > 0:
            cat_centers[cat_name] = np.mean(residuals, axis=0)
    
    return cat_centers


def get_category_centers_embedding(model, tokenizer, categories):
    """从token embedding层获取类别中心"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        embeddings = []
        for word in words[:3]:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        
        if len(embeddings) > 0:
            cat_centers[cat_name] = np.mean(embeddings, axis=0)
    
    return cat_centers


def compute_pca(points):
    """PCA分析"""
    N, d = points.shape
    K = min(N - 1, d)
    
    mean = points.mean(axis=0)
    centered = points - mean
    
    U, S, Vt = svd(centered, full_matrices=False)
    
    scores = U[:, :K] * S[:K]
    directions = Vt[:K, :]
    
    total_var = np.sum(S**2)
    variance_explained = S[:K]**2
    cumvar = np.cumsum(variance_explained) / total_var if total_var > 0 else np.zeros(K)
    
    return {
        "scores": scores,
        "directions": directions,
        "singular_values": S[:K],
        "variance_explained": variance_explained,
        "cumvar": cumvar,
        "mean": mean,
        "K": K,
    }


def procrustes_align(X, Y):
    """Orthogonal Procrustes"""
    M = X.T @ Y
    U, sigma, Vt = svd(M, full_matrices=False)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    scale = np.sum(sigma) / np.sum(X**2) if np.sum(X**2) > 0 else 0
    Y_pred = X @ R
    error = np.sum((Y - Y_pred)**2) / np.sum(Y**2) if np.sum(Y**2) > 0 else 1.0
    
    return R, scale, error, sigma


def compute_rotation_angle(R):
    """计算旋转矩阵的旋转角度"""
    K = R.shape[0]
    trace_val = np.trace(R)
    cos_angle = np.clip((trace_val - 1) / max(K - 1, 1), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg, float(trace_val)


def compute_rotation_spectrum(R):
    """
    分解旋转矩阵R为2D旋转平面的角度谱
    
    旋转矩阵的特征值是共轭复数对e^{±iθ_k}或实数±1
    每个共轭对对应一个2D旋转平面, 角度为θ_k
    
    Returns:
        rotation_angles: [n_pairs] 各2D旋转平面的角度(度), 降序排列
    """
    eigenvalues = np.linalg.eigvals(R)
    
    rotation_angles = []
    processed = set()
    
    for i, ev in enumerate(eigenvalues):
        if i in processed:
            continue
        
        if np.isreal(ev):
            real_val = np.real(ev)
            if real_val < -0.5:
                rotation_angles.append(180.0)
            processed.add(i)
        else:
            theta = np.degrees(np.arccos(np.clip(np.real(ev), -1, 1)))
            rotation_angles.append(theta)
            for j in range(i+1, len(eigenvalues)):
                if j not in processed:
                    if np.abs(np.real(eigenvalues[j]) - np.real(ev)) < 0.01 and \
                       np.abs(np.abs(np.imag(eigenvalues[j])) - np.abs(np.imag(ev))) < 0.01:
                        processed.add(j)
                        break
            processed.add(i)
    
    rotation_angles.sort(reverse=True)
    return rotation_angles


def compute_cosine_dist_matrix(centers, cat_names):
    """计算cosine距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    cos_sim = points_norm @ points_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCVI: 大N(50) Procrustes旋转分析 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"  模型: {info.model_class}, d_model={d_model}, n_layers={n_layers}")
    
    # 选择5个关键层
    layer_candidates = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))
    print(f"  测试层({len(layer_candidates)}个): {layer_candidates}")
    
    all_results = {}
    
    for domain_name, categories in DOMAINS_LARGE.items():
        cat_names = list(categories.keys())
        N = len(cat_names)
        K = N - 1  # 49个PC
        
        print(f"\n--- 领域: {domain_name} (N={N}, K={K}) ---")
        
        # 1. Embedding中心
        print(f"  收集embedding中心...")
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个embedding中心")
            # 用已有的继续
            cat_names = [n for n in cat_names if n in emb_centers]
            N = len(cat_names)
            K = N - 1
            print(f"  调整为N={N}, K={K}")
        
        # 2. 各Residual层中心
        layer_centers = {"emb": emb_centers}
        for layer_idx in layer_candidates:
            print(f"  收集L{layer_idx}中心...", end=" ", flush=True)
            t0 = time.time()
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            t1 = time.time()
            # 只保留在两个center中都有的类别
            common_cats = [n for n in cat_names if n in res_centers]
            if len(common_cats) < 10:
                print(f"跳过(只有{len(common_cats)}个共同中心)")
                continue
            # 更新cat_names为公共子集
            cat_names = common_cats
            N = len(cat_names)
            K = N - 1
            layer_centers[f"L{layer_idx}"] = res_centers
            print(f"OK (N={N}, {t1-t0:.1f}s)")
        
        # 3. PCA at each layer (使用公共cat_names)
        print(f"  PCA分析(N={N}, K={K})...")
        layer_pcas = {}
        for layer_key, centers in layer_centers.items():
            points = np.array([centers[name] for name in cat_names])
            pca = compute_pca(points)
            layer_pcas[layer_key] = pca
        
        # 4. 直接emb→各层 Procrustes
        print(f"  Procrustes分析(emb→各层)...")
        emb_pca = layer_pcas["emb"]
        
        direct_results = []
        
        for layer_key in list(layer_pcas.keys())[1:]:
            X = emb_pca["scores"]
            Y = layer_pcas[layer_key]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            angle_deg, trace_val = compute_rotation_angle(R)
            rot_spectrum = compute_rotation_spectrum(R)
            
            # β_emb
            emb_dist = compute_cosine_dist_matrix(emb_centers, cat_names)
            current_proj = layer_pcas[layer_key]["scores"]
            current_dist = squareform(pdist(current_proj, metric='euclidean'))
            
            upper = np.triu_indices(N, k=1)
            emb_flat = emb_dist[upper]
            cur_flat = current_dist[upper]
            emb_z = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-10)
            cur_z = (cur_flat - cur_flat.mean()) / (cur_flat.std() + 1e-10)
            r_emb, p_emb = pearsonr(emb_z, cur_z)
            
            # 距离保持
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            
            result = {
                "layer_key": layer_key,
                "layer_idx": int(layer_key[1:]) if layer_key != "emb" else -1,
                "direct_angle_deg": float(angle_deg),
                "direct_trace": float(trace_val),
                "direct_error": float(error),
                "beta_emb": float(r_emb),
                "r_dist": float(r_dist),
                "N": N,
                "K": K,
                # 旋转频谱
                "rotation_spectrum": rot_spectrum[:25],
                "n_rotations_gt90": sum(1 for a in rot_spectrum if a > 90),
                "n_rotations_gt60": sum(1 for a in rot_spectrum if a > 60),
                "n_rotations_lt30": sum(1 for a in rot_spectrum if a < 30),
                "max_rotation": max(rot_spectrum) if rot_spectrum else 0,
                "mean_rotation": float(np.mean(rot_spectrum)) if rot_spectrum else 0,
                "median_rotation": float(np.median(rot_spectrum)) if rot_spectrum else 0,
                # PCA方差
                "emb_cumvar_top10": float(emb_pca["cumvar"][min(9, K-1)]),
                "emb_cumvar_top25": float(emb_pca["cumvar"][min(24, K-1)]),
                "res_cumvar_top10": float(layer_pcas[layer_key]["cumvar"][min(9, K-1)]),
                "res_cumvar_top25": float(layer_pcas[layer_key]["cumvar"][min(24, K-1)]),
            }
            
            direct_results.append(result)
            
            # 打印
            n_gt90 = sum(1 for a in rot_spectrum if a > 90)
            n_lt30 = sum(1 for a in rot_spectrum if a < 30)
            top5_rot = ", ".join([f"{a:.1f}°" for a in rot_spectrum[:5]])
            
            print(f"    emb→{layer_key}: θ={angle_deg:.1f}°, "
                  f"β_emb={r_emb:+.3f}, r_dist={r_dist:.3f}, "
                  f"n_gt90={n_gt90}/{len(rot_spectrum)}, "
                  f"n_lt30={n_lt30}/{len(rot_spectrum)}, "
                  f"top5=[{top5_rot}]")
        
        # 5. 层间Procrustes
        print(f"  层间Procrustes分析...")
        layer_keys = list(layer_pcas.keys())
        step_results = []
        
        for i in range(len(layer_keys) - 1):
            key_from = layer_keys[i]
            key_to = layer_keys[i + 1]
            
            X = layer_pcas[key_from]["scores"]
            Y = layer_pcas[key_to]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            angle_deg, trace_val = compute_rotation_angle(R)
            rot_spectrum = compute_rotation_spectrum(R)
            
            step_results.append({
                "from": key_from,
                "to": key_to,
                "step_angle_deg": float(angle_deg),
                "step_error": float(error),
                "rotation_spectrum": rot_spectrum[:25],
                "n_rotations_gt90": sum(1 for a in rot_spectrum if a > 90),
                "mean_rotation": float(np.mean(rot_spectrum)) if rot_spectrum else 0,
            })
            
            n_gt90 = sum(1 for a in rot_spectrum if a > 90)
            print(f"    {key_from}→{key_to}: step_θ={angle_deg:.1f}°, "
                  f"n_gt90={n_gt90}/{len(rot_spectrum)}")
        
        # 6. 旋转频谱详细分析
        print(f"\n  旋转频谱详细分析:")
        
        for dr in direct_results:
            spec = dr["rotation_spectrum"]
            if not spec:
                continue
            
            # 频谱直方图
            bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
            hist = np.histogram(spec, bins=bins)[0]
            
            print(f"    {dr['layer_key']}: θ={dr['direct_angle_deg']:.1f}°, β={dr['beta_emb']:+.3f}")
            print(f"      频谱: ", end="")
            for k_idx in range(len(hist)):
                if hist[k_idx] > 0:
                    print(f"{bins[k_idx]}-{bins[k_idx+1]}°:{hist[k_idx]} ", end="")
            print()
            
            # 小/中/大旋转比例
            n_small = sum(1 for a in spec if a < 30)
            n_medium = sum(1 for a in spec if 30 <= a < 90)
            n_large = sum(1 for a in spec if a >= 90)
            print(f"      小(<30°):{n_small}({n_small/len(spec)*100:.0f}%) "
                  f"中(30-90°):{n_medium}({n_medium/len(spec)*100:.0f}%) "
                  f"大(≥90°):{n_large}({n_large/len(spec)*100:.0f}%)")
        
        all_results[domain_name] = {
            "direct": direct_results,
            "step": step_results,
            "N": N,
            "K": K,
        }
    
    # ============================================================
    # 汇总分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CCVI 汇总分析 — {model_name}")
    print(f"{'='*70}")
    
    # === 1. Animal vs Vehicle 旋转对比 ===
    print("\n--- 1. Animal50 vs Vehicle50 旋转对比 ---")
    
    for domain_name in ["animal50", "vehicle50"]:
        if domain_name not in all_results:
            continue
        
        data = all_results[domain_name]
        direct = data["direct"]
        
        if not direct:
            continue
        
        avg_angle = np.mean([d["direct_angle_deg"] for d in direct])
        avg_beta = np.mean([d["beta_emb"] for d in direct])
        avg_rdist = np.mean([d["r_dist"] for d in direct])
        avg_n_gt90 = np.mean([d["n_rotations_gt90"] for d in direct])
        avg_mean_rot = np.mean([d["mean_rotation"] for d in direct])
        K = data["K"]
        
        print(f"  {domain_name}: direct_θ={avg_angle:.1f}°, "
              f"β_emb={avg_beta:+.3f}, r_dist={avg_rdist:.3f}, "
              f"avg_n_gt90={avg_n_gt90:.1f}/{K}, "
              f"avg_mean_rot={avg_mean_rot:.1f}°")
    
    # === 2. β_emb与旋转频谱的关系 ===
    print("\n--- 2. β_emb与旋转频谱特征的关系 ---")
    
    all_n_gt90 = []
    all_mean_rot = []
    all_max_rot = []
    all_beta = []
    
    for domain_name, data in all_results.items():
        for dr in data["direct"]:
            all_n_gt90.append(dr["n_rotations_gt90"])
            all_mean_rot.append(dr["mean_rotation"])
            all_max_rot.append(dr["max_rotation"])
            all_beta.append(dr["beta_emb"])
    
    if len(all_beta) > 5:
        r1, p1 = pearsonr(all_n_gt90, all_beta)
        r2, p2 = pearsonr(all_mean_rot, all_beta)
        r3, p3 = pearsonr(all_max_rot, all_beta)
        
        print(f"  n_gt90 ↔ β_emb:  r={r1:+.3f}, p={p1:.3f}")
        print(f"  mean_rot ↔ β:    r={r2:+.3f}, p={p2:.3f}")
        print(f"  max_rot ↔ β:     r={r3:+.3f}, p={p3:.3f}")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_tested": layer_candidates,
        "domain_results": all_results,
    }
    
    out_path = TEMP / f"ccvi_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    release_model(model)
    gc.collect()
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    result = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
