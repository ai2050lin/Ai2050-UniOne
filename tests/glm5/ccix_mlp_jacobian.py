"""
CCIX(359): MLP几何修正的维度分解分析
===================================

★★★★★ CCVIII核心发现:
  - MLP是几何保持者: no_mlp在深层给负β_emb!
  - 注意力是几何改变者: no_attn降低β但不变负
  - 三组件动态平衡: 残差(锚) + 注意力(变换) + MLP(修正)

★★★★★ 本实验目标:
  1. MLP增量在语义子空间vs正交子空间的能量分布
  2. MLP对attention几何扭曲的修正程度量化
  3. MLP Jacobian在语义方向vs随机方向的放大/保持差异
  4. ★★★★★ 核心问题: MLP如何在维度层面"修正"几何扭曲?

核心思路:
  h_{l+1} = h_l + Δ_attn + Δ_mlp
  
  如果MLP是"几何修正器":
  - Δ_mlp应该在语义子空间(V_pca)有更多能量 → 修正语义维度
  - Δ_mlp应该使距离结构更接近embedding → 提升β
  - MLP Jacobian应该在语义方向有更大的放大 → 保持语义结构

MLP Jacobian (SwiGLU):
  MLP(x) = W_down @ (silu(W_gate @ x) ⊙ (W_up @ x))
  J_MLP(x) = W_down @ [diag(silu'(g) ⊙ u) @ W_gate + diag(silu(g)) @ W_up]
  其中 g = W_gate @ x, u = W_up @ x
  JVP: J_MLP @ v = W_down @ [(silu'(g) ⊙ u) ⊙ (W_gate @ v) + silu(g) ⊙ (W_up @ v)]

用法:
  python ccix_mlp_jacobian.py --model qwen3
  python ccix_mlp_jacobian.py --model glm4
  python ccix_mlp_jacobian.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, get_layer_weights, release_model,
    MODEL_CONFIGS
)
import transformers

TEMP = Path("tests/glm5_temp")

# 复用CCVIII的领域定义
ANIMAL50 = {
    "dog": ["dog", "puppy", "hound", "canine", "pooch"],
    "cat": ["cat", "kitten", "feline", "tomcat", "kitty"],
    "wolf": ["wolf", "werewolf", "lupine", "coyote", "jackal"],
    "lion": ["lion", "tiger", "leopard", "cheetah", "panther"],
    "bird": ["bird", "sparrow", "robin", "finch", "wren"],
    "eagle": ["eagle", "hawk", "falcon", "vulture", "osprey"],
    "fish": ["fish", "trout", "salmon", "bass", "perch"],
    "shark": ["shark", "whale", "dolphin", "porpoise", "orca"],
    "snake": ["snake", "serpent", "viper", "cobra", "python"],
    "lizard": ["lizard", "gecko", "iguana", "chameleon", "salamander"],
    "horse": ["horse", "stallion", "mare", "pony", "colt"],
    "cow": ["cow", "cattle", "bull", "ox", "heifer"],
    "pig": ["pig", "swine", "hog", "boar", "sow"],
    "sheep": ["sheep", "lamb", "ram", "ewe", "fleece"],
    "goat": ["goat", "billy", "kid", "nanny", "caprine"],
    "chicken": ["chicken", "hen", "rooster", "cock", "poultry"],
    "duck": ["duck", "drake", "goose", "swan", "mallard"],
    "rabbit": ["rabbit", "bunny", "hare", "lagomorph", "cottontail"],
    "mouse": ["mouse", "rat", "rodent", "vole", "hamster"],
    "bear": ["bear", "grizzly", "polar", "cub", "ursine"],
    "elephant": ["elephant", "pachyderm", "tusk", "mammoth", "trunk"],
    "giraffe": ["giraffe", "ruminant", "hoof", "neck", "savanna"],
    "zebra": ["zebra", "stripes", "equine", "mustang", "striped"],
    "monkey": ["monkey", "ape", "primate", "chimp", "baboon"],
    "gorilla": ["gorilla", "silverback", "simian", "primate", "ape"],
    "penguin": ["penguin", "emperor", "frost", "ice", "flightless"],
    "owl": ["owl", "nocturnal", "hoot", "raptor", "bird"],
    "parrot": ["parrot", "macaw", "cockatoo", "tropical", "feather"],
    "turtle": ["turtle", "tortoise", "shell", "reptile", "terrapin"],
    "crocodile": ["crocodile", "alligator", "caiman", "reptile", "scale"],
    "frog": ["frog", "toad", "amphibian", "tadpole", "ribbit"],
    "butterfly": ["butterfly", "moth", "insect", "caterpillar", "cocoon"],
    "bee": ["bee", "honey", "wasp", "hive", "pollinator"],
    "ant": ["ant", "colony", "insect", "worker", "formic"],
    "spider": ["spider", "arachnid", "web", "silk", "tarantula"],
    "crab": ["crab", "lobster", "crustacean", "claw", "shellfish"],
    "octopus": ["octopus", "squid", "tentacle", "cephalopod", "ink"],
    "jellyfish": ["jellyfish", "tentacle", "sting", "medusa", "plankton"],
    "deer": ["deer", "stag", "doe", "fawn", "buck"],
    "fox": ["fox", "vixen", "canine", "crafty", "pelt"],
    "squirrel": ["squirrel", "chipmunk", "rodent", "acorn", "bushy"],
    "beaver": ["beaver", "dam", "rodent", "lodger", "gnaw"],
    "otter": ["otter", "river", "mustelid", "playful", "swim"],
    "kangaroo": ["kangaroo", "marsupial", "joey", "outback", "hop"],
    "koala": ["koala", "marsupial", "eucalyptus", "bear", "pouch"],
    "panda": ["panda", "bamboo", "bear", "china", "black"],
    "rhino": ["rhino", "rhinoceros", "horn", "pachyderm", "charge"],
    "hippo": ["hippo", "hippopotamus", "river", "africa", "water"],
    "camel": ["camel", "dromedary", "hump", "desert", "arabian"],
    "bat": ["bat", "nocturnal", "wing", "cave", "echolocation"],
    "seal": ["seal", "sea", "flipper", "pinniped", "arctic"],
}

VEHICLE50 = {
    "car": ["car", "automobile", "sedan", "vehicle", "auto"],
    "truck": ["truck", "lorry", "freight", "rig", "semi"],
    "bus": ["bus", "coach", "transit", "shuttle", "omnibus"],
    "train": ["train", "locomotive", "railway", "express", "freight"],
    "plane": ["plane", "aircraft", "jet", "airliner", "flight"],
    "helicopter": ["helicopter", "chopper", "rotor", "whirlybird", "heli"],
    "boat": ["boat", "vessel", "craft", "ship", "sailboat"],
    "submarine": ["submarine", "sub", "u-boat", "torpedo", "naval"],
    "bicycle": ["bicycle", "bike", "cycle", "pedal", "two-wheeler"],
    "motorcycle": ["motorcycle", "motorbike", "chopper", "scooter", "hog"],
    "scooter": ["scooter", "moped", "vespa", "kick", "razor"],
    "van": ["van", "minivan", "cargo", "delivery", "panel"],
    "taxi": ["taxi", "cab", "hack", "meter", "fare"],
    "ambulance": ["ambulance", "medic", "emergency", "rescue", "siren"],
    "firetruck": ["firetruck", "fire_engine", "ladder", "pumper", "hose"],
    "tractor": ["tractor", "farm", "plow", "harvester", "agricultural"],
    "bulldozer": ["bulldozer", "crawler", "blade", "earthmover", "dozer"],
    "crane": ["crane", "hoist", "derrick", "winch", "lifting"],
    "forklift": ["forklift", "pallet", "warehouse", "industrial", "lift"],
    "tank": ["tank", "armor", "turret", "military", "tracked"],
    "jeep": ["jeep", "suv", "offroad", "four_by_four", "land"],
    "wagon": ["wagon", "carriage", "cart", "buggy", "stagecoach"],
    "canoe": ["canoe", "kayak", "paddle", "rowing", "outrigger"],
    "raft": ["raft", "float", "pontoon", "inflatable", "log"],
    "yacht": ["yacht", "sail", "luxury", "cruiser", "boating"],
    "jet_ski": ["jet_ski", "watercraft", "pwc", "wave", "sea_doo"],
    "tram": ["tram", "trolley", "streetcar", "light_rail", "cable"],
    "rickshaw": ["rickshaw", "pedicab", "tuk_tuk", "cycle", "auto"],
    "gondola": ["gondola", "cable_car", "ski_lift", "chairlift", "aerial"],
    "hovercraft": ["hovercraft", "air_cushion", "hover", "surface", "acv"],
    "segway": ["segway", "gyroscope", "self_balancing", "electric", "personal"],
    "skateboard": ["skateboard", "deck", "kickflip", "ollie", "grind"],
    "rollerblade": ["rollerblade", "inline", "skate", "blade", "wheels"],
    "snowmobile": ["snowmobile", "sled", "snowmachine", "ski_doo", "winter"],
    "atv": ["atv", "quad", "four_wheeler", "offroad", "all_terrain"],
    "go_kart": ["go_kart", "kart", "racing", "mini", "track"],
    "golf_cart": ["golf_cart", "buggy", "course", "electric", "club"],
    "wheelchair": ["wheelchair", "mobility", "accessible", "handicap", "chair"],
    "stroller": ["stroller", "pram", "baby", "pushchair", "buggy"],
    "wagon_train": ["caravan", "convoy", "procession", "wagons", "trail"],
    "catapult": ["catapult", "trebuchet", "siege", "launcher", "projectile"],
    "chariot": ["chariot", "ancient", "roman", "war", "horse_drawn"],
    "galley": ["galley", "row", "ancient", "warship", "oar"],
    "longship": ["longship", "viking", "norse", "drakkar", "dragon"],
    "galleon": ["galleon", "sailing", "treasure", "spanish", "man_of_war"],
    "clipper": ["clipper", "tea", "fast", "sailing", "merchant"],
    "dinghy": ["dinghy", "small", "rowboat", "tender", "inflatable"],
    "catamaran": ["catamaran", "hull", "twin", "sailing", "multihull"],
    "houseboat": ["houseboat", "floating", "liveaboard", "residence", "barge"],
    "ferry": ["ferry", "crossing", "passenger", "roll_on", "boat"],
}


def compute_pca(points):
    """PCA: 返回scores, components, singular_values, mean, K"""
    N, d = points.shape
    K = min(N - 1, d)
    mean = points.mean(axis=0)
    centered = points - mean
    U, S, Vt = svd(centered, full_matrices=False)
    scores = U[:, :K] * S[:K]
    return scores, Vt[:K, :], S[:K], mean, K


def compute_beta_emb(emb_matrix, res_matrix):
    """计算β_emb: embedding cosine距离 vs residual euclidean距离的相关性"""
    N = emb_matrix.shape[0]
    
    res_scores, _, _, _, K = compute_pca(res_matrix)
    emb_scores, _, _, _, _ = compute_pca(emb_matrix)
    
    norms_emb = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms_emb = np.maximum(norms_emb, 1e-10)
    emb_norm = emb_matrix / norms_emb
    cos_sim = emb_norm @ emb_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    
    res_dist = squareform(pdist(res_scores, metric='euclidean'))
    
    upper = np.triu_indices(N, k=1)
    beta_emb, _ = pearsonr(cos_dist[upper], res_dist[upper])
    
    return beta_emb


def silu(x):
    """SiLU/Swish activation"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def silu_derivative(x):
    """SiLU导数: sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig * (1.0 + x * (1.0 - sig))


def compute_mlp_jvp(W_gate, W_up, W_down, x, v):
    """
    计算MLP Jacobian-vector product: J_MLP(x) @ v
    
    MLP(x) = W_down @ (silu(W_gate @ x) ⊙ (W_up @ x))
    J_MLP(x) @ v = W_down @ [(silu'(g) ⊙ u) ⊙ (W_gate @ v) + silu(g) ⊙ (W_up @ v)]
    其中 g = W_gate @ x, u = W_up @ x
    
    Args:
        W_gate: [intermediate, d_model]
        W_up: [intermediate, d_model]
        W_down: [d_model, intermediate]
        x: [d_model] 输入点
        v: [d_model] 方向向量
    
    Returns:
        J_MLP @ v: [d_model]
    """
    g = W_gate @ x          # [intermediate]
    u = W_up @ x            # [intermediate]
    silu_g = silu(g)        # [intermediate]
    silu_prime_g = silu_derivative(g)  # [intermediate]
    
    # J_f @ v = (silu'(g) ⊙ u) ⊙ (W_gate @ v) + silu(g) ⊙ (W_up @ v)
    W_gate_v = W_gate @ v   # [intermediate]
    W_up_v = W_up @ v       # [intermediate]
    
    jf_v = (silu_prime_g * u) * W_gate_v + silu_g * W_up_v  # [intermediate]
    
    # J_MLP @ v = W_down @ jf_v
    jmlp_v = W_down @ jf_v  # [d_model]
    
    return jmlp_v


def collect_all_data(model, tokenizer, device, model_name, domain_dict, sample_layers):
    """
    收集各层的激活数据: embedding, layer输出, attention增量, MLP增量, MLP输入
    
    Returns:
        emb_matrix: [N, d_model]
        data: {layer_idx: {layer_out, attn_delta, mlp_delta, mlp_input}} 
              每个是 [N, d_model]
    """
    layers = get_layers(model)
    n_layers = len(layers)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    # 获取类别
    categories = list(domain_dict.keys())
    N = len(categories)
    
    # 初始化存储
    emb_list = []
    data = {li: {"layer_out": [], "attn_delta": [], "mlp_delta": [], "mlp_input": []}
            for li in sample_layers}
    
    for cat_idx, cat in enumerate(categories):
        main_word = domain_dict[cat][0]
        tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
        if len(tok_ids) == 0:
            continue
        tok_id = tok_ids[-1]
        
        # Embedding
        with torch.no_grad():
            emb = embed_layer.weight[tok_id].detach().float().cpu().numpy()
        emb_list.append(emb)
        
        # Forward pass with hooks
        input_ids = torch.tensor([[tok_id]], device=device)
        captured = {}
        
        def make_hook(key):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in sample_layers:
            layer = layers[li]
            # 层输出
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
            # Attention输出 (Δ_attn)
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(make_hook(f"A{li}")))
            # MLP输出 (Δ_mlp)
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_hook(f"M{li}")))
            # Post-attention LayerNorm输出 (MLP输入)
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(layer, ln_name):
                    ln_mod = getattr(layer, ln_name)
                    hooks.append(ln_mod.register_forward_hook(make_hook(f"LN{li}")))
                    break
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                print(f"  Forward failed for {cat}: {e}")
        
        for h in hooks:
            h.remove()
        
        # 提取结果
        for li in sample_layers:
            l_key = f"L{li}"
            a_key = f"A{li}"
            m_key = f"M{li}"
            ln_key = f"LN{li}"
            
            if l_key in captured:
                data[li]["layer_out"].append(captured[l_key][0, -1, :].numpy())
            if a_key in captured:
                data[li]["attn_delta"].append(captured[a_key][0, -1, :].numpy())
            if m_key in captured:
                data[li]["mlp_delta"].append(captured[m_key][0, -1, :].numpy())
            if ln_key in captured:
                data[li]["mlp_input"].append(captured[ln_key][0, -1, :].numpy())
        
        del captured
        gc.collect()
        
        if cat_idx % 10 == 0:
            print(f"  已收集 {cat_idx+1}/{N} tokens")
    
    # 转换为numpy
    emb_matrix = np.array(emb_list)  # [N, d_model]
    for li in sample_layers:
        for key in data[li]:
            if len(data[li][key]) == N:
                data[li][key] = np.array(data[li][key])  # [N, d_model]
            else:
                data[li][key] = None
    
    return emb_matrix, data, N


def analyze_subspace_energy(emb_matrix, data, sample_layers, N):
    """
    Part 1: MLP增量在语义子空间vs正交子空间的能量分布
    
    核心思路:
    - Embedding PCA → 语义子空间基 V_pca [d_model, K]
    - 正交补空间基 V_orth [d_model, d_model-K]
    - 将Δ_mlp和Δ_attn投影到两个子空间
    - 计算能量比: semantic_ratio = ||proj_pca||² / ||Δ||²
    """
    print("\n  === Part 1: 子空间能量分析 ===")
    
    # Embedding PCA
    emb_centered = emb_matrix - emb_matrix.mean(axis=0)
    _, _, S_emb, _, K = compute_pca(emb_matrix)
    # PCA components: Vt[:K] 是前K个主成分
    U_emb, S_emb_full, Vt_emb = svd(emb_centered, full_matrices=False)
    V_pca = Vt_emb[:K].T     # [d_model, K] — 语义子空间基
    V_orth = Vt_emb[K:].T    # [d_model, d_model-K] — 正交补基
    
    print(f"  Embedding PCA: K={K}, 前5奇异值={S_emb[:5].round(2)}")
    
    results = []
    
    for li in sample_layers:
        attn_delta = data[li].get("attn_delta")
        mlp_delta = data[li].get("mlp_delta")
        
        if attn_delta is None or mlp_delta is None:
            continue
        if not isinstance(attn_delta, np.ndarray) or not isinstance(mlp_delta, np.ndarray):
            continue
        
        # 对每个token的增量进行投影
        # Δ是 [N, d_model]
        
        # Attn增量在语义子空间的投影
        proj_attn_pca = attn_delta @ V_pca @ V_pca.T  # [N, d_model]
        proj_attn_orth = attn_delta @ V_orth @ V_orth.T
        
        energy_attn_total = np.sum(attn_delta ** 2)
        energy_attn_pca = np.sum(proj_attn_pca ** 2)
        energy_attn_orth = np.sum(proj_attn_orth ** 2)
        
        # MLP增量
        proj_mlp_pca = mlp_delta @ V_pca @ V_pca.T
        proj_mlp_orth = mlp_delta @ V_orth @ V_orth.T
        
        energy_mlp_total = np.sum(mlp_delta ** 2)
        energy_mlp_pca = np.sum(proj_mlp_pca ** 2)
        energy_mlp_orth = np.sum(proj_mlp_orth ** 2)
        
        # 能量比
        sem_ratio_attn = energy_attn_pca / max(energy_attn_total, 1e-20)
        sem_ratio_mlp = energy_mlp_pca / max(energy_mlp_total, 1e-20)
        
        # 随机基线: K/d_model 是随机投影到K维子空间的期望能量比
        random_baseline = K / emb_matrix.shape[1]
        
        result = {
            "layer": li,
            "K": K,
            "d_model": emb_matrix.shape[1],
            "random_baseline": float(random_baseline),
            # Attn
            "energy_attn_total": float(energy_attn_total),
            "energy_attn_pca": float(energy_attn_pca),
            "sem_ratio_attn": float(sem_ratio_attn),
            # MLP
            "energy_mlp_total": float(energy_mlp_total),
            "energy_mlp_pca": float(energy_mlp_pca),
            "sem_ratio_mlp": float(sem_ratio_mlp),
            # 增量比: MLP能量 vs Attn能量
            "mlp_attn_energy_ratio": float(energy_mlp_total / max(energy_attn_total, 1e-20)),
        }
        results.append(result)
        
        print(f"  L{li:2d}: sem_ratio Attn={sem_ratio_attn:.3f} MLP={sem_ratio_mlp:.3f} "
              f"(random={random_baseline:.3f}), "
              f"energy MLP/Attn={result['mlp_attn_energy_ratio']:.3f}")
    
    return results, V_pca, V_orth, K


def analyze_geometry_correction(emb_matrix, data, sample_layers, N):
    """
    Part 2: MLP对几何扭曲的修正程度量化
    
    核心思路:
    - h_after_attn = h_{l+1} - Δ_mlp (移除MLP增量, 只保留attention)
    - h_after_mlp = h_{l+1} (完整层输出)
    - β_attn = β_emb(emb, h_after_attn)
    - β_mlp = β_emb(emb, h_after_mlp)
    - Correction = β_mlp - β_attn
    """
    print("\n  === Part 2: 几何修正分析 ===")
    
    results = []
    
    for li in sample_layers:
        layer_out = data[li].get("layer_out")
        mlp_delta = data[li].get("mlp_delta")
        attn_delta = data[li].get("attn_delta")
        
        if layer_out is None or mlp_delta is None or attn_delta is None:
            continue
        if not isinstance(layer_out, np.ndarray) or not isinstance(mlp_delta, np.ndarray):
            continue
        
        # h_after_mlp = h_{l+1}
        h_after_mlp = layer_out
        
        # h_after_attn = h_{l+1} - Δ_mlp
        h_after_attn = layer_out - mlp_delta
        
        # β after MLP
        beta_mlp = compute_beta_emb(emb_matrix, h_after_mlp)
        
        # β after Attn only
        beta_attn = compute_beta_emb(emb_matrix, h_after_attn)
        
        # Correction
        correction = beta_mlp - beta_attn
        
        result = {
            "layer": li,
            "beta_attn_only": float(beta_attn),
            "beta_mlp_full": float(beta_mlp),
            "correction": float(correction),
        }
        results.append(result)
        
        print(f"  L{li:2d}: β_attn={beta_attn:+.3f}, β_mlp={beta_mlp:+.3f}, "
              f"correction={correction:+.3f}")
    
    return results


def analyze_mlp_jacobian(model, model_name, emb_matrix, data, sample_layers, 
                          V_pca, V_orth, K, N, n_random=50):
    """
    Part 3: MLP Jacobian的方向放大分析
    
    核心思路:
    - 对每个PCA方向v_i, 计算 J_MLP(x_mean) @ v_i
    - 放大系数 = ||J_MLP @ v_i|| / ||v_i||
    - 方向保持 = cos(J_MLP @ v_i, v_i)
    - 与随机方向对比
    """
    print("\n  === Part 3: MLP Jacobian JVP分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    # 生成随机方向 (与PCA方向同数量)
    np.random.seed(42)
    random_dirs = np.random.randn(n_random, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    results = []
    
    for li in sample_layers:
        mlp_input = data[li].get("mlp_input")
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        # 获取MLP权重
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        if lw.W_gate is None or lw.W_up is None or lw.W_down is None:
            print(f"  L{li}: MLP权重不完整, 跳过")
            continue
        
        W_gate = lw.W_gate.astype(np.float32)  # [intermediate, d_model]
        W_up = lw.W_up.astype(np.float32)
        W_down = lw.W_down.astype(np.float32)
        
        # 在均值输入点计算Jacobian
        x_mean = mlp_input.mean(axis=0).astype(np.float32)  # [d_model]
        
        # ★ PCA方向的JVP
        pca_amplifications = []
        pca_preservations = []
        
        for k in range(K):
            v = V_pca[:, k].astype(np.float32)  # 第k个PCA方向
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            
            amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            
            pca_amplifications.append(amp)
            pca_preservations.append(float(cos_pres))
        
        # ★ 随机方向的JVP
        rand_amplifications = []
        rand_preservations = []
        
        for i in range(n_random):
            v = random_dirs[i].astype(np.float32)
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            
            amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            
            rand_amplifications.append(amp)
            rand_preservations.append(float(cos_pres))
        
        # ★ 正交子空间方向的JVP (从V_orth采样)
        orth_amplifications = []
        orth_preservations = []
        n_orth = min(n_random, V_orth.shape[1])
        for k in range(n_orth):
            v = V_orth[:, k].astype(np.float32)
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            
            amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            
            orth_amplifications.append(amp)
            orth_preservations.append(float(cos_pres))
        
        # 汇总统计
        result = {
            "layer": li,
            # PCA方向
            "pca_amp_mean": float(np.mean(pca_amplifications)),
            "pca_amp_std": float(np.std(pca_amplifications)),
            "pca_pres_mean": float(np.mean(pca_preservations)),
            "pca_pres_std": float(np.std(pca_preservations)),
            # 前10 PCA方向 (最重要)
            "pca_top10_amp": float(np.mean(pca_amplifications[:10])),
            "pca_top10_pres": float(np.mean(pca_preservations[:10])),
            # 随机方向
            "rand_amp_mean": float(np.mean(rand_amplifications)),
            "rand_amp_std": float(np.std(rand_amplifications)),
            "rand_pres_mean": float(np.mean(rand_preservations)),
            "rand_pres_std": float(np.std(rand_preservations)),
            # 正交子空间方向
            "orth_amp_mean": float(np.mean(orth_amplifications)),
            "orth_pres_mean": float(np.mean(orth_preservations)),
            # 关键对比
            "amp_ratio_pca_vs_rand": float(np.mean(pca_amplifications) / max(np.mean(rand_amplifications), 1e-10)),
            "pres_diff_pca_vs_rand": float(np.mean(pca_preservations) - np.mean(rand_preservations)),
            "amp_ratio_pca_vs_orth": float(np.mean(pca_amplifications) / max(np.mean(orth_amplifications), 1e-10)),
            "pres_diff_pca_vs_orth": float(np.mean(pca_preservations) - np.mean(orth_preservations)),
            # 逐PCA方向 (前20个)
            "pca_amplifications_top20": [float(x) for x in pca_amplifications[:20]],
            "pca_preservations_top20": [float(x) for x in pca_preservations[:20]],
        }
        results.append(result)
        
        print(f"  L{li:2d}: PCA amp={np.mean(pca_amplifications):.4f}±{np.std(pca_amplifications):.4f}, "
              f"Rand amp={np.mean(rand_amplifications):.4f}±{np.std(rand_amplifications):.4f}, "
              f"Orth amp={np.mean(orth_amplifications):.4f}, "
              f"ratio(PCA/Rand)={result['amp_ratio_pca_vs_rand']:.3f}")
        print(f"         PCA pres={np.mean(pca_preservations):.4f}±{np.std(pca_preservations):.4f}, "
              f"Rand pres={np.mean(rand_preservations):.4f}±{np.std(rand_preservations):.4f}, "
              f"diff={result['pres_diff_pca_vs_rand']:+.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "subspace", "correction", "jacobian"])
    args = parser.parse_args()
    
    model_name = args.model
    analysis = args.analysis
    
    print(f"\n{'='*70}")
    print(f"CCIX: MLP几何修正的维度分解分析 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型 (DS7B用device_map=auto避免CPU OOM)
    if model_name == "deepseek7b":
        cfg = MODEL_CONFIGS[model_name]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"  [CCIX] {model_name} loaded with device_map=auto, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    n_layers = model_info.n_layers
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f"  采样层: {sample_layers}")
    
    domains = [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]
    all_results = {"model": model_name, "d_model": model_info.d_model, "n_layers": n_layers}
    
    for domain_name, domain_dict in domains:
        print(f"\n{'='*70}")
        print(f"Domain: {domain_name}")
        print(f"{'='*70}")
        
        # 收集数据
        print(f"\n  收集激活数据...")
        emb_matrix, data, N = collect_all_data(
            model, tokenizer, device, model_name, domain_dict, sample_layers
        )
        print(f"  收集完成: N={N}, d_model={emb_matrix.shape[1]}")
        
        # Part 1: 子空间能量
        if analysis in ["all", "subspace"]:
            subspace_results, V_pca, V_orth, K = analyze_subspace_energy(
                emb_matrix, data, sample_layers, N
            )
            all_results[f"subspace_{domain_name}"] = subspace_results
        else:
            # 需要V_pca给Part 3用
            emb_centered = emb_matrix - emb_matrix.mean(axis=0)
            _, _, _, Vt_emb = svd(emb_centered, full_matrices=False)[:4]
            K = min(N - 1, emb_matrix.shape[1])
            V_pca = Vt_emb[:K].T
            V_orth = Vt_emb[K:].T
        
        # Part 2: 几何修正
        if analysis in ["all", "correction"]:
            correction_results = analyze_geometry_correction(
                emb_matrix, data, sample_layers, N
            )
            all_results[f"correction_{domain_name}"] = correction_results
        
        # Part 3: MLP Jacobian
        if analysis in ["all", "jacobian"]:
            jacobian_results = analyze_mlp_jacobian(
                model, model_name, emb_matrix, data, sample_layers,
                V_pca, V_orth, K, N
            )
            all_results[f"jacobian_{domain_name}"] = jacobian_results
    
    # 保存结果
    output_path = TEMP / f"ccix_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    
    # ===== 汇总分析 =====
    print(f"\n{'='*70}")
    print(f"CCIX 汇总分析: {model_name}")
    print(f"{'='*70}")
    
    # Part 1 汇总
    for domain_name, _ in domains:
        key = f"subspace_{domain_name}"
        if key in all_results:
            data_r = all_results[key]
            print(f"\n--- {domain_name}: 子空间能量比 (semantic / total) ---")
            print(f"  {'Layer':>5s}  {'Attn_sem':>9s}  {'MLP_sem':>9s}  {'Random':>9s}  {'MLP/Attn_E':>10s}")
            for r in data_r:
                print(f"  L{r['layer']:3d}  {r['sem_ratio_attn']:9.4f}  {r['sem_ratio_mlp']:9.4f}  "
                      f"{r['random_baseline']:9.4f}  {r['mlp_attn_energy_ratio']:10.4f}")
    
    # Part 2 汇总
    for domain_name, _ in domains:
        key = f"correction_{domain_name}"
        if key in all_results:
            data_r = all_results[key]
            print(f"\n--- {domain_name}: 几何修正 ---")
            print(f"  {'Layer':>5s}  {'β_attn':>8s}  {'β_mlp':>8s}  {'Δ(correction)':>13s}")
            for r in data_r:
                print(f"  L{r['layer']:3d}  {r['beta_attn_only']:+8.3f}  {r['beta_mlp_full']:+8.3f}  "
                      f"{r['correction']:+13.3f}")
    
    # Part 3 汇总
    for domain_name, _ in domains:
        key = f"jacobian_{domain_name}"
        if key in all_results:
            data_r = all_results[key]
            print(f"\n--- {domain_name}: MLP Jacobian方向分析 ---")
            print(f"  {'Layer':>5s}  {'PCA_amp':>8s}  {'Rand_amp':>8s}  {'ratio':>6s}  "
                  f"{'PCA_pres':>8s}  {'Rand_pres':>8s}  {'Δpres':>6s}")
            for r in data_r:
                print(f"  L{r['layer']:3d}  {r['pca_amp_mean']:8.4f}  {r['rand_amp_mean']:8.4f}  "
                      f"{r['amp_ratio_pca_vs_rand']:6.3f}  "
                      f"{r['pca_pres_mean']:+8.4f}  {r['rand_pres_mean']:+8.4f}  "
                      f"{r['pres_diff_pca_vs_rand']:+6.4f}")
            
            # 逐PCA方向趋势
            print(f"\n  逐PCA方向放大系数 (前20):")
            for r in data_r:
                amps = r.get("pca_amplifications_top20", [])
                pres = r.get("pca_preservations_top20", [])
                if amps:
                    amp_str = " ".join([f"{a:.3f}" for a in amps[:10]])
                    pres_str = " ".join([f"{p:+.3f}" for p in pres[:10]])
                    print(f"  L{r['layer']:3d} amp: {amp_str}")
                    print(f"        pres: {pres_str}")
    
    # 释放模型
    release_model(model)
    print(f"\nCCIX {model_name} 完成!")


if __name__ == "__main__":
    main()
