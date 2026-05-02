"""
CCVIII(358): 架构组件消融实验
===================================
★★★★★ CCVII核心发现:
  - θ≈90°是维度诅咒, 与训练和语义都无关
  - β_emb受训练影响, 但方向出乎意料: 训练降低了β_emb!
  - 随机Transformer也给出β>0 → 架构自发保持几何
  - 训练"重塑"而非"增加"几何保持

★★★★★ 本实验目标:
  1. Layer-wise β_emb分析: trained vs random, 每层β如何变化?
  2. 组件消融: 零化attention/MLP, β如何变化?
  3. 分离残差连接贡献: y = x + F(x)中的"x"项贡献多少β?
  4. 训练vs随机: 哪个组件的训练差异最大?

核心洞察:
  h_{l+1} = h_l + Attn(LN(h_l)) + MLP(LN(h_l + Attn(LN(h_l))))
  - 残差连接使h_{l+1}包含h_l → 即使Attn/MLP是随机的, 也有β>0
  - Attn贡献: Δ_attn = Attn(LN(h_l))
  - MLP贡献: Δ_mlp = MLP(LN(h_l + Δ_attn))
  - 如果零化Attn: h_{l+1} = h_l + MLP(LN(h_l))  → β?
  - 如果零化MLP: h_{l+1} = h_l + Attn(LN(h_l))  → β?
  - 如果零化两者: h_{l+1} = h_l (纯残差)        → β=1.0!

用法:
  python ccviii_arch_ablation.py --model qwen3
  python ccviii_arch_ablation.py --model glm4
  python ccviii_arch_ablation.py --model deepseek7b
  python ccviii_arch_ablation.py --model qwen3 --analysis layerwise  # 只做layerwise分析
  python ccviii_arch_ablation.py --model qwen3 --analysis ablation   # 只做消融分析
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

# 领域定义 (复用CCVI)
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


def procrustes_align(X, Y):
    """Orthogonal Procrustes"""
    M = X.T @ Y
    U, sigma, Vt = svd(M, full_matrices=False)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Y_pred = X @ R
    error = np.sum((Y - Y_pred)**2) / np.sum(Y**2) if np.sum(Y**2) > 0 else 1.0
    return R, sigma, error


def compute_rotation_angle(R):
    """计算旋转矩阵的旋转角度"""
    K = R.shape[0]
    trace_val = np.trace(R)
    cos_angle = np.clip((trace_val - 1) / max(K - 1, 1), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg, float(trace_val)


def compute_pca(points):
    """PCA"""
    N, d = points.shape
    K = min(N - 1, d)
    mean = points.mean(axis=0)
    centered = points - mean
    U, S, Vt = svd(centered, full_matrices=False)
    scores = U[:, :K] * S[:K]
    return scores, Vt[:K, :], S[:K], mean, K


def compute_beta_emb(emb_matrix, res_matrix):
    """
    计算β_emb: embedding cosine距离 vs residual euclidean距离的相关性
    
    Args:
        emb_matrix: N×D 原始embedding矩阵
        res_matrix: N×D residual PCA scores矩阵
    
    Returns:
        beta_emb, r_dist, theta, N, K
    """
    N = emb_matrix.shape[0]
    
    # PCA for residual
    res_scores, _, _, _, K = compute_pca(res_matrix)
    
    # PCA for embedding
    emb_scores, _, _, _, _ = compute_pca(emb_matrix)
    
    # Procrustes
    emb_c = emb_scores - emb_scores.mean(axis=0)
    res_c = res_scores - res_scores.mean(axis=0)
    R, sigma, error = procrustes_align(emb_c, res_c)
    theta, _ = compute_rotation_angle(R)
    
    # β_emb: embedding cosine距离 vs residual euclidean距离
    norms_emb = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms_emb = np.maximum(norms_emb, 1e-10)
    emb_norm = emb_matrix / norms_emb
    cos_sim = emb_norm @ emb_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    
    res_dist = squareform(pdist(res_scores, metric='euclidean'))
    
    upper = np.triu_indices(N, k=1)
    beta_emb, _ = pearsonr(cos_dist[upper], res_dist[upper])
    
    # r_dist
    emb_dist = squareform(pdist(emb_c, metric='euclidean'))
    res_dist_c = squareform(pdist(res_c, metric='euclidean'))
    r_dist, _ = pearsonr(emb_dist[upper], res_dist_c[upper])
    
    return beta_emb, r_dist, theta, N, K


def get_category_embeddings(model, tokenizer, device, domain_dict):
    """获取类别代表词的embedding (每个类别取第一个词)"""
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    categories = list(domain_dict.keys())
    n_cats = len(categories)
    
    embeddings = np.zeros((n_cats, d_model))
    valid_cats = []
    
    for i, cat in enumerate(categories):
        words = domain_dict[cat]
        main_word = words[0]
        tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
        if len(tok_ids) == 0:
            continue
        
        # 取最后一个token的embedding
        tok_id = tok_ids[-1]
        with torch.no_grad():
            emb = embed_layer.weight[tok_id].detach().cpu().float().numpy()
        embeddings[i] = emb
        valid_cats.append(cat)
    
    return embeddings[:len(valid_cats)], valid_cats


def run_layerwise_analysis(model, tokenizer, device, model_name, domain_name, domain_dict):
    """
    Layer-wise β_emb分析: 逐层收集residual, 计算β_emb
    
    ★★★★★ 关键: 不仅收集层输出, 还收集attention输出和MLP输出
    """
    print(f"\n  === Layer-wise分析: {model_name}/{domain_name} ===")
    
    layers = get_layers(model)
    n_layers = len(layers)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    # 获取类别embeddings
    emb_matrix, valid_cats = get_category_embeddings(model, tokenizer, device, domain_dict)
    N = len(valid_cats)
    print(f"  N={N} categories, d_model={d_model}, n_layers={n_layers}")
    
    # 采样层 (每3层+首尾)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f"  采样层: {sample_layers}")
    
    # 对每个token做前向传播, 用hook收集各层的:
    # 1. 层输出 (after residual)
    # 2. attention输出 (after attention, before MLP residual)
    # 3. MLP输出 (after MLP)
    all_layer_outputs = {}    # {layer_idx: N×D}
    all_attn_outputs = {}     # {layer_idx: N×D}  (attention residual)
    all_mlp_outputs = {}      # {layer_idx: N×D}  (MLP residual)
    all_attn_deltas = {}      # {layer_idx: N×D}  (attention delta)
    all_mlp_deltas = {}       # {layer_idx: N×D}  (MLP delta)
    
    for li in sample_layers:
        all_layer_outputs[li] = []
        all_attn_outputs[li] = []
        all_mlp_outputs[li] = []
        all_attn_deltas[li] = []
        all_mlp_deltas[li] = []
    
    for cat_idx, cat in enumerate(valid_cats):
        main_word = domain_dict[cat][0]
        tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
        if len(tok_ids) == 0:
            continue
        tok_id = tok_ids[-1]
        
        # 构造输入: 单个token
        input_ids = torch.tensor([[tok_id]], device=device)
        
        # Hook收集
        captured = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in sample_layers:
            layer = layers[li]
            # 层输出hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
            # Attention输出hook (self_attn输出)
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(make_hook(f"A{li}")))
            # MLP输出hook
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_hook(f"M{li}")))
        
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
            
            if l_key in captured:
                # 层输出: [1, seq_len, d_model] → 取最后一个token
                layer_out = captured[l_key][0, -1, :].numpy()
                all_layer_outputs[li].append(layer_out)
            
            if a_key in captured:
                # Attention输出: [1, seq_len, d_model]
                attn_out = captured[a_key][0, -1, :].numpy()
                all_attn_outputs[li].append(attn_out)
            
            if m_key in captured:
                # MLP输出: [1, seq_len, d_model]
                mlp_out = captured[m_key][0, -1, :].numpy()
                all_mlp_outputs[li].append(mlp_out)
        
        # 清理
        del captured
        gc.collect()
    
    # 转换为numpy
    for li in sample_layers:
        if len(all_layer_outputs[li]) == N:
            all_layer_outputs[li] = np.array(all_layer_outputs[li])
        if len(all_attn_outputs[li]) == N:
            all_attn_outputs[li] = np.array(all_attn_outputs[li])
        if len(all_mlp_outputs[li]) == N:
            all_mlp_outputs[li] = np.array(all_mlp_outputs[li])
    
    # 计算各层的β_emb
    results = []
    
    for li in sample_layers:
        if li not in all_layer_outputs or not isinstance(all_layer_outputs[li], np.ndarray):
            continue
        
        layer_res = all_layer_outputs[li]
        if layer_res.shape[0] != N:
            continue
        
        # Full layer: emb → layer output
        beta_full, r_full, theta_full, _, K = compute_beta_emb(emb_matrix, layer_res)
        
        result = {
            "layer": li,
            "theta": float(theta_full),
            "beta_emb": float(beta_full),
            "r_dist": float(r_full),
            "N": N,
            "K": K,
        }
        
        # Attention contribution analysis
        if li in all_attn_outputs and isinstance(all_attn_outputs[li], np.ndarray):
            if all_attn_outputs[li].shape[0] == N:
                # Attention output是residual stream after attention
                # 但我们想看attention的增量贡献
                # Δ_attn = Attn(LN(h_l)) - 在residual stream中表现为加法
                # 实际上Attn hook输出的是self_attn模块的输出, 不是residual stream
                # 所以: h_after_attn = h_l + Attn(LN(h_l))
                # 我们需要h_l (前一层输出) 来计算增量
                
                # 如果是第0层, h_l = embedding
                if li == 0:
                    h_before = emb_matrix
                elif li - 1 in all_layer_outputs and isinstance(all_layer_outputs[li-1], np.ndarray):
                    h_before = all_layer_outputs[li-1]
                else:
                    h_before = None
                
                attn_out = all_attn_outputs[li]
                
                # β_attn: emb距离 vs attention输出距离
                beta_attn, r_attn, theta_attn, _, _ = compute_beta_emb(emb_matrix, attn_out)
                result["beta_attn_only"] = float(beta_attn)
                result["theta_attn_only"] = float(theta_attn)
                
                # Δ_attn分析: 如果有前一层输出
                if h_before is not None and h_before.shape[0] == N:
                    delta_attn = attn_out  # attention模块的输出就是Δ_attn
                    # 但这已经是LN(h_l)经过attention后的输出, 不是residual stream的增量
                    # 实际上attention hook给出的是QKV计算后的投影输出
                    
                    # β of delta_attn: emb距离 vs delta_attn距离
                    beta_delta_attn, _, _, _, _ = compute_beta_emb(emb_matrix, delta_attn)
                    result["beta_delta_attn"] = float(beta_delta_attn)
        
        # MLP contribution analysis
        if li in all_mlp_outputs and isinstance(all_mlp_outputs[li], np.ndarray):
            if all_mlp_outputs[li].shape[0] == N:
                mlp_out = all_mlp_outputs[li]
                
                beta_mlp, r_mlp, theta_mlp, _, _ = compute_beta_emb(emb_matrix, mlp_out)
                result["beta_mlp_only"] = float(beta_mlp)
                result["theta_mlp_only"] = float(theta_mlp)
        
        results.append(result)
        
        print(f"  L{li:2d}: θ={theta_full:.1f}°, β_emb={beta_full:+.3f}, "
              f"r_dist={r_full:+.3f}", end="")
        if "beta_attn_only" in result:
            print(f", β_attn={result['beta_attn_only']:+.3f}", end="")
        if "beta_mlp_only" in result:
            print(f", β_mlp={result['beta_mlp_only']:+.3f}", end="")
        print()
    
    return results, emb_matrix, all_layer_outputs


def run_ablation_experiment(model, tokenizer, device, model_name, domain_name, domain_dict):
    """
    组件消融实验: 零化attention/MLP, 观察β_emb变化
    
    ★★★★★ 核心方法:
    用hook在forward过程中拦截组件输出并替换为零
    """
    print(f"\n  === 组件消融实验: {model_name}/{domain_name} ===")
    
    layers = get_layers(model)
    n_layers = len(layers)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    # 获取类别embeddings
    emb_matrix, valid_cats = get_category_embeddings(model, tokenizer, device, domain_dict)
    N = len(valid_cats)
    
    # 只测试几个关键层
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = sorted(set([max(0, min(l, n_layers-1)) for l in test_layers]))
    print(f"  测试层: {test_layers}")
    
    # 消融条件
    ablation_conditions = [
        ("full", None, None),                    # 完整模型
        ("no_attn", "attn", None),               # 零化attention
        ("no_mlp", None, "mlp"),                 # 零化MLP
        ("no_both", "attn", "mlp"),              # 零化两者 (纯残差)
    ]
    
    results = {}
    
    for cond_name, zero_attn, zero_mlp in ablation_conditions:
        print(f"\n  --- 条件: {cond_name} ---")
        layer_residues = {li: [] for li in test_layers}
        
        for cat_idx, cat in enumerate(valid_cats):
            main_word = domain_dict[cat][0]
            tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
            if len(tok_ids) == 0:
                continue
            tok_id = tok_ids[-1]
            
            input_ids = torch.tensor([[tok_id]], device=device)
            
            # Hook: 收集层输出 + 零化指定组件
            captured = {}
            
            def make_capture_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            def make_zero_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        # 返回与输出相同形状的零张量
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return hook
            
            hooks = []
            
            for li in test_layers:
                layer = layers[li]
                # 层输出hook
                hooks.append(layer.register_forward_hook(make_capture_hook(f"L{li}")))
                
                # 零化attention
                if zero_attn == "attn" and hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(make_zero_hook()))
                
                # 零化MLP
                if zero_mlp == "mlp" and hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(make_zero_hook()))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids)
                except Exception as e:
                    print(f"  Forward failed: {e}")
            
            for h in hooks:
                h.remove()
            
            # 提取结果
            for li in test_layers:
                l_key = f"L{li}"
                if l_key in captured:
                    layer_out = captured[l_key][0, -1, :].numpy()
                    layer_residues[li].append(layer_out)
            
            del captured
            gc.collect()
        
        # 计算各层的β_emb
        condition_results = []
        
        for li in test_layers:
            if len(layer_residues[li]) != N:
                continue
            
            res_matrix = np.array(layer_residues[li])
            beta, r, theta, _, K = compute_beta_emb(emb_matrix, res_matrix)
            
            condition_results.append({
                "layer": li,
                "theta": float(theta),
                "beta_emb": float(beta),
                "r_dist": float(r),
            })
            
            print(f"  L{li:2d}: θ={theta:.1f}°, β_emb={beta:+.3f}, r_dist={r:+.3f}")
        
        results[cond_name] = condition_results
    
    return results


def run_residual_contribution_analysis(model, tokenizer, device, model_name, domain_name, domain_dict):
    """
    ★★★★★ 残差连接贡献分析
    
    核心思路:
    h_{l+1} = h_l + Δ_attn + Δ_mlp
    
    其中:
    - h_l: 来自残差连接, 完全保持前一层
    - Δ_attn: attention的贡献
    - Δ_mlp: MLP的贡献
    
    我们可以计算:
    1. β(h_l → h_{l+1}): 完整层变换
    2. β(h_l → h_l): 纯残差 (=1.0, 完美保持)
    3. β(h_l → Δ_attn): attention增量的几何保持
    4. β(h_l → Δ_mlp): MLP增量的几何保持
    
    更精确地: 
    Δ_attn + Δ_mlp 是"新信息", 它保持了多少emb几何?
    """
    print(f"\n  === 残差连接贡献分析: {model_name}/{domain_name} ===")
    
    layers = get_layers(model)
    n_layers = len(layers)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    emb_matrix, valid_cats = get_category_embeddings(model, tokenizer, device, domain_dict)
    N = len(valid_cats)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    # 收集层输出 (连续两层, 以计算增量)
    all_outputs = {li: [] for li in sample_layers}
    # 也收集embedding作为"第-1层"
    all_emb = []
    
    for cat_idx, cat in enumerate(valid_cats):
        main_word = domain_dict[cat][0]
        tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
        if len(tok_ids) == 0:
            continue
        tok_id = tok_ids[-1]
        
        input_ids = torch.tensor([[tok_id]], device=device)
        
        captured = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in sample_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                print(f"  Forward failed: {e}")
        
        for h in hooks:
            h.remove()
        
        # 提取embedding
        with torch.no_grad():
            emb = embed_layer(torch.tensor([[tok_id]], device=device))
            all_emb.append(emb[0, -1, :].detach().float().cpu().numpy())
        
        # 提取层输出
        for li in sample_layers:
            l_key = f"L{li}"
            if l_key in captured:
                all_outputs[li].append(captured[l_key][0, -1, :].numpy())
        
        del captured
        gc.collect()
    
    emb_np = np.array(all_emb)  # N×D
    
    # 计算各层的增量分析
    results = []
    prev_output = emb_np  # "第-1层" = embedding
    
    for li in sample_layers:
        if len(all_outputs[li]) != N:
            continue
        
        curr_output = np.array(all_outputs[li])  # N×D
        delta = curr_output - prev_output  # 增量 Δ = h_{l+1} - h_l
        
        # 1. Full: emb → curr_output
        beta_full, r_full, theta_full, _, K = compute_beta_emb(emb_matrix, curr_output)
        
        # 2. Delta: emb → Δ (增量)
        beta_delta, r_delta, theta_delta, _, _ = compute_beta_emb(emb_matrix, delta)
        
        # 3. 残差比例: ||Δ||² / ||curr_output||²
        delta_norm_sq = np.sum(delta**2)
        curr_norm_sq = np.sum(curr_output**2)
        residual_ratio = 1.0 - delta_norm_sq / max(curr_norm_sq, 1e-10)  # 残差贡献比例
        
        # 4. 增量与embedding的相关性
        # 如果增量主要来自embedding几何, 那么增量的方向与emb相关
        # 用cosine similarity: Δ_i · emb_i (逐token)
        cos_sim_delta_emb = []
        for i in range(N):
            d_norm = np.linalg.norm(delta[i])
            e_norm = np.linalg.norm(emb_matrix[i])
            if d_norm > 1e-10 and e_norm > 1e-10:
                cos = np.dot(delta[i], emb_matrix[i]) / (d_norm * e_norm)
                cos_sim_delta_emb.append(cos)
        mean_cos_delta_emb = np.mean(cos_sim_delta_emb) if cos_sim_delta_emb else 0.0
        
        result = {
            "layer": li,
            "theta_full": float(theta_full),
            "beta_full": float(beta_full),
            "beta_delta": float(beta_delta),
            "theta_delta": float(theta_delta),
            "r_dist_full": float(r_full),
            "r_dist_delta": float(r_delta),
            "residual_ratio": float(residual_ratio),
            "delta_norm_ratio": float(1.0 - residual_ratio),
            "mean_cos_delta_emb": float(mean_cos_delta_emb),
        }
        results.append(result)
        
        print(f"  L{li:2d}: β_full={beta_full:+.3f}, β_Δ={beta_delta:+.3f}, "
              f"残差比={residual_ratio:.3f}, cos(Δ,emb)={mean_cos_delta_emb:+.3f}")
        
        prev_output = curr_output
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "layerwise", "ablation", "residual"])
    args = parser.parse_args()
    
    model_name = args.model
    analysis = args.analysis
    
    print(f"\n{'='*70}")
    print(f"CCVIII: 架构组件消融实验 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    all_results = {"model": model_name, "d_model": model_info.d_model, "n_layers": model_info.n_layers}
    
    domains = [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]
    
    # Part 1: Layer-wise分析
    if analysis in ["all", "layerwise"]:
        print(f"\n{'='*70}")
        print(f"Part 1: Layer-wise β_emb分析")
        print(f"{'='*70}")
        
        for domain_name, domain_dict in domains:
            layerwise_results, emb_matrix, layer_outputs = run_layerwise_analysis(
                model, tokenizer, device, model_name, domain_name, domain_dict
            )
            all_results[f"layerwise_{domain_name}"] = layerwise_results
    
    # Part 2: 组件消融
    if analysis in ["all", "ablation"]:
        print(f"\n{'='*70}")
        print(f"Part 2: 组件消融实验")
        print(f"{'='*70}")
        
        for domain_name, domain_dict in domains:
            ablation_results = run_ablation_experiment(
                model, tokenizer, device, model_name, domain_name, domain_dict
            )
            all_results[f"ablation_{domain_name}"] = ablation_results
    
    # Part 3: 残差连接贡献分析
    if analysis in ["all", "residual"]:
        print(f"\n{'='*70}")
        print(f"Part 3: 残差连接贡献分析")
        print(f"{'='*70}")
        
        for domain_name, domain_dict in domains:
            residual_results = run_residual_contribution_analysis(
                model, tokenizer, device, model_name, domain_name, domain_dict
            )
            all_results[f"residual_{domain_name}"] = residual_results
    
    # 保存结果
    output_path = TEMP / f"ccviii_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    
    # ===== 汇总分析 =====
    print(f"\n{'='*70}")
    print(f"汇总分析: {model_name}")
    print(f"{'='*70}")
    
    # Layer-wise β_emb趋势
    for domain_name, _ in domains:
        key = f"layerwise_{domain_name}"
        if key in all_results:
            data = all_results[key]
            print(f"\n--- {domain_name}: β_emb layer-wise趋势 ---")
            for r in data:
                bar = "█" * max(0, int((r['beta_emb'] + 0.5) * 20))
                extra = ""
                if 'beta_attn_only' in r:
                    extra += f", β_attn={r['beta_attn_only']:+.3f}"
                if 'beta_mlp_only' in r:
                    extra += f", β_mlp={r['beta_mlp_only']:+.3f}"
                print(f"  L{r['layer']:2d}: β={r['beta_emb']:+.3f} {bar}{extra}")
    
    # 消融对比
    for domain_name, _ in domains:
        key = f"ablation_{domain_name}"
        if key in all_results:
            data = all_results[key]
            print(f"\n--- {domain_name}: 消融对比 ---")
            for cond, results in data.items():
                if results:
                    avg_beta = np.mean([r['beta_emb'] for r in results])
                    print(f"  {cond:8s}: avg β_emb={avg_beta:+.3f}")
    
    # 残差贡献
    for domain_name, _ in domains:
        key = f"residual_{domain_name}"
        if key in all_results:
            data = all_results[key]
            print(f"\n--- {domain_name}: 残差连接贡献 ---")
            for r in data:
                print(f"  L{r['layer']:2d}: β_full={r['beta_full']:+.3f}, "
                      f"β_Δ={r['beta_delta']:+.3f}, "
                      f"残差比={r['residual_ratio']:.3f}, "
                      f"cos(Δ,emb)={r['mean_cos_delta_emb']:+.3f}")
    
    # 释放模型
    release_model(model)
    print(f"\nCCVIII {model_name} 完成!")


if __name__ == "__main__":
    main()
