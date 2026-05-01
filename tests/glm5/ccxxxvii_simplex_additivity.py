"""
CCXXXVII(337): 语义单纯形可加性 + 注意力头传播追踪 + 跨领域一致性
======================================================================
核心突破1: ★★★★★ 语义单纯形的可加性验证
  → 从3类开始, 逐一增加1类到10类, 观察维度是否+1
  → 验证: 新增类别是否引入新的正交方向?
  → 关键指标: Δn_sep (增加1类后n_separating_PCs的变化量)
  → 如果Δn_sep=1 → 单纯形可加! N-1维假设完全成立
  → 如果Δn_sep=0 → 新增类别被压缩到现有维度

核心突破2: ★★★★ 注意力头信息传播追踪
  → 对每个注意力头, 计算它从comp_pos到last_pos的"传播贡献"
  → 方法: 在每层, 对每个head计算:
    comp→last attention weight × comp_pos_diff → 传播信号
  → 找到: 哪些头是"比较信息传播头"?

核心突破3: ★★★ 跨领域一致性验证
  → 领域1: 颜色 (red/blue/green/yellow/purple/orange)
  → 领域2: 情感 (happy/sad/angry/scared/surprised/disgusted)
  → 领域3: 职业 (doctor/teacher/engineer/artist/lawyer/chef)
  → 验证: 不同领域是否也遵循N-1维单纯形?

实验设计:
  Part 1: 单纯形可加性 (★核心)
    在最佳语义分离层, 从3类逐一增加到10类
    对每个N, 计算: n_separating_PCs, regularity_score
    关键: Δn_seq = n_sep(N+1) - n_sep(N)
    
  Part 2: 注意力头传播追踪
    在信息传播关键层, 对每个注意力头:
    1. 计算: comp_pos对last_pos的attention weight
    2. 计算: V_com × O → 该头对last_pos残差的贡献
    3. 计算: 该贡献中bigger-smaller差异 → 传播信号
    
  Part 3: 跨领域一致性
    3个领域, 每领域6类, 每类10+词汇
    在最佳层做PCA+F-ratio → n_separating_PCs

用法:
  python ccxxxvii_simplex_additivity.py --model qwen3
  python ccxxxvii_simplex_additivity.py --model glm4
  python ccxxxvii_simplex_additivity.py --model deepseek7b
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
LOG = TEMP / "ccxxxvii_simplex_additivity_log.txt"

# ===== 语义类别定义 =====
# 领域1: Habitat (已知遵循N-1维单纯形)
HABITATS_ORDERED = ["land", "ocean", "sky", "space", "microscopic", "virtual", 
                     "underground", "arctic", "tropical", "freshwater"]

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
    "arctic": ["polar", "penguin", "walrus", "narwhal", "caribou",
               "musk", "snowy", "beluga", "puffin", "seal",
               "reindeer", "ptarmigan", "lemming", "husky", "fox"],
    "tropical": ["parrot", "toucan", "gorilla", "orangutan", "jaguar",
                 "anaconda", "piranha", "sloth", "iguana", "macaw",
                 "chimpanzee", "bamboo", "orchid", "mango", "papaya"],
    "freshwater": ["trout", "bass", "salmon", "catfish", "pike",
                   "perch", "carp", "minnow", "crayfish", "tadpole",
                   "frog", "newt", "turtle", "duck", "heron"],
}

# 领域2: 颜色
COLOR_DOMAIN = {
    "red": ["apple", "rose", "ruby", "cherry", "tomato", "flame", "crimson",
            "scarlet", "brick", "garnet", "maroon", "blood", "rust", "copper", "coral"],
    "blue": ["ocean", "sky", "sapphire", "azure", "navy", "cobalt", "indigo",
             "teal", "cyan", "turquoise", "aquamarine", "cerulean", "lapis", "periwinkle", "cornflower"],
    "green": ["grass", "emerald", "forest", "lime", "mint", "olive", "jade",
              "sage", "moss", "fern", "clover", "basil", "cucumber", "parsley", "matcha"],
    "yellow": ["sun", "gold", "lemon", "banana", "honey", "mustard", "amber",
               "canary", "daffodil", "butter", "saffron", "marigold", "corn", "wheat", "ochre"],
    "purple": ["violet", "lavender", "plum", "grape", "orchid", "amethyst", "magenta",
               "lilac", "mauve", "mulberry", "eggplant", "wisteria", "hyacinth", "iris", "raisin"],
    "orange": ["tangerine", "carrot", "pumpkin", "apricot", "peach", "mango", "cantaloupe",
               "salmon", "coral", "copper", "rust", "amber", "marigold", "papaya", "persimmon"],
}

# 领域3: 情感
EMOTION_DOMAIN = {
    "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation", "euphoria",
              "contentment", "pleasure", "gladness", "merriment", "jubilation", "rapture", "exhilaration", "joviality"],
    "sad": ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay", "woe",
            "anguish", "heartache", "mourning", "dejection", "forlorn", "mournful", "doleful", "lugubrious"],
    "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility", "indignation",
              "animosity", "vexation", "exasperation", "resentment", "bitterness", "animus", "choler", "temper"],
    "scared": ["fear", "terror", "dread", "panic", "fright", "horror", "alarm",
               "anxiety", "apprehension", "trepidation", "phobia", "dismay", "consternation", "awe", "timidity"],
    "surprised": ["astonishment", "amazement", "wonder", "shock", "startle", "stupefaction",
                  "disbelief", "incredulity", "awe", "bewilderment", "stunned", "dumbfounded", "flabbergasted", "thunderstruck", "dumbstruck"],
    "disgusted": ["revulsion", "repugnance", "nausea", "loathing", "abhorrence", "aversion",
                  "distaste", "antipathy", "repulsion", "odium", "dislike", "contempt", "scorn", "squeamishness", "repellent"],
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


def compute_n_separating_pcs(hab_resids, all_vecs, n_pc_extra=2):
    """计算n_separating_PCs"""
    arr = np.array(all_vecs)
    centered = arr - arr.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    n_habs = len(hab_resids)
    n_pc = min(n_habs + n_pc_extra, Vt.shape[0])
    
    hab_proj = {}
    for hab in hab_resids:
        hab_arr = np.array(hab_resids[hab])
        hab_centered = hab_arr - arr.mean(axis=0)
        pc_proj = hab_centered @ Vt[:n_pc].T
        hab_proj[hab] = pc_proj
    
    n_separating = 0
    separating_f_ratios = []
    for pc_i in range(n_pc):
        means = [np.mean(hab_proj[h][:, pc_i]) for h in hab_resids]
        within_vars = [np.var(hab_proj[h][:, pc_i]) for h in hab_resids]
        f_ratio = np.var(means) / max(np.mean(within_vars), 1e-10)
        separating_f_ratios.append(f_ratio)
        if f_ratio > 1.0:
            n_separating += 1
    
    return n_separating, separating_f_ratios[:n_pc]


def compute_simplex_geometry(hab_centers_dict):
    """计算单纯形几何指标"""
    hab_names = list(hab_centers_dict.keys())
    centers_arr = np.array([hab_centers_dict[h] for h in hab_names])
    n_habs = len(hab_names)
    
    if n_habs < 3:
        return None
    
    # 两两距离
    pairwise_dists = squareform(pdist(centers_arr))
    upper_tri = pairwise_dists[np.triu_indices(n_habs, k=1)]
    
    mean_dist = np.mean(upper_tri)
    std_dist = np.std(upper_tri)
    regularity_score = 1.0 - std_dist / max(mean_dist, 1e-10)
    
    # 顶点到质心
    centroid = np.mean(centers_arr, axis=0)
    vertex_radii = [np.linalg.norm(centers_arr[i] - centroid) for i in range(n_habs)]
    mean_radius = np.mean(vertex_radii)
    std_radius = np.std(vertex_radii)
    radius_uniformity = 1.0 - std_radius / max(mean_radius, 1e-10)
    
    # 角度
    angles = []
    for i in range(n_habs):
        for j in range(i + 1, n_habs):
            v1 = centers_arr[i] - centroid
            v2 = centers_arr[j] - centroid
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
    
    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    ideal_angle = np.arccos(-1.0 / n_habs) * 180 / np.pi if n_habs > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    angle_uniformity = 1.0 - std_angle / max(mean_angle, 1e-10) if mean_angle > 0 else 0
    
    return {
        "n_classes": n_habs,
        "regularity_score": round(float(regularity_score), 4),
        "radius_uniformity": round(float(radius_uniformity), 4),
        "mean_angle": round(float(mean_angle), 2),
        "ideal_angle": round(float(ideal_angle), 2),
        "angle_deviation": round(float(angle_deviation), 2),
        "angle_uniformity": round(float(angle_uniformity), 4),
    }


def collect_residuals_at_layer(model, tokenizer, layers, li, domain_dict, prompt_template, 
                                n_words=10, device="cuda"):
    """收集某层某领域的残差"""
    hab_resids = {}
    for hab, words in domain_dict.items():
        word_list = words[:n_words]
        resids = []
        for word in word_list:
            prompt = prompt_template.format(word=word)
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
    
    return hab_resids


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXVII(337): 单纯形可加性 + 注意力头追踪 + 跨领域 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # 确定最佳层
    best_layer_map = {"qwen3": 27, "glm4": 30, "deepseek7b": 14}
    best_layer = best_layer_map.get(model_name, n_layers // 2)
    if best_layer >= n_layers:
        best_layer = n_layers // 2
    
    # ====================================================================
    # Part 1: ★★★★★ 单纯形可加性验证
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: ★★★★★ 单纯形可加性验证")
    log("="*60)
    log(f"  最佳层: L{best_layer}")
    log(f"  从3类逐一增加到10类, 观察Δn_sep")
    
    # 测试多个层
    test_layers_add = sorted(set([
        max(0, best_layer - 6), max(0, best_layer - 3), best_layer, 
        min(n_layers-1, best_layer + 3), min(n_layers-1, best_layer + 6),
        n_layers // 3, n_layers - 1
    ]))
    test_layers_add = [l for l in test_layers_add if 0 <= l < n_layers]
    
    additivity_results = {}
    
    for li in test_layers_add:
        log(f"\n  --- Layer L{li} ---")
        
        # 收集10个habitat的残差
        hab_resids_full = collect_residuals_at_layer(
            model, tokenizer, layers, li, 
            {h: ALL_HABITATS[h] for h in HABITATS_ORDERED},
            "The {word} lives in the", n_words=10, device=device
        )
        
        valid_habs = [h for h in HABITATS_ORDERED if h in hab_resids_full]
        log(f"    Valid habitats: {len(valid_habs)}/{len(HABITATS_ORDERED)}")
        
        if len(valid_habs) < 3:
            log(f"    Skip: insufficient habitats")
            continue
        
        layer_additivity = {}
        prev_n_sep = 0
        
        for n_classes in range(3, len(valid_habs) + 1):
            current_habs = valid_habs[:n_classes]
            current_resids = {h: hab_resids_full[h] for h in current_habs}
            
            # 所有向量
            all_vecs = []
            for h in current_habs:
                all_vecs.extend(current_resids[h])
            
            n_sep, f_ratios = compute_n_separating_pcs(current_resids, all_vecs)
            delta_n_sep = n_sep - prev_n_sep
            
            # 计算几何(如果有足够维度)
            if n_sep >= 2:
                arr = np.array(all_vecs)
                centered = arr - arr.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                
                hab_centers = {}
                for h in current_habs:
                    hab_arr = np.array(current_resids[h])
                    hab_centered = hab_arr - arr.mean(axis=0)
                    pc_proj = hab_centered @ Vt[:n_sep].T
                    hab_centers[h] = np.mean(pc_proj, axis=0)
                
                geom = compute_simplex_geometry(hab_centers)
            else:
                geom = None
            
            log(f"    N={n_classes}: n_sep={n_sep} (expected N-1={n_classes-1}), "
                f"Δn_sep={delta_n_sep}, match={'✓' if n_sep == n_classes-1 else '✗'}"
                + (f", regularity={geom['regularity_score']:.3f}" if geom else ""))
            
            layer_additivity[f"N{n_classes}"] = {
                "n_classes": n_classes,
                "habitats": current_habs,
                "n_separating_PCs": n_sep,
                "expected_N_minus_1": n_classes - 1,
                "delta_n_sep": delta_n_sep,
                "match": n_sep == n_classes - 1,
                "top_f_ratios": [round(float(f), 2) for f in f_ratios[:n_classes+1]],
                "geometry": geom,
            }
            
            prev_n_sep = n_sep
        
        additivity_results[f"L{li}"] = layer_additivity
    
    results["additivity"] = additivity_results
    
    # 汇总可加性
    log("\n  === Additivity Summary ===")
    for layer_key, layer_data in additivity_results.items():
        log(f"  {layer_key}:")
        for n_key, n_data in sorted(layer_data.items()):
            log(f"    {n_key}: n_sep={n_data['n_separating_PCs']}, "
                f"Δ={n_data['delta_n_sep']}, "
                f"match={'✓' if n_data['match'] else '✗'}")
    
    # ====================================================================
    # Part 2: ★★★★ 注意力头信息传播追踪
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: ★★★★ 注意力头信息传播追踪")
    log("="*60)
    
    # 只在3个关键层测试: 早期, 中期, 晚期
    attn_test_layers = sorted(set([
        max(0, best_layer // 3),
        best_layer,
        min(n_layers - 1, best_layer + (n_layers - best_layer) * 2 // 3),
    ]))
    attn_test_layers = [l for l in attn_test_layers if 0 <= l < n_layers]
    
    log(f"  测试层: {attn_test_layers}")
    log(f"  测试对: {SIZE_PAIRS[:4]}")
    
    bigger_id = tokenizer.encode("bigger", add_special_tokens=False)[0]
    smaller_id = tokenizer.encode("smaller", add_special_tokens=False)[0]
    
    attention_prop_results = {}
    
    for big, small in SIZE_PAIRS[:4]:
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
        last_pos = toks_b.input_ids.shape[1] - 1
        
        log(f"\n  {big}/{small}: comp_pos={comp_pos}, last_pos={last_pos}")
        
        pair_attn_data = {}
        
        for li in attn_test_layers:
            layer = layers[li]
            # 获取n_heads - 从模型配置
            cfg = model.config
            n_heads = getattr(cfg, 'num_attention_heads', None) or getattr(cfg, 'num_heads', d_model)
            # Qwen3有GQA, W_o的shape是[d_model, n_kv_heads*d_head], 但d_head=d_model/n_heads
            # 所以d_head = d_model // n_heads
            d_head = d_model // n_heads
            
            # 收集注意力权重和值向量
            captured_attn = {}
            captured_values = {}
            
            def mk_attn_hook(k):
                def hook(m, inp, out):
                    # out = (hidden_states, attn_weights, past_kv)
                    if isinstance(out, tuple):
                        captured_attn[k] = out[1].detach().float().cpu() if len(out) > 1 else None
                    else:
                        captured_attn[k] = None
                return hook
            
            # 更可靠的方法: 使用forward_hook获取输出, 然后分析残差差异
            captured_out_b = {}
            captured_out_s = {}
            
            def mk_out_hook(key, storage):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    storage[key] = o.detach().clone()
                return hook
            
            # 方法1: 用attn_output hook获取注意力层的输出
            # 对于Qwen2/GLM4架构, self_attn的输出是 (hidden, attn_weights, ...)
            # 我们需要attn_weights
            
            # 方法2 (更直接): 分析每个head的O矩阵投影
            # 获取该层的W_o, W_v
            W_o = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()  # [d_model, d_model]
            W_v = layer.self_attn.v_proj.weight.detach().float().cpu().numpy()  # [d_model, d_model]
            
            # W_o: [d_model, n_heads * d_head], W_v: [n_heads * d_head, d_model]
            # 对于每个head h, W_o[:, h*d_head:(h+1)*d_head]是该head的输出投影
            # head_output = W_o[:, h*d_head:(h+1)*d_head] @ V_h @ x
            
            # 收集bigger和smaller在该层的残差
            h1 = layers[li].register_forward_hook(mk_out_hook("r", captured_out_b))
            with torch.no_grad():
                _ = model(**toks_b)
            h1.remove()
            
            h2 = layers[li].register_forward_hook(mk_out_hook("r", captured_out_s))
            with torch.no_grad():
                _ = model(**toks_s)
            h2.remove()
            
            if "r" not in captured_out_b or "r" not in captured_out_s:
                log(f"    L{li}: capture failed")
                continue
            
            res_b = captured_out_b["r"]  # [1, seq_len, d_model]
            res_s = captured_out_s["r"]
            
            # 总差异
            diff_comp = res_b[0, comp_pos, :].float().cpu().numpy() - res_s[0, comp_pos, :].float().cpu().numpy()
            diff_last = res_b[0, last_pos, :].float().cpu().numpy() - res_s[0, last_pos, :].float().cpu().numpy()
            
            # ★★★ 核心方法: 逐head分析W_O的列空间 ★★★
            # 对于每个head h, W_o的第h个块 = W_o[:, h*d_head:(h+1)*d_head]
            # 这个块将head h的d_head维输出投影到d_model维
            # 如果diff_last在W_o[:, h*d_head:(h+1)*d_head]的列空间中有投影,
            # 说明head h对last_pos的差异做出了贡献
            
            head_contributions = []
            for h in range(n_heads):
                # W_o的第h个块
                W_o_h = W_o[:, h * d_head:(h + 1) * d_head]  # [d_model, d_head]
                
                # diff_last在W_o_h列空间中的投影
                proj_coeffs = W_o_h.T @ diff_last  # [d_head]
                proj_energy = float(np.sum(proj_coeffs ** 2))
                diff_last_norm = float(np.linalg.norm(diff_last) ** 2)
                
                contribution_ratio = proj_energy / max(diff_last_norm, 1e-10)
                
                # diff_comp在W_o_h列空间中的投影
                proj_coeffs_comp = W_o_h.T @ diff_comp
                proj_energy_comp = float(np.sum(proj_coeffs_comp ** 2))
                diff_comp_norm = float(np.linalg.norm(diff_comp) ** 2)
                contribution_comp = proj_energy_comp / max(diff_comp_norm, 1e-10)
                
                head_contributions.append({
                    "head": h,
                    "last_contribution": round(contribution_ratio, 6),
                    "comp_contribution": round(contribution_comp, 6),
                    "proj_energy_last": round(proj_energy, 4),
                    "proj_energy_comp": round(proj_energy_comp, 4),
                })
            
            # 按last_contribution排序
            head_contributions.sort(key=lambda x: x["last_contribution"], reverse=True)
            
            log(f"    L{li}: Top-5 heads (by last_pos contribution):")
            for hc in head_contributions[:5]:
                log(f"      H{hc['head']:2d}: last_contrib={hc['last_contribution']:.6f}, "
                    f"comp_contrib={hc['comp_contribution']:.6f}")
            
            pair_attn_data[f"L{li}"] = {
                "n_heads": n_heads,
                "d_head": d_head,
                "comp_pos": comp_pos,
                "last_pos": last_pos,
                "head_contributions": head_contributions,
                "top5_heads": [hc["head"] for hc in head_contributions[:5]],
            }
            
            del captured_out_b, captured_out_s
            gc.collect()
        
        attention_prop_results[f"{big}_{small}"] = pair_attn_data
    
    # 汇总注意力传播
    log("\n  === Attention Propagation Summary ===")
    # 统计: 跨所有比较对, 哪些head最常出现在top-5
    head_frequency = {}
    for pair_key, pair_data in attention_prop_results.items():
        for layer_key, layer_data in pair_data.items():
            for hc in layer_data["head_contributions"][:5]:
                h_key = f"{layer_key}_H{hc['head']}"
                if h_key not in head_frequency:
                    head_frequency[h_key] = 0
                head_frequency[h_key] += 1
    
    sorted_heads = sorted(head_frequency.items(), key=lambda x: x[1], reverse=True)
    log(f"  Most frequent top-5 heads:")
    for h_key, freq in sorted_heads[:10]:
        log(f"    {h_key}: appears {freq} times")
    
    results["attention_propagation"] = attention_prop_results
    
    # ====================================================================
    # Part 3: ★★★ 跨领域一致性验证
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: ★★★ 跨领域一致性验证")
    log("="*60)
    
    domains = {
        "habitat": (ALL_HABITATS, "The {word} lives in the", 
                    ["land", "ocean", "sky", "space", "microscopic", "virtual"]),
        "color": (COLOR_DOMAIN, "The {word} is", 
                  ["red", "blue", "green", "yellow", "purple", "orange"]),
        "emotion": (EMOTION_DOMAIN, "The person felt {word} about the", 
                    ["happy", "sad", "angry", "scared", "surprised", "disgusted"]),
    }
    
    test_layers_cross = sorted(set([
        best_layer, n_layers // 2, n_layers // 3
    ]))
    test_layers_cross = [l for l in test_layers_cross if 0 <= l < n_layers]
    
    cross_domain_results = {}
    
    for domain_name, (domain_dict, prompt_template, domain_order) in domains.items():
        log(f"\n  --- Domain: {domain_name} ---")
        
        for li in test_layers_cross:
            hab_resids = collect_residuals_at_layer(
                model, tokenizer, layers, li,
                {h: domain_dict[h] for h in domain_order if h in domain_dict},
                prompt_template, n_words=10, device=device
            )
            
            valid_habs = [h for h in domain_order if h in hab_resids]
            
            if len(valid_habs) < 3:
                log(f"    L{li}: insufficient classes ({len(valid_habs)})")
                continue
            
            current_resids = {h: hab_resids[h] for h in valid_habs}
            all_vecs = []
            for h in valid_habs:
                all_vecs.extend(current_resids[h])
            
            n_sep, f_ratios = compute_n_separating_pcs(current_resids, all_vecs)
            n_classes = len(valid_habs)
            
            # 几何分析
            if n_sep >= 2:
                arr = np.array(all_vecs)
                centered = arr - arr.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                
                hab_centers = {}
                for h in valid_habs:
                    hab_arr = np.array(current_resids[h])
                    hab_centered = hab_arr - arr.mean(axis=0)
                    pc_proj = hab_centered @ Vt[:n_sep].T
                    hab_centers[h] = np.mean(pc_proj, axis=0)
                
                geom = compute_simplex_geometry(hab_centers)
            else:
                geom = None
            
            match = n_sep == n_classes - 1
            
            log(f"    L{li}: {domain_name} → N={n_classes}, n_sep={n_sep}, "
                f"expected(N-1)={n_classes-1}, match={'✓' if match else '✗'}"
                + (f", regularity={geom['regularity_score']:.3f}" if geom else "")
                + (f", angle_dev={geom['angle_deviation']:.1f}°" if geom else ""))
            
            key = f"{domain_name}_L{li}"
            cross_domain_results[key] = {
                "domain": domain_name,
                "layer": li,
                "n_classes": n_classes,
                "classes": valid_habs,
                "n_separating_PCs": n_sep,
                "expected_N_minus_1": n_classes - 1,
                "match": match,
                "geometry": geom,
                "top_f_ratios": [round(float(f), 2) for f in f_ratios[:n_classes+1]],
            }
    
    results["cross_domain"] = cross_domain_results
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    out_path = TEMP / f"ccxxxvii_simplex_additivity_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nResults saved to {out_path}")
    
    # ====================================================================
    # 最终汇总
    # ====================================================================
    log("\n" + "="*60)
    log("FINAL SUMMARY - CCXXXVII")
    log("="*60)
    
    # 可加性汇总
    log("\n--- Part 1: Additivity Summary ---")
    for layer_key, layer_data in additivity_results.items():
        deltas = []
        matches = []
        for n_key in sorted(layer_data.keys()):
            nd = layer_data[n_key]
            deltas.append(nd["delta_n_sep"])
            matches.append(nd["match"])
        log(f"  {layer_key}: Δn_sep={deltas}, matches={sum(matches)}/{len(matches)}")
    
    # 跨领域汇总
    log("\n--- Part 3: Cross-Domain Summary ---")
    for key, data in cross_domain_results.items():
        log(f"  {key}: N={data['n_classes']}, n_sep={data['n_separating_PCs']}, "
            f"match={'✓' if data['match'] else '✗'}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"CCXXXVII Log - {args.model}\n")
    
    run(args.model)
