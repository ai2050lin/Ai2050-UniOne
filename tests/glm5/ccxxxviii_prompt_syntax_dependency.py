"""
CCXXXVIII(338): Prompt-语法依赖性验证 + 情感领域可加性 + 跨领域维度上限
=========================================================================
★★★★★ 核心假设: N-1维单纯形需要"领域+语法"的双重条件

CCXXXVII发现: 颜色n_sep=0! 三个模型完全一致!
  → 假设: "The red is"语法不自然, 模型无法提取颜色语义
  → 验证: 修正prompt后颜色是否也有N-1维单纯形?

Part 1: ★★★★★ 颜色领域多prompt对比
  → Prompt A: "The {word} is" (原始, 语法不自然 - 颜色词是形容词)
  → Prompt B: "The color {word} is very" (语法自然 - 明确颜色语义)
  → Prompt C: "{Word} is a bright color" (颜色作为属性名词)
  → Prompt D: "I like the color {word}" (主观体验句)
  → 验证: prompt修正后n_sep是否从0变为N-1?

Part 2: ★★★★★ 情感领域可加性验证
  → 从3类情感开始逐增到6类
  → 3类: happy/sad/angry
  → 4类: +scared
  → 5类: +surprised
  → 6类: +disgusted
  → 验证: Δn_sep=1是否持续成立?

Part 3: ★★★★ 多领域多prompt系统性测试
  → 每个领域用2-3种prompt模板
  → habitat: "The {word} lives in the" / "{Word} is found in the"
  → emotion: "The person felt {word}" / "{Word} overwhelmed the person"
  → occupation: "The {word} works at the" / "{Word} is a skilled professional"
  → color: "The color {word} is" / "I see the color {word}"
  → 验证: 语法自然性是否是语义分离的必要条件?

用法:
  python ccxxxviii_prompt_syntax_dependency.py --model qwen3
  python ccxxxviii_prompt_syntax_dependency.py --model glm4
  python ccxxxviii_prompt_syntax_dependency.py --model deepseek7b
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
LOG = TEMP / "ccxxxviii_prompt_syntax_log.txt"

# ===== 语义类别定义 =====
# 颜色领域 - 6类 (CCXXXVII中n_sep=0, 现在用多prompt测试)
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
COLOR_ORDER = ["red", "blue", "green", "yellow", "purple", "orange"]

# 情感领域 - 6类
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
EMOTION_ORDER = ["happy", "sad", "angry", "scared", "surprised", "disgusted"]

# Habitat领域 - 6类 (对比基线)
HABITAT_DOMAIN = {
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
}
HABITAT_ORDER = ["land", "ocean", "sky", "space", "microscopic", "virtual"]

# 职业领域 - 6类 (新领域测试)
OCCUPATION_DOMAIN = {
    "doctor": ["surgeon", "physician", "nurse", "therapist", "pediatrician", "cardiologist",
               "dermatologist", "neurologist", "psychiatrist", "oncologist", "radiologist",
               "anesthesiologist", "pathologist", "pharmacist", "clinician"],
    "teacher": ["professor", "instructor", "educator", "tutor", "lecturer", "mentor",
                "coach", "trainer", "academic", "scholar", "counselor", "advisor",
                "principal", "headmaster", "faculty"],
    "engineer": ["architect", "designer", "developer", "programmer", "mechanic", "technician",
                 "builder", "constructor", "inventor", "fabricator", "planner", "analyst",
                 "consultant", "inspector", "supervisor"],
    "artist": ["painter", "sculptor", "musician", "dancer", "actor", "singer",
               "poet", "writer", "composer", "illustrator", "photographer", "filmmaker",
               "performer", "ceramicist", "printmaker"],
    "lawyer": ["attorney", "advocate", "counsel", "barrister", "solicitor", "prosecutor",
               "defender", "judge", "magistrate", "paralegal", "litigator", "mediator",
               "arbitrator", "notary", "clerk"],
    "chef": ["cook", "baker", "pastry", "butcher", "sous", "line",
             "prep", "saucier", "grillardin", "poissonnier", "garde", "tournant",
             "sommelier", "caterer", "restaurateur"],
}
OCCUPATION_ORDER = ["doctor", "teacher", "engineer", "artist", "lawyer", "chef"]

# ===== Prompt模板定义 =====
# 每个领域多种prompt, 从不自然到自然
PROMPTS = {
    "color": {
        "A_unnatural": "The {word} is",  # 原始CCXXXVII - n_sep=0
        "B_color_explicit": "The color {word} is very",  # 明确颜色语义
        "C_color_attribute": "I see the color {word} everywhere",  # 颜色作为感知对象
        "D_color_subjective": "My favorite color is {word} because",  # 主观体验
    },
    "emotion": {
        "A_felt": "The person felt {word} about the",  # CCXXXVII用过的
        "B_overwhelmed": "{Word} overwhelmed the person completely",  # 情感作为主语
        "C_experienced": "She experienced intense {word} during the",  # 情感作为宾语
    },
    "habitat": {
        "A_lives": "The {word} lives in the",  # CCXXXVII用过的
        "B_found": "{Word} is found in the",  # habitat作为地点属性
    },
    "occupation": {
        "A_works": "The {word} works at the",  # 职业作为主语
        "B_skilled": "{Word} is a skilled professional who",  # 职业作为专有名词
    },
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_n_separating_pcs(class_resids, all_vecs, n_pc_extra=2):
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
    
    return n_separating, separating_f_ratios[:n_pc]


def compute_simplex_geometry(class_centers_dict):
    """计算单纯形几何指标"""
    class_names = list(class_centers_dict.keys())
    centers_arr = np.array([class_centers_dict[c] for c in class_names])
    n_cls = len(class_names)
    
    if n_cls < 3:
        return None
    
    pairwise_dists = squareform(pdist(centers_arr))
    upper_tri = pairwise_dists[np.triu_indices(n_cls, k=1)]
    
    mean_dist = np.mean(upper_tri)
    std_dist = np.std(upper_tri)
    regularity_score = 1.0 - std_dist / max(mean_dist, 1e-10)
    
    centroid = np.mean(centers_arr, axis=0)
    vertex_radii = [np.linalg.norm(centers_arr[i] - centroid) for i in range(n_cls)]
    mean_radius = np.mean(vertex_radii)
    std_radius = np.std(vertex_radii)
    radius_uniformity = 1.0 - std_radius / max(mean_radius, 1e-10)
    
    angles = []
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            v1 = centers_arr[i] - centroid
            v2 = centers_arr[j] - centroid
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
    
    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    ideal_angle = np.arccos(-1.0 / n_cls) * 180 / np.pi if n_cls > 1 else 0
    angle_deviation = abs(mean_angle - ideal_angle)
    angle_uniformity = 1.0 - std_angle / max(mean_angle, 1e-10) if mean_angle > 0 else 0
    
    return {
        "n_classes": n_cls,
        "regularity_score": round(float(regularity_score), 4),
        "radius_uniformity": round(float(radius_uniformity), 4),
        "mean_angle": round(float(mean_angle), 2),
        "ideal_angle": round(float(ideal_angle), 2),
        "angle_deviation": round(float(angle_deviation), 2),
        "angle_uniformity": round(float(angle_uniformity), 4),
    }


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


def analyze_domain_at_layer(model, tokenizer, layers, li, domain_dict, domain_order,
                            prompt_template, n_words=10, device="cuda"):
    """分析某领域在某层的语义分离"""
    class_resids = collect_residuals_at_layer(
        model, tokenizer, layers, li, domain_dict, prompt_template, n_words, device
    )
    
    valid_classes = [c for c in domain_order if c in class_resids]
    if len(valid_classes) < 3:
        return None
    
    current_resids = {c: class_resids[c] for c in valid_classes}
    all_vecs = []
    for c in valid_classes:
        all_vecs.extend(current_resids[c])
    
    n_sep, f_ratios = compute_n_separating_pcs(current_resids, all_vecs)
    n_classes = len(valid_classes)
    
    # 几何分析
    geom = None
    if n_sep >= 2:
        arr = np.array(all_vecs)
        centered = arr - arr.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        
        class_centers = {}
        for c in valid_classes:
            cls_arr = np.array(current_resids[c])
            cls_centered = cls_arr - arr.mean(axis=0)
            pc_proj = cls_centered @ Vt[:n_sep].T
            class_centers[c] = np.mean(pc_proj, axis=0)
        
        geom = compute_simplex_geometry(class_centers)
    
    return {
        "n_classes": n_classes,
        "classes": valid_classes,
        "n_separating_PCs": n_sep,
        "expected_N_minus_1": n_classes - 1,
        "match": n_sep == n_classes - 1,
        "geometry": geom,
        "top_f_ratios": [round(float(f), 2) for f in f_ratios[:n_classes + 1]],
    }


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}")
    log(f"CCXXXVIII(338): Prompt-语法依赖性 + 情感可加性 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # 确定测试层 - 从之前的实验知道最佳层
    best_layer_map = {"qwen3": 27, "glm4": 30, "deepseek7b": 14}
    best_layer = best_layer_map.get(model_name, n_layers // 2)
    if best_layer >= n_layers:
        best_layer = n_layers // 2
    
    # 测试多个层以获得完整图景
    test_layers = sorted(set([
        max(0, best_layer - 6), max(0, best_layer - 3), best_layer,
        min(n_layers - 1, best_layer + 3), min(n_layers - 1, best_layer + 6),
    ]))
    test_layers = [l for l in test_layers if 0 <= l < n_layers]
    
    log(f"  测试层: {test_layers}")
    log(f"  最佳层: L{best_layer}")
    
    # ====================================================================
    # Part 1: ★★★★★ 颜色领域多prompt对比 — 核心突破!
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: ★★★★★ 颜色领域多prompt对比")
    log("  验证: 语法不自然(The red is)导致n_sep=0")
    log("  修正prompt后颜色是否恢复N-1维单纯形?")
    log("="*60)
    
    color_prompt_results = {}
    
    for prompt_name, prompt_template in PROMPTS["color"].items():
        log(f"\n  --- Color Prompt: {prompt_name} ---")
        log(f"    Template: '{prompt_template}'")
        
        for li in test_layers:
            result = analyze_domain_at_layer(
                model, tokenizer, layers, li,
                COLOR_DOMAIN, COLOR_ORDER,
                prompt_template, n_words=10, device=device
            )
            
            if result is None:
                log(f"    L{li}: insufficient classes")
                continue
            
            match_str = "✓" if result["match"] else "✗"
            geom_str = ""
            if result["geometry"]:
                geom_str = f", reg={result['geometry']['regularity_score']:.3f}, angle_dev={result['geometry']['angle_deviation']:.1f}°"
            
            log(f"    L{li}: N={result['n_classes']}, n_sep={result['n_separating_PCs']}, "
                f"expected={result['expected_N_minus_1']}, match={match_str}{geom_str}")
            
            key = f"{prompt_name}_L{li}"
            color_prompt_results[key] = {
                "prompt_name": prompt_name,
                "prompt_template": prompt_template,
                "layer": li,
                **result,
            }
    
    results["color_prompts"] = color_prompt_results
    
    # ====================================================================
    # Part 2: ★★★★★ 情感领域可加性验证
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: ★★★★★ 情感领域可加性验证")
    log("  从3类→6类, 观察Δn_sep是否=1")
    log("="*60)
    
    emotion_prompt_results = {}
    emotion_additivity = {}
    
    # 用最佳prompt模板
    emotion_template = "The person felt {word} about the"
    
    for li in test_layers:
        log(f"\n  --- Layer L{li} ---")
        
        # 先收集所有6类情感的残差
        full_resids = collect_residuals_at_layer(
            model, tokenizer, layers, li,
            EMOTION_DOMAIN, emotion_template,
            n_words=10, device=device
        )
        
        valid_emotions = [e for e in EMOTION_ORDER if e in full_resids]
        log(f"    Valid emotions: {len(valid_emotions)}/{len(EMOTION_ORDER)}")
        
        if len(valid_emotions) < 3:
            log(f"    Skip: insufficient emotions")
            continue
        
        layer_additivity = {}
        prev_n_sep = 0
        
        for n_classes in range(3, len(valid_emotions) + 1):
            current_emotions = valid_emotions[:n_classes]
            current_resids = {e: full_resids[e] for e in current_emotions}
            
            all_vecs = []
            for e in current_emotions:
                all_vecs.extend(current_resids[e])
            
            n_sep, f_ratios = compute_n_separating_pcs(current_resids, all_vecs)
            delta_n_sep = n_sep - prev_n_sep
            
            geom = None
            if n_sep >= 2:
                arr = np.array(all_vecs)
                centered = arr - arr.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                
                cls_centers = {}
                for e in current_emotions:
                    cls_arr = np.array(current_resids[e])
                    cls_centered = cls_arr - arr.mean(axis=0)
                    pc_proj = cls_centered @ Vt[:n_sep].T
                    cls_centers[e] = np.mean(pc_proj, axis=0)
                
                geom = compute_simplex_geometry(cls_centers)
            
            match_str = "✓" if n_sep == n_classes - 1 else "✗"
            geom_str = f", reg={geom['regularity_score']:.3f}" if geom else ""
            
            log(f"    N={n_classes}: n_sep={n_sep} (expected {n_classes-1}), "
                f"Δn_sep={delta_n_sep}, match={match_str}{geom_str}")
            
            layer_additivity[f"N{n_classes}"] = {
                "n_classes": n_classes,
                "emotions": current_emotions,
                "n_separating_PCs": n_sep,
                "expected_N_minus_1": n_classes - 1,
                "delta_n_sep": delta_n_sep,
                "match": n_sep == n_classes - 1,
                "geometry": geom,
            }
            
            prev_n_sep = n_sep
        
        emotion_additivity[f"L{li}"] = layer_additivity
    
    results["emotion_additivity"] = emotion_additivity
    
    # ====================================================================
    # Part 3: ★★★★ 多领域多prompt系统性测试
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: ★★★★ 多领域多prompt系统性测试")
    log("  每个领域用2-3种prompt, 验证语法依赖")
    log("="*60)
    
    all_domains = {
        "habitat": (HABITAT_DOMAIN, HABITAT_ORDER),
        "emotion": (EMOTION_DOMAIN, EMOTION_ORDER),
        "occupation": (OCCUPATION_DOMAIN, OCCUPATION_ORDER),
        "color": (COLOR_DOMAIN, COLOR_ORDER),
    }
    
    # 只在3个关键层测试
    key_layers = sorted(set([
        max(0, best_layer - 3), best_layer,
        min(n_layers - 1, best_layer + 3),
    ]))
    key_layers = [l for l in key_layers if 0 <= l < n_layers]
    
    log(f"  关键层: {key_layers}")
    
    multi_domain_results = {}
    
    for domain_name, (domain_dict, domain_order) in all_domains.items():
        log(f"\n  --- Domain: {domain_name} ---")
        
        for prompt_name, prompt_template in PROMPTS.get(domain_name, {}).items():
            log(f"    Prompt: {prompt_name} = '{prompt_template}'")
            
            for li in key_layers:
                result = analyze_domain_at_layer(
                    model, tokenizer, layers, li,
                    domain_dict, domain_order,
                    prompt_template, n_words=10, device=device
                )
                
                if result is None:
                    log(f"      L{li}: insufficient classes")
                    continue
                
                match_str = "✓" if result["match"] else "✗"
                geom_str = ""
                if result["geometry"]:
                    geom_str = f", reg={result['geometry']['regularity_score']:.3f}"
                
                log(f"      L{li}: N={result['n_classes']}, n_sep={result['n_separating_PCs']}, "
                    f"match={match_str}{geom_str}")
                
                key = f"{domain_name}_{prompt_name}_L{li}"
                multi_domain_results[key] = {
                    "domain": domain_name,
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template,
                    "layer": li,
                    **result,
                }
    
    results["multi_domain"] = multi_domain_results
    
    # ====================================================================
    # Part 4: ★★★ 颜色领域可加性验证 (如果prompt修正成功)
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: ★★★ 颜色领域可加性验证 (最佳prompt)")
    log("="*60)
    
    # 找到颜色领域n_sep最高的prompt
    best_color_prompt = None
    best_color_nsep = 0
    for key, data in color_prompt_results.items():
        if data.get("n_separating_PCs", 0) > best_color_nsep:
            best_color_nsep = data["n_separating_PCs"]
            best_color_prompt = data["prompt_template"]
    
    if best_color_prompt and best_color_nsep >= 2:
        log(f"  最佳颜色prompt: '{best_color_prompt}' (n_sep={best_color_nsep})")
        log(f"  验证颜色领域可加性...")
        
        color_additivity = {}
        
        for li in [best_layer]:  # 只在最佳层
            full_resids = collect_residuals_at_layer(
                model, tokenizer, layers, li,
                COLOR_DOMAIN, best_color_prompt,
                n_words=10, device=device
            )
            
            valid_colors = [c for c in COLOR_ORDER if c in full_resids]
            log(f"    Valid colors: {len(valid_colors)}/{len(COLOR_ORDER)}")
            
            if len(valid_colors) < 3:
                log(f"    Skip: insufficient colors")
                continue
            
            layer_additivity = {}
            prev_n_sep = 0
            
            for n_classes in range(3, len(valid_colors) + 1):
                current_colors = valid_colors[:n_classes]
                current_resids = {c: full_resids[c] for c in current_colors}
                
                all_vecs = []
                for c in current_colors:
                    all_vecs.extend(current_resids[c])
                
                n_sep, f_ratios = compute_n_separating_pcs(current_resids, all_vecs)
                delta_n_sep = n_sep - prev_n_sep
                
                geom = None
                if n_sep >= 2:
                    arr = np.array(all_vecs)
                    centered = arr - arr.mean(axis=0)
                    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                    
                    cls_centers = {}
                    for c in current_colors:
                        cls_arr = np.array(current_resids[c])
                        cls_centered = cls_arr - arr.mean(axis=0)
                        pc_proj = cls_centered @ Vt[:n_sep].T
                        cls_centers[c] = np.mean(pc_proj, axis=0)
                    
                    geom = compute_simplex_geometry(cls_centers)
                
                match_str = "✓" if n_sep == n_classes - 1 else "✗"
                geom_str = f", reg={geom['regularity_score']:.3f}" if geom else ""
                
                log(f"    N={n_classes}: n_sep={n_sep} (expected {n_classes-1}), "
                    f"Δn_sep={delta_n_sep}, match={match_str}{geom_str}")
                
                layer_additivity[f"N{n_classes}"] = {
                    "n_classes": n_classes,
                    "colors": current_colors,
                    "n_separating_PCs": n_sep,
                    "expected_N_minus_1": n_classes - 1,
                    "delta_n_sep": delta_n_sep,
                    "match": n_sep == n_classes - 1,
                    "geometry": geom,
                }
                
                prev_n_sep = n_sep
            
            color_additivity[f"L{li}"] = layer_additivity
        
        results["color_additivity"] = color_additivity
    else:
        log(f"  颜色领域n_sep最高仅{best_color_nsep}, 无法测试可加性")
        log(f"  → 语法修正仍不足以恢复颜色语义分离!")
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    out_path = TEMP / f"ccxxxviii_prompt_syntax_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nResults saved to {out_path}")
    
    # ====================================================================
    # 最终汇总
    # ====================================================================
    log("\n" + "="*60)
    log("FINAL SUMMARY - CCXXXVIII")
    log("="*60)
    
    # Part 1: 颜色prompt汇总
    log("\n--- Part 1: Color Prompt Comparison ---")
    for prompt_name in PROMPTS["color"]:
        results_for_prompt = {k: v for k, v in color_prompt_results.items() 
                              if v["prompt_name"] == prompt_name}
        if results_for_prompt:
            best = max(results_for_prompt.values(), key=lambda x: x.get("n_separating_PCs", 0))
            log(f"  {prompt_name}: best n_sep={best['n_separating_PCs']} "
                f"(L{best['layer']}), match={'✓' if best['match'] else '✗'}")
    
    # Part 2: 情感可加性汇总
    log("\n--- Part 2: Emotion Additivity ---")
    for layer_key, layer_data in emotion_additivity.items():
        deltas = [nd["delta_n_sep"] for nd in layer_data.values()]
        matches = sum(1 for nd in layer_data.values() if nd["match"])
        total = len(layer_data)
        log(f"  {layer_key}: Δn_sep={deltas}, matches={matches}/{total}")
    
    # Part 3: 跨领域跨prompt汇总
    log("\n--- Part 3: Multi-Domain Multi-Prompt ---")
    for domain_name in all_domains:
        domain_results = {k: v for k, v in multi_domain_results.items() 
                          if v["domain"] == domain_name}
        if domain_results:
            best = max(domain_results.values(), key=lambda x: x.get("n_separating_PCs", 0))
            log(f"  {domain_name}: best n_sep={best['n_separating_PCs']} "
                f"({best['prompt_name']}, L{best['layer']}), "
                f"match={'✓' if best['match'] else '✗'}")
    
    # Part 4: 颜色可加性
    if "color_additivity" in results:
        log("\n--- Part 4: Color Additivity ---")
        for layer_key, layer_data in results["color_additivity"].items():
            deltas = [nd["delta_n_sep"] for nd in layer_data.values()]
            matches = sum(1 for nd in layer_data.values() if nd["match"])
            total = len(layer_data)
            log(f"  {layer_key}: Δn_sep={deltas}, matches={matches}/{total}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"CCXXXVIII Log - {args.model}\n")
    
    run(args.model)
