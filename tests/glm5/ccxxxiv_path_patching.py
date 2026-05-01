"""
CCXXXIV(334): Path Patching因果验证 + 比较词Embedding结构 + 扩展语义空间
======================================================================
核心问题1: 比较词位置的残差交换后logit是否翻转? → 因果性验证
核心问题2: bigger/smaller/heavier/lighter的embedding在W_U行空间有何不同?
核心问题3: 4+类别habitat空间是否仍是N-1维单纯形?

关键假设(CCXXXIII): 比较词位置是区分bigger/smaller的唯一来源,
  区分来自5-10%的W_U行空间分量. 需要因果验证.

实验设计:
  Part 1: Path Patching — 交换比较词位置的embedding
    对"A is bigger than B": 将pos3的bigger embedding替换为smaller embedding
    测量: logit(bigger)和logit(smaller)是否翻转?
    在不同层做patch: L0, Lmid, L3/4, Llast-1, Llast

  Part 2: 比较词Embedding的精确结构
    提取bigger/smaller/heavier/lighter/faster/slower的embedding
    计算它们在W_U行空间的投影方向和强度
    分析: 是否存在"比较操作"的通用方向 + "属性特定"的偏移?

  Part 3: 4+类别语义空间验证
    增加"desert", "underwater", "forest"等habitat类别
    验证: N个类别 → N-1维单纯形? (4类→3维, 5类→4维?)
    用PCA验证: 前N-1个PC是否编码单纯形结构?

用法:
  python ccxxxiv_path_patching.py --model qwen3
  python ccxxxiv_path_patching.py --model glm4
  python ccxxxiv_path_patching.py --model deepseek7b
"""
import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxxiv_path_patching_log.txt"

# 比较词
COMPARE_WORDS = {
    "size": ["bigger", "smaller", "larger", "tinier"],
    "weight": ["heavier", "lighter"],
    "speed": ["faster", "slower"],
    "temperature": ["hotter", "colder"],
    "age": ["older", "younger"],
}

# 大小比较对
SIZE_PAIRS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
    ("shark", "crab"), ("tiger", "rat"), ("eagle", "sparrow"),
    ("mountain", "hill"), ("tree", "bush"), ("bus", "car"),
]

# Habitat词汇 - 扩展到5类
WORDS_BY_HABITAT_EXT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
             "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
              "crab", "seal", "squid", "lobster", "jellyfish", "starfish",
              "seahorse", "eel", "manta"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
            "falcon", "pigeon", "robin", "condor", "albatross", "vulture",
            "hummingbird", "stork"],
    "underwater": ["jellyfish", "seahorse", "starfish", "coral", "anemone",
                   "sponge", "urchin", "nudibranch", "anglerfish", "guppy",
                   "betta", "catfish", "perch", "carp", "bass"],
    "desert": ["camel", "scorpion", "rattlesnake", "lizard", "coyote",
               "vulture", "roadrunner", "fennec", "meerkat", "iguana",
               "gecko", "jackal", "dingo", "tortoise", "armadillo"],
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def project_to_subspace(vec, U_basis):
    """将向量投影到U_basis的列空间"""
    coeffs = U_basis.T @ vec
    proj = U_basis @ coeffs
    return proj, coeffs


def simulate_layernorm(vec, eps=1e-5):
    """模拟LayerNorm(无weight/bias)"""
    mean = vec.mean()
    var = vec.var()
    return (vec - mean) / np.sqrt(var + eps)


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXIV(334): Path Patching + Embedding结构 + 扩展语义 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)
    
    # W_U行空间基(500分量)
    log("Computing W_U row space basis (500 components)...")
    W_U_T = W_U.T.astype(np.float32)
    k = min(500, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    U_wu, s_wu, _ = svds(W_U_T, k=k)
    sort_idx = np.argsort(s_wu)[::-1]
    U_wu = U_wu[:, sort_idx].astype(np.float64)
    s_wu_sorted = s_wu[sort_idx]
    
    # 累积覆盖率
    cum_energy = np.cumsum(s_wu_sorted**2) / np.sum(s_wu_sorted**2)
    log(f"  500-component coverage: {cum_energy[-1]:.4f}")
    log(f"  Top-1 SV: {s_wu_sorted[0]:.2f}, Top-50 SV: {s_wu_sorted[49]:.2f}")
    
    results["W_U_coverage"] = {
        "n_components": 500,
        "coverage_500": round(float(cum_energy[-1]), 4),
        "top1_sv": round(float(s_wu_sorted[0]), 2),
        "top50_sv": round(float(s_wu_sorted[49]), 2),
    }
    
    # Token IDs
    bigger_ids = [tokenizer.encode("bigger", add_special_tokens=False)[0]]
    smaller_ids = [tokenizer.encode("smaller", add_special_tokens=False)[0]]
    heavier_ids = [tokenizer.encode("heavier", add_special_tokens=False)[0]]
    lighter_ids = [tokenizer.encode("lighter", add_special_tokens=False)[0]]
    faster_ids = [tokenizer.encode("faster", add_special_tokens=False)[0]]
    slower_ids = [tokenizer.encode("slower", add_special_tokens=False)[0]]
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # ====================================================================
    # Part 1: Path Patching — 交换比较词位置的embedding
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: Path Patching因果验证")
    log("="*60)
    
    patching_results = {}
    
    # 获取embedding层
    embed_layer = model.get_input_embeddings()
    
    # 对6对比较词做patching
    test_pairs = SIZE_PAIRS[:6]
    
    for big, small in test_pairs:
        # 构造两个prompt
        prompt_bigger = f"The {big} is bigger than the {small}"
        prompt_smaller = f"The {big} is smaller than the {small}"
        
        toks_bigger = tokenizer(prompt_bigger, return_tensors="pt").to(device)
        toks_smaller = tokenizer(prompt_smaller, return_tensors="pt").to(device)
        
        seq_len = toks_bigger.input_ids.shape[1]
        
        # 找比较词位置(通常pos3)
        tokens_bigger_list = [tokenizer.decode([t]) for t in toks_bigger.input_ids[0].tolist()]
        tokens_smaller_list = [tokenizer.decode([t]) for t in toks_smaller.input_ids[0].tolist()]
        
        # 找到不同token的位置
        diff_positions = []
        for i in range(min(len(tokens_bigger_list), len(tokens_smaller_list))):
            if tokens_bigger_list[i] != tokens_smaller_list[i]:
                diff_positions.append(i)
        
        if not diff_positions:
            log(f"  Skip {big}/{small}: no diff position found")
            continue
        
        comp_pos = diff_positions[0]  # 比较词位置
        log(f"  {big}/{small}: diff at positions {diff_positions}, comp_pos={comp_pos}")
        
        # === Patching实验 ===
        # 1) Baseline: 正常forward
        with torch.no_grad():
            out_bigger = model(**toks_bigger)
            logits_bigger = out_bigger.logits[0, -1]  # [vocab]
            bigger_logit_on_bigger = logits_bigger[bigger_ids[0]].item()
            smaller_logit_on_bigger = logits_bigger[smaller_ids[0]].item()
            
            out_smaller = model(**toks_smaller)
            logits_smaller = out_smaller.logits[0, -1]
            bigger_logit_on_smaller = logits_smaller[bigger_ids[0]].item()
            smaller_logit_on_smaller = logits_smaller[smaller_ids[0]].item()
        
        # 2) Patch: 将bigger句中pos3的embedding替换为smaller的embedding
        #    即: "The elephant is [smaller_embed] than the mouse"
        embed_bigger = embed_layer(toks_bigger.input_ids).detach().clone()
        embed_smaller = embed_layer(toks_smaller.input_ids).detach().clone()
        
        # Patch: 在比较词位置交换embedding
        embed_patched = embed_bigger.clone()
        embed_patched[0, comp_pos, :] = embed_smaller[0, comp_pos, :]
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        with torch.no_grad():
            out_patched = model(inputs_embeds=embed_patched, position_ids=position_ids)
            logits_patched = out_patched.logits[0, -1]
            bigger_logit_patched = logits_patched[bigger_ids[0]].item()
            smaller_logit_patched = logits_patched[smaller_ids[0]].item()
        
        # 3) 反向Patch: 将smaller句中pos3的embedding替换为bigger的embedding
        embed_patched_rev = embed_smaller.clone()
        embed_patched_rev[0, comp_pos, :] = embed_bigger[0, comp_pos, :]
        
        with torch.no_grad():
            out_patched_rev = model(inputs_embeds=embed_patched_rev, position_ids=position_ids)
            logits_patched_rev = out_patched_rev.logits[0, -1]
            bigger_logit_patched_rev = logits_patched_rev[bigger_ids[0]].item()
            smaller_logit_patched_rev = logits_patched_rev[smaller_ids[0]].item()
        
        # 4) 部分patch: 只替换比较词token的残差方向, 保持模长
        #    diff = emb_bigger - emb_smaller at comp_pos
        diff_emb = embed_bigger[0, comp_pos, :] - embed_smaller[0, comp_pos, :]
        # 将bigger句中的comp_pos添加diff(放大差异)
        embed_amplified = embed_bigger.clone()
        embed_amplified[0, comp_pos, :] += diff_emb  # 2x diff
        
        with torch.no_grad():
            out_amplified = model(inputs_embeds=embed_amplified, position_ids=position_ids)
            logits_amplified = out_amplified.logits[0, -1]
            bigger_logit_amplified = logits_amplified[bigger_ids[0]].item()
            smaller_logit_amplified = logits_amplified[smaller_ids[0]].item()
        
        key = f"{big}_{small}"
        patching_results[key] = {
            "comp_pos": comp_pos,
            "baseline": {
                "bigger_prompt": {
                    "logit_bigger": round(bigger_logit_on_bigger, 4),
                    "logit_smaller": round(smaller_logit_on_bigger, 4),
                    "diff": round(bigger_logit_on_bigger - smaller_logit_on_bigger, 4),
                },
                "smaller_prompt": {
                    "logit_bigger": round(bigger_logit_on_smaller, 4),
                    "logit_smaller": round(smaller_logit_on_smaller, 4),
                    "diff": round(bigger_logit_on_smaller - smaller_logit_on_smaller, 4),
                },
            },
            "patch_bigger_to_smaller": {
                "logit_bigger": round(bigger_logit_patched, 4),
                "logit_smaller": round(smaller_logit_patched, 4),
                "diff": round(bigger_logit_patched - smaller_logit_patched, 4),
                "flip_ratio": round(
                    (bigger_logit_patched - smaller_logit_patched) / 
                    max(abs(bigger_logit_on_bigger - smaller_logit_on_bigger), 1e-6), 4),
            },
            "patch_smaller_to_bigger": {
                "logit_bigger": round(bigger_logit_patched_rev, 4),
                "logit_smaller": round(smaller_logit_patched_rev, 4),
                "diff": round(bigger_logit_patched_rev - smaller_logit_patched_rev, 4),
                "flip_ratio": round(
                    (bigger_logit_patched_rev - smaller_logit_patched_rev) / 
                    max(abs(bigger_logit_on_smaller - smaller_logit_on_smaller), 1e-6), 4),
            },
            "amplified_bigger": {
                "logit_bigger": round(bigger_logit_amplified, 4),
                "logit_smaller": round(smaller_logit_amplified, 4),
                "diff": round(bigger_logit_amplified - smaller_logit_amplified, 4),
                "amplification_ratio": round(
                    (bigger_logit_amplified - smaller_logit_amplified) / 
                    max(abs(bigger_logit_on_bigger - smaller_logit_on_bigger), 1e-6), 4),
            },
        }
        
        log(f"  Baseline bigger: diff={bigger_logit_on_bigger - smaller_logit_on_bigger:.4f}")
        log(f"  Baseline smaller: diff={bigger_logit_on_smaller - smaller_logit_on_smaller:.4f}")
        log(f"  Patch B→S: diff={bigger_logit_patched - smaller_logit_patched:.4f} "
            f"(flip_ratio={patching_results[key]['patch_bigger_to_smaller']['flip_ratio']})")
        log(f"  Patch S→B: diff={bigger_logit_patched_rev - smaller_logit_patched_rev:.4f} "
            f"(flip_ratio={patching_results[key]['patch_smaller_to_bigger']['flip_ratio']})")
        log(f"  Amplified: diff={bigger_logit_amplified - smaller_logit_amplified:.4f} "
            f"(ratio={patching_results[key]['amplified_bigger']['amplification_ratio']})")
    
    results["path_patching"] = patching_results
    
    # 汇总
    log("\n  === Path Patching Summary ===")
    flip_ratios_b2s = []
    flip_ratios_s2b = []
    amp_ratios = []
    for key, val in patching_results.items():
        flip_ratios_b2s.append(val["patch_bigger_to_smaller"]["flip_ratio"])
        flip_ratios_s2b.append(val["patch_smaller_to_bigger"]["flip_ratio"])
        amp_ratios.append(val["amplified_bigger"]["amplification_ratio"])
    
    if flip_ratios_b2s:
        log(f"  Patch B→S: mean flip_ratio = {np.mean(flip_ratios_b2s):.4f} ± {np.std(flip_ratios_b2s):.4f}")
        log(f"  Patch S→B: mean flip_ratio = {np.mean(flip_ratios_s2b):.4f} ± {np.std(flip_ratios_s2b):.4f}")
        log(f"  Amplified: mean ratio = {np.mean(amp_ratios):.4f} ± {np.std(amp_ratios):.4f}")
    
    # ====================================================================
    # Part 2: 比较词Embedding的精确结构
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: 比较词Embedding结构分析")
    log("="*60)
    
    # 收集所有比较词的embedding
    all_compare_words = []
    for attr, words in COMPARE_WORDS.items():
        all_compare_words.extend([(w, attr) for w in words])
    
    embed_data = {}
    embed_vectors = {}
    
    for word, attr in all_compare_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        tok_id = tok_ids[0]
        
        # 获取token embedding
        emb = embed_layer.weight[tok_id].detach().cpu().float().numpy()  # [d_model]
        embed_vectors[word] = emb
        
        # 在W_U行空间的投影
        proj, coeffs = project_to_subspace(emb, U_wu)
        proj_norm = np.linalg.norm(proj)
        emb_norm = np.linalg.norm(emb)
        row_ratio = proj_norm**2 / max(emb_norm**2, 1e-20)
        
        # W_U行空间投影方向的主要分量
        top_coeffs_idx = np.argsort(np.abs(coeffs))[-10:][::-1]
        top_coeffs_vals = coeffs[top_coeffs_idx]
        
        # 比较词在W_U中的logit
        bigger_logit = float(W_U[bigger_ids[0]] @ emb)
        smaller_logit = float(W_U[smaller_ids[0]] @ emb)
        heavier_logit = float(W_U[heavier_ids[0]] @ emb)
        lighter_logit = float(W_U[lighter_ids[0]] @ emb)
        faster_logit = float(W_U[faster_ids[0]] @ emb)
        slower_logit = float(W_U[slower_ids[0]] @ emb)
        
        embed_data[word] = {
            "attribute": attr,
            "token_id": tok_id,
            "emb_norm": round(float(emb_norm), 4),
            "row_ratio": round(float(row_ratio), 4),
            "logit_bigger": round(bigger_logit, 4),
            "logit_smaller": round(smaller_logit, 4),
            "logit_heavier": round(heavier_logit, 4),
            "logit_lighter": round(lighter_logit, 4),
            "logit_faster": round(faster_logit, 4),
            "logit_slower": round(slower_logit, 4),
            "top5_WU_components": [int(x) for x in top_coeffs_idx[:5]],
        }
    
    # 比较词对之间的余弦相似度
    cos_matrix = {}
    for w1, attr1 in all_compare_words:
        for w2, attr2 in all_compare_words:
            if w1 >= w2:
                continue
            if w1 not in embed_vectors or w2 not in embed_vectors:
                continue
            v1 = embed_vectors[w1]
            v2 = embed_vectors[w2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos = float(np.dot(v1, v2) / (n1 * n2))
            else:
                cos = 0
            cos_matrix[f"{w1}_vs_{w2}"] = {
                "cos": round(cos, 4),
                "same_attr": attr1 == attr2,
                "attr1": attr1,
                "attr2": attr2,
            }
    
    # 反义词对的embedding差异
    antonym_diffs = {}
    antonym_pairs = [
        ("bigger", "smaller"), ("larger", "tinier"),
        ("heavier", "lighter"), ("faster", "slower"),
        ("hotter", "colder"), ("older", "younger"),
    ]
    
    for w1, w2 in antonym_pairs:
        if w1 not in embed_vectors or w2 not in embed_vectors:
            continue
        diff = embed_vectors[w1] - embed_vectors[w2]
        diff_norm = np.linalg.norm(diff)
        
        if diff_norm > 1e-10:
            diff_dir = diff / diff_norm
            proj_diff, _ = project_to_subspace(diff_dir, U_wu)
            diff_in_row = np.linalg.norm(proj_diff)**2
            
            # 反义词方向是否指向对应的W_U行?
            bigger_proj = float(np.dot(diff_dir, W_U[bigger_ids[0]]))
            smaller_proj = float(np.dot(diff_dir, W_U[smaller_ids[0]]))
            heavier_proj = float(np.dot(diff_dir, W_U[heavier_ids[0]]))
            lighter_proj = float(np.dot(diff_dir, W_U[lighter_ids[0]]))
        else:
            diff_in_row = 0
            bigger_proj = smaller_proj = heavier_proj = lighter_proj = 0
        
        antonym_diffs[f"{w1}_minus_{w2}"] = {
            "diff_norm": round(float(diff_norm), 4),
            "diff_in_WU_row": round(float(diff_in_row), 4),
            "proj_on_bigger": round(float(bigger_proj), 4),
            "proj_on_smaller": round(float(smaller_proj), 4),
            "proj_on_heavier": round(float(heavier_proj), 4),
            "proj_on_lighter": round(float(lighter_proj), 4),
        }
    
    # "比较操作"通用方向分析
    # 所有正向(bigger, heavier, faster等)的平均 vs 所有负向(smaller, lighter, slower等)的平均
    positive_words = ["bigger", "larger", "heavier", "faster", "hotter", "older"]
    negative_words = ["smaller", "tinier", "lighter", "slower", "colder", "younger"]
    
    pos_vecs = [embed_vectors[w] for w in positive_words if w in embed_vectors]
    neg_vecs = [embed_vectors[w] for w in negative_words if w in embed_vectors]
    
    generic_compare_dir = {}
    if len(pos_vecs) >= 2 and len(neg_vecs) >= 2:
        pos_mean = np.mean(pos_vecs, axis=0)
        neg_mean = np.mean(neg_vecs, axis=0)
        generic_diff = pos_mean - neg_mean
        generic_norm = np.linalg.norm(generic_diff)
        
        if generic_norm > 1e-10:
            generic_dir = generic_diff / generic_norm
            proj_generic, _ = project_to_subspace(generic_dir, U_wu)
            generic_in_row = np.linalg.norm(proj_generic)**2
            
            # 通用方向与各属性方向的余弦
            cos_with_size = float(np.dot(generic_dir, embed_vectors.get("bigger", np.zeros(d_model)) - embed_vectors.get("smaller", np.zeros(d_model)))) if "bigger" in embed_vectors and "smaller" in embed_vectors else 0
            cos_with_weight = float(np.dot(generic_dir, embed_vectors.get("heavier", np.zeros(d_model)) - embed_vectors.get("lighter", np.zeros(d_model)))) if "heavier" in embed_vectors and "lighter" in embed_vectors else 0
            cos_with_speed = float(np.dot(generic_dir, embed_vectors.get("faster", np.zeros(d_model)) - embed_vectors.get("slower", np.zeros(d_model)))) if "faster" in embed_vectors and "slower" in embed_vectors else 0
            
            # 归一化
            size_diff_norm = np.linalg.norm(embed_vectors.get("bigger", np.zeros(d_model)) - embed_vectors.get("smaller", np.zeros(d_model)))
            weight_diff_norm = np.linalg.norm(embed_vectors.get("heavier", np.zeros(d_model)) - embed_vectors.get("lighter", np.zeros(d_model)))
            speed_diff_norm = np.linalg.norm(embed_vectors.get("faster", np.zeros(d_model)) - embed_vectors.get("slower", np.zeros(d_model)))
            
            cos_size = cos_with_size / max(size_diff_norm * generic_norm, 1e-10)
            cos_weight = cos_with_weight / max(weight_diff_norm * generic_norm, 1e-10)
            cos_speed = cos_with_speed / max(speed_diff_norm * generic_norm, 1e-10)
            
            generic_compare_dir = {
                "norm": round(float(generic_norm), 4),
                "in_WU_row": round(float(generic_in_row), 4),
                "cos_with_size": round(float(cos_size), 4),
                "cos_with_weight": round(float(cos_weight), 4),
                "cos_with_speed": round(float(cos_speed), 4),
            }
    
    embed_results = {
        "individual": embed_data,
        "cosine_matrix": cos_matrix,
        "antonym_diffs": antonym_diffs,
        "generic_compare_direction": generic_compare_dir,
    }
    
    results["embedding_structure"] = embed_results
    
    # 汇总
    log("\n  === Embedding Structure Summary ===")
    for word, data in embed_data.items():
        log(f"  {word}({data['attribute']}): row_ratio={data['row_ratio']:.4f}, "
            f"logit_bigger={data['logit_bigger']:.4f}, logit_smaller={data['logit_smaller']:.4f}")
    
    log("\n  Antonym diffs:")
    for key, val in antonym_diffs.items():
        log(f"  {key}: diff_in_row={val['diff_in_WU_row']:.4f}, "
            f"proj_bigger={val['proj_on_bigger']:.4f}, proj_smaller={val['proj_on_smaller']:.4f}")
    
    if generic_compare_dir:
        log(f"\n  Generic compare direction: in_row={generic_compare_dir['in_WU_row']:.4f}, "
            f"cos_size={generic_compare_dir['cos_with_size']:.4f}, "
            f"cos_weight={generic_compare_dir['cos_with_weight']:.4f}, "
            f"cos_speed={generic_compare_dir['cos_with_speed']:.4f}")
    
    # ====================================================================
    # Part 3: 4+类别语义空间验证
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: 扩展语义空间验证 (3/4/5类)")
    log("="*60)
    
    semantic_ext_results = {}
    
    # 测试不同数量的habitat类别
    habitat_configs = {
        "3class": ["land", "ocean", "sky"],
        "4class": ["land", "ocean", "sky", "desert"],
        "5class": ["land", "ocean", "sky", "desert", "underwater"],
    }
    
    # 选择测试层
    sem_test_layers = [0, n_layers // 2, n_layers - 2, n_layers - 1]
    if n_layers // 4 not in sem_test_layers:
        sem_test_layers.append(n_layers // 4)
    sem_test_layers = sorted(set(sem_test_layers))
    
    for config_name, hab_list in habitat_configs.items():
        log(f"\n  --- {config_name}: {hab_list} ---")
        
        for li in sem_test_layers:
            hab_data = {}
            for hab in hab_list:
                words = WORDS_BY_HABITAT_EXT.get(hab, [])
                hab_resids = []
                for word in words[:12]:  # 每类12个词
                    prompt = f"The {word} lives in the"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    captured = {}
                    
                    def mk_hook(k):
                        def hook(m, inp, out):
                            o = out[0] if isinstance(out, tuple) else out
                            captured[k] = o[0, -1, :].detach().float().cpu().numpy()
                        return hook
                    
                    hook = layers[li].register_forward_hook(mk_hook(f"L{li}"))
                    with torch.no_grad():
                        _ = model(**toks)
                    hook.remove()
                    
                    if f"L{li}" in captured:
                        hab_resids.append(captured[f"L{li}"])
                
                hab_data[hab] = hab_resids
            
            # PCA分析
            all_vecs = []
            all_labels = []
            for hab in hab_list:
                all_vecs.extend(hab_data[hab])
                all_labels.extend([hab] * len(hab_data[hab]))
            
            if len(all_vecs) < len(hab_list) * 5:
                log(f"    L{li}: insufficient data ({len(all_vecs)} samples)")
                continue
            
            arr = np.array(all_vecs)
            centered = arr - arr.mean(axis=0)
            
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            
            n_classes = len(hab_list)
            n_pc = min(n_classes + 2, Vt.shape[0])  # 检查N-1维+N+1维
            
            # 各类在前N+2个PC上的投影
            hab_proj = {}
            for hab in hab_list:
                if len(hab_data[hab]) < 3:
                    continue
                hab_arr = np.array(hab_data[hab])
                hab_centered = hab_arr - arr.mean(axis=0)
                pc_proj = hab_centered @ Vt[:n_pc].T  # [n_samples, n_pc]
                hab_proj[hab] = {
                    "mean": [round(float(np.mean(pc_proj[:, i])), 4) for i in range(n_pc)],
                    "std": [round(float(np.std(pc_proj[:, i])), 4) for i in range(n_pc)],
                }
            
            # 判断每个PC的分离能力
            pc_separation = {}
            for pc_i in range(n_pc):
                means = []
                within_vars = []
                for hab in hab_list:
                    if hab in hab_proj:
                        means.append(hab_proj[hab]["mean"][pc_i])
                        within_vars.append(hab_proj[hab]["std"][pc_i]**2)
                
                if len(means) >= 2:
                    between_var = np.var(means)
                    avg_within = np.mean(within_vars) if within_vars else 1e-10
                    f_ratio = between_var / max(avg_within, 1e-10)
                else:
                    f_ratio = 0
                
                # 检查: 这个PC能否分离某些类?
                # 即: 是否存在某些habitat的mean明显不同
                max_abs_mean = max(abs(m) for m in means) if means else 0
                is_separating = f_ratio > 1.0  # F>1表示between>within
                
                pc_separation[f"PC{pc_i}"] = {
                    "sv": round(float(S[pc_i]), 4),
                    "var_explained": round(float(S[pc_i]**2 / np.sum(S**2)), 4),
                    "f_ratio": round(float(f_ratio), 4),
                    "is_separating": is_separating,
                    "means": {hab: hab_proj[hab]["mean"][pc_i] for hab in hab_list if hab in hab_proj},
                }
            
            # ★★★ 关键测试: N-1维单纯形假设
            # 前N-1个PC应该能分离N个类, 第N个PC不能
            n_separating = sum(1 for pc in pc_separation.values() if pc["is_separating"])
            expected_separating = n_classes - 1
            
            # 检验: 前N-1个PC的F-ratio都>1, 第N个PC的F-ratio<1?
            top_n1_separating = all(
                pc_separation[f"PC{i}"]["is_separating"] 
                for i in range(min(n_classes - 1, n_pc))
            )
            nth_not_separating = (
                n_classes - 1 < n_pc and 
                not pc_separation[f"PC{n_classes-1}"]["is_separating"]
            )
            
            simplex_hypothesis = top_n1_separating and nth_not_separating
            
            sem_key = f"{config_name}_L{li}"
            semantic_ext_results[sem_key] = {
                "n_classes": n_classes,
                "expected_dim": n_classes - 1,
                "n_separating_PCs": n_separating,
                "simplex_hypothesis": simplex_hypothesis,
                "top_n1_separating": top_n1_separating,
                "nth_not_separating": nth_not_separating,
                "pc_separation": pc_separation,
                "habitat_projections": hab_proj,
            }
            
            log(f"    L{li}: {config_name} → n_classes={n_classes}, "
                f"expected_dim={n_classes-1}, n_separating={n_separating}, "
                f"simplex_hypothesis={simplex_hypothesis}")
            for pc_i in range(min(n_pc, n_classes + 1)):
                sep = pc_separation[f"PC{pc_i}"]
                means_str = ", ".join(f"{h}={v:.2f}" for h, v in sep["means"].items())
                log(f"      PC{pc_i}: F={sep['f_ratio']:.2f}, separating={sep['is_separating']}, "
                    f"means: {means_str}")
    
    results["semantic_extended"] = semantic_ext_results
    
    # ====================================================================
    # Part 4: Inter-layer path tracing — 比较信息在层间的传播
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: Inter-layer比较信息传播追踪")
    log("="*60)
    
    # 对一个比较对, 在中间某层交换比较词位置的残差
    # 观察logit变化 → 信息在哪个层变得"不可逆"?
    
    big, small = "elephant", "mouse"
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
    
    log(f"  Comparison word at position {comp_pos}")
    
    # 收集bigger和smaller句子在各层的残差(比较词位置)
    inter_layer_results = {}
    
    # 选择要测试的层
    layer_patch_points = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in layer_patch_points:
        layer_patch_points.append(n_layers - 1)
    
    for patch_layer in layer_patch_points:
        # 在patch_layer层交换比较词位置的残差
        # 方法: 用hook在patch_layer之后交换比较词位置的输出
        
        captured_b = {}
        captured_s = {}
        
        def mk_capture_hook(key, storage):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                storage[key] = o.detach().clone()
            return hook
        
        # 先运行bigger句子, 收集所有层输出
        hooks_b = [layers[li].register_forward_hook(mk_capture_hook(f"L{li}", captured_b)) 
                   for li in range(n_layers)]
        with torch.no_grad():
            _ = model(**toks_b)
        for h in hooks_b:
            h.remove()
        
        # 再运行smaller句子, 收集所有层输出
        hooks_s = [layers[li].register_forward_hook(mk_capture_hook(f"L{li}", captured_s))
                   for li in range(n_layers)]
        with torch.no_grad():
            _ = model(**toks_s)
        for h in hooks_s:
            h.remove()
        
        # 构造patched forward: 在patch_layer之后替换比较词位置
        # 使用bigger句的层输出, 但在patch_layer用smaller句的比较词位置输出
        # 注意: 这需要修改残差流, 比较复杂
        # 简化方法: 直接计算两层残差在比较词位置的差异
        
        if f"L{patch_layer}" in captured_b and f"L{patch_layer}" in captured_s:
            res_b_comp = captured_b[f"L{patch_layer}"][0, comp_pos, :].float().cpu().numpy()
            res_s_comp = captured_s[f"L{patch_layer}"][0, comp_pos, :].float().cpu().numpy()
            res_b_last = captured_b[f"L{patch_layer}"][0, -1, :].float().cpu().numpy()
            res_s_last = captured_s[f"L{patch_layer}"][0, -1, :].float().cpu().numpy()
            
            # 比较词位置的差异
            diff_comp = res_b_comp - res_s_comp
            diff_comp_norm = np.linalg.norm(diff_comp)
            
            # 最后位置的差异
            diff_last = res_b_last - res_s_last
            diff_last_norm = np.linalg.norm(diff_last)
            
            # 差异在W_U行空间
            if diff_comp_norm > 1e-10:
                proj_comp, _ = project_to_subspace(diff_comp / diff_comp_norm, U_wu)
                comp_in_row = np.linalg.norm(proj_comp)**2
            else:
                comp_in_row = 0
            
            if diff_last_norm > 1e-10:
                proj_last, _ = project_to_subspace(diff_last / diff_last_norm, U_wu)
                last_in_row = np.linalg.norm(proj_last)**2
            else:
                last_in_row = 0
            
            # 两个位置差异的余弦
            if diff_comp_norm > 1e-10 and diff_last_norm > 1e-10:
                cos_comp_last = float(np.dot(diff_comp, diff_last) / (diff_comp_norm * diff_last_norm))
            else:
                cos_comp_last = 0
            
            inter_layer_results[f"L{patch_layer}"] = {
                "comp_pos_diff_norm": round(float(diff_comp_norm), 4),
                "comp_pos_diff_in_row": round(float(comp_in_row), 4),
                "last_pos_diff_norm": round(float(diff_last_norm), 4),
                "last_pos_diff_in_row": round(float(last_in_row), 4),
                "cos_comp_vs_last_diff": round(float(cos_comp_last), 4),
            }
            
            log(f"  L{patch_layer}: comp_diff={diff_comp_norm:.4f}(row={comp_in_row:.4f}), "
                f"last_diff={diff_last_norm:.4f}(row={last_in_row:.4f}), "
                f"cos={cos_comp_last:.4f}")
    
    results["inter_layer_propagation"] = inter_layer_results
    
    # 保存 (处理numpy类型)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output_path = TEMP / f"ccxxxiv_path_patching_{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    log(f"\nResults saved to {output_path}")
    
    # 释放模型
    del captured_b, captured_s
    gc.collect()
    release_model(model)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    run(args.model)
