"""
CCXXXII(332): 残差向量W_U行空间投影分解 + 语义→logit完整路径
======================================================================
核心问题1: 残差向量本身有多少在W_U行空间? (不是"方向差", 是完整向量)
核心问题2: "A is bigger than B" vs "A is" → W_U行空间分量差是多少?
核心问题3: 用更多SVD分量(200/500)重新验证"bigger方向在W_U零空间"
核心问题4: 推理-语义正交性的精确刻画(子空间维度+重叠)

关键洞察(CCXXXI): bigger方向在W_U零空间(cos≈0), 但残差向量本身
  有足够W_U行空间分量驱动正确输出 → 需要分解残差向量!

实验设计:
  Part 1: W_U行空间完整估计 — 用50/100/200/500个SVD分量,
    测量W_U行空间各维度的覆盖率, 确定合理分量数

  Part 2: 残差向量分解 — 每个prompt的残差向量分解为:
    resid = proj_row(W_U) + proj_null(W_U)
    测量: proj_row的范数占比, proj_row中哪些成分驱动bigger/smaller

  Part 3: 语义→logit路径 — 比较不同prompt的W_U行空间分量:
    "A is bigger than B" vs "A is" vs "A is big"
    → W_U行空间分量如何变化? bigger/smaller logit由哪部分驱动?

  Part 4: 推理-语义正交性精确刻画 — 构建两个子空间:
    语义子空间: habitat分类方向的PCA
    推理子空间: 比较操作方向的PCA
    → 子空间维度, 重叠度, Grassmann距离

用法:
  python ccxxxii_resid_wu_decompose.py --model qwen3
  python ccxxxii_resid_wu_decompose.py --model glm4
  python ccxxxii_resid_wu_decompose.py --model deepseek7b
"""
import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxxii_resid_wu_decompose_log.txt"

# ===== 词汇定义 =====
SIZE_COMPARE = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
    ("shark", "crab"), ("tiger", "rat"), ("eagle", "sparrow"),
    ("mountain", "hill"), ("tree", "bush"), ("bus", "car"),
    ("giraffe", "dog"), ("hippo", "ant"), ("rhino", "bee"),
    ("wolf", "fly"), ("deer", "worm"), ("camel", "flea"),
]

WEIGHT_COMPARE = [
    ("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
    ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
    ("concrete", "foam"), ("brick", "straw"), ("copper", "wool"),
    ("platinum", "air"), ("tungsten", "dust"), ("uranium", "smoke"),
]

SPEED_COMPARE = [
    ("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
    ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
    ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle"),
    ("ferrari", "bicycle"), ("bullet", "cloud"), ("lightning", "glacier"),
]

# 纯属性词汇
SIZE_ATTR_WORDS = ["elephant", "mountain", "whale", "building", "giant",
                   "mouse", "ant", "grain", "atom", "dot",
                   "dinosaur", "planet", "bacteria", "universe", "pebble"]
WEIGHT_ATTR_WORDS = ["iron", "lead", "stone", "steel", "concrete",
                     "feather", "air", "bubble", "smoke", "dust",
                     "tungsten", "platinum", "helium", "vacuum", "foam"]
SPEED_ATTR_WORDS = ["cheetah", "rocket", "lightning", "falcon", "bullet",
                    "snail", "turtle", "sloth", "glacier", "molasses",
                    "missile", "photon", "worm", "snail", "standstill"]

# Habitat词汇 (更多, 用于构建更稳健的语义子空间)
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
             "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra",
             "monkey", "gorilla", "hippo", "rhino", "camel"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
              "crab", "seal", "squid", "lobster", "jellyfish", "starfish",
              "seahorse", "eel", "manta", "clam", "shrimp", "anchovy",
              "anglerfish", "narwhal"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
            "falcon", "pigeon", "robin", "condor", "albatross", "vulture",
            "hummingbird", "stork", "crane", "pelican", "flamingo",
            "penguin", "seagull"],
}

# 模板
TEMPLATE_COMPARE = "The {} is bigger than the {}"
TEMPLATE_WEIGHT_COMPARE = "The {} is heavier than the {}"
TEMPLATE_SPEED_COMPARE = "The {} is faster than the {}"
TEMPLATE_SIZE_ATTR = "The {} is very big"
TEMPLATE_WEIGHT_ATTR = "The {} is very heavy"
TEMPLATE_SPEED_ATTR = "The {} is very fast"
TEMPLATE_HABITAT = "The {} lives in the"
TEMPLATE_NEUTRAL = "The {} is"  # 中性基线
TEMPLATE_COMPARE_REV = "The {} is smaller than the {}"  # 反向比较

HABITAT_TOKENS = {
    "land": ["land", "ground", "earth", "field", "forest", "plains"],
    "ocean": ["ocean", "sea", "water", "river", "lake", "deep"],
    "sky": ["sky", "air", "trees", "mountains", "heights", "clouds"],
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers):
    """收集各层残差(最后一个token位置)"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = toks.input_ids.shape[1]
    last_pos = seq_len - 1
    
    captured = {}
    def mk_hook(k):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
        return hook
    
    hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers]
    with torch.no_grad():
        _ = model(**toks)
    for h in hooks:
        h.remove()
    return captured


def compute_wu_row_space(W_U, n_components_list):
    """
    计算W_U行空间在不同SVD分量数下的基
    
    Returns:
        dict: {n_comp: U_basis [d_model, n_comp]}
    """
    results = {}
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab_size]
    d_model = W_U_T.shape[0]
    
    # 先计算最大分量数的SVD, 然后截取
    max_comp = max(n_components_list)
    k = min(max_comp, min(W_U_T.shape) - 2)
    k = max(k, 10)
    
    log(f"  Computing SVD of W_U^T with k={k}...")
    t0 = time.time()
    U_full, s_full, _ = svds(W_U_T, k=k)
    # svds返回无序, 需要排序
    sort_idx = np.argsort(s_full)[::-1]
    U_full = U_full[:, sort_idx]
    s_full = s_full[sort_idx]
    log(f"  SVD done in {time.time()-t0:.1f}s, top-10 singular values: {s_full[:10].round(2)}")
    
    for n_comp in n_components_list:
        n = min(n_comp, k)
        results[n_comp] = U_full[:, :n].astype(np.float64)
    
    # 返回奇异值(用于覆盖率分析)
    return results, s_full


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
    
    log(f"\n{'='*70}\nCCXXXII(332): 残差W_U分解 + 语义→logit路径 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # 找token IDs
    bigger_tokens = ["bigger", "larger", "greater", "huge", "enormous", "massive", "giant", "immense"]
    smaller_tokens = ["smaller", "tiny", "little", "miniature", "minute", "petite", "diminutive"]
    heavier_tokens = ["heavier", "weightier", "denser", "ponderous", "massive"]
    lighter_tokens = ["lighter", "feathery", "weightless", "airy", "buoyant"]
    faster_tokens = ["faster", "quicker", "swifter", "speedier", "rapid"]
    slower_tokens = ["slower", "sluggish", "leisurely", "unhurried", "languid"]
    
    bigger_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in bigger_tokens 
                 if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    smaller_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in smaller_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    heavier_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in heavier_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    lighter_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in lighter_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    faster_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in faster_tokens 
                  if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    slower_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in slower_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # ====================================================================
    # Part 1: W_U行空间完整估计
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: W_U行空间完整估计")
    log("="*60)
    
    n_comp_list = [50, 100, 200, 500]
    wu_bases, wu_sv = compute_wu_row_space(W_U, n_comp_list)
    
    # 计算各分量数的覆盖率
    total_var = np.sum(wu_sv ** 2)
    cum_var = np.cumsum(wu_sv ** 2) / total_var
    
    coverage_results = {}
    for n_comp in n_comp_list:
        n = min(n_comp, len(wu_sv))
        coverage = float(cum_var[n-1])
        coverage_results[n_comp] = {
            "coverage": round(coverage, 4),
            "top_sv": [round(float(s), 2) for s in wu_sv[:min(10, n)]],
        }
        log(f"  W_U row space with {n_comp} components: coverage={coverage:.4f}")
    
    results["wu_coverage"] = coverage_results
    
    # ====================================================================
    # Part 2: 残差向量分解 — resid = proj_row + proj_null
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: 残差向量分解")
    log("="*60)
    
    # 收集各类残差
    # 2a: size比较残差
    log("  2a: 收集size比较残差...")
    size_compare_resids = {li: [] for li in test_layers}
    for big, small in SIZE_COMPARE:
        prompt = TEMPLATE_COMPARE.format(big, small)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                size_compare_resids[li].append(cap[f"L{li}"])
    
    # 2b: size比较(反向)
    log("  2b: 收集size反向比较残差...")
    size_reverse_resids = {li: [] for li in test_layers}
    for big, small in SIZE_COMPARE:
        prompt = TEMPLATE_COMPARE_REV.format(small, big)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                size_reverse_resids[li].append(cap[f"L{li}"])
    
    # 2c: weight比较残差
    log("  2c: 收集weight比较残差...")
    weight_compare_resids = {li: [] for li in test_layers}
    for heavy, light in WEIGHT_COMPARE:
        prompt = TEMPLATE_WEIGHT_COMPARE.format(heavy, light)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                weight_compare_resids[li].append(cap[f"L{li}"])
    
    # 2d: speed比较残差
    log("  2d: 收集speed比较残差...")
    speed_compare_resids = {li: [] for li in test_layers}
    for fast, slow in SPEED_COMPARE:
        prompt = TEMPLATE_SPEED_COMPARE.format(fast, slow)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                speed_compare_resids[li].append(cap[f"L{li}"])
    
    # 2e: habitat残差
    log("  2e: 收集habitat残差...")
    habitat_resids = {li: {"land": [], "ocean": [], "sky": []} for li in test_layers}
    for hab, words in WORDS_BY_HABITAT.items():
        for word in words:
            prompt = TEMPLATE_HABITAT.format(word)
            cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
            for li in test_layers:
                if f"L{li}" in cap:
                    habitat_resids[li][hab].append(cap[f"L{li}"])
    
    # 2f: 中性句子残差 ("The elephant is")
    log("  2f: 收集中性句子残差...")
    neutral_words = ["elephant", "whale", "lion", "dog", "eagle", 
                     "iron", "rock", "steel", "cheetah", "rocket",
                     "mountain", "ocean", "tree", "house", "car",
                     "bear", "horse", "shark", "falcon", "tiger"]
    neutral_resids = {li: [] for li in test_layers}
    for word in neutral_words:
        prompt = TEMPLATE_NEUTRAL.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                neutral_resids[li].append(cap[f"L{li}"])
    
    # 2g: 属性句子残差 ("The elephant is very big")
    log("  2g: 收集属性句子残差...")
    size_attr_resids = {li: [] for li in test_layers}
    for word in SIZE_ATTR_WORDS:
        prompt = TEMPLATE_SIZE_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                size_attr_resids[li].append(cap[f"L{li}"])
    
    weight_attr_resids = {li: [] for li in test_layers}
    for word in WEIGHT_ATTR_WORDS:
        prompt = TEMPLATE_WEIGHT_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                weight_attr_resids[li].append(cap[f"L{li}"])
    
    speed_attr_resids = {li: [] for li in test_layers}
    for word in SPEED_ATTR_WORDS:
        prompt = TEMPLATE_SPEED_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                speed_attr_resids[li].append(cap[f"L{li}"])
    
    # 使用200分量作为主分析(覆盖50%+行空间)
    n_main = 200
    if n_main not in wu_bases:
        n_main = max(k for k in wu_bases.keys())
    U_wu = wu_bases[n_main]
    
    # 分解各层各类残差
    log("\n  分解残差向量...")
    resid_decompose = {}
    
    for li in test_layers:
        layer_results = {}
        
        # 对每种残差类别, 计算行空间/零空间分解
        categories = {
            "size_compare": size_compare_resids[li],
            "size_reverse": size_reverse_resids[li],
            "weight_compare": weight_compare_resids[li],
            "speed_compare": speed_compare_resids[li],
            "neutral": neutral_resids[li],
            "size_attr": size_attr_resids[li],
            "weight_attr": weight_attr_resids[li],
            "speed_attr": speed_attr_resids[li],
        }
        for hab in ["land", "ocean", "sky"]:
            categories[f"habitat_{hab}"] = habitat_resids[li][hab]
        
        for cat_name, resids in categories.items():
            if len(resids) < 3:
                continue
            
            resid_arr = np.array(resids)  # [n, d_model]
            mean_resid = np.mean(resid_arr, axis=0)
            
            # 分解均值残差
            proj_row, coeffs_row = project_to_subspace(mean_resid, U_wu)
            proj_null = mean_resid - proj_row
            
            norm_total = np.linalg.norm(mean_resid)
            norm_row = np.linalg.norm(proj_row)
            norm_null = np.linalg.norm(proj_null)
            row_ratio = (norm_row ** 2) / max(norm_total ** 2, 1e-20)
            
            # 对每个残差也做分解, 计算统计
            row_ratios = []
            for r in resids:
                pr, _ = project_to_subspace(r, U_wu)
                pn = r - pr
                n_t = np.linalg.norm(r) ** 2
                n_r = np.linalg.norm(pr) ** 2
                row_ratios.append(n_r / max(n_t, 1e-20))
            
            layer_results[cat_name] = {
                "n_samples": len(resids),
                "mean_row_ratio": round(float(np.mean(row_ratios)), 4),
                "std_row_ratio": round(float(np.std(row_ratios)), 4),
                "mean_resid_norm": round(float(norm_total), 4),
                "proj_row_norm": round(float(norm_row), 4),
                "proj_null_norm": round(float(norm_null), 4),
            }
        
        resid_decompose[f"L{li}"] = layer_results
    
    results["resid_decompose"] = resid_decompose
    
    # ====================================================================
    # Part 3: 语义→logit路径
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: 语义→logit路径 — bigger/smaller logit由哪部分驱动?")
    log("="*60)
    
    # 关键问题: bigger/smaller的logit差异来自残差的哪部分?
    # 对于"A is bigger than B", 残差 = proj_row + proj_null
    # logit_bigger = W_U[bigger_id] @ resid / ||resid||
    #              = W_U[bigger_id] @ proj_row / ||resid||  (因为W_U@proj_null≈0)
    # 所以: logit差 = W_U[bigger_id] @ proj_row - W_U[smaller_id] @ proj_row
    
    logit_path = {}
    
    for li in test_layers:
        if len(size_compare_resids[li]) < 3 or len(neutral_resids[li]) < 3:
            continue
        
        # 取比较残差和中性残差的均值
        compare_mean = np.mean(size_compare_resids[li], axis=0)
        neutral_mean = np.mean(neutral_resids[li], axis=0)
        
        # 分解
        proj_row_compare, _ = project_to_subspace(compare_mean, U_wu)
        proj_null_compare = compare_mean - proj_row_compare
        proj_row_neutral, _ = project_to_subspace(neutral_mean, U_wu)
        proj_null_neutral = neutral_mean - proj_row_neutral
        
        # logit计算: W_U[target_id] @ residual (对归一化残差)
        def compute_logits(resid, target_ids_list):
            """计算残差对各target token的logit"""
            logits = {}
            for name, ids in target_ids_list:
                if len(ids) == 0:
                    continue
                w_dirs = W_U[ids]  # [n_tokens, d_model]
                logit_vals = w_dirs @ resid
                logits[name] = {
                    "mean": round(float(np.mean(logit_vals)), 4),
                    "std": round(float(np.std(logit_vals)), 4),
                }
            return logits
        
        target_ids_list = [
            ("bigger", bigger_ids), ("smaller", smaller_ids),
            ("heavier", heavier_ids), ("lighter", lighter_ids),
            ("faster", faster_ids), ("slower", slower_ids),
        ]
        
        # 完整残差的logit
        logits_full_compare = compute_logits(compare_mean, target_ids_list)
        logits_full_neutral = compute_logits(neutral_mean, target_ids_list)
        
        # 只有行空间分量的logit
        logits_row_compare = compute_logits(proj_row_compare, target_ids_list)
        logits_row_neutral = compute_logits(proj_row_neutral, target_ids_list)
        
        # 只有零空间分量的logit
        logits_null_compare = compute_logits(proj_null_compare, target_ids_list)
        
        # ★★★ 关键指标: logit差(bigger-smaller)来自行空间还是零空间?
        bigger_smaller_diff_full = logits_full_compare.get("bigger", {}).get("mean", 0) - logits_full_compare.get("smaller", {}).get("mean", 0)
        bigger_smaller_diff_row = logits_row_compare.get("bigger", {}).get("mean", 0) - logits_row_compare.get("smaller", {}).get("mean", 0)
        bigger_smaller_diff_null = logits_null_compare.get("bigger", {}).get("mean", 0) - logits_null_compare.get("smaller", {}).get("mean", 0)
        
        # 比较prompt vs 中性prompt的行空间logit差
        bigger_logit_compare_vs_neutral = (logits_full_compare.get("bigger", {}).get("mean", 0) - 
                                           logits_full_neutral.get("bigger", {}).get("mean", 0))
        smaller_logit_compare_vs_neutral = (logits_full_compare.get("smaller", {}).get("mean", 0) - 
                                            logits_full_neutral.get("smaller", {}).get("mean", 0))
        
        bigger_logit_row_compare_vs_neutral = (logits_row_compare.get("bigger", {}).get("mean", 0) - 
                                               logits_row_neutral.get("bigger", {}).get("mean", 0))
        
        logit_path[f"L{li}"] = {
            "full_logits_compare": logits_full_compare,
            "full_logits_neutral": logits_full_neutral,
            "row_logits_compare": logits_row_compare,
            "null_logits_compare": logits_null_compare,
            "bigger_smaller_diff": {
                "full": round(bigger_smaller_diff_full, 4),
                "row_only": round(bigger_smaller_diff_row, 4),
                "null_only": round(bigger_smaller_diff_null, 4),
                "row_fraction": round(bigger_smaller_diff_row / max(abs(bigger_smaller_diff_full), 1e-6), 4),
            },
            "compare_vs_neutral": {
                "bigger_logit_diff_full": round(bigger_logit_compare_vs_neutral, 4),
                "smaller_logit_diff_full": round(smaller_logit_compare_vs_neutral, 4),
                "bigger_logit_diff_row": round(bigger_logit_row_compare_vs_neutral, 4),
            },
        }
    
    results["logit_path"] = logit_path
    
    # ====================================================================
    # Part 3b: 用不同SVD分量数重新验证"bigger方向在W_U零空间"
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3b: bigger方向在W_U行空间的投影比 — 不同SVD分量数")
    log("="*60)
    
    dir_projection_verify = {}
    
    for li in test_layers:
        if len(size_compare_resids[li]) < 3:
            continue
        
        # 用habitat中性基线计算bigger方向
        hab_all = []
        for hab in ["land", "ocean", "sky"]:
            hab_all.extend(habitat_resids[li].get(hab, []))
        if len(hab_all) < 6:
            continue
        baseline_mean = np.mean(hab_all, axis=0)
        
        size_mean = np.mean(size_compare_resids[li], axis=0)
        bigger_dir = size_mean - baseline_mean
        bigger_norm = np.linalg.norm(bigger_dir)
        if bigger_norm < 1e-10:
            continue
        bigger_dir = bigger_dir / bigger_norm
        
        # 同样计算weight方向
        weight_dir = None
        if len(weight_compare_resids[li]) >= 3:
            weight_mean = np.mean(weight_compare_resids[li], axis=0)
            weight_dir_raw = weight_mean - baseline_mean
            weight_norm = np.linalg.norm(weight_dir_raw)
            if weight_norm > 1e-10:
                weight_dir = weight_dir_raw / weight_norm
        
        # speed方向
        speed_dir = None
        if len(speed_compare_resids[li]) >= 3:
            speed_mean = np.mean(speed_compare_resids[li], axis=0)
            speed_dir_raw = speed_mean - baseline_mean
            speed_norm = np.linalg.norm(speed_dir_raw)
            if speed_norm > 1e-10:
                speed_dir = speed_dir_raw / speed_norm
        
        layer_verify = {}
        for n_comp, U_basis in wu_bases.items():
            proj_b, _ = project_to_subspace(bigger_dir, U_basis)
            ratio_bigger = np.linalg.norm(proj_b) ** 2
            
            result_entry = {"bigger_in_row_ratio": round(float(ratio_bigger), 4)}
            
            if weight_dir is not None:
                proj_w, _ = project_to_subspace(weight_dir, U_basis)
                result_entry["weight_in_row_ratio"] = round(float(np.linalg.norm(proj_w) ** 2), 4)
                result_entry["cos_bigger_weight"] = round(float(np.dot(bigger_dir, weight_dir)), 4)
            
            if speed_dir is not None:
                proj_s, _ = project_to_subspace(speed_dir, U_basis)
                result_entry["speed_in_row_ratio"] = round(float(np.linalg.norm(proj_s) ** 2), 4)
            
            # 计算方向之间的余弦
            if weight_dir is not None and speed_dir is not None:
                result_entry["cos_bigger_speed"] = round(float(np.dot(bigger_dir, speed_dir)), 4)
                result_entry["cos_weight_speed"] = round(float(np.dot(weight_dir, speed_dir)), 4)
            
            layer_verify[f"n{n_comp}"] = result_entry
        
        dir_projection_verify[f"L{li}"] = layer_verify
    
    results["dir_projection_verify"] = dir_projection_verify
    
    # ====================================================================
    # Part 4: 推理-语义正交性精确刻画
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: 推理-语义正交性精确刻画")
    log("="*60)
    
    orthogonality = {}
    
    for li in test_layers:
        # 构建语义子空间: habitat分类方向
        hab_resids_all = []
        hab_labels = []
        for hab in ["land", "ocean", "sky"]:
            r = habitat_resids[li].get(hab, [])
            if len(r) >= 3:
                hab_resids_all.extend(r)
                hab_labels.extend([hab] * len(r))
        
        # 构建推理子空间: 比较操作方向
        compare_resids_all = []
        compare_labels = []
        for r in size_compare_resids[li]:
            compare_resids_all.append(r)
            compare_labels.append("size")
        for r in weight_compare_resids[li]:
            compare_resids_all.append(r)
            compare_labels.append("weight")
        for r in speed_compare_resids[li]:
            compare_resids_all.append(r)
            compare_labels.append("speed")
        # 也加反向比较
        for r in size_reverse_resids[li]:
            compare_resids_all.append(r)
            compare_labels.append("size_rev")
        
        if len(hab_resids_all) < 10 or len(compare_resids_all) < 10:
            continue
        
        hab_arr = np.array(hab_resids_all)  # [n_hab, d_model]
        comp_arr = np.array(compare_resids_all)  # [n_comp, d_model]
        
        # PCA: Vt的行是主成分方向(在d_model空间中)
        hab_centered = hab_arr - hab_arr.mean(axis=0)
        comp_centered = comp_arr - comp_arr.mean(axis=0)
        
        _, S_hab, Vt_hab = np.linalg.svd(hab_centered, full_matrices=False)
        _, S_comp, Vt_comp = np.linalg.svd(comp_centered, full_matrices=False)
        # Vt_hab: [min(n,d), d_model], Vt_comp: [min(n,d), d_model]
        
        # 有效维度 (90%方差)
        hab_energy = np.cumsum(S_hab**2) / np.sum(S_hab**2)
        comp_energy = np.cumsum(S_comp**2) / np.sum(S_comp**2)
        
        hab_eff_dim_90 = int(np.searchsorted(hab_energy, 0.90)) + 1
        comp_eff_dim_90 = int(np.searchsorted(comp_energy, 0.90)) + 1
        hab_eff_dim_99 = int(np.searchsorted(hab_energy, 0.99)) + 1
        comp_eff_dim_99 = int(np.searchsorted(comp_energy, 0.99)) + 1
        
        # 子空间重叠 (Grassmann距离)
        # 取有效维度的PCA分量(行=主方向在d_model空间)
        n_hab = min(hab_eff_dim_99, Vt_hab.shape[0])
        n_comp = min(comp_eff_dim_99, Vt_comp.shape[0])
        
        # 子空间基: Vt[:n, :].T = [d_model, n]
        basis_hab = Vt_hab[:n_hab].T  # [d_model, n_hab]
        basis_comp = Vt_comp[:n_comp].T  # [d_model, n_comp]
        
        # 子空间重叠: basis_hab^T @ basis_comp → [n_hab, n_comp]
        overlap_matrix = basis_hab.T @ basis_comp  # [n_hab, n_comp]
        sv_overlap = np.linalg.svd(overlap_matrix, compute_uv=False)
        
        # Principal angles
        principal_angles = np.arccos(np.clip(sv_overlap, 0, 1))
        
        # 子空间重叠度 (0=正交, 1=完全重合)
        subspace_overlap = float(np.mean(sv_overlap ** 2))
        
        # 语义方差在推理子空间的投影
        hab_var_in_comp = float(np.mean([
            np.linalg.norm(project_to_subspace(r, basis_comp)[0]) ** 2 / max(np.linalg.norm(r) ** 2, 1e-20)
            for r in hab_centered[:20]  # 采样20个
        ]))
        
        # 推理方差在语义子空间的投影
        comp_var_in_hab = float(np.mean([
            np.linalg.norm(project_to_subspace(r, basis_hab)[0]) ** 2 / max(np.linalg.norm(r) ** 2, 1e-20)
            for r in comp_centered[:20]
        ]))
        
        orthogonality[f"L{li}"] = {
            "habitat_eff_dim_90": hab_eff_dim_90,
            "habitat_eff_dim_99": hab_eff_dim_99,
            "compare_eff_dim_90": comp_eff_dim_90,
            "compare_eff_dim_99": comp_eff_dim_99,
            "subspace_overlap": round(subspace_overlap, 4),
            "principal_angles_deg_top5": [round(float(a * 180 / np.pi), 1) for a in principal_angles[:5]],
            "hab_var_in_compare_subspace": round(hab_var_in_comp, 4),
            "compare_var_in_habitat_subspace": round(comp_var_in_hab, 4),
            "n_hab_samples": len(hab_resids_all),
            "n_compare_samples": len(compare_resids_all),
        }
        
        log(f"  L{li}: hab_dim={hab_eff_dim_99}, comp_dim={comp_eff_dim_99}, "
            f"overlap={subspace_overlap:.4f}, hab→comp={hab_var_in_comp:.4f}, comp→hab={comp_var_in_hab:.4f}")
    
    results["orthogonality"] = orthogonality
    
    # ====================================================================
    # Part 5: 层间推理方向一致性 (用中性基线)
    # ====================================================================
    log("\n" + "="*60)
    log("Part 5: 层间推理方向一致性 + 跨属性方向关系")
    log("="*60)
    
    direction_consistency = {}
    
    for li in test_layers:
        hab_all = []
        for hab in ["land", "ocean", "sky"]:
            hab_all.extend(habitat_resids[li].get(hab, []))
        if len(hab_all) < 6:
            continue
        baseline_mean = np.mean(hab_all, axis=0)
        
        # 各类方向(用中性基线)
        dirs = {}
        
        if len(size_compare_resids[li]) >= 3:
            size_mean = np.mean(size_compare_resids[li], axis=0)
            d = size_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["size_compare"] = d / n
        
        if len(size_reverse_resids[li]) >= 3:
            size_rev_mean = np.mean(size_reverse_resids[li], axis=0)
            d = size_rev_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["size_reverse"] = d / n
        
        if len(weight_compare_resids[li]) >= 3:
            weight_mean = np.mean(weight_compare_resids[li], axis=0)
            d = weight_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["weight_compare"] = d / n
        
        if len(speed_compare_resids[li]) >= 3:
            speed_mean = np.mean(speed_compare_resids[li], axis=0)
            d = speed_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["speed_compare"] = d / n
        
        if len(size_attr_resids[li]) >= 3:
            attr_mean = np.mean(size_attr_resids[li], axis=0)
            d = attr_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["size_attr"] = d / n
        
        if len(weight_attr_resids[li]) >= 3:
            attr_mean = np.mean(weight_attr_resids[li], axis=0)
            d = attr_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["weight_attr"] = d / n
        
        if len(speed_attr_resids[li]) >= 3:
            attr_mean = np.mean(speed_attr_resids[li], axis=0)
            d = attr_mean - baseline_mean
            n = np.linalg.norm(d)
            if n > 1e-10:
                dirs["speed_attr"] = d / n
        
        # 方向间余弦
        cos_matrix = {}
        dir_names = list(dirs.keys())
        for i, n1 in enumerate(dir_names):
            for j, n2 in enumerate(dir_names):
                if j >= i:
                    c = float(np.dot(dirs[n1], dirs[n2]))
                    cos_matrix[f"cos({n1},{n2})"] = round(c, 4)
        
        # ★★★ 关键: 比较操作方向 = compare - attr
        # size_compare = size_attr + operation_size
        # 所以 operation_size = size_compare - size_attr (但在方向空间中)
        if "size_compare" in dirs and "size_attr" in dirs:
            op_size = dirs["size_compare"] - dirs["size_attr"] * np.dot(dirs["size_compare"], dirs["size_attr"])
            n = np.linalg.norm(op_size)
            if n > 1e-10:
                op_size = op_size / n
            else:
                op_size = None
        else:
            op_size = None
        
        if "weight_compare" in dirs and "weight_attr" in dirs:
            op_weight = dirs["weight_compare"] - dirs["weight_attr"] * np.dot(dirs["weight_compare"], dirs["weight_attr"])
            n = np.linalg.norm(op_weight)
            if n > 1e-10:
                op_weight = op_weight / n
            else:
                op_weight = None
        else:
            op_weight = None
        
        if "speed_compare" in dirs and "speed_attr" in dirs:
            op_speed = dirs["speed_compare"] - dirs["speed_attr"] * np.dot(dirs["speed_compare"], dirs["speed_attr"])
            n = np.linalg.norm(op_speed)
            if n > 1e-10:
                op_speed = op_speed / n
            else:
                op_speed = None
        else:
            op_speed = None
        
        # 操作方向之间的余弦
        op_cos = {}
        if op_size is not None and op_weight is not None:
            op_cos["cos(op_size,op_weight)"] = round(float(np.dot(op_size, op_weight)), 4)
        if op_size is not None and op_speed is not None:
            op_cos["cos(op_size,op_speed)"] = round(float(np.dot(op_size, op_speed)), 4)
        if op_weight is not None and op_speed is not None:
            op_cos["cos(op_weight,op_speed)"] = round(float(np.dot(op_weight, op_speed)), 4)
        
        # ★★★ 比较方向 vs 反向比较方向
        if "size_compare" in dirs and "size_reverse" in dirs:
            cos_fwd_rev = float(np.dot(dirs["size_compare"], dirs["size_reverse"]))
            op_cos["cos(size_compare,size_reverse)"] = round(cos_fwd_rev, 4)
        
        direction_consistency[f"L{li}"] = {
            "direction_cosines": cos_matrix,
            "operation_cosines": op_cos,
        }
    
    results["direction_consistency"] = direction_consistency
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    out_file = TEMP / f"ccxxxii_resid_wu_decompose_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, 
                    "n_wu_components": n_main, "results": results}, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到 {out_file}")
    
    # 打印关键摘要
    log("\n" + "="*60)
    log("关键摘要")
    log("="*60)
    
    log("\n1. W_U行空间覆盖率:")
    for n_comp, cr in coverage_results.items():
        log(f"   {n_comp} components: {cr['coverage']:.4f}")
    
    log("\n2. 残差在W_U行空间的投影比:")
    for li in test_layers:
        layer_data = resid_decompose.get(f"L{li}", {})
        parts = []
        for cat in ["size_compare", "neutral", "habitat_land", "habitat_ocean", "habitat_sky"]:
            if cat in layer_data:
                parts.append(f"{cat}={layer_data[cat]['mean_row_ratio']:.4f}")
        log(f"   L{li}: {', '.join(parts)}")
    
    log("\n3. bigger/smaller logit差异来源:")
    for li in test_layers:
        lp = logit_path.get(f"L{li}", {})
        bsd = lp.get("bigger_smaller_diff", {})
        log(f"   L{li}: full={bsd.get('full',0):.4f}, row={bsd.get('row_only',0):.4f}, "
            f"row_fraction={bsd.get('row_fraction',0):.4f}")
    
    log("\n4. 推理-语义正交性:")
    for li in test_layers:
        o = orthogonality.get(f"L{li}", {})
        if o:
            log(f"   L{li}: overlap={o['subspace_overlap']:.4f}, hab_dim={o['habitat_eff_dim_99']}, "
                f"comp_dim={o['compare_eff_dim_99']}, hab→comp={o['hab_var_in_compare_subspace']:.4f}")
    
    log("\n5. bigger方向在W_U行空间投影比(不同分量数):")
    for li in test_layers:
        dpv = dir_projection_verify.get(f"L{li}", {})
        parts = []
        for nc in [50, 100, 200, 500]:
            key = f"n{nc}"
            if key in dpv:
                parts.append(f"n{nc}={dpv[key]['bigger_in_row_ratio']:.4f}")
        log(f"   L{li}: {', '.join(parts)}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"CCXXXII(332): 残差W_U分解 + 语义→logit路径 - {args.model}\n")
    
    run(args.model)
