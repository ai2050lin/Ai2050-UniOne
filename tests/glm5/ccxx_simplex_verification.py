"""
CCXX(320): 单纯形精确验证 — 12/15/20关系 + 非均匀关系共享顶点测试
======================================================================
CCXVII发现N关系→N-1维(浅层), 但:
1. 只有4/6/8/10四个数据点, 缩放律不精确
2. 是否某些关系共享顶点→维度<n_rel-1?
3. 20+关系时dim是否继续≈n_rel-1?

设计:
  - 扩展到12/15/20种关系
  - 精确拟合缩放律: dim = a*n_rel + b
  - 测试非均匀关系组(如4个habitat子关系是否只贡献2-3维)
  - 浅层+深层对比

关系列表(20种):
  animal_land, animal_ocean, animal_sky,  (3 habitat)
  fruit, vegetable, grain,                 (3 food type)
  vehicle, tool, furniture,                (3 man-made)
  metal, wood, fabric,                     (3 material)
  hot, cold, warm,                         (3 temperature)
  big, small, medium                       (3 size)

非均匀子组测试:
  - 3 habitat alone → 2维? (3类→2维四面体)
  - 3 food + 3 habitat → 5维? (6类→5维)
  - 共享顶点测试: 如果"animal_land"和"animal_ocean"共享"animal"维度,
    则3 habitat子关系的维度可能<2

用法:
  python ccxx_simplex_verification.py --model qwen3
  python ccxx_simplex_verification.py --model glm4
  python ccxx_simplex_verification.py --model deepseek7b
"""
import argparse, os, sys, json, gc
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
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxx_simplex_verification_log.txt"

# ===== 词汇和关系定义 =====
# 每种关系: 关键词 → 一组具有该关系的词
RELATION_WORDS = {
    # Habitat (3类)
    "animal_land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer", "bear", "rabbit"],
    "animal_ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "seal", "crab", "squid", "lobster", "turtle"],
    "animal_sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "falcon", "swallow", "pigeon", "robin"],
    # Food type (3类)
    "food_fruit": ["apple", "banana", "mango", "cherry", "peach", "grape", "lemon", "orange", "pear", "plum"],
    "food_vegetable": ["carrot", "potato", "onion", "pepper", "tomato", "cabbage", "celery", "lettuce", "pea", "bean"],
    "food_grain": ["rice", "wheat", "corn", "oat", "barley", "millet", "rye", "sorghum", "quinoa", "buckwheat"],
    # Man-made (3类)
    "object_vehicle": ["car", "truck", "bus", "train", "plane", "boat", "bicycle", "motorcycle", "helicopter", "van"],
    "object_tool": ["hammer", "knife", "drill", "wrench", "saw", "axe", "chisel", "pliers", "screwdriver", "ruler"],
    "object_furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "wardrobe", "bench", "stool"],
    # Material (3类)
    "material_metal": ["iron", "steel", "copper", "gold", "silver", "aluminum", "brass", "bronze", "lead", "tin"],
    "material_wood": ["oak", "pine", "cedar", "maple", "birch", "elm", "ash", "walnut", "cherry_wood", "bamboo"],
    "material_fabric": ["silk", "cotton", "wool", "linen", "velvet", "nylon", "polyester", "satin", "denim", "leather"],
    # Temperature (3类)
    "temp_hot": ["fire", "lava", "sun", "furnace", "oven", "volcano", "desert", "summer", "boiling", "scorching"],
    "temp_cold": ["ice", "snow", "frost", "glacier", "winter", "arctic", "freezing", "frozen", "frostbite", "frigid"],
    "temp_warm": ["spring", "cozy", "blanket", "hearth", "sunshine", "breeze", "mild", "gentle", "comfort", "lukewarm"],
    # Size (3类)
    "size_big": ["mountain", "ocean", "elephant", "continent", "planet", "building", "whale", "tree", "castle", "glacier"],
    "size_small": ["ant", "grain", "dust", "pebble", "drop", "atom", "cell", "molecule", "speck", "crumb"],
    "size_medium": ["dog", "chair", "book", "table", "human", "door", "window", "car", "tree_bush", "basket"],
    # Color (2类 - 额外)
    "color_warm": ["red", "orange", "yellow", "gold", "crimson", "scarlet", "amber", "rust", "coral", "salmon"],
    "color_cool": ["blue", "green", "purple", "teal", "indigo", "violet", "cyan", "navy", "azure", "turquoise"],
}

# 关系分组 - 用于子组测试
RELATION_GROUPS = {
    "habitat": ["animal_land", "animal_ocean", "animal_sky"],
    "food": ["food_fruit", "food_vegetable", "food_grain"],
    "object": ["object_vehicle", "object_tool", "object_furniture"],
    "material": ["material_metal", "material_wood", "material_fabric"],
    "temperature": ["temp_hot", "temp_cold", "temp_warm"],
    "size": ["size_big", "size_small", "size_medium"],
    "color": ["color_warm", "color_cool"],
}

# 模板
TEMPLATE = "The {} is"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_residuals(model, tokenizer, device, layers, word, test_layers):
    """收集词在指定层的残差"""
    prompt = TEMPLATE.format(word)
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


def compute_effective_dim(residuals_by_class, n_classes, F_threshold=10.0):
    """
    计算联合子空间的有效维度
    residuals_by_class: {class_name: [residual_vectors]}
    用ANOVA F值判断SVD模式中哪些显著区分类
    """
    # 收集所有残差
    all_vecs = []
    labels = []
    for ci, (cls_name, vecs) in enumerate(residuals_by_class.items()):
        all_vecs.extend(vecs)
        labels.extend([ci] * len(vecs))
    
    if len(all_vecs) < n_classes + 1:
        return 0, []
    
    X = np.array(all_vecs)
    labels = np.array(labels)
    n_samples, d = X.shape
    
    # 全局均值
    grand_mean = X.mean(axis=0)
    
    # 组内/组间
    SS_between = np.zeros(d)
    SS_within = np.zeros(d)
    for ci in range(n_classes):
        mask = labels == ci
        if mask.sum() == 0:
            continue
        group_mean = X[mask].mean(axis=0)
        SS_between += mask.sum() * (group_mean - grand_mean) ** 2
        SS_within += ((X[mask] - group_mean) ** 2).sum(axis=0)
    
    # 沿SS_between主方向做SVD
    # 使用残差矩阵: 每类均值-全局均值
    class_means = np.zeros((n_classes, d))
    for ci in range(n_classes):
        mask = labels == ci
        if mask.sum() > 0:
            class_means[ci] = X[mask].mean(axis=0) - grand_mean
    
    # SVD of class mean matrix
    U, S, Vt = np.linalg.svd(class_means, full_matrices=False)
    
    # 对每个SVD模式计算ANOVA F
    significant_modes = 0
    mode_F_values = []
    for k in range(min(len(S), n_classes - 1)):
        mode_dir = Vt[k]  # [d]
        # 投影
        proj = X @ mode_dir
        # One-way ANOVA
        groups = [proj[labels == ci] for ci in range(n_classes) if (labels == ci).sum() > 1]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        mode_F_values.append(float(F))
        if F > F_threshold:
            significant_modes += 1
    
    return significant_modes, mode_F_values


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    log(f"\n{'='*70}\nCCXX(320): 单纯形精确验证 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"  test_layers={test_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # ===== Step 1: 收集所有词的残差 =====
    log("\n--- Step 1: 收集残差 ---")
    all_residuals = {}  # {rel_name: {layer: [vec1, vec2, ...]}}
    
    for rel_name, words in RELATION_WORDS.items():
        all_residuals[rel_name] = {li: [] for li in test_layers}
        for word in words:
            try:
                resid = collect_residuals(model, tokenizer, device, layers, word, test_layers)
                for li in test_layers:
                    all_residuals[rel_name][li].append(resid[f"L{li}"])
            except Exception as e:
                log(f"  跳过 {word}: {e}")
    
    log(f"  收集了 {len(all_residuals)} 种关系")
    
    # ===== Step 2: 精确缩放律 - 3/6/9/12/15/18/20关系 =====
    log("\n--- Step 2: 精确缩放律 ---")
    
    all_rel_names = list(RELATION_WORDS.keys())
    
    # 按组排列, 确保每组关系均匀加入
    group_order = ["habitat", "food", "object", "material", "temperature", "size", "color"]
    ordered_rels = []
    for g in group_order:
        ordered_rels.extend(RELATION_GROUPS[g])
    
    # 测试不同关系数
    n_rel_tests = [3, 6, 9, 12, 15, 18, 20]
    
    for li in test_layers:
        log(f"\n  Layer {li}:")
        for n_rel in n_rel_tests:
            if n_rel > len(ordered_rels):
                continue
            
            rel_subset = ordered_rels[:n_rel]
            
            # 构建残差矩阵
            resid_by_class = {}
            for rel_name in rel_subset:
                vecs = all_residuals[rel_name][li]
                if len(vecs) >= 2:
                    resid_by_class[rel_name] = vecs
            
            if len(resid_by_class) < n_rel:
                log(f"    n_rel={n_rel}: 跳过(数据不足)")
                continue
            
            eff_dim, mode_F = compute_effective_dim(resid_by_class, n_rel, F_threshold=10.0)
            eff_dim_F100, _ = compute_effective_dim(resid_by_class, n_rel, F_threshold=100.0)
            
            log(f"    n_rel={n_rel}: dim_F10={eff_dim}, dim_F100={eff_dim_F100}, ratio={eff_dim/n_rel:.3f}")
            
            key = f"scaling_{n_rel}rel_L{li}"
            results[key] = {
                "n_relations": n_rel,
                "layer": li,
                "eff_dim_F10": eff_dim,
                "eff_dim_F100": eff_dim_F100,
                "ratio_F10": round(eff_dim / n_rel, 4),
                "top_F_values": [round(f, 2) for f in mode_F[:10]],
            }
    
    # ===== Step 3: 线性回归拟合缩放律 =====
    log("\n--- Step 3: 缩放律拟合 ---")
    for li in test_layers:
        x_data = []
        y_data = []
        for n_rel in n_rel_tests:
            key = f"scaling_{n_rel}rel_L{li}"
            if key in results:
                x_data.append(n_rel)
                y_data.append(results[key]["eff_dim_F10"])
        
        if len(x_data) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            log(f"  L{li}: dim = {slope:.3f}*n_rel + {intercept:.3f} (R²={r_value**2:.4f}, p={p_value:.6f})")
            results[f"scaling_law_L{li}"] = {
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r_squared": round(r_value ** 2, 4),
                "p_value": round(p_value, 6),
            }
    
    # ===== Step 4: 非均匀子组测试 - 共享顶点 =====
    log("\n--- Step 4: 共享顶点测试 ---")
    
    # 单独测试每个3关系组
    for group_name, rel_list in RELATION_GROUPS.items():
        if len(rel_list) < 2:
            continue
        for li in [0, n_layers - 1]:
            resid_by_class = {}
            for rel_name in rel_list:
                vecs = all_residuals[rel_name][li]
                if len(vecs) >= 2:
                    resid_by_class[rel_name] = vecs
            
            if len(resid_by_class) < 2:
                continue
            
            eff_dim, mode_F = compute_effective_dim(resid_by_class, len(resid_by_class), F_threshold=10.0)
            expected = len(resid_by_class) - 1  # 单纯形期望维度
            deficit = expected - eff_dim  # 如果>0, 说明有共享顶点
            
            log(f"  {group_name} L{li}: n_rel={len(resid_by_class)}, dim={eff_dim}, expected={expected}, deficit={deficit}")
            
            key = f"subgroup_{group_name}_L{li}"
            results[key] = {
                "group": group_name,
                "n_relations": len(resid_by_class),
                "layer": li,
                "eff_dim_F10": eff_dim,
                "expected_dim": expected,
                "deficit": deficit,
                "top_F_values": [round(f, 2) for f in mode_F[:5]],
            }
    
    # ===== Step 5: 跨组组合测试 =====
    log("\n--- Step 5: 跨组组合测试 ---")
    
    # 测试不同组的组合
    combo_tests = [
        ("habitat+food", RELATION_GROUPS["habitat"] + RELATION_GROUPS["food"]),       # 6 rels
        ("habitat+food+object", RELATION_GROUPS["habitat"] + RELATION_GROUPS["food"] + RELATION_GROUPS["object"]),  # 9 rels
        ("3groups_any3", ["animal_land", "animal_ocean", "food_fruit"]),  # 3 rels from 2 groups
        ("2groups_2each", ["animal_land", "animal_ocean", "food_fruit", "food_vegetable"]),  # 4 rels from 2 groups
    ]
    
    for combo_name, rel_list in combo_tests:
        for li in [0, n_layers - 1]:
            resid_by_class = {}
            for rel_name in rel_list:
                if rel_name in all_residuals:
                    vecs = all_residuals[rel_name][li]
                    if len(vecs) >= 2:
                        resid_by_class[rel_name] = vecs
            
            if len(resid_by_class) < 2:
                continue
            
            eff_dim, mode_F = compute_effective_dim(resid_by_class, len(resid_by_class), F_threshold=10.0)
            expected = len(resid_by_class) - 1
            
            log(f"  {combo_name} L{li}: n_rel={len(resid_by_class)}, dim={eff_dim}, expected={expected}, deficit={expected-eff_dim}")
            
            key = f"combo_{combo_name}_L{li}"
            results[key] = {
                "combo": combo_name,
                "n_relations": len(resid_by_class),
                "layer": li,
                "eff_dim_F10": eff_dim,
                "expected_dim": expected,
                "deficit": expected - eff_dim,
            }
    
    # ===== Step 6: 核心判断 =====
    log("\n--- Step 6: 核心判断 ---")
    
    # 检查dim=n_rel-1是否精确
    simplex_checks = []
    for li in [0, n_layers - 1]:
        for n_rel in n_rel_tests:
            key = f"scaling_{n_rel}rel_L{li}"
            if key in results:
                r = results[key]
                if r["eff_dim_F10"] == n_rel - 1:
                    simplex_checks.append(True)
                else:
                    simplex_checks.append(False)
                    log(f"  L{li} n_rel={n_rel}: dim={r['eff_dim_F10']} ≠ {n_rel-1} (偏差={r['eff_dim_F10']-(n_rel-1)})")
    
    n_exact = sum(simplex_checks)
    n_total = len(simplex_checks)
    log(f"\n  dim=n_rel-1 精确匹配: {n_exact}/{n_total} ({100*n_exact/max(n_total,1):.1f}%)")
    
    # 检查deficit
    subgroup_deficits = []
    for group_name in RELATION_GROUPS:
        for li in [0, n_layers - 1]:
            key = f"subgroup_{group_name}_L{li}"
            if key in results:
                subgroup_deficits.append(results[key]["deficit"])
    
    if subgroup_deficits:
        mean_deficit = np.mean(subgroup_deficits)
        log(f"  子组deficit均值: {mean_deficit:.3f} (0=完美单纯形, >0=共享顶点)")
    
    results["core_judgment"] = {
        "simplex_match_rate": round(n_exact / max(n_total, 1), 4),
        "simplex_exact_count": n_exact,
        "simplex_total_count": n_total,
        "mean_subgroup_deficit": round(float(np.mean(subgroup_deficits)), 4) if subgroup_deficits else None,
        "conclusion": "dim=n_rel-1精确" if n_exact > n_total * 0.8 else "dim≈n_rel-1近似",
    }
    
    # 保存结果
    out_path = TEMP / f"ccxx_simplex_verification_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, "results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n结果保存到: {out_path}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    # 清除旧日志
    if args.model == "qwen3":
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("")
    
    run(args.model)
