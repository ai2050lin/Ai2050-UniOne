"""
CCXXVIII(328): size vs weight方向相反的精确分解
======================================================================
CCXXIII发现: size vs weight方向cos≈-0.75! "大"和"重"在语义空间中朝相反方向!
关键问题: 这是比较运算符(bigger/heavier)导致的, 还是属性内容(size/weight)导致的?

实验设计:
  1. "A is bigger than B" vs "A is heavier than B" - 分离比较词效果
  2. "A is big" vs "A is heavy" - 纯属性方向(无比较)
  3. "A is bigger and heavier than B" - 两个属性同时激活
  4. 同一比较词不同属性: "bigger" vs "heavier" 的比较词方向
  5. 不同比较词同一属性: "A is bigger" vs "A is larger" vs "A is greater"
  6. 属性方向vs比较方向的分解: Δ = α*属性方向 + β*比较方向

用法:
  python ccxxviii_size_weight_decompose.py --model qwen3
  python ccxxviii_size_weight_decompose.py --model glm4
  python ccxxviii_size_weight_decompose.py --model deepseek7b
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxviii_size_weight_decompose_log.txt"

# 词汇
ANIMAL_BIG = ["elephant", "whale", "horse", "lion", "bear", "cow", "shark", "tiger", "eagle", "bus"]
ANIMAL_SMALL = ["mouse", "fish", "cat", "rabbit", "fox", "chicken", "crab", "rat", "sparrow", "car"]

HEAVY_ITEMS = ["iron", "rock", "steel", "gold", "lead", "stone", "concrete", "brick", "copper", "platinum"]
LIGHT_ITEMS = ["feather", "leaf", "paper", "cotton", "silk", "grass", "foam", "straw", "wool", "air"]

FAST_ITEMS = ["cheetah", "falcon", "horse", "rocket", "jet", "leopard", "eagle", "tiger", "deer", "lightning"]
SLOW_ITEMS = ["turtle", "snail", "slug", "cart", "boat", "worm", "ant", "sloth", "beetle", "glacier"]

# 模板
TEMPLATES = {
    "compare_size": "The {} is bigger than the {}",
    "compare_weight": "The {} is heavier than the {}",
    "compare_speed": "The {} is faster than the {}",
    "attribute_size": "The {} is big",
    "attribute_weight": "The {} is heavy",
    "attribute_speed": "The {} is fast",
    "compare_size_larger": "The {} is larger than the {}",
    "compare_size_greater": "The {} is greater than the {}",
    "compare_size_huge": "The {} is huge compared to the {}",
    "compare_weight_dense": "The {} is denser than the {}",
    "compare_weight_massive": "The {} is more massive than the {}",
    "combo_size_weight": "The {} is bigger and heavier than the {}",
    "combo_size_speed": "The {} is bigger and faster than the {}",
    "neutral": "The {}",
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_residuals(model, tokenizer, device, layers, template, word_pairs, test_layers):
    """收集指定模板和词对的残差"""
    resids = {li: [] for li in test_layers}
    
    for w1, w2 in word_pairs:
        prompt = template.format(w1, w2)
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
        
        for li in test_layers:
            if f"L{li}" in captured:
                resids[li].append(captured[f"L{li}"])
    
    return resids


def collect_residuals_single(model, tokenizer, device, layers, template, words, test_layers):
    """收集单词模板的残差"""
    resids = {li: [] for li in test_layers}
    
    for w in words:
        prompt = template.format(w)
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
        
        for li in test_layers:
            if f"L{li}" in captured:
                resids[li].append(captured[f"L{li}"])
    
    return resids


def compute_direction(resids_li, exclude_indices=None):
    """计算残差集合的均值方向"""
    if exclude_indices:
        arr = [r for i, r in enumerate(resids_li) if i not in exclude_indices]
    else:
        arr = resids_li
    if len(arr) < 2:
        return None
    mean = np.mean(arr, axis=0)
    overall_mean = np.mean(resids_li, axis=0)
    direction = mean - overall_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    return direction / norm


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    test_layers = [0, n_layers // 2, n_layers - 1]
    
    log(f"\n{'='*70}\nCCXXVIII(328): size vs weight方向相反的精确分解 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # ===== Step 1: 比较方向 vs 属性方向 =====
    log("\n--- Step 1: 比较方向 vs 属性方向 ---")
    
    # 1a: 比较句方向
    size_pairs = list(zip(ANIMAL_BIG, ANIMAL_SMALL))
    weight_pairs = list(zip(HEAVY_ITEMS, LIGHT_ITEMS))
    speed_pairs = list(zip(FAST_ITEMS, SLOW_ITEMS))
    
    resids_compare_size = collect_residuals(model, tokenizer, device, layers, 
                                             TEMPLATES["compare_size"], size_pairs, test_layers)
    resids_compare_weight = collect_residuals(model, tokenizer, device, layers,
                                               TEMPLATES["compare_weight"], weight_pairs, test_layers)
    resids_compare_speed = collect_residuals(model, tokenizer, device, layers,
                                              TEMPLATES["compare_speed"], speed_pairs, test_layers)
    
    log(f"  比较句残差收集完成: size={len(resids_compare_size[test_layers[0]])}, "
        f"weight={len(resids_compare_weight[test_layers[0]])}, "
        f"speed={len(resids_compare_speed[test_layers[0]])}")
    
    # 1b: 属性句方向(无比较)
    all_animals = ANIMAL_BIG + ANIMAL_SMALL
    all_materials = HEAVY_ITEMS + LIGHT_ITEMS
    
    resids_attr_size = collect_residuals_single(model, tokenizer, device, layers,
                                                 TEMPLATES["attribute_size"], all_animals, test_layers)
    resids_attr_weight = collect_residuals_single(model, tokenizer, device, layers,
                                                   TEMPLATES["attribute_weight"], all_materials, test_layers)
    
    log(f"  属性句残差收集完成: size={len(resids_attr_size[test_layers[0]])}, "
        f"weight={len(resids_attr_weight[test_layers[0]])}")
    
    # 1c: 对比比较方向vs属性方向
    for li in test_layers:
        # 比较方向
        size_comp_dir = compute_direction(resids_compare_size[li])
        weight_comp_dir = compute_direction(resids_compare_weight[li])
        speed_comp_dir = compute_direction(resids_compare_speed[li])
        
        # 属性方向
        size_attr_dir = compute_direction(resids_attr_size[li])
        weight_attr_dir = compute_direction(resids_attr_weight[li])
        
        # 余弦比较
        cosines = {}
        for name1, dir1 in [("size_comp", size_comp_dir), ("weight_comp", weight_comp_dir), 
                            ("speed_comp", speed_comp_dir),
                            ("size_attr", size_attr_dir), ("weight_attr", weight_attr_dir)]:
            if dir1 is None:
                continue
            for name2, dir2 in [("size_comp", size_comp_dir), ("weight_comp", weight_comp_dir),
                                ("speed_comp", speed_comp_dir),
                                ("size_attr", size_attr_dir), ("weight_attr", weight_attr_dir)]:
                if dir2 is None:
                    continue
                if name1 < name2:
                    cos_val = float(np.dot(dir1, dir2))
                    cosines[f"cos({name1},{name2})"] = round(cos_val, 4)
        
        results[f"compare_vs_attribute_L{li}"] = {
            "layer": li,
            "cosines": cosines,
        }
        
        log(f"  L{li}:")
        for k, v in sorted(cosines.items()):
            log(f"    {k} = {v}")
    
    # ===== Step 2: 比较词分解 =====
    log("\n--- Step 2: 比较词分解 ---")
    
    # 同一属性, 不同比较词
    resids_larger = collect_residuals(model, tokenizer, device, layers,
                                       TEMPLATES["compare_size_larger"], size_pairs, test_layers)
    resids_greater = collect_residuals(model, tokenizer, device, layers,
                                        TEMPLATES["compare_size_greater"], size_pairs, test_layers)
    resids_dense = collect_residuals(model, tokenizer, device, layers,
                                      TEMPLATES["compare_weight_dense"], weight_pairs, test_layers)
    resids_massive = collect_residuals(model, tokenizer, device, layers,
                                        TEMPLATES["compare_weight_massive"], weight_pairs, test_layers)
    
    for li in test_layers:
        bigger_dir = compute_direction(resids_compare_size[li])
        larger_dir = compute_direction(resids_larger[li])
        greater_dir = compute_direction(resids_greater[li])
        heavier_dir = compute_direction(resids_compare_weight[li])
        denser_dir = compute_direction(resids_dense[li])
        massive_dir = compute_direction(resids_massive[li])
        
        comp_word_cos = {}
        for name1, dir1 in [("bigger", bigger_dir), ("larger", larger_dir), ("greater", greater_dir),
                            ("heavier", heavier_dir), ("denser", denser_dir), ("massive", massive_dir)]:
            if dir1 is None:
                continue
            for name2, dir2 in [("bigger", bigger_dir), ("larger", larger_dir), ("greater", greater_dir),
                                ("heavier", heavier_dir), ("denser", denser_dir), ("massive", massive_dir)]:
                if dir2 is None:
                    continue
                if name1 < name2:
                    cos_val = float(np.dot(dir1, dir2))
                    comp_word_cos[f"cos({name1},{name2})"] = round(cos_val, 4)
        
        results[f"compare_word_decompose_L{li}"] = {
            "layer": li,
            "cosines": comp_word_cos,
        }
        
        # 关键比较: bigger vs heavier, bigger vs larger, heavier vs denser
        key_cos = {}
        for k in ["cos(bigger,heavier)", "cos(bigger,larger)", "cos(heavier,denser)",
                   "cos(bigger,greater)", "cos(heavier,massive)"]:
            if k in comp_word_cos:
                key_cos[k] = comp_word_cos[k]
        
        log(f"  L{li}: 关键余弦 = {key_cos}")
    
    # ===== Step 3: 组合方向分解 =====
    log("\n--- Step 3: 组合方向分解 ---")
    
    # "bigger and heavier" vs 单独
    resids_combo_sw = collect_residuals(model, tokenizer, device, layers,
                                         TEMPLATES["combo_size_weight"], size_pairs, test_layers)
    resids_combo_ss = collect_residuals(model, tokenizer, device, layers,
                                         TEMPLATES["combo_size_speed"], size_pairs, test_layers)
    
    for li in test_layers:
        combo_sw_dir = compute_direction(resids_combo_sw[li])
        combo_ss_dir = compute_direction(resids_combo_ss[li])
        size_comp_dir = compute_direction(resids_compare_size[li])
        weight_comp_dir = compute_direction(resids_compare_weight[li])
        speed_comp_dir = compute_direction(resids_compare_speed[li])
        
        combo_cos = {}
        for name1, dir1 in [("combo_sw", combo_sw_dir), ("combo_ss", combo_ss_dir)]:
            if dir1 is None:
                continue
            for name2, dir2 in [("size_comp", size_comp_dir), ("weight_comp", weight_comp_dir), 
                                ("speed_comp", speed_comp_dir)]:
                if dir2 is None:
                    continue
                cos_val = float(np.dot(dir1, dir2))
                combo_cos[f"cos({name1},{name2})"] = round(cos_val, 4)
        
        results[f"combo_decompose_L{li}"] = {
            "layer": li,
            "cosines": combo_cos,
        }
        
        log(f"  L{li}: 组合方向余弦 = {combo_cos}")
    
    # ===== Step 4: 推理方向的PCA分解 - 比较/属性/正交方向 =====
    log("\n--- Step 4: 推理方向的PCA分解 ---")
    
    for li in test_layers:
        # 合并所有比较残差
        all_resids = (resids_compare_size[li] + resids_compare_weight[li] + 
                     resids_compare_speed[li])
        
        if len(all_resids) < 5:
            continue
        
        X = np.array(all_resids)
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 各个比较方向在前5个PC上的系数
        dim_dirs = {}
        for name, resids in [("size", resids_compare_size[li]), 
                             ("weight", resids_compare_weight[li]),
                             ("speed", resids_compare_speed[li])]:
            if len(resids) < 2:
                continue
            dir_vec = np.mean(resids, axis=0) - mean
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                dim_dirs[name] = dir_vec / norm
        
        # PC系数
        n_pc = min(5, Vt.shape[0])
        pc_coeffs = {}
        for name, dir_vec in dim_dirs.items():
            coeffs = Vt[:n_pc] @ dir_vec
            pc_coeffs[name] = {
                "coeffs": [round(float(c), 4) for c in coeffs],
                "abs_coeffs": [round(float(abs(c)), 4) for c in coeffs],
                "dominant_pc": int(np.argmax(np.abs(coeffs))),
                "dominant_coeff": round(float(coeffs[np.argmax(np.abs(coeffs))]), 4),
            }
        
        # PC能量分布
        total_energy = np.sum(S[:n_pc]**2)
        pc_energy = [round(float(S[i]**2 / total_energy), 4) for i in range(n_pc)]
        
        # 比较运算符方向(所有比较共有的方向)
        # = 前n_rel个样本的均值 vs 后面的均值 (不太对)
        # 更好的方法: 比较运算符方向 = 第一个词位置的方向(总是"大/重/快"的那个)
        # 实际上, 比较运算符方向 = 所有比较句的均值方向 vs neutral baseline
        
        results[f"pca_decompose_L{li}"] = {
            "layer": li,
            "pc_energy": pc_energy,
            "dim_pc_coeffs": pc_coeffs,
        }
        
        log(f"  L{li}: PC能量 = {pc_energy}")
        for name, info_d in pc_coeffs.items():
            log(f"    {name}: dominant_PC={info_d['dominant_pc']}, coeff={info_d['dominant_coeff']}, "
                f"coeffs={info_d['coeffs']}")
    
    # ===== Step 5: 比较运算符方向的提取 =====
    log("\n--- Step 5: 比较运算符方向 ---")
    
    # 比较运算符方向 = "bigger than" vs "heavier than" vs "faster than" 的方向差
    # 如果size vs weight方向相反是因为比较运算符, 那么同一个词对:
    # "elephant is bigger than mouse" vs "elephant is heavier than mouse"
    # 应该只在比较词位置不同
    
    for li in test_layers:
        # 用同一组词对(size_pairs), 但不同比较词
        # bigger方向 vs heavier方向
        bigger_resids = resids_compare_size[li]  # "bigger than"
        heavier_with_size = collect_residuals(model, tokenizer, device, layers,
                                               "The {} is heavier than the {}", size_pairs[:5], test_layers)
        faster_with_size = collect_residuals(model, tokenizer, device, layers,
                                              "The {} is faster than the {}", size_pairs[:5], test_layers)
        
        bigger_dir = compute_direction(bigger_resids)
        heavier_dir_same_words = compute_direction(heavier_with_size[li])
        faster_dir_same_words = compute_direction(faster_with_size[li])
        
        op_cos = {}
        for name1, dir1 in [("bigger", bigger_dir), ("heavier_same", heavier_dir_same_words), 
                            ("faster_same", faster_dir_same_words)]:
            if dir1 is None:
                continue
            for name2, dir2 in [("bigger", bigger_dir), ("heavier_same", heavier_dir_same_words),
                                ("faster_same", faster_dir_same_words)]:
                if dir2 is None:
                    continue
                if name1 < name2:
                    cos_val = float(np.dot(dir1, dir2))
                    op_cos[f"cos({name1},{name2})"] = round(cos_val, 4)
        
        results[f"operator_direction_L{li}"] = {
            "layer": li,
            "same_words_different_operator": op_cos,
        }
        
        log(f"  L{li}: 同词对不同比较词余弦 = {op_cos}")
    
    # 保存结果
    out_path = TEMP / f"ccxxviii_size_weight_decompose_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, "results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n结果保存到: {out_path}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    if args.model == "qwen3":
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("")
    
    run(args.model)
