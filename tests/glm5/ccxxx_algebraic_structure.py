"""
CCXXX(330): 单纯形+推理方向的代数结构
======================================================================
CCXX发现: dim=n_rel-1单纯形(R²=1.0), CCXXIII发现推理方向eff_dim≈2-4。
关键问题: 单纯形+推理方向是否构成某种代数结构?

实验设计:
  1. 推理方向是否是单纯形顶点的线性组合?
  2. 传递性测试: A>B, B>C → A>C, 方向是否满足传递性?
  3. 加法封闭性: size_dir + weight_dir 是否在新方向上?
  4. 格(Lattice)结构: 推理方向是否构成偏序?
  5. 多关系+推理组合: "A is bigger than B, and B lives in the ocean"
  6. 对称性: -size_dir ≈ "smaller"方向?

用法:
  python ccxxx_algebraic_structure.py --model qwen3
  python ccxxx_algebraic_structure.py --model glm4
  python ccxxx_algebraic_structure.py --model deepseek7b
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
LOG = TEMP / "ccxxx_algebraic_structure_log.txt"

# 词汇
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

SIZE_ANIMALS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
    ("shark", "crab"), ("tiger", "rat"), ("eagle", "sparrow"),
    ("mountain", "hill"), ("tree", "bush"), ("bus", "car"),
]

HABITAT_TEMPLATES = {
    "land": "The {} lives on the land",
    "ocean": "The {} lives in the ocean",
    "sky": "The {} lives in the sky",
}

COMPARE_TEMPLATE = "The {} is bigger than the {}"
REVERSE_TEMPLATE = "The {} is smaller than the {}"
COMBO_TEMPLATE = "The {} is bigger than the {}, and the {} lives in the {}"
CHAIN_TEMPLATE_A = "The {} is bigger than the {}"
CHAIN_TEMPLATE_B = "The {} is bigger than the {}"
CHAIN_TEMPLATE_C = "The {} is bigger than the {}"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_residuals(model, tokenizer, device, layers, prompts, test_layers):
    """收集提示列表的残差"""
    resids = {li: [] for li in test_layers}
    
    for prompt in prompts:
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


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    li_mid = n_layers // 2
    li_last = n_layers - 1
    test_layers = [0, li_mid, li_last]
    
    log(f"\n{'='*70}\nCCXXX(330): 单纯形+推理方向的代数结构 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # ===== Step 1: 推理方向与单纯形顶点的关系 =====
    log("\n--- Step 1: 推理方向与单纯形顶点的关系 ---")
    
    # 收集habitat单纯形方向
    hab_prompts = []
    hab_labels = []
    for hab, words in WORDS_BY_HABITAT.items():
        for w in words:
            hab_prompts.append(HABITAT_TEMPLATES[hab].format(w))
            hab_labels.append(hab)
    
    hab_resids = collect_residuals(model, tokenizer, device, layers, hab_prompts, test_layers)
    
    # 收集size比较方向
    size_prompts = [COMPARE_TEMPLATE.format(big, small) for big, small in SIZE_ANIMALS]
    reverse_prompts = [REVERSE_TEMPLATE.format(small, big) for big, small in SIZE_ANIMALS]
    
    size_resids = collect_residuals(model, tokenizer, device, layers, size_prompts, test_layers)
    reverse_resids = collect_residuals(model, tokenizer, device, layers, reverse_prompts, test_layers)
    
    log(f"  收集完成: habitat={len(hab_prompts)}, size={len(size_prompts)}, reverse={len(reverse_prompts)}")
    
    for li in test_layers:
        # 单纯形顶点(habitat方向)
        hab_arr = np.array(hab_resids[li])
        if hab_arr.shape[0] < 5:
            continue
        hab_mean = hab_arr.mean(axis=0)
        
        simplex_dirs = {}
        for hab in ["land", "ocean", "sky"]:
            mask = [i for i, l in enumerate(hab_labels) if l == hab]
            if len(mask) >= 2:
                group_mean = hab_arr[mask].mean(axis=0)
                direction = group_mean - hab_mean
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    simplex_dirs[hab] = direction / norm
        
        # 推理方向(size比较)
        size_arr = np.array(size_resids[li])
        if size_arr.shape[0] >= 3:
            size_mean = size_arr.mean(axis=0)
            # size方向: 所有size比较的均值 - 所有残差均值
            all_mean = np.concatenate([hab_arr, size_arr]).mean(axis=0)
            size_dir = size_mean - all_mean
            sn = np.linalg.norm(size_dir)
            if sn > 1e-10:
                size_dir = size_dir / sn
            else:
                size_dir = None
        else:
            size_dir = None
        
        # reverse方向
        rev_arr = np.array(reverse_resids[li])
        if rev_arr.shape[0] >= 3:
            rev_mean = rev_arr.mean(axis=0)
            all_mean_r = np.concatenate([hab_arr, rev_arr]).mean(axis=0)
            rev_dir = rev_mean - all_mean_r
            rn = np.linalg.norm(rev_dir)
            if rn > 1e-10:
                rev_dir = rev_dir / rn
            else:
                rev_dir = None
        else:
            rev_dir = None
        
        # 推理方向与单纯形顶点的余弦
        reasoning_simplex_cos = {}
        if size_dir is not None:
            for hab, s_dir in simplex_dirs.items():
                cos_val = float(np.dot(size_dir, s_dir))
                reasoning_simplex_cos[f"cos(size_dir,{hab})"] = round(cos_val, 4)
        
        # size_dir vs -size_dir ≈ reverse?
        if size_dir is not None and rev_dir is not None:
            cos_size_rev = float(np.dot(size_dir, rev_dir))
            reasoning_simplex_cos["cos(size_dir,reverse_dir)"] = round(cos_size_rev, 4)
        
        # 推理方向是否在单纯形子空间中?
        if simplex_dirs and size_dir is not None:
            # 单纯形子空间 = simplex_dirs的列空间
            simplex_matrix = np.array(list(simplex_dirs.values()))  # [n_hab, d_model]
            U_s, S_s, Vt_s = np.linalg.svd(simplex_matrix, full_matrices=False)
            
            # size_dir在单纯形子空间的投影
            # Vt_s: [min(n,d), d_model], 取前n_hab个主成分
            n_hab = len(simplex_dirs)
            proj = Vt_s[:n_hab].T @ (Vt_s[:n_hab] @ size_dir)
            proj_norm = np.linalg.norm(proj)
            size_norm = np.linalg.norm(size_dir)
            proj_ratio = proj_norm / size_norm if size_norm > 1e-10 else 0
            
            # 余弦
            cos_proj = float(np.dot(size_dir, proj / (proj_norm + 1e-10))) if proj_norm > 1e-10 else 0
            
            reasoning_simplex_cos["size_in_simplex_ratio"] = round(float(proj_ratio), 4)
            reasoning_simplex_cos["size_in_simplex_cos"] = round(cos_proj, 4)
        
        # 单纯形顶点之间的余弦
        simplex_cos = {}
        hab_names = list(simplex_dirs.keys())
        for i, h1 in enumerate(hab_names):
            for j, h2 in enumerate(hab_names):
                if i < j:
                    cos_val = float(np.dot(simplex_dirs[h1], simplex_dirs[h2]))
                    simplex_cos[f"cos({h1},{h2})"] = round(cos_val, 4)
        
        results[f"reasoning_simplex_L{li}"] = {
            "layer": li,
            "reasoning_to_simplex": reasoning_simplex_cos,
            "simplex_internal": simplex_cos,
        }
        
        log(f"  L{li}: reasoning→simplex = {reasoning_simplex_cos}")
        log(f"        simplex内部 = {simplex_cos}")
    
    # ===== Step 2: 传递性测试 =====
    log("\n--- Step 2: 传递性测试 ---")
    
    # A > B, B > C → A > C?
    # 用chain: elephant > horse > cat
    chains = [
        ("elephant", "horse", "cat"),
        ("whale", "shark", "dolphin"),
        ("mountain", "tree", "bush"),
        ("bus", "car", "bike"),
        ("lion", "fox", "rabbit"),
        ("bear", "dog", "cat"),
    ]
    
    for li in [li_mid, li_last]:
        # 收集 A>B, B>C, A>C 的残差
        ab_prompts = [COMPARE_TEMPLATE.format(a, b) for a, b, c in chains]
        bc_prompts = [COMPARE_TEMPLATE.format(b, c) for a, b, c in chains]
        ac_prompts = [COMPARE_TEMPLATE.format(a, c) for a, b, c in chains]
        
        ab_resids = collect_residuals(model, tokenizer, device, layers, ab_prompts, [li])
        bc_resids = collect_residuals(model, tokenizer, device, layers, bc_prompts, [li])
        ac_resids = collect_residuals(model, tokenizer, device, layers, ac_prompts, [li])
        
        if len(ab_resids[li]) >= 3 and len(bc_resids[li]) >= 3 and len(ac_resids[li]) >= 3:
            # 计算方向
            all_chain = np.concatenate([ab_resids[li], bc_resids[li], ac_resids[li]])
            all_mean = all_chain.mean(axis=0)
            
            ab_dir = np.mean(ab_resids[li], axis=0) - all_mean
            bc_dir = np.mean(bc_resids[li], axis=0) - all_mean
            ac_dir = np.mean(ac_resids[li], axis=0) - all_mean
            
            ab_norm = np.linalg.norm(ab_dir)
            bc_norm = np.linalg.norm(bc_dir)
            ac_norm = np.linalg.norm(ac_dir)
            
            if ab_norm > 1e-10 and bc_norm > 1e-10 and ac_norm > 1e-10:
                ab_dir = ab_dir / ab_norm
                bc_dir = bc_dir / bc_norm
                ac_dir = ac_dir / ac_norm
                
                # 传递性: A>C方向 ≈ A>B方向? (因为A和C差距更大)
                cos_ab_bc = float(np.dot(ab_dir, bc_dir))
                cos_ab_ac = float(np.dot(ab_dir, ac_dir))
                cos_bc_ac = float(np.dot(bc_dir, ac_dir))
                
                # 线性组合: ac_dir ≈ α*ab_dir + β*bc_dir?
                # 最小二乘拟合
                combo_matrix = np.column_stack([ab_dir, bc_dir])  # [d_model, 2]
                alpha_beta = np.linalg.lstsq(combo_matrix, ac_dir, rcond=None)[0]
                reconstructed = combo_matrix @ alpha_beta
                cos_recon = float(np.dot(ac_dir, reconstructed / (np.linalg.norm(reconstructed) + 1e-10)))
                
                results[f"transitivity_L{li}"] = {
                    "layer": li,
                    "n_chains": len(chains),
                    "cos_ab_bc": round(cos_ab_bc, 4),
                    "cos_ab_ac": round(cos_ab_ac, 4),
                    "cos_bc_ac": round(cos_bc_ac, 4),
                    "linear_combo_alpha": round(float(alpha_beta[0]), 4),
                    "linear_combo_beta": round(float(alpha_beta[1]), 4),
                    "linear_combo_cos": round(cos_recon, 4),
                }
                
                log(f"  L{li}: cos(ab,bc)={cos_ab_bc:.4f}, cos(ab,ac)={cos_ab_ac:.4f}, "
                    f"cos(bc,ac)={cos_bc_ac:.4f}")
                log(f"        线性组合: α={alpha_beta[0]:.4f}, β={alpha_beta[1]:.4f}, "
                    f"cos_recon={cos_recon:.4f}")
    
    # ===== Step 3: 对称性测试 =====
    log("\n--- Step 3: 对称性测试 ---")
    
    for li in test_layers:
        size_arr = np.array(size_resids[li])
        rev_arr = np.array(reverse_resids[li])
        
        if size_arr.shape[0] >= 3 and rev_arr.shape[0] >= 3:
            all_arr = np.concatenate([size_arr, rev_arr])
            all_mean = all_arr.mean(axis=0)
            
            size_dir = np.mean(size_arr, axis=0) - all_mean
            rev_dir = np.mean(rev_arr, axis=0) - all_mean
            
            sn = np.linalg.norm(size_dir)
            rn = np.linalg.norm(rev_dir)
            
            if sn > 1e-10 and rn > 1e-10:
                size_dir = size_dir / sn
                rev_dir = rev_dir / rn
                
                cos_sr = float(np.dot(size_dir, rev_dir))
                
                # 反对称性: rev_dir ≈ -size_dir?
                cos_neg = float(np.dot(size_dir, -rev_dir))
                
                results[f"symmetry_L{li}"] = {
                    "layer": li,
                    "cos_bigger_smaller": round(cos_sr, 4),
                    "cos_bigger_neg_smaller": round(cos_neg, 4),
                    "is_antisymmetric": bool(cos_neg > 0.7),
                }
                
                log(f"  L{li}: cos(bigger,smaller)={cos_sr:.4f}, cos(bigger,-smaller)={cos_neg:.4f}")
    
    # ===== Step 4: 多关系+推理组合 =====
    log("\n--- Step 4: 多关系+推理组合 ---")
    
    # "A is bigger than B, and B lives in the ocean"
    combo_prompts_1 = []  # size + ocean
    combo_prompts_2 = []  # size + land
    combo_prompts_3 = []  # size + sky
    
    ocean_words = WORDS_BY_HABITAT["ocean"][:5]
    land_words = WORDS_BY_HABITAT["land"][:5]
    sky_words = WORDS_BY_HABITAT["sky"][:5]
    
    for big, small in SIZE_ANIMALS[:5]:
        for w in ocean_words:
            combo_prompts_1.append(f"The {big} is bigger than the {small}, and the {w} lives in the ocean")
        for w in land_words:
            combo_prompts_2.append(f"The {big} is bigger than the {small}, and the {w} lives on the land")
        for w in sky_words:
            combo_prompts_3.append(f"The {big} is bigger than the {small}, and the {w} lives in the sky")
    
    combo1_resids = collect_residuals(model, tokenizer, device, layers, combo_prompts_1[:10], [li_mid])
    combo2_resids = collect_residuals(model, tokenizer, device, layers, combo_prompts_2[:10], [li_mid])
    combo3_resids = collect_residuals(model, tokenizer, device, layers, combo_prompts_3[:10], [li_mid])
    
    if (len(combo1_resids[li_mid]) >= 3 and len(combo2_resids[li_mid]) >= 3 and 
        len(combo3_resids[li_mid]) >= 3):
        
        combo_all = np.concatenate([combo1_resids[li_mid], combo2_resids[li_mid], combo3_resids[li_mid]])
        combo_mean = combo_all.mean(axis=0)
        
        c1_dir = np.mean(combo1_resids[li_mid], axis=0) - combo_mean
        c2_dir = np.mean(combo2_resids[li_mid], axis=0) - combo_mean
        c3_dir = np.mean(combo3_resids[li_mid], axis=0) - combo_mean
        
        c1n = np.linalg.norm(c1_dir)
        c2n = np.linalg.norm(c2_dir)
        c3n = np.linalg.norm(c3_dir)
        
        if c1n > 1e-10 and c2n > 1e-10 and c3n > 1e-10:
            c1_dir = c1_dir / c1n
            c2_dir = c2_dir / c2n
            c3_dir = c3_dir / c3n
            
            combo_cos = {
                "cos(size+ocean, size+land)": round(float(np.dot(c1_dir, c2_dir)), 4),
                "cos(size+ocean, size+sky)": round(float(np.dot(c1_dir, c3_dir)), 4),
                "cos(size+land, size+sky)": round(float(np.dot(c2_dir, c3_dir)), 4),
            }
            
            results["combo_relation_reasoning"] = {
                "layer": li_mid,
                "cosines": combo_cos,
            }
            
            log(f"  L{li_mid}: 组合方向余弦 = {combo_cos}")
    
    # ===== Step 5: 格结构测试 =====
    log("\n--- Step 5: 格结构测试 ---")
    
    # 如果推理方向构成格(Lattice), 则:
    # 1. 存在meet(∧)和join(∨)运算
    # 2. 满足吸收律: a ∧ (a ∨ b) = a
    # 测试: 不同推理维度的方向是否满足某种偏序关系
    
    for li in [li_mid, li_last]:
        # 收集3种推理方向
        weight_prompts = [f"The {h} is heavier than the {l}" for h, l in 
                         [("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
                          ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
                          ("concrete", "foam"), ("brick", "straw"), ("copper", "wool")]]
        speed_prompts = [f"The {f} is faster than the {s}" for f, s in
                        [("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
                         ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
                         ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle")]]
        
        weight_resids = collect_residuals(model, tokenizer, device, layers, weight_prompts, [li])
        speed_resids = collect_residuals(model, tokenizer, device, layers, speed_prompts, [li])
        
        size_arr = np.array(size_resids[li])
        weight_arr = np.array(weight_resids[li])
        speed_arr = np.array(speed_resids[li])
        
        if size_arr.shape[0] >= 3 and weight_arr.shape[0] >= 3 and speed_arr.shape[0] >= 3:
            all_arr = np.concatenate([size_arr, weight_arr, speed_arr])
            all_mean = all_arr.mean(axis=0)
            
            dirs = {}
            for name, arr in [("size", size_arr), ("weight", weight_arr), ("speed", speed_arr)]:
                d = np.mean(arr, axis=0) - all_mean
                n = np.linalg.norm(d)
                if n > 1e-10:
                    dirs[name] = d / n
            
            if len(dirs) == 3:
                # 格结构: 是否存在某种偏序?
                # 测试: 方向加法的封闭性
                # size + weight → 新方向是否在已知方向上?
                
                dir_names = list(dirs.keys())
                lattice_info = {}
                
                # 两两组合
                for i, d1 in enumerate(dir_names):
                    for j, d2 in enumerate(dir_names):
                        if i < j:
                            combo = dirs[d1] + dirs[d2]
                            combo_norm = np.linalg.norm(combo)
                            if combo_norm > 1e-10:
                                combo = combo / combo_norm
                            
                            # 组合方向与各原始方向的余弦
                            combo_cos = {}
                            for dk, dv in dirs.items():
                                combo_cos[f"cos(combo_{d1}_{d2},{dk})"] = round(float(np.dot(combo, dv)), 4)
                            
                            lattice_info[f"{d1}+{d2}"] = combo_cos
                
                # 3方向之和
                all_combo = sum(dirs.values())
                all_combo_norm = np.linalg.norm(all_combo)
                if all_combo_norm > 1e-10:
                    all_combo = all_combo / all_combo_norm
                
                all_combo_cos = {}
                for dk, dv in dirs.items():
                    all_combo_cos[f"cos(all_combo,{dk})"] = round(float(np.dot(all_combo, dv)), 4)
                
                lattice_info["size+weight+speed"] = all_combo_cos
                
                # 偏序测试: 如果size > weight > speed构成链, 
                # 则size方向应该最"远", speed最"近"
                # 测量: 各方向到均值的距离在所有方向组成的子空间中的投影
                
                results[f"lattice_structure_L{li}"] = {
                    "layer": li,
                    "lattice_info": lattice_info,
                }
                
                log(f"  L{li}: 格结构 = {lattice_info}")
    
    # ===== Step 6: 推理方向的群结构 =====
    log("\n--- Step 6: 推理方向的群结构 ---")
    
    for li in [li_mid, li_last]:
        size_arr = np.array(size_resids[li])
        rev_arr = np.array(reverse_resids[li])
        
        if size_arr.shape[0] >= 3 and rev_arr.shape[0] >= 3:
            all_arr = np.concatenate([size_arr, rev_arr])
            all_mean = all_arr.mean(axis=0)
            
            size_dir = np.mean(size_arr, axis=0) - all_mean
            rev_dir = np.mean(rev_arr, axis=0) - all_mean
            
            sn = np.linalg.norm(size_dir)
            rn = np.linalg.norm(rev_dir)
            
            if sn > 1e-10 and rn > 1e-10:
                size_dir_n = size_dir / sn
                rev_dir_n = rev_dir / rn
                
                # 群结构测试:
                # 1. 逆元: -size_dir ≈ rev_dir? (对称性)
                cos_inverse = float(np.dot(size_dir_n, -rev_dir_n))
                
                # 2. 结合律: 不适用(方向加法自然满足)
                
                # 3. 单位元: 所有方向的均值 ≈ 0 (已中心化)
                
                # 4. 封闭性: size + reverse ≈ 0?
                sum_dir = size_dir_n + rev_dir_n
                sum_norm = np.linalg.norm(sum_dir)
                
                group_info = {
                    "cos_size_neg_reverse": round(cos_inverse, 4),
                    "sum_dir_norm": round(float(sum_norm), 4),
                    "sum_to_size_cos": round(float(np.dot(sum_dir / (sum_norm + 1e-10), size_dir_n)), 4) if sum_norm > 1e-10 else 0,
                    "is_group_likely": bool(cos_inverse > 0.8 and sum_norm < 0.3),
                }
                
                results[f"group_structure_L{li}"] = {
                    "layer": li,
                    **group_info,
                }
                
                log(f"  L{li}: 群结构 = {group_info}")
    
    # 保存结果
    out_path = TEMP / f"ccxxx_algebraic_structure_{model_name}.json"
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
