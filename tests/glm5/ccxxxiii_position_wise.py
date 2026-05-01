"""
CCXXXIII(333): Position-wise分析 + LayerNorm后logit空间 + DS7B 3-4维语义解码
======================================================================
核心问题1: "bigger than"和"smaller than"同方向(cos=0.977), 模型如何区分?
核心问题2: LayerNorm是否旋转零空间分量到行空间?
核心问题3: DS7B 3-4维语义空间编码了什么?

关键假设(CCXXXII): "bigger/smaller同方向"意味着模型编码"比较关系本身",
  而不是"更大/更小的方向". 区分信息可能来自:
  - 不同token位置的残差
  - LayerNorm后的差异
  - 8-33%的W_U行空间分量

实验设计:
  Part 1: Position-wise残差分析 — 收集每个token位置的残差
    对比: "A is bigger than B"各位置的残差方向
    关键: 位置B(最后一个词) vs 位置"is" vs 位置"bigger"

  Part 2: LayerNorm前后对比 — 最终LayerNorm对W_U行空间投影的影响
    计算LayerNorm前后的W_U行空间投影比变化
    验证: LayerNorm是否将零空间分量"旋转"到行空间

  Part 3: DS7B 3-4维语义空间解码
    提取DS7B中间层的3-4个语义PCA分量
    分析: 这些维度是否对应land/ocean/sky?

  Part 4: 因果验证 — bigger/smaller同方向, 但残差向量不同
    计算两个句子的残差向量(不是方向)在W_U行空间的差异
    验证: 残差向量(整体)在W_U行空间的差异是否足以区分bigger/smaller

用法:
  python ccxxxiii_position_wise.py --model qwen3
  python ccxxxiii_position_wise.py --model glm4
  python ccxxxiii_position_wise.py --model deepseek7b
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
LOG = TEMP / "ccxxxiii_position_wise_log.txt"

# 词汇
SIZE_COMPARE = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
    ("shark", "crab"), ("tiger", "rat"), ("eagle", "sparrow"),
    ("mountain", "hill"), ("tree", "bush"), ("bus", "car"),
]

WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
             "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
              "crab", "seal", "squid", "lobster", "jellyfish", "starfish",
              "seahorse", "eel", "manta"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
            "falcon", "pigeon", "robin", "condor", "albatross", "vulture",
            "hummingbird", "stork"],
}

# 模板
TEMPLATES = {
    "bigger": "The {} is bigger than the {}",
    "smaller": "The {} is smaller than the {}",
    "heavier": "The {} is heavier than the {}",
    "bigger_rev": "The {} is smaller than the {}",
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_all_positions(model, tokenizer, device, layers, prompt, test_layers):
    """收集各层所有token位置的残差"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = toks.input_ids.shape[1]
    
    captured = {}
    def mk_hook(k):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[k] = o[0, :, :].detach().float().cpu().numpy()  # [seq_len, d_model]
        return hook
    
    hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers]
    with torch.no_grad():
        _ = model(**toks)
    for h in hooks:
        h.remove()
    
    # 获取token文本
    token_texts = [tokenizer.decode([t]) for t in toks.input_ids[0].tolist()]
    
    return captured, token_texts, seq_len


def simulate_layernorm(vec, eps=1e-5):
    """模拟LayerNorm(无weight/bias)"""
    mean = vec.mean()
    var = vec.var()
    return (vec - mean) / np.sqrt(var + eps)


def project_to_subspace(vec, U_basis):
    """将向量投影到U_basis的列空间, 返回投影和系数"""
    coeffs = U_basis.T @ vec
    proj = U_basis @ coeffs
    return proj, coeffs


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXIII(333): Position-wise + LayerNorm + 3维语义解码 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)
    
    # W_U行空间基(200分量)
    log("Computing W_U row space basis (200 components)...")
    W_U_T = W_U.T.astype(np.float32)
    U_wu, s_wu, _ = svds(W_U_T, k=200)
    sort_idx = np.argsort(s_wu)[::-1]
    U_wu = U_wu[:, sort_idx].astype(np.float64)
    log(f"  Done.")
    
    # Token IDs
    bigger_tokens = ["bigger", "larger", "greater", "huge", "enormous", "massive"]
    smaller_tokens = ["smaller", "tiny", "little", "miniature", "minute", "petite"]
    bigger_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in bigger_tokens 
                 if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    smaller_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in smaller_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # ====================================================================
    # Part 1: Position-wise残差分析
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: Position-wise残差分析")
    log("="*60)
    
    # 对几对比较词, 收集所有位置
    test_pairs = [("elephant", "mouse"), ("whale", "fish"), ("lion", "rabbit"),
                  ("horse", "cat"), ("bear", "fox"), ("shark", "crab")]
    
    position_data = {}
    
    for big, small in test_pairs:
        # "A is bigger than B"
        prompt_bigger = f"The {big} is bigger than the {small}"
        cap_bigger, tokens_bigger, slen_bigger = collect_all_positions(
            model, tokenizer, device, layers, prompt_bigger, test_layers)
        
        # "A is smaller than B"  (注意: 这不是反向比较, 是用smaller词)
        prompt_smaller = f"The {big} is smaller than the {small}"
        cap_smaller, tokens_smaller, slen_smaller = collect_all_positions(
            model, tokenizer, device, layers, prompt_smaller, test_layers)
        
        # "B is smaller than A" (反向比较)
        prompt_rev = f"The {small} is smaller than the {big}"
        cap_rev, tokens_rev, slen_rev = collect_all_positions(
            model, tokenizer, device, layers, prompt_rev, test_layers)
        
        for li in test_layers:
            key = f"pair_{big}_{small}_L{li}"
            
            if f"L{li}" not in cap_bigger or f"L{li}" not in cap_smaller or f"L{li}" not in cap_rev:
                continue
            
            res_bigger = cap_bigger[f"L{li}"]  # [seq_len, d_model]
            res_smaller = cap_smaller[f"L{li}"]
            res_rev = cap_rev[f"L{li}"]
            
            # 找关键位置: "bigger"/"smaller"词的位置和最后一个词的位置
            # 模板: "The <A> is <comp> than the <B>"
            # 位置: 0=The, 1=A, 2=is, 3=comp, 4=than, 5=the, 6=B
            
            n_pos = min(res_bigger.shape[0], 7)
            pos_results = {}
            
            for pos in range(n_pos):
                vec_big = res_bigger[pos]
                vec_small = res_smaller[pos]
                vec_rev = res_rev[pos]
                
                # 计算bigger vs smaller的差异方向
                diff = vec_big - vec_small
                diff_norm = np.linalg.norm(diff)
                
                # 差异在W_U行空间的投影
                if diff_norm > 1e-10:
                    diff_dir = diff / diff_norm
                    proj_diff, _ = project_to_subspace(diff_dir, U_wu)
                    diff_in_row = np.linalg.norm(proj_diff) ** 2
                    
                    # bigger/smaller logit在差异方向
                    bigger_logit_diff = np.mean(W_U[bigger_ids] @ diff)
                    smaller_logit_diff = np.mean(W_U[smaller_ids] @ diff)
                else:
                    diff_in_row = 0
                    bigger_logit_diff = 0
                    smaller_logit_diff = 0
                
                # 各残差在W_U行空间的投影比
                proj_big, _ = project_to_subspace(vec_big, U_wu)
                proj_small, _ = project_to_subspace(vec_small, U_wu)
                row_ratio_big = np.linalg.norm(proj_big) ** 2 / max(np.linalg.norm(vec_big) ** 2, 1e-20)
                row_ratio_small = np.linalg.norm(proj_small) ** 2 / max(np.linalg.norm(vec_small) ** 2, 1e-20)
                
                pos_results[f"pos{pos}"] = {
                    "token_bigger": tokens_bigger[pos] if pos < len(tokens_bigger) else "?",
                    "diff_norm": round(float(diff_norm), 4),
                    "diff_in_WU_row": round(float(diff_in_row), 4),
                    "bigger_logit_on_diff": round(float(bigger_logit_diff), 4),
                    "smaller_logit_on_diff": round(float(smaller_logit_diff), 4),
                    "row_ratio_bigger": round(float(row_ratio_big), 4),
                    "row_ratio_smaller": round(float(row_ratio_small), 4),
                }
            
            position_data[key] = {
                "tokens_bigger": tokens_bigger,
                "tokens_smaller": tokens_smaller,
                "positions": pos_results,
            }
    
    results["position_wise"] = position_data
    
    # 汇总: 哪个位置的bigger-smaller差异最大? 哪个位置最有效区分bigger/smaller?
    log("\n  Position-wise summary (average across pairs):")
    for li in test_layers:
        pos_avg = {}
        for pos in range(7):
            diffs = []
            diff_rows = []
            bigger_logits = []
            smaller_logits = []
            for key, val in position_data.items():
                if f"_L{li}" not in key:
                    continue
                pkey = f"pos{pos}"
                if pkey in val["positions"]:
                    diffs.append(val["positions"][pkey]["diff_norm"])
                    diff_rows.append(val["positions"][pkey]["diff_in_WU_row"])
                    bigger_logits.append(val["positions"][pkey]["bigger_logit_on_diff"])
                    smaller_logits.append(val["positions"][pkey]["smaller_logit_on_diff"])
            
            if diffs:
                avg_diff = np.mean(diffs)
                avg_row = np.mean(diff_rows)
                avg_b = np.mean(bigger_logits)
                avg_s = np.mean(smaller_logits)
                token_name = ""
                for key, val in position_data.items():
                    if f"_L{li}" in key and pkey in val["positions"]:
                        token_name = val["positions"][pkey]["token_bigger"]
                        break
                log(f"    L{li} pos{pos}({token_name}): diff_norm={avg_diff:.4f}, "
                    f"diff_in_row={avg_row:.4f}, bigger_logit={avg_b:.4f}, smaller_logit={avg_s:.4f}")
    
    # ====================================================================
    # Part 2: LayerNorm前后对比
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: LayerNorm前后对比")
    log("="*60)
    
    ln_results = {}
    
    for li in test_layers:
        # 用size比较残差
        size_resids = []
        for big, small in SIZE_COMPARE[:6]:
            prompt = f"The {big} is bigger than the {small}"
            cap, tokens, slen = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
            if f"L{li}" in cap:
                size_resids.append(cap[f"L{li}"][-1])  # 最后token位置
        
        if len(size_resids) < 3:
            continue
        
        mean_resid = np.mean(size_resids, axis=0)
        
        # LayerNorm前
        proj_before, _ = project_to_subspace(mean_resid, U_wu)
        row_ratio_before = np.linalg.norm(proj_before) ** 2 / max(np.linalg.norm(mean_resid) ** 2, 1e-20)
        
        # bigger/smaller logit差异 (LayerNorm前)
        bigger_logit_before = np.mean(W_U[bigger_ids] @ mean_resid)
        smaller_logit_before = np.mean(W_U[smaller_ids] @ mean_resid)
        diff_before = bigger_logit_before - smaller_logit_before
        
        # LayerNorm后
        resid_after_ln = simulate_layernorm(mean_resid)
        proj_after, _ = project_to_subspace(resid_after_ln, U_wu)
        row_ratio_after = np.linalg.norm(proj_after) ** 2 / max(np.linalg.norm(resid_after_ln) ** 2, 1e-20)
        
        # bigger/smaller logit差异 (LayerNorm后)
        bigger_logit_after = np.mean(W_U[bigger_ids] @ resid_after_ln)
        smaller_logit_after = np.mean(W_U[smaller_ids] @ resid_after_ln)
        diff_after = bigger_logit_after - smaller_logit_after
        
        # LayerNorm的方向旋转
        norm_before = np.linalg.norm(mean_resid)
        norm_after = np.linalg.norm(resid_after_ln)
        if norm_before > 1e-10 and norm_after > 1e-10:
            cos_before_after = float(np.dot(mean_resid / norm_before, resid_after_ln / norm_after))
        else:
            cos_before_after = 0
        
        # LayerNorm对W_U行空间投影比的影响
        # 关键: bigger方向在LN前后的W_U行空间投影比
        hab_resids = []
        for hab, words in WORDS_BY_HABITAT.items():
            for word in words[:5]:
                prompt = f"The {word} lives in the"
                cap, _, _ = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
                if f"L{li}" in cap:
                    hab_resids.append(cap[f"L{li}"][-1])
        
        if len(hab_resids) >= 6:
            baseline_mean = np.mean(hab_resids, axis=0)
            bigger_dir_before = mean_resid - baseline_mean
            bigger_norm_before = np.linalg.norm(bigger_dir_before)
            if bigger_norm_before > 1e-10:
                bigger_dir_before = bigger_dir_before / bigger_norm_before
                proj_b_before, _ = project_to_subspace(bigger_dir_before, U_wu)
                bigger_in_row_before = np.linalg.norm(proj_b_before) ** 2
                
                # LN后的bigger方向
                resid_compare_ln = simulate_layernorm(mean_resid)
                baseline_ln = simulate_layernorm(baseline_mean)
                bigger_dir_after = resid_compare_ln - baseline_ln
                bigger_norm_after = np.linalg.norm(bigger_dir_after)
                if bigger_norm_after > 1e-10:
                    bigger_dir_after = bigger_dir_after / bigger_norm_after
                    proj_b_after, _ = project_to_subspace(bigger_dir_after, U_wu)
                    bigger_in_row_after = np.linalg.norm(proj_b_after) ** 2
                else:
                    bigger_in_row_after = 0
            else:
                bigger_in_row_before = 0
                bigger_in_row_after = 0
        else:
            bigger_in_row_before = 0
            bigger_in_row_after = 0
        
        ln_results[f"L{li}"] = {
            "row_ratio_before_ln": round(float(row_ratio_before), 4),
            "row_ratio_after_ln": round(float(row_ratio_after), 4),
            "cos_before_after": round(float(cos_before_after), 4),
            "bigger_smaller_diff_before_ln": round(float(diff_before), 4),
            "bigger_smaller_diff_after_ln": round(float(diff_after), 4),
            "bigger_dir_in_row_before_ln": round(float(bigger_in_row_before), 4),
            "bigger_dir_in_row_after_ln": round(float(bigger_in_row_after), 4),
        }
        
        log(f"  L{li}: row_ratio before/after LN = {row_ratio_before:.4f}/{row_ratio_after:.4f}, "
            f"cos(LN前,LN后) = {cos_before_after:.4f}, "
            f"bigger-smaller diff before/after = {diff_before:.4f}/{diff_after:.4f}")
    
    results["layernorm_comparison"] = ln_results
    
    # ====================================================================
    # Part 3: DS7B 3-4维语义空间解码
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: 语义空间PCA解码")
    log("="*60)
    
    # 对所有模型, 提取中间层语义PCA分量, 分析其含义
    semantic_pca = {}
    
    for li in test_layers:
        # 收集habitat残差
        hab_data = {"land": [], "ocean": [], "sky": []}
        for hab, words in WORDS_BY_HABITAT.items():
            for word in words:
                prompt = f"The {word} lives in the"
                cap, _, _ = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
                if f"L{li}" in cap:
                    hab_data[hab].append(cap[f"L{li}"][-1])
        
        all_hab = []
        all_labels = []
        for hab in ["land", "ocean", "sky"]:
            all_hab.extend(hab_data[hab])
            all_labels.extend([hab] * len(hab_data[hab]))
        
        if len(all_hab) < 15:
            continue
        
        hab_arr = np.array(all_hab)
        hab_centered = hab_arr - hab_arr.mean(axis=0)
        
        # PCA
        _, S_hab, Vt_hab = np.linalg.svd(hab_centered, full_matrices=False)
        
        # 前5个主成分
        n_pc = min(5, Vt_hab.shape[0])
        
        # 计算每个habitat在前n_pc主成分上的投影
        hab_projections = {}
        for hab in ["land", "ocean", "sky"]:
            if len(hab_data[hab]) < 3:
                continue
            hab_resid_arr = np.array(hab_data[hab])
            hab_resid_centered = hab_resid_arr - hab_arr.mean(axis=0)
            # 投影到前n_pc个PC
            pc_projections = hab_resid_centered @ Vt_hab[:n_pc].T  # [n_samples, n_pc]
            hab_projections[hab] = {
                "mean_pc": [round(float(np.mean(pc_projections[:, i])), 4) for i in range(n_pc)],
                "std_pc": [round(float(np.std(pc_projections[:, i])), 4) for i in range(n_pc)],
            }
        
        # 判断: 哪些PC能区分habitat?
        habitat_separation = {}
        for pc_i in range(n_pc):
            means = [hab_projections[hab]["mean_pc"][pc_i] for hab in ["land", "ocean", "sky"] if hab in hab_projections]
            if len(means) == 3:
                # F-statistic
                group_data = []
                for hab in ["land", "ocean", "sky"]:
                    if hab in hab_projections and len(hab_data[hab]) >= 3:
                        group_data.append(np.array(hab_data[hab]))
                
                if len(group_data) == 3:
                    # 简单方差比
                    all_proj = np.concatenate([
                        (np.array(hab_data[hab]) - hab_arr.mean(axis=0)) @ Vt_hab[pc_i]
                        for hab in ["land", "ocean", "sky"]
                    ])
                    between_var = np.var([np.mean(g @ Vt_hab[pc_i]) for g in [
                        np.array(hab_data[hab]) - hab_arr.mean(axis=0) for hab in ["land", "ocean", "sky"]
                    ]])
                    within_var = np.mean([np.var(g @ Vt_hab[pc_i]) for g in [
                        np.array(hab_data[hab]) - hab_arr.mean(axis=0) for hab in ["land", "ocean", "sky"]
                    ]])
                    f_ratio = between_var / max(within_var, 1e-10)
                else:
                    f_ratio = 0
            else:
                f_ratio = 0
            
            habitat_separation[f"PC{pc_i}"] = {
                "sv": round(float(S_hab[pc_i]), 4),
                "var_explained": round(float(S_hab[pc_i]**2 / np.sum(S_hab**2)), 4),
                "land_mean": hab_projections.get("land", {}).get("mean_pc", [0]*n_pc)[pc_i] if pc_i < len(hab_projections.get("land", {}).get("mean_pc", [])) else 0,
                "ocean_mean": hab_projections.get("ocean", {}).get("mean_pc", [0]*n_pc)[pc_i] if pc_i < len(hab_projections.get("ocean", {}).get("mean_pc", [])) else 0,
                "sky_mean": hab_projections.get("sky", {}).get("mean_pc", [0]*n_pc)[pc_i] if pc_i < len(hab_projections.get("sky", {}).get("mean_pc", [])) else 0,
                "f_ratio": round(float(f_ratio), 4),
            }
        
        semantic_pca[f"L{li}"] = {
            "n_samples": len(all_hab),
            "top5_sv": [round(float(s), 4) for s in S_hab[:5]],
            "habitat_projections": hab_projections,
            "habitat_separation": habitat_separation,
        }
        
        log(f"  L{li}: top5 SV = {[round(float(s), 2) for s in S_hab[:5]]}")
        for pc_i in range(min(3, n_pc)):
            sep = habitat_separation.get(f"PC{pc_i}", {})
            log(f"    PC{pc_i}: land={sep.get('land_mean',0):.2f}, ocean={sep.get('ocean_mean',0):.2f}, "
                f"sky={sep.get('sky_mean',0):.2f}, f_ratio={sep.get('f_ratio',0):.2f}")
    
    results["semantic_pca"] = semantic_pca
    
    # ====================================================================
    # Part 4: 因果验证 — 残差向量(整体)在W_U行空间的差异
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: bigger vs smaller残差向量在W_U行空间的差异")
    log("="*60)
    
    causal_results = {}
    
    for li in test_layers:
        bigger_resids = []
        smaller_resids = []
        reverse_resids = []
        
        for big, small in SIZE_COMPARE[:6]:
            # "A is bigger than B"
            prompt = f"The {big} is bigger than the {small}"
            cap, _, _ = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
            if f"L{li}" in cap:
                bigger_resids.append(cap[f"L{li}"][-1])
            
            # "A is smaller than B" (同一句式, 不同比较词)
            prompt = f"The {big} is smaller than the {small}"
            cap, _, _ = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
            if f"L{li}" in cap:
                smaller_resids.append(cap[f"L{li}"][-1])
            
            # "B is smaller than A" (反向比较)
            prompt = f"The {small} is smaller than the {big}"
            cap, _, _ = collect_all_positions(model, tokenizer, device, layers, prompt, [li])
            if f"L{li}" in cap:
                reverse_resids.append(cap[f"L{li}"][-1])
        
        if len(bigger_resids) < 3 or len(smaller_resids) < 3:
            continue
        
        bigger_arr = np.array(bigger_resids)
        smaller_arr = np.array(smaller_resids)
        reverse_arr = np.array(reverse_resids)
        
        bigger_mean = np.mean(bigger_arr, axis=0)
        smaller_mean = np.mean(smaller_arr, axis=0)
        reverse_mean = np.mean(reverse_arr, axis=0) if len(reverse_resids) >= 3 else None
        
        # 分解到W_U行空间和零空间
        proj_bigger, coeffs_bigger = project_to_subspace(bigger_mean, U_wu)
        null_bigger = bigger_mean - proj_bigger
        
        proj_smaller, coeffs_smaller = project_to_subspace(smaller_mean, U_wu)
        null_smaller = smaller_mean - proj_smaller
        
        # ★★★ 关键: 行空间分量差异 vs 零空间分量差异
        diff_total = bigger_mean - smaller_mean
        diff_row = proj_bigger - proj_smaller
        diff_null = null_bigger - null_smaller
        
        # Logit差异分解
        bigger_logit_total = np.mean(W_U[bigger_ids] @ diff_total)
        smaller_logit_total = np.mean(W_U[smaller_ids] @ diff_total)
        bigger_logit_row = np.mean(W_U[bigger_ids] @ diff_row)
        smaller_logit_row = np.mean(W_U[smaller_ids] @ diff_row)
        bigger_logit_null = np.mean(W_U[bigger_ids] @ diff_null)
        smaller_logit_null = np.mean(W_U[smaller_ids] @ diff_null)
        
        # 反向比较
        reverse_data = {}
        if reverse_mean is not None:
            proj_rev, _ = project_to_subspace(reverse_mean, U_wu)
            diff_bigger_rev = bigger_mean - reverse_mean
            diff_row_rev = proj_bigger - proj_rev
            
            bigger_logit_vs_rev = np.mean(W_U[bigger_ids] @ diff_bigger_rev)
            smaller_logit_vs_rev = np.mean(W_U[smaller_ids] @ diff_bigger_rev)
            
            reverse_data = {
                "cos(bigger, reverse)": round(float(
                    np.dot(bigger_mean, reverse_mean) / 
                    max(np.linalg.norm(bigger_mean) * np.linalg.norm(reverse_mean), 1e-10)), 4),
                "bigger_logit_vs_reverse": round(float(bigger_logit_vs_rev), 4),
                "smaller_logit_vs_reverse": round(float(smaller_logit_vs_rev), 4),
            }
        
        causal_results[f"L{li}"] = {
            "bigger_logit_on_diff": round(float(bigger_logit_total), 4),
            "smaller_logit_on_diff": round(float(smaller_logit_total), 4),
            "bigger_logit_from_row": round(float(bigger_logit_row), 4),
            "smaller_logit_from_row": round(float(smaller_logit_row), 4),
            "bigger_logit_from_null": round(float(bigger_logit_null), 4),
            "smaller_logit_from_null": round(float(smaller_logit_null), 4),
            "row_contribution_to_bigger_logit": round(float(bigger_logit_row / max(abs(bigger_logit_total), 1e-6)), 4),
            "reverse_data": reverse_data,
        }
        
        log(f"  L{li}: bigger_logit={bigger_logit_total:.4f}, smaller_logit={smaller_logit_total:.4f}")
        log(f"    from_row: bigger={bigger_logit_row:.4f}, smaller={smaller_logit_row:.4f}")
        log(f"    from_null: bigger={bigger_logit_null:.4f}, smaller={smaller_logit_null:.4f}")
    
    results["causal_verification"] = causal_results
    
    # 保存
    out_file = TEMP / f"ccxxxiii_position_wise_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, 
                    "results": results}, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n结果已保存到 {out_file}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"CCXXXIII(333): Position-wise + LayerNorm + 3维语义解码 - {args.model}\n")
    
    run(args.model)
