"""
CCXXXV(335): 中间层残差Path Patching + 极端语义距离类别 + 2D语义几何精确分析
======================================================================
核心突破1: ★★★★★ 在中间层(L5/L10/L15/L20等)的残差上交换比较词位置信息
  → 确认信息在哪一层变得"不可逆" (flip_ratio从0.7→1.0的转变点)
  → 方法: 用hook在指定层之后替换比较词位置的残差, 然后继续forward

核心突破2: ★★★★ 语义距离更远的类别测试
  → 增加"space"(太空), "virtual"(虚拟), "microscopic"(微观)等极端类别
  → 如果5+极端类别仍只有2维 → 确认2维是语义空间上限
  → 如果有3+维 → 说明之前2维是类别选择问题

核心突破3: ★★★ 2D语义空间的精确几何结构
  → 各类别在2D平面上的角度分布
  → 是否存在"语义距离"度量?
  → 类别之间的angular separation

实验设计:
  Part 1: 逐层残差Path Patching (★核心)
    对"The elephant is bigger than the mouse"中的比较词位置,
    在每一层(或采样层)交换bigger↔smaller的残差表示.
    关键指标: flip_ratio随层数的变化曲线
    
  Part 2: 极端语义类别空间维度测试
    6类: land, ocean, sky, space, microscopic, virtual
    如果仍为2维 → 2维是上限
    如果3+维 → 之前的2维是类别相似性导致的
    
  Part 3: 2D语义平面上的精确几何
    各类别中心在2D平面上的坐标和角度
    类别间的angular separation
    是否形成正多边形或其他规则结构

用法:
  python ccxxxv_residual_patching.py --model qwen3
  python ccxxxv_residual_patching.py --model glm4
  python ccxxxv_residual_patching.py --model deepseek7b
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
LOG = TEMP / "ccxxxv_residual_patching_log.txt"

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

# ★★★ 极端语义距离类别 — 最大化类别间差异
EXTREME_HABITATS = {
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

# 原始3类 + desert + underwater (复用CCXXXIV的数据以对比)
ORIGINAL_HABITATS = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit",
             "fox", "deer", "bear", "wolf", "elephant", "giraffe", "zebra"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle",
              "crab", "seal", "squid", "lobster", "jellyfish", "starfish",
              "seahorse", "eel", "manta"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow",
            "falcon", "pigeon", "robin", "condor", "albatross", "vulture",
            "hummingbird", "stork"],
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


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXV(335): 残差Path Patching + 极端语义 + 2D几何 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    W_U = get_W_U(model)
    
    # W_U行空间基(500分量)
    log("Computing W_U row space basis (500 components)...")
    W_U_T = W_U.T.astype(np.float32)
    k = min(500, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    U_wu, s_wu, _ = svds(W_U_T, k=k)
    sort_idx = np.argsort(s_wu)[::-1]
    U_wu = U_wu[:, sort_idx].astype(np.float64)
    
    # Token IDs
    bigger_id = tokenizer.encode("bigger", add_special_tokens=False)[0]
    smaller_id = tokenizer.encode("smaller", add_special_tokens=False)[0]
    
    embed_layer = model.get_input_embeddings()
    
    # ====================================================================
    # Part 1: 逐层残差Path Patching — ★★★★★ 核心实验
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: 逐层残差Path Patching (核心实验)")
    log("="*60)
    log("  方法: 在指定层L, 用hook替换比较词位置的残差")
    log("  bigger句的comp_pos残差 → smaller句的comp_pos残差")
    log("  然后从L+1层继续forward到最后一层")
    log("  关键指标: flip_ratio随层数L的变化")
    
    # 测试3对比较词, 减少计算量但保证统计有效性
    test_pairs = SIZE_PAIRS[:6]  # 6对
    
    # 采样层 — 密集采样以捕捉转变点
    sample_layers = sorted(set(
        [0] + 
        list(range(1, n_layers, max(1, n_layers // 12))) +
        [n_layers // 4, n_layers // 3, n_layers // 2, 
         2 * n_layers // 3, 3 * n_layers // 4,
         n_layers - 2, n_layers - 1]
    ))
    
    log(f"  测试层: {sample_layers}")
    log(f"  测试对: {test_pairs}")
    
    residual_patching_results = {}
    
    for big, small in test_pairs:
        prompt_bigger = f"The {big} is bigger than the {small}"
        prompt_smaller = f"The {big} is smaller than the {small}"
        
        toks_bigger = tokenizer(prompt_bigger, return_tensors="pt").to(device)
        toks_smaller = tokenizer(prompt_smaller, return_tensors="pt").to(device)
        
        # 找比较词位置
        tokens_b_list = [tokenizer.decode([t]) for t in toks_bigger.input_ids[0].tolist()]
        tokens_s_list = [tokenizer.decode([t]) for t in toks_smaller.input_ids[0].tolist()]
        comp_pos = 0
        for i in range(min(len(tokens_b_list), len(tokens_s_list))):
            if tokens_b_list[i] != tokens_s_list[i]:
                comp_pos = i
                break
        
        seq_len = toks_bigger.input_ids.shape[1]
        log(f"\n  {big}/{small}: comp_pos={comp_pos}, seq_len={seq_len}")
        
        # === Step 1: 获取baseline logits ===
        with torch.no_grad():
            out_b = model(**toks_bigger)
            logits_b = out_b.logits[0, -1]
            diff_baseline = (logits_b[bigger_id] - logits_b[smaller_id]).item()
            
            out_s = model(**toks_smaller)
            logits_s = out_s.logits[0, -1]
            diff_baseline_smaller = (logits_s[bigger_id] - logits_s[smaller_id]).item()
        
        log(f"    Baseline: bigger_diff={diff_baseline:.4f}, smaller_diff={diff_baseline_smaller:.4f}")
        
        # === Step 2: 逐层残差patching ===
        layer_flip_data = {}
        
        for patch_li in sample_layers:
            # 方法: 使用"activation patching"
            # 1) 先forward bigger句子, 在patch_li层后用hook捕获输出
            # 2) 再forward smaller句子, 在patch_li层后用hook捕获输出
            # 3) 构造"patched"输入: bigger句的embed, 但在patch_li层后替换comp_pos的残差为smaller句的
            # 4) 从patch_li+1层开始继续forward
            
            # 采集中间层输出
            captured_b_patch = {}
            captured_s_patch = {}
            
            def mk_hook(key, storage):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    storage[key] = o.detach().clone()
                return hook
            
            # Forward bigger句子, 收集patch_li层输出
            hook_b = layers[patch_li].register_forward_hook(mk_hook("res", captured_b_patch))
            with torch.no_grad():
                _ = model(**toks_bigger)
            hook_b.remove()
            
            # Forward smaller句子, 收集patch_li层输出
            hook_s = layers[patch_li].register_forward_hook(mk_hook("res", captured_s_patch))
            with torch.no_grad():
                _ = model(**toks_smaller)
            hook_s.remove()
            
            if "res" not in captured_b_patch or "res" not in captured_s_patch:
                log(f"    L{patch_li}: capture failed, skip")
                continue
            
            res_b = captured_b_patch["res"]  # [1, seq_len, d_model]
            res_s = captured_s_patch["res"]
            
            # ★ 核心patch操作: 替换bigger句中comp_pos的残差为smaller句的
            # 方法: 从embedding层重新构造输入, 但使用"clean run until patch_li, 
            #       then patch, then continue from patch_li+1"
            # 
            # 简化实现: 直接使用模型forward, 在patch_li层后用hook修改残差
            # 具体步骤:
            # 1. 用bigger句的embed作为输入forward到patch_li
            # 2. 在patch_li后, 将comp_pos的残差替换为smaller句的
            # 3. 继续forward到最后一层
            
            # 实现方式: 用hook在patch_li+1层的输入中修改
            # 即在patch_li层的输出被修改后, patch_li+1层的输入自然改变
            
            # 更简洁的实现: 
            # 使用修改后的res_b作为新输入, 从patch_li+1开始forward
            # 这需要模型支持从中间层开始forward (transformer可以做到)
            
            # 实际上, 最可靠的方法是:
            # 再次forward bigger句, 但在patch_li层后用hook修改输出
            
            patched_res = res_b.clone()
            patched_res[0, comp_pos, :] = res_s[0, comp_pos, :]
            
            # 用hook实现: forward bigger句, 在patch_li后替换输出
            patched_logits = None
            
            def mk_patch_hook():
                def hook(m, inp, out):
                    # out是tuple, 第一个元素是hidden_states
                    if isinstance(out, tuple):
                        new_out = (patched_res,) + out[1:]
                        return new_out
                    else:
                        return patched_res
                return hook
            
            hook_patch = layers[patch_li].register_forward_hook(mk_patch_hook())
            with torch.no_grad():
                out_patched = model(**toks_bigger)
                patched_logits = out_patched.logits[0, -1]
            hook_patch.remove()
            
            diff_patched = (patched_logits[bigger_id] - patched_logits[smaller_id]).item()
            
            # flip_ratio: patched_diff / baseline_diff
            # 如果=1.0, 说明patch没有效果(信息还没传播到comp_pos)
            # 如果≈diff_baseline_smaller/diff_baseline, 说明完美翻转
            # 注意: 这里flip_ratio的定义是patched_diff/baseline_diff
            # 当flip_ratio接近0时, 说明patch使bigger_diff变为0(信息被完全替换)
            # 当flip_ratio接近diff_baseline_smaller/diff_baseline时, 说明翻转
            
            if abs(diff_baseline) > 1e-6:
                flip_ratio = diff_patched / diff_baseline
            else:
                flip_ratio = 0
            
            # 更好的指标: logit_change_ratio
            # patched后的diff应该趋近于smaller_baseline的diff
            # change = (diff_patched - diff_baseline) / (diff_baseline_smaller - diff_baseline)
            # 当change=1时, 完美翻转; 当change=0时, 无效果
            if abs(diff_baseline - diff_baseline_smaller) > 1e-6:
                change_ratio = (diff_patched - diff_baseline) / (diff_baseline_smaller - diff_baseline)
            else:
                change_ratio = 0
            
            # 也在last_pos做patch(交换最后位置的残差)
            patched_res_last = res_b.clone()
            patched_res_last[0, -1, :] = res_s[0, -1, :]
            
            hook_patch_last = layers[patch_li].register_forward_hook(
                lambda m, inp, out: (patched_res_last,) + out[1:] if isinstance(out, tuple) else patched_res_last
            )
            with torch.no_grad():
                out_patched_last = model(**toks_bigger)
                patched_logits_last = out_patched_last.logits[0, -1]
            hook_patch_last.remove()
            
            diff_patched_last = (patched_logits_last[bigger_id] - patched_logits_last[smaller_id]).item()
            
            if abs(diff_baseline) > 1e-6:
                flip_ratio_last = diff_patched_last / diff_baseline
            else:
                flip_ratio_last = 0
            
            if abs(diff_baseline - diff_baseline_smaller) > 1e-6:
                change_ratio_last = (diff_patched_last - diff_baseline) / (diff_baseline_smaller - diff_baseline)
            else:
                change_ratio_last = 0
            
            layer_flip_data[f"L{patch_li}"] = {
                "patch_layer": patch_li,
                "diff_patched_comp": round(diff_patched, 4),
                "flip_ratio_comp": round(flip_ratio, 4),
                "change_ratio_comp": round(change_ratio, 4),
                "diff_patched_last": round(diff_patched_last, 4),
                "flip_ratio_last": round(flip_ratio_last, 4),
                "change_ratio_last": round(change_ratio_last, 4),
            }
            
            log(f"    L{patch_li}: comp_patch: diff={diff_patched:.4f} flip={flip_ratio:.4f} change={change_ratio:.4f} | "
                f"last_patch: diff={diff_patched_last:.4f} flip={flip_ratio_last:.4f} change={change_ratio_last:.4f}")
            
            del captured_b_patch, captured_s_patch, patched_res, patched_res_last
            gc.collect()
        
        pair_key = f"{big}_{small}"
        residual_patching_results[pair_key] = {
            "comp_pos": comp_pos,
            "baseline_bigger_diff": round(diff_baseline, 4),
            "baseline_smaller_diff": round(diff_baseline_smaller, 4),
            "layer_data": layer_flip_data,
        }
    
    # 汇总: flip_ratio随层数变化
    log("\n  === 残差Path Patching Summary ===")
    avg_change_comp = {}  # {layer: mean change_ratio}
    avg_change_last = {}
    
    for pair_key, pair_data in residual_patching_results.items():
        for layer_key, layer_data in pair_data["layer_data"].items():
            li = layer_data["patch_layer"]
            if li not in avg_change_comp:
                avg_change_comp[li] = []
                avg_change_last[li] = []
            avg_change_comp[li].append(layer_data["change_ratio_comp"])
            avg_change_last[li].append(layer_data["change_ratio_last"])
    
    summary_table = []
    for li in sorted(avg_change_comp.keys()):
        mean_comp = np.mean(avg_change_comp[li])
        mean_last = np.mean(avg_change_last[li])
        std_comp = np.std(avg_change_comp[li])
        summary_table.append({
            "layer": li,
            "mean_change_comp": round(mean_comp, 4),
            "std_change_comp": round(std_comp, 4),
            "mean_change_last": round(mean_last, 4),
        })
        log(f"  L{li}: comp_change={mean_comp:.4f}±{std_comp:.4f}, last_change={mean_last:.4f}")
    
    results["residual_patching"] = {
        "per_pair": residual_patching_results,
        "summary": summary_table,
    }
    
    # ★★★ 寻找"转变点": change_ratio从0跳到1的层
    if len(summary_table) >= 3:
        max_change_layer = max(summary_table, key=lambda x: abs(x["mean_change_comp"]))
        log(f"\n  ★★★ 最大变化层: L{max_change_layer['layer']}, change={max_change_layer['mean_change_comp']:.4f}")
        
        # 找change_ratio > 0.5的第一层
        first_significant = None
        for entry in sorted(summary_table, key=lambda x: x["layer"]):
            if abs(entry["mean_change_comp"]) > 0.5:
                first_significant = entry["layer"]
                break
        if first_significant is not None:
            log(f"  ★★★ 首次显著变化(change>0.5): L{first_significant}")
        
        results["residual_patching"]["transition_point"] = {
            "max_change_layer": max_change_layer["layer"],
            "max_change_value": max_change_layer["mean_change_comp"],
            "first_significant_layer": first_significant,
        }
    
    # ====================================================================
    # Part 2: 极端语义距离类别空间维度测试
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: 极端语义距离类别空间维度测试")
    log("="*60)
    log("  测试: land/ocean/sky/space/microscopic/virtual (6类)")
    log("  如果仍为2维 → 2维是语义空间上限")
    log("  如果3+维 → 之前的2维是类别相似性导致的")
    
    # 不同类别组合测试
    habitat_configs = {
        "3class_original": ["land", "ocean", "sky"],
        "4class_extreme": ["land", "ocean", "sky", "space"],
        "5class_extreme": ["land", "ocean", "sky", "space", "microscopic"],
        "6class_extreme": ["land", "ocean", "sky", "space", "microscopic", "virtual"],
        "4class_physical": ["land", "ocean", "sky", "space"],  # 纯物理环境
        "4class_mixed": ["land", "ocean", "space", "virtual"],  # 物理+抽象
    }
    
    # 合并所有habitat词表
    ALL_HABITATS = {}
    ALL_HABITATS.update(ORIGINAL_HABITATS)
    ALL_HABITATS.update(EXTREME_HABITATS)
    
    semantic_results = {}
    
    # 测试3个关键层
    sem_test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    if n_layers - 2 not in sem_test_layers:
        sem_test_layers.append(n_layers - 2)
    sem_test_layers = sorted(set(sem_test_layers))
    
    log(f"  测试层: {sem_test_layers}")
    
    for config_name, hab_list in habitat_configs.items():
        log(f"\n  --- {config_name}: {hab_list} ---")
        
        for li in sem_test_layers:
            hab_data = {}
            valid_habs = []
            
            for hab in hab_list:
                words = ALL_HABITATS.get(hab, [])
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
                        try:
                            _ = model(**toks)
                        except:
                            pass
                    hook.remove()
                    
                    if f"L{li}" in captured:
                        hab_resids.append(captured[f"L{li}"])
                
                if len(hab_resids) >= 5:
                    hab_data[hab] = hab_resids
                    valid_habs.append(hab)
            
            if len(valid_habs) < 3:
                log(f"    L{li}: insufficient valid habitats ({len(valid_habs)})")
                continue
            
            # PCA分析
            all_vecs = []
            all_labels = []
            for hab in valid_habs:
                all_vecs.extend(hab_data[hab])
                all_labels.extend([hab] * len(hab_data[hab]))
            
            arr = np.array(all_vecs)
            centered = arr - arr.mean(axis=0)
            
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            
            n_classes = len(valid_habs)
            n_pc = min(n_classes + 2, Vt.shape[0])
            
            # 各类在PC上的投影
            hab_proj = {}
            for hab in valid_habs:
                hab_arr = np.array(hab_data[hab])
                hab_centered = hab_arr - arr.mean(axis=0)
                pc_proj = hab_centered @ Vt[:n_pc].T
                hab_proj[hab] = {
                    "mean": [round(float(np.mean(pc_proj[:, i])), 4) for i in range(n_pc)],
                    "std": [round(float(np.std(pc_proj[:, i])), 4) for i in range(n_pc)],
                }
            
            # 判断每个PC的分离能力
            pc_separation = {}
            for pc_i in range(n_pc):
                means = []
                within_vars = []
                for hab in valid_habs:
                    if hab in hab_proj:
                        means.append(hab_proj[hab]["mean"][pc_i])
                        within_vars.append(hab_proj[hab]["std"][pc_i]**2)
                
                if len(means) >= 2:
                    between_var = np.var(means)
                    avg_within = np.mean(within_vars) if within_vars else 1e-10
                    f_ratio = between_var / max(avg_within, 1e-10)
                else:
                    f_ratio = 0
                
                is_separating = f_ratio > 1.0
                
                pc_separation[f"PC{pc_i}"] = {
                    "sv": round(float(S[pc_i]), 4),
                    "var_explained": round(float(S[pc_i]**2 / np.sum(S**2)), 4),
                    "f_ratio": round(float(f_ratio), 4),
                    "is_separating": is_separating,
                    "means": {hab: hab_proj[hab]["mean"][pc_i] for hab in valid_habs if hab in hab_proj},
                }
            
            n_separating = sum(1 for pc in pc_separation.values() if pc["is_separating"])
            
            sem_key = f"{config_name}_L{li}"
            semantic_results[sem_key] = {
                "n_classes": n_classes,
                "valid_habitats": valid_habs,
                "expected_dim_simplex": n_classes - 1,
                "n_separating_PCs": n_separating,
                "pc_separation": pc_separation,
                "habitat_projections": hab_proj,
            }
            
            log(f"    L{li}: {config_name} → n_classes={n_classes}, "
                f"expected(N-1)={n_classes-1}, n_separating={n_separating}")
            for pc_i in range(min(n_pc, n_classes + 1)):
                sep = pc_separation[f"PC{pc_i}"]
                means_str = ", ".join(f"{h}={v:.2f}" for h, v in sep["means"].items())
                log(f"      PC{pc_i}: F={sep['f_ratio']:.2f}, separating={sep['is_separating']}, "
                    f"means: {means_str}")
    
    results["semantic_extreme"] = semantic_results
    
    # ====================================================================
    # Part 3: 2D语义平面精确几何分析
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: 2D语义平面精确几何分析")
    log("="*60)
    
    # 使用所有6类, 在最佳语义分离层做详细分析
    # 找到最佳层(3类simplex最强的层)
    
    all_habs_6 = ["land", "ocean", "sky", "space", "microscopic", "virtual"]
    best_layer = n_layers // 2  # 默认中间层
    
    # 如果之前有semantic结果, 找3class F-ratio最高的层
    if semantic_results:
        best_f = 0
        for key, val in semantic_results.items():
            if "3class" in key and "pc_separation" in val:
                for pc_key, pc_val in val["pc_separation"].items():
                    if pc_val["f_ratio"] > best_f:
                        best_f = pc_val["f_ratio"]
                        layer_str = key.split("_L")[-1]
                        try:
                            best_layer = int(layer_str)
                        except:
                            pass
    
    log(f"  分析层: L{best_layer}")
    
    geometry_data = {}
    
    for hab in all_habs_6:
        words = ALL_HABITATS.get(hab, [])
        hab_resids = []
        
        for word in words[:15]:
            prompt = f"The {word} lives in the"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            captured = {}
            
            def mk_hook(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured[k] = o[0, -1, :].detach().float().cpu().numpy()
                return hook
            
            hook = layers[best_layer].register_forward_hook(mk_hook("L"))
            with torch.no_grad():
                try:
                    _ = model(**toks)
                except:
                    pass
            hook.remove()
            
            if "L" in captured:
                hab_resids.append(captured["L"])
        
        if hab_resids:
            geometry_data[hab] = hab_resids
    
    # 在全部数据上做PCA, 投影到2D
    all_vecs = []
    all_labels = []
    for hab, vecs in geometry_data.items():
        all_vecs.extend(vecs)
        all_labels.extend([hab] * len(vecs))
    
    if len(all_vecs) >= 10:
        arr = np.array(all_vecs)
        centered = arr - arr.mean(axis=0)
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # 投影到前4个PC
        n_geom_pc = min(4, Vt.shape[0])
        projections = centered @ Vt[:n_geom_pc].T  # [n_samples, n_geom_pc]
        
        # 各类的2D坐标和角度
        hab_geometry = {}
        for hab in geometry_data:
            hab_idx = [i for i, l in enumerate(all_labels) if l == hab]
            hab_proj = projections[hab_idx]  # [n_hab, n_geom_pc]
            
            # 2D中心
            center_2d = np.mean(hab_proj[:, :2], axis=0)
            center_4d = np.mean(hab_proj[:, :n_geom_pc], axis=0)
            
            # 2D角度(从原点)
            angle_2d = np.arctan2(center_2d[1], center_2d[0]) * 180 / np.pi
            
            # 到原点的距离
            radius_2d = np.linalg.norm(center_2d)
            
            # 类内散度
            intra_spread = np.mean(np.std(hab_proj[:, :2], axis=0))
            
            hab_geometry[hab] = {
                "center_2d": [round(float(center_2d[0]), 4), round(float(center_2d[1]), 4)],
                "center_4d": [round(float(c), 4) for c in center_4d],
                "angle_degrees": round(float(angle_2d), 2),
                "radius_2d": round(float(radius_2d), 4),
                "intra_spread": round(float(intra_spread), 4),
                "n_samples": len(hab_idx),
            }
        
        # 类间angular separation
        hab_angles = {h: hab_geometry[h]["angle_degrees"] for h in hab_geometry}
        sorted_habs = sorted(hab_angles.keys(), key=lambda h: hab_angles[h])
        
        angular_separations = {}
        for i in range(len(sorted_habs)):
            for j in range(i + 1, len(sorted_habs)):
                h1, h2 = sorted_habs[i], sorted_habs[j]
                a1, a2 = hab_angles[h1], hab_angles[h2]
                # 在[-180, 180]范围内取最小角度差
                diff = abs(a2 - a1)
                if diff > 180:
                    diff = 360 - diff
                angular_separations[f"{h1}_vs_{h2}"] = round(diff, 2)
        
        # 理想均匀分布的角度
        n_habs = len(hab_geometry)
        ideal_angular_sep = 360.0 / n_habs if n_habs > 0 else 0
        
        # 实际平均角度间隔
        if sorted_habs:
            actual_seps = []
            for i in range(len(sorted_habs)):
                h1 = sorted_habs[i]
                h2 = sorted_habs[(i + 1) % len(sorted_habs)]
                a1 = hab_angles[h1]
                a2 = hab_angles[h2]
                diff = a2 - a1
                if i == len(sorted_habs) - 1:
                    diff = (hab_angles[sorted_habs[0]] + 360) - a1
                actual_seps.append(abs(diff))
            mean_angular_sep = np.mean(actual_seps)
            angular_uniformity = 1.0 - np.std(actual_seps) / max(mean_angular_sep, 1e-10)
        else:
            mean_angular_sep = 0
            angular_uniformity = 0
        
        log(f"\n  2D Geometry Results (L{best_layer}):")
        log(f"  Ideal angular separation for {n_habs} classes: {ideal_angular_sep:.2f}°")
        log(f"  Actual mean angular separation: {mean_angular_sep:.2f}°")
        log(f"  Angular uniformity: {angular_uniformity:.4f} (1=perfect)")
        
        for hab in sorted_habs:
            g = hab_geometry[hab]
            log(f"  {hab}: center=({g['center_2d'][0]:.2f}, {g['center_2d'][1]:.2f}), "
                f"angle={g['angle_degrees']:.1f}°, radius={g['radius_2d']:.4f}, "
                f"spread={g['intra_spread']:.4f}")
        
        # PC0/PC1的F-ratio (是否6类仍只有2个separating PCs)
        n_separating_6class = 0
        for pc_i in range(min(n_geom_pc, 6)):
            means = [hab_geometry[h]["center_4d"][pc_i] for h in hab_geometry]
            within_vars = []
            for hab in geometry_data:
                hab_idx = [i for i, l in enumerate(all_labels) if l == hab]
                hab_proj_pc = projections[hab_idx, pc_i]
                within_vars.append(np.var(hab_proj_pc))
            between_var = np.var(means)
            avg_within = np.mean(within_vars) if within_vars else 1e-10
            f_ratio = between_var / max(avg_within, 1e-10)
            is_sep = f_ratio > 1.0
            if is_sep:
                n_separating_6class += 1
            log(f"  PC{pc_i}: F={f_ratio:.2f}, separating={is_sep}")
        
        log(f"\n  ★ 6类n_separating_PCs = {n_separating_6class} (expected N-1 = {n_habs - 1})")
        
        results["geometry_2d"] = {
            "analysis_layer": best_layer,
            "habitat_geometry": hab_geometry,
            "angular_separations": angular_separations,
            "ideal_angular_sep": round(ideal_angular_sep, 2),
            "mean_angular_sep": round(mean_angular_sep, 2),
            "angular_uniformity": round(angular_uniformity, 4),
            "n_separating_PCs_6class": n_separating_6class,
            "expected_dim_simplex": n_habs - 1,
            "sorted_by_angle": sorted_habs,
        }
    
    # ====================================================================
    # Part 4: 多比较对验证 — 确保path patching结论不是特殊情况
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: 多比较属性验证 — heavier/lighter, faster/slower")
    log("="*60)
    
    # 除了bigger/smaller, 也测试heavier/lighter和faster/slower
    multi_compare_results = {}
    
    compare_configs = [
        ("bigger", "smaller", [("elephant", "mouse"), ("whale", "fish"), ("horse", "cat")]),
        ("heavier", "lighter", [("elephant", "mouse"), ("whale", "fish"), ("bear", "fox")]),
        ("faster", "slower", [("cheetah", "turtle"), ("eagle", "snail"), ("horse", "slug")]),
    ]
    
    for comp_pos_word, comp_neg_word, pairs in compare_configs:
        comp_pos_id = tokenizer.encode(comp_pos_word, add_special_tokens=False)[0]
        comp_neg_id = tokenizer.encode(comp_neg_word, add_special_tokens=False)[0]
        
        log(f"\n  --- {comp_pos_word}/{comp_neg_word} ---")
        
        flip_ratios_by_layer = {li: [] for li in sample_layers}
        
        for big, small in pairs:
            prompt_pos = f"The {big} is {comp_pos_word} than the {small}"
            prompt_neg = f"The {big} is {comp_neg_word} than the {small}"
            
            toks_pos = tokenizer(prompt_pos, return_tensors="pt").to(device)
            toks_neg = tokenizer(prompt_neg, return_tensors="pt").to(device)
            
            # Baseline
            with torch.no_grad():
                out_pos = model(**toks_pos)
                logits_pos = out_pos.logits[0, -1]
                diff_pos = (logits_pos[comp_pos_id] - logits_pos[comp_neg_id]).item()
                
                out_neg = model(**toks_neg)
                logits_neg = out_neg.logits[0, -1]
                diff_neg = (logits_neg[comp_pos_id] - logits_neg[comp_neg_id]).item()
            
            log(f"    {big}/{small}: pos_diff={diff_pos:.4f}, neg_diff={diff_neg:.4f}")
            
            # 找比较词位置
            tokens_pos_list = [tokenizer.decode([t]) for t in toks_pos.input_ids[0].tolist()]
            tokens_neg_list = [tokenizer.decode([t]) for t in toks_neg.input_ids[0].tolist()]
            comp_p = 0
            for i in range(min(len(tokens_pos_list), len(tokens_neg_list))):
                if tokens_pos_list[i] != tokens_neg_list[i]:
                    comp_p = i
                    break
            
            # 在3个关键层做patching
            key_layers = [0, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1]
            key_layers = sorted(set([l for l in key_layers if l < n_layers]))
            
            for patch_li in key_layers:
                captured_pos = {}
                captured_neg = {}
                
                def mk_h(k, st):
                    def hook(m, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        st[k] = o.detach().clone()
                    return hook
                
                h1 = layers[patch_li].register_forward_hook(mk_h("r", captured_pos))
                with torch.no_grad():
                    _ = model(**toks_pos)
                h1.remove()
                
                h2 = layers[patch_li].register_forward_hook(mk_h("r", captured_neg))
                with torch.no_grad():
                    _ = model(**toks_neg)
                h2.remove()
                
                if "r" not in captured_pos or "r" not in captured_neg:
                    continue
                
                res_pos = captured_pos["r"]
                res_neg = captured_neg["r"]
                
                # Patch: 将pos句的comp_pos残差替换为neg句的
                patched = res_pos.clone()
                patched[0, comp_p, :] = res_neg[0, comp_p, :]
                
                def mk_patch():
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            return (patched,) + out[1:]
                        return patched
                    return hook
                
                h3 = layers[patch_li].register_forward_hook(mk_patch())
                with torch.no_grad():
                    out_patched = model(**toks_pos)
                    logits_patched = out_patched.logits[0, -1]
                h3.remove()
                
                diff_patched = (logits_patched[comp_pos_id] - logits_patched[comp_neg_id]).item()
                
                if abs(diff_pos - diff_neg) > 1e-6:
                    change = (diff_patched - diff_pos) / (diff_neg - diff_pos)
                else:
                    change = 0
                
                flip_ratios_by_layer.setdefault(patch_li, []).append(change)
                
                log(f"    L{patch_li}: patched_diff={diff_patched:.4f}, change_ratio={change:.4f}")
                
                del captured_pos, captured_neg, patched
                gc.collect()
        
        # 汇总
        for li in sorted(flip_ratios_by_layer.keys()):
            vals = flip_ratios_by_layer[li]
            if vals:
                log(f"  {comp_pos_word}/{comp_neg_word} L{li}: mean_change={np.mean(vals):.4f}")
        
        multi_compare_results[f"{comp_pos_word}_{comp_neg_word}"] = {
            "flip_ratios_by_layer": {str(k): [round(v, 4) for v in vs] for k, vs in flip_ratios_by_layer.items() if vs},
        }
    
    results["multi_compare"] = multi_compare_results
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output_path = TEMP / f"ccxxxv_residual_patching_{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    log(f"\nResults saved to {output_path}")
    
    # 释放模型
    release_model(model)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    run(args.model)
