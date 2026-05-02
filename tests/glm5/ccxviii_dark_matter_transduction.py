"""
CCXVIII(368): 暗物质级联转导机制
=====================================

CCXVII发现: 暗物质steering承载75-87%效果, 而W_U-only几乎无效。
这引出核心问题: 暗物质如何影响输出? 如果暗物质不在W_U空间中,
它必须通过Attn/MLP的"非线性转导"被重新定向到W_U空间。

本实验追踪暗物质的转导路径:
1. 在L12注入delta_dark, 观察L13-L18的delta中有多少W_U分量
2. 在L12注入delta_wu, 观察L13-L18的delta中有多少W_U分量
3. 比较两者的"转导效率": 暗物质→W_U的转化率

用法:
  python ccxviii_dark_matter_transduction.py --model qwen3 --exp 1
  python ccxviii_dark_matter_transduction.py --model qwen3 --exp 2
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANS_TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

TEMP = Path("tests/glm5_temp")

CONCEPTS = {
    "apple": {
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
    },
    "love": {
        "templates": ["The word is love", "Feel the love", "Love is strong", "Show your love", "Love and peace"],
        "probe_words": ["heart", "feel", "care", "passion", "emotion", "hate", "romance", "affection"],
    },
    "science": {
        "templates": ["The word is science", "Study of science", "Science advances", "Modern science", "Science is knowledge"],
        "probe_words": ["research", "study", "theory", "experiment", "physics", "art", "biology", "data"],
    },
}

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


def inject_and_collect(model, tokenizer, device, inject_layer, direction, alpha, 
                       capture_layers, baseline_text=BASELINE_TEXT):
    """
    在inject_layer注入direction, 收集capture_layers的状态
    
    Returns: {layer_idx: residual_state}
    """
    all_layers = get_layers(model)
    captured = {}
    injected = [False]
    
    def make_inject_hook():
        def hook(module, inp, output):
            if not injected[0]:
                direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
                if isinstance(output, tuple):
                    modified = output[0].clone()
                    modified[0, -1, :] += alpha * direction_t.to(modified.dtype)
                    return (modified,) + output[1:]
                else:
                    modified = output.clone()
                    modified[0, -1, :] += alpha * direction_t.to(modified.dtype)
                    return modified
                injected[0] = True
            return output
        return hook
    
    def make_capture_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    
    hooks = []
    # 注入hook
    if inject_layer < len(all_layers):
        hooks.append(all_layers[inject_layer].register_forward_hook(make_inject_hook()))
    # 捕获hooks
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_capture_hook(li)))
    
    input_ids = tokenizer.encode(baseline_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
    
    for h in hooks: h.remove()
    gc.collect()
    return captured


# ================================================================
# Exp1: 暗物质级联转导追踪
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts):
    """
    在inject_layer注入delta_dark或delta_wu, 追踪后续层的W_U投影变化
    
    核心问题: 注入暗物质后, 后续层的delta中有多少变成了W_U分量?
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Dark Matter Cascade Transduction")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U行空间基
    print(f"  Computing W_U row space basis...")
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    
    # 收集baseline
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    # 收集概念delta
    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()
    
    # 注入实验
    inject_layers = [l for l in [12, 18] if l < n_layers]
    track_range = 6  # 追踪注入后6层
    alpha = 0.5
    
    results = {}
    
    for inject_l in inject_layers:
        print(f"\n  --- Inject at Layer {inject_l} ---")
        layer_results = {}
        
        for cname, cdata in concepts.items():
            if inject_l not in all_deltas.get(cname, {}):
                continue
            
            delta = all_deltas[cname][inject_l]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            # 分解delta
            proj_wu = U_wu.T @ delta
            delta_wu = U_wu @ proj_wu  # W_U分量
            delta_dark = delta - delta_wu  # 暗物质
            
            # 初始W_U投影比
            init_wu_ratio = np.sum(proj_wu ** 2) / (delta_norm ** 2)
            init_dark_ratio = 1.0 - init_wu_ratio
            
            # 追踪层
            track_layers = [inject_l + k for k in range(1, track_range + 1) if inject_l + k < n_layers]
            capture_layers = [inject_l] + track_layers
            
            # 三种注入: full, wu_only, dark_only
            # 对每种注入, 收集后续层的状态
            for inject_type, inject_dir in [("full", delta), ("wu", delta_wu), ("dark", delta_dark)]:
                injected_states = inject_and_collect(
                    model, tokenizer, device, inject_l, inject_dir, alpha, capture_layers
                )
                
                # 计算注入后各层的delta (相对于baseline)
                transduction_data = {}
                for tl in track_layers:
                    if tl not in injected_states or tl not in bl_all:
                        continue
                    
                    delta_tl = injected_states[tl] - bl_all[tl]
                    delta_tl_norm = np.linalg.norm(delta_tl)
                    if delta_tl_norm < 1e-8:
                        continue
                    
                    # W_U投影比
                    proj_wu_tl = U_wu.T @ delta_tl
                    wu_ratio_tl = np.sum(proj_wu_tl ** 2) / (delta_tl_norm ** 2)
                    
                    # 与原始delta的余弦
                    cos_with_original = float(np.dot(delta_tl, delta) / (delta_tl_norm * delta_norm))
                    
                    # 与delta_dark和delta_wu的余弦
                    cos_with_dark = 0.0
                    cos_with_wu = 0.0
                    n_dark = np.linalg.norm(delta_dark)
                    n_wu = np.linalg.norm(delta_wu)
                    if n_dark > 1e-8:
                        cos_with_dark = float(np.dot(delta_tl, delta_dark) / (delta_tl_norm * n_dark))
                    if n_wu > 1e-8:
                        cos_with_wu = float(np.dot(delta_tl, delta_wu) / (delta_tl_norm * n_wu))
                    
                    transduction_data[str(tl)] = {
                        "delta_norm": float(delta_tl_norm),
                        "wu_ratio": float(wu_ratio_tl),
                        "cos_with_original": cos_with_original,
                        "cos_with_dark": cos_with_dark,
                        "cos_with_wu": cos_with_wu,
                    }
                
                if inject_type not in layer_results:
                    layer_results[inject_type] = {}
                layer_results[inject_type][cname] = {
                    "init_wu_ratio": float(init_wu_ratio) if inject_type == "full" else (1.0 if inject_type == "wu" else 0.0),
                    "init_dark_ratio": float(init_dark_ratio) if inject_type == "full" else (0.0 if inject_type == "wu" else 1.0),
                    "inject_norm": float(np.linalg.norm(inject_dir)),
                    "transduction": transduction_data,
                }
                
                # 打印
                print(f"    {cname} ({inject_type}): ", end="")
                for tl_key, td in sorted(transduction_data.items()):
                    print(f"L{tl_key}:wu={td['wu_ratio']:.3f}/cos={td['cos_with_original']:.3f} ", end="")
                print()
        
        results[str(inject_l)] = layer_results
    
    # 汇总: 每种注入类型在后续层的平均W_U投影比
    print(f"\n  === Transduction Summary ===")
    for inject_l, lr in results.items():
        print(f"  Inject L{inject_l}:")
        for inject_type in ["full", "wu", "dark"]:
            wu_ratios_by_k = {}
            for cname, data in lr[inject_type].items():
                for tl_key, td in data["transduction"].items():
                    k = int(tl_key) - int(inject_l)
                    if k not in wu_ratios_by_k:
                        wu_ratios_by_k[k] = []
                    wu_ratios_by_k[k].append(td["wu_ratio"])
            
            if wu_ratios_by_k:
                ratio_str = ", ".join([f"k={k}:wu={np.mean(v):.3f}" for k, v in sorted(wu_ratios_by_k.items())])
                init_wu = "1.0" if inject_type == "wu" else ("0.0" if inject_type == "dark" else "~0.09")
                print(f"    {inject_type}(init_wu={init_wu}): {ratio_str}")
    
    return results


# ================================================================
# Exp2: 暗物质PCA + alpha缩放测试
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts):
    """
    1. 暗物质的PCA: 暗物质的有效维度
    2. alpha缩放: W_U-only在更大alpha下是否有效
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Dark Matter PCA + Alpha Scaling")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U行空间基
    print(f"  Computing W_U row space basis...")
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    
    # 收集baseline和delta
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    all_deltas = {}
    all_dark = {}  # {layer: [dark_vectors]}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                delta = np.mean(concept_states[l], axis=0) - bl_all[l]
                deltas[l] = delta
                
                # 计算暗物质
                if l not in all_dark:
                    all_dark[l] = []
                proj_wu = U_wu.T @ delta
                delta_dark = delta - U_wu @ proj_wu
                if np.linalg.norm(delta_dark) > 1e-8:
                    all_dark[l].append(delta_dark)
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()
    
    # Part 1: 暗物质PCA
    print(f"\n  --- Part 1: Dark Matter PCA ---")
    key_layers = [l for l in [12, 18, 24] if l < n_layers]
    
    dark_pca_results = {}
    for l in key_layers:
        if l not in all_dark or len(all_dark[l]) < 3:
            continue
        
        X_dark = np.array(all_dark[l])  # [n_dark_vecs, d_model]
        n_samples = X_dark.shape[0]
        X_c = X_dark - X_dark.mean(axis=0, keepdims=True)
        
        # SVD
        if n_samples < d_model:
            XXt = X_c @ X_c.T
            eigenvalues, _ = np.linalg.eigh(XXt)
            eigenvalues = eigenvalues[::-1]
            s = np.sqrt(np.maximum(eigenvalues, 0))
        else:
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)
        
        total_var = np.sum(s ** 2)
        if total_var < 1e-20:
            continue
        
        var_ratio = s ** 2 / total_var
        cum_var = np.cumsum(var_ratio)
        p = var_ratio[var_ratio > 1e-10]
        entropy = -np.sum(p * np.log2(p))
        eff_rank = 2 ** entropy
        
        n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
        
        dark_pca_results[str(l)] = {
            "n_samples": n_samples,
            "effective_rank": float(eff_rank),
            "n_for_90": n_90,
            "n_for_95": n_95,
            "cum_var_at_1": float(cum_var[0]),
            "cum_var_at_5": float(cum_var[min(4, len(cum_var)-1)]),
            "cum_var_at_10": float(cum_var[min(9, len(cum_var)-1)]),
        }
        
        print(f"  L{l}: n={n_samples}, eff_rank={eff_rank:.1f}, "
              f"n_90={n_90}, cum@1={cum_var[0]:.3f}, cum@5={cum_var[min(4,len(cum_var)-1)]:.3f}")
    
    # Part 2: alpha缩放测试
    print(f"\n  --- Part 2: Alpha Scaling (W_U vs Dark) ---")
    
    inject_layer = 18  # 中层
    if inject_layer >= n_layers:
        inject_layer = n_layers // 2
    
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    
    alpha_results = {}
    
    for alpha in alpha_values:
        print(f"\n  alpha={alpha}:")
        alpha_data = {}
        
        for cname, cdata in concepts.items():
            if inject_layer not in all_deltas.get(cname, {}):
                continue
            
            delta = all_deltas[cname][inject_layer]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            # 分解
            proj_wu = U_wu.T @ delta
            delta_wu = U_wu @ proj_wu
            delta_dark = delta - delta_wu
            
            # 获取probe IDs
            probe_ids = {}
            for w in cdata["probe_words"]:
                ids = model.tokenizer.encode(w, add_special_tokens=False) if hasattr(model, 'tokenizer') else []
                # 需要用外部tokenizer
                pass
            
            # 用logit变化测量
            # Baseline logits
            input_ids = tokenizer.encode(BASELINE_TEXT, add_special_tokens=True, return_tensors="pt").to(device)
            with torch.no_grad():
                base_out = model(input_ids=input_ids)
                base_logits = base_out.logits[0, -1, :].detach().float().cpu().numpy()
            
            # 三种steering的logit变化
            effects = {}
            for stype, sdir in [("full", delta), ("wu", delta_wu), ("dark", delta_dark)]:
                dir_tensor = torch.tensor(sdir, dtype=torch.float32, device=device)
                all_layers_list = get_layers(model)
                
                injected_flag = [False]
                def make_hook():
                    def hook(module, inp, output):
                        if not injected_flag[0]:
                            if isinstance(output, tuple):
                                modified = output[0].clone()
                                modified[0, -1, :] += alpha * dir_tensor.to(modified.dtype)
                                return (modified,) + output[1:]
                            else:
                                modified = output.clone()
                                modified[0, -1, :] += alpha * dir_tensor.to(modified.dtype)
                                return modified
                            injected_flag[0] = True
                        return output
                    return hook
                
                hook = all_layers_list[inject_layer].register_forward_hook(make_hook())
                with torch.no_grad():
                    try:
                        inj_out = model(input_ids=input_ids)
                        inj_logits = inj_out.logits[0, -1, :].detach().float().cpu().numpy()
                    except:
                        inj_logits = base_logits.copy()
                hook.remove()
                
                # probe words logit change
                probe_changes = []
                for w in cdata["probe_words"]:
                    wids = tokenizer.encode(w, add_special_tokens=False)
                    if wids:
                        probe_changes.append(float(inj_logits[wids[0]] - base_logits[wids[0]]))
                
                mean_change = np.mean(probe_changes) if probe_changes else 0
                effects[stype] = mean_change
            
            alpha_data[cname] = effects
            print(f"    {cname}: full={effects['full']:.3f}, wu={effects['wu']:.3f}, dark={effects['dark']:.3f}")
        
        # 汇总
        full_vals = [v["full"] for v in alpha_data.values()]
        wu_vals = [v["wu"] for v in alpha_data.values()]
        dark_vals = [v["dark"] for v in alpha_data.values()]
        
        wu_eff = np.mean([w/f if abs(f) > 1e-6 else 0 for w, f in zip(wu_vals, full_vals)])
        dark_eff = np.mean([d/f if abs(f) > 1e-6 else 0 for d, f in zip(dark_vals, full_vals)])
        
        alpha_results[str(alpha)] = {
            "full_mean": float(np.mean(full_vals)),
            "wu_mean": float(np.mean(wu_vals)),
            "dark_mean": float(np.mean(dark_vals)),
            "wu_efficiency": float(wu_eff),
            "dark_efficiency": float(dark_eff),
        }
        print(f"    Summary: full={np.mean(full_vals):.3f}, wu={np.mean(wu_vals):.3f}({wu_eff:.3f}), "
              f"dark={np.mean(dark_vals):.3f}({dark_eff:.3f})")
    
    return {
        "dark_pca": dark_pca_results,
        "alpha_scaling": alpha_results,
    }


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1", choices=["1", "2", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXVIII: Dark Matter Transduction Mechanism — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    all_results = {}

    if args.exp in ["1", "all"]:
        exp1_results = run_exp1(model, tokenizer, device, model_info, CONCEPTS)
        all_results["exp1"] = exp1_results

    if args.exp in ["2", "all"]:
        exp2_results = run_exp2(model, tokenizer, device, model_info, CONCEPTS)
        all_results["exp2"] = exp2_results

    # 保存
    output_path = TEMP / f"ccxviii_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    release_model(model)
    print(f"\nCCXVIII {model_name} 完成!")


if __name__ == "__main__":
    main()
