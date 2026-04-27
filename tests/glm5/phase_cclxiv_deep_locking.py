"""
Phase CCLXIV: 深层锁定机制追踪
================================================
核心问题: CCLXI发现浅层替换有效(switch_rate=0.83), 深层无效(0.00)。
深层为什么能"锁定"? 纠正在哪层发生?

3个子实验:
  Exp1: 替换传播追踪 — 在L0替换, 逐层追踪残差流被纠正的过程
  Exp2: 注意力vs FFN贡献分解 — 纠正主要来自Attn还是FFN?
  Exp3: 深层锁定验证 — 在纠正层之后再次替换, 是否仍被锁定?

填充: KN-1c(编码层级), KN-2c(属性正交性), 核心问题1

用法:
  python phase_cclxiv_deep_locking.py --model qwen3 --exp 1
"""
import argparse, os, sys, json, time, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
)

OUTPUT_DIR = Path("results/causal_fiber")

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano"],
}

TEMPLATES = [
    "The {} is",
    "I saw a {} today",
    "This {} looks",
    "A {} can be",
]


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_full_residuals(model, tokenizer, device, prompt, target_token_str, n_layers):
    """收集所有层的残差流向量"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    target_pos = None
    for i, t in enumerate(tokens):
        if target_token_str.lower() in t.lower().replace("_", "").replace("G", ""):
            target_pos = i
            break
    if target_pos is None:
        target_pos = len(tokens) - 1
    
    captured = {}
    layers = get_layers(model)
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    residuals = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            residuals[li] = captured[key][0, target_pos].numpy()
    
    return residuals, target_pos


def get_top_k_tokens(logits, tokenizer, k=10):
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_k = torch.topk(probs, k)
    result = []
    for i in range(k):
        tok_id = top_k.indices[i].item()
        tok_str = tokenizer.decode([tok_id]).strip()
        prob = top_k.values[i].item()
        result.append({"token": tok_str, "id": tok_id, "prob": prob})
    return result


# ============================================================
# Exp1: 替换传播追踪 — 在L0替换, 逐层追踪纠正过程
# ============================================================
def exp1_propagation_tracking(args):
    """核心实验: 在L0注入概念差分, 逐层追踪残差流变化, 找到'纠正'发生的层
    
    方法:
    1. 收集source和target在每层的残差流
    2. 在L0注入delta = source_L0 - target_L0 (差分注入, 非完整替换)
    3. 追踪干预后每层残差流与原始target残差流的cos
    4. 如果cos在某层突然回升, 说明该层在'纠正'干预
    """
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp1: 差分注入传播追踪 ({args.model}, {n_layers}L, d={d_model})")
    print(f"  方法: L0注入delta=source-target, 逐层追踪残差流vs原始target的cos")
    
    # 选跨类别概念对(纠正效果最明显)
    cross_pairs = [
        ("dog", "apple"), ("cat", "hammer"), ("horse", "rice"),
        ("eagle", "ocean"), ("shark", "desert"), ("snake", "cheese"),
    ]
    
    template = "The {} is"
    all_results = {}
    
    for source, target in cross_pairs:
        print(f"\n  === {source} -> {target} ===")
        
        prompt_source = template.format(source)
        prompt_target = template.format(target)
        
        # 1. 收集source和target的原始残差流
        resid_source, _ = collect_full_residuals(
            model, tokenizer, device, prompt_source, source, n_layers)
        resid_target, pos_target = collect_full_residuals(
            model, tokenizer, device, prompt_target, target, n_layers)
        
        if 0 not in resid_source or 0 not in resid_target:
            print(f"    Skip: missing L0 residual")
            continue
        
        # 2. 计算差分方向: delta = source_L0 - target_L0
        layers_list = get_layers(model)
        delta_L0 = resid_source[0] - resid_target[0]
        delta_norm = float(np.linalg.norm(delta_L0))
        print(f"    delta_norm at L0: {delta_norm:.2f}")
        
        # 3. 差分注入: 在L0将delta加到target位置
        input_ids_target = tokenizer(prompt_target, return_tensors="pt").to(device).input_ids
        
        # 收集干预后每层的残差流
        perlayer_captured = {}
        
        def make_delta_injection_hook():
            """在L0注入差分方向"""
            model_dtype = next(model.parameters()).dtype
            delta_tensor = torch.tensor(delta_L0, dtype=model_dtype, device=device)
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    if pos_target < out.shape[1]:
                        out[0, pos_target] = out[0, pos_target] + delta_tensor
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    if pos_target < out.shape[1]:
                        out[0, pos_target] = out[0, pos_target] + delta_tensor
                    return out
            return hook
        
        def make_perlayer_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    perlayer_captured[layer_idx] = output[0][0, pos_target].detach().float().cpu().numpy()
                else:
                    perlayer_captured[layer_idx] = output[0, pos_target].detach().float().cpu().numpy()
            return hook
        
        # 注册hooks: L0差分注入 + 后续层捕获
        hook_L0 = layers_list[0].register_forward_hook(make_delta_injection_hook())
        hooks_rest = []
        for li in range(1, n_layers):
            hooks_rest.append(layers_list[li].register_forward_hook(make_perlayer_hook(li)))
        
        with torch.no_grad():
            try:
                logits_intervened = model(input_ids_target).logits
            except Exception as e:
                print(f"    Forward failed: {e}")
                hook_L0.remove()
                for h in hooks_rest:
                    h.remove()
                continue
        
        hook_L0.remove()
        for h in hooks_rest:
            h.remove()
        
        intervened_top = get_top_k_tokens(logits_intervened, tokenizer, k=10)
        
        # 4. Baseline logits
        with torch.no_grad():
            logits_baseline = model(input_ids_target).logits
        baseline_top = get_top_k_tokens(logits_baseline, tokenizer, k=10)
        
        # 5. 核心分析: 每层干预残差 vs 原始target残差的cos
        # 如果cos在某层突然回升, 说明该层在"纠正"干预
        per_layer_cos_target = {}  # 干预后各层 vs 原始target各层 (回升=纠正)
        per_layer_cos_source = {}  # 干预后各层 vs 原始source各层 (下降=纠正)
        per_layer_delta_cos = {}   # 干预后各层的delta方向 vs 原始delta方向
        per_layer_norm = {}
        
        original_delta_dir = delta_L0 / (delta_norm + 1e-10)
        
        for li in sorted(perlayer_captured.keys()):
            h_intervened = perlayer_captured[li]
            per_layer_norm[li] = float(np.linalg.norm(h_intervened))
            
            if li in resid_target:
                per_layer_cos_target[li] = proper_cos(h_intervened, resid_target[li])
            if li in resid_source:
                per_layer_cos_source[li] = proper_cos(h_intervened, resid_source[li])
            
            # 计算delta残差 = 干预后 - 原始target
            if li in resid_target:
                current_delta = h_intervened - resid_target[li]
                current_delta_norm = np.linalg.norm(current_delta)
                if current_delta_norm > 1e-10:
                    per_layer_delta_cos[li] = proper_cos(current_delta / current_delta_norm, original_delta_dir)
                else:
                    per_layer_delta_cos[li] = 0.0
        
        # 6. 计算纠正层: delta_cos下降最快的层(差分方向被旋转/削弱)
        correction_layers = []
        prev_delta_cos = per_layer_delta_cos.get(1, 1.0)
        for li in sorted(per_layer_delta_cos.keys()):
            if li <= 1:
                continue
            curr_delta_cos = per_layer_delta_cos[li]
            delta_drop = prev_delta_cos - curr_delta_cos
            if delta_drop > 0.05:  # 差分方向显著偏转
                correction_layers.append((li, round(delta_drop, 4), round(curr_delta_cos, 4)))
            prev_delta_cos = curr_delta_cos
        
        result = {
            "source": source,
            "target": target,
            "delta_norm_L0": round(delta_norm, 2),
            "baseline_top1": baseline_top[0]["token"],
            "intervened_top1": intervened_top[0]["token"],
            "switched": baseline_top[0]["token"] != intervened_top[0]["token"],
            "per_layer_cos_with_target": {str(k): round(v, 4) for k, v in sorted(per_layer_cos_target.items())},
            "per_layer_cos_with_source": {str(k): round(v, 4) for k, v in sorted(per_layer_cos_source.items())},
            "per_layer_delta_cos": {str(k): round(v, 4) for k, v in sorted(per_layer_delta_cos.items())},
            "per_layer_norm": {str(k): round(v, 2) for k, v in sorted(per_layer_norm.items())},
            "correction_layers": correction_layers[:5],
        }
        
        all_results[f"{source}->{target}"] = result
        
        # 打印关键层
        print(f"    Baseline top1: {baseline_top[0]['token']}, Intervened top1: {intervened_top[0]['token']}, Switched: {result['switched']}")
        print(f"    delta_cos (L1-L5): ", end="")
        for li in range(1, min(6, n_layers)):
            if li in per_layer_delta_cos:
                print(f"L{li}={per_layer_delta_cos[li]:.3f} ", end="")
        print()
        print(f"    delta_cos (mid): ", end="")
        mid = n_layers // 2
        for li in range(max(1, mid-2), min(n_layers, mid+3)):
            if li in per_layer_delta_cos:
                print(f"L{li}={per_layer_delta_cos[li]:.3f} ", end="")
        print()
        print(f"    delta_cos (deep): ", end="")
        for li in range(max(1, n_layers-5), n_layers):
            if li in per_layer_delta_cos:
                print(f"L{li}={per_layer_delta_cos[li]:.3f} ", end="")
        print()
        print(f"    cos_with_target (L1-L5): ", end="")
        for li in range(1, min(6, n_layers)):
            if li in per_layer_cos_target:
                print(f"L{li}={per_layer_cos_target[li]:.3f} ", end="")
        print()
        print(f"    cos_with_target (deep): ", end="")
        for li in range(max(1, n_layers-5), n_layers):
            if li in per_layer_cos_target:
                print(f"L{li}={per_layer_cos_target[li]:.3f} ", end="")
        print()
        
        if correction_layers:
            print(f"    Correction layers: {correction_layers[:3]}")
    
    # 6. 汇总: 找到跨概念对一致的纠正层
    correction_counts = defaultdict(int)
    for pair, res in all_results.items():
        for cl, delta, cos in res["correction_layers"]:
            correction_counts[cl] += 1
    
    print(f"\n  === 纠正层汇总 ({args.model}) ===")
    for layer, count in sorted(correction_counts.items(), key=lambda x: -x[1]):
        print(f"    L{layer}: {count}/{len(all_results)} 概念对在此层纠正")
    
    # 保存
    save_dir = OUTPUT_DIR / f"{args.model}_cclxiv"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXIV",
        "exp": 1,
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(all_results),
        "correction_layer_counts": {str(k): v for k, v in sorted(correction_counts.items(), key=lambda x: -x[1])},
        "per_pair_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(save_dir / "exp1_propagation_tracking.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved to {save_dir / 'exp1_propagation_tracking.json'}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: 注意力 vs FFN 贡献分解
# ============================================================
def exp2_attn_vs_ffn(args):
    """分解纠正的来源: 注意力还是FFN?"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp2: Attn vs FFN 贡献分解 ({args.model}, {n_layers}L)")
    print(f"  方法: 在L0替换后, 逐层计算Attn输出和FFN输出与纠正方向的对齐度")
    
    cross_pairs = [
        ("dog", "apple"), ("cat", "hammer"), ("horse", "rice"),
        ("eagle", "ocean"), ("shark", "desert"),
    ]
    template = "The {} is"
    all_results = {}
    
    for source, target in cross_pairs:
        print(f"\n  === {source} -> {target} ===")
        
        prompt_source = template.format(source)
        prompt_target = template.format(target)
        
        # 收集原始残差流
        resid_source, _ = collect_full_residuals(
            model, tokenizer, device, prompt_source, source, n_layers)
        resid_target, pos_target = collect_full_residuals(
            model, tokenizer, device, prompt_target, target, n_layers)
        
        if 0 not in resid_source or 0 not in resid_target:
            continue
        
        # 计算纠正方向: target残差 - source残差 (每层)
        correction_dir = {}
        for li in range(n_layers):
            if li in resid_target and li in resid_source:
                correction_dir[li] = resid_target[li] - resid_source[li]
        
        # 逐层分解: Attn输出和FFN输出
        # 残差流 = h_{l} + Attn(h_l) + FFN(h_l + Attn(h_l))
        # 所以 delta = Attn + FFN
        # 我们需要捕获每层的attn_output和 ffn_output
        
        input_ids_target = tokenizer(prompt_target, return_tensors="pt").to(device).input_ids
        
        # 对target做L0替换的干预, 同时捕获每层的attn和ffn输出
        layers_list = get_layers(model)
        source_vec_L0 = resid_source[0]
        
        attn_outputs = {}
        ffn_outputs = {}
        resid_after_attn = {}
        
        # 注册hook: L0干预 + 捕获attn/ffn输出
        def make_L0_intervention_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    if pos_target < out.shape[1]:
                        new_vec = torch.tensor(source_vec_L0, dtype=out.dtype, device=out.device)
                        out[0, pos_target] = new_vec
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    if pos_target < out.shape[1]:
                        new_vec = torch.tensor(source_vec_L0, dtype=out.dtype, device=out.device)
                        out[0, pos_target] = new_vec
                    return out
            return hook
        
        # 逐层注册attn和ffn hook
        # Transformer lens的结构: 每个layer有 self_attention 和 mlp
        # 我们需要在attn输出后和ffn输出后捕获
        hooks = []
        hooks.append(layers_list[0].register_forward_hook(make_L0_intervention_hook()))
        
        for li in range(n_layers):
            layer = layers_list[li]
            
            # 尝试捕获attention输出
            if hasattr(layer, 'self_attn') or hasattr(layer, 'attention') or hasattr(layer, 'attn'):
                attn_mod = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None) or getattr(layer, 'attn', None)
                if attn_mod is not None:
                    def make_attn_hook(lidx):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                attn_outputs[lidx] = output[0][0, pos_target].detach().float().cpu().numpy()
                            else:
                                attn_outputs[lidx] = output[0, pos_target].detach().float().cpu().numpy()
                        return hook
                    hooks.append(attn_mod.register_forward_hook(make_attn_hook(li)))
            
            # 尝试捕获FFN输出
            if hasattr(layer, 'mlp') or hasattr(layer, 'ffn') or hasattr(layer, 'feed_forward'):
                ffn_mod = getattr(layer, 'mlp', None) or getattr(layer, 'ffn', None) or getattr(layer, 'feed_forward', None)
                if ffn_mod is not None:
                    def make_ffn_hook(lidx):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                ffn_outputs[lidx] = output[0][0, pos_target].detach().float().cpu().numpy()
                            else:
                                ffn_outputs[lidx] = output[0, pos_target].detach().float().cpu().numpy()
                        return hook
                    hooks.append(ffn_mod.register_forward_hook(make_ffn_hook(li)))
        
        with torch.no_grad():
            try:
                logits = model(input_ids_target).logits
            except Exception as e:
                print(f"    Forward failed: {e}")
                for h in hooks:
                    h.remove()
                continue
        
        for h in hooks:
            h.remove()
        
        # 分析: 每层的attn和ffn输出与纠正方向的对齐度
        result = {"source": source, "target": target}
        attn_alignment = {}
        ffn_alignment = {}
        attn_norm = {}
        ffn_norm = {}
        
        for li in range(n_layers):
            if li in correction_dir and li in attn_outputs:
                cos_val = proper_cos(attn_outputs[li], correction_dir[li])
                attn_alignment[li] = cos_val
                attn_norm[li] = float(np.linalg.norm(attn_outputs[li]))
            if li in correction_dir and li in ffn_outputs:
                cos_val = proper_cos(ffn_outputs[li], correction_dir[li])
                ffn_alignment[li] = cos_val
                ffn_norm[li] = float(np.linalg.norm(ffn_outputs[li]))
        
        result["attn_alignment_with_correction"] = {str(k): round(v, 4) for k, v in sorted(attn_alignment.items())}
        result["ffn_alignment_with_correction"] = {str(k): round(v, 4) for k, v in sorted(ffn_alignment.items())}
        result["attn_norm"] = {str(k): round(v, 2) for k, v in sorted(attn_norm.items())}
        result["ffn_norm"] = {str(k): round(v, 2) for k, v in sorted(ffn_norm.items())}
        
        all_results[f"{source}->{target}"] = result
        
        # 打印关键层
        print(f"    Attn align (shallow): ", end="")
        for li in range(1, min(6, n_layers)):
            if li in attn_alignment:
                print(f"L{li}={attn_alignment[li]:.3f} ", end="")
        print()
        print(f"    FFN align (shallow):  ", end="")
        for li in range(1, min(6, n_layers)):
            if li in ffn_alignment:
                print(f"L{li}={ffn_alignment[li]:.3f} ", end="")
        print()
        print(f"    Attn align (mid): ", end="")
        mid = n_layers // 2
        for li in range(max(1, mid-2), min(n_layers, mid+3)):
            if li in attn_alignment:
                print(f"L{li}={attn_alignment[li]:.3f} ", end="")
        print()
        print(f"    FFN align (mid):  ", end="")
        for li in range(max(1, mid-2), min(n_layers, mid+3)):
            if li in ffn_alignment:
                print(f"L{li}={ffn_alignment[li]:.3f} ", end="")
        print()
    
    # 保存
    save_dir = OUTPUT_DIR / f"{args.model}_cclxiv"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXIV",
        "exp": 2,
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(all_results),
        "per_pair_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(save_dir / "exp2_attn_vs_ffn.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved to {save_dir / 'exp2_attn_vs_ffn.json'}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: Template热点消融 — 消融template方向, 观察语法变化
# ============================================================
def exp3_template_hotspot_ablation(args):
    """消融template热点层的template方向, 观察输出是否失去语法结构"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 基于CCLXIII结果的template热点层
    template_hotspot = {
        "qwen3": 18,
        "glm4": 20,
        "deepseek7b": 14,
    }
    hotspot_layer = template_hotspot.get(args.model, n_layers // 2)
    
    print(f"\n  Exp3: Template热点消融 ({args.model}, 热点层=L{hotspot_layer})")
    print(f"  方法: 收集template方向, 消融后观察输出变化")
    
    # 1. 收集template方向: 不同模板在同一概念的残差流差异
    test_words = ["dog", "apple", "hammer", "mountain", "cat", "rice", "knife", "ocean"]
    
    # 对每个词, 收集不同模板的残差流
    template_residuals = defaultdict(dict)  # {word: {template_idx: {layer: vector}}}
    
    for word in test_words:
        for ti, tmpl in enumerate(TEMPLATES):
            prompt = tmpl.format(word)
            resid, _ = collect_full_residuals(model, tokenizer, device, prompt, word, n_layers)
            template_residuals[word][ti] = resid
    
    # 2. 计算template方向: 对每个词, 不同模板的残差流差异
    # template方向 = 残差流 - 跨模板均值(仅保留template变化部分)
    template_directions = {}  # {layer: [direction_per_word_template]}
    
    for li in [0, hotspot_layer, n_layers-1]:
        dirs = []
        for word in test_words:
            vecs = []
            for ti in range(len(TEMPLATES)):
                if ti in template_residuals[word] and li in template_residuals[word][ti]:
                    vecs.append(template_residuals[word][ti][li])
            if len(vecs) >= 2:
                mean_vec = np.mean(vecs, axis=0)
                for v in vecs:
                    deviation = v - mean_vec
                    if np.linalg.norm(deviation) > 1e-10:
                        dirs.append(deviation / np.linalg.norm(deviation))
        if dirs:
            template_directions[li] = np.array(dirs)
    
    # 3. 消融实验: 在template热点层移除template方向
    # 对每个测试句, 计算template成分, 减去它, 观察输出
    print(f"\n  --- 消融template方向(热点层L{hotspot_layer}) ---")
    
    test_prompts = [
        ("The dog is running", "dog"),
        ("I saw a cat today", "cat"),
        ("This apple looks fresh", "apple"),
        ("A hammer can be useful", "hammer"),
    ]
    
    if hotspot_layer not in template_directions:
        print(f"  Skip: no template direction at L{hotspot_layer}")
        release_model(model)
        return {}
    
    tmpl_dirs = template_directions[hotspot_layer]  # (n_dirs, d_model)
    
    # 构建template子空间的投影矩阵
    # P_template = D @ D^T, 其中D是归一化的template方向矩阵
    D = tmpl_dirs.T  # (d_model, n_dirs)
    P_template = D @ D.T  # (d_model, d_model)
    P_orthogonal = np.eye(d_model) - P_template  # 投影到template的正交补
    
    ablation_results = {}
    layers_list = get_layers(model)
    
    for prompt, target_word in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        # 找target位置
        target_pos = None
        for i, t in enumerate(tokens):
            if target_word.lower() in t.lower().replace("_", "").replace("G", ""):
                target_pos = i
                break
        if target_pos is None:
            target_pos = len(tokens) - 1
        
        # Baseline
        with torch.no_grad():
            logits_baseline = model(input_ids).logits
        baseline_top = get_top_k_tokens(logits_baseline, tokenizer, k=10)
        
        # 消融: 在template热点层, 移除template方向
        # 获取模型的dtype
        model_dtype = next(model.parameters()).dtype
        P_orth_tensor = torch.tensor(P_orthogonal, dtype=model_dtype, device=device)
        
        def make_ablation_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    # 对target位置, 投影到template的正交补
                    h = out[0, target_pos]  # (d_model,)
                    h_projected = P_orth_tensor @ h  # 移除template成分
                    out[0, target_pos] = h_projected
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    h = out[0, target_pos]
                    h_projected = P_orth_tensor @ h
                    out[0, target_pos] = h_projected
                    return out
            return hook
        
        hook = layers_list[hotspot_layer].register_forward_hook(make_ablation_hook())
        
        with torch.no_grad():
            try:
                logits_ablated = model(input_ids).logits
            except Exception as e:
                print(f"    Forward failed: {e}")
                hook.remove()
                continue
        
        hook.remove()
        
        ablated_top = get_top_k_tokens(logits_ablated, tokenizer, k=10)
        
        # 检查输出变化
        result = {
            "prompt": prompt,
            "baseline_top5": [(t["token"], round(t["prob"], 4)) for t in baseline_top[:5]],
            "ablated_top5": [(t["token"], round(t["prob"], 4)) for t in ablated_top[:5]],
            "top1_changed": baseline_top[0]["token"] != ablated_top[0]["token"],
            "top1_baseline": baseline_top[0]["token"],
            "top1_ablated": ablated_top[0]["token"],
        }
        ablation_results[prompt] = result
        
        print(f"    '{prompt}' -> baseline='{baseline_top[0]['token']}' ablated='{ablated_top[0]['token']}' changed={result['top1_changed']}")
    
    # 4. 对照: 在非热点层做同样消融
    control_layer = 0
    print(f"\n  --- 对照: 在L{control_layer}消融template方向 ---")
    
    if control_layer in template_directions:
        tmpl_dirs_ctrl = template_directions[control_layer]
        D_ctrl = tmpl_dirs_ctrl.T
        P_template_ctrl = D_ctrl @ D_ctrl.T
        P_orth_ctrl = np.eye(d_model) - P_template_ctrl
        model_dtype = next(model.parameters()).dtype
        P_orth_ctrl_tensor = torch.tensor(P_orth_ctrl, dtype=model_dtype, device=device)
        
        for prompt, target_word in test_prompts[:2]:
            input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            target_pos = len(tokens) - 1
            
            with torch.no_grad():
                logits_baseline = model(input_ids).logits
            baseline_top = get_top_k_tokens(logits_baseline, tokenizer, k=5)
            
            def make_ctrl_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        h = out[0, target_pos]
                        h_projected = P_orth_ctrl_tensor @ h
                        out[0, target_pos] = h_projected
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        h = out[0, target_pos]
                        h_projected = P_orth_ctrl_tensor @ h
                        out[0, target_pos] = h_projected
                        return out
                return hook
            
            hook = layers_list[control_layer].register_forward_hook(make_ctrl_hook())
            with torch.no_grad():
                try:
                    logits_ablated = model(input_ids).logits
                except:
                    hook.remove()
                    continue
            hook.remove()
            
            ablated_top = get_top_k_tokens(logits_ablated, tokenizer, k=5)
            print(f"    L{control_layer}: '{prompt}' -> baseline='{baseline_top[0]['token']}' ablated='{ablated_top[0]['token']}' changed={baseline_top[0]['token'] != ablated_top[0]['token']}")
    
    # 保存
    save_dir = OUTPUT_DIR / f"{args.model}_cclxiv"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXIV",
        "exp": 3,
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "template_hotspot_layer": hotspot_layer,
        "ablation_results": ablation_results,
        "n_template_directions": {str(k): v.shape[0] for k, v in template_directions.items()},
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(save_dir / "exp3_template_ablation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved to {save_dir / 'exp3_template_ablation.json'}")
    
    release_model(model)
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase CCLXIV: Deep Locking Mechanism")
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True,
                        help="1=propagation tracking, 2=attn vs ffn, 3=template ablation")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase CCLXIV: Deep Locking Mechanism")
    print(f"Model: {args.model}, Exp: {args.exp}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    if args.exp == 1:
        exp1_propagation_tracking(args)
    elif args.exp == 2:
        exp2_attn_vs_ffn(args)
    elif args.exp == 3:
        exp3_template_hotspot_ablation(args)
