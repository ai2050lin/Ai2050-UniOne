"""
Phase CCLXI: 中间层因果干预 — 验证概念编码的因果性
================================================
核心问题: CCLX发现深层干预无效, 但中间层(L10-L20)是否有效?

3个子实验:
  Exp1: 中间层概念替换干预 — 在L5/L10/L15/L20替换概念向量, 观察输出切换
  Exp2: 中间层方向注入 — 注入类别方向(动物/食物/工具/自然), 观察输出偏移
  Exp3: 逐层干预精细扫描 — 逐层干预, 精确测量哪一层干预最有效

填充: KN-1c(编码层级), KN-1d(稀疏vs稠密), UN-1(因果性)

用法:
  python phase_cclxi_midlayer_intervention.py --model qwen3 --exp 1
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


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


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


def collect_residuals_at_position(model, tokenizer, device, prompt, target_token_str, n_layers):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    target_pos = None
    for i, t in enumerate(tokens):
        if target_token_str.lower() in t.lower().replace("▁", "").replace("Ġ", ""):
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
    
    residuals = []
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            residuals.append(captured[key][0, target_pos].numpy())
        else:
            residuals.append(None)
    
    return residuals, target_pos


def run_intervention_at_layer(model, tokenizer, device, prompt_source, prompt_target,
                               concept_source, concept_target, n_layers, intervention_layer):
    """在指定层替换残差流(完整替换, 非差分叠加)"""
    resid_source, pos_source = collect_residuals_at_position(
        model, tokenizer, device, prompt_source, concept_source, n_layers)
    resid_target, pos_target = collect_residuals_at_position(
        model, tokenizer, device, prompt_target, concept_target, n_layers)
    
    if resid_source[intervention_layer] is None or resid_target[intervention_layer] is None:
        return {"error": "missing residual"}
    
    # baseline
    input_ids_target = tokenizer(prompt_target, return_tensors="pt").to(device).input_ids
    with torch.no_grad():
        logits_baseline = model(input_ids_target).logits
    baseline_top = get_top_k_tokens(logits_baseline, tokenizer, k=10)
    
    # 干预: 替换
    layers = get_layers(model)
    source_vec = resid_source[intervention_layer]
    
    def intervention_hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0].clone()
            if pos_target < out.shape[1]:
                # 替换: target位置 = source位置的残差
                new_vec = torch.tensor(source_vec, dtype=out.dtype, device=out.device)
                out[0, pos_target] = new_vec
            return (out,) + output[1:]
        else:
            out = output.clone()
            if pos_target < out.shape[1]:
                new_vec = torch.tensor(source_vec, dtype=out.dtype, device=out.device)
                out[0, pos_target] = new_vec
            return out
    
    hook = layers[intervention_layer].register_forward_hook(intervention_hook)
    
    with torch.no_grad():
        try:
            logits_intervened = model(input_ids_target).logits
        except Exception as e:
            hook.remove()
            return {"error": str(e)}
    
    hook.remove()
    
    intervened_top = get_top_k_tokens(logits_intervened, tokenizer, k=10)
    
    # 概率变化
    source_tok_ids = tokenizer.encode(concept_source, add_special_tokens=False)
    target_tok_ids = tokenizer.encode(concept_target, add_special_tokens=False)
    
    baseline_probs = {}
    intervened_probs = {}
    for tok_id in source_tok_ids:
        baseline_probs[concept_source] = torch.softmax(logits_baseline[0, -1], dim=-1)[tok_id].item()
        intervened_probs[concept_source] = torch.softmax(logits_intervened[0, -1], dim=-1)[tok_id].item()
    for tok_id in target_tok_ids:
        baseline_probs[concept_target] = torch.softmax(logits_baseline[0, -1], dim=-1)[tok_id].item()
        intervened_probs[concept_target] = torch.softmax(logits_intervened[0, -1], dim=-1)[tok_id].item()
    
    return {
        "intervention_layer": intervention_layer,
        "concept_source": concept_source,
        "concept_target": concept_target,
        "baseline_top1": baseline_top[0]["token"],
        "intervened_top1": intervened_top[0]["token"],
        "baseline_top5": [(t["token"], round(t["prob"], 4)) for t in baseline_top[:5]],
        "intervened_top5": [(t["token"], round(t["prob"], 4)) for t in intervened_top[:5]],
        "baseline_probs": baseline_probs,
        "intervened_probs": intervened_probs,
        "switched": baseline_top[0]["token"] != intervened_top[0]["token"],
        "source_prob_lift": intervened_probs.get(concept_source, 0) - baseline_probs.get(concept_source, 0),
        "target_prob_drop": baseline_probs.get(concept_target, 0) - intervened_probs.get(concept_target, 0),
    }


# ============================================================
# Exp1: 中间层概念替换干预(精细层扫描)
# ============================================================
def exp1_midlayer_intervention(args):
    """中间层概念替换: 精确扫描每层的干预效果"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp1: 中间层概念替换干预 ({args.model}, {n_layers}层, d={d_model})")
    
    # 异类概念对(干预效果应更明显)
    cross_cat_pairs = [
        ("dog", "apple", "animals", "food"),
        ("hammer", "mountain", "tools", "nature"),
        ("cat", "knife", "animals", "tools"),
        ("rice", "ocean", "food", "nature"),
        ("eagle", "cheese", "animals", "food"),
        ("drill", "river", "tools", "nature"),
    ]
    
    template = "The {} is"
    
    # 逐层扫描
    results = []
    
    for w_source, w_target, cat_s, cat_t in cross_cat_pairs:
        prompt_s = template.format(w_source)
        prompt_t = template.format(w_target)
        pair_result = {
            "source": w_source, "target": w_target,
            "source_cat": cat_s, "target_cat": cat_t,
            "layer_results": {}
        }
        
        # 每3层扫描一次(覆盖全面但不会太慢)
        scan_layers = list(range(0, n_layers, 3)) + [n_layers - 1]
        scan_layers = sorted(set(scan_layers))
        
        for li in scan_layers:
            try:
                res = run_intervention_at_layer(
                    model, tokenizer, device,
                    prompt_s, prompt_t, w_source, w_target,
                    n_layers, li
                )
                pair_result["layer_results"][str(li)] = {
                    "switched": res.get("switched", False),
                    "source_prob_lift": round(res.get("source_prob_lift", 0), 6),
                    "target_prob_drop": round(res.get("target_prob_drop", 0), 6),
                    "baseline_top1": res.get("baseline_top1", "?"),
                    "intervened_top1": res.get("intervened_top1", "?"),
                }
            except Exception as e:
                pair_result["layer_results"][str(li)] = {"error": str(e)}
            
            # 每层都清理GPU缓存
            if li % 6 == 0:
                torch.cuda.empty_cache()
        
        results.append(pair_result)
        # 打印关键信息
        best_li = None
        best_lift = -999
        for li_str, lr in pair_result["layer_results"].items():
            if "error" not in lr and lr.get("source_prob_lift", 0) > best_lift:
                best_lift = lr["source_prob_lift"]
                best_li = li_str
        print(f"    {w_source}({cat_s})→{w_target}({cat_t}): best=L{best_li}(lift={best_lift:.4f})")
    
    # 汇总: 每层的平均switch_rate和prob_lift
    summary = {}
    all_layers = set()
    for r in results:
        all_layers.update(r["layer_results"].keys())
    
    for li_str in sorted(all_layers, key=int):
        switches = []
        lifts = []
        drops = []
        for r in results:
            lr = r["layer_results"].get(li_str, {})
            if "error" in lr:
                continue
            switches.append(lr.get("switched", False))
            lifts.append(lr.get("source_prob_lift", 0))
            drops.append(lr.get("target_prob_drop", 0))
        
        if switches:
            summary[li_str] = {
                "switch_rate": float(np.mean(switches)),
                "mean_source_lift": float(np.mean(lifts)),
                "mean_target_drop": float(np.mean(drops)),
                "n_pairs": len(switches),
            }
    
    # 找最佳干预层
    best_layer = max(summary.keys(), key=lambda k: summary[k]["mean_source_lift"]) if summary else None
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXI",
        "exp": 1,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
        "summary": summary,
        "best_intervention_layer": best_layer,
    }
    
    with open(out_dir / "exp1_midlayer_intervention.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp1 总结 ===")
    print(f"  最佳干预层: L{best_layer}")
    for li_str in sorted(summary.keys(), key=int):
        s = summary[li_str]
        marker = " <<<" if li_str == best_layer else ""
        print(f"    L{li_str}: switch={s['switch_rate']:.3f}, "
              f"source_lift={s['mean_source_lift']:.4f}, "
              f"target_drop={s['mean_target_drop']:.4f}{marker}")
    
    release_model(model)
    return output


# ============================================================
# Exp2: 中间层方向注入(类别方向)
# ============================================================
def exp2_direction_injection(args):
    """注入类别方向, 观察输出是否向该类别偏移"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp2: 中间层方向注入 ({args.model})")
    
    # 计算类别方向: 类内均值 - 全局均值
    template = "The {} is"
    all_words = []
    for cat, words in CONCEPTS.items():
        all_words.extend(words)
    
    # 收集所有词的中间层残差
    print("  收集概念残差...")
    concept_resids = {}  # concept -> {layer: mean_vec}
    mid_layer = n_layers // 2
    
    for cat, words in CONCEPTS.items():
        for word in words:
            prompt = template.format(word)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, word, n_layers)
            concept_resids[word] = resid
    
    # 计算类别方向(在中间层)
    cat_directions = {}
    global_mean = np.zeros(d_model)
    n_valid = 0
    
    for word, resid in concept_resids.items():
        if resid[mid_layer] is not None:
            global_mean += resid[mid_layer]
            n_valid += 1
    if n_valid > 0:
        global_mean /= n_valid
    
    for cat, words in CONCEPTS.items():
        cat_vecs = []
        for word in words:
            if word in concept_resids and concept_resids[word][mid_layer] is not None:
                cat_vecs.append(concept_resids[word][mid_layer] - global_mean)
        if cat_vecs:
            cat_dir = np.mean(cat_vecs, axis=0)
            norm = np.linalg.norm(cat_dir)
            if norm > 1e-10:
                cat_dir = cat_dir / norm
            cat_directions[cat] = cat_dir
    
    # 注入实验: 对一个中性prompt注入类别方向
    neutral_prompts = [
        "The thing is",
        "It is a",
        "This object is",
    ]
    
    results = {}
    injection_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    injection_layers = sorted(set(injection_layers))
    betas = [1.0, 3.0, 5.0, 10.0]
    
    for prompt in neutral_prompts:
        prompt_result = {"prompt": prompt, "injection_results": {}}
        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
        
        # baseline
        with torch.no_grad():
            logits_base = model(input_ids).logits
        base_top = get_top_k_tokens(logits_base, tokenizer, k=10)
        prompt_result["baseline_top5"] = [(t["token"], round(t["prob"], 4)) for t in base_top[:5]]
        
        for li in injection_layers:
            layer_result = {}
            for cat, direction in cat_directions.items():
                cat_result = {}
                for beta in betas:
                    # 注入方向
                    layers = get_layers(model)
                    
                    def make_inject_hook(layer_idx, dir_vec, beta_val, pos=0):
                        def hook(module, input, output):
                            if layer_idx != li:
                                return output
                            if isinstance(output, tuple):
                                out = output[0].clone()
                                delta = torch.tensor(
                                    beta_val * dir_vec, dtype=out.dtype, device=out.device)
                                out[0, pos] += delta
                                return (out,) + output[1:]
                            else:
                                out = output.clone()
                                delta = torch.tensor(
                                    beta_val * dir_vec, dtype=out.dtype, device=out.device)
                                out[0, pos] += delta
                                return out
                        return hook
                    
                    hook = layers[li].register_forward_hook(
                        make_inject_hook(li, direction, beta))
                    
                    with torch.no_grad():
                        try:
                            logits_inj = model(input_ids).logits
                            inj_top = get_top_k_tokens(logits_inj, tokenizer, k=5)
                            
                            # 计算该类别词的概率提升
                            cat_word_probs = {}
                            for word in CONCEPTS[cat][:3]:
                                tok_ids = tokenizer.encode(word, add_special_tokens=False)
                                if tok_ids:
                                    base_p = torch.softmax(logits_base[0, -1], dim=-1)[tok_ids[0]].item()
                                    inj_p = torch.softmax(logits_inj[0, -1], dim=-1)[tok_ids[0]].item()
                                    cat_word_probs[word] = {
                                        "baseline_prob": round(base_p, 6),
                                        "injected_prob": round(inj_p, 6),
                                        "lift": round(inj_p - base_p, 6),
                                    }
                            
                            cat_result[f"beta_{beta}"] = {
                                "top1": inj_top[0]["token"],
                                "cat_word_probs": cat_word_probs,
                            }
                        except:
                            cat_result[f"beta_{beta}"] = {"error": "forward failed"}
                    
                    hook.remove()
                
                layer_result[cat] = cat_result
            
            prompt_result["injection_results"][str(li)] = layer_result
            torch.cuda.empty_cache()
        
        results[prompt] = prompt_result
    
    # 汇总
    summary = {}
    for li in injection_layers:
        li_str = str(li)
        lifts_by_cat = defaultdict(list)
        for prompt, pr in results.items():
            if li_str in pr.get("injection_results", {}):
                for cat, cat_res in pr["injection_results"][li_str].items():
                    for beta_key, br in cat_res.items():
                        if "cat_word_probs" in br:
                            for word, wp in br["cat_word_probs"].items():
                                lifts_by_cat[cat].append(wp["lift"])
        
        if lifts_by_cat:
            summary[li_str] = {
                cat: {"mean_lift": float(np.mean(v)), "n": len(v)}
                for cat, v in lifts_by_cat.items()
            }
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXI",
        "exp": 2,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
        "summary": summary,
    }
    
    with open(out_dir / "exp2_direction_injection.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp2 总结 ===")
    for li_str in sorted(summary.keys(), key=int):
        s = summary[li_str]
        lifts = [v["mean_lift"] for v in s.values()]
        print(f"    L{li_str}: mean_lift={np.mean(lifts):.6f}, " +
              ", ".join(f"{c}={v['mean_lift']:.6f}" for c, v in s.items()))
    
    release_model(model)
    return output


# ============================================================
# Exp3: 逻辑推理注意力头消融
# ============================================================
def exp3_logic_head_ablation(args):
    """消融中间层注意力头, 观察逻辑推理能力变化"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp3: 逻辑推理注意力头消融 ({args.model})")
    
    # 逻辑推理测试句
    logic_sentences = [
        ("The dog is an animal", "animal"),
        ("The apple is a fruit", "fruit"),
        ("The hammer is a tool", "tool"),
        ("The mountain is a landscape", "landscape"),
        ("If it rains, the ground gets wet", "wet"),
        ("All birds can fly, so the eagle can fly", "fly"),
    ]
    
    # 否定逻辑测试句
    negation_sentences = [
        ("The dog is not a plant", "plant"),
        ("The apple is not a tool", "tool"),
        ("The hammer is not food", "food"),
    ]
    
    # 获取注意力头数
    layer0 = get_layers(model)[0]
    n_heads = layer0.self_attn.num_heads if hasattr(layer0.self_attn, 'num_heads') else d_model // 64
    
    print(f"  注意力头数: {n_heads}/层")
    
    # 选择要消融的层(中间层为主)
    ablation_layers = list(range(max(0, n_layers//4), min(n_layers, 3*n_layers//4), 2))
    
    results = {"ablation_by_layer": {}, "ablation_by_head": {}}
    
    for li in ablation_layers:
        layer_result = {"logic_correct": [], "negation_correct": []}
        
        # 逐头消融
        head_results = {}
        for head_idx in range(min(n_heads, 8)):  # 最多测8个头
            head_correct = 0
            head_total = 0
            
            for sentence, expected_word in logic_sentences[:4]:
                input_ids = tokenizer(sentence, return_tensors="pt").to(device).input_ids
                
                # baseline
                with torch.no_grad():
                    logits_base = model(input_ids).logits
                base_top1 = get_top_k_tokens(logits_base, tokenizer, k=1)[0]["token"]
                
                # 消融该头
                layers = get_layers(model)
                
                def make_head_zero_hook(layer_idx, h_idx, n_h):
                    def hook(module, input, output):
                        if layer_idx != li:
                            return output
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            # 将该头的输出置零
                            d_head = out.shape[-1] // n_h
                            out[:, :, h_idx*d_head:(h_idx+1)*d_head] = 0
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            d_head = out.shape[-1] // n_h
                            out[:, :, h_idx*d_head:(h_idx+1)*d_head] = 0
                            return out
                    return hook
                
                hook = layers[li].register_forward_hook(
                    make_head_zero_hook(li, head_idx, n_heads))
                
                with torch.no_grad():
                    try:
                        logits_abl = model(input_ids).logits
                    except:
                        hook.remove()
                        continue
                
                hook.remove()
                
                abl_top1 = get_top_k_tokens(logits_abl, tokenizer, k=1)[0]["token"]
                
                # 检查是否包含expected_word
                base_has = expected_word.lower() in base_top1.lower()
                abl_has = expected_word.lower() in abl_top1.lower()
                
                if base_has and not abl_has:
                    head_correct += 1  # 消融导致丢失→该头重要
                head_total += 1
            
            if head_total > 0:
                head_results[str(head_idx)] = {
                    "importance": head_correct / head_total,
                    "n_tested": head_total,
                }
        
        layer_result["head_importance"] = head_results
        results["ablation_by_layer"][str(li)] = layer_result
        print(f"    L{li}: " + ", ".join(
            f"H{h}={v['importance']:.2f}" for h, v in sorted(head_results.items(), key=lambda x: -x[1]['importance'])[:3]))
        
        torch.cuda.empty_cache()
    
    # 汇总: 找最重要的头
    all_head_importance = defaultdict(list)
    for li_str, lr in results["ablation_by_layer"].items():
        for h_str, hr in lr.get("head_importance", {}).items():
            all_head_importance[f"L{li_str}_H{h_str}"].append(hr["importance"])
    
    top_heads = sorted(all_head_importance.items(), key=lambda x: -np.mean(x[1]))[:10]
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxi"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXI",
        "exp": 3,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads},
        "results": results,
        "top_logic_heads": [(h, round(float(np.mean(imp)), 3)) for h, imp in top_heads],
    }
    
    with open(out_dir / "exp3_logic_head_ablation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp3 总结 ===")
    print(f"  逻辑推理关键注意力头(前5):")
    for h, imp in top_heads[:5]:
        print(f"    {h}: importance={float(np.mean(imp)):.3f}")
    
    release_model(model)
    return output


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Phase CCLXI: 中间层因果干预")
    print(f"Model: {args.model}, Exp: {args.exp}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    if args.exp == 1:
        exp1_midlayer_intervention(args)
    elif args.exp == 2:
        exp2_direction_injection(args)
    elif args.exp == 3:
        exp3_logic_head_ablation(args)
