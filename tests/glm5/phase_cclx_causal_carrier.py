"""
Phase CCLX: 因果载体追踪 — 从观察转向因果干预
================================================
核心问题: 概念编码是因果载体还是衍生现象?

4个子实验:
  Exp1: 残差流概念替换干预 — 在某层替换概念向量, 观察输出是否切换
  Exp2: 跨任务概念追踪 — 同一概念在不同句式(陈述/否定/疑问)中编码变化
  Exp3: 概念-逻辑交互 — "X is Y" vs "X is not Y" 中X的编码差异
  Exp4: 大样本PCA探针 — 100+概念, PCA降维, 修正d>>n过拟合

填充: KN-2a(跨任务概念), KN-2b(概念稳定性), UN-1(知识×逻辑交互)

用法:
  python phase_cclx_causal_carrier.py --model qwen3 --exp 1
  python phase_cclx_causal_carrier.py --model qwen3 --exp 2
  python phase_cclx_causal_carrier.py --model qwen3 --exp 3
  python phase_cclx_causal_carrier.py --model qwen3 --exp 4
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
    collect_layer_outputs,
)

OUTPUT_DIR = Path("results/causal_fiber")

# ===== 概念定义 =====
CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano"],
}
ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
for cat, items in CONCEPTS.items():
    for item in items:
        ALL_CONCEPTS.append(item)
        CONCEPT_CATEGORIES[item] = cat

# 抽象链
ABSTRACTION_CHAINS = [
    ["apple", "fruit", "food", "substance", "object"],
    ["dog", "animal", "organism", "entity", "thing"],
    ["hammer", "tool", "instrument", "device", "object"],
    ["mountain", "terrain", "landscape", "environment", "space"],
    ["rice", "grain", "crop", "produce", "matter"],
]


def proper_cos(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_top_k_tokens(logits, tokenizer, k=10):
    """从logits获取top-k token及其概率"""
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
    """
    收集各层残差流在target_token位置的表示
    
    Returns:
        residuals: list of np.ndarray [d_model], 按层序
        target_pos: int, target token在序列中的位置
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    # 找target token位置
    target_pos = None
    for i, t in enumerate(tokens):
        if target_token_str.lower() in t.lower().replace("▁", "").replace("Ġ", ""):
            target_pos = i
            break
    if target_pos is None:
        target_pos = len(tokens) - 1  # fallback: last position
    
    # 用hook收集各层输出
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


def run_intervention(model, tokenizer, device, prompt_source, prompt_target,
                     concept_source, concept_target, n_layers, intervention_layer):
    """
    因果干预实验: 在intervention_layer替换残差流
    
    流程:
    1. 运行source prompt, 收集各层残差
    2. 运行target prompt, 收集各层残差
    3. 在intervention_layer, 用source的残差替换target的残差
    4. 观察最终输出是否从target概念切换到source概念
    
    Returns:
        dict: 干预结果
    """
    # 1. 收集source和target的残差
    resid_source, pos_source = collect_residuals_at_position(
        model, tokenizer, device, prompt_source, concept_source, n_layers)
    resid_target, pos_target = collect_residuals_at_position(
        model, tokenizer, device, prompt_target, concept_target, n_layers)
    
    # 2. 获取baseline输出
    input_ids_target = tokenizer(prompt_target, return_tensors="pt").to(device).input_ids
    with torch.no_grad():
        logits_baseline = model(input_ids_target).logits
    baseline_top = get_top_k_tokens(logits_baseline, tokenizer, k=10)
    
    # 3. 干预: 在intervention_layer替换残差
    input_ids_source = tokenizer(prompt_source, return_tensors="pt").to(device).input_ids
    
    # 使用hook进行干预
    intervention_result = {}
    layers = get_layers(model)
    
    def make_intervention_hook(li, source_resid, pos_s, pos_t):
        def hook(module, input, output):
            if li == intervention_layer:
                if isinstance(output, tuple):
                    # 替换target位置的残差为source位置的残差
                    out = output[0].clone()
                    if pos_t < out.shape[1] and source_resid is not None:
                        # 计算差分向量: source概念 - target概念
                        delta = source_resid - out[0, pos_t].detach().float().cpu().numpy()
                        delta_tensor = torch.tensor(delta, dtype=out.dtype, device=out.device)
                        out[0, pos_t] += delta_tensor
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    if pos_t < out.shape[1] and source_resid is not None:
                        delta = source_resid - out[0, pos_t].detach().float().cpu().numpy()
                        delta_tensor = torch.tensor(delta, dtype=out.dtype, device=out.device)
                        out[0, pos_t] += delta_tensor
                    return out
            return output
        return hook
    
    hook = layers[intervention_layer].register_forward_hook(
        make_intervention_hook(intervention_layer, resid_source[intervention_layer], 
                               pos_source, pos_target))
    
    with torch.no_grad():
        try:
            logits_intervened = model(input_ids_target).logits
        except Exception as e:
            print(f"  Intervention forward failed: {e}")
            hook.remove()
            return {"error": str(e)}
    
    hook.remove()
    
    intervened_top = get_top_k_tokens(logits_intervened, tokenizer, k=10)
    
    # 4. 比较输出
    # source概念的token id
    source_tok_ids = tokenizer.encode(concept_source, add_special_tokens=False)
    target_tok_ids = tokenizer.encode(concept_target, add_special_tokens=False)
    
    # baseline中source和target概念的概率
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
        "baseline_top": baseline_top[:5],
        "intervened_top": intervened_top[:5],
        "baseline_probs": baseline_probs,
        "intervened_probs": intervened_probs,
        "switched": baseline_top[0]["token"] != intervened_top[0]["token"],
    }


# ============================================================
# Exp1: 残差流概念替换干预
# ============================================================
def exp1_intervention(args):
    """残差流概念替换干预: 在某层替换概念向量, 观察输出是否切换"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp1: 残差流概念替换干预 ({args.model}, {n_layers}层)")
    
    # 概念对: 同类 vs 异类
    same_cat_pairs = []
    diff_cat_pairs = []
    cats = list(CONCEPTS.keys())
    for ci, cat1 in enumerate(cats):
        for cj, cat2 in enumerate(cats):
            if ci >= cj:
                continue
            for w1 in CONCEPTS[cat1][:3]:
                for w2 in CONCEPTS[cat2][:3]:
                    if cat1 == cat2:
                        same_cat_pairs.append((w1, w2, cat1))
                    else:
                        diff_cat_pairs.append((w1, w2, cat1, cat2))
    
    # 同类对也要加上同类内部的
    for cat in cats:
        words = CONCEPTS[cat][:3]
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                same_cat_pairs.append((words[i], words[j], cat))
    
    # 选择10个同类对, 10个异类对
    same_cat_pairs = same_cat_pairs[:10]
    diff_cat_pairs = diff_cat_pairs[:10]
    
    templates = [
        "The {} is",
        "I saw a {} today",
        "This {} looks",
    ]
    
    results = {
        "same_cat": [],
        "diff_cat": [],
    }
    
    # 采样层(不要每层都试, 太慢)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    # 同类对干预
    print(f"\n  --- 同类概念对干预 (n={len(same_cat_pairs)}) ---")
    for w1, w2, cat in same_cat_pairs:
        prompt1 = templates[0].format(w1)
        prompt2 = templates[0].format(w2)
        pair_result = {"pair": (w1, w2), "category": cat, "layer_results": {}}
        
        for li in sample_layers:
            try:
                res = run_intervention(model, tokenizer, device, 
                                       prompt1, prompt2, w1, w2, n_layers, li)
                pair_result["layer_results"][str(li)] = {
                    "switched": res.get("switched", False),
                    "baseline_top1": res.get("baseline_top1", "?"),
                    "intervened_top1": res.get("intervened_top1", "?"),
                    "source_prob_baseline": res.get("baseline_probs", {}).get(w1, 0),
                    "source_prob_intervened": res.get("intervened_probs", {}).get(w1, 0),
                    "target_prob_baseline": res.get("baseline_probs", {}).get(w2, 0),
                    "target_prob_intervened": res.get("intervened_probs", {}).get(w2, 0),
                }
            except Exception as e:
                pair_result["layer_results"][str(li)] = {"error": str(e)}
        
        results["same_cat"].append(pair_result)
        print(f"    {w1}→{w2}: " + 
              ", ".join(f"L{li}={'Y' if pair_result['layer_results'].get(str(li),{}).get('switched') else 'N'}" 
                       for li in sample_layers[:5]))
    
    # 异类对干预
    print(f"\n  --- 异类概念对干预 (n={len(diff_cat_pairs)}) ---")
    for w1, w2, cat1, cat2 in diff_cat_pairs:
        prompt1 = templates[0].format(w1)
        prompt2 = templates[0].format(w2)
        pair_result = {"pair": (w1, w2), "categories": (cat1, cat2), "layer_results": {}}
        
        for li in sample_layers:
            try:
                res = run_intervention(model, tokenizer, device, 
                                       prompt1, prompt2, w1, w2, n_layers, li)
                pair_result["layer_results"][str(li)] = {
                    "switched": res.get("switched", False),
                    "baseline_top1": res.get("baseline_top1", "?"),
                    "intervened_top1": res.get("intervened_top1", "?"),
                    "source_prob_baseline": res.get("baseline_probs", {}).get(w1, 0),
                    "source_prob_intervened": res.get("intervened_probs", {}).get(w1, 0),
                    "target_prob_baseline": res.get("baseline_probs", {}).get(w2, 0),
                    "target_prob_intervened": res.get("intervened_probs", {}).get(w2, 0),
                }
            except Exception as e:
                pair_result["layer_results"][str(li)] = {"error": str(e)}
        
        results["diff_cat"].append(pair_result)
        print(f"    {w1}({cat1})→{w2}({cat2}): " + 
              ", ".join(f"L{li}={'Y' if pair_result['layer_results'].get(str(li),{}).get('switched') else 'N'}" 
                       for li in sample_layers[:5]))
    
    # 汇总统计
    summary = {"same_cat": {}, "diff_cat": {}}
    for group in ["same_cat", "diff_cat"]:
        switch_by_layer = defaultdict(list)
        prob_change_by_layer = defaultdict(list)
        for pair in results[group]:
            for li_str, lr in pair["layer_results"].items():
                if "error" in lr:
                    continue
                switch_by_layer[li_str].append(lr.get("switched", False))
                prob_change_by_layer[li_str].append(
                    lr.get("source_prob_intervened", 0) - lr.get("source_prob_baseline", 0))
        
        for li_str in switch_by_layer:
            switches = switch_by_layer[li_str]
            prob_changes = prob_change_by_layer[li_str]
            summary[group][li_str] = {
                "switch_rate": float(np.mean(switches)),
                "mean_prob_change": float(np.mean(prob_changes)),
                "n_pairs": len(switches),
            }
    
    # 保存
    out_dir = OUTPUT_DIR / f"{args.model}_cclx"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLX",
        "exp": 1,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
        "summary": summary,
    }
    
    with open(out_dir / "exp1_intervention.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印总结
    print(f"\n  === Exp1 总结 ===")
    for group in ["same_cat", "diff_cat"]:
        print(f"\n  {group}:")
        for li_str in sorted(summary[group].keys(), key=int):
            s = summary[group][li_str]
            print(f"    L{li_str}: switch_rate={s['switch_rate']:.3f}, "
                  f"prob_change={s['mean_prob_change']:.4f}")
    
    release_model(model)
    return output


# ============================================================
# Exp2: 跨任务概念追踪
# ============================================================
def exp2_cross_task(args):
    """跨任务概念追踪: 同一概念在不同句式中的编码变化"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    
    print(f"\n  Exp2: 跨任务概念追踪 ({args.model})")
    
    # 选择10个概念
    concepts = ["dog", "apple", "hammer", "mountain", "rice", 
                "cat", "cheese", "knife", "river", "bread"]
    
    # 不同句式模板
    templates = {
        "declarative": "The {} is",
        "negative": "The {} is not",
        "question": "Is the {}",
        "plural": "The {}s are",
        "relative": "I know the {} that",
    }
    
    results = {}
    for concept in concepts:
        cat = CONCEPT_CATEGORIES.get(concept, "unknown")
        concept_result = {"category": cat, "template_results": {}}
        
        # 收集各模板下的残差
        residuals_by_template = {}
        for tname, template in templates.items():
            prompt = template.format(concept)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, concept, n_layers)
            residuals_by_template[tname] = resid
        
        # 计算跨模板cos相似度(以declarative为参考)
        ref = residuals_by_template["declarative"]
        for tname, resid in residuals_by_template.items():
            cos_by_layer = []
            for li in range(n_layers):
                if ref[li] is not None and resid[li] is not None:
                    cos_by_layer.append(proper_cos(ref[li], resid[li]))
                else:
                    cos_by_layer.append(None)
            concept_result["template_results"][tname] = {
                "cos_with_declarative": {
                    str(li): c for li, c in enumerate(cos_by_layer) if c is not None
                }
            }
        
        # 计算模板间的概念特异性方向(cos)
        # 概念特异性 = 该模板的残差 - 所有模板残差的均值
        all_resids_at_layer = defaultdict(list)
        for tname, resid in residuals_by_template.items():
            for li in range(n_layers):
                if resid[li] is not None:
                    all_resids_at_layer[li].append(resid[li])
        
        concept_result["specificity_cos"] = {}
        for li in range(n_layers):
            if len(all_resids_at_layer[li]) < 2:
                continue
            mean_vec = np.mean(all_resids_at_layer[li], axis=0)
            spec_dirs = {t: r[li] - mean_vec for t, r in residuals_by_template.items() if r[li] is not None}
            
            # 计算模板对间的spec cos
            pair_cos = {}
            tnames = list(spec_dirs.keys())
            for i in range(len(tnames)):
                for j in range(i+1, len(tnames)):
                    c = proper_cos(spec_dirs[tnames[i]], spec_dirs[tnames[j]])
                    pair_cos[f"{tnames[i]}_vs_{tnames[j]}"] = c
            
            concept_result["specificity_cos"][str(li)] = {
                "mean_pair_cos": float(np.mean(list(pair_cos.values()))) if pair_cos else 0,
                "pair_cos": pair_cos,
            }
        
        results[concept] = concept_result
        # 简要打印
        last_li = str(n_layers - 1)
        mean_cos = concept_result.get("specificity_cos", {}).get(last_li, {}).get("mean_pair_cos", 0)
        print(f"    {concept}({cat}): L{n_layers-1} spec_cos={mean_cos:.3f}")
    
    # 汇总
    summary = {}
    for li_str in [str(n_layers // 4), str(n_layers // 2), str(3 * n_layers // 4), str(n_layers - 1)]:
        all_spec_cos = []
        for concept, cr in results.items():
            if li_str in cr.get("specificity_cos", {}):
                all_spec_cos.append(cr["specificity_cos"][li_str]["mean_pair_cos"])
        if all_spec_cos:
            summary[li_str] = {
                "mean_spec_cos": float(np.mean(all_spec_cos)),
                "std_spec_cos": float(np.std(all_spec_cos)),
                "n_concepts": len(all_spec_cos),
            }
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclx"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLX",
        "exp": 2,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "results": results,
        "summary": summary,
    }
    
    with open(out_dir / "exp2_cross_task.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp2 总结 ===")
    for li_str, s in sorted(summary.items(), key=lambda x: int(x[0])):
        print(f"    L{li_str}: mean_spec_cos={s['mean_spec_cos']:.3f}±{s['std_spec_cos']:.3f}")
    
    release_model(model)
    return output


# ============================================================
# Exp3: 概念-逻辑交互
# ============================================================
def exp3_concept_logic(args):
    """概念-逻辑交互: "X is Y" vs "X is not Y" 中X的编码差异"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    
    print(f"\n  Exp3: 概念-逻辑交互 ({args.model})")
    
    # 概念-属性对
    concept_attr_pairs = [
        ("dog", "animal"),
        ("apple", "fruit"),
        ("hammer", "tool"),
        ("mountain", "landscape"),
        ("salmon", "fish"),
        ("rice", "grain"),
        ("eagle", "bird"),
        ("knife", "weapon"),
        ("ocean", "water"),
        ("cat", "pet"),
    ]
    
    templates = {
        "affirmative": "The {} is a {}",
        "negative": "The {} is not a {}",
        "uncertain": "Maybe the {} is a {}",
    }
    
    results = {}
    for concept, attr in concept_attr_pairs:
        cat = CONCEPT_CATEGORIES.get(concept, "unknown")
        pair_result = {"concept": concept, "attribute": attr, "category": cat}
        
        # 收集各模板下LAST位置的残差(而非concept位置)
        # 因为concept位置的表示只依赖前面的token, 不受"is not"影响
        # LAST位置才能反映整个句子的语义
        residuals = {}
        for tname, template in templates.items():
            prompt = template.format(concept, attr)
            # 收集last position的残差
            input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
            seq_len = input_ids.shape[1]
            
            captured = {}
            layers_list = get_layers(model)
            
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            hooks = []
            for li in range(n_layers):
                hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids)
                except:
                    pass
            
            for h in hooks:
                h.remove()
            
            resid = []
            for li in range(n_layers):
                key = f"L{li}"
                if key in captured:
                    resid.append(captured[key][0, -1].numpy())  # LAST position
                else:
                    resid.append(None)
            
            residuals[tname] = resid
        
        # 1. 概念编码在肯定vs否定句中的差异
        cos_aff_neg = []
        for li in range(n_layers):
            if residuals["affirmative"][li] is not None and residuals["negative"][li] is not None:
                cos_aff_neg.append(proper_cos(
                    residuals["affirmative"][li], residuals["negative"][li]))
            else:
                cos_aff_neg.append(None)
        
        # 2. 肯定vs否定中概念特异性方向的变化
        # 差分向量: neg - aff
        delta_aff_neg = []
        delta_norm = []
        for li in range(n_layers):
            if residuals["affirmative"][li] is not None and residuals["negative"][li] is not None:
                d = residuals["negative"][li] - residuals["affirmative"][li]
                delta_aff_neg.append(d)
                delta_norm.append(float(np.linalg.norm(d)))
            else:
                delta_aff_neg.append(None)
                delta_norm.append(0)
        
        # 3. 差分向量与属性方向的关系
        W_U = get_W_U(model)
        attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
        attr_dir = W_U[attr_tok_ids[0]] if attr_tok_ids else np.zeros(model_info.d_model)
        attr_dir_norm = attr_dir / max(np.linalg.norm(attr_dir), 1e-10)
        
        delta_attr_cos = []
        for li in range(n_layers):
            if delta_aff_neg[li] is not None:
                delta_normed = delta_aff_neg[li] / max(np.linalg.norm(delta_aff_neg[li]), 1e-10)
                delta_attr_cos.append(proper_cos(delta_normed, attr_dir_norm))
            else:
                delta_attr_cos.append(None)
        
        pair_result["cos_affirmative_vs_negative"] = {
            str(li): c for li, c in enumerate(cos_aff_neg) if c is not None
        }
        pair_result["delta_norm"] = {
            str(li): n for li, n in enumerate(delta_norm)
        }
        pair_result["delta_attr_cos"] = {
            str(li): c for li, c in enumerate(delta_attr_cos) if c is not None
        }
        
        results[f"{concept}_{attr}"] = pair_result
        # 打印关键层
        last_li = str(n_layers - 1)
        mid_li = n_layers // 2
        aff_neg_val = cos_aff_neg[mid_li] if mid_li < len(cos_aff_neg) and cos_aff_neg[mid_li] is not None else None
        delta_val = delta_attr_cos[mid_li] if mid_li < len(delta_attr_cos) and delta_attr_cos[mid_li] is not None else None
        an_str = f"{aff_neg_val:.3f}" if aff_neg_val is not None else "N/A"
        da_str = f"{delta_val:.3f}" if delta_val is not None else "N/A"
        print(f"    {concept}→{attr}: cos(aff,neg)@L{mid_li}={an_str}, delta_attr_cos@L{mid_li}={da_str}")
    
    # 汇总
    summary = {}
    for li_str in [str(n_layers // 4), str(n_layers // 2), str(3 * n_layers // 4), str(n_layers - 1)]:
        li = int(li_str)
        all_cos = [r["cos_affirmative_vs_negative"].get(li_str) for r in results.values() 
                   if li_str in r.get("cos_affirmative_vs_negative", {})]
        all_delta_cos = [r["delta_attr_cos"].get(li_str) for r in results.values()
                        if li_str in r.get("delta_attr_cos", {})]
        all_delta_norm = [r["delta_norm"].get(li_str, 0) for r in results.values()]
        
        valid_cos = [c for c in all_cos if c is not None]
        valid_delta_cos = [c for c in all_delta_cos if c is not None]
        
        summary[li_str] = {
            "mean_cos_aff_neg": float(np.mean(valid_cos)) if valid_cos else 0,
            "mean_delta_attr_cos": float(np.mean(valid_delta_cos)) if valid_delta_cos else 0,
            "mean_delta_norm": float(np.mean(all_delta_norm)),
            "n_pairs": len(valid_cos),
        }
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclx"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLX",
        "exp": 3,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "results": results,
        "summary": summary,
    }
    
    with open(out_dir / "exp3_concept_logic.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp3 总结 ===")
    for li_str, s in sorted(summary.items(), key=lambda x: int(x[0])):
        print(f"    L{li_str}: cos(aff,neg)={s['mean_cos_aff_neg']:.3f}, "
              f"delta→attr_cos={s['mean_delta_attr_cos']:.3f}, "
              f"delta_norm={s['mean_delta_norm']:.2f}")
    
    release_model(model)
    return output


# ============================================================
# Exp4: 大样本PCA降维探针
# ============================================================
def exp4_large_sample_probe(args):
    """大样本PCA降维探针: 修正d>>n过拟合问题"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    
    print(f"\n  Exp4: 大样本PCA降维探针 ({args.model})")
    
    # 扩展概念集: 4类×15概念 = 60概念
    extended_concepts = {
        "animals": ["dog", "cat", "horse", "eagle", "shark", "snake", 
                     "lion", "bear", "whale", "rabbit", "tiger", "deer", 
                     "wolf", "fox", "owl"],
        "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango",
                     "banana", "grape", "pizza", "steak", "milk", "egg",
                     "pasta", "corn", "potato"],
        "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel",
                     "pliers", "screwdriver", "axe", "ruler", "scissors", "shovel",
                     "clamp", "file", "level"],
        "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano",
                     "lake", "cave", "cliff", "valley", "island", "glacier",
                     "canyon", "swamp", "reef"],
    }
    
    # 也更新全局CONCEPT_CATEGORIES以包含新概念
    for cat, items in extended_concepts.items():
        for item in items:
            CONCEPT_CATEGORIES[item] = cat
    
    all_concepts = []
    all_categories = []
    for cat, items in extended_concepts.items():
        for item in items:
            all_concepts.append(item)
            all_categories.append(cat)
    
    n_concepts = len(all_concepts)
    n_categories = len(extended_concepts)
    
    print(f"  概念数: {n_concepts}, 类别数: {n_categories}")
    
    templates = [
        "The {} is",
        "I saw a {} today",
    ]
    
    # 收集残差流
    print(f"  收集残差流...")
    residuals_by_concept = defaultdict(lambda: defaultdict(list))  # concept -> layer -> [resids]
    
    for ci, concept in enumerate(all_concepts):
        for template in templates:
            prompt = template.format(concept)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, concept, n_layers)
            for li in range(n_layers):
                if resid[li] is not None:
                    residuals_by_concept[concept][li].append(resid[li])
        
        if (ci + 1) % 10 == 0:
            print(f"    {ci+1}/{n_concepts} concepts done")
    
    # 对每个概念, 平均跨模板的残差
    mean_residuals = {}
    for concept in all_concepts:
        mean_residuals[concept] = {}
        for li in range(n_layers):
            if li in residuals_by_concept[concept] and residuals_by_concept[concept][li]:
                mean_residuals[concept][li] = np.mean(residuals_by_concept[concept][li], axis=0)
    
    # PCA降维后探针
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    probe_results = {}
    random_baselines = {}
    
    for li in sample_layers:
        # 构建数据矩阵
        X = []
        y = []
        for concept in all_concepts:
            if li in mean_residuals[concept]:
                X.append(mean_residuals[concept][li])
                y.append(all_categories.index(CONCEPT_CATEGORIES[concept]))
        
        if len(X) < 10:
            continue
        
        X = np.array(X)
        y = np.array(y)
        
        # 不同PCA维度的探针准确率
        pca_dims = [2, 5, 10, 15, 20, 30, 50]
        li_result = {"n_samples": len(X), "pca_probes": {}}
        
        for n_comp in pca_dims:
            if n_comp >= len(X):
                continue
            
            # PCA降维
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            
            # 5-fold交叉验证
            clf = LogisticRegression(max_iter=1000, C=1.0)
            try:
                cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy')
                li_result["pca_probes"][str(n_comp)] = {
                    "mean_acc": float(np.mean(scores)),
                    "std_acc": float(np.std(scores)),
                    "explained_var": float(np.sum(pca.explained_variance_ratio_)),
                }
            except Exception as e:
                li_result["pca_probes"][str(n_comp)] = {"error": str(e)}
        
        # 原始维度(无PCA)的探针 - 预期会过拟合
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
            scores_raw = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
            li_result["raw_probe"] = {
                "mean_acc": float(np.mean(scores_raw)),
                "std_acc": float(np.std(scores_raw)),
            }
        except Exception as e:
            li_result["raw_probe"] = {"error": str(e)}
        
        # 随机基线(打乱标签)
        n_permutations = 100
        random_accs = []
        y_shuffled = y.copy()
        for _ in range(n_permutations):
            np.random.shuffle(y_shuffled)
            try:
                scaler_r = StandardScaler()
                X_scaled_r = scaler_r.fit_transform(X)
                pca_r = PCA(n_components=5)
                X_pca_r = pca_r.fit_transform(X_scaled_r)
                clf_r = LogisticRegression(max_iter=500, C=1.0)
                cv_r = StratifiedKFold(n_splits=min(3, len(np.unique(y_shuffled))), 
                                       shuffle=True, random_state=42)
                scores_r = cross_val_score(clf_r, X_pca_r, y_shuffled, cv=cv_r, scoring='accuracy')
                random_accs.append(float(np.mean(scores_r)))
            except:
                pass
        
        li_result["random_baseline_pca5"] = {
            "mean_acc": float(np.mean(random_accs)) if random_accs else 0,
            "std_acc": float(np.std(random_accs)) if random_accs else 0,
            "n_permutations": len(random_accs),
        }
        
        probe_results[str(li)] = li_result
        print(f"    L{li}: pca5_acc={li_result['pca_probes'].get('5',{}).get('mean_acc','N/A'):.3f}, "
              f"raw_acc={li_result['raw_probe'].get('mean_acc','N/A'):.3f}, "
              f"random_pca5={li_result['random_baseline_pca5']['mean_acc']:.3f}")
    
    # 保存
    out_dir = OUTPUT_DIR / f"{args.model}_cclx"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLX",
        "exp": 4,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "n_concepts": n_concepts,
        "n_categories": n_categories,
        "results": probe_results,
    }
    
    with open(out_dir / "exp4_large_probe.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp4 总结 ===")
    print(f"  概念: {n_concepts}, 类别: {n_categories}")
    for li_str, r in sorted(probe_results.items(), key=lambda x: int(x[0])):
        pca5 = r.get("pca_probes", {}).get("5", {})
        raw = r.get("raw_probe", {})
        rand = r.get("random_baseline_pca5", {})
        print(f"    L{li_str}: PCA5={pca5.get('mean_acc','?'):.3f}±{pca5.get('std_acc','?'):.3f}, "
              f"Raw={raw.get('mean_acc','?'):.3f}, "
              f"Random={rand.get('mean_acc','?'):.3f}±{rand.get('std_acc','?'):.3f}")
    
    release_model(model)
    return output


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Phase CCLX: 因果载体追踪")
    print(f"Model: {args.model}, Exp: {args.exp}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    if args.exp == 1:
        exp1_intervention(args)
    elif args.exp == 2:
        exp2_cross_task(args)
    elif args.exp == 3:
        exp3_concept_logic(args)
    elif args.exp == 4:
        exp4_large_sample_probe(args)
