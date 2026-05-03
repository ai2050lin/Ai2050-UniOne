"""
CCL-H(250.8): 语法控制精细化 - 从"可控"到"可用"
==================================================
核心问题: CCL-G实现了100%语法角色操控, 但输出不连贯(garbage)
  → 需要找到最小有效扰动: 只改变语法角色, 不破坏语义

三合一实验:
  Exp1 (Phase 5A): ★★★★★ 最小有效扰动
    → 测试alpha=0.01, 0.05, 0.1, 0.2, 0.5, 1.0
    → 测量: KL散度, token变化率, 输出连贯性, 语法角色方向余弦
    → 目标: 找到alpha_min使得语法角色方向余弦显著变化但KL仍小

  Exp2 (Phase 5B): ★★★★ 跨句子语法方向泛化
    → 从句子集A提取语法方向, 应用到句子集B
    → 测量泛化成功率: 语法方向是否跨句子通用?

  Exp3 (Phase 5C): ★★★ 语法方向组合与精细控制
    → 测试d_nsubj + d_amod组合
    → 测试沿V_grammar PCA基方向的精细操控
    → 理解V_grammar坐标系统的几何结构
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 数据集 =====
# 训练集: 用于提取语法方向
SYNTAX_DATA = {
    "nsubj": {
        "sentences": [
            "The cat sat on the mat",
            "The dog ran through the park",
            "The bird sang a beautiful song",
            "The child played with the toys",
            "The student read the textbook",
            "The teacher explained the lesson",
            "The scientist discovered the formula",
            "The writer published the novel",
        ],
        "target_words": [
            "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
        ],
    },
    "dobj": {
        "sentences": [
            "She chased the cat away",
            "He found the dog outside",
            "They watched the bird closely",
            "We helped the child today",
            "I praised the student loudly",
            "You thanked the teacher warmly",
            "He remembered the scientist well",
            "She admired the writer greatly",
        ],
        "target_words": [
            "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
        ],
    },
    "amod": {
        "sentences": [
            "The beautiful cat sat quietly",
            "The large dog ran swiftly",
            "The small bird sang softly",
            "The young child played happily",
            "The bright student read carefully",
            "The wise teacher explained clearly",
            "The famous scientist discovered something",
            "The talented writer published recently",
        ],
        "target_words": [
            "beautiful", "large", "small", "young", "bright", "wise", "famous", "talented",
        ],
    },
    "advmod": {
        "sentences": [
            "The cat ran quickly home",
            "The dog barked loudly today",
            "The bird sang softly outside",
            "The child played happily inside",
            "The student read carefully alone",
            "The teacher spoke clearly again",
            "The scientist worked diligently there",
            "The writer typed rapidly now",
        ],
        "target_words": [
            "quickly", "loudly", "softly", "happily", "carefully", "clearly",
            "diligently", "rapidly",
        ],
    },
}

# 泛化测试集: 与训练集完全不同的句子
GENERALIZATION_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom",
            "The doctor treated the patient",
            "The artist painted the portrait",
            "The soldier defended the castle",
        ],
        "target_words": ["king", "doctor", "artist", "soldier"],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
        ],
        "target_words": ["king", "doctor", "artist", "soldier"],
    },
}

# 精细控制测试集
REFINEMENT_TEST = [
    ("The cat sat on the mat", "cat", "nsubj"),
    ("The dog ran through the park", "dog", "nsubj"),
    ("She chased the cat away", "cat", "dobj"),
    ("He found the dog outside", "dog", "dobj"),
    ("The beautiful cat sat quietly", "beautiful", "amod"),
    ("The large dog ran swiftly", "large", "amod"),
]


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def get_last_layer_hidden(model, tokenizer, device, sentence):
    """获取最后层的hidden states"""
    layers = get_layers(model)
    last_layer = layers[-1]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().clone()
        else:
            captured['h'] = output.detach().float().clone()
    
    h_handle = last_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
        base_logits = output.logits.detach().float()
    
    h_handle.remove()
    
    if 'h' not in captured:
        return None, None, toks
    
    return captured['h'], base_logits, toks


def compute_logits_from_hidden(model, hidden_states):
    """从hidden states通过final norm + lm_head计算logits"""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        normed = model.model.norm(hidden_states.to(model.model.norm.weight.device).to(model.model.norm.weight.dtype))
    else:
        normed = hidden_states
    
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(normed.to(model.lm_head.weight.dtype))
    else:
        logits = normed
    
    return logits.detach().float()


def collect_causal_directions(model, tokenizer, device, sentences, target_words, dep_type, n_max=None):
    """收集因果梯度方向"""
    layers = get_layers(model)
    last_layer = layers[-1]
    
    directions = []
    if n_max is None:
        n_max = len(sentences)
    
    for sent, target in zip(sentences[:n_max], target_words[:n_max]):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens, target)
        if dep_idx is None:
            continue
        
        captured_h = {}
        grad_h = {}
        
        def capture_and_grad_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone().detach().requires_grad_(True)
                captured_h['h'] = h
                def grad_callback(grad):
                    grad_h['grad'] = grad
                h.register_hook(grad_callback)
                return (h,) + output[1:]
            else:
                h = output.clone().detach().requires_grad_(True)
                captured_h['h'] = h
                def grad_callback(grad):
                    grad_h['grad'] = grad
                h.register_hook(grad_callback)
                return h
        
        hook_handle = last_layer.register_forward_hook(capture_and_grad_hook)
        
        output = model(**toks)
        logits = output.logits
        
        top_token = torch.argmax(logits[0, dep_idx]).item()
        top_prob = torch.softmax(logits[0, dep_idx].float(), dim=-1)[top_token]
        
        top_prob.backward()
        hook_handle.remove()
        
        if 'grad' in grad_h:
            grad_direction = grad_h['grad'][0, dep_idx, :].detach().float().cpu().numpy()
            grad_norm = np.linalg.norm(grad_direction)
            if grad_norm > 1e-10:
                grad_dir_normed = grad_direction / grad_norm
                directions.append({
                    'direction': grad_dir_normed,
                    'raw_direction': grad_direction,
                    'grad_norm': float(grad_norm),
                    'dep_type': dep_type,
                    'sentence': sent,
                    'target': target,
                    'dep_idx': dep_idx,
                })
        
        del output, logits
        torch.cuda.empty_cache()
    
    return directions


def perturb_and_analyze(model, tokenizer, h, dep_idx, base_probs, base_top_tokens, 
                        direction, alpha, h_norm):
    """
    沿方向扰动并全面分析效果
    
    Returns:
        dict: {kl, top5, token_changed, direction_cos, prob_shift}
    """
    perturbation = alpha * h_norm * direction
    pert_t = torch.tensor(perturbation, dtype=h.dtype, device=h.device)
    
    mod_h = h.clone()
    mod_h[0, dep_idx, :] += pert_t
    
    pert_logits = compute_logits_from_hidden(model, mod_h)
    pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
    
    # KL散度
    kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
    
    # Top tokens
    top_indices = np.argsort(pert_probs)[::-1][:10]
    top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
    
    # Token是否改变
    base_top1 = base_top_tokens[0][0] if base_top_tokens else ""
    pert_top1 = top_tokens[0][0] if top_tokens else ""
    token_changed = base_top1 != pert_top1
    
    # 概率转移: 原top-5的概率总和
    base_top5_prob = sum(p for _, p in base_top_tokens[:5])
    new_top5_in_old = sum(pert_probs[base_probs.argsort()[::-1][:5]])
    prob_shift = float(base_top5_prob - new_top5_in_old)
    
    return {
        "kl": float(kl),
        "top5": top_tokens[:5],
        "token_changed": token_changed,
        "prob_shift": float(prob_shift),
        "top1_prob": float(top_tokens[0][1]) if top_tokens else 0,
    }


def is_coherent_token(token_str):
    """简单判断token是否是连贯的英语"""
    # 去除空格前缀
    t = token_str.strip()
    if not t:
        return False
    # 排除特殊字符、标点序列、非ASCII等
    if any(c in t for c in ['{', '}', '$', '?', ')', '(', '.', '\n', '\t', ';', '#']):
        return len(t) > 3  # 长标点序列不连贯
    # 排除纯数字
    if t.isdigit():
        return False
    # 排除Unicode特殊字符
    try:
        t.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True


# ================================================================
# Exp1 (Phase 5A): 最小有效扰动
# ================================================================

def exp1_minimal_perturbation(model, tokenizer, device, model_info):
    """
    ★★★★★ 最小有效扰动
    测试不同alpha下语法方向扰动的效果:
    - 很小的alpha: 语法角色方向余弦变化多少?
    - 中等alpha: token是否改变? 连贯性如何?
    - 大alpha: 完全改变(已知)
    
    关键创新: 不只看top token变化, 还看:
    1. 方向余弦: 扰动后h'与各语法角色质心的余弦
    2. 连贯性: 扰动后top token是否有意义
    3. 渐进性: alpha从0.01到5.0的连续变化
    """
    print("\n" + "="*70)
    print("Exp1 (Phase 5A): 最小有效扰动")
    print("="*70)
    
    # 1. 收集4种语法角色的因果方向
    print("\n[1] Collecting causal directions for 4 roles...")
    
    all_dirs_data = []
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:6],
            data["target_words"][:6],
            role,
        )
        for d in dirs:
            all_dirs_data.append(d)
        print(f"  {role}: {len(dirs)} directions")
    
    if len(all_dirs_data) < 8:
        print("  ERROR: Too few directions")
        return None
    
    # 2. 计算语法角色质心方向
    print("\n[2] Computing grammar role centroids...")
    
    role_centroids = {}
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        role_dirs = [d['direction'] for d in all_dirs_data if d['dep_type'] == role]
        if role_dirs:
            centroid = np.mean(role_dirs, axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-10)
            role_centroids[role] = centroid
    
    # nsubj→dobj转换方向
    d_nsubj = role_centroids.get("nsubj")
    d_dobj = role_centroids.get("dobj")
    d_amod = role_centroids.get("amod")
    d_advmod = role_centroids.get("advmod")
    
    if d_nsubj is None or d_dobj is None:
        print("  ERROR: Missing nsubj or dobj centroids")
        return None
    
    # 质心间余弦
    nsubj_dobj_cos = float(np.dot(d_nsubj, d_dobj))
    print(f"  nsubj-dobj cos: {nsubj_dobj_cos:.4f}")
    
    # 3. ★★★ 对测试句子, 测试不同alpha的扰动效果
    print("\n[3] Testing perturbations at different alpha values...")
    
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    test_cases = [
        ("The cat sat on the mat", "cat", "nsubj", d_dobj, "nsubj→dobj"),
        ("The dog ran through the park", "dog", "nsubj", d_dobj, "nsubj→dobj"),
        ("She chased the cat away", "cat", "dobj", d_nsubj, "dobj→nsubj"),
        ("He found the dog outside", "dog", "dobj", d_nsubj, "dobj→nsubj"),
        ("The beautiful cat sat quietly", "beautiful", "amod", d_advmod, "amod→advmod"),
        ("The large dog ran swiftly", "large", "amod", d_advmod, "amod→advmod"),
    ]
    
    alpha_results = {a: {"kl": [], "change_rate": [], "coherent_rate": [], "direction_cos": [],
                          "top1_prob": [], "prob_shift": []} for a in alphas}
    
    per_case_results = []
    
    for sent, target, orig_role, perturb_dir, desc in test_cases:
        print(f"\n  [{desc}] {sent} / '{target}'")
        
        h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
        if h is None:
            continue
        
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        h_norm = float(torch.norm(h[0, dep_idx, :]).float())
        base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
        base_top_indices = np.argsort(base_probs)[::-1][:10]
        base_top_tokens = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
        
        # 原始hidden与各角色质心的余弦
        h_vec = h[0, dep_idx, :].float().cpu().numpy()
        h_normed = h_vec / max(np.linalg.norm(h_vec), 1e-10)
        orig_cos = {r: float(np.dot(h_normed, c)) for r, c in role_centroids.items()}
        print(f"    h_norm={h_norm:.2f}, orig cos: {', '.join(f'{r}={c:.3f}' for r, c in sorted(orig_cos.items(), key=lambda x: abs(x[1]), reverse=True)[:3])}")
        
        case_data = {
            "sentence": sent, "target": target, "orig_role": orig_role,
            "desc": desc, "h_norm": h_norm, "orig_cos": orig_cos,
            "base_top5": base_top_tokens[:5],
            "alphas": {},
        }
        
        for alpha in alphas:
            result = perturb_and_analyze(
                model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
                perturb_dir, alpha, h_norm
            )
            
            # 扰动后hidden与各角色质心的余弦
            pert_h = h_vec + alpha * h_norm * perturb_dir
            pert_h_normed = pert_h / max(np.linalg.norm(pert_h), 1e-10)
            pert_cos = {r: float(np.dot(pert_h_normed, c)) for r, c in role_centroids.items()}
            
            coherent = is_coherent_token(result["top5"][0][0]) if result["top5"] else False
            
            alpha_results[alpha]["kl"].append(result["kl"])
            alpha_results[alpha]["change_rate"].append(1.0 if result["token_changed"] else 0.0)
            alpha_results[alpha]["coherent_rate"].append(1.0 if coherent else 0.0)
            alpha_results[alpha]["direction_cos"].append(pert_cos)
            alpha_results[alpha]["top1_prob"].append(result["top1_prob"])
            alpha_results[alpha]["prob_shift"].append(result["prob_shift"])
            
            case_data["alphas"][str(alpha)] = {
                "kl": result["kl"],
                "top5": result["top5"],
                "token_changed": result["token_changed"],
                "coherent": coherent,
                "pert_cos": pert_cos,
                "top1_prob": result["top1_prob"],
            }
            
            if alpha in [0.05, 0.2, 1.0]:
                print(f"    α={alpha}: KL={result['kl']:.4f}, changed={result['token_changed']}, "
                      f"coherent={coherent}, top1={result['top5'][0] if result['top5'] else 'N/A'}")
                # 扰动后与目标角色的余弦变化
                target_role = desc.split("→")[1]
                if target_role in pert_cos:
                    cos_shift = pert_cos[target_role] - orig_cos.get(target_role, 0)
                    print(f"      cos({target_role}): {orig_cos.get(target_role, 0):.4f} → {pert_cos[target_role]:.4f} (Δ={cos_shift:+.4f})")
        
        per_case_results.append(case_data)
    
    # 4. ★★★ 汇总分析
    print("\n" + "="*50)
    print("Alpha Scan Summary:")
    print("="*50)
    
    summary = {}
    for alpha in alphas:
        s = alpha_results[alpha]
        mean_kl = np.mean(s["kl"]) if s["kl"] else 0
        change_rate = np.mean(s["change_rate"]) if s["change_rate"] else 0
        coherent_rate = np.mean(s["coherent_rate"]) if s["coherent_rate"] else 0
        mean_top1_prob = np.mean(s["top1_prob"]) if s["top1_prob"] else 0
        mean_prob_shift = np.mean(s["prob_shift"]) if s["prob_shift"] else 0
        
        summary[str(alpha)] = {
            "mean_kl": float(mean_kl),
            "change_rate": float(change_rate),
            "coherent_rate": float(coherent_rate),
            "mean_top1_prob": float(mean_top1_prob),
            "mean_prob_shift": float(mean_prob_shift),
        }
        
        print(f"  α={alpha:5.2f}: KL={mean_kl:8.4f}, change={change_rate:.0%}, "
              f"coherent={coherent_rate:.0%}, top1_prob={mean_top1_prob:.4f}")
    
    # 5. 找到最优alpha
    # 条件: change_rate > 0 且 coherent_rate > 0.5
    best_alpha = None
    for alpha in alphas:
        s = summary[str(alpha)]
        if s["change_rate"] > 0 and s["coherent_rate"] > 0.5:
            if best_alpha is None or alpha < best_alpha:
                best_alpha = alpha
    
    if best_alpha:
        print(f"\n  ★ Best alpha (change>0 AND coherent>50%): {best_alpha}")
    else:
        # 放宽条件: 找change_rate最大的coherent case
        for alpha in alphas:
            s = summary[str(alpha)]
            if s["coherent_rate"] > 0:
                best_alpha = alpha
                break
        print(f"\n  ★ Best alpha (relaxed): {best_alpha}")
    
    # 6. 随机方向基线
    print("\n[4] Random direction baseline...")
    d_model = model_info.d_model
    random_kls_by_alpha = {a: [] for a in alphas[:5]}  # 只测前5个alpha
    
    for _ in range(10):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        for sent, target, orig_role, _, _ in test_cases[:2]:  # 只用2个句子
            h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
            if h is None:
                continue
            tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target)
            if dep_idx is None:
                continue
            
            h_norm = float(torch.norm(h[0, dep_idx, :]).float())
            base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
            base_top_indices = np.argsort(base_probs)[::-1][:10]
            base_top_toks = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
            
            for alpha in alphas[:5]:
                result = perturb_and_analyze(
                    model, tokenizer, h, dep_idx, base_probs, base_top_toks,
                    rand_dir, alpha, h_norm
                )
                random_kls_by_alpha[alpha].append(result["kl"])
    
    random_summary = {}
    for alpha in alphas[:5]:
        rkl = np.mean(random_kls_by_alpha[alpha]) if random_kls_by_alpha[alpha] else 0
        random_summary[str(alpha)] = float(rkl)
        print(f"  α={alpha:.2f}: random KL={rkl:.4f}, grammar KL={summary[str(alpha)]['mean_kl']:.4f}, "
              f"lift={summary[str(alpha)]['mean_kl'] / max(rkl, 1e-10):.2f}x")
    
    results = {
        "model": model_info.name,
        "alphas": alphas,
        "summary": summary,
        "random_summary": random_summary,
        "best_alpha": best_alpha,
        "per_case_results": per_case_results,
    }
    
    return results


# ================================================================
# Exp2 (Phase 5B): 跨句子语法方向泛化
# ================================================================

def exp2_cross_sentence_generalization(model, tokenizer, device, model_info):
    """
    ★★★★ 跨句子语法方向泛化
    从训练集提取语法方向, 应用到完全不同的泛化测试集
    → 测量: 语法方向是否跨句子通用?
    
    方法:
    1. 从训练集(cat/dog/bird等)提取d_nsubj, d_dobj
    2. 应用到泛化集(king/doctor/artist等)
    3. 测量KL散度变化和token变化
    """
    print("\n" + "="*70)
    print("Exp2 (Phase 5B): 跨句子语法方向泛化")
    print("="*70)
    
    # 1. 从训练集提取语法方向
    print("\n[1] Extracting grammar directions from training set...")
    
    train_dirs = {}
    for role in ["nsubj", "dobj"]:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:6],
            data["target_words"][:6],
            role,
        )
        if dirs:
            centroid = np.mean([d['direction'] for d in dirs], axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-10)
            train_dirs[role] = centroid
        print(f"  {role}: {len(dirs)} directions, centroid norm={np.linalg.norm(centroid) if dirs else 0:.4f}")
    
    if "nsubj" not in train_dirs or "dobj" not in train_dirs:
        print("  ERROR: Missing training directions")
        return None
    
    # 2. 从泛化测试集也提取方向(作为对比)
    print("\n[2] Extracting grammar directions from generalization set...")
    
    gen_dirs = {}
    for role in ["nsubj", "dobj"]:
        data = GENERALIZATION_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"],
            data["target_words"],
            role,
        )
        if dirs:
            centroid = np.mean([d['direction'] for d in dirs], axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-10)
            gen_dirs[role] = centroid
        print(f"  {role}: {len(dirs)} directions")
    
    # 3. 训练方向 vs 泛化方向的余弦相似度
    print("\n[3] Cross-set direction similarity...")
    
    for role in ["nsubj", "dobj"]:
        if role in train_dirs and role in gen_dirs:
            cos = float(np.dot(train_dirs[role], gen_dirs[role]))
            print(f"  {role}: train-gen cos={cos:.4f}")
    
    # 4. ★★★ 应用训练方向到泛化测试集
    print("\n[4] Applying training directions to generalization set...")
    
    alphas = [0.1, 0.2, 0.5, 1.0, 2.0]
    
    results_list = []
    
    for role in ["nsubj", "dobj"]:
        gen_data = GENERALIZATION_DATA[role]
        other_role = "dobj" if role == "nsubj" else "nsubj"
        perturb_dir = train_dirs.get(other_role)
        
        if perturb_dir is None:
            continue
        
        for sent, target in zip(gen_data["sentences"], gen_data["target_words"]):
            h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
            if h is None:
                continue
            
            tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target)
            if dep_idx is None:
                continue
            
            h_norm = float(torch.norm(h[0, dep_idx, :]).float())
            base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
            base_top_indices = np.argsort(base_probs)[::-1][:10]
            base_top_tokens = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
            
            # 也用泛化方向做对比
            gen_perturb_dir = gen_dirs.get(other_role)
            
            case_result = {
                "sentence": sent, "target": target, "orig_role": role,
                "target_role": other_role,
                "base_top5": base_top_tokens[:5],
                "train_dir_results": {},
                "gen_dir_results": {},
            }
            
            for alpha in alphas:
                # 训练方向
                train_result = perturb_and_analyze(
                    model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
                    perturb_dir, alpha, h_norm
                )
                
                # 泛化方向
                if gen_perturb_dir is not None:
                    gen_result = perturb_and_analyze(
                        model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
                        gen_perturb_dir, alpha, h_norm
                    )
                    case_result["gen_dir_results"][str(alpha)] = {
                        "kl": gen_result["kl"],
                        "token_changed": gen_result["token_changed"],
                        "top5": gen_result["top5"],
                        "coherent": is_coherent_token(gen_result["top5"][0][0]) if gen_result["top5"] else False,
                    }
                
                case_result["train_dir_results"][str(alpha)] = {
                    "kl": train_result["kl"],
                    "token_changed": train_result["token_changed"],
                    "top5": train_result["top5"],
                    "coherent": is_coherent_token(train_result["top5"][0][0]) if train_result["top5"] else False,
                }
            
            results_list.append(case_result)
            
            # 打印α=0.5的结果
            tr = case_result["train_dir_results"].get("0.5", {})
            print(f"  [{role}→{other_role}] '{target}': train_dir KL={tr.get('kl', 0):.4f}, "
                  f"changed={tr.get('token_changed', False)}")
            if "0.5" in case_result.get("gen_dir_results", {}):
                gr = case_result["gen_dir_results"]["0.5"]
                print(f"    gen_dir  KL={gr.get('kl', 0):.4f}, changed={gr.get('token_changed', False)}")
    
    # 5. 汇总: 训练方向 vs 泛化方向的效果对比
    print("\n" + "="*50)
    print("Generalization Summary:")
    print("="*50)
    
    gen_summary = {}
    for alpha in alphas:
        train_kls = []
        gen_kls = []
        train_changes = []
        gen_changes = []
        
        for case in results_list:
            if str(alpha) in case["train_dir_results"]:
                train_kls.append(case["train_dir_results"][str(alpha)]["kl"])
                train_changes.append(1.0 if case["train_dir_results"][str(alpha)]["token_changed"] else 0.0)
            if str(alpha) in case.get("gen_dir_results", {}):
                gen_kls.append(case["gen_dir_results"][str(alpha)]["kl"])
                gen_changes.append(1.0 if case["gen_dir_results"][str(alpha)]["token_changed"] else 0.0)
        
        gen_summary[str(alpha)] = {
            "train_mean_kl": float(np.mean(train_kls)) if train_kls else 0,
            "gen_mean_kl": float(np.mean(gen_kls)) if gen_kls else 0,
            "train_change_rate": float(np.mean(train_changes)) if train_changes else 0,
            "gen_change_rate": float(np.mean(gen_changes)) if gen_changes else 0,
        }
        
        s = gen_summary[str(alpha)]
        print(f"  α={alpha:.1f}: train KL={s['train_mean_kl']:.4f} ({s['train_change_rate']:.0%}), "
              f"gen KL={s['gen_mean_kl']:.4f} ({s['gen_change_rate']:.0%})")
    
    # 方向余弦
    direction_similarity = {}
    for role in ["nsubj", "dobj"]:
        if role in train_dirs and role in gen_dirs:
            direction_similarity[role] = float(np.dot(train_dirs[role], gen_dirs[role]))
    
    results = {
        "model": model_info.name,
        "direction_similarity": direction_similarity,
        "gen_summary": gen_summary,
        "results_list": results_list,
    }
    
    return results


# ================================================================
# Exp3 (Phase 5C): 语法方向组合与精细控制
# ================================================================

def exp3_compositional_control(model, tokenizer, device, model_info):
    """
    ★★★ 语法方向组合与精细控制
    1. 测试d_nsubj + d_amod组合是否产生组合语法效果
    2. 沿V_grammar PCA基方向精细操控
    3. 理解V_grammar坐标系统
    """
    print("\n" + "="*70)
    print("Exp3 (Phase 5C): 语法方向组合与精细控制")
    print("="*70)
    
    # 1. 收集4种语法角色的因果方向
    print("\n[1] Collecting causal directions for 4 roles...")
    
    all_dirs_data = []
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:6],
            data["target_words"][:6],
            role,
        )
        for d in dirs:
            all_dirs_data.append(d)
        print(f"  {role}: {len(dirs)} directions")
    
    if len(all_dirs_data) < 8:
        print("  ERROR: Too few directions")
        return None
    
    # 2. 计算角色质心
    role_centroids = {}
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        role_dirs = [d['direction'] for d in all_dirs_data if d['dep_type'] == role]
        if role_dirs:
            centroid = np.mean(role_dirs, axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-10)
            role_centroids[role] = centroid
    
    # 3. PCA on 因果方向 → V_grammar的基
    D = np.array([d['direction'] for d in all_dirs_data])
    pca = PCA()
    pca.fit(D)
    var = pca.explained_variance_ratio_
    
    cumvar = np.cumsum(var)
    k_90 = np.searchsorted(cumvar, 0.90) + 1
    
    print(f"  PCA: k_90={k_90}, top5 var={[f'{v*100:.1f}%' for v in var[:5]]}")
    
    # 4. ★★★ 组合方向测试
    print("\n[2] Compositional direction test...")
    
    test_sent = "The cat sat on the mat"
    test_target = "cat"
    
    h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, test_sent)
    if h is None:
        return None
    
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, test_target)
    if dep_idx is None:
        return None
    
    h_norm = float(torch.norm(h[0, dep_idx, :]).float())
    base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
    base_top_indices = np.argsort(base_probs)[::-1][:10]
    base_top_tokens = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
    
    print(f"  Baseline top-5: {base_top_tokens[:5]}")
    
    # 测试组合方向
    alpha_small = 0.2  # 使用较小的alpha
    
    combinations = [
        ("nsubj_only", role_centroids.get("nsubj"), {"nsubj": 1.0}),
        ("dobj_only", role_centroids.get("dobj"), {"dobj": 1.0}),
        ("amod_only", role_centroids.get("amod"), {"amod": 1.0}),
        ("advmod_only", role_centroids.get("advmod"), {"advmod": 1.0}),
        ("nsubj+dobj", None, {"nsubj": 0.5, "dobj": 0.5}),
        ("nsubj+amod", None, {"nsubj": 0.5, "amod": 0.5}),
        ("dobj+advmod", None, {"dobj": 0.5, "advmod": 0.5}),
        ("nsubj-amod", None, {"nsubj": 0.7, "amod": -0.3}),
    ]
    
    combo_results = {}
    
    for name, direction, weights in combinations:
        if direction is None:
            # 组合方向
            combo_dir = np.zeros_like(role_centroids["nsubj"])
            for role, w in weights.items():
                if role in role_centroids:
                    combo_dir += w * role_centroids[role]
            norm = np.linalg.norm(combo_dir)
            if norm > 1e-10:
                combo_dir = combo_dir / norm
            direction = combo_dir
        
        result = perturb_and_analyze(
            model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
            direction, alpha_small, h_norm
        )
        
        # 扰动后与各角色的余弦
        pert_h = h[0, dep_idx, :].float().cpu().numpy() + alpha_small * h_norm * direction
        pert_h_normed = pert_h / max(np.linalg.norm(pert_h), 1e-10)
        role_cos = {r: float(np.dot(pert_h_normed, c)) for r, c in role_centroids.items()}
        
        combo_results[name] = {
            "weights": weights,
            "kl": result["kl"],
            "top5": result["top5"],
            "token_changed": result["token_changed"],
            "coherent": is_coherent_token(result["top5"][0][0]) if result["top5"] else False,
            "role_cosine": role_cos,
        }
        
        print(f"\n  {name} (weights={weights}):")
        print(f"    KL={result['kl']:.4f}, changed={result['token_changed']}, "
              f"coherent={combo_results[name]['coherent']}")
        print(f"    top5: {result['top5'][:3]}")
        print(f"    role cos: {', '.join(f'{r}={c:.3f}' for r, c in sorted(role_cos.items(), key=lambda x: abs(x[1]), reverse=True)[:3])}")
    
    # 5. ★★★ PCA基方向精细操控
    print("\n[3] PCA basis direction fine control...")
    
    n_pc_test = min(k_90, 6)
    
    pc_results = {}
    for pc_idx in range(n_pc_test):
        pc_dir = pca.components_[pc_idx]
        
        # 测试正负方向
        for sign_name, sign in [("+", 1.0), ("-", -1.0)]:
            result = perturb_and_analyze(
                model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
                sign * pc_dir, alpha_small, h_norm
            )
            
            # 与角色的余弦
            role_cos = {r: float(np.dot(pc_dir * sign, c)) for r, c in role_centroids.items()}
            
            key = f"PC{pc_idx+1}_{sign_name}"
            pc_results[key] = {
                "variance": float(var[pc_idx]),
                "kl": result["kl"],
                "top5": result["top5"],
                "token_changed": result["token_changed"],
                "coherent": is_coherent_token(result["top5"][0][0]) if result["top5"] else False,
                "role_cosine": role_cos,
            }
        
        # 找最相关的角色
        pos_cos = pc_results[f"PC{pc_idx+1}_+"]["role_cosine"]
        max_role = max(pos_cos, key=lambda k: abs(pos_cos[k]))
        max_cos = abs(pos_cos[max_role])
        print(f"  PC{pc_idx+1} (var={var[pc_idx]*100:.1f}%): ~{max_role}(cos={max_cos:.3f}), "
              f"KL+={pc_results[f'PC{pc_idx+1}_+']['kl']:.4f}")
    
    # 6. ★★★ 多alpha PCA操控
    print("\n[4] Multi-alpha PCA control...")
    
    # 选最可解释的PC (PC1或PC2)
    best_pc = 0  # PC1
    best_pc_dir = pca.components_[best_pc]
    
    multi_alpha_results = {}
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        result = perturb_and_analyze(
            model, tokenizer, h, dep_idx, base_probs, base_top_tokens,
            best_pc_dir, alpha, h_norm
        )
        multi_alpha_results[str(alpha)] = {
            "kl": result["kl"],
            "top5": result["top5"],
            "token_changed": result["token_changed"],
            "coherent": is_coherent_token(result["top5"][0][0]) if result["top5"] else False,
        }
    
    # 7. 第二个测试句子
    test_sent2 = "She chased the cat away"
    test_target2 = "cat"
    
    h2, base_logits2, toks2 = get_last_layer_hidden(model, tokenizer, device, test_sent2)
    if h2 is not None:
        tokens2 = [safe_decode(tokenizer, t) for t in toks2.input_ids[0].tolist()]
        dep_idx2 = find_token_index(tokens2, test_target2)
        
        if dep_idx2 is not None:
            h_norm2 = float(torch.norm(h2[0, dep_idx2, :]).float())
            base_probs2 = torch.softmax(base_logits2[0, dep_idx2].float(), dim=-1).cpu().numpy()
            base_top2 = [(safe_decode(tokenizer, int(i)), float(base_probs2[i])) 
                         for i in np.argsort(base_probs2)[::-1][:10]]
            
            multi_alpha2 = {}
            for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                result = perturb_and_analyze(
                    model, tokenizer, h2, dep_idx2, base_probs2, base_top2,
                    best_pc_dir, alpha, h_norm2
                )
                multi_alpha2[str(alpha)] = {
                    "kl": result["kl"],
                    "top5": result["top5"],
                    "token_changed": result["token_changed"],
                    "coherent": is_coherent_token(result["top5"][0][0]) if result["top5"] else False,
                }
    
    results = {
        "model": model_info.name,
        "pca_k90": int(k_90),
        "pca_top5_var": [float(v) for v in var[:5]],
        "combo_results": combo_results,
        "pc_results": pc_results,
        "multi_alpha_pc1": multi_alpha_results,
        "multi_alpha_pc1_sent2": multi_alpha2 if h2 is not None else None,
    }
    
    return results


# ================================================================
# 主函数
# ================================================================

def run_experiment(model_name, exp_num):
    """运行指定实验"""
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    try:
        if exp_num == 1:
            result = exp1_minimal_perturbation(model, tokenizer, device, model_info)
        elif exp_num == 2:
            result = exp2_cross_sentence_generalization(model, tokenizer, device, model_info)
        elif exp_num == 3:
            result = exp3_compositional_control(model, tokenizer, device, model_info)
        else:
            print(f"Unknown experiment: {exp_num}")
            result = None
    finally:
        release_model(model)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="CCL-H: Grammar Control Refinement")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment number (0=all, 1, 2, 3)")
    args = parser.parse_args()
    
    model_name = args.model
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "glm5_temp")
    os.makedirs(out_dir, exist_ok=True)
    
    exp_names = {1: "exp1", 2: "exp2", 3: "exp3"}
    
    exps_to_run = [1, 2, 3] if args.exp == 0 else [args.exp]
    
    for exp_num in exps_to_run:
        print(f"\n{'#'*70}")
        print(f"# Running Exp{exp_num} on {model_name}")
        print(f"{'#'*70}")
        
        result = run_experiment(model_name, exp_num)
        
        if result is not None:
            out_file = os.path.join(out_dir, f"cclh_{exp_names[exp_num]}_{model_name}_results.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
