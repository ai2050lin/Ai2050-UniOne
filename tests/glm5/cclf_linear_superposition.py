"""
CCL-F(250.6): 因果方向线性子空间 + 扩展语法角色 + 流形坐标
============================================================
核心问题: 6-13维因果方向流形是线性子空间还是弯曲流形?
用户核心洞察: 最小作用量原理 → 应该是线性子空间(最经济)

三合一实验:
  Exp1 (Phase 4J): ★★★★★ 因果方向的线性叠加性
    → 如果两个因果方向的线性组合仍是因果方向 → 语法有线性结构
    → 这是最小作用量原理的关键检验
    → 方法: 
      a. 收集多个因果方向 d_i
      b. 线性组合: d_new = α*d_i + β*d_j
      c. 测量 d_new 的因果效应(KL)
      d. 对比: d_new的KL vs d_i的KL vs 随机方向的KL
      e. 如果 d_new 的KL ≥ min(KL_i, KL_j) → 线性叠加有效

  Exp2 (Phase 4K): ★★★★ 扩展语法角色空间
    → 除nsubj/dobj外, 测试更多语法角色
    → 不同语法角色的因果方向是否在同一流形中?
    → 使用主语(nsubj), 宾语(dobj), 修饰语(amod), 状语(advmod)

  Exp3 (Phase 4L): ★★★ 流形坐标发现
    → 在因果方向流形中, 寻找可解释的坐标
    → 分析: 流形各主成分与语言特征的关联
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 扩展数据集: 4种语法角色 =====
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
            "The cat slept on the sofa",
            "The dog barked at the stranger",
            "The bird flew over the house",
            "The child smiled at the teacher",
        ],
        "target_words": [
            "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
            "cat", "dog", "bird", "child",
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
            "They fed the cat carefully",
            "I walked the dog slowly",
            "We rescued the bird quickly",
            "She comforted the child gently",
        ],
        "target_words": [
            "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
            "cat", "dog", "bird", "child",
        ],
    },
    "amod": {
        # 形容词修饰语: "the [adj] [noun]" 中 [adj] 的位置
        "sentences": [
            "The beautiful cat sat quietly",
            "The large dog ran swiftly",
            "The small bird sang softly",
            "The young child played happily",
            "The bright student read carefully",
            "The wise teacher explained clearly",
            "The famous scientist discovered something",
            "The talented writer published recently",
            "The cute cat slept peacefully",
            "The strong dog barked loudly",
            "The tiny bird flew gracefully",
            "The happy child smiled broadly",
        ],
        "target_words": [
            "beautiful", "large", "small", "young", "bright", "wise", "famous", "talented",
            "cute", "strong", "tiny", "happy",
        ],
    },
    "advmod": {
        # 副词修饰语: 动词后的副词
        "sentences": [
            "The cat ran quickly home",
            "The dog barked loudly today",
            "The bird sang softly outside",
            "The child played happily inside",
            "The student read carefully alone",
            "The teacher spoke clearly again",
            "The scientist worked diligently there",
            "The writer typed rapidly now",
            "The cat jumped suddenly up",
            "The dog walked slowly away",
            "The bird flew gently down",
            "The child laughed freely here",
        ],
        "target_words": [
            "quickly", "loudly", "softly", "happily", "carefully", "clearly",
            "diligently", "rapidly", "suddenly", "slowly", "gently", "freely",
        ],
    },
}


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
            captured['h'] = output[0].detach().float()
        else:
            captured['h'] = output.detach().float()
    
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
        
        # Autograd
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
                    'hidden_states': captured_h['h'].detach(),
                    'base_probs': torch.softmax(logits[0, dep_idx].detach().float(), dim=-1).cpu().numpy(),
                })
        
        del output, logits
        torch.cuda.empty_cache()
    
    return directions


def perturb_and_measure_kl(model, hidden_states, dep_idx, base_probs, direction, alpha, h_norm):
    """沿方向扰动hidden states并测量KL散度"""
    perturbation = alpha * h_norm * direction
    pert_t = torch.tensor(perturbation, dtype=hidden_states.dtype, device=hidden_states.device)
    
    mod_h = hidden_states.clone()
    mod_h[0, dep_idx, :] += pert_t
    
    pert_logits = compute_logits_from_hidden(model, mod_h)
    pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
    
    kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
    return kl


# ================================================================
# Exp1: 因果方向的线性叠加性 (Phase 4J)
# ================================================================

def exp1_linear_superposition(model, tokenizer, device, model_info):
    """
    ★★★★★ 因果方向的线性叠加性
    核心检验: 如果语法流形是线性子空间 → 线性组合仍应是因果方向
    
    方法:
    1. 收集nsubj和dobj的因果方向 d_1, d_2, ..., d_n
    2. 构造线性组合: d_new = α*d_i + β*d_j, 然后归一化
    3. 测量 d_new 的因果效应(KL)
    4. 对比:
       a. d_new的KL vs d_i的KL vs d_j的KL
       b. d_new的KL vs 随机方向的KL
       c. 同角色组合(nsubj+nsubj) vs 跨角色组合(nsubj+dobj)
    """
    print("\n" + "="*70)
    print("Exp1 (Phase 4J): 因果方向的线性叠加性")
    print("="*70)
    
    # 1. 收集因果方向
    print("\n[1] Collecting causal directions...")
    
    nsubj_dirs = collect_causal_directions(
        model, tokenizer, device,
        SYNTAX_DATA["nsubj"]["sentences"][:8],
        SYNTAX_DATA["nsubj"]["target_words"][:8],
        "nsubj",
    )
    print(f"  nsubj: {len(nsubj_dirs)} directions")
    
    dobj_dirs = collect_causal_directions(
        model, tokenizer, device,
        SYNTAX_DATA["dobj"]["sentences"][:8],
        SYNTAX_DATA["dobj"]["target_words"][:8],
        "dobj",
    )
    print(f"  dobj: {len(dobj_dirs)} directions")
    
    all_dirs = nsubj_dirs + dobj_dirs
    n_nsubj = len(nsubj_dirs)
    
    if len(all_dirs) < 4:
        print("  ERROR: Too few directions")
        return None
    
    # 2. 计算h_norm用于缩放alpha
    h_norms = [np.linalg.norm(d['hidden_states'][0, d['dep_idx'], :].float().cpu().numpy()) for d in all_dirs]
    h_norm_mean = float(np.mean(h_norms))
    
    # 3. ★★★ 基线: 每个方向的因果效应
    print("\n[2] Measuring baseline causal effects...")
    alpha_test = 2.0  # 使用α=2.0作为标准测试
    
    dir_kls = {}
    for i, d in enumerate(all_dirs):
        kl = perturb_and_measure_kl(
            model, d['hidden_states'], d['dep_idx'],
            d['base_probs'], d['direction'], alpha_test, h_norm_mean
        )
        dir_kls[i] = kl
    
    nsubj_kls = [dir_kls[i] for i in range(len(nsubj_dirs))]
    dobj_kls = [dir_kls[i] for i in range(len(nsubj_dirs), len(all_dirs))]
    
    print(f"  nsubj KL(α=2): mean={np.mean(nsubj_kls):.4f}, std={np.std(nsubj_kls):.4f}")
    print(f"  dobj KL(α=2): mean={np.mean(dobj_kls):.4f}, std={np.std(dobj_kls):.4f}")
    
    # 4. ★★★ 同角色线性叠加测试
    print("\n[3] Same-role linear superposition test...")
    
    same_role_results = {"nsubj": [], "dobj": []}
    
    for role, dirs in [("nsubj", nsubj_dirs), ("dobj", dobj_dirs)]:
        if len(dirs) < 2:
            continue
        
        combo_kls = []
        individual_kls = []
        
        for i in range(min(len(dirs), 4)):
            for j in range(i + 1, min(len(dirs), 4)):
                # 线性组合: d_new = d_i + d_j, 归一化
                d_new = dirs[i]['direction'] + dirs[j]['direction']
                d_new_norm = np.linalg.norm(d_new)
                if d_new_norm < 1e-10:
                    continue
                d_new = d_new / d_new_norm
                
                # 用d_i的句子测试
                kl_new = perturb_and_measure_kl(
                    model, dirs[i]['hidden_states'], dirs[i]['dep_idx'],
                    dirs[i]['base_probs'], d_new, alpha_test, h_norm_mean
                )
                
                kl_i = dir_kls[i if role == "nsubj" else n_nsubj + i]
                kl_j = dir_kls[j if role == "nsubj" else n_nsubj + j]
                
                combo_kls.append(kl_new)
                individual_kls.append((kl_i, kl_j))
        
        if combo_kls:
            mean_combo = np.mean(combo_kls)
            mean_indiv = np.mean([max(a, b) for a, b in individual_kls])
            min_indiv = np.mean([min(a, b) for a, b in individual_kls])
            
            same_role_results[role] = {
                "combo_kl_mean": float(mean_combo),
                "individual_kl_max_mean": float(mean_indiv),
                "individual_kl_min_mean": float(min_indiv),
                "superposition_ratio": float(mean_combo / max(mean_indiv, 1e-10)),
                "n_pairs": len(combo_kls),
            }
            
            print(f"  {role}: combo_KL={mean_combo:.4f}, "
                  f"max_indiv_KL={mean_indiv:.4f}, min_indiv_KL={min_indiv:.4f}, "
                  f"ratio={mean_combo/max(mean_indiv, 1e-10):.3f}")
    
    # 5. ★★★ 跨角色线性叠加测试
    print("\n[4] Cross-role linear superposition test...")
    
    cross_combo_kls = []
    cross_individual_kls = []
    
    for i in range(min(len(nsubj_dirs), 4)):
        for j in range(min(len(dobj_dirs), 4)):
            d_new = nsubj_dirs[i]['direction'] + dobj_dirs[j]['direction']
            d_new_norm = np.linalg.norm(d_new)
            if d_new_norm < 1e-10:
                continue
            d_new = d_new / d_new_norm
            
            # 用nsubj句子测试
            kl_on_nsubj = perturb_and_measure_kl(
                model, nsubj_dirs[i]['hidden_states'], nsubj_dirs[i]['dep_idx'],
                nsubj_dirs[i]['base_probs'], d_new, alpha_test, h_norm_mean
            )
            
            # 用dobj句子测试
            kl_on_dobj = perturb_and_measure_kl(
                model, dobj_dirs[j]['hidden_states'], dobj_dirs[j]['dep_idx'],
                dobj_dirs[j]['base_probs'], d_new, alpha_test, h_norm_mean
            )
            
            kl_nsubj_i = dir_kls[i]
            kl_dobj_j = dir_kls[len(nsubj_dirs) + j]
            
            cross_combo_kls.append((kl_on_nsubj, kl_on_dobj))
            cross_individual_kls.append((kl_nsubj_i, kl_dobj_j))
    
    if cross_combo_kls:
        mean_cross_nsubj = np.mean([k[0] for k in cross_combo_kls])
        mean_cross_dobj = np.mean([k[1] for k in cross_combo_kls])
        mean_indiv_nsubj = np.mean([k[0] for k in cross_individual_kls])
        mean_indiv_dobj = np.mean([k[1] for k in cross_individual_kls])
        
        cross_results = {
            "cross_on_nsubj_kl": float(mean_cross_nsubj),
            "cross_on_dobj_kl": float(mean_cross_dobj),
            "indiv_nsubj_kl": float(mean_indiv_nsubj),
            "indiv_dobj_kl": float(mean_indiv_dobj),
            "ratio_nsubj": float(mean_cross_nsubj / max(mean_indiv_nsubj, 1e-10)),
            "ratio_dobj": float(mean_cross_dobj / max(mean_indiv_dobj, 1e-10)),
            "n_pairs": len(cross_combo_kls),
        }
        
        print(f"  Cross on nsubj: combo_KL={mean_cross_nsubj:.4f}, indiv_KL={mean_indiv_nsubj:.4f}, "
              f"ratio={mean_cross_nsubj/max(mean_indiv_nsubj, 1e-10):.3f}")
        print(f"  Cross on dobj:  combo_KL={mean_cross_dobj:.4f}, indiv_KL={mean_indiv_dobj:.4f}, "
              f"ratio={mean_cross_dobj/max(mean_indiv_dobj, 1e-10):.3f}")
    else:
        cross_results = {}
    
    # 6. ★★★ 随机方向基线
    print("\n[5] Random direction baseline...")
    
    random_kls = []
    d_model = all_dirs[0]['direction'].shape[0]
    
    for trial in range(20):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        # 在多个句子上测试随机方向
        trial_kls = []
        for d in all_dirs[:4]:
            kl = perturb_and_measure_kl(
                model, d['hidden_states'], d['dep_idx'],
                d['base_probs'], rand_dir, alpha_test, h_norm_mean
            )
            trial_kls.append(kl)
        random_kls.append(np.mean(trial_kls))
    
    mean_random_kl = np.mean(random_kls)
    
    print(f"  Random KL(α=2): mean={mean_random_kl:.4f}")
    print(f"  ★ Causal / Random lift: {np.mean(list(dir_kls.values())) / max(mean_random_kl, 1e-10):.2f}x")
    
    # 7. ★★★ 加权线性叠加测试
    print("\n[6] Weighted superposition test...")
    
    # 测试不同权重: d_new = α*d_i + (1-α)*d_j
    weighted_results = []
    
    for i in range(min(len(nsubj_dirs), 3)):
        for j in range(min(len(dobj_dirs), 3)):
            for w in [0.2, 0.4, 0.6, 0.8]:
                d_new = w * nsubj_dirs[i]['direction'] + (1 - w) * dobj_dirs[j]['direction']
                d_new_norm = np.linalg.norm(d_new)
                if d_new_norm < 1e-10:
                    continue
                d_new = d_new / d_new_norm
                
                kl_nsubj = perturb_and_measure_kl(
                    model, nsubj_dirs[i]['hidden_states'], nsubj_dirs[i]['dep_idx'],
                    nsubj_dirs[i]['base_probs'], d_new, alpha_test, h_norm_mean
                )
                
                kl_dobj = perturb_and_measure_kl(
                    model, dobj_dirs[j]['hidden_states'], dobj_dirs[j]['dep_idx'],
                    dobj_dirs[j]['base_probs'], d_new, alpha_test, h_norm_mean
                )
                
                weighted_results.append({
                    "weight_nsubj": w,
                    "weight_dobj": 1 - w,
                    "kl_on_nsubj": float(kl_nsubj),
                    "kl_on_dobj": float(kl_dobj),
                })
    
    # 按权重汇总
    weight_summary = {}
    for wr in weighted_results:
        w = wr["weight_nsubj"]
        if w not in weight_summary:
            weight_summary[w] = {"kl_nsubj": [], "kl_dobj": []}
        weight_summary[w]["kl_nsubj"].append(wr["kl_on_nsubj"])
        weight_summary[w]["kl_dobj"].append(wr["kl_on_dobj"])
    
    print("  Weight(nsubj) | KL_on_nsubj | KL_on_dobj")
    for w in sorted(weight_summary.keys()):
        mn = np.mean(weight_summary[w]["kl_nsubj"])
        md = np.mean(weight_summary[w]["kl_dobj"])
        print(f"  w={w:.1f}          | {mn:.4f}       | {md:.4f}")
    
    # 8. ★★★ 减法组合: d_i - d_j (差异方向)
    print("\n[7] Subtraction test: d_i - d_j...")
    
    sub_results = []
    for i in range(min(len(nsubj_dirs), 3)):
        for j in range(min(len(dobj_dirs), 3)):
            d_diff = nsubj_dirs[i]['direction'] - dobj_dirs[j]['direction']
            d_diff_norm = np.linalg.norm(d_diff)
            if d_diff_norm < 1e-10:
                continue
            d_diff = d_diff / d_diff_norm
            
            kl_nsubj = perturb_and_measure_kl(
                model, nsubj_dirs[i]['hidden_states'], nsubj_dirs[i]['dep_idx'],
                nsubj_dirs[i]['base_probs'], d_diff, alpha_test, h_norm_mean
            )
            
            sub_results.append(float(kl_nsubj))
    
    if sub_results:
        print(f"  Subtraction KL: mean={np.mean(sub_results):.4f}, "
              f"vs individual mean={np.mean(nsubj_kls):.4f}")
    
    results = {
        "model": model_info.name,
        "alpha_test": alpha_test,
        "h_norm_mean": float(h_norm_mean),
        "individual_kls": {
            "nsubj_mean": float(np.mean(nsubj_kls)),
            "dobj_mean": float(np.mean(dobj_kls)),
            "all_mean": float(np.mean(list(dir_kls.values()))),
        },
        "random_baseline_kl": float(mean_random_kl),
        "same_role_superposition": same_role_results,
        "cross_role_superposition": cross_results,
        "weighted_superposition": weight_summary,
        "subtraction_kl_mean": float(np.mean(sub_results)) if sub_results else 0.0,
    }
    
    # 汇总判断
    print("\n" + "="*50)
    print("★★★ 线性叠加性判断 ★★★")
    all_individual_kl = np.mean(list(dir_kls.values()))
    
    same_role_ratios = []
    for role, data in same_role_results.items():
        if "superposition_ratio" in data:
            same_role_ratios.append(data["superposition_ratio"])
    
    if same_role_ratios:
        mean_same_ratio = np.mean(same_role_ratios)
        print(f"  同角色叠加比: {mean_same_ratio:.3f}")
        if mean_same_ratio > 0.5:
            print("  → 同角色线性叠加有效! (ratio > 0.5)")
        else:
            print("  → 同角色线性叠加效果弱")
    
    if cross_results and "ratio_nsubj" in cross_results:
        print(f"  跨角色叠加比(nsubj): {cross_results['ratio_nsubj']:.3f}")
        print(f"  跨角色叠加比(dobj):  {cross_results['ratio_dobj']:.3f}")
    
    print(f"  因果/随机 lift: {all_individual_kl / max(mean_random_kl, 1e-10):.2f}x")
    
    return results


# ================================================================
# Exp2: 扩展语法角色空间 (Phase 4K)
# ================================================================

def exp2_extended_syntax_roles(model, tokenizer, device, model_info):
    """
    ★★★★ 扩展语法角色空间
    4种语法角色的因果方向是否在同一流形中?
    """
    print("\n" + "="*70)
    print("Exp2 (Phase 4K): 扩展语法角色空间")
    print("="*70)
    
    # 1. 收集4种语法角色的因果方向
    print("\n[1] Collecting causal directions for 4 syntax roles...")
    
    all_role_dirs = {}
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:8],
            data["target_words"][:8],
            role,
        )
        all_role_dirs[role] = dirs
        print(f"  {role}: {len(dirs)} directions collected")
    
    # 2. 汇总所有方向
    all_dirs = []
    all_labels = []
    for role, dirs in all_role_dirs.items():
        for d in dirs:
            all_dirs.append(d['direction'])
            all_labels.append(role)
    
    if len(all_dirs) < 8:
        print("  ERROR: Too few directions")
        return None
    
    D = np.array(all_dirs)  # [n_dirs, d_model]
    
    # 3. PCA on 所有因果方向
    print("\n[2] PCA on all causal directions...")
    pca = PCA()
    pca.fit(D)
    var = pca.explained_variance_ratio_
    
    var_probs = var / np.sum(var)
    var_probs = var_probs[var_probs > 1e-15]
    entropy = -np.sum(var_probs * np.log(var_probs))
    eff_rank = np.exp(entropy)
    
    cumvar = np.cumsum(var)
    k_90 = np.searchsorted(cumvar, 0.90) + 1
    k_95 = np.searchsorted(cumvar, 0.95) + 1
    
    print(f"  Eff rank: {eff_rank:.1f}")
    print(f"  PC1: {var[0]*100:.2f}%, top5: {np.sum(var[:5])*100:.2f}%")
    print(f"  k_90: {k_90}, k_95: {k_95}")
    
    # 4. 各角色的质心和分布
    print("\n[3] Role centroid analysis...")
    role_centroids = {}
    for role, dirs in all_role_dirs.items():
        if len(dirs) >= 2:
            role_dirs = np.array([d['direction'] for d in dirs])
            centroid = np.mean(role_dirs, axis=0)
            centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)
            role_centroids[role] = centroid_norm
            
            # 簇内距离
            intra_dist = np.mean([np.linalg.norm(d - centroid) for d in role_dirs])
            print(f"  {role}: centroid_norm_ok, intra_dist={intra_dist:.4f}, n={len(dirs)}")
    
    # 5. ★★★ 角色间余弦矩阵
    print("\n[4] Inter-role cosine matrix...")
    roles = list(role_centroids.keys())
    cos_matrix = np.zeros((len(roles), len(roles)))
    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):
            cos_matrix[i, j] = float(np.dot(role_centroids[r1], role_centroids[r2]))
    
    print("  " + " " * 10 + "  ".join(f"{r:>10}" for r in roles))
    for i, r1 in enumerate(roles):
        row = "  ".join(f"{cos_matrix[i, j]:>10.4f}" for j in range(len(roles)))
        print(f"  {r1:>10}  {row}")
    
    # 6. ★★★ LDA分类: 4角色
    print("\n[5] Multi-role LDA classification...")
    y = np.array(all_labels)
    
    # 2-class: nsubj vs dobj
    nd_mask = np.isin(y, ['nsubj', 'dobj'])
    if np.sum(nd_mask) >= 4:
        lda_2 = LinearDiscriminantAnalysis(n_components=1)
        try:
            lda_2.fit(D[nd_mask], y[nd_mask])
            score_2 = lda_2.score(D[nd_mask], y[nd_mask])
            print(f"  nsubj vs dobj: {score_2:.3f}")
        except:
            score_2 = 0.5
    
    # 4-class: all roles
    lda_4 = LinearDiscriminantAnalysis(n_components=min(3, len(set(y)) - 1))
    try:
        lda_4.fit(D, y)
        score_4 = lda_4.score(D, y)
        print(f"  4-role classification: {score_4:.3f}")
    except:
        score_4 = 0.25
    
    # 7. ★★★ 各角色在PCA流形上的投影
    print("\n[6] Role projections on causal manifold...")
    
    k_manifold = k_95
    U_manifold = pca.components_[:k_manifold].T  # [d_model, k_manifold]
    
    role_proj_stats = {}
    for role, dirs in all_role_dirs.items():
        if len(dirs) < 2:
            continue
        role_dirs = np.array([d['direction'] for d in dirs])
        # 投影到流形
        proj = role_dirs @ U_manifold  # [n_dirs, k_manifold]
        # 流形投影能量比
        total_energy = np.sum(role_dirs ** 2, axis=1)
        manifold_energy = np.sum(proj ** 2, axis=1)
        energy_ratio = np.mean(manifold_energy) / max(np.mean(total_energy), 1e-10)
        
        role_proj_stats[role] = {
            "manifold_energy_ratio": float(energy_ratio),
            "n_dirs": len(dirs),
        }
        print(f"  {role}: manifold energy ratio = {energy_ratio:.4f}")
    
    # 8. 随机基线
    print("\n[7] Random baseline for eff_rank...")
    n_random = 20
    random_eff_ranks = []
    for _ in range(n_random):
        rand_dirs = np.random.randn(D.shape[0], D.shape[1])
        rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        pca_rand = PCA()
        pca_rand.fit(rand_dirs)
        var_rand = pca_rand.explained_variance_ratio_
        vp = var_rand / np.sum(var_rand)
        vp = vp[vp > 1e-15]
        random_eff_ranks.append(np.exp(-np.sum(vp * np.log(vp))))
    
    print(f"  Causal eff_rank: {eff_rank:.1f}")
    print(f"  Random eff_rank: {np.mean(random_eff_ranks):.1f}")
    print(f"  ★ Structure lift: {np.mean(random_eff_ranks) / max(eff_rank, 0.1):.2f}x")
    
    results = {
        "model": model_info.name,
        "n_roles": len(all_role_dirs),
        "eff_rank": float(eff_rank),
        "pc1_variance": float(var[0]),
        "top5_variance": float(np.sum(var[:5])),
        "k_90": int(k_90),
        "k_95": int(k_95),
        "random_eff_rank_mean": float(np.mean(random_eff_ranks)),
        "structure_lift": float(np.mean(random_eff_ranks) / max(eff_rank, 0.1)),
        "lda_2class": float(score_2) if 'score_2' in dir() else 0.0,
        "lda_4class": float(score_4),
        "inter_role_cos_matrix": {roles[i]: {roles[j]: float(cos_matrix[i, j]) for j in range(len(roles))} for i in range(len(roles))},
        "role_projection_stats": role_proj_stats,
        "n_dirs_per_role": {role: len(dirs) for role, dirs in all_role_dirs.items()},
    }
    
    return results


# ================================================================
# Exp3: 流形坐标发现 (Phase 4L)
# ================================================================

def exp3_manifold_coordinates(model, tokenizer, device, model_info):
    """
    ★★★ 流形坐标发现
    在因果方向流形中, 寻找可解释的坐标
    
    方法:
    1. 收集所有4种语法角色的因果方向
    2. 在流形的PCA子空间中, 分析每个PC与语言特征的关联
    3. 测试: PC坐标是否能预测语法角色?
    """
    print("\n" + "="*70)
    print("Exp3 (Phase 4L): 流形坐标发现")
    print("="*70)
    
    # 1. 收集所有语法角色的因果方向
    print("\n[1] Collecting all causal directions...")
    
    all_dirs_data = []
    for role in ["nsubj", "dobj", "amod", "advmod"]:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:8],
            data["target_words"][:8],
            role,
        )
        for d in dirs:
            all_dirs_data.append(d)
        print(f"  {role}: {len(dirs)} directions")
    
    if len(all_dirs_data) < 8:
        print("  ERROR: Too few directions")
        return None
    
    D = np.array([d['direction'] for d in all_dirs_data])
    labels = [d['dep_type'] for d in all_dirs_data]
    sentences = [d['sentence'] for d in all_dirs_data]
    targets = [d['target'] for d in all_dirs_data]
    
    # 2. PCA on 因果方向
    print("\n[2] PCA on causal directions...")
    pca = PCA()
    pca.fit(D)
    var = pca.explained_variance_ratio_
    
    cumvar = np.cumsum(var)
    k_90 = np.searchsorted(cumvar, 0.90) + 1
    
    print(f"  PC1: {var[0]*100:.2f}%, top5: {np.sum(var[:5])*100:.2f}%")
    print(f"  k_90: {k_90}")
    
    # 3. ★★★ 在PCA子空间中的投影坐标
    print("\n[3] Projection coordinates...")
    
    n_pc = min(k_90, 10)
    proj = pca.transform(D)[:, :n_pc]  # [n_dirs, n_pc]
    
    # 4. ★★★ 各PC与语法角色的关联
    print("\n[4] PC vs syntax role association...")
    
    unique_roles = sorted(set(labels))
    
    pc_role_stats = {}
    for pc_idx in range(min(n_pc, 6)):
        pc_values = proj[:, pc_idx]
        
        role_means = {}
        role_stds = {}
        for role in unique_roles:
            mask = np.array(labels) == role
            if np.sum(mask) > 0:
                role_means[role] = float(np.mean(pc_values[mask]))
                role_stds[role] = float(np.std(pc_values[mask]))
        
        # ANOVA-like: 角色间方差 / 角色内方差
        grand_mean = np.mean(pc_values)
        ss_between = sum(len([l for l in labels if l == r]) * (m - grand_mean)**2 
                         for r, m in role_means.items())
        ss_within = sum(np.sum((pc_values[np.array(labels) == r] - role_means[r])**2) 
                        for r in unique_roles if r in role_means and np.sum(np.array(labels) == r) > 0)
        
        f_ratio = ss_between / max(ss_within, 1e-10)
        
        pc_role_stats[f"PC{pc_idx+1}"] = {
            "variance_ratio": float(var[pc_idx]),
            "role_means": role_means,
            "role_stds": role_stds,
            "f_ratio": float(f_ratio),
        }
        
        means_str = ", ".join(f"{r}={role_means[r]:.3f}" for r in unique_roles if r in role_means)
        print(f"  PC{pc_idx+1} (var={var[pc_idx]*100:.1f}%): {means_str}, F={f_ratio:.2f}")
    
    # 5. ★★★ 坐标的可解释性: 角色分类
    print("\n[5] Role classification using PC coordinates...")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    
    X_coord = proj[:, :n_pc]
    y_role = np.array(labels)
    
    # 简单逻辑回归
    try:
        clf = LogisticRegression(max_iter=1000, multi_class='ovr')
        loo = LeaveOneOut()
        correct = 0
        total = 0
        for train_idx, test_idx in loo.split(X_coord):
            if len(set(y_role[train_idx])) < 2:
                continue
            clf.fit(X_coord[train_idx], y_role[train_idx])
            pred = clf.predict(X_coord[test_idx])
            if pred[0] == y_role[test_idx[0]]:
                correct += 1
            total += 1
        
        loo_acc = correct / max(total, 1)
        print(f"  LOO accuracy: {loo_acc:.3f} ({correct}/{total})")
    except:
        loo_acc = 0.0
        print(f"  LOO classification failed")
    
    # 6. ★★★ PC与LDA方向的对齐
    print("\n[6] PC vs LDA alignment...")
    
    lda_4 = LinearDiscriminantAnalysis(n_components=min(3, len(unique_roles) - 1))
    try:
        lda_4.fit(D, y_role)
        lda_dirs = lda_4.coef_  # [n_classes-1, d_model] or [n_classes, d_model]
        
        for li in range(min(lda_dirs.shape[0], 3)):
            lda_dir = lda_dirs[li] / max(np.linalg.norm(lda_dirs[li]), 1e-10)
            cos_with_pc1 = float(np.dot(lda_dir, pca.components_[0]))
            cos_with_pc2 = float(np.dot(lda_dir, pca.components_[1])) if n_pc > 1 else 0.0
            print(f"  LDA{li+1} || PC1={cos_with_pc1:.4f}, PC2={cos_with_pc2:.4f}")
    except:
        print("  LDA failed")
    
    # 7. ★★★ 2D可视化数据 (PC1 vs PC2)
    print("\n[7] 2D projection data...")
    proj_2d = proj[:, :2]
    
    viz_data = []
    for i in range(len(all_dirs_data)):
        viz_data.append({
            "PC1": float(proj_2d[i, 0]),
            "PC2": float(proj_2d[i, 1]),
            "role": labels[i],
            "sentence": sentences[i],
            "target": targets[i],
        })
    
    # 计算角色在PC1-PC2平面上的分散度
    role_spread = {}
    for role in unique_roles:
        mask = np.array(labels) == role
        if np.sum(mask) >= 2:
            role_proj = proj_2d[mask]
            centroid = np.mean(role_proj, axis=0)
            spread = np.mean(np.linalg.norm(role_proj - centroid, axis=1))
            role_spread[role] = float(spread)
    
    print("  Role spread in PC1-PC2 plane:")
    for role, spread in role_spread.items():
        print(f"    {role}: {spread:.4f}")
    
    results = {
        "model": model_info.name,
        "n_directions": len(all_dirs_data),
        "n_roles": len(unique_roles),
        "eff_rank": float(np.exp(-np.sum((var / np.sum(var))[var / np.sum(var) > 1e-15] * 
                                          np.log((var / np.sum(var))[var / np.sum(var) > 1e-15])))),
        "pc1_variance": float(var[0]),
        "k_90": int(k_90),
        "n_pc_used": int(n_pc),
        "loo_accuracy": float(loo_acc),
        "pc_role_stats": pc_role_stats,
        "role_spread": role_spread,
        "viz_data_sample": viz_data[:16],  # 只保存前16个用于可视化
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CCL-F: Linear Superposition + Extended Roles + Manifold Coordinates")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"\nModel: {model_info.name} ({model_info.model_class}), Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_linear_superposition(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_extended_syntax_roles(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_manifold_coordinates(model, tokenizer, device, model_info)
        
        if results:
            out_path = f"tests/glm5_temp/cclf_exp{args.exp}_{args.model}_results.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {out_path}")
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
