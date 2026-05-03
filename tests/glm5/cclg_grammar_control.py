"""
CCL-G(250.7): V_grammar基向量语义 + 多层演化 + 语法角色操控
================================================================
核心问题: V_grammar是12-20维线性子空间 → 能否用线性代数操控语法?

三合一实验:
  Exp1 (Phase 4M): ★★★★★ V_grammar基向量的语义发现
    → 沿每个基向量(流形PC)扰动, 观察生成文本的变化
    → 如果基向量有可解释含义 → 建立"语法坐标系"
    → 方法:
      a. 收集4角色因果方向, PCA得到V_grammar的基
      b. 沿每个PC方向扰动h, 生成top-10 token变化
      c. 分析: PC方向是否有可解释的语言学含义?

  Exp2 (Phase 4N): ★★★★ 多层V_grammar演化
    → 逐层收集因果方向, 分析V_grammar的层间变化
    → V_grammar的维度如何随层增长?
    → 是否存在"语法子空间临界层"?
    → 方法:
      a. 在1/4, 1/2, 3/4, last层收集因果方向
      b. 分析每层的有效秩、LDA分类能力
      c. 层间子空间重叠度(投影矩阵相似度)

  Exp3 (Phase 4O): ★★★★★ 语法角色操控实验
    → 构造"nsubj→dobj"转换矩阵
    → 测试: 矩阵乘法是否精确改变语法角色?
    → 如果成功 → 可以用线性代数操控语言 → AGI核心
    → 方法:
      a. 收集nsubj和dobj的因果方向质心
      b. 构造转换矩阵: M = d_dobj_centroid @ d_nsubj_centroid^+
      c. 在nsubj句子上应用M, 检查输出是否变为dobj模式
      d. 精确度: top-token变化率, KL散度, 角色分类改变率
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Fix Windows GBK encoding for console output
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

# 测试句子(用于4O操控实验)
CONTROL_TEST_DATA = {
    "nsubj_to_dobj": [
        ("The cat sat on the mat", "cat"),      # nsubj → 应转为dobj模式
        ("The dog ran through the park", "dog"),
        ("The bird sang a beautiful song", "bird"),
        ("The student read the textbook", "student"),
    ],
    "dobj_to_nsubj": [
        ("She chased the cat away", "cat"),     # dobj → 应转为nsubj模式
        ("He found the dog outside", "dog"),
        ("They watched the bird closely", "bird"),
        ("I praised the student loudly", "student"),
    ],
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


def get_layer_hidden(model, tokenizer, device, sentence, target_layer_idx):
    """获取指定层的hidden states"""
    layers = get_layers(model)
    target_layer = layers[target_layer_idx]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float()
        else:
            captured['h'] = output.detach().float()
    
    h_handle = target_layer.register_forward_hook(hook_fn)
    
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


def collect_causal_directions_at_layer(model, tokenizer, device, sentences, target_words, dep_type, layer_idx, n_max=None):
    """在指定层收集因果梯度方向"""
    layers = get_layers(model)
    target_layer = layers[layer_idx]
    
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
        
        # Autograd at target layer
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
        
        hook_handle = target_layer.register_forward_hook(capture_and_grad_hook)
        
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
                    'layer_idx': layer_idx,
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


def perturb_and_get_top_tokens(model, hidden_states, dep_idx, direction, alpha, h_norm, tokenizer, n_top=10):
    """沿方向扰动并返回top token变化"""
    perturbation = alpha * h_norm * direction
    pert_t = torch.tensor(perturbation, dtype=hidden_states.dtype, device=hidden_states.device)
    
    mod_h = hidden_states.clone()
    mod_h[0, dep_idx, :] += pert_t
    
    pert_logits = compute_logits_from_hidden(model, mod_h)
    pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
    
    top_indices = np.argsort(pert_probs)[::-1][:n_top]
    top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
    
    return top_tokens, pert_probs


# ================================================================
# Exp1 (Phase 4M): V_grammar基向量语义发现
# ================================================================

def exp1_basis_semantics(model, tokenizer, device, model_info):
    """
    ★★★★★ V_grammar基向量的语义发现
    沿每个流形PC方向扰动, 观察生成文本的变化
    """
    print("\n" + "="*70)
    print("Exp1 (Phase 4M): V_grammar基向量语义发现")
    print("="*70)
    
    # 1. 收集4种语法角色的因果方向
    print("\n[1] Collecting causal directions for 4 roles...")
    
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
    
    # 2. PCA on 因果方向 → V_grammar的基
    print("\n[2] PCA on causal directions → V_grammar basis...")
    
    pca = PCA()
    pca.fit(D)
    var = pca.explained_variance_ratio_
    
    cumvar = np.cumsum(var)
    k_90 = np.searchsorted(cumvar, 0.90) + 1
    k_95 = np.searchsorted(cumvar, 0.95) + 1
    
    var_probs = var / np.sum(var)
    var_probs = var_probs[var_probs > 1e-15]
    entropy = -np.sum(var_probs * np.log(var_probs))
    eff_rank = np.exp(entropy)
    
    print(f"  Eff rank: {eff_rank:.1f}")
    print(f"  k_90: {k_90}, k_95: {k_95}")
    print(f"  PC1-5 variance: {[f'{v*100:.1f}%' for v in var[:5]]}")
    
    # 3. ★★★ 沿每个PC方向扰动, 观察token变化
    print("\n[3] Perturbation along each PC direction...")
    
    n_pc_test = min(k_90, 8)  # 测试前8个PC
    
    # 用一个固定的句子做测试
    test_sent = "The cat sat on the mat"
    test_target = "cat"
    
    toks = tokenizer(test_sent, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
    test_idx = find_token_index(tokens, test_target)
    
    if test_idx is None:
        print("  ERROR: Cannot find test target")
        return None
    
    # 获取baseline hidden states和logits
    layers = get_layers(model)
    last_layer = layers[-1]
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().clone()
        else:
            captured['h'] = output.detach().clone()
    
    h_handle = last_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
        base_logits = output.logits.detach().float()
    
    h_handle.remove()
    
    if 'h' not in captured:
        print("  ERROR: Cannot capture hidden states")
        return None
    
    h_test = captured['h']
    h_norm_test = float(torch.norm(h_test[0, test_idx, :]).float())
    
    base_probs = torch.softmax(base_logits[0, test_idx].float(), dim=-1).cpu().numpy()
    base_top_indices = np.argsort(base_probs)[::-1][:10]
    base_top_tokens = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
    
    print(f"  Baseline top-5: {base_top_tokens[:5]}")
    print(f"  h_norm: {h_norm_test:.2f}")
    
    # 4. 沿每个PC方向扰动
    pc_semantic_results = {}
    alpha_test = 3.0  # 较大alpha使变化更明显
    
    for pc_idx in range(n_pc_test):
        pc_dir = pca.components_[pc_idx]  # [d_model]
        pc_var = var[pc_idx]
        
        # 正方向扰动
        top_tokens_pos, probs_pos = perturb_and_get_top_tokens(
            model, h_test, test_idx, pc_dir, alpha_test, h_norm_test, tokenizer, n_top=10
        )
        
        # 负方向扰动
        top_tokens_neg, probs_neg = perturb_and_get_top_tokens(
            model, h_test, test_idx, -pc_dir, alpha_test, h_norm_test, tokenizer, n_top=10
        )
        
        # KL散度
        kl_pos = float(np.sum(base_probs * np.log(base_probs / (probs_pos + 1e-10) + 1e-10)))
        kl_neg = float(np.sum(base_probs * np.log(base_probs / (probs_neg + 1e-10) + 1e-10)))
        
        # Token变化分析
        base_top_set = set([t for t, _ in base_top_tokens[:5]])
        pos_top_set = set([t for t, _ in top_tokens_pos[:5]])
        neg_top_set = set([t for t, _ in top_tokens_neg[:5]])
        
        pos_new = pos_top_set - base_top_set
        neg_new = neg_top_set - base_top_set
        
        # 与语法角色的余弦
        role_cos = {}
        for role in ["nsubj", "dobj", "amod", "advmod"]:
            role_dirs = [d['direction'] for d in all_dirs_data if d['dep_type'] == role]
            if role_dirs:
                centroid = np.mean(role_dirs, axis=0)
                centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)
                role_cos[role] = float(np.dot(pc_dir, centroid_norm))
        
        pc_result = {
            "variance_ratio": float(pc_var),
            "kl_pos": float(kl_pos),
            "kl_neg": float(kl_neg),
            "top5_baseline": base_top_tokens[:5],
            "top5_pos": top_tokens_pos[:5],
            "top5_neg": top_tokens_neg[:5],
            "new_tokens_pos": list(pos_new),
            "new_tokens_neg": list(neg_new),
            "role_cosine": role_cos,
        }
        pc_semantic_results[f"PC{pc_idx+1}"] = pc_result
        
        print(f"\n  PC{pc_idx+1} (var={pc_var*100:.1f}%): KL+={kl_pos:.2f}, KL-={kl_neg:.2f}")
        print(f"    +dir top-5: {top_tokens_pos[:5]}")
        print(f"    -dir top-5: {top_tokens_neg[:5]}")
        print(f"    Role cos: {', '.join(f'{r}={c:.3f}' for r, c in sorted(role_cos.items(), key=lambda x: abs(x[1]), reverse=True))}")
    
    # 5. ★★★ 随机方向基线
    print("\n[4] Random direction baseline...")
    d_model = pca.components_.shape[1]
    random_kls = []
    for _ in range(20):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        _, probs_rand = perturb_and_get_top_tokens(
            model, h_test, test_idx, rand_dir, alpha_test, h_norm_test, tokenizer, n_top=5
        )
        kl_rand = float(np.sum(base_probs * np.log(base_probs / (probs_rand + 1e-10) + 1e-10)))
        random_kls.append(kl_rand)
    
    mean_random_kl = np.mean(random_kls)
    print(f"  Random KL(α={alpha_test}): mean={mean_random_kl:.4f}")
    
    # 6. ★★★ PC方向的因果效应 vs 随机
    pc_kls = [pc_semantic_results[f"PC{i+1}"]["kl_pos"] + pc_semantic_results[f"PC{i+1}"]["kl_neg"] 
              for i in range(n_pc_test)]
    print(f"  PC KL sum: mean={np.mean(pc_kls):.4f}")
    print(f"  ★ PC / Random lift: {np.mean(pc_kls) / max(2 * mean_random_kl, 1e-10):.2f}x")
    
    # 7. ★★★ PC方向的可解释性评分
    print("\n[5] PC interpretability scoring...")
    
    pc_scores = {}
    for i in range(n_pc_test):
        key = f"PC{i+1}"
        r = pc_semantic_results[key]
        
        # 因果效应评分
        causal_score = (r["kl_pos"] + r["kl_neg"]) / max(2 * mean_random_kl, 1e-10)
        
        # 角色区分评分: max(|cos|) across roles
        max_role_cos = max(abs(c) for c in r["role_cosine"].values()) if r["role_cosine"] else 0
        
        # Token变化评分: 新token比例
        n_base = len(base_top_tokens[:5])
        new_frac = (len(r["new_tokens_pos"]) + len(r["new_tokens_neg"])) / max(2 * n_base, 1)
        
        pc_scores[key] = {
            "causal_lift": float(causal_score),
            "max_role_cosine": float(max_role_cos),
            "token_change_fraction": float(new_frac),
            "interpretability": float(causal_score * (1 + max_role_cos)),
        }
        
        print(f"  {key}: causal_lift={causal_score:.2f}, max_role_cos={max_role_cos:.3f}, "
              f"interp={pc_scores[key]['interpretability']:.2f}")
    
    results = {
        "model": model_info.name,
        "eff_rank": float(eff_rank),
        "k_90": int(k_90),
        "k_95": int(k_95),
        "n_pc_tested": int(n_pc_test),
        "alpha_test": alpha_test,
        "random_kl_mean": float(mean_random_kl),
        "pc_kl_mean": float(np.mean(pc_kls)),
        "pc_causal_vs_random_lift": float(np.mean(pc_kls) / max(2 * mean_random_kl, 1e-10)),
        "pc_semantics": {k: {kk: vv for kk, vv in v.items() if kk not in ['top5_baseline', 'top5_pos', 'top5_neg']} 
                        for k, v in pc_semantic_results.items()},
        "pc_scores": pc_scores,
    }
    
    return results


# ================================================================
# Exp2 (Phase 4N): 多层V_grammar演化
# ================================================================

def exp2_layer_evolution(model, tokenizer, device, model_info):
    """
    ★★★★ 多层V_grammar演化
    逐层收集因果方向, 分析V_grammar的层间变化
    """
    print("\n" + "="*70)
    print("Exp2 (Phase 4N): 多层V_grammar演化")
    print("="*70)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 1. 选择采样层 - 只用3层减少GPU内存
    sample_layers = sorted(set([
        n_layers // 4, 
        n_layers // 2, 
        n_layers - 1
    ]))
    print(f"  Sample layers: {sample_layers}")
    
    # 2. 用前向hook收集各层hidden states, 做差分分析 (避免中间层反向传播OOM)
    print("\n[1] Collecting hidden states at each layer via forward hooks...")
    
    layer_data = {}
    n_sentences_per_role = 4
    
    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")
        torch.cuda.empty_cache()
        
        # 收集nsubj和dobj的hidden states
        nsubj_hiddens = []
        dobj_hiddens = []
        
        for sent, target in zip(SYNTAX_DATA["nsubj"]["sentences"][:n_sentences_per_role],
                                SYNTAX_DATA["nsubj"]["target_words"][:n_sentences_per_role]):
            h, _, toks = get_layer_hidden(model, tokenizer, device, sent, li)
            if h is not None:
                input_ids = toks.input_ids
                tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens, target)
                if dep_idx is not None:
                    nsubj_hiddens.append(h[0, dep_idx, :].float().cpu().numpy())
        
        for sent, target in zip(SYNTAX_DATA["dobj"]["sentences"][:n_sentences_per_role],
                                SYNTAX_DATA["dobj"]["target_words"][:n_sentences_per_role]):
            h, _, toks = get_layer_hidden(model, tokenizer, device, sent, li)
            if h is not None:
                input_ids = toks.input_ids
                tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens, target)
                if dep_idx is not None:
                    dobj_hiddens.append(h[0, dep_idx, :].float().cpu().numpy())
        
        print(f"    nsubj: {len(nsubj_hiddens)} hiddens, dobj: {len(dobj_hiddens)} hiddens")
        
        if len(nsubj_hiddens) < 2 or len(dobj_hiddens) < 2:
            print(f"    Too few hiddens, skipping layer {li}")
            continue
        
        # 差分分析: nsubj_hidden - dobj_hidden的方向
        all_hiddens = np.array(nsubj_hiddens + dobj_hiddens)
        y = np.array(["nsubj"] * len(nsubj_hiddens) + ["dobj"] * len(dobj_hiddens))
        
        # PCA
        pca = PCA()
        pca.fit(all_hiddens)
        var = pca.explained_variance_ratio_
        var_probs = var / np.sum(var)
        var_probs = var_probs[var_probs > 1e-15]
        entropy = -np.sum(var_probs * np.log(var_probs))
        eff_rank = np.exp(entropy)
        
        cumvar = np.cumsum(var)
        k_90 = np.searchsorted(cumvar, 0.90) + 1
        k_95 = np.searchsorted(cumvar, 0.95) + 1
        
        # LDA分类
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(all_hiddens, y)
            lda_acc = lda.score(all_hiddens, y)
        except:
            lda_acc = 0.5
        
        # nsubj vs dobj质心余弦
        nsubj_centroid = np.mean(nsubj_hiddens, axis=0)
        dobj_centroid = np.mean(dobj_hiddens, axis=0)
        nsubj_norm = nsubj_centroid / max(np.linalg.norm(nsubj_centroid), 1e-10)
        dobj_norm = dobj_centroid / max(np.linalg.norm(dobj_centroid), 1e-10)
        role_cos = float(np.dot(nsubj_norm, dobj_norm))
        
        # 质心距离
        centroid_dist = float(np.linalg.norm(nsubj_centroid - dobj_centroid))
        
        # 簇内一致性
        nsubj_hiddens_arr = np.array(nsubj_hiddens)
        dobj_hiddens_arr = np.array(dobj_hiddens)
        nsubj_intra = float(np.mean([
            np.dot(nsubj_hiddens_arr[i], nsubj_centroid / max(np.linalg.norm(nsubj_centroid), 1e-10))
            for i in range(len(nsubj_hiddens_arr))
        ]))
        dobj_intra = float(np.mean([
            np.dot(dobj_hiddens_arr[i], dobj_centroid / max(np.linalg.norm(dobj_centroid), 1e-10))
            for i in range(len(dobj_hiddens_arr))
        ]))
        
        # Fisher discriminant ratio
        nsubj_var = np.mean(np.sum((nsubj_hiddens_arr - nsubj_centroid)**2, axis=1))
        dobj_var = np.mean(np.sum((dobj_hiddens_arr - dobj_centroid)**2, axis=1))
        fisher_ratio = centroid_dist**2 / max((nsubj_var + dobj_var) / 2, 1e-10)
        
        layer_data[li] = {
            "eff_rank": float(eff_rank),
            "pc1_variance": float(var[0]),
            "k_90": int(k_90),
            "k_95": int(k_95),
            "lda_accuracy": float(lda_acc),
            "nsubj_dobj_cos": float(role_cos),
            "centroid_distance": centroid_dist,
            "nsubj_intra_consistency": float(nsubj_intra),
            "dobj_intra_consistency": float(dobj_intra),
            "fisher_ratio": float(fisher_ratio),
            "n_nsubj": len(nsubj_hiddens),
            "n_dobj": len(dobj_hiddens),
        }
        
        print(f"    Eff rank: {eff_rank:.1f}, k_90: {k_90}, k_95: {k_95}")
        print(f"    LDA: {lda_acc:.3f}, nsubj-dobj cos: {role_cos:.3f}")
        print(f"    Centroid dist: {centroid_dist:.2f}, Fisher ratio: {fisher_ratio:.2f}")
        print(f"    nsubj intra: {nsubj_intra:.3f}, dobj intra: {dobj_intra:.3f}")
        
        torch.cuda.empty_cache()
    
    # 3. ★★★ 演化趋势分析
    print("\n[2] Evolution trends...")
    
    layer_indices = sorted(layer_data.keys())
    
    eff_ranks = [layer_data[li]["eff_rank"] for li in layer_indices]
    lda_accs = [layer_data[li]["lda_accuracy"] for li in layer_indices]
    nsubj_dobj_coss = [layer_data[li]["nsubj_dobj_cos"] for li in layer_indices]
    nsubj_intras = [layer_data[li]["nsubj_intra_consistency"] for li in layer_indices]
    dobj_intras = [layer_data[li]["dobj_intra_consistency"] for li in layer_indices]
    fisher_ratios = [layer_data[li]["fisher_ratio"] for li in layer_indices]
    centroid_dists = [layer_data[li]["centroid_distance"] for li in layer_indices]
    
    print(f"  Eff rank evolution: {eff_ranks}")
    print(f"  LDA accuracy evolution: {[f'{a:.3f}' for a in lda_accs]}")
    print(f"  nsubj-dobj cos evolution: {[f'{c:.3f}' for c in nsubj_dobj_coss]}")
    print(f"  Fisher ratio evolution: {[f'{f:.2f}' for f in fisher_ratios]}")
    print(f"  Centroid dist evolution: {[f'{d:.2f}' for d in centroid_dists]}")
    print(f"  nsubj intra evolution: {[f'{c:.3f}' for c in nsubj_intras]}")
    print(f"  dobj intra evolution: {[f'{c:.3f}' for c in dobj_intras]}")
    
    # 判断是否存在临界层
    critical_info = {}
    if len(eff_ranks) > 1:
        rank_diffs = [eff_ranks[i+1] - eff_ranks[i] for i in range(len(eff_ranks)-1)]
        max_diff_idx = np.argmax(np.abs(rank_diffs))
        critical_info = {
            "max_rank_change_layer": f"L{layer_indices[max_diff_idx]}-L{layer_indices[max_diff_idx+1]}",
            "rank_change": float(rank_diffs[max_diff_idx]),
        }
        print(f"  ★ Max rank change between L{layer_indices[max_diff_idx]}-L{layer_indices[max_diff_idx+1]}: {rank_diffs[max_diff_idx]:.1f}")
    
    results = {
        "model": model_info.name,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "layer_data": {str(k): v for k, v in layer_data.items()},
        "eff_rank_evolution": {str(li): layer_data[li]["eff_rank"] for li in layer_indices},
        "lda_evolution": {str(li): layer_data[li]["lda_accuracy"] for li in layer_indices},
        "fisher_evolution": {str(li): layer_data[li]["fisher_ratio"] for li in layer_indices},
        "critical_info": critical_info,
    }
    
    return results


# ================================================================
# Exp3 (Phase 4O): ★★★★★ 语法角色操控实验
# ================================================================

def exp3_syntax_control(model, tokenizer, device, model_info):
    """
    ★★★★★ 语法角色操控实验
    构造语法角色转换矩阵, 测试矩阵乘法是否精确改变语法角色
    """
    print("\n" + "="*70)
    print("Exp3 (Phase 4O): ★★★★★ 语法角色操控实验")
    print("="*70)
    
    # 1. 收集nsubj和dobj的因果方向
    print("\n[1] Collecting nsubj and dobj causal directions...")
    
    nsubj_dirs = collect_causal_directions(
        model, tokenizer, device,
        SYNTAX_DATA["nsubj"]["sentences"][:8],
        SYNTAX_DATA["nsubj"]["target_words"][:8],
        "nsubj",
    )
    dobj_dirs = collect_causal_directions(
        model, tokenizer, device,
        SYNTAX_DATA["dobj"]["sentences"][:8],
        SYNTAX_DATA["dobj"]["target_words"][:8],
        "dobj",
    )
    
    print(f"  nsubj: {len(nsubj_dirs)} directions")
    print(f"  dobj: {len(dobj_dirs)} directions")
    
    if len(nsubj_dirs) < 3 or len(dobj_dirs) < 3:
        print("  ERROR: Too few directions")
        return None
    
    # 2. 构造转换向量 (简单版: 质心差异方向)
    print("\n[2] Constructing transformation vectors...")
    
    nsubj_centroid = np.mean([d['direction'] for d in nsubj_dirs], axis=0)
    dobj_centroid = np.mean([d['direction'] for d in dobj_dirs], axis=0)
    
    # 转换方向: nsubj → dobj
    d_nsubj_to_dobj = dobj_centroid - nsubj_centroid
    d_nsubj_to_dobj_norm = np.linalg.norm(d_nsubj_to_dobj)
    if d_nsubj_to_dobj_norm > 1e-10:
        d_nsubj_to_dobj = d_nsubj_to_dobj / d_nsubj_to_dobj_norm
    
    # 转换方向: dobj → nsubj
    d_dobj_to_nsubj = nsubj_centroid - dobj_centroid
    d_dobj_to_nsubj_norm = np.linalg.norm(d_dobj_to_nsubj)
    if d_dobj_to_nsubj_norm > 1e-10:
        d_dobj_to_nsubj = d_dobj_to_nsubj / d_dobj_to_nsubj_norm
    
    print(f"  nsubj centroid norm: {np.linalg.norm(nsubj_centroid):.4f}")
    print(f"  dobj centroid norm: {np.linalg.norm(dobj_centroid):.4f}")
    print(f"  Centroid cos(nsubj, dobj): {np.dot(nsubj_centroid/np.linalg.norm(nsubj_centroid), dobj_centroid/np.linalg.norm(dobj_centroid)):.4f}")
    
    # 3. ★★★ 高维转换: 基于PCA子空间的线性变换
    print("\n[3] Constructing PCA-based linear transformation...")
    
    all_dirs = nsubj_dirs + dobj_dirs
    D = np.array([d['direction'] for d in all_dirs])
    
    pca = PCA()
    pca.fit(D)
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    k_95 = np.searchsorted(cumvar, 0.95) + 1
    
    print(f"  k_95: {k_95}")
    
    # 在PCA子空间中构造转换
    U_pca = pca.components_[:k_95].T  # [d_model, k_95]
    
    # nsubj和dobj质心在PCA子空间中的坐标
    nsubj_coord = U_pca.T @ nsubj_centroid  # [k_95]
    dobj_coord = U_pca.T @ dobj_centroid     # [k_95]
    
    # 转换向量(在PCA坐标中)
    delta_coord = dobj_coord - nsubj_coord  # [k_95]
    
    # 投影回原始空间
    d_nsubj_to_dobj_pca = U_pca @ delta_coord
    d_nsubj_to_dobj_pca_norm = np.linalg.norm(d_nsubj_to_dobj_pca)
    if d_nsubj_to_dobj_pca_norm > 1e-10:
        d_nsubj_to_dobj_pca = d_nsubj_to_dobj_pca / d_nsubj_to_dobj_pca_norm
    
    print(f"  PCA-based direction cos with centroid-based: {np.dot(d_nsubj_to_dobj_pca, d_nsubj_to_dobj):.4f}")
    
    # 4. ★★★★★ 测试1: 在nsubj句子上沿转换方向扰动
    print("\n[4] ★★★ Test 1: Perturbing nsubj sentences toward dobj direction...")
    
    h_norms = [np.linalg.norm(d['hidden_states'][0, d['dep_idx'], :].float().cpu().numpy()) for d in all_dirs]
    h_norm_mean = float(np.mean(h_norms))
    
    test_alphas = [1.0, 2.0, 3.0, 5.0]
    
    nsubj_control_results = []
    
    for d in nsubj_dirs:
        sent = d['sentence']
        h = d['hidden_states']
        dep_idx = d['dep_idx']
        base_probs = d['base_probs']
        
        sent_result = {
            "sentence": sent,
            "target": d['target'],
            "base_top5": [],  # 填充below
        }
        
        # Baseline top-5
        base_top_indices = np.argsort(base_probs)[::-1][:5]
        base_top5 = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
        sent_result["base_top5"] = base_top5
        
        for alpha in test_alphas:
            # 质心方向扰动
            kl_centroid = perturb_and_measure_kl(
                model, h, dep_idx, base_probs, d_nsubj_to_dobj, alpha, h_norm_mean
            )
            
            # PCA方向扰动
            kl_pca = perturb_and_measure_kl(
                model, h, dep_idx, base_probs, d_nsubj_to_dobj_pca, alpha, h_norm_mean
            )
            
            # 获取扰动后的top tokens
            top_tokens_perturbed, _ = perturb_and_get_top_tokens(
                model, h, dep_idx, d_nsubj_to_dobj, alpha, h_norm_mean, tokenizer, n_top=10
            )
            
            sent_result[f"alpha_{alpha}"] = {
                "kl_centroid": float(kl_centroid),
                "kl_pca": float(kl_pca),
                "top5_perturbed": top_tokens_perturbed[:5],
                "token_changed": base_top5[0][0] != top_tokens_perturbed[0][0],
            }
        
        nsubj_control_results.append(sent_result)
        
        # 只显示前2个句子的详细信息
        if len(nsubj_control_results) <= 2:
            print(f"\n  Sentence: {sent}")
            print(f"  Base top-3: {base_top5[:3]}")
            for alpha in test_alphas:
                r = sent_result[f"alpha_{alpha}"]
                print(f"  α={alpha}: KL_c={r['kl_centroid']:.2f}, KL_pca={r['kl_pca']:.2f}, "
                      f"top1_changed={r['token_changed']}, new_top3={r['top5_perturbed'][:3]}")
    
    # 5. ★★★★★ 测试2: 在dobj句子上沿nsubj方向扰动
    print("\n[5] ★★★ Test 2: Perturbing dobj sentences toward nsubj direction...")
    
    dobj_control_results = []
    
    for d in dobj_dirs:
        sent = d['sentence']
        h = d['hidden_states']
        dep_idx = d['dep_idx']
        base_probs = d['base_probs']
        
        sent_result = {
            "sentence": sent,
            "target": d['target'],
            "base_top5": [],
        }
        
        base_top_indices = np.argsort(base_probs)[::-1][:5]
        base_top5 = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
        sent_result["base_top5"] = base_top5
        
        for alpha in test_alphas:
            kl_centroid = perturb_and_measure_kl(
                model, h, dep_idx, base_probs, d_dobj_to_nsubj, alpha, h_norm_mean
            )
            
            kl_pca = perturb_and_measure_kl(
                model, h, dep_idx, base_probs, -d_nsubj_to_dobj_pca, alpha, h_norm_mean
            )
            
            top_tokens_perturbed, _ = perturb_and_get_top_tokens(
                model, h, dep_idx, d_dobj_to_nsubj, alpha, h_norm_mean, tokenizer, n_top=10
            )
            
            sent_result[f"alpha_{alpha}"] = {
                "kl_centroid": float(kl_centroid),
                "kl_pca": float(kl_pca),
                "top5_perturbed": top_tokens_perturbed[:5],
                "token_changed": base_top5[0][0] != top_tokens_perturbed[0][0],
            }
        
        dobj_control_results.append(sent_result)
        
        if len(dobj_control_results) <= 2:
            print(f"\n  Sentence: {sent}")
            print(f"  Base top-3: {base_top5[:3]}")
            for alpha in test_alphas:
                r = sent_result[f"alpha_{alpha}"]
                print(f"  α={alpha}: KL_c={r['kl_centroid']:.2f}, KL_pca={r['kl_pca']:.2f}, "
                      f"top1_changed={r['token_changed']}, new_top3={r['top5_perturbed'][:3]}")
    
    # 6. ★★★ 汇总统计
    print("\n[6] Summary statistics...")
    
    # nsubj → dobj
    nsubj_kls_by_alpha = {}
    nsubj_change_rate = {}
    for alpha in test_alphas:
        kls = [r[f"alpha_{alpha}"]["kl_centroid"] for r in nsubj_control_results if f"alpha_{alpha}" in r]
        changes = [r[f"alpha_{alpha}"]["token_changed"] for r in nsubj_control_results if f"alpha_{alpha}" in r]
        nsubj_kls_by_alpha[alpha] = kls
        nsubj_change_rate[alpha] = np.mean(changes) if changes else 0.0
    
    # dobj → nsubj
    dobj_kls_by_alpha = {}
    dobj_change_rate = {}
    for alpha in test_alphas:
        kls = [r[f"alpha_{alpha}"]["kl_centroid"] for r in dobj_control_results if f"alpha_{alpha}" in r]
        changes = [r[f"alpha_{alpha}"]["token_changed"] for r in dobj_control_results if f"alpha_{alpha}" in r]
        dobj_kls_by_alpha[alpha] = kls
        dobj_change_rate[alpha] = np.mean(changes) if changes else 0.0
    
    print("\n  nsubj → dobj control:")
    for alpha in test_alphas:
        mean_kl = np.mean(nsubj_kls_by_alpha[alpha]) if nsubj_kls_by_alpha[alpha] else 0
        print(f"    α={alpha}: mean KL={mean_kl:.2f}, top1 change rate={nsubj_change_rate[alpha]:.2f}")
    
    print("\n  dobj → nsubj control:")
    for alpha in test_alphas:
        mean_kl = np.mean(dobj_kls_by_alpha[alpha]) if dobj_kls_by_alpha[alpha] else 0
        print(f"    α={alpha}: mean KL={mean_kl:.2f}, top1 change rate={dobj_change_rate[alpha]:.2f}")
    
    # 7. ★★★ 与随机方向对比
    print("\n[7] Random direction baseline for control...")
    
    d_model = D.shape[1]
    random_control_kls = []
    
    for _ in range(20):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        trial_kls = []
        for d in nsubj_dirs[:4]:
            kl = perturb_and_measure_kl(
                model, d['hidden_states'], d['dep_idx'],
                d['base_probs'], rand_dir, 3.0, h_norm_mean
            )
            trial_kls.append(kl)
        random_control_kls.append(np.mean(trial_kls))
    
    mean_random_kl = np.mean(random_control_kls)
    
    nsubj_kl_alpha3 = np.mean(nsubj_kls_by_alpha[3.0]) if nsubj_kls_by_alpha.get(3.0) else 0
    print(f"  Random KL(α=3): {mean_random_kl:.4f}")
    print(f"  nsubj→dobj KL(α=3): {nsubj_kl_alpha3:.4f}")
    print(f"  ★ Control / Random lift: {nsubj_kl_alpha3 / max(mean_random_kl, 1e-10):.2f}x")
    
    results = {
        "model": model_info.name,
        "n_nsubj_dirs": len(nsubj_dirs),
        "n_dobj_dirs": len(dobj_dirs),
        "centroid_cos": float(np.dot(
            nsubj_centroid / np.linalg.norm(nsubj_centroid), 
            dobj_centroid / np.linalg.norm(dobj_centroid)
        )),
        "pca_k95": int(k_95),
        "pca_centroid_cos": float(np.dot(d_nsubj_to_dobj_pca, d_nsubj_to_dobj)),
        "nsubj_control_summary": {
            str(alpha): {
                "mean_kl": float(np.mean(nsubj_kls_by_alpha[alpha])) if nsubj_kls_by_alpha.get(alpha) else 0,
                "change_rate": float(nsubj_change_rate.get(alpha, 0)),
            }
            for alpha in test_alphas
        },
        "dobj_control_summary": {
            str(alpha): {
                "mean_kl": float(np.mean(dobj_kls_by_alpha[alpha])) if dobj_kls_by_alpha.get(alpha) else 0,
                "change_rate": float(dobj_change_rate.get(alpha, 0)),
            }
            for alpha in test_alphas
        },
        "random_kl_alpha3": float(mean_random_kl),
        "control_vs_random_lift": float(nsubj_kl_alpha3 / max(mean_random_kl, 1e-10)),
        "nsubj_control_sample": nsubj_control_results[:3],
        "dobj_control_sample": dobj_control_results[:3],
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CCL-G: V_grammar Basis Semantics + Layer Evolution + Syntax Control")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"\nModel: {model_info.name} ({model_info.model_class}), Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_basis_semantics(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_layer_evolution(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_syntax_control(model, tokenizer, device, model_info)
        
        if results:
            out_path = f"tests/glm5_temp/cclg_exp{args.exp}_{args.model}_results.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {out_path}")
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
