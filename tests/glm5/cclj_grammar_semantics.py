"""
CCL-J(251): 语法角色操控的语义化与跨层验证
==================================================
基于CCL-I的发现: 探针方向可以80-93%翻转语法角色分类

Phase 7三合一实验:
  Exp1 (7A): ★★★★★ 探针方向操控的精细化
    → 测试更小alpha(0.02-0.5)的探针方向操控
    → 找到alpha_min: 最小使探针翻转的alpha
    → 分析翻转后概率分布的精细变化
    → 测量"语义保持度": 翻转后top token是否仍然是实词

  Exp2 (7B): ★★★★ 探针方向的几何理解
    → 4个角色方向构成什么几何结构?
    → 正四面体检验: 4方向两两余弦是否接近-1/3?
    → 分类边界分析: 6对角色之间的边界法线
    → 对称性检验: 是否存在旋转对称性?

  Exp3 (7C): ★★★ 跨层语法角色操控
    → 在1/4, 1/2, 3/4, 最后一层训练探针
    → 跨层探针准确率变化曲线
    → 中间层操控 + 后续层自动传播的效果
    → 哪一层操控效果最好?
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 数据集(复用CCL-I) =====
EXTENDED_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom",
            "The doctor treated the patient",
            "The artist painted the portrait",
            "The soldier defended the castle",
            "The cat sat on the mat",
            "The dog ran through the park",
            "The bird sang a beautiful song",
            "The child played with the toys",
            "The student read the textbook",
            "The teacher explained the lesson",
            "The woman drove the car",
            "The man fixed the roof",
            "The girl sang a song",
            "The boy kicked the ball",
            "The president signed the bill",
            "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "woman", "man",
            "girl", "boy", "president", "chef",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "She chased the cat away",
            "He found the dog outside",
            "They watched the bird closely",
            "We helped the child today",
            "I praised the student loudly",
            "You thanked the teacher warmly",
            "The police arrested the man quickly",
            "The company hired the woman recently",
            "The coach trained the girl daily",
            "The teacher praised the boy warmly",
            "The nation elected the president fairly",
            "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "man", "woman",
            "girl", "boy", "president", "chef",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The beautiful cat sat quietly",
            "The large dog ran swiftly",
            "The small bird sang softly",
            "The young child played happily",
            "The bright student read carefully",
            "The wise teacher explained clearly",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The little girl smiled sweetly",
            "The smart boy answered quickly",
            "The powerful president decided firmly",
            "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong",
            "beautiful", "large", "small", "young",
            "bright", "wise", "old", "tall",
            "little", "smart", "powerful", "skilled",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever",
            "The doctor worked carefully always",
            "The artist painted beautifully daily",
            "The soldier fought bravely there",
            "The cat ran quickly home",
            "The dog barked loudly today",
            "The bird sang softly outside",
            "The child played happily inside",
            "The student read carefully alone",
            "The teacher spoke clearly again",
            "The woman drove slowly home",
            "The man spoke quietly now",
            "The girl laughed happily then",
            "The boy ran fast away",
            "The president spoke firmly today",
            "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely",
            "quickly", "loudly", "softly", "happily",
            "carefully", "clearly", "slowly", "quietly",
            "happily", "fast", "firmly", "quickly",
        ],
    },
}

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

# 功能词列表(用于判断top token是否是实词)
FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'me',
    'my', 'your', 'his', 'her', 'their', 'our', 'what', 'which', 'who',
    'whom', 'whose', ',', '.', '!', '?', ';', ':', '\'', '"', '-',
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
    'brings', 'in', 'for', 'on', 'away', 'outside', 'today',
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def is_content_word(token_str):
    """判断token是否是实词(非功能词)"""
    t = token_str.strip().lower().lstrip('▁').lstrip(' ')
    if not t:
        return False
    if t in FUNCTION_WORDS:
        return False
    if len(t) <= 1:
        return False
    return True


def is_coherent_token(token_str):
    """简单判断token是否是连贯的英语"""
    t = token_str.strip()
    if not t:
        return False
    if any(c in t for c in ['{', '}', '$', ')', '(', '\n', '\t', ';', '#']):
        return len(t) > 3
    if t.isdigit():
        return False
    try:
        t.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True


def get_layer_hidden(model, tokenizer, device, sentence, layer_idx):
    """获取指定层的hidden states"""
    layers = get_layers(model)
    target_layer = layers[layer_idx]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().clone()
        else:
            captured['h'] = output.detach().float().clone()
    
    h_handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
    
    h_handle.remove()
    
    if 'h' not in captured:
        return None, toks
    
    return captured['h'], toks


def get_last_layer_hidden(model, tokenizer, device, sentence):
    """获取最后层的hidden states + logits"""
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


def compute_logits_from_intermediate_hidden(model, hidden_states, start_layer):
    """
    从中间层的hidden states计算logits
    将hidden_states作为start_layer的输出, 传播通过后续层
    """
    layers = get_layers(model)
    
    # 需要通过后续层传播
    h = hidden_states
    for li in range(start_layer + 1, len(layers)):
        layer = layers[li]
        # 需要手动调用layer forward
        # 但这需要attention mask等参数, 比较复杂
        # 简化方案: 只测最后一层
        pass
    
    # 对于最后一层, 直接用final norm + lm_head
    return compute_logits_from_hidden(model, h)


def train_probe_at_layer(model, tokenizer, device, model_info, layer_idx, data_dict, role_names):
    """在指定层训练语法角色探针"""
    
    all_hidden = []
    all_labels = []
    
    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target in zip(data["sentences"], data["target_words"]):
            h, toks = get_layer_hidden(model, tokenizer, device, sent, layer_idx)
            if h is None:
                continue
            
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            h_vec = h[0, dep_idx, :].float().cpu().numpy()
            all_hidden.append(h_vec)
            all_labels.append(role_idx)
    
    if len(set(all_labels)) < 2:
        return None
    
    X = np.array(all_hidden)
    y = np.array(all_labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    
    # 交叉验证
    n_classes = len(set(y))
    cv_folds = min(5, min(np.bincount(y)))
    if cv_folds >= 2:
        cv_scores = cross_val_score(probe, X_scaled, y, cv=cv_folds, scoring='accuracy')
        cv_acc = float(cv_scores.mean())
    else:
        cv_acc = -1.0
    
    probe.fit(X_scaled, y)
    train_acc = float(probe.score(X_scaled, y))
    
    # 提取探针方向(转换回原始空间)
    probe_weights = probe.coef_
    probe_intercept = probe.intercept_
    
    scale_factors = scaler.scale_
    mean_factors = scaler.mean_
    
    probe_weights_orig = probe_weights / scale_factors[np.newaxis, :]
    
    probe_directions = {}
    for i, role in enumerate(role_names):
        w = probe_weights_orig[i]
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            probe_directions[role] = w / w_norm
    
    return {
        'probe': probe,
        'scaler': scaler,
        'probe_directions': probe_directions,
        'probe_weights_orig': probe_weights_orig,
        'train_acc': train_acc,
        'cv_acc': cv_acc,
        'n_samples': len(X),
    }


# ================================================================
# Exp1 (7A): 探针方向操控的精细化
# ================================================================

def exp1_fine_grained_manipulation(model, tokenizer, device, model_info):
    """
    ★★★★★ 探针方向操控的精细化
    
    核心思路:
    1. 加载Exp1训练的探针
    2. 用更精细的alpha(0.02-0.5)测试操控
    3. 测量:
       - alpha_min: 最小使探针翻转的alpha
       - 语义保持度: 翻转后top token是否是实词
       - KL曲线: KL随alpha的变化
       - 探针概率曲线: 各角色概率随alpha的变化
    """
    print("\n" + "="*70)
    print("Exp1 (7A): 探针方向操控的精细化")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    
    # 1. 加载或训练探针
    print("\n[1] Loading probe...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    probe_path = os.path.join(results_dir, f"ccli_probe_{model_info.name}.npz")
    dir_path = os.path.join(results_dir, f"ccli_directions_{model_info.name}.npz")
    
    if not os.path.exists(probe_path):
        print("  Probe not found, training from scratch...")
        probe_result = train_probe_at_layer(
            model, tokenizer, device, model_info, 
            model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
        )
        if probe_result is None:
            print("  ERROR: Failed to train probe")
            return None
        probe = probe_result['probe']
        scaler = probe_result['scaler']
        probe_directions = probe_result['probe_directions']
    else:
        probe_data = np.load(probe_path)
        dir_data = np.load(dir_path)
        
        X_train = probe_data['X_train']
        y_train = probe_data['y_train']
        scaler = StandardScaler()
        scaler.mean_ = probe_data['scaler_mean']
        scaler.scale_ = probe_data['scaler_scale']
        
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        X_scaled = scaler.transform(X_train)
        probe.fit(X_scaled, y_train)
        
        probe_directions = {}
        for role in ROLE_NAMES:
            key_p = f"probe_{role}"
            if key_p in dir_data:
                probe_directions[role] = dir_data[key_p]
        
        print(f"  Loaded probe, accuracy: {probe.score(X_scaled, y_train):.4f}")
    
    print(f"  Probe directions: {list(probe_directions.keys())}")
    
    # 2. 精细alpha操控测试
    print("\n[2] Fine-grained alpha manipulation...")
    
    manipulation_tests = [
        ("The cat sat on the mat", "cat", "nsubj", "dobj"),
        ("The dog ran through the park", "dog", "nsubj", "dobj"),
        ("She chased the cat away", "cat", "dobj", "nsubj"),
        ("He found the dog outside", "dog", "dobj", "nsubj"),
        ("The beautiful cat sat quietly", "beautiful", "amod", "advmod"),
        ("The large dog ran swiftly", "large", "amod", "advmod"),
        ("The cat ran quickly home", "quickly", "advmod", "amod"),
        ("The dog barked loudly today", "loudly", "advmod", "amod"),
        # 更多测试对
        ("The king ruled the kingdom", "king", "nsubj", "dobj"),
        ("She visited the doctor recently", "doctor", "dobj", "nsubj"),
        ("The brave king fought hard", "brave", "amod", "nsubj"),
        ("The king ruled wisely forever", "wisely", "advmod", "amod"),
    ]
    
    fine_alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
    
    all_case_results = []
    alpha_flip_stats = {a: {"flip": 0, "total": 0} for a in fine_alphas}
    alpha_content_word_stats = {a: {"content": 0, "total": 0} for a in fine_alphas}
    
    for sent, target, src_role, tgt_role in manipulation_tests:
        print(f"\n  [{src_role}→{tgt_role}] {sent} / '{target}'")
        
        h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
        if h is None:
            continue
        
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        h_vec = h[0, dep_idx, :].float().cpu().numpy()
        h_norm = float(np.linalg.norm(h_vec))
        
        base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
        base_top_idx = np.argmax(base_probs)
        base_top_token = safe_decode(tokenizer, int(base_top_idx))
        base_top_prob = float(base_probs[base_top_idx])
        
        # 原始探针预测
        h_scaled = scaler.transform(h_vec.reshape(1, -1))
        orig_pred = int(probe.predict(h_scaled)[0])
        orig_prob = probe.predict_proba(h_scaled)[0]
        
        # 构造操控方向
        if tgt_role not in probe_directions or src_role not in probe_directions:
            continue
        
        probe_dir = probe_directions[tgt_role] - probe_directions[src_role]
        probe_dir_norm = np.linalg.norm(probe_dir)
        if probe_dir_norm < 1e-10:
            continue
        probe_dir = probe_dir / probe_dir_norm
        
        case_data = {
            "sentence": sent, "target": target,
            "src_role": src_role, "tgt_role": tgt_role,
            "orig_probe_pred": ROLE_NAMES[orig_pred],
            "base_top_token": base_top_token,
            "base_top_prob": base_top_prob,
            "alpha_results": [],
        }
        
        alpha_min_flip = None  # 最小翻转alpha
        
        for alpha in fine_alphas:
            perturbation = alpha * (h_norm / max(np.linalg.norm(probe_dir), 1e-10)) * probe_dir
            pert_h = h_vec + perturbation
            
            # 探针预测
            pert_h_scaled = scaler.transform(pert_h.reshape(1, -1))
            pert_pred = int(probe.predict(pert_h_scaled)[0])
            pert_pred_role = ROLE_NAMES[pert_pred]
            pert_prob = probe.predict_proba(pert_h_scaled)[0]
            
            probe_flipped = (pert_pred_role == tgt_role)
            
            # Logits变化
            pert_h_t = torch.tensor(pert_h, dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
            pert_h_full = h.clone()
            pert_h_full[0, dep_idx, :] = pert_h_t[0, 0, :]
            
            pert_logits = compute_logits_from_hidden(model, pert_h_full)
            pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
            
            kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
            
            top_idx = np.argmax(pert_probs)
            top_token = safe_decode(tokenizer, int(top_idx))
            top_prob = float(pert_probs[top_idx])
            
            content_word = is_content_word(top_token)
            coherent = is_coherent_token(top_token)
            
            alpha_result = {
                "alpha": alpha,
                "kl": kl,
                "probe_flipped": probe_flipped,
                "probe_pred": pert_pred_role,
                "tgt_prob": float(pert_prob[ROLE_NAMES.index(tgt_role)]),
                "src_prob": float(pert_prob[ROLE_NAMES.index(src_role)]),
                "top_token": top_token,
                "top_prob": top_prob,
                "content_word": content_word,
                "coherent": coherent,
            }
            case_data["alpha_results"].append(alpha_result)
            
            # 统计
            alpha_flip_stats[alpha]["total"] += 1
            if probe_flipped:
                alpha_flip_stats[alpha]["flip"] += 1
                if alpha_min_flip is None:
                    alpha_min_flip = alpha
            
            if probe_flipped:
                alpha_content_word_stats[alpha]["total"] += 1
                if content_word:
                    alpha_content_word_stats[alpha]["content"] += 1
            
            # 打印关键alpha
            if alpha in [0.05, 0.1, 0.2, 0.5]:
                print(f"    α={alpha}: probe→{pert_pred_role} flip={probe_flipped} "
                      f"KL={kl:.3f} top={top_token}({top_prob:.3f}) "
                      f"content={content_word} "
                      f"prob[{tgt_role}]={pert_prob[ROLE_NAMES.index(tgt_role)]:.3f}")
        
        case_data["alpha_min_flip"] = alpha_min_flip
        all_case_results.append(case_data)
    
    # 3. 汇总: alpha vs flip rate, alpha vs content word rate
    print("\n" + "="*50)
    print("Fine-grained Manipulation Summary")
    print("="*50)
    
    print("\n  Alpha vs Probe Flip Rate:")
    for alpha in fine_alphas:
        s = alpha_flip_stats[alpha]
        rate = s["flip"] / max(s["total"], 1)
        print(f"    α={alpha:.2f}: flip_rate={rate:.1%} ({s['flip']}/{s['total']})")
    
    print("\n  Alpha vs Content Word Rate (among flipped cases):")
    for alpha in fine_alphas:
        s = alpha_content_word_stats[alpha]
        rate = s["content"] / max(s["total"], 1)
        print(f"    α={alpha:.2f}: content_rate={rate:.1%} ({s['content']}/{s['total']})")
    
    print("\n  Alpha_min (minimum alpha for flip) per case:")
    alpha_mins = [c["alpha_min_flip"] for c in all_case_results if c["alpha_min_flip"] is not None]
    if alpha_mins:
        print(f"    mean alpha_min = {np.mean(alpha_mins):.3f}")
        print(f"    median alpha_min = {np.median(alpha_mins):.3f}")
        print(f"    min = {min(alpha_mins):.3f}, max = {max(alpha_mins):.3f}")
    
    results = {
        "model": model_info.name,
        "alpha_flip_rates": {str(a): alpha_flip_stats[a]["flip"] / max(alpha_flip_stats[a]["total"], 1) for a in fine_alphas},
        "alpha_content_rates": {str(a): alpha_content_word_stats[a]["content"] / max(alpha_content_word_stats[a]["total"], 1) for a in fine_alphas},
        "alpha_min_stats": {
            "mean": float(np.mean(alpha_mins)) if alpha_mins else None,
            "median": float(np.median(alpha_mins)) if alpha_mins else None,
            "min": float(min(alpha_mins)) if alpha_mins else None,
            "max": float(max(alpha_mins)) if alpha_mins else None,
        },
        "per_case": all_case_results,
    }
    
    return results


# ================================================================
# Exp2 (7B): 探针方向的几何理解
# ================================================================

def exp2_geometric_structure(model, tokenizer, device, model_info):
    """
    ★★★★ 探针方向的几何理解
    
    核心思路:
    1. 计算探针方向的完整余弦矩阵
    2. 正四面体检验: 4个方向两两余弦≈-1/3?
    3. 分类边界分析: 6对角色之间的边界法线
    4. 对称性检验: 旋转群分析
    5. 降维可视化: t-SNE / PCA
    """
    print("\n" + "="*70)
    print("Exp2 (7B): 探针方向的几何理解")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    
    # 1. 训练探针(复用)
    print("\n[1] Training probe...")
    probe_result = train_probe_at_layer(
        model, tokenizer, device, model_info,
        model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    if probe_result is None:
        print("  ERROR: Failed to train probe")
        return None
    
    probe = probe_result['probe']
    scaler = probe_result['scaler']
    probe_directions = probe_result['probe_directions']
    
    print(f"  Probe train acc: {probe_result['train_acc']:.4f}")
    print(f"  Directions: {list(probe_directions.keys())}")
    
    # 2. 完整余弦矩阵
    print("\n[2] Cosine similarity matrix of probe directions:")
    
    cos_matrix = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            if r1 in probe_directions and r2 in probe_directions:
                cos_matrix[i, j] = float(np.dot(probe_directions[r1], probe_directions[r2]))
    
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_matrix[i, j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    
    # 3. 正四面体检验
    print("\n[3] Regular tetrahedron test:")
    print("  If 4 directions form a regular tetrahedron,")
    print("  pairwise cosine should be -1/3 ≈ -0.3333")
    
    # 收集非对角线元素
    off_diag = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag.append(cos_matrix[i, j])
    
    mean_off = np.mean(off_diag)
    std_off = np.std(off_diag)
    tetrahedron_error = abs(mean_off - (-1/3))
    
    print(f"  Mean off-diagonal cosine: {mean_off:.4f}")
    print(f"  Std off-diagonal cosine: {std_off:.4f}")
    print(f"  Expected (tetrahedron): -0.3333")
    print(f"  Error from tetrahedron: {tetrahedron_error:.4f}")
    
    if tetrahedron_error < 0.1:
        print("  ★ APPROXIMATELY TETRAHEDRON! (error < 0.1)")
    elif tetrahedron_error < 0.2:
        print("  ~ Weakly tetrahedral (error < 0.2)")
    else:
        print("  ✗ Not tetrahedral")
    
    # 4. 分类边界法线分析
    print("\n[4] Classification boundary normal vectors:")
    print("  For each pair (A, B), boundary normal = w_B - w_A")
    
    boundary_normals = {}
    boundary_cos_matrix = {}
    
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            if i >= j:
                continue
            if r1 in probe_directions and r2 in probe_directions:
                n = probe_directions[r2] - probe_directions[r1]
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-10:
                    n = n / n_norm
                boundary_normals[(r1, r2)] = n
    
    # 边界法线之间的余弦
    pairs = list(boundary_normals.keys())
    print(f"\n  Number of boundary pairs: {len(pairs)}")
    
    print("\n  Boundary normal cosine matrix:")
    pair_labels = [f"{a}-{b}" for a, b in pairs]
    print("        ", "  ".join(f"{l:>12s}" for l in pair_labels))
    for i, p1 in enumerate(pairs):
        row_vals = []
        for j, p2 in enumerate(pairs):
            c = float(np.dot(boundary_normals[p1], boundary_normals[p2]))
            row_vals.append(c)
            boundary_cos_matrix[(p1, p2)] = c
        row = "  ".join(f"{v:12.4f}" for v in row_vals)
        print(f"  {pair_labels[i]:>12s} {row}")
    
    # 5. 对称性分析
    print("\n[5] Symmetry analysis:")
    
    # 检查是否存在近似旋转对称性
    # 如果方向构成正四面体, 存在12阶旋转群(四面体群)
    
    # 简化检验: 检查所有方向到质心的距离是否相等
    center = np.mean([probe_directions[r] for r in ROLE_NAMES if r in probe_directions], axis=0)
    dists_to_center = []
    for r in ROLE_NAMES:
        if r in probe_directions:
            d = np.linalg.norm(probe_directions[r] - center)
            dists_to_center.append(d)
    
    if dists_to_center:
        print(f"  Distances to center: {', '.join(f'{d:.4f}' for d in dists_to_center)}")
        print(f"  Mean distance: {np.mean(dists_to_center):.4f}")
        print(f"  Std distance: {np.std(dists_to_center):.4f}")
        if np.std(dists_to_center) / max(np.mean(dists_to_center), 1e-10) < 0.1:
            print("  ★ Distances approximately equal → symmetric!")
        else:
            print("  Distances not equal → asymmetric")
    
    # 6. 降维可视化: PCA到2D/3D
    print("\n[6] PCA projection of probe directions:")
    
    dir_matrix = np.array([probe_directions[r] for r in ROLE_NAMES if r in probe_directions])
    
    if dir_matrix.shape[0] >= 4:
        # 用全部训练数据做PCA背景
        all_h = []
        for role_idx, role in enumerate(ROLE_NAMES):
            data = EXTENDED_DATA[role]
            for sent, target in zip(data["sentences"][:8], data["target_words"][:8]):
                h, toks = get_layer_hidden(model, tokenizer, device, sent, model_info.n_layers - 1)
                if h is None:
                    continue
                tokens = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
                dep_idx = find_token_index(tokens, target)
                if dep_idx is None:
                    continue
                all_h.append(h[0, dep_idx, :].float().cpu().numpy())
        
        if all_h:
            all_h = np.array(all_h)
            pca_bg = PCA(n_components=3)
            pca_bg.fit(all_h)
            
            # 投影探针方向
            for r in ROLE_NAMES:
                if r in probe_directions:
                    proj = pca_bg.transform(probe_directions[r].reshape(1, -1))[0]
                    print(f"  {r}: PC1={proj[0]:.4f}, PC2={proj[1]:.4f}, PC3={proj[2]:.4f}")
            
            # 投影质心
            center_proj = pca_bg.transform(center.reshape(1, -1))[0]
            print(f"  center: PC1={center_proj[0]:.4f}, PC2={center_proj[1]:.4f}, PC3={center_proj[2]:.4f}")
    
    # 7. 角度分析
    print("\n[7] Angle analysis:")
    
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            if i >= j:
                continue
            if r1 in probe_directions and r2 in probe_directions:
                cos_val = cos_matrix[i, j]
                angle = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
                print(f"  angle({r1}, {r2}) = {angle:.1f}° (cos={cos_val:.4f})")
    
    # 正四面体预期角度: arccos(-1/3) ≈ 109.47°
    tetra_angle = np.degrees(np.arccos(-1/3))
    print(f"\n  Regular tetrahedron expected angle: {tetra_angle:.2f}°")
    
    # 8. 子空间维度分析
    print("\n[8] Subspace dimensionality:")
    
    # 探针方向张成的子空间
    pca_dirs = PCA()
    pca_dirs.fit(dir_matrix)
    print(f"  PCA variance of 4 probe directions: {pca_dirs.explained_variance_ratio_}")
    cumvar = np.cumsum(pca_dirs.explained_variance_ratio_)
    print(f"  Cumulative variance: {cumvar}")
    
    # 如果前1-2个PC就能解释>90%, 说明4个方向接近共面/共线
    if cumvar[0] > 0.9:
        print("  ★ Nearly 1D! Directions are nearly collinear")
    elif len(cumvar) > 1 and cumvar[1] > 0.9:
        print("  ★ Nearly 2D! Directions are nearly coplanar")
    elif len(cumvar) > 2 and cumvar[2] > 0.9:
        print("  ~ 3D subspace")
    else:
        print("  Full 4D (tetrahedron-like)")
    
    results = {
        "model": model_info.name,
        "cos_matrix": cos_matrix.tolist(),
        "mean_off_diag": float(mean_off),
        "std_off_diag": float(std_off),
        "tetrahedron_error": float(tetrahedron_error),
        "is_approximately_tetrahedron": tetrahedron_error < 0.1,
        "dists_to_center": dists_to_center,
        "pca_variance": pca_dirs.explained_variance_ratio_.tolist(),
        "angles": {
            f"{r1}-{r2}": float(np.degrees(np.arccos(np.clip(cos_matrix[i, j], -1, 1))))
            for i, r1 in enumerate(ROLE_NAMES) for j, r2 in enumerate(ROLE_NAMES) if i < j
        },
    }
    
    return results


# ================================================================
# Exp3 (7C): 跨层语法角色操控
# ================================================================

def exp3_cross_layer_manipulation(model, tokenizer, device, model_info):
    """
    ★★★ 跨层语法角色操控
    
    核心思路:
    1. 在多个层(1/4, 1/2, 3/4, 最后一层)训练探针
    2. 比较各层探针的准确率 → 语法角色在哪些层开始编码?
    3. 在各层做探针方向操控 → 哪层操控效果最好?
    4. 分析跨层探针方向的一致性
    """
    print("\n" + "="*70)
    print("Exp3 (7C): 跨层语法角色操控")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    n_layers = model_info.n_layers
    
    # 采样层
    layer_indices = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))
    print(f"  Testing layers: {layer_indices}")
    
    # 1. 各层训练探针
    print("\n[1] Training probes at each layer...")
    
    layer_probes = {}
    for li in layer_indices:
        print(f"\n  Layer {li}/{n_layers-1}...")
        result = train_probe_at_layer(
            model, tokenizer, device, model_info,
            li, EXTENDED_DATA, ROLE_NAMES
        )
        if result is not None:
            layer_probes[li] = result
            print(f"    train_acc={result['train_acc']:.4f}, cv_acc={result['cv_acc']:.4f}, "
                  f"n_samples={result['n_samples']}")
        else:
            print(f"    Failed to train probe at layer {li}")
    
    # 2. 准确率随层变化曲线
    print("\n[2] Probe accuracy vs layer depth:")
    print(f"  {'Layer':>6s}  {'Train Acc':>10s}  {'CV Acc':>10s}  {'N Samples':>10s}")
    for li in sorted(layer_probes.keys()):
        r = layer_probes[li]
        print(f"  {li:>6d}  {r['train_acc']:>10.4f}  {r['cv_acc']:>10.4f}  {r['n_samples']:>10d}")
    
    # 3. 跨层探针方向余弦
    print("\n[3] Cross-layer direction consistency:")
    
    # 对每个角色, 各层方向之间的余弦
    for role in ROLE_NAMES:
        print(f"\n  {role}:")
        layer_dirs = {li: layer_probes[li]['probe_directions'].get(role) 
                     for li in layer_probes if role in layer_probes[li]['probe_directions']}
        
        if len(layer_dirs) < 2:
            print(f"    Not enough layers with direction")
            continue
        
        # 逐对余弦
        li_list = sorted(layer_dirs.keys())
        for i in range(len(li_list)):
            for j in range(i+1, len(li_list)):
                li1, li2 = li_list[i], li_list[j]
                d1, d2 = layer_dirs[li1], layer_dirs[li2]
                cos = float(np.dot(d1, d2))
                print(f"    cos(L{li1}, L{li2}) = {cos:.4f}")
    
    # 4. 各层操控测试
    print("\n[4] Cross-layer manipulation test:")
    
    manipulation_tests = [
        ("The cat sat on the mat", "cat", "nsubj", "dobj"),
        ("She chased the cat away", "cat", "dobj", "nsubj"),
        ("The beautiful cat sat quietly", "beautiful", "amod", "advmod"),
        ("The cat ran quickly home", "quickly", "advmod", "amod"),
    ]
    
    alphas = [0.5, 1.0, 2.0]
    
    layer_manipulation_results = {}
    
    for li in sorted(layer_probes.keys()):
        probe_data = layer_probes[li]
        probe = probe_data['probe']
        scaler_l = probe_data['scaler']
        probe_dirs_l = probe_data['probe_directions']
        
        layer_results = []
        
        for sent, target, src_role, tgt_role in manipulation_tests:
            h, toks = get_layer_hidden(model, tokenizer, device, sent, li)
            if h is None:
                continue
            
            tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target)
            if dep_idx is None:
                continue
            
            h_vec = h[0, dep_idx, :].float().cpu().numpy()
            h_norm = float(np.linalg.norm(h_vec))
            
            # 原始探针预测
            h_scaled = scaler_l.transform(h_vec.reshape(1, -1))
            orig_pred = int(probe.predict(h_scaled)[0])
            
            # 构造操控方向
            if tgt_role not in probe_dirs_l or src_role not in probe_dirs_l:
                continue
            
            probe_dir = probe_dirs_l[tgt_role] - probe_dirs_l[src_role]
            probe_dir_norm = np.linalg.norm(probe_dir)
            if probe_dir_norm < 1e-10:
                continue
            probe_dir = probe_dir / probe_dir_norm
            
            for alpha in alphas:
                perturbation = alpha * h_norm * probe_dir
                pert_h = h_vec + perturbation
                
                # 探针预测
                pert_h_scaled = scaler_l.transform(pert_h.reshape(1, -1))
                pert_pred = int(probe.predict(pert_h_scaled)[0])
                pert_pred_role = ROLE_NAMES[pert_pred]
                
                probe_flipped = (pert_pred_role == tgt_role)
                
                # 如果是最后一层, 计算logits变化
                kl = None
                top_token = None
                if li == n_layers - 1:
                    pert_h_t = torch.tensor(pert_h, dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
                    pert_h_full = h.clone()
                    pert_h_full[0, dep_idx, :] = pert_h_t[0, 0, :]
                    
                    pert_logits = compute_logits_from_hidden(model, pert_h_full)
                    
                    # 获取base logits
                    with torch.no_grad():
                        base_output = model(**toks)
                    base_logits_last = base_output.logits.detach().float()
                    base_probs = torch.softmax(base_logits_last[0, dep_idx].float(), dim=-1).cpu().numpy()
                    pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                    
                    kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
                    top_idx = np.argmax(pert_probs)
                    top_token = safe_decode(tokenizer, int(top_idx))
                
                layer_results.append({
                    "alpha": alpha,
                    "probe_flipped": probe_flipped,
                    "kl": kl,
                    "top_token": top_token,
                })
        
        # 汇总该层
        flip_rate = np.mean([r["probe_flipped"] for r in layer_results]) if layer_results else 0
        mean_kl = np.mean([r["kl"] for r in layer_results if r["kl"] is not None]) if any(r["kl"] is not None for r in layer_results) else None
        
        layer_manipulation_results[li] = {
            "flip_rate": float(flip_rate),
            "mean_kl": mean_kl,
            "n_tests": len(layer_results),
        }
        
        kl_str = f"{mean_kl:.3f}" if mean_kl is not None else "N/A"
        print(f"\n  Layer {li}: flip_rate={flip_rate:.1%}, "
              f"mean_KL={kl_str}, "
              f"n_tests={len(layer_results)}")
    
    # 5. 汇总
    print("\n" + "="*50)
    print("Cross-layer Summary")
    print("="*50)
    
    print(f"\n  {'Layer':>6s}  {'Train Acc':>10s}  {'Flip Rate':>10s}  {'Mean KL':>10s}")
    for li in sorted(layer_manipulation_results.keys()):
        lr = layer_manipulation_results[li]
        pr = layer_probes[li]
        kl_str = f"{lr['mean_kl']:.3f}" if lr['mean_kl'] is not None else "N/A"
        print(f"  {li:>6d}  {pr['train_acc']:>10.4f}  {lr['flip_rate']:>10.1%}  {kl_str:>10s}")
    
    results = {
        "model": model_info.name,
        "layer_indices": layer_indices,
        "layer_accuracies": {str(li): layer_probes[li]['train_acc'] for li in layer_probes},
        "layer_manipulation": {str(li): layer_manipulation_results.get(li, {}) for li in layer_indices},
    }
    
    return results


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CCL-J: Grammar Semantics & Cross-Layer")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    
    print(f"\n{'='*70}")
    print(f"Model: {model_info.name} ({model_info.model_class})")
    print(f"Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    print(f"{'='*70}")
    
    try:
        if args.exp == 1:
            results = exp1_fine_grained_manipulation(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_geometric_structure(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_cross_layer_manipulation(model, tokenizer, device, model_info)
        
        if results is not None:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
            os.makedirs(results_dir, exist_ok=True)
            
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(make_serializable(v) for v in obj)
                return obj
            
            results = make_serializable(results)
            
            exp_names = {1: "exp1_fine_grained", 2: "exp2_geometric", 3: "exp3_cross_layer"}
            out_path = os.path.join(results_dir, f"cclj_{exp_names[args.exp]}_{model_info.name}_results.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  Results saved to {out_path}")
    
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
