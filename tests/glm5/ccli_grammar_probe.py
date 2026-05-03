"""
CCL-I(250.9): 语法角色探针与真正的角色操控
==================================================
核心问题: CCL-H发现"因果梯度≠语法角色方向" — 因果梯度改变top token,
但不改变语法角色. 需要训练语法角色探针(probe), 用探针方向做操控.

三合一实验:
  Exp1 (Phase 6A): ★★★★★ 语法角色线性探针
    → 在hidden states上训练线性分类器(4分类: nsubj/dobj/amod/advmod)
    → 分析: 探针的权重方向是否与V_grammar对齐?
    → 关键: 探针方向 vs 因果梯度方向 的余弦相似度

  Exp2 (Phase 6B): ★★★★★ 探针方向操控
    → 沿探针分类边界方向扰动, 检查语法角色分类是否改变
    → 同时检查输出文本是否仍然连贯
    → 对比: 探针方向 vs 因果梯度方向 的操控效果

  Exp3 (Phase 6C): ★★★★ V_grammar的通用/特异分解
    → 分解V_grammar = V_universal ⊕ V_sentence_specific
    → 多句子PCA, 找跨句子不变的主方向
    → 测试V_universal方向的跨句子操控效果
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 数据集 =====
# 训练集: 用于提取语法方向和训练探针
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

# 扩展数据集: 更多句子以增强探针训练
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

# 泛化测试集
GENERALIZATION_DATA = {
    "nsubj": {
        "sentences": [
            "The knight guarded the castle",
            "The nurse cared for the patient",
            "The pilot flew the airplane",
            "The driver steered the truck",
        ],
        "target_words": ["knight", "nurse", "pilot", "driver"],
    },
    "dobj": {
        "sentences": [
            "They respected the knight deeply",
            "She thanked the nurse warmly",
            "He watched the pilot carefully",
            "We hired the driver recently",
        ],
        "target_words": ["knight", "nurse", "pilot", "driver"],
    },
    "amod": {
        "sentences": [
            "The noble knight stood tall",
            "The gentle nurse smiled warmly",
            "The experienced pilot landed safely",
            "The careful driver turned slowly",
        ],
        "target_words": ["noble", "gentle", "experienced", "careful"],
    },
    "advmod": {
        "sentences": [
            "The knight fought valiantly there",
            "The nurse worked patiently always",
            "The pilot landed smoothly today",
            "The driver turned carefully now",
        ],
        "target_words": ["valiantly", "patiently", "smoothly", "carefully"],
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


# ================================================================
# Exp1 (Phase 6A): 语法角色线性探针
# ================================================================

def exp1_grammar_probe(model, tokenizer, device, model_info):
    """
    ★★★★★ 语法角色线性探针
    
    核心思路:
    1. 从多个句子收集hidden states + 语法角色标签
    2. 训练线性探针(4分类): nsubj/dobj/amod/advmod
    3. 分析探针权重方向:
       - 与V_grammar的PCA基对齐度
       - 与因果梯度方向的余弦
       - 探针方向是否比因果梯度方向更适合操控
    
    关键问题:
    → 探针权重 w_role 使得 w_role^T h + b = 0 是分类边界
    → 沿 w_role 扰动 h, 理论上可以改变分类结果
    → 这是"真正的语法角色操控方向"
    """
    print("\n" + "="*70)
    print("Exp1 (Phase 6A): 语法角色线性探针")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    
    # 1. 收集hidden states和标签
    print("\n[1] Collecting hidden states for probe training...")
    
    all_hidden = []
    all_labels = []
    all_sentences = []
    all_targets = []
    
    for role_idx, role in enumerate(ROLE_NAMES):
        data = EXTENDED_DATA[role]
        n_collected = 0
        for sent, target in zip(data["sentences"], data["target_words"]):
            h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
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
            all_sentences.append(sent)
            all_targets.append(target)
            n_collected += 1
        
        print(f"  {role}: {n_collected} samples")
    
    X = np.array(all_hidden)  # [N, d_model]
    y = np.array(all_labels)  # [N]
    print(f"  Total: {X.shape[0]} samples, dim={X.shape[1]}")
    
    # 2. 训练线性探针
    print("\n[2] Training linear probe (LogisticRegression)...")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LogisticRegression with L2 (linear probe)
    probe = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
    )
    
    # 交叉验证
    cv_scores = cross_val_score(probe, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 在全部数据上训练
    probe.fit(X_scaled, y)
    train_acc = probe.score(X_scaled, y)
    print(f"  Train accuracy: {train_acc:.4f}")
    
    # 3. 提取探针权重方向
    print("\n[3] Extracting probe weight directions...")
    
    # probe.coef_ shape: [n_classes, d_model]
    # 对于multinomial LR, 方向 = w_target - w_source
    probe_weights = probe.coef_  # [4, d_model] (after StandardScaler)
    probe_intercept = probe.intercept_  # [4]
    
    # 转换回原始空间(去除StandardScaler的影响)
    # StandardScaler: X_scaled = (X - mean) / std
    # LR在scaled空间: w_scaled^T X_scaled + b = 0
    # 原始空间: (w_scaled / std)^T X + (b - w_scaled^T mean/std) = 0
    # 所以原始权重: w_orig = w_scaled / std, b_orig = b - w_scaled^T (mean/std)
    scale_factors = scaler.scale_  # [d_model]
    mean_factors = scaler.mean_   # [d_model]
    
    probe_weights_orig = probe_weights / scale_factors[np.newaxis, :]  # [4, d_model]
    probe_intercept_orig = probe_intercept - np.sum(probe_weights * (mean_factors / scale_factors)[np.newaxis, :], axis=1)
    
    # 归一化探针权重方向
    probe_directions = {}
    for i, role in enumerate(ROLE_NAMES):
        w = probe_weights_orig[i]
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            probe_directions[role] = w / w_norm
        else:
            probe_directions[role] = w
        print(f"  {role}: |w|={w_norm:.4f}, intercept={probe_intercept_orig[i]:.4f}")
    
    # 4. 分析探针方向 vs 因果梯度方向
    print("\n[4] Comparing probe directions vs causal gradient directions...")
    
    # 收集因果梯度方向
    causal_dirs = {}
    for role in ROLE_NAMES:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:6],
            data["target_words"][:6],
            role,
        )
        if dirs:
            causal_dirs[role] = np.mean([d['direction'] for d in dirs], axis=0)
            causal_dirs[role] = causal_dirs[role] / max(np.linalg.norm(causal_dirs[role]), 1e-10)
        print(f"  {role}: {len(dirs)} causal directions collected")
    
    # 计算探针方向 vs 因果梯度方向的余弦相似度
    print("\n  Probe direction vs Causal gradient direction:")
    probe_vs_causal = {}
    for role in ROLE_NAMES:
        if role in probe_directions and role in causal_dirs:
            cos = float(np.dot(probe_directions[role], causal_dirs[role]))
            probe_vs_causal[role] = cos
            print(f"    {role}: cos(probe, causal) = {cos:.4f}")
        else:
            print(f"    {role}: missing direction")
    
    # 5. 探针方向之间的余弦矩阵
    print("\n  Probe direction cosine matrix:")
    cos_matrix = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            if r1 in probe_directions and r2 in probe_directions:
                cos_matrix[i, j] = float(np.dot(probe_directions[r1], probe_directions[r2]))
    
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_matrix[i, j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    
    # 6. 探针方向 vs V_grammar PCA基
    print("\n[5] Probe directions vs V_grammar PCA basis...")
    
    # 收集所有因果方向做PCA
    all_causal_dirs = []
    for role in ROLE_NAMES:
        if role in causal_dirs:
            all_causal_dirs.append(causal_dirs[role])
    
    if len(all_causal_dirs) >= 4:
        all_causal_matrix = np.array(all_causal_dirs)  # [4, d_model]
        pca = PCA(n_components=min(4, len(all_causal_dirs)))
        pca.fit(all_causal_matrix)
        
        print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
        
        # 探针方向在V_grammar PCA基上的投影
        for role in ROLE_NAMES:
            if role in probe_directions:
                proj = pca.transform(probe_directions[role].reshape(1, -1))[0]
                print(f"  {role} probe in V_grammar PCA coords: {proj}")
    
    # 7. LDA探针(替代方案)
    print("\n[6] Training LDA probe for comparison...")
    
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(X_scaled, y)
    lda_acc = lda.score(X_scaled, y)
    print(f"  LDA train accuracy: {lda_acc:.4f}")
    
    # LDA方向
    lda_directions = {}
    lda_scalings = lda.scalings_  # [d_model, n_components]
    for i, role in enumerate(ROLE_NAMES):
        if i < lda_scalings.shape[1]:
            w = lda_scalings[:, i] / scale_factors  # 转换回原始空间
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                lda_directions[role] = w / w_norm
    
    # 8. 在泛化集上测试探针
    print("\n[7] Testing probe on generalization set...")
    
    gen_hidden = []
    gen_labels = []
    for role_idx, role in enumerate(ROLE_NAMES):
        data = GENERALIZATION_DATA[role]
        for sent, target in zip(data["sentences"], data["target_words"]):
            h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
            if h is None:
                continue
            tokens = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            h_vec = h[0, dep_idx, :].float().cpu().numpy()
            gen_hidden.append(h_vec)
            gen_labels.append(role_idx)
    
    if gen_hidden:
        X_gen = np.array(gen_hidden)
        y_gen = np.array(gen_labels)
        X_gen_scaled = scaler.transform(X_gen)
        gen_acc = probe.score(X_gen_scaled, y_gen)
        print(f"  Generalization accuracy: {gen_acc:.4f} ({len(gen_hidden)} samples)")
    else:
        gen_acc = 0.0
    
    results = {
        "model": model_info.name,
        "n_train": len(X),
        "n_gen": len(gen_hidden) if gen_hidden else 0,
        "cv_accuracy": float(cv_scores.mean()),
        "train_accuracy": float(train_acc),
        "gen_accuracy": float(gen_acc),
        "probe_vs_causal": probe_vs_causal,
        "cos_matrix": cos_matrix.tolist(),
        "pca_variance": pca.explained_variance_ratio_.tolist() if len(all_causal_dirs) >= 4 else [],
    }
    
    # 保存探针和scaler用于Exp2
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(results_dir, exist_ok=True)
    
    np.savez(
        os.path.join(results_dir, f"ccli_probe_{model_info.name}.npz"),
        probe_weights_orig=probe_weights_orig,
        probe_intercept_orig=probe_intercept_orig,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        X_train=X,
        y_train=y,
    )
    
    # 保存方向向量
    dir_data = {}
    for role in ROLE_NAMES:
        if role in probe_directions:
            dir_data[f"probe_{role}"] = probe_directions[role]
        if role in causal_dirs:
            dir_data[f"causal_{role}"] = causal_dirs[role]
    
    np.savez(
        os.path.join(results_dir, f"ccli_directions_{model_info.name}.npz"),
        **dir_data,
    )
    
    print("\n  Probe and directions saved.")
    return results


# ================================================================
# Exp2 (Phase 6B): 探针方向操控
# ================================================================

def exp2_probe_manipulation(model, tokenizer, device, model_info):
    """
    ★★★★★ 探针方向操控
    
    核心思路:
    1. 加载Exp1训练的探针
    2. 沿探针分类边界方向扰动hidden state
    3. 检查: 探针的分类结果是否改变?
    4. 同时检查: 输出token是否改变? 是否连贯?
    5. 对比: 探针方向 vs 因果梯度方向 的操控效果
    
    关键验证:
    → 如果沿探针方向扰动, 探针分类从nsubj→dobj
    → 同时输出token仍然连贯
    → 则证明: 探针方向是"真正的语法角色操控方向"
    """
    print("\n" + "="*70)
    print("Exp2 (Phase 6B): 探针方向操控")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    
    # 1. 加载探针和方向
    print("\n[1] Loading probe and directions...")
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    
    probe_path = os.path.join(results_dir, f"ccli_probe_{model_info.name}.npz")
    dir_path = os.path.join(results_dir, f"ccli_directions_{model_info.name}.npz")
    
    if not os.path.exists(probe_path) or not os.path.exists(dir_path):
        print("  ERROR: Probe not found. Run Exp1 first.")
        return None
    
    probe_data = np.load(probe_path)
    dir_data = np.load(dir_path)
    
    probe_weights_orig = probe_data['probe_weights_orig']
    probe_intercept_orig = probe_data['probe_intercept_orig']
    scaler_mean = probe_data['scaler_mean']
    scaler_scale = probe_data['scaler_scale']
    
    probe_directions = {}
    causal_dirs = {}
    for role in ROLE_NAMES:
        key_p = f"probe_{role}"
        key_c = f"causal_{role}"
        if key_p in dir_data:
            probe_directions[role] = dir_data[key_p]
        if key_c in dir_data:
            causal_dirs[role] = dir_data[key_c]
    
    print(f"  Loaded probe weights: {probe_weights_orig.shape}")
    print(f"  Probe directions: {list(probe_directions.keys())}")
    print(f"  Causal directions: {list(causal_dirs.keys())}")
    
    # 2. 重新训练探针分类器(用于快速预测)
    print("\n[2] Retraining probe for prediction...")
    
    X_train = probe_data['X_train']
    y_train = probe_data['y_train']
    
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    X_scaled = scaler.transform(X_train)
    probe.fit(X_scaled, y_train)
    print(f"  Probe retrained, accuracy: {probe.score(X_scaled, y_train):.4f}")
    
    # 3. 定义操控测试
    print("\n[3] Testing probe-based manipulation...")
    
    # 操控方向: 从角色A到角色B = probe_direction[B] - probe_direction[A]
    # 这沿分类边界移动, 应该改变分类结果
    
    manipulation_tests = [
        ("The cat sat on the mat", "cat", "nsubj", "dobj"),
        ("The dog ran through the park", "dog", "nsubj", "dobj"),
        ("She chased the cat away", "cat", "dobj", "nsubj"),
        ("He found the dog outside", "dog", "dobj", "nsubj"),
        ("The beautiful cat sat quietly", "beautiful", "amod", "advmod"),
        ("The large dog ran swiftly", "large", "amod", "advmod"),
        ("The cat ran quickly home", "quickly", "advmod", "amod"),
        ("The dog barked loudly today", "loudly", "advmod", "amod"),
    ]
    
    alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # 操控方式1: 探针方向 (w_target - w_source)
    # 操控方式2: 因果梯度方向
    # 操控方式3: 随机方向 (基线)
    
    results_by_method = {
        "probe": {"success": [], "coherent": [], "kl": [], "probe_flip": []},
        "causal": {"success": [], "coherent": [], "kl": [], "probe_flip": []},
        "random": {"success": [], "coherent": [], "kl": [], "probe_flip": []},
    }
    
    per_case_results = []
    
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
        base_top_indices = np.argsort(base_probs)[::-1][:5]
        base_top_tokens = [(safe_decode(tokenizer, int(idx)), float(base_probs[idx])) for idx in base_top_indices]
        
        # 原始探针预测
        h_scaled = scaler.transform(h_vec.reshape(1, -1))
        orig_pred = int(probe.predict(h_scaled)[0])
        orig_pred_role = ROLE_NAMES[orig_pred]
        orig_prob = probe.predict_proba(h_scaled)[0]
        
        print(f"    h_norm={h_norm:.2f}, probe predicts: {orig_pred_role} "
              f"(probs: {', '.join(f'{ROLE_NAMES[i]}={p:.3f}' for i, p in enumerate(orig_prob))})")
        print(f"    Base top-5: {', '.join(f'{t}({p:.3f})' for t, p in base_top_tokens[:3])}")
        
        # 构造操控方向
        # 方法1: 探针方向 = probe_direction[tgt] - probe_direction[src]
        if tgt_role in probe_directions and src_role in probe_directions:
            probe_dir = probe_directions[tgt_role] - probe_directions[src_role]
            probe_dir_norm = np.linalg.norm(probe_dir)
            if probe_dir_norm > 1e-10:
                probe_dir = probe_dir / probe_dir_norm
        else:
            probe_dir = None
        
        # 方法2: 因果梯度方向
        causal_dir = causal_dirs.get(tgt_role)
        
        # 方法3: 随机方向
        d_model = model_info.d_model
        random_dir = np.random.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        
        case_data = {
            "sentence": sent, "target": target,
            "src_role": src_role, "tgt_role": tgt_role,
            "orig_probe_pred": orig_pred_role,
            "alphas": {},
        }
        
        for method_name, direction in [("probe", probe_dir), ("causal", causal_dir), ("random", random_dir)]:
            if direction is None:
                continue
            
            method_results = []
            for alpha in alphas:
                # 扰动
                perturbation = alpha * (h_norm / np.linalg.norm(direction)) * direction
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
                
                top_indices = np.argsort(pert_probs)[::-1][:5]
                top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
                
                token_changed = base_top_tokens[0][0] != top_tokens[0][0]
                coherent = is_coherent_token(top_tokens[0][0]) if top_tokens else False
                
                method_results.append({
                    "alpha": alpha,
                    "kl": kl,
                    "token_changed": token_changed,
                    "coherent": coherent,
                    "probe_flipped": probe_flipped,
                    "probe_pred": pert_pred_role,
                    "top3": top_tokens[:3],
                    "tgt_prob": float(pert_prob[ROLE_NAMES.index(tgt_role)]),
                })
                
                # 汇总
                results_by_method[method_name]["kl"].append(kl)
                results_by_method[method_name]["coherent"].append(1.0 if coherent else 0.0)
                results_by_method[method_name]["probe_flip"].append(1.0 if probe_flipped else 0.0)
            
            case_data["alphas"][method_name] = method_results
            
            # 打印关键alpha的结果
            for mr in method_results:
                if mr["alpha"] in [0.5, 1.0, 2.0]:
                    print(f"    {method_name} α={mr['alpha']}: probe→{mr['probe_pred']}, "
                          f"flip={mr['probe_flipped']}, KL={mr['kl']:.3f}, "
                          f"changed={mr['token_changed']}, coherent={mr['coherent']}, "
                          f"top1={mr['top3'][0] if mr['top3'] else 'N/A'}")
        
        per_case_results.append(case_data)
    
    # 4. 汇总
    print("\n" + "="*50)
    print("Manipulation Summary:")
    print("="*50)
    
    for method_name in ["probe", "causal", "random"]:
        s = results_by_method[method_name]
        n = len(s["kl"]) if s["kl"] else 1
        mean_kl = np.mean(s["kl"]) if s["kl"] else 0
        coherent_rate = np.mean(s["coherent"]) if s["coherent"] else 0
        flip_rate = np.mean(s["probe_flip"]) if s["probe_flip"] else 0
        
        print(f"\n  {method_name.upper()} method:")
        print(f"    Mean KL: {mean_kl:.4f}")
        print(f"    Coherent rate: {coherent_rate:.1%}")
        print(f"    Probe flip rate: {flip_rate:.1%}  ← KEY METRIC")
    
    # 关键对比: probe flip rate
    probe_flip = np.mean(results_by_method["probe"]["probe_flip"]) if results_by_method["probe"]["probe_flip"] else 0
    causal_flip = np.mean(results_by_method["causal"]["probe_flip"]) if results_by_method["causal"]["probe_flip"] else 0
    random_flip = np.mean(results_by_method["random"]["probe_flip"]) if results_by_method["random"]["probe_flip"] else 0
    
    print(f"\n  ★ PROBE FLIP RATE (key metric):")
    print(f"    Probe direction: {probe_flip:.1%}")
    print(f"    Causal gradient: {causal_flip:.1%}")
    print(f"    Random baseline: {random_flip:.1%}")
    
    results = {
        "model": model_info.name,
        "summary": {
            "probe_flip_rate": float(probe_flip),
            "causal_flip_rate": float(causal_flip),
            "random_flip_rate": float(random_flip),
            "probe_mean_kl": float(np.mean(results_by_method["probe"]["kl"])) if results_by_method["probe"]["kl"] else 0,
            "causal_mean_kl": float(np.mean(results_by_method["causal"]["kl"])) if results_by_method["causal"]["kl"] else 0,
            "probe_coherent_rate": float(np.mean(results_by_method["probe"]["coherent"])) if results_by_method["probe"]["coherent"] else 0,
            "causal_coherent_rate": float(np.mean(results_by_method["causal"]["coherent"])) if results_by_method["causal"]["coherent"] else 0,
        },
        "per_case_results": per_case_results,
    }
    
    return results


# ================================================================
# Exp3 (Phase 6C): V_grammar通用/特异分解
# ================================================================

def exp3_grammar_decomposition(model, tokenizer, device, model_info):
    """
    ★★★★ V_grammar通用/特异分解
    
    核心思路:
    1. 从多个不同词汇的句子集收集因果方向
    2. PCA分解: 找跨句子集不变的主方向(V_universal)
    3. 残差: 句子特异的成分(V_specific)
    4. 测试: V_universal方向是否跨句子操控成功?
    
    方法:
    → 对每种语法角色, 从不同句子集收集方向
    → 对所有方向做PCA
    → PC1-3 (高方差): V_universal
    → PC4+ (低方差): V_specific
    """
    print("\n" + "="*70)
    print("Exp3 (Phase 6C): V_grammar通用/特异分解")
    print("="*70)
    
    ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]
    
    # 1. 从两个不同的句子集收集因果方向
    print("\n[1] Collecting causal directions from multiple sentence sets...")
    
    all_dirs_by_role = {role: [] for role in ROLE_NAMES}
    
    # 句子集1: 原始数据
    for role in ROLE_NAMES:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:6],
            data["target_words"][:6],
            role,
        )
        for d in dirs:
            all_dirs_by_role[role].append({
                'direction': d['direction'],
                'source': 'train',
            })
        print(f"  {role} (train): {len(dirs)} dirs")
    
    # 句子集2: 扩展数据(不同词汇)
    for role in ROLE_NAMES:
        data = EXTENDED_DATA[role]
        # 只取不在SYNTAX_DATA中的句子
        new_sents = []
        new_targets = []
        for s, t in zip(data["sentences"], data["target_words"]):
            if s not in SYNTAX_DATA[role]["sentences"]:
                new_sents.append(s)
                new_targets.append(t)
        
        dirs = collect_causal_directions(
            model, tokenizer, device,
            new_sents[:6],
            new_targets[:6],
            role,
        )
        for d in dirs:
            all_dirs_by_role[role].append({
                'direction': d['direction'],
                'source': 'extended',
            })
        print(f"  {role} (extended): {len(dirs)} dirs")
    
    # 2. 对每种角色做PCA分解
    print("\n[2] PCA decomposition per role...")
    
    universal_dirs = {}
    specific_dirs = {}
    pca_results = {}
    
    for role in ROLE_NAMES:
        dirs = [d['direction'] for d in all_dirs_by_role[role]]
        if len(dirs) < 3:
            print(f"  {role}: too few directions ({len(dirs)}), skipping")
            continue
        
        dirs_matrix = np.array(dirs)  # [N, d_model]
        
        pca = PCA()
        pca.fit(dirs_matrix)
        
        var_ratio = pca.explained_variance_ratio_
        cumvar = np.cumsum(var_ratio)
        
        # 找到90%方差的截止点
        k90 = np.searchsorted(cumvar, 0.9) + 1
        k90 = max(k90, 2)  # 至少2个
        
        print(f"  {role}: N={len(dirs)}, k90={k90}, "
              f"var[0:3]={var_ratio[:3]}")
        
        # V_universal = 前 k90 个主方向
        universal_components = pca.components_[:k90]  # [k90, d_model]
        
        # V_specific = 后面的主方向
        specific_components = pca.components_[k90:] if k90 < len(pca.components_) else np.array([])
        
        # 平均通用方向
        universal_mean = np.mean(universal_components, axis=0)
        universal_mean = universal_mean / max(np.linalg.norm(universal_mean), 1e-10)
        universal_dirs[role] = universal_mean
        
        if len(specific_components) > 0:
            specific_mean = np.mean(specific_components, axis=0)
            specific_mean = specific_mean / max(np.linalg.norm(specific_mean), 1e-10)
            specific_dirs[role] = specific_mean
        
        pca_results[role] = {
            "n_dirs": len(dirs),
            "k90": int(k90),
            "var_top3": var_ratio[:3].tolist(),
            "cumvar_top3": cumvar[:3].tolist(),
        }
    
    # 3. 通用方向 vs 因果梯度方向的余弦
    print("\n[3] Universal direction vs causal gradient direction...")
    
    causal_dirs = {}
    for role in ROLE_NAMES:
        data = SYNTAX_DATA[role]
        dirs = collect_causal_directions(
            model, tokenizer, device,
            data["sentences"][:4],
            data["target_words"][:4],
            role,
        )
        if dirs:
            causal_dirs[role] = np.mean([d['direction'] for d in dirs], axis=0)
            causal_dirs[role] = causal_dirs[role] / max(np.linalg.norm(causal_dirs[role]), 1e-10)
    
    for role in ROLE_NAMES:
        if role in universal_dirs and role in causal_dirs:
            cos = float(np.dot(universal_dirs[role], causal_dirs[role]))
            print(f"  {role}: cos(universal, causal) = {cos:.4f}")
    
    # 4. 测试通用方向的跨句子操控效果
    print("\n[4] Testing universal direction cross-sentence manipulation...")
    
    test_cases = [
        ("The knight guarded the castle", "knight", "nsubj", "dobj"),
        ("The nurse cared for the patient", "nurse", "nsubj", "dobj"),
        ("They respected the knight deeply", "knight", "dobj", "nsubj"),
        ("She thanked the nurse warmly", "nurse", "dobj", "nsubj"),
    ]
    
    alphas = [0.5, 1.0, 2.0]
    
    universal_results = []
    causal_cross_results = []
    
    for sent, target, src_role, tgt_role in test_cases:
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
        
        # 通用方向操控
        if tgt_role in universal_dirs and src_role in universal_dirs:
            uni_dir = universal_dirs[tgt_role] - universal_dirs[src_role]
            uni_dir_norm = np.linalg.norm(uni_dir)
            if uni_dir_norm > 1e-10:
                uni_dir = uni_dir / uni_dir_norm
                
                for alpha in alphas:
                    perturbation = alpha * h_norm * uni_dir
                    pert_h = h_vec + perturbation
                    
                    pert_h_t = torch.tensor(pert_h, dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
                    pert_h_full = h.clone()
                    pert_h_full[0, dep_idx, :] = pert_h_t[0, 0, :]
                    
                    pert_logits = compute_logits_from_hidden(model, pert_h_full)
                    pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                    
                    kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
                    
                    top_indices = np.argsort(pert_probs)[::-1][:5]
                    top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
                    
                    coherent = is_coherent_token(top_tokens[0][0]) if top_tokens else False
                    
                    print(f"    Universal α={alpha}: KL={kl:.3f}, coherent={coherent}, "
                          f"top1={top_tokens[0] if top_tokens else 'N/A'}")
                    
                    universal_results.append({
                        "alpha": alpha, "kl": kl, "coherent": coherent,
                        "top1": top_tokens[0] if top_tokens else None,
                    })
        
        # 因果梯度方向操控(对比)
        if tgt_role in causal_dirs:
            for alpha in alphas:
                perturbation = alpha * h_norm * causal_dirs[tgt_role]
                pert_h = h_vec + perturbation
                
                pert_h_t = torch.tensor(pert_h, dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
                pert_h_full = h.clone()
                pert_h_full[0, dep_idx, :] = pert_h_t[0, 0, :]
                
                pert_logits = compute_logits_from_hidden(model, pert_h_full)
                pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                
                kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
                
                top_indices = np.argsort(pert_probs)[::-1][:5]
                top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
                
                coherent = is_coherent_token(top_tokens[0][0]) if top_tokens else False
                
                print(f"    Causal α={alpha}: KL={kl:.3f}, coherent={coherent}, "
                      f"top1={top_tokens[0] if top_tokens else 'N/A'}")
                
                causal_cross_results.append({
                    "alpha": alpha, "kl": kl, "coherent": coherent,
                    "top1": top_tokens[0] if top_tokens else None,
                })
    
    # 5. 通用方向 vs 特异方向的操控对比
    print("\n[5] Universal vs Specific direction comparison...")
    
    # 在训练集句子上测试
    for sent, target, src_role, tgt_role in [
        ("The cat sat on the mat", "cat", "nsubj", "dobj"),
        ("She chased the cat away", "cat", "dobj", "nsubj"),
    ]:
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
        
        print(f"\n  [{src_role}→{tgt_role}] {sent} / '{target}'")
        
        for dir_name, dir_dict in [("universal", universal_dirs), ("specific", specific_dirs)]:
            if tgt_role in dir_dict and src_role in dir_dict:
                d = dir_dict[tgt_role] - dir_dict[src_role]
                d_norm = np.linalg.norm(d)
                if d_norm > 1e-10:
                    d = d / d_norm
                    for alpha in [1.0, 2.0]:
                        perturbation = alpha * h_norm * d
                        pert_h = h_vec + perturbation
                        
                        pert_h_t = torch.tensor(pert_h, dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
                        pert_h_full = h.clone()
                        pert_h_full[0, dep_idx, :] = pert_h_t[0, 0, :]
                        
                        pert_logits = compute_logits_from_hidden(model, pert_h_full)
                        pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                        
                        kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
                        top_indices = np.argsort(pert_probs)[::-1][:3]
                        top_tokens = [(safe_decode(tokenizer, int(idx)), float(pert_probs[idx])) for idx in top_indices]
                        coherent = is_coherent_token(top_tokens[0][0]) if top_tokens else False
                        
                        print(f"    {dir_name} α={alpha}: KL={kl:.3f}, coherent={coherent}, "
                              f"top1={top_tokens[0] if top_tokens else 'N/A'}")
    
    # 6. 汇总
    uni_kl = np.mean([r["kl"] for r in universal_results]) if universal_results else 0
    uni_coherent = np.mean([1.0 if r["coherent"] else 0.0 for r in universal_results]) if universal_results else 0
    causal_kl = np.mean([r["kl"] for r in causal_cross_results]) if causal_cross_results else 0
    causal_coherent = np.mean([1.0 if r["coherent"] else 0.0 for r in causal_cross_results]) if causal_cross_results else 0
    
    print(f"\n  Universal direction: mean_KL={uni_kl:.3f}, coherent={uni_coherent:.1%}")
    print(f"  Causal direction:    mean_KL={causal_kl:.3f}, coherent={causal_coherent:.1%}")
    
    results = {
        "model": model_info.name,
        "pca_results": pca_results,
        "universal_mean_kl": float(uni_kl),
        "universal_coherent_rate": float(uni_coherent),
        "causal_mean_kl": float(causal_kl),
        "causal_coherent_rate": float(causal_coherent),
    }
    
    return results


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CCL-I: Grammar Probe & Role Manipulation")
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
            results = exp1_grammar_probe(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_probe_manipulation(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_grammar_decomposition(model, tokenizer, device, model_info)
        
        if results is not None:
            # 保存结果
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
            os.makedirs(results_dir, exist_ok=True)
            
            # 需要序列化的数据转为可JSON化的
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(v) for v in obj]
                if isinstance(obj, tuple):
                    return [make_serializable(v) for v in obj]
                if isinstance(obj, bool):
                    return obj
                return obj
            
            results = make_serializable(results)
            
            result_path = os.path.join(results_dir, f"ccli_exp{args.exp}_{args.model}_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {result_path}")
    
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
