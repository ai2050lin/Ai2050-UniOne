"""
CCXXVIII(376): 语法变换的几何解码

核心问题:
  1. 同一概念在不同语法角色(主语/宾语/修饰语)下, 在V_sem中如何变化?
  2. 语法变换是否对应V_sem中的线性变换?
  3. 跨概念的语法变换是否一致? (如果一致, 说明语法=群结构)

理论背景(CCXVII建立):
  V_mid = V_sem ⊕ V_nonsem
  V_sem: ~5维语义子空间, 包含~25%方差, ~180%类别分离
  假设: 语法角色 = V_sem上的线性变换群

实验设计:
  Exp1: 语法角色解码
    - 构造句子: "The [NOUN] [VERB]s" (主语) vs "[VERB] the [NOUN]" (宾语)
      vs "The [ADJ] [NOUN]" (被修饰) vs "[NOUN]" (孤立)
    - 提取NOUN在各语法角色下的表示
    - 在V_sem中: 不同语法角色的NOUN是否落在不同区域?
    - 量化: 语法角色可解码性(probe_acc), V_sem中的角色距离

  Exp2: 语法变换的线性性
    - 对同一概念: rep(主语) - rep(孤立) = Δ_subject
    - 对同一概念: rep(宾语) - rep(孤立) = Δ_object
    - 问题: Δ_subject和Δ_object是否跨概念一致?
    - 如果一致: 存在全局的语法变换矩阵 T_sub, T_obj
    - 量化: 跨概念Δ的余弦相似度, 线性拟合误差

  Exp3: V_sem vs V_nonsem中的语法信息
    - 分别在V_sem(Top-5 PCs)和V_nonsem(其余PCs)中
    - 测量语法角色的可解码性
    - 如果V_sem中可解码: 语法是语义结构的一部分
    - 如果V_nonsem中也可解码: 语法独立于语义
"""

import sys, os, argparse, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
TEMP.mkdir(parents=True, exist_ok=True)

# ===== 实验素材 =====
# 选择20个名词，构造4种语法角色
TEST_NOUNS = [
    # 动物(5)
    "cat", "dog", "bird", "fish", "horse",
    # 物品(5)
    "hammer", "knife", "wheel", "rope", "sword",
    # 自然(5)
    "rain", "snow", "wind", "sun", "stone",
    # 抽象(5)
    "time", "truth", "power", "freedom", "justice",
]

# 4种语法角色的句子模板
# role: subject(主语), object(宾语), modified(被修饰), isolated(孤立)
def make_sentences(noun):
    """为每个名词生成4种语法角色的句子"""
    return {
        "subject": f"The {noun} runs",
        "object": f"Sees the {noun}",
        "modified": f"The red {noun}",
        "isolated": f"{noun}",
    }

# 10个动词用于Exp1的额外验证
TEST_VERBS = ["runs", "eats", "sees", "makes", "finds", "takes", "gives", "knows", "loves", "needs"]

# 10个形容词用于Exp1的额外验证  
TEST_ADJS = ["red", "big", "old", "new", "good", "bad", "dark", "cold", "hot", "fast"]


def extract_noun_rep_at_role(model, tokenizer, device, noun, sentence, model_info):
    """
    提取名词在特定语法角色句子中的表示
    关键: 精确定位名词的最后一个token位置
    """
    n_layers = model_info.n_layers
    
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # 找到名词的token位置
    noun_tokens = tokenizer(noun, add_special_tokens=False)['input_ids']
    sentence_tokens = input_ids[0].tolist()
    
    # 在句子token中找到名词的起始位置
    target_pos = None
    for i in range(len(sentence_tokens) - len(noun_tokens) + 1):
        if sentence_tokens[i:i+len(noun_tokens)] == noun_tokens:
            target_pos = i + len(noun_tokens) - 1  # 名词最后一个token
            break
    
    if target_pos is None:
        # fallback: 用最后一个非特殊token
        target_pos = len(sentence_tokens) - 1
    
    with torch.no_grad():
        try:
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # 收集每层的表示
            reps = {}
            for layer_idx in range(n_layers + 1):
                reps[layer_idx] = hidden_states[layer_idx][0, target_pos, :].detach().cpu().float().numpy()
            return reps
        except Exception as e:
            print(f"  Error extracting rep for '{sentence}': {e}")
            return None


def extract_all_role_reps(model_name, nouns, model=None, tokenizer=None, device=None):
    """
    为所有名词在所有语法角色下提取表示
    返回: {noun: {role: {layer_idx: vector}}}
    """
    if model is None:
        model, tokenizer, device = load_model(model_name)
        own_model = True
    else:
        own_model = False
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    all_reps = {}
    for ni, noun in enumerate(nouns):
        if ni % 5 == 0:
            print(f"  Noun {ni}/{len(nouns)}: {noun}")
        
        sentences = make_sentences(noun)
        all_reps[noun] = {}
        
        for role, sentence in sentences.items():
            reps = extract_noun_rep_at_role(model, tokenizer, device, noun, sentence, model_info)
            if reps is not None:
                all_reps[noun][role] = reps
    
    if own_model:
        release_model(model)
        print("Model released.")
    
    return all_reps, n_layers, model_info.d_model


# ===== Exp1: 语法角色解码 =====
def run_exp1(model_name):
    """语法角色可解码性: 在不同层和不同子空间中, 能否区分语法角色?"""
    print(f"\n{'='*60}")
    print(f"Exp1: 语法角色解码 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 提取所有角色表示
    all_reps, _, d_model = extract_all_role_reps(
        model_name, TEST_NOUNS, model=model, tokenizer=tokenizer, device=device
    )
    
    # 同时提取80个概念的表示用于训练PCA(复用CCXXVI的概念集)
    REPRESENTATIVE_CONCEPTS = [
        "cat", "dog", "bird", "fish", "lion", "tiger", "eagle", "whale",
        "red", "blue", "green", "yellow", "white", "black", "purple", "orange",
        "happy", "sad", "angry", "fear", "love", "hate", "hope", "joy",
        "wood", "stone", "metal", "glass", "paper", "cloth", "plastic", "rubber",
        "rain", "snow", "wind", "storm", "sun", "cloud", "fog", "ice",
        "hand", "foot", "head", "eye", "heart", "brain", "blood", "bone",
        "bread", "rice", "meat", "fruit", "water", "milk", "salt", "sugar",
        "hammer", "knife", "sword", "wheel", "rope", "nail", "axe", "saw",
        "time", "space", "truth", "beauty", "justice", "freedom", "power", "knowledge",
        "king", "queen", "child", "mother", "father", "friend", "enemy", "teacher",
    ]
    
    # 提取80概念在中间层的表示, 用于训练PCA
    print("  Extracting 80 concepts for PCA training...")
    concept_reps = {}
    for ci, concept in enumerate(REPRESENTATIVE_CONCEPTS):
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        target_pos = input_ids.shape[1] - 1
        with torch.no_grad():
            try:
                outputs = model(input_ids, output_hidden_states=True)
                concept_reps[concept] = outputs.hidden_states[mid_layer+1][0, target_pos, :].detach().cpu().float().numpy()
            except Exception as e:
                print(f"  Error for {concept}: {e}")
    
    release_model(model)
    print("Model released.")
    
    # 训练PCA on 80概念
    concept_names = list(concept_reps.keys())
    concept_vecs = np.array([concept_reps[n] for n in concept_names])
    concept_centered = concept_vecs - concept_vecs.mean(axis=0)
    
    pca = PCA(n_components=50)
    pca.fit(concept_centered)
    
    # 分析: 在不同层中, 语法角色的可解码性
    roles = ["subject", "object", "modified", "isolated"]
    layer_results = {}
    
    # 采样关键层
    sample_layers = [0, mid_layer//3, mid_layer//2, mid_layer, 
                     mid_layer + mid_layer//3, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l <= n_layers]))
    
    for layer_idx in sample_layers:
        # 收集该层所有名词×角色的表示
        X = []  # 表示向量
        y_role = []  # 语法角色标签
        y_noun = []  # 名词标签
        
        for noun in TEST_NOUNS:
            if noun not in all_reps:
                continue
            for role in roles:
                if role in all_reps[noun]:
                    vec = all_reps[noun][role][layer_idx]
                    X.append(vec)
                    y_role.append(role)
                    y_noun.append(noun)
        
        if len(X) < 10:
            continue
        
        X = np.array(X)
        X_centered = X - X.mean(axis=0)
        
        # 1. 在全空间中解码语法角色
        pca_full = PCA(n_components=min(30, X.shape[0]-1, X.shape[1]))
        X_pca = pca_full.fit_transform(X_centered)
        
        # 留一法交叉验证(按名词分组)
        # 简化: 直接用训练准确率 + 交叉验证
        try:
            clf_role = LogisticRegression(max_iter=1000, C=1.0)
            cv_scores = cross_val_score(clf_role, X_pca, y_role, cv=min(5, len(set(y_noun))), 
                                        scoring='accuracy')
            role_acc_cv = float(cv_scores.mean())
        except:
            role_acc_cv = -1.0
        
        clf_role.fit(X_pca, y_role)
        role_acc_train = float(clf_role.score(X_pca, y_role))
        
        # 2. 在V_sem(Top-5 PCs)中解码语法角色
        X_sem = pca_full.transform(X_centered)[:, :5]  # Top-5 PCs
        try:
            clf_sem = LogisticRegression(max_iter=1000, C=1.0)
            cv_scores_sem = cross_val_score(clf_sem, X_sem, y_role, cv=min(5, len(set(y_noun))),
                                            scoring='accuracy')
            role_acc_sem_cv = float(cv_scores_sem.mean())
        except:
            role_acc_sem_cv = -1.0
        
        clf_sem.fit(X_sem, y_role)
        role_acc_sem = float(clf_sem.score(X_sem, y_role))
        
        # 3. 在V_nonsem(PCs 6-50)中解码语法角色
        X_nonsem = pca_full.transform(X_centered)[:, 5:50]
        try:
            clf_nonsem = LogisticRegression(max_iter=1000, C=1.0)
            cv_scores_nonsem = cross_val_score(clf_nonsem, X_nonsem, y_role, cv=min(5, len(set(y_noun))),
                                               scoring='accuracy')
            role_acc_nonsem_cv = float(cv_scores_nonsem.mean())
        except:
            role_acc_nonsem_cv = -1.0
        
        clf_nonsem.fit(X_nonsem, y_role)
        role_acc_nonsem = float(clf_nonsem.score(X_nonsem, y_role))
        
        # 4. 计算角色间距离矩阵
        role_centers = {}
        for role in roles:
            mask = np.array([r == role for r in y_role])
            if mask.sum() > 0:
                role_centers[role] = X_centered[mask].mean(axis=0)
        
        role_dist_matrix = {}
        for r1 in roles:
            for r2 in roles:
                if r1 in role_centers and r2 in role_centers:
                    dist = np.linalg.norm(role_centers[r1] - role_centers[r2])
                    cos = float(np.dot(role_centers[r1], role_centers[r2]) / 
                               (np.linalg.norm(role_centers[r1]) * np.linalg.norm(role_centers[r2]) + 1e-10))
                    role_dist_matrix[f"{r1}-{r2}"] = {"dist": float(dist), "cos": cos}
        
        # 5. 语法角色在V_sem中的分布
        X_sem_all = pca_full.transform(X_centered)[:, :5]
        sem_by_role = {}
        for role in roles:
            mask = np.array([r == role for r in y_role])
            if mask.sum() > 0:
                sem_by_role[role] = {
                    "pc0_mean": float(X_sem_all[mask, 0].mean()),
                    "pc1_mean": float(X_sem_all[mask, 1].mean()),
                    "pc2_mean": float(X_sem_all[mask, 2].mean()),
                    "pc0_std": float(X_sem_all[mask, 0].std()),
                    "pc1_std": float(X_sem_all[mask, 1].std()),
                }
        
        layer_results[layer_idx] = {
            "n_samples": len(X),
            "role_acc_full_train": role_acc_train,
            "role_acc_full_cv": role_acc_cv,
            "role_acc_sem_train": role_acc_sem,
            "role_acc_sem_cv": role_acc_sem_cv,
            "role_acc_nonsem_train": role_acc_nonsem,
            "role_acc_nonsem_cv": role_acc_nonsem_cv,
            "role_dist_matrix": role_dist_matrix,
            "sem_by_role": sem_by_role,
        }
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    Full space:  train={role_acc_train:.3f}, cv={role_acc_cv:.3f}")
        print(f"    V_sem(5d):   train={role_acc_sem:.3f}, cv={role_acc_sem_cv:.3f}")
        print(f"    V_nonsem:    train={role_acc_nonsem:.3f}, cv={role_acc_nonsem_cv:.3f}")
        print(f"    Role centers in V_sem:")
        for role in roles:
            if role in sem_by_role:
                s = sem_by_role[role]
                print(f"      {role:10s}: PC0={s['pc0_mean']:+.3f}, PC1={s['pc1_mean']:+.3f}, PC2={s['pc2_mean']:+.3f}")
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "n_nouns": len(TEST_NOUNS),
        "roles": roles,
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    out_path = TEMP / f"ccxxviii_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


# ===== Exp2: 语法变换的线性性 =====
def run_exp2(model_name):
    """语法变换是否跨概念一致? Δ_role = rep(role) - rep(isolated) 是否有统一方向?"""
    print(f"\n{'='*60}")
    print(f"Exp2: 语法变换线性性 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 提取所有角色表示
    all_reps, _, d_model = extract_all_role_reps(
        model_name, TEST_NOUNS, model=model, tokenizer=tokenizer, device=device
    )
    
    release_model(model)
    print("Model released.")
    
    # 计算每个名词的Δ_role = rep(role) - rep(isolated)
    roles = ["subject", "object", "modified"]
    deltas = {role: {} for role in roles}  # {role: {noun: {layer: delta_vec}}}
    
    for noun in TEST_NOUNS:
        if noun not in all_reps:
            continue
        if "isolated" not in all_reps[noun]:
            continue
        
        iso_reps = all_reps[noun]["isolated"]
        
        for role in roles:
            if role not in all_reps[noun]:
                continue
            deltas[role][noun] = {}
            for layer_idx in range(n_layers + 1):
                if layer_idx in all_reps[noun][role] and layer_idx in iso_reps:
                    deltas[role][noun][layer_idx] = (
                        all_reps[noun][role][layer_idx] - iso_reps[layer_idx]
                    )
    
    # 分析: 在关键层中, Δ_role是否跨概念一致
    sample_layers = [0, mid_layer//3, mid_layer//2, mid_layer, 
                     mid_layer + mid_layer//3, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l <= n_layers]))
    
    layer_results = {}
    
    for layer_idx in sample_layers:
        layer_data = {}
        
        for role in roles:
            # 收集该层所有名词的Δ
            delta_vecs = []
            delta_norms = []
            for noun in TEST_NOUNS:
                if noun in deltas[role] and layer_idx in deltas[role][noun]:
                    d = deltas[role][noun][layer_idx]
                    delta_vecs.append(d)
                    delta_norms.append(np.linalg.norm(d))
            
            if len(delta_vecs) < 3:
                continue
            
            delta_vecs = np.array(delta_vecs)
            delta_norms = np.array(delta_norms)
            
            # 1. Δ的平均范数
            mean_norm = float(delta_norms.mean())
            std_norm = float(delta_norms.std())
            
            # 2. Δ的方向一致性: 计算所有Δ两两之间的余弦相似度
            # 归一化
            norms = np.linalg.norm(delta_vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            delta_normalized = delta_vecs / norms
            
            # 平均方向
            mean_direction = delta_normalized.mean(axis=0)
            mean_dir_norm = np.linalg.norm(mean_direction)
            
            # 一致性指标: ||mean_direction|| (1.0=完全一致, 1/sqrt(n)=随机)
            consistency = float(mean_dir_norm)
            random_baseline = float(1.0 / np.sqrt(len(delta_vecs)))
            
            # 3. 用平均方向做线性拟合
            # rep(role) ≈ rep(isolated) + α * Δ_mean
            # 拟合误差
            if mean_dir_norm > 1e-10:
                mean_dir_unit = mean_direction / mean_dir_norm
                
                # 对每个名词, 计算投影和残差
                proj_lengths = []
                residuals = []
                for i, noun in enumerate([n for n in TEST_NOUNS if n in deltas[role] and layer_idx in deltas[role][n]]):
                    d = delta_vecs[i]
                    proj = np.dot(d, mean_dir_unit)
                    residual = d - proj * mean_dir_unit
                    proj_lengths.append(proj)
                    residuals.append(np.linalg.norm(residual))
                
                mean_proj = float(np.mean(proj_lengths))
                mean_residual = float(np.mean(residuals))
                residual_ratio = mean_residual / (mean_proj + 1e-10)
            else:
                mean_proj = 0.0
                mean_residual = float(mean_norm)
                residual_ratio = float('inf')
            
            # 4. Δ在V_sem vs V_nonsem中的能量
            # 需要PCA - 用全部Δ来训练
            pca_delta = PCA(n_components=min(20, len(delta_vecs)-1, delta_vecs.shape[1]))
            pca_delta.fit(delta_vecs)
            
            # 前5个分量的方差
            var_top5 = float(pca_delta.explained_variance_ratio_[:5].sum()) if len(pca_delta.explained_variance_ratio_) >= 5 else float(pca_delta.explained_variance_ratio_.sum())
            
            layer_data[role] = {
                "mean_norm": mean_norm,
                "std_norm": std_norm,
                "consistency": consistency,
                "random_baseline": random_baseline,
                "consistency_vs_random": consistency / random_baseline if random_baseline > 0 else 0,
                "mean_proj_on_mean_dir": mean_proj,
                "mean_residual": mean_residual,
                "residual_ratio": residual_ratio,
                "delta_var_top5": var_top5,
                "delta_var_top1": float(pca_delta.explained_variance_ratio_[0]) if len(pca_delta.explained_variance_ratio_) > 0 else 0,
            }
            
            print(f"\n  Layer {layer_idx}, Role={role}:")
            print(f"    Mean Δ norm: {mean_norm:.4f} ± {std_norm:.4f}")
            print(f"    Direction consistency: {consistency:.4f} (random={random_baseline:.4f}, ratio={consistency/random_baseline:.2f})")
            print(f"    Linear fit: proj={mean_proj:.4f}, residual={mean_residual:.4f}, ratio={residual_ratio:.4f}")
            print(f"    Δ variance: top1={pca_delta.explained_variance_ratio_[0]:.4f}, top5={var_top5:.4f}")
        
        # 5. 跨角色对比: 不同角色的Δ是否正交?
        cross_role = {}
        for r1 in roles:
            for r2 in roles:
                if r1 >= r2:
                    continue
                if r1 not in layer_data or r2 not in layer_data:
                    continue
                
                # 计算两种角色的平均Δ的余弦相似度
                d1_vecs = []
                d2_vecs = []
                for noun in TEST_NOUNS:
                    if noun in deltas[r1] and layer_idx in deltas[r1][noun] and \
                       noun in deltas[r2] and layer_idx in deltas[r2][noun]:
                        d1_vecs.append(deltas[r1][noun][layer_idx])
                        d2_vecs.append(deltas[r2][noun][layer_idx])
                
                if len(d1_vecs) < 3:
                    continue
                
                d1_mean = np.mean(d1_vecs, axis=0)
                d2_mean = np.mean(d2_vecs, axis=0)
                
                cos = float(np.dot(d1_mean, d2_mean) / 
                           (np.linalg.norm(d1_mean) * np.linalg.norm(d2_mean) + 1e-10))
                angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                
                cross_role[f"{r1}_vs_{r2}"] = {"cos": cos, "angle": angle}
                print(f"    {r1} vs {r2}: cos={cos:.4f}, angle={angle:.1f}°")
        
        layer_data["cross_role"] = cross_role
        layer_results[layer_idx] = layer_data
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "n_nouns": len(TEST_NOUNS),
        "roles": roles,
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    out_path = TEMP / f"ccxxviii_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


# ===== Exp3: 语法变换在V_sem中的群结构 =====
def run_exp3(model_name):
    """
    验证语法变换是否构成群:
    - 同一概念: subject→object 的变换 = Δ_obj - Δ_sub
    - 这个变换是否跨概念一致?
    - 是否满足某种群性质?
    """
    print(f"\n{'='*60}")
    print(f"Exp3: 语法变换群结构 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 提取所有角色表示
    all_reps, _, d_model = extract_all_role_reps(
        model_name, TEST_NOUNS, model=model, tokenizer=tokenizer, device=device
    )
    
    # 额外: 提取"双重角色"的句子
    # subject+modified: "The red cat runs"
    # object+modified: "Sees the red cat"
    double_reps = {}
    print("\n  Extracting double-role representations...")
    for noun in TEST_NOUNS[:10]:  # 只用10个名词(节省时间)
        double_sentences = {
            "subj_mod": f"The red {noun} runs",
            "obj_mod": f"Sees the red {noun}",
            "subj_verb2": f"The {noun} eats and runs",
        }
        double_reps[noun] = {}
        for role_key, sentence in double_sentences.items():
            reps = extract_noun_rep_at_role(model, tokenizer, device, noun, sentence, model_info)
            if reps is not None:
                double_reps[noun][role_key] = reps
    
    release_model(model)
    print("Model released.")
    
    # 分析中间层的群结构
    roles = ["subject", "object", "modified"]
    
    # 1. 基本变换: T_sub = Δ_sub, T_obj = Δ_obj, T_mod = Δ_mod
    # 2. 组合变换: T_sub+mod ≈ Δ_subj_mod, T_obj+mod ≈ Δ_obj_mod
    # 3. 群性质: T_sub+mod ≈ T_sub + T_mod? (加法群)
    
    mid_results = {}
    
    for layer_idx in [mid_layer]:
        print(f"\n  === Layer {layer_idx} (mid) ===")
        
        # 收集变换向量
        transforms = {role: [] for role in roles}
        double_transforms = {"subj_mod": [], "obj_mod": []}
        
        for noun in TEST_NOUNS:
            if noun not in all_reps or "isolated" not in all_reps[noun]:
                continue
            iso = all_reps[noun]["isolated"].get(layer_idx)
            if iso is None:
                continue
            
            for role in roles:
                if role in all_reps[noun] and layer_idx in all_reps[noun][role]:
                    transforms[role].append(all_reps[noun][role][layer_idx] - iso)
            
            # 双重角色
            if noun in double_reps:
                for drole in ["subj_mod", "obj_mod"]:
                    if drole in double_reps[noun] and layer_idx in double_reps[noun][drole]:
                        double_transforms[drole].append(double_reps[noun][drole][layer_idx] - iso)
        
        # 计算平均变换
        mean_transforms = {}
        for role in roles:
            if len(transforms[role]) > 0:
                mean_transforms[role] = np.mean(transforms[role], axis=0)
        
        mean_double = {}
        for drole in ["subj_mod", "obj_mod"]:
            if len(double_transforms[drole]) > 0:
                mean_double[drole] = np.mean(double_transforms[drole], axis=0)
        
        # 3.1 验证: T_sub+mod ≈ T_sub + T_mod?
        results_31 = {}
        if "subject" in mean_transforms and "modified" in mean_transforms and "subj_mod" in mean_double:
            T_sub = mean_transforms["subject"]
            T_mod = mean_transforms["modified"]
            T_sub_mod_actual = mean_double["subj_mod"]
            T_sub_mod_predicted = T_sub + T_mod
            
            cos_pred_actual = float(np.dot(T_sub_mod_predicted, T_sub_mod_actual) / 
                                    (np.linalg.norm(T_sub_mod_predicted) * np.linalg.norm(T_sub_mod_actual) + 1e-10))
            norm_pred = float(np.linalg.norm(T_sub_mod_predicted))
            norm_actual = float(np.linalg.norm(T_sub_mod_actual))
            residual = float(np.linalg.norm(T_sub_mod_predicted - T_sub_mod_actual))
            
            results_31 = {
                "cos_predicted_vs_actual": cos_pred_actual,
                "norm_predicted": norm_pred,
                "norm_actual": norm_actual,
                "residual_norm": residual,
                "residual_ratio": residual / (norm_actual + 1e-10),
            }
            print(f"\n  T_sub+mod = T_sub + T_mod?")
            print(f"    cos(predicted, actual) = {cos_pred_actual:.4f}")
            print(f"    ||predicted|| = {norm_pred:.4f}, ||actual|| = {norm_actual:.4f}")
            print(f"    residual = {residual:.4f} (ratio={residual/norm_actual:.4f})")
        
        # 3.2 T_obj+mod ≈ T_obj + T_mod?
        results_32 = {}
        if "object" in mean_transforms and "modified" in mean_transforms and "obj_mod" in mean_double:
            T_obj = mean_transforms["object"]
            T_mod = mean_transforms["modified"]
            T_obj_mod_actual = mean_double["obj_mod"]
            T_obj_mod_predicted = T_obj + T_mod
            
            cos_pred_actual = float(np.dot(T_obj_mod_predicted, T_obj_mod_actual) / 
                                    (np.linalg.norm(T_obj_mod_predicted) * np.linalg.norm(T_obj_mod_actual) + 1e-10))
            norm_pred = float(np.linalg.norm(T_obj_mod_predicted))
            norm_actual = float(np.linalg.norm(T_obj_mod_actual))
            residual = float(np.linalg.norm(T_obj_mod_predicted - T_obj_mod_actual))
            
            results_32 = {
                "cos_predicted_vs_actual": cos_pred_actual,
                "norm_predicted": norm_pred,
                "norm_actual": norm_actual,
                "residual_norm": residual,
                "residual_ratio": residual / (norm_actual + 1e-10),
            }
            print(f"\n  T_obj+mod = T_obj + T_mod?")
            print(f"    cos(predicted, actual) = {cos_pred_actual:.4f}")
            print(f"    ||predicted|| = {norm_pred:.4f}, ||actual|| = {norm_actual:.4f}")
            print(f"    residual = {residual:.4f} (ratio={residual/norm_actual:.4f})")
        
        # 3.3 变换之间的正交性
        ortho_results = {}
        role_pairs = [("subject", "object"), ("subject", "modified"), ("object", "modified")]
        for r1, r2 in role_pairs:
            if r1 in mean_transforms and r2 in mean_transforms:
                cos = float(np.dot(mean_transforms[r1], mean_transforms[r2]) / 
                           (np.linalg.norm(mean_transforms[r1]) * np.linalg.norm(mean_transforms[r2]) + 1e-10))
                angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                ortho_results[f"{r1}_vs_{r2}"] = {"cos": cos, "angle": angle}
                print(f"\n  T_{r1} vs T_{r2}: cos={cos:.4f}, angle={angle:.1f}°")
        
        # 3.4 逐名词验证加法性
        additive_per_noun = []
        for noun in TEST_NOUNS[:10]:
            if noun not in all_reps or "isolated" not in all_reps[noun]:
                continue
            iso = all_reps[noun]["isolated"].get(layer_idx)
            if iso is None:
                continue
            
            d_sub = all_reps[noun].get("subject", {}).get(layer_idx)
            d_mod = all_reps[noun].get("modified", {}).get(layer_idx)
            d_sub_mod = double_reps.get(noun, {}).get("subj_mod", {}).get(layer_idx)
            
            if d_sub is None or d_mod is None or d_sub_mod is None:
                continue
            
            T_sub = d_sub - iso
            T_mod = d_mod - iso
            T_sub_mod = d_sub_mod - iso
            
            predicted = T_sub + T_mod
            cos_pa = float(np.dot(predicted, T_sub_mod) / 
                          (np.linalg.norm(predicted) * np.linalg.norm(T_sub_mod) + 1e-10))
            residual = float(np.linalg.norm(predicted - T_sub_mod))
            
            additive_per_noun.append({
                "noun": noun,
                "cos_predicted_actual": cos_pa,
                "residual_norm": residual,
                "actual_norm": float(np.linalg.norm(T_sub_mod)),
            })
        
        if additive_per_noun:
            mean_cos = float(np.mean([x["cos_predicted_actual"] for x in additive_per_noun]))
            mean_residual_ratio = float(np.mean([x["residual_norm"] / (x["actual_norm"] + 1e-10) for x in additive_per_noun]))
            print(f"\n  Per-noun additivity:")
            print(f"    Mean cos(predicted, actual) = {mean_cos:.4f}")
            print(f"    Mean residual/actual = {mean_residual_ratio:.4f}")
            for x in additive_per_noun:
                print(f"      {x['noun']:10s}: cos={x['cos_predicted_actual']:.4f}, residual_ratio={x['residual_norm']/(x['actual_norm']+1e-10):.4f}")
        
        mid_results = {
            "additivity_sub_mod": results_31,
            "additivity_obj_mod": results_32,
            "orthogonality": ortho_results,
            "additive_per_noun": additive_per_noun,
            "mean_additive_cos": float(np.mean([x["cos_predicted_actual"] for x in additive_per_noun])) if additive_per_noun else None,
            "mean_residual_ratio": float(np.mean([x["residual_norm"] / (x["actual_norm"] + 1e-10) for x in additive_per_noun])) if additive_per_noun else None,
        }
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "n_nouns": len(TEST_NOUNS),
        "mid_layer_results": mid_results,
    }
    
    out_path = TEMP / f"ccxxviii_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


# ===== 主函数 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    if args.exp == 1:
        run_exp1(args.model)
    elif args.exp == 2:
        run_exp2(args.model)
    elif args.exp == 3:
        run_exp3(args.model)
