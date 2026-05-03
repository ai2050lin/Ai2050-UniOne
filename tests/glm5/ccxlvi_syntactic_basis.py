"""
CCXLVI(406): 完整语法基提取 — PC1-PC5的角色编码解析
======================================================

CCXLV关键发现:
  - d_head = PC1 (cos=0.82-0.90)
  - d_level ≠ PC2, 仅对齐PC6/10 (cos=0.13-0.21)
  - V_syn约3-5维, PC2-PC5尚未提取
  - 自然句一致性仅56-65%

本实验目标:
  Exp1: 从6种角色的大样本中提取PC1-PC5
    - 每个PC对齐什么语法特征?
    - 5维是否足以做自然句方向判断?

  Exp2: PC2-PC5的语义解释
    - 每个PC方向上, 不同角色的投影值排序
    - 与已知语法特征的对比(动词性/数/格/语态)

  Exp3: 完整5维语法基的自然句验证
    - 用(d_head, PC2, PC3, PC4, PC5)做自然句方向判断
    - 是否比单独d_head (56-65%)显著提升?

  Exp4: 语法基的跨层分析
    - PC1-PC5在不同层是否一致?
    - 语法信息的层间传递
"""

import sys, os, argparse, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
TEMP.mkdir(parents=True, exist_ok=True)

# ===== 大样本数据集 =====
WORDS = [
    "cat", "dog", "bird", "fish", "lion", "tiger", "king", "queen",
    "eagle", "whale", "horse", "wolf", "sword", "shield", "fire", "water",
    "mountain", "river", "sun", "moon", "hammer", "nail", "rope", "stone",
    "teacher", "student", "doctor", "patient", "mother", "child",
]

ROLE_TEMPLATES = {
    "subject": [
        "The {} chases the dog",
        "The {} sees the cat",
        "The {} hits the bird",
        "The {} follows the lion",
    ],
    "object": [
        "The cat chases the {}",
        "The dog sees the {}",
        "The bird hits the {}",
        "The lion follows the {}",
    ],
    "modifier": [
        "The red {} chases the dog",
        "The big {} sees the cat",
        "The old {} hits the bird",
        "The fast {} follows the lion",
    ],
    "indirect_obj": [
        "The king gives the {} the sword",
        "The queen gives the {} the crown",
        "The teacher gives the {} the book",
        "The doctor gives the {} the medicine",
    ],
    "prep_obj": [
        "The cat looks at the {}",
        "The dog runs to the {}",
        "The bird flies toward the {}",
        "The lion walks near the {}",
    ],
    "adverbial": [
        "The cat runs {}",
        "The dog jumps {}",
        "The bird flies {}",
        "The lion moves {}",
    ],
}

ADVERBIAL_WORDS = ["quickly", "slowly", "fast", "hard", "swiftly", "carefully"]

# ===== 复杂自然句 (从CCXLV复用) =====
COMPLEX_SENTENCES = [
    {"text": "The scientist who discovered the cure believes that the medicine will save millions",
     "pairs": [("scientist", "cure", "head"), ("medicine", "millions", "head"),
               ("cure", "scientist", "dep"), ("believes", "scientist", "dep")]},
    {"text": "The painting was stolen by the thief who escaped from the museum",
     "pairs": [("painting", "stolen", "dep"), ("thief", "escaped", "head"),
               ("museum", "escaped", "dep"), ("painting", "thief", "dep")]},
    {"text": "The teacher carefully gave the student a difficult assignment",
     "pairs": [("teacher", "gave", "dep"), ("student", "gave", "dep"),
               ("assignment", "gave", "dep"), ("teacher", "student", "head")]},
    {"text": "The cat and the dog chased the rabbit and the squirrel",
     "pairs": [("cat", "chased", "dep"), ("dog", "chased", "dep"),
               ("rabbit", "chased", "dep"), ("squirrel", "chased", "dep")]},
    {"text": "The king's advisor's daughter married the prince",
     "pairs": [("king", "advisor", "head"), ("advisor", "daughter", "head"),
               ("daughter", "married", "dep"), ("prince", "married", "dep")]},
    {"text": "The book on the table in the room by the window",
     "pairs": [("book", "table", "head"), ("table", "room", "head"),
               ("room", "window", "head"), ("window", "room", "dep")]},
    {"text": "The boy wanted to give the girl a flower",
     "pairs": [("boy", "wanted", "dep"), ("girl", "give", "dep"),
               ("flower", "give", "dep")]},
    {"text": "The elephant is bigger than the mouse",
     "pairs": [("elephant", "bigger", "dep"), ("mouse", "bigger", "dep"),
               ("elephant", "mouse", "head")]},
    {"text": "It was the king who gave the sword to the knight",
     "pairs": [("king", "gave", "head"), ("sword", "gave", "dep"),
               ("knight", "gave", "dep")]},
    {"text": "Although the storm was fierce the ship survived",
     "pairs": [("storm", "fierce", "dep"), ("ship", "survived", "dep"),
               ("storm", "ship", "dep")]},
    {"text": "The very tall extremely intelligent young scientist won",
     "pairs": [("scientist", "tall", "head"), ("scientist", "intelligent", "head"),
               ("scientist", "young", "head"), ("scientist", "won", "dep")]},
    {"text": "There is a dragon in the cave behind the mountain",
     "pairs": [("dragon", "cave", "dep"), ("cave", "mountain", "head"),
               ("mountain", "cave", "dep")]},
    {"text": "The queen herself prepared the feast for the guests",
     "pairs": [("queen", "prepared", "dep"), ("feast", "prepared", "dep"),
               ("guests", "prepared", "dep")]},
    {"text": "The warrior however decided to fight the dragon alone",
     "pairs": [("warrior", "decided", "dep"), ("dragon", "fight", "dep")]},
    {"text": "The hero decided to try to begin to fight the monster",
     "pairs": [("hero", "decided", "dep"), ("monster", "fight", "dep")]},
    {"text": "The cat that the dog that the boy chased bit ran away",
     "pairs": [("cat", "ran", "dep"), ("dog", "bit", "dep"),
               ("boy", "chased", "dep"), ("cat", "dog", "head")]},
]


def collect_role_representations(model, tokenizer, device, model_info, n_words=16, target_layer=None):
    """收集多种角色的大量表示"""
    d_model = model_info.d_model
    layers = get_layers(model)
    if target_layer is None:
        target_layer = len(layers) // 2
    
    role_reps = {role: [] for role in ROLE_TEMPLATES}
    
    captured = {}
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["h"] = output[0].detach().float().cpu()
            else:
                captured["h"] = output.detach().float().cpu()
        return hook
    
    hook = layers[target_layer].register_forward_hook(make_hook())
    
    for role, templates in ROLE_TEMPLATES.items():
        for word in WORDS[:n_words]:
            for tmpl in templates[:2]:  # 每词2个模板
                if role == "adverbial":
                    adv = ADVERBIAL_WORDS[hash(word) % len(ADVERBIAL_WORDS)]
                    prompt = tmpl.format(adv)
                    target = adv
                else:
                    prompt = tmpl.format(word)
                    target = word
                
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    try:
                        _ = model(**toks)
                    except:
                        continue
                
                if "h" in captured:
                    h = captured["h"][0].numpy()  # [seq_len, d_model]
                    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
                    
                    for i, t in enumerate(tokens):
                        if target.lower() in t.lower():
                            role_reps[role].append(h[i])
                            break
    
    hook.remove()
    
    # 统计
    total = sum(len(v) for v in role_reps.values())
    print(f"  Collected {total} samples: " + ", ".join(f"{r}={len(v)}" for r, v in role_reps.items()))
    
    return role_reps


def extract_syntactic_basis(role_reps, n_components=5):
    """从角色表示中提取语法基(PC1-PC5)"""
    # 合并所有表示
    all_reps = []
    all_labels = []
    role_to_idx = {}
    
    for role, reps_list in role_reps.items():
        if len(reps_list) < 2:
            continue
        role_to_idx[role] = len(role_to_idx)
        all_reps.extend(reps_list)
        all_labels.extend([role_to_idx[role]] * len(reps_list))
    
    X = np.array(all_reps)  # [n, d_model]
    y = np.array(all_labels)
    
    # PCA on角色中心
    centroids = {}
    for role, idx in role_to_idx.items():
        mask = y == idx
        if np.sum(mask) >= 2:
            centroids[role] = np.mean(X[mask], axis=0)
    
    C = np.array(list(centroids.values()))
    C_centered = C - np.mean(C, axis=0)
    
    pca = PCA(n_components=min(n_components, len(C_centered)))
    pca.fit(C_centered)
    
    components = pca.components_  # [n_comp, d_model]
    explained = pca.explained_variance_ratio_
    
    return components, explained, role_to_idx, X, y, centroids


def compute_role_projections(centroids, components):
    """计算每个角色中心在每个PC上的投影"""
    role_names = list(centroids.keys())
    C = np.array([centroids[r] for r in role_names])
    C_centered = C - np.mean(C, axis=0)
    
    projections = {}
    for i in range(len(components)):
        pc = components[i]
        proj_vals = {}
        for j, role in enumerate(role_names):
            proj_vals[role] = float(np.dot(C_centered[j], pc))
        projections[f"PC{i+1}"] = proj_vals
    
    return projections


# ===== Exp1: 提取PC1-PC5 =====
def run_exp1(model, tokenizer, device, model_info):
    """提取PC1-PC5语法基"""
    print("\n" + "="*70)
    print("Exp1: PC1-PC5语法基提取")
    print("="*70)
    
    # 收集大样本
    role_reps = collect_role_representations(
        model, tokenizer, device, model_info, n_words=16
    )
    
    # 提取语法基
    components, explained, role_to_idx, X, y, centroids = extract_syntactic_basis(
        role_reps, n_components=5
    )
    
    print(f"\n  PCA方差解释率:")
    cumvar = np.cumsum(explained)
    for i in range(len(explained)):
        print(f"    PC{i+1}: {explained[i]:.3f} (cum: {cumvar[i]:.3f})")
    
    # 计算角色投影
    projections = compute_role_projections(centroids, components)
    
    print(f"\n  角色在PC1-PC5上的投影值:")
    print(f"  {'Role':<15}", end="")
    for i in range(len(components)):
        print(f"  PC{i+1:>5}", end="")
    print()
    print("  " + "-" * (15 + 8 * len(components)))
    
    for role in projections["PC1"]:
        print(f"  {role:<15}", end="")
        for i in range(len(components)):
            val = projections[f"PC{i+1}"][role]
            print(f"  {val:>+7.1f}", end="")
        print()
    
    # 对比d_head (subject - object方向)
    subj_centroid = centroids.get("subject", np.zeros_like(list(centroids.values())[0]))
    obj_centroid = centroids.get("object", np.zeros_like(list(centroids.values())[0]))
    d_head_raw = subj_centroid - obj_centroid
    d_head_norm = d_head_raw / max(np.linalg.norm(d_head_raw), 1e-10)
    
    print(f"\n  d_head与PC1-PC5的对齐度:")
    for i in range(len(components)):
        cos = abs(np.dot(d_head_norm, components[i]))
        print(f"    |cos(d_head, PC{i+1})| = {cos:.3f}")
    
    # 交叉验证: 用k维做角色分类
    print(f"\n  角色分类准确率 (交叉验证):")
    
    # Binary: subject vs object
    subj_idx = role_to_idx.get("subject", -1)
    obj_idx = role_to_idx.get("object", -1)
    if subj_idx >= 0 and obj_idx >= 0:
        mask = (y == subj_idx) | (y == obj_idx)
        X_bin = X[mask]
        y_bin = y[mask]
        
        for n_dims in [1, 2, 3, 5]:
            pca_cv = PCA(n_components=min(n_dims, X_bin.shape[1]))
            X_pca = pca_cv.fit_transform(X_bin)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = min(3, len(X_bin) // 2)
            scores = cross_val_score(clf, X_pca, y_bin, cv=cv)
            print(f"    Binary(subj/obj) {n_dims}D: {np.mean(scores):.3f}")
    
    # Multi-role
    for n_dims in [1, 2, 3, 5, 10]:
        pca_cv = PCA(n_components=min(n_dims, X.shape[1]))
        X_pca = pca_cv.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv = min(3, len(X) // max(len(role_to_idx), 2))
        try:
            scores = cross_val_score(clf, X_pca, y, cv=cv)
            print(f"    Multi-role {n_dims}D: {np.mean(scores):.3f}")
        except:
            pass
    
    results = {
        "explained_variance": {f"PC{i+1}": float(explained[i]) for i in range(len(explained))},
        "cumulative_variance": {f"PC{i+1}": float(cumvar[i]) for i in range(len(cumvar))},
        "projections": projections,
        "d_head_alignment": {f"PC{i+1}": float(abs(np.dot(d_head_norm, components[i]))) 
                            for i in range(len(components))},
    }
    
    return results, components, centroids


# ===== Exp2: PC2-PC5语义解释 =====
def run_exp2(model, tokenizer, device, model_info, components, centroids):
    """分析PC2-PC5编码的语法特征"""
    print("\n" + "="*70)
    print("Exp2: PC2-PC5语义解释")
    print("="*70)
    
    d_model = model_info.d_model
    layers = get_layers(model)
    target_layer = len(layers) // 2
    
    results = {}
    
    # 对每个PC, 测试其对不同语法特征的区分能力
    # 特征1: 名词类 vs 修饰类 (subject/object/indirect/possessor vs modifier/adverbial)
    # 特征2: head vs dependent (subject/possessor vs object/prep_obj)
    # 特征3: 动词相关 vs 名词相关
    # 特征4: 格/角色 (subject vs indirect_obj vs prep_obj)
    
    # 定义语法特征对
    feature_pairs = {
        "noun_vs_modifier": (["subject", "object", "indirect_obj", "prep_obj"], ["modifier", "adverbial"]),
        "head_vs_dependent": (["subject", "indirect_obj"], ["object", "prep_obj"]),
        "argument_vs_adjunct": (["subject", "object", "indirect_obj"], ["modifier", "adverbial", "prep_obj"]),
        "core_vs_oblique": (["subject", "object", "indirect_obj"], ["prep_obj"]),
    }
    
    for pc_idx in range(min(5, len(components))):
        pc = components[pc_idx]
        pc_name = f"PC{pc_idx+1}"
        print(f"\n  --- {pc_name} ---")
        
        # 对每个语法特征, 计算两组中心的投影差
        for feat_name, (group1, group2) in feature_pairs.items():
            g1_centroids = [centroids[r] for r in group1 if r in centroids]
            g2_centroids = [centroids[r] for r in group2 if r in centroids]
            
            if len(g1_centroids) == 0 or len(g2_centroids) == 0:
                continue
            
            g1_mean = np.mean(g1_centroids, axis=0)
            g2_mean = np.mean(g2_centroids, axis=0)
            g1_centered = g1_mean - np.mean(list(centroids.values()), axis=0)
            g2_centered = g2_mean - np.mean(list(centroids.values()), axis=0)
            
            proj_g1 = np.dot(g1_centered, pc)
            proj_g2 = np.dot(g2_centered, pc)
            separation = abs(proj_g1 - proj_g2)
            
            print(f"    {feat_name}: g1={proj_g1:+.1f}, g2={proj_g2:+.1f}, sep={separation:.1f}")
            
            if pc_name not in results:
                results[pc_name] = {}
            results[pc_name][feat_name] = {
                "group1_proj": float(proj_g1),
                "group2_proj": float(proj_g2),
                "separation": float(separation),
            }
    
    # 综合判断: 每个PC最敏感的语法特征
    print(f"\n  ★ 综合判断:")
    for pc_name in results:
        best_feat = max(results[pc_name], key=lambda f: results[pc_name][f]["separation"])
        sep = results[pc_name][best_feat]["separation"]
        print(f"    {pc_name}: 最敏感特征={best_feat} (separation={sep:.1f})")
    
    return results


# ===== Exp3: 5维语法基的自然句验证 =====
def run_exp3(model, tokenizer, device, model_info, components, centroids):
    """用5维语法基做自然句方向判断"""
    print("\n" + "="*70)
    print("Exp3: 5维语法基的自然句验证")
    print("="*70)
    
    d_model = model_info.d_model
    layers = get_layers(model)
    target_layer = len(layers) // 2
    
    n_pcs = len(components)
    
    # 1维: d_head (PC1) — 基线
    # 2维: PC1+PC2
    # 5维: PC1-PC5
    # 方法: head词在head-direction上的投影应>dep词
    
    # 定义"head-direction": subject中心 - object中心 的方向
    subj_c = centroids.get("subject")
    obj_c = centroids.get("object")
    
    if subj_c is None or obj_c is None:
        print("  SKIP: Need subject and object centroids")
        return {"status": "skipped"}
    
    # 用LDA训练最佳分类方向 (比简单PCA更好)
    all_reps = []
    all_labels = []
    for role, reps_list in ROLE_TEMPLATES.items():
        # 需要重新收集或用之前的数据
        pass
    
    # 简化方法: 用角色中心在PC空间中的位置做模板匹配
    # head角色: subject, possessor (如果有的话)
    # dep角色: object, prep_obj
    
    # 计算所有角色中心在PC空间中的坐标
    all_centroid_vecs = list(centroids.values())
    all_role_names = list(centroids.keys())
    C = np.array(all_centroid_vecs)
    C_centered = C - np.mean(C, axis=0)
    
    # PC空间中的角色坐标
    role_coords = {}
    for i, role in enumerate(all_role_names):
        coords = []
        for pc in components:
            coords.append(float(np.dot(C_centered[i], pc)))
        role_coords[role] = coords
    
    print(f"  角色在PC1-{n_pcs}空间中的坐标:")
    for role, coords in role_coords.items():
        coord_str = ", ".join([f"{c:+.1f}" for c in coords])
        print(f"    {role}: ({coord_str})")
    
    # 定义head/dep方向: 在PC空间中, head角色(如subject)的坐标减去dep角色(如object)
    head_roles = ["subject", "indirect_obj"]
    dep_roles = ["object", "prep_obj"]
    
    # 在PC空间中计算head vs dep的最佳分离方向
    head_coords = np.array([role_coords[r] for r in head_roles if r in role_coords])
    dep_coords = np.array([role_coords[r] for r in dep_roles if r in role_coords])
    
    if len(head_coords) == 0 or len(dep_coords) == 0:
        print("  SKIP: Insufficient head/dep roles")
        return {"status": "skipped"}
    
    # head vs dep分类方向 (在PC空间中)
    head_mean = np.mean(head_coords, axis=0)
    dep_mean = np.mean(dep_coords, axis=0)
    head_dep_direction = head_mean - dep_mean
    hd_norm = np.linalg.norm(head_dep_direction)
    if hd_norm > 0:
        head_dep_direction = head_dep_direction / hd_norm
    
    print(f"\n  Head-dep分类方向(PC空间): {[f'{v:.3f}' for v in head_dep_direction]}")
    
    # 在自然句上测试
    results_by_dim = {}
    
    for n_use in [1, 2, 3, 5]:  # 用1/2/3/5个PC
        if n_use > n_pcs:
            continue
        
        # 截取前n_use个维度
        hd_dir_subset = head_dep_direction[:n_use]
        norm_sub = np.linalg.norm(hd_dir_subset)
        if norm_sub > 0:
            hd_dir_subset = hd_dir_subset / norm_sub
        
        correct = 0
        total = 0
        details = []
        
        for sent_data in COMPLEX_SENTENCES:
            text = sent_data["text"]
            pairs = sent_data["pairs"]
            
            target_tokens = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
            
            # 获取token表示
            captured = {}
            def make_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured["h"] = output[0].detach().float().cpu()
                    else:
                        captured["h"] = output.detach().float().cpu()
                return hook
            
            hook = layers[target_layer].register_forward_hook(make_hook())
            toks = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    _ = model(**toks)
                except:
                    hook.remove()
                    continue
            hook.remove()
            
            if "h" not in captured:
                continue
            
            h = captured["h"][0].numpy()
            tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
            mean_c = np.mean(list(centroids.values()), axis=0)
            
            # 获取每个目标token在PC空间中的坐标
            token_coords = {}
            for target in target_tokens:
                for i, t in enumerate(tokens):
                    if target.lower() in t.lower():
                        h_c = h[i] - mean_c
                        coords = [float(np.dot(h_c, pc)) for pc in components[:n_use]]
                        token_coords[target] = coords
                        break
            
            # 判断每对
            for w1, w2, direction in pairs:
                if w1 not in token_coords or w2 not in token_coords:
                    continue
                
                c1 = np.array(token_coords[w1][:n_use])
                c2 = np.array(token_coords[w2][:n_use])
                
                # 投影到head-dep方向
                proj1 = np.dot(c1, hd_dir_subset)
                proj2 = np.dot(c2, hd_dir_subset)
                
                if direction == "head":
                    is_correct = proj1 > proj2
                else:
                    is_correct = proj1 < proj2
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = correct / max(total, 1)
        results_by_dim[f"{n_use}D"] = {
            "correct": correct,
            "total": total,
            "accuracy": float(accuracy),
        }
        print(f"  {n_use}D PCA + head-dep方向: {correct}/{total} = {accuracy:.3f}")
    
    return results_by_dim


# ===== Exp4: 语法基跨层分析 =====
def run_exp4(model, tokenizer, device, model_info):
    """语法基在不同层的一致性"""
    print("\n" + "="*70)
    print("Exp4: 语法基跨层分析")
    print("="*70)
    
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 采样层
    n_layers = len(layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    # 对每个层提取PC1并检查与中间层PC1的对齐度
    layer_pc1s = {}
    
    for li in sample_layers:
        role_reps = collect_role_representations(
            model, tokenizer, device, model_info, n_words=8, target_layer=li
        )
        
        # 提取PC1
        all_reps = []
        for role, reps_list in role_reps.items():
            all_reps.extend(reps_list)
        
        if len(all_reps) < 10:
            continue
        
        X = np.array(all_reps)
        pca = PCA(n_components=1)
        pca.fit(X - np.mean(X, axis=0))
        pc1 = pca.components_[0]
        pc1 = pc1 / max(np.linalg.norm(pc1), 1e-10)
        
        layer_pc1s[li] = pc1
        print(f"  Layer {li}: PC1 norm={np.linalg.norm(pc1):.4f}")
    
    # 层间PC1对齐度
    print(f"\n  PC1跨层对齐度 (cos):")
    layer_list = sorted(layer_pc1s.keys())
    results = {"layer_alignment": {}}
    
    for i, li in enumerate(layer_list):
        for j, lj in enumerate(layer_list):
            if j <= i:
                continue
            cos = abs(np.dot(layer_pc1s[li], layer_pc1s[lj]))
            print(f"    L{li} vs L{lj}: {cos:.3f}")
            results["layer_alignment"][f"L{li}_L{lj}"] = float(cos)
    
    # 相邻层对齐度
    adjacent_cos = []
    for i in range(len(layer_list) - 1):
        li = layer_list[i]
        lj = layer_list[i + 1]
        cos = abs(np.dot(layer_pc1s[li], layer_pc1s[lj]))
        adjacent_cos.append(cos)
    
    if adjacent_cos:
        print(f"\n  相邻层PC1对齐度: mean={np.mean(adjacent_cos):.3f}, min={np.min(adjacent_cos):.3f}")
        results["adjacent_mean"] = float(np.mean(adjacent_cos))
        results["adjacent_min"] = float(np.min(adjacent_cos))
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    
    model_name = args.model
    exp = args.exp
    
    print(f"\n{'='*70}")
    print(f"CCXLVI: 完整语法基提取 — {model_name} — Exp{exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 缓存文件
    cache_path = TEMP / f"ccxlvi_basis_{model_name}.npz"
    
    if exp == 1:
        results, components, centroids = run_exp1(model, tokenizer, device, model_info)
        # 保存components和centroids供后续实验使用
        np.savez(cache_path, 
                 components=components,
                 centroids=np.array(list(centroids.values())),
                 centroid_roles=list(centroids.keys()))
        results_out = results
    
    elif exp in [2, 3]:
        # 加载之前保存的基
        if not cache_path.exists():
            print(f"  ERROR: Need to run Exp1 first! Cache not found: {cache_path}")
            release_model(model)
            return
        
        data = np.load(cache_path, allow_pickle=True)
        components = data['components']
        centroid_roles = list(data['centroid_roles'])
        centroid_vals = data['centroids']
        centroids = dict(zip(centroid_roles, centroid_vals))
        
        print(f"  Loaded cached basis: {components.shape[0]} PCs, {len(centroids)} roles")
        
        if exp == 2:
            results_out = run_exp2(model, tokenizer, device, model_info, components, centroids)
        else:
            results_out = run_exp3(model, tokenizer, device, model_info, components, centroids)
    
    elif exp == 4:
        results_out = run_exp4(model, tokenizer, device, model_info)
    
    # 保存结果
    out_path = TEMP / f"ccxlvi_exp{exp}_{model_name}_results.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_out = convert(results_out)
    results_out["model"] = model_name
    results_out["exp"] = exp
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"CCXLVI Exp{exp} ({model_name}) COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
