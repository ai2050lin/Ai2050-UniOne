"""
CCXXXII(383): 位置控制的角色解析轨迹 — 角色信息何时真正被解析？

核心问题:
  CCXXXI Exp1的CV=1.000从L1就开始, 但这是位置混淆(主语=pos1, 宾语=pos4)
  需要在同一位置比较主语vs宾语, 才能追踪角色信息的真正解析时机

实验设计:
  Exp1: 2×2因子设计逐层追踪
    - 使用关系从句构造4种条件(与CCXXX Exp2相同)
    - Subj+Early: "The cat that sees the dog runs"  → cat@pos1, subject
    - Subj+Late:  "The dog that the cat sees runs"  → cat@pos4, subject
    - Obj+Early:  "The cat that the dog sees runs"  → cat@pos1, object
    - Obj+Late:   "The dog sees the cat that runs"   → cat@pos4, object
    - 纯角色效应: δ_role = (se+sl-oe-ol)/2 → 控制位置后
    - 逐层追踪: L0→L_max的||δ_role||, V_sem%, CV_acc变化
  
  Exp2: V_sem^⊥中的最优V_syn提取
    - CCXXXI发现V_sem^⊥(5d) CV=1.0, 但V_syn_delta(5d) CV=0.71-0.79
    - 在V_sem^⊥中直接做PCA, 找最优5维V_syn
    - 逐层追踪V_syn的角色解码能力
  
  Exp3: 角色变换的名词依赖性
    - 每个名词的δ_role方向是否相同?
    - 在V_syn中, 不同名词的主语→宾语变换是否平行?
    - 这决定了V_syn的结构: 统一方向 vs 名词特定方向
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

NOUN_PAIRS = [
    ("cat", "dog"), ("bird", "fish"), ("lion", "tiger"), ("eagle", "whale"), ("horse", "wolf"),
    ("king", "queen"), ("mother", "child"), ("friend", "enemy"), ("teacher", "student"), ("doctor", "patient"),
    ("hammer", "knife"), ("sword", "wheel"), ("rope", "nail"), ("stone", "glass"), ("wood", "metal"),
    ("rain", "snow"), ("wind", "storm"), ("sun", "moon"), ("fire", "water"), ("mountain", "river"),
]

TRANSITIVE_VERBS = ["chases", "sees", "finds", "takes", "watches"]

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


def find_noun_position(tokenizer, sentence, noun):
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    for prefix in ['', ' ']:
        noun_tokens = tokenizer(prefix + noun, add_special_tokens=False)['input_ids']
        for i in range(len(input_ids) - len(noun_tokens) + 1):
            if input_ids[i:i+len(noun_tokens)] == noun_tokens:
                return i + len(noun_tokens) - 1
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip()
        if decoded == noun:
            return i
    return None


# ============================================================
# Exp1: 2×2因子设计逐层追踪
# ============================================================
def run_exp1(model_name):
    print(f"\n{'='*60}")
    print(f"CCXXXII Exp1: Position-Controlled Role Resolution — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 训练PCA(用中间层)
    mid_layer = n_layers // 2
    concept_reps = []
    for concept in REPRESENTATIVE_CONCEPTS:
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            last_pos = inputs['input_ids'].shape[1] - 1
            rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
            concept_reps.append(rep)
    
    X_pca = np.array(concept_reps)
    pca = PCA(n_components=50)
    pca.fit(X_pca)
    pc_axes = pca.components_[:5]
    
    # 收集4种条件的逐层数据
    conditions = ["subj_early", "subj_late", "obj_early", "obj_late"]
    
    # 逐层收集
    layer_data = {f"L{l}": {c: [] for c in conditions} for l in range(n_layers)}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for verb in TRANSITIVE_VERBS[:3]:
            for noun, other in [(noun_a, noun_b), (noun_b, noun_a)]:
                sent_se = f"The {noun} that {verb} the {other} runs"
                sent_sl = f"The {other} that the {noun} {verb} runs"
                sent_oe = f"The {noun} that the {other} {verb} runs"
                sent_ol = f"The {other} {verb} the {noun} that runs"
                
                sentences = {
                    "subj_early": sent_se,
                    "subj_late": sent_sl,
                    "obj_early": sent_oe,
                    "obj_late": sent_ol,
                }
                
                for cond_name, sentence in sentences.items():
                    noun_pos = find_noun_position(tokenizer, sentence, noun)
                    if noun_pos is None:
                        continue
                    
                    inputs = tokenizer(sentence, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            outputs = model(inputs['input_ids'], output_hidden_states=True)
                            for l in range(n_layers):
                                rep = outputs.hidden_states[l][0, noun_pos, :].detach().cpu().float().numpy()
                                layer_data[f"L{l}"][cond_name].append({
                                    "noun": noun,
                                    "verb": verb,
                                    "rep": rep,
                                })
                        except Exception as e:
                            pass
    
    # 逐层分析2×2因子
    results = {"model": model_name, "exp": 1, "n_layers": n_layers}
    
    print("\n--- Position-Controlled Role Resolution ---")
    print(f"  {'Layer':>5} | {'Role CV':>7} | {'||δ_role||':>10} | {'||δ_pos||':>10} | {'V_sem%':>6} | {'Consist':>8}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")
    
    # 采样层以节省计算
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    trajectory = []
    
    for l in sample_layers:
        key = f"L{l}"
        data = layer_data[key]
        
        # 找有效样本(4个条件都有)
        valid_samples = []
        noun_set = set()
        for noun in [n for pair in NOUN_PAIRS for n in pair]:
            for verb in TRANSITIVE_VERBS[:3]:
                reps = {}
                for c in conditions:
                    matches = [d for d in data[c] if d["noun"] == noun and d["verb"] == verb]
                    if matches:
                        reps[c] = matches[0]["rep"]
                
                if len(reps) == 4:
                    valid_samples.append(reps)
                    noun_set.add(noun)
        
        n_valid = len(valid_samples)
        if n_valid < 10:
            continue
        
        # 2×2因子分析
        role_deltas = []
        pos_deltas = []
        
        for reps in valid_samples:
            se = reps["subj_early"]
            sl = reps["subj_late"]
            oe = reps["obj_early"]
            ol = reps["obj_late"]
            
            # 纯角色效应: (se + sl - oe - ol) / 2
            d_role = (se + sl - oe - ol) / 2
            # 纯位置效应: (se + oe - sl - ol) / 2
            d_pos = (se + oe - sl - ol) / 2
            
            role_deltas.append(d_role)
            pos_deltas.append(d_pos)
        
        role_deltas = np.array(role_deltas)
        pos_deltas = np.array(pos_deltas)
        
        # 角色分类
        # 构造分类数据: subj reps vs obj reps (控制位置)
        X_role = []
        y_role = []
        for reps in valid_samples:
            X_role.append(reps["subj_early"])  # subj
            X_role.append(reps["obj_early"])    # obj
            y_role.append(0)
            y_role.append(1)
        
        X_role = np.array(X_role)
        y_role = np.array(y_role)
        
        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv_folds = min(5, n_valid // 5)
        if cv_folds >= 2:
            cv_scores = cross_val_score(clf, X_role, y_role, cv=cv_folds, scoring='accuracy')
            role_cv = cv_scores.mean()
        else:
            role_cv = 0.5
        
        # δ_role统计
        mean_norm_role = float(np.mean(np.linalg.norm(role_deltas, axis=1)))
        mean_norm_pos = float(np.mean(np.linalg.norm(pos_deltas, axis=1)))
        
        mean_d = role_deltas.mean(axis=0)
        consistency = np.mean([np.dot(d, mean_d) / (np.linalg.norm(d) * np.linalg.norm(mean_d) + 1e-10) 
                              for d in role_deltas])
        
        vsem = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                       for d in role_deltas])
        
        entry = {
            "layer": l, "role_cv": float(role_cv), "n_valid": n_valid,
            "mean_norm_role": mean_norm_role, "mean_norm_pos": mean_norm_pos,
            "consistency": float(consistency), "vsem_energy": float(vsem),
        }
        trajectory.append(entry)
        
        print(f"  L{l:>4} | {role_cv:>7.3f} | {mean_norm_role:>10.2f} | {mean_norm_pos:>10.2f} | {vsem*100:>5.1f}% | {consistency:>8.3f}")
    
    results["trajectory"] = trajectory
    
    # 找角色解析层(位置控制后)
    for entry in trajectory:
        if entry["role_cv"] > 0.65 and entry["mean_norm_role"] > 0.5:
            results["resolution_layer"] = entry["layer"]
            results["resolution_cv"] = entry["role_cv"]
            print(f"\n  ★ Position-controlled role resolution: L{entry['layer']} (CV={entry['role_cv']:.3f})")
            break
    
    # 峰值
    if trajectory:
        best = max(trajectory, key=lambda x: x["role_cv"])
        results["peak_layer"] = best["layer"]
        results["peak_cv"] = best["role_cv"]
    
    # 三段分析
    early = [e for e in trajectory if e["layer"] < n_layers // 3]
    mid = [e for e in trajectory if n_layers // 3 <= e["layer"] < 2 * n_layers // 3]
    late = [e for e in trajectory if e["layer"] >= 2 * n_layers // 3]
    
    for name, seg in [("Early", early), ("Mid", mid), ("Late", late)]:
        if seg:
            print(f"  {name}: mean_cv={np.mean([e['role_cv'] for e in seg]):.3f}, "
                  f"mean_||δ_role||={np.mean([e['mean_norm_role'] for e in seg]):.2f}")
    
    out_path = TEMP / f"ccxxxii_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp2: V_sem^⊥中的最优V_syn提取
# ============================================================
def run_exp2(model_name):
    print(f"\n{'='*60}")
    print(f"CCXXXII Exp2: Optimal V_syn from V_sem^⊥ — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 收集语法角色数据
    print("Collecting syntactic data...")
    syn_reps = {"subj": [], "obj": []}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for verb in TRANSITIVE_VERBS:
            sent_subj = f"The {noun_a} {verb} the {noun_b}"
            sent_obj = f"The {noun_b} {verb} the {noun_a}"
            
            pos_subj = find_noun_position(tokenizer, sent_subj, noun_a)
            pos_obj = find_noun_position(tokenizer, sent_obj, noun_a)
            
            if pos_subj is None or pos_obj is None:
                continue
            
            for sentence, noun_pos, role in [
                (sent_subj, pos_subj, "subj"),
                (sent_obj, pos_obj, "obj"),
            ]:
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        outputs = model(inputs['input_ids'], output_hidden_states=True)
                        rep = outputs.hidden_states[mid_layer][0, noun_pos, :].detach().cpu().float().numpy()
                        syn_reps[role].append({"noun": noun_a, "verb": verb, "rep": rep})
                    except:
                        pass
    
    n_subj = len(syn_reps["subj"])
    n_obj = len(syn_reps["obj"])
    print(f"  Collected: {n_subj} subj, {n_obj} obj")
    
    # V_sem PCA
    concept_reps = []
    for concept in REPRESENTATIVE_CONCEPTS:
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            last_pos = inputs['input_ids'].shape[1] - 1
            rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
            concept_reps.append(rep)
    
    pca_sem = PCA(n_components=50)
    pca_sem.fit(np.array(concept_reps))
    
    # 计算V_sem^⊥的投影矩阵
    V_sem_5 = pca_sem.components_[:5]  # [5, d]
    d_model = V_sem_5.shape[1]
    
    # 正交补空间的基: I - V_sem^T (V_sem V_sem^T)^{-1} V_sem
    # 更简洁: 投影到V_sem^⊥ = I - V_sem^T V_sem (因为V_sem已正交归一化)
    proj_vsem = V_sem_5.T @ V_sem_5  # [d, d]
    proj_perp = np.eye(d_model) - proj_vsem  # 投影到V_sem^⊥的矩阵
    
    # 在V_sem^⊥中做PCA
    X_all = np.array([d["rep"] for d in syn_reps["subj"]] + [d["rep"] for d in syn_reps["obj"]])
    y_all = np.array([0] * n_subj + [1] * n_obj)
    
    # 投影到V_sem^⊥
    X_perp = (proj_perp @ X_all.T).T  # [N, d]
    
    # 在V_sem^⊥空间做PCA
    pca_perp = PCA(n_components=50)
    pca_perp.fit(X_perp)
    
    results = {"model": model_name, "exp": 2, "n_subj": n_subj, "n_obj": n_obj}
    
    # 对比不同子空间的角色解码
    print("\n--- Role Classification in Different Subspaces ---")
    
    subspaces = {}
    
    # 1. V_sem(5d)
    X_vsem = (V_sem_5 @ X_all.T).T
    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv_folds = min(5, min(n_subj, n_obj) // 5)
    cv_vsem = cross_val_score(clf, X_vsem, y_all, cv=cv_folds, scoring='accuracy').mean()
    subspaces["V_sem(5d)"] = float(cv_vsem)
    print(f"  V_sem(5d):          CV = {cv_vsem:.3f}")
    
    # 2. V_sem^⊥(5d) — 最优5维
    X_perp5 = (pca_perp.components_[:5] @ X_perp.T).T
    cv_perp5 = cross_val_score(clf, X_perp5, y_all, cv=cv_folds, scoring='accuracy').mean()
    subspaces["V_sem^⊥(5d)"] = float(cv_perp5)
    print(f"  V_sem^⊥(5d):        CV = {cv_perp5:.3f}")
    
    # 3. V_sem^⊥(10d)
    X_perp10 = (pca_perp.components_[:10] @ X_perp.T).T
    cv_perp10 = cross_val_score(clf, X_perp10, y_all, cv=cv_folds, scoring='accuracy').mean()
    subspaces["V_sem^⊥(10d)"] = float(cv_perp10)
    print(f"  V_sem^⊥(10d):       CV = {cv_perp10:.3f}")
    
    # 4. V_syn_delta(5d) — 用δ_role PCA
    deltas = []
    for d_subj in syn_reps["subj"]:
        for d_obj in syn_reps["obj"]:
            if d_subj["noun"] == d_obj["noun"] and d_subj["verb"] == d_obj["verb"]:
                deltas.append(d_subj["rep"] - d_obj["rep"])
    
    if deltas:
        deltas = np.array(deltas)
        pca_delta = PCA(n_components=min(20, len(deltas)-1))
        pca_delta.fit(deltas)
        
        X_delta5 = (pca_delta.components_[:5] @ X_all.T).T
        cv_delta5 = cross_val_score(clf, X_delta5, y_all, cv=cv_folds, scoring='accuracy').mean()
        subspaces["V_syn_delta(5d)"] = float(cv_delta5)
        print(f"  V_syn_delta(5d):    CV = {cv_delta5:.3f}")
        
        # 5. V_syn_perp(5d) — V_sem^⊥中最优的5维(=pca_perp[:5])
        # 6. V_syn_delta在V_sem^⊥中的投影
        V_delta_5 = pca_delta.components_[:5]
        V_delta_5_perp = (proj_perp @ V_delta_5.T).T  # 投影到V_sem^⊥
        # 正交化
        from scipy.linalg import orth
        V_delta_5_perp_orth = orth(V_delta_5_perp.T).T
        
        if V_delta_5_perp_orth.shape[0] >= 5:
            X_delta_perp5 = (V_delta_5_perp_orth[:5] @ X_all.T).T
            cv_delta_perp5 = cross_val_score(clf, X_delta_perp5, y_all, cv=cv_folds, scoring='accuracy').mean()
            subspaces["V_syn_delta^⊥(5d)"] = float(cv_delta_perp5)
            print(f"  V_syn_delta^⊥(5d):  CV = {cv_delta_perp5:.3f}")
        
        # 7. V_syn_perp vs V_syn_delta的对齐
        if V_delta_5_perp_orth.shape[0] >= 5:
            C_align = pca_perp.components_[:5] @ V_delta_5_perp_orth[:5].T
            svd_align = np.linalg.svd(C_align, compute_uv=False)
            results["vsyn_perp_vs_vsyn_delta_alignment"] = {
                "singular_values": svd_align.tolist(),
                "max_alignment": float(svd_align[0]),
            }
            print(f"  V_syn^⊥ vs V_syn_delta alignment: max={svd_align[0]:.4f}")
    
    results["subspaces"] = subspaces
    
    # PCA分析: V_sem^⊥中角色相关PC
    print("\n--- PC-wise Role Discriminability in V_sem^⊥ ---")
    
    pc_role_scores = []
    for pc_idx in range(min(20, pca_perp.components_.shape[0])):
        # 该PC上的值
        pc_vals = pca_perp.components_[pc_idx] @ X_perp.T
        
        subj_vals = pc_vals[:n_subj]
        obj_vals = pc_vals[n_subj:]
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(subj_vals) + np.var(obj_vals)) / 2)
        d_cohen = (np.mean(subj_vals) - np.mean(obj_vals)) / (pooled_std + 1e-10)
        
        # 单PC分类
        X_pc = pc_vals.reshape(-1, 1)
        cv_pc = cross_val_score(clf, X_pc, y_all, cv=cv_folds, scoring='accuracy').mean()
        
        pc_role_scores.append({
            "pc": pc_idx,
            "d_cohen": float(d_cohen),
            "cv_acc": float(cv_pc),
            "variance_pct": float(pca_perp.explained_variance_ratio_[pc_idx] * 100),
        })
        
        if pc_idx < 10:
            print(f"  PC{pc_idx}: d={d_cohen:.3f}, CV={cv_pc:.3f}, var={pca_perp.explained_variance_ratio_[pc_idx]*100:.2f}%")
    
    results["pc_role_scores"] = pc_role_scores
    
    # δ_role在V_sem^⊥中的能量
    if deltas is not None and len(deltas) > 0:
        # 投影δ_role到V_sem^⊥
        deltas_perp = (proj_perp @ deltas.T).T
        
        # δ_role在V_sem^⊥(5d)中的能量
        perp5_energy = np.mean([
            np.sum((pca_perp.components_[:5] @ d) ** 2) / (np.linalg.norm(d) ** 2 + 1e-10)
            for d in deltas_perp
        ])
        
        # δ_role在V_sem^⊥(20d)中的能量
        perp20_energy = np.mean([
            np.sum((pca_perp.components_[:20] @ d) ** 2) / (np.linalg.norm(d) ** 2 + 1e-10)
            for d in deltas_perp
        ])
        
        # δ_role在V_sem^⊥的全部能量(=1-V_sem能量)
        total_perp_energy = np.mean([
            np.linalg.norm(proj_perp @ d) ** 2 / (np.linalg.norm(d) ** 2 + 1e-10)
            for d in deltas
        ])
        
        results["delta_perp_energy"] = {
            "perp5d": float(perp5_energy),
            "perp20d": float(perp20_energy),
            "total_perp": float(total_perp_energy),
        }
        
        print(f"\n  δ_role in V_sem^⊥: 5d={perp5_energy*100:.1f}%, 20d={perp20_energy*100:.1f}%, total={total_perp_energy*100:.1f}%")
    
    out_path = TEMP / f"ccxxxii_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp3: 角色变换的名词依赖性
# ============================================================
def run_exp3(model_name):
    print(f"\n{'='*60}")
    print(f"CCXXXII Exp3: Noun-Dependent Role Transform — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 收集每个名词的δ_role
    print("Collecting per-noun δ_role...")
    noun_deltas = {}  # {noun: [δ_role1, δ_role2, ...]}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for verb in TRANSITIVE_VERBS:
            sent_subj = f"The {noun_a} {verb} the {noun_b}"
            sent_obj = f"The {noun_b} {verb} the {noun_a}"
            
            pos_subj = find_noun_position(tokenizer, sent_subj, noun_a)
            pos_obj = find_noun_position(tokenizer, sent_obj, noun_a)
            
            if pos_subj is None or pos_obj is None:
                continue
            
            inputs_subj = tokenizer(sent_subj, return_tensors='pt').to(device)
            inputs_obj = tokenizer(sent_obj, return_tensors='pt').to(device)
            
            with torch.no_grad():
                try:
                    out_subj = model(inputs_subj['input_ids'], output_hidden_states=True)
                    out_obj = model(inputs_obj['input_ids'], output_hidden_states=True)
                    
                    rep_subj = out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy()
                    rep_obj = out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy()
                    
                    delta = rep_subj - rep_obj
                    
                    if noun_a not in noun_deltas:
                        noun_deltas[noun_a] = []
                    noun_deltas[noun_a].append(delta)
                except:
                    pass
    
    # 计算每个名词的平均δ_role方向
    noun_mean_deltas = {}
    for noun, deltas in noun_deltas.items():
        noun_mean_deltas[noun] = np.mean(deltas, axis=0)
    
    results = {"model": model_name, "exp": 3, "n_nouns": len(noun_mean_deltas)}
    
    # 1. 名词间δ_role方向的平行性
    print("\n--- Cross-Noun δ_role Parallelism ---")
    
    nouns = list(noun_mean_deltas.keys())
    mean_deltas = np.array([noun_mean_deltas[n] for n in nouns])
    
    # 全局平均方向
    global_mean = mean_deltas.mean(axis=0)
    global_norm = np.linalg.norm(global_mean)
    
    if global_norm > 1e-10:
        # 每个名词的δ_role与全局方向的cos
        cos_to_global = [np.dot(d, global_mean) / (np.linalg.norm(d) * global_norm + 1e-10) 
                        for d in mean_deltas]
        
        mean_cos = np.mean(cos_to_global)
        
        # 名词间两两cos
        pairwise_cos = []
        for i in range(len(nouns)):
            for j in range(i+1, len(nouns)):
                cos_ij = np.dot(mean_deltas[i], mean_deltas[j]) / (
                    np.linalg.norm(mean_deltas[i]) * np.linalg.norm(mean_deltas[j]) + 1e-10)
                pairwise_cos.append(cos_ij)
        
        mean_pairwise = np.mean(pairwise_cos)
        
        results["parallelism"] = {
            "mean_cos_to_global": float(mean_cos),
            "mean_pairwise_cos": float(mean_pairwise),
            "n_nouns": len(nouns),
        }
        
        print(f"  Mean cos(δ_noun, δ_global): {mean_cos:.3f}")
        print(f"  Mean pairwise cos(δ_i, δ_j): {mean_pairwise:.3f}")
        
        # 按类别分析
        from collections import defaultdict
        categories = {
            "animals": ["cat", "dog", "bird", "fish", "lion", "tiger", "eagle", "whale", "horse", "wolf"],
            "tools": ["hammer", "knife", "sword", "wheel", "rope", "nail", "stone", "glass", "wood", "metal"],
            "social": ["king", "queen", "mother", "child", "friend", "enemy", "teacher", "student", "doctor", "patient"],
            "nature": ["rain", "snow", "wind", "storm", "sun", "moon", "fire", "water", "mountain", "river"],
        }
        
        cat_parallelism = {}
        for cat_name, cat_nouns in categories.items():
            cat_deltas = [noun_mean_deltas[n] for n in cat_nouns if n in noun_mean_deltas]
            if len(cat_deltas) >= 2:
                cat_mean = np.mean(cat_deltas, axis=0)
                cat_cos = [np.dot(d, cat_mean) / (np.linalg.norm(d) * np.linalg.norm(cat_mean) + 1e-10) 
                          for d in cat_deltas]
                cat_parallelism[cat_name] = {
                    "mean_cos": float(np.mean(cat_cos)),
                    "n_nouns": len(cat_deltas),
                }
                print(f"  {cat_name}: within-cat cos = {np.mean(cat_cos):.3f} ({len(cat_deltas)} nouns)")
        
        results["category_parallelism"] = cat_parallelism
    
    # 2. 投影到V_sem^⊥后的平行性
    print("\n--- Parallelism in V_sem^⊥ ---")
    
    # 训练V_sem PCA
    concept_reps = []
    for concept in REPRESENTATIVE_CONCEPTS:
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            last_pos = inputs['input_ids'].shape[1] - 1
            rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
            concept_reps.append(rep)
    
    pca_sem = PCA(n_components=50)
    pca_sem.fit(np.array(concept_reps))
    V_sem_5 = pca_sem.components_[:5]
    
    proj_perp = np.eye(V_sem_5.shape[1]) - V_sem_5.T @ V_sem_5
    
    # 投影到V_sem^⊥
    perp_deltas = {n: proj_perp @ d for n, d in noun_mean_deltas.items()}
    perp_mean = np.mean(list(perp_deltas.values()), axis=0)
    perp_norm = np.linalg.norm(perp_mean)
    
    if perp_norm > 1e-10:
        cos_perp_global = [np.dot(d, perp_mean) / (np.linalg.norm(d) * perp_norm + 1e-10) 
                          for d in perp_deltas.values()]
        
        pairwise_perp_cos = []
        perp_list = list(perp_deltas.values())
        for i in range(len(perp_list)):
            for j in range(i+1, len(perp_list)):
                cos_ij = np.dot(perp_list[i], perp_list[j]) / (
                    np.linalg.norm(perp_list[i]) * np.linalg.norm(perp_list[j]) + 1e-10)
                pairwise_perp_cos.append(cos_ij)
        
        results["parallelism_perp"] = {
            "mean_cos_to_global": float(np.mean(cos_perp_global)),
            "mean_pairwise_cos": float(np.mean(pairwise_perp_cos)),
        }
        
        print(f"  Mean cos(δ_noun^⊥, δ_global^⊥): {np.mean(cos_perp_global):.3f}")
        print(f"  Mean pairwise cos(δ_i^⊥, δ_j^⊥): {np.mean(pairwise_perp_cos):.3f}")
    
    # 3. δ_role的范数 vs 名词语义位置
    print("\n--- δ_role Norm vs Semantic Position ---")
    
    # 在V_sem中, 名词的语义位置
    noun_sem_pos = {}
    for noun in nouns:
        if noun in noun_mean_deltas:
            prompt = f"The word is {noun}"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(inputs['input_ids'], output_hidden_states=True)
                last_pos = inputs['input_ids'].shape[1] - 1
                rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
                noun_sem_pos[noun] = rep
    
    if noun_sem_pos:
        # 语义centroid
        sem_centroid = np.mean(list(noun_sem_pos.values()), axis=0)
        
        # δ_role范数 vs 距语义centroid距离
        delta_norms = [np.linalg.norm(noun_mean_deltas[n]) for n in nouns if n in noun_sem_pos]
        sem_dists = [np.linalg.norm(noun_sem_pos[n] - sem_centroid) for n in nouns if n in noun_sem_pos]
        
        if len(delta_norms) >= 5:
            corr = np.corrcoef(delta_norms, sem_dists)[0, 1]
            results["norm_vs_sem_dist"] = {
                "correlation": float(corr),
                "mean_delta_norm": float(np.mean(delta_norms)),
                "std_delta_norm": float(np.std(delta_norms)),
            }
            print(f"  Correlation(||δ_role||, distance_to_sem_centroid): {corr:.3f}")
            print(f"  ||δ_role||: mean={np.mean(delta_norms):.2f}, std={np.std(delta_norms):.2f}")
    
    out_path = TEMP / f"ccxxxii_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


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
