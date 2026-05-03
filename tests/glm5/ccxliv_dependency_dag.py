"""
CCXLIV(404): 依存方向精细测试 + d_level提取 + 依存DAG构建

核心问题:
  CCXLIII确认d_syn编码"依存方向"(头-依赖), 但:
  1. 只测试了5种角色, 需要更系统的依存类型映射
  2. d_level方向尚未提取和验证
  3. 从方向到依存DAG的构建尚未完成

实验设计:
  Exp1: 8+种依存类型的d_syn投影映射表
    - subject (nsubj): 主语
    - object (dobj): 直接宾语
    - indirect_object (iobj): 间接宾语
    - modifier (amod): 形容词修饰语
    - possessor (poss): 所有格
    - prep_object (pobj): 介词宾语
    - complement (acomp): 表语/补语
    - adverbial (advmod): 状语修饰语
    - relative_clause (acl): 关系从句中的名词
    - reflexive (refl): 反身代词宾语

  Exp2: d_level方向提取与验证
    - d_head = d_syn方向 (头vs依赖)
    - d_level = V_sem^⊥中的第二语法方向
    - 验证: d_head ⊥ d_level?
    - 2维语法空间能否编码8+种角色?

  Exp3: 依存DAG构建与验证
    - 从(d_head, d_level)坐标构建依存图
    - 验证: 依存图的边方向是否与语言学一致?
    - 验证: 依存DAG与语言学依存树的同构性
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
DITRANSITIVE_VERBS = ["gives", "shows", "tells", "sends", "offers"]
PASSIVE_VERBS = ["chased", "seen", "found", "taken", "watched"]

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

ADJECTIVES = ["red", "big", "small", "old", "young", "fast", "slow", "dark"]
ADVERBS = ["quickly", "slowly", "carefully", "silently", "boldly", "gently", "quietly", "softly"]
POSSESSIVE_NOUNS = ["cat", "dog", "king", "queen", "teacher", "doctor", "child", "mother"]


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


def find_token_position(tokenizer, sentence, target_word):
    """更通用的token位置查找"""
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    for prefix in ['', ' ']:
        target_tokens = tokenizer(prefix + target_word, add_special_tokens=False)['input_ids']
        for i in range(len(input_ids) - len(target_tokens) + 1):
            if input_ids[i:i+len(target_tokens)] == target_tokens:
                return i + len(target_tokens) - 1
    # fallback
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == target_word.lower():
            return i
    return None


def compute_perp_basis(V_sem_5):
    d = V_sem_5.shape[1]
    proj_vsem = V_sem_5.T @ V_sem_5
    proj_perp = np.eye(d) - proj_vsem
    return proj_perp


def extract_d_syn_from_reps(subj_reps, obj_reps, proj_perp):
    subj_perp = (proj_perp @ np.array(subj_reps).T).T
    obj_perp = (proj_perp @ np.array(obj_reps).T).T
    d_role = subj_perp - obj_perp
    d_syn = d_role.mean(axis=0)
    d_syn_norm = np.linalg.norm(d_syn)
    if d_syn_norm > 1e-10:
        d_syn_unit = d_syn / d_syn_norm
    else:
        d_syn_unit = np.zeros_like(d_syn)
    return d_syn_unit, d_syn_norm


# ============================================================
# Exp1: 8+种依存类型的d_syn投影映射表
# ============================================================
def run_exp1(model_name):
    """
    系统测试8+种依存类型的d_syn投影
    
    依存类型:
    1. subject (nsubj): "The cat chases the dog"
    2. object (dobj): "The cat chases the dog"
    3. indirect_object (iobj): "The cat gives the dog the fish"
    4. modifier (amod): "The red cat chases the dog"
    5. possessor (poss): "The cat's tail is long"
    6. prep_object (pobj): "The cat looks at the dog"
    7. complement (acomp): "The cat seems happy" / "The cat becomes big"
    8. adverbial (advmod): "The cat runs quickly"
    9. passive_subject (nsubjpass): "The dog is chased by the cat"
    10. agent (agent): "The dog is chased by the cat" (by-phrase)
    """
    print(f"\n{'='*60}")
    print(f"CCXLIV Exp1: 10 Dependency Types d_syn Map — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

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
    V_sem_5 = pca_sem.components_[:5]
    proj_perp = compute_perp_basis(V_sem_5)

    # 从主动句提取d_syn (= d_head方向)
    print("Extracting d_syn (= d_head) from active sentences...")
    subj_reps = []
    obj_reps = []
    for noun_a, noun_b in NOUN_PAIRS:
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
                    subj_reps.append(out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy())
                    obj_reps.append(out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy())
                except:
                    pass

    d_syn_unit, d_syn_norm = extract_d_syn_from_reps(subj_reps, obj_reps, proj_perp)
    print(f"  ||d_syn|| = {d_syn_norm:.2f}")

    # ==========================================
    # 收集10种依存类型的表示
    # ==========================================
    print("\nCollecting 10 dependency type representations...")

    dep_reps = {
        "subject": [],
        "object": [],
        "indirect_object": [],
        "modifier": [],
        "possessor": [],
        "prep_object": [],
        "complement": [],
        "adverbial": [],
        "passive_subject": [],
        "agent": [],
    }

    test_nouns = ["cat", "dog", "bird", "fish", "lion", "tiger", "horse", "wolf",
                  "king", "queen", "hammer", "knife", "rain", "snow", "wood", "stone"]
    test_nouns_extra = test_nouns + ["bear", "fox", "deer", "hawk", "eagle", "shark"]

    for noun in test_nouns_extra:
        # 1. subject: "The noun chases the dog"
        for verb in ["chases", "sees", "finds"]:
            sent = f"The {noun} {verb} the dog"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["subject"].append(rep)
                    except:
                        pass

        # 2. object: "The dog chases the noun"
        for verb in ["chases", "sees", "finds"]:
            sent = f"The dog {verb} the {noun}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["object"].append(rep)
                    except:
                        pass

        # 3. indirect_object: "The king gives the noun the sword"
        for verb in ["gives", "shows", "tells"]:
            sent = f"The king {verb} the {noun} the sword"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["indirect_object"].append(rep)
                    except:
                        pass

        # 4. modifier (adjective): "The red noun chases the dog"
        for adj in ["red", "big", "small", "old"]:
            sent = f"The {adj} {noun} chases the dog"
            pos = find_noun_position(tokenizer, sent, adj)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["modifier"].append(rep)
                    except:
                        pass

        # 5. possessor: "The noun's tail is long"
        for body in ["tail", "head", "eye", "leg"]:
            sent = f"The {noun}'s {body} is long"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["possessor"].append(rep)
                    except:
                        pass

        # 6. prep_object: "The cat looks at the noun"
        for prep_phrase in ["looks at", "runs to", "comes from"]:
            sent = f"The cat {prep_phrase} the {noun}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["prep_object"].append(rep)
                    except:
                        pass

        # 7. complement (predicative): "The noun seems happy"
        for adj in ["happy", "big", "strong", "old"]:
            sent = f"The {noun} seems {adj}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["complement"].append(rep)
                    except:
                        pass

        # 8. adverbial: "The cat runs quickly"
        # noun is subject, adverb is target
        # We want the adverb's representation
        for adv in ADVERBS[:4]:
            sent = f"The {noun} runs {adv}"
            pos = find_token_position(tokenizer, sent, adv.replace("ly", ""))  # try without -ly
            if pos is None:
                pos = find_token_position(tokenizer, sent, adv)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["adverbial"].append(rep)
                    except:
                        pass

    # 9-10: 被动语态
    for noun_a, noun_b in NOUN_PAIRS[:15]:
        for i, verb in enumerate(TRANSITIVE_VERBS[:5]):
            pverb = PASSIVE_VERBS[i]
            # passive_subject: "The noun is chased by the dog"
            sent = f"The {noun_a} is {pverb} by the {noun_b}"
            pos = find_noun_position(tokenizer, sent, noun_a)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["passive_subject"].append(rep)
                    except:
                        pass

            # agent (by-phrase): "The dog is chased by the noun"
            sent = f"The {noun_b} is {pverb} by the {noun_a}"
            pos = find_noun_position(tokenizer, sent, noun_a)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        dep_reps["agent"].append(rep)
                    except:
                        pass

    # 打印统计
    for dep, reps in dep_reps.items():
        print(f"  {dep}: {len(reps)} representations")

    # ==========================================
    # 计算10种依存类型的d_syn投影
    # ==========================================
    print("\n--- d_syn Projection by Dependency Type ---")

    results = {
        "model": model_name, "exp": 1,
        "d_syn_norm": float(d_syn_norm),
        "dep_counts": {k: len(v) for k, v in dep_reps.items()},
    }

    dep_projections = {}
    for dep, reps in dep_reps.items():
        if len(reps) > 0:
            reps_arr = np.array(reps)
            reps_perp = (proj_perp @ reps_arr.T).T
            projections = np.array([np.dot(r, d_syn_unit) * d_syn_norm for r in reps_perp])
            dep_projections[dep] = {
                "mean": float(projections.mean()),
                "std": float(projections.std()),
                "median": float(np.median(projections)),
                "n": len(projections),
            }
            print(f"  {dep:20s}: d_syn proj mean={projections.mean():+.2f}, std={projections.std():.2f}, n={len(projections)}")

    results["dep_projections"] = dep_projections

    # ==========================================
    # 依存类型排序 + 层级分析
    # ==========================================
    print("\n--- Dependency Type Ranking by d_syn Projection ---")
    sorted_deps = sorted(dep_projections.items(), key=lambda x: x[1]["mean"], reverse=True)
    for rank, (dep, stats) in enumerate(sorted_deps, 1):
        bar = "+" * int(max(0, stats["mean"] / 50))
        print(f"  {rank:2d}. {dep:20s}: {stats['mean']:+8.2f}  {bar}")

    # ==========================================
    # 依存类型分类
    # ==========================================
    print("\n--- Multi-class Classification ---")

    # 合并所有表示
    all_reps = []
    all_labels = []
    all_dep_perp = []
    for dep, reps in dep_reps.items():
        if len(reps) > 0:
            reps_arr = np.array(reps)
            reps_perp = (proj_perp @ reps_arr.T).T
            all_reps.extend(reps)
            all_dep_perp.extend(reps_perp)
            all_labels.extend([dep] * len(reps))

    all_dep_perp = np.array(all_dep_perp)
    all_labels = np.array(all_labels)

    if len(all_dep_perp) > 50:
        pca_perp = PCA(n_components=20)
        all_perp_pca = pca_perp.fit_transform(all_dep_perp)

        # 2-way: subject vs object
        mask_so = np.isin(all_labels, ["subject", "object"])
        if mask_so.sum() > 20:
            X = all_perp_pca[mask_so, :5]
            y = (all_labels[mask_so] == "subject").astype(int)
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            print(f"  subject vs object: {scores.mean():.3f}")
            results["cv_subj_obj"] = float(scores.mean())

        # 3-way: head vs dependent vs other
        head_deps = ["subject", "passive_subject", "modifier", "possessor", "complement"]
        dep_deps = ["object", "indirect_object", "prep_object", "agent", "adverbial"]
        
        head_mask = np.isin(all_labels, head_deps)
        dep_mask = np.isin(all_labels, dep_deps)
        mask_3 = head_mask | dep_mask
        
        if mask_3.sum() > 30:
            X = all_perp_pca[mask_3, :5]
            y_3 = np.array(["head" if l in head_deps else "dependent" for l in all_labels[mask_3]])
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X, y_3, cv=5, scoring='accuracy')
            print(f"  head vs dependent: {scores.mean():.3f} (chance=0.500)")
            results["cv_head_dep"] = float(scores.mean())

        # 5-way: major types
        major_5 = ["subject", "object", "modifier", "prep_object", "complement"]
        mask_5 = np.isin(all_labels, major_5)
        if mask_5.sum() > 50:
            min_count = min((all_labels[mask_5] == d).sum() for d in major_5)
            if min_count >= 5:
                balanced = []
                bal_labels = []
                rng = np.random.RandomState(42)
                for dep in major_5:
                    indices = np.where((all_labels == dep) & mask_5)[0]
                    selected = rng.choice(indices, size=min(min_count, len(indices)), replace=False)
                    balanced.append(all_perp_pca[selected, :5])
                    bal_labels.extend([dep] * len(selected))
                if len(bal_labels) >= 50:
                    X = np.vstack(balanced)
                    y = np.array(bal_labels)
                    clf = LogisticRegression(max_iter=1000)
                    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                    print(f"  5-way (subj/obj/mod/pobj/comp): {scores.mean():.3f} (chance=0.200)")
                    results["cv_5way"] = float(scores.mean())

        # 10-way
        min_count = min((all_labels == d).sum() for d in dep_reps if (all_labels == d).sum() > 0)
        if min_count >= 5 and len(all_labels) >= 100:
            balanced = []
            bal_labels = []
            rng = np.random.RandomState(42)
            for dep in dep_reps:
                indices = np.where(all_labels == dep)[0]
                if len(indices) >= 5:
                    selected = rng.choice(indices, size=min(min_count, len(indices)), replace=False)
                    balanced.append(all_perp_pca[selected, :10])
                    bal_labels.extend([dep] * len(selected))
            if len(bal_labels) >= 100:
                X = np.vstack(balanced)
                y = np.array(bal_labels)
                clf = LogisticRegression(max_iter=1000)
                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                print(f"  10-way (all deps): {scores.mean():.3f} (chance=0.100)")
                results["cv_10way"] = float(scores.mean())

    # ==========================================
    # PC0/PC1分布
    # ==========================================
    if len(all_dep_perp) > 50:
        print("\n--- PC0/PC1 Distribution by Dependency Type ---")
        for pc_idx in [0, 1]:
            print(f"\n  PC{pc_idx}:")
            for dep in ["subject", "object", "modifier", "complement", "possessor", 
                        "prep_object", "indirect_object", "adverbial", "passive_subject", "agent"]:
                mask = all_labels == dep
                if mask.sum() > 0:
                    vals = all_perp_pca[mask, pc_idx]
                    print(f"    {dep:20s}: mean={vals.mean():+.2f}, std={vals.std():.2f}")

        results["pca_explained"] = pca_perp.explained_variance_ratio_[:10].tolist()

    out_path = TEMP / f"ccxliv_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp2: d_level方向提取与验证
# ============================================================
def run_exp2(model_name):
    """
    提取d_level方向并验证d_head ⊥ d_level

    方法:
    1. d_head = d_syn方向 (从subj/obj对比提取)
    2. d_level = V_sem^⊥中的第二语法方向
       - 从subject vs modifier对比提取
       - 或从object vs prep_object对比提取
    3. 验证正交性
    4. 2维语法空间(d_head, d_level)的编码能力
    """
    print(f"\n{'='*60}")
    print(f"CCXLIV Exp2: d_level Extraction & Orthogonality — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

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
    V_sem_5 = pca_sem.components_[:5]
    proj_perp = compute_perp_basis(V_sem_5)

    # ==========================================
    # 收集各类角色表示
    # ==========================================
    print("Collecting role representations...")

    role_reps = {"subject": [], "object": [], "modifier": [], "prep_object": [],
                 "complement": [], "possessor": [], "indirect_object": []}

    test_nouns = ["cat", "dog", "bird", "fish", "lion", "tiger", "horse", "wolf",
                  "king", "queen", "hammer", "knife", "rain", "snow", "wood", "stone"]

    for noun in test_nouns:
        # subject
        for verb in ["chases", "sees", "finds"]:
            sent = f"The {noun} {verb} the dog"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["subject"].append(rep)
                    except:
                        pass

        # object
        for verb in ["chases", "sees", "finds"]:
            sent = f"The dog {verb} the {noun}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["object"].append(rep)
                    except:
                        pass

        # modifier
        for adj in ["red", "big", "small", "old"]:
            sent = f"The {adj} {noun} chases the dog"
            pos = find_noun_position(tokenizer, sent, adj)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["modifier"].append(rep)
                    except:
                        pass

        # prep_object
        for prep_phrase in ["looks at", "runs to", "comes from"]:
            sent = f"The cat {prep_phrase} the {noun}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["prep_object"].append(rep)
                    except:
                        pass

        # complement
        for adj in ["happy", "big", "strong", "old"]:
            sent = f"The {noun} seems {adj}"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["complement"].append(rep)
                    except:
                        pass

        # possessor
        for body in ["tail", "head", "eye", "leg"]:
            sent = f"The {noun}'s {body} is long"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["possessor"].append(rep)
                    except:
                        pass

        # indirect_object
        for verb in ["gives", "shows", "tells"]:
            sent = f"The king {verb} the {noun} the sword"
            pos = find_noun_position(tokenizer, sent, noun)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                        role_reps["indirect_object"].append(rep)
                    except:
                        pass

    for role, reps in role_reps.items():
        print(f"  {role}: {len(reps)} representations")

    # ==========================================
    # 提取d_head方向
    # ==========================================
    print("\n--- Extracting d_head (= d_syn) ---")
    d_head_unit, d_head_norm = extract_d_syn_from_reps(
        role_reps["subject"], role_reps["object"], proj_perp
    )
    print(f"  ||d_head|| = {d_head_norm:.2f}")

    # ==========================================
    # 提取d_level方向
    # ==========================================
    # 方法1: subject vs modifier (都是head, 但不同层级)
    print("\n--- Extracting d_level (subject vs modifier) ---")
    
    subj_perp = (proj_perp @ np.array(role_reps["subject"]).T).T
    mod_perp = (proj_perp @ np.array(role_reps["modifier"]).T).T
    
    # 先移除d_head分量
    subj_resid = subj_perp - np.outer(subj_perp @ d_head_unit, d_head_unit)
    mod_resid = mod_perp - np.outer(mod_perp @ d_head_unit, d_head_unit)
    
    d_level_raw = subj_resid.mean(axis=0) - mod_resid.mean(axis=0)
    d_level_norm = np.linalg.norm(d_level_raw)
    if d_level_norm > 1e-10:
        d_level_unit = d_level_raw / d_level_norm
    else:
        d_level_unit = np.zeros_like(d_level_raw)
    print(f"  ||d_level|| = {d_level_norm:.2f}")

    # 方法2: object vs prep_object (都是dependent, 但不同层级)
    print("\n--- Extracting d_level2 (object vs prep_object) ---")
    obj_perp = (proj_perp @ np.array(role_reps["object"]).T).T
    pobj_perp = (proj_perp @ np.array(role_reps["prep_object"]).T).T
    
    obj_resid = obj_perp - np.outer(obj_perp @ d_head_unit, d_head_unit)
    pobj_resid = pobj_perp - np.outer(pobj_perp @ d_head_unit, d_head_unit)
    
    d_level2_raw = obj_resid.mean(axis=0) - pobj_resid.mean(axis=0)
    d_level2_norm = np.linalg.norm(d_level2_raw)
    if d_level2_norm > 1e-10:
        d_level2_unit = d_level2_raw / d_level2_norm
    else:
        d_level2_unit = np.zeros_like(d_level2_raw)
    print(f"  ||d_level2|| = {d_level2_norm:.2f}")

    # ==========================================
    # ★★★ 验证正交性
    # ==========================================
    print("\n--- ★★★ Orthogonality Verification ---")

    # d_head ⊥ d_level?
    cos_head_level = np.dot(d_head_unit, d_level_unit)
    print(f"  cos(d_head, d_level)  = {cos_head_level:.6f}")
    print(f"  |cos| < 0.1? {'YES ★' if abs(cos_head_level) < 0.1 else 'NO'}")

    # d_head ⊥ d_level2?
    cos_head_level2 = np.dot(d_head_unit, d_level2_unit)
    print(f"  cos(d_head, d_level2) = {cos_head_level2:.6f}")
    print(f"  |cos| < 0.1? {'YES ★' if abs(cos_head_level2) < 0.1 else 'NO'}")

    # d_level ⊥ d_level2?
    cos_level_level2 = np.dot(d_level_unit, d_level2_unit)
    print(f"  cos(d_level, d_level2) = {cos_level_level2:.6f}")
    print(f"  |cos| < 0.1? {'YES ★' if abs(cos_level_level2) < 0.1 else 'NO'}")

    # 所有方向 ⊥ V_sem?
    for name, unit in [("d_head", d_head_unit), ("d_level", d_level_unit), ("d_level2", d_level2_unit)]:
        max_vsem_cos = max(abs(np.dot(unit, V_sem_5[i])) for i in range(5))
        print(f"  max|cos({name}, V_sem_i)| = {max_vsem_cos:.6f}")

    results = {
        "model": model_name, "exp": 2,
        "d_head_norm": float(d_head_norm),
        "d_level_norm": float(d_level_norm),
        "d_level2_norm": float(d_level2_norm),
        "cos_head_level": float(cos_head_level),
        "cos_head_level2": float(cos_head_level2),
        "cos_level_level2": float(cos_level_level2),
    }

    # ==========================================
    # 2维语法空间 (d_head, d_level) 的编码
    # ==========================================
    print("\n--- 2D Syntax Space Encoding ---")

    # 每种角色在(d_head, d_level)中的坐标
    role_coords = {}
    for role, reps in role_reps.items():
        if len(reps) > 0:
            reps_arr = np.array(reps)
            reps_perp = (proj_perp @ reps_arr.T).T
            head_proj = np.array([np.dot(r, d_head_unit) * d_head_norm for r in reps_perp])
            level_proj = np.array([np.dot(r, d_level_unit) * d_level_norm for r in reps_perp])
            role_coords[role] = {
                "head_mean": float(head_proj.mean()),
                "head_std": float(head_proj.std()),
                "level_mean": float(level_proj.mean()),
                "level_std": float(level_proj.std()),
            }
            print(f"  {role:20s}: d_head={head_proj.mean():+8.2f}, d_level={level_proj.mean():+8.2f}")

    results["role_coords_2d"] = role_coords

    # 也计算d_level2的坐标
    role_coords_3d = {}
    for role, reps in role_reps.items():
        if len(reps) > 0:
            reps_arr = np.array(reps)
            reps_perp = (proj_perp @ reps_arr.T).T
            head_proj = np.array([np.dot(r, d_head_unit) * d_head_norm for r in reps_perp])
            level_proj = np.array([np.dot(r, d_level_unit) * d_level_norm for r in reps_perp])
            level2_proj = np.array([np.dot(r, d_level2_unit) * d_level2_norm for r in reps_perp])
            role_coords_3d[role] = {
                "head_mean": float(head_proj.mean()),
                "level_mean": float(level_proj.mean()),
                "level2_mean": float(level2_proj.mean()),
            }
    results["role_coords_3d"] = role_coords_3d

    # ==========================================
    # 2维/3维分类能力
    # ==========================================
    print("\n--- Classification with 2D/3D Syntax Space ---")

    all_reps = []
    all_labels = []
    all_perp = []
    for role, reps in role_reps.items():
        if len(reps) > 0:
            reps_arr = np.array(reps)
            reps_perp = (proj_perp @ reps_arr.T).T
            all_perp.extend(reps_perp)
            all_labels.extend([role] * len(reps))

    all_perp = np.array(all_perp)
    all_labels = np.array(all_labels)

    if len(all_perp) > 50:
        # (d_head, d_level) 坐标
        head_projs = np.array([np.dot(r, d_head_unit) * d_head_norm for r in all_perp])
        level_projs = np.array([np.dot(r, d_level_unit) * d_level_norm for r in all_perp])
        level2_projs = np.array([np.dot(r, d_level2_unit) * d_level2_norm for r in all_perp])

        X_1d = head_projs.reshape(-1, 1)
        X_2d = np.column_stack([head_projs, level_projs])
        X_3d = np.column_stack([head_projs, level_projs, level2_projs])

        # subject vs object
        mask_so = np.isin(all_labels, ["subject", "object"])
        if mask_so.sum() > 20:
            for name, X in [("1D(d_head)", X_1d), ("2D(d_head+d_level)", X_2d), ("3D(+d_level2)", X_3d)]:
                X_sub = X[mask_so]
                y_sub = (all_labels[mask_so] == "subject").astype(int)
                clf = LogisticRegression(max_iter=1000)
                scores = cross_val_score(clf, X_sub, y_sub, cv=5, scoring='accuracy')
                print(f"  subj vs obj {name}: {scores.mean():.3f}")

        # head vs dependent
        head_roles = ["subject", "modifier", "complement", "possessor"]
        dep_roles = ["object", "prep_object", "indirect_object"]
        mask_hd = np.isin(all_labels, head_roles + dep_roles)
        if mask_hd.sum() > 30:
            for name, X in [("1D", X_1d), ("2D", X_2d), ("3D", X_3d)]:
                X_sub = X[mask_hd]
                y_sub = np.array(["head" if l in head_roles else "dep" for l in all_labels[mask_hd]])
                clf = LogisticRegression(max_iter=1000)
                scores = cross_val_score(clf, X_sub, y_sub, cv=5, scoring='accuracy')
                print(f"  head vs dep {name}: {scores.mean():.3f}")

        # 7-way (all roles with enough samples)
        valid_roles = [r for r in role_reps if len(role_reps[r]) >= 10]
        if len(valid_roles) >= 5:
            mask_all = np.isin(all_labels, valid_roles)
            min_count = min((all_labels[mask_all] == r).sum() for r in valid_roles)
            if min_count >= 5:
                balanced = []
                bal_labels = []
                rng = np.random.RandomState(42)
                for role in valid_roles:
                    indices = np.where(all_labels == role)[0]
                    selected = rng.choice(indices, size=min(min_count, len(indices)), replace=False)
                    balanced.append(X_3d[selected])
                    bal_labels.extend([role] * len(selected))
                if len(bal_labels) >= 50:
                    X_bal = np.vstack(balanced)
                    y_bal = np.array(bal_labels)
                    clf = LogisticRegression(max_iter=1000)
                    scores = cross_val_score(clf, X_bal, y_bal, cv=5, scoring='accuracy')
                    print(f"  {len(valid_roles)}-way (3D): {scores.mean():.3f} (chance={1/len(valid_roles):.3f})")
                    results["cv_nway_3d"] = float(scores.mean())

    out_path = TEMP / f"ccxliv_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp3: 依存DAG构建与验证
# ============================================================
def run_exp3(model_name):
    """
    从(d_head, d_level)坐标构建依存DAG
    
    方法:
    1. 对句子中每个token, 计算(d_head, d_level)坐标
    2. 根据d_head坐标确定head-dependent方向
    3. 根据d_level坐标确定语法层级
    4. 构建依存图, 验证与语言学依存树的一致性
    
    测试:
    - 简单SVO句: "The cat chases the dog"
      期望: cat→head, chases→root, dog→dependent
    - 带修饰语: "The red cat chases the big dog"
      期望: red→modifier(head), cat→head, big→modifier, dog→dep
    - 被动句: "The dog is chased by the cat"
      期望: dog→head(passive_subj), cat→dep(agent)
    - 双宾语: "The king gives the dog the sword"
      期望: king→head, dog→iobj(dep), sword→dobj(dep)
    """
    print(f"\n{'='*60}")
    print(f"CCXLIV Exp3: Dependency DAG Construction — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

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
    V_sem_5 = pca_sem.components_[:5]
    proj_perp = compute_perp_basis(V_sem_5)

    # 提取d_head和d_level
    print("Extracting d_head and d_level...")
    
    subj_reps, obj_reps = [], []
    for noun_a, noun_b in NOUN_PAIRS:
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
                    subj_reps.append(out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy())
                    obj_reps.append(out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy())
                except:
                    pass

    d_head_unit, d_head_norm = extract_d_syn_from_reps(subj_reps, obj_reps, proj_perp)
    print(f"  ||d_head|| = {d_head_norm:.2f}")

    # 提取d_level (subject vs modifier)
    mod_reps = []
    for noun in ["cat", "dog", "bird", "fish", "lion", "tiger", "horse", "wolf",
                 "king", "queen", "hammer", "knife", "rain", "snow"]:
        for adj in ["red", "big", "small", "old", "young", "fast", "slow", "dark"]:
            sent = f"The {adj} {noun} chases the dog"
            pos = find_noun_position(tokenizer, sent, adj)
            if pos is not None:
                inputs = tokenizer(sent, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        mod_reps.append(out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy())
                    except:
                        pass

    if len(subj_reps) > 0 and len(mod_reps) > 0:
        subj_perp = (proj_perp @ np.array(subj_reps).T).T
        mod_perp = (proj_perp @ np.array(mod_reps).T).T
        
        subj_resid = subj_perp - np.outer(subj_perp @ d_head_unit, d_head_unit)
        mod_resid = mod_perp - np.outer(mod_perp @ d_head_unit, d_head_unit)
        
        d_level_raw = subj_resid.mean(axis=0) - mod_resid.mean(axis=0)
        d_level_norm = np.linalg.norm(d_level_raw)
        if d_level_norm > 1e-10:
            d_level_unit = d_level_raw / d_level_norm
        else:
            d_level_unit = np.zeros_like(d_level_raw)
    else:
        d_level_unit = np.zeros(d_model)
        d_level_norm = 0.0
    
    print(f"  ||d_level|| = {d_level_norm:.2f}")

    # ==========================================
    # 对完整句子构建依存图
    # ==========================================
    print("\n--- Dependency DAG Construction ---")

    test_sentences = [
        # (sentence, expected_structure)
        ("The cat chases the dog", 
         {"cat": "subject", "chases": "root", "dog": "object"}),
        ("The red cat chases the big dog",
         {"red": "modifier", "cat": "subject", "chases": "root", "big": "modifier", "dog": "object"}),
        ("The dog is chased by the cat",
         {"dog": "passive_subject", "chased": "root", "cat": "agent"}),
        ("The king gives the dog the sword",
         {"king": "subject", "gives": "root", "dog": "indirect_object", "sword": "object"}),
        ("The cat's tail is long",
         {"cat": "possessor", "tail": "subject", "long": "complement"}),
        ("The bird looks at the fish",
         {"bird": "subject", "looks": "root", "fish": "prep_object"}),
        ("The lion seems strong",
         {"lion": "complement_subject", "seems": "root", "strong": "complement"}),
        ("The cat runs quickly",
         {"cat": "subject", "runs": "root", "quickly": "adverbial"}),
    ]

    results = {"model": model_name, "exp": 3, "sentences": []}

    for sent, expected in test_sentences:
        print(f"\n  Sentence: \"{sent}\"")
        inputs = tokenizer(sent, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        # 对每个token计算(d_head, d_level)坐标
        token_coords = []
        seq_len = input_ids.shape[1]
        
        for t in range(seq_len):
            rep = out.hidden_states[mid_layer][0, t, :].detach().cpu().float().numpy()
            rep_perp = proj_perp @ rep
            head_coord = np.dot(rep_perp, d_head_unit) * d_head_norm
            level_coord = np.dot(rep_perp, d_level_unit) * d_level_norm
            
            token_text = tokenizer.decode([input_ids[0, t].item()]).strip()
            token_coords.append({
                "token": token_text,
                "pos": t,
                "d_head": float(head_coord),
                "d_level": float(level_coord),
            })

        # 打印每个token的坐标
        print(f"    {'Token':12s} {'d_head':>10s} {'d_level':>10s} {'Expected':>20s}")
        print(f"    {'─'*55}")
        for tc in token_coords:
            token = tc["token"]
            expected_role = expected.get(token, "?")
            marker = "★" if expected_role != "?" else " "
            print(f"  {marker} {token:12s} {tc['d_head']:+10.2f} {tc['d_level']:+10.2f} {expected_role:>20s}")

        # 构建依存图: d_head最高的是root, 每个token依存于d_head值最近的上级
        # 简单启发式: d_head>0的是head类, d_head<0的是dependent类
        head_threshold = 0.0  # 简单二分
        
        sent_result = {
            "sentence": sent,
            "token_coords": token_coords,
            "expected": expected,
        }
        results["sentences"].append(sent_result)

    # ==========================================
    # 批量验证: d_head能否区分head/dependent
    # ==========================================
    print("\n--- Batch Head/Dep Classification ---")

    # 收集head和dependent token的d_head值
    head_dheads = []
    dep_dheads = []
    correct = 0
    total = 0

    for sent, expected in test_sentences:
        inputs = tokenizer(sent, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for token_text, role in expected.items():
            pos = find_token_position(tokenizer, sent, token_text)
            if pos is None:
                continue
            rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
            rep_perp = proj_perp @ rep
            d_head_val = np.dot(rep_perp, d_head_unit) * d_head_norm

            is_head_role = role in ["subject", "root", "modifier", "possessor", 
                                    "complement_subject", "complement"]
            if is_head_role:
                head_dheads.append(d_head_val)
            else:
                dep_dheads.append(d_head_val)

            # 预测: d_head > threshold → head
            predicted_head = d_head_val > head_threshold
            actual_head = is_head_role
            if predicted_head == actual_head:
                correct += 1
            total += 1

    if len(head_dheads) > 0 and len(dep_dheads) > 0:
        print(f"  Head tokens d_head:   mean={np.mean(head_dheads):+.2f}, std={np.std(head_dheads):.2f}")
        print(f"  Dep tokens d_head:    mean={np.mean(dep_dheads):+.2f}, std={np.std(dep_dheads):.2f}")
        print(f"  Classification acc:   {correct}/{total} = {correct/total:.3f}")

    results["head_dep_classification"] = {
        "head_dhead_mean": float(np.mean(head_dheads)) if head_dheads else 0,
        "dep_dhead_mean": float(np.mean(dep_dheads)) if dep_dheads else 0,
        "accuracy": float(correct/total) if total > 0 else 0,
    }

    # ==========================================
    # 依存方向一致性
    # ==========================================
    print("\n--- Dependency Direction Consistency ---")

    # 对于每对(head, dependent), d_head(head)应该 > d_head(dependent)
    # 测试: subject > object, modifier > noun_it_modifies, etc.
    direction_correct = 0
    direction_total = 0

    test_pairs_dir = [
        # (sentence, head_token, dep_token)
        ("The cat chases the dog", "cat", "dog"),
        ("The red cat chases the dog", "cat", "dog"),
        ("The red cat chases the big dog", "cat", "dog"),
        ("The king gives the dog the sword", "king", "dog"),
        ("The king gives the dog the sword", "king", "sword"),
        ("The dog is chased by the cat", "dog", "cat"),
        ("The bird looks at the fish", "bird", "fish"),
        ("The cat's tail is long", "cat", "tail"),
    ]

    for sent, head_tok, dep_tok in test_pairs_dir:
        inputs = tokenizer(sent, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(inputs['input_ids'], output_hidden_states=True)

        head_pos = find_token_position(tokenizer, sent, head_tok)
        dep_pos = find_token_position(tokenizer, sent, dep_tok)

        if head_pos is None or dep_pos is None:
            continue

        head_rep = out.hidden_states[mid_layer][0, head_pos, :].detach().cpu().float().numpy()
        dep_rep = out.hidden_states[mid_layer][0, dep_pos, :].detach().cpu().float().numpy()

        head_dhead = np.dot(proj_perp @ head_rep, d_head_unit) * d_head_norm
        dep_dhead = np.dot(proj_perp @ dep_rep, d_head_unit) * d_head_norm

        is_correct = head_dhead > dep_dhead
        direction_correct += int(is_correct)
        direction_total += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status} \"{sent}\": {head_tok}({head_dhead:+.2f}) > {dep_tok}({dep_dhead:+.2f})")

    if direction_total > 0:
        print(f"\n  Direction consistency: {direction_correct}/{direction_total} = {direction_correct/direction_total:.3f}")

    results["direction_consistency"] = {
        "correct": direction_correct,
        "total": direction_total,
        "accuracy": float(direction_correct/direction_total) if direction_total > 0 else 0,
    }

    out_path = TEMP / f"ccxliv_exp3_{model_name}_results.json"
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
