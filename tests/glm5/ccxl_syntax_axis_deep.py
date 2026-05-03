"""
CCXL(400): 语法编码轴深度解析 — 从子空间到语法运算

核心目标:
  CCXXXII发现了:
  1. V_sem^⊥(5d) CV=1.0, PC0的d_cohen>10 → 存在极强角色编码轴
  2. Exp1位置控制后role_cv=0.5 → 分类方法有bug(不同名词对比较不公平)
  3. δ_role跨名词cos≈0.55-0.76 → 角色方向有一致性但非完全平行
  4. δ_role在V_sem^⊥(5d)中能量仅~49% → 51%在高维空间

  本实验解决三个关键问题:

  Exp1: 修正后的位置控制角色解析
    - Bug修复: 之前用subj_early vs obj_early分类,但这些样本对应不同名词对
    - 修正: 用同一名词的subj vs obj做配对分类
    - 追踪: 逐层的角色分类准确率(真正控制位置)

  Exp2: V_sem^⊥中角色编码轴的精细结构
    - PC0的d_cohen>10 → 这几乎是一个完美的"角色方向"
    - 分析: PC0方向与V_sem各轴的关系
    - 分析: PC0方向上,不同名词的角色偏移是否平行(线性可加?)
    - 分析: 在PC0方向上定义"角色坐标" → 正值=主语, 负值=宾语?

  Exp3: ★★★ 语法运算的代数结构
    - 核心假设: V_sem^⊥中存在一个"语法方向"d_syn
    - 使得: rep(主语位置) ≈ rep(基础) + α·d_syn
    - 使得: rep(宾语位置) ≈ rep(基础) - β·d_syn
    - 如果α=β对所有名词成立 → 语法运算是线性算子!
    - 测试: d_syn的加法可逆性(主语→宾语=沿d_syn反方向移动)
    - 测试: d_syn的跨动词一致性
    - 测试: d_syn能否泛化到新名词?

  Exp4: 多角色扩展 (主语/宾语/修饰语/独立)
    - "The red cat chases the dog" → cat=主语, dog=宾语, red=修饰语
    - "The cat sees the red dog" → cat=主语, dog=宾语, red=修饰语
    - 三种角色的编码: 是否在V_sem^⊥中形成三角形/四面体?
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

# 用于PCA训练的80个概念
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

# 修饰语(用于Exp4)
ADJECTIVES = ["red", "big", "old", "fast", "strong"]


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


def compute_perp_basis(V_sem_5):
    """计算V_sem的正交补空间投影矩阵"""
    d = V_sem_5.shape[1]
    proj_vsem = V_sem_5.T @ V_sem_5  # [d, d]
    proj_perp = np.eye(d) - proj_vsem
    return proj_perp


# ============================================================
# Exp1: 修正后的位置控制角色解析
# ============================================================
def run_exp1(model_name):
    """
    修正: 用同一名词的subj vs obj做配对分类, 不是跨名词对比较
    关键改进: 每个样本对是同一名词在主语/宾语位置的表示
    """
    print(f"\n{'='*60}")
    print(f"CCXL Exp1: Fixed Position-Controlled Role Resolution — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2

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

    X_pca = np.array(concept_reps)
    pca_sem = PCA(n_components=50)
    pca_sem.fit(X_pca)
    V_sem_5 = pca_sem.components_[:5]
    proj_perp = compute_perp_basis(V_sem_5)

    # 收集2×2因子数据
    conditions = ["subj_early", "subj_late", "obj_early", "obj_late"]

    sample_layers = list(range(0, n_layers, max(1, n_layers // 12))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    # 存储逐层数据
    all_layer_data = {}  # {layer: {noun_verb: {cond: rep}}}

    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")

        for verb in TRANSITIVE_VERBS[:3]:
            # 4种条件
            for noun, other in [(noun_a, noun_b), (noun_b, noun_a)]:
                sentences = {
                    "subj_early": f"The {noun} that {verb} the {other} runs",
                    "subj_late": f"The {other} that the {noun} {verb} runs",
                    "obj_early": f"The {noun} that the {other} {verb} runs",
                    "obj_late": f"The {other} {verb} the {noun} that runs",
                }

                key = f"{noun}_{verb}"

                for cond_name, sentence in sentences.items():
                    noun_pos = find_noun_position(tokenizer, sentence, noun)
                    if noun_pos is None:
                        continue

                    inputs = tokenizer(sentence, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            outputs = model(inputs['input_ids'], output_hidden_states=True)
                            for l in sample_layers:
                                if l not in all_layer_data:
                                    all_layer_data[l] = {}
                                nk = f"{key}_{cond_name}"
                                rep = outputs.hidden_states[l][0, noun_pos, :].detach().cpu().float().numpy()
                                all_layer_data[l][nk] = rep
                        except:
                            pass

    # 逐层分析
    results = {"model": model_name, "exp": 1, "n_layers": n_layers}

    print(f"\n--- Fixed Position-Controlled Role Resolution ---")
    print(f"  {'Layer':>5} | {'CV_full':>7} | {'CV_perp5':>8} | {'||δ_role||':>10} | {'||δ_pos||':>10} | {'V_sem%':>6}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    trajectory = []

    for l in sample_layers:
        data = all_layer_data.get(l, {})

        # 收集配对数据: 对每个noun_verb, 需要4个条件都有
        valid_pairs = []
        for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
            for verb in TRANSITIVE_VERBS[:3]:
                for noun in [noun_a, noun_b]:
                    key = f"{noun}_{verb}"
                    reps = {}
                    for c in conditions:
                        nk = f"{key}_{c}"
                        if nk in data:
                            reps[c] = data[nk]
                    if len(reps) == 4:
                        valid_pairs.append((noun, verb, reps))

        n_valid = len(valid_pairs)
        if n_valid < 10:
            continue

        # 方法1: 用同一名词的subj vs obj做分类(全空间)
        X_subj_obj = []
        y_subj_obj = []
        for noun, verb, reps in valid_pairs:
            # subj = (subj_early + subj_late) / 2 (平均掉位置效应)
            subj_rep = (reps["subj_early"] + reps["subj_late"]) / 2
            obj_rep = (reps["obj_early"] + reps["obj_late"]) / 2
            X_subj_obj.append(subj_rep)
            X_subj_obj.append(obj_rep)
            y_subj_obj.append(0)  # subj
            y_subj_obj.append(1)  # obj

        X_subj_obj = np.array(X_subj_obj)
        y_subj_obj = np.array(y_subj_obj)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv_folds = min(5, n_valid // 5)
        if cv_folds >= 2:
            cv_full = cross_val_score(clf, X_subj_obj, y_subj_obj, cv=cv_folds, scoring='accuracy').mean()
        else:
            cv_full = 0.5

        # 方法2: 在V_sem^⊥(5d)中分类
        X_perp = (proj_perp @ X_subj_obj.T).T
        pca_perp = PCA(n_components=min(5, n_valid * 2 - 1))
        X_perp5 = pca_perp.fit_transform(X_perp)
        cv_perp5 = cross_val_score(clf, X_perp5, y_subj_obj, cv=cv_folds, scoring='accuracy').mean()

        # δ_role和δ_pos统计
        role_deltas = []
        pos_deltas = []
        for noun, verb, reps in valid_pairs:
            se, sl, oe, ol = reps["subj_early"], reps["subj_late"], reps["obj_early"], reps["obj_late"]
            d_role = (se + sl - oe - ol) / 2
            d_pos = (se + oe - sl - ol) / 2
            role_deltas.append(d_role)
            pos_deltas.append(d_pos)

        role_deltas = np.array(role_deltas)
        pos_deltas = np.array(pos_deltas)

        mean_norm_role = float(np.mean(np.linalg.norm(role_deltas, axis=1)))
        mean_norm_pos = float(np.mean(np.linalg.norm(pos_deltas, axis=1)))

        vsem = np.mean([np.sum((np.dot(d, V_sem_5.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10)
                       for d in role_deltas])

        entry = {
            "layer": l, "cv_full": float(cv_full), "cv_perp5": float(cv_perp5),
            "n_valid": n_valid, "mean_norm_role": mean_norm_role,
            "mean_norm_pos": mean_norm_pos, "vsem_energy": float(vsem),
        }
        trajectory.append(entry)

        print(f"  L{l:>4} | {cv_full:>7.3f} | {cv_perp5:>8.3f} | {mean_norm_role:>10.2f} | {mean_norm_pos:>10.2f} | {vsem*100:>5.1f}%")

    results["trajectory"] = trajectory

    # 找角色解析层
    for entry in trajectory:
        if entry["cv_full"] > 0.65:
            results["resolution_layer"] = entry["layer"]
            results["resolution_cv"] = entry["cv_full"]
            print(f"\n  ★ Fixed role resolution: L{entry['layer']} (CV={entry['cv_full']:.3f})")
            break

    out_path = TEMP / f"ccxl_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp2: V_sem^⊥中角色编码轴的精细结构
# ============================================================
def run_exp2(model_name):
    """
    分析V_sem^⊥中角色编码轴:
    - PC0的d_cohen>10 → 角色方向
    - 角色坐标: 每个名词在PC0上的值 → 正=主语, 负=宾语?
    - 线性可加性: α·d_syn是否对所有名词相同?
    """
    print(f"\n{'='*60}")
    print(f"CCXL Exp2: Syntax Axis Fine Structure — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

    # 收集语法角色数据(同句改写)
    print("Collecting syntactic role data...")
    subj_reps = []
    obj_reps = []
    noun_labels_s = []
    verb_labels_s = []

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

                    subj_reps.append(rep_subj)
                    obj_reps.append(rep_obj)
                    noun_labels_s.append(noun_a)
                    verb_labels_s.append(verb)
                except:
                    pass

    n_subj = len(subj_reps)
    n_obj = len(obj_reps)
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
    V_sem_5 = pca_sem.components_[:5]
    proj_perp = compute_perp_basis(V_sem_5)

    # 投影到V_sem^⊥
    X_all = np.array(subj_reps + obj_reps)
    y_all = np.array([0] * n_subj + [1] * n_obj)
    X_perp = (proj_perp @ X_all.T).T

    # 在V_sem^⊥中做PCA
    pca_perp = PCA(n_components=50)
    pca_perp.fit(X_perp)

    results = {"model": model_name, "exp": 2, "n_subj": n_subj, "n_obj": n_obj}

    # === 2A: 角色编码轴分析 ===
    print("\n--- Syntax Axis Analysis ---")

    # PC0是角色编码主轴
    d_syn = pca_perp.components_[0]  # V_sem^⊥中的第1个PC方向(在原始空间)
    # 注意: d_syn是V_sem^⊥中的方向, 但定义在原始d_model维空间

    # 每个样本在PC0上的坐标
    coords_pc0 = pca_perp.transform(X_perp)[:, 0]

    subj_coords = coords_pc0[:n_subj]
    obj_coords = coords_pc0[n_subj:]

    print(f"  PC0 coordinates: subj mean={np.mean(subj_coords):.2f}, obj mean={np.mean(obj_coords):.2f}")
    print(f"  PC0 coordinates: subj std={np.std(subj_coords):.2f}, obj std={np.std(obj_coords):.2f}")

    # PC0坐标是否完美分隔角色?
    # 如果subj全在正侧, obj全在负侧(或反过来)
    sign_consistency = 0
    for sc, oc in zip(subj_coords, obj_coords):
        if np.sign(sc) != np.sign(oc):  # 不同侧
            sign_consistency += 1
    sign_rate = sign_consistency / min(n_subj, n_obj)
    print(f"  PC0 sign consistency: {sign_rate:.3f} (subj and obj on opposite sides)")

    results["pc0_analysis"] = {
        "subj_mean": float(np.mean(subj_coords)),
        "obj_mean": float(np.mean(obj_coords)),
        "subj_std": float(np.std(subj_coords)),
        "obj_std": float(np.std(obj_coords)),
        "sign_consistency": float(sign_rate),
        "variance_pct": float(pca_perp.explained_variance_ratio_[0] * 100),
    }

    # === 2B: 角色偏移的线性结构 ===
    print("\n--- Linear Structure of Role Shift ---")

    # 计算δ_role = rep(subj) - rep(obj) for each matched pair
    deltas = []
    delta_nouns = []
    delta_verbs = []

    # 匹配同名词同动词的subj/obj对
    for i in range(n_subj):
        for j in range(n_obj):
            if noun_labels_s[i] == noun_labels_s[j] and verb_labels_s[i] == verb_labels_s[j]:
                # 注意: noun_labels_s对应subj_reps, 但obj_reps的noun_labels也存了
                pass

    # 简化: 直接用配对的名词和动词
    noun_verb_set = {}
    for i in range(n_subj):
        key = f"{noun_labels_s[i]}_{verb_labels_s[i]}"
        if key not in noun_verb_set:
            noun_verb_set[key] = {}
        noun_verb_set[key]["subj"] = subj_reps[i]
        noun_verb_set[key]["subj_idx"] = i

    for i in range(n_obj):
        # obj_reps也是对noun_a的, 但noun_labels_s可能不对
        # 修正: obj_reps[i]对应noun_labels_s[i]作为宾语
        key = f"{noun_labels_s[i]}_{verb_labels_s[i]}"
        if key in noun_verb_set and "obj" not in noun_verb_set[key]:
            noun_verb_set[key]["obj"] = obj_reps[i]
            noun_verb_set[key]["obj_idx"] = i

    matched_pairs = []
    for key, val in noun_verb_set.items():
        if "subj" in val and "obj" in val:
            noun, verb = key.rsplit("_", 1)
            delta = val["subj"] - val["obj"]
            matched_pairs.append({
                "noun": noun, "verb": verb,
                "delta": delta,
                "subj_rep": val["subj"], "obj_rep": val["obj"],
            })

    print(f"  Matched pairs: {len(matched_pairs)}")

    if len(matched_pairs) < 5:
        results["linear_structure"] = "insufficient_data"
    else:
        # δ_role在PC0方向上的投影
        delta_pc0_coords = []
        delta_norms = []
        for p in matched_pairs:
            d = p["delta"]
            # 投影到V_sem^⊥再取PC0坐标
            d_perp = proj_perp @ d
            d_perp_pca = pca_perp.transform(d_perp.reshape(1, -1))
            delta_pc0_coords.append(d_perp_pca[0, 0])
            delta_norms.append(np.linalg.norm(d))

        # δ_role方向是否一致?
        delta_vecs = np.array([p["delta"] for p in matched_pairs])
        delta_norms_arr = np.linalg.norm(delta_vecs, axis=1, keepdims=True)
        delta_norms_arr = np.maximum(delta_norms_arr, 1e-10)
        delta_normalized = delta_vecs / delta_norms_arr

        mean_dir = delta_normalized.mean(axis=0)
        consistency = float(np.linalg.norm(mean_dir))

        # 按名词分组的δ_role方向一致性
        noun_deltas = {}
        for p in matched_pairs:
            noun = p["noun"]
            if noun not in noun_deltas:
                noun_deltas[noun] = []
            d_norm = np.linalg.norm(p["delta"])
            if d_norm > 1e-10:
                noun_deltas[noun].append(p["delta"] / d_norm)

        noun_consistency = {}
        for noun, deltas_n in noun_deltas.items():
            if len(deltas_n) >= 2:
                dn = np.array(deltas_n)
                mean_dn = dn.mean(axis=0)
                noun_consistency[noun] = float(np.linalg.norm(mean_dn))

        mean_noun_consist = np.mean(list(noun_consistency.values())) if noun_consistency else 0.0

        # δ_role的PC0坐标: 是否所有δ都指向PC0的同一侧?
        pc0_direction_sign = np.mean(np.sign(delta_pc0_coords))

        # α和β: δ_role在d_syn方向上的分量
        # 如果δ_role ≈ α·d_syn, 则α = δ_role · d_syn / ||d_syn||^2
        # 但d_syn在V_sem^⊥空间, 需要投影
        alphas = []
        for p in matched_pairs:
            d_perp = proj_perp @ p["delta"]
            # d_syn是pca_perp.components_[0], 在V_sem^⊥空间
            # α = d_perp · d_syn (因为d_syn已归一化)
            alpha = np.dot(d_perp, d_syn)
            alphas.append(alpha)

        alphas = np.array(alphas)

        results["linear_structure"] = {
            "global_consistency": consistency,
            "mean_noun_consistency": mean_noun_consist,
            "pc0_direction_sign": float(pc0_direction_sign),
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_cv": float(np.std(alphas) / (np.abs(np.mean(alphas)) + 1e-10)),
            "n_matched": len(matched_pairs),
        }

        print(f"  Global δ consistency: {consistency:.4f}")
        print(f"  Per-noun δ consistency: {mean_noun_consist:.4f}")
        print(f"  PC0 direction sign: {pc0_direction_sign:.3f}")
        print(f"  α (role shift in d_syn): mean={np.mean(alphas):.2f}, std={np.std(alphas):.2f}, CV={np.std(alphas)/(np.abs(np.mean(alphas))+1e-10):.3f}")

        # === 2C: d_syn与V_sem的关系 ===
        print("\n--- d_syn vs V_sem Relationship ---")

        # d_syn与V_sem各轴的余弦
        cos_with_vsem = []
        for i in range(5):
            cos = float(np.dot(d_syn, V_sem_5[i]) / (np.linalg.norm(d_syn) * np.linalg.norm(V_sem_5[i]) + 1e-10))
            cos_with_vsem.append(cos)

        # d_syn与V_sem^⊥中前5个PC的关系(已经正交, 但看角色信息分布)
        pc_role_scores = []
        for pc_idx in range(min(20, pca_perp.components_.shape[0])):
            pc_vals = pca_perp.transform(X_perp)[:, pc_idx]
            subj_vals = pc_vals[:n_subj]
            obj_vals = pc_vals[n_subj:]

            pooled_std = np.sqrt((np.var(subj_vals) + np.var(obj_vals)) / 2)
            d_cohen = abs(np.mean(subj_vals) - np.mean(obj_vals)) / (pooled_std + 1e-10)

            # 单PC分类
            X_pc = pc_vals.reshape(-1, 1)
            cv_folds = min(5, min(n_subj, n_obj) // 5)
            if cv_folds >= 2:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                cv_pc = cross_val_score(clf, X_pc, y_all, cv=cv_folds, scoring='accuracy').mean()
            else:
                cv_pc = 0.5

            pc_role_scores.append({
                "pc": pc_idx,
                "d_cohen": float(d_cohen),
                "cv_acc": float(cv_pc),
                "variance_pct": float(pca_perp.explained_variance_ratio_[pc_idx] * 100),
            })

        results["dsyn_vs_vsem"] = {
            "cos_with_vsem_pcs": cos_with_vsem,
        }
        results["pc_role_scores"] = pc_role_scores

        print(f"  cos(d_syn, V_sem PCs): {[f'{c:.4f}' for c in cos_with_vsem]}")
        print(f"  Top-5 PC role scores:")
        for i in range(min(5, len(pc_role_scores))):
            ps = pc_role_scores[i]
            print(f"    PC{ps['pc']}: d={ps['d_cohen']:.2f}, CV={ps['cv_acc']:.3f}, var={ps['variance_pct']:.2f}%")

        # === 2D: α的名词依赖性 ===
        print("\n--- α (Role Shift Magnitude) by Noun Category ---")

        categories = {
            "animals": ["cat", "dog", "bird", "fish", "lion", "tiger", "eagle", "whale", "horse", "wolf"],
            "tools": ["hammer", "knife", "sword", "wheel", "rope", "nail", "stone", "glass", "wood", "metal"],
            "social": ["king", "queen", "mother", "child", "friend", "enemy", "teacher", "student", "doctor", "patient"],
            "nature": ["rain", "snow", "wind", "storm", "sun", "moon", "fire", "water", "mountain", "river"],
        }

        cat_alphas = {}
        for cat_name, cat_nouns in categories.items():
            cat_a = [alphas[i] for i, p in enumerate(matched_pairs) if p["noun"] in cat_nouns]
            if cat_a:
                cat_alphas[cat_name] = {
                    "mean": float(np.mean(cat_a)),
                    "std": float(np.std(cat_a)),
                    "n": len(cat_a),
                }
                print(f"  {cat_name}: α_mean={np.mean(cat_a):.2f}, α_std={np.std(cat_a):.2f} (n={len(cat_a)})")

        results["alpha_by_category"] = cat_alphas

    out_path = TEMP / f"ccxl_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp3: 语法运算的代数结构
# ============================================================
def run_exp3(model_name):
    """
    核心测试: 语法角色变换是否是线性运算?
    
    假设: 存在一个"语法方向"d_syn, 使得:
      rep(主语) = rep(基础) + α·d_syn + 语义贡献
      rep(宾语) = rep(基础) - β·d_syn + 语义贡献
    
    测试:
    1. 加法可逆性: rep(subj) + rep(obj) ≈ 2·rep(基础)?
    2. d_syn的一致性: 不同名词/动词的d_syn是否相同?
    3. d_syn的泛化: 用部分数据提取d_syn, 能否对新名词分类?
    """
    print(f"\n{'='*60}")
    print(f"CCXL Exp3: Algebraic Structure of Syntax Operation — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

    # 收集数据: 同句改写 + 孤立表示
    print("Collecting data (syntactic + isolated)...")

    noun_reps = {}  # {noun: {condition: rep}}
    # conditions: "subj_verbX", "obj_verbX", "isolated"

    # 收集孤立表示
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")

        for noun in [noun_a, noun_b]:
            if noun not in noun_reps:
                noun_reps[noun] = {}

            # 孤立表示
            prompt = f"The word is {noun}"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                try:
                    outputs = model(inputs['input_ids'], output_hidden_states=True)
                    last_pos = inputs['input_ids'].shape[1] - 1
                    rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
                    noun_reps[noun]["isolated"] = rep
                except:
                    pass

    # 收集同句改写表示
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
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

                    if noun_a not in noun_reps:
                        noun_reps[noun_a] = {}
                    noun_reps[noun_a][f"subj_{verb}"] = rep_subj
                    noun_reps[noun_a][f"obj_{verb}"] = rep_obj
                except:
                    pass

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

    results = {"model": model_name, "exp": 3}

    # === 3A: 加法可逆性测试 ===
    print("\n--- Additive Invertibility Test ---")

    # 对于有subj, obj, isolated数据的名词:
    # rep(subj) + rep(obj) ≈ 2·rep(isolated)?
    # 或者: rep(subj) - rep(isolated) ≈ -(rep(obj) - rep(isolated))?
    # 即: δ_subj ≈ -δ_obj?

    invertibility_data = []
    for noun, conds in noun_reps.items():
        if "isolated" not in conds:
            continue
        iso = conds["isolated"]
        for verb in TRANSITIVE_VERBS:
            sk = f"subj_{verb}"
            ok = f"obj_{verb}"
            if sk in conds and ok in conds:
                delta_subj = conds[sk] - iso
                delta_obj = conds[ok] - iso

                # 投影到V_sem^⊥
                delta_subj_perp = proj_perp @ delta_subj
                delta_obj_perp = proj_perp @ delta_obj

                invertibility_data.append({
                    "noun": noun, "verb": verb,
                    "delta_subj": delta_subj, "delta_obj": delta_obj,
                    "delta_subj_perp": delta_subj_perp, "delta_obj_perp": delta_obj_perp,
                })

    print(f"  Data points: {len(invertibility_data)}")

    if len(invertibility_data) >= 5:
        # 测试1: δ_subj ≈ -δ_obj? (在V_sem^⊥中)
        cos_inv_perp = []
        cos_inv_full = []
        for d in invertibility_data:
            ds = d["delta_subj_perp"]
            do = d["delta_obj_perp"]
            n1, n2 = np.linalg.norm(ds), np.linalg.norm(do)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_inv_perp.append(np.dot(ds, -do) / (n1 * n2))

            ds_f = d["delta_subj"]
            do_f = d["delta_obj"]
            n1f, n2f = np.linalg.norm(ds_f), np.linalg.norm(do_f)
            if n1f > 1e-10 and n2f > 1e-10:
                cos_inv_full.append(np.dot(ds_f, -do_f) / (n1f * n2f))

        # 测试2: 范数比 ||δ_subj|| / ||δ_obj||
        norm_ratios = []
        for d in invertibility_data:
            ns = np.linalg.norm(d["delta_subj_perp"])
            no = np.linalg.norm(d["delta_obj_perp"])
            if no > 1e-10:
                norm_ratios.append(ns / no)

        # 测试3: rep(subj) + rep(obj) vs 2·rep(isolated)
        midpoint_errors = []
        for d in invertibility_data:
            noun = d["noun"]
            iso = noun_reps[noun]["isolated"]
            midpoint = (noun_reps[noun][f"subj_{d['verb']}"] + noun_reps[noun][f"obj_{d['verb']}"]) / 2
            err_perp = np.linalg.norm(proj_perp @ (midpoint - iso)) / (np.linalg.norm(proj_perp @ iso) + 1e-10)
            midpoint_errors.append(err_perp)

        results["invertibility"] = {
            "cos_subj_vs_neg_obj_perp": {
                "mean": float(np.mean(cos_inv_perp)),
                "std": float(np.std(cos_inv_perp)),
                "n": len(cos_inv_perp),
            },
            "cos_subj_vs_neg_obj_full": {
                "mean": float(np.mean(cos_inv_full)),
                "std": float(np.std(cos_inv_full)),
            },
            "norm_ratio_subj_obj": {
                "mean": float(np.mean(norm_ratios)),
                "std": float(np.std(norm_ratios)),
            },
            "midpoint_error_perp": {
                "mean": float(np.mean(midpoint_errors)),
                "std": float(np.std(midpoint_errors)),
            },
        }

        print(f"  cos(δ_subj^⊥, -δ_obj^⊥): mean={np.mean(cos_inv_perp):.4f}, std={np.std(cos_inv_perp):.4f}")
        print(f"  cos(δ_subj, -δ_obj):       mean={np.mean(cos_inv_full):.4f}, std={np.std(cos_inv_full):.4f}")
        print(f"  ||δ_subj^⊥|| / ||δ_obj^⊥||: mean={np.mean(norm_ratios):.4f}, std={np.std(norm_ratios):.4f}")
        print(f"  ||midpoint - isolated||^⊥ / ||isolated||^⊥: mean={np.mean(midpoint_errors):.4f}")

        # === 3B: d_syn的提取与泛化 ===
        print("\n--- d_syn Extraction and Generalization ---")

        # 从V_sem^⊥中的subj/obj数据提取d_syn
        all_subj_perp = []
        all_obj_perp = []
        all_nouns = []
        for d in invertibility_data:
            all_subj_perp.append(d["delta_subj_perp"])
            all_obj_perp.append(d["delta_obj_perp"])
            all_nouns.append(d["noun"])

        all_subj_perp = np.array(all_subj_perp)
        all_obj_perp = np.array(all_obj_perp)

        # d_syn = mean(rep_subj^⊥ - rep_obj^⊥) 归一化
        # 但我们用的是 δ_subj和δ_obj (减去了isolated), 所以 d_syn应该从原始subj/obj表示提取
        # 重新收集
        raw_subj_perp = []
        raw_obj_perp = []
        raw_nouns = []

        for noun, conds in noun_reps.items():
            for verb in TRANSITIVE_VERBS:
                sk = f"subj_{verb}"
                ok = f"obj_{verb}"
                if sk in conds and ok in conds:
                    raw_subj_perp.append(proj_perp @ conds[sk])
                    raw_obj_perp.append(proj_perp @ conds[ok])
                    raw_nouns.append(noun)

        raw_subj_perp = np.array(raw_subj_perp)
        raw_obj_perp = np.array(raw_obj_perp)

        # d_syn = mean(subj^⊥ - obj^⊥) 归一化
        d_role_perp = raw_subj_perp - raw_obj_perp
        d_syn_vec = d_role_perp.mean(axis=0)
        d_syn_norm = np.linalg.norm(d_syn_vec)
        if d_syn_norm > 1e-10:
            d_syn_unit = d_syn_vec / d_syn_norm
        else:
            d_syn_unit = np.zeros_like(d_syn_vec)

        # 用d_syn分类: 每个样本在d_syn上的投影
        subj_proj = np.array([np.dot(s, d_syn_unit) for s in raw_subj_perp])
        obj_proj = np.array([np.dot(o, d_syn_unit) for o in raw_obj_perp])

        # 分类阈值
        threshold = (np.mean(subj_proj) + np.mean(obj_proj)) / 2
        acc_subj = np.mean(subj_proj > threshold)
        acc_obj = np.mean(obj_proj < threshold)
        acc_total = (acc_subj * len(subj_proj) + acc_obj * len(obj_proj)) / (len(subj_proj) + len(obj_proj))

        print(f"  d_syn classification (all data): acc={acc_total:.4f}")
        print(f"    subj_proj: mean={np.mean(subj_proj):.2f}, std={np.std(subj_proj):.2f}")
        print(f"    obj_proj:  mean={np.mean(obj_proj):.2f}, std={np.std(obj_proj):.2f}")

        # Leave-one-noun-out 泛化测试
        unique_nouns = list(set(raw_nouns))
        if len(unique_nouns) >= 5:
            loo_accs = []
            for test_noun in unique_nouns:
                train_idx = [i for i, n in enumerate(raw_nouns) if n != test_noun]
                test_idx = [i for i, n in enumerate(raw_nouns) if n == test_noun]

                if len(train_idx) < 10 or len(test_idx) < 2:
                    continue

                # 训练: 从训练集提取d_syn
                train_d_role = raw_subj_perp[train_idx] - raw_obj_perp[train_idx]
                train_d_syn = train_d_role.mean(axis=0)
                train_norm = np.linalg.norm(train_d_syn)
                if train_norm < 1e-10:
                    continue
                train_d_syn_unit = train_d_syn / train_norm

                # 测试: 在测试集上分类
                test_subj_proj = np.array([np.dot(raw_subj_perp[i], train_d_syn_unit) for i in test_idx])
                test_obj_proj = np.array([np.dot(raw_obj_perp[i], train_d_syn_unit) for i in test_idx])

                test_threshold = (np.mean(test_subj_proj) + np.mean(test_obj_proj)) / 2
                test_acc_s = np.mean(test_subj_proj > test_threshold)
                test_acc_o = np.mean(test_obj_proj < test_threshold)
                test_acc = (test_acc_s + test_acc_o) / 2
                loo_accs.append(test_acc)

            if loo_accs:
                results["generalization"] = {
                    "loo_mean_acc": float(np.mean(loo_accs)),
                    "loo_std_acc": float(np.std(loo_accs)),
                    "n_test_nouns": len(loo_accs),
                }
                print(f"  Leave-one-noun-out generalization: mean={np.mean(loo_accs):.4f}, std={np.std(loo_accs):.4f}")

        # === 3C: d_syn的跨动词一致性 ===
        print("\n--- d_syn Cross-Verb Consistency ---")

        verb_d_syns = {}
        for verb in TRANSITIVE_VERBS:
            verb_subj = []
            verb_obj = []
            for i, n in enumerate(raw_nouns):
                # 需要重新匹配动词...
                pass

        # 简化: 从noun_reps中按动词提取
        verb_d_syns = {}
        for verb in TRANSITIVE_VERBS:
            sk = f"subj_{verb}"
            ok = f"obj_{verb}"
            subjs = []
            objs = []
            for noun, conds in noun_reps.items():
                if sk in conds and ok in conds:
                    subjs.append(proj_perp @ conds[sk])
                    objs.append(proj_perp @ conds[ok])

            if len(subjs) >= 5:
                d_v = np.array(subjs) - np.array(objs)
                d_v_mean = d_v.mean(axis=0)
                d_v_norm = np.linalg.norm(d_v_mean)
                if d_v_norm > 1e-10:
                    verb_d_syns[verb] = d_v_mean / d_v_norm

        if len(verb_d_syns) >= 2:
            verbs = list(verb_d_syns.keys())
            pairwise_cos = []
            for i in range(len(verbs)):
                for j in range(i + 1, len(verbs)):
                    cos = np.dot(verb_d_syns[verbs[i]], verb_d_syns[verbs[j]])
                    pairwise_cos.append(cos)

            results["cross_verb_consistency"] = {
                "mean_pairwise_cos": float(np.mean(pairwise_cos)),
                "std_pairwise_cos": float(np.std(pairwise_cos)),
                "n_verbs": len(verbs),
            }
            print(f"  Cross-verb d_syn consistency: mean_cos={np.mean(pairwise_cos):.4f}, std={np.std(pairwise_cos):.4f}")

    out_path = TEMP / f"ccxl_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp4: 多角色扩展 (主语/宾语/修饰语/孤立)
# ============================================================
def run_exp4(model_name):
    """
    扩展到3-4种语法角色:
    - 主语: "The [N1] [V]s the [N2]" → N1在主语位置
    - 宾语: "The [N2] [V]s the [N1]" → N1在宾语位置
    - 修饰语: "The [ADJ] [N1] [V]s the [N2]" → ADJ在修饰语位置
    - 孤立: "The word is [N1]" → N1无语法角色

    分析: 这4种角色在V_sem^⊥中是否形成可辨识的几何结构?
    """
    print(f"\n{'='*60}")
    print(f"CCXL Exp4: Multi-Role Geometry — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2

    # 收集4种角色的表示
    role_data = {"subject": [], "object": [], "modifier": [], "isolated": []}

    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")

        for verb in TRANSITIVE_VERBS[:3]:
            # 主语: noun_a作主语
            sent_subj = f"The {noun_a} {verb} the {noun_b}"
            pos_subj = find_noun_position(tokenizer, sent_subj, noun_a)
            if pos_subj is not None:
                inputs = tokenizer(sent_subj, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy()
                        role_data["subject"].append({"noun": noun_a, "verb": verb, "rep": rep})
                    except:
                        pass

            # 宾语: noun_a作宾语
            sent_obj = f"The {noun_b} {verb} the {noun_a}"
            pos_obj = find_noun_position(tokenizer, sent_obj, noun_a)
            if pos_obj is not None:
                inputs = tokenizer(sent_obj, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        out = model(inputs['input_ids'], output_hidden_states=True)
                        rep = out.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy()
                        role_data["object"].append({"noun": noun_a, "verb": verb, "rep": rep})
                    except:
                        pass

            # 修饰语: 形容词作修饰语
            for adj in ADJECTIVES[:3]:
                sent_mod = f"The {adj} {noun_a} {verb} the {noun_b}"
                pos_mod = find_noun_position(tokenizer, sent_mod, adj)
                if pos_mod is not None:
                    inputs = tokenizer(sent_mod, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            out = model(inputs['input_ids'], output_hidden_states=True)
                            rep = out.hidden_states[mid_layer][0, pos_mod, :].detach().cpu().float().numpy()
                            role_data["modifier"].append({"noun": adj, "verb": verb, "rep": rep})
                        except:
                            pass

        # 孤立
        for noun in [noun_a, noun_b]:
            prompt = f"The word is {noun}"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                try:
                    out = model(inputs['input_ids'], output_hidden_states=True)
                    last_pos = inputs['input_ids'].shape[1] - 1
                    rep = out.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
                    role_data["isolated"].append({"noun": noun, "verb": "none", "rep": rep})
                except:
                    pass

    for role, data in role_data.items():
        print(f"  {role}: {len(data)} samples")

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

    results = {"model": model_name, "exp": 4,
               "n_samples": {r: len(d) for r, d in role_data.items()}}

    # 分析: 在V_sem^⊥中, 4种角色的几何
    print("\n--- Multi-Role Geometry in V_sem^⊥ ---")

    # 4种角色的centroid(在V_sem^⊥中)
    centroids_perp = {}
    for role, data in role_data.items():
        if len(data) < 5:
            continue
        reps = np.array([d["rep"] for d in data])
        reps_perp = (proj_perp @ reps.T).T
        centroids_perp[role] = reps_perp.mean(axis=0)

    # Centroid间的距离
    roles = list(centroids_perp.keys())
    print(f"  Roles with data: {roles}")

    if len(roles) >= 2:
        # 成对距离
        dist_matrix = {}
        for i, r1 in enumerate(roles):
            for j, r2 in enumerate(roles):
                if i >= j:
                    continue
                dist = np.linalg.norm(centroids_perp[r1] - centroids_perp[r2])
                cos = np.dot(centroids_perp[r1], centroids_perp[r2]) / (
                    np.linalg.norm(centroids_perp[r1]) * np.linalg.norm(centroids_perp[r2]) + 1e-10)
                dist_matrix[f"{r1}_vs_{r2}"] = {
                    "distance": float(dist),
                    "cosine": float(cos),
                }
                print(f"  {r1} vs {r2}: dist={dist:.2f}, cos={cos:.4f}")

        results["centroid_distances"] = dist_matrix

    # 分类: 4类角色(如果数据够)
    if all(len(role_data[r]) >= 10 for r in ["subject", "object", "isolated"]):
        # 二分类: subj vs obj
        X_so = np.array([d["rep"] for d in role_data["subject"]] +
                        [d["rep"] for d in role_data["object"]])
        y_so = np.array([0] * len(role_data["subject"]) + [1] * len(role_data["object"]))

        X_so_perp = (proj_perp @ X_so.T).T
        pca_so = PCA(n_components=min(5, len(X_so) - 1))
        X_so_pca = pca_so.fit_transform(X_so_perp)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv_folds = min(5, min(len(role_data["subject"]), len(role_data["object"])) // 5)
        if cv_folds >= 2:
            cv_so = cross_val_score(clf, X_so_pca, y_so, cv=cv_folds, scoring='accuracy').mean()
        else:
            cv_so = 0.5

        # 三分类: subj vs obj vs isolated
        X_3 = np.array([d["rep"] for d in role_data["subject"]] +
                        [d["rep"] for d in role_data["object"]] +
                        [d["rep"] for d in role_data["isolated"]])
        y_3 = np.array([0] * len(role_data["subject"]) +
                       [1] * len(role_data["object"]) +
                       [2] * len(role_data["isolated"]))

        X_3_perp = (proj_perp @ X_3.T).T
        pca_3 = PCA(n_components=min(10, len(X_3) - 1))
        X_3_pca = pca_3.fit_transform(X_3_perp)

        cv_folds_3 = min(5, min(len(role_data["subject"]), len(role_data["object"]),
                                len(role_data["isolated"])) // 5)
        if cv_folds_3 >= 2:
            cv_3 = cross_val_score(clf, X_3_pca, y_3, cv=cv_folds_3, scoring='accuracy').mean()
        else:
            cv_3 = None

        results["classification"] = {
            "subj_vs_obj_cv": float(cv_so),
            "three_class_cv": float(cv_3) if cv_3 is not None else None,
        }

        print(f"\n  Classification in V_sem^⊥:")
        print(f"    Subj vs Obj: CV={cv_so:.3f}")
        if cv_3 is not None:
            print(f"    Subj vs Obj vs Isolated: CV={cv_3:.3f} (chance={1/3:.3f})")

    # 四分类(如果modifier数据够)
    if all(len(role_data[r]) >= 10 for r in ["subject", "object", "modifier", "isolated"]):
        X_4 = np.array([d["rep"] for d in role_data["subject"]] +
                        [d["rep"] for d in role_data["object"]] +
                        [d["rep"] for d in role_data["modifier"]] +
                        [d["rep"] for d in role_data["isolated"]])
        y_4 = np.array([0] * len(role_data["subject"]) +
                       [1] * len(role_data["object"]) +
                       [2] * len(role_data["modifier"]) +
                       [3] * len(role_data["isolated"]))

        X_4_perp = (proj_perp @ X_4.T).T
        pca_4 = PCA(n_components=min(10, len(X_4) - 1))
        X_4_pca = pca_4.fit_transform(X_4_perp)

        cv_folds_4 = min(5, min(len(role_data[r]) for r in ["subject", "object", "modifier", "isolated"]) // 5)
        if cv_folds_4 >= 2:
            cv_4 = cross_val_score(clf, X_4_pca, y_4, cv=cv_folds_4, scoring='accuracy').mean()
            results["classification"]["four_class_cv"] = float(cv_4)
            print(f"    Subj vs Obj vs Mod vs Isolated: CV={cv_4:.3f} (chance={1/4:.3f})")

    out_path = TEMP / f"ccxl_exp4_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        run_exp1(args.model)
    elif args.exp == 2:
        run_exp2(args.model)
    elif args.exp == 3:
        run_exp3(args.model)
    elif args.exp == 4:
        run_exp4(args.model)
