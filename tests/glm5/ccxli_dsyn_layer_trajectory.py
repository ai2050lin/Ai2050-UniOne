"""
CCXLI(401): d_syn层轨迹 — 语法主语标记如何逐层形成？

核心问题:
  CCXL发现d_syn是1维的"主语标记方向"，α(主语偏移)≈30(Qwen3)
  但: d_syn方向在各层是否相同？主语标记何时出现？

实验设计:
  Exp1: 逐层提取d_syn，追踪其方向一致性
    - 在每一层提取d_syn = mean(subj^⊥ - obj^⊥)归一化
    - 计算相邻层d_syn的cos → 方向是否稳定？
    - 计算d_syn与L_mid层d_syn的cos → 方向是否收敛？

  Exp2: 逐层追踪α(主语偏移幅度)
    - 在每一层，每个名词的主语/宾语投影到d_syn_mid
    - 追踪α随层数的变化 → 主语标记何时出现？
    - 对比: L0(嵌入层)的α ≈ 0？哪层开始显著？

  Exp3: d_syn与位置编码的关系
    - 位置编码是否包含d_syn信息？
    - 纯位置嵌入的PCA是否有d_syn方向？
    - 用2×2因子设计分离: 位置贡献 vs 角色贡献
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


def compute_perp_basis(V_sem_5):
    d = V_sem_5.shape[1]
    proj_vsem = V_sem_5.T @ V_sem_5
    proj_perp = np.eye(d) - proj_vsem
    return proj_perp


# ============================================================
# Exp1+2: 逐层d_syn轨迹 (合并以节省GPU时间)
# ============================================================
def run_exp12(model_name):
    """
    逐层提取d_syn方向和α(主语偏移幅度)
    """
    print(f"\n{'='*60}")
    print(f"CCXLI Exp1+2: Layer-wise d_syn Trajectory — {model_name}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model

    # 采样层(避免每层都做，太慢)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 18))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    # V_sem PCA (用中间层)
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

    # 收集逐层同句改写表示
    print("Collecting layer-wise syntactic representations...")

    # 存储结构: {layer: {noun_verb_role: rep}}
    all_reps = {l: {} for l in sample_layers}

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

                    for l in sample_layers:
                        rep_subj = out_subj.hidden_states[l][0, pos_subj, :].detach().cpu().float().numpy()
                        rep_obj = out_obj.hidden_states[l][0, pos_obj, :].detach().cpu().float().numpy()

                        all_reps[l][f"{noun_a}_{verb}_subj"] = rep_subj
                        all_reps[l][f"{noun_a}_{verb}_obj"] = rep_obj
                except:
                    pass

    # 逐层提取d_syn
    print("\n--- Layer-wise d_syn Trajectory ---")

    layer_d_syns = {}  # {layer: d_syn_unit_vector}

    for l in sample_layers:
        data = all_reps[l]

        # 收集配对的主语/宾语
        subj_reps = []
        obj_reps = []

        for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
            for verb in TRANSITIVE_VERBS:
                sk = f"{noun_a}_{verb}_subj"
                ok = f"{noun_a}_{verb}_obj"
                if sk in data and ok in data:
                    subj_reps.append(data[sk])
                    obj_reps.append(data[ok])

        if len(subj_reps) < 10:
            continue

        subj_reps = np.array(subj_reps)
        obj_reps = np.array(obj_reps)

        # 投影到V_sem^⊥
        subj_perp = (proj_perp @ subj_reps.T).T
        obj_perp = (proj_perp @ obj_reps.T).T

        # d_syn = mean(subj - obj) 归一化
        d_role = subj_perp - obj_perp
        d_syn = d_role.mean(axis=0)
        d_syn_norm = np.linalg.norm(d_syn)

        if d_syn_norm > 1e-10:
            d_syn_unit = d_syn / d_syn_norm
        else:
            d_syn_unit = np.zeros_like(d_syn)

        layer_d_syns[l] = d_syn_unit

    # 分析d_syn方向一致性
    results = {"model": model_name, "exp": "1+2", "n_layers": n_layers}

    # 参考方向: 中间层的d_syn
    ref_layer = mid_layer
    if ref_layer not in layer_d_syns:
        ref_layer = min(layer_d_syns.keys(), key=lambda x: abs(x - mid_layer))

    d_syn_ref = layer_d_syns.get(ref_layer)

    print(f"  {'Layer':>5} | {'cos_to_ref':>10} | {'||d_syn||':>10} | {'α_subj':>8} | {'α_obj':>8} | {'α_ratio':>8} | {'CV_acc':>6}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    trajectory = []

    for l in sample_layers:
        if l not in layer_d_syns or l not in all_reps:
            continue

        data = all_reps[l]
        d_syn_l = layer_d_syns[l]

        # cos with reference
        if d_syn_ref is not None:
            cos_ref = float(np.dot(d_syn_l, d_syn_ref))
        else:
            cos_ref = 0.0

        # 提取α和CV
        subj_reps = []
        obj_reps = []
        for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
            for verb in TRANSITIVE_VERBS:
                sk = f"{noun_a}_{verb}_subj"
                ok = f"{noun_a}_{verb}_obj"
                if sk in data and ok in data:
                    subj_reps.append(data[sk])
                    obj_reps.append(data[ok])

        if len(subj_reps) < 10:
            continue

        subj_reps = np.array(subj_reps)
        obj_reps = np.array(obj_reps)

        subj_perp = (proj_perp @ subj_reps.T).T
        obj_perp = (proj_perp @ obj_reps.T).T

        # 投影到d_syn方向
        subj_proj = np.array([np.dot(s, d_syn_l) for s in subj_perp])
        obj_proj = np.array([np.dot(o, d_syn_l) for o in obj_perp])

        alpha_subj = float(np.mean(subj_proj))
        alpha_obj = float(np.mean(obj_proj))

        # 分类准确率
        X_role = np.vstack([subj_perp, obj_perp])
        y_role = np.array([0] * len(subj_perp) + [1] * len(obj_perp))

        # 用d_syn方向1维分类
        all_proj = np.concatenate([subj_proj, obj_proj])
        threshold = (np.mean(subj_proj) + np.mean(obj_proj)) / 2
        pred_subj = np.mean(subj_proj > threshold)
        pred_obj = np.mean(obj_proj < threshold)
        cv_1d = float((pred_subj + pred_obj) / 2)

        # d_syn norm
        d_role = subj_perp - obj_perp
        d_syn_norm_l = float(np.linalg.norm(d_role.mean(axis=0)))

        entry = {
            "layer": l, "cos_to_ref": cos_ref,
            "d_syn_norm": d_syn_norm_l,
            "alpha_subj": alpha_subj, "alpha_obj": alpha_obj,
            "alpha_ratio": abs(alpha_subj) / (abs(alpha_obj) + 1e-10),
            "cv_1d": cv_1d, "n_samples": len(subj_reps),
        }
        trajectory.append(entry)

        print(f"  L{l:>4} | {cos_ref:>10.4f} | {d_syn_norm_l:>10.2f} | {alpha_subj:>8.2f} | {alpha_obj:>8.2f} | {abs(alpha_subj)/(abs(alpha_obj)+1e-10):>8.2f} | {cv_1d:>6.3f}")

    results["trajectory"] = trajectory

    # 分析关键拐点
    # 找α_subj首次显著>0的层
    for entry in trajectory:
        if entry["alpha_subj"] > 1.0 and entry["cv_1d"] > 0.9:
            results["subject_marking_onset"] = {
                "layer": entry["layer"],
                "alpha_subj": entry["alpha_subj"],
                "cv_1d": entry["cv_1d"],
            }
            print(f"\n  ★ Subject marking onset: L{entry['layer']} (α_subj={entry['alpha_subj']:.2f}, CV={entry['cv_1d']:.3f})")
            break

    # 相邻层d_syn方向一致性
    if len(trajectory) >= 2:
        adj_cos = []
        for i in range(len(trajectory) - 1):
            l1 = trajectory[i]["layer"]
            l2 = trajectory[i + 1]["layer"]
            if l1 in layer_d_syns and l2 in layer_d_syns:
                cos = float(np.dot(layer_d_syns[l1], layer_d_syns[l2]))
                adj_cos.append({"from": l1, "to": l2, "cos": cos})

        results["adjacent_layer_cos"] = adj_cos

        print(f"\n  Adjacent layer d_syn consistency:")
        for ac in adj_cos:
            print(f"    L{ac['from']}→L{ac['to']}: cos={ac['cos']:.4f}")

    # 三段分析
    early = [e for e in trajectory if e["layer"] < n_layers // 3]
    mid = [e for e in trajectory if n_layers // 3 <= e["layer"] < 2 * n_layers // 3]
    late = [e for e in trajectory if e["layer"] >= 2 * n_layers // 3]

    for name, seg in [("Early", early), ("Mid", mid), ("Late", late)]:
        if seg:
            results[f"{name}_summary"] = {
                "mean_cos_ref": float(np.mean([e["cos_to_ref"] for e in seg])),
                "mean_alpha_subj": float(np.mean([e["alpha_subj"] for e in seg])),
                "mean_alpha_obj": float(np.mean([e["alpha_obj"] for e in seg])),
                "mean_cv_1d": float(np.mean([e["cv_1d"] for e in seg])),
            }

    out_path = TEMP / f"ccxli_exp12_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp3: d_syn与位置编码的关系
# ============================================================
def run_exp3(model_name):
    """
    分析位置编码是否包含d_syn信息
    """
    print(f"\n{'='*60}")
    print(f"CCXLI Exp3: d_syn vs Position Encoding — {model_name}")
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

    # 1. 提取d_syn (参考方向)
    print("Extracting d_syn reference...")
    subj_reps_ref = []
    obj_reps_ref = []

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

                    subj_reps_ref.append(rep_subj)
                    obj_reps_ref.append(rep_obj)
                except:
                    pass

    subj_perp_ref = (proj_perp @ np.array(subj_reps_ref).T).T
    obj_perp_ref = (proj_perp @ np.array(obj_reps_ref).T).T
    d_syn_ref = (subj_perp_ref - obj_perp_ref).mean(axis=0)
    d_syn_norm = np.linalg.norm(d_syn_ref)
    d_syn_unit = d_syn_ref / d_syn_norm if d_syn_norm > 1e-10 else d_syn_ref

    # 2. 位置编码与d_syn
    print("\n--- Position Encoding vs d_syn ---")

    # 2a. 纯位置编码: 用相同token在不同位置
    # "The cat" at position 1 vs "The cat" at position 4
    # 构造不同长度的句子来控制token位置

    # 简化测试: 用固定模板，测量不同位置的token表示差异
    pos_data = {}  # {position: [reps]}

    for noun_a, noun_b in NOUN_PAIRS[:5]:
        for verb in TRANSITIVE_VERBS[:2]:
            # cat在主语位置(pos1) vs cat在宾语位置(pos4)
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

                    # 收集L0(纯嵌入层, 含位置编码)的表示
                    rep_subj_L0 = out_subj.hidden_states[0][0, pos_subj, :].detach().cpu().float().numpy()
                    rep_obj_L0 = out_obj.hidden_states[0][0, pos_obj, :].detach().cpu().float().numpy()

                    # 位置差异
                    pos_diff = rep_subj_L0 - rep_obj_L0
                    pos_diff_perp = proj_perp @ pos_diff

                    # cos with d_syn
                    cos_pos_dsyn = float(np.dot(pos_diff_perp, d_syn_unit) / (np.linalg.norm(pos_diff_perp) + 1e-10))

                    if "cos_pos_dsyn" not in pos_data:
                        pos_data["cos_pos_dsyn"] = []
                    pos_data["cos_pos_dsyn"].append(cos_pos_dsyn)

                    # 位置差异的范数
                    if "pos_diff_norm" not in pos_data:
                        pos_data["pos_diff_norm"] = []
                    pos_data["pos_diff_norm"].append(np.linalg.norm(pos_diff_perp))

                except:
                    pass

    results = {"model": model_name, "exp": 3}

    if pos_data.get("cos_pos_dsyn"):
        results["position_vs_dsyn"] = {
            "cos_pos_diff_with_dsyn": {
                "mean": float(np.mean(pos_data["cos_pos_dsyn"])),
                "std": float(np.std(pos_data["cos_pos_dsyn"])),
                "n": len(pos_data["cos_pos_dsyn"]),
            },
            "pos_diff_norm_perp": {
                "mean": float(np.mean(pos_data["pos_diff_norm"])),
                "std": float(np.std(pos_data["pos_diff_norm"])),
            },
        }

        print(f"  cos(position_diff^⊥, d_syn): mean={np.mean(pos_data['cos_pos_dsyn']):.4f}, std={np.std(pos_data['cos_pos_dsyn']):.4f}")
        print(f"  ||position_diff^⊥||: mean={np.mean(pos_data['pos_diff_norm']):.2f}")

    # 2b. 用2×2因子设计分离位置vs角色
    print("\n--- 2×2 Factor Design: Position vs Role ---")

    conditions = ["subj_early", "subj_late", "obj_early", "obj_late"]
    factor_data = {c: [] for c in conditions}

    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS[:10]):
        for verb in TRANSITIVE_VERBS[:2]:
            for noun, other in [(noun_a, noun_b), (noun_b, noun_a)]:
                sentences = {
                    "subj_early": f"The {noun} that {verb} the {other} runs",
                    "subj_late": f"The {other} that the {noun} {verb} runs",
                    "obj_early": f"The {noun} that the {other} {verb} runs",
                    "obj_late": f"The {other} {verb} the {noun} that runs",
                }

                for cond, sentence in sentences.items():
                    noun_pos = find_noun_position(tokenizer, sentence, noun)
                    if noun_pos is None:
                        continue

                    inputs = tokenizer(sentence, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            out = model(inputs['input_ids'], output_hidden_states=True)
                            # 在mid_layer收集
                            rep = out.hidden_states[mid_layer][0, noun_pos, :].detach().cpu().float().numpy()
                            factor_data[cond].append({
                                "noun": noun, "verb": verb,
                                "rep": rep, "pos": noun_pos,
                            })
                        except:
                            pass

    # 2×2分析
    valid_pairs = []
    for noun in set(d["noun"] for c in factor_data.values() for d in c):
        for verb in TRANSITIVE_VERBS[:2]:
            reps = {}
            for c in conditions:
                matches = [d for d in factor_data[c] if d["noun"] == noun and d["verb"] == verb]
                if matches:
                    reps[c] = matches[0]["rep"]
            if len(reps) == 4:
                valid_pairs.append(reps)

    if len(valid_pairs) >= 5:
        role_deltas = []
        pos_deltas = []

        for reps in valid_pairs:
            se, sl, oe, ol = reps["subj_early"], reps["subj_late"], reps["obj_early"], reps["obj_late"]

            d_role = (se + sl - oe - ol) / 2
            d_pos = (se + oe - sl - ol) / 2

            role_deltas.append(d_role)
            pos_deltas.append(d_pos)

        role_deltas = np.array(role_deltas)
        pos_deltas = np.array(pos_deltas)

        # 投影到V_sem^⊥
        role_deltas_perp = (proj_perp @ role_deltas.T).T
        pos_deltas_perp = (proj_perp @ pos_deltas.T).T

        # cos with d_syn
        cos_role_dsyn = [float(np.dot(d, d_syn_unit) / (np.linalg.norm(d) + 1e-10)) for d in role_deltas_perp]
        cos_pos_dsyn = [float(np.dot(d, d_syn_unit) / (np.linalg.norm(d) + 1e-10)) for d in pos_deltas_perp]

        # cos between role and pos deltas
        cos_role_pos = [float(np.dot(r, p) / (np.linalg.norm(r) * np.linalg.norm(p) + 1e-10))
                       for r, p in zip(role_deltas_perp, pos_deltas_perp)]

        results["factor_analysis"] = {
            "n_valid": len(valid_pairs),
            "cos_role_delta_with_dsyn": {
                "mean": float(np.mean(cos_role_dsyn)),
                "std": float(np.std(cos_role_dsyn)),
            },
            "cos_pos_delta_with_dsyn": {
                "mean": float(np.mean(cos_pos_dsyn)),
                "std": float(np.std(cos_pos_dsyn)),
            },
            "cos_role_vs_pos_delta": {
                "mean": float(np.mean(cos_role_pos)),
                "std": float(np.std(cos_role_pos)),
            },
        }

        print(f"  cos(δ_role^⊥, d_syn): mean={np.mean(cos_role_dsyn):.4f}")
        print(f"  cos(δ_pos^⊥, d_syn):  mean={np.mean(cos_pos_dsyn):.4f}")
        print(f"  cos(δ_role^⊥, δ_pos^⊥): mean={np.mean(cos_role_pos):.4f}")

    out_path = TEMP / f"ccxli_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[12, 3])
    args = parser.parse_args()

    if args.exp == 12:
        run_exp12(args.model)
    elif args.exp == 3:
        run_exp3(args.model)
