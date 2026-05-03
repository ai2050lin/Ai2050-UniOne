"""
CCXLIII(403): 被动语态分离 + 模型内部Hook因果干预 + 多角色扩展

核心问题:
  1. d_syn编码的是"语法角色"还是"位置"?
     - 主动: "The cat chases the dog" → cat=subj@pos1
     - 被动: "The dog is chased by the cat" → cat=by-obj@pos5
     - 如果d_syn编码角色: cat的d_syn投影应该随句法角色改变
     - 如果d_syn编码位置: cat的d_syn投影应该与位置相关

  2. 模型内部hook干预: 在forward中注入d_syn, 看输出是否改变
     - Exp3只修改了表示空间, 没有修改模型forward
     - 需要在模型中间层hook注入d_syn, 看输出logit变化

  3. 多角色扩展:
     - 主语/宾语只是2种角色
     - 修饰语: "The red cat chases the dog"
     - 间接宾语: "The cat gives the dog the fish"
     - 介词宾语: "The cat looks at the dog"

实验设计:
  Exp1: 被动语态d_syn编码
    - 对比主动/被动句中同一名词的d_syn投影
    - 如果cat在主动句(主语)的投影>0, 被动句(介词宾语)投影≈0 → 证明d_syn编码角色

  Exp2: 模型内部Hook因果干预
    - 在中间层forward时注入+/-d_syn到目标位置
    - 测量输出logit的变化(概率偏移)
    - 对比: d_syn vs 随机方向 vs 语义方向

  Exp3: 多角色d_syn结构
    - 4种角色: 主语/宾语/修饰语/介词宾语
    - 提取每种角色的表示, 在V_sem^⊥中分析
    - 是否存在多个独立的语法方向?
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


def extract_d_syn_from_reps(subj_reps, obj_reps, proj_perp):
    """从已收集的主语/宾语表示提取d_syn"""
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
# Exp1: 被动语态d_syn编码
# ============================================================
def run_exp1(model_name):
    """
    核心实验: 分离位置和语法角色

    主动句: "The cat chases the dog"
      cat = 主语 @ pos1 → d_syn投影应该>0
      dog = 宾语 @ pos4 → d_syn投影应该≈0

    被动句: "The dog is chased by the cat"
      dog = 主语 @ pos1 → d_syn投影应该>0
      cat = 介词宾语 @ pos5 → d_syn投影应该≈0 or <0

    如果d_syn编码位置: cat@pos1和cat@pos5的投影都>0 (主语总是在pos1)
    如果d_syn编码角色: cat@pos1投影>0, cat@pos5投影≈0 (角色变了)
    """
    print(f"\n{'='*60}")
    print(f"CCXLIII Exp1: Passive Voice — Position vs Role — {model_name}")
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

    # 从主动句提取d_syn
    print("Extracting d_syn from active sentences...")
    subj_reps_active = []
    obj_reps_active = []

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
                    rep_subj = out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy()
                    rep_obj = out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy()
                    subj_reps_active.append(rep_subj)
                    obj_reps_active.append(rep_obj)
                except:
                    pass

    d_syn_unit, d_syn_norm = extract_d_syn_from_reps(subj_reps_active, obj_reps_active, proj_perp)
    print(f"  ||d_syn|| = {d_syn_norm:.2f}")

    # ==========================================
    # 核心: 被动语态测试
    # ==========================================
    print("\n--- Passive Voice d_syn Projection ---")

    results = {
        "model": model_name, "exp": 1,
        "d_syn_norm": float(d_syn_norm),
        "active_subj_projections": [],
        "active_obj_projections": [],
        "passive_subj_projections": [],
        "passive_byobj_projections": [],
    }

    test_pairs = NOUN_PAIRS[:15]
    test_verbs = TRANSITIVE_VERBS[:5]

    for noun_a, noun_b in test_pairs:
        for i, verb in enumerate(test_verbs):
            pverb = PASSIVE_VERBS[i]

            # 主动句: "The cat chases the dog"
            sent_active = f"The {noun_a} {verb} the {noun_b}"
            pos_a_active = find_noun_position(tokenizer, sent_active, noun_a)  # 主语位置
            pos_b_active = find_noun_position(tokenizer, sent_active, noun_b)  # 宾语位置

            # 被动句: "The dog is chased by the cat"
            sent_passive = f"The {noun_b} is {pverb} by the {noun_a}"
            pos_b_passive = find_noun_position(tokenizer, sent_passive, noun_b)  # 主语位置
            pos_a_passive = find_noun_position(tokenizer, sent_passive, noun_a)  # by-宾语位置

            if pos_a_active is None or pos_b_active is None or pos_b_passive is None or pos_a_passive is None:
                continue

            inputs_active = tokenizer(sent_active, return_tensors='pt').to(device)
            inputs_passive = tokenizer(sent_passive, return_tensors='pt').to(device)

            with torch.no_grad():
                try:
                    # 主动句表示
                    out_active = model(inputs_active['input_ids'], output_hidden_states=True)
                    rep_a_active = out_active.hidden_states[mid_layer][0, pos_a_active, :].detach().cpu().float().numpy()
                    rep_b_active = out_active.hidden_states[mid_layer][0, pos_b_active, :].detach().cpu().float().numpy()

                    # 被动句表示
                    out_passive = model(inputs_passive['input_ids'], output_hidden_states=True)
                    rep_b_passive = out_passive.hidden_states[mid_layer][0, pos_b_passive, :].detach().cpu().float().numpy()
                    rep_a_passive = out_passive.hidden_states[mid_layer][0, pos_a_passive, :].detach().cpu().float().numpy()

                    # 在V_sem^⊥中投影到d_syn
                    a_active_perp = proj_perp @ rep_a_active
                    a_active_proj = np.dot(a_active_perp, d_syn_unit) * d_syn_norm

                    b_active_perp = proj_perp @ rep_b_active
                    b_active_proj = np.dot(b_active_perp, d_syn_unit) * d_syn_norm

                    b_passive_perp = proj_perp @ rep_b_passive
                    b_passive_proj = np.dot(b_passive_perp, d_syn_unit) * d_syn_norm

                    a_passive_perp = proj_perp @ rep_a_passive
                    a_passive_proj = np.dot(a_passive_perp, d_syn_unit) * d_syn_norm

                    results["active_subj_projections"].append(float(a_active_proj))
                    results["active_obj_projections"].append(float(b_active_proj))
                    results["passive_subj_projections"].append(float(b_passive_proj))
                    results["passive_byobj_projections"].append(float(a_passive_proj))

                except Exception as e:
                    pass

    # 统计
    active_subj = np.array(results["active_subj_projections"])
    active_obj = np.array(results["active_obj_projections"])
    passive_subj = np.array(results["passive_subj_projections"])
    passive_byobj = np.array(results["passive_byobj_projections"])

    if len(active_subj) > 0:
        print(f"\n  ★ Active sentence:")
        print(f"    noun_a (subj@pos1) d_syn proj: mean={active_subj.mean():.2f}, std={active_subj.std():.2f}")
        print(f"    noun_b (obj@pos4)  d_syn proj: mean={active_obj.mean():.2f}, std={active_obj.std():.2f}")

        print(f"\n  ★ Passive sentence:")
        print(f"    noun_b (subj@pos1)     d_syn proj: mean={passive_subj.mean():.2f}, std={passive_subj.std():.2f}")
        print(f"    noun_a (by-obj@pos5)  d_syn proj: mean={passive_byobj.mean():.2f}, std={passive_byobj.std():.2f}")

        # ★★★ 关键判据
        print(f"\n  ★★★ Position vs Role Test:")
        # 同一名词(noun_a)在主动(主语) vs 被动(by-宾语)的d_syn投影变化
        # 如果编码位置: noun_a的投影应该只与pos相关(pos1 vs pos5)
        # 如果编码角色: noun_a的投影应该与角色相关(主语 vs by-宾语)
        if passive_byobj.mean() < 0:
            ratio = abs(active_subj.mean() / (passive_byobj.mean() + 1e-10))
            print(f"    noun_a: active_subj={active_subj.mean():.2f} vs passive_byobj={passive_byobj.mean():.2f}")
            print(f"    ★ ROLE ENCODING: d_syn flips sign when role changes! ratio={ratio:.1f}")
        else:
            ratio = active_subj.mean() / (passive_byobj.mean() + 1e-10)
            print(f"    noun_a: active_subj={active_subj.mean():.2f} vs passive_byobj={passive_byobj.mean():.2f}")
            if ratio > 2:
                print(f"    ★ PARTIAL ROLE ENCODING: d_syn decreases but doesn't flip. ratio={ratio:.1f}")
            else:
                print(f"    ✗ POSITION ENCODING: d_syn similar for same noun. ratio={ratio:.1f}")

        # 被动句主语的投影应该>0 (与主动句主语类似)
        # 被动句by-宾语的投影应该≈0 (与主动句宾语类似)
        # 用分类器验证
        from sklearn.linear_model import LogisticRegression
        X_subj = np.vstack([
            active_subj.reshape(-1, 1),
            passive_subj.reshape(-1, 1),
        ])
        X_obj = np.vstack([
            active_obj.reshape(-1, 1),
            passive_byobj.reshape(-1, 1),
        ])
        y_subj = np.ones(len(X_subj))
        y_obj = np.zeros(len(X_obj))

        X = np.vstack([X_subj, X_obj])
        y = np.hstack([y_subj, y_obj])

        if len(X) > 10:
            clf = LogisticRegression()
            scores = cross_val_score(clf, X, y, cv=min(5, len(X)//5), scoring='accuracy')
            print(f"\n    d_syn projection → role classification CV_acc = {scores.mean():.3f} ± {scores.std():.3f}")
            print(f"    (chance = 0.500)")
            results["role_cv_from_projection"] = float(scores.mean())

        results["active_subj_mean"] = float(active_subj.mean())
        results["active_obj_mean"] = float(active_obj.mean())
        results["passive_subj_mean"] = float(passive_subj.mean())
        results["passive_byobj_mean"] = float(passive_byobj.mean())

    out_path = TEMP / f"ccxliii_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp2: 模型内部Hook因果干预
# ============================================================
def run_exp2(model_name):
    """
    在模型forward过程中注入d_syn, 直接测试对输出的影响

    方法:
    1. 取一个主动句 "The cat chases the dog"
    2. 基线: 正常forward, 记录输出logits
    3. 干预: 在mid_layer的dog位置注入+β·d_syn
       → 如果d_syn编码主语, dog应该被"当成主语"
       → 模型输出应该变化(预测不同的后续词)
    4. 对照: 注入随机方向(同样幅度)
    """
    print(f"\n{'='*60}")
    print(f"CCXLIII Exp2: Hook Causal Intervention — {model_name}")
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

    # 提取d_syn
    print("Extracting d_syn...")
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

    # 随机方向对照
    rng = np.random.RandomState(42)
    random_dir = rng.randn(d_model)
    random_dir_perp = proj_perp @ random_dir
    random_dir_norm = np.linalg.norm(random_dir_perp)
    if random_dir_norm > 1e-10:
        random_dir_unit = random_dir_perp / random_dir_norm
    else:
        random_dir_unit = np.zeros_like(random_dir_perp)

    # V_sem方向对照(第一个PC)
    vsem_dir = V_sem_5[0]
    vsem_dir_norm = np.linalg.norm(vsem_dir)
    if vsem_dir_norm > 1e-10:
        vsem_dir_unit = vsem_dir / vsem_dir_norm
    else:
        vsem_dir_unit = np.zeros_like(vsem_dir)

    # ==========================================
    # Hook干预测试
    # ==========================================
    print("\n--- Hook Intervention Test ---")

    results = {
        "model": model_name, "exp": 2,
        "d_syn_norm": float(d_syn_norm),
    }

    betas = [0.5, 1.0, 2.0]
    test_pairs = NOUN_PAIRS[:10]
    test_verbs = TRANSITIVE_VERBS[:3]

    for direction_name, direction_unit in [("d_syn", d_syn_unit), ("random", random_dir_unit), ("vsem_pc0", vsem_dir_unit)]:
        print(f"\n  Direction: {direction_name}")
        direction_results = []

        for noun_a, noun_b in test_pairs:
            for verb in test_verbs:
                sentence = f"The {noun_a} {verb} the {noun_b}"
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                input_ids = inputs['input_ids']

                # 找宾语位置(注入目标)
                obj_pos = find_noun_position(tokenizer, sentence, noun_b)
                if obj_pos is None:
                    continue

                # 基线
                with torch.no_grad():
                    out_base = model(input_ids, output_hidden_states=True)
                    logits_base = out_base.logits[0, -1, :].detach().cpu().float()
                    probs_base = torch.softmax(logits_base, dim=-1).numpy()

                for beta in betas:
                    injection_strength = beta * d_syn_norm

                    # Hook注入
                    def make_hook(target_pos, direction, strength):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                hidden = output[0].clone()
                                dir_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
                                hidden[0, target_pos, :] += strength * dir_tensor
                                return (hidden,) + output[1:]
                            return output
                        return hook

                    layers = list(get_layers(model))
                    hook = layers[mid_layer].register_forward_hook(
                        make_hook(obj_pos, direction_unit, injection_strength)
                    )

                    with torch.no_grad():
                        try:
                            out_int = model(input_ids, output_hidden_states=True)
                            logits_int = out_int.logits[0, -1, :].detach().cpu().float()
                            probs_int = torch.softmax(logits_int, dim=-1).numpy()

                            # KL散度
                            kl_div = float(np.sum(probs_base * np.log(probs_base / (probs_int + 1e-10) + 1e-10)))

                            # Top-1预测变化
                            top1_base_id = np.argmax(probs_base)
                            top1_int_id = np.argmax(probs_int)
                            top1_changed = int(top1_base_id != top1_int_id)

                            # 主语相关词的概率变化
                            # 在SVO句中, 模型倾向于预测与主语相关的词
                            # 如果注入+d_syn到宾语位置, 主语相关词的概率应该增加
                            subj_pos = find_noun_position(tokenizer, sentence, noun_a)
                            if subj_pos is not None:
                                # 取noun_a的token id
                                subj_token_ids = tokenizer.encode(" " + noun_a, add_special_tokens=False)
                                obj_token_ids = tokenizer.encode(" " + noun_b, add_special_tokens=False)
                                verb_token_ids = tokenizer.encode(" " + verb, add_special_tokens=False)

                                # 主语/宾语/动词的token概率
                                subj_prob_base = sum(probs_base[tid] for tid in subj_token_ids if tid < len(probs_base))
                                subj_prob_int = sum(probs_int[tid] for tid in subj_token_ids if tid < len(probs_int))
                                obj_prob_base = sum(probs_base[tid] for tid in obj_token_ids if tid < len(probs_base))
                                obj_prob_int = sum(probs_int[tid] for tid in obj_token_ids if tid < len(probs_int))

                                direction_results.append({
                                    "beta": beta,
                                    "kl_div": kl_div,
                                    "top1_changed": top1_changed,
                                    "subj_prob_base": float(subj_prob_base),
                                    "subj_prob_int": float(subj_prob_int),
                                    "obj_prob_base": float(obj_prob_base),
                                    "obj_prob_int": float(obj_prob_int),
                                })
                            else:
                                direction_results.append({
                                    "beta": beta,
                                    "kl_div": kl_div,
                                    "top1_changed": top1_changed,
                                })

                        except:
                            pass

                    hook.remove()

        # 汇总
        if direction_results:
            for beta in betas:
                beta_res = [r for r in direction_results if r["beta"] == beta]
                if beta_res:
                    mean_kl = np.mean([r["kl_div"] for r in beta_res])
                    mean_top1_change = np.mean([r["top1_changed"] for r in beta_res])
                    print(f"    β={beta:.1f}: KL={mean_kl:.4f}, top1_change={mean_top1_change:.3f}")

                    if "subj_prob_base" in beta_res[0]:
                        mean_subj_base = np.mean([r["subj_prob_base"] for r in beta_res])
                        mean_subj_int = np.mean([r["subj_prob_int"] for r in beta_res])
                        mean_obj_base = np.mean([r["obj_prob_base"] for r in beta_res])
                        mean_obj_int = np.mean([r["obj_prob_int"] for r in beta_res])
                        print(f"           subj_prob: {mean_subj_base:.6f}→{mean_subj_int:.6f}, "
                              f"obj_prob: {mean_obj_base:.6f}→{mean_obj_int:.6f}")

            results[f"{direction_name}_intervention"] = direction_results

    out_path = TEMP / f"ccxliii_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp3: 多角色d_syn结构
# ============================================================
def run_exp3(model_name):
    """
    扩展到更多语法角色:
    1. 主语: "The cat chases the dog" → cat@pos1
    2. 宾语: "The cat chases the dog" → dog@pos4
    3. 修饰语: "The red cat chases the dog" → red@pos1
    4. 介词宾语: "The cat looks at the dog" → dog@pos5
    5. 孤立: "The word is cat" → cat@last

    问题:
    - 是否存在多个独立的d_syn方向?
    - 4+种角色能否用1维d_syn编码?
    - 还是需要多维子空间?
    """
    print(f"\n{'='*60}")
    print(f"CCXLIII Exp3: Multi-Role d_syn Structure — {model_name}")
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

    # 收集5种角色的表示
    print("Collecting multi-role representations...")
    role_reps = {
        "subject": [],
        "object": [],
        "modifier": [],
        "prep_object": [],
        "isolated": [],
    }

    test_nouns = ["cat", "dog", "bird", "fish", "lion", "tiger", "horse", "wolf", "king", "queen",
                  "hammer", "knife", "rain", "snow", "wood", "stone"]

    for noun in test_nouns:
        # 主语: "The noun chases the dog"
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

        # 宾语: "The dog chases the noun"
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

        # 修饰语: "The red noun chases the dog"
        for adj in ADJECTIVES[:3]:
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

        # 介词宾语: "The cat looks at the noun"
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

        # 孤立: "The word is noun"
        sent = f"The word is {noun}"
        pos = find_noun_position(tokenizer, sent, noun)
        if pos is not None:
            inputs = tokenizer(sent, return_tensors='pt').to(device)
            with torch.no_grad():
                try:
                    out = model(inputs['input_ids'], output_hidden_states=True)
                    rep = out.hidden_states[mid_layer][0, pos, :].detach().cpu().float().numpy()
                    role_reps["isolated"].append(rep)
                except:
                    pass

    for role, reps in role_reps.items():
        print(f"  {role}: {len(reps)} representations")

    # ==========================================
    # 分析: V_sem^⊥中的多角色结构
    # ==========================================
    print("\n--- Multi-Role Analysis in V_sem^⊥ ---")

    results = {"model": model_name, "exp": 3, "role_counts": {k: len(v) for k, v in role_reps.items()}}

    # 合并所有角色的表示
    all_reps = []
    all_labels = []
    for role, reps in role_reps.items():
        if len(reps) > 0:
            all_reps.extend(reps)
            all_labels.extend([role] * len(reps))

    all_reps = np.array(all_reps)
    all_labels = np.array(all_labels)

    if len(all_reps) > 10:
        # 在V_sem^⊥中做PCA
        all_perp = (proj_perp @ all_reps.T).T
        pca_perp = PCA(n_components=20)
        all_perp_pca = pca_perp.fit_transform(all_perp)

        # PC0的角色分布
        print("\n  PC0 distribution by role:")
        pc0_by_role = {}
        for role in ["subject", "object", "modifier", "prep_object", "isolated"]:
            mask = all_labels == role
            if mask.sum() > 0:
                pc0_vals = all_perp_pca[mask, 0]
                pc0_by_role[role] = {"mean": float(pc0_vals.mean()), "std": float(pc0_vals.std())}
                print(f"    {role:12s}: PC0 mean={pc0_vals.mean():+.2f}, std={pc0_vals.std():.2f}")

        # PC1的角色分布
        print("\n  PC1 distribution by role:")
        pc1_by_role = {}
        for role in ["subject", "object", "modifier", "prep_object", "isolated"]:
            mask = all_labels == role
            if mask.sum() > 0:
                pc1_vals = all_perp_pca[mask, 1]
                pc1_by_role[role] = {"mean": float(pc1_vals.mean()), "std": float(pc1_vals.std())}
                print(f"    {role:12s}: PC1 mean={pc1_vals.mean():+.2f}, std={pc1_vals.std():.2f}")

        results["pc0_by_role"] = pc0_by_role
        results["pc1_by_role"] = pc1_by_role
        results["perp_pca_explained"] = pca_perp.explained_variance_ratio_[:10].tolist()

        # 多角色分类
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # 2分类: subj vs obj
        mask_so = np.isin(all_labels, ["subject", "object"])
        if mask_so.sum() > 20:
            X_so = all_perp_pca[mask_so, :5]
            y_so = (all_labels[mask_so] == "subject").astype(int)
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X_so, y_so, cv=5, scoring='accuracy')
            print(f"\n  Subj vs Obj CV_acc (5 PCs) = {scores.mean():.3f} ± {scores.std():.3f}")
            results["subj_obj_cv"] = float(scores.mean())

        # 3分类: subj vs obj vs iso
        mask_3 = np.isin(all_labels, ["subject", "object", "isolated"])
        if mask_3.sum() > 30:
            X_3 = all_perp_pca[mask_3, :5]
            y_3 = all_labels[mask_3]
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X_3, y_3, cv=5, scoring='accuracy')
            print(f"  Subj vs Obj vs Iso CV_acc (5 PCs) = {scores.mean():.3f} ± {scores.std():.3f} (chance=0.333)")
            results["3way_cv"] = float(scores.mean())

        # 5分类
        min_count = min(len(v) for v in role_reps.values() if len(v) > 0)
        if min_count >= 5 and len(all_labels) >= 50:
            # 平衡样本
            balanced_reps = []
            balanced_labels = []
            rng = np.random.RandomState(42)
            for role in ["subject", "object", "modifier", "prep_object", "isolated"]:
                mask = all_labels == role
                if mask.sum() >= 5:
                    indices = np.where(mask)[0]
                    selected = rng.choice(indices, size=min(min_count, len(indices)), replace=False)
                    balanced_reps.append(all_perp_pca[selected, :5])
                    balanced_labels.extend([role] * len(selected))

            if len(balanced_labels) >= 50:
                X_5 = np.vstack(balanced_reps)
                y_5 = np.array(balanced_labels)
                clf = LogisticRegression(max_iter=1000)
                scores = cross_val_score(clf, X_5, y_5, cv=5, scoring='accuracy')
                print(f"  5-Role CV_acc (5 PCs) = {scores.mean():.3f} ± {scores.std():.3f} (chance=0.200)")
                results["5way_cv"] = float(scores.mean())

        # ==========================================
        # d_syn方向对不同角色的投影
        # ==========================================
        print("\n--- d_syn Projection by Role ---")

        # 从subj/obj提取d_syn
        d_syn_unit, d_syn_norm = extract_d_syn_from_reps(
            role_reps["subject"], role_reps["object"], proj_perp
        )

        role_projections = {}
        for role, reps in role_reps.items():
            if len(reps) > 0:
                reps_arr = np.array(reps)
                reps_perp = (proj_perp @ reps_arr.T).T
                projections = np.array([np.dot(r, d_syn_unit) * d_syn_norm for r in reps_perp])
                role_projections[role] = {
                    "mean": float(projections.mean()),
                    "std": float(projections.std()),
                    "median": float(np.median(projections)),
                }
                print(f"  {role:12s}: d_syn proj mean={projections.mean():+.2f}, std={projections.std():.2f}")

        results["d_syn_projections_by_role"] = role_projections

        # ==========================================
        # 多维语法方向提取
        # ==========================================
        print("\n--- Multi-Dimensional Syntax Directions ---")

        # 用LDA提取最优分类方向
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        mask_3 = np.isin(all_labels, ["subject", "object", "isolated"])
        if mask_3.sum() > 30:
            X_3 = all_perp_pca[mask_3, :10]
            y_3 = all_labels[mask_3]

            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(X_3, y_3)

            # LD1/LD2的角色分布
            print("  LDA directions:")
            for role in ["subject", "object", "isolated"]:
                mask = y_3 == role
                if mask.sum() > 0:
                    print(f"    {role:12s}: LD1 mean={X_lda[mask, 0].mean():+.2f}, LD2 mean={X_lda[mask, 1].mean():+.2f}")

            results["lda_by_role"] = {
                role: {
                    "ld1_mean": float(X_lda[y_3 == role, 0].mean()),
                    "ld2_mean": float(X_lda[y_3 == role, 1].mean()),
                }
                for role in ["subject", "object", "isolated"]
                if (y_3 == role).sum() > 0
            }

    out_path = TEMP / f"ccxliii_exp3_{model_name}_results.json"
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
