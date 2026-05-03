"""
CCXLII(402): 语法角色因果干预 — d_syn是否是语法角色的因果方向？

核心问题:
  CCXL-CCXLI发现d_syn是1维的"主语标记方向"
  但: d_syn是"发现"而非"因果证明"
  → 需要因果干预: 沿d_syn注入偏移, 看能否改变语法角色

实验设计:
  Exp1: d_syn注入干预
    - 取一个宾语位置的token, 在中间层注入+β·d_syn
    - 预测: 模型将把这个token解读为更像主语
    - 测量: 干预后模型输出的变化(是否预测主语相关的后续词?)

  Exp2: 反向d_syn注入
    - 取一个主语位置的token, 在中间层注入-β·d_syn
    - 预测: 模型将把这个token解读为更像宾语
    - 测量: 干预后模型输出的变化

  Exp3: 控制实验 — 沿随机方向注入
    - 在V_sem^⊥中取随机方向(非d_syn)
    - 注入相同幅度的偏移
    - 预测: 不应改变语法角色解读
    - 这是因果推断的关键对照!

  Exp4: 位置与角色分离(被动语态)
    - "The cat chases the dog" → cat=主语@pos1
    - "The dog is chased by the cat" → cat=介词宾语@pos5
    - 在被动语态中, d_syn对cat的编码是否改变?
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


def extract_d_syn(model, tokenizer, device, model_info, proj_perp):
    """提取d_syn方向"""
    mid_layer = model_info.n_layers // 2
    
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
                    
                    rep_subj = out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy()
                    rep_obj = out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy()
                    
                    subj_reps.append(rep_subj)
                    obj_reps.append(rep_obj)
                except:
                    pass
    
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
# Exp1+2: d_syn注入因果干预
# ============================================================
def run_exp12(model_name):
    """
    核心因果干预测试:
    1. 在宾语位置注入+d_syn → 模型是否把宾语解读为主语?
    2. 在主语位置注入-d_syn → 模型是否把主语解读为宾语?
    """
    print(f"\n{'='*60}")
    print(f"CCXLII Exp1+2: d_syn Causal Intervention — {model_name}")
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
    d_syn_unit, d_syn_norm = extract_d_syn(model, tokenizer, device, model_info, proj_perp)
    print(f"  ||d_syn|| = {d_syn_norm:.2f}")

    # 也提取随机方向作为对照
    rng = np.random.RandomState(42)
    random_dir = rng.randn(d_model)
    random_dir_perp = proj_perp @ random_dir
    random_dir_perp_norm = np.linalg.norm(random_dir_perp)
    if random_dir_perp_norm > 1e-10:
        random_dir_unit = random_dir_perp / random_dir_perp_norm
    else:
        random_dir_unit = np.zeros_like(random_dir_perp)

    # 测试不同注入强度
    betas = [0.5, 1.0, 2.0, 5.0, 10.0]
    # betas相对于d_syn_norm的倍数

    results = {"model": model_name, "exp": "1+2", "d_syn_norm": float(d_syn_norm)}

    # 测试句子: "The cat chases the dog"
    # cat=主语(pos1), dog=宾语(pos4)
    # 期望: 干预后模型输出更倾向于主语/宾语相关的词

    test_pairs = NOUN_PAIRS[:10]
    test_verbs = TRANSITIVE_VERBS[:3]

    # ==========================================
    # 测试1: 在宾语位置注入+d_syn → 宾语→主语?
    # ==========================================
    print("\n--- Test 1: Inject +d_syn at Object Position ---")

    intervention_results_obj = []

    for noun_a, noun_b in test_pairs:
        for verb in test_verbs:
            sentence = f"The {noun_a} {verb} the {noun_b}"
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            
            # 找宾语位置
            obj_pos = find_noun_position(tokenizer, sentence, noun_b)
            if obj_pos is None:
                continue
            
            # 基线: 无干预的输出概率
            with torch.no_grad():
                out_base = model(input_ids, output_hidden_states=True)
                logits_base = out_base.logits[0, -1, :].detach().cpu().float()
                probs_base = torch.softmax(logits_base, dim=-1).numpy()
            
            # 干预: 在mid_layer注入+d_syn到宾语位置
            for beta in betas:
                injection_strength = beta * d_syn_norm  # 相对于d_syn的倍数
                
                # 用hook注入
                captured = {}
                modified_hidden = None
                
                def make_injection_hook(target_pos, direction, strength, layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                            # 注入到目标位置
                            dir_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
                            hidden[0, target_pos, :] += strength * dir_tensor
                            return (hidden,) + output[1:]
                        return output
                    return hook
                
                # 注册hook到中间层
                layers = list(model.model.layers)
                hook = layers[mid_layer].register_forward_hook(
                    make_injection_hook(obj_pos, d_syn_unit, injection_strength, mid_layer)
                )
                
                with torch.no_grad():
                    try:
                        out_intervened = model(input_ids, output_hidden_states=True)
                        logits_int = out_intervened.logits[0, -1, :].detach().cpu().float()
                        probs_int = torch.softmax(logits_int, dim=-1).numpy()
                        
                        # KL散度
                        kl_div = float(np.sum(probs_base * np.log(probs_base / (probs_int + 1e-10) + 1e-10)))
                        
                        # top-5预测变化
                        top5_base = np.argsort(probs_base)[-5:][::-1]
                        top5_int = np.argsort(probs_int)[-5:][::-1]
                        
                        intervention_results_obj.append({
                            "noun_a": noun_a, "noun_b": noun_b, "verb": verb,
                            "beta": beta, "kl_div": kl_div,
                            "top1_base": tokenizer.decode([top5_base[0]]).strip(),
                            "top1_int": tokenizer.decode([top5_int[0]]).strip(),
                            "top5_overlap": len(set(top5_base) & set(top5_int)),
                        })
                    except:
                        pass
                
                hook.remove()

    # 汇总
    if intervention_results_obj:
        for beta in betas:
            beta_results = [r for r in intervention_results_obj if r["beta"] == beta]
            if beta_results:
                mean_kl = np.mean([r["kl_div"] for r in beta_results])
                mean_overlap = np.mean([r["top5_overlap"] for r in beta_results])
                print(f"  β={beta:.1f}: mean_KL={mean_kl:.4f}, top5_overlap={mean_overlap:.1f}/5")
        
        results["inject_dsyn_at_obj"] = intervention_results_obj

    # ==========================================
    # 测试2: 在主语位置注入-d_syn → 主语→宾语?
    # ==========================================
    print("\n--- Test 2: Inject -d_syn at Subject Position ---")

    intervention_results_subj = []

    for noun_a, noun_b in test_pairs:
        for verb in test_verbs:
            sentence = f"The {noun_a} {verb} the {noun_b}"
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            
            subj_pos = find_noun_position(tokenizer, sentence, noun_a)
            if subj_pos is None:
                continue
            
            with torch.no_grad():
                out_base = model(input_ids, output_hidden_states=True)
                logits_base = out_base.logits[0, -1, :].detach().cpu().float()
                probs_base = torch.softmax(logits_base, dim=-1).numpy()
            
            for beta in betas:
                injection_strength = beta * d_syn_norm
                
                layers = list(model.model.layers)
                hook = layers[mid_layer].register_forward_hook(
                    make_injection_hook(subj_pos, -d_syn_unit, injection_strength, mid_layer)
                )
                
                with torch.no_grad():
                    try:
                        out_intervened = model(input_ids, output_hidden_states=True)
                        logits_int = out_intervened.logits[0, -1, :].detach().cpu().float()
                        probs_int = torch.softmax(logits_int, dim=-1).numpy()
                        
                        kl_div = float(np.sum(probs_base * np.log(probs_base / (probs_int + 1e-10) + 1e-10)))
                        
                        top5_base = np.argsort(probs_base)[-5:][::-1]
                        top5_int = np.argsort(probs_int)[-5:][::-1]
                        
                        intervention_results_subj.append({
                            "noun_a": noun_a, "noun_b": noun_b, "verb": verb,
                            "beta": beta, "kl_div": kl_div,
                            "top1_base": tokenizer.decode([top5_base[0]]).strip(),
                            "top1_int": tokenizer.decode([top5_int[0]]).strip(),
                            "top5_overlap": len(set(top5_base) & set(top5_int)),
                        })
                    except:
                        pass
                
                hook.remove()

    if intervention_results_subj:
        for beta in betas:
            beta_results = [r for r in intervention_results_subj if r["beta"] == beta]
            if beta_results:
                mean_kl = np.mean([r["kl_div"] for r in beta_results])
                mean_overlap = np.mean([r["top5_overlap"] for r in beta_results])
                print(f"  β={beta:.1f}: mean_KL={mean_kl:.4f}, top5_overlap={mean_overlap:.1f}/5")
        
        results["inject_neg_dsyn_at_subj"] = intervention_results_subj

    # ==========================================
    # 测试3: 对照 — 随机方向注入
    # ==========================================
    print("\n--- Test 3: Control — Inject Random Direction ---")

    intervention_results_random = []

    for noun_a, noun_b in test_pairs[:5]:
        for verb in test_verbs[:2]:
            sentence = f"The {noun_a} {verb} the {noun_b}"
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            
            obj_pos = find_noun_position(tokenizer, sentence, noun_b)
            if obj_pos is None:
                continue
            
            with torch.no_grad():
                out_base = model(input_ids, output_hidden_states=True)
                logits_base = out_base.logits[0, -1, :].detach().cpu().float()
                probs_base = torch.softmax(logits_base, dim=-1).numpy()
            
            for beta in [1.0, 5.0, 10.0]:
                injection_strength = beta * d_syn_norm
                
                layers = list(model.model.layers)
                hook = layers[mid_layer].register_forward_hook(
                    make_injection_hook(obj_pos, random_dir_unit, injection_strength, mid_layer)
                )
                
                with torch.no_grad():
                    try:
                        out_intervened = model(input_ids, output_hidden_states=True)
                        logits_int = out_intervened.logits[0, -1, :].detach().cpu().float()
                        probs_int = torch.softmax(logits_int, dim=-1).numpy()
                        
                        kl_div = float(np.sum(probs_base * np.log(probs_base / (probs_int + 1e-10) + 1e-10)))
                        
                        intervention_results_random.append({
                            "noun_a": noun_a, "noun_b": noun_b, "verb": verb,
                            "beta": beta, "kl_div": kl_div,
                        })
                    except:
                        pass
                
                hook.remove()

    if intervention_results_random:
        for beta in [1.0, 5.0, 10.0]:
            beta_results = [r for r in intervention_results_random if r["beta"] == beta]
            if beta_results:
                mean_kl = np.mean([r["kl_div"] for r in beta_results])
                print(f"  Random β={beta:.1f}: mean_KL={mean_kl:.4f}")
        
        results["inject_random_at_obj"] = intervention_results_random

    out_path = TEMP / f"ccxlii_exp12_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp3: 表示空间中的d_syn干预 (更直接的因果测试)
# ============================================================
def run_exp3(model_name):
    """
    更直接的因果测试:
    1. 取一个主语位置的表示rep_subj
    2. 在V_sem^⊥中: rep_subj^⊥ + β·d_syn → 投影回全空间
    3. 用修改后的表示做角色分类 → 是否改变分类结果?
    
    这测试: d_syn方向上的偏移是否足以改变角色判断
    """
    print(f"\n{'='*60}")
    print(f"CCXLII Exp3: Direct d_syn Intervention in Representation — {model_name}")
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
    d_syn_unit, d_syn_norm = extract_d_syn(model, tokenizer, device, model_info, proj_perp)

    # 收集主语/宾语表示
    print("Collecting subject/object representations...")
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

                    rep_subj = out_subj.hidden_states[mid_layer][0, pos_subj, :].detach().cpu().float().numpy()
                    rep_obj = out_obj.hidden_states[mid_layer][0, pos_obj, :].detach().cpu().float().numpy()

                    subj_reps.append(rep_subj)
                    obj_reps.append(rep_obj)
                except:
                    pass

    subj_reps = np.array(subj_reps)
    obj_reps = np.array(obj_reps)
    n_subj = len(subj_reps)
    n_obj = len(obj_reps)

    print(f"  Collected: {n_subj} subj, {n_obj} obj")

    # 训练角色分类器(在V_sem^⊥中)
    X_all = np.vstack([subj_reps, obj_reps])
    y_all = np.array([0] * n_subj + [1] * n_obj)

    X_perp = (proj_perp @ X_all.T).T
    pca_perp = PCA(n_components=10)
    X_perp_pca = pca_perp.fit_transform(X_perp)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_perp_pca, y_all)

    results = {"model": model_name, "exp": 3}

    # ==========================================
    # 干预测试: 在d_syn方向上偏移表示
    # ==========================================
    print("\n--- d_syn Direction Intervention ---")

    # 对每个主语表示, 沿-d_syn偏移 → 分类是否变为宾语?
    # 对每个宾语表示, 沿+d_syn偏移 → 分类是否变为主语?

    alphas = np.linspace(-2.0, 2.0, 21)  # 偏移系数(相对于d_syn_norm)

    subj_flip_rates = []
    obj_flip_rates = []

    for alpha in alphas:
        injection = alpha * d_syn_norm * d_syn_unit  # 在原始空间的偏移

        # 主语→宾语: rep_subj - α·d_syn_norm·d_syn_unit
        subj_modified = subj_reps + injection  # 注入(可能是+/-)
        subj_mod_perp = (proj_perp @ subj_modified.T).T
        subj_mod_pca = pca_perp.transform(subj_mod_perp)
        subj_pred = clf.predict(subj_mod_pca)
        subj_flip = np.mean(subj_pred == 1)  # 被分类为宾语的比例

        # 宾语→主语: rep_obj + α·d_syn_norm·d_syn_unit
        obj_modified = obj_reps + injection
        obj_mod_perp = (proj_perp @ obj_modified.T).T
        obj_mod_pca = pca_perp.transform(obj_mod_perp)
        obj_pred = clf.predict(obj_mod_pca)
        obj_flip = np.mean(obj_pred == 0)  # 被分类为主语的比例

        subj_flip_rates.append(float(subj_flip))
        obj_flip_rates.append(float(obj_flip))

        if abs(alpha) < 0.01 or abs(alpha - 1.0) < 0.1 or abs(alpha + 1.0) < 0.1:
            print(f"  α={alpha:+.1f}: subj→obj_rate={subj_flip:.3f}, obj→subj_rate={obj_flip:.3f}")

    results["subj_flip_by_alpha"] = {f"α={a:.1f}": r for a, r in zip(alphas, subj_flip_rates)}
    results["obj_flip_by_alpha"] = {f"α={a:.1f}": r for a, r in zip(alphas, obj_flip_rates)}

    # ==========================================
    # 对照: 随机方向偏移
    # ==========================================
    print("\n--- Random Direction Control ---")

    rng = np.random.RandomState(42)
    random_flip_subj = []
    random_flip_obj = []

    for trial in range(5):
        random_dir = rng.randn(d_model)
        random_dir_perp = proj_perp @ random_dir
        r_norm = np.linalg.norm(random_dir_perp)
        if r_norm < 1e-10:
            continue
        random_dir_unit = random_dir_perp / r_norm

        alpha = -1.0  # 与d_syn干预相同的偏移量
        injection = alpha * d_syn_norm * random_dir_unit

        subj_modified = subj_reps + injection
        subj_mod_perp = (proj_perp @ subj_modified.T).T
        subj_mod_pca = pca_perp.transform(subj_mod_perp)
        subj_pred = clf.predict(subj_mod_pca)
        subj_flip = np.mean(subj_pred == 1)

        obj_modified = obj_reps + injection
        obj_mod_perp = (proj_perp @ obj_modified.T).T
        obj_mod_pca = pca_perp.transform(obj_mod_perp)
        obj_pred = clf.predict(obj_mod_pca)
        obj_flip = np.mean(obj_pred == 0)

        random_flip_subj.append(float(subj_flip))
        random_flip_obj.append(float(obj_flip))

    print(f"  Random α=-1.0: subj→obj_rate={np.mean(random_flip_subj):.3f}, obj→subj_rate={np.mean(random_flip_obj):.3f}")

    results["random_control"] = {
        "subj_flip_mean": float(np.mean(random_flip_subj)),
        "obj_flip_mean": float(np.mean(random_flip_obj)),
        "subj_flip_std": float(np.std(random_flip_subj)),
        "obj_flip_std": float(np.std(random_flip_obj)),
    }

    # ★★★ 因果效应度量
    # d_syn在α=-1时的翻转率 vs 随机方向的翻转率
    dsyn_subj_flip_at_neg1 = subj_flip_rates[alphas.tolist().index(min(alphas, key=lambda x: abs(x + 1.0)))]
    dsyn_obj_flip_at_pos1 = obj_flip_rates[alphas.tolist().index(min(alphas, key=lambda x: abs(x - 1.0)))]

    causal_effect_subj = dsyn_subj_flip_at_neg1 - np.mean(random_flip_subj)
    causal_effect_obj = dsyn_obj_flip_at_pos1 - np.mean(random_flip_obj)

    results["causal_effect"] = {
        "d_syn_subj_flip_at_neg1": float(dsyn_subj_flip_at_neg1),
        "d_syn_obj_flip_at_pos1": float(dsyn_obj_flip_at_pos1),
        "random_subj_flip": float(np.mean(random_flip_subj)),
        "random_obj_flip": float(np.mean(random_flip_obj)),
        "causal_effect_subj": float(causal_effect_subj),
        "causal_effect_obj": float(causal_effect_obj),
    }

    print(f"\n  ★ Causal Effect:")
    print(f"    d_syn(-1) flips subj→obj: {dsyn_subj_flip_at_neg1:.3f}")
    print(f"    Random(-1) flips subj→obj: {np.mean(random_flip_subj):.3f}")
    print(f"    Causal effect (subj): {causal_effect_subj:.3f}")
    print(f"    d_syn(+1) flips obj→subj: {dsyn_obj_flip_at_pos1:.3f}")
    print(f"    Random(+1) flips obj→subj: {np.mean(random_flip_obj):.3f}")
    print(f"    Causal effect (obj): {causal_effect_obj:.3f}")

    out_path = TEMP / f"ccxlii_exp3_{model_name}_results.json"
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
