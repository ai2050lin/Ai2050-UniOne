"""
CCXXXI(382): 层级角色解析 + 注意力分析 — 角色信息何时被解析？

核心问题:
  CCXXX发现cos(role,interact)=-1.0, 即在句首位置, 主语和宾语的表示几乎相同
  → 角色信息在句首尚未被解析
  → 问题: 角色信息何时被解析? 哪一层? 哪些注意力头?

实验设计:
  Exp1: 层级角色解析轨迹
    - 对同一句子, 追踪cat位置的表示随层数的变化
    - 逐层计算: 能否区分主语vs宾语?
    - 找到"角色解析层" — 角色信息首次显著出现的层
  
  Exp2: 注意力模式与角色解析
    - 提取关键层的attention pattern
    - 分析: 动词对名词的注意力是否编码角色信息?
    - 分析: 哪些头对角色解析贡献最大?
  
  Exp3: V_syn专用子空间
    - 用语法角色数据训练专用PCA
    - 分离"纯语法子空间"V_syn
    - 验证: V_syn中角色是否线性可分?
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

# ===== 名词和动词 =====
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
    """找到名词在句子中的token位置"""
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
    """找到目标词在句子中的token位置(宽松匹配)"""
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    
    for prefix in ['', ' ']:
        target_tokens = tokenizer(prefix + target_word, add_special_tokens=False)['input_ids']
        for i in range(len(input_ids) - len(target_tokens) + 1):
            if input_ids[i:i+len(target_tokens)] == target_tokens:
                return list(range(i, i + len(target_tokens)))
    
    # 宽松匹配
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == target_word.lower() or decoded == target_word.lower().rstrip('s'):
            return [i]
    
    return None


# ============================================================
# Exp1: 层级角色解析轨迹
# ============================================================
def run_exp1(model_name):
    """
    逐层追踪角色信息的解析过程
    
    设计:
    - 简单SVO: "The [A] [verb] the [B]" vs "The [B] [verb] the [A]"
    - 对[A], 在每层比较: 主语时的表示 vs 宾语时的表示
    - 用线性分类器逐层判断: 该层能否区分主语/宾语?
    """
    print(f"\n{'='*60}")
    print(f"CCXXXI Exp1: Layer-wise Role Resolution — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 收集逐层数据
    # 对每个名词, 在每个动词下, 收集subject和object的逐层表示
    layer_data = {f"L{l}": {"subj": [], "obj": []} for l in range(n_layers)}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for verb in TRANSITIVE_VERBS:
            # A是主语: "The [A] [verb] the [B]"
            sent_ab = f"The {noun_a} {verb} the {noun_b}"
            # A是宾语: "The [B] [verb] the [A]"  
            sent_ba = f"The {noun_b} {verb} the {noun_a}"
            
            # 找A在两个句子中的位置
            pos_subj = find_noun_position(tokenizer, sent_ab, noun_a)
            pos_obj = find_noun_position(tokenizer, sent_ba, noun_a)
            
            if pos_subj is None or pos_obj is None:
                continue
            
            # 前向传播(只做一次per句子, 提取所有层)
            for sentence, noun_pos, role_label in [
                (sent_ab, pos_subj, "subj"),
                (sent_ba, pos_obj, "obj"),
            ]:
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        outputs = model(inputs['input_ids'], output_hidden_states=True)
                        for l in range(n_layers):
                            rep = outputs.hidden_states[l][0, noun_pos, :].detach().cpu().float().numpy()
                            layer_data[f"L{l}"][role_label].append({
                                "noun": noun_a,
                                "verb": verb,
                                "rep": rep,
                                "sentence": sentence,
                            })
                    except Exception as e:
                        pass
    
    # 逐层分析
    results = {"model": model_name, "exp": 1, "n_layers": n_layers}
    
    print("\n--- Layer-wise Role Discriminability ---")
    print(f"  {'Layer':>5} | {'CV_acc':>6} | {'||δ_role||':>10} | {'Consist':>8} | {'V_sem%':>6} | {'CV_rand':>6}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    
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
    
    # 逐层分析(采样部分层以节省时间)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 20))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    resolution_trajectory = []
    
    for l in sample_layers:
        key = f"L{l}"
        subj_reps = [d["rep"] for d in layer_data[key]["subj"]]
        obj_reps = [d["rep"] for d in layer_data[key]["obj"]]
        
        if len(subj_reps) < 10 or len(obj_reps) < 10:
            continue
        
        X = np.array(subj_reps + obj_reps)
        y = np.array([0] * len(subj_reps) + [1] * len(obj_reps))
        
        # 分类准确率
        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv_folds = min(5, min(len(subj_reps), len(obj_reps)) // 5)
        if cv_folds >= 2:
            cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
            cv_acc = cv_scores.mean()
        else:
            cv_acc = 0.5
        
        # 随机基线
        y_rand = np.random.permutation(y)
        if cv_folds >= 2:
            cv_rand = cross_val_score(clf, X, y_rand, cv=cv_folds, scoring='accuracy').mean()
        else:
            cv_rand = 0.5
        
        # δ_role分析
        deltas = [s - o for s, o in zip(subj_reps[:len(obj_reps)], obj_reps[:len(subj_reps)])]
        if deltas:
            deltas = np.array(deltas)
            mean_norm = float(np.mean(np.linalg.norm(deltas, axis=1)))
            mean_d = deltas.mean(axis=0)
            consistency = np.mean([np.dot(d, mean_d) / (np.linalg.norm(d) * np.linalg.norm(mean_d) + 1e-10) 
                                  for d in deltas])
            vsem = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                           for d in deltas])
        else:
            mean_norm = 0
            consistency = 0
            vsem = 0
        
        resolution_trajectory.append({
            "layer": l, "cv_acc": cv_acc, "cv_rand": cv_rand,
            "mean_norm": mean_norm, "consistency": consistency,
            "vsem_energy": vsem,
        })
        
        print(f"  L{l:>4} | {cv_acc:>6.3f} | {mean_norm:>10.2f} | {consistency:>8.3f} | {vsem*100:>5.1f}% | {cv_rand:>6.3f}")
    
    results["resolution_trajectory"] = resolution_trajectory
    
    # 找到角色解析层
    if resolution_trajectory:
        # 角色解析层: CV准确率首次超过 0.6 (或随机+0.1) 的层
        for entry in resolution_trajectory:
            if entry["cv_acc"] > max(entry["cv_rand"] + 0.1, 0.6):
                resolution_layer = entry["layer"]
                resolution_acc = entry["cv_acc"]
                break
        else:
            resolution_layer = -1
            resolution_acc = 0
        
        # 峰值层
        best_entry = max(resolution_trajectory, key=lambda x: x["cv_acc"])
        
        results["resolution_layer"] = resolution_layer
        results["resolution_acc"] = resolution_acc
        results["peak_layer"] = best_entry["layer"]
        results["peak_acc"] = best_entry["cv_acc"]
        
        print(f"\n  ★ Role resolution layer: L{resolution_layer} (acc={resolution_acc:.3f})")
        print(f"  ★ Peak layer: L{best_entry['layer']} (acc={best_entry['cv_acc']:.3f})")
        
        # 分3段分析: 早期/中期/晚期的角色解码
        early_layers = [e for e in resolution_trajectory if e["layer"] < n_layers // 3]
        mid_layers = [e for e in resolution_trajectory if n_layers // 3 <= e["layer"] < 2 * n_layers // 3]
        late_layers = [e for e in resolution_trajectory if e["layer"] >= 2 * n_layers // 3]
        
        for name, segment in [("Early(0-1/3)", early_layers), ("Mid(1/3-2/3)", mid_layers), ("Late(2/3-1)", late_layers)]:
            if segment:
                mean_acc = np.mean([e["cv_acc"] for e in segment])
                mean_norm = np.mean([e["mean_norm"] for e in segment])
                print(f"  {name}: mean_acc={mean_acc:.3f}, mean_||δ||={mean_norm:.2f}")
    
    # 保存
    # 把numpy数组转为list以便JSON序列化
    for entry in resolution_trajectory:
        for k, v in entry.items():
            if isinstance(v, (np.floating, np.integer)):
                entry[k] = float(v)
    
    out_path = TEMP / f"ccxxxi_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp2: 注意力模式与角色解析
# ============================================================
def run_exp2(model_name):
    """
    分析注意力模式与语法角色解析的关系
    
    设计:
    - 对简单SVO句子, 提取动词位置处的attention pattern
    - 比较主语动词和宾语动词的注意力模式
    - 分析: 哪些注意力头对角色编码贡献最大?
    """
    print(f"\n{'='*60}")
    print(f"CCXXXI Exp2: Attention & Role Resolution — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    layers = get_layers(model)
    
    # 只分析几个关键层
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 收集注意力pattern
    # 对每个句子, 提取动词token对各名词token的attention权重
    
    attention_data = {f"L{l}": [] for l in key_layers}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS[:10]):  # 减少到10对
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/10")
        
        for verb in TRANSITIVE_VERBS[:3]:
            sent = f"The {noun_a} {verb} the {noun_b}"
            
            # 找各词位置
            pos_a = find_noun_position(tokenizer, sent, noun_a)
            pos_b = find_noun_position(tokenizer, sent, noun_b)
            pos_verb = find_token_position(tokenizer, sent, verb.rstrip('s'))
            
            if pos_a is None or pos_b is None or pos_verb is None:
                continue
            
            verb_pos = pos_verb[0] if isinstance(pos_verb, list) else pos_verb
            
            # 前向传播 + 提取attention
            inputs = tokenizer(sent, return_tensors='pt').to(device)
            
            # 用hook提取attention pattern
            attn_patterns = {}
            
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # output包含: (hidden_states, attn_weights, past_key_value)
                    # 或者: output[1] 是 attention weights
                    if isinstance(output, tuple) and len(output) >= 2:
                        # 有些模型返回attn_weights
                        attn = output[1]
                        if attn is not None and isinstance(attn, torch.Tensor):
                            attn_patterns[f"L{layer_idx}"] = attn.detach().float().cpu()
                return hook
            
            hooks = []
            for l in key_layers:
                layer = layers[l]
                # Register hook on self_attn
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
            
            with torch.no_grad():
                try:
                    # 尝试用output_attentions
                    outputs = model(inputs['input_ids'], output_attentions=True, output_hidden_states=True)
                    
                    # 如果模型直接返回attentions
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        for l_idx, l in enumerate(key_layers):
                            if l < len(outputs.attentions) and outputs.attentions[l] is not None:
                                attn_patterns[f"L{l}"] = outputs.attentions[l].detach().float().cpu()
                except Exception as e:
                    print(f"  Error: {e}")
            
            for h in hooks:
                h.remove()
            
            # 分析attention pattern
            for l in key_layers:
                key = f"L{l}"
                if key not in attn_patterns:
                    continue
                
                attn = attn_patterns[key]  # [1, n_heads, seq_len, seq_len]
                if attn.dim() != 4:
                    continue
                
                n_heads = attn.shape[1]
                seq_len = attn.shape[2]
                
                # 动词对各token的attention
                if verb_pos < seq_len:
                    verb_attn = attn[0, :, verb_pos, :]  # [n_heads, seq_len]
                    
                    # 动词→主语名词 的注意力
                    attn_to_subj = verb_attn[:, pos_a].mean().item() if pos_a < seq_len else 0
                    # 动词→宾语名词 的注意力
                    attn_to_obj = verb_attn[:, pos_b].mean().item() if pos_b < seq_len else 0
                    
                    # 逐头的注意力差异
                    head_attn_subj = verb_attn[:, pos_a].numpy() if pos_a < seq_len else np.zeros(n_heads)
                    head_attn_obj = verb_attn[:, pos_b].numpy() if pos_b < seq_len else np.zeros(n_heads)
                    
                    # 主语→动词, 宾语→动词 的注意力(反向)
                    subj_to_verb = attn[0, :, pos_a, verb_pos].mean().item() if pos_a < seq_len else 0
                    obj_to_verb = attn[0, :, pos_b, verb_pos].mean().item() if pos_b < seq_len else 0
                    
                    attention_data[key].append({
                        "noun_a": noun_a,
                        "noun_b": noun_b,
                        "verb": verb,
                        "attn_verb_to_subj": attn_to_subj,
                        "attn_verb_to_obj": attn_to_obj,
                        "attn_subj_to_verb": subj_to_verb,
                        "attn_obj_to_verb": obj_to_verb,
                        "head_attn_subj": head_attn_subj.tolist(),
                        "head_attn_obj": head_attn_obj.tolist(),
                        "n_heads": n_heads,
                    })
    
    # ===== 分析 =====
    results = {"model": model_name, "exp": 2}
    
    for l in key_layers:
        key = f"L{l}"
        data = attention_data[key]
        
        if not data:
            continue
        
        # 平均注意力
        mean_verb_to_subj = np.mean([d["attn_verb_to_subj"] for d in data])
        mean_verb_to_obj = np.mean([d["attn_verb_to_obj"] for d in data])
        mean_subj_to_verb = np.mean([d["attn_subj_to_verb"] for d in data])
        mean_obj_to_verb = np.mean([d["attn_obj_to_verb"] for d in data])
        
        # 逐头分析
        n_heads = data[0]["n_heads"]
        head_subj = np.mean([d["head_attn_subj"] for d in data], axis=0)
        head_obj = np.mean([d["head_attn_obj"] for d in data], axis=0)
        head_diff = head_subj - head_obj  # 正值=更关注主语
        
        # 找到角色区分头: verb→subj 和 verb→obj 差异最大的头
        head_role_score = np.abs(head_diff)
        top_heads = np.argsort(head_role_score)[::-1][:5]
        
        results[key] = {
            "mean_verb_to_subj": float(mean_verb_to_subj),
            "mean_verb_to_obj": float(mean_verb_to_obj),
            "mean_subj_to_verb": float(mean_subj_to_verb),
            "mean_obj_to_verb": float(mean_obj_to_verb),
            "ratio_subj_vs_obj": float(mean_verb_to_subj / (mean_verb_to_obj + 1e-10)),
            "head_attn_subj": head_subj.tolist(),
            "head_attn_obj": head_obj.tolist(),
            "head_role_score": head_role_score.tolist(),
            "top5_role_heads": top_heads.tolist(),
        }
        
        print(f"\n  L{l}:")
        print(f"    Verb→Subj: {mean_verb_to_subj:.4f}, Verb→Obj: {mean_verb_to_obj:.4f}, "
              f"ratio={mean_verb_to_subj/(mean_verb_to_obj+1e-10):.2f}")
        print(f"    Subj→Verb: {mean_subj_to_verb:.4f}, Obj→Verb: {mean_obj_to_verb:.4f}")
        print(f"    Top-5 role heads: {top_heads.tolist()}")
        print(f"    Head scores: {[f'{s:.3f}' for s in head_role_score[top_heads]]}")
    
    # 保存
    out_path = TEMP / f"ccxxxi_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp3: V_syn专用子空间
# ============================================================
def run_exp3(model_name):
    """
    用语法角色数据训练专用PCA, 分离纯语法子空间V_syn
    
    设计:
    - 收集大量主语/宾语表示(在中间层)
    - 用这些数据训练PCA → V_syn
    - 在V_syn中: 主语/宾语是否线性可分?
    - V_syn与V_sem的正交性?
    """
    print(f"\n{'='*60}")
    print(f"CCXXXI Exp3: V_syn Subspace Isolation — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 步骤1: 收集语法角色数据
    print("Step 1: Collecting syntactic role representations...")
    
    syn_data = {"subj": [], "obj": []}  # 每个entry: {noun, verb, rep}
    
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
                        syn_data[role].append({"noun": noun_a, "verb": verb, "rep": rep})
                    except:
                        pass
    
    n_subj = len(syn_data["subj"])
    n_obj = len(syn_data["obj"])
    print(f"  Collected: {n_subj} subject reps, {n_obj} object reps")
    
    # 步骤2: 训练V_sem PCA(用语义概念)
    print("\nStep 2: Training V_sem PCA...")
    concept_reps = []
    for concept in REPRESENTATIVE_CONCEPTS:
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], output_hidden_states=True)
            last_pos = inputs['input_ids'].shape[1] - 1
            rep = outputs.hidden_states[mid_layer][0, last_pos, :].detach().cpu().float().numpy()
            concept_reps.append(rep)
    
    X_sem = np.array(concept_reps)
    pca_sem = PCA(n_components=50)
    pca_sem.fit(X_sem)
    
    # 步骤3: 训练V_syn PCA(用语法角色数据的差值)
    print("\nStep 3: Training V_syn PCA...")
    
    # 方法A: 直接用所有表示做PCA
    all_syn_reps = np.array([d["rep"] for d in syn_data["subj"]] + [d["rep"] for d in syn_data["obj"]])
    pca_syn_raw = PCA(n_components=50)
    pca_syn_raw.fit(all_syn_reps)
    
    # 方法B: 用δ_role做PCA(只保留角色变化方向)
    # 配对: 同名词同动词的主语-宾语差
    deltas = []
    for d_subj in syn_data["subj"]:
        for d_obj in syn_data["obj"]:
            if d_subj["noun"] == d_obj["noun"] and d_subj["verb"] == d_obj["verb"]:
                deltas.append(d_subj["rep"] - d_obj["rep"])
    
    if deltas:
        deltas = np.array(deltas)
        pca_syn_delta = PCA(n_components=min(20, len(deltas)-1))
        pca_syn_delta.fit(deltas)
        
        print(f"  δ_role PCA: top-1={pca_syn_delta.explained_variance_ratio_[0]*100:.1f}%, "
              f"top-5={np.sum(pca_syn_delta.explained_variance_ratio_[:5])*100:.1f}%")
    else:
        pca_syn_delta = None
    
    # 步骤4: V_syn中角色解码
    print("\nStep 4: Role classification in different subspaces...")
    
    results = {"model": model_name, "exp": 3, "n_subj": n_subj, "n_obj": n_obj}
    
    # 4a. 在V_sem(5d)中解码
    X_all = np.array([d["rep"] for d in syn_data["subj"]] + [d["rep"] for d in syn_data["obj"]])
    y_all = np.array([0] * n_subj + [1] * n_obj)
    
    # V_sem投影
    X_vsem = pca_sem.components_[:5] @ X_all.T  # [5, N]
    X_vsem = X_vsem.T  # [N, 5]
    
    if n_subj >= 10 and n_obj >= 10:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        cv_folds = min(5, min(n_subj, n_obj) // 5)
        cv_vsem = cross_val_score(clf, X_vsem, y_all, cv=cv_folds, scoring='accuracy').mean()
    else:
        cv_vsem = 0
    
    print(f"  V_sem(5d) role CV: {cv_vsem:.3f}")
    results["cv_vsem5"] = float(cv_vsem)
    
    # 4b. 在V_syn_delta中解码
    if pca_syn_delta is not None:
        for n_dims in [2, 5, 10]:
            X_vsyn = pca_syn_delta.components_[:n_dims] @ X_all.T
            X_vsyn = X_vsyn.T
            
            if n_subj >= 10 and n_obj >= 10:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                cv_vsyn = cross_val_score(clf, X_vsyn, y_all, cv=cv_folds, scoring='accuracy').mean()
            else:
                cv_vsyn = 0
            
            print(f"  V_syn_delta({n_dims}d) role CV: {cv_vsyn:.3f}")
            results[f"cv_vsyn_delta{n_dims}"] = float(cv_vsyn)
    
    # 4c. 在V_sem的残差空间中解码
    # 去除V_sem(5d)后的残差
    proj_vsem5 = pca_sem.components_[:5].T @ (pca_sem.components_[:5] @ X_all.T)  # [d, N]
    X_residual = X_all - proj_vsem5.T  # [N, d]
    
    # 在残差空间做PCA
    pca_resid = PCA(n_components=50)
    pca_resid.fit(X_residual)
    
    for n_dims in [5, 10, 20]:
        X_resid_proj = pca_resid.components_[:n_dims] @ X_residual.T
        X_resid_proj = X_resid_proj.T
        
        if n_subj >= 10 and n_obj >= 10:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv_resid = cross_val_score(clf, X_resid_proj, y_all, cv=cv_folds, scoring='accuracy').mean()
        else:
            cv_resid = 0
        
        print(f"  Residual(V_sem^⊥)({n_dims}d) role CV: {cv_resid:.3f}")
        results[f"cv_resid{n_dims}"] = float(cv_resid)
    
    # 步骤5: V_sem与V_syn的正交性
    print("\nStep 5: V_sem vs V_syn orthogonality...")
    
    if pca_syn_delta is not None:
        # 计算 V_sem PCs 与 V_syn_delta PCs 的子空间对齐
        V_sem_5 = pca_sem.components_[:5]  # [5, d]
        V_syn_5 = pca_syn_delta.components_[:5]  # [5, d]
        
        # 子空间余弦矩阵: C[i,j] = cos(V_sem_i, V_syn_j)
        C = V_sem_5 @ V_syn_5.T  # [5, 5]
        
        # SVD of C: 最大奇异值 = 子空间最大对齐
        svd_vals = np.linalg.svd(C, compute_uv=False)
        
        # 子空间距离: Grassmann distance
        max_alignment = svd_vals[0]
        
        # 平均对齐
        mean_alignment = np.mean(np.abs(C))
        
        results["vsem_vsyn_alignment"] = {
            "max_singular_value": float(max_alignment),
            "mean_abs_cos": float(mean_alignment),
            "singular_values": svd_vals.tolist(),
        }
        
        print(f"  Max alignment (top singular value): {max_alignment:.4f}")
        print(f"  Mean |cos|: {mean_alignment:.4f}")
        print(f"  All singular values: {[f'{v:.4f}' for v in svd_vals]}")
        
        # 逐PC分析
        print("\n  PC alignment matrix (V_sem × V_syn):")
        print("  " + " ".join([f"Vsyn{i:>2}" for i in range(5)]))
        for i in range(5):
            row = " ".join([f"{C[i,j]:>6.3f}" for j in range(5)])
            print(f"  Vsem{i}: {row}")
    
    # 步骤6: δ_role在V_sem和V_syn中的投影
    print("\nStep 6: δ_role projection analysis...")
    
    if deltas is not None and len(deltas) > 0:
        # δ_role在V_sem中的能量
        vsem_energy = np.mean([
            np.sum((pca_sem.components_[:5] @ d) ** 2) / (np.linalg.norm(d) ** 2 + 1e-10)
            for d in deltas
        ])
        
        # δ_role在V_syn_delta中的能量
        if pca_syn_delta is not None:
            vsyn_energy = np.mean([
                np.sum((pca_syn_delta.components_[:5] @ d) ** 2) / (np.linalg.norm(d) ** 2 + 1e-10)
                for d in deltas
            ])
            
            # δ_role在V_sem+V_syn联合空间中的能量
            # 联合空间的基: V_sem PCs + V_syn PCs (正交化)
            combined = np.vstack([pca_sem.components_[:5], pca_syn_delta.components_[:5]])
            # Gram-Schmidt正交化
            from scipy.linalg import orth
            combined_orth = orth(combined.T).T  # [rank, d]
            
            combined_energy = np.mean([
                np.sum((combined_orth @ d) ** 2) / (np.linalg.norm(d) ** 2 + 1e-10)
                for d in deltas
            ])
            
            print(f"  V_sem(5d) energy: {vsem_energy*100:.1f}%")
            print(f"  V_syn(5d) energy: {vsyn_energy*100:.1f}%")
            print(f"  Combined(orth) energy: {combined_energy*100:.1f}%")
            
            results["delta_projection"] = {
                "vsem5_energy": float(vsem_energy),
                "vsyn5_energy": float(vsyn_energy),
                "combined_energy": float(combined_energy),
            }
    
    # 保存
    out_path = TEMP / f"ccxxxi_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
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
