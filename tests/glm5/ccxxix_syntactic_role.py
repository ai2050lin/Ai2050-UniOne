"""
CCXXIX(378): 同句法内语法角色对比 — 消除上下文效应

核心改进:
  CCXXVIII发现: Δ = rep(sentence) - rep(isolated) 被"上下文效应"主导
  语法角色差异(||δ_role||)远小于上下文效应(||Δ_context||)
  
  解决方案: 使用同句改写(minimal pair), 让名词在不同语法角色下
  出现在相同的句子框架中, 消除上下文效应:
  
  "The cat chases the dog" → cat是主语, dog是宾语
  "The dog chases the cat" → dog是主语, cat是宾语
  
  Δ = rep(cat_主语) - rep(cat_宾语) = 纯语法角色差异!

实验设计:
  Exp1: 同句改写对比
    - 20个名词对(cat/dog, king/queen, etc.)
    - 每对构造: "The [A] [VERB]s the [B]" vs "The [B] [VERB]s the [A]"
    - 测量: 同一名词在主语vs宾语位置的差异
    - 在V_sem中: 主语方向 vs 宾语方向

  Exp2: 语法角色的几何结构
    - 主语变换 T_sub = rep(主语) - rep(某基线)
    - 宾语变换 T_obj = rep(宾语) - rep(某基线)
    - 问题: T_sub和T_obj是否跨概念一致?
    - T_sub与T_obj是否正交?

  Exp3: 语法角色vs语义类别的交互
    - 不同语义类别的名词, 语法角色效应是否不同?
    - 例如: "cat chases" vs "freedom requires"
    - 语义×语法的交互项是否显著?
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

# ===== 同句改写对 =====
# 每对名词在同一句话中互换主语/宾语位置
NOUN_PAIRS = [
    # 动物对
    ("cat", "dog"), ("bird", "fish"), ("lion", "tiger"), ("eagle", "whale"), ("horse", "wolf"),
    # 人物对
    ("king", "queen"), ("mother", "child"), ("friend", "enemy"), ("teacher", "student"), ("doctor", "patient"),
    # 物品对
    ("hammer", "knife"), ("sword", "wheel"), ("rope", "nail"), ("stone", "glass"), ("wood", "metal"),
    # 自然对
    ("rain", "snow"), ("wind", "storm"), ("sun", "moon"), ("fire", "water"), ("mountain", "river"),
]

# 动词: 选择20个及物动词
TRANSITIVE_VERBS = [
    "chases", "sees", "loves", "hates", "follows",
    "finds", "takes", "knows", "watches", "catches",
    "helps", "needs", "likes", "remembers", "understands",
    "fears", "protects", "attacks", "avoids", "trusts",
]

# 80个概念(用于PCA训练)
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

CATEGORIES = {
    "animals": REPRESENTATIVE_CONCEPTS[0:8],
    "colors": REPRESENTATIVE_CONCEPTS[8:16],
    "emotions": REPRESENTATIVE_CONCEPTS[16:24],
    "materials": REPRESENTATIVE_CONCEPTS[24:32],
    "weather": REPRESENTATIVE_CONCEPTS[32:40],
    "body_parts": REPRESENTATIVE_CONCEPTS[40:48],
    "foods": REPRESENTATIVE_CONCEPTS[48:56],
    "tools": REPRESENTATIVE_CONCEPTS[56:64],
    "abstract": REPRESENTATIVE_CONCEPTS[64:72],
    "social": REPRESENTATIVE_CONCEPTS[72:80],
}


def find_noun_position(tokenizer, sentence, noun):
    """找到名词在句子中的token位置(兼容有无前导空格的情况)"""
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    
    # 尝试多种编码方式
    for prefix in ['', ' ']:
        noun_tokens = tokenizer(prefix + noun, add_special_tokens=False)['input_ids']
        for i in range(len(input_ids) - len(noun_tokens) + 1):
            if input_ids[i:i+len(noun_tokens)] == noun_tokens:
                return i + len(noun_tokens) - 1  # 名词最后一个token
    
    # fallback: 通过解码找到包含名词的token
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip()
        if decoded == noun:
            return i
    
    # 最终fallback: 返回None
    return None


def extract_reps_with_pca(model_name, noun_pairs, verbs, model=None, tokenizer=None, device=None):
    """
    提取所有同句改写条件下的表示, 同时训练PCA
    
    返回: reps_dict, pca_model, model_info
    reps_dict: {noun: {role: {verb: {layer: vector}}}}
    role: "subject" or "object"
    """
    if model is None:
        model, tokenizer, device = load_model(model_name)
        own_model = True
    else:
        own_model = False
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 提取同句改写表示
    reps_dict = {}  # {noun: {role: {verb: {layer: vec}}}}
    
    for pair_idx, (noun_a, noun_b) in enumerate(noun_pairs):
        if pair_idx % 5 == 0:
            print(f"  Pair {pair_idx}/{len(noun_pairs)}: {noun_a}/{noun_b}")
        
        for noun in [noun_a, noun_b]:
            if noun not in reps_dict:
                reps_dict[noun] = {"subject": {}, "object": {}}
        
        # 每个动词构造一对句子
        for verb in verbs[:5]:  # 用5个动词减少时间
            # "The [A] [VERB]s the [B]" → A是主语, B是宾语
            sent_ab = f"The {noun_a} {verb} the {noun_b}"
            # "The [B] [VERB]s the [A]" → B是主语, A是宾语
            sent_ba = f"The {noun_b} {verb} the {noun_a}"
            
            for sentence, subj_noun, obj_noun in [
                (sent_ab, noun_a, noun_b),
                (sent_ba, noun_b, noun_a),
            ]:
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                input_ids = inputs['input_ids']
                
                # 找主语和宾语的位置
                subj_pos = find_noun_position(tokenizer, sentence, subj_noun)
                obj_pos = find_noun_position(tokenizer, sentence, obj_noun)
                
                if subj_pos is None or obj_pos is None:
                    continue
                
                with torch.no_grad():
                    try:
                        outputs = model(input_ids, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                        
                        for layer_idx in range(n_layers + 1):
                            # 主语位置的表示
                            if verb not in reps_dict[subj_noun]["subject"]:
                                reps_dict[subj_noun]["subject"][verb] = {}
                            reps_dict[subj_noun]["subject"][verb][layer_idx] = \
                                hidden_states[layer_idx][0, subj_pos, :].detach().cpu().float().numpy()
                            
                            # 宾语位置的表示
                            if verb not in reps_dict[obj_noun]["object"]:
                                reps_dict[obj_noun]["object"][verb] = {}
                            reps_dict[obj_noun]["object"][verb][layer_idx] = \
                                hidden_states[layer_idx][0, obj_pos, :].detach().cpu().float().numpy()
                    except Exception as e:
                        print(f"  Error for '{sentence}': {e}")
    
    # 训练PCA(用80概念)
    print("  Training PCA on 80 concepts...")
    concept_reps = {}
    mid_layer = n_layers // 2
    for concept in REPRESENTATIVE_CONCEPTS:
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        target_pos = input_ids.shape[1] - 1
        with torch.no_grad():
            try:
                outputs = model(input_ids, output_hidden_states=True)
                concept_reps[concept] = outputs.hidden_states[mid_layer+1][0, target_pos, :].detach().cpu().float().numpy()
            except:
                pass
    
    concept_vecs = np.array([concept_reps[c] for c in REPRESENTATIVE_CONCEPTS if c in concept_reps])
    concept_centered = concept_vecs - concept_vecs.mean(axis=0)
    pca = PCA(n_components=50)
    pca.fit(concept_centered)
    
    if own_model:
        release_model(model)
        print("Model released.")
    
    return reps_dict, pca, model_info


# ===== Exp1: 同句改写对比 =====
def run_exp1(model_name):
    """同句改写: 同一名词在主语vs宾语位置的差异"""
    print(f"\n{'='*60}")
    print(f"Exp1: 同句改写语法角色对比 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    d_model = model_info.d_model
    
    # 提取表示
    reps_dict, pca, _ = extract_reps_with_pca(
        model_name, NOUN_PAIRS, TRANSITIVE_VERBS, model=model, tokenizer=tokenizer, device=device
    )
    
    release_model(model)
    print("Model released.")
    
    # 分析: 同一名词, 主语位置 vs 宾语位置
    # key: 对每个名词, 在每个动词下:
    #   δ_role = rep(主语) - rep(宾语)  (纯语法角色差异!)
    
    sample_layers = [0, mid_layer//3, mid_layer//2, mid_layer, 
                     mid_layer + mid_layer//3, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l <= n_layers]))
    
    layer_results = {}
    
    for layer_idx in sample_layers:
        # 收集所有名词×动词的δ_role
        deltas = []  # δ = rep(subject) - rep(object)
        delta_norms = []
        nouns_list = []
        verbs_list = []
        
        for noun in reps_dict:
            for verb in reps_dict[noun]["subject"]:
                if verb in reps_dict[noun]["object"]:
                    if layer_idx in reps_dict[noun]["subject"][verb] and \
                       layer_idx in reps_dict[noun]["object"][verb]:
                        subj_rep = reps_dict[noun]["subject"][verb][layer_idx]
                        obj_rep = reps_dict[noun]["object"][verb][layer_idx]
                        delta = subj_rep - obj_rep
                        deltas.append(delta)
                        delta_norms.append(np.linalg.norm(delta))
                        nouns_list.append(noun)
                        verbs_list.append(verb)
        
        if len(deltas) < 5:
            continue
        
        deltas = np.array(deltas)
        delta_norms = np.array(delta_norms)
        
        # 1. δ的基本统计
        mean_norm = float(delta_norms.mean())
        std_norm = float(delta_norms.std())
        
        # 2. δ的方向一致性
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        delta_normalized = deltas / norms
        
        mean_direction = delta_normalized.mean(axis=0)
        consistency = float(np.linalg.norm(mean_direction))
        random_baseline = float(1.0 / np.sqrt(len(deltas)))
        
        # 3. δ在V_sem和V_nonsem中的投影
        delta_centered = deltas - deltas.mean(axis=0)
        
        # 投影到V_sem(Top-5 PCs)
        sem_proj = pca.transform(delta_centered)[:, :5]
        sem_energy = float(np.sum(sem_proj**2) / np.sum(delta_centered**2 + 1e-10))
        
        # 投影到V_nonsem(PCs 6-50)
        nonsem_proj = pca.transform(delta_centered)[:, 5:50]
        nonsem_energy = float(np.sum(nonsem_proj**2) / np.sum(delta_centered**2 + 1e-10))
        
        # 4. δ的PCA
        pca_delta = PCA(n_components=min(20, len(deltas)-1, deltas.shape[1]))
        pca_delta.fit(deltas)
        
        # 5. 用δ区分主语/宾语角色的准确率
        # (验证δ确实编码了语法角色)
        # 用线性探针: 从δ预测名词类别
        # 这不太合理, 换一个: δ的符号是否一致?
        # 更好的验证: δ指向"主语方向"
        
        # 6. δ的平均方向与V_sem主轴的关系
        if consistency > 1e-10:
            mean_dir_unit = mean_direction / consistency
            cos_with_pcs = []
            for pc_idx in range(5):
                pc_dir = pca.components_[pc_idx]
                cos = float(np.dot(mean_dir_unit, pc_dir) / (np.linalg.norm(pc_dir) + 1e-10))
                cos_with_pcs.append(cos)
        else:
            cos_with_pcs = [0.0] * 5
        
        # 7. 跨动词一致性: 同一名词, 不同动词的δ是否一致?
        cross_verb_consistency = {}
        for noun in reps_dict:
            noun_deltas = []
            for verb in reps_dict[noun]["subject"]:
                if verb in reps_dict[noun]["object"]:
                    if layer_idx in reps_dict[noun]["subject"][verb] and \
                       layer_idx in reps_dict[noun]["object"][verb]:
                        d = reps_dict[noun]["subject"][verb][layer_idx] - \
                            reps_dict[noun]["object"][verb][layer_idx]
                        norm = np.linalg.norm(d)
                        if norm > 1e-10:
                            noun_deltas.append(d / norm)
            
            if len(noun_deltas) >= 2:
                # 计算两两余弦相似度
                noun_deltas = np.array(noun_deltas)
                mean_dir = noun_deltas.mean(axis=0)
                cv_consist = float(np.linalg.norm(mean_dir))
                cross_verb_consistency[noun] = cv_consist
        
        mean_cv_consist = float(np.mean(list(cross_verb_consistency.values()))) if cross_verb_consistency else 0.0
        
        layer_results[layer_idx] = {
            "n_deltas": len(deltas),
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "consistency": consistency,
            "random_baseline": random_baseline,
            "consistency_ratio": consistency / random_baseline if random_baseline > 0 else 0,
            "sem_energy": sem_energy,
            "nonsem_energy": nonsem_energy,
            "delta_var_top1": float(pca_delta.explained_variance_ratio_[0]) if len(pca_delta.explained_variance_ratio_) > 0 else 0,
            "delta_var_top5": float(pca_delta.explained_variance_ratio_[:5].sum()) if len(pca_delta.explained_variance_ratio_) >= 5 else float(pca_delta.explained_variance_ratio_.sum()),
            "cos_with_vsem_pcs": cos_with_pcs,
            "cross_verb_consistency": mean_cv_consist,
        }
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    N deltas: {len(deltas)}")
        print(f"    Mean ||δ||: {mean_norm:.4f} ± {std_norm:.4f}")
        print(f"    Direction consistency: {consistency:.4f} (random={random_baseline:.4f}, ratio={consistency/random_baseline:.2f})")
        print(f"    V_sem energy: {sem_energy:.4f}, V_nonsem energy: {nonsem_energy:.4f}")
        print(f"    δ PCA: top1={pca_delta.explained_variance_ratio_[0]:.4f}, top5={pca_delta.explained_variance_ratio_[:5].sum():.4f}")
        print(f"    cos(δ_direction, V_sem PCs): {[f'{c:.3f}' for c in cos_with_pcs]}")
        print(f"    Cross-verb consistency: {mean_cv_consist:.4f}")
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "n_noun_pairs": len(NOUN_PAIRS),
        "n_verbs": 5,
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    out_path = TEMP / f"ccxxix_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


# ===== Exp2: 语法角色变换的跨概念一致性 =====
def run_exp2(model_name):
    """
    验证: 语法角色变换是否跨概念一致?
    即: rep(主语) - rep(宾语) 是否对所有名词都指向同一方向?
    
    改进: 用PCA分析δ的维度结构
    """
    print(f"\n{'='*60}")
    print(f"Exp2: 语法角色变换跨概念一致性 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 提取表示
    reps_dict, pca, _ = extract_reps_with_pca(
        model_name, NOUN_PAIRS, TRANSITIVE_VERBS, model=model, tokenizer=tokenizer, device=device
    )
    
    release_model(model)
    print("Model released.")
    
    # 分析中间层
    layer_results = {}
    
    for layer_idx in [mid_layer]:
        print(f"\n  === Layer {layer_idx} (mid) ===")
        
        # 收集: 每个名词的平均δ
        noun_deltas = {}  # {noun: mean_delta}
        noun_deltas_by_verb = {}  # {noun: {verb: delta}}
        
        for noun in reps_dict:
            deltas_for_noun = []
            deltas_by_verb = {}
            for verb in reps_dict[noun]["subject"]:
                if verb in reps_dict[noun]["object"]:
                    if layer_idx in reps_dict[noun]["subject"][verb] and \
                       layer_idx in reps_dict[noun]["object"][verb]:
                        d = reps_dict[noun]["subject"][verb][layer_idx] - \
                            reps_dict[noun]["object"][verb][layer_idx]
                        deltas_for_noun.append(d)
                        deltas_by_verb[verb] = d
            
            if len(deltas_for_noun) > 0:
                noun_deltas[noun] = np.mean(deltas_for_noun, axis=0)
                noun_deltas_by_verb[noun] = deltas_by_verb
        
        # 1. 跨名词δ的余弦相似度矩阵
        nouns_with_deltas = list(noun_deltas.keys())
        delta_vecs = np.array([noun_deltas[n] for n in nouns_with_deltas])
        
        # 归一化
        norms = np.linalg.norm(delta_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        delta_normalized = delta_vecs / norms
        
        # 余弦相似度矩阵
        cos_matrix = delta_normalized @ delta_normalized.T
        
        # 上三角平均(排除对角线)
        n = len(nouns_with_deltas)
        upper_tri = cos_matrix[np.triu_indices(n, k=1)]
        mean_cos = float(upper_tri.mean())
        std_cos = float(upper_tri.std())
        min_cos = float(upper_tri.min())
        max_cos = float(upper_tri.max())
        
        print(f"\n  Cross-noun δ consistency:")
        print(f"    Mean pairwise cos: {mean_cos:.4f} ± {std_cos:.4f}")
        print(f"    Range: [{min_cos:.4f}, {max_cos:.4f}]")
        
        # 2. δ的PCA维度分析
        pca_delta = PCA(n_components=min(10, len(delta_vecs)-1, delta_vecs.shape[1]))
        pca_delta.fit(delta_vecs)
        
        print(f"\n  δ PCA variance:")
        for i in range(min(5, len(pca_delta.explained_variance_ratio_))):
            print(f"    PC{i}: {pca_delta.explained_variance_ratio_[i]:.4f}")
        
        # 3. δ在V_sem各主轴上的投影
        delta_centered = delta_vecs - delta_vecs.mean(axis=0)
        delta_in_sem = pca.transform(delta_centered)[:, :5]
        
        print(f"\n  δ in V_sem (per-PC energy):")
        for pc_idx in range(5):
            energy = float(np.mean(delta_in_sem[:, pc_idx]**2))
            print(f"    PC{pc_idx}: energy={energy:.4f}")
        
        # 4. 按语义类别分析
        noun_to_cat = {}
        for cat, words in CATEGORIES.items():
            for w in words:
                noun_to_cat[w] = cat
        
        cat_deltas = {}
        for noun in nouns_with_deltas:
            cat = noun_to_cat.get(noun, "unknown")
            if cat not in cat_deltas:
                cat_deltas[cat] = []
            cat_deltas[cat].append(noun_deltas[noun])
        
        print(f"\n  δ by semantic category:")
        cat_results = {}
        for cat, deltas in sorted(cat_deltas.items()):
            if len(deltas) < 2:
                continue
            cat_vecs = np.array(deltas)
            cat_norms = np.linalg.norm(cat_vecs, axis=1, keepdims=True)
            cat_norms = np.maximum(cat_norms, 1e-10)
            cat_normalized = cat_vecs / cat_norms
            cat_mean_dir = cat_normalized.mean(axis=0)
            cat_consist = float(np.linalg.norm(cat_mean_dir))
            cat_mean_norm = float(np.mean(np.linalg.norm(cat_vecs, axis=1)))
            
            cat_results[cat] = {
                "n_nouns": len(deltas),
                "consistency": cat_consist,
                "mean_norm": cat_mean_norm,
            }
            print(f"    {cat:12s}: n={len(deltas):2d}, consist={cat_consist:.4f}, mean_norm={cat_mean_norm:.4f}")
        
        # 5. 跨类别δ方向对比
        cat_mean_dirs = {}
        for cat, deltas in cat_deltas.items():
            if len(deltas) >= 2:
                cat_vecs = np.array(deltas)
                cat_norms = np.linalg.norm(cat_vecs, axis=1, keepdims=True)
                cat_norms = np.maximum(cat_norms, 1e-10)
                cat_mean_dirs[cat] = (cat_vecs / cat_norms).mean(axis=0)
        
        cross_cat_cos = {}
        cat_names = sorted(cat_mean_dirs.keys())
        for i, c1 in enumerate(cat_names):
            for j, c2 in enumerate(cat_names):
                if i >= j:
                    continue
                cos = float(np.dot(cat_mean_dirs[c1], cat_mean_dirs[c2]) / 
                           (np.linalg.norm(cat_mean_dirs[c1]) * np.linalg.norm(cat_mean_dirs[c2]) + 1e-10))
                angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                cross_cat_cos[f"{c1}_vs_{c2}"] = {"cos": cos, "angle": angle}
        
        if cross_cat_cos:
            angles = [v["angle"] for v in cross_cat_cos.values()]
            print(f"\n  Cross-category δ angles: mean={np.mean(angles):.1f}°, range=[{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
        
        layer_results[layer_idx] = {
            "n_nouns_with_deltas": len(nouns_with_deltas),
            "mean_pairwise_cos": mean_cos,
            "std_pairwise_cos": std_cos,
            "delta_var_top1": float(pca_delta.explained_variance_ratio_[0]) if len(pca_delta.explained_variance_ratio_) > 0 else 0,
            "delta_var_top5": float(pca_delta.explained_variance_ratio_[:5].sum()) if len(pca_delta.explained_variance_ratio_) >= 5 else float(pca_delta.explained_variance_ratio_.sum()),
            "category_results": cat_results,
            "cross_category_angles": {k: v["angle"] for k, v in cross_cat_cos.items()} if cross_cat_cos else {},
        }
    
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "n_noun_pairs": len(NOUN_PAIRS),
        "n_verbs": 5,
        "layer_results": {str(k): v for k, v in layer_results.items()},
    }
    
    out_path = TEMP / f"ccxxix_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


# ===== Exp3: 语法角色可解码性(消除上下文效应后) =====
def run_exp3(model_name):
    """
    在同句改写条件下, 验证V_sem中语法角色的可解码性
    对比: 
      A. 只用名词的语义向量 → 能否区分语义类别? (正对照)
      B. 用δ_role = rep(subject) - rep(object) → 能否区分语法角色? (核心)
      C. 用rep(subject)或rep(object)本身 → 能否区分语义类别? (验证表示质量)
    """
    print(f"\n{'='*60}")
    print(f"Exp3: 语法角色可解码性(消除上下文后) — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 提取表示
    reps_dict, pca, _ = extract_reps_with_pca(
        model_name, NOUN_PAIRS, TRANSITIVE_VERBS, model=model, tokenizer=tokenizer, device=device
    )
    
    release_model(model)
    print("Model released.")
    
    # 分析中间层
    for layer_idx in [mid_layer]:
        print(f"\n  === Layer {layer_idx} (mid) ===")
        
        # 收集: 主语位置的表示, 宾语位置的表示
        subj_reps = []  # 名词在主语位置的表示
        obj_reps = []   # 名词在宾语位置的表示
        delta_roles = []  # δ = subj - obj
        noun_labels = []
        verb_labels = []
        
        for noun in reps_dict:
            for verb in reps_dict[noun]["subject"]:
                if verb in reps_dict[noun]["object"]:
                    if layer_idx in reps_dict[noun]["subject"][verb] and \
                       layer_idx in reps_dict[noun]["object"][verb]:
                        subj_reps.append(reps_dict[noun]["subject"][verb][layer_idx])
                        obj_reps.append(reps_dict[noun]["object"][verb][layer_idx])
                        delta_roles.append(
                            reps_dict[noun]["subject"][verb][layer_idx] - 
                            reps_dict[noun]["object"][verb][layer_idx]
                        )
                        noun_labels.append(noun)
                        verb_labels.append(verb)
        
        if len(subj_reps) < 10:
            print("  Not enough data!")
            return {}
        
        subj_reps = np.array(subj_reps)
        obj_reps = np.array(obj_reps)
        delta_roles = np.array(delta_roles)
        
        # 名词→类别映射
        noun_to_cat = {}
        for cat, words in CATEGORIES.items():
            for w in words:
                noun_to_cat[w] = cat
        cat_labels = [noun_to_cat.get(n, "unknown") for n in noun_labels]
        
        # 1. 主语表示的语义类别可解码性
        print("\n  [A] Subject rep → semantic category:")
        pca_subj = PCA(n_components=min(30, subj_reps.shape[0]-1, subj_reps.shape[1]))
        X_subj = pca_subj.fit_transform(subj_reps - subj_reps.mean(axis=0))
        try:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = cross_val_score(clf, X_subj, cat_labels, cv=min(5, len(set(cat_labels))), scoring='accuracy')
            print(f"    CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
        except:
            print(f"    CV failed")
        
        # 2. 宾语表示的语义类别可解码性
        print("\n  [B] Object rep → semantic category:")
        pca_obj = PCA(n_components=min(30, obj_reps.shape[0]-1, obj_reps.shape[1]))
        X_obj = pca_obj.fit_transform(obj_reps - obj_reps.mean(axis=0))
        try:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = cross_val_score(clf, X_obj, cat_labels, cv=min(5, len(set(cat_labels))), scoring='accuracy')
            print(f"    CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
        except:
            print(f"    CV failed")
        
        # 3. δ_role的语义类别可解码性
        print("\n  [C] δ_role = subj - obj → semantic category:")
        pca_delta = PCA(n_components=min(30, delta_roles.shape[0]-1, delta_roles.shape[1]))
        X_delta = pca_delta.fit_transform(delta_roles - delta_roles.mean(axis=0))
        try:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = cross_val_score(clf, X_delta, cat_labels, cv=min(5, len(set(cat_labels))), scoring='accuracy')
            print(f"    CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
        except:
            print(f"    CV failed")
        
        # 4. δ_role在V_sem vs V_nonsem中的能量
        delta_centered = delta_roles - delta_roles.mean(axis=0)
        delta_in_sem = pca.transform(delta_centered)[:, :5]
        delta_in_nonsem = pca.transform(delta_centered)[:, 5:50]
        
        total_energy = float(np.sum(delta_centered**2))
        sem_energy = float(np.sum(delta_in_sem**2))
        nonsem_energy = float(np.sum(delta_in_nonsem**2))
        
        print(f"\n  [D] δ_role energy distribution:")
        print(f"    V_sem (5d): {sem_energy/total_energy*100:.1f}%")
        print(f"    V_nonsem:   {nonsem_energy/total_energy*100:.1f}%")
        print(f"    Total:      {(sem_energy+nonsem_energy)/total_energy*100:.1f}%")
        
        # 5. 主语 vs 宾语表示的平均差异方向
        mean_subj = subj_reps.mean(axis=0)
        mean_obj = obj_reps.mean(axis=0)
        role_direction = mean_subj - mean_obj
        role_dir_norm = np.linalg.norm(role_direction)
        
        if role_dir_norm > 1e-10:
            role_dir_unit = role_direction / role_dir_norm
            # 这个方向与V_sem主轴的余弦
            cos_with_pcs = []
            for pc_idx in range(5):
                cos = float(np.dot(role_dir_unit, pca.components_[pc_idx]))
                cos_with_pcs.append(cos)
            
            print(f"\n  [E] Mean role direction (subj→obj):")
            print(f"    ||direction|| = {role_dir_norm:.4f}")
            print(f"    cos with V_sem PCs: {[f'{c:.4f}' for c in cos_with_pcs]}")
        
        # 6. 每个名词的平均δ, 在V_sem中的可视化
        noun_mean_delta = {}
        for noun in reps_dict:
            deltas = []
            for verb in reps_dict[noun]["subject"]:
                if verb in reps_dict[noun]["object"]:
                    if layer_idx in reps_dict[noun]["subject"][verb] and \
                       layer_idx in reps_dict[noun]["object"][verb]:
                        deltas.append(
                            reps_dict[noun]["subject"][verb][layer_idx] - 
                            reps_dict[noun]["object"][verb][layer_idx]
                        )
            if deltas:
                noun_mean_delta[noun] = np.mean(deltas, axis=0)
        
        if noun_mean_delta:
            delta_vecs = np.array([noun_mean_delta[n] for n in sorted(noun_mean_delta.keys())])
            delta_centered_n = delta_vecs - delta_vecs.mean(axis=0)
            delta_in_sem_n = pca.transform(delta_centered_n)[:, :5]
            
            print(f"\n  [F] Per-noun δ in V_sem (sample):")
            sorted_nouns = sorted(noun_mean_delta.keys())
            for i, noun in enumerate(sorted_nouns[:10]):
                cat = noun_to_cat.get(noun, "?")
                print(f"    {noun:10s} ({cat:10s}): PC0={delta_in_sem_n[i,0]:+.3f}, PC1={delta_in_sem_n[i,1]:+.3f}, PC2={delta_in_sem_n[i,2]:+.3f}")
    
    return {"model": model_name, "status": "completed"}


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
