"""
CCXXX(380): 位置-语法分离实验 — 位置编码vs语法角色编码的解耦

核心问题:
  CCXXIX发现 δ_role = rep(noun_主语) - rep(noun_宾语) 与V_sem正交(V_sem能量<1.3%)
  但CCXXIX的设计中, 主语在句子早期位置, 宾语在晚期位置:
    "The cat chases the dog" → cat在pos1(主语), dog在pos4(宾语)
    "The dog chases the cat" → cat在pos4(宾语), dog在pos1(主语)
  
  δ_role = rep(cat_pos1_subj) - rep(cat_pos4_obj) = 位置效应 + 语法角色效应
  
  关键: 位置效应有多大? 能否解释δ_role的全部?
  
实验设计:
  Exp1: 位置效应测量
    - 同一名词, 同一语法角色(主语), 不同位置:
      "The cat chases the dog"  → cat在pos~1
      "Soon the cat chases the dog" → cat在pos~2 (加1个前置词)
      "Yes, the cat chases the dog" → cat在pos~2-3 (加2个前置词)
    - δ_pos = rep(pos_shifted) - rep(pos_original)
    - 测量: ||δ_pos||, V_sem能量, 方向一致性, 与δ_role的关系

  Exp2: 角色-位置分解
    - δ_role = δ_position + δ_role_pure
    - 位置偏移量 = pos4 - pos1 ≈ 3个token
    - δ_position ≈ 3 × mean_δ_pos_per_token (从Exp1估计)
    - δ_role_pure = δ_role - δ_position
    - 分析δ_role_pure的几何特性

  Exp3: 全因子设计(2角色 × 2位置)
    - 使用关系从句构造:
      Subj+Early: "The cat that sees the dog runs" → cat在pos1, 主语
      Subj+Late:  "The dog that the cat sees runs" → cat在pos4, 主语(关系从句中)
      Obj+Early:  "The cat that the dog sees runs" → cat在pos1, 宾语(关系从句中)
      Obj+Late:   "The dog sees the cat that runs" → cat在pos4, 宾语
    - 2×2因子分析: δ_role_pure, δ_pos_pure, 交互效应
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

FILLER_WORDS = ["Soon", "Then", "Now", "Yes", "Still"]  # 前置副词

TRANSITIVE_VERBS = ["chases", "sees", "finds", "takes", "watches"]

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


def train_pca(model, tokenizer, device, model_info):
    """训练V_sem PCA"""
    n_layers = model_info.n_layers
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
    
    X = np.array(concept_reps)
    pca = PCA(n_components=50)
    pca.fit(X)
    
    return pca, mid_layer


# ============================================================
# Exp1: 位置效应测量
# ============================================================
def run_exp1(model_name):
    """测量位置效应: 同名词同角色, 不同位置"""
    print(f"\n{'='*60}")
    print(f"CCXXX Exp1: Position Effect — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 训练PCA
    print("Training PCA...")
    pca, _ = train_pca(model, tokenizer, device, model_info)
    
    # 收集位置变体表示
    # 条件: "The [X] [VERB]s the [Y]" (baseline)
    #        "[FILLER] the [X] [VERB]s the [Y]" (shifted)
    
    position_reps = {}  # {noun: {condition: {verb: {layer: vec}}}}
    # conditions: "base", "shift1", "shift2"
    
    all_nouns = [n for pair in NOUN_PAIRS for n in pair]
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for noun in [noun_a, noun_b]:
            if noun not in position_reps:
                position_reps[noun] = {}
            
            for verb in TRANSITIVE_VERBS[:3]:  # 3个动词减少时间
                # Baseline: "The [noun] [verb] the [noun_b]"
                # 这里noun是主语
                other = noun_b if noun == noun_a else noun_a
                
                conditions = {
                    "base": f"The {noun} {verb} the {other}",
                    "shift1": f"Soon the {noun} {verb} the {other}",
                    "shift2": f"Yes, the {noun} {verb} the {other}",
                }
                
                for cond_name, sentence in conditions.items():
                    if cond_name not in position_reps[noun]:
                        position_reps[noun][cond_name] = {}
                    
                    noun_pos = find_noun_position(tokenizer, sentence, noun)
                    if noun_pos is None:
                        print(f"  WARNING: Cannot find '{noun}' in '{sentence}'")
                        continue
                    
                    inputs = tokenizer(sentence, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            outputs = model(inputs['input_ids'], output_hidden_states=True)
                            # 收集多个层
                            for layer_idx in [0, n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]:
                                if verb not in position_reps[noun][cond_name]:
                                    position_reps[noun][cond_name][verb] = {}
                                position_reps[noun][cond_name][verb][layer_idx] = \
                                    outputs.hidden_states[layer_idx][0, noun_pos, :].detach().cpu().float().numpy()
                        except Exception as e:
                            print(f"  Error: '{sentence}': {e}")
    
    # ===== 分析 =====
    results = {"model": model_name, "exp": 1}
    
    # 1. 位置效应δ_pos
    print("\n--- Position Effect Analysis ---")
    
    for layer_idx in [0, n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]:
        deltas_shift1 = []
        deltas_shift2 = []
        
        for noun in all_nouns:
            for verb in TRANSITIVE_VERBS[:3]:
                try:
                    base = position_reps[noun]["base"][verb][layer_idx]
                    shift1 = position_reps[noun]["shift1"][verb][layer_idx]
                    shift2 = position_reps[noun]["shift2"][verb][layer_idx]
                    
                    d1 = shift1 - base
                    d2 = shift2 - base
                    deltas_shift1.append(d1)
                    deltas_shift2.append(d2)
                except KeyError:
                    continue
        
        if not deltas_shift1:
            continue
        
        deltas_shift1 = np.array(deltas_shift1)
        deltas_shift2 = np.array(deltas_shift2)
        
        # 统计量
        norms1 = np.linalg.norm(deltas_shift1, axis=1)
        norms2 = np.linalg.norm(deltas_shift2, axis=1)
        
        # 方向一致性
        mean_d1 = deltas_shift1.mean(axis=0)
        mean_d2 = deltas_shift2.mean(axis=0)
        
        cos_values_1 = [np.dot(d, mean_d1) / (np.linalg.norm(d) * np.linalg.norm(mean_d1) + 1e-10) 
                        for d in deltas_shift1]
        cos_values_2 = [np.dot(d, mean_d2) / (np.linalg.norm(d) * np.linalg.norm(mean_d2) + 1e-10) 
                        for d in deltas_shift2]
        
        consistency_1 = np.mean(cos_values_1)
        consistency_2 = np.mean(cos_values_2)
        
        # 随机基线
        rand_d1 = np.random.randn(*deltas_shift1.shape)
        rand_mean1 = rand_d1.mean(axis=0)
        rand_consistency = np.mean([np.dot(d, rand_mean1) / (np.linalg.norm(d) * np.linalg.norm(rand_mean1) + 1e-10) 
                                   for d in rand_d1])
        ratio_1 = consistency_1 / max(rand_consistency, 1e-10)
        ratio_2 = consistency_2 / max(rand_consistency, 1e-10)
        
        # V_sem能量
        pc_axes = pca.components_[:5]  # Top-5 PCs
        vsem_energy_1 = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                                 for d in deltas_shift1])
        vsem_energy_2 = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                                 for d in deltas_shift2])
        
        # δ_shift1 vs δ_shift2的关系
        cross_cos = np.dot(mean_d1, mean_d2) / (np.linalg.norm(mean_d1) * np.linalg.norm(mean_d2) + 1e-10)
        
        # PCA分析δ_pos的结构
        pca_pos = PCA(n_components=min(10, deltas_shift1.shape[0]-1))
        pca_pos.fit(deltas_shift1)
        top1_var = pca_pos.explained_variance_ratio_[0]
        top5_var = np.sum(pca_pos.explained_variance_ratio_[:5])
        
        layer_key = f"L{layer_idx}"
        results[layer_key] = {
            "mean_norm_shift1": float(np.mean(norms1)),
            "mean_norm_shift2": float(np.mean(norms2)),
            "consistency_shift1": float(consistency_1),
            "consistency_shift2": float(consistency_2),
            "consistency_ratio_1": float(ratio_1),
            "consistency_ratio_2": float(ratio_2),
            "vsem_energy_shift1": float(vsem_energy_1),
            "vsem_energy_shift2": float(vsem_energy_2),
            "cross_cos_1vs2": float(cross_cos),
            "pca_top1_var": float(top1_var),
            "pca_top5_var": float(top5_var),
        }
        
        print(f"  L{layer_idx}: ||δ_pos1||={np.mean(norms1):.2f}, ||δ_pos2||={np.mean(norms2):.2f}, "
              f"consist1={consistency_1:.3f}(×{ratio_1:.1f}), consist2={consistency_2:.3f}(×{ratio_2:.1f}), "
              f"V_sem%={vsem_energy_1*100:.1f}/{vsem_energy_2*100:.1f}, "
              f"cos(δ1,δ2)={cross_cos:.3f}, top1={top1_var*100:.1f}%")
    
    # 2. 位置效应vs语法角色效应的比较
    print("\n--- Position vs Role Effect Comparison ---")
    
    # 收集语法角色δ(与CCXXIX相同的设计)
    role_reps = {}  # {noun: {role: {verb: {layer: vec}}}}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        for noun in [noun_a, noun_b]:
            if noun not in role_reps:
                role_reps[noun] = {"subject": {}, "object": {}}
        
        for verb in TRANSITIVE_VERBS[:3]:
            sent_ab = f"The {noun_a} {verb} the {noun_b}"
            sent_ba = f"The {noun_b} {verb} the {noun_a}"
            
            for sentence, subj_noun, obj_noun in [(sent_ab, noun_a, noun_b), (sent_ba, noun_b, noun_a)]:
                subj_pos = find_noun_position(tokenizer, sentence, subj_noun)
                obj_pos = find_noun_position(tokenizer, sentence, obj_noun)
                
                if subj_pos is None or obj_pos is None:
                    continue
                
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        outputs = model(inputs['input_ids'], output_hidden_states=True)
                        for layer_idx in [0, n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]:
                            if verb not in role_reps[subj_noun]["subject"]:
                                role_reps[subj_noun]["subject"][verb] = {}
                            role_reps[subj_noun]["subject"][verb][layer_idx] = \
                                outputs.hidden_states[layer_idx][0, subj_pos, :].detach().cpu().float().numpy()
                            
                            if verb not in role_reps[obj_noun]["object"]:
                                role_reps[obj_noun]["object"][verb] = {}
                            role_reps[obj_noun]["object"][verb][layer_idx] = \
                                outputs.hidden_states[layer_idx][0, obj_pos, :].detach().cpu().float().numpy()
                    except Exception as e:
                        print(f"  Error: {e}")
    
    # 比较δ_role vs δ_pos
    for layer_idx in [mid_layer]:
        deltas_role = []
        deltas_pos = []
        
        for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
            for noun in [noun_a, noun_b]:
                for verb in TRANSITIVE_VERBS[:3]:
                    try:
                        subj_rep = role_reps[noun]["subject"][verb][layer_idx]
                        obj_rep = role_reps[noun]["object"][verb][layer_idx]
                        base_rep = position_reps[noun]["base"][verb][layer_idx]
                        shift1_rep = position_reps[noun]["shift1"][verb][layer_idx]
                        
                        d_role = subj_rep - obj_rep
                        d_pos = shift1_rep - base_rep
                        
                        deltas_role.append(d_role)
                        deltas_pos.append(d_pos)
                    except KeyError:
                        continue
        
        if not deltas_role:
            continue
        
        deltas_role = np.array(deltas_role)
        deltas_pos = np.array(deltas_pos)
        
        # 大小比较
        mean_norm_role = np.mean(np.linalg.norm(deltas_role, axis=1))
        mean_norm_pos = np.mean(np.linalg.norm(deltas_pos, axis=1))
        
        # 方向比较
        mean_role = deltas_role.mean(axis=0)
        mean_pos = deltas_pos.mean(axis=0)
        cos_role_pos = np.dot(mean_role, mean_pos) / (np.linalg.norm(mean_role) * np.linalg.norm(mean_pos) + 1e-10)
        angle_role_pos = np.degrees(np.arccos(np.clip(cos_role_pos, -1, 1)))
        
        # 逐样本cos
        pairwise_cos = [np.dot(dr, dp) / (np.linalg.norm(dr) * np.linalg.norm(dp) + 1e-10) 
                       for dr, dp in zip(deltas_role, deltas_pos)]
        mean_pairwise_cos = np.mean(pairwise_cos)
        
        # 位置效应能否预测角色效应?
        # 如果δ_role ≈ k × δ_pos (位置完全解释角色), 则cos应接近1
        # 实际位置差 ≈ 3个token (从pos1到pos4)
        # δ_role ≈ rep(subj_pos1) - rep(obj_pos4) = rep(subj_pos1) - rep(subj_pos4) + rep(subj_pos4) - rep(obj_pos4)
        # 简化: 如果位置效应是线性的, δ_role ≈ 3 × δ_pos + δ_role_pure
        
        # V_sem能量对比
        pc_axes = pca.components_[:5]
        vsem_role = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                            for d in deltas_role])
        vsem_pos = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                           for d in deltas_pos])
        
        results["comparison"] = {
            "mean_norm_role": float(mean_norm_role),
            "mean_norm_pos": float(mean_norm_pos),
            "norm_ratio_role_vs_pos": float(mean_norm_role / (mean_norm_pos + 1e-10)),
            "cos_role_pos_mean": float(cos_role_pos),
            "angle_role_pos": float(angle_role_pos),
            "mean_pairwise_cos": float(mean_pairwise_cos),
            "vsem_energy_role": float(vsem_role),
            "vsem_energy_pos": float(vsem_pos),
        }
        
        print(f"  ||δ_role||={mean_norm_role:.2f}, ||δ_pos(1token)||={mean_norm_pos:.2f}, "
              f"ratio={mean_norm_role/mean_norm_pos:.1f}")
        print(f"  cos(δ_role, δ_pos)={cos_role_pos:.3f}, angle={angle_role_pos:.1f}°")
        print(f"  Mean pairwise cos={mean_pairwise_cos:.3f}")
        print(f"  V_sem%: role={vsem_role*100:.1f}%, pos={vsem_pos*100:.1f}%")
        
        # 3. 位置校正后的纯角色效应
        print("\n--- Position-Corrected Pure Role Effect ---")
        
        # 估计每token位置效应
        # 从Exp1: shift1是1个token的位移, shift2是2个token的位移
        # 位置效应应该是近似线性的(先验证)
        
        # 收集shift2的δ
        deltas_pos2 = []
        for noun in all_nouns:
            for verb in TRANSITIVE_VERBS[:3]:
                try:
                    base = position_reps[noun]["base"][verb][layer_idx]
                    shift2 = position_reps[noun]["shift2"][verb][layer_idx]
                    deltas_pos2.append(shift2 - base)
                except KeyError:
                    continue
        
        deltas_pos2 = np.array(deltas_pos2)
        mean_dpos2 = deltas_pos2.mean(axis=0)
        mean_dpos1 = deltas_pos.mean(axis=0)
        
        # 线性度检查: δ_pos2 ≈ 2 × δ_pos1?
        cos_linear = np.dot(mean_dpos2, mean_dpos1) / (np.linalg.norm(mean_dpos2) * np.linalg.norm(mean_dpos1) + 1e-10)
        norm_ratio_2vs1 = np.linalg.norm(mean_dpos2) / (np.linalg.norm(mean_dpos1) + 1e-10)
        
        print(f"  Position linearity: cos(δ_pos2, δ_pos1)={cos_linear:.3f}, "
              f"||δ_pos2||/||δ_pos1||={norm_ratio_2vs1:.2f} (expect ~2.0 if linear)")
        
        # 位置校正: δ_role ≈ 3 × δ_pos_per_token + δ_role_pure
        # 主语在pos1, 宾语在pos4, 差3个位置
        # 但实际上, 名词在两个句子中的绝对位置不同:
        # "The cat chases the dog": cat=1, dog=4
        # "The dog chases the cat": cat=4, dog=1
        # δ_role = rep(cat@1_subj) - rep(cat@4_obj)
        # 位置差=3, 所以位置效应 ≈ 3 × δ_pos_per_token
        
        # 但更精确的做法: δ_role_pure = δ_role - position_projection
        # position_projection = proj(δ_role onto δ_pos_direction) * δ_pos_direction
        
        pos_direction = mean_dpos1 / (np.linalg.norm(mean_dpos1) + 1e-10)
        
        # 对每个δ_role, 计算位置校正
        pure_role_deltas = []
        for d_role in deltas_role:
            # 位置投影: 假设位置差=3, 则位置分量 ≈ 3 × |δ_pos_per_token| × pos_direction
            pos_component = np.dot(d_role, pos_direction) * pos_direction
            pure_role = d_role - pos_component
            pure_role_deltas.append(pure_role)
        
        pure_role_deltas = np.array(pure_role_deltas)
        
        # 纯角色效应的统计
        pure_norms = np.linalg.norm(pure_role_deltas, axis=1)
        mean_pure_norm = np.mean(pure_norms)
        
        # 方向一致性
        mean_pure = pure_role_deltas.mean(axis=0)
        pure_consistency = np.mean([np.dot(d, mean_pure) / (np.linalg.norm(d) * np.linalg.norm(mean_pure) + 1e-10) 
                                   for d in pure_role_deltas])
        
        rand_pr = np.random.randn(*pure_role_deltas.shape)
        rand_pr_mean = rand_pr.mean(axis=0)
        rand_consist = np.mean([np.dot(d, rand_pr_mean) / (np.linalg.norm(d) * np.linalg.norm(rand_pr_mean) + 1e-10) 
                               for d in rand_pr])
        pure_ratio = pure_consistency / max(rand_consist, 1e-10)
        
        # V_sem能量
        vsem_pure = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                            for d in pure_role_deltas])
        
        # 位置分量在δ_role中的占比
        pos_fraction = 1 - (mean_pure_norm / (mean_norm_role + 1e-10)) ** 2
        
        results["pure_role"] = {
            "mean_norm_pure": float(mean_pure_norm),
            "mean_norm_role": float(mean_norm_role),
            "pos_fraction": float(pos_fraction),
            "pure_consistency": float(pure_consistency),
            "pure_consistency_ratio": float(pure_ratio),
            "vsem_energy_pure": float(vsem_pure),
            "vsem_energy_role": float(vsem_role),
            "position_linearity_cos": float(cos_linear),
            "position_linearity_ratio": float(norm_ratio_2vs1),
        }
        
        print(f"  ||δ_role_pure||={mean_pure_norm:.2f} vs ||δ_role||={mean_norm_role:.2f}")
        print(f"  Position fraction: {pos_fraction*100:.1f}%")
        print(f"  Pure role consistency: {pure_consistency:.3f} (×{pure_ratio:.1f})")
        print(f"  V_sem%: pure_role={vsem_pure*100:.1f}% vs raw_role={vsem_role*100:.1f}%")
    
    # 保存结果
    out_path = TEMP / f"ccxxx_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp2: 全因子设计 — 2角色 × 2位置
# ============================================================
def run_exp2(model_name):
    """
    使用关系从句构造2×2因子设计:
    Subj+Early: "The cat that sees the dog runs"  → cat@pos~1, subject
    Subj+Late:  "The dog that the cat sees runs"  → cat@pos~4, subject  
    Obj+Early:  "The cat that the dog sees runs"  → cat@pos~1, object
    Obj+Late:   "The dog sees the cat that runs"   → cat@pos~4, object
    """
    print(f"\n{'='*60}")
    print(f"CCXXX Exp2: 2×2 Factorial (Role × Position) — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 训练PCA
    print("Training PCA...")
    pca, _ = train_pca(model, tokenizer, device, model_info)
    
    # 4种条件
    conditions = ["subj_early", "subj_late", "obj_early", "obj_late"]
    
    factorial_reps = {}  # {noun: {condition: {verb: {layer: vec}}}}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for noun in [noun_a, noun_b]:
            if noun not in factorial_reps:
                factorial_reps[noun] = {c: {} for c in conditions}
        
        other_a = noun_b  # noun_a的配对
        other_b = noun_a  # noun_b的配对
        
        for verb in TRANSITIVE_VERBS[:3]:
            # Verb root (去掉s)
            verb_root = verb.rstrip('s')
            
            # 构造4种句子 (对每个名词)
            for noun, other in [(noun_a, noun_b), (noun_b, noun_a)]:
                # Subj+Early: "The [noun] that [verb]s the [other] runs"
                sent_se = f"The {noun} that {verb} the {other} runs"
                # Subj+Late:  "The [other] that the [noun] [verb]s runs" 
                sent_sl = f"The {other} that the {noun} {verb} runs"
                # Obj+Early:  "The [noun] that the [other] [verb]s runs"
                sent_oe = f"The {noun} that the {other} {verb} runs"
                # Obj+Late:   "The [other] [verb]s the [noun] that runs"
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
                        # 尝试更宽松的搜索
                        tokens = tokenizer(sentence)['input_ids']
                        decoded = [tokenizer.decode([t]).strip() for t in tokens]
                        # print(f"  Tokens for '{sentence}': {decoded}")
                        # print(f"  Looking for '{noun}'")
                        continue
                    
                    if verb not in factorial_reps[noun][cond_name]:
                        factorial_reps[noun][cond_name][verb] = {}
                    
                    inputs = tokenizer(sentence, return_tensors='pt').to(device)
                    with torch.no_grad():
                        try:
                            outputs = model(inputs['input_ids'], output_hidden_states=True)
                            for layer_idx in [0, n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]:
                                factorial_reps[noun][cond_name][verb][layer_idx] = \
                                    outputs.hidden_states[layer_idx][0, noun_pos, :].detach().cpu().float().numpy()
                        except Exception as e:
                            print(f"  Error: '{sentence}': {e}")
    
    # ===== 2×2因子分析 =====
    results = {"model": model_name, "exp": 2}
    
    all_nouns = [n for pair in NOUN_PAIRS for n in pair]
    
    for layer_idx in [mid_layer]:
        print(f"\n--- Layer {layer_idx} (mid) ---")
        
        # 收集4种条件的表示
        data = {c: [] for c in conditions}
        
        for noun in all_nouns:
            for verb in TRANSITIVE_VERBS[:3]:
                reps = {}
                for c in conditions:
                    try:
                        reps[c] = factorial_reps[noun][c][verb][layer_idx]
                    except KeyError:
                        reps = {}
                        break
                
                if len(reps) == 4:
                    for c in conditions:
                        data[c].append({"noun": noun, "verb": verb, "rep": reps[c]})
        
        n_valid = len(data["subj_early"])
        print(f"  Valid samples: {n_valid}")
        
        if n_valid < 10:
            print("  Too few valid samples, skipping")
            continue
        
        # 计算效应
        # δ_role = mean(subj_early + subj_late) - mean(obj_early + obj_late)
        # δ_pos = mean(subj_early + obj_early) - mean(subj_late + obj_late)
        # δ_interaction = (subj_early - subj_late) - (obj_early - obj_late)
        
        role_deltas = []    # 主效应: 角色
        pos_deltas = []     # 主效应: 位置
        interact_deltas = [] # 交互效应
        
        for i in range(n_valid):
            se = data["subj_early"][i]["rep"]
            sl = data["subj_late"][i]["rep"]
            oe = data["obj_early"][i]["rep"]
            ol = data["obj_late"][i]["rep"]
            
            # 角色: (se + sl)/2 - (oe + ol)/2 = (se + sl - oe - ol)/2
            d_role = (se + sl - oe - ol) / 2
            # 位置: (se + oe)/2 - (sl + ol)/2 = (se + oe - sl - ol)/2
            d_pos = (se + oe - sl - ol) / 2
            # 交互: (se - sl) - (oe - ol) = se - sl - oe + ol
            d_interact = se - sl - oe + ol
            
            role_deltas.append(d_role)
            pos_deltas.append(d_pos)
            interact_deltas.append(d_interact)
        
        role_deltas = np.array(role_deltas)
        pos_deltas = np.array(pos_deltas)
        interact_deltas = np.array(interact_deltas)
        
        # 统计量
        def analyze_deltas(deltas, name):
            norms = np.linalg.norm(deltas, axis=1)
            mean_d = deltas.mean(axis=0)
            consist = np.mean([np.dot(d, mean_d) / (np.linalg.norm(d) * np.linalg.norm(mean_d) + 1e-10) 
                              for d in deltas])
            
            # 随机基线
            rand_d = np.random.randn(*deltas.shape)
            rand_mean = rand_d.mean(axis=0)
            rand_c = np.mean([np.dot(d, rand_mean) / (np.linalg.norm(d) * np.linalg.norm(rand_mean) + 1e-10) 
                             for d in rand_d])
            ratio = consist / max(rand_c, 1e-10)
            
            # V_sem能量
            pc_axes = pca.components_[:5]
            vsem = np.mean([np.sum((np.dot(d, pc_axes.T) ** 2)) / (np.linalg.norm(d) ** 2 + 1e-10) 
                           for d in deltas])
            
            # PCA
            pca_d = PCA(n_components=min(10, deltas.shape[0]-1))
            pca_d.fit(deltas)
            
            return {
                "mean_norm": float(np.mean(norms)),
                "consistency": float(consist),
                "consistency_ratio": float(ratio),
                "vsem_energy": float(vsem),
                "pca_top1": float(pca_d.explained_variance_ratio_[0]),
                "pca_top5": float(np.sum(pca_d.explained_variance_ratio_[:5])),
            }, mean_d
        
        role_stats, mean_role = analyze_deltas(role_deltas, "role")
        pos_stats, mean_pos = analyze_deltas(pos_deltas, "position")
        interact_stats, mean_interact = analyze_deltas(interact_deltas, "interaction")
        
        # 效应间关系
        cos_role_pos = np.dot(mean_role, mean_pos) / (np.linalg.norm(mean_role) * np.linalg.norm(mean_pos) + 1e-10)
        cos_role_interact = np.dot(mean_role, mean_interact) / (np.linalg.norm(mean_role) * np.linalg.norm(mean_interact) + 1e-10)
        cos_pos_interact = np.dot(mean_pos, mean_interact) / (np.linalg.norm(mean_pos) * np.linalg.norm(mean_interact) + 1e-10)
        
        # 效应大小比较
        total_norm = role_stats["mean_norm"] ** 2 + pos_stats["mean_norm"] ** 2 + interact_stats["mean_norm"] ** 2
        
        print(f"  Role effect:     ||δ||={role_stats['mean_norm']:.2f}, "
              f"consist={role_stats['consistency']:.3f}(×{role_stats['consistency_ratio']:.1f}), "
              f"V_sem%={role_stats['vsem_energy']*100:.1f}, top1={role_stats['pca_top1']*100:.1f}%")
        print(f"  Position effect: ||δ||={pos_stats['mean_norm']:.2f}, "
              f"consist={pos_stats['consistency']:.3f}(×{pos_stats['consistency_ratio']:.1f}), "
              f"V_sem%={pos_stats['vsem_energy']*100:.1f}, top1={pos_stats['pca_top1']*100:.1f}%")
        print(f"  Interaction:     ||δ||={interact_stats['mean_norm']:.2f}, "
              f"consist={interact_stats['consistency']:.3f}(×{interact_stats['consistency_ratio']:.1f}), "
              f"V_sem%={interact_stats['vsem_energy']*100:.1f}, top1={interact_stats['pca_top1']*100:.1f}%")
        print(f"  Cos(role, pos)={cos_role_pos:.3f}, cos(role, interact)={cos_role_interact:.3f}, "
              f"cos(pos, interact)={cos_pos_interact:.3f}")
        print(f"  Variance partition: role={role_stats['mean_norm']**2/total_norm*100:.1f}%, "
              f"pos={pos_stats['mean_norm']**2/total_norm*100:.1f}%, "
              f"interact={interact_stats['mean_norm']**2/total_norm*100:.1f}%")
        
        results[f"L{layer_idx}"] = {
            "role": role_stats,
            "position": pos_stats,
            "interaction": interact_stats,
            "cos_role_pos": float(cos_role_pos),
            "cos_role_interact": float(cos_role_interact),
            "cos_pos_interact": float(cos_pos_interact),
            "variance_partition": {
                "role_pct": float(role_stats['mean_norm']**2 / total_norm * 100),
                "pos_pct": float(pos_stats['mean_norm']**2 / total_norm * 100),
                "interact_pct": float(interact_stats['mean_norm']**2 / total_norm * 100),
            },
            "n_valid": n_valid,
        }
        
        # 逐名词验证: 纯角色效应是否跨名词一致
        print("\n  Per-noun pure role effect (2×2 factorial)...")
        
        # 对每个名词, 计算纯角色效应
        noun_role_deltas = {}
        for i in range(n_valid):
            noun = data["subj_early"][i]["noun"]
            se = data["subj_early"][i]["rep"]
            sl = data["subj_late"][i]["rep"]
            oe = data["obj_early"][i]["rep"]
            ol = data["obj_late"][i]["rep"]
            
            d = (se + sl - oe - ol) / 2
            if noun not in noun_role_deltas:
                noun_role_deltas[noun] = []
            noun_role_deltas[noun].append(d)
        
        # 类内和跨类一致性
        cat_consistencies = {}
        for cat_name, cat_concepts in CATEGORIES.items():
            cat_deltas = []
            for concept in cat_concepts:
                if concept in noun_role_deltas:
                    cat_deltas.extend(noun_role_deltas[concept])
            
            if len(cat_deltas) < 2:
                continue
            
            cat_deltas = np.array(cat_deltas)
            cat_mean = cat_deltas.mean(axis=0)
            cat_consist = np.mean([np.dot(d, cat_mean) / (np.linalg.norm(d) * np.linalg.norm(cat_mean) + 1e-10) 
                                  for d in cat_deltas])
            cat_consistencies[cat_name] = float(cat_consist)
        
        results[f"L{layer_idx}"]["category_consistencies"] = cat_consistencies
        print(f"  Category consistencies: {cat_consistencies}")
    
    # 保存结果
    out_path = TEMP / f"ccxxx_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp3: V_sem中位置vs角色的分离
# ============================================================
def run_exp3(model_name):
    """
    在V_sem的5个主分量中, 分析位置信息和角色信息的分离:
    1. 每个PC编码的是位置还是角色?
    2. 位置和角色是否在不同的PC上?
    """
    print(f"\n{'='*60}")
    print(f"CCXXX Exp3: V_sem Position vs Role Separation — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    mid_layer = n_layers // 2
    
    # 训练PCA
    print("Training PCA...")
    pca, _ = train_pca(model, tokenizer, device, model_info)
    pc_axes = pca.components_[:10]  # Top-10 PCs
    
    # 收集CCXXIX式的角色数据 + 位置变体
    # 条件: subject_base, subject_shifted, object_base
    all_data = []  # list of {noun, verb, role, position, rep}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        if noun_idx % 5 == 0:
            print(f"  Pair {noun_idx}/{len(NOUN_PAIRS)}")
        
        for verb in TRANSITIVE_VERBS[:3]:
            # Subj baseline: "The [A] [verb] the [B]"
            sent_subj = f"The {noun_a} {verb} the {noun_b}"
            # Obj baseline: "The [B] [verb] the [A]"  
            sent_obj = f"The {noun_b} {verb} the {noun_a}"
            # Subj shifted: "Soon the [A] [verb] the [B]"
            sent_subj_shift = f"Soon the {noun_a} {verb} the {noun_b}"
            
            sentences = [
                (sent_subj, noun_a, "subject", "early"),
                (sent_obj, noun_a, "object", "late"),
                (sent_subj_shift, noun_a, "subject", "shifted"),
            ]
            
            for sentence, noun, role, pos_type in sentences:
                noun_pos = find_noun_position(tokenizer, sentence, noun)
                if noun_pos is None:
                    continue
                
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    try:
                        outputs = model(inputs['input_ids'], output_hidden_states=True)
                        rep = outputs.hidden_states[mid_layer][0, noun_pos, :].detach().cpu().float().numpy()
                        
                        # 投影到V_sem
                        pc_projections = np.dot(rep, pc_axes.T)  # [10]
                        
                        all_data.append({
                            "noun": noun,
                            "verb": verb,
                            "role": role,
                            "position": pos_type,
                            "rep": rep,
                            "pc_proj": pc_projections,
                        })
                    except Exception as e:
                        pass
    
    if len(all_data) < 20:
        print("Too few data points!")
        release_model(model)
        return {}
    
    results = {"model": model_name, "exp": 3, "n_data": len(all_data)}
    
    # 1. 每个PC编码的是位置还是角色?
    print("\n--- PC-wise Role vs Position Decodability ---")
    
    for pc_idx in range(10):
        # 收集该PC上的值
        subj_early = [d["pc_proj"][pc_idx] for d in all_data if d["role"] == "subject" and d["position"] == "early"]
        subj_shifted = [d["pc_proj"][pc_idx] for d in all_data if d["role"] == "subject" and d["position"] == "shifted"]
        obj_late = [d["pc_proj"][pc_idx] for d in all_data if d["role"] == "object" and d["position"] == "late"]
        
        if len(subj_early) < 5 or len(obj_late) < 5:
            continue
        
        # 角色区分: subject vs object (都在baseline位置)
        subj_vals = np.array(subj_early)
        obj_vals = np.array(obj_late)
        
        # 简单分类: 用中位数分割
        all_role_vals = np.concatenate([subj_vals, obj_vals])
        role_labels = np.array([0]*len(subj_vals) + [1]*len(obj_vals))
        
        # Cohen's d for role
        d_role = (np.mean(subj_vals) - np.mean(obj_vals)) / (np.std(all_role_vals) + 1e-10)
        
        # 位置区分: subject_early vs subject_shifted
        if len(subj_shifted) >= 5:
            early_vals = np.array(subj_early)
            shifted_vals = np.array(subj_shifted)
            all_pos_vals = np.concatenate([early_vals, shifted_vals])
            d_pos = (np.mean(early_vals) - np.mean(shifted_vals)) / (np.std(all_pos_vals) + 1e-10)
        else:
            d_pos = 0
        
        results[f"PC{pc_idx}"] = {
            "mean_subj_early": float(np.mean(subj_vals)),
            "mean_obj_late": float(np.mean(obj_vals)),
            "d_role": float(d_role),
            "d_position": float(d_pos),
        }
        
        print(f"  PC{pc_idx}: d_role={d_role:.3f}, d_pos={d_pos:.3f}, "
              f"subj_early={np.mean(subj_vals):.2f}, obj_late={np.mean(obj_vals):.2f}")
    
    # 2. 5维V_sem中的角色/位置解码
    print("\n--- Role/Position Classification in V_sem ---")
    
    X_pc = np.array([d["pc_proj"][:5] for d in all_data])
    
    # 角色分类 (subject vs object, baseline only)
    baseline_data = [d for d in all_data if d["position"] in ["early", "late"]]
    if len(baseline_data) >= 20:
        X_role = np.array([d["pc_proj"][:5] for d in baseline_data])
        y_role = np.array([0 if d["role"] == "subject" else 1 for d in baseline_data])
        
        if len(np.unique(y_role)) == 2:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            role_cv = cross_val_score(clf, X_role, y_role, cv=min(5, len(y_role)//5), scoring='accuracy')
            print(f"  Role CV in V_sem(5d): {role_cv.mean():.3f} ± {role_cv.std():.3f}")
            results["role_cv_vsem"] = float(role_cv.mean())
    
    # 位置分类 (early vs shifted, subject only)
    pos_data = [d for d in all_data if d["role"] == "subject" and d["position"] in ["early", "shifted"]]
    if len(pos_data) >= 20:
        X_pos = np.array([d["pc_proj"][:5] for d in pos_data])
        y_pos = np.array([0 if d["position"] == "early" else 1 for d in pos_data])
        
        if len(np.unique(y_pos)) == 2:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            pos_cv = cross_val_score(clf, X_pos, y_pos, cv=min(5, len(y_pos)//5), scoring='accuracy')
            print(f"  Position CV in V_sem(5d): {pos_cv.mean():.3f} ± {pos_cv.std():.3f}")
            results["pos_cv_vsem"] = float(pos_cv.mean())
    
    # 3. δ_role和δ_pos在V_sem各PC上的投影
    print("\n--- Projection of δ_role and δ_pos onto V_sem PCs ---")
    
    # 计算每个名词的δ_role和δ_pos
    role_proj_per_pc = {i: [] for i in range(10)}
    pos_proj_per_pc = {i: [] for i in range(10)}
    
    for noun_idx, (noun_a, noun_b) in enumerate(NOUN_PAIRS):
        for noun in [noun_a, noun_b]:
            for verb in TRANSITIVE_VERBS[:3]:
                # 找该noun的所有条件
                subj_e = [d for d in all_data if d["noun"] == noun and d["verb"] == verb 
                         and d["role"] == "subject" and d["position"] == "early"]
                obj_l = [d for d in all_data if d["noun"] == noun and d["verb"] == verb 
                        and d["role"] == "object" and d["position"] == "late"]
                subj_s = [d for d in all_data if d["noun"] == noun and d["verb"] == verb 
                         and d["role"] == "subject" and d["position"] == "shifted"]
                
                if subj_e and obj_l:
                    d_role = subj_e[0]["rep"] - obj_l[0]["rep"]
                    for pc_idx in range(10):
                        proj = np.dot(d_role, pc_axes[pc_idx])
                        role_proj_per_pc[pc_idx].append(proj)
                
                if subj_e and subj_s:
                    d_pos = subj_e[0]["rep"] - subj_s[0]["rep"]
                    for pc_idx in range(10):
                        proj = np.dot(d_pos, pc_axes[pc_idx])
                        pos_proj_per_pc[pc_idx].append(proj)
    
    print("  PC  | δ_role proj | δ_pos proj | Role/Pos ratio")
    print("  ----|-------------|------------|---------------")
    for pc_idx in range(10):
        if role_proj_per_pc[pc_idx] and pos_proj_per_pc[pc_idx]:
            mean_role_proj = np.mean(np.abs(role_proj_per_pc[pc_idx]))
            mean_pos_proj = np.mean(np.abs(pos_proj_per_pc[pc_idx]))
            ratio = mean_role_proj / (mean_pos_proj + 1e-10)
            results[f"PC{pc_idx}_role_proj"] = float(mean_role_proj)
            results[f"PC{pc_idx}_pos_proj"] = float(mean_pos_proj)
            print(f"  PC{pc_idx} | {mean_role_proj:11.2f} | {mean_pos_proj:10.2f} | {ratio:13.2f}")
    
    # 保存结果
    out_path = TEMP / f"ccxxx_exp3_{model_name}_results.json"
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
