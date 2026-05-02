"""
CCXII(362): 概念本质判别器 — Vector vs Subspace vs Trajectory
==============================================================

★★★★★ 核心问题:
  概念到底是 A.向量 B.子空间 C.动态轨道?

★★★★★ 实验设计:
  Experiment 1: Vector Steering Test
    - 找到概念方向 v = mean(h(concept)) - mean(h(control))
    - 在不同层注入 h' = h + α*v
    - 如果同一方向在所有层都有效 → 支持向量模型
    
  Experiment 2: Subspace vs Vector Test
    - 收集概念的多上下文表示, PCA得到子空间
    - 单方向 vs 多方向子空间注入
    - 如果多维子空间远强于单方向 → 支持子空间模型
    
  Experiment 3: Trajectory Replay Test (最关键!)
    - 记录概念token的完整层轨迹 (h1,h2,...,hL)
    - 在另一个token上重放轨迹
    - 有序重放 vs 打乱层序 → 如果打乱失效 → 支持轨道模型
    
  Experiment 4: Layer Permutation Test (终极判别!)
    - apple轨迹: L5→L6→L7→L8
    - 打乱: L7→L5→L8→L6
    - 如果打乱仍有效 → 静态结构
    - 如果打乱崩溃 → 动态轨道

★★★★★ 判别标准:
  向量: 任意层加同一方向都有效
  子空间: 需多维组合才有效, 单方向不稳定
  轨道: 必须按层序列注入, 打乱层顺序失效

用法:
  python ccxii_concept_nature.py --model qwen3
  python ccxii_concept_nature.py --model qwen3 --exp 1
  python ccxii_concept_nature.py --model qwen3 --exp 3
"""

import argparse, os, sys, json, gc, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model,
    MODEL_CONFIGS
)

TEMP = Path("tests/glm5_temp")


# ==========================================
# 概念定义
# ==========================================

# 核心概念 + 多上下文模板
CONCEPTS = {
    "apple": {
        "templates": [
            "The word is apple",
            "I ate an apple", 
            "A red apple",
            "The apple fell",
            "Apple is a fruit",
        ],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
        "control_words": ["the", "a", "is", "was", "and"],
    },
    "dog": {
        "templates": [
            "The word is dog",
            "A big dog",
            "The dog barked",
            "My pet dog",
            "Dog is an animal",
        ],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
        "control_words": ["the", "a", "is", "was", "and"],
    },
    "king": {
        "templates": [
            "The word is king",
            "The king ruled",
            "A wise king",
            "The king and queen",
            "King is a ruler",
        ],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
        "control_words": ["the", "a", "is", "was", "and"],
    },
    "doctor": {
        "templates": [
            "The word is doctor",
            "The doctor helped",
            "A good doctor",
            "Visit the doctor",
            "Doctor treats patients",
        ],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
        "control_words": ["the", "a", "is", "was", "and"],
    },
    "red": {
        "templates": [
            "The color is red",
            "A red car",
            "The red apple",
            "Red is a color",
            "She wore red",
        ],
        "probe_words": ["color", "bright", "blood", "fire", "blue", "green", "yellow", "orange"],
        "control_words": ["the", "a", "is", "was", "and"],
    },
}


def get_last_token_hidden(model, tokenizer, device, text, layers_to_capture=None):
    """获取指定层最后一个token的隐藏状态 (使用output_hidden_states)"""
    model_info_temp = get_model_info(model, "qwen3")
    if layers_to_capture is None:
        layers_to_capture = list(range(model_info_temp.n_layers))
    
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        except:
            outputs = model(input_ids=input_ids)
    
    result = {}
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        for li in layers_to_capture:
            if li + 1 < len(outputs.hidden_states):
                result[li] = outputs.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy()
    
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    
    gc.collect()
    
    return result, logits


def get_residual_stream_states(model, tokenizer, device, text, sample_layers):
    """获取残差流在各层的状态 (使用output_hidden_states)"""
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        except:
            outputs = model(input_ids=input_ids)
    
    result = {}
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        for li in sample_layers:
            if li + 1 < len(outputs.hidden_states):  # hidden_states[0]=embedding, [i+1]=after layer i
                result[li] = outputs.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy()
    
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    
    gc.collect()
    
    return result, logits


def steering_forward(model, tokenizer, device, text, steering_vector, layer, alpha):
    """
    在指定层注入steering vector到残差流
    steering_vector: [d_model] numpy array
    返回: logits
    """
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    
    sv_tensor = torch.tensor(steering_vector, dtype=torch.float32, device=device)
    
    def make_steering_hook(target_layer, sv, a):
        """在target层输出后注入steering vector"""
        def hook(module, inp, output):
            # output是Transformer层的输出 (可能是tuple)
            if isinstance(output, tuple):
                h = output[0]
                h_mod = h.clone()
                h_mod[:, -1, :] += a * sv.to(h.device)
                return (h_mod,) + output[1:]
            else:
                h = output
                h_mod = h.clone()
                h_mod[:, -1, :] += a * sv.to(h.device)
                return h_mod
        return hook
    
    hooks = []
    all_layers = get_layers(model)
    
    # Hook at the target layer's output
    hooks.append(all_layers[layer].register_forward_hook(
        make_steering_hook(layer, sv_tensor, alpha)
    ))
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Steering forward failed: {e}")
            for h in hooks:
                h.remove()
            return None
    
    for h in hooks:
        h.remove()
    
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    
    gc.collect()
    
    return logits


def trajectory_replay_forward(model, tokenizer, device, base_text, trajectory, replay_layers, permute_order=None):
    """
    在base_text上重放trajectory
    
    trajectory: {layer: state_vector} — 每层最后一个token的残差流状态
    replay_layers: 要重放的层列表
    permute_order: 如果提供, 按照这个顺序映射层
      e.g. {5:7, 6:5, 7:8, 8:6} 表示在L5放L7的状态, L6放L5的状态...
    
    返回: logits
    """
    input_ids = tokenizer.encode(base_text, add_special_tokens=True, return_tensors="pt").to(device)
    
    hooks = []
    all_layers = get_layers(model)
    
    def make_replay_hook(target_layer, source_state):
        """在target_layer输出后替换残差流状态"""
        source_tensor = torch.tensor(source_state, dtype=torch.float32, device=device)
        
        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] = source_tensor.to(h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] = source_tensor.to(h.device)
                return h
        return hook
    
    for li in replay_layers:
        if permute_order is not None and li in permute_order:
            source_li = permute_order[li]
        else:
            source_li = li
        
        if source_li in trajectory:
            # Hook at layer output
            hooks.append(all_layers[li].register_forward_hook(
                make_replay_hook(li, trajectory[source_li])
            ))
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Trajectory replay forward failed: {e}")
            for h in hooks:
                h.remove()
            return None
    
    for h in hooks:
        h.remove()
    
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    
    return logits


def compute_concept_score(logits, tokenizer, probe_words, concept_name, vocab_size=None):
    """计算概念激活分数: probe_words的平均概率 vs 基线"""
    if vocab_size is None:
        vocab_size = len(logits)
    
    # 获取probe_words的token ids
    probe_ids = []
    for word in probe_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            probe_ids.append(ids[0])
        # 也尝试不带空格
        ids2 = tokenizer.encode(word, add_special_tokens=False)
        if len(ids2) == 1 and ids2[0] not in probe_ids:
            probe_ids.append(ids2[0])
    
    probe_ids = list(set(probe_ids))
    if len(probe_ids) == 0:
        return 0.0, {}
    
    # log_softmax
    log_probs = logits - np.max(logits)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
    
    probe_log_probs = [log_probs[tid] for tid in probe_ids if tid < len(logits)]
    
    if len(probe_log_probs) == 0:
        return 0.0, {}
    
    score = float(np.mean(probe_log_probs))
    
    # 也记录每个probe word的概率
    word_scores = {}
    for word in probe_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1 and ids[0] < len(logits):
            word_scores[word] = float(np.exp(log_probs[ids[0]]))
    
    return score, word_scores


# ==========================================
# Experiment 1: Vector Steering Test
# ==========================================

def experiment1_vector_steering(model, tokenizer, device, model_name, model_info):
    """向量假说检验: 同一方向在不同层的因果效果"""
    print("\n" + "="*70)
    print("Experiment 1: Vector Steering Test")
    print("="*70)
    
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    sample_layers = sorted(set([0, 1, 3, 6, 9, 12, 18, 24, 30] + [n_layers-1]))
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    results = []
    
    for concept_name, concept_data in CONCEPTS.items():
        print(f"\n  --- Concept: {concept_name} ---")
        
        # 收集概念表示 (多上下文)
        concept_states = {}
        for template in concept_data["templates"]:
            states, _ = get_residual_stream_states(
                model, tokenizer, device, template, sample_layers
            )
            for li, state in states.items():
                if li not in concept_states:
                    concept_states[li] = []
                concept_states[li].append(state)
        
        # 收集控制表示
        control_states = {}
        for word in concept_data["control_words"]:
            states, _ = get_residual_stream_states(
                model, tokenizer, device, word, sample_layers
            )
            for li, state in states.items():
                if li not in control_states:
                    control_states[li] = []
                control_states[li].append(state)
        
        # 计算概念方向: v = mean(concept) - mean(control)
        concept_directions = {}
        for li in sample_layers:
            if li in concept_states and li in control_states:
                concept_mean = np.mean(concept_states[li], axis=0)
                control_mean = np.mean(control_states[li], axis=0)
                concept_directions[li] = concept_mean - control_mean
        
        # 基线测试: 无干预时probe words的概率
        baseline_text = "The word is"  # 中性上下文
        _, baseline_logits = get_residual_stream_states(
            model, tokenizer, device, baseline_text, sample_layers
        )
        baseline_score, baseline_word_scores = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"], concept_name
        )
        
        # 在不同层注入概念方向
        alphas = [0.5, 1.0, 2.0, 5.0]
        
        for alpha in alphas:
            for inject_layer in sample_layers:
                if inject_layer not in concept_directions:
                    continue
                
                # 使用"最典型"层的概念方向 (中层)
                source_layer = min(sample_layers, key=lambda l: abs(l - n_layers//2))
                if source_layer not in concept_directions:
                    source_layer = list(concept_directions.keys())[0]
                
                steering_vec = concept_directions[source_layer]
                
                # 注入
                steered_logits = steering_forward(
                    model, tokenizer, device, baseline_text,
                    steering_vec, inject_layer, alpha
                )
                
                if steered_logits is None:
                    continue
                
                steered_score, steered_word_scores = compute_concept_score(
                    steered_logits, tokenizer, concept_data["probe_words"], concept_name
                )
                
                delta = steered_score - baseline_score
                
                result = {
                    "concept": concept_name,
                    "alpha": alpha,
                    "inject_layer": inject_layer,
                    "source_layer": source_layer,
                    "baseline_score": baseline_score,
                    "steered_score": steered_score,
                    "delta": delta,
                    "word_scores": {k: steered_word_scores.get(k, 0) - baseline_word_scores.get(k, 0) 
                                   for k in concept_data["probe_words"][:4]},
                }
                results.append(result)
                
                if alpha == 1.0:
                    print(f"    L{inject_layer:2d}: baseline={baseline_score:.3f}, "
                          f"steered={steered_score:.3f}, delta={delta:+.3f}")
        
        # 清理
        del concept_states, control_states, concept_directions
        gc.collect()
    
    return results


# ==========================================
# Experiment 2: Subspace vs Vector Test
# ==========================================

def experiment2_subspace_vs_vector(model, tokenizer, device, model_name, model_info):
    """子空间假说检验: 单方向 vs 多维子空间"""
    print("\n" + "="*70)
    print("Experiment 2: Subspace vs Vector Test")
    print("="*70)
    
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    target_layer = n_layers // 2  # 在中层注入
    
    results = []
    
    for concept_name, concept_data in CONCEPTS.items():
        print(f"\n  --- Concept: {concept_name} ---")
        
        # 收集多上下文概念表示
        concept_states_list = []
        for template in concept_data["templates"]:
            states, _ = get_residual_stream_states(
                model, tokenizer, device, template, [target_layer]
            )
            if target_layer in states:
                concept_states_list.append(states[target_layer])
        
        if len(concept_states_list) < 3:
            print(f"    Not enough states, skipping")
            continue
        
        concept_matrix = np.array(concept_states_list)  # [n_templates, d_model]
        concept_centered = concept_matrix - concept_matrix.mean(axis=0)
        
        # PCA
        U_pca, S_pca, Vt_pca = svd(concept_centered, full_matrices=False)
        # Vt_pca[:k] 是前k个主方向
        
        # 基线
        baseline_text = "The word is"
        _, baseline_logits = get_residual_stream_states(
            model, tokenizer, device, baseline_text, [target_layer]
        )
        baseline_score, _ = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"], concept_name
        )
        
        # 测试: 1维 vs 3维 vs 5维子空间注入
        n_dirs_list = [1, 2, 3, 5]
        alpha = 2.0
        
        for n_dirs in n_dirs_list:
            # 构造子空间扰动: 沿前n_dirs个主方向的组合
            # 使用同一alpha但不同维度
            subspace_vec = np.zeros(d_model, dtype=np.float32)
            for k in range(min(n_dirs, len(Vt_pca))):
                subspace_vec += alpha * S_pca[k] / S_pca[0] * Vt_pca[k]
            
            # 1. 单方向 (只第1个PC)
            if n_dirs == 1:
                single_vec = alpha * Vt_pca[0]
                single_logits = steering_forward(
                    model, tokenizer, device, baseline_text,
                    single_vec, target_layer, 1.0
                )
                if single_logits is not None:
                    single_score, _ = compute_concept_score(
                        single_logits, tokenizer, concept_data["probe_words"], concept_name
                    )
                else:
                    single_score = baseline_score
            else:
                single_score = None
            
            # 2. 多维子空间
            multi_logits = steering_forward(
                model, tokenizer, device, baseline_text,
                subspace_vec, target_layer, 1.0
            )
            
            if multi_logits is not None:
                multi_score, _ = compute_concept_score(
                    multi_logits, tokenizer, concept_data["probe_words"], concept_name
                )
            else:
                multi_score = baseline_score
            
            result = {
                "concept": concept_name,
                "n_dirs": n_dirs,
                "alpha": alpha,
                "target_layer": target_layer,
                "baseline_score": baseline_score,
                "subspace_score": multi_score,
                "subspace_delta": multi_score - baseline_score,
                "single_score": single_score,
                "single_delta": (single_score - baseline_score) if single_score else None,
            }
            results.append(result)
            
            if single_score is not None:
                print(f"    {n_dirs}dir: baseline={baseline_score:.3f}, "
                      f"single={single_score:.3f}(Δ={single_score-baseline_score:+.3f}), "
                      f"subspace={multi_score:.3f}(Δ={multi_score-baseline_score:+.3f})")
            else:
                print(f"    {n_dirs}dir: baseline={baseline_score:.3f}, "
                      f"subspace={multi_score:.3f}(Δ={multi_score-baseline_score:+.3f})")
        
        del concept_states_list, concept_matrix
        gc.collect()
    
    return results


# ==========================================
# Experiment 3: Trajectory Replay Test
# ==========================================

def experiment3_trajectory_replay(model, tokenizer, device, model_name, model_info):
    """轨道假说检验: 完整轨迹重放 vs 单层注入"""
    print("\n" + "="*70)
    print("Experiment 3: Trajectory Replay Test")
    print("="*70)
    
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # 选择4个连续层用于重放
    replay_start = max(1, n_layers // 4)
    replay_layers = list(range(replay_start, replay_start + 4))
    replay_layers = [l for l in replay_layers if l < n_layers]
    
    print(f"  Replay layers: {replay_layers}")
    
    results = []
    
    for concept_name, concept_data in CONCEPTS.items():
        print(f"\n  --- Concept: {concept_name} ---")
        
        # 收集概念轨迹
        # 使用最典型的模板
        template = concept_data["templates"][0]
        trajectory, concept_logits = get_residual_stream_states(
            model, tokenizer, device, template, replay_layers
        )
        
        if len(trajectory) < len(replay_layers):
            print(f"    Incomplete trajectory, skipping")
            continue
        
        # 基线: 在neutral context上
        base_text = "The word is"  # 中性上下文
        _, baseline_logits = get_residual_stream_states(
            model, tokenizer, device, base_text, replay_layers
        )
        baseline_score, _ = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"], concept_name
        )
        
        # 也获取概念本身的分数
        concept_score, _ = compute_concept_score(
            concept_logits, tokenizer, concept_data["probe_words"], concept_name
        )
        
        # Test 1: 完整有序轨迹重放
        ordered_logits = trajectory_replay_forward(
            model, tokenizer, device, base_text, trajectory, replay_layers
        )
        if ordered_logits is not None:
            ordered_score, _ = compute_concept_score(
                ordered_logits, tokenizer, concept_data["probe_words"], concept_name
            )
        else:
            ordered_score = baseline_score
        
        # Test 2: 打乱层序重放 (关键判别!)
        if len(replay_layers) >= 3:
            # 反转层序: L8→L7→L6→L5
            reversed_order = {replay_layers[i]: replay_layers[-(i+1)] 
                            for i in range(len(replay_layers))}
            reversed_logits = trajectory_replay_forward(
                model, tokenizer, device, base_text, trajectory, 
                replay_layers, permute_order=reversed_order
            )
            if reversed_logits is not None:
                reversed_score, _ = compute_concept_score(
                    reversed_logits, tokenizer, concept_data["probe_words"], concept_name
                )
            else:
                reversed_score = baseline_score
        else:
            reversed_score = None
        
        # Test 3: 随机打乱层序
        if len(replay_layers) >= 3:
            rng = np.random.RandomState(42)
            shuffled = list(replay_layers)
            rng.shuffle(shuffled)
            # 确保不是原序也不是反转
            attempts = 0
            while (shuffled == replay_layers or 
                   (shuffled == list(reversed(replay_layers)) and attempts < 10)):
                rng.shuffle(shuffled)
                attempts += 1
            
            random_order = {replay_layers[i]: shuffled[i] 
                          for i in range(len(replay_layers))}
            random_logits = trajectory_replay_forward(
                model, tokenizer, device, base_text, trajectory,
                replay_layers, permute_order=random_order
            )
            if random_logits is not None:
                random_score, _ = compute_concept_score(
                    random_logits, tokenizer, concept_data["probe_words"], concept_name
                )
            else:
                random_score = baseline_score
        else:
            random_score = None
        
        # Test 4: 只在单层注入 (对比)
        for li in replay_layers[::2]:  # 只测试部分层
            single_replay = {li: trajectory[li]}
            single_logits = trajectory_replay_forward(
                model, tokenizer, device, base_text, single_replay, [li]
            )
            if single_logits is not None:
                single_score, _ = compute_concept_score(
                    single_logits, tokenizer, concept_data["probe_words"], concept_name
                )
            else:
                single_score = baseline_score
            
            result = {
                "concept": concept_name,
                "test_type": "single_layer",
                "layer": li,
                "baseline_score": baseline_score,
                "test_score": single_score,
                "delta": single_score - baseline_score,
            }
            results.append(result)
            
            print(f"    Single L{li}: baseline={baseline_score:.3f}, "
                  f"score={single_score:.3f}(Δ={single_score-baseline_score:+.3f})")
        
        # 完整轨迹测试结果
        result = {
            "concept": concept_name,
            "replay_layers": replay_layers,
            "baseline_score": baseline_score,
            "concept_score": concept_score,
            "ordered_score": ordered_score,
            "ordered_delta": ordered_score - baseline_score,
            "reversed_score": reversed_score,
            "reversed_delta": (reversed_score - baseline_score) if reversed_score is not None else None,
            "random_score": random_score,
            "random_delta": (random_score - baseline_score) if random_score is not None else None,
        }
        results.append(result)
        
        rev_str = f"reversed={reversed_score:.3f}(Δ={reversed_score-baseline_score:+.3f})" if reversed_score else "reversed=N/A"
        rnd_str = f"random={random_score:.3f}(Δ={random_score-baseline_score:+.3f})" if random_score else "random=N/A"
        print(f"    Ordered: {ordered_score:.3f}(Δ={ordered_score-baseline_score:+.3f}), "
              f"{rev_str}, {rnd_str}")
        print(f"    Concept itself: {concept_score:.3f}")
        
        del trajectory
        gc.collect()
    
    return results


# ==========================================
# Experiment 4: Layer Permutation Test (终极判别)
# ==========================================

def experiment4_layer_permutation(model, tokenizer, device, model_name, model_info):
    """终极判别: 层排列测试"""
    print("\n" + "="*70)
    print("Experiment 4: Layer Permutation Test (终极判别)")
    print("="*70)
    
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # 使用6个层: 浅→中→深
    perm_layers = sorted(set([
        max(0, n_layers//6), 
        max(1, n_layers//4),
        n_layers//3, 
        n_layers//2,
        2*n_layers//3,
        min(n_layers-1, 5*n_layers//6)
    ]))
    # 去掉超过n_layers的
    perm_layers = [l for l in perm_layers if l < n_layers]
    
    if len(perm_layers) < 4:
        print("  Not enough layers for permutation test")
        return []
    
    print(f"  Permutation layers: {perm_layers}")
    
    results = []
    
    # 只测试3个核心概念
    test_concepts = {"apple": CONCEPTS["apple"], "king": CONCEPTS["king"], "red": CONCEPTS["red"]}
    
    for concept_name, concept_data in test_concepts.items():
        print(f"\n  --- Concept: {concept_name} ---")
        
        # 收集完整轨迹
        template = concept_data["templates"][0]
        trajectory, concept_logits = get_residual_stream_states(
            model, tokenizer, device, template, perm_layers
        )
        
        if len(trajectory) < len(perm_layers):
            print(f"    Incomplete trajectory, skipping")
            continue
        
        # 基线
        base_text = "The word is"
        _, baseline_logits = get_residual_stream_states(
            model, tokenizer, device, base_text, perm_layers
        )
        baseline_score, _ = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"], concept_name
        )
        
        # Test 1: 正常有序
        ordered_logits = trajectory_replay_forward(
            model, tokenizer, device, base_text, trajectory, perm_layers
        )
        ordered_score = baseline_score
        if ordered_logits is not None:
            ordered_score, _ = compute_concept_score(
                ordered_logits, tokenizer, concept_data["probe_words"], concept_name
            )
        
        # Test 2: 反转层序
        reversed_order = {perm_layers[i]: perm_layers[-(i+1)] for i in range(len(perm_layers))}
        rev_logits = trajectory_replay_forward(
            model, tokenizer, device, base_text, trajectory, perm_layers, permute_order=reversed_order
        )
        rev_score = baseline_score
        if rev_logits is not None:
            rev_score, _ = compute_concept_score(
                rev_logits, tokenizer, concept_data["probe_words"], concept_name
            )
        
        # Test 3: 多种随机排列
        rng = np.random.RandomState(42)
        random_scores = []
        for trial in range(3):
            shuffled = list(perm_layers)
            # 确保不是原序
            while shuffled == perm_layers:
                rng.shuffle(shuffled)
            random_order = {perm_layers[i]: shuffled[i] for i in range(len(perm_layers))}
            
            rnd_logits = trajectory_replay_forward(
                model, tokenizer, device, base_text, trajectory, perm_layers, permute_order=random_order
            )
            if rnd_logits is not None:
                rnd_score, _ = compute_concept_score(
                    rnd_logits, tokenizer, concept_data["probe_words"], concept_name
                )
                random_scores.append(rnd_score)
        
        avg_random = np.mean(random_scores) if random_scores else baseline_score
        
        # Test 4: 只重放中间2层
        mid_layers = perm_layers[len(perm_layers)//2-1:len(perm_layers)//2+1]
        partial_logits = trajectory_replay_forward(
            model, tokenizer, device, base_text, trajectory, mid_layers
        )
        partial_score = baseline_score
        if partial_logits is not None:
            partial_score, _ = compute_concept_score(
                partial_logits, tokenizer, concept_data["probe_words"], concept_name
            )
        
        # Test 5: 每层独立注入 (每层单独, 不连续)
        single_scores = {}
        for li in perm_layers:
            single_replay = {li: trajectory[li]}
            single_logits = trajectory_replay_forward(
                model, tokenizer, device, base_text, single_replay, [li]
            )
            if single_logits is not None:
                s_score, _ = compute_concept_score(
                    single_logits, tokenizer, concept_data["probe_words"], concept_name
                )
                single_scores[li] = s_score
        
        result = {
            "concept": concept_name,
            "perm_layers": perm_layers,
            "baseline_score": baseline_score,
            "ordered_score": ordered_score,
            "ordered_delta": ordered_score - baseline_score,
            "reversed_score": rev_score,
            "reversed_delta": rev_score - baseline_score,
            "random_scores": random_scores,
            "avg_random_score": avg_random,
            "avg_random_delta": avg_random - baseline_score,
            "partial_score": partial_score,
            "partial_delta": partial_score - baseline_score,
            "single_scores": single_scores,
            "max_single_delta": max([s - baseline_score for s in single_scores.values()]) if single_scores else 0,
        }
        results.append(result)
        
        # 判别
        ordered_d = ordered_score - baseline_score
        reversed_d = rev_score - baseline_score
        random_d = avg_random - baseline_score
        partial_d = partial_score - baseline_score
        max_single_d = max([s - baseline_score for s in single_scores.values()]) if single_scores else 0
        
        # ★★★ 判别逻辑 ★★★
        if ordered_d > 0 and reversed_d < ordered_d * 0.5:
            verdict = "TRAJECTORY! (ordered >> reversed)"
        elif ordered_d > 0 and random_d < ordered_d * 0.5:
            verdict = "TRAJECTORY! (ordered >> random)"
        elif ordered_d > 0 and partial_d < ordered_d * 0.3:
            verdict = "TRAJECTORY! (full >> partial)"
        elif max_single_d > 0 and max_single_d >= ordered_d * 0.7:
            verdict = "VECTOR? (single layer sufficient)"
        else:
            verdict = "MIXED (need more analysis)"
        
        print(f"    Ordered: Δ={ordered_d:+.3f}")
        print(f"    Reversed: Δ={reversed_d:+.3f}")
        print(f"    Random avg: Δ={random_d:+.3f}")
        print(f"    Partial (2 layers): Δ={partial_d:+.3f}")
        print(f"    Max single layer: Δ={max_single_d:+.3f}")
        print(f"    ★★★ VERDICT: {verdict}")
        
        del trajectory
        gc.collect()
    
    return results


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3"])
    parser.add_argument("--exp", type=int, default=0, 
                       choices=[0, 1, 2, 3, 4],
                       help="0=all, 1=vector, 2=subspace, 3=trajectory, 4=permutation")
    args = parser.parse_args()
    
    model_name = args.model
    exp = args.exp
    
    print(f"\n{'#'*70}")
    print(f"CCXII: Concept Nature Discriminator — {model_name}")
    print(f"Vector vs Subspace vs Trajectory")
    print(f"{'#'*70}")
    
    # 加载模型 (需要output_hidden_states)
    model, tokenizer, device = load_model(model_name)
    
    # 确保模型输出hidden states
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True
    
    model_info = get_model_info(model, model_name)
    print(f"  d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    all_results = {
        "model": model_name, 
        "d_model": model_info.d_model, 
        "n_layers": model_info.n_layers
    }
    
    # 运行实验
    if exp in [0, 1]:
        exp1_results = experiment1_vector_steering(model, tokenizer, device, model_name, model_info)
        all_results["exp1_vector_steering"] = exp1_results
    
    if exp in [0, 2]:
        exp2_results = experiment2_subspace_vs_vector(model, tokenizer, device, model_name, model_info)
        all_results["exp2_subspace_vs_vector"] = exp2_results
    
    if exp in [0, 3]:
        exp3_results = experiment3_trajectory_replay(model, tokenizer, device, model_name, model_info)
        all_results["exp3_trajectory_replay"] = exp3_results
    
    if exp in [0, 4]:
        exp4_results = experiment4_layer_permutation(model, tokenizer, device, model_name, model_info)
        all_results["exp4_layer_permutation"] = exp4_results
    
    # 保存
    output_path = TEMP / f"ccxii_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")
    
    # 最终判别
    print(f"\n{'#'*70}")
    print(f"CCXII Final Verdict: {model_name}")
    print(f"{'#'*70}")
    
    if "exp1_vector_steering" in all_results:
        exp1 = all_results["exp1_vector_steering"]
        # 统计: 多少层注入有效?
        effective_layers = sum(1 for r in exp1 if r.get("delta", 0) > 0.5 and r.get("alpha") == 1.0)
        total_layers_tested = sum(1 for r in exp1 if r.get("alpha") == 1.0)
        print(f"\n  Exp1 (Vector): {effective_layers}/{total_layers_tested} layers effective")
        if effective_layers > total_layers_tested * 0.5:
            print(f"    → Supports VECTOR model (works at many layers)")
        else:
            print(f"    → Weak VECTOR support (only specific layers)")
    
    if "exp2_subspace_vs_vector" in all_results:
        exp2 = all_results["exp2_subspace_vs_vector"]
        n_dirs_results = {}
        for r in exp2:
            nd = r.get("n_dirs", 0)
            if nd not in n_dirs_results:
                n_dirs_results[nd] = []
            n_dirs_results[nd].append(r.get("subspace_delta", 0))
        
        print(f"\n  Exp2 (Subspace):")
        for nd in sorted(n_dirs_results.keys()):
            avg_delta = np.mean(n_dirs_results[nd])
            print(f"    {nd} directions: avg_delta={avg_delta:+.3f}")
    
    if "exp3_trajectory_replay" in all_results:
        exp3 = all_results["exp3_trajectory_replay"]
        print(f"\n  Exp3 (Trajectory):")
        for r in exp3:
            if "ordered_delta" in r:
                od = r.get("ordered_delta", 0)
                rd = r.get("reversed_delta", None)
                rnd = r.get("random_delta", None)
                concept = r.get("concept", "?")
                print(f"    {concept}: ordered Δ={od:+.3f}, "
                      f"reversed Δ={rd:+.3f}" if rd is not None else f"    {concept}: ordered Δ={od:+.3f}")
    
    if "exp4_layer_permutation" in all_results:
        exp4 = all_results["exp4_layer_permutation"]
        print(f"\n  Exp4 (Permutation — ULTIMATE):")
        for r in exp4:
            concept = r.get("concept", "?")
            od = r.get("ordered_delta", 0)
            rd = r.get("reversed_delta", 0)
            avg_rd = r.get("avg_random_delta", 0)
            pd = r.get("partial_delta", 0)
            msd = r.get("max_single_delta", 0)
            
            print(f"    {concept}:")
            print(f"      Ordered:   Δ={od:+.3f}")
            print(f"      Reversed:  Δ={rd:+.3f} ({'OK' if abs(rd) > abs(od)*0.5 else 'DEGRADED'})")
            print(f"      Random:    Δ={avg_rd:+.3f} ({'OK' if abs(avg_rd) > abs(od)*0.5 else 'DEGRADED'})")
            print(f"      Partial:   Δ={pd:+.3f}")
            print(f"      Single:    Δ={msd:+.3f}")
            
            # 判别
            if od > 0:
                if rd < od * 0.3 or avg_rd < od * 0.3:
                    print(f"      ★★★ TRAJECTORY! (permutation degrades effect)")
                elif msd >= od * 0.7:
                    print(f"      ★★★ VECTOR! (single layer sufficient)")
                elif pd >= od * 0.7:
                    print(f"      ★★★ SUBSPACE! (partial path works)")
                else:
                    print(f"      ★★★ MIXED (needs deeper analysis)")
            else:
                print(f"      ★ No significant steering effect")
    
    release_model(model)
    print(f"\nCCXII {model_name} 完成!")


if __name__ == "__main__":
    main()
