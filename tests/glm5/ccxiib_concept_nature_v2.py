"""
CCXII-B: 概念本质判别器 — 改进版 (加性残差注入)
==============================================

关键改进:
  - 使用加性干预 h' = h + α*(concept_trajectory - baseline) 而非替换
  - 这样不会破坏模型的正常计算流程
  - 可以测试不同层的加性注入效果

判别标准:
  A 向量: 任意层加同一方向都有效, 效果相似
  B 子空间: 需多维组合才有效
  C 轨道: 必须按层序列注入, 单层弱, 多层强, 层序重要
"""

import argparse, os, sys, json, gc, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model,
    MODEL_CONFIGS
)

TEMP = Path("tests/glm5_temp")

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
    },
}


def get_hidden_states(model, tokenizer, device, text, sample_layers):
    """获取残差流状态 (使用output_hidden_states)"""
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        except:
            outputs = model(input_ids=input_ids)
    
    result = {}
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        for li in sample_layers:
            if li + 1 < len(outputs.hidden_states):
                result[li] = outputs.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy()
    
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return result, logits


def compute_concept_score(logits, tokenizer, probe_words):
    """计算概念激活分数"""
    log_probs = logits - np.max(logits)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
    
    probe_log_probs = []
    word_scores = {}
    for word in probe_words:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            if len(ids) == 1 and ids[0] < len(logits):
                probe_log_probs.append(log_probs[ids[0]])
                word_scores[word] = float(np.exp(log_probs[ids[0]]))
                break
    
    score = float(np.mean(probe_log_probs)) if probe_log_probs else -20.0
    return score, word_scores


def additive_steering_forward(model, tokenizer, device, text, deltas, alpha=1.0):
    """
    加性残差注入: 在指定层加入 delta
    
    deltas: {layer: vector} — 要添加的残差增量
    alpha: 缩放因子
    """
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    
    def make_hook(layer_idx, delta_vec, a):
        delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += a * delta_tensor.to(h.device)
                return h
        return hook
    
    hooks = []
    all_layers = get_layers(model)
    for li, delta in deltas.items():
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li, delta, alpha)))
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Additive steering failed: {e}")
            for h in hooks: h.remove()
            return None
    
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3"])
    args = parser.parse_args()
    model_name = args.model
    
    print(f"\n{'#'*70}")
    print(f"CCXII-B: Concept Nature Discriminator (Additive) — {model_name}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True
    
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")
    
    # 采样层
    sample_layers = sorted(set([0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]))
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    results = {}
    
    for concept_name, concept_data in CONCEPTS.items():
        print(f"\n{'='*60}")
        print(f"  Concept: {concept_name}")
        print(f"{'='*60}")
        
        # 1. 收集概念轨迹 (多上下文平均)
        concept_trajectories = {}  # {layer: [state1, state2, ...]}
        for template in concept_data["templates"]:
            states, _ = get_hidden_states(model, tokenizer, device, template, sample_layers)
            for li, state in states.items():
                if li not in concept_trajectories:
                    concept_trajectories[li] = []
                concept_trajectories[li].append(state)
        
        # 平均轨迹
        concept_mean = {li: np.mean(states, axis=0) for li, states in concept_trajectories.items()}
        
        # 2. 收集基线轨迹
        baseline_text = "The word is"
        baseline_states, baseline_logits = get_hidden_states(
            model, tokenizer, device, baseline_text, sample_layers
        )
        
        # 3. 概念方向: 逐层差异
        concept_deltas = {}
        for li in sample_layers:
            if li in concept_mean and li in baseline_states:
                concept_deltas[li] = concept_mean[li] - baseline_states[li]
        
        # 4. 全局概念方向 (中层差异)
        mid_layer = n_layers // 2
        if mid_layer not in concept_deltas:
            mid_layer = list(concept_deltas.keys())[len(concept_deltas)//2] if concept_deltas else None
        global_direction = concept_deltas.get(mid_layer, None)
        
        # 基线分数
        baseline_score, baseline_words = compute_concept_score(
            baseline_logits, tokenizer, concept_data["probe_words"]
        )
        print(f"  Baseline score: {baseline_score:.3f}")
        
        # ==========================================
        # Test A: 向量假说 — 同一方向在所有层的效果
        # ==========================================
        print(f"\n  --- Test A: Same Direction at Different Layers ---")
        
        alpha = 1.0
        vector_results = []
        
        if global_direction is not None:
            for inject_layer in sample_layers:
                deltas = {inject_layer: global_direction}
                logits = additive_steering_forward(
                    model, tokenizer, device, baseline_text, deltas, alpha
                )
                if logits is not None:
                    score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                    delta = score - baseline_score
                    vector_results.append({
                        "inject_layer": inject_layer,
                        "source": "global",
                        "score": score,
                        "delta": delta,
                    })
                    print(f"    L{inject_layer:2d} +global_dir: score={score:.3f}, Δ={delta:+.3f}")
        
        # ==========================================
        # Test B: 逐层差异方向 — 每层用自己的delta
        # ==========================================
        print(f"\n  --- Test B: Per-Layer Delta (Trajectory Hypothesis) ---")
        
        perlayer_results = []
        for inject_layer in sample_layers:
            if inject_layer not in concept_deltas:
                continue
            deltas = {inject_layer: concept_deltas[inject_layer]}
            logits = additive_steering_forward(
                model, tokenizer, device, baseline_text, deltas, alpha
            )
            if logits is not None:
                score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                delta = score - baseline_score
                perlayer_results.append({
                    "inject_layer": inject_layer,
                    "source": "per_layer",
                    "score": score,
                    "delta": delta,
                })
                print(f"    L{inject_layer:2d} +layer_delta: score={score:.3f}, Δ={delta:+.3f}")
        
        # ==========================================
        # Test C: 多层累积注入 — 轨道vs向量判别
        # ==========================================
        print(f"\n  --- Test C: Multi-Layer Cumulative Injection ---")
        
        cumulative_results = []
        
        # C1: 有序累积 (L0, L0+L3, L0+L3+L6, ...)
        for n_inject in [1, 2, 3, 4, 5, 6, 8, 10, 14]:
            inject_layers = sample_layers[:n_inject]
            deltas = {li: concept_deltas[li] for li in inject_layers if li in concept_deltas}
            if not deltas:
                continue
            logits = additive_steering_forward(
                model, tokenizer, device, baseline_text, deltas, alpha
            )
            if logits is not None:
                score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                delta = score - baseline_score
                cumulative_results.append({
                    "type": "ordered",
                    "n_layers": len(deltas),
                    "layers": list(deltas.keys()),
                    "score": score,
                    "delta": delta,
                })
                print(f"    Ordered {len(deltas)} layers: Δ={delta:+.3f}")
        
        # C2: 反向累积 (L35, L35+L33, ...)
        reversed_layers = list(reversed(sample_layers))
        for n_inject in [1, 2, 3, 4, 5, 6, 8, 10, 14]:
            inject_layers = reversed_layers[:n_inject]
            deltas = {li: concept_deltas[li] for li in inject_layers if li in concept_deltas}
            if not deltas:
                continue
            logits = additive_steering_forward(
                model, tokenizer, device, baseline_text, deltas, alpha
            )
            if logits is not None:
                score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                delta = score - baseline_score
                cumulative_results.append({
                    "type": "reversed",
                    "n_layers": len(deltas),
                    "layers": list(deltas.keys()),
                    "score": score,
                    "delta": delta,
                })
                print(f"    Reversed {len(deltas)} layers: Δ={delta:+.3f}")
        
        # C3: 随机层序累积
        rng = np.random.RandomState(42)
        shuffled_layers = list(sample_layers)
        rng.shuffle(shuffled_layers)
        for n_inject in [1, 3, 6, 10, 14]:
            inject_layers = shuffled_layers[:n_inject]
            deltas = {li: concept_deltas[li] for li in inject_layers if li in concept_deltas}
            if not deltas:
                continue
            logits = additive_steering_forward(
                model, tokenizer, device, baseline_text, deltas, alpha
            )
            if logits is not None:
                score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                delta = score - baseline_score
                cumulative_results.append({
                    "type": "random",
                    "n_layers": len(deltas),
                    "layers": list(deltas.keys()),
                    "score": score,
                    "delta": delta,
                })
                print(f"    Random {len(deltas)} layers: Δ={delta:+.3f}")
        
        # C4: 只用中间层 (跳过浅层和深层)
        mid_only = sample_layers[4:10]  # 大约L12-L27
        deltas_mid = {li: concept_deltas[li] for li in mid_only if li in concept_deltas}
        if deltas_mid:
            logits = additive_steering_forward(
                model, tokenizer, device, baseline_text, deltas_mid, alpha
            )
            if logits is not None:
                score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                delta = score - baseline_score
                cumulative_results.append({
                    "type": "mid_only",
                    "n_layers": len(deltas_mid),
                    "layers": list(deltas_mid.keys()),
                    "score": score,
                    "delta": delta,
                })
                print(f"    Mid-only {len(deltas_mid)} layers: Δ={delta:+.3f}")
        
        # ==========================================
        # Test D: 子空间 vs 向量 — 单PC vs 多PC
        # ==========================================
        print(f"\n  --- Test D: Subspace vs Vector ---")
        
        # 用多上下文收集概念表示
        concept_matrix = []
        for li in [9, 12, 18, 24]:  # 中间4层
            if li in concept_trajectories and len(concept_trajectories[li]) >= 3:
                for state in concept_trajectories[li]:
                    concept_matrix.append(state)
        
        subspace_results = []
        if len(concept_matrix) >= 5:
            concept_matrix = np.array(concept_matrix)
            centered = concept_matrix - concept_matrix.mean(axis=0)
            _, S_pca, Vt_pca = svd(centered, full_matrices=False)
            
            target_layer = n_layers // 2
            
            for n_dirs in [1, 3, 5, 10]:
                # 构造子空间向量
                subspace_vec = np.zeros(d_model, dtype=np.float32)
                for k in range(min(n_dirs, len(Vt_pca))):
                    subspace_vec += (S_pca[k] / S_pca[0]) * Vt_pca[k]
                
                deltas = {target_layer: subspace_vec}
                logits = additive_steering_forward(
                    model, tokenizer, device, baseline_text, deltas, 2.0
                )
                if logits is not None:
                    score, words = compute_concept_score(logits, tokenizer, concept_data["probe_words"])
                    delta = score - baseline_score
                    subspace_results.append({
                        "n_dirs": n_dirs,
                        "score": score,
                        "delta": delta,
                    })
                    print(f"    {n_dirs}-PC subspace: Δ={delta:+.3f}")
        
        # 汇总
        results[concept_name] = {
            "baseline_score": baseline_score,
            "vector_results": vector_results,
            "perlayer_results": perlayer_results,
            "cumulative_results": cumulative_results,
            "subspace_results": subspace_results,
        }
        
        # 判别
        print(f"\n  --- VERDICT for {concept_name} ---")
        
        # A: 向量性
        if vector_results:
            effective_count = sum(1 for r in vector_results if r["delta"] > 0.3)
            total = len(vector_results)
            avg_delta = np.mean([r["delta"] for r in vector_results])
            print(f"    Vector: {effective_count}/{total} layers effective, avg Δ={avg_delta:+.3f}")
            if effective_count > total * 0.5:
                print(f"    → Strong VECTOR support")
            elif effective_count > total * 0.2:
                print(f"    → Partial VECTOR support (layer-dependent)")
            else:
                print(f"    → Weak VECTOR support")
        
        # C: 轨道性
        ordered_5 = [r for r in cumulative_results if r["type"] == "ordered" and r["n_layers"] >= 4]
        reversed_5 = [r for r in cumulative_results if r["type"] == "reversed" and r["n_layers"] >= 4]
        random_5 = [r for r in cumulative_results if r["type"] == "random" and r["n_layers"] >= 4]
        
        if ordered_5 and reversed_5:
            od = np.mean([r["delta"] for r in ordered_5])
            rd = np.mean([r["delta"] for r in reversed_5])
            ratio = od / max(abs(rd), 0.01) if rd != 0 else float('inf')
            print(f"    Trajectory: ordered Δ={od:+.3f}, reversed Δ={rd:+.3f}, ratio={ratio:.2f}")
            if od > 0 and rd < od * 0.3:
                print(f"    → Strong TRAJECTORY support (reversed << ordered)")
            elif od > 0 and rd < od * 0.7:
                print(f"    → Partial TRAJECTORY support")
            else:
                print(f"    → Weak TRAJECTORY support")
        
        # 单层 vs 多层
        max_single = max([r["delta"] for r in perlayer_results], default=0)
        if cumulative_results:
            max_cumulative = max([r["delta"] for r in cumulative_results if r["n_layers"] >= 4], default=0)
            print(f"    Max single-layer Δ={max_single:+.3f}, max multi-layer Δ={max_cumulative:+.3f}")
            if max_cumulative > max_single * 1.5:
                print(f"    → SUBSPACE/TRAJECTORY (multi-layer much better)")
            elif max_cumulative > max_single * 1.2:
                print(f"    → MIXED (multi-layer slightly better)")
            else:
                print(f"    → VECTOR (single layer comparable)")
        
        del concept_trajectories, concept_mean, baseline_states, concept_deltas
        gc.collect()
    
    # 保存结果
    output_path = TEMP / f"ccxiib_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")
    
    # 最终总结
    print(f"\n{'#'*70}")
    print(f"CCXII-B Final Summary")
    print(f"{'#'*70}")
    
    for concept_name, data in results.items():
        print(f"\n  {concept_name}:")
        
        # 向量支持度
        vr = data.get("vector_results", [])
        if vr:
            eff = sum(1 for r in vr if r["delta"] > 0.3)
            print(f"    Vector: {eff}/{len(vr)} layers effective")
        
        # 轨道支持度
        cr = data.get("cumulative_results", [])
        ordered = [r for r in cr if r["type"] == "ordered" and r["n_layers"] >= 4]
        reversed_r = [r for r in cr if r["type"] == "reversed" and r["n_layers"] >= 4]
        if ordered and reversed_r:
            od = np.mean([r["delta"] for r in ordered])
            rd = np.mean([r["delta"] for r in reversed_r])
            print(f"    Trajectory: ordered={od:+.3f} vs reversed={rd:+.3f}")
        
        # 单层vs多层
        pr = data.get("perlayer_results", [])
        max_single = max([r["delta"] for r in pr], default=0)
        max_multi = max([r["delta"] for r in cr if r["n_layers"] >= 4], default=0)
        print(f"    Single max={max_single:+.3f}, Multi max={max_multi:+.3f}")
    
    release_model(model)
    print(f"\nCCXII-B {model_name} 完成!")


if __name__ == "__main__":
    main()
