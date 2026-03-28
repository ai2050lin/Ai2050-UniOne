"""
Phase 3: 螺旋轨迹分类 & 中间层深度分析 & 注意力头功能标注 & 跨模型验证
========================================================================
基于Phase 2的"螺旋收敛假说"，深入验证:
  1. 偏转轨迹分类: 语法/逻辑/风格/语义的特征性螺旋轨迹
  2. 中间层深度分析: 第3-8层分歧区域的逐层细粒度分析
  3. 注意力头功能标注: activation patching确定头的语言功能
  4. 跨模型一致性: GPT-2 + Qwen2.5-0.5B + Qwen2.5-1.5B

使用模型: GPT-2, Qwen2.5-0.5B, Qwen2.5-1.5B
硬件: RTX 3080 8GB
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 3 扩展测试集 - 每个维度更多变体
TEST_CASES = {
    "syntax_svo": "The cat catches the mouse.",
    "syntax_passive": "The mouse is caught by the cat.",
    "syntax_question": "Does the cat catch the mouse?",
    "syntax_embed": "The cat, which is black, catches the mouse.",
    "logic_valid": "All birds can fly. Penguins are birds. So penguins can fly.",
    "logic_invalid": "All birds can fly. Penguins are birds. So penguins cannot fly.",
    "logic_conditional": "If it rains, the ground gets wet. It is raining.",
    "logic_negation": "The sky is not blue.",
    "style_formal": "The aforementioned research demonstrates significant findings.",
    "style_casual": "That research shows some really cool stuff.",
    "style_academic": "Empirical evidence suggests a statistically significant correlation.",
    "style_poetic": "The moonlight weaves through whispering branches of silver.",
    "semantic_concrete": "The cat purrs softly on the mat.",
    "semantic_abstract": "Justice requires fairness and equality.",
    "semantic_temporal": "Yesterday the conference will have been completed.",
    "semantic_spatial": "The bridge extends across the river to the northern shore.",
}


def load_model(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float32, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    config = {
        "model_name": model_name,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers,
        "num_heads": model.config.n_head if hasattr(model.config, 'n_head') else model.config.num_attention_heads,
        "num_kv_heads": getattr(model.config, 'num_key_value_heads', None),
        "vocab_size": model.config.vocab_size,
    }
    print(f"  hidden={config['hidden_size']}, layers={config['num_layers']}, heads={config['num_heads']}")
    return model, tokenizer, config


def get_layer_modules(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    return []


def run_forward_capture(model, tokenizer, sentence, max_length=64):
    """Run forward pass, capture residual stream and attention per layer."""
    hooks = []
    residual_cache = {}
    attn_cache = {}
    layers = get_layer_modules(model)

    def make_res_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            residual_cache[idx] = h.detach().cpu()
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_res_hook(i)))

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    token_ids = inputs["input_ids"][0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    with torch.no_grad():
        output = model(**inputs, output_attentions=True)

    attentions = output.attentions
    for h in hooks:
        h.remove()

    return residual_cache, attentions, token_ids, tokens


# ============================================================
# 分析1: 偏转轨迹特征提取与分类
# ============================================================
def analyze_spiral_trajectories(model, tokenizer, config):
    """
    为每个句子提取残差流的螺旋轨迹特征:
    - 每层偏转角度序列
    - 偏转角度的标准差和自相关
    - 螺旋半径变化 (残差范数)
    - 角速度 (偏转角/范数比)
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: SPIRAL TRAJECTORY CLASSIFICATION")
    print("=" * 60)

    dimension_groups = {
        "syntax": ["syntax_svo", "syntax_passive", "syntax_question", "syntax_embed"],
        "logic": ["logic_valid", "logic_invalid", "logic_conditional", "logic_negation"],
        "style": ["style_formal", "style_casual", "style_academic", "style_poetic"],
        "semantic": ["semantic_concrete", "semantic_abstract", "semantic_temporal", "semantic_spatial"],
    }

    results = {}
    all_trajectories = {}

    for case_name, sentence in TEST_CASES.items():
        residual_cache, _, _, _ = run_forward_capture(model, tokenizer, sentence)

        # 提取逐层 mean direction
        layer_directions = []
        layer_norms = []
        for l in sorted(residual_cache.keys()):
            residual = residual_cache[l][0]  # [seq, hidden]
            mean_vec = residual.mean(dim=0)
            norm = float(torch.norm(mean_vec))
            direction = mean_vec / (norm + 1e-8)
            layer_directions.append(direction.numpy())
            layer_norms.append(norm)

        # 计算每层偏转角
        step_angles = []
        for i in range(1, len(layer_directions)):
            cos = np.clip(np.dot(layer_directions[i-1], layer_directions[i]), -1, 1)
            angle = float(np.degrees(np.arccos(cos)))
            step_angles.append(angle)

        # 螺旋特征
        if len(step_angles) > 1:
            angles_arr = np.array(step_angles)
            norm_arr = np.array(layer_norms[1:])

            # 角速度 (偏转角/范数比)
            angular_velocity = angles_arr / (norm_arr + 1e-8)

            # 自相关: 偏转角的周期性
            if len(angles_arr) > 3:
                autocorr = np.correlate(angles_arr - angles_arr.mean(), angles_arr - angles_arr.mean(), mode='full')
                mid = len(autocorr) // 2
                # 归一化自相关
                norm_autocorr = autocorr[mid:] / (autocorr[mid] + 1e-8)
                # 周期性指标: 第一个谷值的位置
                first_trough = None
                for k in range(1, min(5, len(norm_autocorr))):
                    if k < len(norm_autocorr) and norm_autocorr[k] < 0:
                        first_trough = k
                        break
            else:
                norm_autocorr = []
                first_trough = None

            # 螺距: 两次完整旋转间范数的变化比
            full_rotation_step = int(round(360.0 / (np.mean(angles_arr) + 1e-8)))

            trajectory_features = {
                "sentence": sentence,
                "step_angles": step_angles,
                "mean_angle": float(np.mean(angles_arr)),
                "std_angle": float(np.std(angles_arr)),
                "min_angle": float(np.min(angles_arr)),
                "max_angle": float(np.max(angles_arr)),
                "angular_velocity_mean": float(np.mean(angular_velocity)),
                "angular_velocity_std": float(np.std(angular_velocity)),
                "norm_growth_ratio": float(layer_norms[-1] / (layer_norms[0] + 1e-8)),
                "autocorr_first_trough": first_trough,
                "full_rotation_step": full_rotation_step,
                "layer_norms": layer_norms,
                "total_angle": float(np.sum(angles_arr)),
            }

            all_trajectories[case_name] = trajectory_features
            results[case_name] = {
                "mean_angle": trajectory_features["mean_angle"],
                "std_angle": trajectory_features["std_angle"],
                "angular_velocity_mean": trajectory_features["angular_velocity_mean"],
                "norm_growth_ratio": trajectory_features["norm_growth_ratio"],
                "autocorr_first_trough": first_trough,
                "full_rotation_step": full_rotation_step,
                "step_angles": step_angles,
            }

    # 维度级汇总
    dimension_summary = {}
    for dim_name, case_list in dimension_groups.items():
        valid_cases = [c for c in case_list if c in all_trajectories]
        if not valid_cases:
            continue
        dim_features = {
            "cases": valid_cases,
            "mean_mean_angle": float(np.mean([all_trajectories[c]["mean_angle"] for c in valid_cases])),
            "mean_std_angle": float(np.mean([all_trajectories[c]["std_angle"] for c in valid_cases])),
            "mean_angular_velocity": float(np.mean([all_trajectories[c]["angular_velocity_mean"] for c in valid_cases])),
            "mean_norm_growth": float(np.mean([all_trajectories[c]["norm_growth_ratio"] for c in valid_cases])),
            "angle_std_across_cases": float(np.std([all_trajectories[c]["mean_angle"] for c in valid_cases])),
            "norm_growth_std_across_cases": float(np.std([all_trajectories[c]["norm_growth_ratio"] for c in valid_cases])),
        }
        # 检查同维度内的轨迹相似性
        if len(valid_cases) >= 2:
            all_step_angles = [all_trajectories[c]["step_angles"] for c in valid_cases]
            min_len = min(len(a) for a in all_step_angles)
            # 对齐后的相关性
            pairwise_corrs = []
            for i in range(len(all_step_angles)):
                for j in range(i+1, len(all_step_angles)):
                    a = np.array(all_step_angles[i][:min_len])
                    b = np.array(all_step_angles[j][:min_len])
                    if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                        corr = float(np.corrcoef(a, b)[0, 1])
                        pairwise_corrs.append(corr)
            dim_features["intra_dimension_correlation"] = float(np.mean(pairwise_corrs)) if pairwise_corrs else 0
        dimension_summary[dim_name] = dim_features

        print(f"  [{dim_name}] mean_angle={dim_features['mean_mean_angle']:.2f}+-{dim_features['mean_std_angle']:.2f}, "
              f"ang_vel={dim_features['mean_angular_velocity']:.6f}, "
              f"norm_growth={dim_features['mean_norm_growth']:.2f}x, "
              f"intra_corr={dim_features.get('intra_dimension_correlation', 'N/A')}")

    return {"per_case": results, "dimension_summary": dimension_summary}


# ============================================================
# 分析2: 中间层深度分析 (逐token + 逐层)
# ============================================================
def analyze_middle_layers(model, tokenizer, config):
    """
    对中间层(分歧区域)做逐token分析:
    - 每个token在各层的方向变化
    - 找出"分歧爆发点"——哪个token在哪一层开始与其他token分叉
    - 语法树位置与分歧的对应关系
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: MIDDLE LAYER DEEP ANALYSIS")
    print("=" * 60)

    # 选取关键句子对
    analysis_pairs = [
        {
            "name": "syntax_active_passive",
            "sent_a": "The cat catches the mouse.",
            "sent_b": "The mouse is caught by the cat.",
            "dimension": "syntax",
        },
        {
            "name": "logic_valid_invalid",
            "sent_a": "All birds can fly. Penguins are birds. So penguins can fly.",
            "sent_b": "All birds can fly. Penguins are birds. So penguins cannot fly.",
            "dimension": "logic",
        },
        {
            "name": "style_formal_casual",
            "sent_a": "The aforementioned research demonstrates significant findings.",
            "sent_b": "That research shows some really cool stuff.",
            "dimension": "style",
        },
        {
            "name": "semantic_concrete_abstract",
            "sent_a": "The cat purrs softly on the mat.",
            "sent_b": "Justice requires fairness and equality.",
            "dimension": "semantic",
        },
    ]

    results = {}

    for pair in analysis_pairs:
        name = pair["name"]
        print(f"\n  --- {name} ---")

        res_a, _, ids_a, toks_a = run_forward_capture(model, tokenizer, pair["sent_a"])
        res_b, _, ids_b, toks_b = run_forward_capture(model, tokenizer, pair["sent_b"])

        num_layers = len(res_a)
        min_seq = min(len(toks_a), len(toks_b))

        # 逐token逐层的余弦相似度矩阵: [token_pos, layer]
        cos_matrix = np.zeros((min_seq, num_layers))
        angle_matrix = np.zeros((min_seq, num_layers))
        norm_matrix_a = np.zeros((min_seq, num_layers))
        norm_matrix_b = np.zeros((min_seq, num_layers))

        for t in range(min_seq):
            for l in sorted(res_a.keys()):
                if l >= num_layers or l >= len(res_b):
                    continue
                va = res_a[l][0, t]
                vb = res_b[l][0, t]
                na = torch.norm(va)
                nb = torch.norm(vb)
                norm_matrix_a[t, l] = float(na)
                norm_matrix_b[t, l] = float(nb)
                if na > 1e-6 and nb > 1e-6:
                    cos = float(torch.dot(va, vb) / (na * nb))
                    cos = np.clip(cos, -1, 1)
                    cos_matrix[t, l] = cos
                    angle_matrix[t, l] = float(np.degrees(np.arccos(cos)))

        # 找分歧爆发点: 对每个token, cos下降最快的层
        divergence_layers = []
        divergence_magnitudes = []
        for t in range(min_seq):
            cos_t = cos_matrix[t]
            # cos下降量
            drops = cos_t[:-1] - cos_t[1:] if len(cos_t) > 1 else []
            if len(drops) > 0:
                max_drop_layer = int(np.argmax(drops))
                max_drop_val = float(np.max(drops))
                divergence_layers.append(max_drop_layer)
                divergence_magnitudes.append(max_drop_val)

        # 逐层摘要: 该层所有token的平均分歧程度
        layer_divergence = np.mean(1 - cos_matrix, axis=0)

        # 找全局分歧最大层
        max_div_layer = int(np.argmax(layer_divergence))
        max_div_value = float(layer_divergence[max_div_layer])

        pair_result = {
            "name": name,
            "dimension": pair["dimension"],
            "sentence_a": pair["sent_a"],
            "sentence_b": pair["sent_b"],
            "aligned_length": min_seq,
            "tokens_a": [t.encode("ascii", "replace").decode() for t in toks_a[:min_seq]],
            "tokens_b": [t.encode("ascii", "replace").decode() for t in toks_b[:min_seq]],
            "max_divergence_layer": max_div_layer,
            "max_divergence_value": max_div_value,
            "layer_divergence_profile": layer_divergence.tolist(),
            "per_token_divergence_layer": divergence_layers,
            "per_token_divergence_magnitude": divergence_magnitudes,
        }

        # 打印top3分歧token
        if divergence_magnitudes:
            top3_idx = np.argsort(divergence_magnitudes)[-3:][::-1]
            print(f"    Max divergence at layer {max_div_layer} (val={max_div_value:.4f})")
            for idx in top3_idx:
                ta = pair_result["tokens_a"][idx] if idx < len(pair_result["tokens_a"]) else "?"
                tb = pair_result["tokens_b"][idx] if idx < len(pair_result["tokens_b"]) else "?"
                print(f"    Token pos {idx}: '{ta}' vs '{tb}', "
                      f"divergence_layer={divergence_layers[idx]}, "
                      f"magnitude={divergence_magnitudes[idx]:.4f}")

        results[name] = pair_result

    return results


# ============================================================
# 分析3: 注意力头功能标注 (Logit Attribution)
# ============================================================
def analyze_attention_heads(model, tokenizer, config):
    """
    通过逐头注意力模式分析确定头功能:
    - Prev-token heads (上三角高注意力)
    - Induction heads (重复token间高注意力)
    - Duplicate token heads (相同token间高注意力)
    - 远程依赖头 (距离>3的高注意力)
    - 对每个句子, 标注每个头的dominant功能
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: ATTENTION HEAD FUNCTION PROFILING")
    print("=" * 60)

    num_heads = config["num_heads"]
    num_layers = config["num_layers"]

    # 用多个句子建立头的功能轮廓
    probe_sentences = [
        "The cat sat on the mat. The cat was happy.",
        "All birds can fly. Penguins are birds.",
        "The sky is not blue today.",
        "Research demonstrates findings.",
    ]

    # 每个头的功能得分
    head_scores = {l: {h: {"prev_token": 0, "duplicate": 0, "remote": 0, "first_token": 0, "uniformity": 0}
                      for h in range(num_heads)} for l in range(num_layers)}

    for sent in probe_sentences:
        _, attentions, _, tokens = run_forward_capture(model, tokenizer, sent)
        if attentions is None:
            continue

        for l, attn_layer in enumerate(attentions):
            attn = attn_layer[0].cpu().numpy()  # [batch, heads, seq, seq] -> [heads, seq, seq]
            if l >= num_layers:
                break

            for h in range(min(num_heads, attn.shape[0])):
                a = attn[h]  # [seq, seq]
                seq_len = a.shape[0]

                # 归一化为概率
                a_softmax = np.exp(a - a.max(axis=-1, keepdims=True))
                a_softmax = a_softmax / (a_softmax.sum(axis=-1, keepdims=True) + 1e-10)

                # Prev-token: 对角线偏移1
                if seq_len > 1:
                    diag_vals = np.array([a_softmax[i, max(0, i-1)] for i in range(seq_len)])
                    head_scores[l][h]["prev_token"] += float(diag_vals.mean())

                # Duplicate token: 相同token ID之间
                token_ids = tokenizer.encode(sent)
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i != j and i < len(token_ids) and j < len(token_ids):
                            if token_ids[i] == token_ids[j]:
                                head_scores[l][h]["duplicate"] += a_softmax[i, j] / seq_len

                # Remote: 距离>3
                if seq_len > 4:
                    for i in range(seq_len):
                        remote_sum = 0
                        count = 0
                        for j in range(seq_len):
                            if abs(i - j) > 3:
                                remote_sum += a_softmax[i, j]
                                count += 1
                        if count > 0:
                            head_scores[l][h]["remote"] += remote_sum / seq_len

                # First token (BOS) attention
                if seq_len > 1:
                    head_scores[l][h]["first_token"] += float(a_softmax[:, 0].mean())

                # Uniformity (entropy-based)
                for i in range(seq_len):
                    ent = -np.sum(a_softmax[i] * np.log(a_softmax[i] + 1e-10))
                    max_ent = np.log(seq_len)
                    head_scores[l][h]["uniformity"] += ent / (max_ent + 1e-10)
                head_scores[l][h]["uniformity"] /= seq_len

    # 平均并分类
    num_probes = len(probe_sentences)
    head_profile = {}
    for l in range(num_layers):
        for h in range(num_heads):
            for k in head_scores[l][h]:
                head_scores[l][h][k] /= num_probes

            # 分类: dominant function
            scores = head_scores[l][h]
            dominant = max(scores, key=lambda x: scores[x])
            head_profile[f"layer{l}_head{h}"] = {
                "layer": l,
                "head": h,
                "scores": scores,
                "dominant_function": dominant,
            }

    # 统计各功能头的分布
    function_counts = {"prev_token": 0, "duplicate": 0, "remote": 0, "first_token": 0, "uniformity": 0}
    for hp in head_profile.values():
        function_counts[hp["dominant_function"]] += 1

    print(f"  Total heads: {len(head_profile)}")
    for func, count in function_counts.items():
        print(f"    {func}: {count} heads")

    # 打印每层功能分布
    print("\n  Per-layer function distribution:")
    layer_dist = {l: {"prev_token": 0, "duplicate": 0, "remote": 0, "first_token": 0, "uniformity": 0}
                  for l in range(num_layers)}
    for hp in head_profile.values():
        layer_dist[hp["layer"]][hp["dominant_function"]] += 1

    for l in range(num_layers):
        dist = layer_dist[l]
        dom = max(dist, key=dist.get)
        print(f"    Layer {l:2d}: {dict(dist)} (dominant: {dom})")

    return {"head_profile": head_profile, "function_counts": function_counts, "layer_distribution": layer_dist}


# ============================================================
# 分析4: 跨模型一致性验证
# ============================================================
def cross_model_validation(model, tokenizer, config):
    """
    验证Phase 2发现是否跨模型一致:
    - 每层偏转角度
    - 方向累积
    - 有效秩变化趋势
    - 跨维度收敛模式
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: CROSS-MODEL CONSISTENCY CHECK")
    print("=" * 60)

    results = {}

    for case_name, sentence in TEST_CASES.items():
        residual_cache, _, _, _ = run_forward_capture(model, tokenizer, sentence)

        layer_data = []
        prev_dir = None
        for l in sorted(residual_cache.keys()):
            residual = residual_cache[l][0]
            mean_vec = residual.mean(dim=0)
            norm = float(torch.norm(mean_vec))
            direction = mean_vec / (norm + 1e-8)

            step_angle = 0.0
            if prev_dir is not None:
                cos = np.clip(float(torch.dot(direction, prev_dir)), -1, 1)
                step_angle = float(np.degrees(np.arccos(cos)))
            prev_dir = direction

            # 有效秩
            if residual.shape[0] >= 3:
                diffs = []
                for i in range(residual.shape[0]):
                    for j in range(i+1, residual.shape[0]):
                        diffs.append((residual[i] - residual[j]).numpy())
                diffs = torch.tensor(np.array(diffs))
                try:
                    _, S, _ = torch.linalg.svd(diffs.float(), full_matrices=False)
                    S = S[S > 1e-6]
                    eff_rank = float(S.sum() / S[0]) if len(S) > 0 else 0
                except Exception:
                    eff_rank = 0
            else:
                eff_rank = 0

            layer_data.append({
                "layer": l,
                "norm": norm,
                "step_angle": step_angle,
                "effective_rank": eff_rank,
            })

        if layer_data:
            total_angle = sum(ld["step_angle"] for ld in layer_data)
            mean_step = np.mean([ld["step_angle"] for ld in layer_data[1:]]) if len(layer_data) > 1 else 0
            results[case_name] = {
                "layer_data": layer_data,
                "total_angle": total_angle,
                "mean_step_angle": mean_step,
                "norm_growth": layer_data[-1]["norm"] / (layer_data[0]["norm"] + 1e-8),
                "rank_trend": layer_data[-1]["effective_rank"] - layer_data[0]["effective_rank"],
            }

    # 汇总统计
    all_mean_steps = [r["mean_step_angle"] for r in results.values() if r["mean_step_angle"] > 0]
    all_total_angles = [r["total_angle"] for r in results.values()]
    all_norm_growths = [r["norm_growth"] for r in results.values()]
    all_rank_trends = [r["rank_trend"] for r in results.values()]

    summary = {
        "model_name": config["model_name"],
        "num_test_cases": len(results),
        "mean_step_angle_avg": float(np.mean(all_mean_steps)) if all_mean_steps else 0,
        "mean_step_angle_std": float(np.std(all_mean_steps)) if all_mean_steps else 0,
        "total_angle_avg": float(np.mean(all_total_angles)) if all_total_angles else 0,
        "norm_growth_avg": float(np.mean(all_norm_growths)) if all_norm_growths else 0,
        "rank_trend_avg": float(np.mean(all_rank_trends)) if all_rank_trends else 0,
        "rank_trend_positive": sum(1 for t in all_rank_trends if t > 0),
        "rank_trend_negative": sum(1 for t in all_rank_trends if t < 0),
    }

    print(f"  Model: {config['model_name']}")
    print(f"  Mean step angle: {summary['mean_step_angle_avg']:.2f} +/- {summary['mean_step_angle_std']:.2f}")
    print(f"  Total angle (avg): {summary['total_angle_avg']:.1f}")
    print(f"  Norm growth (avg): {summary['norm_growth_avg']:.2f}x")
    print(f"  Rank trend: avg={summary['rank_trend_avg']:.2f}, "
          f"positive={summary['rank_trend_positive']}, negative={summary['rank_trend_negative']}")

    return {"per_case": results, "summary": summary}


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": timestamp, "phase": 3, "models": {}}

    # ====== GPT-2 ======
    print("=" * 60)
    print("PHASE 3: GPT-2")
    print("=" * 60)
    model_gpt2, tok_gpt2, cfg_gpt2 = load_model("gpt2")

    results["models"]["gpt2"] = {
        "config": cfg_gpt2,
        "spiral_trajectories": analyze_spiral_trajectories(model_gpt2, tok_gpt2, cfg_gpt2),
        "middle_layers": analyze_middle_layers(model_gpt2, tok_gpt2, cfg_gpt2),
        "attention_heads": analyze_attention_heads(model_gpt2, tok_gpt2, cfg_gpt2),
        "cross_model": cross_model_validation(model_gpt2, tok_gpt2, cfg_gpt2),
    }

    del model_gpt2
    torch.cuda.empty_cache()

    # ====== Qwen2.5-0.5B ======
    print("\n" + "=" * 60)
    print("PHASE 3: Qwen2.5-0.5B")
    print("=" * 60)
    try:
        model_qwen, tok_qwen, cfg_qwen = load_model("Qwen/Qwen2.5-0.5B")
        results["models"]["qwen2.5-0.5b"] = {
            "config": cfg_qwen,
            "spiral_trajectories": analyze_spiral_trajectories(model_qwen, tok_qwen, cfg_qwen),
            "middle_layers": analyze_middle_layers(model_qwen, tok_qwen, cfg_qwen),
            "cross_model": cross_model_validation(model_qwen, tok_qwen, cfg_qwen),
        }
        del model_qwen
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  Qwen-0.5B failed: {e}")
        results["models"]["qwen2.5-0.5b"] = {"error": str(e)}

    # ====== Qwen2.5-1.5B ======
    print("\n" + "=" * 60)
    print("PHASE 3: Qwen2.5-1.5B")
    print("=" * 60)
    try:
        model_qwen15, tok_qwen15, cfg_qwen15 = load_model("Qwen/Qwen2.5-1.5B")
        results["models"]["qwen2.5-1.5b"] = {
            "config": cfg_qwen15,
            "spiral_trajectories": analyze_spiral_trajectories(model_qwen15, tok_qwen15, cfg_qwen15),
            "cross_model": cross_model_validation(model_qwen15, tok_qwen15, cfg_qwen15),
        }
        del model_qwen15
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  Qwen-1.5B failed: {e}")
        results["models"]["qwen2.5-1.5b"] = {"error": str(e)}

    # Save
    output_path = OUTPUT_DIR / f"phase3_spiral_trajectory_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
