"""
Phase 4: 维度信息定位 & 子空间投影分析 & 因果验证
===================================================
Phase 3得出"通用螺旋假说":螺旋轨迹是架构级不变量，不携带维度信息。
Phase 4的核心问题: 语法/逻辑/风格/语义的差异到底编码在哪里?

分析模块:
  1. 残差流子空间投影分析: 在不同子空间中测量维度分离度
  2. 个别token轨迹偏离分析: 哪些token偏离了通用螺旋
  3. 逐层维度判别力分析: 哪一层对哪个维度的判别力最强
  4. 因果干预: 零化/扰动特定维度，观察下游变化
  5. FFN权重谱分析: 旋转器的方向偏好

使用模型: GPT-2 + Qwen2.5-0.5B
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
from itertools import combinations

OUTPUT_DIR = Path("d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 维度对比句子对
DIMENSION_PAIRS = {
    "syntax": [
        ("The cat catches the mouse.", "The mouse is caught by the cat.", "active_passive"),
        ("The cat catches the mouse.", "Does the cat catch the mouse?", "declarative_question"),
    ],
    "logic": [
        ("All birds can fly. Penguins are birds. So penguins can fly.",
         "All birds can fly. Penguins are birds. So penguins cannot fly.", "valid_invalid"),
        ("The sky is blue.", "The sky is not blue.", "affirm_negate"),
    ],
    "style": [
        ("The committee has reached a consensus regarding this matter.",
         "Everyone agreed on what to do about it.", "formal_casual"),
        ("The empirical evidence suggests a significant correlation.",
         "The data shows something interesting.", "academic_plain"),
    ],
    "semantic": [
        ("The cat purrs softly on the mat.", "Justice requires fairness and equality.", "concrete_abstract"),
        ("The bridge extends across the river.", "Democracy demands equal representation.", "concrete_abstract2"),
    ],
}

# 用于维度判别的独立测试句子
INDEPENDENT_SENTENCES = {
    "syntax": [
        "The dog chased the ball across the park.",
        "The ball was chased by the dog across the park.",
        "What did the dog chase across the park?",
    ],
    "logic": [
        "All mammals are warm-blooded. Whales are mammals. Therefore whales are warm-blooded.",
        "All mammals are warm-blooded. Whales are mammals. Therefore whales are cold-blooded.",
    ],
    "style": [
        "The aforementioned study provides compelling evidence.",
        "This study gives pretty good proof.",
    ],
    "semantic": [
        "The children played happily in the garden.",
        "Freedom requires sacrifice and responsibility.",
    ],
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


def capture_all_layers(model, tokenizer, sentence, max_length=64):
    """Capture residual stream at all layers."""
    hooks = []
    residual_cache = {}
    layers = get_layer_modules(model)

    def make_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            residual_cache[idx] = h.detach().cpu()
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    token_ids = inputs["input_ids"][0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return residual_cache, token_ids, tokens


def compute_token_pair_residuals(model, tokenizer, sent_a, sent_b, max_length=64):
    """Get aligned residual streams for sentence pair."""
    res_a, ids_a, toks_a = capture_all_layers(model, tokenizer, sent_a, max_length)
    res_b, ids_b, toks_b = capture_all_layers(model, tokenizer, sent_b, max_length)

    min_seq = min(len(toks_a), len(toks_b))
    num_layers = min(len(res_a), len(res_b))

    # Return aligned residuals: list of [min_seq, hidden] tensors
    aligned = []
    for l in range(num_layers):
        ra = res_a[l][0, :min_seq]  # [min_seq, hidden]
        rb = res_b[l][0, :min_seq]
        aligned.append((ra, rb))

    return aligned, min_seq, toks_a[:min_seq], toks_b[:min_seq], num_layers


# ============================================================
# 分析1: 子空间投影 - 在不同子空间中测量维度分离度
# ============================================================
def analyze_subspace_projection(model, tokenizer, config):
    """
    将残差流投影到不同子空间:
    - 全空间 (baseline)
    - Top-K PCA子空间
    - 随机子空间
    - 正交于均值的子空间 (去除全局方向)
    测量每个子空间中句子对的维度分离度
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: SUBSPACE PROJECTION")
    print("=" * 60)

    results = {}
    hidden = config["hidden_size"]

    for dim_name, pairs in DIMENSION_PAIRS.items():
        print(f"\n  --- {dim_name} ---")
        dim_results = []

        for sent_a, sent_b, pair_name in pairs:
            aligned, min_seq, toks_a, toks_b, num_layers = compute_token_pair_residuals(
                model, tokenizer, sent_a, sent_b
            )

            # 收集所有层的残差差值用于PCA
            all_diffs = []
            for l in range(num_layers):
                ra, rb = aligned[l]
                diff = (ra - rb).mean(dim=0)  # 平均token差
                all_diffs.append(diff.numpy())
            all_diffs = np.array(all_diffs)  # [layers, hidden]

            # PCA: 找到差异的主方向
            mean_diff = all_diffs.mean(axis=0)
            _, S, Vt = np.linalg.svd(all_diffs - all_diffs.mean(axis=0), full_matrices=False)

            # 分析几个关键层
            key_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            layer_results = {}

            for l in key_layers:
                ra, rb = aligned[l]
                diff = ra - rb  # [seq, hidden]
                mean_d = diff.mean(dim=0)

                # 在不同子空间中测量差异
                # (a) 全空间
                full_norm = float(torch.norm(mean_d))

                # (b) Top-K PCA
                for k in [1, 5, 10, 50]:
                    if k <= min(Vt.shape[0], Vt.shape[1]):
                        subspace = Vt[:k].T  # [hidden, k]
                        proj = mean_d.numpy() @ subspace @ subspace.T
                        proj_norm = float(np.linalg.norm(proj))
                        ratio = proj_norm / (full_norm + 1e-8)
                        layer_results[f"pca{k}_ratio"] = ratio

                # (c) 正交于全局均值的子空间
                global_mean = (ra.mean(dim=0) + rb.mean(dim=0)) / 2
                global_mean_norm = float(torch.norm(global_mean))
                # 投影掉全局均值方向
                if global_mean_norm > 1e-6:
                    global_dir = global_mean / global_mean_norm
                    proj_on_global = float(torch.dot(mean_d, global_dir))
                    orthogonal = mean_d - proj_on_global * global_dir
                    orth_norm = float(torch.norm(orthogonal))
                    layer_results["orthogonal_to_global_norm"] = orth_norm
                    layer_results["orthogonal_ratio"] = orth_norm / (full_norm + 1e-8)

                # (d) 逐token差异的标准差 (离散度)
                per_token_norms = torch.norm(diff, dim=-1)  # [seq]
                layer_results["per_token_norm_std"] = float(per_token_norms.std())
                layer_results["per_token_norm_mean"] = float(per_token_norms.mean())

                # (e) 差异的Top-1方向解释的方差比
                diff_matrix = diff.numpy()
                if diff_matrix.shape[0] >= 2:
                    _, s_diff, _ = np.linalg.svd(diff_matrix, full_matrices=False)
                    layer_results["diff_top1_sv_ratio"] = float(s_diff[0] / (s_diff.sum() + 1e-8))

            results[f"{dim_name}_{pair_name}"] = layer_results

            # 打印关键信息
            last_layer = key_layers[-1]
            lr = layer_results
            print(f"    [{pair_name}] L{last_layer}: "
                  f"pca1={lr.get('pca1_ratio', 0):.3f}, pca5={lr.get('pca5_ratio', 0):.3f}, "
                  f"orth_ratio={lr.get('orthogonal_ratio', 0):.3f}, "
                  f"per_tok_std={lr.get('per_token_norm_std', 0):.4f}, "
                  f"diff_top1={lr.get('diff_top1_sv_ratio', 0):.3f}")

    return results


# ============================================================
# 分析2: 个别token轨迹偏离
# ============================================================
def analyze_token_deviation(model, tokenizer, config):
    """
    对每个句子对，计算每个token偏离"通用螺旋"的程度。
    通用螺旋 = 所有token的平均轨迹。
    偏离度 = 每个token的轨迹与平均轨迹的余弦距离。
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: TOKEN-LEVEL DEVIATION FROM UNIVERSAL SPIRAL")
    print("=" * 60)

    results = {}

    for dim_name, pairs in DIMENSION_PAIRS.items():
        print(f"\n  --- {dim_name} ---")
        dim_results = {}

        for sent_a, sent_b, pair_name in pairs:
            aligned, min_seq, toks_a, toks_b, num_layers = compute_token_pair_residuals(
                model, tokenizer, sent_a, sent_b
            )

            # 对每个句子，计算通用螺旋 (所有token的平均方向轨迹)
            universal_spiral_a = []
            universal_spiral_b = []
            for l in range(num_layers):
                ra, rb = aligned[l]
                mean_a = ra.mean(dim=0)
                mean_b = rb.mean(dim=0)
                na = torch.norm(mean_a)
                nb = torch.norm(mean_b)
                if na > 1e-6:
                    universal_spiral_a.append((mean_a / na).numpy())
                if nb > 1e-6:
                    universal_spiral_b.append((mean_b / nb).numpy())

            # 计算每个token的偏离
            token_deviations_a = []
            token_deviations_b = []
            for t in range(min_seq):
                deviations_a = []
                deviations_b = []
                for l in range(num_layers):
                    if l >= len(universal_spiral_a) or l >= len(universal_spiral_b):
                        continue
                    ra_t = aligned[l][0][t]
                    rb_t = aligned[l][1][t]
                    na = torch.norm(ra_t)
                    nb = torch.norm(rb_t)

                    if na > 1e-6 and l < len(universal_spiral_a):
                        dir_t = ra_t / na
                        cos = np.clip(float(np.dot(dir_t.numpy(), universal_spiral_a[l])), -1, 1)
                        deviations_a.append(float(np.degrees(np.arccos(cos))))
                    if nb > 1e-6 and l < len(universal_spiral_b):
                        dir_t = rb_t / nb
                        cos = np.clip(float(np.dot(dir_t.numpy(), universal_spiral_b[l])), -1, 1)
                        deviations_b.append(float(np.degrees(np.arccos(cos))))

                token_deviations_a.append(deviations_a)
                token_deviations_b.append(deviations_b)

            # 找偏离最大的token
            max_dev_a = [(i, np.mean(d)) for i, d in enumerate(token_deviations_a) if d]
            max_dev_b = [(i, np.mean(d)) for i, d in enumerate(token_deviations_b) if d]

            if max_dev_a:
                max_dev_a.sort(key=lambda x: x[1], reverse=True)
                max_dev_b.sort(key=lambda x: x[1], reverse=True)
                top_tokens_a = max_dev_a[:3]
                top_tokens_b = max_dev_b[:3]

                avg_dev_a = np.mean([d[1] for d in max_dev_a])
                avg_dev_b = np.mean([d[1] for d in max_dev_b])

                dim_results[pair_name] = {
                    "avg_deviation_a": avg_dev_a,
                    "avg_deviation_b": avg_dev_b,
                    "top_deviant_tokens_a": [(toks_a[i], d) for i, d in top_tokens_a],
                    "top_deviant_tokens_b": [(toks_b[i], d) for i, d in top_tokens_b],
                }

                print(f"    [{pair_name}] avg_dev: A={avg_dev_a:.2f}°, B={avg_dev_b:.2f}°")
                print(f"      Top-A: {[(toks_a[i].encode('ascii','replace').decode(), f'{d:.1f}') for i,d in top_tokens_a]}")
                print(f"      Top-B: {[(toks_b[i].encode('ascii','replace').decode(), f'{d:.1f}') for i,d in top_tokens_b]}")

        results[dim_name] = dim_results

    return results


# ============================================================
# 分析3: 逐层维度判别力
# ============================================================
def analyze_layer_discriminability(model, tokenizer, config):
    """
    对每一层，计算它对语法/逻辑/风格/语义四个维度的"判别力"。
    判别力 = 同维度句子对的相似度 vs 跨维度句子对的相似度之差。
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: PER-LAYER DIMENSION DISCRIMINABILITY")
    print("=" * 60)

    # 收集所有独立句子的残差流
    all_residuals = {}
    all_labels = {}

    for dim_name, sentences in INDEPENDENT_SENTENCES.items():
        for i, sent in enumerate(sentences):
            label = f"{dim_name}_{i}"
            res_cache, _, _ = capture_all_layers(model, tokenizer, sent)
            all_residuals[label] = res_cache
            all_labels[label] = dim_name

    num_layers = config["num_layers"]
    dimensions = list(all_labels.values())
    unique_dims = list(set(dimensions))

    layer_disc = {l: {dim: 0 for dim in unique_dims} for l in range(num_layers)}

    # 对每一层计算维度判别力
    for l in range(num_layers):
        # 获取所有句子在该层的mean residual
        layer_vectors = {}
        for label, res_cache in all_residuals.items():
            if l in res_cache:
                vec = res_cache[l][0].mean(dim=0)  # [hidden]
                layer_vectors[label] = vec

        if len(layer_vectors) < 4:
            continue

        # 计算同维度内的平均余弦相似度
        intra_dim = {dim: [] for dim in unique_dims}
        inter_dim = []

        labels = list(layer_vectors.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                vi = layer_vectors[labels[i]]
                vj = layer_vectors[labels[j]]
                ni = torch.norm(vi)
                nj = torch.norm(vj)
                if ni > 1e-6 and nj > 1e-6:
                    cos = float(torch.dot(vi, vj) / (ni * nj))
                    if all_labels[labels[i]] == all_labels[labels[j]]:
                        intra_dim[all_labels[labels[i]]].append(cos)
                    else:
                        inter_dim.append(cos)

        # 判别力 = 同维度相似度 - 跨维度相似度 (越负越好)
        for dim in unique_dims:
            if intra_dim[dim] and inter_dim:
                disc = np.mean(intra_dim[dim]) - np.mean(inter_dim)
                layer_disc[l][dim] = disc

    # 打印
    print("\n  Layer | syntax | logic  | style  | semantic")
    print("  " + "-" * 50)
    for l in range(num_layers):
        d = layer_disc[l]
        vals = " | ".join(f"{d[dim]:+.4f}" for dim in unique_dims)
        if any(abs(d[dim]) > 0.001 for dim in unique_dims):
            print(f"  L{l:2d}  | {vals}")

    # 找每维度的最大判别力层
    best_layers = {}
    for dim in unique_dims:
        best_l = max(range(num_layers), key=lambda l: layer_disc[l][dim])
        best_layers[dim] = {"layer": best_l, "score": layer_disc[best_l][dim]}
        print(f"\n  Best layer for {dim}: L{best_l} (score={layer_disc[best_l][dim]:+.4f})")

    return {"per_layer": layer_disc, "best_layers": best_layers}


# ============================================================
# 分析4: FFN权重谱分析
# ============================================================
def analyze_ffn_spectrum(model, tokenizer, config):
    """
    分析FFN权重矩阵的谱性质:
    - 每层FFN的singular value分布
    - FFN变换的"旋转角度"谱 (对随机向量的方向改变)
    - 检测是否有特定维度被所有层一致偏好
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: FFN WEIGHT SPECTRUM")
    print("=" * 60)

    hidden = config["hidden_size"]
    layers = get_layer_modules(model)
    num_layers = len(layers)

    results = {}

    # 对每层FFN提取权重
    for l in range(min(num_layers, 12)):  # 限制层数避免太慢
        layer = layers[l]

        # 提取FFN (MLP) 权重
        mlp = None
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
        elif hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward

        if mlp is None:
            continue

        # 获取up/down projection权重
        W_up = None
        W_down = None
        if hasattr(mlp, 'c_fc'):
            W_up = mlp.c_fc.weight.data.cpu()  # [hidden, intermediate]
            W_down = mlp.c_proj.weight.data.cpu()  # [intermediate, hidden]
        elif hasattr(mlp, 'gate_proj'):
            W_up = mlp.gate_proj.weight.data.cpu()
            W_down = mlp.down_proj.weight.data.cpu()

        if W_up is None or W_down is None:
            continue

        # 组合变换 W = W_down @ W_up (简化分析)
        # 但维度太大，改用随机投影测试
        # 生成100个随机方向，测量变换后的方向变化
        torch.manual_seed(42)
        random_dirs = torch.randn(100, hidden)
        random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True)

        # 模拟FFN变换: x -> W_down(activation(W_up @ x))
        # 简化: 只看线性部分 W_down @ W_up
        # PyTorch Linear weights: [out_features, in_features]
        # x: [hidden] -> W_up @ x: [intermediate] -> W_down @ y: [hidden]
        # W_up needs transpose: [in, out]
        try:
            if W_up.shape[0] == hidden:
                # W_up is [hidden, intermediate], no transpose needed
                W_combined = W_down.float() @ W_up.float()
            else:
                # W_up is [intermediate, hidden], need transpose
                W_combined = W_down.float() @ W_up.float().T
        except Exception:
            # Fallback: try both orders
            try:
                W_combined = W_down.float() @ W_up.float().T
            except Exception:
                continue

        if W_combined.shape[0] != hidden or W_combined.shape[1] != hidden:
            continue

        # 对每个随机方向计算变换后的角度
        transformed = random_dirs.float() @ W_combined.T  # [100, hidden]
        transformed = transformed / (transformed.norm(dim=-1, keepdim=True) + 1e-8)

        angles = []
        for i in range(100):
            cos = np.clip(float(torch.dot(random_dirs[i].float(), transformed[i])), -1, 1)
            angles.append(float(np.degrees(np.arccos(cos))))

        # SVD of combined weight
        try:
            _, S, Vt = torch.linalg.svd(W_combined.float(), full_matrices=False)
            top_sv_ratio = float(S[0] / S.sum())
            eff_rank = float(S.sum() / S[0])
            sv_decay = float(S[0] / S[min(9, len(S)-1)]) if len(S) > 9 else 0
        except Exception:
            top_sv_ratio = 0
            eff_rank = 0
            sv_decay = 0

        # 变换的范数缩放
        norm_changes = (transformed * (torch.linalg.norm(W_combined.float(), dim=0)[None, :] + 1e-8)).norm(dim=-1)
        avg_norm_change = float(norm_changes.mean())

        results[f"layer{l}"] = {
            "mean_rotation_angle": float(np.mean(angles)),
            "std_rotation_angle": float(np.std(angles)),
            "median_rotation_angle": float(np.median(angles)),
            "top_sv_ratio": top_sv_ratio,
            "effective_rank": eff_rank,
            "sv_decay_top10": sv_decay,
            "avg_norm_change": avg_norm_change,
        }

        print(f"  Layer {l}: rotation={np.mean(angles):.1f}+/-{np.std(angles):.1f} deg, "
              f"top_sv={top_sv_ratio:.3f}, eff_rank={eff_rank:.1f}, "
              f"sv_decay={sv_decay:.1f}")

    # 分析跨层的方向偏好一致性
    # 对每层FFN组合权重的Top-1右奇异向量，计算跨层相关性
    print("\n  Cross-layer direction consistency:")
    layer_top_dirs = {}
    for key, data in results.items():
        l = int(key.replace("layer", ""))
        layer = layers[l]
        mlp = layer.mlp if hasattr(layer, 'mlp') else layer.feed_forward
        if mlp is None:
            continue
        if hasattr(mlp, 'c_fc'):
            W_up = mlp.c_fc.weight.data.cpu().float()
            W_down = mlp.c_proj.weight.data.cpu().float()
        elif hasattr(mlp, 'gate_proj'):
            W_up = mlp.gate_proj.weight.data.cpu().float()
            W_down = mlp.down_proj.weight.data.cpu().float()
        else:
            continue
        if W_up.shape[0] == hidden:
            W_combined = W_down @ W_up
        else:
            W_combined = W_down @ W_up.T
        try:
            _, S, Vt = torch.linalg.svd(W_combined, full_matrices=False)
            layer_top_dirs[l] = Vt[0]
        except Exception:
            pass

    # 计算相邻层的Top-1方向相关性
    adj_corrs = []
    sorted_layers = sorted(layer_top_dirs.keys())
    for i in range(len(sorted_layers)-1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        cos = float(torch.dot(layer_top_dirs[l1], layer_top_dirs[l2]))
        adj_corrs.append(cos)
        print(f"    L{l1}->L{l2}: cos={cos:.4f}")

    if adj_corrs:
        print(f"    Mean adjacent correlation: {np.mean(adj_corrs):.4f}")

    return results


# ============================================================
# 分析5: 因果干预 (零化维度差异方向)
# ============================================================
def analyze_causal_intervention(model, tokenizer, config):
    """
    因果干预实验:
    1. 找到句子对在某层的差异方向
    2. 在后续层中零化/缩小这个差异方向
    3. 观察输出概率的变化
    简化版: 用logit difference作为代理指标
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 5: CAUSAL INTERVENTION (LOGIT DIFF)")
    print("=" * 60)

    results = {}
    hidden = config["hidden_size"]
    device = model.device

    # 选取对比句子对
    test_pairs = [
        ("The cat catches the mouse.", "The mouse is caught by the cat.", "syntax"),
        ("The sky is blue.", "The sky is not blue.", "logic"),
        ("The committee has reached a consensus.", "Everyone agreed on it.", "style"),
        ("The cat purrs softly.", "Justice requires fairness.", "semantic"),
    ]

    for sent_a, sent_b, dim_name in test_pairs:
        print(f"\n  [{dim_name}] A: {sent_a}")
        print(f"  [{dim_name}] B: {sent_b}")

        # 正常前向传播,获取logit difference
        inputs_a = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=64)
        inputs_b = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=64)
        inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
        inputs_b = {k: v.to(device) for k, v in inputs_b.items()}

        with torch.no_grad():
            logits_a = model(**inputs_a).logits[0, -1, :].cpu()
            logits_b = model(**inputs_b).logits[0, -1, :].cpu()

        # Logit difference (top-10 tokens)
        top_k = 10
        top_probs_a = torch.softmax(logits_a, dim=-1)
        top_probs_b = torch.softmax(logits_b, dim=-1)
        top_indices = torch.topk(top_probs_a + top_probs_b, top_k).indices
        logit_diff_normal = float(torch.norm(logits_a[top_indices] - logits_b[top_indices]))

        # JS divergence of output distributions
        p = top_probs_a[top_indices].numpy()
        q = top_probs_b[top_indices].numpy()
        m = 0.5 * (p + q)
        js_normal = 0.5 * np.sum(p * np.log(p / (m + 1e-10) + 1e-10)) + \
                     0.5 * np.sum(q * np.log(q / (m + 1e-10) + 1e-10))

        # 获取残差流差异
        res_a, _, toks_a = capture_all_layers(model, tokenizer, sent_a)
        res_b, _, toks_b = capture_all_layers(model, tokenizer, sent_b)

        # 在中间层的差异方向
        mid_layer = config["num_layers"] // 2
        if mid_layer in res_a and mid_layer in res_b:
            diff_dir = (res_a[mid_layer][0].mean(0) - res_b[mid_layer][0].mean(0))
            diff_dir = diff_dir / (torch.norm(diff_dir) + 1e-8)

            # 计算差异方向上的"能量"在各层的分布
            layer_energy = []
            for l in range(config["num_layers"]):
                if l in res_a and l in res_b:
                    d = res_a[l][0].mean(0) - res_b[l][0].mean(0)
                    proj = float(torch.dot(d, diff_dir))
                    layer_energy.append(proj)

        pair_result = {
            "dimension": dim_name,
            "sentence_a": sent_a,
            "sentence_b": sent_b,
            "logit_diff_normal": logit_diff_normal,
            "js_divergence_normal": js_normal,
            "layer_energy_profile": layer_energy if 'layer_energy' in dir() else [],
            "top_predicted_a": tokenizer.decode([torch.argmax(logits_a).item()]),
            "top_predicted_b": tokenizer.decode([torch.argmax(logits_b).item()]),
        }

        # 逐层logit diff分析
        layer_logit_diffs = []
        for l in range(config["num_layers"]):
            if l not in res_a or l not in res_b:
                continue
            # 计算该层残差流的差异范数
            ra = res_a[l][0].mean(0)
            rb = res_b[l][0].mean(0)
            layer_logit_diffs.append(float(torch.norm(ra - rb)))

        pair_result["layer_residual_diff"] = layer_logit_diffs

        results[f"{dim_name}"] = pair_result

        print(f"    logit_diff={logit_diff_normal:.4f}, JS={js_normal:.4f}")
        print(f"    predict_A='{pair_result['top_predicted_a']}', predict_B='{pair_result['top_predicted_b']}'")
        if layer_energy:
            print(f"    energy_profile: {[f'{e:.2f}' for e in layer_energy[:5]]}...")

    return results


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": timestamp, "phase": 4, "models": {}}

    # ====== GPT-2 ======
    print("=" * 60)
    print("PHASE 4: GPT-2")
    print("=" * 60)
    model_gpt2, tok_gpt2, cfg_gpt2 = load_model("gpt2")

    results["models"]["gpt2"] = {
        "config": cfg_gpt2,
        "subspace_projection": analyze_subspace_projection(model_gpt2, tok_gpt2, cfg_gpt2),
        "token_deviation": analyze_token_deviation(model_gpt2, tok_gpt2, cfg_gpt2),
        "layer_discriminability": analyze_layer_discriminability(model_gpt2, tok_gpt2, cfg_gpt2),
        "ffn_spectrum": analyze_ffn_spectrum(model_gpt2, tok_gpt2, cfg_gpt2),
        "causal_intervention": analyze_causal_intervention(model_gpt2, tok_gpt2, cfg_gpt2),
    }

    del model_gpt2
    torch.cuda.empty_cache()

    # ====== Qwen2.5-0.5B ======
    print("\n" + "=" * 60)
    print("PHASE 4: Qwen2.5-0.5B")
    print("=" * 60)
    try:
        model_qwen, tok_qwen, cfg_qwen = load_model("Qwen/Qwen2.5-0.5B")
        results["models"]["qwen2.5-0.5b"] = {
            "config": cfg_qwen,
            "subspace_projection": analyze_subspace_projection(model_qwen, tok_qwen, cfg_qwen),
            "token_deviation": analyze_token_deviation(model_qwen, tok_qwen, cfg_qwen),
            "ffn_spectrum": analyze_ffn_spectrum(model_qwen, tok_qwen, cfg_qwen),
            "causal_intervention": analyze_causal_intervention(model_qwen, tok_qwen, cfg_qwen),
        }
        del model_qwen
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  Qwen failed: {e}")
        results["models"]["qwen2.5-0.5b"] = {"error": str(e)}

    # Save
    output_path = OUTPUT_DIR / f"phase4_dimension_localization_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
