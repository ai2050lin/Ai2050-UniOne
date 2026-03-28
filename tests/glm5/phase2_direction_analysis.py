"""
Phase 2: 残差流方向分析 & Phase 1 修复
===========================================
修复 Phase 1 的硬伤 + 深入方向分析:
  1. 修复有效秩计算 (token间差异矩阵, 而非token内协方差)
  2. 修复 pairwise attention diff (逐头+对齐+熵方法)
  3. 残差流方向变化分析 (检测语法/逻辑/风格在方向上的差异)
  4. 跨维度方向差异分析 (同一句子在残差流中各维度的贡献)
  5. 验证 Qwen Embedding SVD

使用模型: GPT-2 + Qwen2.5-0.5B
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试句子 (与Phase1一致, 用于对比)
TEST_CASES = {
    "syntax_svo": "The cat catches the mouse.",
    "syntax_passive": "The mouse is caught by the cat.",
    "logic_syllogism": "All birds can fly. Penguins are birds.",
    "logic_negation": "The sky is not blue.",
    "style_formal": "The aforementioned research demonstrates significant findings.",
    "style_casual": "That research shows some really cool stuff.",
    "semantic_animal": "The cat purrs softly on the mat.",
    "semantic_abstract": "Justice requires fairness and equality.",
}


def load_model(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float32, device_map="auto",
        attn_implementation="eager",  # 需要eager才能output_attentions
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
    """获取可hook的层模块列表"""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    return []


def run_forward_capture(model, tokenizer, sentence, max_length=64):
    """运行前向传播并捕获所有层的残差流和注意力"""
    hooks = []
    residual_cache = {}
    attn_cache = {}

    layers = get_layer_modules(model)
    num_layers = len(layers)

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

    model.config.output_attentions = True
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)

    attentions = output.attentions if output.attentions else None
    model.config.output_attentions = False

    for h in hooks:
        h.remove()

    return residual_cache, attentions, token_ids, tokens


# ============================================================
# 修复1: 有效秩 - 使用token间差异矩阵
# ============================================================
def compute_effective_rank_fixed(residual):
    """
    修复版有效秩计算:
    - 原方法: token内协方差矩阵的特征值 → 抹掉了方向差异
    - 新方法: token间差异矩阵的SVD → 保留了方向差异
    """
    # residual: [seq_len, hidden_size]
    if residual.shape[0] < 3:
        return 0.0, 0.0, 0.0

    # 方法A: Token间差异矩阵
    # 计算所有token对的差异向量
    diff_matrix = []
    for i in range(residual.shape[0]):
        for j in range(i + 1, residual.shape[0]):
            diff_matrix.append(residual[i] - residual[j])
    diff_matrix = torch.stack(diff_matrix)  # [N, hidden_size]

    # 对差异矩阵做SVD
    try:
        _, S, _ = torch.linalg.svd(diff_matrix.float(), full_matrices=False)
        S = S[S > 1e-6]
        eff_rank = float(torch.sum(S) / torch.max(S)) if len(S) > 0 else 0.0
        eff_dim = float(torch.sum(S / S.sum() > 0.01)) if len(S) > 0 else 0.0
        top_sv_ratio = float(S[0] / S.sum()) if len(S) > 0 else 0.0
    except Exception:
        eff_rank = 0.0
        eff_dim = 0.0
        top_sv_ratio = 0.0

    # 方法B: 标准有效秩 (基于核范数比F范数) 作为对比
    cov = (residual - residual.mean(0)).T @ (residual - residual.mean(0))
    eigenvalues = torch.linalg.eigvalsh(cov.float())
    eigenvalues = eigenvalues[eigenvalues > 1e-6]
    if len(eigenvalues) > 0:
        nuclear_norm = float(torch.sum(eigenvalues))
        spectral_norm = float(torch.max(eigenvalues))
        standard_eff_rank = nuclear_norm / spectral_norm
    else:
        standard_eff_rank = 0.0

    return eff_rank, eff_dim, standard_eff_rank


# ============================================================
# 修复2: Pairwise Attention Diff (逐头分析 + 信息论度量)
# ============================================================
def compute_attention_difference(att_a, att_b, tokens_a, tokens_b):
    """
    修复版注意力差异计算:
    - 原方法: 简单取绝对差均值 → token不对齐时容易为0
    - 新方法: 逐头分析 + 对齐 + KL散度 + 逐功能指标
    """
    if att_a is None or att_b is None:
        return None

    results = {}
    # att: [1, heads, seq, seq]
    # att是tuple of (batch, heads, seq, seq)，取第一层
    a = att_a[0].cpu().numpy()[0]  # [heads, seq_a, seq_a]
    b = att_b[0].cpu().numpy()[0]  # [heads, seq_b, seq_b]

    num_heads = min(a.shape[0], b.shape[0])
    min_seq = min(a.shape[1], b.shape[1])

    head_diffs = []
    head_kl_divs = []
    head_js_divs = []

    for h in range(num_heads):
        a_h = a[h, :min_seq, :min_seq]
        b_h = b[h, :min_seq, :min_seq]

        # 确保每行是概率分布 (softmax后再比较)
        a_h_softmax = np.exp(a_h - a_h.max(axis=-1, keepdims=True))
        a_h_softmax = a_h_softmax / a_h_softmax.sum(axis=-1, keepdims=True)
        b_h_softmax = np.exp(b_h - b_h.max(axis=-1, keepdims=True))
        b_h_softmax = b_h_softmax / b_h_softmax.sum(axis=-1, keepdims=True)

        # L1距离
        l1 = np.abs(a_h_softmax - b_h_softmax).mean()
        head_diffs.append(l1)

        # KL散度 (行平均)
        kl_rows = []
        for row in range(min_seq):
            p = a_h_softmax[row]
            q = b_h_softmax[row]
            # 避免log(0)
            kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            kl_rows.append(kl)
        head_kl_divs.append(np.mean(kl_rows))

        # JS散度
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / (m + 1e-10) + 1e-10)) + \
             0.5 * np.sum(q * np.log(q / (m + 1e-10) + 1e-10))
        head_js_divs.append(js)

    # 功能性注意力差异
    # 比较prev-token pattern (对角线-1)
    diag_diffs = []
    for h in range(num_heads):
        a_h = a[h, :min_seq, :min_seq]
        b_h = b[h, :min_seq, :min_seq]
        if min_seq > 2:
            a_diag = np.diag(a_h, k=-1)[:-1]
            b_diag = np.diag(b_h, k=-1)[:-1]
            diag_diffs.append(float(np.abs(a_diag - b_diag).mean()))

    # 远程依赖差异
    remote_diffs = []
    for h in range(num_heads):
        a_h = a[h, :min_seq, :min_seq]
        b_h = b[h, :min_seq, :min_seq]
        if min_seq > 4:
            # 比较距离>3的位置的注意力总和
            a_remote = []
            b_remote = []
            for i in range(min_seq):
                for j in range(max(0, i-3), min(min_seq, i+4)):
                    pass
                a_remote_row = a_h[i, :max(0,i-3)].sum() + a_h[i, min(min_seq,i+4):].sum()
                b_remote_row = b_h[i, :max(0,i-3)].sum() + b_h[i, min(min_seq,i+4):].sum()
                a_remote.append(a_remote_row)
                b_remote.append(b_remote_row)
            remote_diffs.append(float(np.abs(np.array(a_remote) - np.array(b_remote)).mean()))

    results = {
        "num_heads": num_heads,
        "aligned_seq_len": min_seq,
        "mean_l1_diff": float(np.mean(head_diffs)),
        "max_head_l1": float(np.max(head_diffs)),
        "max_l1_head": int(np.argmax(head_diffs)),
        "mean_kl_divergence": float(np.mean(head_kl_divs)),
        "mean_js_divergence": float(np.mean(head_js_divs)),
        "max_js_head": int(np.argmax(head_js_divs)),
        "per_head_l1": [float(x) for x in head_diffs],
        "per_head_kl": [float(x) for x in head_kl_divs],
        "mean_diag_diff": float(np.mean(diag_diffs)) if diag_diffs else 0.0,
        "mean_remote_diff": float(np.mean(remote_diffs)) if remote_diffs else 0.0,
    }

    return results


# ============================================================
# 新分析3: 残差流方向变化
# ============================================================
def analyze_residual_directions(model, tokenizer, config):
    """分析残差流方向变化 - 关键新分析"""
    print("\n" + "=" * 60)
    print("RESIDUAL DIRECTION ANALYSIS")
    print("=" * 60)

    results = {}
    all_residuals = {}

    # 对每个测试句子捕获残差流
    for case_name, sentence in TEST_CASES.items():
        residual_cache, _, token_ids, tokens = run_forward_capture(
            model, tokenizer, sentence
        )
        all_residuals[case_name] = residual_cache
        results[case_name] = {
            "sentence": sentence,
            "num_tokens": len(tokens),
            "tokens": tokens[:10],  # 前10个token
        }

    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]

    # --- 3a. 逐层方向变化分析 ---
    # 对每个句子, 计算相邻层之间残差流方向的变化角度
    for case_name, residual_cache in all_residuals.items():
        layer_directions = []
        for l in range(num_layers):
            if l not in residual_cache:
                continue
            residual = residual_cache[l][0]  # [seq_len, hidden_size]
            # 每个token的残差流方向 (归一化)
            norms = torch.norm(residual, dim=-1, keepdim=True)
            directions = residual / (norms + 1e-8)  # [seq_len, hidden_size]
            # 层的方向重心 (平均方向)
            mean_dir = directions.mean(dim=0)  # [hidden_size]
            mean_dir = mean_dir / (torch.norm(mean_dir) + 1e-8)
            layer_directions.append(mean_dir.numpy())

        # 计算相邻层方向的角度变化
        angle_changes = []
        for i in range(1, len(layer_directions)):
            cos_angle = np.clip(np.dot(layer_directions[i-1], layer_directions[i]), -1, 1)
            angle = float(np.arccos(cos_angle))
            angle_changes.append({
                "from_layer": i - 1,
                "to_layer": i,
                "angle_deg": float(np.degrees(angle)),
                "cosine_similarity": float(cos_angle),
            })

        results[case_name]["direction_angle_changes"] = angle_changes
        if angle_changes:
            avg_angle = np.mean([a["angle_deg"] for a in angle_changes])
            print(f"  [{case_name}] avg direction change: {avg_angle:.2f}°")

    # --- 3b. 跨维度方向差异: 语法/逻辑/风格/语义 ---
    dimension_pairs = [
        ("syntax_svo", "syntax_passive", "syntax_active_vs_passive"),
        ("logic_syllogism", "logic_negation", "logic_vs_negation"),
        ("style_formal", "style_casual", "style_formal_vs_casual"),
        ("semantic_animal", "semantic_abstract", "semantic_animal_vs_abstract"),
    ]

    cross_dimension_results = {}
    for k1, k2, pair_name in dimension_pairs:
        if k1 not in all_residuals or k2 not in all_residuals:
            continue

        layer_angle_diffs = []
        layer_cos_diffs = []

        for l in range(num_layers):
            if l not in all_residuals[k1] or l not in all_residuals[k2]:
                continue

            r1 = all_residuals[k1][l][0]  # [seq1, hidden]
            r2 = all_residuals[k2][l][0]  # [seq2, hidden]

            # 对齐: 取较短序列
            min_seq = min(r1.shape[0], r2.shape[0])
            r1 = r1[:min_seq]
            r2 = r2[:min_seq]

            # 每个token对的方向差异
            token_cos_sims = []
            for t in range(min_seq):
                n1 = torch.norm(r1[t])
                n2 = torch.norm(r2[t])
                if n1 > 1e-6 and n2 > 1e-6:
                    cos = float(torch.dot(r1[t], r2[t]) / (n1 * n2))
                    token_cos_sims.append(cos)

            if token_cos_sims:
                mean_cos = float(np.mean(token_cos_sims))
                mean_angle = float(np.degrees(np.arccos(np.clip(mean_cos, -1, 1))))
                layer_cos_diffs.append(mean_cos)
                layer_angle_diffs.append(mean_angle)

        if layer_angle_diffs:
            cross_dimension_results[pair_name] = {
                "case1": k1,
                "case2": k2,
                "layer_cosine_similarities": layer_cos_diffs,
                "layer_angle_differences": layer_angle_diffs,
                "min_angle_layer": int(np.argmin(layer_angle_diffs)),
                "max_angle_layer": int(np.argmax(layer_angle_diffs)),
                "trend": "diverging" if layer_angle_diffs[-1] > layer_angle_diffs[0] else "converging",
            }
            print(f"  [{pair_name}] angle: {layer_angle_diffs[0]:.1f}° -> {layer_angle_diffs[-1]:.1f}° "
                  f"({cross_dimension_results[pair_name]['trend']})")

    results["cross_dimension_direction"] = cross_dimension_results

    # --- 3c. 方向主导成分分析 ---
    # 在最后一层, 分析每个句子残差流的主导方向
    print("\n  --- Direction PCA at last layer ---")
    for case_name, residual_cache in all_residuals.items():
        last_layer = max(residual_cache.keys())
        residual = residual_cache[last_layer][0]  # [seq, hidden]

        # Token间的差异矩阵
        diffs = []
        for i in range(residual.shape[0]):
            for j in range(i+1, residual.shape[0]):
                diffs.append((residual[i] - residual[j]).numpy())
        diffs = np.array(diffs)

        if len(diffs) > 1:
            _, S, Vt = np.linalg.svd(diffs, full_matrices=False)
            top3_var = (S[:3]**2).sum() / (S**2).sum()
            results[case_name]["last_layer_direction_variance_top3"] = float(top3_var)
            results[case_name]["last_layer_effective_rank"] = float(S.sum() / S[0]) if S[0] > 0 else 0
            print(f"  [{case_name}] top3_dir_var={top3_var:.4f}, eff_rank={S.sum()/S[0]:.1f}")

    return results


# ============================================================
# 修复4: Qwen Embedding SVD (加强版)
# ============================================================
def analyze_embedding_fixed(model, tokenizer, config):
    """修复版Embedding分析 - 特别针对Qwen"""
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS (FIXED)")
    print("=" * 60)

    results = {}

    # 获取embedding权重
    embed_module = model.get_input_embeddings()
    W_embed = embed_module.weight.data.cpu().numpy()
    results["shape"] = list(W_embed.shape)

    # SVD
    try:
        U, S, Vt = np.linalg.svd(W_embed, full_matrices=False)
        results["singular_values_top20"] = S[:20].tolist()
        results["cumvar_top20"] = (np.cumsum(S[:20]**2) / np.sum(S**2)).tolist()
        results["total_variance"] = float(np.sum(S**2))

        # 检查数值稳定性
        results["max_sv"] = float(S[0])
        results["min_sv_top100"] = float(S[min(99, len(S)-1)])
        results["condition_number_top100"] = float(S[0] / S[min(99, len(S)-1)])

        print(f"  Shape: {W_embed.shape}")
        print(f"  Top 5 SV: {[f'{s:.2f}' for s in S[:5]]}")
        print(f"  CumVar(top5): {results['cumvar_top20'][4]:.4f}")
        print(f"  Condition number (top100): {results['condition_number_top100']:.2f}")

        # 主成分分析
        top_components = []
        for i in range(min(5, len(S))):
            direction = Vt[i]
            projections = W_embed @ direction
            top_idx = np.argsort(projections)[-5:][::-1]
            bot_idx = np.argsort(projections)[:5]
            top_tokens = [tokenizer.decode([idx]).strip() for idx in top_idx]
            bot_tokens = [tokenizer.decode([idx]).strip() for idx in bot_idx]
            top_components.append({
                "pc": i,
                "sv": float(S[i]),
                "var_ratio": float(S[i]**2 / np.sum(S**2)),
                "top_tokens": top_tokens,
                "bot_tokens": bot_tokens,
            })
        results["top_components"] = top_components

        for c in top_components[:3]:
            print(f"  PC{c['pc']} (var={c['var_ratio']*100:.1f}%): "
                  f"top={[t.encode('ascii','replace').decode() for t in c['top_tokens'][:3]]}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ERROR in SVD: {e}")

    # Qwen可能有RMSNorm在embedding后, 检查
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Qwen风格 - embedding后可能接RMSNorm
        if hasattr(model.model, 'layers'):
            print("  Note: Qwen architecture detected - embedding followed by RMSNorm")

    # Embedding空间统计
    row_norms = np.linalg.norm(W_embed, axis=1)
    results["mean_embed_norm"] = float(np.mean(row_norms))
    results["std_embed_norm"] = float(np.std(row_norms))
    results["min_embed_norm"] = float(np.min(row_norms))
    results["max_embed_norm"] = float(np.max(row_norms))

    # 检查是否有零向量
    zero_count = np.sum(row_norms < 1e-6)
    results["zero_vector_count"] = int(zero_count)

    print(f"  Embed norm: mean={np.mean(row_norms):.4f}, std={np.std(row_norms):.4f}")
    print(f"  Zero vectors: {zero_count}")

    return results


# ============================================================
# 新分析: Token级方向轨迹
# ============================================================
def analyze_token_trajectories(model, tokenizer, config):
    """追踪单个token在残差流中的完整轨迹"""
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL TRAJECTORY ANALYSIS")
    print("=" * 60)

    # 选取有结构差异的句子对
    sentence_pairs = [
        {
            "name": "active_vs_passive",
            "sent_a": "The cat catches the mouse.",
            "sent_b": "The mouse is caught by the cat.",
            "focus_word": "cat",  # 主语位置变化
        },
        {
            "name": "affirm_vs_negate",
            "sent_a": "The sky is blue.",
            "sent_b": "The sky is not blue.",
            "focus_word": "sky",
        },
        {
            "name": "formal_vs_casual",
            "sent_a": "The committee has reached a consensus regarding this matter.",
            "sent_b": "Everyone agreed on what to do about it.",
            "focus_word": None,  # 位置对齐方式不同
        },
    ]

    results = {}

    for pair_info in sentence_pairs:
        name = pair_info["name"]
        sent_a = pair_info["sent_a"]
        sent_b = pair_info["sent_b"]

        print(f"\n  --- {name} ---")
        print(f"    A: {sent_a}")
        print(f"    B: {sent_b}")

        res_a, _, ids_a, toks_a = run_forward_capture(model, tokenizer, sent_a)
        res_b, _, ids_b, toks_b = run_forward_capture(model, tokenizer, sent_b)

        # 对齐token: 用BOS对齐
        min_seq = min(len(toks_a), len(toks_b))

        # 计算每个位置的逐层余弦相似度
        position_layer_cosine = []
        for t in range(min_seq):
            layer_cosines = []
            layer_angles = []
            for l in sorted(res_a.keys()):
                if l not in res_b:
                    continue
                v_a = res_a[l][0, t]
                v_b = res_b[l][0, t]
                n_a = torch.norm(v_a)
                n_b = torch.norm(v_b)
                if n_a > 1e-6 and n_b > 1e-6:
                    cos = float(torch.dot(v_a, v_b) / (n_a * n_b))
                    angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                    layer_cosines.append(cos)
                    layer_angles.append(angle)

            position_layer_cosine.append({
                "token_a": toks_a[t].encode("ascii", "replace").decode() if t < len(toks_a) else "?",
                "token_b": toks_b[t].encode("ascii", "replace").decode() if t < len(toks_b) else "?",
                "layer_cosines": layer_cosines,
                "layer_angles": layer_angles,
                "first_layer_cos": layer_cosines[0] if layer_cosines else 0,
                "last_layer_cos": layer_cosines[-1] if layer_cosines else 0,
                "cos_divergence": (layer_cosines[-1] - layer_cosines[0]) if len(layer_cosines) >= 2 else 0,
            })

        # 汇总
        pair_result = {
            "name": name,
            "sentence_a": sent_a,
            "sentence_b": sent_b,
            "aligned_length": min_seq,
            "position_analysis": position_layer_cosine,
        }

        # 找出分歧最大的位置
        divergences = [p["cos_divergence"] for p in position_layer_cosine]
        if divergences:
            max_div_idx = int(np.argmax(np.abs(divergences)))
            pair_result["max_divergence_position"] = max_div_idx
            pair_result["max_divergence_token"] = position_layer_cosine[max_div_idx]["token_a"]
            print(f"    Max divergence at pos {max_div_idx}: "
                  f"'{position_layer_cosine[max_div_idx]['token_a']}' vs "
                  f"'{position_layer_cosine[max_div_idx]['token_b']}' "
                  f"(cos: {position_layer_cosine[max_div_idx]['first_layer_cos']:.3f} -> "
                  f"{position_layer_cosine[max_div_idx]['last_layer_cos']:.3f})")

        results[name] = pair_result

    return results


# ============================================================
# 新分析: 方向偏转累积效应
# ============================================================
def analyze_direction_accumulation(model, tokenizer, config):
    """
    分析FFN对残差流方向的偏转是否累积
    核心假说: 每层FFN做微小方向偏转, 多层累积后产生大方向变化
    """
    print("\n" + "=" * 60)
    print("DIRECTION ACCUMULATION ANALYSIS")
    print("=" * 60)

    results = {}

    for case_name, sentence in TEST_CASES.items():
        residual_cache, _, _, _ = run_forward_capture(model, tokenizer, sentence)

        layer_data = []
        prev_direction = None

        for l in sorted(residual_cache.keys()):
            residual = residual_cache[l][0]  # [seq, hidden]
            # 使用所有token的均值方向
            mean_vec = residual.mean(dim=0)
            direction = mean_vec / (torch.norm(mean_vec) + 1e-8)

            if prev_direction is not None:
                # 单层偏转角度
                cos_change = float(torch.dot(direction, prev_direction))
                cos_change = np.clip(cos_change, -1, 1)
                single_step_angle = float(np.degrees(np.arccos(cos_change)))
            else:
                single_step_angle = 0.0

            prev_direction = direction
            layer_data.append({
                "layer": l,
                "step_angle": single_step_angle,
                "mean_norm": float(torch.norm(residual, dim=-1).mean()),
            })

        # 计算累积偏转角度
        cumulative_angle = 0.0
        for ld in layer_data:
            cumulative_angle += ld["step_angle"]
            ld["cumulative_angle"] = cumulative_angle

        results[case_name] = {
            "sentence": sentence,
            "layer_data": layer_data,
            "total_accumulated_angle": float(cumulative_angle),
            "mean_step_angle": float(np.mean([ld["step_angle"] for ld in layer_data[1:]])) if len(layer_data) > 1 else 0,
        }

        # 打印
        angles = [ld["step_angle"] for ld in layer_data[1:]]
        if angles:
            print(f"  [{case_name}] mean_step={np.mean(angles):.2f}°, "
                  f"total_accum={cumulative_angle:.1f}°")

    return results


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": timestamp, "phase": 2, "models": {}}

    # ====== GPT-2 ======
    print("=" * 60)
    print("PHASE 2: GPT-2")
    print("=" * 60)
    model_gpt2, tok_gpt2, cfg_gpt2 = load_model("gpt2")

    results["models"]["gpt2"] = {
        "config": cfg_gpt2,
        "embedding_fixed": analyze_embedding_fixed(model_gpt2, tok_gpt2, cfg_gpt2),
        "residual_directions": analyze_residual_directions(model_gpt2, tok_gpt2, cfg_gpt2),
        "token_trajectories": analyze_token_trajectories(model_gpt2, tok_gpt2, cfg_gpt2),
        "direction_accumulation": analyze_direction_accumulation(model_gpt2, tok_gpt2, cfg_gpt2),
    }

    # 修复版有效秩 (对所有测试句子)
    print("\n" + "=" * 60)
    print("FIXED EFFECTIVE RANK: GPT-2")
    print("=" * 60)
    fixed_rank_results = {}
    for case_name, sentence in TEST_CASES.items():
        residual_cache, _, _, _ = run_forward_capture(model_gpt2, tok_gpt2, sentence)
        layer_rank_data = []
        for l in sorted(residual_cache.keys()):
            eff_rank, eff_dim, std_rank = compute_effective_rank_fixed(residual_cache[l][0])
            layer_rank_data.append({
                "layer": l,
                "effective_rank_diff_matrix": eff_rank,
                "effective_dim_diff_matrix": eff_dim,
                "standard_effective_rank": std_rank,
            })
        fixed_rank_results[case_name] = {
            "sentence": sentence,
            "layer_ranks": layer_rank_data,
        }
        if layer_rank_data:
            print(f"  [{case_name}] diff_rank: {layer_rank_data[0]['effective_rank_diff_matrix']:.1f} "
                  f"-> {layer_rank_data[-1]['effective_rank_diff_matrix']:.1f} "
                  f"(std_rank: {layer_rank_data[0]['standard_effective_rank']:.1f} "
                  f"-> {layer_rank_data[-1]['standard_effective_rank']:.1f})")

    results["models"]["gpt2"]["fixed_effective_rank"] = fixed_rank_results

    # 修复版pairwise attention diff
    print("\n" + "=" * 60)
    print("FIXED PAIRWISE ATTENTION DIFF: GPT-2")
    print("=" * 60)
    pair_sentences = [
        ("The tall building stands in the city center.",
         "Building tall stands in the city center.",
         "syntax_scrambled"),
        ("All birds can fly. Penguins are birds. So penguins can fly.",
         "All birds can fly. Penguins are birds. So penguins cannot fly.",
         "logic_valid_vs_invalid"),
        ("The committee has reached a consensus regarding this matter.",
         "Everyone agreed on what to do about it.",
         "style_formal_vs_casual"),
    ]

    fixed_attn_results = {}
    for sent_a, sent_b, pair_name in pair_sentences:
        res_a, attn_a, _, _ = run_forward_capture(model_gpt2, tok_gpt2, sent_a)
        res_b, attn_b, _, _ = run_forward_capture(model_gpt2, tok_gpt2, sent_b)

        diff = compute_attention_difference(attn_a, attn_b, None, None)
        if diff:
            fixed_attn_results[pair_name] = {
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                **diff,
            }
            print(f"  [{pair_name}] L1={diff['mean_l1_diff']:.6f}, "
                  f"KL={diff['mean_kl_divergence']:.6f}, "
                  f"JS={diff['mean_js_divergence']:.6f}, "
                  f"max_head={diff['max_js_head']}")

    results["models"]["gpt2"]["fixed_pairwise_attention"] = fixed_attn_results

    del model_gpt2
    torch.cuda.empty_cache()

    # ====== Qwen2.5-0.5B ======
    print("\n" + "=" * 60)
    print("PHASE 2: Qwen2.5-0.5B")
    print("=" * 60)
    try:
        model_qwen, tok_qwen, cfg_qwen = load_model("Qwen/Qwen2.5-0.5B")

        results["models"]["qwen2.5-0.5b"] = {
            "config": cfg_qwen,
            "embedding_fixed": analyze_embedding_fixed(model_qwen, tok_qwen, cfg_qwen),
            "residual_directions": analyze_residual_directions(model_qwen, tok_qwen, cfg_qwen),
            "direction_accumulation": analyze_direction_accumulation(model_qwen, tok_qwen, cfg_qwen),
        }

        # 修复版有效秩
        print("\n" + "=" * 60)
        print("FIXED EFFECTIVE RANK: Qwen2.5-0.5B")
        print("=" * 60)
        fixed_rank_qwen = {}
        for case_name, sentence in TEST_CASES.items():
            residual_cache, _, _, _ = run_forward_capture(model_qwen, tok_qwen, sentence)
            layer_rank_data = []
            for l in sorted(residual_cache.keys()):
                eff_rank, eff_dim, std_rank = compute_effective_rank_fixed(residual_cache[l][0])
                layer_rank_data.append({
                    "layer": l,
                    "effective_rank_diff_matrix": eff_rank,
                    "effective_dim_diff_matrix": eff_dim,
                    "standard_effective_rank": std_rank,
                })
            fixed_rank_qwen[case_name] = {
                "sentence": sentence,
                "layer_ranks": layer_rank_data,
            }
            if layer_rank_data:
                print(f"  [{case_name}] diff_rank: {layer_rank_data[0]['effective_rank_diff_matrix']:.1f} "
                      f"-> {layer_rank_data[-1]['effective_rank_diff_matrix']:.1f}")

        results["models"]["qwen2.5-0.5b"]["fixed_effective_rank"] = fixed_rank_qwen

        del model_qwen
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  Qwen failed: {e}")
        results["models"]["qwen2.5-0.5b"] = {"error": str(e)}

    # Save
    output_path = OUTPUT_DIR / f"phase2_direction_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
