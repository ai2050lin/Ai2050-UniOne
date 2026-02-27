# -*- coding: utf-8 -*-
"""
Phase XXXVI 拼图 #5: GPT-2 权重矩阵低秩近似
=============================================
核心问题: DNN 学到的变换能否用极少数"基本算子"来表示?
- 权重矩阵的奇异值衰减有多快?
- 秩-r 近似保留多少信息?
- 这些"基本算子"到底在做什么变换?

如果权重本质低秩,大脑的数学结构就可以用少数算子组合来表达。
"""

import torch
import numpy as np
import json
import os
import sys

def main():
    print("=" * 70)
    print("Phase XXXVI 拼图 #5: GPT-2 权重矩阵低秩分析")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")

    print("\n[1/4] 加载 GPT-2 Small...")
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("错误: 请安装 transformer_lens")
        sys.exit(1)

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    # ==========================================
    # 测量 1: MLP 权重矩阵的奇异值谱
    # ==========================================
    print("\n[2/4] 分析 MLP 权重矩阵的奇异值谱...")

    mlp_svd_results = []

    for l in range(n_layers):
        # W_in: [d_model, d_mlp] = [768, 3072]
        W_in = model.blocks[l].mlp.W_in.detach().cpu().numpy()
        # W_out: [d_mlp, d_model] = [3072, 768]
        W_out = model.blocks[l].mlp.W_out.detach().cpu().numpy()

        U_in, S_in, Vt_in = np.linalg.svd(W_in, full_matrices=False)
        U_out, S_out, Vt_out = np.linalg.svd(W_out, full_matrices=False)

        # 计算不同秩的近似保留率
        var_in = (S_in ** 2) / (S_in ** 2).sum()
        var_out = (S_out ** 2) / (S_out ** 2).sum()
        cumvar_in = np.cumsum(var_in)
        cumvar_out = np.cumsum(var_out)

        # 找到保留 90%, 95%, 99% 信息需要的秩
        rank_90_in = int(np.argmax(cumvar_in >= 0.90)) + 1
        rank_95_in = int(np.argmax(cumvar_in >= 0.95)) + 1
        rank_99_in = int(np.argmax(cumvar_in >= 0.99)) + 1

        rank_90_out = int(np.argmax(cumvar_out >= 0.90)) + 1
        rank_95_out = int(np.argmax(cumvar_out >= 0.95)) + 1
        rank_99_out = int(np.argmax(cumvar_out >= 0.99)) + 1

        layer_result = {
            "layer": l,
            "W_in_shape": list(W_in.shape),
            "W_out_shape": list(W_out.shape),
            "W_in_top10_sv": S_in[:10].tolist(),
            "W_out_top10_sv": S_out[:10].tolist(),
            "W_in_rank_90": rank_90_in,
            "W_in_rank_95": rank_95_in,
            "W_in_rank_99": rank_99_in,
            "W_out_rank_90": rank_90_out,
            "W_out_rank_95": rank_95_out,
            "W_out_rank_99": rank_99_out,
            "W_in_sv_ratio_1_10": float(S_in[0] / S_in[9]) if S_in[9] > 0 else float('inf'),
            "W_out_sv_ratio_1_10": float(S_out[0] / S_out[9]) if S_out[9] > 0 else float('inf'),
        }
        mlp_svd_results.append(layer_result)

        if l % 3 == 0 or l == n_layers - 1:
            print(f"  L{l:>2} W_in:  90%={rank_90_in:>3}, 95%={rank_95_in:>3}, 99%={rank_99_in:>3} / {min(W_in.shape)} | "
                  f"SV1/SV10={S_in[0]/S_in[9]:.1f}x")
            print(f"  L{l:>2} W_out: 90%={rank_90_out:>3}, 95%={rank_95_out:>3}, 99%={rank_99_out:>3} / {min(W_out.shape)} | "
                  f"SV1/SV10={S_out[0]/S_out[9]:.1f}x")

    # ==========================================
    # 测量 2: Attention QKV 权重矩阵分析
    # ==========================================
    print("\n[3/4] 分析 Attention QKV 权重矩阵...")

    attn_svd_results = []

    for l in range(n_layers):
        # W_Q, W_K, W_V: [n_heads, d_model, d_head] = [12, 768, 64]
        W_Q = model.blocks[l].attn.W_Q.detach().cpu().numpy()  # [n_heads, d_model, d_head]
        W_K = model.blocks[l].attn.W_K.detach().cpu().numpy()
        W_V = model.blocks[l].attn.W_V.detach().cpu().numpy()

        # 对每个 Head 分析 W_Q 的秩
        head_ranks = []
        for h in range(n_heads):
            wq = W_Q[h]  # [d_model, d_head] = [768, 64]
            U, S, Vt = np.linalg.svd(wq, full_matrices=False)
            var = (S ** 2) / (S ** 2).sum()
            cumvar = np.cumsum(var)
            rank_95 = int(np.argmax(cumvar >= 0.95)) + 1
            head_ranks.append(rank_95)

        # 全连接形式的 QK 乘积矩阵: W_Q @ W_K^T = [d_model, d_model]
        # 这才是 Attention 真正的"关联算子"
        W_QK_combined = []
        for h in range(n_heads):
            wqk = W_Q[h] @ W_K[h].T  # [d_model, d_model]
            U, S, Vt = np.linalg.svd(wqk, full_matrices=False)
            var = (S ** 2) / (S ** 2).sum()
            cumvar = np.cumsum(var)
            rank_95 = int(np.argmax(cumvar >= 0.95)) + 1
            W_QK_combined.append({
                "head": h,
                "rank_95": rank_95,
                "top3_sv": S[:3].tolist(),
                "sv_ratio_1_5": float(S[0] / S[4]) if len(S) > 4 and S[4] > 0 else float('inf'),
            })

        attn_result = {
            "layer": l,
            "W_Q_head_ranks_95": head_ranks,
            "W_QK_combined": W_QK_combined,
            "avg_QK_rank_95": float(np.mean([x["rank_95"] for x in W_QK_combined])),
        }
        attn_svd_results.append(attn_result)

        if l % 3 == 0 or l == n_layers - 1:
            avg_qk = attn_result["avg_QK_rank_95"]
            print(f"  L{l:>2} W_QK 关联算子平均秩(95%): {avg_qk:.1f} / {d_model}")

    # ==========================================
    # 测量 3: 低秩近似后的输出精度
    # ==========================================
    print("\n[4/4] 测试低秩近似后的模型输出保真度...")

    test_text = "The theory of general relativity describes gravity as a geometric property"
    tokens = model.to_tokens(test_text)

    # 原始输出
    with torch.no_grad():
        original_logits = model(tokens).cpu()

    # 对 MLP W_in 做不同秩的近似,观察输出变化
    rank_tests = [16, 32, 64, 128, 256, 512]
    fidelity_results = []

    for test_rank in rank_tests:
        # 临时替换所有层的 W_in
        original_weights = []
        for l in range(n_layers):
            W = model.blocks[l].mlp.W_in.detach().cpu().numpy()
            original_weights.append(W.copy())
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            # 秩-r 近似
            W_approx = U[:, :test_rank] @ np.diag(S[:test_rank]) @ Vt[:test_rank, :]
            model.blocks[l].mlp.W_in.data = torch.tensor(W_approx, dtype=torch.float32).to(device)

        with torch.no_grad():
            approx_logits = model(tokens).cpu()

        # 恢复原始权重
        for l in range(n_layers):
            model.blocks[l].mlp.W_in.data = torch.tensor(original_weights[l], dtype=torch.float32).to(device)

        # 计算保真度指标
        cos_sim = torch.nn.functional.cosine_similarity(
            original_logits.flatten().unsqueeze(0),
            approx_logits.flatten().unsqueeze(0)
        ).item()

        # Top-1 预测保持率
        orig_preds = original_logits[0].argmax(dim=-1)
        approx_preds = approx_logits[0].argmax(dim=-1)
        top1_match = (orig_preds == approx_preds).float().mean().item()

        result = {
            "rank": test_rank,
            "rank_ratio": f"{test_rank}/{min(d_model, d_mlp)}",
            "cosine_similarity": float(cos_sim),
            "top1_prediction_match": float(top1_match),
        }
        fidelity_results.append(result)
        print(f"  Rank {test_rank:>3} ({test_rank/min(d_model,d_mlp)*100:>5.1f}%) | "
              f"Cosine={cos_sim:.6f} | Top-1={top1_match*100:.1f}%")

    # ==========================================
    # 报告
    # ==========================================
    os.makedirs("tempdata", exist_ok=True)

    # 汇总
    avg_mlp_rank95 = float(np.mean([r["W_in_rank_95"] for r in mlp_svd_results]))
    avg_attn_qk_rank = float(np.mean([r["avg_QK_rank_95"] for r in attn_svd_results]))

    report = {
        "experiment": "Phase XXXVI E4: GPT-2 权重矩阵低秩分析",
        "model": "gpt2-small",
        "mlp_svd_results": mlp_svd_results,
        "attn_svd_results": attn_svd_results,
        "fidelity_results": fidelity_results,
        "summary": {
            "avg_mlp_W_in_rank_95": avg_mlp_rank95,
            "avg_mlp_W_in_rank_95_ratio": f"{avg_mlp_rank95:.0f}/{min(d_model,d_mlp)}",
            "avg_attn_QK_rank_95": avg_attn_qk_rank,
        }
    }

    report_path = "tempdata/exp_weight_lowrank_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[完成] 报告: {report_path}")
    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    print(f"1. MLP W_in 平均 95% 秩: {avg_mlp_rank95:.0f} / {min(d_model,d_mlp)}"
          f" ({avg_mlp_rank95/min(d_model,d_mlp)*100:.1f}%)")
    print(f"2. Attention QK 关联算子平均 95% 秩: {avg_attn_qk_rank:.0f} / {d_model}")

    # 找到保真度达到 99% 的最小秩
    for r in fidelity_results:
        if r["top1_prediction_match"] >= 0.99:
            print(f"3. Top-1 预测 99%+ 保持率所需最小秩: {r['rank']} ({r['rank']/min(d_model,d_mlp)*100:.1f}%)")
            break
    else:
        print(f"3. 最高测试秩 {rank_tests[-1]} 时 Top-1 保持率: {fidelity_results[-1]['top1_prediction_match']*100:.1f}%")

if __name__ == "__main__":
    main()
