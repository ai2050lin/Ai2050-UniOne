# -*- coding: utf-8 -*-
"""
Phase XXXVI 关键实验: 从空白 Tensor 自发涌现稀疏字典结构
=========================================================
核心问题: 不用 BP,仅靠"侧抑制 + 预测误差"能否自发涌现
          与 GPT-2 MLP 相同的稀疏字典结构?

实验设计:
  1. 创建一个随机初始化的 [d_input, d_hidden] 权重矩阵
  2. 用合成数据流冲刷它,但不用 BP
  3. 仅使用三种局部规则: 
     a) Hebbian: 同时激活的神经元增强连接
     b) 侧抑制: 同层神经元竞争性压制(Sanger Rule)
     c) 预测误差: 高层->低层的预测误差信号局部修正权重
  4. 观察 W 是否自发涌现:
     - 稀疏激活 (峰度 > 0)
     - 专家化 (不同输入激活不同神经元子集)
     - 低秩结构 (少数奇异值主导)

如果成功:  证明大脑不需要 BP 也能形成与 DNN 等价的编码结构
如果失败:  说明预测误差的具体公式需要修正

使用 GPU 加速, 实时报告进度。
"""

import torch
import numpy as np
import json
import os


def main():
    print("=" * 70)
    print("关键实验: 从空白 Tensor 自发涌现稀疏字典结构")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")

    # ==========================================
    # 实验参数
    # ==========================================
    d_input = 64       # 输入维度 (模拟感觉输入)
    d_hidden = 256     # 隐藏层维度 (模拟皮层神经元数)
    d_predict = 64     # 预测层维度 (模拟高层预测)
    n_categories = 5   # 数据中的隐含类别数
    n_steps = 5000     # 数据冲刷步数
    batch_size = 64

    # 学习率
    lr_hebbian = 0.001     # Hebbian 可塑性
    lr_inhibit = 0.0005    # 侧抑制
    lr_predict = 0.005     # 预测误差修正

    print(f"  输入维度: {d_input}, 隐藏层: {d_hidden}, 预测层: {d_predict}")
    print(f"  隐含类别: {n_categories}, 冲刷步数: {n_steps}")

    # ==========================================
    # 生成合成数据 (有隐含结构的数据流)
    # ==========================================
    print("\n[1/4] 生成有隐含结构的合成数据流...")

    # 每个类别有一个"原型向量"
    prototypes = torch.randn(n_categories, d_input, device=device)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)  # 归一化

    def generate_batch(batch_size):
        """生成带噪声的类别数据"""
        labels = torch.randint(0, n_categories, (batch_size,), device=device)
        data = prototypes[labels] + 0.3 * torch.randn(batch_size, d_input, device=device)
        # 下一帧预测目标: 同类别的另一个样本
        next_labels = labels  # 简化: 下一帧仍是同类别
        targets = prototypes[next_labels] + 0.3 * torch.randn(batch_size, d_input, device=device)
        return data, targets, labels

    # ==========================================
    # 初始化空白网络 (随机矩阵)
    # ==========================================
    print("\n[2/4] 初始化空白网络...")

    # Layer 1: 输入 -> 隐藏 (编码层)
    W1 = torch.randn(d_input, d_hidden, device=device) * 0.01
    b1 = torch.zeros(d_hidden, device=device)

    # Layer 2: 隐藏 -> 预测 (预测层)
    W2 = torch.randn(d_hidden, d_predict, device=device) * 0.01
    b2 = torch.zeros(d_predict, device=device)

    # 读出矩阵 (从预测到输入空间的解码,用于计算预测误差)
    W_decode = torch.randn(d_predict, d_input, device=device) * 0.01

    print(f"  W1: {list(W1.shape)}, W2: {list(W2.shape)}, W_decode: {list(W_decode.shape)}")

    # ==========================================
    # 局部规则冲刷
    # ==========================================
    print("\n[3/4] 开始局部规则冲刷 (无 BP, 无全局损失)...")
    print(f"  {'Step':>6} | {'Sparsity':>8} | {'Kurtosis':>8} | {'PredErr':>7} | {'W1Rank95':>8} | {'SV1/10':>6}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*6}")

    history = []

    for step in range(n_steps):
        # 生成数据
        x, x_next, labels = generate_batch(batch_size)

        # ---- 前向传播 (纯推理) ----
        # 隐藏层
        h_pre = x @ W1 + b1                    # [batch, d_hidden]
        h = torch.relu(h_pre)                    # ReLU 激活

        # 侧抑制: 对同一样本的隐藏层做 WTA (Winner-Takes-All)
        # 保留 Top-K 最活跃的神经元, 压制其他
        k = max(1, d_hidden // 10)  # 10% 存活
        topk_vals, topk_idx = h.topk(k, dim=-1)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(1, topk_idx, topk_vals)

        # 预测层
        p = h_sparse @ W2 + b2                  # [batch, d_predict]

        # 解码预测
        x_pred = p @ W_decode                    # [batch, d_input]

        # 预测误差
        pred_error = x_next - x_pred              # [batch, d_input]
        pred_err_norm = pred_error.norm(dim=-1).mean().item()

        # ---- 局部学习规则 (无 BP!) ----

        # 规则 1: Hebbian 增强 (fire together, wire together)
        # W1 += lr * x^T @ h_sparse (共现增强)
        delta_W1_hebb = lr_hebbian * (x.T @ h_sparse) / batch_size

        # 规则 2: Sanger Rule 侧抑制 (正交化)
        # 这是 Oja 规则的推广: 强制不同神经元提取不同特征
        # W1_i += lr * y_i * (x - sum_{j<=i} y_j * W1_j)
        h_sorted, sort_idx = h_sparse.abs().mean(dim=0).sort(descending=True)
        delta_W1_inhibit = torch.zeros_like(W1)
        for i in range(min(k, 20)):  # 只对最活跃的前 20 个做
            idx = sort_idx[i]
            y_i = h_sparse[:, idx:idx+1]  # [batch, 1]
            # 减去已经被更强神经元解释的部分
            reconstruction = h_sparse[:, sort_idx[:i]] @ W1[:, sort_idx[:i]].T  # [batch, d_input]
            residual = x - reconstruction
            delta_W1_inhibit[:, idx] = lr_inhibit * (residual.T @ y_i).squeeze() / batch_size

        # 规则 3: 预测误差驱动的局部修正
        # W_decode += lr * p^T @ pred_error (预测层修正)
        delta_W_decode = lr_predict * (p.T @ pred_error) / batch_size
        # W2 += lr * h_sparse^T @ (pred_error @ W_decode^T)  (误差反传一层,但仍是局部的!)
        error_signal = pred_error @ W_decode.T  # [batch, d_predict] -> 这个信号传回隐藏层
        delta_W2 = lr_predict * (h_sparse.T @ error_signal) / batch_size

        # ---- 应用更新 ----
        with torch.no_grad():
            W1 += delta_W1_hebb + delta_W1_inhibit
            W2 += delta_W2
            W_decode += delta_W_decode

            # 权重衰减 (LTD)
            W1 *= 0.9999
            W2 *= 0.9999
            W_decode *= 0.9999

        # ---- 定期采样 ----
        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # 生成大批数据用于统计
                x_eval, _, labels_eval = generate_batch(1000)
                h_eval_pre = x_eval @ W1 + b1
                h_eval = torch.relu(h_eval_pre)

                # 稀疏度
                h_np = h_eval.cpu().numpy().flatten()
                sparsity = float((np.abs(h_np) < 0.01).mean())
                kurtosis = float(np.mean(((h_np - h_np.mean()) / (h_np.std() + 1e-8))**4) - 3)

                # W1 权重秩
                W1_np = W1.cpu().numpy()
                _, S_W1, _ = np.linalg.svd(W1_np, full_matrices=False)
                var_W1 = (S_W1 ** 2) / (S_W1 ** 2).sum()
                cumvar = np.cumsum(var_W1)
                rank_95 = int(np.argmax(cumvar >= 0.95)) + 1
                sv_ratio = float(S_W1[0] / S_W1[9]) if len(S_W1) > 9 and S_W1[9] > 0 else float('inf')

                # 专家化检查
                category_activations = {}
                for cat in range(n_categories):
                    mask = (labels_eval == cat)
                    if mask.sum() > 0:
                        cat_h = h_eval[mask].mean(dim=0).cpu().numpy()
                        top_neurons = set(np.argsort(np.abs(cat_h))[-20:])
                        category_activations[cat] = top_neurons

                # 计算跨类别重叠
                overlaps = []
                for i in range(n_categories):
                    for j in range(i+1, n_categories):
                        if i in category_activations and j in category_activations:
                            overlap = len(category_activations[i] & category_activations[j])
                            overlaps.append(overlap)
                avg_overlap = float(np.mean(overlaps)) if overlaps else 20.0

            snapshot = {
                "step": step,
                "sparsity": sparsity,
                "kurtosis": kurtosis,
                "pred_error": pred_err_norm,
                "W1_rank_95": rank_95,
                "sv_ratio_1_10": sv_ratio,
                "avg_category_overlap_top20": avg_overlap,
            }
            history.append(snapshot)

            print(f"  {step:>6} | {sparsity:>7.1%} | {kurtosis:>8.1f} | {pred_err_norm:>7.3f} | "
                  f"{rank_95:>8} | {sv_ratio:>6.1f}")

    # ==========================================
    # 最终评估
    # ==========================================
    print("\n[4/4] 最终评估...")

    final = history[-1]
    initial = history[0]

    print(f"\n  === 涌现对比: 初始 vs 最终 ===")
    print(f"  {'指标':>20} | {'初始':>10} | {'最终':>10} | {'变化':>10}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"  {'稀疏度':>20} | {initial['sparsity']:>9.1%} | {final['sparsity']:>9.1%} | "
          f"{'UP' if final['sparsity'] > initial['sparsity'] else 'DOWN'}")
    print(f"  {'峰度':>20} | {initial['kurtosis']:>10.1f} | {final['kurtosis']:>10.1f} | "
          f"{'UP' if final['kurtosis'] > initial['kurtosis'] else 'DOWN'}")
    print(f"  {'W1 秩(95%)':>20} | {initial['W1_rank_95']:>10} | {final['W1_rank_95']:>10} | "
          f"{'DOWN' if final['W1_rank_95'] < initial['W1_rank_95'] else 'UP'}")
    print(f"  {'SV1/SV10':>20} | {initial['sv_ratio_1_10']:>10.1f} | {final['sv_ratio_1_10']:>10.1f} | "
          f"{'UP' if final['sv_ratio_1_10'] > initial['sv_ratio_1_10'] else 'DOWN'}")
    print(f"  {'预测误差':>20} | {initial['pred_error']:>10.3f} | {final['pred_error']:>10.3f} | "
          f"{'DOWN' if final['pred_error'] < initial['pred_error'] else 'UP'}")
    print(f"  {'跨类别重叠/20':>20} | {initial['avg_category_overlap_top20']:>10.1f} | "
          f"{final['avg_category_overlap_top20']:>10.1f} | "
          f"{'DOWN=Good' if final['avg_category_overlap_top20'] < initial['avg_category_overlap_top20'] else 'UP=Bad'}")

    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "experiment": "关键实验: 自发涌现稀疏字典结构验证",
        "parameters": {
            "d_input": d_input, "d_hidden": d_hidden, "d_predict": d_predict,
            "n_categories": n_categories, "n_steps": n_steps,
            "lr_hebbian": lr_hebbian, "lr_inhibit": lr_inhibit, "lr_predict": lr_predict,
        },
        "history": history,
        "conclusion": {
            "sparsity_emerged": final['kurtosis'] > 3,
            "expert_emerged": final['avg_category_overlap_top20'] < initial['avg_category_overlap_top20'] * 0.7,
            "low_rank_emerged": final['W1_rank_95'] < initial['W1_rank_95'] * 0.8,
            "prediction_improved": final['pred_error'] < initial['pred_error'] * 0.5,
        }
    }

    report_path = "tempdata/exp_spontaneous_emergence_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[完成] 报告: {report_path}")

    # 总结
    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    c = report["conclusion"]

    verdict_count = sum([c["sparsity_emerged"], c["expert_emerged"],
                         c["low_rank_emerged"], c["prediction_improved"]])

    print(f"  稀疏性涌现: {'YES' if c['sparsity_emerged'] else 'NO'} (峰度={final['kurtosis']:.1f})")
    print(f"  专家化涌现: {'YES' if c['expert_emerged'] else 'NO'} (重叠: {initial['avg_category_overlap_top20']:.1f}->{final['avg_category_overlap_top20']:.1f})")
    print(f"  低秩涌现:   {'YES' if c['low_rank_emerged'] else 'NO'} (秩: {initial['W1_rank_95']}->{final['W1_rank_95']})")
    print(f"  预测改善:   {'YES' if c['prediction_improved'] else 'NO'} (误差: {initial['pred_error']:.3f}->{final['pred_error']:.3f})")
    print(f"\n  总体判定: {verdict_count}/4 项涌现成功")

    if verdict_count >= 3:
        print("  [!!] 局部规则 CAN 自发涌现与 DNN 等价的稀疏编码 -> 大脑假设获得强支持!")
    elif verdict_count >= 2:
        print("  [OK] 部分涌现成功, 预测误差规则需要改进")
    else:
        print("  [XX] 涌现失败, 局部规则设计需要根本性修改")

if __name__ == "__main__":
    main()
