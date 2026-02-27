# -*- coding: utf-8 -*-
"""
Phase XXXVI 拼图 #6: Z113 训练相变点追踪
=========================================
核心问题: 模型什么时候从"死记硬背"突变为"抽象理解"(Grokking)?
- 训练过程中表示空间的拓扑结构如何变化?
- 是否存在明确的"相变点"?
- 相变前后,权重矩阵的秩和稀疏度是否发生质变?

这个实验要找到"涌现"的数学签名——临界点方程。
使用 GPU 训练,实时报告进度。
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os

# 强制 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

P = 113  # Z_113 群
EMBED_DIM = 128
HIDDEN_DIM = 256

class Z113Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(P, EMBED_DIM)
        self.W1 = nn.Linear(EMBED_DIM * 2, HIDDEN_DIM)
        self.W2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.W_out = nn.Linear(HIDDEN_DIM, P)

    def forward(self, a, b):
        e_a = self.E(a)
        e_b = self.E(b)
        x = torch.cat([e_a, e_b], dim=-1)
        h1 = torch.relu(self.W1(x))
        h2 = torch.relu(self.W2(h1))
        return self.W_out(h2), h1, h2


def compute_metrics(model, A_all, B_all, C_all, device):
    """计算当前模型的各种结构指标"""
    model.eval()
    with torch.no_grad():
        logits, h1, h2 = model(A_all.to(device), B_all.to(device))
        preds = logits.argmax(dim=-1)
        accuracy = (preds == C_all.to(device)).float().mean().item()

        # 1. 嵌入矩阵 SVD
        E = model.E.weight.detach().cpu().numpy()
        E_centered = E - E.mean(axis=0)
        _, S_E, _ = np.linalg.svd(E_centered, full_matrices=False)
        var_E = (S_E ** 2) / (S_E ** 2).sum()
        cumvar_E = np.cumsum(var_E)
        intrinsic_dim_95 = int(np.argmax(cumvar_E >= 0.95)) + 1

        # 2. 隐藏层激活稀疏度
        h1_cpu = h1.detach().cpu().numpy()
        h1_sparsity = float((np.abs(h1_cpu) < 0.01).mean())
        h1_kurtosis = float(np.mean(((h1_cpu - h1_cpu.mean()) / (h1_cpu.std() + 1e-8))**4) - 3)

        # 3. W1 权重矩阵的秩
        W1 = model.W1.weight.detach().cpu().numpy()
        _, S_W1, _ = np.linalg.svd(W1, full_matrices=False)
        var_W1 = (S_W1 ** 2) / (S_W1 ** 2).sum()
        cumvar_W1 = np.cumsum(var_W1)
        W1_rank_95 = int(np.argmax(cumvar_W1 >= 0.95)) + 1

        # 4. 嵌入空间的前两个主成分是否形成圆(模运算的标志)
        U_E = (E_centered @ np.linalg.svd(E_centered, full_matrices=False)[2][:2].T)
        # 计算到原点的距离标准差(如果是圆,标准差很小)
        distances = np.sqrt((U_E ** 2).sum(axis=1))
        circularity = float(1.0 - distances.std() / (distances.mean() + 1e-8))

    model.train()
    return {
        "accuracy": accuracy,
        "intrinsic_dim_95": intrinsic_dim_95,
        "h1_sparsity": h1_sparsity,
        "h1_kurtosis": h1_kurtosis,
        "W1_rank_95": W1_rank_95,
        "circularity": circularity,
        "top5_sv_E": S_E[:5].tolist(),
        "sv_ratio_E": float(S_E[0] / S_E[1]) if S_E[1] > 0 else float('inf'),
    }


def main():
    print("=" * 70)
    print("Phase XXXVI 拼图 #6: Z113 训练相变点追踪")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")

    # 全数据集
    A_all, B_all = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
    A_all, B_all = A_all.flatten(), B_all.flatten()
    C_all = (A_all + B_all) % P
    total = len(A_all)  # 113*113 = 12769

    # 训练/测试分割 (50/50 为了观察 Grokking)
    perm = torch.randperm(total)
    train_size = total // 2
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    A_train, B_train, C_train = A_all[train_idx], B_all[train_idx], C_all[train_idx]
    A_test, B_test, C_test = A_all[test_idx], B_all[test_idx], C_all[test_idx]

    print(f"  训练集: {len(A_train)}, 测试集: {len(A_test)}")

    model = Z113Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    total_epochs = 15000
    checkpoint_interval = 100
    history = []

    print(f"\n[训练] 共 {total_epochs} 步,每 {checkpoint_interval} 步采样一次...")
    print(f"  {'Epoch':>6} | {'TrainAcc':>8} | {'TestAcc':>8} | {'IntDim':>6} | {'Sparse':>7} | {'Kurt':>7} | {'W1Rank':>6} | {'Circ':>5}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5}")

    grokking_epoch = None
    prev_test_acc = 0

    for epoch in range(total_epochs):
        # Mini-batch 训练
        model.train()
        perm_train = torch.randperm(len(A_train))
        batch_size = 512
        total_loss = 0
        n_batches = 0

        for i in range(0, len(A_train), batch_size):
            idx = perm_train[i:i+batch_size]
            logits, _, _ = model(A_train[idx].to(device), B_train[idx].to(device))
            loss = loss_fn(logits, C_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # 定期采样
        if epoch % checkpoint_interval == 0 or epoch == total_epochs - 1:
            # 训练集准确率
            model.eval()
            with torch.no_grad():
                train_logits, _, _ = model(A_train.to(device), B_train.to(device))
                train_acc = (train_logits.argmax(-1) == C_train.to(device)).float().mean().item()

                test_logits, _, _ = model(A_test.to(device), B_test.to(device))
                test_acc = (test_logits.argmax(-1) == C_test.to(device)).float().mean().item()

            # 计算结构指标
            metrics = compute_metrics(model, A_all, B_all, C_all, device)
            metrics["epoch"] = epoch
            metrics["train_accuracy"] = train_acc
            metrics["test_accuracy"] = test_acc
            metrics["loss"] = total_loss / n_batches
            history.append(metrics)

            # 检测 Grokking
            if grokking_epoch is None and test_acc > 0.9 and prev_test_acc < 0.5:
                grokking_epoch = epoch
                print(f"  *** GROKKING DETECTED at epoch {epoch}! ***")

            if epoch % (checkpoint_interval * 5) == 0 or epoch == total_epochs - 1 or (grokking_epoch and epoch == grokking_epoch):
                print(f"  {epoch:>6} | {train_acc:>7.1%} | {test_acc:>7.1%} | "
                      f"{metrics['intrinsic_dim_95']:>6} | {metrics['h1_sparsity']:>6.1%} | "
                      f"{metrics['h1_kurtosis']:>7.1f} | {metrics['W1_rank_95']:>6} | "
                      f"{metrics['circularity']:>5.3f}")

            prev_test_acc = test_acc

            # 早停: 测试集满分
            if test_acc > 0.99:
                print(f"\n  [早停] 测试集准确率 {test_acc:.1%} > 99%, 已完成泛化!")
                break

    # ==========================================
    # 分析相变
    # ==========================================
    print("\n" + "=" * 70)
    print("相变分析")
    print("=" * 70)

    if grokking_epoch:
        print(f"Grokking 相变点: Epoch {grokking_epoch}")

        # 找到相变前后的快照
        pre_phase = [h for h in history if h["epoch"] < grokking_epoch]
        post_phase = [h for h in history if h["epoch"] >= grokking_epoch]

        if pre_phase and post_phase:
            pre = pre_phase[-1]
            post = post_phase[0] if len(post_phase) > 0 else post_phase[-1]

            print(f"\n  相变前 (Epoch {pre['epoch']}):")
            print(f"    内禀维度: {pre['intrinsic_dim_95']}")
            print(f"    稀疏度: {pre['h1_sparsity']:.1%}")
            print(f"    峰度: {pre['h1_kurtosis']:.1f}")
            print(f"    W1 秩: {pre['W1_rank_95']}")
            print(f"    圆度: {pre['circularity']:.3f}")

            print(f"\n  相变后 (Epoch {post['epoch']}):")
            print(f"    内禀维度: {post['intrinsic_dim_95']}")
            print(f"    稀疏度: {post['h1_sparsity']:.1%}")
            print(f"    峰度: {post['h1_kurtosis']:.1f}")
            print(f"    W1 秩: {post['W1_rank_95']}")
            print(f"    圆度: {post['circularity']:.3f}")
    else:
        print("未检测到明确的 Grokking 相变")
        if history:
            final = history[-1]
            print(f"  最终状态: Train={final['train_accuracy']:.1%}, Test={final['test_accuracy']:.1%}")
            print(f"  内禀维度={final['intrinsic_dim_95']}, 峰度={final['h1_kurtosis']:.1f}")

    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    report = {
        "experiment": "Phase XXXVI E5: Z113 训练相变追踪",
        "grokking_epoch": grokking_epoch,
        "total_epochs": epoch + 1,
        "history": history,
    }
    report_path = "tempdata/exp_phase_transition_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 报告: {report_path}")

if __name__ == "__main__":
    main()
