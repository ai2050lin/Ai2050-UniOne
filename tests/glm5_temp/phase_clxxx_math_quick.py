"""Phase CLXXX-Math Quick: 纯数学验证 — 快速版"""
import numpy as np, json, sys
from pathlib import Path
from datetime import datetime

d_model = 4096
n_func = 5
n_trials = 100  # 快速版

print("Phase CLXXX-Math Quick: 随机子空间对齐度验证", flush=True)

# 核心测试：不同W行数的随机矩阵中，5维随机正交子空间的能量占比
W_rows_list = [4096, 2560, 512, 13696]
labels = ["W_Q(4096x4096)", "W_K(2560x4096)", "W_K(512x4096,GQA)", "W_down(13696x4096)"]

for W_rows, label in zip(W_rows_list, labels):
    ratios = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        W = rng.standard_normal((W_rows, d_model))
        A = rng.standard_normal((n_func, d_model))
        Q, R = np.linalg.qr(A.T)
        V = Q[:, :n_func].T
        func_e = np.sum((W @ V.T) ** 2)
        total_e = np.sum(W ** 2)
        ratios.append(func_e / total_e)
    ratios = np.array(ratios)
    theory = n_func / d_model
    print(f"  {label}: {np.mean(ratios):.6f} +/- {np.std(ratios):.6f} (theory={theory:.6f}, err={abs(np.mean(ratios)-theory):.6f})", flush=True)

# 维度扫描
print("\n维度扫描 (W=4096x4096):", flush=True)
for n_dim in [1, 2, 5, 10, 20, 50, 100]:
    ratios = []
    for trial in range(50):
        rng = np.random.default_rng(trial + 1000)
        W = rng.standard_normal((4096, d_model))
        A = rng.standard_normal((n_dim, d_model))
        Q, R = np.linalg.qr(A.T)
        V = Q[:, :n_dim].T
        func_e = np.sum((W @ V.T) ** 2)
        total_e = np.sum(W ** 2)
        ratios.append(func_e / total_e)
    ratios = np.array(ratios)
    print(f"  n={n_dim}: {np.mean(ratios):.6f} +/- {np.std(ratios):.6f} (theory={n_dim/d_model:.6f})", flush=True)

# 结构化矩阵测试
print("\n结构化矩阵测试 (低秩):", flush=True)
n_trials_s = 50
ratios_rand, ratios_aligned = [], []
for trial in range(n_trials_s):
    rng = np.random.default_rng(trial)
    W = rng.standard_normal((4096, d_model))
    top_dirs = rng.standard_normal((50, d_model))
    Q_top, _ = np.linalg.qr(top_dirs.T)
    for i in range(50):
        W += 10 * np.outer(rng.standard_normal(4096), Q_top[:, i])
    A = rng.standard_normal((n_func, d_model))
    Q, R = np.linalg.qr(A.T)
    V_rand = Q[:, :n_func].T
    ratios_rand.append(np.sum((W @ V_rand.T) ** 2) / np.sum(W ** 2))
    U_w, s_w, Vt_w = np.linalg.svd(W, full_matrices=False)
    V_aligned = Vt_w[:n_func, :]
    ratios_aligned.append(np.sum((W @ V_aligned.T) ** 2) / np.sum(W ** 2))

ratios_rand = np.array(ratios_rand)
ratios_aligned = np.array(ratios_aligned)
print(f"  随机方向: {np.mean(ratios_rand):.6f} +/- {np.std(ratios_rand):.6f}", flush=True)
print(f"  对齐方向(top SVD): {np.mean(ratios_aligned):.6f}", flush=True)
print(f"  对齐/随机比: {np.mean(ratios_aligned)/np.mean(ratios_rand):.1f}x", flush=True)

print(f"\n关键结论: 理论预期5/4096={5/4096:.6f}={5/4096*100:.4f}%", flush=True)
print(f"Phase CLXXIX实测Q/K/V功能对齐: 0.12-0.33%", flush=True)
print(f"→ 如果模型权重中随机方向对齐度也≈0.122%, 则功能对齐可能是统计假象", flush=True)
print(f"→ 需要Phase CLXXX模型实验验证", flush=True)

results = {
    'theory_ratio': 5/4096,
    'theory_pct': 5/4096*100,
    'measured_clxxix_pct': '0.12-0.33',
    'conclusion': '理论5/4096=0.122%≈CLXXIX测量值, 需模型实验区分',
    'timestamp': datetime.now().isoformat()
}
out_dir = Path("results/phase_clxxx_math")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "theory_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print("Done!", flush=True)
