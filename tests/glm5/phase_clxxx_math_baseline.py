"""
Phase CLXXX-Math: 纯数学验证 — 随机子空间对齐度的理论预期
===========================================================
不需要GPU，只用numpy随机矩阵验证理论

核心理论:
  对于随机正交方向 V ∈ R^(n×d)，权重矩阵 W ∈ R^(m×d)
  V在W中的能量占比 E[V@W^T]^2 / E[||W||^2] 的理论期望值

  如果W的各行独立从某分布采样，则:
    E[ Σ_i (v_k · w_i)^2 ] = Σ_i E[(v_k · w_i)^2]
  
  对随机正交v_k:
    E[(v_k · w_i)^2] = ||w_i||^2 / d  (均匀分布假设)
  
  所以:
    E[ratio] = n / d (子空间维度/总维度)

  对于 n=5, d=4096:
    E[ratio] = 5/4096 ≈ 0.122%
    
  这恰好等于我们测到的Q/K/V功能对齐度0.12-0.33%!

  但是！这不完全正确——W_Q等不是随机矩阵，它们是训练后的权重。
  关键问题是：训练后的权重矩阵，其行空间是否均匀分布？
  
  如果训练后W_Q的行空间仍然"近似均匀"，那么随机方向的期望对齐度
  就约等于 n/d ≈ 0.122%。
  
  但如果训练创造了特殊的结构（如W_Q集中在某些子空间），
  那么随机方向和功能方向的对齐度可能不同。

  所以真正需要测试的是：
  1. 理论基线 n/d 是否确实等于0.122%
  2. 训练后的W_Q/W_K/W_V/W_O中，随机方向的对齐度是否偏离 n/d
  3. 功能方向的对齐度是否显著偏离随机方向的对齐度
"""

import numpy as np
from datetime import datetime

def test_random_matrix_alignment():
    """纯数学验证：随机矩阵中随机子空间的对齐度"""
    print("="*60)
    print("纯数学验证: 随机矩阵中随机子空间的对齐度")
    print("="*60)
    
    # 模拟d_model=4096的情况
    d_model = 4096
    
    # 不同W矩阵行数(模拟n_heads*head_dim)
    W_rows_list = [4096, 2560, 512, 13696]  # W_Q, W_K/V(GQA), W_down
    labels = ["W_Q (4096×4096)", "W_K (2560×4096, Qwen3)", 
              "W_K (512×4096, GLM4 GQA)", "W_down (13696×4096, GLM4)"]
    
    n_func = 5  # 功能方向数
    
    # 不同随机种子
    n_trials = 500
    
    for W_rows, label in zip(W_rows_list, labels):
        print(f"\n--- {label} ---")
        
        ratios = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial)
            
            # 生成随机W矩阵 (模拟训练后的权重)
            W = rng.standard_normal((W_rows, d_model))
            
            # 生成随机正交方向
            A = rng.standard_normal((n_func, d_model))
            Q, R = np.linalg.qr(A.T)
            V = Q[:, :n_func].T  # [5, 4096]
            
            # 计算对齐度
            W_func = W @ V.T  # [W_rows, n_func]
            func_energy = np.sum(W_func ** 2)
            total_energy = np.sum(W ** 2)
            ratio = func_energy / total_energy
            
            ratios.append(ratio)
        
        ratios = np.array(ratios)
        theory = n_func / d_model
        
        print(f"  实测: {np.mean(ratios):.6f} ± {np.std(ratios):.6f}")
        print(f"  理论: {theory:.6f}")
        print(f"  误差: {abs(np.mean(ratios) - theory):.6f}")
        print(f"  P95:  {np.percentile(ratios, 95):.6f}")
        print(f"  P99:  {np.percentile(ratios, 99):.6f}")
    
    # 更完整的维度扫描
    print("\n\n" + "="*60)
    print("维度扫描: n/d理论值 vs 实测值")
    print("="*60)
    
    W_rows = 4096
    print(f"\n  W矩阵: {W_rows}×{d_model}")
    print(f"  {'n_dim':>6} | {'theory':>10} | {'measured':>10} | {'std':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for n_dim in [1, 2, 5, 10, 20, 50, 100, 200]:
        ratios = []
        for trial in range(200):
            rng = np.random.default_rng(trial + 1000)
            W = rng.standard_normal((W_rows, d_model))
            A = rng.standard_normal((n_dim, d_model))
            Q, R = np.linalg.qr(A.T)
            V = Q[:, :n_dim].T
            func_e = np.sum((W @ V.T) ** 2)
            total_e = np.sum(W ** 2)
            ratios.append(func_e / total_e)
        
        ratios = np.array(ratios)
        theory = n_dim / d_model
        print(f"  {n_dim:>6} | {theory:>10.6f} | {np.mean(ratios):>10.6f} | {np.std(ratios):>10.6f}")
    
    # 关键结论
    print("\n\n" + "!"*60)
    print("关键理论结论:")
    print("!"*60)
    print(f"""
  1. 随机正交子空间(维度n)在随机矩阵中的期望能量占比 = n/d_model
  2. 当n=5, d_model=4096时:
     理论值 = 5/4096 = {5/4096:.6f} = {5/4096*100:.4f}%
  3. 这恰好等于Phase CLXXIX中测到的Q/K/V功能对齐度!
     W_Q功能对齐: 0.12-0.33%
     随机基线预期: 0.122%
  
  但这不代表功能-内容分离一定是假象!
  因为W_Q/W_K/W_V不是随机矩阵——它们是训练后的权重。
  训练可能创造了特殊的行空间结构。
  
  关键区分实验(需要实际模型):
    - 如果功能方向的对齐度 >> 随机方向的对齐度(都是5维)
      → 功能方向"选择"了W_Q中特殊对齐的子空间 → 真实结构
    - 如果功能方向的对齐度 ≈ 随机方向的对齐度
      → 功能方向没有选择特殊子空间 → 统计假象
    
  同理，W_U能量占比:
    - 如果5维功能子空间的W_U能量 >> 5维随机子空间的W_U能量
      → 功能方向"选择"了W_U中高能量的子空间 → 真实结构
    - 如果功能方向的W_U能量 ≈ 随机方向的W_U能量
      → 功能方向只是随机方向 → 功能极薄也是假象
""")


def test_structured_matrix():
    """测试：有结构的矩阵中，随机vs特定方向的对齐度差异"""
    print("\n" + "="*60)
    print("结构化矩阵测试: 低秩结构对对齐度的影响")
    print("="*60)
    
    d_model = 4096
    W_rows = 4096
    n_func = 5
    n_trials = 200
    
    # 情况1: 完全随机矩阵
    print("\n--- 情况1: 随机矩阵 ---")
    ratios_rand = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        W = rng.standard_normal((W_rows, d_model))
        A = rng.standard_normal((n_func, d_model))
        Q, R = np.linalg.qr(A.T)
        V = Q[:, :n_func].T
        ratios_rand.append(np.sum((W @ V.T) ** 2) / np.sum(W ** 2))
    
    print(f"  随机方向: {np.mean(ratios_rand):.6f} ± {np.std(ratios_rand):.6f}")
    print(f"  理论: {n_func/d_model:.6f}")
    
    # 情况2: 低秩矩阵 (前k个奇异值远大于其余)
    print("\n--- 情况2: 低秩矩阵(前50奇异值=10x其余) ---")
    ratios_rand_lr = []
    ratios_aligned = []
    
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        # 生成低秩矩阵
        U = rng.standard_normal((W_rows, 50))
        V_base = rng.standard_normal((50, d_model))
        S = np.diag(np.concatenate([np.ones(50) * 10, np.ones(min(W_rows, d_model)-50)]))
        
        # 简单构造: 主要能量在前50维
        W = rng.standard_normal((W_rows, d_model))
        # 增加前50维的能量
        top_dirs = rng.standard_normal((50, d_model))
        Q_top, _ = np.linalg.qr(top_dirs.T)
        for i in range(50):
            W += 10 * np.outer(rng.standard_normal(W_rows), Q_top[:, i])
        
        # 随机方向对齐度
        A = rng.standard_normal((n_func, d_model))
        Q, R = np.linalg.qr(A.T)
        V_rand = Q[:, :n_func].T
        ratios_rand_lr.append(np.sum((W @ V_rand.T) ** 2) / np.sum(W ** 2))
        
        # "对齐"方向: 选择W的高能量方向
        # SVD取前5个右奇异向量
        U_w, s_w, Vt_w = np.linalg.svd(W, full_matrices=False)
        V_aligned = Vt_w[:n_func, :]  # [5, 4096] - W的高能量方向
        ratios_aligned.append(np.sum((W @ V_aligned.T) ** 2) / np.sum(W ** 2))
    
    print(f"  随机方向: {np.mean(ratios_rand_lr):.6f} ± {np.std(ratios_rand_lr):.6f}")
    print(f"  对齐方向(W的top奇异向量): {np.mean(ratios_aligned):.6f}")
    print(f"  理论(n/d): {n_func/d_model:.6f}")
    print(f"  对齐/随机比: {np.mean(ratios_aligned)/np.mean(ratios_rand_lr):.1f}x")
    
    print("""
  关键洞察:
    - 在低秩矩阵中，"对齐"方向(W的top奇异向量)的对齐度远高于随机方向
    - 如果功能方向确实选择了W_U/W_Q的特殊方向，对齐度会远高于随机基线
    - 反之，如果功能方向对齐度≈随机基线，说明功能方向并没有选择特殊子空间
    """)


if __name__ == "__main__":
    test_random_matrix_alignment()
    test_structured_matrix()
    
    # 保存理论结果
    import json
    from pathlib import Path
    
    results = {
        'theory_ratio_5_4096': 5/4096,
        'theory_ratio_pct': 5/4096 * 100,
        'qkv_alignment_measured_pct': 0.12,  # Phase CLXXIX测量值
        'conclusion': '理论预期5/4096=0.122%≈实测0.12-0.33%, 需要实际模型测试区分',
        'timestamp': datetime.now().isoformat(),
    }
    
    out_dir = Path("results/phase_clxxx_math")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "theory_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("理论验证完成! 结果保存在 results/phase_clxxx_math/theory_results.json")
