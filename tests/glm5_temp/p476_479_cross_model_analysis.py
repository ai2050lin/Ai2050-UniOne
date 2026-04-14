"""
Phase XCIX 跨模型统一分析
汇总三模型的P476-P479结果
"""

import numpy as np

# ===== 架构参数汇总 =====
models = {
    'qwen3': {
        'd_model': 2560, 'n_layers': 36, 'intermediate_size': 9728,
        'FFN_ratio': 3.800, 'd_over_n': 71.1, 'n_heads': 32, 'head_dim': 80,
        'delta': 0.185, 'mean_alpha': 0.196, 'focusing_ratio': 0.84,
        'signal_growth_exp': 0.929,  # ||h_L|| ~ L^0.929
    },
    'glm4': {
        'd_model': 4096, 'n_layers': 40, 'intermediate_size': 13696,
        'FFN_ratio': 3.344, 'd_over_n': 102.4, 'n_heads': 32, 'head_dim': 128,
        'delta': 0.205, 'mean_alpha': 0.669, 'focusing_ratio': 1.00,
        'signal_growth_exp': 1.873,  # ||h_L|| ~ L^1.873
    },
    'deepseek7b': {
        'd_model': 3584, 'n_layers': 28, 'intermediate_size': 18944,
        'FFN_ratio': 5.286, 'd_over_n': 128.0, 'n_heads': 28, 'head_dim': 128,
        'delta': 0.233, 'mean_alpha': 0.455, 'focusing_ratio': 1.00,
        'signal_growth_exp': 0.915,  # ||h_L|| ~ L^0.915
    },
}

# ===== P477: 训练残留汇总 =====
p477_data = {
    'qwen3': {
        'mean_norm_ratio': 2.244, 'mean_mp_deviation': 259289,
        'mean_s_max_ratio': 5.027, 'mean_info_ratio': 0.992,
        'best_predictor': 's_max_ratio', 'best_corr': 0.138,
    },
    'glm4': {
        'mean_norm_ratio': 1.613, 'mean_mp_deviation': 197047,
        'mean_s_max_ratio': 3.717, 'mean_info_ratio': 0.679,
        'best_predictor': 'norm_ratio', 'best_corr': 0.389,
    },
    'deepseek7b': {
        'mean_norm_ratio': 3.369, 'mean_mp_deviation': 1508755,
        'mean_s_max_ratio': 8.637, 'mean_info_ratio': 0.689,
        'best_predictor': 'info_ratio', 'best_corr': 0.249,
    },
}

# ===== P479: 多变量模型汇总 =====
p479_data = {
    'qwen3': {
        'best_single': 'ln_norm', 'best_single_corr': 0.590,
        'final_features': ['ln_norm', 'layer_frac', 'PR'],
        'r2_final': 0.489, 'r2_adj': 0.387,
    },
    'glm4': {
        'best_single': 'norm_ratio', 'best_single_corr': 0.389,
        'final_features': ['norm_ratio', 'kappa', 'ln_norm', 'layer_frac', 'F_W', 'delta_W', 'PR'],
        'r2_final': 0.776, 'r2_adj': 0.655,
    },
    'deepseek7b': {
        'best_single': 'ln_norm', 'best_single_corr': 0.515,
        'final_features': ['ln_norm', 'delta_W', 'sigma_W', 'PR'],
        'r2_final': 0.508, 'r2_adj': 0.423,
    },
}

print("=" * 70)
print("Phase XCIX: Cross-Model Unified Analysis")
print("=" * 70)

# 1. 信号增长指数与架构参数的关系
print("\n### 1. Signal Growth Exponent vs Architecture")
print(f"    {'Model':>12} {'d/n_L':>6} {'FFN_r':>6} {'alpha_exp':>10} {'mean_alpha':>10} {'focus_r':>8}")
for name, m in models.items():
    print(f"    {name:>12} {m['d_over_n']:6.1f} {m['FFN_ratio']:6.3f} "
          f"{m['signal_growth_exp']:10.3f} {m['mean_alpha']:10.3f} {m['focusing_ratio']:8.2f}")

# 相关性分析
d_over_n = np.array([m['d_over_n'] for m in models.values()])
ffn_ratios = np.array([m['FFN_ratio'] for m in models.values()])
growth_exps = np.array([m['signal_growth_exp'] for m in models.values()])
mean_alphas = np.array([m['mean_alpha'] for m in models.values()])
focus_ratios = np.array([m['focusing_ratio'] for m in models.values()])

# 注意: 只有3个数据点, 相关性需要谨慎
print(f"\n  Correlations (n=3, use with caution):")
corr_d_growth = np.corrcoef(d_over_n, growth_exps)[0, 1]
corr_d_alpha = np.corrcoef(d_over_n, mean_alphas)[0, 1]
corr_ffn_growth = np.corrcoef(ffn_ratios, growth_exps)[0, 1]
corr_ffn_alpha = np.corrcoef(ffn_ratios, mean_alphas)[0, 1]
corr_d_focus = np.corrcoef(d_over_n, focus_ratios)[0, 1]

print(f"    d/n_L vs growth_exponent: {corr_d_growth:.3f}")
print(f"    d/n_L vs mean_alpha: {corr_d_alpha:.3f}")
print(f"    FFN_ratio vs growth_exponent: {corr_ffn_growth:.3f}")
print(f"    FFN_ratio vs mean_alpha: {corr_ffn_alpha:.3f}")
print(f"    d/n_L vs focusing_ratio: {corr_d_focus:.3f}")

# 2. 训练残留跨模型对比
print(f"\n### 2. Training Residual Cross-Model")
print(f"    {'Model':>12} {'NR':>6} {'SMR':>6} {'IR':>6} {'best_pred':>12} {'best_c':>7}")
for name, d in p477_data.items():
    print(f"    {name:>12} {d['mean_norm_ratio']:6.3f} {d['mean_s_max_ratio']:6.3f} "
          f"{d['mean_info_ratio']:6.3f} {d['best_predictor']:>12} {d['best_corr']:7.3f}")

# norm_ratio vs d/n_L
norm_ratios = np.array([p477_data[m]['mean_norm_ratio'] for m in models.keys()])
print(f"\n  norm_ratio vs d/n_L: corr = {np.corrcoef(d_over_n, norm_ratios)[0, 1]:.3f}")
print(f"  norm_ratio vs FFN_ratio: corr = {np.corrcoef(ffn_ratios, norm_ratios)[0, 1]:.3f}")

# 3. 多变量模型对比
print(f"\n### 3. Multivariate Model Comparison")
print(f"    {'Model':>12} {'best_single':>12} {'best_c':>7} {'R2':>6} {'R2_adj':>7} {'n_feat':>6}")
for name, d in p479_data.items():
    print(f"    {name:>12} {d['best_single']:>12} {d['best_single_corr']:7.3f} "
          f"{d['r2_final']:6.3f} {d['r2_adj']:7.3f} {len(d['final_features']):6d}")

# 4. 关键发现: 信号增长指数
print(f"\n### 4. Key Discovery: Signal Growth Exponent")
print(f"    ||h_L|| / ||h_0|| ~ L^beta")
print(f"    Qwen3: beta = 0.929 (near-linear)")
print(f"    GLM4:  beta = 1.873 (near-quadratic!)")
print(f"    DS7B:  beta = 0.915 (near-linear)")
print(f"")
print(f"    GLM4 is the outlier: its signal grows quadratically!")
print(f"    This may explain why GLM4 has mean_alpha=0.669 >> Qwen3's 0.196")
print(f"")

# GLM4 的特殊之处
print(f"### 5. What Makes GLM4 Special?")
print(f"    GLM4 has:")
print(f"    - Highest signal growth: beta=1.873 vs ~0.92 for others")
print(f"    - Highest focusing_ratio: 1.00 (vs Qwen3's 0.84)")
print(f"    - Highest mean_alpha: 0.669 (vs Qwen3's 0.196)")
print(f"    - But NOT the highest d/n_L (102.4 vs DS7B's 128.0)")
print(f"    - And NOT the highest FFN_ratio (3.344 vs DS7B's 5.286)")
print(f"")
print(f"    => Signal growth exponent beta is NOT determined by d/n_L or FFN_ratio!")
print(f"    => It's determined by the training dynamics / weight structure!")
print(f"")

# 6. ln_norm (LayerNorm权重范数) 作为alpha的最佳单变量预测因子
print(f"### 6. LayerNorm Weight Norm as Alpha Predictor")
print(f"    Qwen3: ln_norm vs alpha corr = 0.590")
print(f"    GLM4:  ln_norm vs alpha corr = 0.203")
print(f"    DS7B:  ln_norm vs alpha corr = 0.515")
print(f"")
print(f"    ln_norm is the best single predictor in 2/3 models!")
print(f"    => LayerNorm weight magnitude influences signal focusing")
print(f"    => This makes physical sense: larger LN weights -> more amplification -> more focusing")
print(f"")

# 7. 总结
print(f"### 7. Phase XCIX Summary")
print(f"")
print(f"**Core Finding 1**: Signal growth follows power law ||h_L|| ~ L^beta")
print(f"    - Qwen3: beta=0.929, GLM4: beta=1.873, DS7B: beta=0.915")
print(f"    - GLM4 is the outlier with near-quadratic growth")
print(f"    - beta is NOT determined by d/n_L or FFN_ratio")
print(f"")
print(f"**Core Finding 2**: LayerNorm weight norm (ln_norm) is the best single predictor of alpha")
print(f"    - In 2/3 models (Qwen3, DS7B), ln_norm has the highest correlation with alpha")
print(f"    - Physical interpretation: larger LN weights amplify signals more -> more focusing")
print(f"")
print(f"**Core Finding 3**: Training residual (norm_ratio) varies 2x across models")
print(f"    - GLM4: 1.613, Qwen3: 2.244, DS7B: 3.369")
print(f"    - DS7B has the most training deviation from initialization")
print(f"    - But norm_ratio is NOT a good predictor of alpha within models")
print(f"")
print(f"**Core Finding 4**: Multi-variable models achieve R2=0.49-0.78")
print(f"    - Still 22-51% variance unexplained")
print(f"    - No universal set of features works across models")
print(f"    - alpha is influenced by multiple factors, none dominant")
