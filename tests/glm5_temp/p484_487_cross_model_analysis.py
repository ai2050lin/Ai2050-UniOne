"""P484-P487 跨模型统一分析"""
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

RESULT_DIR = Path(__file__).parent

# 三模型的核心结果汇总
print("="*70)
print("Phase CI 跨模型统一分析 (P484-P487)")
print("="*70)

# ========== P484: Gamma定义的重新审视 ==========
print("\n### P484: Gamma vs Alpha 跨模型比较 ###")

models_p484 = {
    'Qwen3': {
        'gamma_k50_vs_alpha': 0.927,
        'gamma_k200_vs_alpha': 0.775,
        'gamma_vs_layer_frac': 0.923,  # k=200
        'gamma_k50_mean': 0.026,
        'gamma_k200_mean': 0.095,
    },
    'GLM4': {
        'gamma_k50_vs_alpha': 0.197,
        'gamma_k200_vs_alpha': 0.491,
        'gamma_vs_layer_frac': -0.704,  # k=200, 负相关!
        'gamma_k50_mean': 0.013,
        'gamma_k200_mean': 0.045,
    },
    'DS7B': {
        'gamma_k50_vs_alpha': 0.421,
        'gamma_k200_vs_alpha': 0.465,
        'gamma_vs_layer_frac': -0.452,  # k=200, 负相关
        'gamma_k50_mean': 0.012,
        'gamma_k200_mean': 0.054,
    }
}

for model, data in models_p484.items():
    print(f"\n  {model}:")
    print(f"    gamma_k50 vs alpha: corr={data['gamma_k50_vs_alpha']:.3f}")
    print(f"    gamma_k200 vs alpha: corr={data['gamma_k200_vs_alpha']:.3f}")
    print(f"    gamma vs layer_frac: corr={data['gamma_vs_layer_frac']:.3f}")
    print(f"    gamma_k50_mean: {data['gamma_k50_mean']:.4f}")

# ========== P485: Alpha多种定义 ==========
print("\n### P485: 4种Alpha定义跨模型比较 ###")

alpha_comparison = {
    'Qwen3': {
        'attr_vs_spectral': -0.470,
        'attr_vs_entropy_norm': 0.523,
        'attr_best_predictor': 'd_eff(0.528)',
        'attr_vs_layer_frac': 0.628,
    },
    'GLM4': {
        'attr_vs_spectral': 0.378,
        'attr_vs_entropy_norm': 0.036,
        'attr_best_predictor': 'PR(0.168)',
        'attr_vs_layer_frac': -0.408,
    },
    'DS7B': {
        'attr_vs_spectral': 0.373,
        'attr_vs_entropy_norm': -0.190,
        'attr_best_predictor': 'd_eff(-0.252)',
        'attr_vs_layer_frac': -0.324,
    }
}

for model, data in alpha_comparison.items():
    print(f"\n  {model}:")
    print(f"    attr vs spectral: {data['attr_vs_spectral']:.3f}")
    print(f"    attr vs entropy_norm: {data['attr_vs_entropy_norm']:.3f}")
    print(f"    attr best_predictor: {data['attr_best_predictor']}")
    print(f"    attr vs layer_frac: {data['attr_vs_layer_frac']:.3f}")

# ========== P486: 统一量 ==========
print("\n### P486: 统一量跨模型比较 ###")

unified_comparison = {
    'Qwen3': {
        'd_eff_vs_alpha': 0.528,
        'd_eff_vs_gamma': 0.513,
        'PR_vs_alpha': 0.515,
        'PR_vs_gamma': 0.462,
        'fisher_vs_alpha': 0.255,
        'fisher_vs_gamma': 0.647,
        'd_eff_delta_vs_alpha': -0.421,
        'd_eff_delta_vs_gamma': -0.786,
        'norm_ratio_vs_alpha': 0.314,
        'norm_ratio_vs_gamma': 0.707,
        'alpha_vs_gamma': 0.544,
    },
    'GLM4': {
        'd_eff_vs_alpha': -0.074,
        'd_eff_vs_gamma': 0.656,
        'PR_vs_alpha': 0.168,
        'PR_vs_gamma': 0.807,
        'fisher_vs_alpha': 0.373,
        'fisher_vs_gamma': 0.821,
        'd_eff_delta_vs_alpha': 0.408,
        'd_eff_delta_vs_gamma': 0.186,
        'norm_ratio_vs_alpha': 0.099,
        'norm_ratio_vs_gamma': 0.921,
        'alpha_vs_gamma': 0.275,
    },
    'DS7B': {
        'd_eff_vs_alpha': -0.252,
        'd_eff_vs_gamma': 0.134,
        'PR_vs_alpha': -0.039,
        'PR_vs_gamma': 0.650,
        'fisher_vs_alpha': 0.139,
        'fisher_vs_gamma': 0.025,
        'd_eff_delta_vs_alpha': 0.522,
        'd_eff_delta_vs_gamma': 0.878,
        'norm_ratio_vs_alpha': -0.183,
        'norm_ratio_vs_gamma': -0.102,
        'alpha_vs_gamma': 0.392,
    }
}

# 找出跨模型一致的统一量
print("\n  === 跨模型一致的统一量 ===")
features = ['d_eff', 'PR', 'fisher', 'd_eff_delta', 'norm_ratio']
for target in ['alpha', 'gamma']:
    print(f"\n  特征 vs {target}:")
    for feat in features:
        corrs = [unified_comparison[m][f'{feat}_vs_{target}'] for m in unified_comparison]
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs)
        sign_consistent = all(c > 0 for c in corrs) or all(c < 0 for c in corrs)
        print(f"    {feat}: mean_corr={mean_corr:.3f}, std={std_corr:.3f}, "
              f"sign_consistent={sign_consistent}, corrs={[round(c,3) for c in corrs]}")

# ========== P487: 聚焦模式分类 ==========
print("\n### P487: 聚焦模式分类 ###")
print("  Qwen3: 36层全low_focus (PR>0.3但top10<0.15)")
print("  GLM4: 40层全low_focus (PR>0.3但top10<0.15)")
print("  DS7B: 28层全low_focus (PR>0.3但top10<0.15)")
print("  → 当前分类阈值不合理: 所有层都是low_focus!")
print("  → 需要基于PR和top10的连续指标, 而非硬阈值分类")

# ========== 综合分析 ==========
print("\n" + "="*70)
print("Phase CI 综合结论")
print("="*70)

print("""
1. Gamma-Alpha关系是模型相关的:
   - Qwen3: gamma_k50 vs alpha = 0.927 (极强)
   - GLM4: gamma_k50 vs alpha = 0.197 (弱)
   - DS7B: gamma_k50 vs alpha = 0.421 (中等)
   → alpha_attribution不是gamma的通用预测因子

2. Gamma的层位置趋势是模型相关的:
   - Qwen3: gamma随层增加 (corr=0.92)
   - GLM4: gamma随层减少 (corr=-0.70)
   - DS7B: gamma随层减少 (corr=-0.45)
   → Qwen3与GLM4/DS7B的gamma层间演化方向相反!

3. Alpha_attribution的层位置趋势也是模型相关的:
   - Qwen3: alpha随层增加 (0.63)
   - GLM4: alpha随层减少 (-0.41)
   - DS7B: alpha几乎不随层变化 (-0.32, range: -0.04~0.04)
   → Qwen3是唯一"alpha随层增加"的模型

4. 跨模型一致的统一量:
   - PR vs gamma: mean=0.64, std=0.15, 方向一致(全正) [OK]
   - norm_ratio vs gamma: mean=0.51, std=0.45, 方向不一致 [NO]
   - fisher vs gamma: mean=0.50, std=0.34, 方向一致 [OK]
   - d_eff_delta vs gamma: mean=0.09, std=0.55, 方向不一致 [NO]
   -> PR(参与比)是最跨模型一致的gamma预测因子!

5. Alpha_attribution的定义问题:
   - DS7B的alpha_attribution几乎为0(range: -0.04~0.04)
   - GLM4的alpha_attribution以负值为主
   - 只有Qwen3的alpha_attribution有有意义的层间变化
   → 属性干预方法可能在某些模型中不敏感

6. Alpha_spectral vs Alpha_attribution:
   - Qwen3: corr=-0.470 (负相关!)
   - GLM4: corr=0.378
   - DS7B: corr=0.373
   → 两种alpha定义捕捉不同的信息, 不能互换

核心发现: PR(参与比)是跨模型最一致的gamma预测因子!
物理意义: PR衡量W_down谱的"有效自由度" → PR越大→越多的奇异值参与→信号被更均匀地传播→gamma越大
""")

# 保存结果
out_path = RESULT_DIR / "p484_487_cross_model_summary.txt"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("Phase CI 跨模型统一分析 (P484-P487)\n")
    f.write("="*60 + "\n\n")
    f.write("1. Gamma-Alpha关系模型相关: Qwen3=0.93, GLM4=0.20, DS7B=0.42\n")
    f.write("2. Gamma层位置趋势相反: Qwen3正(+0.92), GLM4/DS7B负(-0.70/-0.45)\n")
    f.write("3. Alpha层位置趋势: Qwen3正(+0.63), GLM4负(-0.41), DS7B几乎不变\n")
    f.write("4. PR是跨模型最一致的gamma预测因子: mean_corr=0.64, 方向一致\n")
    f.write("5. Alpha_attribution在DS7B中几乎为0, 方法可能不敏感\n")
    f.write("6. Alpha_spectral与Alpha_attribution不可互换(Qwen3负相关-0.47)\n")

print(f"\n结果已保存到 {out_path}")
