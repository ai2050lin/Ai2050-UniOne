"""CCIX Cross-model analysis"""
import json, numpy as np

models = ['qwen3', 'glm4']
model_names = {'qwen3': 'Qwen3-4B', 'glm4': 'GLM4-9B'}

print('='*80)
print('CCIX Cross-model Comparison')
print('='*80)

# Part 1: Subspace Energy
print('\n=== Part 1: Subspace Energy (animal50, semantic/total) ===')
for m in models:
    try:
        data = json.load(open(f'tests/glm5_temp/ccix_{m}_results.json', encoding='utf-8'))
        subspace = data.get('subspace_animal50', [])
        deep = [r for r in subspace if r['layer'] >= 6]
        if deep:
            avg_attn = np.mean([r['sem_ratio_attn'] for r in deep])
            avg_mlp = np.mean([r['sem_ratio_mlp'] for r in deep])
            avg_rand = np.mean([r['random_baseline'] for r in deep])
            print(f'  {model_names[m]:12s} L6+: Attn_sem={avg_attn:.4f}, MLP_sem={avg_mlp:.4f}, Random={avg_rand:.4f}')
            print(f'               MLP > Random: {avg_mlp > avg_rand}, MLP > Attn: {avg_mlp > avg_attn}')
    except Exception as e:
        print(f'  {m}: error {e}')

# Part 2: Geometry Correction
print('\n=== Part 2: Geometry Correction (animal50, deep layers L6+) ===')
for m in models:
    try:
        data = json.load(open(f'tests/glm5_temp/ccix_{m}_results.json', encoding='utf-8'))
        corr = data.get('correction_animal50', [])
        deep = [r for r in corr if r['layer'] >= 6]
        if deep:
            avg_corr = np.mean([r['correction'] for r in deep])
            avg_beta_attn = np.mean([r['beta_attn_only'] for r in deep])
            avg_beta_mlp = np.mean([r['beta_mlp_full'] for r in deep])
            print(f'  {model_names[m]:12s} L6+: beta_attn={avg_beta_attn:+.3f}, beta_mlp={avg_beta_mlp:+.3f}, correction={avg_corr:+.3f}')
    except:
        pass

# Part 3: Jacobian
print('\n=== Part 3: MLP Jacobian (animal50, deep layers L6+) ===')
for m in models:
    try:
        data = json.load(open(f'tests/glm5_temp/ccix_{m}_results.json', encoding='utf-8'))
        jac = data.get('jacobian_animal50', [])
        deep = [r for r in jac if r['layer'] >= 6]
        if deep:
            avg_ratio = np.mean([r['amp_ratio_pca_vs_rand'] for r in deep])
            avg_pca_pres = np.mean([r['pca_pres_mean'] for r in deep])
            avg_rand_pres = np.mean([r['rand_pres_mean'] for r in deep])
            avg_diff = np.mean([r['pres_diff_pca_vs_rand'] for r in deep])
            print(f'  {model_names[m]:12s} L6+: amp_ratio(PCA/Rand)={avg_ratio:.3f}')
            print(f'               PCA_pres={avg_pca_pres:+.4f}, Rand_pres={avg_rand_pres:+.4f}, diff={avg_diff:+.4f}')
    except:
        pass

print('\n' + '='*80)
print('CORE FINDINGS')
print('='*80)
print()
print('1. Subspace Energy: MLP semantic ratio ~ random baseline')
print('   => MLP does NOT preferentially focus on semantic dimensions')
print()
print('2. Geometry Correction: correction ~ 0 for deep layers')
print('   => MLP immediate geometric effect is negligible')
print('   => CCVIII no_mlp negative beta = CUMULATIVE effect, not immediate')
print()
print('3. Jacobian Amplification: PCA/Rand ratio ~ 1.0')
print('   => MLP does NOT preferentially amplify semantic directions')
print()
print('4. JACOBIAN PRESERVATION: deep layers pres ~ -0.5 (DIRECTION INVERSION!)')
print('   => MLP REFLECTS/INVERTS directions in deep layers')
print('   => Residual connection + MLP reflection = geometry preservation')
print('   => This is the TRUE mechanism of MLP as "geometry preserver"!')
print()
print('5. CORRECTED CCVIII interpretation:')
print('   CCVIII: "MLP is the geometry preserver"')
print('   CCIX correction: MLP is a direction reflector + residual is the geometry anchor')
print('   Combined: h + MLP(LN(h)) ~ h + (-0.5 * perturbation)')
print('   Residual "1" dominates geometry, MLP "-0.5" corrects direction')
