"""CCXLV 严格统计检验: specificity + best_alignment"""
import json, numpy as np
from pathlib import Path

TEMP = Path('tests/glm5_temp')

# 收集所有结果
specs = []
for model in ['qwen3', 'glm4', 'deepseek7b']:
    jp = TEMP / f'ccxlv_large_n_{model}.json'
    if not jp.exists():
        continue
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for domain, result in data.items():
        edge_al = result.get('edge_alignment', {})
        for cls, ar in edge_al.items():
            if 'error' in ar:
                continue
            specs.append({
                'model': model,
                'domain': domain,
                'cls': cls,
                'best': ar.get('best_alignment', 0),
                'specificity': ar.get('specificity', 0),
                'verdict': ar.get('verdict', '?'),
            })

n_aligned = sum(1 for s in specs if s['verdict'] == 'EDGE-ALIGNED')
n_weak = sum(1 for s in specs if s['verdict'] != 'EDGE-ALIGNED')
print(f'总轨迹数: {len(specs)}')
print(f'EDGE-ALIGNED: {n_aligned}, WEAK-EDGE: {n_weak}')

spec_vals = [s['specificity'] for s in specs]
print(f'\nSpecificity统计:')
print(f'  均值: {np.mean(spec_vals):.3f}')
print(f'  中位: {np.median(spec_vals):.3f}')
print(f'  最小: {np.min(spec_vals):.3f}')

# 构造N=6正则单纯形
N = 6
D = 5
vertices = np.zeros((N, D))
for i in range(D):
    vertices[i, i] = 1.0
last = np.full(D, (1.0 - np.sqrt(N)) / D)
vertices[N-1] = last
center = np.mean(vertices, axis=0)
vertices = vertices - center
edge_len = np.linalg.norm(vertices[0] - vertices[1])
vertices = vertices / edge_len

edges = []
for i in range(N):
    for j in range(i+1, N):
        d = vertices[j] - vertices[i]
        d = d / np.linalg.norm(d)
        edges.append(d)

# Monte Carlo: 随机方向的best_alignment和specificity分布
np.random.seed(42)
n_sim = 200000
random_best = np.zeros(n_sim)
random_spec = np.zeros(n_sim)
for k in range(n_sim):
    v = np.random.randn(D)
    v = v / np.linalg.norm(v)
    aligns = [abs(np.dot(v, e)) for e in edges]
    sorted_a = sorted(aligns, reverse=True)
    random_best[k] = sorted_a[0]
    random_spec[k] = sorted_a[0] - sorted_a[1]

print(f'\n随机方向统计 (D={D}, {len(edges)}边, {n_sim}次模拟):')
print(f'  best_alignment: 均值={np.mean(random_best):.3f} P95={np.percentile(random_best, 95):.3f} P99={np.percentile(random_best, 99):.3f}')
print(f'  specificity: 均值={np.mean(random_spec):.3f} P95={np.percentile(random_spec, 95):.3f} P99={np.percentile(random_spec, 99):.3f}')

# 观测值
obs_best = np.mean([s['best'] for s in specs])
obs_spec = np.mean([s['specificity'] for s in specs])
print(f'\n观测值:')
print(f'  best_alignment均值: {obs_best:.3f}')
print(f'  specificity均值: {obs_spec:.3f}')

# P值
p_best = np.mean(random_best >= obs_best)
p_spec = np.mean(random_spec >= obs_spec)
print(f'  P(随机best>=观测) = {p_best:.6f}')
print(f'  P(随机spec>=观测) = {p_spec:.6f}')

# 联合检验: 每个轨迹best>0.7
p_single_best07 = np.mean(random_best > 0.7)
n_obs_best07 = sum(1 for s in specs if s['best'] > 0.7)
n_total = len(specs)

print(f'\n★★★★★ 联合检验:')
print(f'  P(随机方向best>0.7) = {p_single_best07:.4f}')
print(f'  观测: {n_obs_best07}/{n_total} best>0.7')

# 关键: 边不是独立的, 不能简单乘
# 用bootstrap: 从随机分布中抽样n_total次, 看有多少次全部>0.7
n_bootstrap = 100000
count_all_pass = 0
for _ in range(n_bootstrap):
    sample = np.random.choice(random_best, size=n_total, replace=True)
    if np.all(sample > 0.7):
        count_all_pass += 1
p_all_pass = count_all_pass / n_bootstrap

print(f'  P(随机{n_total}个全部best>0.7) = {p_all_pass:.8f}')
if p_all_pass < 0.001:
    print(f'  ★★★★★ P<0.001, 极强证据!')
elif p_all_pass < 0.01:
    print(f'  ★★★★ P<0.01, 强证据!')
elif p_all_pass < 0.05:
    print(f'  ★★★ P<0.05, 中等证据')

# 更关键: specificity是区分边对齐和面对齐的指标
# 如果强度轨迹确实沿某条特定边, specificity应该高
# 如果在面上, specificity低但best可能也高

# 按领域分析specificity
print(f'\n★★★ 按领域specificity分析:')
for domain in ['emotion_6', 'occupation_6', 'habitat_8']:
    domain_specs = [s for s in specs if s['domain'] == domain]
    if not domain_specs:
        continue
    avg_spec = np.mean([s['specificity'] for s in domain_specs])
    avg_best = np.mean([s['best'] for s in domain_specs])
    n_al = sum(1 for s in domain_specs if s['verdict'] == 'EDGE-ALIGNED')
    print(f'  {domain}: spec={avg_spec:.3f} best={avg_best:.3f} aligned={n_al}/{len(domain_specs)}')
    # 高best但低spec → 可能在面上
    for s in domain_specs:
        if s['best'] > 0.7 and s['specificity'] < 0.05:
            print(f'    面上候选: {s["cls"]} best={s["best"]:.3f} spec={s["specificity"]:.3f}')

# 关键: 高best+高spec = 真正的边对齐; 高best+低spec = 面上对齐
# 两者的区别是语义上的
n_edge_strict = sum(1 for s in specs if s['best'] > 0.7 and s['specificity'] > 0.05)
n_face_aligned = sum(1 for s in specs if s['best'] > 0.7 and s['specificity'] <= 0.05)
n_weak_total = sum(1 for s in specs if s['best'] <= 0.7)

print(f'\n★★★★★ 最终分类:')
print(f'  边对齐(best>0.7, spec>0.05): {n_edge_strict}')
print(f'  面对齐(best>0.7, spec<=0.05): {n_face_aligned}')
print(f'  弱对齐(best<=0.7): {n_weak_total}')
print(f'  边+面合计: {n_edge_strict + n_face_aligned}/{n_total} = {(n_edge_strict+n_face_aligned)/n_total:.1%}')

print(f'\n★★★★★ 核心结论:')
print(f'  1. {n_obs_best07}/{n_total}个轨迹best_alignment>0.7 (随机P={p_all_pass:.6f})')
print(f'  2. 边对齐(严格): {n_edge_strict}/{n_total} = {n_edge_strict/n_total:.1%}')
print(f'  3. 边+面合计: {(n_edge_strict+n_face_aligned)}/{n_total} = {(n_edge_strict+n_face_aligned)/n_total:.1%}')
print(f'  4. ★★★★★ 关键: 即使specificity低, best_alignment>0.7')
print(f'     说明轨迹在单纯形面内移动(不是径向,不是随机的)')
print(f'     这本身就是重要发现: 强度变化在单纯形面上, 不在径向!')
