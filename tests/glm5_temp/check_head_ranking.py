"""检查每个模型关键层的head alignment详细排名"""
import json

# DS7B L27
print("="*60)
print("DS7B L27 head alignment ranking:")
d = json.load(open('results/causal_fiber/deepseek7b_direct_head/s1_direct_head.json'))
l27 = d['L27']
heads = l27['all_heads']
top = sorted(heads.items(), key=lambda x: x[1]['alignment'], reverse=True)[:10]
for n, v in top:
    feat_str = ', '.join(f"{k}={val['mean']:.3f}" for k, val in sorted(v['feature_cos'].items(), key=lambda x: -x[1]['mean'])[:3])
    print(f"  {n}: align={v['alignment']:.4f}, norm={v['mean_norm']:.1f} | {feat_str}")

# DS7B 各层top alignment对比
print("\nDS7B 各层 top alignment:")
for lk in sorted(d.keys()):
    ld = d[lk]
    if 'error' in ld:
        continue
    top_h = sorted(ld['all_heads'].items(), key=lambda x: x[1]['alignment'], reverse=True)[:1]
    if top_h:
        n, v = top_h[0]
        print(f"  {lk}: {n} align={v['alignment']:.4f}")

# Qwen3 L35
print("\n" + "="*60)
print("Qwen3 L35 head alignment ranking:")
d = json.load(open('results/causal_fiber/qwen3_direct_head/s1_direct_head.json'))
l35 = d['L35']
heads = l35['all_heads']
top = sorted(heads.items(), key=lambda x: x[1]['alignment'], reverse=True)[:10]
for n, v in top:
    feat_str = ', '.join(f"{k}={val['mean']:.3f}" for k, val in sorted(v['feature_cos'].items(), key=lambda x: -x[1]['mean'])[:3])
    print(f"  {n}: align={v['alignment']:.4f}, norm={v['mean_norm']:.1f} | {feat_str}")

# Qwen3 各层top alignment对比
print("\nQwen3 各层 top alignment:")
for lk in sorted(d.keys()):
    ld = d[lk]
    if 'error' in ld:
        continue
    top_h = sorted(ld['all_heads'].items(), key=lambda x: x[1]['alignment'], reverse=True)[:1]
    if top_h:
        n, v = top_h[0]
        print(f"  {lk}: {n} align={v['alignment']:.4f}")

# GLM4 L32
print("\n" + "="*60)
print("GLM4 L32 head alignment ranking:")
d = json.load(open('results/causal_fiber/glm4_direct_head/s1_direct_head.json'))
l32 = d['L32']
heads = l32['all_heads']
top = sorted(heads.items(), key=lambda x: x[1]['alignment'], reverse=True)[:10]
for n, v in top:
    feat_str = ', '.join(f"{k}={val['mean']:.3f}" for k, val in sorted(v['feature_cos'].items(), key=lambda x: -x[1]['mean'])[:3])
    print(f"  {n}: align={v['alignment']:.4f}, norm={v['mean_norm']:.1f} | {feat_str}")

# GLM4 各层top alignment对比
print("\nGLM4 各层 top alignment:")
for lk in sorted(d.keys()):
    ld = d[lk]
    if 'error' in ld:
        print(f"  {lk}: {ld['error']}")
        continue
    top_h = sorted(ld['all_heads'].items(), key=lambda x: x[1]['alignment'], reverse=True)[:1]
    if top_h:
        n, v = top_h[0]
        print(f"  {lk}: {n} align={v['alignment']:.4f}")
