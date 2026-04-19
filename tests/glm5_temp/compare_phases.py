"""对比Phase CCIV (8bit hook) 和 Phase CCV (direct head) 的结果"""
import json

# Phase CCIV: 8bit W_o投影法
print("="*70)
print("Phase CCIV (8bit W_o projection): DS7B S2 head hook")
print("="*70)
try:
    with open('results/causal_fiber/deepseek7b_8bit/s2_results.json') as f:
        d = json.load(f)
    for lk in sorted(d.keys()):
        ld = d[lk]
        if 'error' in ld:
            print(f'  {lk}: {ld["error"]}')
            continue
        top_heads = ld.get('top_heads', [])
        for h_name, cos in top_heads[:3]:
            print(f'  {lk}: {h_name} cos={cos:.4f}')
except Exception as e:
    print(f'Error: {e}')

print()
print("="*70)
print("Phase CCV (direct head output): DS7B S1 head alignment")
print("="*70)
with open('results/causal_fiber/deepseek7b_direct_head/s1_direct_head.json') as f:
    d = json.load(f)
for lk in sorted(d.keys()):
    ld = d[lk]
    if 'error' in ld:
        print(f'  {lk}: {ld["error"]}')
        continue
    top_heads = ld.get('top_heads', [])
    for h_name, alignment, norm in top_heads[:3]:
        print(f'  {lk}: {h_name} align={alignment:.4f}')

# 对比Qwen3
print()
print("="*70)
print("Phase CCIV (8bit W_o projection): Qwen3 S2 head hook")
print("="*70)
try:
    # Qwen3的8bit结果可能在qwen3_hook目录
    import os
    qwen3_dirs = [d for d in os.listdir('results/causal_fiber') if 'qwen3' in d]
    print(f'Available qwen3 dirs: {qwen3_dirs}')
    # 尝试qwen3_hook
    for dname in ['qwen3_hook', 'qwen3_megasample']:
        p = f'results/causal_fiber/{dname}/s2_results.json'
        if os.path.exists(p):
            print(f'\n  {dname}:')
            with open(p) as f:
                d = json.load(f)
            for lk in sorted(d.keys()):
                ld = d[lk]
                if isinstance(ld, dict) and 'top_heads' in ld:
                    for h_name, cos in ld['top_heads'][:3]:
                        print(f'    {lk}: {h_name} cos={cos:.4f}')
except Exception as e:
    print(f'Error: {e}')

print()
print("="*70)
print("Phase CCV (direct head output): Qwen3 S1 head alignment")
print("="*70)
with open('results/causal_fiber/qwen3_direct_head/s1_direct_head.json') as f:
    d = json.load(f)
for lk in sorted(d.keys()):
    ld = d[lk]
    if 'error' in ld:
        print(f'  {lk}: {ld["error"]}')
        continue
    top_heads = ld.get('top_heads', [])
    for h_name, alignment, norm in top_heads[:3]:
        print(f'  {lk}: {h_name} align={alignment:.4f}')
