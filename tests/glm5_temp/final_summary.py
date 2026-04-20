"""汇总Phase CCVI严格验证结果"""
import json

print("="*70)
print("Phase CCVI: 严格验证 — 双方法对比 + 特征分解 + Bootstrap")
print("="*70)

for model in ['deepseek7b', 'qwen3', 'glm4']:
    print(f"\n{'='*70}")
    print(f"  {model.upper()}")
    print(f"{'='*70}")
    
    # S4: 层间趋势
    try:
        with open(f'results/causal_fiber/{model}_rigorous/s4_layer_trend.json') as f:
            s4 = json.load(f)
        print("\n  S4 层间趋势 (best_single = 单特征alignment均值, best_cross = 跨特征alignment):")
        for lk in sorted(s4.keys()):
            ld = s4[lk]
            if 'error' in ld:
                print(f"    {lk}: {ld['error']}")
            else:
                print(f"    {lk}: single={ld['best_single_feature_alignment']:.4f}, cross={ld['best_cross_feature_alignment']:.4f}, head={ld['best_head']}")
    except:
        pass
    
    # S1: 双方法对比
    try:
        with open(f'results/causal_fiber/{model}_rigorous/s1_dual_method.json') as f:
            s1 = json.load(f)
        print("\n  S1 双方法对比 (direct=直接head输出, projected=W_o投影):")
        for lk in sorted(s1.keys()):
            ld = s1[lk]
            if 'error' in ld:
                print(f"    {lk}: {ld['error']}")
            else:
                top = ld.get('sorted_by_direct', [])
                if top:
                    h_name, direct, proj = top[0]
                    print(f"    {lk}: top={h_name} direct={direct:.4f} projected={proj:.4f} (差={proj-direct:.4f})")
    except:
        pass
    
    # S2: 特征分解
    try:
        with open(f'results/causal_fiber/{model}_rigorous/s2_feature_decomposed.json') as f:
            s2 = json.load(f)
        print("\n  S2 特征分解 (top head的per-feature alignment):")
        for lk in sorted(s2.keys()):
            ld = s2[lk]
            if 'error' in ld:
                print(f"    {lk}: {ld['error']}")
            else:
                sorted_heads = ld.get('sorted_by_single_feature', [])
                if sorted_heads:
                    h_name, cross, avg_single = sorted_heads[0]
                    # 获取per-feature details
                    all_heads = ld.get('all_heads', {})
                    h_data = all_heads.get(h_name, {})
                    feat_aligns = h_data.get('feature_alignments', {})
                    feat_str = ', '.join(f"{k}={v['alignment']:.3f}" for k, v in sorted(feat_aligns.items(), key=lambda x: -x[1]['alignment']))
                    print(f"    {lk}: top={h_name} cross={cross:.4f} avg_single={avg_single:.4f}")
                    print(f"         {feat_str}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # S3: Bootstrap
    try:
        with open(f'results/causal_fiber/{model}_rigorous/s3_bootstrap.json') as f:
            s3 = json.load(f)
        print("\n  S3 Bootstrap (95% CI for top head):")
        for lk in sorted(s3.keys()):
            ld = s3[lk]
            if 'error' in ld:
                print(f"    {lk}: {ld['error']}")
            else:
                top_heads = ld.get('top_heads', [])
                if top_heads:
                    h_name, boot_data = top_heads[0]
                    for fname, bd in sorted(boot_data.items(), key=lambda x: -x[1]['observed'])[:3]:
                        sig = "YES" if bd['ci_low'] > 0.3 else "no"
                        print(f"    {lk} {h_name} {fname}: {bd['observed']:.4f} [{bd['ci_low']:.4f}, {bd['ci_high']:.4f}] sig>{0.3}:{sig}")
    except Exception as e:
        print(f"    Error: {e}")
