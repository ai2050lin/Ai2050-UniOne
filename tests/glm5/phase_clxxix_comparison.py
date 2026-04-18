"""
Phase CLXXIX: 4-bit vs FP16 对比分析
======================================
对比GLM-4-9B在4-bit NF4量化和非量化(FP16)下的CLXXIX测试结果差异

数据来源:
- 4-bit: results/phase_clxxix_4bit/glm4_4bit_results.json (本次测试)
- FP16: AGI_GLM5_MEMO.md中记录的之前测试结果
"""

import json
import numpy as np
from pathlib import Path

# ============================================================
# FP16结果 (来自MEMO, Phase CLXXIX之前测试)
# ============================================================

FP16_P791 = {
    # Layer: (W_Q func%, W_K func%, W_V func%, W_O func%)
    0:  (0.26, 0.29, 0.12, 0.13),
    20: (0.12, 0.12, 0.12, 0.12),
    39: (0.13, 0.13, 0.12, 0.13),
}

FP16_P792 = {
    'n_heads': 32,
    'n_kv_heads': 2,
    'functional_heads': 0,
    'content_heads': 32,
    'mixed_heads': 0,
    'top_q_func': 0.14,  # % from MEMO
}

FP16_P793_WO = {
    'func_ratio': 0.12,  # %
    'content_ratio': 99.88,
    'leak_ratio': 854,
}

FP16_P793_FFN = {
    # (func_energy, content_energy, leak_ratio)
    # GLM4 FFN泄漏333-1826x
    'func_energy_range': (0.001, 0.002),
    'content_energy_range': (0.55, 0.64),
    'leak_ratio_range': (333, 1826),
}

FP16_P794 = {
    'status': 'SKIPPED (SDPA不支持output_attentions=True)',
}


# ============================================================
# 4-bit结果
# ============================================================

RESULTS_DIR = Path(r"d:\Ai2050\TransformerLens-Project\results\phase_clxxix_4bit")

def load_4bit_results():
    path = RESULTS_DIR / "glm4_4bit_results.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_p791(fp16_data, bit4_data):
    """P791: Q/K/V与功能/内容空间的对齐度对比"""
    print("\n" + "="*70)
    print("P791: Q/K/V与功能/内容空间的对齐度 — 4-bit vs FP16")
    print("="*70)
    
    bit4_p791 = bit4_data['p791_qkv_alignment']
    
    # 选共同层: 0, 20, 39
    common_layers = [0, 20, 39]
    
    print(f"\n{'Layer':<8} {'Matrix':<6} {'FP16 func%':<14} {'4-bit func%':<14} {'差异':<10} {'相对变化':<10}")
    print("-" * 70)
    
    diffs_summary = {}
    
    for layer in common_layers:
        fp16_vals = fp16_data[layer]
        if str(layer) not in bit4_p791 and layer not in bit4_p791:
            continue
        
        bit4_layer = bit4_p791.get(str(layer), bit4_p791.get(layer, {}))
        
        for i, name in enumerate(['W_Q', 'W_K', 'W_V', 'W_O']):
            fp16_val = fp16_vals[i]
            bit4_val = bit4_layer.get(name, {}).get('func_ratio', 0) * 100  # 转为%
            
            diff = bit4_val - fp16_val
            rel_change = (diff / (fp16_val + 1e-30)) * 100
            
            print(f"{layer:<8} {name:<6} {fp16_val:<14.2f} {bit4_val:<14.4f} {diff:+.4f}%   {rel_change:+.1f}%")
            
            key = f"L{layer}_{name}"
            diffs_summary[key] = {
                'fp16': fp16_val,
                '4bit': bit4_val,
                'diff': diff,
                'rel_change': rel_change,
            }
    
    # 完整4-bit数据 (所有层)
    print(f"\n4-bit完整跨层数据:")
    all_layers = sorted([int(k) for k in bit4_p791.keys() if str(k).isdigit() or isinstance(k, int)])
    for layer in all_layers:
        bit4_layer = bit4_p791.get(str(layer), bit4_p791.get(layer, {}))
        vals = {name: bit4_layer.get(name, {}).get('func_ratio', 0) * 100
                for name in ['W_Q', 'W_K', 'W_V', 'W_O']}
        print(f"  Layer {layer}: Q={vals['W_Q']:.4f}%, K={vals['W_K']:.4f}%, "
              f"V={vals['W_V']:.4f}%, O={vals['W_O']:.4f}%")
    
    return diffs_summary


def compare_p792(fp16_data, bit4_data):
    """P792: 注意力头的功能/内容特化对比"""
    print("\n" + "="*70)
    print("P792: 注意力头的功能/内容特化 — 4-bit vs FP16")
    print("="*70)
    
    bit4_p792 = bit4_data['p792_head_specialization']
    
    print(f"\n{'指标':<25} {'FP16':<15} {'4-bit':<15}")
    print("-" * 55)
    
    # 头数分布
    fp16_func = fp16_data['functional_heads']
    fp16_content = fp16_data['content_heads']
    fp16_mixed = 32 - fp16_func - fp16_content
    
    bit4_func = bit4_p792.get('cross_layer', {}).get('10', bit4_p792.get('cross_layer', {}).get(10, {})).get('functional', 0) if bit4_p792.get('cross_layer') else 0
    bit4_content = bit4_p792.get('cross_layer', {}).get('10', bit4_p792.get('cross_layer', {}).get(10, {})).get('content', 0) if bit4_p792.get('cross_layer') else 0
    bit4_mixed = 32 - bit4_func - bit4_content
    
    # 从heads字段统计
    heads = bit4_p792.get('heads', {})
    bit4_func_h = sum(1 for h in heads.values() if h.get('type') == 'functional')
    bit4_content_h = sum(1 for h in heads.values() if h.get('type') == 'content')
    bit4_mixed_h = sum(1 for h in heads.values() if h.get('type') == 'mixed')
    
    print(f"{'功能头':<25} {fp16_func:<15} {bit4_func_h:<15}")
    print(f"{'内容头':<25} {fp16_content:<15} {bit4_content_h:<15}")
    print(f"{'混合头':<25} {fp16_mixed:<15} {bit4_mixed_h:<15}")
    
    # Top Q功能对齐
    top_q_fp16 = fp16_data['top_q_func']
    if heads:
        top_q_4bit = max(h.get('q_func_ratio', 0) for h in heads.values()) * 100
    else:
        top_q_4bit = 0
    
    print(f"{'Top Q功能对齐(%)':<25} {top_q_fp16:<15.2f} {top_q_4bit:<15.4f}")
    
    # 所有头的Q功能对齐分布
    if heads:
        q_ratios = [h.get('q_func_ratio', 0) * 100 for h in heads.values()]
        print(f"\n4-bit Q功能对齐分布:")
        print(f"  Mean: {np.mean(q_ratios):.4f}%")
        print(f"  Std:  {np.std(q_ratios):.4f}%")
        print(f"  Min:  {np.min(q_ratios):.4f}%")
        print(f"  Max:  {np.max(q_ratios):.4f}%")
    
    # 跨层头分布
    cross_layer = bit4_p792.get('cross_layer', {})
    print(f"\n4-bit跨层头分布:")
    for layer_idx, data in sorted(cross_layer.items()):
        print(f"  Layer {layer_idx}: func={data.get('functional', 0)}, "
              f"content={data.get('content', 0)}, mixed={data.get('mixed', 0)}")


def compare_p793(fp16_data, bit4_data):
    """P793: 功能干预经过注意力层的非线性效应对比"""
    print("\n" + "="*70)
    print("P793: 功能干预经过注意力层的非线性效应 — 4-bit vs FP16")
    print("="*70)
    
    bit4_p793 = bit4_data['p793_intervention']
    
    # W_O输出分布
    print(f"\n--- W_O输出分布 ---")
    print(f"{'指标':<20} {'FP16':<15} {'4-bit(平均)':<15}")
    print("-" * 50)
    
    fp16_wo_func = fp16_data['W_O']['func_ratio']
    fp16_wo_leak = fp16_data['W_O']['leak_ratio']
    
    # 4-bit: 取所有功能方向的平均
    wo_funcs = []
    wo_leaks = []
    for label, data in bit4_p793.items():
        if isinstance(data, dict) and 'after_W_O' in data:
            wo_funcs.append(data['after_W_O']['func_ratio'] * 100)
            wo_leaks.append(data['after_W_O']['leak_ratio'])
    
    bit4_wo_func = np.mean(wo_funcs) if wo_funcs else 0
    bit4_wo_leak = np.mean(wo_leaks) if wo_leaks else 0
    
    print(f"{'功能占比(%)':<20} {fp16_wo_func:<15.2f} {bit4_wo_func:<15.4f}")
    print(f"{'泄漏比':<20} {fp16_wo_leak:<15.0f} {bit4_wo_leak:<15.1f}")
    
    # FFN变换
    print(f"\n--- FFN变换分析 ---")
    print(f"{'指标':<20} {'FP16':<25} {'4-bit(平均)':<15}")
    print("-" * 60)
    
    fp16_ffn_func = fp16_data['FFN']['func_energy_range']
    fp16_ffn_content = fp16_data['FFN']['content_energy_range']
    fp16_ffn_leak = fp16_data['FFN']['leak_ratio_range']
    
    ffn_funcs = []
    ffn_contents = []
    ffn_leaks = []
    for label, data in bit4_p793.items():
        if isinstance(data, dict) and 'after_FFN' in data:
            ffn_funcs.append(data['after_FFN']['func_energy'])
            ffn_contents.append(data['after_FFN']['content_energy'])
            ffn_leaks.append(data['after_FFN']['leak_ratio'])
    
    bit4_ffn_func = (np.min(ffn_funcs), np.max(ffn_funcs)) if ffn_funcs else (0, 0)
    bit4_ffn_content = (np.min(ffn_contents), np.max(ffn_contents)) if ffn_contents else (0, 0)
    bit4_ffn_leak = (np.min(ffn_leaks), np.max(ffn_leaks)) if ffn_leaks else (0, 0)
    
    print(f"{'功能能量':<20} {fp16_ffn_func[0]:.3f}-{fp16_ffn_func[1]:.3f}     "
          f"{bit4_ffn_func[0]:.4f}-{bit4_ffn_func[1]:.4f}")
    print(f"{'内容能量':<20} {fp16_ffn_content[0]:.2f}-{fp16_ffn_content[1]:.2f}     "
          f"{bit4_ffn_content[0]:.4f}-{bit4_ffn_content[1]:.4f}")
    print(f"{'泄漏比':<20} {fp16_ffn_leak[0]:.0f}-{fp16_ffn_leak[1]:.0f}x       "
          f"{bit4_ffn_leak[0]:.1f}-{bit4_ffn_leak[1]:.1f}x")
    
    # 各方向详细对比
    print(f"\n--- 各功能方向FFN泄漏比 ---")
    print(f"{'方向':<12} {'4-bit泄漏比':<15} {'4-bit功能能量':<15} {'4-bit内容能量':<15}")
    print("-" * 60)
    for label, data in bit4_p793.items():
        if isinstance(data, dict) and 'after_FFN' in data:
            ffn = data['after_FFN']
            print(f"{label:<12} {ffn['leak_ratio']:<15.1f} {ffn['func_energy']:<15.4f} "
                  f"{ffn['content_energy']:<15.4f}")


def compare_p794(fp16_data, bit4_data):
    """P794: 注意力模式的功能调制对比"""
    print("\n" + "="*70)
    print("P794: 注意力模式的功能调制 — 4-bit vs FP16")
    print("="*70)
    
    bit4_p794 = bit4_data['p794_attention_modulation']
    
    if fp16_data.get('status') == 'SKIPPED':
        print("\n  FP16版: P794跳过 (SDPA不支持output_attentions=True)")
        print("  4-bit版: 使用eager attention, 可获取注意力权重")
        print("  → 4-bit版首次完成了P794!")
    
    # 打印4-bit结果
    print(f"\n  4-bit注意力模式调制结果:")
    
    # 各功能维度
    modulations = {}
    for key, data in bit4_p794.items():
        if key.endswith('_diff'):
            dim_name = key.replace('_diff', '')
            modulations[dim_name] = data
            print(f"    {dim_name}: top1_diff={data.get('top1_diff', 0):+.4f}, "
                  f"entropy_diff={data.get('entropy_diff', 0):+.4f}, "
                  f"diag_diff={data.get('diag_diff', 0):+.4f}, "
                  f"total={data.get('total_modulation', 0):.4f}")
    
    # 调制强度排序
    if modulations:
        sorted_mods = sorted(modulations.items(), key=lambda x: x[1].get('total_modulation', 0), reverse=True)
        print(f"\n  调制强度排序:")
        for dim, data in sorted_mods:
            print(f"    {dim}: {data.get('total_modulation', 0):.4f}")
        
        # 分析调制模式
        print(f"\n  关键发现:")
        strongest = sorted_mods[0]
        weakest = sorted_mods[-1]
        print(f"    最强调制维度: {strongest[0]} (modulation={strongest[1].get('total_modulation', 0):.4f})")
        print(f"    最弱调制维度: {weakest[0]} (modulation={weakest[1].get('total_modulation', 0):.4f})")
        print(f"    强弱比: {strongest[1].get('total_modulation', 0) / (weakest[1].get('total_modulation', 0) + 1e-30):.1f}x")
        
        # 极性vs其他
        if 'polarity' in modulations:
            pol = modulations['polarity']
            print(f"    极性(polarity)调制: top1={pol.get('top1_diff', 0):+.4f}, "
                  f"entropy={pol.get('entropy_diff', 0):+.4f}, diag={pol.get('diag_diff', 0):+.4f}")
            print(f"    → 极性变化导致注意力分散度增加(entropy↑), 集中度下降(top1↓)")


def generate_comparison_summary():
    """生成综合对比总结"""
    print("\n" + "="*70)
    print("综合对比总结: GLM-4-9B 4-bit vs FP16 (Phase CLXXIX)")
    print("="*70)
    
    bit4_data = load_4bit_results()
    
    # 1. P791对比
    p791_diffs = compare_p791(FP16_P791, bit4_data)
    
    # 2. P792对比
    compare_p792(FP16_P792, bit4_data)
    
    # 3. P793对比
    compare_p793({'W_O': FP16_P793_WO, 'FFN': FP16_P793_FFN}, bit4_data)
    
    # 4. P794对比
    compare_p794(FP16_P794, bit4_data)
    
    # 综合分析
    print("\n" + "="*70)
    print("量化影响分析")
    print("="*70)
    
    print("""
1. P791 对齐度:
   - FP16: 0.12%-0.29%, 4-bit: 0.12%-0.28%
   - 量化后功能对齐度基本不变 (差异<0.01%)
   - 结论: 4-bit量化对Q/K/V权重矩阵的功能空间占比影响极小

2. P792 头特化:
   - FP16和4-bit: 都是32/32内容头, 0功能头
   - 结论: 4-bit量化不改变注意力头的功能/内容分化模式

3. P793 W_O泄漏:
   - FP16: 泄漏854x, 4-bit: 泄漏856x
   - 结论: W_O的功能→内容泄漏比在量化后几乎不变

4. P793 FFN泄漏:
   - FP16: 泄漏333-1826x (功能能量0.001-0.002)
   - 4-bit: 泄漏45-60x (功能能量0.0002, 内容能量0.01)
   - ★ 这是最大差异! 4-bit的FFN泄漏比降低了约10-30倍
   - 但4-bit的FFN输出norm也小了约5-6倍(0.10 vs 0.55-0.64)
   - 归一化后: 4-bit FFN功能保留率更高

5. P794 注意力调制:
   - FP16版跳过, 4-bit版首次完成(使用eager attention)
   - 极性(polarity)调制最强(0.1275), 时态(tense)最弱(0.0084)
   - 极性变化使注意力显著分散(entropy↑0.06, top1↓0.02)
    """)


if __name__ == "__main__":
    generate_comparison_summary()
