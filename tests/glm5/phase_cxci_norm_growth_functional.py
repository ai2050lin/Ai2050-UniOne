"""
Phase CXCI-CXCII: Norm增长的功能意义
=====================================
核心问题: norm增长是"有意义的"还是"噪声放大"?

CXCI: norm增长是否承载功能信号?
- P893: 逐层PCA验证 — norm增长方向的方差是否编码功能
- P894: norm增长方向与W_U解码方向的对齐度

CXCII: norm增长与功能信号的定量关系
- P895: 信号转移假设 — 功能信号是否从"方向信号"转移到"幅度信号"
- P896: 低频vs高频维度的norm增长差异
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_model, release_model, get_layers, get_model_info, get_W_U

# 功能对 (与CXC相同)
FUNC_PAIRS = {
    'syntax': [
        ("The cat sits quietly", "Cat the sits quietly"),
        ("I am very happy", "I is very happy"),
        ("He can run fast", "He can runs fast"),
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),
        ("Big house near the lake", "Small house near the lake"),
        ("Red apple on the table", "Green apple on the table"),
    ],
    'polarity': [
        ("She is very happy today", "She is very sad today"),
        ("Good result from the test", "Bad result from the test"),
        ("Love the beautiful place", "Hate the beautiful place"),
    ],
    'logic': [
        ("Because it rained, the ground is wet", "It rained, but the ground is dry"),
        ("Since she studied hard, she passed", "She studied hard, but she failed"),
        ("Due to the heat, the ice melted", "Despite the heat, the ice stayed frozen"),
        ("If it rains, then the ground gets wet, and it rained",
         "If it rains, then the ground gets wet, but it didn't rain"),
        ("First she cooked dinner, then she ate it", "First she ate dinner, then she cooked it"),
        ("After the rain stopped, the sun came out", "Before the rain stopped, the sun came out"),
    ],
}


def get_residual_at_layer(model, tokenizer, device, text, layer_idx):
    """获取某层残差流的最后一个token表示"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().float().cpu()
        else:
            captured['hs'] = output.detach().float().cpu()
    
    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    if 'hs' in captured:
        return captured['hs'][0, -1, :].numpy()
    return None


def analyze_norm_direction(vecs_a, vecs_b, W_U_sv, n_freq_bands=8):
    """分析norm增长方向与功能信号的关系"""
    deltas = [a - b for a, b in zip(vecs_a, vecs_b)]
    mean_delta = np.mean(deltas, axis=0)
    delta_norm = np.linalg.norm(mean_delta)
    
    if delta_norm < 1e-10:
        return None
    
    # 归一化方向
    delta_dir = mean_delta / delta_norm
    
    # 与W_U奇异向量的对齐
    n_sv = min(W_U_sv.shape[1], 50)
    cos_sv = [np.dot(delta_dir, W_U_sv[:, i]) for i in range(n_sv)]
    
    # 频段分析: 将维度分成低频/中频/高频
    d = len(mean_delta)
    band_size = d // n_freq_bands
    band_energies = []
    for b in range(n_freq_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_freq_bands - 1 else d
        band_energy = np.linalg.norm(mean_delta[start:end])
        band_energies.append(band_energy)
    
    # 归一化频段能量
    total_energy = sum(band_energies)
    band_ratios = [e / total_energy for e in band_energies] if total_energy > 0 else [0] * n_freq_bands
    
    return {
        'delta_norm': float(delta_norm),
        'cos_sv_top10': [float(c) for c in cos_sv[:10]],
        'cos_sv_max': float(max(abs(c) for c in cos_sv)),
        'cos_sv_argmax': int(np.argmax([abs(c) for c in cos_sv])),
        'band_energies': [float(e) for e in band_energies],
        'band_ratios': [float(r) for r in band_ratios],
    }


def signal_transfer_test(all_layer_vecs, func_name, sample_layers):
    """P895: 信号转移测试 — 功能信号是否从方向转移到幅度"""
    results = {}
    
    for layer_idx in sample_layers:
        if func_name not in all_layer_vecs.get(layer_idx, {}):
            continue
        vecs_a, vecs_b = all_layer_vecs[layer_idx][func_name]
        
        # 方向信号: delta的归一化方向（与内容无关的纯方向差异）
        deltas = [a - b for a, b in zip(vecs_a, vecs_b)]
        mean_delta = np.mean(deltas, axis=0)
        delta_norm = np.linalg.norm(mean_delta)
        
        # 幅度信号: norm的增长
        norms_a = [np.linalg.norm(v) for v in vecs_a]
        norms_b = [np.linalg.norm(v) for v in vecs_b]
        mean_norm = (np.mean(norms_a) + np.mean(norms_b)) / 2
        
        # 方向信号强度 = delta_norm / mean_norm (与CXC一致)
        direction_signal = delta_norm / mean_norm if mean_norm > 0 else 0
        
        # 幅度信号强度 = |norm_a - norm_b| / mean_norm
        norm_diff = abs(np.mean(norms_a) - np.mean(norms_b))
        amplitude_signal = norm_diff / mean_norm if mean_norm > 0 else 0
        
        # PCA有效维度
        all_vecs = np.vstack([vecs_a + vecs_b])
        if len(all_vecs) >= 3:
            pca = PCA()
            pca.fit(all_vecs)
            var_ratio = pca.explained_variance_ratio_
            cumvar = np.cumsum(var_ratio)
            eff_dim = np.searchsorted(cumvar, 0.95) + 1
        else:
            eff_dim = 0
        
        results[layer_idx] = {
            'direction_signal': float(direction_signal),
            'amplitude_signal': float(amplitude_signal),
            'mean_norm': float(mean_norm),
            'delta_norm': float(delta_norm),
            'eff_dim_95': int(eff_dim),
            'norm_a_mean': float(np.mean(norms_a)),
            'norm_b_mean': float(np.mean(norms_b)),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'glm4', 'deepseek7b'])
    args = parser.parse_args()
    
    log_path = f'tmp/cxci_{args.model}.log'
    os.makedirs('tmp', exist_ok=True)
    log_file = open(log_path, 'w', buffering=1)
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    model, tokenizer, device = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    print(f'Model: {args.model}, L={n_layers}, d={d_model}', flush=True)
    
    # 采样层
    sample_layers = list(range(0, n_layers, 2))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    # 获取W_U的SVD
    print('Computing W_U SVD...', flush=True)
    W_U = get_W_U(model)  # [vocab_size, d_model]
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)
    k = min(200, min(W_U_T.shape) - 2)
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    sort_idx = np.argsort(s_wut)[::-1]
    U_wut = U_wut[:, sort_idx]
    s_wut = s_wut[sort_idx]
    print(f'W_U SVD done: top-10 singular values = {s_wut[:10].tolist()}', flush=True)
    
    results = {
        'model': args.model,
        'n_layers': n_layers,
        'd_model': d_model,
        'P893_norm_direction_WU_alignment': {},
        'P894_freq_band_analysis': {},
        'P895_signal_transfer': {},
        'P896_norm_growth_summary': {},
    }
    
    # 收集所有层的所有功能对向量
    print('\n=== Collecting residual vectors ===', flush=True)
    all_layer_vecs = {}
    
    for layer_idx in sample_layers:
        all_layer_vecs[layer_idx] = {}
        for func_name, pairs in FUNC_PAIRS.items():
            vecs_a = []
            vecs_b = []
            for text_a, text_b in pairs:
                va = get_residual_at_layer(model, tokenizer, device, text_a, layer_idx)
                vb = get_residual_at_layer(model, tokenizer, device, text_b, layer_idx)
                if va is not None and vb is not None:
                    vecs_a.append(va)
                    vecs_b.append(vb)
            if len(vecs_a) >= 2:
                all_layer_vecs[layer_idx][func_name] = (vecs_a, vecs_b)
    
    # ===== P893: norm增长方向与W_U对齐 =====
    print('\n=== P893: Norm Direction vs W_U Alignment ===', flush=True)
    
    for layer_idx in sample_layers:
        layer_results = {}
        for func_name in FUNC_PAIRS.keys():
            if func_name not in all_layer_vecs.get(layer_idx, {}):
                continue
            vecs_a, vecs_b = all_layer_vecs[layer_idx][func_name]
            analysis = analyze_norm_direction(vecs_a, vecs_b, U_wut)
            if analysis is not None:
                layer_results[func_name] = analysis
        
        results['P893_norm_direction_WU_alignment'][str(layer_idx)] = layer_results
        
        if layer_idx % 6 == 0 or layer_idx == n_layers - 1:
            wu_str = {k: f"cos_max={v['cos_sv_max']:.4f}" for k, v in layer_results.items()}
            print(f'L{layer_idx}: {wu_str}', flush=True)
    
    # ===== P894: 频段分析 =====
    print('\n=== P894: Frequency Band Analysis ===', flush=True)
    
    for func_name in FUNC_PAIRS.keys():
        func_band_data = []
        for layer_idx in sample_layers:
            layer_data = results['P893_norm_direction_WU_alignment'].get(str(layer_idx), {})
            if func_name in layer_data:
                func_band_data.append({
                    'layer': layer_idx,
                    'band_ratios': layer_data[func_name]['band_ratios'],
                })
        
        results['P894_freq_band_analysis'][func_name] = func_band_data
        
        if func_band_data:
            # 比较L0 vs 末端的频段分布
            l0_bands = func_band_data[0]['band_ratios']
            last_bands = func_band_data[-1]['band_ratios']
            print(f'{func_name}: L0 low={l0_bands[0]:.3f} high={l0_bands[-1]:.3f} | '
                  f'Last low={last_bands[0]:.3f} high={last_bands[-1]:.3f}', flush=True)
    
    # ===== P895: 信号转移测试 =====
    print('\n=== P895: Signal Transfer Test ===', flush=True)
    
    for func_name in FUNC_PAIRS.keys():
        transfer = signal_transfer_test(all_layer_vecs, func_name, sample_layers)
        results['P895_signal_transfer'][func_name] = {str(k): v for k, v in transfer.items()}
        
        if transfer:
            l0_data = transfer.get(0, {})
            last_data = transfer.get(n_layers - 1, {})
            dir_l0 = l0_data.get('direction_signal', 0)
            dir_last = last_data.get('direction_signal', 0)
            amp_l0 = l0_data.get('amplitude_signal', 0)
            amp_last = last_data.get('amplitude_signal', 0)
            print(f'{func_name}: dir_signal L0={dir_l0:.4f}→Last={dir_last:.4f} | '
                  f'amp_signal L0={amp_l0:.4f}→Last={amp_last:.4f}', flush=True)
    
    # ===== P896: Norm增长总结 =====
    print('\n=== P896: Norm Growth Summary ===', flush=True)
    
    norm_growth_data = {}
    for layer_idx in sample_layers:
        layer_norms = {}
        for func_name in FUNC_PAIRS.keys():
            if func_name not in all_layer_vecs.get(layer_idx, {}):
                continue
            vecs_a, vecs_b = all_layer_vecs[layer_idx][func_name]
            norms_a = [np.linalg.norm(v) for v in vecs_a]
            norms_b = [np.linalg.norm(v) for v in vecs_b]
            layer_norms[func_name] = {
                'mean_norm': float((np.mean(norms_a) + np.mean(norms_b)) / 2),
                'norm_a_mean': float(np.mean(norms_a)),
                'norm_b_mean': float(np.mean(norms_b)),
            }
        
        norm_growth_data[str(layer_idx)] = layer_norms
        
        if layer_idx % 6 == 0 or layer_idx == n_layers - 1:
            norm_str = {k: f"norm={v['mean_norm']:.1f}" for k, v in layer_norms.items()}
            print(f'L{layer_idx}: {norm_str}', flush=True)
    
    results['P896_norm_growth_summary'] = norm_growth_data
    
    # ===== 综合分析 =====
    print('\n=== COMPREHENSIVE ANALYSIS ===', flush=True)
    
    # 1. Norm增长 vs 功能信号衰减
    print('\n1. Norm Growth vs Signal Decay:', flush=True)
    for func_name in FUNC_PAIRS.keys():
        transfer = results['P895_signal_transfer'].get(func_name, {})
        if not transfer:
            continue
        
        norms = []
        dir_signals = []
        for layer_idx in sample_layers:
            td = transfer.get(str(layer_idx), {})
            if td:
                norms.append(td['mean_norm'])
                dir_signals.append(td['direction_signal'])
        
        if len(norms) > 2:
            norm_corr = np.corrcoef(norms, dir_signals)[0, 1]
            print(f'  {func_name}: norm-dir_signal corr = {norm_corr:.3f}', flush=True)
            
            if norm_corr < -0.5:
                print(f'    >> Norm增长伴随方向信号衰减 → 支持"稀释假说"', flush=True)
            elif norm_corr > 0.5:
                print(f'    >> Norm增长伴随方向信号增强 → 支持"信号增强假说"', flush=True)
            else:
                print(f'    >> 无显著相关 → 独立过程', flush=True)
    
    # 2. W_U对齐度随层变化
    print('\n2. W_U Alignment vs Layer:', flush=True)
    for func_name in FUNC_PAIRS.keys():
        cos_values = []
        for layer_idx in sample_layers:
            ld = results['P893_norm_direction_WU_alignment'].get(str(layer_idx), {})
            if func_name in ld:
                cos_values.append((layer_idx, ld[func_name]['cos_sv_max']))
        
        if len(cos_values) > 2:
            # 趋势: W_U对齐度是否随层增加?
            layers_arr = [c[0] for c in cos_values]
            cos_arr = [c[1] for c in cos_values]
            slope, intercept, r, p, se = stats.linregress(layers_arr, cos_arr)
            print(f'  {func_name}: W_U alignment slope={slope:.6f}, r={r:.3f}, p={p:.4f}', flush=True)
    
    # 3. 信号转移结论
    print('\n3. Signal Transfer Conclusion:', flush=True)
    for func_name in FUNC_PAIRS.keys():
        transfer = results['P895_signal_transfer'].get(func_name, {})
        if not transfer:
            continue
        
        l0 = transfer.get('0', {})
        last = transfer.get(str(n_layers - 1), {})
        
        dir_ratio = last.get('direction_signal', 0) / l0.get('direction_signal', 1)
        amp_ratio = last.get('amplitude_signal', 0) / l0.get('amplitude_signal', 1e-10)
        norm_ratio = last.get('mean_norm', 1) / l0.get('mean_norm', 1)
        
        print(f'  {func_name}: dir_signal ratio={dir_ratio:.3f}, '
              f'amp_signal ratio={amp_ratio:.3f}, norm_ratio={norm_ratio:.1f}', flush=True)
    
    # Save results
    out_dir = Path('results/phase_cxci')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {out_path}', flush=True)
    
    release_model(model)
    print(f'\nPhase CXCI PASSED for {args.model}', flush=True)


if __name__ == '__main__':
    main()
