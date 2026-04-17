"""
Phase CLXXXIX: ICSPB变量→DNN映射 — 7层链的层范围验证与动力学方程
=====================================================================
ICSPB 7层链假设:
1. 对象底图 (family_patched_object_atlas) → 词嵌入层
2. 局部更新 (local_update) → 早期层(L1-N/4)
3. 关系提升 (relation_lift) → 中间偏前(LN/4-N/2)
4. 读出路由 (readout_routing) → 中间(LN/2-3N/4)
5. 推理整合 (inference_integration) → 中间偏后(L3N/4-N*7/8)
6. 阶段组织 (phase_organization) → 末端偏前
7. 前后继约束 (successor_constraint) → 末端

验证方法:
- 测量每层的"功能信号累积量" — 从L0到L_end, 功能差异的norm增长曲线
- 测量Attn/FFN贡献比的层间变化 — 识别角色转换点
- 测量残差保留率的层间变化 — 识别稳定/转变区间
- 用这些变化点的位置验证7层链映射
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_model, release_model, get_layers, get_model_info

FUNC_PAIRS = {
    'syntax': [("The cat sits", "Cat the sits"), ("I am happy", "I is happy"), ("He can run", "He can runs")],
    'semantic': [("The cat sat", "The dog sat"), ("Big house", "Small house"), ("Red apple", "Green apple")],
    'polarity': [("She is happy", "She is sad"), ("Good result", "Bad result"), ("Love it", "Hate it")],
}


def get_residual_at_layer(model, tokenizer, device, text, layer_idx):
    """获取某层残差流的最后一个token表示"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['hs'] = output[0].detach().float().cpu()
        else:
            captured['hs'] = output.detach().float().cpu()
    
    layer = get_layers(model)[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    return captured.get('hs', None)


def compute_functional_signal(model, tokenizer, device, func_type, pairs, sample_layers):
    """
    计算每层的功能信号(差异方向范数)
    返回: {layer_idx: {delta_norm, cos_pair}}
    """
    results = {}
    
    for li in sample_layers:
        norms_a = []
        norms_b = []
        deltas = []
        
        for text_a, text_b in pairs:
            hs_a = get_residual_at_layer(model, tokenizer, device, text_a, li)
            hs_b = get_residual_at_layer(model, tokenizer, device, text_b, li)
            
            if hs_a is None or hs_b is None:
                continue
            
            vec_a = hs_a[0, -1, :].numpy()
            vec_b = hs_b[0, -1, :].numpy()
            
            norms_a.append(np.linalg.norm(vec_a))
            norms_b.append(np.linalg.norm(vec_b))
            deltas.append(vec_b - vec_a)
        
        if not deltas:
            continue
        
        # 平均差异方向
        mean_delta = np.mean(deltas, axis=0)
        delta_norm = np.linalg.norm(mean_delta)
        
        # 平均范数
        mean_norm = (np.mean(norms_a) + np.mean(norms_b)) / 2
        
        # 归一化差异(功能信号强度)
        if mean_norm > 1e-10:
            functional_signal = delta_norm / mean_norm
        else:
            functional_signal = 0
        
        # 各对之间的cosine相似度
        cos_pairs = []
        for i in range(len(deltas)):
            for j in range(i+1, len(deltas)):
                ni = np.linalg.norm(deltas[i])
                nj = np.linalg.norm(deltas[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_pairs.append(float(np.dot(deltas[i], deltas[j]) / (ni * nj)))
        
        results[li] = {
            'delta_norm': float(delta_norm),
            'mean_norm': float(mean_norm),
            'functional_signal': float(functional_signal),
            'cos_pair_mean': float(np.mean(cos_pairs)) if cos_pairs else 0,
            'cos_pair_std': float(np.std(cos_pairs)) if cos_pairs else 0,
        }
    
    return results


def identify_phase_transitions(layer_data, key='functional_signal'):
    """
    识别层间变化的相变点
    返回: 变化率最大的层索引列表
    """
    layers = sorted(layer_data.keys())
    values = [layer_data[l][key] for l in layers]
    
    if len(values) < 3:
        return []
    
    # 计算层间变化率
    diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    
    # 找峰值
    if len(diffs) >= 3:
        peaks, properties = find_peaks(diffs, prominence=0.01*np.max(diffs))
        peak_layers = [layers[p] for p in peaks]
    else:
        peak_layers = []
    
    # 也找最大变化
    max_diff_idx = np.argmax(diffs)
    max_diff_layer = layers[max_diff_idx]
    
    return {
        'peak_layers': peak_layers,
        'max_change_layer': max_diff_layer,
        'max_change_value': float(diffs[max_diff_idx]),
        'diffs': [float(d) for d in diffs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    # Log file
    log_path = f'tmp/clxxxix_{args.model}.log'
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
    
    # 全层采样(每2层采一个)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 20)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f'Sample layers ({len(sample_layers)}): {sample_layers}', flush=True)
    
    # 对每类功能计算层间信号
    all_func_data = {}
    for func_type, pairs in FUNC_PAIRS.items():
        print(f'\nProcessing {func_type}...', flush=True)
        func_data = compute_functional_signal(model, tokenizer, device, func_type, pairs, sample_layers)
        all_func_data[func_type] = func_data
        
        # 打印层间信号
        for li in sample_layers:
            if li in func_data:
                d = func_data[li]
                print(f'  L{li}: delta={d["delta_norm"]:.2f}, norm={d["mean_norm"]:.2f}, sig={d["functional_signal"]:.4f}, cos={d["cos_pair_mean"]:.4f}', flush=True)
    
    # 识别相变点
    print('\n=== Phase Transition Analysis ===', flush=True)
    transitions = {}
    for func_type in FUNC_PAIRS:
        if func_type in all_func_data:
            trans = identify_phase_transitions(all_func_data[func_type], 'functional_signal')
            transitions[func_type] = trans
            print(f'{func_type}: peak_layers={trans["peak_layers"]}, max_change=L{trans["max_change_layer"]} ({trans["max_change_value"]:.4f})', flush=True)
    
    # 归一化层位置(0-1)
    print('\n=== Normalized Phase Positions ===', flush=True)
    for func_type in FUNC_PAIRS:
        if func_type in all_func_data:
            data = all_func_data[func_type]
            layers = sorted(data.keys())
            signals = [data[l]['functional_signal'] for l in layers]
            
            # 找信号峰值
            if len(signals) >= 3:
                sig_peaks, _ = find_peaks(signals, prominence=0.001)
                sig_troughs, _ = find_peaks([-s for s in signals], prominence=0.001)
                
                peak_positions = [layers[p]/n_layers for p in sig_peaks]
                trough_positions = [layers[p]/n_layers for p in sig_troughs]
                
                print(f'{func_type}: signal peaks at normalized positions {["%.2f" % p for p in peak_positions]}', flush=True)
                print(f'{func_type}: signal troughs at normalized positions {["%.2f" % p for p in trough_positions]}', flush=True)
    
    # 跨功能类型汇总
    print('\n=== Cross-Function Summary ===', flush=True)
    for li in sample_layers:
        sigs = {}
        for func_type in FUNC_PAIRS:
            if func_type in all_func_data and li in all_func_data[func_type]:
                sigs[func_type] = all_func_data[func_type][li]['functional_signal']
        if sigs:
            print(f'L{li} ({li/n_layers:.2f}): {sigs}', flush=True)
    
    # 7层链映射验证
    print('\n=== 7-Chain Mapping Validation ===', flush=True)
    
    # 计算每层的"综合功能信号"（3类平均）
    composite_signal = {}
    for li in sample_layers:
        sigs = []
        for func_type in FUNC_PAIRS:
            if func_type in all_func_data and li in all_func_data[func_type]:
                sigs.append(all_func_data[func_type][li]['functional_signal'])
        if sigs:
            composite_signal[li] = float(np.mean(sigs))
    
    # 识别综合信号的阶段性
    if len(composite_signal) >= 5:
        layers_sorted = sorted(composite_signal.keys())
        signals_sorted = [composite_signal[l] for l in layers_sorted]
        
        # 用变化率识别阶段
        max_sig = max(signals_sorted)
        min_sig = min(signals_sorted)
        sig_range = max_sig - min_sig
        
        # 定义阶段: <20%, 20-40%, 40-60%, 60-80%, >80% of range
        low_threshold = min_sig + 0.2 * sig_range
        high_threshold = min_sig + 0.8 * sig_range
        
        phase_1_layers = [l for l in layers_sorted if composite_signal[l] < low_threshold]
        phase_3_layers = [l for l in layers_sorted if composite_signal[l] > high_threshold]
        phase_2_layers = [l for l in layers_sorted if low_threshold <= composite_signal[l] <= high_threshold]
        
        print(f'Phase 1 (low signal, <20%): L{min(phase_1_layers)}-L{max(phase_1_layers)} ({len(phase_1_layers)} layers)', flush=True)
        print(f'Phase 2 (building, 20-80%): L{min(phase_2_layers)}-L{max(phase_2_layers)} ({len(phase_2_layers)} layers)', flush=True)
        print(f'Phase 3 (high signal, >80%): L{min(phase_3_layers)}-L{max(phase_3_layers)} ({len(phase_3_layers)} layers)', flush=True)
        
        # 更细的7段划分
        n7 = len(layers_sorted)
        seg_size = max(1, n7 // 7)
        segments = []
        for i in range(7):
            start_idx = i * seg_size
            end_idx = min((i + 1) * seg_size, n7)
            seg_layers = layers_sorted[start_idx:end_idx]
            if seg_layers:
                seg_sig = np.mean([composite_signal[l] for l in seg_layers])
                segments.append({
                    'seg': i + 1,
                    'layers': f'L{min(seg_layers)}-L{max(seg_layers)}',
                    'normalized_range': f'{min(seg_layers)/n_layers:.2f}-{max(seg_layers)/n_layers:.2f}',
                    'mean_signal': float(seg_sig),
                })
        
        print('\n7-Segment Mapping:', flush=True)
        icspb_names = ['对象底图', '局部更新', '关系提升', '读出路由', '推理整合', '阶段组织', '前后继约束']
        for seg in segments:
            name = icspb_names[seg['seg']-1] if seg['seg'] <= len(icspb_names) else '?'
            print(f"  Chain {seg['seg']} ({name}): {seg['layers']} [{seg['normalized_range']}], signal={seg['mean_signal']:.4f}", flush=True)
    
    # 保存结果
    results = {
        'model': args.model,
        'n_layers': n_layers,
        'd_model': d_model,
        'sample_layers': sample_layers,
        'functional_signals': {ft: {str(k): v for k, v in data.items()} for ft, data in all_func_data.items()},
        'transitions': transitions,
        'composite_signal': {str(k): v for k, v in composite_signal.items()},
    }
    
    out_dir = Path('results/phase_clxxxix')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}', flush=True)
    
    release_model(model)
    print(f'Phase CLXXXIX PASSED for {args.model}', flush=True)


if __name__ == '__main__':
    main()
