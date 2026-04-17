"""
Phase CXC: 逻辑暗物质探测 — 逻辑推理功能对设计与测量
=====================================================
核心问题: 逻辑推理("因为A所以B")的编码是高维线性还是非线性?

实验设计:
1. 4类功能对: syntax/semantic/polarity + 新增 logic
2. logic功能对: 对比逻辑成立 vs 逻辑违反的句子
3. 测量: 逻辑功能信号在7段链中的位置
4. 对比: 逻辑信号 vs 属性信号的强度差异

P890: 逻辑功能对的残差流差异
P891: 逻辑信号 vs 属性信号的线性可分性对比
P892: 逻辑信号在7段链中的定位
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_model, release_model, get_layers, get_model_info

# 4类功能对
FUNC_PAIRS = {
    'syntax': [
        ("The cat sits quietly", "Cat the sits quietly"),   # 语序
        ("I am very happy", "I is very happy"),              # 主谓一致
        ("He can run fast", "He can runs fast"),             # 情态动词
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),  # 实体替换
        ("Big house near the lake", "Small house near the lake"), # 属性替换
        ("Red apple on the table", "Green apple on the table"),   # 颜色替换
    ],
    'polarity': [
        ("She is very happy today", "She is very sad today"),  # 积极/消极
        ("Good result from the test", "Bad result from the test"), # 好/坏
        ("Love the beautiful place", "Hate the beautiful place"),  # 爱/恨
    ],
    'logic': [
        # 因果推理: 因果成立 vs 因果违反
        ("Because it rained, the ground is wet", "It rained, but the ground is dry"),
        ("Since she studied hard, she passed", "She studied hard, but she failed"),
        ("Due to the heat, the ice melted", "Despite the heat, the ice stayed frozen"),
        # 条件推理: 前件真后件真 vs 前件真后件假
        ("If it rains, then the ground gets wet, and it rained", 
         "If it rains, then the ground gets wet, but it didn't rain"),
        ("When the alarm rings, we evacuate, and the alarm rang",
         "When the alarm rings, we evacuate, but we stayed"),
        # 时序推理: 正常时序 vs 违反时序
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
        return captured['hs'][0, -1, :].numpy()  # Last token
    return None


def compute_functional_signal(vecs_a, vecs_b):
    """计算功能信号: delta_norm/mean_norm"""
    deltas = [a - b for a, b in zip(vecs_a, vecs_b)]
    delta_norms = [np.linalg.norm(d) for d in deltas]
    mean_norms = [(np.linalg.norm(a) + np.linalg.norm(b)) / 2 for a, b in zip(vecs_a, vecs_b)]
    
    mean_delta = np.mean(delta_norms)
    mean_total = np.mean(mean_norms)
    functional_signal = mean_delta / mean_total if mean_total > 0 else 0
    
    # Cosine similarity between pairs
    cos_pairs = [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10) 
                 for a, b in zip(vecs_a, vecs_b)]
    
    return {
        'delta_norm_mean': float(mean_delta),
        'delta_norm_std': float(np.std(delta_norms)),
        'mean_norm_mean': float(mean_total),
        'functional_signal': float(functional_signal),
        'cos_pair_mean': float(np.mean(cos_pairs)),
        'cos_pair_std': float(np.std(cos_pairs)),
        'n_pairs': len(deltas),
    }


def linear_probe_test(vecs_a, vecs_b, n_components=5):
    """线性probe测试: 用logistic regression区分两类"""
    X = np.vstack([vecs_a + vecs_b])
    y = np.array([0]*len(vecs_a) + [1]*len(vecs_b))
    
    if len(X) < 4:
        return {'accuracy': 0.5, 'pca_5var': 0, 'note': 'too few samples'}
    
    # Full dimension logistic regression
    try:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        acc_full = clf.score(X, y)
    except:
        acc_full = 0.5
    
    # PCA + logistic regression
    try:
        pca = PCA(n_components=min(n_components, len(X)-1))
        X_pca = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_[:n_components].sum()
        clf2 = LogisticRegression(max_iter=1000, C=1.0)
        clf2.fit(X_pca, y)
        acc_pca = clf2.score(X_pca, y)
    except:
        var_explained = 0
        acc_pca = 0.5
    
    return {
        'accuracy_full': float(acc_full),
        'accuracy_pca5': float(acc_pca),
        'pca_5var_explained': float(var_explained),
        'n_samples': len(X),
    }


def cross_function_orthogonality(all_vecs, layer_idx):
    """测量功能间的正交性: 逻辑vs属性"""
    func_names = list(all_vecs.keys())
    results = {}
    
    for i, fn1 in enumerate(func_names):
        for fn2 in func_names[i+1:]:
            # 计算功能差异方向
            deltas1 = [a - b for a, b in all_vecs[fn1]]
            deltas2 = [a - b for a, b in all_vecs[fn2]]
            
            # 平均方向
            dir1 = np.mean(deltas1, axis=0)
            dir2 = np.mean(deltas2, axis=0)
            
            cos = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-10)
            
            pair_key = f"{fn1}_vs_{fn2}"
            results[pair_key] = {
                'cos': float(cos),
                'orthogonality': float(1 - abs(cos)),  # 1=完全正交, 0=完全对齐
            }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    # Output to log file
    log_path = f'tmp/cxc_{args.model}.log'
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
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    
    print(f'Model: {args.model}, L={n_layers}, H={n_heads}, d={d_model}', flush=True)
    
    # Sample layers (every 2 layers for fine-grained analysis)
    sample_layers = list(range(0, n_layers, 2))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    print(f'Sample layers: {sample_layers}', flush=True)
    
    results = {
        'model': args.model,
        'n_layers': n_layers,
        'd_model': d_model,
        'sample_layers': sample_layers,
        'P890_functional_signal': {},
        'P891_linear_separability': {},
        'P892_cross_function_orthogonality': {},
        'composite_signals': {},
    }
    
    # ===== P890: 各层各类功能信号 =====
    print('\n=== P890: Functional Signal per Layer ===', flush=True)
    
    for layer_idx in sample_layers:
        layer_results = {}
        
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
                sig = compute_functional_signal(vecs_a, vecs_b)
                layer_results[func_name] = sig
        
        results['P890_functional_signal'][str(layer_idx)] = layer_results
        
        if layer_idx % 6 == 0 or layer_idx == n_layers - 1:
            sig_str = {k: f"{v['functional_signal']:.4f}" for k, v in layer_results.items()}
            print(f'L{layer_idx} ({layer_idx/n_layers:.0%}): {sig_str}', flush=True)
    
    # ===== P891: 线性可分性对比 =====
    print('\n=== P891: Linear Separability ===', flush=True)
    
    for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        layer_results = {}
        all_vecs = {}
        
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
                probe = linear_probe_test(vecs_a, vecs_b)
                layer_results[func_name] = probe
                all_vecs[func_name] = list(zip(vecs_a, vecs_b))
        
        results['P891_linear_separability'][str(layer_idx)] = layer_results
        
        acc_str = {k: f"full={v['accuracy_full']:.2f}, pca5={v['accuracy_pca5']:.2f}" 
                   for k, v in layer_results.items()}
        print(f'L{layer_idx}: {acc_str}', flush=True)
    
    # ===== P892: 功能间正交性 (重点: logic vs 属性) =====
    print('\n=== P892: Cross-Function Orthogonality ===', flush=True)
    
    for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        all_vecs = {}
        
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
                all_vecs[func_name] = list(zip(vecs_a, vecs_b))
        
        if len(all_vecs) >= 2:
            ortho = cross_function_orthogonality(all_vecs, layer_idx)
            results['P892_cross_function_orthogonality'][str(layer_idx)] = ortho
            
            logic_pairs = {k: f"cos={v['cos']:.4f}" for k, v in ortho.items() if 'logic' in k}
            print(f'L{layer_idx}: {logic_pairs}', flush=True)
    
    # ===== Composite Signal Summary =====
    print('\n=== Composite Signals per Layer ===', flush=True)
    
    for layer_idx in sample_layers:
        layer_data = results['P890_functional_signal'].get(str(layer_idx), {})
        
        composite = {}
        for func_name in FUNC_PAIRS.keys():
            if func_name in layer_data:
                composite[func_name] = layer_data[func_name]['functional_signal']
        
        # Logic vs Attribute ratio
        attr_signals = [v for k, v in composite.items() if k != 'logic']
        logic_signal = composite.get('logic', 0)
        attr_mean = np.mean(attr_signals) if attr_signals else 0
        
        composite['logic_vs_attr_ratio'] = logic_signal / attr_mean if attr_mean > 0 else 0
        composite['attr_mean'] = float(attr_mean)
        
        results['composite_signals'][str(layer_idx)] = composite
        
        if layer_idx % 6 == 0 or layer_idx == n_layers - 1:
            print(f'L{layer_idx} ({layer_idx/n_layers:.0%}): logic={logic_signal:.4f}, attr_mean={attr_mean:.4f}, ratio={composite["logic_vs_attr_ratio"]:.2f}', flush=True)
    
    # Save results
    out_dir = Path('results/phase_cxc')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {out_path}', flush=True)
    
    # ===== Summary =====
    print('\n=== SUMMARY ===', flush=True)
    
    # Find logic signal peak
    logic_peaks = []
    for layer_idx in sample_layers:
        cs = results['composite_signals'].get(str(layer_idx), {})
        if 'logic' in cs:
            logic_peaks.append((layer_idx, cs['logic']))
    
    if logic_peaks:
        peak_layer, peak_signal = max(logic_peaks, key=lambda x: x[1])
        print(f'Logic signal peak: L{peak_layer} = {peak_signal:.4f}', flush=True)
    
    # Logic vs attribute ratio summary
    ratios = []
    for layer_idx in sample_layers:
        cs = results['composite_signals'].get(str(layer_idx), {})
        if 'logic_vs_attr_ratio' in cs:
            ratios.append((layer_idx, cs['logic_vs_attr_ratio']))
    
    if ratios:
        print(f'Logic/Attr ratio range: {min(r[1] for r in ratios):.2f} - {max(r[1] for r in ratios):.2f}', flush=True)
        mean_ratio = np.mean([r[1] for r in ratios])
        print(f'Mean Logic/Attr ratio: {mean_ratio:.2f}', flush=True)
        
        if mean_ratio < 0.5:
            print('>> Logic signal is MUCH WEAKER than attribute signals - supports "dark matter" hypothesis', flush=True)
        elif mean_ratio < 0.8:
            print('>> Logic signal is WEAKER than attribute signals - partially supports "dark matter"', flush=True)
        else:
            print('>> Logic signal is COMPARABLE to attribute signals - logic is NOT dark matter!', flush=True)
    
    # Logic orthogonality check
    print('\nLogic vs Attribute Orthogonality:', flush=True)
    for layer_idx in [n_layers//4, n_layers//2, 3*n_layers//4]:
        ortho = results['P892_cross_function_orthogonality'].get(str(layer_idx), {})
        for key, val in ortho.items():
            if 'logic' in key:
                ortho_type = "ORTHOGONAL" if val['orthogonality'] > 0.9 else "ALIGNED" if val['orthogonality'] < 0.5 else "PARTIAL"
                print(f'  L{layer_idx} {key}: cos={val["cos"]:.4f} [{ortho_type}]', flush=True)
    
    release_model(model)
    print(f'\nPhase CXC PASSED for {args.model}', flush=True)


if __name__ == '__main__':
    main()
