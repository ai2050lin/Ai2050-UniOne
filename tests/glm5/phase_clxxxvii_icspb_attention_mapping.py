"""
Phase CLXXXVII (简化版): ICSPB变量→DNN映射 — 注意力头功能分工与门控路由
精简设计: 3个采样层, 每类功能3对, 专注核心指标
P871: 注意力头功能标签(简化)
P872: g(门控路由) = 注意力权重功能条件性
P873: q(条件门控场) = softmax后功能调制
P874: b(上下文偏置) = 同token不同上下文的表示差异
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy import stats

# Add test directory to path for model_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_model, release_model, get_layers, get_model_info

# 简化的功能对
FUNC_PAIRS = {
    'syntax': [("The cat sits", "Cat the sits"), ("I am happy", "I is happy"), ("He can run", "He can runs")],
    'semantic': [("The cat sat", "The dog sat"), ("Big house", "Small house"), ("Red apple", "Green apple")],
    'polarity': [("She is happy", "She is sad"), ("Good result", "Bad result"), ("Love it", "Hate it")],
}

# load_model is imported from model_utils

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
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    if 'hs' in captured and captured['hs'].dim() == 3:
        return captured['hs'][0, -1].numpy()  # last token
    return None

def get_attn_weights(model, tokenizer, device, text, layer_idx):
    """获取某层的注意力权重"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured['attn'] = output[1].detach().float().cpu()
    
    layer = model.model.layers[layer_idx]
    # 注意: 需要output_attentions=True或者在self_attn上hook
    # 对于大多数模型, self_attn的输出不含attn weights
    # 改为: 在self_attn.o_proj之前hook
    handle = layer.self_attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    if 'attn' in captured:
        return captured['attn']
    return None

def run_p871(model, tokenizer, device, n_layers, n_heads, head_dim):
    """P871: 注意力头的功能标签 — 用残差流差异向量的head维度probe"""
    print("\n=== P871: 注意力头功能标签 ===", flush=True)
    
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    results = {}
    
    for layer_idx in sample_layers:
        print(f"  Layer {layer_idx}:", flush=True)
        
        diffs = []
        labels = []
        
        for func_type, pairs in FUNC_PAIRS.items():
            for text_a, text_b in pairs:
                hs_a = get_residual_at_layer(model, tokenizer, device, text_a, layer_idx)
                hs_b = get_residual_at_layer(model, tokenizer, device, text_b, layer_idx)
                
                if hs_a is not None and hs_b is not None:
                    diff = hs_a - hs_b
                    diffs.append(diff)
                    labels.append(func_type)
        
        if len(diffs) < 5:
            print(f"    Not enough data", flush=True)
            continue
        
        X = np.array(diffs)
        y = np.array(labels)
        
        # 按head维度分组probe
        head_results = {}
        for h in range(min(n_heads, 32)):  # 最多分析32个head
            start = h * head_dim
            end = start + head_dim
            X_h = X[:, start:end]
            
            try:
                clf = LogisticRegression(max_iter=300)
                clf.fit(X_h, y)
                acc = clf.score(X_h, y)
                
                # 随机基线
                y_rand = np.random.permutation(y)
                clf_r = LogisticRegression(max_iter=300)
                clf_r.fit(X_h, y_rand)
                acc_rand = clf_r.score(X_h, y_rand)
                
                # 每类功能的敏感度
                func_sens = {}
                for ft in FUNC_PAIRS.keys():
                    mask = y == ft
                    if mask.sum() > 0:
                        func_norm = np.mean(np.linalg.norm(X_h[mask], axis=1))
                        other_norm = np.mean(np.linalg.norm(X_h[~mask], axis=1))
                        func_sens[ft] = func_norm / max(other_norm, 1e-10)
                
                best_func = max(func_sens, key=func_sens.get) if func_sens else 'none'
                
                head_results[h] = {
                    'acc': round(acc, 3),
                    'acc_rand': round(acc_rand, 3),
                    'ratio': round(acc / max(acc_rand, 0.01), 2),
                    'best_func': best_func,
                    'sens': {k: round(v, 2) for k, v in func_sens.items()},
                }
            except Exception as e:
                head_results[h] = {'error': str(e)}
        
        # 报告top heads
        sorted_heads = sorted(head_results.items(), key=lambda x: x[1].get('ratio', 0), reverse=True)
        print(f"    Top-3 heads:", flush=True)
        for h, d in sorted_heads[:3]:
            if 'error' not in d:
                print(f"      H{h}: acc={d['acc']:.3f} ({d['ratio']:.1f}x rand), best={d['best_func']}", flush=True)
        
        results[f'L{layer_idx}'] = head_results
    
    return results

def run_p872(model, tokenizer, device, n_layers):
    """P872: g(门控路由) — 注意力权重是否功能条件化"""
    print("\n=== P872: g(门控路由) ===", flush=True)
    
    sample_layers = [n_layers//4, n_layers//2, n_layers-1]
    results = {}
    
    for layer_idx in sample_layers:
        print(f"  Layer {layer_idx}:", flush=True)
        
        # 用残差流差异向量的方向来间接度量门控路由
        # 如果不同功能的差异方向cos很小 → g是功能条件化的
        
        func_diffs = {ft: [] for ft in FUNC_PAIRS.keys()}
        
        for func_type, pairs in FUNC_PAIRS.items():
            for text_a, text_b in pairs:
                hs_a = get_residual_at_layer(model, tokenizer, device, text_a, layer_idx)
                hs_b = get_residual_at_layer(model, tokenizer, device, text_b, layer_idx)
                if hs_a is not None and hs_b is not None:
                    diff = hs_a - hs_b
                    diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
                    func_diffs[func_type].append(diff_norm)
        
        # 计算功能间的方向差异
        func_mean_dirs = {}
        for ft, dirs in func_diffs.items():
            if dirs:
                func_mean_dirs[ft] = np.mean(dirs, axis=0)
                func_mean_dirs[ft] /= (np.linalg.norm(func_mean_dirs[ft]) + 1e-10)
        
        cross_cos = {}
        ft_names = list(func_mean_dirs.keys())
        for i, ft1 in enumerate(ft_names):
            for ft2 in ft_names[i+1:]:
                cos = np.dot(func_mean_dirs[ft1], func_mean_dirs[ft2])
                cross_cos[f'{ft1}_vs_{ft2}'] = round(float(cos), 4)
        
        mean_cos = np.mean(list(cross_cos.values())) if cross_cos else 0
        g_functionality = 1.0 - mean_cos
        
        print(f"    Cross-func cos: {cross_cos}", flush=True)
        print(f"    Mean cos: {mean_cos:.4f}, g_functionality: {g_functionality:.4f}", flush=True)
        
        results[f'L{layer_idx}'] = {
            'cross_cos': cross_cos,
            'mean_cos': round(float(mean_cos), 4),
            'g_functionality': round(float(g_functionality), 4),
        }
    
    return results

def run_p873(model, tokenizer, device, n_layers):
    """P873: q(条件门控场) — 不同功能类型的残差流方差差异"""
    print("\n=== P873: q(条件门控场) ===", flush=True)
    
    sample_layers = [0, n_layers//4, n_layers//2, n_layers-1]
    results = {}
    
    for layer_idx in sample_layers:
        print(f"  Layer {layer_idx}:", flush=True)
        
        func_variances = {}
        for func_type, pairs in FUNC_PAIRS.items():
            diffs = []
            for text_a, text_b in pairs:
                hs_a = get_residual_at_layer(model, tokenizer, device, text_a, layer_idx)
                hs_b = get_residual_at_layer(model, tokenizer, device, text_b, layer_idx)
                if hs_a is not None and hs_b is not None:
                    diffs.append(np.linalg.norm(hs_a - hs_b))
            if diffs:
                func_variances[func_type] = {
                    'mean_diff_norm': round(float(np.mean(diffs)), 4),
                    'std_diff_norm': round(float(np.std(diffs)), 4),
                }
        
        # q度量: 不同功能的差异范数差异
        norms = [v['mean_diff_norm'] for v in func_variances.values()]
        q_modulation = float(np.std(norms)) / max(float(np.mean(norms)), 1e-10)
        
        print(f"    Func variances: {func_variances}", flush=True)
        print(f"    q_modulation (CV): {q_modulation:.4f}", flush=True)
        
        results[f'L{layer_idx}'] = {
            'func_variances': func_variances,
            'q_modulation': round(q_modulation, 4),
        }
    
    return results

def run_p874(model, tokenizer, device, n_layers):
    """P874: b(上下文偏置) — 同token不同上下文的表示差异"""
    print("\n=== P874: b(上下文偏置) ===", flush=True)
    
    sample_layers = [0, n_layers//4, n_layers//2, n_layers-1]
    results = {}
    
    # 同一词在不同上下文中
    target = "happy"
    contexts = [
        f"She is {target} now",
        f"He felt {target} today",
        f"They look {target} always",
        f"The weather is {target} today",
        f"I am {target} about this",
    ]
    
    for layer_idx in sample_layers:
        print(f"  Layer {layer_idx}:", flush=True)
        
        reps = []
        for ctx in contexts:
            hs = get_residual_at_layer(model, tokenizer, device, ctx, layer_idx)
            if hs is not None:
                reps.append(hs)
        
        if len(reps) < 2:
            print(f"    Not enough reps", flush=True)
            continue
        
        reps = np.array(reps)
        mean_rep = np.mean(reps, axis=0)
        
        # 上下文偏置 = 表示的方差/范数²
        context_var = np.mean(np.sum((reps - mean_rep) ** 2, axis=1))
        mean_norm_sq = np.mean(np.sum(reps ** 2, axis=1))
        relative_var = context_var / (mean_norm_sq + 1e-10)
        
        # 上下文间cos
        cos_sims = []
        for i in range(len(reps)):
            for j in range(i+1, len(reps)):
                cos = np.dot(reps[i], reps[j]) / (np.linalg.norm(reps[i]) * np.linalg.norm(reps[j]) + 1e-10)
                cos_sims.append(cos)
        
        mean_cos = np.mean(cos_sims)
        b_bias = 1.0 - mean_cos
        
        print(f"    Relative variance: {relative_var:.4f}", flush=True)
        print(f"    Mean cos: {mean_cos:.4f}, b_bias: {b_bias:.4f}", flush=True)
        
        results[f'L{layer_idx}'] = {
            'relative_var': round(float(relative_var), 4),
            'mean_cos': round(float(mean_cos), 4),
            'b_bias': round(float(b_bias), 4),
        }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, device = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size
    print(f'Model loaded! L={n_layers}, H={n_heads}, d={d_model}', flush=True)
    
    results = {
        'model': args.model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_model': d_model,
    }
    
    # 运行4个实验
    results['p871'] = run_p871(model, tokenizer, device, n_layers, n_heads, head_dim)
    results['p872'] = run_p872(model, tokenizer, device, n_layers)
    results['p873'] = run_p873(model, tokenizer, device, n_layers)
    results['p874'] = run_p874(model, tokenizer, device, n_layers)
    
    # 保存
    out_dir = Path('results/phase_clxxxvii')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to {out_path}", flush=True)
    
    # 汇总关键发现
    print("\n=== CLXXXVII 关键发现 ===", flush=True)
    print(f"Model: {args.model}", flush=True)
    
    # P871: 最佳头
    if 'p871' in results:
        for layer_key, heads in results['p871'].items():
            sorted_h = sorted(heads.items(), key=lambda x: x[1].get('ratio', 0), reverse=True)
            if sorted_h and 'error' not in sorted_h[0][1]:
                top = sorted_h[0]
                print(f"  {layer_key} best head: H{top[0]} acc={top[1]['acc']:.3f} ({top[1]['ratio']:.1f}x), func={top[1].get('best_func','?')}", flush=True)
    
    # P872: g功能条件性
    if 'p872' in results:
        for layer_key, data in results['p872'].items():
            print(f"  {layer_key} g_functionality: {data.get('g_functionality', 'N/A')}", flush=True)
    
    # P874: b偏置
    if 'p874' in results:
        for layer_key, data in results['p874'].items():
            print(f"  {layer_key} b_bias: {data.get('b_bias', 'N/A')}", flush=True)
    
    del model
    torch.cuda.empty_cache()
    print("GPU released.", flush=True)

if __name__ == '__main__':
    main()
