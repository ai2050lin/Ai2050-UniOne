"""
Phase CLXXXVIII: ICSPB变量→DNN映射 — FFN功能注入与a/p/f映射
===============================================================
P875: a(局部激活密度) = FFN稀疏激活率(每层)的功能条件性
P876: p(可塑性预算) = FFN输出对残差流功能信号的贡献比
P877: f(跨区共享纤维流) = 残差连接保留率 + 跨层功能信号相似度

设计:
- 对3个采样层, 用3类功能对, 前向pass中同时hook Attn和FFN输出
- 测量FFN稀疏率(a), Attn/FFN贡献比(p), 残差保留率(f)
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_model, release_model, get_layers, get_model_info, get_W_U

# 功能对（与CLXXXVII一致）
FUNC_PAIRS = {
    'syntax': [("The cat sits", "Cat the sits"), ("I am happy", "I is happy"), ("He can run", "He can runs")],
    'semantic': [("The cat sat", "The dog sat"), ("Big house", "Small house"), ("Red apple", "Green apple")],
    'polarity': [("She is happy", "She is sad"), ("Good result", "Bad result"), ("Love it", "Hate it")],
}


def get_attn_ffn_outputs(model, tokenizer, device, text, layer_idx, threshold=0.0):
    """
    Hook Attn和FFN的输出, 返回:
    - residual_in: 层输入残差流
    - attn_out: 注意力层输出
    - ffn_out: FFN层输出  
    - residual_out: 层输出残差流
    - ffn_intermediate: FFN中间激活(用于计算稀疏率)
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    captured = {}
    
    layers = get_layers(model)
    layer = layers[layer_idx]
    
    # Hook 1: 层输入（通过input_layernorm捕获）
    def hook_input(mod, inp, out):
        if isinstance(out, tuple):
            captured['ln_input'] = out[0].detach().float().cpu()
        else:
            captured['ln_input'] = out.detach().float().cpu()
    
    # Hook 2: 注意力输出
    def hook_attn(mod, inp, out):
        if isinstance(out, tuple):
            captured['attn_out'] = out[0].detach().float().cpu()
        else:
            captured['attn_out'] = out.detach().float().cpu()
    
    # Hook 3: FFN中间激活（gate_proj输出后silu激活）
    def hook_ffn_intermediate(mod, inp, out):
        if isinstance(out, tuple):
            captured['ffn_intermediate'] = out[0].detach().float().cpu()
        else:
            captured['ffn_intermediate'] = out.detach().float().cpu()
    
    # Hook 4: FFN输出（down_proj输出）
    def hook_ffn_out(mod, inp, out):
        if isinstance(out, tuple):
            captured['ffn_out'] = out[0].detach().float().cpu()
        else:
            captured['ffn_out'] = out.detach().float().cpu()
    
    # Hook 5: 层输出
    def hook_layer_out(mod, inp, out):
        if isinstance(out, tuple):
            captured['layer_out'] = out[0].detach().float().cpu()
        else:
            captured['layer_out'] = out.detach().float().cpu()
    
    handles = []
    
    # Input layernorm
    for ln_name in ['input_layernorm', 'ln_1', 'layernorm']:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            handles.append(ln.register_forward_hook(hook_input))
            break
    
    # Self attention output
    handles.append(layer.self_attn.register_forward_hook(hook_attn))
    
    # FFN: hook gate_proj or gate_up_proj for intermediate activations
    mlp = layer.mlp
    if hasattr(mlp, 'gate_up_proj'):
        handles.append(mlp.gate_up_proj.register_forward_hook(hook_ffn_intermediate))
    elif hasattr(mlp, 'gate_proj'):
        handles.append(mlp.gate_proj.register_forward_hook(hook_ffn_intermediate))
    
    # FFN down_proj output
    handles.append(mlp.down_proj.register_forward_hook(hook_ffn_out))
    
    # Layer output
    handles.append(layer.register_forward_hook(hook_layer_out))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in handles:
        h.remove()
    
    return captured


def compute_p875_activation_density(ffn_intermediate, threshold_ratio=0.01):
    """
    P875: a(局部激活密度)
    
    计算FFN中间激活中超过阈值的比例。
    在SwiGLU架构中, gate_proj输出经silu后, 大部分值接近0。
    
    对于gate_up_proj(合并), 输出shape=[1, seq_len, 2*intermediate]
    我们只取gate部分(前半)来计算稀疏率。
    """
    if ffn_intermediate is None:
        return 0.0, 0.0
    
    arr = ffn_intermediate[0, -1, :].numpy()  # last token
    
    # 如果是合并的gate_up, 只取前半(gate部分)
    half = len(arr) // 2
    gate_act = arr[:half]
    
    # 稀疏率: |激活值>阈值| / 总数
    max_val = np.max(np.abs(gate_act))
    if max_val < 1e-10:
        return 0.0, 0.0
    
    threshold = threshold_ratio * max_val
    active_ratio = float(np.mean(np.abs(gate_act) > threshold))
    
    # 均值激活强度
    mean_act = float(np.mean(np.abs(gate_act)))
    
    return active_ratio, mean_act


def compute_p876_plasticity_budget(attn_out, ffn_out, residual_in):
    """
    P876: p(可塑性预算) = Attn/FFN对残差流功能信号的贡献比
    
    计算方法:
    - residual_delta = layer_out - residual_in
    - attn_contribution = ||attn_out||
    - ffn_contribution = ||ffn_out||
    - p_attn = attn_contribution / (attn + ffn)
    - p_ffn = ffn_contribution / (attn + ffn)
    """
    if attn_out is None or ffn_out is None or residual_in is None:
        return {}
    
    # Last token
    attn_vec = attn_out[0, -1, :].numpy()
    ffn_vec = ffn_out[0, -1, :].numpy()
    res_in_vec = residual_in[0, -1, :].numpy()
    
    attn_norm = np.linalg.norm(attn_vec)
    ffn_norm = np.linalg.norm(ffn_vec)
    total = attn_norm + ffn_norm
    
    if total < 1e-10:
        return {'p_attn': 0.5, 'p_ffn': 0.5, 'attn_norm': 0, 'ffn_norm': 0}
    
    # 残差连接保留率
    layer_delta = attn_vec + ffn_vec  # 修改量
    residual_ratio = np.linalg.norm(res_in_vec) / (np.linalg.norm(res_in_vec) + np.linalg.norm(layer_delta) + 1e-10)
    
    return {
        'p_attn': float(attn_norm / total),
        'p_ffn': float(ffn_norm / total),
        'attn_norm': float(attn_norm),
        'ffn_norm': float(ffn_norm),
        'residual_ratio': float(residual_ratio),
        'delta_norm': float(np.linalg.norm(layer_delta)),
    }


def compute_p877_fiber_flow(residual_in, residual_out, layer_delta):
    """
    P877: f(跨区共享纤维流)
    
    测量残差连接的信号保留率和层修改的功能相关性。
    
    f_preserve = cos(residual_out, residual_in) — 残差保留度
    f_modify = ||layer_delta|| / ||residual_in|| — 修改强度
    """
    if residual_in is None or residual_out is None:
        return {}
    
    res_in = residual_in[0, -1, :].numpy()
    res_out = residual_out[0, -1, :].numpy()
    
    # Cosine similarity (残差保留度)
    norm_in = np.linalg.norm(res_in)
    norm_out = np.linalg.norm(res_out)
    
    if norm_in < 1e-10 or norm_out < 1e-10:
        return {'f_preserve': 0, 'f_modify': 0}
    
    cos_preserve = float(np.dot(res_in, res_out) / (norm_in * norm_out))
    
    # 修改强度
    delta = res_out - res_in
    f_modify = float(np.linalg.norm(delta) / norm_in)
    
    return {
        'f_preserve': cos_preserve,
        'f_modify': f_modify,
        'res_in_norm': float(norm_in),
        'res_out_norm': float(norm_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    # Output to log file for PowerShell compatibility
    log_path = f'tmp/clxxxviii_{args.model}.log'
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
    print(f'Model loaded! L={n_layers}, H={n_heads}, d={d_model}', flush=True)
    
    model_info = get_model_info(model, args.model)
    
    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    print(f'Sample layers: {sample_layers}', flush=True)
    
    results = {
        'model': args.model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_model': d_model,
        'sample_layers': sample_layers,
        'P875_activation_density': {},
        'P876_plasticity_budget': {},
        'P877_fiber_flow': {},
    }
    
    # 对每个采样层, 每类功能对
    for li in sample_layers:
        layer_key = f'L{li}'
        results['P875_activation_density'][layer_key] = {}
        results['P876_plasticity_budget'][layer_key] = {}
        results['P877_fiber_flow'][layer_key] = {}
        
        for func_type, pairs in FUNC_PAIRS.items():
            a_ratios = []
            a_means = []
            p_attns = []
            p_ffns = []
            f_preserves = []
            f_modifies = []
            
            for text_a, text_b in pairs:
                # 获取text_a的Attn/FFN输出
                cap_a = get_attn_ffn_outputs(model, tokenizer, device, text_a, li)
                cap_b = get_attn_ffn_outputs(model, tokenizer, device, text_b, li)
                
                # P875: 激活密度
                a_ratio_a, a_mean_a = compute_p875_activation_density(cap_a.get('ffn_intermediate'))
                a_ratio_b, a_mean_b = compute_p875_activation_density(cap_b.get('ffn_intermediate'))
                a_ratios.append((a_ratio_a + a_ratio_b) / 2)
                a_means.append((a_mean_a + a_mean_b) / 2)
                
                # P876: 可塑性预算
                p_info_a = compute_p876_plasticity_budget(
                    cap_a.get('attn_out'), cap_a.get('ffn_out'), cap_a.get('ln_input'))
                p_info_b = compute_p876_plasticity_budget(
                    cap_b.get('attn_out'), cap_b.get('ffn_out'), cap_b.get('ln_input'))
                
                if 'p_attn' in p_info_a:
                    p_attns.append((p_info_a['p_attn'] + p_info_b['p_attn']) / 2)
                    p_ffns.append((p_info_a['p_ffn'] + p_info_b['p_ffn']) / 2)
                
                # P877: 纤维流
                f_info_a = compute_p877_fiber_flow(
                    cap_a.get('ln_input'), cap_a.get('layer_out'), None)
                f_info_b = compute_p877_fiber_flow(
                    cap_b.get('ln_input'), cap_b.get('layer_out'), None)
                
                if 'f_preserve' in f_info_a:
                    f_preserves.append((f_info_a['f_preserve'] + f_info_b['f_preserve']) / 2)
                    f_modifies.append((f_info_a['f_modify'] + f_info_b['f_modify']) / 2)
            
            # 汇总
            results['P875_activation_density'][layer_key][func_type] = {
                'active_ratio_mean': float(np.mean(a_ratios)),
                'active_ratio_std': float(np.std(a_ratios)),
                'mean_act_intensity': float(np.mean(a_means)),
            }
            results['P876_plasticity_budget'][layer_key][func_type] = {
                'p_attn_mean': float(np.mean(p_attns)) if p_attns else 0,
                'p_ffn_mean': float(np.mean(p_ffns)) if p_ffns else 0,
                'p_attn_std': float(np.std(p_attns)) if p_attns else 0,
            }
            results['P877_fiber_flow'][layer_key][func_type] = {
                'f_preserve_mean': float(np.mean(f_preserves)) if f_preserves else 0,
                'f_modify_mean': float(np.mean(f_modifies)) if f_modifies else 0,
                'f_preserve_std': float(np.std(f_preserves)) if f_preserves else 0,
            }
            
            print(f'  L{li} {func_type}: a={np.mean(a_ratios):.4f}, p_attn={np.mean(p_attns) if p_attns else 0:.4f}, p_ffn={np.mean(p_ffns) if p_ffns else 0:.4f}, f_pres={np.mean(f_preserves) if f_preserves else 0:.4f}, f_mod={np.mean(f_modifies) if f_modifies else 0:.4f}', flush=True)
    
    # 跨功能类型汇总
    print('\n=== Cross-function Summary ===', flush=True)
    for li in sample_layers:
        layer_key = f'L{li}'
        
        a_by_func = {ft: results['P875_activation_density'][layer_key][ft]['active_ratio_mean'] for ft in FUNC_PAIRS}
        p_attn_by_func = {ft: results['P876_plasticity_budget'][layer_key][ft]['p_attn_mean'] for ft in FUNC_PAIRS}
        p_ffn_by_func = {ft: results['P876_plasticity_budget'][layer_key][ft]['p_ffn_mean'] for ft in FUNC_PAIRS}
        f_pres_by_func = {ft: results['P877_fiber_flow'][layer_key][ft]['f_preserve_mean'] for ft in FUNC_PAIRS}
        
        # a的功能条件性: 功能间的变异系数
        a_vals = list(a_by_func.values())
        a_cv = float(np.std(a_vals) / (np.mean(a_vals) + 1e-10))
        
        # p的功能条件性
        p_attn_vals = list(p_attn_by_func.values())
        p_ffn_vals = list(p_ffn_by_func.values())
        p_cv = float(np.std(p_attn_vals) / (np.mean(p_attn_vals) + 1e-10))
        
        print(f'L{li}: a_cv={a_cv:.4f} a={a_by_func} | p_attn={p_attn_by_func} p_ffn={p_ffn_by_func} | f_pres={f_pres_by_func} | p_cv={p_cv:.4f}', flush=True)
    
    # 保存结果
    out_dir = Path('results/phase_clxxxviii')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}', flush=True)
    
    release_model(model)
    print(f'Phase CLXXXVIII PASSED for {args.model}', flush=True)


if __name__ == '__main__':
    main()
