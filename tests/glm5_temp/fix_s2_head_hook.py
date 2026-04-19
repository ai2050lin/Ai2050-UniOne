"""
S2修复版: Head Hook因果定位
=============================
问题: o_proj的forward hook在GQA下拿不到concat_heads, 因为HuggingFace实现中
      attention计算在内部完成, o_proj的input已经是attn_output的形状

修复策略: 不hook o_proj, 而是直接用W_o权重矩阵分解attn_output到每个head的子空间
         对于4bit模型, dequantize W_o后使用
         对于非4bit模型, 直接使用W_o

另一个策略: hook self_attn的输出(attn_output), 然后用W_o的伪逆反推每个head的贡献

最可靠策略: 使用W_o的列空间投影法:
  attn_output = W_o @ concat_heads
  W_o [d_model, n_heads*d_head]
  每个head h对应的W_o_h = W_o[:, h*d_head:(h+1)*_head]
  head_h在attn_output空间中的投影 = W_o_h @ (W_o_h^T @ attn_output) / ||W_o_h||^2

运行: 逐模型运行
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json, gc, time
import numpy as np
import torch
from pathlib import Path


# 复用数据生成
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
from causal_megasample import (
    generate_polarity_pairs, generate_tense_pairs, generate_semantic_pairs,
    generate_sentiment_pairs, generate_number_pairs, load_model_fast
)


def test_s2_fixed(model_name, model, tokenizer, device, d_model, n_layers):
    """
    S2修复版: 用W_o列空间投影法分离每个head的因果贡献
    
    方法:
    1. Hook self_attn获取attn_output
    2. 提取W_o权重矩阵
    3. 按head分离W_o的列: W_o_h = W_o[:, h*d_head:(h+1)*d_head]
    4. 对每个head, 计算attn_output在W_o_h子空间中的投影
    5. 计算极性/时态差分在每个head投影中的cos对齐
    """
    print("\n" + "=" * 70)
    print("S2修复版: Head W_o投影因果定位")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:200]
    tense_pairs = generate_tense_pairs()[:200]
    
    # 架构信息
    layer0 = model.model.layers[0]
    n_heads = layer0.self_attn.config.num_attention_heads
    d_head = d_model // n_heads
    if hasattr(layer0.self_attn.config, 'head_dim') and layer0.self_attn.config.head_dim is not None:
        d_head = layer0.self_attn.config.head_dim
    
    n_kv_heads = n_heads
    if hasattr(layer0.self_attn.config, 'num_key_value_heads'):
        n_kv_heads = layer0.self_attn.config.num_key_value_heads
    
    print(f"  n_heads={n_heads}, d_head={d_head}, n_kv_heads={n_kv_heads}, d_model={d_model}")
    
    # 关键层
    if model_name == 'deepseek7b':
        target_layers = [n_layers - 1, n_layers - 2, n_layers - 3]
    elif model_name == 'glm4':
        target_layers = [0, n_layers - 1]
    else:
        target_layers = [n_layers - 1, n_layers - 2]
    
    print(f"  target_layers={target_layers}")
    
    results = {'n_heads': n_heads, 'd_head': d_head, 'n_kv_heads': n_kv_heads,
               'target_layers': target_layers}
    
    for li in target_layers:
        print(f"\n  --- L{li} ---")
        layer = model.model.layers[li]
        
        # 提取W_o权重
        W_o_raw = layer.self_attn.o_proj.weight
        try:
            W_o = W_o_raw.detach().cpu().float().numpy()
            print(f"  W_o shape: {W_o.shape}")
        except:
            print(f"  W_o dequantize failed, skipping L{li}")
            results[f'L{li}'] = {'error': 'W_o_dequantize_failed'}
            continue
        
        # 判断W_o形状是否正确
        # 标准形状: [d_model, n_heads*d_head] 或 [d_model, d_model]
        # 4bit量化可能产生异常形状
        
        # 确保W_o是2D的
        if W_o.ndim != 2:
            print(f"  W_o ndim={W_o.ndim}, 无法处理, 跳过L{li}")
            results[f'L{li}'] = {'error': f'W_o_ndim_{W_o.ndim}'}
            continue
        
        # 检查W_o维度是否匹配d_model
        # 4bit量化后的W_o可能不是标准形状
        if W_o.shape[0] != d_model and W_o.shape[1] != d_model:
            print(f"  W_o shape {W_o.shape} 不匹配 d_model={d_model}, 跳过L{li}")
            results[f'L{li}'] = {'error': f'W_o_shape_mismatch_{W_o.shape}'}
            continue
        
        # 确保W_o的行是d_model
        if W_o.shape[0] != d_model:
            W_o = W_o.T
        
        # 分离每个head的W_o列
        expected_cols = n_heads * d_head
        if W_o.shape[1] != expected_cols:
            print(f"  W_o列数{W_o.shape[1]}≠预期{n_heads}*{d_head}={expected_cols}, 尝试调整d_head")
            actual_d_head = W_o.shape[1] // n_heads
            if actual_d_head > 0:
                d_head_local = actual_d_head
                print(f"  调整d_head={d_head_local}")
            else:
                print(f"  无法分离head, 跳过L{li}")
                results[f'L{li}'] = {'error': 'cannot_separate_heads'}
                continue
        else:
            d_head_local = d_head
        
        head_subspaces = []
        for h in range(n_heads):
            start = h * d_head_local
            end = (h + 1) * d_head_local
            if end <= W_o.shape[1]:
                W_o_h = W_o[:, start:end]  # [d_model, d_head_local]
                head_subspaces.append(W_o_h)
            else:
                head_subspaces.append(None)
        
        n_valid = sum(1 for s in head_subspaces if s is not None)
        print(f"  有效head子空间: {n_valid}/{n_heads}")
        
        if n_valid == 0:
            print(f"  无有效head子空间, 跳过L{li}")
            results[f'L{li}'] = {'error': 'no_valid_head_subspaces'}
            continue
        
        # 对极性和时态, 提取attn_output
        def get_attn_reprs(pairs):
            texts_a = [a for a, b in pairs]
            texts_b = [b for a, b in pairs]
            all_texts = texts_a + texts_b
            n_half = len(texts_a)
            
            captured = []
            
            def make_hook(storage):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].detach().cpu().float()
                    else:
                        h = output.detach().cpu().float()
                    storage.append(h[0])  # [seq, d_model]
                return hook_fn
            
            h = layer.self_attn.register_forward_hook(make_hook(captured))
            
            reprs_a = []
            reprs_b = []
            for i, text in enumerate(all_texts):
                captured.clear()
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try:
                        _ = model(**toks)
                    except:
                        continue
                if captured:
                    out = captured[0].numpy()[-1]  # last token, [d_model]
                    if i < n_half:
                        reprs_a.append(out)
                    else:
                        reprs_b.append(out)
            
            h.remove()
            return reprs_a, reprs_b
        
        print(f"  提取极性attn_output...")
        pol_a, pol_b = get_attn_reprs(pol_pairs)
        print(f"  极性: n={len(pol_a)}/{len(pol_b)}")
        
        print(f"  提取时态attn_output...")
        tense_a, tense_b = get_attn_reprs(tense_pairs)
        print(f"  时态: n={len(tense_a)}/{len(tense_b)}")
        
        if len(pol_a) == 0 or len(pol_b) == 0 or len(tense_a) == 0 or len(tense_b) == 0:
            results[f'L{li}'] = {'error': 'insufficient_data'}
            continue
        
        # 整体attn因果cos
        pol_diff = np.mean(pol_b, axis=0) - np.mean(pol_a, axis=0)
        tense_diff = np.mean(tense_b, axis=0) - np.mean(tense_a, axis=0)
        pol_diff_norm = np.linalg.norm(pol_diff)
        tense_diff_norm = np.linalg.norm(tense_diff)
        
        if pol_diff_norm > 1e-10 and tense_diff_norm > 1e-10:
            overall_cos = float(np.dot(pol_diff / pol_diff_norm, tense_diff / tense_diff_norm))
        else:
            overall_cos = 0.0
        
        print(f"  整体attn因果cos: {overall_cos:+.4f}")
        
        # 每个head的因果对齐
        head_results = []
        
        for h_idx in range(n_heads):
            if head_subspaces[h_idx] is None:
                continue
            
            W_o_h = head_subspaces[h_idx]  # [d_model, d_head]
            
            # 极性差分在head h子空间中的投影
            # proj = W_o_h @ (W_o_h^T @ diff) / ||W_o_h||_F^2
            # 但更准确的方法: 用最小二乘投影
            # head_h_attn = W_o_h @ (W_o_h^+ @ diff), 其中W_o_h^+是伪逆
            
            # 简化: 直接用正交投影
            # proj_pol = W_o_h @ (W_o_h^T @ pol_diff)
            pol_proj = W_o_h @ (W_o_h.T @ pol_diff)
            tense_proj = W_o_h @ (W_o_h.T @ tense_diff)
            
            pol_proj_norm = np.linalg.norm(pol_proj)
            tense_proj_norm = np.linalg.norm(tense_proj)
            
            if pol_proj_norm > 1e-10 and tense_proj_norm > 1e-10:
                head_cos = float(np.dot(pol_proj / pol_proj_norm, tense_proj / tense_proj_norm))
            else:
                head_cos = 0.0
            
            head_results.append({
                'head': h_idx,
                'head_cos': head_cos,
                'pol_proj_norm': float(pol_proj_norm),
                'tense_proj_norm': float(tense_proj_norm),
                'pol_energy': float(pol_proj_norm ** 2),
                'tense_energy': float(tense_proj_norm ** 2),
            })
        
        # 按cos绝对值排序
        head_results.sort(key=lambda x: abs(x['head_cos']), reverse=True)
        
        print(f"\n  {'Head':>6} {'cos':>8} {'pol_proj':>10} {'tense_proj':>11} {'energy':>10}")
        print("  " + "-" * 55)
        for hr in head_results[:15]:
            total_e = hr['pol_energy'] + hr['tense_energy']
            mark = "★★★" if abs(hr['head_cos']) > 0.5 else ("★★" if abs(hr['head_cos']) > 0.3 else "")
            print(f"  h{hr['head']:>4} {hr['head_cos']:>+8.4f} {hr['pol_proj_norm']:>10.2f} {hr['tense_proj_norm']:>11.2f} {total_e:>10.2f} {mark}")
        
        high_cos = [hr for hr in head_results if abs(hr['head_cos']) > 0.5]
        med_cos = [hr for hr in head_results if 0.3 < abs(hr['head_cos']) <= 0.5]
        low_cos = [hr for hr in head_results if abs(hr['head_cos']) <= 0.3]
        
        print(f"\n  统计: ★★★(>0.5)={len(high_cos)}, ★★(0.3-0.5)={len(med_cos)}, ≤0.3={len(low_cos)}")
        
        # 也按能量排序看top heads
        head_results_by_energy = sorted(head_results, key=lambda x: x['pol_energy'] + x['tense_energy'], reverse=True)
        print(f"\n  按能量排序Top-5:")
        for hr in head_results_by_energy[:5]:
            total_e = hr['pol_energy'] + hr['tense_energy']
            print(f"    h{hr['head']}: cos={hr['head_cos']:+.4f}, energy={total_e:.2f}")
        
        results[f'L{li}'] = {
            'overall_cos': float(overall_cos),
            'head_results': head_results[:32],
            'n_high_cos': len(high_cos),
            'n_med_cos': len(med_cos),
            'n_low_cos': len(low_cos),
        }
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}, d_model={d_model}, n_layers={n_layers}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    result_dir = Path(f'results/causal_fiber/{model_name}_megasample')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    t0 = time.time()
    s2 = test_s2_fixed(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / 's2_results.json', 'w') as f:
        json.dump(convert_keys(s2), f, indent=2, default=str)
    print(f"\nS2 saved! ({time.time()-t0:.0f}s)")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nDone!")
