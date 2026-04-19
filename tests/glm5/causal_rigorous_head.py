"""
Phase CCVI: 因果汇聚的严格验证 — 双方法对比 + 大样本 + 特征分解
================================================================

关键问题:
  Phase CCIV (W_o投影): DS7B L27 h12 cos=0.508 → 因果汇聚!
  Phase CCV (直接head输出): DS7B L27 h25 align=0.308 → 无明显汇聚!
  → 两种方法结论矛盾! 需要严格验证!

方案:
  S1: 双方法对比 — 同一数据, 同时计算W_o投影alignment和直接head alignment
  S2: 特征分解alignment — 每个特征单独计算, 消除跨特征"稀释"
  S3: 因果干预验证 — 对top head做activation patching (如果可行)
  S4: 统计显著性 — bootstrap置信区间

样本: 500 pairs/特征 = 2500 pairs (适中, 不太慢)

运行:
  python tests/glm5/causal_rigorous_head.py --model deepseek7b
  python tests/glm5/causal_rigorous_head.py --model qwen3
  python tests/glm5/causal_rigorous_head.py --model glm4
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, gc, time, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from causal_megasample import (
    generate_polarity_pairs, generate_tense_pairs, generate_semantic_pairs,
    generate_sentiment_pairs, generate_number_pairs
)

FEATURE_GENERATORS = {
    'polarity': generate_polarity_pairs,
    'tense': generate_tense_pairs,
    'semantic': generate_semantic_pairs,
    'sentiment': generate_sentiment_pairs,
    'number': generate_number_pairs,
}
FNAMES = list(FEATURE_GENERATORS.keys())

N_PAIRS = 500  # 每特征500对, 适中


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, device_map="auto",
                                                      trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cpu',
                                                      trust_remote_code=True, local_files_only=True)
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    info = {
        'd_model': cfg.hidden_size, 'n_layers': cfg.num_hidden_layers,
        'n_heads': cfg.num_attention_heads, 'd_head': cfg.hidden_size // cfg.num_attention_heads,
        'n_kv_heads': getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        'device': device
    }
    return model, tokenizer, info


def collect_dual_method_data(model, tokenizer, info, target_layers):
    """
    同时收集:
    1. 直接head输出差分 (o_proj input的head切片)
    2. W_o投影后的差分 (o_proj output)
    """
    d_model = info['d_model']
    n_heads = info['n_heads']
    d_head = info['d_head']
    device = info['device']
    
    all_data = {}
    
    for li in target_layers:
        layer = model.model.layers[li]
        o_proj = layer.self_attn.o_proj
        
        if o_proj.weight.is_meta:
            all_data[li] = None
            continue
        
        # 获取W_o
        W_o = o_proj.weight.detach().cpu().float()
        if W_o.shape[0] != d_model:
            W_o = W_o.T
        
        layer_data = {
            'W_o': W_o.numpy(),
            'head_diffs': defaultdict(lambda: defaultdict(list)),  # h_idx -> fname -> [diff_vecs]
            'projected_diffs': defaultdict(lambda: defaultdict(list)),  # h_idx -> fname -> [projected_diffs]
            'resid_diffs': defaultdict(list),  # fname -> [resid_diffs]
        }
        
        for fname in FNAMES:
            pairs = FEATURE_GENERATORS[fname]()
            pairs = pairs[:N_PAIRS]
            
            hook_storage = []
            
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    h = input[0].detach().cpu().float()
                    hook_storage.append(h)
            
            handle = o_proj.register_forward_hook(hook_fn)
            
            reprs_a = []
            reprs_b = []
            
            for text_a, text_b in pairs:
                hook_storage.clear()
                toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if hook_storage:
                    reprs_a.append(hook_storage[0][0, -1, :].numpy())
                
                hook_storage.clear()
                toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if hook_storage:
                    reprs_b.append(hook_storage[0][0, -1, :].numpy())
            
            handle.remove()
            
            n = min(len(reprs_a), len(reprs_b))
            W_o_np = layer_data['W_o']
            for i in range(n):
                diff = reprs_a[i] - reprs_b[i]  # [n_heads*d_head]
                
                # 直接head输出差分
                for h_idx in range(n_heads):
                    s, e = h_idx * d_head, (h_idx + 1) * d_head
                    if e <= len(diff):
                        layer_data['head_diffs'][h_idx][fname].append(diff[s:e])
                
                # W_o投影差分: y = W_o @ x, 对每个head单独投影
                for h_idx in range(n_heads):
                    s, e = h_idx * d_head, (h_idx + 1) * d_head
                    if e <= len(diff):
                        x_h = diff[s:e]
                        y_h = W_o_np[:, s:e] @ x_h  # [d_model]
                        layer_data['projected_diffs'][h_idx][fname].append(y_h)
                
                # 全残差投影差分
                y_full = W_o_np @ diff
                layer_data['resid_diffs'][fname].append(y_full)
        
        all_data[li] = layer_data
        total = sum(sum(len(v2) for v2 in v1.values()) for v1 in layer_data['head_diffs'].values())
        print(f"  L{li}: {total} head-diffs collected")
    
    return all_data


def compute_alignment(diffs_list):
    """计算一组差分向量的方向一致性alignment"""
    if len(diffs_list) < 5:
        return None, None
    
    arr = np.array(diffs_list)
    # 归一化
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    arr_norm = arr / norms
    
    # 平均方向
    mean_dir = arr_norm.mean(axis=0)
    mean_norm = np.linalg.norm(mean_dir)
    if mean_norm < 1e-10:
        return None, None
    
    # alignment: 每个向量与平均方向的cosine
    cosines = arr_norm @ mean_dir / mean_norm
    return float(np.mean(cosines)), float(np.std(cosines))


def test_s1_dual_method(info, all_data):
    """S1: 双方法对比 — 直接head输出 vs W_o投影"""
    print("\n" + "="*70)
    print("S1: 双方法对比 (直接head输出 vs W_o投影)")
    print("="*70)
    
    n_heads = info['n_heads']
    results = {}
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        head_results = {}
        for h_idx in range(n_heads):
            # 直接head输出alignment
            direct_diffs = []
            for fname in FNAMES:
                direct_diffs.extend(layer_data['head_diffs'][h_idx].get(fname, []))
            direct_align, direct_std = compute_alignment(direct_diffs)
            
            # W_o投影alignment
            proj_diffs = []
            for fname in FNAMES:
                proj_diffs.extend(layer_data['projected_diffs'][h_idx].get(fname, []))
            proj_align, proj_std = compute_alignment(proj_diffs)
            
            if direct_align is not None:
                head_results[f'h{h_idx}'] = {
                    'direct_alignment': direct_align,
                    'direct_std': direct_std,
                    'projected_alignment': proj_align if proj_align else 0,
                    'projected_std': proj_std if proj_std else 0,
                    'n_diffs': len(direct_diffs)
                }
        
        # 排序
        sorted_direct = sorted(head_results.items(), key=lambda x: x[1]['direct_alignment'], reverse=True)
        sorted_proj = sorted(head_results.items(), key=lambda x: x[1]['projected_alignment'], reverse=True)
        
        print(f"\n  L{li}:")
        print(f"    直接head输出 Top5:")
        for h_name, hd in sorted_direct[:5]:
            print(f"      {h_name}: direct={hd['direct_alignment']:.4f}, projected={hd['projected_alignment']:.4f}")
        print(f"    W_o投影 Top5:")
        for h_name, hd in sorted_proj[:5]:
            print(f"      {h_name}: direct={hd['direct_alignment']:.4f}, projected={hd['projected_alignment']:.4f}")
        
        results[f'L{li}'] = {
            'sorted_by_direct': [(n, d['direct_alignment'], d['projected_alignment']) for n, d in sorted_direct[:10]],
            'sorted_by_projected': [(n, d['direct_alignment'], d['projected_alignment']) for n, d in sorted_proj[:10]],
            'all_heads': head_results
        }
    
    return results


def test_s2_feature_decomposed(info, all_data):
    """S2: 特征分解alignment — 每个特征单独计算"""
    print("\n" + "="*70)
    print("S2: 特征分解alignment (消除跨特征稀释)")
    print("="*70)
    
    n_heads = info['n_heads']
    results = {}
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        head_results = {}
        for h_idx in range(n_heads):
            feat_aligns = {}
            all_direct_diffs = []
            
            for fname in FNAMES:
                diffs = layer_data['head_diffs'][h_idx].get(fname, [])
                if len(diffs) < 5:
                    continue
                align, std = compute_alignment(diffs)
                if align is not None:
                    feat_aligns[fname] = {'alignment': align, 'std': std, 'n': len(diffs)}
                all_direct_diffs.extend(diffs)
            
            # 跨特征alignment
            cross_align, cross_std = compute_alignment(all_direct_diffs)
            
            if feat_aligns:
                head_results[f'h{h_idx}'] = {
                    'cross_feature_alignment': cross_align if cross_align else 0,
                    'feature_alignments': feat_aligns,
                    'avg_single_feature': float(np.mean([v['alignment'] for v in feat_aligns.values()])),
                    'n_diffs': len(all_direct_diffs)
                }
        
        # 按avg_single_feature排序
        sorted_heads = sorted(head_results.items(), key=lambda x: x[1]['avg_single_feature'], reverse=True)
        
        print(f"\n  L{li}:")
        print(f"    Top5 by avg single-feature alignment:")
        for h_name, hd in sorted_heads[:5]:
            feat_str = ', '.join(f"{k}={v['alignment']:.3f}" for k, v in sorted(hd['feature_alignments'].items(), key=lambda x: -x[1]['alignment']))
            print(f"      {h_name}: cross={hd['cross_feature_alignment']:.4f}, avg_single={hd['avg_single_feature']:.4f} | {feat_str}")
        
        results[f'L{li}'] = {
            'sorted_by_single_feature': [(n, d['cross_feature_alignment'], d['avg_single_feature']) for n, d in sorted_heads[:10]],
            'all_heads': head_results
        }
    
    return results


def test_s3_bootstrap_significance(info, all_data, n_bootstrap=1000):
    """S3: Bootstrap置信区间 — 验证alignment的统计显著性"""
    print("\n" + "="*70)
    print("S3: Bootstrap显著性检验")
    print("="*70)
    
    n_heads = info['n_heads']
    results = {}
    
    np.random.seed(42)
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        head_results = {}
        for h_idx in range(n_heads):
            # 收集每个特征的差分
            feat_diffs = {}
            for fname in FNAMES:
                diffs = layer_data['head_diffs'][h_idx].get(fname, [])
                if len(diffs) >= 10:
                    feat_diffs[fname] = diffs
            
            if not feat_diffs:
                continue
            
            # 计算每个特征的alignment
            feat_aligns = {}
            for fname, diffs in feat_diffs.items():
                arr = np.array(diffs)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                arr_norm = arr / norms
                mean_dir = arr_norm.mean(axis=0)
                mean_norm = np.linalg.norm(mean_dir)
                if mean_norm > 1e-10:
                    cosines = arr_norm @ mean_dir / mean_norm
                    feat_aligns[fname] = float(np.mean(cosines))
            
            if not feat_aligns:
                continue
            
            # Bootstrap: 对每个特征做bootstrap
            bootstrap_results = {}
            for fname, diffs in feat_diffs.items():
                arr = np.array(diffs)
                n = len(arr)
                boot_aligns = []
                for _ in range(n_bootstrap):
                    idx = np.random.randint(0, n, size=n)
                    boot_arr = arr[idx]
                    norms = np.linalg.norm(boot_arr, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    boot_norm = boot_arr / norms
                    mean_dir = boot_norm.mean(axis=0)
                    mean_norm = np.linalg.norm(mean_dir)
                    if mean_norm > 1e-10:
                        cosines = boot_norm @ mean_dir / mean_norm
                        boot_aligns.append(float(np.mean(cosines)))
                
                if boot_aligns:
                    bootstrap_results[fname] = {
                        'mean': float(np.mean(boot_aligns)),
                        'ci_low': float(np.percentile(boot_aligns, 2.5)),
                        'ci_high': float(np.percentile(boot_aligns, 97.5)),
                        'observed': feat_aligns[fname]
                    }
            
            if bootstrap_results:
                head_results[f'h{h_idx}'] = bootstrap_results
        
        # 按平均alignment排序
        sorted_heads = sorted(head_results.items(), 
                            key=lambda x: np.mean([v['observed'] for v in x[1].values()]), 
                            reverse=True)
        
        print(f"\n  L{li}:")
        for h_name, boot_data in sorted_heads[:5]:
            for fname, bd in sorted(boot_data.items(), key=lambda x: -x[1]['observed']):
                print(f"    {h_name} {fname}: {bd['observed']:.4f} [{bd['ci_low']:.4f}, {bd['ci_high']:.4f}]")
        
        results[f'L{li}'] = {
            'top_heads': sorted_heads[:10],
            'all_heads': head_results
        }
    
    return results


def test_s4_layer_comparison(info, all_data):
    """S4: 层间对比 — 每层最强head的alignment变化趋势"""
    print("\n" + "="*70)
    print("S4: 层间alignment趋势")
    print("="*70)
    
    n_heads = info['n_heads']
    results = {}
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        # 每层每个head的单特征avg alignment
        best_single = 0
        best_cross = 0
        best_head = None
        
        for h_idx in range(n_heads):
            feat_aligns = []
            all_diffs = []
            for fname in FNAMES:
                diffs = layer_data['head_diffs'][h_idx].get(fname, [])
                if len(diffs) >= 5:
                    arr = np.array(diffs)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    arr_norm = arr / norms
                    mean_dir = arr_norm.mean(axis=0)
                    mean_norm = np.linalg.norm(mean_dir)
                    if mean_norm > 1e-10:
                        cosines = arr_norm @ mean_dir / mean_norm
                        feat_aligns.append(float(np.mean(cosines)))
                    all_diffs.extend(diffs)
            
            if feat_aligns:
                avg_single = float(np.mean(feat_aligns))
                if avg_single > best_single:
                    best_single = avg_single
                    best_head = h_idx
            
            # 跨特征alignment
            if len(all_diffs) >= 10:
                arr = np.array(all_diffs)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                arr_norm = arr / norms
                mean_dir = arr_norm.mean(axis=0)
                mean_norm = np.linalg.norm(mean_dir)
                if mean_norm > 1e-10:
                    cosines = arr_norm @ mean_dir / mean_norm
                    cross = float(np.mean(cosines))
                    if cross > best_cross:
                        best_cross = cross
        
        results[f'L{li}'] = {
            'best_single_feature_alignment': best_single,
            'best_cross_feature_alignment': best_cross,
            'best_head': best_head
        }
        
        print(f"  L{li}: best_single={best_single:.4f}, best_cross={best_cross:.4f}, best_head=h{best_head}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b', 'qwen3'])
    parser.add_argument('--test', type=str, default='all', choices=['s1', 's2', 's3', 's4', 'all'])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Loading {model_name}...")
    model, tokenizer, info = load_model(model_name)
    print(f"  d_model={info['d_model']}, n_layers={info['n_layers']}, n_heads={info['n_heads']}, d_head={info['d_head']}")
    
    out_dir = Path(f"results/causal_fiber/{model_name}_rigorous")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 采样层: 更密集, 尤其末层
    n_layers = info['n_layers']
    sample_layers = sorted(set(
        list(range(max(0, n_layers - 8), n_layers)) +  # 末8层
        list(range(0, n_layers, max(1, n_layers // 8)))  # 每8层采样
    ))
    print(f"  采样层: {sample_layers}")
    
    # 收集数据
    print("\n收集双方法数据...")
    all_data = collect_dual_method_data(model, tokenizer, info, sample_layers)
    
    if args.test in ['s1', 'all']:
        s1_results = test_s1_dual_method(info, all_data)
        with open(out_dir / 's1_dual_method.json', 'w') as f:
            json.dump(s1_results, f, indent=2, default=str)
    
    if args.test in ['s2', 'all']:
        s2_results = test_s2_feature_decomposed(info, all_data)
        with open(out_dir / 's2_feature_decomposed.json', 'w') as f:
            json.dump(s2_results, f, indent=2, default=str)
    
    if args.test in ['s3', 'all']:
        s3_results = test_s3_bootstrap_significance(info, all_data)
        with open(out_dir / 's3_bootstrap.json', 'w') as f:
            json.dump(s3_results, f, indent=2, default=str)
    
    if args.test in ['s4', 'all']:
        s4_results = test_s4_layer_comparison(info, all_data)
        with open(out_dir / 's4_layer_trend.json', 'w') as f:
            json.dump(s4_results, f, indent=2, default=str)
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Free VRAM: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f}GB")
