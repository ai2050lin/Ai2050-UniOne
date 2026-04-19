"""
Phase CCV: 直接Head输出Hook + W_o SVD分析
==========================================

核心改进:
  S1: 直接hook o_proj的input → 分离每个head的独立输出 → 计算head因果一致性
  S2: W_o SVD分析 → 检查L27的cos=0.5是否因为W_o坍缩
  S3: Head-Feature特异性矩阵 → 因果原子定位

运行:
  python tests/glm5/causal_direct_head.py --model deepseek7b
  python tests/glm5/causal_direct_head.py --model glm4
  python tests/glm5/causal_direct_head.py --model qwen3
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


def collect_head_diffs(model, tokenizer, info, target_layers):
    """通用: 收集指定层的head级差分向量"""
    d_model = info['d_model']
    n_heads = info['n_heads']
    d_head = info['d_head']
    device = info['device']
    
    # 结果: layer -> head_idx -> fname -> [diff_vecs]
    all_data = {}
    
    for li in target_layers:
        layer = model.model.layers[li]
        o_proj = layer.self_attn.o_proj
        
        if o_proj.weight.is_meta:
            all_data[li] = None
            continue
        
        layer_data = defaultdict(lambda: defaultdict(list))
        
        for fname in FNAMES:
            pairs = FEATURE_GENERATORS[fname]()
            
            # 用hook捕获o_proj的input
            hook_storage = []
            
            def hook_fn(module, input, output):
                # input[0] 是 o_proj 的输入: [batch, seq, n_heads*d_head]
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
                    reprs_a.append(hook_storage[0][0, -1, :].numpy())  # [n_heads*d_head]
                
                hook_storage.clear()
                toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if hook_storage:
                    reprs_b.append(hook_storage[0][0, -1, :].numpy())
            
            handle.remove()
            
            n = min(len(reprs_a), len(reprs_b))
            for i in range(n):
                diff = reprs_a[i] - reprs_b[i]  # [n_heads*d_head]
                for h_idx in range(n_heads):
                    s, e = h_idx * d_head, (h_idx + 1) * d_head
                    if e <= len(diff):
                        layer_data[h_idx][fname].append(diff[s:e])
        
        all_data[li] = dict(layer_data)
        total = sum(sum(len(v2) for v2 in v1.values()) for v1 in layer_data.values())
        print(f"  L{li}: {total} diffs collected")
    
    return all_data


# ============================================================
# S1: 直接Head因果一致性
# ============================================================

def test_s1(info, all_data):
    print("\n" + "=" * 70)
    print("S1: 直接Head因果一致性 (绕过W_o投影)")
    print("=" * 70)
    
    results = {}
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        head_results = {}
        for h_idx, feat_diffs in layer_data.items():
            all_diffs = []
            all_labels = []
            for fname, diffs in feat_diffs.items():
                for d in diffs:
                    norm = np.linalg.norm(d)
                    if norm > 1e-10:
                        all_diffs.append(d / norm)
                        all_labels.append(fname)
            
            if len(all_diffs) < 10:
                continue
            
            diff_arr = np.array(all_diffs)
            mean_dir = diff_arr.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm < 1e-10:
                continue
            
            # 方向一致性: 每个差分与平均方向的对齐度
            alignment = diff_arr @ mean_dir / mean_norm
            mean_cos = float(np.mean(alignment))
            std_cos = float(np.std(alignment))
            
            # Per-feature alignment
            feature_cos = {}
            for fname in FNAMES:
                mask = np.array([l == fname for l in all_labels])
                if mask.sum() < 5:
                    continue
                f_diffs = diff_arr[mask]
                f_mean = f_diffs.mean(axis=0)
                f_norm = np.linalg.norm(f_mean)
                if f_norm < 1e-10:
                    continue
                f_alignment = f_diffs @ f_mean / f_norm
                feature_cos[fname] = {
                    'mean': float(np.mean(f_alignment)),
                    'std': float(np.std(f_alignment)),
                    'n': int(mask.sum())
                }
            
            # 平均差分范数
            raw_norms = []
            for fname, diffs in feat_diffs.items():
                for d in diffs:
                    raw_norms.append(np.linalg.norm(d))
            mean_raw_norm = float(np.mean(raw_norms))
            
            head_results[f'h{h_idx}'] = {
                'alignment': mean_cos,
                'std': std_cos,
                'mean_norm': mean_raw_norm,
                'feature_cos': feature_cos,
                'n_diffs': len(all_diffs)
            }
        
        sorted_heads = sorted(head_results.items(), key=lambda x: x[1]['alignment'], reverse=True)
        
        print(f"\n  L{li}:")
        for h_name, h_data in sorted_heads[:5]:
            print(f"    {h_name}: alignment={h_data['alignment']:.4f}, norm={h_data['mean_norm']:.4f}")
            for fname in FNAMES:
                if fname in h_data['feature_cos']:
                    fc = h_data['feature_cos'][fname]
                    print(f"      {fname}: alignment={fc['mean']:.4f}")
        
        results[f'L{li}'] = {
            'top_heads': [(h_name, h_data['alignment'], h_data['mean_norm']) for h_name, h_data in sorted_heads[:10]],
            'all_heads': {h_name: h_data for h_name, h_data in sorted_heads}
        }
    
    return results


# ============================================================
# S2: W_o SVD分析
# ============================================================

def test_s2(model, info, target_layers):
    print("\n" + "=" * 70)
    print("S2: W_o SVD分析")
    print("=" * 70)
    
    d_model = info['d_model']
    n_heads = info['n_heads']
    d_head = info['d_head']
    results = {}
    
    for li in target_layers:
        o_proj = model.model.layers[li].self_attn.o_proj
        if o_proj.weight.is_meta:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        W_o = o_proj.weight.detach().cpu().float()
        if W_o.shape[0] != d_model:
            W_o = W_o.T
        
        U, S, Vt = torch.linalg.svd(W_o, full_matrices=False)
        S = S.numpy()
        
        S2 = S ** 2
        S2_cum = np.cumsum(S2) / S2.sum()
        eff_rank_90 = int(np.searchsorted(S2_cum, 0.90)) + 1
        eff_rank_99 = int(np.searchsorted(S2_cum, 0.99)) + 1
        cond = float(S[0] / S[-1]) if S[-1] > 0 else float('inf')
        top1_ratio = float(S2[0] / S2.sum())
        top5_ratio = float(S2[:5].sum() / S2.sum())
        
        # Head子空间重叠度
        head_corrs = []
        for h1 in range(min(n_heads, 8)):
            for h2 in range(h1+1, min(n_heads, 8)):
                W1 = W_o[:, h1*d_head:(h1+1)*d_head]
                W2 = W_o[:, h2*d_head:(h2+1)*d_head]
                c = torch.nn.functional.cosine_similarity(W1.flatten().unsqueeze(0), W2.flatten().unsqueeze(0)).item()
                head_corrs.append(c)
        mean_head_corr = float(np.mean(head_corrs)) if head_corrs else 0
        
        print(f"\n  L{li}: top5_S={S[:5].tolist()}")
        print(f"    eff_rank: 90%→{eff_rank_90}, 99%→{eff_rank_99}, cond={cond:.1f}")
        print(f"    S^2: top1={top1_ratio*100:.1f}%, top5={top5_ratio*100:.1f}%")
        print(f"    Head子空间平均相关: {mean_head_corr:.4f}")
        
        results[f'L{li}'] = {
            'singular_values_top10': S[:10].tolist(),
            'eff_rank_90': eff_rank_90, 'eff_rank_99': eff_rank_99,
            'condition_number': cond,
            'top1_ratio': top1_ratio, 'top5_ratio': top5_ratio,
            'mean_head_corr': mean_head_corr
        }
    
    return results


# ============================================================
# S3: Head-Feature特异性矩阵
# ============================================================

def test_s3(info, all_data):
    print("\n" + "=" * 70)
    print("S3: Head-Feature特异性矩阵 (因果原子)")
    print("=" * 70)
    
    n_heads = info['n_heads']
    results = {}
    
    for li, layer_data in all_data.items():
        if layer_data is None:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        # 因果范数矩阵: [n_heads, n_features]
        norm_matrix = np.zeros((n_heads, len(FNAMES)))
        
        for h_idx, feat_diffs in layer_data.items():
            for fi, fname in enumerate(FNAMES):
                diffs = feat_diffs.get(fname, [])
                total_norm = sum(np.linalg.norm(d) for d in diffs)
                norm_matrix[h_idx, fi] = total_norm
        
        # 归一化为特异性
        feat_totals = norm_matrix.sum(axis=0, keepdims=True) + 1e-10
        specificity = norm_matrix / feat_totals
        
        # 每个head的主导特征
        head_dom = {}
        for h_idx in range(n_heads):
            if norm_matrix[h_idx].sum() < 1e-10:
                continue
            dom_fi = np.argmax(specificity[h_idx])
            head_dom[f'h{h_idx}'] = {
                'dominant_feature': FNAMES[dom_fi],
                'dominant_score': float(specificity[h_idx, dom_fi]),
                'norm_profile': {FNAMES[fi]: float(norm_matrix[h_idx, fi]) for fi in range(len(FNAMES))},
                'specificity': {FNAMES[fi]: float(specificity[h_idx, fi]) for fi in range(len(FNAMES))}
            }
        
        # 每个特征的主导head
        feat_dom = {}
        for fi, fname in enumerate(FNAMES):
            dom_hi = np.argmax(specificity[:, fi])
            feat_dom[fname] = {
                'dominant_head': f'h{dom_hi}',
                'dominant_score': float(specificity[dom_hi, fi])
            }
        
        print(f"\n  L{li}:")
        # Top 5 head by total norm
        total_norms = norm_matrix.sum(axis=1)
        top_h = np.argsort(total_norms)[::-1][:5]
        for h_idx in top_h:
            if f'h{h_idx}' in head_dom:
                hd = head_dom[f'h{h_idx}']
                print(f"    h{h_idx}: norm={total_norms[h_idx]:.2f}, dominant={hd['dominant_feature']}({hd['dominant_score']:.3f})")
        
        for fname in FNAMES:
            fd = feat_dom[fname]
            print(f"    {fname} → {fd['dominant_head']} (score={fd['dominant_score']:.3f})")
        
        results[f'L{li}'] = {
            'head_dominant': head_dom,
            'feat_dominant': feat_dom,
            'specificity_matrix': specificity.tolist()
        }
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b', 'qwen3'])
    parser.add_argument('--test', type=str, default='all', choices=['s1', 's2', 's3', 'all'])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Loading {model_name}...")
    model, tokenizer, info = load_model(model_name)
    print(f"  d_model={info['d_model']}, n_layers={info['n_layers']}, n_heads={info['n_heads']}, d_head={info['d_head']}")
    
    out_dir = Path(f"results/causal_fiber/{model_name}_direct_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 采样层
    n_layers = info['n_layers']
    sample_layers = sorted(set(
        list(range(max(0, n_layers - 5), n_layers)) +
        list(range(0, n_layers, max(1, n_layers // 5)))
    ))
    print(f"  采样层: {sample_layers}")
    
    # 收集数据(一次遍历, 三测试共享)
    if args.test in ['s1', 's3', 'all']:
        print("\n收集head级差分向量...")
        all_data = collect_head_diffs(model, tokenizer, info, sample_layers)
    
    if args.test in ['s1', 'all']:
        s1_results = test_s1(info, all_data)
        with open(out_dir / 's1_direct_head.json', 'w') as f:
            json.dump(s1_results, f, indent=2, default=str)
    
    if args.test in ['s2', 'all']:
        s2_results = test_s2(model, info, sample_layers)
        with open(out_dir / 's2_wo_svd.json', 'w') as f:
            json.dump(s2_results, f, indent=2, default=str)
    
    if args.test in ['s3', 'all']:
        s3_results = test_s3(info, all_data)
        with open(out_dir / 's3_causal_atoms.json', 'w') as f:
            json.dump(s3_results, f, indent=2, default=str)
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Free VRAM: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f}GB")
