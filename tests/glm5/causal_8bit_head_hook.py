"""
Phase CCIV: 8bit量化模型的Head Hook因果定位
=============================================

4bit量化的GLM4/DS7B的W_o返回8M×1的压缩张量，无法dequantize。
8bit量化的W_o形状正确([d_model, d_model])，可以分离head。

本脚本:
  S2-8bit: 对DS7B和GLM4使用8bit量化，运行S2 head hook因果定位
  S1-8bit: 同时重跑S1大样本因果PCA（验证8bit与4bit的一致性）

运行:
  python tests/glm5/causal_8bit_head_hook.py --model deepseek7b
  python tests/glm5/causal_8bit_head_hook.py --model glm4
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, gc, time, argparse
import numpy as np
import torch
from pathlib import Path

from causal_megasample import (
    generate_polarity_pairs, generate_tense_pairs, generate_semantic_pairs,
    generate_sentiment_pairs, generate_number_pairs
)


def load_model_8bit(model_name):
    """8bit量化加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    PATHS = {
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # GLM4需要CPU offload
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, local_files_only=True
    )
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def get_model_info(model):
    """获取模型结构信息"""
    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_head = d_model // n_heads
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    return d_model, n_layers, n_heads, d_head, n_kv_heads


# ============================================================
# S2-8bit: Head Hook因果定位 (W_o投影法)
# ============================================================

def test_s2_8bit(model_name, model, tokenizer, device, d_model, n_layers, n_heads, d_head):
    """
    S2-8bit: Head Hook因果定位
    
    方法: 对每个差分向量，用W_o投影到每个head的子空间，计算cos
    8bit的W_o可以正确dequantize为[d_model, n_heads*d_head]
    """
    print("\n" + "=" * 70)
    print("S2-8bit: Head Hook因果定位 (W_o投影法, 8bit量化)")
    print("=" * 70)
    
    feature_generators = {
        'polarity': generate_polarity_pairs,
        'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs,
        'sentiment': generate_sentiment_pairs,
        'number': generate_number_pairs,
    }
    fnames = list(feature_generators.keys())
    
    # 采样层: 最后5层 + 中间5层
    total_layers = n_layers
    sample_layers = sorted(set(
        list(range(max(0, total_layers - 5), total_layers)) +
        list(range(0, total_layers, max(1, total_layers // 5)))
    ))
    print(f"  采样层: {sample_layers}")
    
    results = {}
    
    for li in sample_layers:
        layer = model.model.layers[li]
        t0 = time.time()
        
        # 获取W_o并dequantize
        o_proj = layer.self_attn.o_proj
        W_o_raw = o_proj.weight
        # 处理meta tensor (CPU offload时某些层可能在meta device)
        if W_o_raw.is_meta:
            print(f"\n  L{li}: W_o在meta device上, 跳过此层")
            results[f'L{li}'] = {'error': 'meta_tensor_offloaded'}
            continue
        W_o = W_o_raw.detach().cpu().float()  # 8bit自动dequantize
        if W_o.shape[0] != d_model:
            W_o = W_o.T
        
        print(f"\n  L{li}: W_o shape={W_o.shape}")
        
        # 分离每个head的W_o列
        expected_cols = n_heads * d_head
        if W_o.shape[1] != expected_cols:
            print(f"  WARNING: W_o列数{W_o.shape[1]}≠预期{expected_cols}")
            actual_d_head = W_o.shape[1] // n_heads
            if actual_d_head > 0:
                d_head_local = actual_d_head
            else:
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
            results[f'L{li}'] = {'error': 'no_valid_heads'}
            continue
        
        # 收集差分向量
        all_diffs = []  # [diff_vec, label]
        
        for fname in fnames:
            pairs = feature_generators[fname]()
            
            captured = []
            def make_hook(storage):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple): h = output[0].detach().cpu().float()
                    else: h = output.detach().cpu().float()
                    storage.append(h[0])
                return hook_fn
            
            h = layer.register_forward_hook(make_hook(captured))
            
            reprs_a = []
            reprs_b = []
            
            for i, (text_a, text_b) in enumerate(pairs):
                captured.clear()
                toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64)
                # 确保input_ids在正确设备上
                toks = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in toks.items()}
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if captured:
                    reprs_a.append(captured[0].numpy()[-1])
                
                captured.clear()
                toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64)
                toks = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in toks.items()}
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if captured:
                    reprs_b.append(captured[0].numpy()[-1])
            
            h.remove()
            
            n_pairs = min(len(reprs_a), len(reprs_b))
            for i in range(n_pairs):
                diff = reprs_a[i] - reprs_b[i]
                all_diffs.append((diff, fname))
        
        print(f"  收集差分向量: {len(all_diffs)} (5特征)")
        
        if len(all_diffs) == 0:
            results[f'L{li}'] = {'error': 'no_diffs'}
            continue
        
        # 计算每个head的因果cos
        # 方法: diff投影到W_o_h的列空间 → cos(diff, proj)
        # 如果cos接近1，说明diff完全在head h的子空间内
        
        # 整体attn子空间cos
        all_W_o = W_o  # [d_model, n_heads*d_head]
        overall_subspace_norm = torch.linalg.norm(all_W_o @ torch.linalg.pinv(all_W_o), 'fro')
        
        head_results = {}
        for h_idx in range(n_heads):
            if head_subspaces[h_idx] is None:
                continue
            
            W_h = head_subspaces[h_idx]  # [d_model, d_head_local]
            
            # 对每个特征计算平均cos
            feature_cos = {}
            for fname in fnames:
                diffs_f = [d for d, l in all_diffs if l == fname]
                if len(diffs_f) == 0:
                    continue
                
                cos_vals = []
                for diff in diffs_f:
                    diff_t = torch.tensor(diff, dtype=torch.float32)
                    # 投影到head子空间: proj = W_h @ (W_h^+ @ diff)
                    # 简化: proj = W_h @ pinv(W_h) @ diff
                    # 但直接用最小二乘更稳定
                    # x = lstsq(W_h, diff) → proj = W_h @ x
                    try:
                        x = torch.linalg.lstsq(W_h, diff_t)
                        proj = W_h @ x.solution
                        cos = torch.nn.functional.cosine_similarity(
                            diff_t.unsqueeze(0), proj.unsqueeze(0)
                        ).item()
                        cos_vals.append(cos)
                    except:
                        continue
                
                if cos_vals:
                    feature_cos[fname] = {
                        'mean': float(np.mean(cos_vals)),
                        'std': float(np.std(cos_vals)),
                        'n': len(cos_vals)
                    }
            
            # 全特征平均cos
            all_cos = []
            for diff, _ in all_diffs:
                diff_t = torch.tensor(diff, dtype=torch.float32)
                try:
                    x = torch.linalg.lstsq(W_h, diff_t)
                    proj = W_h @ x.solution
                    cos = torch.nn.functional.cosine_similarity(
                        diff_t.unsqueeze(0), proj.unsqueeze(0)
                    ).item()
                    all_cos.append(cos)
                except:
                    continue
            
            if all_cos:
                head_results[f'h{h_idx}'] = {
                    'mean_cos': float(np.mean(all_cos)),
                    'std_cos': float(np.std(all_cos)),
                    'feature_cos': feature_cos,
                    'n': len(all_cos)
                }
        
        # 排序找top head
        sorted_heads = sorted(head_results.items(), key=lambda x: abs(x[1]['mean_cos']), reverse=True)
        
        t1 = time.time()
        print(f"  耗时: {t1-t0:.1f}s")
        print(f"  Top 5 heads (by |cos|):")
        for h_name, h_data in sorted_heads[:5]:
            print(f"    {h_name}: cos={h_data['mean_cos']:+.4f} ± {h_data['std_cos']:.4f}")
            # 各特征cos
            for fname in fnames:
                if fname in h_data['feature_cos']:
                    print(f"      {fname}: {h_data['feature_cos'][fname]['mean']:+.4f}")
        
        results[f'L{li}'] = {
            'top_heads': [(h_name, h_data['mean_cos']) for h_name, h_data in sorted_heads[:10]],
            'all_heads': {h_name: h_data for h_name, h_data in sorted_heads},
            'n_diffs': len(all_diffs)
        }
    
    return results


# ============================================================
# S1-8bit: 大样本因果PCA (验证8bit与4bit一致性)
# ============================================================

def test_s1_8bit(model_name, model, tokenizer, device, d_model, n_layers, n_pairs_per_feature=100):
    """
    S1-8bit: 精简版大样本因果PCA
    只对最后一层做PCA，验证8bit与4bit/FP16的一致性
    """
    print("\n" + "=" * 70)
    print(f"S1-8bit: 因果PCA (最后一层, {n_pairs_per_feature}对/特征)")
    print("=" * 70)
    
    feature_generators = {
        'polarity': generate_polarity_pairs,
        'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs,
        'sentiment': generate_sentiment_pairs,
        'number': generate_number_pairs,
    }
    fnames = list(feature_generators.keys())
    last_layer = n_layers - 1
    layer = model.model.layers[last_layer]
    
    all_diffs = []
    all_labels = []
    
    for fname in fnames:
        pairs = feature_generators[fname]()[:n_pairs_per_feature]
        
        captured = []
        def make_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple): h = output[0].detach().cpu().float()
                else: h = output.detach().cpu().float()
                storage.append(h[0])
            return hook_fn
        
        h = layer.register_forward_hook(make_hook(captured))
        
        reprs_a = []
        reprs_b = []
        
        for text_a, text_b in pairs:
            captured.clear()
            toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64)
            toks = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in toks.items()}
            with torch.no_grad():
                try: _ = model(**toks)
                except: continue
            if captured:
                reprs_a.append(captured[0].numpy()[-1])
            
            captured.clear()
            toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64)
            toks = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in toks.items()}
            with torch.no_grad():
                try: _ = model(**toks)
                except: continue
            if captured:
                reprs_b.append(captured[0].numpy()[-1])
        
        h.remove()
        
        n = min(len(reprs_a), len(reprs_b))
        for i in range(n):
            diff = reprs_a[i] - reprs_b[i]
            all_diffs.append(diff)
            all_labels.append(fname)
        
        print(f"  {fname}: {n} pairs")
    
    if len(all_diffs) < 10:
        print("  ERROR: 太少差分向量")
        return None
    
    # PCA
    X = np.array(all_diffs)
    mean = X.mean(axis=0)
    X_c = X - mean
    
    # SVD
    n_components = min(100, len(X_c) - 1, X_c.shape[1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_c)
    
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    
    # 归一化PCA
    X_norm = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-10)
    pca_norm = PCA(n_components=n_components)
    pca_norm.fit(X_norm)
    var_norm = pca_norm.explained_variance_ratio_
    
    print(f"\n  原始PCA: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%, 前20PC={cum[min(19,len(cum)-1)]*100:.1f}%")
    print(f"  归一化PCA: PC1={var_norm[0]*100:.1f}%, PC2={var_norm[1]*100:.1f}%")
    
    # PC主导特征
    pc_dom = []
    for pc_i in range(min(10, n_components)):
        comp = pca.components_[pc_i]
        scores = {}
        for fname in fnames:
            mask = np.array([l == fname for l in all_labels])
            if mask.sum() > 0:
                proj = X_c[mask] @ comp
                scores[fname] = float(np.mean(np.abs(proj)))
        dom = max(scores, key=scores.get) if scores else 'unknown'
        pc_dom.append({'pc': pc_i, 'dominant': dom, 'score': scores.get(dom, 0)})
        print(f"  PC{pc_i}: dominant={dom}, score={scores.get(dom, 0):.1f}")
    
    return {
        'n_total_diffs': len(all_diffs),
        'raw_pca': {
            'top10_var': var[:10].tolist(),
            'cumulative_var_top20': cum[:20].tolist() if len(cum) >= 20 else cum.tolist(),
        },
        'norm_pca': {
            'top10_var': var_norm[:10].tolist(),
        },
        'pc_dominance': pc_dom,
    }


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b'])
    parser.add_argument('--test', type=str, default='all', choices=['s1', 's2', 'all'])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Loading {model_name} in 8bit...")
    model, tokenizer, device = load_model_8bit(model_name)
    d_model, n_layers, n_heads, d_head, n_kv_heads = get_model_info(model)
    
    print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, n_kv_heads={n_kv_heads}")
    print(f"  Device: {device}")
    
    out_dir = Path(f"results/causal_fiber/{model_name}_8bit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test in ['s1', 'all']:
        s1_results = test_s1_8bit(model_name, model, tokenizer, device, d_model, n_layers, n_pairs_per_feature=150)
        if s1_results:
            with open(out_dir / 's1_results.json', 'w') as f:
                json.dump(s1_results, f, indent=2)
            print(f"S1 results saved to {out_dir / 's1_results.json'}")
    
    if args.test in ['s2', 'all']:
        s2_results = test_s2_8bit(model_name, model, tokenizer, device, d_model, n_layers, n_heads, d_head)
        with open(out_dir / 's2_results.json', 'w') as f:
            json.dump(s2_results, f, indent=2, default=str)
        print(f"S2 results saved to {out_dir / 's2_results.json'}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Free VRAM: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f}GB")
