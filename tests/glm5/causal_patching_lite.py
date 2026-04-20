"""
Phase CCVII-Lite: Activation Patching 精简版
=============================================
只做最关键的: tense特征的跨层因果追踪 (逐层逐head)
样本: 200对 (够用但不太慢)
"""
import os, sys, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from causal_megasample import generate_tense_pairs, generate_polarity_pairs, generate_number_pairs


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
    cfg = model.config
    info = {
        'd_model': cfg.hidden_size, 'n_layers': cfg.num_hidden_layers,
        'n_heads': cfg.num_attention_heads, 'd_head': cfg.hidden_size // cfg.num_attention_heads,
        'device': next(model.parameters()).device
    }
    return model, tokenizer, info


def run_patching(model, tokenizer, info, feature_name, n_pairs=200):
    """对指定特征做跨层跨head的activation patching"""
    n_layers = info['n_layers']
    n_heads = info['n_heads']
    d_head = info['d_head']
    device = info['device']
    
    if feature_name == 'tense':
        pairs = generate_tense_pairs()[:n_pairs]
    elif feature_name == 'polarity':
        pairs = generate_polarity_pairs()[:n_pairs]
    elif feature_name == 'number':
        pairs = generate_number_pairs()[:n_pairs]
    else:
        return {}
    
    # 采样层 (每4层 + 末5层)
    sample_layers = sorted(set(
        list(range(max(0, n_layers - 5), n_layers)) +
        list(range(0, n_layers, max(1, n_layers // 6)))
    ))
    
    results = {}
    
    for li in sample_layers:
        o_proj = model.model.layers[li].self_attn.o_proj
        if o_proj.weight.is_meta:
            results[f'L{li}'] = {'error': 'meta_device'}
            continue
        
        head_effects = {}
        
        for h_idx in range(n_heads):
            l2_dists = []
            
            for text_a, text_b in pairs:
                # source_hook: 捕获text_a的head输出
                source_head_out = [None]
                def src_hook(module, input, output, hi=h_idx):
                    if isinstance(input, tuple) and len(input) > 0:
                        h = input[0].detach()
                        s, e = hi * d_head, (hi + 1) * d_head
                        source_head_out[0] = h[0, -1, s:e].clone()
                
                handle = o_proj.register_forward_hook(src_hook)
                try:
                    toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad():
                        _ = model(**toks)
                except:
                    handle.remove()
                    continue
                handle.remove()
                
                if source_head_out[0] is None:
                    continue
                
                saved_src = source_head_out[0]
                
                # target_hook: patch text_b的head输出
                def tgt_hook(module, input, output, src=saved_src, hi=h_idx):
                    if isinstance(input, tuple) and len(input) > 0:
                        h = input[0]
                        s, e = hi * d_head, (hi + 1) * d_head
                        h[0, -1, s:e] = src.to(h.device)
                
                # Clean run (text_b, no patching)
                try:
                    toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad():
                        clean_out = model(**toks)
                    clean_logit = clean_out.logits[0, -1, :].detach().cpu().float()
                except:
                    continue
                
                # Patched run (text_b, with patching)
                handle = o_proj.register_forward_hook(tgt_hook)
                try:
                    toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad():
                        patched_out = model(**toks)
                    patched_logit = patched_out.logits[0, -1, :].detach().cpu().float()
                except:
                    handle.remove()
                    continue
                handle.remove()
                
                # 因果效应 = logit L2距离
                l2 = float(torch.norm(patched_logit - clean_logit))
                l2_dists.append(l2)
            
            if l2_dists:
                head_effects[f'h{h_idx}'] = {
                    'avg_l2': float(np.mean(l2_dists)),
                    'std_l2': float(np.std(l2_dists)),
                    'n': len(l2_dists)
                }
        
        # 排序
        sorted_heads = sorted(head_effects.items(), key=lambda x: x[1]['avg_l2'], reverse=True)
        
        top5 = [(n, d['avg_l2'], d['std_l2'], d['n']) for n, d in sorted_heads[:5]]
        avg_all = float(np.mean([d['avg_l2'] for d in head_effects.values()]))
        
        results[f'L{li}'] = {
            'top5_heads': top5,
            'avg_all_heads': avg_all,
            'all_heads': head_effects
        }
        
        top_str = ', '.join(f"{n}={l2:.2f}" for n, l2, _, _ in top5)
        print(f"  L{li}: avg={avg_all:.2f}, top: {top_str}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b', 'qwen3'])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Loading {model_name}...")
    model, tokenizer, info = load_model(model_name)
    print(f"  d_model={info['d_model']}, n_layers={info['n_layers']}, n_heads={info['n_heads']}")
    
    out_dir = Path(f"results/causal_fiber/{model_name}_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for fname in ['tense', 'polarity', 'number']:
        print(f"\n=== Patching feature: {fname} ===")
        t0 = time.time()
        res = run_patching(model, tokenizer, info, fname, n_pairs=200)
        print(f"  Done in {time.time()-t0:.0f}s")
        all_results[fname] = res
    
    with open(out_dir / 'patching_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Results saved to {out_dir / 'patching_results.json'}")
