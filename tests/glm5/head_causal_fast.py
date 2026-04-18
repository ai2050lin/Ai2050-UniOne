"""
Phase CLXXXVI (优化版): 注意力头级别因果追踪
=============================================

高效实现:
1. 预先计算基线, 避免重复forward
2. 只在关键层做逐头ablation
3. 用eager attention获取注意力权重
4. 同时计算否定句和肯定句的ablation效果

运行:
  python tests/glm5/head_causal_fast.py --model qwen3
  python tests/glm5/head_causal_fast.py --model glm4
  python tests/glm5/head_causal_fast.py --model deepseek7b
"""

import os
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


POLARITY_PAIRS = [
    ("The cat is here", "The cat is not here"),
    ("The dog is happy", "The dog is not happy"),
    ("The book was found", "The book was not found"),
    ("I like the car", "I do not like the car"),
    ("She knows the answer", "She does not know the answer"),
    ("The house is big", "The house is not big"),
    ("The river flows north", "The river does not flow north"),
    ("He can swim", "He cannot swim"),
    ("The bird will come", "The bird will not come"),
    ("The door was closed", "The door was not closed"),
    ("The phone is working", "The phone is not working"),
    ("The flower has bloomed", "The flower has not bloomed"),
    ("I understand the plan", "I do not understand the plan"),
    ("She likes the movie", "She does not like the movie"),
    ("The bridge is safe", "The bridge is not safe"),
    ("The child was playing", "The child was not playing"),
    ("The star is visible", "The star is not visible"),
    ("The cloud disappeared", "The cloud did not disappear"),
    ("The key works well", "The key does not work well"),
    ("The table holds weight", "The table does not hold weight"),
]


def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation='eager',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True,
            attn_implementation='eager',
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def get_not_token_id(tokenizer):
    test_ids = tokenizer.encode("The cat is not here", add_special_tokens=False)
    for tid in test_ids:
        if tokenizer.decode([tid]).strip().lower() == "not":
            return tid
    ids = tokenizer.encode(" not", add_special_tokens=False)
    return ids[0] if ids else None


def make_ablate_hook(head_idx, n_heads, head_dim):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, :, head_idx*head_dim:(head_idx+1)*head_dim] = 0.0
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[:, :, head_idx*head_dim:(head_idx+1)*head_dim] = 0.0
            return h
    return hook_fn


def run_experiment(model_name):
    print(f"\n{'='*70}")
    print(f"Phase CLXXXVI: Head Causal Tracing - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    layers = model.model.layers
    n_layers = len(layers)
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    head_dim = d_model // n_heads
    not_tok_id = get_not_token_id(tokenizer)
    
    print(f"  n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
    print(f"  not_tok_id={not_tok_id}")
    
    pairs = POLARITY_PAIRS[:12]  # 12 pairs for efficiency
    
    # Step 1: Baseline
    print("\n[Step 1] Computing baselines...")
    base_neg_not = []
    base_aff_not = []
    for aff, neg in pairs:
        # 否定句
        toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float()
        val = float(logits[not_tok_id]) if not_tok_id else 0.0
        if not np.isfinite(val):
            val = 0.0
        base_neg_not.append(val)
        
        # 肯定句
        toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :].detach().cpu().float()
        val = float(logits[not_tok_id]) if not_tok_id else 0.0
        if not np.isfinite(val):
            val = 0.0
        base_aff_not.append(val)
    
    avg_neg = np.mean(base_neg_not)
    avg_aff = np.mean(base_aff_not)
    base_diff = avg_neg - avg_aff
    print(f"  Baseline: neg_not={avg_neg:.3f}, aff_not={avg_aff:.3f}, diff={base_diff:.3f}")
    
    # Step 2: Full layer ablation scan
    print("\n[Step 2] Full layer ablation scan...")
    layer_full_impact = {}
    
    for li in range(n_layers):
        neg_deltas = []
        aff_deltas = []
        
        for i, (aff, neg) in enumerate(pairs):
            hooks = []
            for h in range(n_heads):
                hooks.append(layers[li].self_attn.register_forward_hook(
                    make_ablate_hook(h, n_heads, head_dim)))
            
            # 否定句
            toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
            neg_deltas.append(float(logits[not_tok_id]) - base_neg_not[i] if not_tok_id else 0.0)
            
            # 肯定句
            toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
            aff_deltas.append(float(logits[not_tok_id]) - base_aff_not[i] if not_tok_id else 0.0)
            
            for h in hooks:
                h.remove()
        
        avg_neg_delta = np.mean(neg_deltas)
        avg_aff_delta = np.mean(aff_deltas)
        pol_delta = avg_neg_delta - avg_aff_delta
        
        layer_full_impact[li] = {
            'neg_delta': float(avg_neg_delta),
            'aff_delta': float(avg_aff_delta),
            'pol_delta': float(pol_delta),
        }
        
        marker = " *" if abs(avg_neg_delta) > 0.3 or abs(pol_delta) > 0.2 else ""
        print(f"  L{li:2d}: neg_d={avg_neg_delta:+.4f}, aff_d={avg_aff_delta:+.4f}, pol_d={pol_delta:+.4f}{marker}")
    
    # Find top layers
    top_layers_neg = sorted(layer_full_impact.items(), key=lambda x: abs(x[1]['neg_delta']), reverse=True)[:5]
    top_layers_pol = sorted(layer_full_impact.items(), key=lambda x: abs(x[1]['pol_delta']), reverse=True)[:5]
    
    print(f"\n  Top-5 by neg impact: {[(li, round(d['neg_delta'],4)) for li, d in top_layers_neg]}")
    print(f"  Top-5 by pol impact: {[(li, round(d['pol_delta'],4)) for li, d in top_layers_pol]}")
    
    # Step 3: Per-head ablation on key layers
    key_layers = sorted(set([li for li, _ in top_layers_neg[:3]] + [li for li, _ in top_layers_pol[:3]]))
    print(f"\n[Step 3] Per-head ablation on key layers: {key_layers}")
    
    head_results = {}
    
    for li in key_layers:
        print(f"\n  L{li} per-head ablation...")
        heads_data = []
        
        for h_idx in range(n_heads):
            neg_deltas = []
            aff_deltas = []
            
            for i, (aff, neg) in enumerate(pairs):
                hook = layers[li].self_attn.register_forward_hook(
                    make_ablate_hook(h_idx, n_heads, head_dim))
                
                # 否定句
                toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                hook.remove()
                neg_deltas.append(float(logits[not_tok_id]) - base_neg_not[i] if not_tok_id else 0.0)
                
                # 肯定句
                hook = layers[li].self_attn.register_forward_hook(
                    make_ablate_hook(h_idx, n_heads, head_dim))
                toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
                hook.remove()
                aff_deltas.append(float(logits[not_tok_id]) - base_aff_not[i] if not_tok_id else 0.0)
            
            avg_neg_d = np.mean(neg_deltas)
            avg_aff_d = np.mean(aff_deltas)
            pol_d = avg_neg_d - avg_aff_d
            
            heads_data.append({
                'head': h_idx,
                'neg_delta': round(float(avg_neg_d), 4),
                'aff_delta': round(float(avg_aff_d), 4),
                'pol_delta': round(float(pol_d), 4),
            })
        
        # Sort by |pol_delta|
        sorted_heads = sorted(heads_data, key=lambda x: abs(x['pol_delta']), reverse=True)
        print(f"  Top-5 by polarity:")
        for r in sorted_heads[:5]:
            print(f"    H{r['head']:2d}: neg_d={r['neg_delta']:+.4f}, aff_d={r['aff_delta']:+.4f}, pol_d={r['pol_delta']:+.4f}")
        
        sorted_neg = sorted(heads_data, key=lambda x: abs(x['neg_delta']), reverse=True)
        print(f"  Top-5 by neg impact:")
        for r in sorted_neg[:5]:
            print(f"    H{r['head']:2d}: neg_d={r['neg_delta']:+.4f}, aff_d={r['aff_delta']:+.4f}, pol_d={r['pol_delta']:+.4f}")
        
        head_results[li] = heads_data
    
    # Step 4: Attention weight analysis
    print(f"\n[Step 4] Attention weight analysis...")
    
    attn_results = {}
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    for li in sample_layers:
        head_attn_to_not = defaultdict(list)
        
        for aff, neg in pairs:
            # Find not position
            toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            input_ids = toks.input_ids[0].tolist()
            not_pos = None
            for i, tid in enumerate(input_ids):
                if tokenizer.decode([tid]).strip().lower() == "not":
                    not_pos = i
                    break
            
            if not_pos is None:
                continue
            
            last_pos = len(input_ids) - 1
            
            with torch.no_grad():
                outputs = model(**toks, output_attentions=True)
            
            if outputs.attentions is None:
                continue
            
            attn = outputs.attentions[li][0].detach().cpu().float().numpy()
            
            for h in range(n_heads):
                if not_pos < attn.shape[-1] and last_pos < attn.shape[-2]:
                    head_attn_to_not[h].append(float(attn[h, last_pos, not_pos]))
        
        # Average
        avg_attn = {}
        for h in range(n_heads):
            if head_attn_to_not[h]:
                avg_attn[h] = np.mean(head_attn_to_not[h])
        
        sorted_attn = sorted(avg_attn.items(), key=lambda x: x[1], reverse=True)
        
        layer_avg = np.mean(list(avg_attn.values())) if avg_attn else 0
        attn_results[li] = {
            'avg_attn_to_not': float(layer_avg),
            'top5_heads': [(h, round(a, 4)) for h, a in sorted_attn[:5]],
        }
        
        print(f"  L{li}: avg_attn_last->not={layer_avg:.4f}, top5={[(h,round(a,4)) for h,a in sorted_attn[:5]]}")
    
    # Step 5: Verification - Top polarity heads vs random heads
    print(f"\n[Step 5] Verification: Top polarity heads vs random...")
    
    # Collect top polarity heads across key layers
    all_pol_heads = []
    for li, heads_data in head_results.items():
        sorted_heads = sorted(heads_data, key=lambda x: abs(x['pol_delta']), reverse=True)
        for r in sorted_heads[:3]:
            all_pol_heads.append((li, r['head'], r['pol_delta']))
    
    all_pol_heads.sort(key=lambda x: abs(x[2]), reverse=True)
    top5_pol = all_pol_heads[:5]
    
    # Random 5 heads
    np.random.seed(42)
    all_possible = [(li, h) for li in head_results for h in range(n_heads)]
    rand5_idx = np.random.choice(len(all_possible), size=5, replace=False)
    rand5 = [all_possible[i] for i in rand5_idx]
    
    # Ablate top5 and random5
    for label, heads_list in [("Top5_pol", top5_pol), ("Random5", rand5)]:
        neg_logits = []
        aff_logits = []
        
        for aff, neg in pairs:
            # 否定句
            hooks = []
            for item in heads_list:
                li = item[0]
                h_idx = item[1]
                hooks.append(layers[li].self_attn.register_forward_hook(
                    make_ablate_hook(h_idx, n_heads, head_dim)))
            
            toks = tokenizer(neg, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
            for h in hooks: h.remove()
            neg_logits.append(float(logits[not_tok_id]) if not_tok_id else 0.0)
            
            # 肯定句
            hooks = []
            for item in heads_list:
                li = item[0]
                h_idx = item[1]
                hooks.append(layers[li].self_attn.register_forward_hook(
                    make_ablate_hook(h_idx, n_heads, head_dim)))
            
            toks = tokenizer(aff, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = model(**toks).logits[0, -1, :].detach().cpu().float().numpy()
            for h in hooks: h.remove()
            aff_logits.append(float(logits[not_tok_id]) if not_tok_id else 0.0)
        
        diff = np.mean(neg_logits) - np.mean(aff_logits)
        reduction = base_diff - diff
        print(f"  {label}: diff={diff:.3f}, reduction={reduction:.3f}")
    
    # Save results
    result_dir = Path(f"results/head_causal/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'model': model_name,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'head_dim': head_dim,
        'baseline': {
            'neg_not': float(avg_neg),
            'aff_not': float(avg_aff),
            'diff': float(base_diff),
        },
        'layer_full_impact': {str(k): v for k, v in layer_full_impact.items()},
        'head_results': {str(k): v for k, v in head_results.items()},
        'attn_results': {str(k): v for k, v in attn_results.items()},
        'top5_pol_heads': [(int(li), int(h), float(d)) for li, h, d in top5_pol],
    }
    
    with open(result_dir / "head_causal_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {result_dir / 'head_causal_results.json'}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    run_experiment(args.model)
