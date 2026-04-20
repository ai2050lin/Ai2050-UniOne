"""
Phase CCVII-Fast: Activation Patching 快速版
=============================================
关键优化: 只patch之前发现的高alignment head, 不遍历所有head
每层只patch top-3 head (从Phase CCVI S2结果中选)
3个特征, 300对, 5层, 每层3个head -> 快很多!
"""
import os, sys, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, numpy as np, torch
from pathlib import Path
from causal_megasample import generate_tense_pairs, generate_polarity_pairs, generate_number_pairs


# Phase CCVI发现的每层top head (by single-feature alignment)
TOP_HEADS = {
    'deepseek7b': {
        0: [6, 9, 15],    # L0: h6=0.624, h9, h15
        7: [24, 11, 9],   # L7
        14: [6, 14, 27],  # L14
        21: [9, 8, 23],   # L21
        27: [25, 22, 18], # L27
    },
    'qwen3': {
        0: [4, 25, 5],
        9: [24, 31, 30],
        18: [11, 9, 26],
        27: [23, 24, 22],
        35: [21, 20, 6],
    },
    'glm4': {
        0: [27, 25, 2],
        10: [7, 11, 1],
        20: [24, 4, 21],
        30: [2, 14, 9],
        35: [30, 26, 28],
    }
}

N_PAIRS = 300


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b', 'qwen3'])
    args = parser.parse_args()
    
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    print(f"Loading {args.model}...")
    path = PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    if args.model in ["glm4", "deepseek7b"]:
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
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, device={device}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    features = {
        'tense': generate_tense_pairs,
        'polarity': generate_polarity_pairs,
        'number': generate_number_pairs,
    }
    
    model_heads = TOP_HEADS.get(args.model, {})
    if not model_heads:
        # Fallback: 5层均匀采样, 每层测3个head
        model_heads = {li: [0, n_heads//2, n_heads-1] for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]}
    
    all_results = {}
    t0 = time.time()
    
    for fname, gen_fn in features.items():
        print(f"\n=== {fname} patching ===")
        pairs = gen_fn()[:N_PAIRS]
        feat_results = {}
        
        for li, heads in model_heads.items():
            if li >= n_layers:
                continue
            o_proj = model.model.layers[li].self_attn.o_proj
            if o_proj.weight.is_meta:
                feat_results[f'L{li}'] = {'error': 'meta_device'}
                continue
            
            head_effects = {}
            
            for h_idx in heads:
                l2_list = []
                cos_list = []
                
                for text_a, text_b in pairs:
                    # 1. 捕获source head output
                    src_out = [None]
                    def src_hook(mod, inp, out, hi=h_idx):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            h = inp[0].detach()
                            s, e = hi * d_head, (hi + 1) * d_head
                            src_out[0] = h[0, -1, s:e].clone()
                    
                    h1 = o_proj.register_forward_hook(src_hook)
                    try:
                        toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64).to(device)
                        with torch.no_grad(): _ = model(**toks)
                    except: h1.remove(); continue
                    h1.remove()
                    if src_out[0] is None: continue
                    saved = src_out[0]
                    
                    # 2. Clean run
                    try:
                        toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                        with torch.no_grad(): out = model(**toks)
                        clean = out.logits[0, -1, :].detach().cpu().float()
                    except: continue
                    
                    # 3. Patched run
                    def tgt_hook(mod, inp, out, src=saved, hi=h_idx):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            h = inp[0]
                            s, e = hi * d_head, (hi + 1) * d_head
                            h[0, -1, s:e] = src.to(h.device)
                    
                    h2 = o_proj.register_forward_hook(tgt_hook)
                    try:
                        toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                        with torch.no_grad(): out = model(**toks)
                        patched = out.logits[0, -1, :].detach().cpu().float()
                    except: h2.remove(); continue
                    h2.remove()
                    
                    l2 = float(torch.norm(patched - clean))
                    cos = float(torch.nn.functional.cosine_similarity(clean.unsqueeze(0), patched.unsqueeze(0)))
                    l2_list.append(l2)
                    cos_list.append(cos)
                
                if l2_list:
                    head_effects[f'h{h_idx}'] = {
                        'avg_l2': float(np.mean(l2_list)),
                        'std_l2': float(np.std(l2_list)),
                        'avg_cos_sim': float(np.mean(cos_list)),
                        'n': len(l2_list)
                    }
            
            if head_effects:
                sorted_h = sorted(head_effects.items(), key=lambda x: x[1]['avg_l2'], reverse=True)
                feat_results[f'L{li}'] = {'heads': head_effects}
                
                top_str = ', '.join(f"{n}: l2={d['avg_l2']:.2f} cos={d['avg_cos_sim']:.6f}" for n, d in sorted_h)
                print(f"  L{li}: {top_str}")
        
        all_results[fname] = feat_results
    
    with open(out_dir / 'patching_fast.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s! Results saved.")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model.upper()} Activation Patching")
    print(f"{'='*60}")
    for fname in all_results:
        print(f"\n  {fname}:")
        for lk in sorted(all_results[fname].keys()):
            ld = all_results[fname][lk]
            if 'error' in ld:
                print(f"    {lk}: {ld['error']}")
                continue
            heads = ld['heads']
            sorted_h = sorted(heads.items(), key=lambda x: x[1]['avg_l2'], reverse=True)
            for n, d in sorted_h:
                print(f"    {lk} {n}: l2={d['avg_l2']:.4f}, cos={d['avg_cos_sim']:.6f}, n={d['n']}")


if __name__ == '__main__':
    main()
