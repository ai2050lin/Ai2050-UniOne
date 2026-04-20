"""Phase CCVII-Quick: Activation Patching 快速验证 - 仅tense, 5层, 150对"""
import os, sys, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from causal_megasample import generate_tense_pairs, generate_polarity_pairs, generate_number_pairs


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
    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3个特征, 150对, 5层
    features = {
        'tense': generate_tense_pairs,
        'polarity': generate_polarity_pairs,
        'number': generate_number_pairs,
    }
    
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    print(f"  Sample layers: {sample_layers}")
    
    all_results = {}
    t0 = time.time()
    
    for fname, gen_fn in features.items():
        print(f"\n=== {fname} patching ===")
        pairs = gen_fn()[:150]
        feat_results = {}
        
        for li in sample_layers:
            o_proj = model.model.layers[li].self_attn.o_proj
            if o_proj.weight.is_meta:
                feat_results[f'L{li}'] = {'error': 'meta_device'}
                continue
            
            head_effects = {}
            
            for h_idx in range(n_heads):
                l2_list = []
                
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
                    
                    # 2. Clean run (text_b, no patch)
                    try:
                        toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                        with torch.no_grad(): out = model(**toks)
                        clean = out.logits[0, -1, :].detach().cpu().float()
                    except: continue
                    
                    # 3. Patched run (text_b, with head patch from text_a)
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
                    l2_list.append(l2)
                
                if l2_list:
                    head_effects[f'h{h_idx}'] = float(np.mean(l2_list))
            
            if head_effects:
                sorted_h = sorted(head_effects.items(), key=lambda x: x[1], reverse=True)
                top5 = sorted_h[:5]
                avg_all = float(np.mean(list(head_effects.values())))
                feat_results[f'L{li}'] = {
                    'top5': [(n, v) for n, v in top5],
                    'avg_all': avg_all,
                    'all_heads': head_effects
                }
                top_str = ', '.join(f"{n}={v:.2f}" for n, v in top5)
                print(f"  L{li}: avg={avg_all:.2f}, top: {top_str}")
        
        all_results[fname] = feat_results
    
    with open(out_dir / 'patching_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s! Results saved to {out_dir / 'patching_results.json'}")
    
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
