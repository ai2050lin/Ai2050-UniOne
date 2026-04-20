"""
Phase CCVII-C: 正确的Activation Patching
==========================================
关键bug修复: 
  - 之前patch整个layer output → 后续层都看到patched input → 效果与层无关
  - 正确做法: 只在指定层替换最后一个token的hidden state, 不替换其他层
  
方案: 对每层, 用register_forward_hook在OUTPUT上做patch
  1. 先运行source, 收集每层的output[0, -1, :] (最后token)
  2. 再运行target, 在指定层用hook替换output[0, -1, :]为source的
  3. 后续层继续正常计算 → 真正的因果追踪!

3个特征, 300对, 7层采样
"""
import os, sys, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, numpy as np, torch
from pathlib import Path
from causal_megasample import generate_tense_pairs, generate_polarity_pairs, generate_number_pairs

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
    d_model = cfg.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}, device={device}")
    
    out_dir = Path(f"results/causal_fiber/{args.model}_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    features = {
        'tense': generate_tense_pairs,
        'polarity': generate_polarity_pairs,
        'number': generate_number_pairs,
    }
    
    # 采样层
    sample_layers = sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))
    print(f"  Sample layers: {sample_layers}")
    
    all_results = {}
    t0 = time.time()
    
    for fname, gen_fn in features.items():
        print(f"\n=== {fname} causal patching ===")
        pairs = gen_fn()[:N_PAIRS]
        feat_results = {}
        
        for li in sample_layers:
            if li >= n_layers:
                continue
            
            layer = model.model.layers[li]
            
            l2_list = []
            cos_list = []
            n_done = 0
            
            for text_a, text_b in pairs:
                # Step 1: 运行source (text_a), 收集所有层的最后token output
                src_outputs = {}
                hooks = []
                
                def make_hook(layer_idx):
                    def hook_fn(mod, inp, out):
                        if isinstance(out, tuple):
                            h = out[0].detach()
                        else:
                            h = out.detach()
                        src_outputs[layer_idx] = h[0, -1, :].clone()  # [d_model]
                    return hook_fn
                
                for lidx in sample_layers:
                    if lidx < n_layers:
                        h = model.model.layers[lidx].register_forward_hook(make_hook(lidx))
                        hooks.append(h)
                
                try:
                    toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad(): _ = model(**toks)
                except:
                    for h in hooks: h.remove()
                    continue
                for h in hooks: h.remove()
                
                if li not in src_outputs:
                    continue
                saved_src = src_outputs[li]  # [d_model]
                
                # Step 2: Clean run (text_b, no patching)
                try:
                    toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad(): out = model(**toks)
                    clean = out.logits[0, -1, :].detach().cpu().float()
                except:
                    continue
                
                # Step 3: Patched run (text_b, patch layer li output)
                def patch_hook(mod, inp, out, src=saved_src, _li=li):
                    if isinstance(out, tuple):
                        h = out[0]
                        h[0, -1, :] = src.to(h.device)
                    else:
                        out[0, -1, :] = src.to(out.device)
                
                h_patch = layer.register_forward_hook(patch_hook)
                try:
                    toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad(): out = model(**toks)
                    patched = out.logits[0, -1, :].detach().cpu().float()
                except:
                    h_patch.remove()
                    continue
                h_patch.remove()
                
                l2 = float(torch.norm(patched - clean))
                cos = float(torch.nn.functional.cosine_similarity(clean.unsqueeze(0), patched.unsqueeze(0)))
                l2_list.append(l2)
                cos_list.append(cos)
                n_done += 1
            
            if l2_list:
                feat_results[f'L{li}'] = {
                    'avg_l2': float(np.mean(l2_list)),
                    'std_l2': float(np.std(l2_list)),
                    'avg_cos_sim': float(np.mean(cos_list)),
                    'n': len(l2_list)
                }
                print(f"  L{li}: l2={np.mean(l2_list):.4f}+-{np.std(l2_list):.4f}, cos={np.mean(cos_list):.6f}, n={len(l2_list)}")
        
        all_results[fname] = feat_results
    
    with open(out_dir / 'causal_patching.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s!")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model.upper()} Causal Patching (correct)")
    print(f"{'='*60}")
    for fname in all_results:
        print(f"\n  {fname}:")
        for lk in sorted(all_results[fname].keys()):
            ld = all_results[fname][lk]
            print(f"    {lk}: l2={ld['avg_l2']:.4f}+-{ld['std_l2']:.4f}, cos={ld['avg_cos_sim']:.6f}")


if __name__ == '__main__':
    main()
