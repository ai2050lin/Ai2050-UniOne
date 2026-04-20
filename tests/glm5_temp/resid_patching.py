"""
Phase CCVII-B: Residual Stream Patching
=========================================
关键发现: 单个head patching l2=0.00 -> 无效!
方案: patch整个residual stream (layer output), 而不是单个head
3个特征, 300对, 全部层
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
    
    # 采样层 (每4层)
    sample_layers = sorted(set(
        [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    print(f"  Sample layers: {sample_layers}")
    
    all_results = {}
    t0 = time.time()
    
    for fname, gen_fn in features.items():
        print(f"\n=== {fname} residual stream patching ===")
        pairs = gen_fn()[:N_PAIRS]
        feat_results = {}
        
        for li in sample_layers:
            if li >= n_layers:
                continue
            
            # Hook在layer的output上 (不是o_proj的input)
            layer = model.model.layers[li]
            
            l2_list = []
            cos_list = []
            
            for text_a, text_b in pairs:
                # 1. 捕获source的layer output
                src_resid = [None]
                def src_hook(mod, inp, out, _li=li):
                    # out可能是tuple, 第一个元素是hidden_states
                    if isinstance(out, tuple):
                        src_resid[0] = out[0].detach()
                    else:
                        src_resid[0] = out.detach()
                
                h1 = layer.register_forward_hook(src_hook)
                try:
                    toks = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad(): _ = model(**toks)
                except: h1.remove(); continue
                h1.remove()
                if src_resid[0] is None: continue
                saved = src_resid[0].clone()  # [batch, seq, d_model]
                
                # 2. Clean run (text_b, no patch)
                try:
                    toks = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=64).to(device)
                    with torch.no_grad(): out = model(**toks)
                    clean = out.logits[0, -1, :].detach().cpu().float()
                except: continue
                
                # 3. Patched run (text_b, patch layer output from text_a)
                def tgt_hook(mod, inp, out, src=saved, _li=li):
                    if isinstance(out, tuple):
                        # 替换hidden_states
                        new_out = (src.to(out[0].device),) + out[1:]
                        return new_out
                    else:
                        return src.to(out.device)
                
                h2 = layer.register_forward_hook(tgt_hook)
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
                feat_results[f'L{li}'] = {
                    'avg_l2': float(np.mean(l2_list)),
                    'std_l2': float(np.std(l2_list)),
                    'avg_cos_sim': float(np.mean(cos_list)),
                    'n': len(l2_list)
                }
                print(f"  L{li}: l2={np.mean(l2_list):.4f}+-{np.std(l2_list):.4f}, cos={np.mean(cos_list):.6f}, n={len(l2_list)}")
        
        all_results[fname] = feat_results
    
    with open(out_dir / 'resid_patching.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s! Results saved.")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model.upper()} Residual Stream Patching")
    print(f"{'='*60}")
    for fname in all_results:
        print(f"\n  {fname}:")
        for lk in sorted(all_results[fname].keys()):
            ld = all_results[fname][lk]
            print(f"    {lk}: l2={ld['avg_l2']:.4f}+-{ld['std_l2']:.4f}, cos={ld['avg_cos_sim']:.6f}")


if __name__ == '__main__':
    main()
