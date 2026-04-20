"""
快速验证: 比较两种patching方式的l2差异
只测10对, 只测L0和L27的tense
"""
import os, sys, gc, time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, local_files_only=True
)
model.eval()
device = next(model.parameters()).device
n_layers = model.config.num_hidden_layers
print(f"Loaded: n_layers={n_layers}, device={device}")

pairs = [
    ("The cat sat quietly on the mat", "The cat sits quietly on the mat"),
    ("She walked to the store yesterday", "She walks to the store today"),
    ("He played guitar every evening", "He plays guitar every evening"),
    ("They worked on the project", "They work on the project"),
    ("The dog ran across the field", "The dog runs across the field"),
    ("She wrote a long letter", "She writes a long letter"),
    ("He drove the car fast", "He drives the car fast"),
    ("They built a new house", "They build a new house"),
    ("The bird flew over the lake", "The bird flies over the lake"),
    ("She cooked dinner for us", "She cooks dinner for us"),
]

for layer_idx in [0, 27]:
    print(f"\n=== Layer {layer_idx} ===")
    
    l2s_old = []
    l2s_new = []
    
    for a, b in pairs:
        # === Old method: register hooks for ALL layers, direct modification ===
        src_outputs = {}
        hooks = []
        sample_layers = [0, 4, 9, 14, 18, 23, 27]
        
        def make_hook(li):
            def hook_fn(mod, inp, out):
                if isinstance(out, tuple):
                    h = out[0].detach()
                else:
                    h = out.detach()
                src_outputs[li] = h[0, -1, :].clone()
            return hook_fn
        
        for lidx in sample_layers:
            if lidx < n_layers:
                h = model.model.layers[lidx].register_forward_hook(make_hook(lidx))
                hooks.append(h)
        
        src_ids = tokenizer(a, return_tensors='pt', truncation=True, max_length=64)['input_ids'].to(device)
        with torch.no_grad():
            _ = model(src_ids)
        for h in hooks: h.remove()
        
        if layer_idx not in src_outputs:
            print(f"  Skipping (no src for L{layer_idx})")
            continue
        saved_src = src_outputs[layer_idx]
        
        # Clean run
        clean_ids = tokenizer(b, return_tensors='pt', truncation=True, max_length=64)['input_ids'].to(device)
        with torch.no_grad():
            clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
        
        # Old patch: direct modification
        def patch_old(mod, inp, out, sv=saved_src):
            if isinstance(out, tuple):
                out[0][0, -1, :] = sv.to(out[0].device)
            else:
                out[0, -1, :] = sv.to(out.device)
        
        h_patch = model.model.layers[layer_idx].register_forward_hook(patch_old)
        with torch.no_grad():
            patched_old = model(clean_ids).logits[0, -1].detach().cpu().float()
        h_patch.remove()
        
        diff_old = patched_old - clean_logits
        l2_old = torch.norm(diff_old).item()
        l2s_old.append(l2_old)
        
        # === New method: only target layer hook, clone + return ===
        src_hidden = {}
        def capture_new(mod, inp, out):
            if isinstance(out, tuple):
                src_hidden['val'] = out[0][0, -1, :].detach().clone()
            else:
                src_hidden['val'] = out[0, -1, :].detach().clone()
        
        h = model.model.layers[layer_idx].register_forward_hook(capture_new)
        with torch.no_grad():
            _ = model(src_ids)
        h.remove()
        
        src_vec = src_hidden['val']
        
        # Clean run (again, fresh)
        with torch.no_grad():
            clean_logits2 = model(clean_ids).logits[0, -1].detach().cpu().float()
        
        # New patch: clone + return
        def patch_new(mod, inp, out, sv=src_vec):
            if isinstance(out, tuple):
                hidden = out[0].clone()
                hidden[0, -1, :] = sv.to(hidden.device)
                return (hidden,) + out[1:]
            else:
                hidden = out.clone()
                hidden[0, -1, :] = sv.to(hidden.device)
                return hidden
        
        h = model.model.layers[layer_idx].register_forward_hook(patch_new)
        with torch.no_grad():
            patched_new = model(clean_ids).logits[0, -1].detach().cpu().float()
        h.remove()
        
        diff_new = patched_new - clean_logits2
        l2_new = torch.norm(diff_new).item()
        l2s_new.append(l2_new)
        
        print(f"  old={l2_old:.1f}, new={l2_new:.1f}, ratio={l2_old/max(l2_new,0.01):.2f}")
    
    print(f"  OLD avg={np.mean(l2s_old):.1f}, NEW avg={np.mean(l2s_new):.1f}")

print("\nDone!")
