"""极简验证: 只测1对, 比较两种patching方式"""
import os, gc
import torch
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

a = "The cat sat quietly on the mat"
b = "The cat sits quietly on the mat"

for layer_idx in [0, 27]:
    print(f"\n=== Layer {layer_idx} ===")
    
    # Method A: Old (multi-hook, direct mod)
    src_outputs = {}
    hooks = []
    def make_hook(li):
        def hook_fn(mod, inp, out):
            if isinstance(out, tuple):
                src_outputs[li] = out[0][0, -1, :].detach().clone()
            else:
                src_outputs[li] = out[0, -1, :].detach().clone()
        return hook_fn
    
    for lidx in [0, 4, 9, 14, 18, 23, 27]:
        if lidx < n_layers:
            h = model.model.layers[lidx].register_forward_hook(make_hook(lidx))
            hooks.append(h)
    
    src_ids = tokenizer(a, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad(): _ = model(src_ids)
    for h in hooks: h.remove()
    
    saved_src = src_outputs[layer_idx]
    
    clean_ids = tokenizer(b, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
    
    # Old patch
    def patch_old(mod, inp, out, sv=saved_src):
        if isinstance(out, tuple):
            out[0][0, -1, :] = sv.to(out[0].device)
        else:
            out[0, -1, :] = sv.to(out.device)
    
    h = model.model.layers[layer_idx].register_forward_hook(patch_old)
    with torch.no_grad():
        patched_old = model(clean_ids).logits[0, -1].detach().cpu().float()
    h.remove()
    
    l2_old = torch.norm(patched_old - clean_logits).item()
    cos_old = torch.nn.functional.cosine_similarity(clean_logits.unsqueeze(0), patched_old.unsqueeze(0)).item()
    
    # Method B: New (single-hook, clone+return)
    src_hidden = {}
    def capture(mod, inp, out):
        if isinstance(out, tuple):
            src_hidden['val'] = out[0][0, -1, :].detach().clone()
        else:
            src_hidden['val'] = out[0, -1, :].detach().clone()
    
    h = model.model.layers[layer_idx].register_forward_hook(capture)
    with torch.no_grad(): _ = model(src_ids)
    h.remove()
    
    src_vec = src_hidden['val']
    
    with torch.no_grad():
        clean_logits2 = model(clean_ids).logits[0, -1].detach().cpu().float()
    
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
    
    l2_new = torch.norm(patched_new - clean_logits2).item()
    cos_new = torch.nn.functional.cosine_similarity(clean_logits2.unsqueeze(0), patched_new.unsqueeze(0)).item()
    
    print(f"  Old: l2={l2_old:.1f}, cos={cos_old:.4f}")
    print(f"  New: l2={l2_new:.1f}, cos={cos_new:.4f}")
    print(f"  Ratio: old/new = {l2_old/max(l2_new, 0.01):.2f}")

print("\nDone!")
