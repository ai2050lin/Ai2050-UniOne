"""快速验证patching实现是否正确"""
import os, sys, gc, json, time
import torch
import numpy as np
from pathlib import Path

# 模型加载
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
print(f"Loaded: device={device}")

# 测试对
a = "The cat sat quietly on the mat"
b = "The cat sits quietly on the mat"

# ===== 方法1: 旧脚本方式 (直接修改out[0]) =====
print("\n=== Method 1: Direct modification ===")
for layer_idx in [0, 14, 27]:
    # Source run
    src_hidden = {}
    def capture_src(mod, inp, out, li=layer_idx):
        if isinstance(out, tuple):
            src_hidden[li] = out[0][0, -1, :].detach().clone()
        else:
            src_hidden[li] = out[0, -1, :].detach().clone()
    
    h = model.model.layers[layer_idx].register_forward_hook(capture_src)
    src_ids = tokenizer(a, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        _ = model(src_ids)
    h.remove()
    
    src_vec = src_hidden[layer_idx]
    
    # Clean run
    clean_ids = tokenizer(b, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
    
    # Patched run (direct mod)
    def patch_direct(mod, inp, out, sv=src_vec):
        if isinstance(out, tuple):
            out[0][0, -1, :] = sv.to(out[0].device)
        else:
            out[0, -1, :] = sv.to(out.device)
    
    h = model.model.layers[layer_idx].register_forward_hook(patch_direct)
    with torch.no_grad():
        patched1 = model(clean_ids).logits[0, -1].detach().cpu().float()
    h.remove()
    
    diff1 = patched1 - clean_logits
    l2_1 = torch.norm(diff1).item()
    cos1 = torch.nn.functional.cosine_similarity(clean_logits.unsqueeze(0), patched1.unsqueeze(0)).item()
    print(f"  L{layer_idx}: l2={l2_1:.1f}, cos={cos1:.4f}")

# ===== 方法2: 新脚本方式 (clone + return) =====
print("\n=== Method 2: Clone + return ===")
for layer_idx in [0, 14, 27]:
    # Source run
    src_hidden = {}
    def capture_src2(mod, inp, out, li=layer_idx):
        if isinstance(out, tuple):
            src_hidden[li] = out[0][0, -1, :].detach().clone()
        else:
            src_hidden[li] = out[0, -1, :].detach().clone()
    
    h = model.model.layers[layer_idx].register_forward_hook(capture_src2)
    src_ids = tokenizer(a, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        _ = model(src_ids)
    h.remove()
    
    src_vec = src_hidden[layer_idx]
    
    # Clean run
    clean_ids = tokenizer(b, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
    
    # Patched run (clone + return)
    def patch_clone(mod, inp, out, sv=src_vec):
        if isinstance(out, tuple):
            hidden = out[0].clone()
            hidden[0, -1, :] = sv.to(hidden.device)
            return (hidden,) + out[1:]
        else:
            hidden = out.clone()
            hidden[0, -1, :] = sv.to(hidden.device)
            return hidden
    
    h = model.model.layers[layer_idx].register_forward_hook(patch_clone)
    with torch.no_grad():
        patched2 = model(clean_ids).logits[0, -1].detach().cpu().float()
    h.remove()
    
    diff2 = patched2 - clean_logits
    l2_2 = torch.norm(diff2).item()
    cos2 = torch.nn.functional.cosine_similarity(clean_logits.unsqueeze(0), patched2.unsqueeze(0)).item()
    print(f"  L{layer_idx}: l2={l2_2:.1f}, cos={cos2:.4f}")

print("\nDone!")
