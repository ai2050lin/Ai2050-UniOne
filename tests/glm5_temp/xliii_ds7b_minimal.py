# -*- coding: utf-8 -*-
"""DeepSeek7B 快速验证 - device_map=auto 直接GPU加载"""
import torch, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import functools
print = functools.partial(print, flush=True)

MP = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("[1] device_map=auto GPU direct load")
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
try:
    mdl = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    dev = next(mdl.parameters()).device
    print(f"  OK! device={dev}, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB, {time.time()-t0:.0f}s")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  Falling back to CPU->GPU...")
    mdl = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    mdl = mdl.to("cuda")
    mdl.eval()
    dev = next(mdl.parameters()).device
    print(f"  CPU->GPU OK! device={dev}, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB, {time.time()-t0:.0f}s")

tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True, use_fast=False)
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("[2] Inference test")
inp = tok("The apple is", return_tensors="pt").to(dev)
with torch.no_grad():
    out = mdl(**inp, output_hidden_states=True)
print(f"  OK! n_layers={len(out.hidden_states)}, L0={out.hidden_states[0].shape}")

print("[3] Quick data collection (5 words x 2 templates)")
words = ["apple", "banana", "cat", "red", "sweet"]
templates = ["The {w} is", "A {w} can be"]
n_layers = len(out.hidden_states)
d_model = out.hidden_states[0].shape[-1]
del out

hs_data = {}
for word in words:
    word_hs = []
    for t in templates:
        prompt = t.replace("{w}", word)
        inp = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        word_hs.append([h[0,-1].float().cpu() for h in out.hidden_states])
        del out, inp
    hs_data[word] = word_hs
    print(f"  {word}: collected")

print(f"[4] Cleanup")
del mdl, tok, hs_data
import gc
gc.collect()
torch.cuda.empty_cache()
print("DONE!")
