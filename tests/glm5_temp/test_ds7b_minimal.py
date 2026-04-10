# -*- coding: utf-8 -*-
"""DeepSeek7B 最小化加载测试 - 逐步验证"""
import torch, sys, io, time, gc
import functools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("[1] CUDA检查")
print(f"  CUDA可用: {torch.cuda.is_available()}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n[2] 加载Tokenizer")
from transformers import AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True,
    local_files_only=True, use_fast=False
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  OK! vocab={tok.vocab_size}, 耗时={time.time()-t0:.1f}s")

print("\n[3] 加载模型(CPU, low_cpu_mem_usage=True)")
from transformers import AutoModelForCausalLM
torch.cuda.empty_cache()
gc.collect()
t0 = time.time()
mdl = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, trust_remote_code=True,
    local_files_only=True, low_cpu_mem_usage=True,
    attn_implementation="eager", device_map="cpu"
)
print(f"  CPU加载完成, 耗时={time.time()-t0:.1f}s")

print("\n[4] 迁移到GPU")
t1 = time.time()
mdl = mdl.to("cuda")
mdl.eval()
print(f"  GPU迁移完成, 耗时={time.time()-t1:.1f}s")
print(f"  GPU显存: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

print("\n[5] 推理测试")
inp = tok("Hello world", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = mdl(**inp, output_hidden_states=True)
print(f"  推理OK! n_layers={len(out.hidden_states)}, L0 shape={out.hidden_states[0].shape}")

print("\n[6] FFN权重检查")
layers = mdl.model.layers
mlp0 = layers[0].mlp
if hasattr(mlp0, 'gate_proj'):
    print(f"  gate_proj.weight: {mlp0.gate_proj.weight.shape}")
    print(f"  up_proj.weight: {mlp0.up_proj.weight.shape}")
    print(f"  down_proj.weight: {mlp0.down_proj.weight.shape}")

print("\n[7] 清理")
del mdl, out
torch.cuda.empty_cache()
gc.collect()
print("  完成!")
