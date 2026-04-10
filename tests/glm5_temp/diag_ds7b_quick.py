"""DeepSeek7B 快速诊断 — 仅测试关键差异"""
import torch, os, sys, time, gc, traceback
import functools
print = functools.partial(print, flush=True)

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("[1] 环境检查")
print(f"  CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB" if torch.cuda.is_available() else "  CUDA不可用!")

import psutil
mem = psutil.virtual_memory()
print(f"  RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB ({mem.percent}%)")

# 核心差异测试: low_cpu_mem_usage=True vs False
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n[2] 测试: low_cpu_mem_usage=True (GPT5方式)")
try:
    torch.cuda.empty_cache(); gc.collect()
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cpu"
    )
    mdl = mdl.to("cuda"); mdl.eval()
    print(f"  加载+迁移: {time.time()-t0:.1f}s, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB")
    
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad(): out = mdl(**inp, output_hidden_states=True)
    print(f"  推理OK: n_layers={len(out.hidden_states)}")
    
    del mdl, out; torch.cuda.empty_cache(); gc.collect()
    print("  SUCCESS")
except Exception as e:
    print(f"  FAIL: {e}")
    try: del mdl
    except: pass
    torch.cuda.empty_cache(); gc.collect()

print("\n[3] 测试: 无low_cpu_mem_usage (GLM5原方式)")
try:
    torch.cuda.empty_cache(); gc.collect()
    mem1 = psutil.virtual_memory()
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cpu"
    )
    mem2 = psutil.virtual_memory()
    print(f"  CPU加载: {time.time()-t0:.1f}s, RAM: {mem1.used/1e9:.1f}→{mem2.used/1e9:.1f}GB (Δ{(mem2.used-mem1.used)/1e9:.1f}GB)")
    
    mdl = mdl.to("cuda"); mdl.eval()
    print(f"  GPU迁移完成: GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB")
    
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad(): out = mdl(**inp, output_hidden_states=True)
    print(f"  推理OK: n_layers={len(out.hidden_states)}")
    
    del mdl, out; torch.cuda.empty_cache(); gc.collect()
    print("  SUCCESS")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    try: del mdl
    except: pass
    torch.cuda.empty_cache(); gc.collect()

print("\n[4] 测试: device_map='auto'")
try:
    torch.cuda.empty_cache(); gc.collect()
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    print(f"  加载完成: {time.time()-t0:.1f}s, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB")
    print("  SUCCESS - device_map=auto可用")
    del mdl; torch.cuda.empty_cache(); gc.collect()
except Exception as e:
    print(f"  FAIL(符合04-09记录): {e}")
    try: del mdl
    except: pass
    torch.cuda.empty_cache(); gc.collect()

print("\n完成!")
