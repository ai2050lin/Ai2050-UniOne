# -*- coding: utf-8 -*-
"""DeepSeek7B 最小化加载测试 v2 - 写文件代替print"""
import torch, sys, time, gc

log_file = r"D:\develop\TransformerLens-main\tests\glm5_temp\ds7b_minimal_log.txt"

def log(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

# 清空日志
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=== DeepSeek7B Minimal Test v2 ===\n")

log("[1] Starting...")
log(f"  Python: {sys.version}")
log(f"  CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    log("[2] Loading tokenizer...")
    from transformers import AutoTokenizer
    model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"  Tokenizer OK! vocab={tok.vocab_size}, time={time.time()-t0:.1f}s")
except Exception as e:
    log(f"  Tokenizer FAILED: {e}")
    sys.exit(1)

try:
    log("[3] Loading model to CPU (low_cpu_mem_usage=True)...")
    from transformers import AutoModelForCausalLM
    torch.cuda.empty_cache()
    gc.collect()
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    log(f"  CPU load OK! time={time.time()-t0:.1f}s")
except Exception as e:
    log(f"  CPU load FAILED: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

try:
    log("[4] Moving to GPU...")
    t1 = time.time()
    mdl = mdl.to("cuda")
    mdl.eval()
    log(f"  GPU move OK! time={time.time()-t1:.1f}s")
    log(f"  GPU mem: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
except Exception as e:
    log(f"  GPU move FAILED: {e}")
    sys.exit(1)

try:
    log("[5] Inference test...")
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    log(f"  Inference OK! n_layers={len(out.hidden_states)}")
except Exception as e:
    log(f"  Inference FAILED: {e}")
    sys.exit(1)

log("[6] Cleanup")
del mdl, out
torch.cuda.empty_cache()
gc.collect()
log("ALL DONE!")
