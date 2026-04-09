"""直接测试DS7B - 不重定向, 逐步执行"""
import os
# 在import torch之前就设置环境变量抑制所有warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

p = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("Step 1: Import transformers", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Step 2: Load tokenizer", flush=True)
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"  vocab={tok.vocab_size}", flush=True)

print("Step 3: Load model with device_map=auto", flush=True)
import time
t0 = time.time()
try:
    mdl = AutoModelForCausalLM.from_pretrained(
        p,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mdl.eval()
    print(f"  OK layers={len(mdl.model.layers)} d={mdl.config.hidden_size} t={time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    
    # 尝试CPU加载
    print("Step 3b: Retry on CPU only", flush=True)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            p,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        mdl.eval()
        print(f"  CPU OK layers={len(mdl.model.layers)}", flush=True)
    except Exception as e2:
        print(f"  CPU also FAILED: {e2}", flush=True)
        traceback.print_exc()
        raise

print("Step 4: GPU info", flush=True)
if torch.cuda.is_available():
    print(f"  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print("Step 5: Generate", flush=True)
inp = tok("The capital of France is", return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**inp, max_new_tokens=15, do_sample=False)
result = tok.decode(out[0], skip_special_tokens=True)
print(f"  result: {result}", flush=True)

print("Step 6: Hidden states", flush=True)
inp2 = tok("cat sat on mat", return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out2 = mdl(**inp2, output_hidden_states=True)
h0 = out2.hidden_states[0][0,-1].float()
hf = out2.hidden_states[-1][0,-1].float()
cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
print(f"  n={len(out2.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}", flush=True)

del mdl; del tok; import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE", flush=True)
