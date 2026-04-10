# -*- coding: utf-8 -*-
import torch, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

MP = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("[1] CPU load...")
from transformers import AutoModelForCausalLM
mdl = AutoModelForCausalLM.from_pretrained(
    MP, dtype=torch.bfloat16, trust_remote_code=True,
    local_files_only=True, low_cpu_mem_usage=True,
    attn_implementation="eager", device_map="cpu"
)
print("  OK")

print("[2] GPU move...")
mdl = mdl.to("cuda")
mdl.eval()
print(f"  OK, GPU mem={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

print("[3] Tokenizer...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True)
print("  OK")

print("[4] Inference...")
try:
    inp = tok("Hello", return_tensors="pt").to("cuda")
    print(f"  Input shape: {inp['input_ids'].shape}")
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    print(f"  OK! n_layers={len(out.hidden_states)}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("[5] Cleanup")
del mdl
torch.cuda.empty_cache()
print("Done")
