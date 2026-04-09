"""极简模型测试 v3 - 所有输出写文件"""
import sys, time, gc, os

# 重定向所有stdout/stderr到文件
LOG = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_full.log"
sys.stdout = open(LOG, "w", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

import torch
import warnings
warnings.filterwarnings("ignore")

MODELS = {
    "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "gemma4": r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
}

target = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
path = MODELS.get(target, "")
if not path:
    print(f"Unknown: {target}")
    sys.exit(1)

print(f"=== {target} ===", flush=True)
print(f"path: {path}", flush=True)
print(f"start: {time.strftime('%H:%M:%S')}", flush=True)
print(f"cuda: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()

print(f"[1/4] tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"  vocab={tok.vocab_size}", flush=True)

print(f"[2/4] model (bfloat16, auto)...", flush=True)
mdl = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
mdl.eval()
print(f"  layers={len(mdl.model.layers)} d={mdl.config.hidden_size} t={time.time()-t0:.1f}s", flush=True)

print(f"[3/4] GPU...", flush=True)
print(f"  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print(f"[4/4] generate...", flush=True)
inp = tok("The capital of France is", return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**inp, max_new_tokens=15, do_sample=False)
print(f"  result: {tok.decode(out[0], skip_special_tokens=True)}", flush=True)

# hidden states
print(f"[extra] hidden states...", flush=True)
inp2 = tok("cat sat on mat", return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out2 = mdl(**inp2, output_hidden_states=True)
h0 = out2.hidden_states[0][0,-1].float()
hf = out2.hidden_states[-1][0,-1].float()
cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
print(f"  n={len(out2.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}", flush=True)

# cleanup
del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
print(f"DONE {target} total={time.time()-t0:.1f}s", flush=True)
