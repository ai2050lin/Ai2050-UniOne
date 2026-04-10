import torch, sys, time, traceback
MP = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
LF = r"D:\develop\TransformerLens-main\tests\glm5_temp\ds7b_auto_log.txt"
def L(m):
    with open(LF, 'a') as f: f.write(f"{time.strftime('%H:%M:%S')} {m}\n"); f.flush()
open(LF,'w').write("=== DS7B device_map=auto test ===\n")
L(f"CUDA: {torch.cuda.is_available()}")
L(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1e9:.1f} GB")

try:
    L("[1] device_map=auto...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    dev = next(mdl.parameters()).device
    L(f"  OK! dev={dev}, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB, {time.time()-t0:.0f}s")
except Exception as e:
    L(f"  FAIL: {e}")
    L(traceback.format_exc())
    sys.exit(1)

try:
    L("[2] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    L("  OK")
except Exception as e:
    L(f"  FAIL: {e}")
    sys.exit(1)

try:
    L("[3] Inference...")
    inp = tok("Hello", return_tensors="pt").to(dev)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    L(f"  OK! n_layers={len(out.hidden_states)}")
except Exception as e:
    L(f"  FAIL: {e}")
    L(traceback.format_exc())
    sys.exit(1)

try:
    L("[4] Quick data (5 words)...")
    words = ["apple","banana","cat","red","sweet"]
    for w in words:
        inp = tok(f"The {w} is", return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        L(f"  {w}: OK")
        del out, inp
except Exception as e:
    L(f"  FAIL: {e}")
    sys.exit(1)

L("[5] Cleanup")
del mdl, tok
import gc; gc.collect(); torch.cuda.empty_cache()
L("ALL DONE!")
