import torch, sys, time, traceback
MP = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
LF = r"D:\develop\TransformerLens-main\tests\glm5_temp\ds7b_test_log.txt"

def L(m):
    with open(LF, 'a') as f: f.write(f"{time.strftime('%H:%M:%S')} {m}\n"); f.flush()

open(LF,'w').write("=== DS7B Step Test ===\n")

try:
    L("[1] CPU load...")
    from transformers import AutoModelForCausalLM
    mdl = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu")
    L("  CPU OK")
except Exception as e:
    L(f"  CPU FAIL: {e}\n{traceback.format_exc()}")
    sys.exit(1)

try:
    L("[2] GPU move...")
    mdl = mdl.to("cuda"); mdl.eval()
    L(f"  GPU OK mem={torch.cuda.memory_allocated(0)/1e9:.2f}GB")
except Exception as e:
    L(f"  GPU FAIL: {e}\n{traceback.format_exc()}")
    sys.exit(1)

try:
    L("[3] Tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    L("  TOK OK")
except Exception as e:
    L(f"  TOK FAIL: {e}")
    sys.exit(1)

try:
    L("[4] Inference...")
    inp = tok("Hello", return_tensors="pt").to("cuda")
    L(f"  input_ids shape={inp['input_ids'].shape}")
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    L(f"  INFERENCE OK n_layers={len(out.hidden_states)}")
except Exception as e:
    L(f"  INFERENCE FAIL: {e}\n{traceback.format_exc()}")
    sys.exit(1)

L("[5] Cleanup")
del mdl, out; torch.cuda.empty_cache()
L("ALL DONE!")
