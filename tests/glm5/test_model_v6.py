"""测试DS7B - 简化: 只加载+1次forward, 不generate"""
import sys, time, gc, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

from pathlib import Path as _Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

target = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
p = MODEL_MAP[target]
RESULT = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_result.txt"

def log_msg(msg):
    with open(RESULT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

with open(RESULT, "w", encoding="utf-8") as f:
    f.write(f"=== {target} test ({time.strftime('%H:%M:%S')}) ===\n")

log_msg(f"path: {p}")
log_msg(f"cuda: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
log_msg(f"GPU total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

t0 = time.time()

try:
    log_msg("[1/3] tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log_msg(f"  vocab={tok.vocab_size}")
    
    log_msg("[2/3] model (attn_implementation=eager)...")
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager"
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log_msg(f"  layers={n_layers}, d_model={d_model}, t={time.time()-t0:.1f}s")
    log_msg(f"  GPU VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    log_msg("[3/3] forward pass (no generate)...")
    inp = tok("The cat sat on the mat", return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    h0 = out.hidden_states[0][0,-1].float()
    hf = out.hidden_states[-1][0,-1].float()
    cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
    log_msg(f"  n_hidden={len(out.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}")
    
    # 简单generate测试
    log_msg("[bonus] short generate...")
    inp2 = tok("1+1=", return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out2 = mdl.generate(**inp2, max_new_tokens=5, do_sample=False)
    log_msg(f"  result: {tok.decode(out2[0], skip_special_tokens=True)}")
    
    del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
    log_msg(f"DONE {target} total={time.time()-t0:.1f}s")

except Exception as e:
    log_msg(f"ERROR: {e}")
    import traceback
    log_msg(traceback.format_exc())
    try:
        del mdl; del tok
    except: pass
    gc.collect(); torch.cuda.empty_cache()
