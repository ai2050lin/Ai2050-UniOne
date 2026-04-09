"""测试DeepSeek7B - 手动加载safetensors权重, 避免from_pretrained崩溃"""
import sys, time, gc, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

from pathlib import Path as _Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

target = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
p = str(MODEL_MAP[target])
RESULT = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_result.txt"

def log_msg(msg):
    with open(RESULT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

with open(RESULT, "w", encoding="utf-8") as f:
    f.write(f"=== {target} manual load ({time.strftime('%H:%M:%S')}) ===\n")

log_msg(f"path: {p}")
t0 = time.time()

try:
    # Step 1: tokenizer
    log_msg("[1/5] tokenizer...")
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log_msg(f"  vocab={tok.vocab_size}")

    # Step 2: config
    log_msg("[2/5] config...")
    cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
    log_msg(f"  model_type={cfg.model_type}, d={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

    # Step 3: 手动加载safetensors
    log_msg("[3/5] loading safetensors manually...")
    import glob
    st_files = sorted(glob.glob(os.path.join(p, "*.safetensors")))
    log_msg(f"  found {len(st_files)} safetensors files")
    
    all_state_dict = {}
    for sf in st_files:
        log_msg(f"  loading {os.path.basename(sf)}...")
        sd = load_file(sf)
        log_msg(f"    {len(sd)} tensors")
        all_state_dict.update(sd)
    log_msg(f"  total {len(all_state_dict)} tensors, t={time.time()-t0:.1f}s")

    # Step 4: 用config创建空模型, 加载权重
    log_msg("[4/5] creating model from config + state_dict...")
    with torch.device("meta"):
        mdl = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    # to_empty先分配GPU内存, 然后load_state_dict填充数据
    mdl = mdl.to_empty(device="cuda")
    mdl.load_state_dict(all_state_dict, assign=True)
    del all_state_dict
    mdl = mdl.to(dtype=torch.bfloat16)
    mdl.eval()
    log_msg(f"  model on GPU: layers={len(mdl.model.layers)}")
    log_msg(f"  GPU VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB, t={time.time()-t0:.1f}s")

    # Step 5: 测试
    log_msg("[5/5] forward pass...")
    inp = tok("The cat sat on the mat", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inp, output_hidden_states=True)
    h0 = out.hidden_states[0][0,-1].float()
    hf = out.hidden_states[-1][0,-1].float()
    cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
    log_msg(f"  n={len(out.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}")

    # generate
    log_msg("[bonus] generate...")
    inp2 = tok("1+1=", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out2 = mdl.generate(**inp2, max_new_tokens=5, do_sample=False)
    log_msg(f"  result: {tok.decode(out2[0], skip_special_tokens=True)}")

    del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
    log_msg(f"DONE {target} total={time.time()-t0:.1f}s")

except Exception as e:
    log_msg(f"ERROR: {e}")
    import traceback
    log_msg(traceback.format_exc())
    gc.collect(); torch.cuda.empty_cache()
