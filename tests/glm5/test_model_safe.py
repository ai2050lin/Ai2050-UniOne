"""测试DeepSeek7B - 不用meta device, 直接CPU构建再移GPU"""
import sys, time, gc, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    f.write(f"=== {target} CPU-first ({time.strftime('%H:%M:%S')}) ===\n")

log_msg(f"cuda: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
log_msg(f"torch={torch.__version__}, cuda={torch.version.cuda}")
t0 = time.time()

try:
    # 先清理GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 1: tokenizer
    log_msg("[1/5] tokenizer...")
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log_msg(f"  vocab={tok.vocab_size}")

    # Step 2: 手动加载safetensors到CPU
    log_msg("[2/5] loading safetensors to CPU...")
    import glob
    st_files = sorted(glob.glob(os.path.join(p, "*.safetensors")))
    all_state_dict = {}
    for sf in st_files:
        sd = load_file(sf)
        # 确保bfloat16
        for k, v in sd.items():
            if v.dtype != torch.bfloat16:
                sd[k] = v.to(torch.bfloat16)
        all_state_dict.update(sd)
    log_msg(f"  {len(all_state_dict)} tensors (bfloat16), t={time.time()-t0:.1f}s")

    # Step 3: 直接CPU创建模型+加载权重(不用meta device)
    log_msg("[3/5] creating model on CPU + load_state_dict...")
    cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    mdl.load_state_dict(all_state_dict, assign=True)
    del all_state_dict
    mdl.eval()
    log_msg(f"  CPU model: layers={len(mdl.model.layers)}, t={time.time()-t0:.1f}s")

    # Step 4: 移到GPU
    log_msg("[4/5] moving to GPU...")
    mdl = mdl.to("cuda")
    log_msg(f"  GPU VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB, t={time.time()-t0:.1f}s")

    # Step 5: 测试
    log_msg("[5/5] forward pass...")
    inp = tok("cat sat on mat", return_tensors="pt").to("cuda")
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
