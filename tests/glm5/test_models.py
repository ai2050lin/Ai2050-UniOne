"""
逐个测试4个模型(Qwen3, DeepSeek7B, GLM4, Gemma4)能否正常加载和推理
每次只测一个, 避免GPU内存溢出
"""
import torch
import sys
import time
import gc
import os

MODEL_MAP = {
    "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "gemma4": r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
}

LOG_FILE = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_results.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
    # 安全打印
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

def test_model(name, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    log(f"\n{'='*60}")
    log(f"[TEST] {name}: {path}")
    log(f"{'='*60}")
    
    t0 = time.time()
    
    # 1. 加载tokenizer
    log(f"  [1/4] Loading tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        log(f"  tokenizer OK, vocab_size={tok.vocab_size}")
    except Exception as e:
        log(f"  tokenizer FAILED: {e}")
        return False
    
    # 2. 加载模型
    log(f"  [2/4] Loading model (bfloat16, device_map=auto)...")
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl.eval()
        t1 = time.time()
        log(f"  model OK, n_layers={len(mdl.model.layers)}, d_model={mdl.config.hidden_size}")
        log(f"  load time: {t1-t0:.1f}s")
    except Exception as e:
        log(f"  model FAILED: {e}")
        return False
    
    # 3. GPU内存
    log(f"  [3/4] GPU check...")
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        log(f"  GPU VRAM: {vram:.2f} GB used")
    
    # 4. 推理测试
    log(f"  [4/4] Inference test...")
    try:
        prompt = "The capital of France is"
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=15, do_sample=False)
        result = tok.decode(out[0], skip_special_tokens=True)
        log(f"  input:  '{prompt}'")
        log(f"  output: '{result}'")
    except Exception as e:
        log(f"  inference FAILED: {e}")
        # 释放内存
        del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
        return False
    
    # 额外: hidden states
    try:
        inputs2 = tok("The cat sat on the mat", return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out2 = mdl(**inputs2, output_hidden_states=True)
        h0 = out2.hidden_states[0][0, -1].float()
        hf = out2.hidden_states[-1][0, -1].float()
        cos_sim = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
        log(f"  hidden_states: {len(out2.hidden_states)} layers")
        log(f"  ||h_0||={h0.norm().item():.2f}, ||h_L||={hf.norm().item():.2f}, cos(h0,hL)={cos_sim:.4f}")
    except Exception as e:
        log(f"  hidden_states FAILED: {e}")
    
    # 释放内存
    log(f"  Releasing model...")
    del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
    
    t2 = time.time()
    log(f"  {name} test PASSED! total time: {t2-t0:.1f}s")
    return True

if __name__ == "__main__":
    # 清空日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Model Test - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 从命令行选择要测的模型, 或测全部
    if len(sys.argv) > 1:
        targets = [sys.argv[1]]
    else:
        targets = list(MODEL_MAP.keys())
    
    results = {}
    for name in targets:
        if name not in MODEL_MAP:
            log(f"[SKIP] Unknown model: {name}")
            continue
        ok = test_model(name, MODEL_MAP[name])
        results[name] = "PASS" if ok else "FAIL"
        # 等GPU完全释放
        time.sleep(5)
        gc.collect()
        torch.cuda.empty_cache()
    
    log(f"\n{'='*60}")
    log(f"SUMMARY:")
    for name, status in results.items():
        log(f"  {name}: {status}")
    log(f"{'='*60}")
