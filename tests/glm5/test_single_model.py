"""极简模型测试 - 直接运行不重定向"""
import sys, time, gc, os
import torch

MODELS = {
    "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "gemma4": r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
}

RESULT_FILE = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_result.txt"

def test_one(name, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    lines = []
    def p(s):
        lines.append(s)
        try: print(s, flush=True)
        except: pass
    
    p(f"=== {name} ===")
    p(f"path: {path}")
    
    # tokenizer
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        p(f"[OK] tokenizer vocab={tok.vocab_size}")
    except Exception as e:
        p(f"[FAIL] tokenizer: {e}")
        return lines, False
    
    # model
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        mdl.eval()
        p(f"[OK] model layers={len(mdl.model.layers)} d={mdl.config.hidden_size} t={time.time()-t0:.1f}s")
    except Exception as e:
        p(f"[FAIL] model: {e}")
        return lines, False
    
    # gpu
    if torch.cuda.is_available():
        p(f"[OK] GPU VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # generate
    try:
        inp = tok("The capital of France is", return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=15, do_sample=False)
        p(f"[OK] output: {tok.decode(out[0], skip_special_tokens=True)}")
    except Exception as e:
        p(f"[FAIL] generate: {e}")
        del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
        return lines, False
    
    # hidden states
    try:
        inp2 = tok("cat sat on mat", return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out2 = mdl(**inp2, output_hidden_states=True)
        h0 = out2.hidden_states[0][0,-1].float()
        hf = out2.hidden_states[-1][0,-1].float()
        cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
        p(f"[OK] hidden: n={len(out2.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}")
    except Exception as e:
        p(f"[WARN] hidden: {e}")
    
    # cleanup
    del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
    p(f"[DONE] {name}")
    return lines, True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if target not in MODELS:
        print(f"Unknown: {target}. Available: {list(MODELS.keys())}")
        sys.exit(1)
    
    lines, ok = test_one(target, MODELS[target])
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nResult saved to {RESULT_FILE}")
    print(f"STATUS: {'PASS' if ok else 'FAIL'}")
