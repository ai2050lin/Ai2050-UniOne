"""极简模型测试 - 完全不依赖print输出"""
import sys, time, gc, os
import torch

MODELS = {
    "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
    "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "gemma4": r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
}

RESULT_FILE = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_result.txt"

# 抑制warnings
import warnings
warnings.filterwarnings("ignore")

def test_one(name, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"path: {path}\n")
        f.write(f"start: {time.strftime('%H:%M:%S')}\n")
        f.flush()
        
        # tokenizer
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            f.write(f"[OK] tokenizer vocab={tok.vocab_size}\n"); f.flush()
        except Exception as e:
            f.write(f"[FAIL] tokenizer: {e}\n"); f.flush()
            return False
        
        # model
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            mdl.eval()
            f.write(f"[OK] model layers={len(mdl.model.layers)} d={mdl.config.hidden_size} t={time.time()-t0:.1f}s\n"); f.flush()
        except Exception as e:
            f.write(f"[FAIL] model: {e}\n"); f.flush()
            return False
        
        # gpu
        if torch.cuda.is_available():
            f.write(f"[OK] GPU VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB\n"); f.flush()
        
        # generate
        try:
            inp = tok("The capital of France is", return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=15, do_sample=False)
            result = tok.decode(out[0], skip_special_tokens=True)
            f.write(f"[OK] generate: {result}\n"); f.flush()
        except Exception as e:
            f.write(f"[FAIL] generate: {e}\n"); f.flush()
            del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
            return False
        
        # hidden states
        try:
            inp2 = tok("cat sat on mat", return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out2 = mdl(**inp2, output_hidden_states=True)
            h0 = out2.hidden_states[0][0,-1].float()
            hf = out2.hidden_states[-1][0,-1].float()
            cos = torch.nn.functional.cosine_similarity(h0.unsqueeze(0), hf.unsqueeze(0)).item()
            f.write(f"[OK] hidden: n={len(out2.hidden_states)} ||h0||={h0.norm():.1f} ||hL||={hf.norm():.1f} cos={cos:.4f}\n"); f.flush()
        except Exception as e:
            f.write(f"[WARN] hidden: {e}\n"); f.flush()
        
        # cleanup
        del mdl; del tok; gc.collect(); torch.cuda.empty_cache()
        f.write(f"[DONE] {name} total={time.time()-t0:.1f}s\n"); f.flush()
    
    return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if target not in MODELS:
        with open(RESULT_FILE, "w") as f:
            f.write(f"Unknown: {target}\n")
        sys.exit(1)
    
    ok = test_one(target, MODELS[target])
    # 用exit code传递结果
    sys.exit(0 if ok else 1)
