"""
P64: 第一性原理推导 + 9维语义空间提取
=========================================
核心问题：
1. 为什么effective rank=9？——理论解释
2. 9维有效空间中，每维对应什么语义？
3. 9个主成分与unembed矩阵的关系

实验设计：
- Exp1: 对比 h_final PCA 和 unembed PCA 的主成分对齐程度
- Exp2: 用大量文本(30条)重新计算effective rank，验证稳定性
- Exp3: 提取9个主成分，逐维解释其对logit的贡献
- Exp4: 测试"softmax温度"与rank的关系
- Exp5: 随机投影下是否保持rank=9（排除是高维效应）
- Exp6: 逐文本对分析9维坐标，找语义模式

四模型串行: Qwen3 → DS7B → GLM4 → Gemma4
"""

import torch
import numpy as np
import os, json, sys
from pathlib import Path as _Path
from datetime import datetime

class Logger:
    def __init__(self, log_dir, name):
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(safe)
        self.f.write(safe + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

MODEL_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

TEXTS_30 = [
    "The cat sat on the mat.",
    "A beautiful sunset over the ocean.",
    "Mathematical proof by induction.",
    "The stock market crashed today.",
    "for i in range(10): print(i)",
    "def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)",
    "春天来了，花开满园。",
    "机器学习是人工智能的子领域。",
    "Einstein's theory of relativity changed physics.",
    "The quantum computer solved the problem in seconds.",
    "Climate change is a global challenge.",
    "Python is the most popular programming language.",
    "The neural network has 100 layers.",
    "Attention is all you need for transformers.",
    "Gradient descent minimizes the loss function.",
    "今天天气很好，适合出去散步。",
    "深度学习模型需要大量训练数据。",
    "The universe is expanding at an accelerating rate.",
    "DNA carries genetic information in all living organisms.",
    "The election results surprised everyone last night.",
    "class NeuralNetwork: def __init__(self, layers): self.layers=layers",
    "import numpy as np; x = np.random.randn(100, 256)",
    "The restaurant serves excellent Italian cuisine.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Renaissance was a period of great cultural achievement.",
    "区块链技术正在改变金融行业。",
    "The Mars rover collected soil samples successfully.",
    "Machine translation accuracy improved significantly.",
    "The concert featured a symphony orchestra performance.",
    "Evolution by natural selection explains biodiversity.",
    "Recursive functions call themselves until a base case is reached.",
]

def load_model(model_name, log):
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def get_h_final(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens.input_ids, output_hidden_states=True)
    h = outputs.hidden_states[-1][0, -1].float()
    return h

def get_all_h_final(model, tokenizer, texts):
    return torch.stack([get_h_final(model, tokenizer, t) for t in texts])

def get_unembed(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.data.float()  # (vocab_size, d_model)
    elif hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings().weight.data.float()
    else:
        raise ValueError("Cannot find lm_head")

def compute_effective_rank(matrix, threshold=0.01):
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    eff_rank = (S > threshold * S[0]).sum().item()
    return eff_rank, S

def decode_tokens(tokenizer, ids, logit_contrib=None):
    """Safely decode token IDs to strings"""
    results = []
    for tid in ids:
        try:
            s = tokenizer.decode([tid.item()]).replace("\n", "\\n")[:20]
        except Exception:
            s = f"[{tid.item()}]"
        if logit_contrib is not None:
            s += f"({logit_contrib[tid]:.1f})"
        results.append(s)
    return results

def exp1_h_vs_unembed(model, tokenizer, h_all, log):
    """Compare h_final PCA with unembed PCA alignment"""
    log(f"\n  Exp1: h_final PCA vs unembed PCA alignment")
    
    W = get_unembed(model)  # (vocab_size, d_model)
    n_s = min(5000, W.shape[0])
    W_s = W[torch.randperm(W.shape[0])[:n_s]]
    _, S_u, Vt_u = torch.linalg.svd(W_s - W_s.mean(0), full_matrices=False)
    
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S_h, Vt_h = torch.linalg.svd(h_c, full_matrices=False)
    
    alignments = []
    K = min(10, len(S_h), len(S_u))
    for k in range(K):
        cos = torch.nn.functional.cosine_similarity(Vt_h[k:k+1], Vt_u[k:k+1]).item()
        alignments.append(cos)
    
    log(f"    h_final top-10 SV: {[f'{s:.2f}' for s in S_h[:10].tolist()]}")
    log(f"    unembed top-10 SV: {[f'{s:.1f}' for s in S_u[:10].tolist()]}")
    log(f"    Top-{K} alignment (cos): {[f'{c:.4f}' for c in alignments]}")
    log(f"    Average alignment: {np.mean(alignments):.4f}")
    return {"avg_align": np.mean(alignments), "alignments": alignments, "S_h": S_h[:10].tolist()}

def exp2_rank_stability(model, tokenizer, texts_30, log):
    """Test rank stability across different text set sizes"""
    log(f"\n  Exp2: Rank stability (sample sizes)")
    h_all = get_all_h_final(model, tokenizer, texts_30)
    
    ranks = {}
    for n in [5, 10, 15, 20, 25, 30]:
        idx = torch.randperm(30)[:n]
        r, _ = compute_effective_rank(h_all[idx])
        ranks[n] = r
    
    full_rank, S = compute_effective_rank(h_all)
    log(f"    Rank vs N: {ranks}")
    log(f"    Full (30) rank: {full_rank}, top SV: {[f'{s:.2f}' for s in S[:10].tolist()]}")
    return {"ranks": ranks, "full_rank": full_rank}

def exp3_semantic_components(model, tokenizer, h_all, texts, log):
    """Extract 9 PCs, find top tokens each PC activates"""
    log(f"\n  Exp3: 9 principal components semantic analysis")
    
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S_h, Vt_h = torch.linalg.svd(h_c, full_matrices=False)
    
    W = get_unembed(model)  # (vocab_size, d_model)
    K = 9
    
    total_var = (S_h ** 2).sum().item()
    
    for k in range(K):
        pc = Vt_h[k]  # (d_model,)
        logit_c = (W @ pc)  # (vocab_size,)
        top_vals, top_ids = torch.topk(logit_c, 8)
        toks = decode_tokens(tokenizer, top_ids, logit_c)
        var_pct = S_h[k].item()**2 / total_var * 100
        log(f"    PC{k}: SV={S_h[k]:.2f}, var={var_pct:.1f}%, top: {', '.join(toks)}")
    
    # 9-PC logit reconstruction
    log(f"    Logit reconstruction (K=9 vs full):")
    cos_list, top1_list = [], []
    for i in range(min(10, len(h_all))):
        h = h_all[i:i+1]
        logits_full = (h @ W.T).squeeze()
        coords = (h - h_mean) @ Vt_h[:K].T
        h_recon = h_mean + coords @ Vt_h[:K]
        logits_recon = (h_recon @ W.T).squeeze()
        cos = torch.nn.functional.cosine_similarity(logits_full.unsqueeze(0), logits_recon.unsqueeze(0)).item()
        top1 = (logits_full.argmax() == logits_recon.argmax()).item()
        cos_list.append(cos)
        top1_list.append(top1)
    
    log(f"    avg cos={np.mean(cos_list):.4f}, avg top1={np.mean(top1_list):.2f}")
    return {"recon_cos": np.mean(cos_list), "recon_top1": np.mean(top1_list)}

def exp4_temperature_rank(model, tokenizer, texts, log):
    """Test if rank is scale-invariant"""
    log(f"\n  Exp4: Scale invariance of effective rank")
    
    h_all = get_all_h_final(model, tokenizer, texts[:15])
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ranks = {}
    for s in scales:
        r, _ = compute_effective_rank(h_all * s)
        ranks[str(s)] = r
    
    all_same = len(set(ranks.values())) == 1
    log(f"    Rank vs scale: {ranks}")
    log(f"    Scale invariant: {'YES' if all_same else 'NO'}")
    return {"ranks": ranks, "invariant": all_same}

def exp5_random_projection(model, tokenizer, texts, log):
    """Random projection: does rank=9 survive dimensionality reduction?"""
    log(f"\n  Exp5: Random projection rank test")
    
    h_all = get_all_h_final(model, tokenizer, texts[:15])
    d_model = h_all.shape[1]
    full_rank, _ = compute_effective_rank(h_all)
    
    results = []
    for td in [32, 64, 128, 256, 512, 1024]:
        if td >= d_model:
            continue
        torch.manual_seed(42)
        P = torch.randn(d_model, td) / np.sqrt(d_model)
        h_proj = h_all @ P
        r, _ = compute_effective_rank(h_proj)
        results.append((td, r))
        log(f"    Proj to {td}D: rank={r}")
    
    log(f"    Full {d_model}D: rank={full_rank}")
    return {"full_rank": full_rank, "proj": results}

def exp6_pairwise_analysis(model, tokenizer, h_all, texts, log):
    """Pairwise text diff projected onto 9 PCs"""
    log(f"\n  Exp6: Pairwise text differences on 9 PCs")
    
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S_h, Vt_h = torch.linalg.svd(h_c, full_matrices=False)
    K = 9
    
    # Text categories: 0=general, 2=math, 4=code, 6=chinese, 1=general, 5=code
    pairs = [(0,1,"gen-gen"), (0,2,"gen-math"), (0,4,"gen-code"), (0,6,"gen-cn"), (2,4,"math-code"), (4,5,"code-code"), (6,7,"cn-cn")]
    for i, j, label in pairs:
        if i >= len(texts) or j >= len(texts):
            continue
        diff = h_c[i] - h_c[j]
        coords = (diff @ Vt_h[:K].T).tolist()
        abs_c = [abs(c) for c in coords]
        dom = np.argmax(abs_c)
        log(f"    {label} [{texts[i][:25]}...] vs [{texts[j][:25]}...]:")
        log(f"      dom=PC{dom}({abs_c[dom]:.1f}), coords={[f'{c:.1f}' for c in coords]}")

def run_suite(model_name, texts, log):
    log(f"\n{'='*60}")
    log(f"Model: {model_name}")
    log(f"{'='*60}")
    
    model, tokenizer = load_model(model_name, log)
    
    log(f"  Computing h_final for {len(texts)} texts...")
    h_all = get_all_h_final(model, tokenizer, texts)
    log(f"  h_all shape: {h_all.shape}")
    
    results = {}
    
    try: results["exp1"] = exp1_h_vs_unembed(model, tokenizer, h_all, log)
    except Exception as e: log(f"  Exp1 FAILED: {e}")
    
    try: results["exp2"] = exp2_rank_stability(model, tokenizer, TEXTS_30, log)
    except Exception as e: log(f"  Exp2 FAILED: {e}")
    
    try: results["exp3"] = exp3_semantic_components(model, tokenizer, h_all, texts, log)
    except Exception as e: log(f"  Exp3 FAILED: {e}")
    
    try: results["exp4"] = exp4_temperature_rank(model, tokenizer, texts, log)
    except Exception as e: log(f"  Exp4 FAILED: {e}")
    
    try: results["exp5"] = exp5_random_projection(model, tokenizer, texts, log)
    except Exception as e: log(f"  Exp5 FAILED: {e}")
    
    try: results["exp6"] = exp6_pairwise_analysis(model, tokenizer, h_all, texts, log)
    except Exception as e: log(f"  Exp6 FAILED: {e}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage706_first_principles_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    global log
    log = Logger(run_dir, "results")
    log("P64: First Principles Derivation + 9D Semantic Space")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS_30)}")
    
    all_results = {}
    for mn in MODEL_ORDER:
        try:
            results = run_suite(mn, TEXTS_30, log)
            all_results[mn] = results
        except Exception as e:
            log(f"  FATAL {mn}: {e}")
    
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Conclusion
    log(f"\n{'='*60}")
    log("CROSS-MODEL CONCLUSION")
    log(f"{'='*60}")
    for mn, res in all_results.items():
        log(f"\n  {mn}:")
        if "exp2" in res:
            log(f"    Rank stability: {res['exp2']['ranks']}")
        if "exp1" in res:
            log(f"    h vs unembed align: {res['exp1']['avg_align']:.4f}")
        if "exp3" in res:
            log(f"    9-PC recon: cos={res['exp3']['recon_cos']:.4f}, top1={res['exp3']['recon_top1']:.2f}")
        if "exp4" in res:
            log(f"    Scale invariant: {res['exp4']['invariant']}")
        if "exp5" in res:
            log(f"    Random proj: {res['exp5']['proj']}")
    
    log(f"\nResults saved to {run_dir}")
    log.close()

if __name__ == "__main__":
    main()
