"""
P65-P67: Phase IV — 大样本有效维度 + 语义方向因果验证 + 跨模型PC对齐
====================================================================
P65: 用100条文本重新计算effective rank，确定真正的信息维度
P66: 因果验证——干预PC0/PC1/PC2坐标，观察logit变化
P67: 跨模型语义方向对齐——不同模型的PC0是否编码相同语义？

四模型串行: Qwen3 → DS7B → GLM4 → Gemma4
"""

import torch
import numpy as np
import os, json
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

# 100 diverse texts with category labels
TEXTS_100 = [
    # Category: general_english (20)
    ("The cat sat on the mat.", "gen_en"), ("A beautiful sunset over the ocean.", "gen_en"),
    ("The stock market crashed today.", "gen_en"), ("Climate change is a global challenge.", "gen_en"),
    ("The restaurant serves excellent Italian cuisine.", "gen_en"), ("The Renaissance was a period of great cultural achievement.", "gen_en"),
    ("The election results surprised everyone last night.", "gen_en"), ("The concert featured a symphony orchestra performance.", "gen_en"),
    ("She walked through the garden with a smile.", "gen_en"), ("The ancient castle stood on top of the hill.", "gen_en"),
    ("Breakfast is the most important meal of the day.", "gen_en"), ("The children played happily in the park.", "gen_en"),
    ("A sudden storm interrupted the picnic.", "gen_en"), ("The library has thousands of rare books.", "gen_en"),
    ("He traveled across three continents last year.", "gen_en"), ("The museum exhibit attracted record visitors.", "gen_en"),
    ("The teacher explained the concept clearly.", "gen_en"), ("Spring flowers bloom in every color.", "gen_en"),
    ("The old man told stories by the fireplace.", "gen_en"), ("Music brings people together across cultures.", "gen_en"),
    # Category: math_science (20)
    ("Mathematical proof by induction.", "math"), ("Einstein's theory of relativity changed physics.", "math"),
    ("The quantum computer solved the problem in seconds.", "math"), ("DNA carries genetic information in all living organisms.", "math"),
    ("Photosynthesis converts sunlight into chemical energy.", "math"), ("Evolution by natural selection explains biodiversity.", "math"),
    ("The Pythagorean theorem states a squared plus b squared equals c squared.", "math"),
    ("The derivative of velocity is acceleration.", "math"), ("Newton's third law states every action has an equal opposite reaction.", "math"),
    ("Entropy always increases in isolated systems.", "math"), ("The speed of light is approximately 299,792 km per second.", "math"),
    ("Euler's identity connects five fundamental constants.", "math"), ("The periodic table organizes elements by atomic number.", "math"),
    ("Mitochondria are the powerhouse of the cell.", "math"), ("The Doppler effect explains why sirens change pitch.", "math"),
    ("Maxwell's equations unify electricity and magnetism.", "math"), ("Thermodynamics governs heat and energy transfer.", "math"),
    ("The double helix structure of DNA was discovered by Watson and Crick.", "math"),
    ("Heisenberg's uncertainty principle limits measurement precision.", "math"), ("The gravitational constant is approximately 6.674 times 10 to the negative 11th.", "math"),
    # Category: code (20)
    ("for i in range(10): print(i)", "code"), ("def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)", "code"),
    ("class NeuralNetwork: def __init__(self, layers): self.layers=layers", "code"),
    ("import numpy as np; x = np.random.randn(100, 256)", "code"),
    ("Recursive functions call themselves until a base case is reached.", "code"),
    ("Python is the most popular programming language.", "code"), ("The neural network has 100 layers.", "code"),
    ("Attention is all you need for transformers.", "code"), ("Gradient descent minimizes the loss function.", "code"),
    ("x = torch.tensor([1.0, 2.0, 3.0])", "code"), ("model.train() sets the model to training mode.", "code"),
    ("def forward(self, x): return self.linear(x)", "code"), ("batch_size = 32; learning_rate = 0.001", "code"),
    ("optimizer = Adam(model.parameters(), lr=1e-3)", "code"), ("loss = nn.CrossEntropyLoss()(pred, target)", "code"),
    ("if torch.cuda.is_available(): model = model.cuda()", "code"),
    ("embedding = nn.Embedding(vocab_size, hidden_dim)", "code"),
    ("output = F.softmax(logits, dim=-1)", "code"),
    ("self.attention = nn.MultiheadAttention(d_model, n_heads)", "code"),
    ("hidden = self.rnn(embedded, hidden_state)", "code"),
    # Category: chinese (20)
    ("春天来了，花开满园。", "chinese"), ("机器学习是人工智能的子领域。", "chinese"),
    ("今天天气很好，适合出去散步。", "chinese"), ("深度学习模型需要大量训练数据。", "chinese"),
    ("区块链技术正在改变金融行业。", "chinese"), ("人工智能正在快速发展。", "chinese"),
    ("这本书非常有趣，值得一读。", "chinese"), ("中国经济持续增长。", "chinese"),
    ("量子计算是未来的重要技术。", "chinese"), ("自然语言处理有许多应用场景。", "chinese"),
    ("他每天早上六点起床锻炼身体。", "chinese"), ("这个项目的预算已经超支了。", "chinese"),
    ("研究表明，运动有助于健康。", "chinese"), ("数据科学家需要掌握统计学和编程。", "chinese"),
    ("气候变化对农业产生了重大影响。", "chinese"), ("机器人技术正在迅速发展。", "chinese"),
    ("这部电影讲述了二战期间的故事。", "chinese"), ("太阳系有八大行星。", "chinese"),
    ("互联网改变了人们的生活方式。", "chinese"), ("学生应该培养独立思考的能力。", "chinese"),
    # Category: mixed_reasoning (20)
    ("If all humans are mortal and Socrates is human, then Socrates is mortal.", "reasoning"),
    ("The probability of rain given cloudy skies is approximately 70 percent.", "reasoning"),
    ("To solve this equation, first isolate the variable on one side.", "reasoning"),
    ("The hypothesis is testable and falsifiable under controlled conditions.", "reasoning"),
    ("Therefore, we can conclude that the experiment supports the theory.", "reasoning"),
    ("The correlation does not imply causation in this observational study.", "reasoning"),
    ("Given the premises, the conclusion follows logically by deduction.", "reasoning"),
    ("The null hypothesis cannot be rejected at the 5 percent significance level.", "reasoning"),
    ("This approach reduces computational complexity from O(n squared) to O(n log n).", "reasoning"),
    ("We need to control for confounding variables in the regression model.", "reasoning"),
    ("The confidence interval is 95 percent with a margin of error of plus or minus 3 percent.", "reasoning"),
    ("Assuming the model is correctly specified, the estimator is unbiased.", "reasoning"),
    ("The algorithm converges in polynomial time under these constraints.", "reasoning"),
    ("By induction, the property holds for all natural numbers.", "reasoning"),
    ("The posterior distribution updates our prior belief given the evidence.", "reasoning"),
    ("This function is monotonically increasing on the given interval.", "reasoning"),
    ("The optimal solution lies at the boundary of the feasible region.", "reasoning"),
    ("We can approximate the integral using the trapezoidal rule.", "reasoning"),
    ("The expected value of the random variable is the sum of all possible outcomes weighted by their probabilities.", "reasoning"),
    ("This recurrence relation has a closed-form solution of O(n squared).", "reasoning"),
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
    return outputs.hidden_states[-1][0, -1].float()

def get_all_h(model, tokenizer, texts):
    return torch.stack([get_h_final(model, tokenizer, t) for t, _ in texts])

def get_unembed(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.data.float()
    elif hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings().weight.data.float()
    else:
        raise ValueError("Cannot find lm_head")

def compute_effective_rank(matrix, threshold=0.01):
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    return (S > threshold * S[0]).sum().item(), S

# ===== P65: Large sample effective rank =====
def p65_large_sample_rank(h_all, categories, log):
    """P65: Compute effective rank with 100 texts, also by category"""
    log(f"\n  P65: Large sample effective rank analysis")
    
    N = h_all.shape[0]
    full_rank, S = compute_effective_rank(h_all)
    log(f"    100 texts full rank: {full_rank}")
    log(f"    Top-20 SV: {[f'{s:.1f}' for s in S[:20].tolist()]}")
    
    # Per-category rank
    cat_names = set(categories)
    cat_ranks = {}
    for cat in sorted(cat_names):
        idx = [i for i, c in enumerate(categories) if c == cat]
        h_cat = h_all[idx]
        r, _ = compute_effective_rank(h_cat)
        cat_ranks[cat] = (r, len(idx))
    
    log(f"    Per-category ranks:")
    for cat, (r, n) in sorted(cat_ranks.items()):
        log(f"      {cat}({n} texts): rank={r}")
    
    # Rank growth curve: sample 10,20,30,...,100
    rank_curve = {}
    for n in [10, 20, 30, 50, 75, 100]:
        if n > N:
            continue
        torch.manual_seed(42)
        idx = torch.randperm(N)[:n]
        r, _ = compute_effective_rank(h_all[idx])
        rank_curve[n] = r
    
    log(f"    Rank growth: {rank_curve}")
    
    # Cumulative variance explained
    total = (S ** 2).sum().item()
    cumvar = torch.cumsum(S ** 2, 0) / total
    for k in [5, 10, 20, 30, 50]:
        if k <= len(cumvar):
            log(f"    Top-{k} var: {cumvar[k-1]:.4f}")
    
    return {"full_rank": full_rank, "cat_ranks": cat_ranks, "rank_curve": rank_curve}

# ===== P66: Causal intervention on PCs =====
def p66_causal_pc_intervention(model, tokenizer, h_all, texts, log):
    """P66: Causal test - shift PC coordinates and measure logit changes"""
    log(f"\n  P66: Causal PC intervention")
    
    W = get_unembed(model)  # (vocab_size, d_model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    
    # Pick 5 texts for detailed analysis
    test_idx = [0, 22, 42, 62, 82]  # one from each category
    K = min(5, Vt.shape[0])
    
    results = {}
    for ti in test_idx:
        if ti >= len(texts):
            continue
        h = h_all[ti:ti+1]
        logits_orig = (h @ W.T).squeeze()
        top3_orig = torch.topk(logits_orig, 3)
        
        log(f"\n    Text[{ti}] ({texts[ti][1]}): {texts[ti][0][:40]}...")
        log(f"      Original top3: ", end="")
        
        for k in range(K):
            pc = Vt[k]  # direction
            # Shift by +1 std along this PC
            shift = pc * S[k] * 0.5  # half std
            h_shifted = h + shift.unsqueeze(0)
            logits_shifted = (h_shifted @ W.T).squeeze()
            
            # Shift by -1 std
            h_shifted_neg = h - shift.unsqueeze(0)
            logits_neg = (h_shifted_neg @ W.T).squeeze()
            
            # KL divergence
            import torch.nn.functional as F
            kl_pos = F.kl_div(F.log_softmax(logits_shifted, -1), F.softmax(logits_orig, -1), reduction='sum').item()
            kl_neg = F.kl_div(F.log_softmax(logits_neg, -1), F.softmax(logits_orig, -1), reduction='sum').item()
            
            top1_pos = logits_shifted.argmax().item()
            top1_neg = logits_neg.argmax().item()
            top1_orig = logits_orig.argmax().item()
            
            changed_pos = top1_pos != top1_orig
            changed_neg = top1_neg != top1_orig
            
            log(f"\n      PC{k}(SV={S[k]:.1f}): KL+={kl_pos:.3f}, KL-={kl_neg:.3f}, "
                f"top1_change+={'YES' if changed_pos else 'no'}, top1_change-={'YES' if changed_neg else 'no'}")
    
    return {"tested": len(test_idx)}

# ===== P67: Cross-model PC alignment =====
def p67_cross_model_alignment(pca_data, log):
    """P67: Compare PC directions across models"""
    log(f"\n  P67: Cross-model PC alignment (summary)")
    
    models = list(pca_data.keys())
    if len(models) < 2:
        log("    Need at least 2 models")
        return {}
    
    # Compare each pair
    alignments = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            Vt1 = pca_data[m1]["Vt"]
            Vt2 = pca_data[m2]["Vt"]
            K = min(5, Vt1.shape[0], Vt2.shape[0])
            
            # Align by category means: project category means onto top PCs
            means1 = pca_data[m1]["cat_means"]
            means2 = pca_data[m2]["cat_means"]
            common_cats = set(means1.keys()) & set(means2.keys())
            
            if not common_cats:
                continue
            
            # For each pair of categories, compute the direction in PC space
            cat_pairs = [("gen_en", "chinese"), ("gen_en", "code"), ("gen_en", "math"), ("code", "math")]
            pair_scores = []
            for c1, c2 in cat_pairs:
                if c1 not in common_cats or c2 not in common_cats:
                    continue
                diff1 = means1[c1] - means1[c2]  # direction in h-space (model 1)
                diff2 = means2[c1] - means2[c2]  # direction in h-space (model 2)
                # Normalize and compute cos
                cos = torch.nn.functional.cosine_similarity(
                    diff1.unsqueeze(0), diff2.unsqueeze(0)
                ).item()
                pair_scores.append((f"{c1}-{c2}", cos))
            
            log(f"    {m1} vs {m2} category direction alignment:")
            for pair, cos in pair_scores:
                log(f"      {pair}: cos={cos:.4f}")
            avg = np.mean([c for _, c in pair_scores]) if pair_scores else 0
            log(f"      Average: {avg:.4f}")
            alignments[f"{m1}-{m2}"] = pair_scores
    
    return alignments

def run_suite(model_name, log):
    log(f"\n{'='*60}")
    log(f"Model: {model_name}")
    log(f"{'='*60}")
    
    model, tokenizer = load_model(model_name, log)
    
    texts = [(t, c) for t, c in TEXTS_100]
    categories = [c for _, c in texts]
    
    log(f"  Computing h_final for {len(texts)} texts...")
    h_all = get_all_h(model, tokenizer, texts)
    log(f"  h_all shape: {h_all.shape}")
    
    # PCA for later use
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    
    # Category means
    cat_means = {}
    for cat in set(categories):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)
    
    results = {}
    
    # P65
    try:
        results["p65"] = p65_large_sample_rank(h_all, categories, log)
    except Exception as e:
        log(f"  P65 FAILED: {e}")
    
    # P66
    try:
        results["p66"] = p66_causal_pc_intervention(model, tokenizer, h_all, texts, log)
    except Exception as e:
        log(f"  P66 FAILED: {e}")
    
    results["pca"] = {"Vt": Vt, "S": S, "cat_means": cat_means, "h_mean": h_mean}
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage707_large_sample_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    global log
    log = Logger(run_dir, "results")
    log("P65-P67: Phase IV - Large Sample Rank + Causal PC + Cross-Model Alignment")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS_100)} (5 categories x 20 each)")
    log(f"Categories: {sorted(set(c for _, c in TEXTS_100))}")
    
    all_results = {}
    pca_data = {}
    
    for mn in MODEL_ORDER:
        try:
            results = run_suite(mn, log)
            all_results[mn] = results
            if "pca" in results:
                pca_data[mn] = results["pca"]
        except Exception as e:
            log(f"  FATAL {mn}: {e}")
    
    # P67: Cross-model (after all models done)
    log(f"\n{'='*60}")
    log("P67: Cross-Model PC Alignment")
    log(f"{'='*60}")
    try:
        p67_result = p67_cross_model_alignment(pca_data, log)
    except Exception as e:
        log(f"  P67 FAILED: {e}")
    
    # Save JSON (without tensors)
    save_data = {}
    for mn, res in all_results.items():
        save_data[mn] = {}
        for k, v in res.items():
            if k == "pca":
                continue
            save_data[mn][k] = v
    
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    # Conclusion
    log(f"\n{'='*60}")
    log("CROSS-MODEL CONCLUSION")
    log(f"{'='*60}")
    for mn, res in all_results.items():
        log(f"\n  {mn}:")
        if "p65" in res:
            log(f"    100-text rank: {res['p65']['full_rank']}")
            log(f"    Rank growth: {res['p65']['rank_curve']}")
            log(f"    Category ranks: {dict((k, v[0]) for k, v in res['p65']['cat_ranks'].items())}")
    
    log(f"\nResults saved to {run_dir}")
    log.close()

if __name__ == "__main__":
    main()
