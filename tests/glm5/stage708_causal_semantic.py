"""
P66-P68: Phase IV-B — 语义方向因果干预(修复版) + 跨模型对齐(修复版) + 增强因果验证
========================================================================
P66: 因果验证——干预PC坐标，观察logit/top-k/类别变化（修复Logger bug）
P67: 跨模型语义方向对齐——用类别均值差方向对齐（修复维度不匹配）
P68: 增强因果——系统扫描前10个PC的因果效应量

四模型串行: Qwen3 → DS7B → GLM4 → Gemma4
增大样本: 200条文本(5类别x40)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, json
from pathlib import Path as _Path
from datetime import datetime

class Logger:
    def __init__(self, log_dir, name):
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            # Windows GBK fallback
            safe = msg.encode('gbk', errors='replace').decode('gbk')
            print(safe)
        self.f.write(msg + "\n")
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

# ===== 200 diverse texts (5 categories x 40 each) =====
def build_texts():
    T = []
    # gen_en (40)
    gen_en = [
        "The cat sat on the mat.", "A beautiful sunset over the ocean.", "The stock market crashed today.",
        "Climate change is a global challenge.", "The restaurant serves excellent Italian cuisine.",
        "The Renaissance was a period of great cultural achievement.", "The election results surprised everyone last night.",
        "The concert featured a symphony orchestra performance.", "She walked through the garden with a smile.",
        "The ancient castle stood on top of the hill.", "Breakfast is the most important meal of the day.",
        "The children played happily in the park.", "A sudden storm interrupted the picnic.",
        "The library has thousands of rare books.", "He traveled across three continents last year.",
        "The museum exhibit attracted record visitors.", "The teacher explained the concept clearly.",
        "Spring flowers bloom in every color.", "The old man told stories by the fireplace.",
        "Music brings people together across cultures.", "The river flows gently through the valley.",
        "She picked up the phone and dialed the number.", "The train arrived at the station on time.",
        "A group of friends gathered for dinner.", "The novel tells the story of a young hero.",
        "He opened the window to let in fresh air.", "The dog barked loudly at the stranger.",
        "She finished her homework before dinner.", "The city skyline looked stunning at dusk.",
        "The athlete broke the world record.", "Winter snow covered the mountains.",
        "The chef prepared a delicious meal.", "They decided to go hiking in the forest.",
        "The painting hung on the gallery wall.", "A rainbow appeared after the rain.",
        "The baby laughed at the colorful toy.", "The festival attracted thousands of tourists.",
        "The bridge connected the two towns.", "He wrote a letter to his old friend.",
        "The newspaper reported on the latest events.",
    ]
    for t in gen_en:
        T.append((t, "gen_en"))

    # math_science (40)
    math_sci = [
        "Mathematical proof by induction.", "Einstein's theory of relativity changed physics.",
        "The quantum computer solved the problem in seconds.", "DNA carries genetic information in all living organisms.",
        "Photosynthesis converts sunlight into chemical energy.", "Evolution by natural selection explains biodiversity.",
        "The Pythagorean theorem states a squared plus b squared equals c squared.",
        "The derivative of velocity is acceleration.", "Newton's third law states every action has an equal opposite reaction.",
        "Entropy always increases in isolated systems.", "The speed of light is approximately 299,792 km per second.",
        "Euler's identity connects five fundamental constants.", "The periodic table organizes elements by atomic number.",
        "Mitochondria are the powerhouse of the cell.", "The Doppler effect explains why sirens change pitch.",
        "Maxwell's equations unify electricity and magnetism.", "Thermodynamics governs heat and energy transfer.",
        "The double helix structure of DNA was discovered by Watson and Crick.",
        "Heisenberg's uncertainty principle limits measurement precision.",
        "The gravitational constant is approximately 6.674 times 10 to the negative 11th.",
        "The Schrodinger equation describes quantum mechanical systems.",
        "Protein folding determines biological function.", "The strong nuclear force binds protons in the nucleus.",
        "Fermat's last theorem was proven by Andrew Wiles.", "The Higgs boson gives particles mass.",
        "General relativity predicts gravitational waves.", "Quantum entanglement enables instant correlations.",
        "The ideal gas law relates pressure, volume, and temperature.",
        "Avogadro's number is approximately 6.022 times 10 to the 23rd.",
        "Ohm's law states voltage equals current times resistance.", "The Bohr model describes hydrogen atom energy levels.",
        "Chaos theory shows deterministic systems can be unpredictable.",
        "Bayes' theorem updates probabilities with new evidence.",
        "The Fourier transform decomposes signals into frequencies.",
        "Topology studies properties preserved under continuous deformation.",
        "Game theory analyzes strategic decision making.", "Information theory quantifies data transmission limits.",
        "Catalysis accelerates chemical reactions.", "Plate tectonics explains continental drift.",
        "The immune system defends against pathogens.", "The electromagnetic spectrum ranges from radio to gamma rays.",
        "Superconductivity allows zero resistance current flow.",
    ]
    for t in math_sci:
        T.append((t, "math"))

    # code (40)
    code = [
        "for i in range(10): print(i)",
        "def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)",
        "class NeuralNetwork: def __init__(self, layers): self.layers=layers",
        "import numpy as np; x = np.random.randn(100, 256)",
        "Recursive functions call themselves until a base case is reached.",
        "Python is the most popular programming language.", "The neural network has 100 layers.",
        "Attention is all you need for transformers.", "Gradient descent minimizes the loss function.",
        "x = torch.tensor([1.0, 2.0, 3.0])", "model.train() sets the model to training mode.",
        "def forward(self, x): return self.linear(x)", "batch_size = 32; learning_rate = 0.001",
        "optimizer = Adam(model.parameters(), lr=1e-3)", "loss = nn.CrossEntropyLoss()(pred, target)",
        "if torch.cuda.is_available(): model = model.cuda()",
        "embedding = nn.Embedding(vocab_size, hidden_dim)",
        "output = F.softmax(logits, dim=-1)",
        "self.attention = nn.MultiheadAttention(d_model, n_heads)",
        "hidden = self.rnn(embedded, hidden_state)",
        "while True: data = queue.get(); process(data)",
        "try: result = json.loads(response) except: pass",
        "sorted_list = sorted(items, key=lambda x: x['score'])",
        "from collections import defaultdict; d = defaultdict(list)",
        "np.matmul(A, B) computes matrix multiplication.",
        "model.eval() disables dropout for inference.",
        "torch.no_grad() disables gradient computation.",
        "self.linear = nn.Linear(in_features, out_features)",
        " DataLoader splits data into batches for training.",
        "checkpoint = torch.load(model_path); model.load_state_dict(checkpoint)",
        "scheduler.step() adjusts the learning rate.",
        "def train_epoch(model, loader, optimizer): loss_sum=0",
        "x = x.view(batch_size, -1) reshapes the tensor.",
        "ReLU activation sets negative values to zero.",
        "BatchNorm normalizes across the batch dimension.",
        "Dropout randomly zeros elements during training.",
        "self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)",
        "max_pool = F.max_pool2d(x, kernel_size=2)",
        "residual = x + self.block(x) defines a skip connection.",
        "self.norm = nn.LayerNorm(d_model) normalizes features.",
        "criterion = nn.MSELoss() for regression tasks.",
        "accuracy = (preds == labels).float().mean().item()",
    ]
    for t in code:
        T.append((t, "code"))

    # chinese (40)
    chinese = [
        "春天来了，花开满园。", "机器学习是人工智能的子领域。", "今天天气很好，适合出去散步。",
        "深度学习模型需要大量训练数据。", "区块链技术正在改变金融行业。", "人工智能正在快速发展。",
        "这本书非常有趣，值得一读。", "中国经济持续增长。", "量子计算是未来的重要技术。",
        "自然语言处理有许多应用场景。", "他每天早上六点起床锻炼身体。", "这个项目的预算已经超支了。",
        "研究表明，运动有助于健康。", "数据科学家需要掌握统计学和编程。", "气候变化对农业产生了重大影响。",
        "机器人技术正在迅速发展。", "这部电影讲述了二战期间的故事。", "太阳系有八大行星。",
        "互联网改变了人们的生活方式。", "学生应该培养独立思考的能力。", "科技创新是推动社会进步的重要力量。",
        "城市化进程带来了许多挑战。", "教育公平是社会发展的基石。", "传统手工艺需要得到保护和传承。",
        "旅游业的发展促进了文化交流。", "食品安全问题引起了广泛关注。", "可再生能源是未来的发展趋势。",
        "环境保护需要全社会的共同努力。", "医疗改革关系到每个人的切身利益。",
        "智能制造正在改变传统工业模式。", "全球化进程深刻影响着世界格局。",
        "青年一代肩负着时代赋予的使命。", "文化多样性是人类社会的宝贵财富。",
        "太空探索代表了人类的科学精神。", "心理健康问题日益受到重视。",
        "电子商务改变了人们的消费习惯。", "人口老龄化带来了社会保障压力。",
        "高等教育普及率不断提高。", "交通拥堵是城市发展的常见问题。",
        "食品安全监管需要加强。", "志愿服务体现了公民的社会责任感。",
    ]
    for t in chinese:
        T.append((t, "chinese"))

    # reasoning (40)
    reasoning = [
        "If all humans are mortal and Socrates is human, then Socrates is mortal.",
        "The probability of rain given cloudy skies is approximately 70 percent.",
        "To solve this equation, first isolate the variable on one side.",
        "The hypothesis is testable and falsifiable under controlled conditions.",
        "Therefore, we can conclude that the experiment supports the theory.",
        "The correlation does not imply causation in this observational study.",
        "Given the premises, the conclusion follows logically by deduction.",
        "The null hypothesis cannot be rejected at the 5 percent significance level.",
        "This approach reduces computational complexity from O(n squared) to O(n log n).",
        "We need to control for confounding variables in the regression model.",
        "The confidence interval is 95 percent with a margin of error of plus or minus 3 percent.",
        "Assuming the model is correctly specified, the estimator is unbiased.",
        "The algorithm converges in polynomial time under these constraints.",
        "By induction, the property holds for all natural numbers.",
        "The posterior distribution updates our prior belief given the evidence.",
        "This function is monotonically increasing on the given interval.",
        "The optimal solution lies at the boundary of the feasible region.",
        "We can approximate the integral using the trapezoidal rule.",
        "The expected value of the random variable is the sum of all possible outcomes weighted by their probabilities.",
        "This recurrence relation has a closed-form solution of O(n squared).",
        "Both necessary and sufficient conditions must be satisfied for equivalence.",
        "The proof proceeds by contradiction: assume the opposite and derive a false statement.",
        "A sufficient statistic captures all information about the parameter in the data.",
        "The Markov property states future states depend only on the current state.",
        "Gradient ascent finds the maximum of a differentiable function.",
        "Cross-validation provides a robust estimate of out-of-sample performance.",
        "The law of large numbers guarantees convergence of sample means.",
        "Mutual information measures the dependence between two random variables.",
        "Pareto efficiency means no individual can be made better off without making another worse off.",
        "Dynamic programming solves problems by combining solutions to subproblems.",
        "A greedy algorithm makes locally optimal choices at each step.",
        "The Nash equilibrium is a state where no player benefits from changing strategy.",
        "Kernel methods map data into higher dimensional feature spaces.",
        "Regularization prevents overfitting by adding a penalty term to the loss.",
        "Ensemble methods combine multiple weak learners into a strong predictor.",
        "The bias-variance tradeoff is fundamental to supervised learning.",
        "Transfer learning leverages knowledge from source tasks to improve target tasks.",
        "Contrastive learning pulls positive pairs together and pushes negative pairs apart.",
        "The attention mechanism computes weighted sums of values based on query-key similarity.",
        "Curriculum learning presents training examples in increasing order of difficulty.",
        "Meta-learning learns to learn by optimizing across multiple tasks.",
    ]
    for t in reasoning:
        T.append((t, "reasoning"))

    return T

TEXTS = build_texts()

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

def safe_decode(tokens, tokenizer, n=5):
    """Safely decode top-k token ids, handle GBK encoding errors"""
    try:
        words = []
        for tid in tokens.tolist()[:n]:
            try:
                w = tokenizer.decode([tid]).strip()
                if w:
                    # Filter out problematic Unicode chars that cause GBK errors
                    w_safe = w.encode('ascii', errors='replace').decode('ascii')
                    if w_safe:
                        words.append(w_safe)
            except:
                pass
        return " ".join(words) if words else "decode_error"
    except:
        return "decode_error"

def compute_effective_rank(matrix, threshold=0.01):
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    return (S > threshold * S[0]).sum().item(), S

# ===== P65: Large sample rank (200 texts) =====
def p65_large_sample(h_all, categories, log):
    log(f"\n  P65: 200-text effective rank")
    N = h_all.shape[0]
    full_rank, S = compute_effective_rank(h_all)
    log(f"    200 texts rank: {full_rank} (N={N})")

    top_svs = [f"{s:.1f}" for s in S[:15].tolist()]
    log(f"    Top-15 SV: {top_svs}")

    cat_names = sorted(set(categories))
    cat_ranks = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        h_cat = h_all[idx]
        r, _ = compute_effective_rank(h_cat)
        cat_ranks[cat] = (r, len(idx))

    log(f"    Per-category ranks:")
    for cat, (r, n) in cat_ranks.items():
        log(f"      {cat}({n}): rank={r}")

    # Rank growth curve
    rank_curve = {}
    for n in [10, 20, 40, 80, 120, 160, 200]:
        if n > N:
            continue
        torch.manual_seed(42)
        idx = torch.randperm(N)[:n]
        r, _ = compute_effective_rank(h_all[idx])
        rank_curve[n] = r

    log(f"    Rank growth: {json.dumps({str(k):v for k,v in rank_curve.items()})}")

    # Cumulative variance explained
    total = (S ** 2).sum().item()
    cumvar = torch.cumsum(S ** 2, 0) / total
    for k in [3, 5, 10, 20, 30, 50, 100]:
        if k <= len(cumvar):
            log(f"    Top-{k} var: {cumvar[k-1]*100:.1f}%")

    return {"full_rank": full_rank, "cat_ranks": cat_ranks, "rank_curve": rank_curve,
            "top5_var": cumvar[4].item() if len(cumvar)>=5 else None}


# ===== P66: Causal PC intervention (FIXED) =====
def p66_causal_pc(model, tokenizer, h_all, texts, categories, log):
    log(f"\n  P66: Causal PC intervention (200 texts)")

    W = get_unembed(model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)

    K = min(10, Vt.shape[0])

    # Pick 2 representative texts per category
    cat_names = sorted(set(categories))
    test_indices = []
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat][:2]
        test_indices.extend(idx)

    # Shift magnitude: ±0.5 std, ±1.0 std, ±2.0 std
    shifts = [0.5, 1.0, 2.0]

    log(f"    Testing {len(test_indices)} texts x {K} PCs x {len(shifts)} shifts")

    pc_results = {}  # pc_k -> {shift: {avg_kl, top1_change_rate, top5_change_rate}}

    for k in range(K):
        pc_dir = Vt[k]
        pc_std = S[k].item()

        shift_data = {}
        for shift_mult in shifts:
            kls = []
            top1_changed = 0
            top5_changed = 0
            total = 0

            for ti in test_indices:
                h = h_all[ti:ti+1]
                logits_orig = (h @ W.T).squeeze()
                top1_orig = logits_orig.argmax().item()
                top5_orig = set(torch.topk(logits_orig, 5).indices.tolist())

                for sign in [1, -1]:
                    h_shifted = h + sign * pc_dir * pc_std * shift_mult
                    logits_new = (h_shifted @ W.T).squeeze()

                    kl = F.kl_div(
                        F.log_softmax(logits_new, -1),
                        F.softmax(logits_orig, -1),
                        reduction='sum'
                    ).item()
                    kls.append(kl)

                    top1_new = logits_new.argmax().item()
                    top5_new = set(torch.topk(logits_new, 5).indices.tolist())
                    if top1_new != top1_orig:
                        top1_changed += 1
                    if top5_new != top5_orig:
                        top5_changed += 1
                    total += 1

            avg_kl = np.mean(kls)
            t1_rate = top1_changed / max(total, 1)
            t5_rate = top5_changed / max(total, 1)
            shift_data[f"{shift_mult}std"] = {
                "avg_kl": avg_kl,
                "top1_change": t1_rate,
                "top5_change": t5_rate,
            }

        pc_results[f"PC{k}"] = shift_data
        log(f"    PC{k}(SV={S[k]:.1f}, var={S[k]**2/(S**2).sum()*100:.1f}%):")
        for sm, sd in shift_data.items():
            log(f"      {sm}: KL={sd['avg_kl']:.2f}, top1_chg={sd['top1_change']*100:.0f}%, top5_chg={sd['top5_change']*100:.0f}%")

    # Overall summary: which PCs are most causal?
    log(f"\n    Causality ranking (by top1_change at 1std):")
    causal_scores = []
    for k in range(K):
        pc_key = f"PC{k}"
        if pc_key in pc_results and "1.0std" in pc_results[pc_key]:
            score = pc_results[pc_key]["1.0std"]["top1_change"]
            causal_scores.append((k, score, S[k].item()))
    causal_scores.sort(key=lambda x: -x[1])
    for k, score, sv in causal_scores:
        log(f"      PC{k}: top1_change={score*100:.0f}%, SV={sv:.1f}")

    return pc_results


# ===== P67: Cross-model alignment (FIXED - use per-text CCA) =====
def p67_cross_model(all_pca_data, all_h_data, categories, log):
    """P67: Compare semantic encoding across models using CCA on h_final vectors
    
    Since models have different hidden dims, we use Canonical Correlation Analysis (CCA)
    to find correlated subspaces.
    """
    log(f"\n  P67: Cross-model CCA alignment")

    models = list(all_pca_data.keys())
    if len(models) < 2:
        log("    Need >= 2 models, skipping")
        return {}

    # For each pair, compute CCA on category-level representations
    # Use category means as representations (5 categories per model)
    cat_names = sorted(set(categories))

    alignments = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            log(f"\n    {m1} vs {m2}:")

            # Get category mean vectors from h_final
            h1 = all_h_data[m1]  # (206, d1) numpy
            h2 = all_h_data[m2]  # (206, d2) numpy

            # Method 1: Procrustes alignment on category means
            # Compare direction of category differences
            means1 = {}
            means2 = {}
            for cat in cat_names:
                idx = [k for k, c in enumerate(categories) if c == cat]
                means1[cat] = np.mean(h1[idx], axis=0)
                means2[cat] = np.mean(h2[idx], axis=0)

            cat_pairs = [("chinese", "gen_en"), ("chinese", "code"), ("chinese", "math"),
                         ("gen_en", "code"), ("gen_en", "math"), ("code", "math")]

            # Build direction matrices for Procrustes
            D1 = np.stack([means1[c2] - means1[c1] for c1, c2 in cat_pairs])  # (6, d1)
            D2 = np.stack([means2[c2] - means2[c1] for c1, c2 in cat_pairs])  # (6, d2)

            # Normalize each row (direction only)
            D1_norm = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-8)
            D2_norm = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-8)

            # Compute correlation matrix between directions
            # R = D1_norm @ D1_norm^T vs D2_norm @ D2_norm^T
            R1 = D1_norm @ D1_norm.T  # (6, 6) - within-model1 direction similarity
            R2 = D2_norm @ D2_norm.T  # (6, 6) - within-model2 direction similarity

            # Correlation of correlation matrices
            corr = np.corrcoef(R1.flatten(), R2.flatten())[0, 1]
            log(f"      Direction structure correlation: r={corr:.4f}")

            # Per-pair analysis: use CCA to find shared subspace
            # CCA on category means: X (5xd1) and Y (5xd2)
            X = np.stack([means1[c] for c in cat_names])  # (5, d1)
            Y = np.stack([means2[c] for c in cat_names])  # (5, d2)

            # Simple CCA via SVD of cross-covariance
            Xc = X - X.mean(0)
            Yc = Y - Y.mean(0)
            Cxy = Xc.T @ Yc  # (d1, d2)
            U, S_cc, Vt = np.linalg.svd(Cxy, full_matrices=False)

            # Canonical correlations
            n_canonical = min(5, len(S_cc))
            log(f"      CCA canonical correlations (top {n_canonical}):")
            total_cc = 0
            for k in range(n_canonical):
                log(f"        CC{k+1}: {S_cc[k]:.4f}")
                total_cc += S_cc[k]
            log(f"      Sum of top-3 CC: {sum(S_cc[:3]):.4f}")

            # Interpretation: which category pairs are best aligned?
            pair_scores = []
            for pi, (c1, c2) in enumerate(cat_pairs):
                d1_dir = D1_norm[pi]
                d2_dir = D2_norm[pi]
                # Project d2 into d1's space via CCA
                d2_projected = d2_dir @ Vt[:n_canonical].T @ U[:, :n_canonical].T
                sim = np.dot(d1_dir, d2_projected) / (np.linalg.norm(d1_dir) * np.linalg.norm(d2_projected) + 1e-8)
                pair_scores.append((f"{c1} vs {c2}", float(sim)))
                log(f"      {c1} vs {c2}: CCA-aligned cos={sim:.4f}")

            avg = np.mean([c for _, c in pair_scores])
            log(f"      Average CCA-aligned cosine: {avg:.4f}")
            alignments[f"{m1}-{m2}"] = {
                "direction_corr": float(corr),
                "cca_cc": [float(s) for s in S_cc[:n_canonical]],
                "avg_aligned_cos": float(avg),
                "pairs": pair_scores,
            }

    return alignments


# ===== P68: PC semantic interpretation =====
def p68_pc_semantics(model, tokenizer, h_all, categories, log):
    """P68: Detailed semantic analysis of top-10 PCs"""
    log(f"\n  P68: PC semantic interpretation")

    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)

    W = get_unembed(model)  # (vocab_size, d_model)
    K = min(10, Vt.shape[0])

    # For each PC: project onto vocab, find top positive/negative tokens
    pc_semantics = {}
    for k in range(K):
        pc_dir = Vt[k]
        # Project PC direction onto unembed: pc_dir @ W.T -> (vocab_size,)
        proj = (pc_dir @ W.T)
        var_pct = S[k]**2 / (S**2).sum() * 100

        top_pos = torch.topk(proj, 10)
        top_neg = torch.topk(proj, 10, largest=False)

        pos_words = safe_decode(top_pos.indices, tokenizer, 10)
        neg_words = safe_decode(top_neg.indices, tokenizer, 10)

        # Category correlation: for each category, compute mean projection on this PC
        cat_proj = {}
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            h_cat = h_all[idx]
            proj_vals = (h_cat - h_mean) @ pc_dir  # (n_cat,)
            cat_proj[cat] = {"mean": proj_vals.mean().item(), "std": proj_vals.std().item()}

        log(f"\n    PC{k} (var={var_pct:.1f}%):")
        log(f"      Top+ tokens: {pos_words}")
        log(f"      Top- tokens: {neg_words}")
        log(f"      Category projections:")
        for cat, vals in cat_proj.items():
            log(f"        {cat}: mean={vals['mean']:.1f}, std={vals['std']:.1f}")

        # Find most distinguishing category pair for this PC
        cat_names = sorted(set(categories))
        max_diff = 0
        max_pair = ("", "")
        for i, c1 in enumerate(cat_names):
            for c2 in cat_names[i+1:]:
                diff = abs(cat_proj[c1]["mean"] - cat_proj[c2]["mean"])
                if diff > max_diff:
                    max_diff = diff
                    max_pair = (c1, c2)
        log(f"      Most distinguishing: {max_pair[0]} vs {max_pair[1]} (diff={max_diff:.1f})")

        pc_semantics[f"PC{k}"] = {
            "var_pct": var_pct,
            "cat_proj": {cat: vals["mean"] for cat, vals in cat_proj.items()},
            "max_pair": list(max_pair),
            "max_diff": max_diff,
        }

    return pc_semantics


def run_suite(model_name, log):
    log(f"\n{'='*70}")
    log(f"Model: {model_name}")
    log(f"{'='*70}")

    model, tokenizer = load_model(model_name, log)

    categories = [c for _, c in TEXTS]
    log(f"  Computing h_final for {len(TEXTS)} texts...")
    h_all = get_all_h(model, tokenizer, TEXTS)
    log(f"  h_all shape: {h_all.shape}, norm range: [{h_all.norm(dim=1).min():.1f}, {h_all.norm(dim=1).max():.1f}]")

    # Category means
    cat_means = {}
    for cat in set(categories):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)

    results = {}

    # P65
    try:
        results["p65"] = p65_large_sample(h_all, categories, log)
    except Exception as e:
        log(f"  P65 FAILED: {e}")

    # P66
    try:
        results["p66"] = p66_causal_pc(model, tokenizer, h_all, TEXTS, categories, log)
    except Exception as e:
        log(f"  P66 FAILED: {e}")

    # P68
    try:
        results["p68"] = p68_pc_semantics(model, tokenizer, h_all, categories, log)
    except Exception as e:
        log(f"  P68 FAILED: {e}")

    # Store PCA data for P67
    results["pca_for_p67"] = {
        "cat_means": {k: v.numpy() for k, v in cat_means.items()},
    }
    results["h_for_p67"] = h_all.numpy()

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage708_causal_semantic_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    global log
    log = Logger(run_dir, "results")
    log("P66-P68: Phase IV-B - Causal PC + Cross-Model + PC Semantics (200 texts)")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS)} (5 categories x 40 each)")
    cat_counts = {}
    for _, c in TEXTS:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    log(f"Categories: {cat_counts}")

    all_results = {}
    all_pca = {}
    all_h = {}
    categories = [c for _, c in TEXTS]

    for mn in MODEL_ORDER:
        try:
            results = run_suite(mn, log)
            all_results[mn] = results
            if "pca_for_p67" in results:
                all_pca[mn] = results["pca_for_p67"]
            if "h_for_p67" in results:
                all_h[mn] = results["h_for_p67"]
        except Exception as e:
            log(f"  FATAL {mn}: {e}")
            import traceback
            traceback.print_exc()

    # P67: Cross-model (after all models done)
    log(f"\n{'='*70}")
    log("P67: Cross-Model CCA Alignment")
    log(f"{'='*70}")
    try:
        p67_result = p67_cross_model(all_pca, all_h, categories, log)
        all_results["p67"] = p67_result
    except Exception as e:
        log(f"  P67 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Save JSON
    save_data = {}
    for key, val in all_results.items():
        save_data[key] = {}
        if isinstance(val, dict):
            for k, v in val.items():
                if k == "pca_for_p67":
                    continue
                try:
                    json.dumps(v)
                    save_data[key][k] = v
                except (TypeError, ValueError):
                    save_data[key][k] = str(v)

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    # Final summary
    log(f"\n{'='*70}")
    log("FINAL SUMMARY")
    log(f"{'='*70}")

    for mn, res in all_results.items():
        if mn == "p67":
            continue
        log(f"\n  {mn}:")
        if "p65" in res:
            log(f"    Rank(200): {res['p65']['full_rank']}")
            log(f"    Growth: {json.dumps({str(k):v for k,v in res['p65']['rank_curve'].items()})}")
        if "p66" in res:
            # Find most causal PC
            best = None
            for pc_key, shift_data in res["p66"].items():
                if "1.0std" in shift_data and shift_data["1.0std"]["top1_change"] > (best[1] if best else 0):
                    best = (pc_key, shift_data["1.0std"]["top1_change"])
            if best:
                log(f"    Most causal PC: {best[0]} (top1_chg={best[1]*100:.0f}% at 1std)")

    log(f"\nResults saved to {run_dir}")
    log.close()

if __name__ == "__main__":
    main()
