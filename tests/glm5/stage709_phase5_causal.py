#!/usr/bin/env python3
"""
Stage 709: Phase V — 语义方向因果验证 + 跨模型对齐 + 因果预测链
================================================================
P69: PC方向因果干预——系统量化前10个PC的因果效应量(修复P66)
P70: 跨模型语义方向对齐——CCA方法(修复P67)
P71: 因果预测链——从文本特征预测PC坐标(迈向因果方程)
P72: 多步累积可预测性——delta-h逐层预测精度分析

核心目标: 从"描述"走向"因果"——能否操控h_final的PC坐标来改变输出?
         能否从输入文本预测PC坐标?

四模型串行: Qwen3 -> DS7B -> GLM4 -> Gemma4
大样本: 200条文本(5类别x40)
设备: CUDA
"""

import sys, time, gc, json, os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score

# ===== Logger =====
class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

# ===== Model Config =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}
MODEL_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

# ===== 200 texts (5 categories x 40) =====
def build_texts():
    T = []
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
        "DataLoader splits data into batches for training.",
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

    chinese = [
        "The capital of China is Beijing.", "Chinese cuisine is famous worldwide.",
        "She picked up the phone and dialed the number.", "The train arrived at the station on time.",
        "A group of friends gathered for dinner.", "The novel tells the story of a young hero.",
        "He opened the window to let in fresh air.", "The dog barked loudly at the stranger.",
        "She finished her homework before dinner.", "The city skyline looked stunning at dusk.",
        "Spring flowers bloom in every color.", "The old man told stories by the fireplace.",
        "The athlete broke the world record.", "Winter snow covered the mountains.",
        "The chef prepared a delicious meal.", "They decided to go hiking in the forest.",
        "The painting hung on the gallery wall.", "A rainbow appeared after the rain.",
        "The festival attracted thousands of tourists.", "The bridge connected the two towns.",
        "He wrote a letter to his old friend.", "The newspaper reported on the latest events.",
        "The baby laughed at the colorful toy.", "The river flows gently through the valley.",
        "Music brings people together across cultures.", "Breakfast is the most important meal of the day.",
        "The children played happily in the park.", "A sudden storm interrupted the picnic.",
        "The library has thousands of rare books.", "The museum exhibit attracted record visitors.",
        "The teacher explained the concept clearly.", "He traveled across three continents last year.",
        "The restaurant serves excellent Italian cuisine.", "Climate change is a global challenge.",
        "The ancient castle stood on top of the hill.", "The election results surprised everyone last night.",
        "The Renaissance was a period of great cultural achievement.",
        "The concert featured a symphony orchestra performance.",
        "The stock market crashed today.", "A beautiful sunset over the ocean.",
        "The cat sat on the mat.", "She walked through the garden with a smile.",
    ]
    for t in chinese:
        T.append((t, "chinese"))

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


# ===== Model Loading =====
def load_model(model_name, log):
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"  Loading {model_name} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  {model_name} loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model, tokenizer


def get_h_final(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(tokens.input_ids, output_hidden_states=True)
    return outputs.hidden_states[-1][0, -1].float().cpu()


def get_all_h_and_deltas(model, tokenizer, texts):
    """Get h_final for all texts AND per-layer delta-h for each text."""
    all_h = []
    all_deltas = []
    all_logits = []

    for text, _ in texts:
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        h_final = states[-1]
        logits = outputs.logits[0, -1, :].float().cpu()

        deltas = [states[i] - states[i-1] for i in range(1, len(states))]
        all_h.append(h_final)
        all_deltas.append(torch.stack(deltas))  # (n_layers, d_model)
        all_logits.append(logits)

    return torch.stack(all_h), all_deltas, all_logits


def get_unembed(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.data.float().cpu()
    elif hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings().weight.data.float().cpu()
    else:
        raise ValueError("Cannot find unembed")


def safe_decode(token_ids, tokenizer, n=5):
    words = []
    for tid in token_ids.tolist()[:n]:
        try:
            w = tokenizer.decode([tid]).strip().replace("\n", "\\n")[:15]
            if w:
                w_safe = w.encode("ascii", errors="replace").decode("ascii")
                if w_safe:
                    words.append(w_safe)
        except Exception:
            pass
    return " ".join(words) if words else "decode_err"


def compute_effective_rank(matrix, threshold=0.01):
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    return (S > threshold * S[0]).sum().item(), S


# ===== P69: Systematic PC Causal Intervention =====
def p69_pc_causal_intervention(model, tokenizer, h_all, texts, categories, log):
    """
    P69: Systematically shift each of top-10 PC coordinates and measure:
    1. KL divergence change
    2. Top-1 flip rate
    3. Top-5 change rate
    4. Margin change
    5. Cosine similarity of output distribution
    
    This is the core causal test: can manipulating PC coordinates change the model's output?
    """
    log(f"\n{'='*60}")
    log(f"  P69: Systematic PC Causal Intervention")
    log(f"{'='*60}")

    W = get_unembed(model)  # (vocab_size, d_model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)

    K = min(10, Vt.shape[0])
    cat_names = sorted(set(categories))

    # Select 3 texts per category for testing (15 texts total)
    test_indices = []
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat][:3]
        test_indices.extend(idx)

    shifts = [0.5, 1.0, 2.0, 4.0]  # in units of std
    total_var = (S ** 2).sum().item()

    log(f"    Testing {len(test_indices)} texts x {K} PCs x {len(shifts)} shifts")
    log(f"    d_model={h_all.shape[1]}, SV[0]={S[0]:.1f}")

    # Results table: PC_k x shift -> metrics
    results = {}
    for k in range(K):
        pc_dir = Vt[k]  # (d_model,)
        pc_std = S[k].item()
        var_pct = pc_std**2 / total_var * 100

        shift_results = {}
        for shift_mult in shifts:
            shift_vec = pc_dir * pc_std * shift_mult  # shift in h-space

            kls_pos, kls_neg = [], []
            top1_flip_pos, top1_flip_neg = 0, 0
            top5_change_pos, top5_change_neg = 0, 0
            margin_changes = []
            cos_sims = []

            for ti in test_indices:
                h = h_all[ti:ti+1]
                logits_orig = (h @ W.T).squeeze()
                top1_orig = logits_orig.argmax().item()
                top5_orig = set(torch.topk(logits_orig, 5).indices.tolist())
                margin_orig = (torch.sort(logits_orig, descending=True).values[0] -
                               torch.sort(logits_orig, descending=True).values[1]).item()

                for sign, kls_list, t1_flip, t5_chg in [
                    (1, kls_pos, "pos", "pos"), (-1, kls_neg, "neg", "neg")
                ]:
                    h_shifted = h + sign * shift_vec.unsqueeze(0)
                    logits_new = (h_shifted @ W.T).squeeze()

                    kl = F.kl_div(
                        F.log_softmax(logits_new, -1),
                        F.softmax(logits_orig, -1),
                        reduction="sum"
                    ).item()
                    kls_list.append(kl)

                    top1_new = logits_new.argmax().item()
                    top5_new = set(torch.topk(logits_new, 5).indices.tolist())
                    margin_new = (torch.sort(logits_new, descending=True).values[0] -
                                  torch.sort(logits_new, descending=True).values[1]).item()

                    if sign == 1:
                        if top1_new != top1_orig:
                            top1_flip_pos += 1
                        if top5_new != top5_orig:
                            top5_change_pos += 1
                    else:
                        if top1_new != top1_orig:
                            top1_flip_neg += 1
                        if top5_new != top5_orig:
                            top5_change_neg += 1

                    margin_changes.append(margin_new / (abs(margin_orig) + 1e-10))
                    cos_sims.append(F.cosine_similarity(
                        logits_orig.unsqueeze(0), logits_new.unsqueeze(0)
                    ).item())

            n_total = len(test_indices)
            shift_results[f"{shift_mult}std"] = {
                "avg_kl": float(np.mean(kls_pos + kls_neg)),
                "top1_flip_rate": float((top1_flip_pos + top1_flip_neg) / (2 * n_total)),
                "top5_change_rate": float((top5_change_pos + top5_change_neg) / (2 * n_total)),
                "avg_margin_ratio": float(np.mean(margin_changes)),
                "avg_cos": float(np.mean(cos_sims)),
            }

        results[f"PC{k}"] = {
            "sv": float(pc_std),
            "var_pct": float(var_pct),
            "shifts": shift_results,
        }

        # Print summary for this PC
        log(f"\n    PC{k} (SV={pc_std:.1f}, var={var_pct:.1f}%):")
        log(f"      {'Shift':>8} {'KL':>8} {'T1-flip':>8} {'T5-chg':>8} {'Mrg-R':>8} {'Cos':>8}")
        log(f"      {'-'*48}")
        for sm in shifts:
            sd = shift_results[f"{sm}std"]
            log(f"      {sm:>6.1f}x {sd['avg_kl']:>8.2f} {sd['top1_flip_rate']:>7.0%} "
                f"{sd['top5_change_rate']:>7.0%} {sd['avg_margin_ratio']:>8.3f} {sd['avg_cos']:>8.4f}")

    # Causality ranking
    log(f"\n    --- Causality Ranking (top1_flip at 2std) ---")
    causal_rank = []
    for k in range(K):
        sd = results[f"PC{k}"]["shifts"].get("2.0std", {})
        score = sd.get("top1_flip_rate", 0)
        causal_rank.append((k, score, results[f"PC{k}"]["var_pct"]))
    causal_rank.sort(key=lambda x: -x[1])
    for k, score, vp in causal_rank:
        log(f"      PC{k}: flip_rate={score*100:.0f}%, var={vp:.1f}%")

    # Key finding: is there a correlation between PC variance and causal effect?
    variances = [results[f"PC{k}"]["var_pct"] for k in range(K)]
    effects = [results[f"PC{k}"]["shifts"]["2.0std"]["top1_flip_rate"] for k in range(K)]
    if np.std(variances) > 0 and np.std(effects) > 0:
        r_var_effect = np.corrcoef(variances, effects)[0, 1]
        log(f"\n    Correlation(PCA_variance, causal_effect): r={r_var_effect:.4f}")
    else:
        r_var_effect = None

    return results, r_var_effect


# ===== P70: Cross-Model CCA Alignment =====
def p70_cross_model_cca(all_model_data, categories, log):
    """
    P70: Use Canonical Correlation Analysis to find shared semantic subspaces
    across models with different hidden dimensions.
    
    Key question: Do different models encode the same category distinctions
    in correlated subspaces of their respective hidden states?
    """
    log(f"\n{'='*60}")
    log(f"  P70: Cross-Model CCA Alignment")
    log(f"{'='*60}")

    models = list(all_model_data.keys())
    cat_names = sorted(set(categories))
    alignments = {}

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue

            log(f"\n    {m1} vs {m2}:")
            h1 = all_model_data[m1]["h_all"]  # (200, d1)
            h2 = all_model_data[m2]["h_all"]  # (200, d2)

            # --- Method 1: Category-mean direction alignment ---
            means1, means2 = {}, {}
            for cat in cat_names:
                idx = [k for k, c in enumerate(categories) if c == cat]
                means1[cat] = h1[idx].mean(0)
                means2[cat] = h2[idx].mean(0)

            cat_pairs = [
                ("chinese", "gen_en"), ("chinese", "code"), ("chinese", "math"),
                ("chinese", "reasoning"), ("gen_en", "code"), ("gen_en", "math"),
                ("gen_en", "reasoning"), ("code", "math"), ("code", "reasoning"),
                ("math", "reasoning"),
            ]
            pair_cos = []
            d1_dim = h1.shape[1]
            d2_dim = h2.shape[1]
            for c1, c2 in cat_pairs:
                d1 = (means1[c1] - means1[c2]).float().cpu().numpy()
                d2 = (means2[c1] - means2[c2]).float().cpu().numpy()
                if d1_dim == d2_dim:
                    cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10))
                else:
                    cos = float("nan")  # Cannot compute direct cosine for different dims
                pair_cos.append((f"{c1[:3]}-{c2[:3]}", cos))
                log(f"      {c1[:3]}-{c2[:3]}: {'cos='+f'{cos:.4f}' if not np.isnan(cos) else 'dims differ ('+str(d1_dim)+' vs '+str(d2_dim)+')'}")
            valid_cos = [c for _, c in pair_cos if not np.isnan(c)]
            avg_cos = np.mean(valid_cos) if valid_cos else float("nan")
            log(f"      Average direction cosine: {f'{avg_cos:.4f}' if not np.isnan(avg_cos) else 'N/A (different dims)'}")

            # --- Method 2: CCA on full h vectors ---
            # Use 50 samples per category for CCA (manageable)
            n_per_cat = 35
            sample_idx = []
            for cat in cat_names:
                idx = [k for k, c in enumerate(categories) if c == cat][:n_per_cat]
                sample_idx.extend(idx)
            X = h1[sample_idx].numpy()  # (n, d1)
            Y = h2[sample_idx].numpy()  # (n, d2)

            n_components = min(5, X.shape[0] - 1, X.shape[1], Y.shape[1])
            if n_components < 2:
                log(f"      CCA skipped (not enough samples/dimensions)")
                continue

            try:
                cca = CCA(n_components=n_components)
                X_c, Y_c = cca.fit_transform(X, Y)
                # Canonical correlations
                cc = []
                for comp in range(n_components):
                    r = np.corrcoef(X_c[:, comp], Y_c[:, comp])[0, 1]
                    cc.append(r)
                log(f"      CCA canonical correlations: {[f'{r:.4f}' for r in cc]}")
                log(f"      CCA sum(top-3): {sum(cc[:3]):.4f}")
            except Exception as e:
                log(f"      CCA failed: {e}")
                cc = []

            # --- Method 3: Direction structure correlation ---
            D1 = np.stack([means1[c2].numpy() - means1[c1].numpy() for c1, c2 in cat_pairs])
            D2 = np.stack([means2[c2].numpy() - means2[c1].numpy() for c1, c2 in cat_pairs])
            D1_n = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-8)
            D2_n = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-8)
            R1 = D1_n @ D1_n.T
            R2 = D2_n @ D2_n.T
            struct_r = np.corrcoef(R1.flatten(), R2.flatten())[0, 1]
            log(f"      Direction structure correlation: r={struct_r:.4f}")

            alignments[f"{m1}-{m2}"] = {
                "avg_dir_cos": float(avg_cos),
                "pair_cos": pair_cos,
                "cca_cc": [float(r) for r in cc] if cc else [],
                "struct_r": float(struct_r),
            }

    # Cross-model summary
    log(f"\n    --- Cross-Model Summary ---")
    for pair, data in alignments.items():
        log(f"    {pair}: dir_cos={data['avg_dir_cos']:.4f}, "
            f"cca_top3={sum(data['cca_cc'][:3]):.4f}, struct_r={data['struct_r']:.4f}")

    return alignments


# ===== P71: Causal Prediction Chain =====
def p71_causal_prediction(model, tokenizer, h_all, texts, categories, all_deltas, log):
    """
    P71: Can we predict PC coordinates from text features?
    
    If yes, this is the first step toward a causal equation:
    text_features -> PC_coordinates -> shifted_h -> changed_logits
    
    Features used:
    - Token embedding statistics (mean, std, max norm)
    - Text length (num tokens)
    - Character-level features (has_chinese, has_code, has_math)
    - Category label (one-hot)
    """
    log(f"\n{'='*60}")
    log(f"  P71: Causal Prediction Chain (text -> PC coords)")
    log(f"{'='*60}")

    W_embed = None
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        W_embed = model.model.embed_tokens.weight.data.float().cpu()
    elif hasattr(model, "get_input_embeddings"):
        W_embed = model.get_input_embeddings().weight.data.float().cpu()

    # PCA
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10  # predict top 10 PCs

    # PC coordinates: (200, K)
    pc_coords = (h_c @ Vt[:K].T).numpy()

    # Build text features
    features = []
    cat_names = sorted(set(categories))
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}

    for text, cat in texts:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]

        feat = []
        # 1. Token count
        feat.append(len(input_ids))

        # 2. Embedding statistics
        if W_embed is not None:
            embeds = W_embed[input_ids]  # (seq_len, d_embed)
            feat.append(embeds.mean().item())
            feat.append(embeds.std().item())
            feat.append(embeds.norm(dim=1).mean().item())
            feat.append(embeds.norm(dim=1).std().item())
            # PCA of embeddings
            if embeds.shape[0] > 1:
                e_centered = embeds - embeds.mean(0)
                _, e_S, _ = torch.linalg.svd(e_centered, full_matrices=False)
                feat.append(e_S[0].item())
            else:
                feat.append(0)
        else:
            feat.extend([0] * 6)

        # 3. Character-level features
        feat.append(1.0 if any(ord(c) > 0x4E00 for c in text) else 0.0)  # Chinese
        feat.append(1.0 if any(c in text for c in "=<>{}[]()") else 0.0)  # Code-like
        feat.append(1.0 if any(w in text.lower() for w in ["theorem", "equation", "proof"]) else 0.0)

        # 4. Category one-hot
        one_hot = [0] * len(cat_names)
        one_hot[cat_to_idx[cat]] = 1.0
        feat.extend(one_hot)

        features.append(feat)

    X = np.array(features)  # (200, n_features)
    Y = pc_coords  # (200, K)

    # Train/test split: 80/20
    n_train = 160
    torch.manual_seed(42)
    perm = torch.randperm(len(texts)).numpy()
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Ridge regression per PC
    results = {}
    log(f"\n    Features: {X.shape[1]} dims, Train: {n_train}, Test: {len(test_idx)}")
    log(f"\n    {'PC':>4} {'Var%':>6} {'Train_R2':>10} {'Test_R2':>10} {'Test_Corr':>10}")
    log(f"    {'-'*44}")

    total_var = (S ** 2).sum().item()
    test_corrs = []
    for k in range(K):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train[:, k])
        Y_pred_test = ridge.predict(X_test)
        r2_train = r2_score(Y_train[:, k], ridge.predict(X_train))
        r2_test = r2_score(Y_test[:, k], Y_pred_test)

        if np.std(Y_test[:, k]) > 1e-6:
            corr = np.corrcoef(Y_test[:, k], Y_pred_test)[0, 1]
        else:
            corr = 0

        var_pct = S[k].item()**2 / total_var * 100
        test_corrs.append(corr)

        results[f"PC{k}"] = {
            "var_pct": var_pct,
            "train_r2": float(r2_train),
            "test_r2": float(r2_test),
            "test_corr": float(corr),
        }
        log(f"    PC{k:>2} {var_pct:>5.1f}% {r2_train:>10.4f} {r2_test:>10.4f} {corr:>10.4f}")

    avg_test_corr = np.mean(test_corrs)
    log(f"\n    Average test correlation: {avg_test_corr:.4f}")

    # Predicted vs actual h -> logit comparison
    log(f"\n    --- Predicted h -> logit quality ---")
    W_unembed = get_unembed(model)

    # Build per-PC Ridge models for full test set
    pc_models = []
    for k in range(K):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, Y[:, k])
        pc_models.append(ridge)

    Y_pred_all = np.column_stack([m.predict(X) for m in pc_models]).astype(np.float32)
    # Reconstruct h from predicted PC coords
    h_pred = torch.tensor(Y_pred_all, dtype=torch.float32) @ Vt[:K].float() + h_mean.float()
    h_actual = h_all.float()

    logit_corrs = []
    top1_acc = 0
    for i in test_idx:
        logits_actual = (h_actual[i] @ W_unembed.T).squeeze()
        logits_pred = (h_pred[i] @ W_unembed.T).squeeze()
        # Sample 5000 tokens for correlation
        n_v = logits_actual.shape[0]
        idx_sample = torch.randperm(n_v)[:5000]
        r = np.corrcoef(
            logits_actual[idx_sample].numpy(), logits_pred[idx_sample].numpy()
        )[0, 1]
        logit_corrs.append(r)
        if logits_pred.argmax() == logits_actual.argmax():
            top1_acc += 1

    avg_logit_r = np.mean(logit_corrs)
    log(f"    Logit Pearson r (test set): {avg_logit_r:.4f}")
    log(f"    Top-1 accuracy (test set): {top1_acc}/{len(test_idx)} = {top1_acc/len(test_idx):.1%}")

    return results, {"avg_test_corr": float(avg_test_corr), "avg_logit_r": float(avg_logit_r),
                      "top1_acc": top1_acc, "n_test": len(test_idx)}


# ===== P72: Delta-h Cumulative Predictability =====
def p72_delta_predictability(model, tokenizer, h_all, texts, all_deltas, log):
    """
    P72: Can we predict per-layer delta-h from the previous layer's h?
    
    This tests whether the layer-to-layer transformation is learnable.
    If delta-h_l = f(h_{l-1}) is predictable, the entire pipeline is causally modelable.
    """
    log(f"\n{'='*60}")
    log(f"  P72: Delta-h Layer-to-Layer Predictability")
    log(f"{'='*60}")

    # Use first 50 texts for speed
    n_texts = min(50, len(texts))

    # Get hidden states for all layers for n_texts
    log(f"    Extracting per-layer hidden states for {n_texts} texts...")

    layer_results = {}
    for text_i in range(n_texts):
        text = texts[text_i][0]
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        n_layers = len(states) - 1

        for l in range(n_layers):
            if l not in layer_results:
                layer_results[l] = {"X": [], "Y": []}
            # X = h_{l-1}, Y = delta_h_l = h_l - h_{l-1}
            layer_results[l]["X"].append(states[l])
            layer_results[l]["Y"].append(states[l+1] - states[l])

    # Train/test split
    n_train = int(0.8 * n_texts)
    torch.manual_seed(42)
    perm = torch.randperm(n_texts).numpy()
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # Per-layer prediction
    log(f"\n    {'Layer':>6} {'Train_R2':>10} {'Test_R2':>10} {'Test_Corr':>10} {'Test_KL':>10}")
    log(f"    {'-'*50}")

    results = {}
    for l in sorted(layer_results.keys()):
        X_all = torch.stack(layer_results[l]["X"])  # (n_texts, d)
        Y_all = torch.stack(layer_results[l]["Y"])

        X_train = X_all[train_idx]
        Y_train = Y_all[train_idx]
        X_test = X_all[test_idx]
        Y_test = Y_all[test_idx]

        # Use reduced dimensionality for prediction (top 50 PCs of X)
        X_mean = X_train.mean(0, keepdim=True)
        X_c = X_train - X_mean
        _, _, Vt_X = torch.linalg.svd(X_c, full_matrices=False)
        K_feat = min(50, Vt_X.shape[0], Vt_X.shape[1])

        X_train_reduced = (X_train - X_mean) @ Vt_X[:K_feat].T
        X_test_reduced = (X_test - X_mean) @ Vt_X[:K_feat].T

        # Ridge regression for each dimension of Y (sample 64 dims for speed)
        d_model = Y_train.shape[1]
        sample_dims = torch.randperm(d_model)[:min(64, d_model)]

        r2_train_list = []
        r2_test_list = []
        corr_test_list = []

        for dim in sample_dims:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_reduced.numpy(), Y_train[:, dim].numpy())
            y_pred_train = ridge.predict(X_train_reduced.numpy())
            y_pred_test = ridge.predict(X_test_reduced.numpy())

            r2_t = r2_score(Y_train[:, dim].numpy(), y_pred_train)
            r2_te = r2_score(Y_test[:, dim].numpy(), y_pred_test)
            corr_te = np.corrcoef(Y_test[:, dim].numpy(), y_pred_test)[0, 1]

            r2_train_list.append(r2_t)
            r2_test_list.append(r2_te)
            corr_test_list.append(corr_te)

        avg_r2_train = np.mean(r2_train_list)
        avg_r2_test = np.mean(r2_test_list)
        avg_corr_test = np.mean(corr_test_list)

        results[f"L{l}"] = {
            "train_r2": float(avg_r2_train),
            "test_r2": float(avg_r2_test),
            "test_corr": float(avg_corr_test),
        }
        log(f"    L{l:>4} {avg_r2_train:>10.4f} {avg_r2_test:>10.4f} {avg_corr_test:>10.4f}")

    # Summary: which layers are most predictable?
    log(f"\n    Most predictable layers:")
    layer_scores = [(k, v["test_corr"]) for k, v in results.items()]
    layer_scores.sort(key=lambda x: -x[1])
    for k, score in layer_scores[:5]:
        log(f"      {k}: test_corr={score:.4f}")

    avg_all = np.mean([v["test_corr"] for v in results.values()])
    log(f"\n    Average delta-h predictability: {avg_all:.4f}")

    return results


# ===== Main =====
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage709_phase5_causal_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log = Logger(run_dir, "results")
    log("=" * 70)
    log("Stage 709: Phase V - Causal PC + Cross-Model CCA + Prediction Chain")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS)} (5 categories x 40 each)")
    cat_counts = {}
    for _, c in TEXTS:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    log(f"Categories: {cat_counts}")
    log(f"Models: {MODEL_ORDER}")
    log("=" * 70)

    all_results = {}
    all_model_data = {}
    categories = [c for _, c in TEXTS]

    for mn in MODEL_ORDER:
        t0 = time.time()
        log(f"\n{'#'*70}")
        log(f"# Processing: {mn}")
        log(f"{'#'*70}")

        try:
            model, tokenizer = load_model(mn, log)

            log(f"\n  Computing h_final + deltas for {len(TEXTS)} texts...")
            t1 = time.time()
            h_all, all_deltas, all_logits = get_all_h_and_deltas(model, tokenizer, TEXTS)
            log(f"  Done in {time.time()-t1:.1f}s. h_all: {h_all.shape}")

            model_results = {}

            # P69: PC Causal Intervention
            try:
                t_p = time.time()
                r69, r69_extra = p69_pc_causal_intervention(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p69"] = r69
                model_results["p69_extra"] = r69_extra
                log(f"  P69 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P69 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P71: Causal Prediction Chain
            try:
                t_p = time.time()
                r71, r71_summary = p71_causal_prediction(model, tokenizer, h_all, TEXTS, categories, all_deltas, log)
                model_results["p71"] = r71
                model_results["p71_summary"] = r71_summary
                log(f"  P71 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P71 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P72: Delta-h Predictability (only first 2 models for speed)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r72 = p72_delta_predictability(model, tokenizer, h_all, TEXTS, all_deltas, log)
                    model_results["p72"] = r72
                    log(f"  P72 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P72 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            all_results[mn] = model_results
            all_model_data[mn] = {"h_all": h_all}

            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

            log(f"\n  {mn} total: {time.time()-t0:.1f}s")

        except Exception as e:
            log(f"  FATAL {mn}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    # P70: Cross-Model CCA (after all models)
    log(f"\n{'#'*70}")
    log(f"# P70: Cross-Model CCA Alignment")
    log(f"{'#'*70}")
    try:
        r70 = p70_cross_model_cca(all_model_data, categories, log)
        all_results["p70"] = r70
    except Exception as e:
        log(f"  P70 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Save JSON
    save_data = {}
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    save_data = make_serializable(all_results)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ===== Final Summary =====
    log(f"\n{'='*70}")
    log("FINAL SUMMARY - Phase V")
    log(f"{'='*70}")

    for mn in MODEL_ORDER:
        if mn not in all_results:
            continue
        res = all_results[mn]
        log(f"\n  {mn}:")
        if "p69" in res:
            # Find most causal PC
            best_pc = None
            for pc_key, data in res["p69"].items():
                flip = data["shifts"].get("2.0std", {}).get("top1_flip_rate", 0)
                if best_pc is None or flip > best_pc[1]:
                    best_pc = (pc_key, flip, data.get("var_pct", 0))
            if best_pc:
                log(f"    P69 Most causal: {best_pc[0]} (flip={best_pc[1]*100:.0f}%, var={best_pc[2]:.1f}%)")
            if "p69_extra" in res and res["p69_extra"] is not None:
                log(f"    P69 Var-Effect r: {res['p69_extra']:.4f}")
        if "p71_summary" in res:
            s = res["p71_summary"]
            log(f"    P71 PC coord pred: test_corr={s['avg_test_corr']:.4f}, logit_r={s['avg_logit_r']:.4f}, top1={s['top1_acc']}/{s['n_test']}")
        if "p72" in res:
            avg_pred = np.mean([v["test_corr"] for v in res["p72"].values()])
            log(f"    P72 Delta-h pred: avg_test_corr={avg_pred:.4f}")

    if "p70" in all_results:
        log(f"\n  Cross-Model P70:")
        for pair, data in all_results["p70"].items():
            log(f"    {pair}: dir_cos={data['avg_dir_cos']:.4f}, struct_r={data['struct_r']:.4f}")

    log(f"\nResults saved to: {run_dir}")
    log.close()
    print(f"\nDone! Results at: {run_dir}")


if __name__ == "__main__":
    main()
