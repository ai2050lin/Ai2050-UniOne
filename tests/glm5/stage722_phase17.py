#!/usr/bin/env python3
"""
Stage 722: Phase XVII — 精确梯度方向+操控子空间维度估计+架构对比+定性分析
================================================================================
Phase XVI发现:
- gradient⊥centroid (cos≈0.03), "表征空间"⊥"操控空间"
- PC0是最有效的操控方向(GLM4: 1-cos=0.056)
- 前半层>>后半层(ratio 3~28x)
- 所有cos_shift为负, 没有方向能引导文本靠近目标
- Qwen3完全免疫, DS7B对随机方向反常敏感

Phase XVII目标:
P126: 精确梯度方向计算 — 直接对logits计算grad, 不依赖hidden_states梯度
P127: 操控子空间维度估计 — 多次采样不同目标token集, PCA梯度方向矩阵
P128: 架构对比分析 — GLM4 vs Qwen3的RMSNorm/LayerNorm/attention差异
P129: 操控成功案例定性分析 — 大量生成+人工可读的文本对比
P130: 分类器梯度方向 — 训练线性分类器, 用其权重/梯度作为操控方向

用法: python stage722_phase17.py --model glm4
      python stage722_phase17.py --model qwen3
      python stage722_phase17.py --model deepseek7b
      python stage722_phase17.py --model all
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

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

log = None

# ===== Text dataset (expanded) =====
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
    for t in gen_en: T.append((t, "gen_en"))
    math_sci = [
        "Mathematical proof by induction.", "Einstein's theory of relativity changed physics.",
        "The quantum computer solved the problem in seconds.", "DNA carries genetic information in all living organisms.",
        "Photosynthesis converts sunlight into chemical energy.", "Evolution by natural selection explains biodiversity.",
        "The Pythagorean theorem states a squared plus b squared equals c squared.",
        "The derivative of velocity is acceleration.", "Newton's third law states every action has an equal opposite reaction.",
        "Entropy always increases in isolated systems.", "The speed of light is approximately 299792 km per second.",
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
    for t in math_sci: T.append((t, "math_sci"))
    code = [
        "for i in range(10): print(i)", "def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)",
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
    for t in code: T.append((t, "code"))
    chinese = [
        "The capital of China is Beijing.", "Chinese cuisine is famous worldwide.",
        "The Great Wall stretches across northern China.", "Mandarin is spoken by over a billion people.",
        "China has a rich history spanning thousands of years.", "The Silk Road connected China to the West.",
        "Chinese calligraphy is considered one of the highest art forms.",
        "The Yangtze River is the longest in Asia.", "Confucius was one of China's greatest philosophers.",
        "The Terracotta Army was built to protect the first emperor.",
        "Chinese New Year is the most important traditional holiday.",
        "The Forbidden City was the imperial palace for centuries.",
        "Tai Chi is a traditional Chinese martial art.", "Dim sum is a style of Cantonese cuisine.",
        "The panda is a symbol of wildlife conservation in China.",
        "Shanghai is one of the world's largest cities by population.",
        "The Ming Dynasty built many of China's most famous landmarks.",
        "Tea cultivation originated in ancient China.",
        "Chinese opera combines music, dance, and drama.",
        "The Dragon Boat Festival celebrates the poet Qu Yuan.",
        "Acupuncture is a traditional Chinese medical practice.",
        "The Forbidden City has nine thousand nine hundred and ninety nine rooms.",
        "Chinese painting emphasizes the beauty of nature.",
        "The Summer Palace is a masterpiece of Chinese landscape design.",
        "The Spring Festival marks the beginning of the lunar new year.",
        "The Three Gorges Dam is the world's largest power station by installed capacity.",
        "The Mid-Autumn Festival celebrates family reunion under the full moon.",
        "The Chinese writing system uses thousands of unique characters.",
        "Beijing duck is one of the most famous dishes in Chinese cuisine.",
        "The invention of paper money originated in China during the Tang Dynasty.",
        "The Yellow River is known as the cradle of Chinese civilization.",
        "The Chinese zodiac cycles through twelve animal signs each year.",
        "Chinese traditional medicine uses herbs and acupuncture to treat illness.",
        "The Summer Olympics were held in Beijing in 2008.",
        "Gunpowder was one of the four great inventions of ancient China.",
        "The Li River in Guilin is famous for its karst landscape scenery.",
        "Confucianism has profoundly influenced Chinese society for over two millennia.",
    ]
    for t in chinese: T.append((t, "chinese"))
    poetry = [
        "Shall I compare thee to a summer's day?", "Two roads diverged in a wood, and I took the one less traveled by.",
        "I wandered lonely as a cloud that floats on high.", "To be or not to be, that is the question.",
        "The road goes ever on and on.", "Hope is the thing with feathers that perches in the soul.",
        "In the middle of difficulty lies opportunity.", "Do not go gentle into that good night.",
        "Because I could not stop for Death, he kindly stopped for me.",
        "I think that I shall never see a poem lovely as a tree.",
        "A thing of beauty is a joy forever.", "Water, water, everywhere, nor any drop to drink.",
        "The fog comes on little cat feet.", "I celebrate myself, and sing myself.",
        "Stopping by woods on a snowy evening.", "April is the cruellest month.",
        "Beauty is truth, truth beauty.", "O wild West Wind, thou breath of Autumn's being.",
        "The world is too much with us, late and soon.",
        "If winter comes, can spring be far behind?", "How do I love thee? Let me count the ways.",
        "The love song of J. Alfred Prufrock measured out his life with coffee spoons.",
        "I have measured out my life with coffee spoons.", "Out of the ash I rise with my red hair.",
        "The falcon cannot hear the falconer.", "Things fall apart, the center cannot hold.",
        "This is the way the world ends, not with a bang but a whimper.",
        "I contain multitudes.", "The only people for me are the mad ones.",
        "In Xanadu did Kubla Khan a stately pleasure dome decree.",
        "A host of golden daffodils beside the lake beneath the trees.",
        "Tyger Tyger burning bright in the forests of the night.",
        "My heart aches, and a drowsy numbness pains my sense.",
        "Much have I travell'd in the realms of gold.",
        "On First Looking into Chapman's Homer I felt like some watcher of the skies.",
        "She walks in beauty like the night of cloudless climes and starry skies.",
        "The mind is its own place, and in itself can make a heaven of hell.",
        "Sylvia Plath explored themes of mental illness and identity.",
        "Seamus Heaney won the Nobel Prize in Literature for his poetry.",
    ]
    for t in poetry: T.append((t, "poetry"))
    philosophy = [
        "The unexamined life is not worth living.", "I think therefore I am.",
        "To be is to be perceived.", "Existence precedes essence.",
        "Happiness depends upon ourselves.", "Man is by nature a social animal.",
        "Know thyself.", "The mind is everything. What you think you become.",
        "We are what we repeatedly do. Excellence is not an act but a habit.",
        "It is the mark of an educated mind to entertain a thought without accepting it.",
        "No man's knowledge here can go beyond his experience.",
        "The greatest happiness of the greatest number is the foundation of morals.",
        "We live in the best of all possible worlds.", "Existence precedes essence.",
        "Freedom is what we do with what is done to us.",
        "The death of God is the death of absolute morality.", "Truth is subjectivity.",
        "Man is condemned to be free.", "Hell is other people.",
        "One cannot step twice in the same river.", "The only constant is change.",
        "All men by nature desire knowledge.", "Plato's cave allegory shows the limits of perception.",
        "Kant's categorical imperative demands universal moral laws.",
        "Utilitarianism seeks the greatest good for the greatest number.",
        "Nihilism argues that life has no inherent meaning or value.",
        "Stoicism teaches acceptance of things we cannot change.",
        "Phenomenology studies the structures of conscious experience.",
        "Pragmatism judges truth by its practical consequences.",
        "Absurdism recognizes the conflict between human desire for meaning and the universe's silence.",
        "Determinism holds that all events are caused by previous events.",
        "Free will debates whether our choices are truly our own.",
        "Moral relativism argues that moral judgments are culture dependent.",
        "Epistemology questions the nature and limits of human knowledge.",
        "Metaphysics explores the fundamental nature of reality.",
        "Ethics examines what constitutes a good life and right action.",
        "Political philosophy asks what constitutes a just society.",
        "Aesthetics studies the nature of beauty and art.",
        "The social contract theory explains why people form governments.",
        "Virtue ethics focuses on developing good character traits.",
        "Consequentialism evaluates actions based on their outcomes.",
        "Deontology holds that some actions are inherently right or wrong.",
        "Empiricism claims knowledge comes from sensory experience.",
        "Rationalism argues that reason is the primary source of knowledge.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
    return T


# ===== Model Config =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

MODEL_ORDER = [m for m in MODEL_MAP if m in MODEL_MAP]


def load_model(mname):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[mname]
    log(f"  Loading {mname} from {p.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', 2048)
    log(f"  {mname}: n_layers={n_layers}, d_model={d_model}, device={model.device}")
    return model, tokenizer, n_layers, d_model


def get_unembed(model):
    if hasattr(model, 'lm_head'):
        um = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        um = model.get_output_embeddings()
    else:
        return None, None
    w = um.weight.detach().to(torch.float32)
    b = um.bias.detach().to(torch.float32) if (hasattr(um, 'bias') and um.bias is not None) else None
    return w, b


def get_final_h(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
    return h, inputs


def compute_perplexity(model, tokenizer, text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    avg_loss = outputs.loss.item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return ppl, avg_loss


def compute_centroids(model, tokenizer, texts, n_samples=40):
    cat_h = defaultdict(list)
    indices = list(range(0, len(texts), max(1, len(texts) // n_samples)))[:n_samples]
    for idx in indices:
        h, _ = get_final_h(model, tokenizer, texts[idx][0])
        cat_h[texts[idx][1]].append(h)
    centroids = {}
    for cat, hs in cat_h.items():
        centroids[cat] = torch.stack(hs).mean(dim=0)
    return centroids


def kl_divergence(p, q, eps=1e-10):
    p = F.softmax(p.float(), dim=-1) + eps
    q = F.softmax(q.float(), dim=-1) + eps
    return (p * (p / q).log()).sum().item()


def get_layers_container(model):
    for attr in ['model', 'transformer', 'language_model']:
        container = getattr(model, attr, None)
        if container is not None:
            return container
    return model


def get_actual_layers(model):
    container = get_layers_container(model)
    for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
        layers = getattr(container, attr, None)
        if layers is not None:
            return layers
    return None


model_device_global = None
actual_layers_global = None


def make_inject_hook(direction, scale):
    d = direction.to(model_device_global, dtype=torch.bfloat16)
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + d * scale,) + output[1:]
        return output + d * scale
    return hook


# ===== P126: 精确梯度方向计算 =====
def p126_precise_gradient(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    Phase XVI的P121失败原因: h.grad不传播
    修复: 直接用 d(log_softmax(Wx+b)) / dx = W^T @ (softmax - one_hot) 计算
    这是精确的解析梯度, 不需要backward
    """
    log(f"\n{'='*70}")
    log(f"P126: Precise Gradient Direction (Analytical)")
    log(f"{'='*70}")
    
    uw, ub = get_unembed(model)
    if uw is None:
        log("  WARNING: Cannot get unembed weights, skipping P126")
        return {}
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    cat_pairs = [("code", "gen_en"), ("math_sci", "poetry"), ("chinese", "philosophy")]
    
    # Get target tokens per category
    log("  Computing target tokens per category...")
    cat_target_tokens = {}
    for cat in categories:
        if cat not in centroids:
            continue
        centroid = centroids[cat].to(uw.device)
        cos_sims = F.cosine_similarity(centroid.unsqueeze(0), uw, dim=1)
        topk_indices = cos_sims.topk(min(200, uw.size(0))).indices
        cat_target_tokens[cat] = topk_indices
    
    results = {}
    inj_layer_indices = list(range(min(10, n_layers)))
    
    for src_cat, tgt_cat in cat_pairs:
        log(f"\n  --- {src_cat} -> {tgt_cat} ---")
        if tgt_cat not in cat_target_tokens:
            continue
        
        src_texts = [(t, c) for t, c in texts if c == src_cat][:15]
        target_token_ids = cat_target_tokens[tgt_cat]
        
        grad_directions = []
        centroid_diff = centroids[tgt_cat] - centroids[src_cat]
        centroid_diff_norm = centroid_diff / centroid_diff.norm()
        
        for i, (text, cat) in enumerate(src_texts[:12]):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()  # [1, d_model] on CUDA
            
            # Analytical gradient: d(mean_log_softmax_target) / d(h)
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            probs = F.softmax(logits, dim=-1)  # [1, vocab]
            
            # grad = W^T @ (mean_target_onehot - probs)
            W_t = uw.T.to(h.device)  # [d_model, vocab]
            
            mean_target_vec = torch.zeros(uw.size(0), device=h.device)
            mean_target_vec[target_token_ids] = 1.0 / len(target_token_ids)
            
            grad = W_t @ (mean_target_vec - probs.squeeze(0))  # [d_model] on CUDA
            grad_norm = grad.norm().item()
            
            if grad_norm > 1e-8:
                grad_normalized = (grad / grad_norm).cpu()
                grad_directions.append(grad_normalized)
                
                # Also compute gradient for source category (should be opposite)
                src_target_ids = cat_target_tokens.get(src_cat, target_token_ids)
                mean_src_vec = torch.zeros(uw.size(0), device=h.device)
                mean_src_vec[src_target_ids] = 1.0 / len(src_target_ids)
                grad_src = (W_t @ (mean_src_vec - probs.squeeze(0))).cpu()
                grad_src_norm = grad_src / max(grad_src.norm().item(), 1e-8)
                
                if i < 5:
                    cos_gc = F.cosine_similarity(grad_normalized.unsqueeze(0), centroid_diff_norm.unsqueeze(0)).item()
                    cos_gs = F.cosine_similarity(grad_normalized.unsqueeze(0), grad_src_norm.unsqueeze(0)).item()
                    log(f"    text {i}: grad_norm={grad_norm:.4f}, "
                        f"cos(grad, centroid)={cos_gc:.4f}, "
                        f"cos(grad_tgt, grad_src)={cos_gs:.4f}")
        
        if not grad_directions:
            continue
        
        avg_grad = torch.stack(grad_directions).mean(dim=0)
        avg_grad = avg_grad / avg_grad.norm()
        
        cos_avg_centroid = F.cosine_similarity(avg_grad.unsqueeze(0), centroid_diff_norm.unsqueeze(0)).item()
        log(f"  avg_grad vs centroid_diff: cos={cos_avg_centroid:.4f}")
        log(f"  Number of gradient samples: {len(grad_directions)}")
        
        # Gradient direction variance (how consistent are gradients?)
        grad_matrix = torch.stack(grad_directions)  # [N, d_model]
        grad_std = grad_matrix.std(dim=0).mean().item()
        grad_mean_norm = grad_matrix.mean(dim=0).norm().item()
        log(f"  Gradient consistency: mean_norm={grad_mean_norm:.4f}, avg_std={grad_std:.4f}, ratio={grad_std/grad_mean_norm:.4f}")
        
        # Test gradient direction as injection
        test_texts = [(t, c) for t, c in texts if c == src_cat][:10]
        
        # Compare: gradient direction vs centroid direction vs random
        rand_dir = torch.randn(d_model)
        rand_dir = rand_dir / rand_dir.norm()
        
        for dir_name, direction in [("gradient", avg_grad), ("centroid", centroid_diff_norm), ("random", rand_dir)]:
            one_cos_vals = []
            kl_vals = []
            rank_improves = []
            text_changes = []
            
            for text, _ in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                nat_logits = nat_out.logits[:, -1, :]
                
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(direction, 0.10))
                    handles.append(hh)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                finally:
                    for hh in handles:
                        hh.remove()
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                inj_logits = inj_out.logits[:, -1, :]
                
                one_cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
                kl_vals.append(kl_divergence(nat_logits, inj_logits))
                
                nat_cos_tgt = F.cosine_similarity(nat_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                inj_cos_tgt = F.cosine_similarity(inj_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos_tgt > nat_cos_tgt else 0)
                
                nat_top5 = nat_logits[0].topk(5).indices.tolist()
                inj_top5 = inj_logits[0].topk(5).indices.tolist()
                text_changes.append(1 if nat_top5 != inj_top5 else 0)
            
            avg_1cos = np.mean(one_cos_vals)
            avg_kl = np.mean(kl_vals)
            rank_rate = np.mean(rank_improves) * 100
            tc_rate = np.mean(text_changes) * 100
            
            log(f"  [{dir_name}] 1-cos={avg_1cos:.6f}, KL={avg_kl:.4f}, "
                f"rank_improve={rank_rate:.0f}%, top5_change={tc_rate:.0f}%")
            results[f"{src_cat}->{tgt_cat}_{dir_name}"] = {
                "1cos": avg_1cos, "kl": avg_kl, "rank_improve": rank_rate,
                "top5_change": tc_rate, "cos_avg_centroid": cos_avg_centroid,
                "grad_consistency": grad_std/grad_mean_norm
            }
        
        # Scale sweep for gradient direction
        log(f"  --- Gradient Scale Sweep ---")
        scales = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
        for scale in scales:
            one_cos_vals = []
            rank_improves = []
            for text, _ in test_texts[:8]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(avg_grad, scale))
                    handles.append(hh)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                finally:
                    for hh in handles:
                        hh.remove()
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                one_cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
                nat_cos_tgt = F.cosine_similarity(nat_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                inj_cos_tgt = F.cosine_similarity(inj_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos_tgt > nat_cos_tgt else 0)
            log(f"  scale={scale:.2f}: 1-cos={np.mean(one_cos_vals):.6f}, "
                f"rank_improve={np.mean(rank_improves)*100:.0f}%")
            results[f"grad_scale_{src_cat}_{tgt_cat}_s{scale}"] = {
                "1cos": np.mean(one_cos_vals), "rank_improve": np.mean(rank_improves)*100
            }
    
    return results


# ===== P127: 操控子空间维度估计 =====
def p127_manipulation_subspace(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    多次采样不同目标token集, 收集梯度方向, PCA分析梯度方向的子空间维度
    """
    log(f"\n{'='*70}")
    log(f"P127: Manipulation Subspace Dimension Estimation")
    log(f"{'='*70}")
    
    uw, ub = get_unembed(model)
    if uw is None:
        log("  WARNING: Cannot get unembed weights, skipping P127")
        return {}
    
    W_t = uw.T  # [d_model, vocab] on CPU, will move to device per text
    
    # Collect gradient directions for many different objectives
    all_grads = []
    grad_labels = []
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    
    log("  Computing gradients for multiple objectives...")
    
    # For each category, compute gradient direction from 10 different texts
    for cat in categories:
        if cat not in centroids:
            continue
        centroid = centroids[cat].to(uw.device)
        cos_sims = F.cosine_similarity(centroid.unsqueeze(0), uw, dim=1)
        topk_ids = cos_sims.topk(min(200, uw.size(0))).indices
        
        cat_texts = [(t, c) for t, c in texts if c == cat][:10]
        
        for text, _ in cat_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            probs = F.softmax(logits, dim=-1)
            
            W_t_dev = uw.T.to(h.device)
            mean_target = torch.zeros(uw.size(0), device=h.device)
            mean_target[topk_ids] = 1.0 / len(topk_ids)
            grad = (W_t_dev @ (mean_target - probs.squeeze(0))).cpu()
            grad_norm = grad.norm().item()
            if grad_norm > 1e-8:
                all_grads.append(grad / grad_norm)
                grad_labels.append(cat)
    
    # Also compute "anti-gradient" directions (gradient for minimizing target)
    log("  Computing anti-gradient directions...")
    for cat in categories[:3]:
        if cat not in centroids:
            continue
        centroid = centroids[cat].to(uw.device)
        cos_sims = F.cosine_similarity(centroid.unsqueeze(0), uw, dim=1)
        # Bottom-k tokens (least aligned with category)
        botk_ids = cos_sims.topk(min(200, uw.size(0)), largest=False).indices
        
        cat_texts = [(t, c) for t, c in texts if c == cat][:10]
        for text, _ in cat_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            probs = F.softmax(logits, dim=-1)
            
            W_t_dev = uw.T.to(h.device)
            mean_target = torch.zeros(uw.size(0), device=h.device)
            mean_target[botk_ids] = 1.0 / len(botk_ids)
            grad = (W_t_dev @ (mean_target - probs.squeeze(0))).cpu()
            grad_norm = grad.norm().item()
            if grad_norm > 1e-8:
                all_grads.append(grad / grad_norm)
                grad_labels.append(f"anti_{cat}")
    
    # Also add entropy gradient (maximize entropy = uniform distribution)
    log("  Computing entropy gradient directions...")
    entropy_texts = texts[::max(1, len(texts)//20)][:20]
    for text, _ in entropy_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][:, -1, :].float()
        logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
        probs = F.softmax(logits, dim=-1)
        # grad of entropy = -W^T @ (probs - uniform) = W^T @ (uniform - probs)
        uniform = torch.ones(uw.size(0), device=h.device) / uw.size(0)
        W_t_dev = uw.T.to(h.device)
        grad = (W_t_dev @ (uniform - probs.squeeze(0))).cpu()
        grad_norm = grad.norm().item()
        if grad_norm > 1e-8:
            all_grads.append(grad / grad_norm)
            grad_labels.append("entropy")
    
    # PCA of gradient directions
    grad_matrix = torch.stack(all_grads)  # [N, d_model]
    log(f"  Total gradient samples: {grad_matrix.shape[0]}")
    
    grad_centered = grad_matrix - grad_matrix.mean(dim=0)
    U, S, Vt = torch.linalg.svd(grad_centered, full_matrices=False)
    
    total_var = (S**2).sum().item()
    log(f"\n  Gradient subspace PCA (top-15):")
    cum_var = 0
    effective_dim = 0
    for i in range(min(15, len(S))):
        var_ratio = (S[i]**2).item() / total_var
        cum_var += var_ratio
        if cum_var < 0.90 and cum_var - var_ratio < 0.90:
            effective_dim = i + 1
        if i < 10:
            log(f"    PC{i}: var={var_ratio:.4f}, cum={cum_var:.4f}, singular={S[i].item():.4f}")
    
    # Find dimensions needed for 50%, 80%, 90%, 95% variance
    for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
        cum = 0
        for i in range(len(S)):
            cum += (S[i]**2).item() / total_var
            if cum >= threshold:
                log(f"  Dimensions for {int(threshold*100)}% variance: {i+1}")
                break
    
    # Test: how many gradient PCs are orthogonal to centroid directions?
    log(f"\n  Gradient PCs vs Centroid directions:")
    for cat1 in categories[:3]:
        for cat2 in categories[3:6]:
            if cat1 not in centroids or cat2 not in centroids:
                continue
            cent_diff = centroids[cat2] - centroids[cat1]
            cent_diff = cent_diff / cent_diff.norm()
            cos_vals = []
            for i in range(min(8, Vt.size(0))):
                c = F.cosine_similarity(Vt[i].unsqueeze(0), cent_diff.unsqueeze(0)).item()
                cos_vals.append(abs(c))
            log(f"    {cat1}-{cat2}: max|cos(PC, cent)|={max(cos_vals):.4f}, "
                f"mean={np.mean(cos_vals):.4f}")
    
    # Test: inject top gradient PCs vs centroid direction
    log(f"\n  --- Gradient PC Injection vs Centroid ---")
    inj_layer_indices = list(range(min(10, n_layers)))
    test_texts = texts[:15]
    
    # Top gradient PCs
    for pc_idx in range(min(5, Vt.size(0))):
        pc_dir = Vt[pc_idx].float()
        pc_dir = pc_dir / pc_dir.norm()
        cos_vals = []
        for text, _ in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in inj_layer_indices:
                hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(pc_dir, 0.10))
                handles.append(hh)
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                for hh in handles:
                    hh.remove()
            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        log(f"  grad_PC{pc_idx}: 1-cos={np.mean(cos_vals):.6f}")
    
    results = {
        "n_grad_samples": grad_matrix.shape[0],
        "top10_var_ratios": [(S[i]**2).item()/total_var for i in range(min(10, len(S)))],
        "effective_dims": effective_dim
    }
    return results


# ===== P128: 架构对比分析 =====
def p128_architecture_analysis(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    分析模型架构对操控响应的影响:
    1. 每层的RMSNorm/LayerNorm缩放因子
    2. 每层的输出范数分布
    3. 注入在不同位置的效果(residual stream vs attention output vs FFN output)
    """
    log(f"\n{'='*70}")
    log(f"P128: Architecture Analysis")
    log(f"{'='*70}")
    
    actual_layers = actual_layers_global
    
    # 1. Layer output norm analysis
    log("  Layer output norm distribution:")
    sample_text = texts[0][0]
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    layer_norms = []
    for i, hs in enumerate(outputs.hidden_states):
        norm = hs[:, -1, :].norm().item()
        layer_norms.append(norm)
        if i < 5 or i >= len(outputs.hidden_states) - 3:
            log(f"    Layer {i}: norm={norm:.2f}")
    
    log(f"  First layer norm: {layer_norms[0]:.2f}")
    log(f"  Last layer norm: {layer_norms[-1]:.2f}")
    log(f"  Norm ratio (last/first): {layer_norms[-1]/max(layer_norms[0], 1):.2f}")
    
    # 2. Norm scaling factor per layer
    log(f"\n  Inter-layer norm ratios:")
    for i in range(1, min(len(layer_norms), 11)):
        ratio = layer_norms[i] / max(layer_norms[i-1], 1)
        log(f"    L{i-1}->L{i}: ratio={ratio:.4f}")
    
    # 3. Injection amplification per layer (same direction, different layers)
    log(f"\n  Injection effect per individual layer:")
    direction = centroids.get("code", centroids[list(centroids.keys())[0]]) - centroids.get("gen_en", centroids[list(centroids.keys())[1]])
    direction = direction / direction.norm()
    
    per_layer_effects = []
    for layer_idx in range(min(n_layers, 20)):
        cos_vals = []
        test_batch = texts[:5]
        for text, _ in test_batch:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            
            hh = actual_layers[layer_idx].register_forward_hook(make_inject_hook(direction, 0.10))
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                hh.remove()
            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        
        avg_effect = np.mean(cos_vals)
        per_layer_effects.append(avg_effect)
        if layer_idx < 10 or layer_idx % 5 == 0:
            log(f"    Layer {layer_idx}: 1-cos={avg_effect:.6f}")
    
    # 4. Cumulative injection (add layers one by one)
    log(f"\n  Cumulative injection (adding layers):")
    cum_results = []
    for n_inject in [1, 2, 3, 5, 10, 15, 20]:
        if n_inject > n_layers:
            break
        cos_vals = []
        for text, _ in texts[:5]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in range(n_inject):
                hh = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, 0.10))
                handles.append(hh)
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                for hh in handles:
                    hh.remove()
            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        
        avg_effect = np.mean(cos_vals)
        cum_results.append((n_inject, avg_effect))
        log(f"    {n_inject} layers: 1-cos={avg_effect:.6f}")
    
    # 5. Injection at last layer only (just before unembed)
    log(f"\n  Last-layer-only injection:")
    for layer_idx in [n_layers-1, n_layers-2, n_layers-3, n_layers-5]:
        cos_vals = []
        for text, _ in texts[:5]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            hh = actual_layers[layer_idx].register_forward_hook(make_inject_hook(direction, 0.10))
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                hh.remove()
            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        log(f"    Layer {layer_idx}: 1-cos={np.mean(cos_vals):.6f}")
    
    # 6. Direct unembed space manipulation (bypass all layers)
    log(f"\n  Direct logits manipulation (bypass layers):")
    uw_local, _ = get_unembed(model)
    if uw_local is not None:
        for text, _ in texts[:3]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs)
            nat_logits = nat_out.logits[:, -1, :].float()
            
            # Add direction to logits directly
            dir_on_dev = direction.to(nat_logits.device)
            uw_on_dev = uw_local.to(nat_logits.device)
            direction_logits = dir_on_dev @ uw_on_dev.T  # project to vocab space
            direction_logits = direction_logits / max(direction_logits.norm().item(), 1e-8) * 0.5
            inj_logits = nat_logits + direction_logits.unsqueeze(0)
            
            nat_top5 = tokenizer.convert_ids_to_tokens(nat_logits[0].topk(5).indices.tolist())
            inj_top5 = tokenizer.convert_ids_to_tokens(inj_logits[0].topk(5).indices.tolist())
            
            log(f"    text: {text[:50]}...")
            log(f"    nat_top5: {nat_top5}")
            log(f"    inj_top5: {inj_top5}")
    
    results = {
        "layer_norms": layer_norms[:20],
        "norm_ratio_last_first": layer_norms[-1]/max(layer_norms[0], 1),
        "per_layer_effects": per_layer_effects[:20],
        "cumulative": [(n, e) for n, e in cum_results]
    }
    return results


# ===== P129: 操控成功案例定性分析 =====
def p129_qualitative_analysis(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    大量生成, 找到操控成功改变语义的案例, 详细分析
    """
    log(f"\n{'='*70}")
    log(f"P129: Qualitative Analysis of Successful Manipulation")
    log(f"{'='*70}")
    
    inj_layer_indices = list(range(min(10, n_layers)))
    
    # Use gradient direction and centroid direction
    uw, ub = get_unembed(model)
    centroid_diff = centroids["gen_en"] - centroids["code"]
    centroid_diff_norm = centroid_diff / centroid_diff.norm()
    
    # Compute gradient direction for gen_en
    if uw is not None:
        W_t = uw.T
        centroid_gen = centroids["gen_en"].to(uw.device)
        cos_sims = F.cosine_similarity(centroid_gen.unsqueeze(0), uw, dim=1)
        topk_ids = cos_sims.topk(min(200, uw.size(0))).indices
        
        code_texts = [(t, c) for t, c in texts if c == "code"][:3]
        grad_directions = []
        for text, _ in code_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            probs = F.softmax(logits, dim=-1)
            W_t_dev = uw.T.to(h.device)
            mean_target = torch.zeros(uw.size(0), device=h.device)
            mean_target[topk_ids] = 1.0 / len(topk_ids)
            grad = (W_t_dev @ (mean_target - probs.squeeze(0))).cpu()
            if grad.norm().item() > 1e-8:
                grad_directions.append(grad / grad.norm())
        if grad_directions:
            avg_grad = torch.stack(grad_directions).mean(dim=0)
            avg_grad = avg_grad / avg_grad.norm()
        else:
            avg_grad = centroid_diff_norm
    else:
        avg_grad = centroid_diff_norm
    
    directions = {"centroid": centroid_diff_norm, "gradient": avg_grad}
    
    # Generate with each direction, more tokens
    results = {}
    src_texts = [(t, c) for t, c in texts if c == "code"][:5]
    n_gen = 20
    scales = [0.08, 0.15, 0.25]
    
    for dir_name, direction in directions.items():
        for scale in scales:
            log(f"\n  --- {dir_name} direction, scale={scale}, {n_gen} tokens ---")
            
            for text_idx, (text, _) in enumerate(src_texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                
                # Natural generation
                nat_ids = inputs["input_ids"].clone()
                for _ in range(n_gen):
                    with torch.no_grad():
                        out = model(input_ids=nat_ids)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if tok.item() == tokenizer.eos_token_id:
                        break
                    nat_ids = torch.cat([nat_ids, tok], dim=1)
                
                # Injected generation
                inj_ids = inputs["input_ids"].clone()
                for _ in range(n_gen):
                    handles = []
                    for lidx in inj_layer_indices:
                        hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(direction, scale))
                        handles.append(hh)
                    try:
                        with torch.no_grad():
                            out = model(input_ids=inj_ids)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    finally:
                        for hh in handles:
                            hh.remove()
                    if tok.item() == tokenizer.eos_token_id:
                        break
                    inj_ids = torch.cat([inj_ids, tok], dim=1)
                
                nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
                inj_text = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
                
                text_diff = 1 if nat_text.strip() != inj_text.strip() else 0
                
                # Analyze generated tokens
                n_new_nat = nat_ids.shape[1] - inputs["input_ids"].shape[1]
                n_new_inj = inj_ids.shape[1] - inputs["input_ids"].shape[1]
                
                # Count how many of the new tokens differ
                min_len = min(nat_ids.shape[1], inj_ids.shape[1])
                n_diff_tokens = (nat_ids[0, :min_len] != inj_ids[0, :min_len]).sum().item()
                
                if text_idx < 2:  # Show first 2 examples
                    log(f"\n    [{dir_name}/s{scale}] Example {text_idx+1}:")
                    log(f"    NAT ({n_new_nat} tokens): {nat_text[:200]}")
                    log(f"    INJ ({n_new_inj} tokens): {inj_text[:200]}")
                    log(f"    diff_tokens={n_diff_tokens}/{min_len}, text_changed={text_diff}")
                    
                    try:
                        nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                        inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
                        log(f"    PPL: nat={nat_ppl:.1f}, inj={inj_ppl:.1f}, dPPL={inj_ppl-nat_ppl:+.1f}")
                    except:
                        pass
                
                results[f"{dir_name}_s{scale}_ex{text_idx}"] = {
                    "n_diff_tokens": n_diff_tokens, "min_len": min_len,
                    "text_changed": text_diff,
                    "n_new_nat": n_new_nat, "n_new_inj": n_new_inj
                }
    
    return results


# ===== P130: 分类器梯度方向 =====
def p130_classifier_gradient(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    训练一个简单的线性分类器, 用分类器的法向量作为操控方向
    """
    log(f"\n{'='*70}")
    log(f"P130: Linear Classifier Weight as Manipulation Direction")
    log(f"{'='*70}")
    
    # Collect hidden states and labels
    log("  Collecting hidden states for classifier training...")
    X = []
    y = []
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    
    sample_indices = list(range(0, len(texts), max(1, len(texts)//80)))[:80]
    for idx in sample_indices:
        text, cat = texts[idx]
        if cat not in cat_to_id:
            continue
        h, _ = get_final_h(model, tokenizer, text)
        X.append(h)
        y.append(cat_to_id[cat])
    
    X = torch.stack(X)  # [N, d_model]
    y = torch.tensor(y)
    log(f"  X shape: {X.shape}, classes: {len(cat_to_id)}")
    
    # Train linear classifier with gradient descent
    n_classes = len(cat_to_id)
    W_cls = torch.randn(d_model, n_classes, requires_grad=True) * 0.01
    b_cls = torch.zeros(n_classes, requires_grad=True)
    
    lr = 0.1
    for epoch in range(200):
        logits = X @ W_cls + b_cls
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            if W_cls.grad is not None:
                W_cls -= lr * W_cls.grad
            if b_cls.grad is not None:
                b_cls -= lr * b_cls.grad
        W_cls.grad.zero_()
        b_cls.grad.zero_()
        if epoch % 50 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            log(f"    epoch {epoch}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    # Final accuracy
    with torch.no_grad():
        logits = X @ W_cls + b_cls
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    log(f"  Final accuracy: {acc:.4f}")
    
    # Detach weights for use as directions
    W_cls_det = W_cls.detach()
    
    # Use classifier weight columns as manipulation directions
    log(f"\n  Testing classifier weight directions:")
    inj_layer_indices = list(range(min(10, n_layers)))
    
    # Test code vs gen_en: use (W_gen_en - W_code) as direction
    if "code" in cat_to_id and "gen_en" in cat_to_id:
        cls_direction = W_cls_det[:, cat_to_id["gen_en"]] - W_cls_det[:, cat_to_id["code"]]
        cls_norm = cls_direction.norm().item()
        if cls_norm > 1e-8:
            cls_direction = cls_direction / cls_norm
        else:
            cls_direction = torch.randn(d_model)
            cls_direction = cls_direction / cls_direction.norm()
        
        centroid_dir = centroids["gen_en"] - centroids["code"]
        centroid_dir = centroid_dir / centroid_dir.norm()
        
        cos_cls_cent = F.cosine_similarity(cls_direction.unsqueeze(0), centroid_dir.unsqueeze(0)).item()
        log(f"  cos(classifier_dir, centroid_dir)={cos_cls_cent:.4f}")
        
        for dir_name, direction in [("classifier", cls_direction), ("centroid", centroid_dir)]:
            one_cos_vals = []
            kl_vals = []
            rank_improves = []
            
            test_texts = [(t, c) for t, c in texts if c == "code"][:10]
            for text, _ in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(direction, 0.10))
                    handles.append(hh)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                finally:
                    for hh in handles:
                        hh.remove()
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                one_cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
                kl_vals.append(kl_divergence(nat_out.logits[:, -1, :], inj_out.logits[:, -1, :]))
                nat_cos = F.cosine_similarity(nat_h.cpu(), centroids["gen_en"].unsqueeze(0)).item()
                inj_cos = F.cosine_similarity(inj_h.cpu(), centroids["gen_en"].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos > nat_cos else 0)
            
            log(f"  [{dir_name}] 1-cos={np.mean(one_cos_vals):.6f}, KL={np.mean(kl_vals):.4f}, "
                f"rank_improve={np.mean(rank_improves)*100:.0f}%")
    
    # All pair directions from classifier
    log(f"\n  All-pair classifier directions:")
    pair_results = {}
    for cat1 in ["code", "math_sci", "chinese"]:
        for cat2 in ["gen_en", "poetry", "philosophy"]:
            if cat1 not in cat_to_id or cat2 not in cat_to_id:
                continue
            dir_vec = W_cls_det[:, cat_to_id[cat2]] - W_cls_det[:, cat_to_id[cat1]]
            dir_norm = dir_vec.norm().item()
            if dir_norm < 1e-8:
                continue
            dir_vec = dir_vec / dir_norm
            
            cent_vec = centroids[cat2] - centroids[cat1]
            cent_vec = cent_vec / cent_vec.norm()
            
            cos_cc = F.cosine_similarity(dir_vec.unsqueeze(0), cent_vec.unsqueeze(0)).item()
            
            test_src = [(t, c) for t, c in texts if c == cat1][:8]
            rank_improves = []
            for text, _ in test_src:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(dir_vec, 0.10))
                    handles.append(hh)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                finally:
                    for hh in handles:
                        hh.remove()
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                nat_cos = F.cosine_similarity(nat_h.cpu(), centroids[cat2].unsqueeze(0)).item()
                inj_cos = F.cosine_similarity(inj_h.cpu(), centroids[cat2].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos > nat_cos else 0)
            
            log(f"  {cat1}->{cat2}: cos(cls,cent)={cos_cc:.4f}, rank_improve={np.mean(rank_improves)*100:.0f}%")
            pair_results[f"{cat1}_{cat2}"] = {"cos": cos_cc, "rank_improve": np.mean(rank_improves)*100}
    
    results = {"accuracy": acc, "cos_cls_centroid": cos_cls_cent, "pairs": pair_results}
    return results


# ===== Main =====
def main():
    global log, model_device_global, actual_layers_global
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="glm4")
    args = parser.parse_args()
    
    models_to_run = [args.model] if args.model != "all" else MODEL_ORDER
    
    for mname in models_to_run:
        if mname not in MODEL_MAP:
            log(f"Unknown model: {mname}")
            continue
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = os.path.join("d:\\develop\\TransformerLens-main\\tests\\glm5_temp",
                              f"stage722_phase17_{mname}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        log = Logger(out_dir, "results")
        log(f"\n{'#'*70}")
        log(f"# Stage 722 Phase XVII — {mname}")
        log(f"# Time: {ts}")
        log(f"{'#'*70}")
        
        model, tokenizer, n_layers, d_model = load_model(mname)
        model_device_global = model.device
        actual_layers_global = get_actual_layers(model)
        
        texts = build_texts()
        log(f"  Total texts: {len(texts)}")
        
        log("  Computing centroids...")
        centroids = compute_centroids(model, tokenizer, texts, n_samples=40)
        for cat, c in centroids.items():
            log(f"    {cat}: norm={c.norm():.2f}")
        
        all_results = {}
        
        # P126
        try:
            t0 = time.time()
            r = p126_precise_gradient(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P126"] = r
            log(f"  P126 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P126 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P127
        try:
            t0 = time.time()
            r = p127_manipulation_subspace(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P127"] = r
            log(f"  P127 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P127 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P128
        try:
            t0 = time.time()
            r = p128_architecture_analysis(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P128"] = r
            log(f"  P128 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P128 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P129 (only GLM4 since others are not manipulable)
        if mname == "glm4":
            try:
                t0 = time.time()
                r = p129_qualitative_analysis(model, tokenizer, n_layers, d_model, texts, centroids)
                all_results["P129"] = r
                log(f"  P129 done in {time.time()-t0:.0f}s")
            except Exception as e:
                log(f"  P129 error: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        
        # P130
        try:
            t0 = time.time()
            r = p130_classifier_gradient(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P130"] = r
            log(f"  P130 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P130 error: {e}")
            import traceback; traceback.print_exc()
        
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log(f"\n  Results saved to {out_dir}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        if log:
            log.close()


if __name__ == "__main__":
    main()
