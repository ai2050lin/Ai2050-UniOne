#!/usr/bin/env python3
"""
Stage 721: Phase XVI — 寻找因果性操控方向
================================================================================
Phase XV发现:
- Centroid差方向不是因果性操控方向（注入后远离目标）
- GLM4有多层超线性协同(3.4x), Qwen3/DS7B是次线性
- 连续操控在长期生成中改善质量(dPPL<0 after 10 steps)

Phase XVI目标:
P121: 梯度上升方向搜索 — 对目标类别token概率做梯度上升, 找最优操控方向
P122: PCA主成分操控 — 提取hidden state PCA, 测试各主成分方向的操控效果
P123: Contrastive方向 — 正例vs负例对比, 训练线性分类器提取最优方向
P124: Attention Head操控 — 精确操控特定attention head输出
P125: 方向对比综合评估 — 5种方向方法的效果对比

用法: python stage721_phase16.py --model glm4
      python stage721_phase16.py --model qwen3
      python stage721_phase16.py --model deepseek7b
      python stage721_phase16.py --model all
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

# ===== Text dataset =====
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


def gram_schmidt(vectors):
    """Gram-Schmidt正交化"""
    orthogonal = []
    for v in vectors:
        v = v.float()
        for u in orthogonal:
            v = v - torch.dot(v, u) * u
        norm = v.norm()
        if norm > 1e-6:
            v = v / norm
            orthogonal.append(v)
    return orthogonal


# ===== P121: 梯度上升方向搜索 =====
def p121_gradient_ascent(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    核心思路: 对每个类别, 找到最能增加该类别token概率的hidden state方向
    方法: 
    1. 对目标类别文本, 前向传播得到hidden state h 和 logits
    2. 计算目标类别centroid在unembed空间中的"偏好token集合"(cos top-k tokens)
    3. 构造目标函数 = mean(log P(target_tokens | h)), 对h做梯度上升
    4. 梯度方向就是"因果性操控方向"
    """
    log(f"\n{'='*70}")
    log(f"P121: Gradient Ascent Direction Search")
    log(f"{'='*70}")
    
    uw, ub = get_unembed(model)
    if uw is None:
        log("  WARNING: Cannot get unembed weights, skipping P121")
        return {}
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    cat_pairs = [("code", "gen_en"), ("math_sci", "poetry"), ("chinese", "philosophy")]
    
    # Get target tokens per category (top-k tokens most aligned with centroid)
    log("  Computing target tokens per category...")
    cat_target_tokens = {}
    for cat in categories:
        if cat not in centroids:
            continue
        centroid = centroids[cat].to(uw.device)
        # Cosine similarity between centroid and each unembed row
        cos_sims = F.cosine_similarity(centroid.unsqueeze(0), uw, dim=1)
        topk_indices = cos_sims.topk(min(200, uw.size(0))).indices
        cat_target_tokens[cat] = topk_indices
    
    results = {}
    
    for src_cat, tgt_cat in cat_pairs:
        log(f"\n  --- {src_cat} -> {tgt_cat} ---")
        if tgt_cat not in cat_target_tokens:
            continue
            
        # Source texts
        src_texts = [(t, c) for t, c in texts if c == src_cat][:12]
        target_token_ids = cat_target_tokens[tgt_cat]
        
        grad_directions = []
        
        for i, (text, cat) in enumerate(src_texts[:8]):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            
            # Forward pass with gradients enabled for hidden states
            model.zero_grad()
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()  # [1, d_model]
            
            # Compute logits from unembed
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            
            # Objective: maximize log prob of target tokens
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_prob = log_probs[0, target_token_ids].mean()
            
            # Gradient ascent: direction = grad(target_log_prob) w.r.t. h
            target_log_prob.backward(retain_graph=False)
            if h.grad is not None:
                grad = h.grad.detach().cpu().squeeze(0)
                grad_norm = grad.norm().item()
                if grad_norm > 1e-8:
                    grad_normalized = grad / grad_norm
                    grad_directions.append(grad_normalized)
                    if i < 3:
                        log(f"    text {i}: grad_norm={grad_norm:.4f}, "
                            f"cos(grad, centroid_diff)={F.cosine_similarity(grad_normalized.unsqueeze(0), (centroids[tgt_cat] - centroids[src_cat]).unsqueeze(0)).item():.4f}")
        
        if not grad_directions:
            continue
        
        # Average gradient direction
        avg_grad = torch.stack(grad_directions).mean(dim=0)
        avg_grad = avg_grad / avg_grad.norm()
        
        # Also compute centroid diff direction for comparison
        centroid_diff = centroids[tgt_cat] - centroids[src_cat]
        centroid_diff = centroid_diff / centroid_diff.norm()
        
        cos_grad_centroid = F.cosine_similarity(avg_grad.unsqueeze(0), centroid_diff.unsqueeze(0)).item()
        log(f"  avg_grad vs centroid_diff: cos={cos_grad_centroid:.4f}")
        
        # Test gradient direction as injection
        inj_layer_indices = list(range(min(10, n_layers)))
        
        test_texts = [(t, c) for t, c in texts if c == src_cat][:8]
        
        # Compare: gradient direction vs centroid direction
        for dir_name, direction in [("gradient", avg_grad), ("centroid", centroid_diff)]:
            one_cos_vals = []
            kl_vals = []
            text_diffs = []
            rank_improves = []
            
            for text, _ in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                
                # Natural
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                # Injected (5 tokens)
                handles = []
                for lidx in inj_layer_indices:
                    h = actual_layers_global[lidx].register_forward_hook(make_inject_hook(direction, 0.10))
                    handles.append(h)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                finally:
                    for hh in handles:
                        hh.remove()
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                
                one_cos = (1 - F.cosine_similarity(nat_h, inj_h).item())
                one_cos_vals.append(one_cos)
                
                kl = kl_divergence(nat_out.logits[:, -1, :], inj_out.logits[:, -1, :])
                kl_vals.append(kl)
                
                # Rank improvement: did inj_h get closer to tgt centroid?
                nat_cos_tgt = F.cosine_similarity(nat_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                inj_cos_tgt = F.cosine_similarity(inj_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos_tgt > nat_cos_tgt else 0)
            
            avg_1cos = np.mean(one_cos_vals)
            avg_kl = np.mean(kl_vals)
            rank_rate = np.mean(rank_improves) * 100
            
            log(f"  [{dir_name}] 1-cos={avg_1cos:.6f}, KL={avg_kl:.4f}, "
                f"rank_improve={rank_rate:.0f}%")
            
            results[f"{src_cat}->{tgt_cat}_{dir_name}"] = {
                "1cos": avg_1cos, "kl": avg_kl, "rank_improve": rank_rate,
                "cos_grad_centroid": cos_grad_centroid
            }
    
    # Scale sweep for gradient direction on best pair
    log(f"\n  --- Gradient Direction Scale Sweep (code->gen_en) ---")
    src_cat, tgt_cat = "code", "gen_en"
    src_texts_t = [(t, c) for t, c in texts if c == src_cat][:8]
    
    if "code" in cat_target_tokens and len(grad_directions) > 0:
        avg_grad_code = torch.stack(grad_directions).mean(dim=0)
        avg_grad_code = avg_grad_code / avg_grad_code.norm()
        
        scales = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
        for scale in scales:
            one_cos_vals = []
            kl_vals = []
            rank_improves = []
            for text, _ in src_texts_t:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(avg_grad_code, scale))
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
                nat_cos_tgt = F.cosine_similarity(nat_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                inj_cos_tgt = F.cosine_similarity(inj_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos_tgt > nat_cos_tgt else 0)
            
            log(f"  scale={scale:.2f}: 1-cos={np.mean(one_cos_vals):.6f}, "
                f"KL={np.mean(kl_vals):.4f}, rank_improve={np.mean(rank_improves)*100:.0f}%")
            results[f"scale_sweep_code_gen_en_s{scale}"] = {
                "1cos": np.mean(one_cos_vals), "kl": np.mean(kl_vals),
                "rank_improve": np.mean(rank_improves)*100
            }
    
    return results


# ===== P122: PCA主成分操控 =====
def p122_pca_directions(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    提取所有类别文本hidden states的PCA主成分, 测试各主成分方向的操控效果
    """
    log(f"\n{'='*70}")
    log(f"P122: PCA Principal Component Manipulation")
    log(f"{'='*70}")
    
    # Collect hidden states from all categories
    log("  Collecting hidden states for PCA...")
    all_h = []
    all_labels = []
    sample_texts = texts[::max(1, len(texts)//60)][:60]
    
    for text, cat in sample_texts:
        h, _ = get_final_h(model, tokenizer, text)
        all_h.append(h)
        all_labels.append(cat)
    
    H = torch.stack(all_h)  # [N, d_model]
    log(f"  H shape: {H.shape}")
    
    # PCA via SVD
    H_centered = H - H.mean(dim=0)
    U, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
    
    log(f"  Top-10 explained variance ratios:")
    total_var = (S**2).sum().item()
    for i in range(min(10, len(S))):
        var_ratio = (S[i]**2).item() / total_var
        log(f"    PC{i}: var={var_ratio:.4f}, singular={S[i].item():.2f}")
    
    # Test each PC direction
    inj_layer_indices = list(range(min(10, n_layers)))
    test_texts = texts[:20]
    
    results = {}
    
    # Test top 8 PCs
    for pc_idx in range(min(8, Vt.size(0))):
        pc_dir = Vt[pc_idx].float()
        pc_dir = pc_dir / pc_dir.norm()
        
        one_cos_vals = []
        kl_vals = []
        
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
            
            one_cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
            kl_vals.append(kl_divergence(nat_out.logits[:, -1, :], inj_out.logits[:, -1, :]))
        
        avg_1cos = np.mean(one_cos_vals)
        avg_kl = np.mean(kl_vals)
        log(f"  PC{pc_idx}: 1-cos={avg_1cos:.6f}, KL={avg_kl:.4f}, "
            f"var_ratio={((S[pc_idx]**2).item()/total_var):.4f}")
        results[f"PC{pc_idx}"] = {"1cos": avg_1cos, "kl": avg_kl,
                                   "var_ratio": (S[pc_idx]**2).item()/total_var}
    
    # Test top PC with different scales
    log(f"\n  --- PC0 Scale Sweep ---")
    pc0 = Vt[0].float()
    pc0 = pc0 / pc0.norm()
    
    scales = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    for scale in scales:
        one_cos_vals = []
        kl_vals = []
        for text, _ in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in inj_layer_indices:
                hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(pc0, scale))
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
        log(f"  scale={scale:.2f}: 1-cos={np.mean(one_cos_vals):.6f}, KL={np.mean(kl_vals):.4f}")
        results[f"PC0_scale_{scale}"] = {"1cos": np.mean(one_cos_vals), "kl": np.mean(kl_vals)}
    
    # PC semantic interpretation: which category does each PC separate?
    log(f"\n  --- PC Semantic Analysis ---")
    H_proj = H_centered @ Vt[:8].T  # [N, 8]
    for pc_idx in range(min(8, Vt.size(0))):
        pc_vals = H_proj[:, pc_idx]
        cat_means = defaultdict(list)
        for j, label in enumerate(all_labels):
            cat_means[label].append(pc_vals[j].item())
        log(f"  PC{pc_idx} category means:")
        for cat in sorted(cat_means.keys()):
            log(f"    {cat}: {np.mean(cat_means[cat]):.4f}")
    
    return results


# ===== P123: Contrastive方向 =====
def p123_contrastive_direction(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    用线性分类器从正例(目标类别)vs负例(其他类别)学习最优分离方向
    """
    log(f"\n{'='*70}")
    log(f"P123: Contrastive Linear Classifier Direction")
    log(f"{'='*70}")
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    cat_pairs = [("code", "gen_en"), ("math_sci", "poetry"), ("chinese", "philosophy")]
    
    results = {}
    
    for tgt_cat, neg_cat in cat_pairs:
        log(f"\n  --- Contrastive: {tgt_cat} vs {neg_cat} ---")
        
        # Collect positive and negative samples
        pos_h = []
        neg_h = []
        for text, cat in texts:
            if cat == tgt_cat:
                h, _ = get_final_h(model, tokenizer, text)
                pos_h.append(h)
            elif cat == neg_cat:
                h, _ = get_final_h(model, tokenizer, text)
                neg_h.append(h)
            if len(pos_h) >= 20 and len(neg_h) >= 20:
                break
        
        pos_h = torch.stack(pos_h)  # [N_pos, d_model]
        neg_h = torch.stack(neg_h)  # [N_neg, d_model]
        
        # Method 1: Mean difference (Fisher-like)
        mean_pos = pos_h.mean(dim=0)
        mean_neg = neg_h.mean(dim=0)
        fisher_dir = mean_pos - mean_neg
        fisher_dir = fisher_dir / fisher_dir.norm()
        
        # Method 2: Within-class scatter weighted (LDA-like)
        S_w = ((pos_h - mean_pos).T @ (pos_h - mean_pos) + 
               (neg_h - mean_neg).T @ (neg_h - mean_neg))
        S_w_inv = torch.linalg.pinv(S_w.float())
        lda_dir = S_w_inv @ (mean_pos - mean_neg).float()
        lda_dir = lda_dir / lda_dir.norm()
        
        # Method 3: SVM-like (just centroid diff, already tested)
        centroid_dir = centroids[tgt_cat] - centroids[neg_cat]
        centroid_dir = centroid_dir / centroid_dir.norm()
        
        # Cosine between different methods
        cos_fish_lda = F.cosine_similarity(fisher_dir.unsqueeze(0), lda_dir.unsqueeze(0)).item()
        cos_fish_cent = F.cosine_similarity(fisher_dir.unsqueeze(0), centroid_dir.unsqueeze(0)).item()
        cos_lda_cent = F.cosine_similarity(lda_dir.unsqueeze(0), centroid_dir.unsqueeze(0)).item()
        log(f"  cos(fisher, lda)={cos_fish_lda:.4f}, cos(fisher, centroid)={cos_fish_cent:.4f}, "
            f"cos(lda, centroid)={cos_lda_cent:.4f}")
        
        # Test each direction
        inj_layer_indices = list(range(min(10, n_layers)))
        test_src = [(t, c) for t, c in texts if c == neg_cat][:8]
        
        for dir_name, direction in [("fisher", fisher_dir), ("lda", lda_dir), ("centroid", centroid_dir)]:
            one_cos_vals = []
            kl_vals = []
            rank_improves = []
            
            for text, _ in test_src:
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
                nat_cos_tgt = F.cosine_similarity(nat_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                inj_cos_tgt = F.cosine_similarity(inj_h.cpu(), centroids[tgt_cat].unsqueeze(0)).item()
                rank_improves.append(1 if inj_cos_tgt > nat_cos_tgt else 0)
            
            avg_1cos = np.mean(one_cos_vals)
            avg_kl = np.mean(kl_vals)
            rank_rate = np.mean(rank_improves) * 100
            
            log(f"  [{dir_name}] 1-cos={avg_1cos:.6f}, KL={avg_kl:.4f}, "
                f"rank_improve={rank_rate:.0f}%")
            results[f"{tgt_cat}_vs_{neg_cat}_{dir_name}"] = {
                "1cos": avg_1cos, "kl": avg_kl, "rank_improve": rank_rate
            }
        
        # Scale sweep for Fisher direction
        log(f"  --- Fisher Scale Sweep ---")
        scales = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
        for scale in scales:
            one_cos_vals = []
            rank_improves = []
            for text, _ in test_src:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                handles = []
                for lidx in inj_layer_indices:
                    hh = actual_layers_global[lidx].register_forward_hook(make_inject_hook(fisher_dir, scale))
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
            results[f"{tgt_cat}_vs_{neg_cat}_fisher_s{scale}"] = {
                "1cos": np.mean(one_cos_vals), "rank_improve": np.mean(rank_improves)*100
            }
    
    return results


# ===== P124: Attention Head操控 =====
def p124_attention_head_manipulation(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    精确操控特定attention head的输出而非整个hidden state
    """
    log(f"\n{'='*70}")
    log(f"P124: Attention Head-Level Manipulation")
    log(f"{'='*70}")
    
    # Find attention heads
    log("  Probing attention head structure...")
    
    actual_layers = actual_layers_global
    head_configs = []
    
    for lidx in range(min(6, n_layers)):
        layer = actual_layers[lidx]
        # Try to find attention module
        attn = None
        for attr in ['self_attn', 'attention', 'attn']:
            attn = getattr(layer, attr, None)
            if attn is not None:
                break
        
        if attn is not None:
            n_heads = getattr(attn, 'num_heads', getattr(attn, 'n_head', 
                         getattr(attn.config, 'num_attention_heads', None)))
            head_dim = getattr(attn, 'head_dim', d_model // n_heads if n_heads else d_model // 8)
            if n_heads is None:
                # Try to infer from q_proj shape
                q_proj = getattr(attn, 'q_proj', getattr(attn, 'query', None))
                if q_proj is not None:
                    out_features = q_proj.out_features
                    n_heads = getattr(attn, 'num_heads', 8)  # fallback
                    head_dim = out_features // n_heads
            if n_heads:
                head_configs.append((lidx, n_heads, head_dim, attn))
                log(f"    Layer {lidx}: n_heads={n_heads}, head_dim={head_dim}")
    
    if not head_configs:
        log("  WARNING: Cannot find attention heads, skipping P124")
        return {}
    
    # Test: inject into specific attention head outputs
    results = {}
    direction = centroids.get("code", centroids[list(centroids.keys())[0]]) - centroids.get("gen_en", centroids[list(centroids.keys())[1]])
    direction = direction / direction.norm()
    
    inj_layer_indices = list(range(min(10, n_layers)))
    
    # Method 1: Full hidden state injection (baseline)
    log(f"\n  --- Baseline: Full hidden state injection ---")
    test_texts = texts[:15]
    
    full_cos_vals = []
    full_kl_vals = []
    for text, _ in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            nat_out = model(**inputs, output_hidden_states=True)
        nat_h = nat_out.hidden_states[-1][:, -1, :].float()
        handles = []
        for lidx in inj_layer_indices:
            hh = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, 0.10))
            handles.append(hh)
        try:
            with torch.no_grad():
                inj_out = model(**inputs, output_hidden_states=True)
        finally:
            for hh in handles:
                hh.remove()
        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
        full_cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        full_kl_vals.append(kl_divergence(nat_out.logits[:, -1, :], inj_out.logits[:, -1, :]))
    
    log(f"  Full: 1-cos={np.mean(full_cos_vals):.6f}, KL={np.mean(full_kl_vals):.4f}")
    
    # Method 2: Single layer injection
    log(f"\n  --- Single Layer Injection ---")
    for layer_only in range(min(5, n_layers)):
        cos_vals = []
        for text, _ in test_texts[:8]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            hh = actual_layers[layer_only].register_forward_hook(make_inject_hook(direction, 0.10))
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                hh.remove()
            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            cos_vals.append(1 - F.cosine_similarity(nat_h, inj_h).item())
        log(f"  Layer {layer_only} only: 1-cos={np.mean(cos_vals):.6f}")
        results[f"layer_{layer_only}_only"] = {"1cos": np.mean(cos_vals)}
    
    # Method 3: First half vs second half layers
    log(f"\n  --- Layer Split Analysis ---")
    mid = n_layers // 2
    for name, layer_range in [("first_half", range(0, mid)), ("second_half", range(mid, n_layers))]:
        cos_vals = []
        for text, _ in test_texts[:8]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in layer_range:
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
        log(f"  {name}: 1-cos={np.mean(cos_vals):.6f}")
        results[name] = {"1cos": np.mean(cos_vals)}
    
    return results


# ===== P125: 方向对比综合评估 =====
def p125_direction_comparison(model, tokenizer, n_layers, d_model, texts, centroids):
    """
    综合5种方向生成方法, 在相同条件下对比操控效果
    生成10个token, 比较PPL变化和文本改变
    """
    log(f"\n{'='*70}")
    log(f"P125: Comprehensive Direction Comparison (Generation)")
    log(f"{'='*70}")
    
    categories = ["code", "math_sci", "poetry", "philosophy", "chinese", "gen_en"]
    
    # Compute 5 directions
    log("  Computing 5 directions...")
    
    # 1. Centroid diff (code -> gen_en)
    dir_centroid = centroids["gen_en"] - centroids["code"]
    dir_centroid = dir_centroid / dir_centroid.norm()
    
    # 2. Gradient direction
    uw, ub = get_unembed(model)
    grad_dirs = []
    if uw is not None:
        gen_en_texts = [(t, c) for t, c in texts if c == "gen_en"][:8]
        for text, cat in gen_en_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            model.zero_grad()
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float()
            logits = F.linear(h, uw.to(h.device), ub.to(h.device) if ub is not None else None)
            log_probs = F.log_softmax(logits, dim=-1)
            # Maximize entropy to get "generic" direction
            entropy = -(F.softmax(logits, dim=-1) * log_probs).sum()
            entropy.backward(retain_graph=False)
            if h.grad is not None:
                grad = h.grad.detach().cpu().squeeze(0)
                if grad.norm() > 1e-8:
                    grad_dirs.append(grad / grad.norm())
        if grad_dirs:
            dir_gradient = torch.stack(grad_dirs).mean(dim=0)
            dir_gradient = dir_gradient / dir_gradient.norm()
        else:
            dir_gradient = torch.randn(d_model)
            dir_gradient = dir_gradient / dir_gradient.norm()
    else:
        dir_gradient = torch.randn(d_model)
        dir_gradient = dir_gradient / dir_gradient.norm()
    
    # 3. PCA direction
    all_h = []
    sample_texts = texts[::max(1, len(texts)//40)][:40]
    for text, cat in sample_texts:
        h, _ = get_final_h(model, tokenizer, text)
        all_h.append(h)
    H = torch.stack(all_h)
    H_centered = H - H.mean(dim=0)
    U, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
    dir_pca = Vt[0].float()
    dir_pca = dir_pca / dir_pca.norm()
    
    # 4. Random direction (orthogonal to centroid)
    rand_dir = torch.randn(d_model)
    rand_dir = rand_dir - torch.dot(rand_dir, dir_centroid) * dir_centroid
    rand_dir = rand_dir / rand_dir.norm()
    
    # 5. Fisher direction
    code_h, gen_h = [], []
    for text, cat in texts:
        if cat == "code" and len(code_h) < 20:
            h, _ = get_final_h(model, tokenizer, text)
            code_h.append(h)
        elif cat == "gen_en" and len(gen_h) < 20:
            h, _ = get_final_h(model, tokenizer, text)
            gen_h.append(h)
        if len(code_h) >= 20 and len(gen_h) >= 20:
            break
    dir_fisher = torch.stack(gen_h).mean(dim=0) - torch.stack(code_h).mean(dim=0)
    dir_fisher = dir_fisher / dir_fisher.norm()
    
    # Cosine similarities between directions
    dirs = {"centroid": dir_centroid, "gradient": dir_gradient, "pca": dir_pca, 
            "random": rand_dir, "fisher": dir_fisher}
    log("  Direction cosine similarities:")
    dir_names = list(dirs.keys())
    cos_matrix = {}
    for i in range(len(dir_names)):
        for j in range(i+1, len(dir_names)):
            c = F.cosine_similarity(dirs[dir_names[i]].unsqueeze(0), dirs[dir_names[j]].unsqueeze(0)).item()
            cos_matrix[f"{dir_names[i]}_{dir_names[j]}"] = c
            log(f"    cos({dir_names[i]}, {dir_names[j]})={c:.4f}")
    
    # Generate 10 tokens with each direction
    inj_layer_indices = list(range(min(10, n_layers)))
    src_texts = [(t, c) for t, c in texts if c == "code"][:6]
    n_gen = 10
    scale = 0.10
    
    results = {"cos_matrix": cos_matrix}
    
    for dir_name, direction in dirs.items():
        log(f"\n  --- Generating with {dir_name} direction ---")
        
        ppl_inj_vals = []
        ppl_nat_vals = []
        text_diffs = []
        cos_shift_vals = []
        
        for text, cat in src_texts:
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
            text_diffs.append(text_diff)
            
            try:
                nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
                ppl_nat_vals.append(nat_ppl)
                ppl_inj_vals.append(inj_ppl)
            except:
                pass
            
            # Cosine shift to gen_en centroid
            try:
                h_nat, _ = get_final_h(model, tokenizer, nat_text)
                h_inj, _ = get_final_h(model, tokenizer, inj_text)
                nat_cos = F.cosine_similarity(h_nat.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                inj_cos = F.cosine_similarity(h_inj.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                cos_shift_vals.append(inj_cos - nat_cos)
            except:
                pass
            
            if len(text_diffs) <= 2:
                log(f"    nat: {nat_text[:80]}...")
                log(f"    inj: {inj_text[:80]}...")
        
        avg_ppl_nat = np.mean(ppl_nat_vals) if ppl_nat_vals else 0
        avg_ppl_inj = np.mean(ppl_inj_vals) if ppl_inj_vals else 0
        avg_text_diff = np.mean(text_diffs) * 100
        avg_cos_shift = np.mean(cos_shift_vals) if cos_shift_vals else 0
        
        log(f"  [{dir_name}] PPL_nat={avg_ppl_nat:.1f}, PPL_inj={avg_ppl_inj:.1f}, "
            f"dPPL={avg_ppl_inj-avg_ppl_nat:+.1f}, "
            f"Text_change={avg_text_diff:.0f}%, cos_shift={avg_cos_shift:+.4f}")
        
        results[dir_name] = {
            "ppl_nat": avg_ppl_nat, "ppl_inj": avg_ppl_inj,
            "dppl": avg_ppl_inj - avg_ppl_nat,
            "text_change": avg_text_diff, "cos_shift": avg_cos_shift
        }
    
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
                              f"stage721_phase16_{mname}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        log = Logger(out_dir, "results")
        log(f"\n{'#'*70}")
        log(f"# Stage 721 Phase XVI — {mname}")
        log(f"# Time: {ts}")
        log(f"{'#'*70}")
        
        # Load
        model, tokenizer, n_layers, d_model = load_model(mname)
        model_device_global = model.device
        actual_layers_global = get_actual_layers(model)
        
        # Build texts and centroids
        texts = build_texts()
        log(f"  Total texts: {len(texts)}")
        
        log("  Computing centroids...")
        centroids = compute_centroids(model, tokenizer, texts, n_samples=40)
        for cat, c in centroids.items():
            log(f"    {cat}: norm={c.norm():.2f}")
        
        all_results = {}
        
        # P121
        try:
            t0 = time.time()
            r = p121_gradient_ascent(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P121"] = r
            log(f"  P121 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P121 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P122
        try:
            t0 = time.time()
            r = p122_pca_directions(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P122"] = r
            log(f"  P122 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P122 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P123
        try:
            t0 = time.time()
            r = p123_contrastive_direction(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P123"] = r
            log(f"  P123 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P123 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P124
        try:
            t0 = time.time()
            r = p124_attention_head_manipulation(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P124"] = r
            log(f"  P124 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P124 error: {e}")
            import traceback; traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # P125
        try:
            t0 = time.time()
            r = p125_direction_comparison(model, tokenizer, n_layers, d_model, texts, centroids)
            all_results["P125"] = r
            log(f"  P125 done in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"  P125 error: {e}")
            import traceback; traceback.print_exc()
        
        # Save results
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
