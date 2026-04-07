#!/usr/bin/env python3
"""
Stage 720: Phase XV — GLM4语义定向验证 + 逐层消融 + 多维度操控 + 连续生成操控
================================================================================
Phase XIV发现:
- GLM4存在"甜蜜操控区间" scale=0.08~0.15 (Text=50~70%, dPPL<0)
- Qwen3几乎不可操控, DS7B需要极大scale
- 累积注入是唯一有效方法

Phase XV目标:
P117: GLM4语义定向验证 — 生成后用centroid距离验证类别偏移方向
P118: 逐层注入贡献消融 — 精确量化每层对操控效果的贡献
P119: 多维度同时操控 — 同时偏移多个语义方向
P120: 连续多步操控 — 15步生成中持续注入, 测长文本引导能力

用法: python stage720_phase15.py --model glm4
      python stage720_phase15.py --model qwen3
      python stage720_phase15.py --model deepseek7b
      python stage720_phase15.py --model all
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


def make_inject_hook(direction, scale):
    d = direction.to(model_device_global, dtype=torch.bfloat16)
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + d * scale,) + output[1:]
        return output + d * scale
    return hook


model_device_global = None


def generate_with_hooks(model, tokenizer, input_ids, inj_layers, direction, scale, n_tokens=10):
    """Generate n_tokens with cumulative injection at specified layers."""
    gen_ids = input_ids.clone()
    d = direction.to(model_device_global, dtype=torch.bfloat16)
    for _ in range(n_tokens):
        handles = []
        for lidx in inj_layers:
            h = actual_layers_global[lidx].register_forward_hook(
                lambda m, inp, out, _d=d, _s=scale: 
                    (out[0] + _d * _s,) + out[1:] if isinstance(out, tuple) else out + _d * _s
            )
            handles.append(h)
        try:
            with torch.no_grad():
                out = model(input_ids=gen_ids)
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        finally:
            for h in handles:
                h.remove()
        if tok.item() == tokenizer.eos_token_id:
            break
        gen_ids = torch.cat([gen_ids, tok], dim=1)
    return gen_ids


def generate_natural(model, tokenizer, input_ids, n_tokens=10):
    """Generate n_tokens naturally without injection."""
    gen_ids = input_ids.clone()
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=gen_ids)
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if tok.item() == tokenizer.eos_token_id:
            break
        gen_ids = torch.cat([gen_ids, tok], dim=1)
    return gen_ids


actual_layers_global = None


# ===== P117: GLM4语义定向验证 =====
def p117_semantic_verification(model, tokenizer, n_layers, d_model, texts, centroids):
    """验证操控后文本是否真正偏向目标类别"""
    log(f"\n{'='*70}")
    log(f"P117: Semantic Direction Verification")
    log(f"{'='*70}")
    
    categories = sorted(centroids.keys())
    # Choose category pairs for targeted shifting
    cat_pairs = []
    for i in range(min(len(categories), 6)):
        for j in range(i+1, min(len(categories), 6)):
            cat_pairs.append((categories[i], categories[j]))
    
    log(f"  Categories: {categories}")
    log(f"  Testing {len(cat_pairs)} category pairs")
    log(f"  Scale: 0.10 (sweet spot from Phase XIV)")
    
    # For each pair: take source cat text, inject toward target cat direction
    # Then measure: 1) centroid distance shift, 2) generated text category
    results = []
    
    scale = 0.10
    inj_count = n_layers // 4
    container = get_layers_container(model)
    layers_attr = None
    for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
        if hasattr(container, attr):
            layers_attr = attr
            break
    actual_layers = getattr(container, layers_attr)
    inj_layer_indices = list(range(inj_count))
    
    n_per_pair = 8  # texts per source category
    
    for src_cat, tgt_cat in cat_pairs[:8]:  # limit to 8 pairs
        direction = (centroids[tgt_cat] - centroids[src_cat])
        direction = direction / (direction.norm() + 1e-8)
        
        src_texts = [(t, c) for t, c in texts if c == src_cat][:n_per_pair]
        if len(src_texts) < 3:
            continue
        
        correct_shift = 0
        total_cos_shifts = []
        
        for text, cat in src_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            
            # Natural centroid distances
            h_nat, _ = get_final_h(model, tokenizer, text)
            nat_dists = {}
            for c in categories:
                nat_dists[c] = F.cosine_similarity(h_nat.unsqueeze(0), centroids[c].unsqueeze(0)).item()
            
            # Injected centroid distances (after generating 5 tokens)
            gen_ids = inputs["input_ids"].clone()
            handles = []
            for lidx in inj_layer_indices:
                h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
                handles.append(h)
            try:
                with torch.no_grad():
                    for _ in range(5):
                        out = model(input_ids=gen_ids)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        gen_ids = torch.cat([gen_ids, tok], dim=1)
            finally:
                for h in handles:
                    h.remove()
            
            inj_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            h_inj, _ = get_final_h(model, tokenizer, inj_text)
            inj_dists = {}
            for c in categories:
                inj_dists[c] = F.cosine_similarity(h_inj.unsqueeze(0), centroids[c].unsqueeze(0)).item()
            
            # Check if target category distance decreased (approached)
            nat_rank_src = sorted(nat_dists.items(), key=lambda x: -x[1]).index((src_cat, nat_dists[src_cat]))
            nat_rank_tgt = sorted(nat_dists.items(), key=lambda x: -x[1]).index((tgt_cat, nat_dists[tgt_cat]))
            inj_rank_src = sorted(inj_dists.items(), key=lambda x: -x[1]).index((src_cat, inj_dists[src_cat]))
            inj_rank_tgt = sorted(inj_dists.items(), key=lambda x: -x[1]).index((tgt_cat, inj_dists[tgt_cat]))
            
            tgt_cos_shift = inj_dists[tgt_cat] - nat_dists[tgt_cat]
            total_cos_shifts.append(tgt_cos_shift)
            
            if inj_rank_tgt < nat_rank_tgt:  # target rank improved
                correct_shift += 1
        
        if total_cos_shifts:
            avg_shift = np.mean(total_cos_shifts)
            shift_rate = correct_shift / len(src_texts) * 100
            log(f"  {src_cat[:8]:>8s} -> {tgt_cat[:8]:>8s}: "
                f"avg_cos_shift={avg_shift:+.4f}, rank_improve={shift_rate:.0f}%")
            results.append({
                "src": src_cat, "tgt": tgt_cat,
                "avg_cos_shift": avg_shift,
                "rank_improve_rate": shift_rate,
                "n_texts": len(src_texts),
            })
    
    # Summary
    if results:
        avg_all_shift = np.mean([r["avg_cos_shift"] for r in results])
        avg_improve = np.mean([r["rank_improve_rate"] for r in results])
        log(f"\n  Summary: avg_cos_shift={avg_all_shift:+.4f}, avg_rank_improve={avg_improve:.1f}%")
        log(f"  Interpretation:")
        if avg_all_shift > 0:
            log(f"    -> Injection DOES shift generated text toward target category!")
        else:
            log(f"    -> Injection shifts AWAY from target (centroid direction is not causal)")
        
        # Also test with larger scales
        log(f"\n  Scale sweep for verification:")
        for s in [0.05, 0.10, 0.15, 0.20]:
            shifts = []
            for src_cat, tgt_cat in cat_pairs[:4]:
                direction = (centroids[tgt_cat] - centroids[src_cat])
                direction = direction / (direction.norm() + 1e-8)
                src_texts = [(t, c) for t, c in texts if c == src_cat][:5]
                for text, cat in src_texts:
                    h_nat, _ = get_final_h(model, tokenizer, text)
                    nat_d = F.cosine_similarity(h_nat.unsqueeze(0), centroids[tgt_cat].unsqueeze(0)).item()
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                    gen_ids = inputs["input_ids"].clone()
                    handles = []
                    for lidx in inj_layer_indices:
                        h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, s))
                        handles.append(h)
                    try:
                        with torch.no_grad():
                            for _ in range(5):
                                out = model(input_ids=gen_ids)
                                tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                                if tok.item() == tokenizer.eos_token_id:
                                    break
                                gen_ids = torch.cat([gen_ids, tok], dim=1)
                    finally:
                        for h in handles:
                            h.remove()
                    inj_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    h_inj, _ = get_final_h(model, tokenizer, inj_text)
                    inj_d = F.cosine_similarity(h_inj.unsqueeze(0), centroids[tgt_cat].unsqueeze(0)).item()
                    shifts.append(inj_d - nat_d)
            if shifts:
                log(f"    scale={s:.2f}: avg_cos_shift={np.mean(shifts):+.4f}")
    
    return results


# ===== P118: 逐层注入贡献消融 =====
def p118_layer_ablation(model, tokenizer, n_layers, d_model, texts, centroids):
    """精确量化每层对操控效果的贡献"""
    log(f"\n{'='*70}")
    log(f"P118: Per-Layer Contribution Ablation")
    log(f"{'='*70}")
    
    # Choose a direction: code - gen_en (most distinct categories)
    if "code" in centroids and "gen_en" in centroids:
        direction = (centroids["code"] - centroids["gen_en"])
    else:
        cats = sorted(centroids.keys())
        direction = (centroids[cats[1]] - centroids[cats[0]])
    direction = direction / (direction.norm() + 1e-8)
    
    scale = 0.10
    inj_count = n_layers // 4
    container = get_layers_container(model)
    layers_attr = None
    for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
        if hasattr(container, attr):
            layers_attr = attr
            break
    actual_layers = getattr(container, layers_attr)
    
    test_texts = texts[:20]
    
    # Baseline: no injection
    log(f"\n  Baseline (no injection):")
    baseline_cos = []
    for text, cat in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
        baseline_cos.append(1.0)  # cos with itself
    log(f"    avg_cos=1.000 (reference)")
    
    # Full injection baseline (all first-quarter layers)
    log(f"\n  Full injection ({inj_count} layers, scale={scale}):")
    full_cos = []
    for text, cat in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            nat_out = model(**inputs, output_hidden_states=True)
            nat_h = nat_out.hidden_states[-1][:, -1, :].float()
        handles = []
        for lidx in range(inj_count):
            h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
            handles.append(h)
        try:
            with torch.no_grad():
                inj_out = model(**inputs, output_hidden_states=True)
                inj_h = inj_out.hidden_states[-1][:, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        full_cos.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
    log(f"    avg_cos={np.mean(full_cos):.6f}, 1-cos={1-np.mean(full_cos):.6f}")
    
    # Per-layer ablation: inject at all layers EXCEPT one
    log(f"\n  Leave-one-out ablation (remove each layer from full injection):")
    contributions = []
    for leave_out in range(min(inj_count, 12)):  # limit to first 12 layers for speed
        ablation_cos = []
        for text, cat in test_texts[:15]:  # 15 texts for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in range(inj_count):
                if lidx == leave_out:
                    continue
                h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
                handles.append(h)
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
                    inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            finally:
                for h in handles:
                    h.remove()
            ablation_cos.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
        
        avg_abl_cos = np.mean(ablation_cos)
        avg_full_cos = np.mean(full_cos[:15])
        # Contribution = how much removing this layer REDUCES the effect
        contribution = (1 - avg_abl_cos) - (1 - avg_full_cos)
        contributions.append({
            "layer": leave_out,
            "ablation_1-cos": 1 - avg_abl_cos,
            "full_1-cos": 1 - avg_full_cos,
            "contribution": contribution,
        })
        log(f"    L{leave_out:2d}: ablation_1-cos={1-avg_abl_cos:.6f}, "
            f"contribution={contribution:+.6f}")
    
    # Single-layer injection
    log(f"\n  Single-layer injection (each layer alone, scale={scale}):")
    single_results = []
    for lidx in range(min(inj_count, 12)):
        single_cos = []
        for text, cat in test_texts[:15]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
                    inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            finally:
                h.remove()
            single_cos.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
        
        avg_s_cos = np.mean(single_cos)
        single_results.append({"layer": lidx, "1-cos": 1 - avg_s_cos})
        log(f"    L{lidx:2d}: 1-cos={1-avg_s_cos:.6f}")
    
    # Rank layers by contribution
    if contributions:
        log(f"\n  Layer ranking by contribution (leave-one-out):")
        sorted_contrib = sorted(contributions, key=lambda x: -x["contribution"])
        for i, c in enumerate(sorted_contrib):
            log(f"    #{i+1}: L{c['layer']:2d} contribution={c['contribution']:+.6f}")
        
        top3 = sorted_contrib[:3]
        top3_layers = [c["layer"] for c in top3]
        log(f"\n  Top-3 most important layers: {top3_layers}")
        
        # Test: inject ONLY at top-3 layers
        log(f"\n  Top-3 only injection ({top3_layers}):")
        top3_cos = []
        for text, cat in test_texts[:15]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = []
            for lidx in top3_layers:
                h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
                handles.append(h)
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
                    inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            finally:
                for h in handles:
                    h.remove()
            top3_cos.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
        log(f"    avg_cos={np.mean(top3_cos):.6f}, 1-cos={1-np.mean(top3_cos):.6f}")
        log(f"    Compare: full({inj_count}L) 1-cos={1-np.mean(full_cos[:15]):.6f}")
        log(f"    Top-3 achieves {100*(1-np.mean(top3_cos))/(1-np.mean(full_cos[:15])+1e-10):.0f}% of full injection effect")
    
    return {"contributions": contributions, "single": single_results}


# ===== P119: 多维度同时操控 =====
def p119_multi_direction(model, tokenizer, n_layers, d_model, texts, centroids):
    """同时偏移多个语义方向"""
    log(f"\n{'='*70}")
    log(f"P119: Multi-Direction Simultaneous Manipulation")
    log(f"{'='*70}")
    
    categories = sorted(centroids.keys())
    inj_count = n_layers // 4
    container = get_layers_container(model)
    layers_attr = None
    for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
        if hasattr(container, attr):
            layers_attr = attr
            break
    actual_layers = getattr(container, layers_attr)
    inj_layer_indices = list(range(inj_count))
    
    test_texts = texts[:15]
    
    # Test: single direction vs 2 directions vs 3 directions
    # Use orthogonal directions via Gram-Schmidt
    def orthogonalize(dirs):
        """Gram-Schmidt orthogonalization"""
        result = [dirs[0] / (dirs[0].norm() + 1e-8)]
        for d in dirs[1:]:
            proj = sum(r * torch.dot(d, r) for r in result)
            orth = d - proj
            norm = orth.norm()
            if norm > 1e-8:
                result.append(orth / norm)
        return result
    
    # Get multiple directions
    base_dirs = []
    for i in range(min(3, len(categories))):
        if i == 0:
            base_dirs.append(centroids[categories[i]])
        else:
            base_dirs.append(centroids[categories[i]] - centroids[categories[0]])
    
    orth_dirs = orthogonalize(base_dirs)
    
    log(f"  Categories used: {categories[:4]}")
    log(f"  {len(orth_dirs)} orthogonal directions constructed")
    log(f"  Injecting at {inj_count} layers")
    
    scale = 0.10
    
    configs = [
        ("1-dir (code)", [orth_dirs[0]]),
        ("2-dir (code+math)", orth_dirs[:2]),
        ("3-dir (code+math+chinese)", orth_dirs[:3]),
        ("1-dir (gen_en)", [centroids[categories[0]] / (centroids[categories[0]].norm() + 1e-8)]),
        ("2-dir (gen_en+code)", [
            centroids[categories[0]] / (centroids[categories[0]].norm() + 1e-8),
            orth_dirs[0]
        ]),
    ]
    
    results = []
    for name, dirs in configs:
        combined_dir = sum(dirs) / len(dirs)
        combined_dir = combined_dir / (combined_dir.norm() + 1e-8)
        
        cos_vals = []
        kl_vals = []
        text_changed = []
        
        for text, cat in test_texts:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                handles = []
                for lidx in inj_layer_indices:
                    h = actual_layers[lidx].register_forward_hook(make_inject_hook(combined_dir, scale))
                    handles.append(h)
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_logits = inj_out.logits[:, -1, :].float()
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                
                cos_vals.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
                kl_vals.append(kl_divergence(nat_logits, inj_logits))
                
                # Generate 5 tokens
                gen_ids = inputs["input_ids"].clone()
                handles = []
                for lidx in inj_layer_indices:
                    h = actual_layers[lidx].register_forward_hook(make_inject_hook(combined_dir, scale))
                    handles.append(h)
                try:
                    with torch.no_grad():
                        for _ in range(5):
                            out = model(input_ids=gen_ids)
                            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            if tok.item() == tokenizer.eos_token_id:
                                break
                            gen_ids = torch.cat([gen_ids, tok], dim=1)
                finally:
                    for h in handles:
                        h.remove()
                
                nat_ids = inputs["input_ids"].clone()
                with torch.no_grad():
                    for _ in range(5):
                        out = model(input_ids=nat_ids)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        nat_ids = torch.cat([nat_ids, tok], dim=1)
                
                inj_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
                text_changed.append(1 if nat_text.strip() != inj_text.strip() else 0)
            except:
                continue
        
        if cos_vals:
            n = len(cos_vals)
            r = {
                "config": name,
                "n_dirs": len(dirs),
                "1-cos": 1 - np.mean(cos_vals),
                "avg_kl": np.mean(kl_vals),
                "text_change_rate": sum(text_changed) / n * 100,
                "n": n,
            }
            results.append(r)
            log(f"  {name:30s}: 1-cos={r['1-cos']:.6f}, KL={r['avg_kl']:.4f}, "
                f"Text={r['text_change_rate']:.0f}%")
    
    # Test scale combinations
    log(f"\n  Scale interaction (2 dirs, different scale ratios):")
    for s1 in [0.05, 0.10, 0.20]:
        for s2 in [0.05, 0.10, 0.20]:
            cos_vals = []
            for text, cat in test_texts[:10]:
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
                    with torch.no_grad():
                        nat_out = model(**inputs, output_hidden_states=True)
                        nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                    handles = []
                    for lidx in inj_layer_indices:
                        # Combined direction with different scales
                        combined = orth_dirs[0] * s1 + orth_dirs[1] * s2
                        combined = combined / (combined.norm() + 1e-8)
                        h = actual_layers[lidx].register_forward_hook(make_inject_hook(combined, 0.10))
                        handles.append(h)
                    try:
                        with torch.no_grad():
                            inj_out = model(**inputs, output_hidden_states=True)
                            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                    finally:
                        for h in handles:
                            h.remove()
                    cos_vals.append(F.cosine_similarity(nat_h, inj_h, dim=-1).item())
                except:
                    continue
            if cos_vals:
                log(f"    s1={s1:.2f}, s2={s2:.2f}: 1-cos={1-np.mean(cos_vals):.6f}")
    
    return results


# ===== P120: 连续多步操控 =====
def p120_continuous_manipulation(model, tokenizer, n_layers, d_model, texts, centroids):
    """多步生成中持续注入, 测长文本引导能力"""
    log(f"\n{'='*70}")
    log(f"P120: Continuous Multi-Step Manipulation")
    log(f"{'='*70}")
    
    categories = sorted(centroids.keys())
    inj_count = n_layers // 4
    container = get_layers_container(model)
    layers_attr = None
    for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
        if hasattr(container, attr):
            layers_attr = attr
            break
    actual_layers = getattr(container, layers_attr)
    inj_layer_indices = list(range(inj_count))
    
    global model_device_global, actual_layers_global
    model_device_global = model.device
    actual_layers_global = actual_layers
    
    # Direction: code - gen_en
    if "code" in centroids and "gen_en" in centroids:
        direction = (centroids["code"] - centroids["gen_en"])
    else:
        direction = (centroids[categories[1]] - centroids[categories[0]])
    direction = direction / (direction.norm() + 1e-8)
    
    scale = 0.10
    n_gen_steps = [5, 10, 15, 20]
    
    test_texts = [(t, c) for t, c in texts if c == "gen_en"][:8]
    if not test_texts:
        test_texts = texts[:8]
    
    log(f"  Direction: {'code-gen_en' if 'code' in centroids else f'{categories[1]}-{categories[0]}'}")
    log(f"  Scale: {scale}, Injecting at {inj_count} layers")
    log(f"  Generation steps: {n_gen_steps}")
    log(f"  Test texts: {len(test_texts)} (from gen_en)")
    
    results = []
    
    for n_steps in n_gen_steps:
        cos_at_step = defaultdict(list)
        kl_at_step = defaultdict(list)
        ppl_inj_vals = []
        ppl_nat_vals = []
        
        for text, cat in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            
            # Injected generation
            inj_ids = inputs["input_ids"].clone()
            for step in range(n_steps):
                handles = []
                for lidx in inj_layer_indices:
                    h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, scale))
                    handles.append(h)
                try:
                    with torch.no_grad():
                        out = model(input_ids=inj_ids, output_hidden_states=True)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if step in [0, 4, 9, 14, 19]:
                        h_step = out.hidden_states[-1][:, -1, :].float()
                        cos_at_step[step].append(h_step.norm().item())
                finally:
                    for h in handles:
                        h.remove()
                if tok.item() == tokenizer.eos_token_id:
                    break
                inj_ids = torch.cat([inj_ids, tok], dim=1)
            
            # Natural generation
            nat_ids = inputs["input_ids"].clone()
            for step in range(n_steps):
                with torch.no_grad():
                    out = model(input_ids=nat_ids)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if tok.item() == tokenizer.eos_token_id:
                    break
                nat_ids = torch.cat([nat_ids, tok], dim=1)
            
            inj_text = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
            nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
            
            # PPL
            try:
                inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
                nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                ppl_inj_vals.append(inj_ppl)
                ppl_nat_vals.append(nat_ppl)
            except Exception as e:
                log(f"      PPL error: {e}")
                pass
            
            # Final hidden state comparison
            cos_final = 0.0
            kl_final = 0.0
            nat_prox = {}
            inj_prox = {}
            try:
                h_nat, _ = get_final_h(model, tokenizer, nat_text)
                h_inj, _ = get_final_h(model, tokenizer, inj_text)
                cos_final = F.cosine_similarity(h_nat.unsqueeze(0), h_inj.unsqueeze(0)).item()
                
                # KL of final token distribution
                inputs_nat = tokenizer(nat_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                inputs_inj = tokenizer(inj_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                with torch.no_grad():
                    nat_logits = model(**inputs_nat).logits[:, -1, :].float()
                    inj_logits = model(**inputs_inj).logits[:, -1, :].float()
                kl_final = kl_divergence(nat_logits, inj_logits)
                
                text_diff = 1 if nat_text.strip() != inj_text.strip() else 0
                
                # Category proximity shift
                for c in categories:
                    if c in centroids:
                        nat_prox[c] = F.cosine_similarity(h_nat.unsqueeze(0), centroids[c].unsqueeze(0)).item()
                        inj_prox[c] = F.cosine_similarity(h_inj.unsqueeze(0), centroids[c].unsqueeze(0)).item()
            except Exception as e:
                log(f"      Analysis error: {e}")
                text_diff = 0
            
            # Save sample texts
            if len(results) == 0 or True:
                log(f"\n  Sample (n_steps={n_steps}):")
                log(f"    NAT: {nat_text[:150]}...")
                log(f"    INJ: {inj_text[:150]}...")
                log(f"    cos_final={cos_final:.4f}, KL={kl_final:.4f}, diff={text_diff}")
                if "code" in centroids and "gen_en" in centroids:
                    log(f"    nat_prox: gen_en={nat_prox.get('gen_en',0):.4f}, code={nat_prox.get('code',0):.4f}")
                    log(f"    inj_prox: gen_en={inj_prox.get('gen_en',0):.4f}, code={inj_prox.get('code',0):.4f}")
        
        # Aggregate
        avg_inj_ppl = np.mean(ppl_inj_vals) if ppl_inj_vals else float('inf')
        avg_nat_ppl = np.mean(ppl_nat_vals) if ppl_nat_vals else float('inf')
        r = {
            "n_steps": n_steps,
            "avg_inj_ppl": avg_inj_ppl,
            "avg_nat_ppl": avg_nat_ppl,
            "ppl_delta": avg_inj_ppl - avg_nat_ppl,
            "n": len(test_texts),
        }
        results.append(r)
        log(f"\n  n_steps={n_steps:2d}: inj_PPL={avg_inj_ppl:.1f}, nat_PPL={avg_nat_ppl:.1f}, "
            f"dPPL={r['ppl_delta']:+.1f}")
    
    # Test different scales for continuous injection
    log(f"\n  Scale sweep for continuous 10-step generation:")
    for s in [0.05, 0.08, 0.10, 0.12, 0.15]:
        cos_vals = []
        kl_vals = []
        text_changed = []
        
        for text, cat in test_texts[:6]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            
            inj_ids = inputs["input_ids"].clone()
            for step in range(10):
                handles = []
                for lidx in inj_layer_indices:
                    h = actual_layers[lidx].register_forward_hook(make_inject_hook(direction, s))
                    handles.append(h)
                try:
                    with torch.no_grad():
                        out = model(input_ids=inj_ids)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                finally:
                    for h in handles:
                        h.remove()
                if tok.item() == tokenizer.eos_token_id:
                    break
                inj_ids = torch.cat([inj_ids, tok], dim=1)
            
            nat_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for step in range(10):
                    out = model(input_ids=nat_ids)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if tok.item() == tokenizer.eos_token_id:
                        break
                    nat_ids = torch.cat([nat_ids, tok], dim=1)
            
            inj_text = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
            nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
            text_changed.append(1 if nat_text.strip() != inj_text.strip() else 0)
            
            h_inj, _ = get_final_h(model, tokenizer, inj_text)
            cos_vals.append(F.cosine_similarity(nat_h.unsqueeze(0), h_inj.unsqueeze(0)).item())
            
            inputs_inj = tokenizer(inj_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                inj_logits = model(**inputs_inj).logits[:, -1, :].float()
            kl_vals.append(kl_divergence(
                nat_out.logits[:, -1, :].float(), inj_logits
            ))
        
        if cos_vals:
            log(f"    scale={s:.2f}: 1-cos={1-np.mean(cos_vals):.6f}, "
                f"KL={np.mean(kl_vals):.4f}, Text={sum(text_changed)/len(text_changed)*100:.0f}%")
    
    return results


# ===== Main =====
def main():
    global log, model_device_global, actual_layers_global
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="glm4",
                       choices=list(MODEL_MAP.keys()) + ["all"])
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    
    models_to_test = MODEL_ORDER if args.model == "all" else [args.model]
    
    texts = build_texts()
    print(f"Dataset: {len(texts)} texts, {len(set(t[1] for t in texts))} categories")
    
    for mname in models_to_test:
        print(f"\n{'#'*70}")
        print(f"# {mname.upper()} — Phase XV")
        print(f"{'#'*70}")
        
        out_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage720_phase15_{mname}_{ts}"
        log = Logger(out_dir, "results")
        log(f"Stage 720 Phase XV — {mname}")
        log(f"Time: {ts}")
        
        model, tokenizer, n_layers, d_model = load_model(mname)
        model_device_global = model.device
        
        container = get_layers_container(model)
        layers_attr = None
        for attr in ['layers', 'decoder_layers', 'encoder_layers', 'block']:
            if hasattr(container, attr):
                layers_attr = attr
                break
        actual_layers_global = getattr(container, layers_attr)
        
        # Compute centroids
        log(f"\nComputing centroids...")
        centroids = compute_centroids(model, tokenizer, texts, n_samples=40)
        log(f"  Categories: {sorted(centroids.keys())}")
        
        # Compute centroid distances
        cats = sorted(centroids.keys())
        log(f"\n  Centroid distances (cosine):")
        for i in range(min(len(cats), 6)):
            for j in range(i+1, min(len(cats), 6)):
                d = F.cosine_similarity(centroids[cats[i]].unsqueeze(0), centroids[cats[j]].unsqueeze(0)).item()
                log(f"    {cats[i]:>10s} vs {cats[j]:>10s}: cos={d:.4f}")
        
        # Run experiments
        t0 = time.time()
        
        p117_semantic_verification(model, tokenizer, n_layers, d_model, texts, centroids)
        
        p118_layer_ablation(model, tokenizer, n_layers, d_model, texts, centroids)
        
        p119_multi_direction(model, tokenizer, n_layers, d_model, texts, centroids)
        
        p120_continuous_manipulation(model, tokenizer, n_layers, d_model, texts, centroids)
        
        elapsed = time.time() - t0
        log(f"\n{'='*70}")
        log(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        log(f"\n  Done with {mname}")
        log.close()
        log = None
        
        # Free GPU
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU freed for {mname}. Waiting 30s before next model...")
        time.sleep(30)


if __name__ == "__main__":
    main()
