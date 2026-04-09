#!/usr/bin/env python3
"""
Stage 723: Phase XVIII — 突破操控不可能三角
================================================================================
Phase XVII核心发现:
- 精确梯度方向与centroid弱正相关(cos≈0.17), 不同目标类别的梯度几乎相同(cos>0.94)
- 操控子空间维度: DS7B(2维50%方差) << GLM4(5维50%方差)
- 累积注入近似线性(GLM4: ~0.002/层), DS7B完全拮抗
- 最后3层注入无效, 甜蜜操控窗口: scale∈[0.08, 0.15]
- "操控不可能三角": 高方向性-低质量损失-高成功率无法同时满足

Phase XVIII目标: 突破操控瓶颈
P131: 跨层差异化注入 — 不同层注入不同方向(PC0前半, PC1后半), 测量协同效果
P132: Logit空间直接操控 — 结合hidden state注入+logit空间定向修改
P133: Norm bypass操控 — 在RMSNorm之前/之后注入, 绕过"语义固化层"
P134: 二阶Hessian方向 — 计算Hessian对角线, 用牛顿法更新操控方向
P135: Adversarial多步迭代 — PGD式多步优化, 在logit空间直接最大化目标类别概率

用法: python stage723_phase18.py --model glm4
      python stage723_phase18.py --model qwen3
      python stage723_phase18.py --model deepseek7b
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
        "The quadratic formula solves ax^2+bx+c=0.", "Quantum entanglement connects distant particles.",
        "The Navier-Stokes equations describe fluid dynamics.", "Fermat's Last Theorem was proven by Andrew Wiles.",
        "The Riemann hypothesis remains unproven.", "Thermodynamics governs energy conservation.",
        "The periodic table organizes chemical elements.", "Bayesian inference updates probability distributions.",
        "The Fourier transform converts time to frequency domain.", "Topology studies properties preserved under deformation.",
        "Godel's incompleteness theorem limits formal systems.", "The standard model describes fundamental particles.",
        "Catalan numbers count binary trees.", "Group theory classifies symmetries.",
        "Information theory defines entropy as H=-sum(p*log(p)).", "The Boltzmann distribution describes statistical mechanics.",
        "Lambda calculus provides a foundation for computation.", "The Goldbach conjecture proposes that every even number is the sum of two primes.",
        "Neural networks learn through gradient descent.", "The Drake equation estimates alien civilizations.",
        "Chaos theory shows sensitive dependence on initial conditions.", "Graph theory studies networks of nodes and edges.",
        "The Pythagorean theorem: a^2+b^2=c^2.", "Differential equations model continuous change.",
        "Number theory studies integer properties.", "Machine learning finds patterns in data.",
        "The double-slit experiment demonstrates wave-particle duality.", "Category theory abstracts mathematical structures.",
        "Statistical mechanics bridges microscopic to macroscopic.", "The halting problem is undecidable.",
        "Euler's identity: e^(i*pi)+1=0.", "Linear algebra studies vector spaces and linear maps.",
        "The scientific method requires hypothesis testing.", "Cryptography secures digital communications.",
        "Game theory analyzes strategic decision making.", "Computational complexity classifies problem difficulty.",
        "DNA encodes genetic information in base pairs.", "The Higgs boson was confirmed in 2012.",
    ]
    for t in math_sci: T.append((t, "math_sci"))
    poetry = [
        "Roses are red, violets are blue.", "The road not taken diverged in a yellow wood.",
        "Shall I compare thee to a summer's day?", "Two roads diverged in a wood, and I took the one less traveled.",
        "The fog comes on little cat feet.", "Hope is the thing with feathers that perches in the soul.",
        "In Xanadu did Kubla Khan a stately pleasure-dome decree.", "Do not go gentle into that good night.",
        "The Waste Land is T.S. Eliot's masterpiece.", "Emily Dickinson wrote about death and immortality.",
        "Haiku: an old silent pond / a frog jumps into the pond / splash! silence again.",
        "The Raven by Edgar Allan Poe features nevermore.", "i carry your heart with me by e.e. cummings.",
        "Daffodils by William Wordsworth celebrates nature.", "The Love Song of J. Alfred Prufrock is by T.S. Eliot.",
        "Ode to a Nightingale by John Keats explores beauty.", "Robert Frost won four Pulitzer Prizes for poetry.",
        "Free verse poetry breaks traditional meter.", "Sonnet 18 is Shakespeare's most famous poem.",
        "The Lake Isle of Innisfree by Yeats yearns for escape.", "Langston Hughes captured the Harlem Renaissance.",
        "Poetry uses metaphor, simile, and imagery.", "Alliteration repeats initial consonant sounds.",
        "Blank verse has meter but no rhyme.", "Sylvia Plath explored mental illness in her poetry.",
        "The Odyssey is an epic poem by Homer.", "Maya Angelou wrote Still I Rise.",
        "Concrete poetry uses visual arrangement of text.", "Rhyme scheme patterns include ABAB and AABB.",
        "Enjambment carries meaning across line breaks.", "Apostrophe addresses absent or dead persons.",
        "Onomatopoeia uses words that sound like their meaning.", "William Blake was both poet and artist.",
        "Limericks are humorous five-line poems with AABBA rhyme.", "War poetry by Wilfred Owen exposes horror.",
        "Haiku follows a 5-7-5 syllable structure.", "The Iliad tells the story of the Trojan War.",
        "Walt Whitman pioneered free verse in America.", "Metrical patterns create rhythm in poetry.",
        "Symbolism uses objects to represent abstract ideas.",
    ]
    for t in poetry: T.append((t, "poetry"))
    code = [
        "def fibonacci(n): return n if n<2 else fibonacci(n-1)+fibonacci(n-2)",
        "for i in range(len(arr)): print(arr[i])",
        "class Node: def __init__(self, val): self.val=val; self.next=None",
        "x = [i**2 for i in range(10)]", "import numpy as np; A=np.random.randn(3,3)",
        "def quicksort(arr): if len(arr)<=1: return arr; pivot=arr[0]",
        "SELECT * FROM users WHERE age > 18 ORDER BY name;",
        "git commit -m 'fix: resolve null pointer exception'",
        "docker run -d -p 8080:80 nginx", "npm install express --save",
        "const sum = arr.reduce((a,b) => a+b, 0);",
        "public static void main(String[] args) { System.out.println('Hello'); }",
        "func handler(w http.ResponseWriter, r *http.Request) { fmt.Fprintf(w, 'OK') }",
        "print(f'The answer is {42}')", "x = torch.randn(2, 3, requires_grad=True)",
        "model.fit(X_train, y_train, epochs=100, batch_size=32)",
        "var xhr = new XMLHttpRequest(); xhr.open('GET', '/api/data');",
        "CREATE TABLE IF NOT EXISTS orders (id INT PRIMARY KEY, total DECIMAL);",
        "python -m pytest tests/ -v --cov=src", "curl -X POST -H 'Content-Type: application/json' -d '{}' localhost:5000/api",
        "async function fetchData() { const res = await fetch(url); return res.json(); }",
        "HashMap<String, Integer> map = new HashMap<>(); map.put('key', 42);",
        "float angle = atan2(y, x) * 180 / PI;",
        "df.groupby('category').agg({'price': ['mean', 'std']})",
        "git rebase -i HEAD~5", "docker-compose up -d --build",
        "int** matrix = (int**)malloc(n * sizeof(int*));",
        "from flask import Flask; app = Flask(__name__)",
        "pthread_create(&thread, NULL, worker, arg);",
        "try: result = json.loads(response.text) except: pass",
        "module.exports = router; app.use('/api', router);",
        "ALTER TABLE users ADD COLUMN email VARCHAR(255);",
        "ssh -i ~/.ssh/key.pem user@host", "for node in cluster: node.send(msg)",
        "val df = spark.read.parquet('data/input/')", "redis.set('cache:key', json.dumps(value))",
        "rm -rf build/ && cmake .. && make -j4",
    ]
    for t in code: T.append((t, "code"))
    chinese = [
        "人工智能正在改变世界。",
        "中国的经济发展取得了显著成就。",
        "教育是国之大计，党之大计。",
        "科技自立自强是国家发展的战略支撑。",
        "文化自信是一个民族最基本的力量。",
        "绿水青山就是金山银山。",
        "人民对美好生活的向往就是我们的奋斗目标。",
        "创新是引领发展的第一动力。",
        "数字经济成为新的增长引擎。",
        "乡村振兴战略全面推进。",
        "高质量发展是时代的要求。",
        "对外开放的基本国策不会改变。",
        "中国式现代化道路越走越宽广。",
        "碳中和目标推动能源转型。",
        "中医药传承创新发展。",
        "传统文化的创造性转化和创新性发展。",
        "国家治理体系和治理能力现代化。",
        "构建人类命运共同体。",
        "一带一路倡议促进共同发展。",
        "太空探索取得重大突破。",
        "量子计算研究达到世界先进水平。",
        "基因编辑技术造福人类健康。",
        "新能源汽车产业蓬勃发展。",
        "5G网络覆盖不断扩展。",
        "芯片自主研发持续突破。",
        "航天员完成太空行走任务。",
        "深海探测达到万米深度。",
        "大数据分析助力科学决策。",
        "智慧城市建设改善民生。",
        "教育均衡发展缩小城乡差距。",
        "医疗保障体系不断完善。",
        "养老服务体系日益健全。",
        "粮食安全是国家安全的重要基础。",
        "生态环境持续改善。",
        "文化遗产保护传承力度加大。",
        "青年是国家的未来和希望。",
        "互联网普及率持续提高。",
        "移动支付便捷人们生活。",
        "电子商务改变消费方式。",
        "在线教育打破时空限制。",
    ]
    for t in chinese: T.append((t, "chinese"))
    philosophy = [
        "I think, therefore I am.", "The unexamined life is not worth living.",
        "To be or not to be, that is the question.", "Knowledge is power.",
        "The only true wisdom is in knowing you know nothing.", "Existence precedes essence.",
        "Man is condemned to be free.", "The categorical imperative demands universal maxims.",
        "All perception is theory-laden.", "Truth is correspondence between thought and reality.",
        "Utilitarianism maximizes happiness.", "The social contract governs political legitimacy.",
        "Phenomenology studies structures of experience.", "Hermeneutics is the art of interpretation.",
        "Nihilism denies inherent meaning.", "Pragmatism judges truth by practical consequences.",
        "Rationalism prioritizes reason as source of knowledge.", "Empiricism derives knowledge from experience.",
        "The mind-body problem asks how mental relates to physical.", "Free will debates determinism versus agency.",
        "Moral relativism versus moral absolutism.", "The problem of evil challenges theism.",
        "Epistemology studies the nature of knowledge.", "Metaphysics examines the fundamental nature of reality.",
        "Consciousness remains the hard problem of philosophy.", "Personal identity questions persistence over time.",
        "Ethics weighs right versus wrong actions.", "Aesthetics explores beauty and artistic value.",
        "Political philosophy examines justice and power.", "Language philosophy asks how words have meaning.",
        "Logic provides formal systems for valid reasoning.", "Philosophy of science examines scientific methodology.",
        "Eastern philosophy emphasizes harmony and balance.", "Stoicism teaches virtue as the highest good.",
        "Skepticism questions whether knowledge is possible.", "Idealism claims reality is fundamentally mental.",
        "Materialism asserts that matter is fundamental.", "Dualism separates mind from body.",
        "Phenomenological reduction brackets assumptions.", "Structuralism analyzes underlying patterns.",
        "Postmodernism questions grand narratives.", "The Turing test evaluates machine intelligence.",
        "Chinese philosophy: yin and yang balance.", "Buddhist philosophy examines suffering and liberation.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
    return T

# ===== Model paths =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}


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


def get_texts_with_h(model, tokenizer, texts, n_per_cat=40):
    """Get hidden states for texts."""
    cat_h = defaultdict(list)
    indices_per_cat = defaultdict(list)
    for i, (t, c) in enumerate(texts):
        if len(indices_per_cat[c]) < n_per_cat:
            indices_per_cat[c].append(i)
    for c, idxs in indices_per_cat.items():
        for idx in idxs:
            t = texts[idx][0]
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
            cat_h[c].append(h)
    return cat_h


def compute_centroids(cat_h):
    centroids = {}
    for c, hs in cat_h.items():
        if len(hs) > 0:
            centroids[c] = torch.stack(hs).mean(dim=0)
    return centroids


def compute_pca_directions(cat_h, centroids, n_components=8):
    """Compute PCA directions across all hidden states."""
    all_h = []
    for c, hs in cat_h.items():
        all_h.extend(hs)
    H = torch.stack(all_h)  # [N, d]
    # Center
    mean = H.mean(dim=0)
    H_c = H - mean
    # SVD
    U, S, Vt = torch.linalg.svd(H_c, full_matrices=False)
    return Vt[:n_components]  # [n_components, d]


# =====================================================================
# P131: Cross-layer differentiated injection
# =====================================================================
def p131_cross_layer_diff(model, tokenizer, texts, centroids, n_layers, d_model, uw, ub):
    log("\n" + "="*80)
    log("P131: Cross-layer differentiated injection")
    log("="*80)
    
    # Use PC0 direction from PCA of hidden states
    pca_dirs = compute_pca_directions(
        get_texts_with_h(model, tokenizer, texts, n_per_cat=40),
        centroids
    )
    pc0 = pca_dirs[0]  # highest variance direction
    pc1 = pca_dirs[1]  # second
    
    # Also get centroid difference direction
    if "code" in centroids and "gen_en" in centroids:
        centroid_dir = centroids["code"] - centroids["gen_en"]
        centroid_dir = centroid_dir / centroid_dir.norm()
    
    dev = next(model.parameters()).device
    
    test_texts = [t for t, c in texts if c == "code"][:8]
    
    injection_strategies = {
        "uniform_pc0": [(0.0, 1.0, pc0)],  # all layers same direction
        "uniform_centroid": [(0.0, 1.0, centroid_dir)],
        "front_pc0_back_pc1": [(0.0, 0.5, pc0), (0.5, 1.0, pc1)],  # different directions
        "front_pc0_back_neg_pc0": [(0.0, 0.5, pc0), (0.5, 1.0, -pc0)],  # opposing
        "front_pc0_back_zero": [(0.0, 0.5, pc0), (0.5, 1.0, torch.zeros_like(pc0))],  # decay
        "increasing_scale": None,  # build below
        "alternating_sign": None,  # build below
        "front_centroid_back_neg": [(0.0, 0.5, centroid_dir), (0.5, 1.0, -centroid_dir)],
    }
    
    # Build increasing scale strategy
    inc_segments = []
    for i in range(5):
        frac_start = i / 5
        frac_end = (i + 1) / 5
        scale = (i + 1) / 5  # increasing scale
        inc_segments.append((frac_start, frac_end, pc0 * scale))
    injection_strategies["increasing_scale"] = inc_segments
    
    # Build alternating sign strategy
    alt_segments = []
    for i in range(5):
        sign = 1 if i % 2 == 0 else -1
        frac_start = i / 5
        frac_end = (i + 1) / 5
        alt_segments.append((frac_start, frac_end, pc0 * sign))
    injection_strategies["alternating_sign"] = alt_segments
    
    scales = [0.05, 0.10, 0.15]
    
    log(f"\n  Testing {len(injection_strategies)} injection strategies x {len(scales)} scales x {len(test_texts)} texts")
    
    for strat_name, segments in injection_strategies.items():
        for scale in scales:
            one_cos_list = []
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                # Get natural hidden state
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                nat_h = nat_out.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
                
                # Inject at each layer based on strategy
                def make_hook(segments, scale, dev):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                        else:
                            h = output
                        # Determine which segment this layer falls into
                        # We don't know the exact layer index from the hook,
                        # so we'll use a different approach
                        return output
                    return hook_fn
                
                # Use output manipulation instead: inject into intermediate hidden states
                # by modifying the model's forward pass layer by layer
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    all_hs = [h[:, -1, :].float() for h in outputs.hidden_states]
                
                # Apply differentiated injection per layer group
                h_inj = all_hs[0].cpu().clone()  # [1, d]
                for layer_idx in range(1, min(n_layers, len(all_hs))):
                    frac = layer_idx / n_layers
                    # Find which segment this layer falls into
                    delta = torch.zeros(1, d_model)  # on CPU
                    for seg_start, seg_end, seg_dir in segments:
                        if seg_start <= frac < seg_end:
                            delta[0] = seg_dir.cpu() * scale
                            break
                    h_inj = all_hs[layer_idx].cpu() + delta  # both on CPU
                
                # Compute cos similarity change
                inj_h = h_inj.squeeze(0)
                # Compare to target centroid (gen_en)
                if "gen_en" in centroids:
                    nat_cos = F.cosine_similarity(nat_h.unsqueeze(0), centroids["gen_en"].unsqueeze(0))
                    inj_cos = F.cosine_similarity(inj_h.unsqueeze(0), centroids["gen_en"].unsqueeze(0))
                    one_cos_list.append((inj_cos.item() - nat_cos.item()))
            
            avg_change = np.mean(one_cos_list) if one_cos_list else 0
            log(f"  {strat_name:30s} scale={scale:.2f}: avg_cos_change={avg_change:.6f}  n={len(one_cos_list)}")


# =====================================================================
# P132: Logit-space direct manipulation + hidden state injection
# =====================================================================
def p132_logit_manipulation(model, tokenizer, texts, centroids, uw, ub):
    log("\n" + "="*80)
    log("P132: Logit-space direct manipulation + hidden state injection")
    log("="*80)
    
    if uw is None:
        log("  SKIP: No unembed matrix found")
        return
    
    dev = next(model.parameters()).device
    uw_dev = uw.to(dev)
    ub_dev = ub.to(dev) if ub is not None else None
    
    # For each category, find top-K representative tokens
    cat_top_tokens = {}
    for cat, centroid in centroids.items():
        centroid_dev = centroid.to(dev)
        cos_sims = F.cosine_similarity(centroid_dev.unsqueeze(0), uw_dev, dim=1)
        topk = cos_sims.topk(100)
        cat_top_tokens[cat] = topk.indices.tolist()
    
    test_texts = [t for t, c in texts if c == "code"][:10]
    
    strategies = [
        ("none", 0.0, 0.0),        # no manipulation
        ("hs_only", 0.10, 0.0),     # hidden state only
        ("logit_boost_50", 0.0, 0.5),  # boost target logits
        ("logit_suppress_50", 0.0, -0.5),  # suppress source logits
        ("hs+boost_50", 0.10, 0.5), # both
        ("hs+suppress_50", 0.10, -0.5),  # both
        ("boost_100", 0.0, 1.0),    # larger boost
        ("suppress_100", 0.0, -1.0),
        ("hs+boost_100", 0.10, 1.0),
        ("logit_temp_low", 0.0, 0.0),  # temperature reduction
    ]
    
    target_cat = "gen_en"
    source_cat = "code"
    
    for strat_name, hs_scale, logit_scale in strategies:
        if strat_name == "logit_temp_low":
            logit_scale = 0.0
            temp_factor = 0.5
        else:
            temp_factor = 1.0
        
        results = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                nat_h = outputs.hidden_states[-1][:, -1, :].float()  # [1, d]
                nat_logits = outputs.logits[:, -1, :].float()  # [1, vocab]
            
            # 1. Hidden state injection
            if hs_scale > 0 and source_cat in centroids and target_cat in centroids:
                centroid_diff = (centroids[target_cat] - centroids[source_cat]).to(dev)
                centroid_diff = centroid_diff / centroid_diff.norm()
                inj_h = nat_h + centroid_diff.unsqueeze(0) * hs_scale
            else:
                inj_h = nat_h
            
            # 2. Compute logits from injected h
            inj_logits = F.linear(inj_h, uw_dev, ub_dev)  # [1, vocab]
            
            # 3. Logit manipulation
            if logit_scale != 0.0 and target_cat in cat_top_tokens and source_cat in cat_top_tokens:
                logit_delta = torch.zeros_like(inj_logits)
                # Boost target tokens
                target_ids = cat_top_tokens[target_cat]
                logit_delta[0, target_ids] += logit_scale * 0.1
                # Suppress source tokens
                source_ids = cat_top_tokens[source_cat]
                logit_delta[0, source_ids] -= logit_scale * 0.1
                inj_logits = inj_logits + logit_delta
            
            # 4. Temperature
            if temp_factor != 1.0:
                inj_logits = inj_logits / temp_factor
            
            # Evaluate
            nat_probs = F.softmax(nat_logits, dim=-1)
            inj_probs = F.softmax(inj_logits, dim=-1)
            
            # KL divergence
            kl = F.kl_div(inj_probs.log(), nat_probs, reduction='batchmean').item()
            
            # Top-5 change
            nat_top5 = nat_logits[0].topk(5).indices.tolist()
            inj_top5 = inj_logits[0].topk(5).indices.tolist()
            top5_change = len(set(nat_top5) - set(inj_top5))
            
            # Target probability increase
            if target_cat in cat_top_tokens:
                nat_target_prob = nat_probs[0, cat_top_tokens[target_cat]].sum().item()
                inj_target_prob = inj_probs[0, cat_top_tokens[target_cat]].sum().item()
                target_prob_change = inj_target_prob - nat_target_prob
            else:
                target_prob_change = 0
            
            results.append({
                "kl": kl,
                "top5_change": top5_change,
                "target_prob_change": target_prob_change,
            })
        
        avg_kl = np.mean([r["kl"] for r in results])
        avg_top5 = np.mean([r["top5_change"] for r in results])
        avg_tpc = np.mean([r["target_prob_change"] for r in results])
        
        nat_top5_tokens = tokenizer.convert_ids_to_tokens(nat_top5) if 'nat_top5' in dir() else []
        inj_top5_tokens = tokenizer.convert_ids_to_tokens(inj_top5) if 'inj_top5' in dir() else []
        
        log(f"  {strat_name:25s}: KL={avg_kl:.4f}  top5_change={avg_top5:.1f}  "
            f"target_prob_delta={avg_tpc:+.6f}")


# =====================================================================
# P133: Norm bypass manipulation
# =====================================================================
def p133_norm_bypass(model, tokenizer, texts, centroids, n_layers, d_model):
    log("\n" + "="*80)
    log("P133: Norm bypass manipulation")
    log("="*80)
    
    dev = next(model.parameters()).device
    
    # Check architecture for norm layers
    has_rmsnorm = False
    has_layernorm = False
    norm_positions = []
    
    for name, module in model.named_modules():
        if 'norm' in name.lower() or 'ln' in name.lower():
            if 'RMSNorm' in str(type(module)):
                has_rmsnorm = True
            elif 'LayerNorm' in str(type(module)):
                has_layernorm = True
            norm_positions.append(name)
    
    log(f"  Architecture: RMSNorm={has_rmsnorm}, LayerNorm={has_layernorm}")
    log(f"  Found {len(norm_positions)} norm layers")
    if len(norm_positions) <= 15:
        for p in norm_positions:
            log(f"    {p}")
    else:
        for p in norm_positions[:5]:
            log(f"    {p}")
        log(f"    ... ({len(norm_positions)-5} more)")
    
    # Strategy: compare injection before vs after final norm
    test_texts = [t for t, c in texts if c == "code"][:8]
    
    if "code" in centroids and "gen_en" in centroids:
        direction = centroids["code"] - centroids["gen_en"]
        direction = direction / direction.norm()
    
    scales = [0.05, 0.10, 0.15, 0.20]
    
    # For each text, compare:
    # 1. Standard injection at last layer
    # 2. Injection scaled by layer norm value
    # 3. Injection with norm compensation (renormalize after injection)
    
    for scale in scales:
        results_std = []
        results_norm_comp = []
        results_norm_aware = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                nat_h = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
            
            # Standard injection
            inj_h_std = nat_h + direction.cpu() * scale
            nat_norm = nat_h.norm().item()
            inj_norm = inj_h_std.norm().item()
            
            # Norm-compensated injection (inject, then renormalize to original norm)
            inj_h_normcomp = nat_h + direction.cpu() * scale
            inj_h_normcomp = inj_h_normcomp * (nat_norm / max(inj_h_normcomp.norm().item(), 1e-8))
            
            # Norm-aware injection (scale direction by layer's expected norm change)
            # The idea: instead of adding a fixed offset, multiply by a rotation matrix
            # that preserves norm but changes direction
            # We approximate this by projecting the direction to be perpendicular to nat_h
            dir_perp = direction.cpu() - F.cosine_similarity(
                direction.cpu().unsqueeze(0), nat_h.unsqueeze(0), dim=1
            ).item() * nat_h / nat_norm * nat_norm
            inj_h_naware = nat_h + dir_perp * scale
            inj_h_naware = inj_h_naware * (nat_norm / max(inj_h_naware.norm().item(), 1e-8))
            
            if "gen_en" in centroids:
                nat_cos = F.cosine_similarity(nat_h.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                
                std_cos = F.cosine_similarity(inj_h_std.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                results_std.append(std_cos - nat_cos)
                
                nc_cos = F.cosine_similarity(inj_h_normcomp.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                results_norm_comp.append(nc_cos - nat_cos)
                
                na_cos = F.cosine_similarity(inj_h_naware.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                results_norm_aware.append(na_cos - nat_cos)
        
        if results_std:
            log(f"  scale={scale:.2f}: std={np.mean(results_std):.6f}  "
                f"norm_comp={np.mean(results_norm_comp):.6f}  "
                f"norm_aware={np.mean(results_norm_aware):.6f}")
    
    # Layer-by-layer norm analysis
    log(f"\n  Layer-by-layer norm analysis:")
    text = test_texts[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    for i in range(min(n_layers, len(outputs.hidden_states))):
        h = outputs.hidden_states[i][:, -1, :].float()
        norm_val = h.norm().item()
        if i % 5 == 0 or i == n_layers - 1:
            log(f"    L{i:2d}: norm={norm_val:.4f}")


# =====================================================================
# P134: Second-order Hessian direction
# =====================================================================
def p134_hessian_direction(model, tokenizer, texts, centroids, uw, ub, d_model):
    log("\n" + "="*80)
    log("P134: Second-order Hessian diagonal estimation")
    log("="*80)
    
    if uw is None:
        log("  SKIP: No unembed matrix found")
        return
    
    dev = next(model.parameters()).device
    uw_dev = uw.to(dev)
    ub_dev = ub.to(dev) if ub is not None else None
    
    # For a few texts, estimate Hessian diagonal of log P(target|h) w.r.t. h
    # H_ii = d^2 log P / dh_i^2 ≈ [log P(h + eps*e_i) - 2*log P(h) + log P(h - eps*e_i)] / eps^2
    
    test_texts = [t for t, c in texts if c == "code"][:5]
    eps = 0.01
    n_samples = min(200, d_model)  # sample dimensions for efficiency
    
    target_cat = "gen_en"
    if target_cat not in centroids:
        target_cat = list(centroids.keys())[0]
    
    centroid_dev = centroids[target_cat].to(dev)
    cos_sims = F.cosine_similarity(centroid_dev.unsqueeze(0), uw_dev, dim=1)
    topk_ids = cos_sims.topk(min(100, uw_dev.size(0))).indices
    
    log(f"  Target category: {target_cat}, top-{len(topk_ids)} tokens")
    log(f"  eps={eps}, sampling {n_samples} dimensions")
    
    for text_idx, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1][:, -1, :].float()  # [1, d]
        
        # Compute base log probability of target tokens
        logits = F.linear(h, uw_dev, ub_dev)
        log_probs = F.log_softmax(logits, dim=-1)
        base_lp = log_probs[0, topk_ids].mean().item()
        
        # Sample random dimensions
        rand_dims = torch.randperm(h.size(-1))[:n_samples]
        h_diag = torch.zeros(h.size(-1))
        
        for dim in rand_dims:
            # Forward perturbation
            h_plus = h.clone()
            h_plus[0, dim] += eps
            lp_plus = F.log_softmax(F.linear(h_plus, uw_dev, ub_dev), dim=-1)[0, topk_ids].mean().item()
            
            # Backward perturbation
            h_minus = h.clone()
            h_minus[0, dim] -= eps
            lp_minus = F.log_softmax(F.linear(h_minus, uw_dev, ub_dev), dim=-1)[0, topk_ids].mean().item()
            
            h_diag[dim] = (lp_plus - 2 * base_lp + lp_minus) / (eps * eps)
        
        # Analyze Hessian diagonal
        h_diag_vals = h_diag[h_diag != 0]  # non-zero entries
        if len(h_diag_vals) > 0:
            h_mean = h_diag_vals.mean().item()
            h_std = h_diag_vals.std().item()
            h_min = h_diag_vals.min().item()
            h_max = h_diag_vals.max().item()
            n_neg = (h_diag_vals < 0).sum().item()
            n_pos = (h_diag_vals >= 0).sum().item()
            
            log(f"  text[{text_idx}]: H_mean={h_mean:.4f}  std={h_std:.4f}  "
                f"min={h_min:.4f}  max={h_max:.4f}  neg:{n_neg}/{len(h_diag_vals)}")
            
            # Find dimensions with largest Hessian (most curvature = most important for manipulation)
            top_h_dims = h_diag.abs().topk(20)
            log(f"    Top-20 Hessian dimensions (abs): mean|H|={top_h_dims.values.mean():.4f}")
            
            # Newton step direction: -H^{-1} * gradient
            # Approximate: for diagonal Hessian, step_i = -gradient_i / H_ii
            # Compute gradient
            probs = F.softmax(logits, dim=-1)
            mean_target_vec = torch.zeros(uw_dev.size(0), device=dev)
            mean_target_vec[topk_ids] = 1.0 / len(topk_ids)
            grad = (uw_dev.T @ (mean_target_vec - probs.squeeze(0)))
            grad_norm = grad.norm().item()
            
            # Newton step (only for sampled dimensions where H_diag != 0)
            newton_step = torch.zeros_like(h_diag)
            mask = (h_diag != 0)
            newton_step[mask] = -grad[mask] / (h_diag[mask] + 1e-4)  # regularized
            
            # Compare gradient direction vs Newton direction
            newton_step_norm = newton_step.norm().item()
            if newton_step_norm > 1e-8 and grad_norm > 1e-8:
                cos_gn = F.cosine_similarity(grad.unsqueeze(0), newton_step.unsqueeze(0)).item()
                log(f"    cos(gradient, newton_step)={cos_gn:.4f}  "
                    f"|newton|={newton_step_norm:.4f}  |grad|={grad_norm:.4f}")
                
                # Test Newton direction effectiveness
                h_newton = h + newton_step.unsqueeze(0) * 0.1
                nat_h = h.squeeze(0).cpu()
                inj_h = h_newton.squeeze(0).cpu()
                if "gen_en" in centroids:
                    nat_cos = F.cosine_similarity(nat_h.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                    inj_cos = F.cosine_similarity(inj_h.unsqueeze(0), centroids["gen_en"].unsqueeze(0)).item()
                    log(f"    Newton cos_change={inj_cos - nat_cos:.6f}")


# =====================================================================
# P135: Adversarial multi-step iterative optimization (PGD-style)
# =====================================================================
def p135_adversarial_pgd(model, tokenizer, texts, centroids, uw, ub, d_model):
    log("\n" + "="*80)
    log("P135: Adversarial PGD-style multi-step optimization")
    log("="*80)
    
    if uw is None:
        log("  SKIP: No unembed matrix found")
        return
    
    dev = next(model.parameters()).device
    uw_dev = uw.to(dev)
    ub_dev = ub.to(dev) if ub is not None else None
    
    # PGD in hidden state space: maximize P(target tokens | h + delta)
    # But delta must be bounded by epsilon (max perturbation budget)
    
    test_texts = [t for t, c in texts if c == "code"][:6]
    
    configs = [
        ("pgd_grad", 10, 0.01, 0.10),   # 10 steps, step_size=0.01, eps=0.10
        ("pgd_grad", 20, 0.005, 0.10),   # 20 steps, smaller step
        ("pgd_grad", 10, 0.01, 0.20),    # larger budget
        ("pgd_grad", 20, 0.01, 0.20),    # more steps + larger budget
        ("pgd_newton", 10, 0.01, 0.10),  # Newton-guided
        ("pgd_centroid_init", 10, 0.01, 0.10),  # start from centroid direction
    ]
    
    for target_cat in ["gen_en", "poetry"]:
        if target_cat not in centroids:
            continue
        
        centroid_dev = centroids[target_cat].to(dev)
        cos_sims = F.cosine_similarity(centroid_dev.unsqueeze(0), uw_dev, dim=1)
        topk_ids = cos_sims.topk(min(100, uw_dev.size(0))).indices
        
        log(f"\n  Target: code -> {target_cat}")
        
        for config_name, n_steps, step_size, eps in configs:
            cos_changes = []
            ppl_changes = []
            
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    nat_h = outputs.hidden_states[-1][:, -1, :].float().clone()  # [1, d]
                
                # Natural perplexity
                nat_logits = F.linear(nat_h, uw_dev, ub_dev)
                nat_loss = F.cross_entropy(nat_logits, inputs["input_ids"][:, -1:].view(-1)).item()
                
                # Initialize delta
                if "centroid_init" in config_name and "code" in centroids:
                    delta = ((centroids[target_cat] - centroids["code"]).to(dev) * 0.05).unsqueeze(0)
                else:
                    delta = torch.zeros_like(nat_h)
                
                # PGD loop
                for step in range(n_steps):
                    h_adv = nat_h + delta
                    
                    # Compute gradient of target probability
                    logits = F.linear(h_adv, uw_dev, ub_dev)
                    probs = F.softmax(logits, dim=-1)
                    mean_target = torch.zeros(uw_dev.size(0), device=dev)
                    mean_target[topk_ids] = 1.0 / len(topk_ids)
                    grad = uw_dev.T @ (mean_target - probs.squeeze(0))  # [d]
                    grad = grad / max(grad.norm().item(), 1e-8)
                    
                    if "newton" in config_name:
                        # Scale by confidence (low confidence → bigger step)
                        target_prob = probs[0, topk_ids].mean().item()
                        adaptive_step = step_size * (1 - target_prob + 0.1)
                    else:
                        adaptive_step = step_size
                    
                    # Ascend step
                    delta = delta + adaptive_step * grad.unsqueeze(0)
                    
                    # Project back to epsilon ball
                    delta_norm = delta.norm().item()
                    if delta_norm > eps:
                        delta = delta * (eps / delta_norm)
                
                # Evaluate final result
                h_final = nat_h + delta
                final_logits = F.linear(h_final, uw_dev, ub_dev)
                final_probs = F.softmax(final_logits, dim=-1)
                
                nat_h_cpu = nat_h.squeeze(0).cpu()
                inj_h_cpu = h_final.squeeze(0).cpu()
                
                nat_cos = F.cosine_similarity(nat_h_cpu.unsqueeze(0), centroids[target_cat].unsqueeze(0)).item()
                inj_cos = F.cosine_similarity(inj_h_cpu.unsqueeze(0), centroids[target_cat].unsqueeze(0)).item()
                cos_changes.append(inj_cos - nat_cos)
                
                # PPL estimate from logits
                final_loss = F.cross_entropy(final_logits, inputs["input_ids"][:, -1:].view(-1)).item()
                ppl_changes.append(final_loss - nat_loss)
                
                # Target probability
                target_prob = final_probs[0, topk_ids].mean().item()
                nat_target = F.softmax(nat_logits, dim=-1)[0, topk_ids].mean().item()
            
            avg_cos = np.mean(cos_changes)
            avg_ppl = np.mean(ppl_changes)
            log(f"  {config_name:30s} steps={n_steps:2d} eps={eps:.2f}: "
                f"cos_change={avg_cos:+.6f}  loss_change={avg_ppl:+.4f}  "
                f"target_prob: {nat_target:.4f}->{target_prob:.4f}")
    
    # Also test: multi-target optimization (maximize code+poetry simultaneously)
    log(f"\n  Multi-target optimization (maximize P(poetry)+P(chinese) simultaneously):")
    multi_targets = ["poetry", "chinese"]
    multi_topk = []
    for tc in multi_targets:
        if tc in centroids:
            c_dev = centroids[tc].to(dev)
            cs = F.cosine_similarity(c_dev.unsqueeze(0), uw_dev, dim=1)
            multi_topk.extend(cs.topk(50).indices.tolist())
    multi_topk = list(set(multi_topk))[:100]
    
    if multi_topk:
        for eps in [0.05, 0.10, 0.15]:
            cos_changes = []
            for text in test_texts[:4]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    nat_h = outputs.hidden_states[-1][:, -1, :].float().clone()
                
                delta = torch.zeros_like(nat_h)
                for step in range(15):
                    h_adv = nat_h + delta
                    logits = F.linear(h_adv, uw_dev, ub_dev)
                    probs = F.softmax(logits, dim=-1)
                    mean_t = torch.zeros(uw_dev.size(0), device=dev)
                    mean_t[multi_topk] = 1.0 / len(multi_topk)
                    grad = uw_dev.T @ (mean_t - probs.squeeze(0))
                    grad = grad / max(grad.norm().item(), 1e-8)
                    delta = delta + 0.01 * grad.unsqueeze(0)
                    if delta.norm().item() > eps:
                        delta = delta * (eps / delta.norm().item())
                
                h_final = nat_h + delta
                for tc in multi_targets:
                    nat_c = F.cosine_similarity(nat_h.squeeze(0).cpu().unsqueeze(0), centroids[tc].unsqueeze(0)).item()
                    inj_c = F.cosine_similarity(h_final.squeeze(0).cpu().unsqueeze(0), centroids[tc].unsqueeze(0)).item()
                    cos_changes.append(inj_c - nat_c)
            
            log(f"    eps={eps:.2f}: avg_cos_change={np.mean(cos_changes):.6f}  (multi-target)")


# =====================================================================
# MAIN
# =====================================================================
def main():
    global log
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="glm4")
    args = parser.parse_args()
    
    models_to_run = [args.model]
    
    for mname in models_to_run:
        if mname not in MODEL_MAP:
            log_str = f"Model {mname} not found. Available: {list(MODEL_MAP.keys())}"
            print(log_str)
            continue
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage723_phase18_{mname}_{ts}"
        log = Logger(log_dir, "results")
        
        log(f"\n{'#'*80}")
        log(f"# Stage 723: Phase XVIII — Break manipulation impossible triangle")
        log(f"# Model: {mname}")
        log(f"# Time: {ts}")
        log(f"{'#'*80}")
        
        # Load model
        model, tokenizer, n_layers, d_model = load_model(mname)
        texts = build_texts()
        log(f"  Texts: {len(texts)} total")
        
        # Get centroids
        log(f"\n  Computing centroids...")
        cat_h = get_texts_with_h(model, tokenizer, texts, n_per_cat=40)
        centroids = compute_centroids(cat_h)
        log(f"  Centroids: {list(centroids.keys())}")
        
        # Get unembed
        uw, ub = get_unembed(model)
        
        # Run experiments
        try:
            p131_cross_layer_diff(model, tokenizer, texts, centroids, n_layers, d_model, uw, ub)
        except Exception as e:
            log(f"  P131 ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            p132_logit_manipulation(model, tokenizer, texts, centroids, uw, ub)
        except Exception as e:
            log(f"  P132 ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            p133_norm_bypass(model, tokenizer, texts, centroids, n_layers, d_model)
        except Exception as e:
            log(f"  P133 ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            p134_hessian_direction(model, tokenizer, texts, centroids, uw, ub, d_model)
        except Exception as e:
            log(f"  P134 ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            p135_adversarial_pgd(model, tokenizer, texts, centroids, uw, ub, d_model)
        except Exception as e:
            log(f"  P135 ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        log(f"\n{'#'*80}")
        log(f"# Phase XVIII complete for {mname}")
        log(f"{'#'*80}")
        log.close()
        
        # Free GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)


if __name__ == "__main__":
    main()
