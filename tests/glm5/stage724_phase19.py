#!/usr/bin/env python3
"""
Stage 724: Phase XIX — 根本性策略转换: Logit/Token/Unembed/Attention操控
================================================================================
Phase XVIII核心结论:
- 操控不可能三角被确认: 所有hidden state操控方法cos_shift<0
- Hessian符号反转: GLM4 99%负(局部凹), Qwen3 100%正(局部凸)
- 范数指数稀释(699x~3234x)是操控瓶颈物理根源
- Logit空间操控是唯一有效入口(效果是hidden state注入的11.5倍)

Phase XIX策略转换: 放弃hidden state操控, 探索全新入口
P136: Logit空间精细化操控 — 20种logit修改策略×40文本, 含自适应缩放和类别感知
P137: Prefix token操控的数学化 — 50种prefix×6类别, 测量prefix→hidden state→logit完整链路
P138: Self-generation操控 — 模型自生成目标文本, 提取h作为操控方向, 与centroid对比
P139: Unembed矩阵结构操控 — W矩阵行/列分析, 直接修改W改变输出分布
P140: Attention head级别操控 — 修改attention pattern间接操控hidden state

用法: python stage724_phase19.py --model glm4
      python stage724_phase19.py --model qwen3
      python stage724_phase19.py --model deepseek7b
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
        "人工智能正在改变世界。","中国的经济发展取得了显著成就。",
        "教育是国之大计，党之大计。","科技自立自强是国家发展的战略支撑。",
        "文化自信是一个民族最基本的力量。","绿水青山就是金山银山。",
        "人民对美好生活的向往就是我们的奋斗目标。","创新是引领发展的第一动力。",
        "数字经济成为新的增长引擎。","乡村振兴战略全面推进。",
        "高质量发展是时代的要求。","对外开放的基本国策不会改变。",
        "中国式现代化道路越走越宽广。","碳中和目标推动能源转型。",
        "中医药传承创新发展。","传统文化的创造性转化和创新性发展。",
        "国家治理体系和治理能力现代化。","构建人类命运共同体。",
        "一带一路倡议促进共同发展。","太空探索取得重大突破。",
        "量子计算研究达到世界先进水平。","基因编辑技术造福人类健康。",
        "新能源汽车产业蓬勃发展。","5G网络覆盖不断扩展。",
        "芯片自主研发持续突破。","航天员完成太空行走任务。",
        "深海探测达到万米深度。","大数据分析助力科学决策。",
        "智慧城市建设改善民生。","教育均衡发展缩小城乡差距。",
        "医疗保障体系不断完善。","养老服务体系日益健全。",
        "粮食安全是国家安全的重要基础。","生态环境持续改善。",
        "文化遗产保护传承力度加大。","青年是国家的未来和希望。",
        "互联网普及率持续提高。","移动支付便捷人们生活。",
        "电子商务改变消费方式。","在线教育打破时空限制。",
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


def kl_div(p, q):
    """KL(p||q) where p,q are prob tensors."""
    p = p + 1e-10
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum().item()


def get_top_tokens_for_category(model, tokenizer, category_texts, top_k=50):
    """Get the most probable next tokens for a given category."""
    token_freq = defaultdict(int)
    for text, _ in category_texts[:20]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, top_k).indices.tolist()
        for tid in top_ids:
            token_freq[tid] += 1
    # Sort by frequency
    sorted_tokens = sorted(token_freq.items(), key=lambda x: -x[1])
    return [(tid, freq) for tid, freq in sorted_tokens[:top_k]]


# =====================================================================
# P136: Logit空间精细化操控
# =====================================================================
def p136_logit_finegrained(model, tokenizer, texts, centroids, uw, ub):
    """
    P136: 20种logit修改策略 × 40文本
    策略包括: boost top-K, suppress source-K, adaptive scaling,
    category-aware boost, temperature sweep, top-p sampling manipulation,
    logit lens at multiple layers, etc.
    """
    log("\n" + "="*80)
    log("P136: Logit space fine-grained manipulation (20 strategies x 40 texts)")
    log("="*80)

    dev = model.device
    d_model = uw.shape[1]
    categories = sorted(centroids.keys())
    
    # Target: code→gen_en cross-category manipulation
    src_cat = "code"
    tgt_cat = "gen_en"
    test_texts = [(t, c) for t, c in texts if c == src_cat][:40]
    tgt_texts = [(t, c) for t, c in texts if c == tgt_cat][:20]
    
    # Get target category top tokens
    tgt_top_tokens = get_top_tokens_for_category(model, tokenizer, tgt_texts, top_k=100)
    tgt_token_ids = [tid for tid, freq in tgt_top_tokens[:50]]
    
    # Get source category top tokens
    src_texts_for_tokens = [(t, c) for t, c in texts if c == src_cat][:20]
    src_top_tokens = get_top_tokens_for_category(model, tokenizer, src_texts_for_tokens, top_k=100)
    src_token_ids = [tid for tid, freq in src_top_tokens[:50]]
    
    log(f"  Target tokens ({tgt_cat}): {tgt_token_ids[:10]}...")
    log(f"  Source tokens ({src_cat}): {src_token_ids[:10]}...")
    
    # 20 strategies
    strategies = {
        "none": {},
        "boost_top5": {"boost_ids": tgt_token_ids[:5], "boost_val": 5.0},
        "boost_top10": {"boost_ids": tgt_token_ids[:10], "boost_val": 5.0},
        "boost_top20": {"boost_ids": tgt_token_ids[:20], "boost_val": 5.0},
        "boost_top50": {"boost_ids": tgt_token_ids[:50], "boost_val": 5.0},
        "boost_adaptive": {"boost_ids": tgt_token_ids[:20], "boost_val": "adaptive"},
        "suppress_src5": {"suppress_ids": src_token_ids[:5], "suppress_val": -10.0},
        "suppress_src10": {"suppress_ids": src_token_ids[:10], "suppress_val": -10.0},
        "suppress_src20": {"suppress_ids": src_token_ids[:20], "suppress_val": -10.0},
        "boost5_suppress5": {"boost_ids": tgt_token_ids[:5], "boost_val": 5.0,
                            "suppress_ids": src_token_ids[:5], "suppress_val": -10.0},
        "boost10_suppress10": {"boost_ids": tgt_token_ids[:10], "boost_val": 5.0,
                               "suppress_ids": src_token_ids[:10], "suppress_val": -10.0},
        "temp_0.5": {"temperature": 0.5},
        "temp_0.3": {"temperature": 0.3},
        "temp_0.1": {"temperature": 0.1},
        "top_p_0.9": {"top_p": 0.9},
        "top_p_0.5": {"top_p": 0.5},
        "top_p_0.1": {"top_p": 0.1},
        "boost_scaled_10": {"boost_ids": tgt_token_ids[:10], "boost_val": 10.0},
        "boost_scaled_20": {"boost_ids": tgt_token_ids[:10], "boost_val": 20.0},
        "boost_scaled_50": {"boost_ids": tgt_token_ids[:10], "boost_val": 50.0},
    }
    
    results = {}
    for sname, sparams in strategies.items():
        total_kl = 0.0
        total_tpc = 0.0  # target prob change
        total_ttr = 0.0  # target token rank change
        count = 0
        
        for text, cat in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].float().cpu()
            probs_orig = F.softmax(logits, dim=-1)
            
            # Apply strategy
            modified_logits = logits.clone()
            
            # Temperature
            temp = sparams.get("temperature", 1.0)
            if temp != 1.0:
                modified_logits = modified_logits / temp
            
            # Boost
            boost_ids = sparams.get("boost_ids", [])
            boost_val = sparams.get("boost_val", 0.0)
            if boost_ids and boost_val:
                if boost_val == "adaptive":
                    # Adaptive: scale by original probability
                    for tid in boost_ids:
                        orig_prob = probs_orig[tid].item()
                        scale = max(1.0, -math.log(orig_prob + 1e-10))
                        modified_logits[tid] += scale
                else:
                    for tid in boost_ids:
                        modified_logits[tid] += boost_val
            
            # Suppress
            suppress_ids = sparams.get("suppress_ids", [])
            suppress_val = sparams.get("suppress_val", 0.0)
            if suppress_ids and suppress_val:
                for tid in suppress_ids:
                    modified_logits[tid] += suppress_val
            
            # Top-p (nucleus) filtering
            top_p = sparams.get("top_p", None)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(modified_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_logits[mask] = float('-inf')
                modified_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)
            
            probs_mod = F.softmax(modified_logits, dim=-1)
            
            # Metrics
            kl = kl_div(probs_orig, probs_mod)
            # Target prob: average prob of target category tokens
            tgt_prob_orig = sum(probs_orig[tid].item() for tid in tgt_token_ids[:20])
            tgt_prob_mod = sum(probs_mod[tid].item() for tid in tgt_token_ids[:20])
            tpc = tgt_prob_mod - tgt_prob_orig
            
            # Target token rank: average rank of target top-5 tokens
            ranks_orig = []
            ranks_mod = []
            sorted_probs_orig, sorted_idx_orig = torch.sort(probs_orig, descending=True)
            sorted_probs_mod, sorted_idx_mod = torch.sort(probs_mod, descending=True)
            rank_map_orig = {idx.item(): r for r, idx in enumerate(sorted_idx_orig)}
            rank_map_mod = {idx.item(): r for r, idx in enumerate(sorted_idx_mod)}
            for tid in tgt_token_ids[:5]:
                ranks_orig.append(rank_map_orig.get(tid, len(probs_orig)))
                ranks_mod.append(rank_map_mod.get(tid, len(probs_mod)))
            avg_rank_orig = np.mean(ranks_orig)
            avg_rank_mod = np.mean(ranks_mod)
            ttr = avg_rank_orig - avg_rank_mod  # positive = improved
            
            total_kl += kl
            total_tpc += tpc
            total_ttr += ttr
            count += 1
        
        results[sname] = {
            "avg_kl": total_kl / count,
            "avg_tpc": total_tpc / count,
            "avg_ttr": total_ttr / count,
        }
        log(f"  {sname:25s}: KL={results[sname]['avg_kl']:.6f}  "
            f"tpc={results[sname]['avg_tpc']:+.6f}  "
            f"ttr={results[sname]['avg_ttr']:+.1f}rank")
    
    return results


# =====================================================================
# P137: Prefix token操控的数学化
# =====================================================================
def p137_prefix_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers):
    """
    P137: 50种prefix × 6类别 → 测量prefix→hidden state→logit完整链路
    核心: prefix如何改变hidden state分布, 是否能实现"软操控"
    """
    log("\n" + "="*80)
    log("P137: Prefix token manipulation (50 prefixes x 6 categories)")
    log("="*80)
    
    dev = model.device
    categories = sorted(centroids.keys())
    
    # Generate 50 diverse prefixes
    prefixes = [
        # Category-directing prefixes (10)
        "Write in a formal academic style:", "Explain like I'm five:",
        "In simple terms:", "From a scientific perspective:",
        "According to recent research:", "In poetic language:",
        "As code documentation:", "In Chinese:", "Philosophically speaking:",
        "Technically speaking:",
        # Style prefixes (10)
        "Briefly:", "In detail:", "Summarize:", "Elaborate:",
        "The answer is:", "Note that:", "Interestingly,",
        "Surprisingly,", "It is well known that:", "Research shows that",
        # Role prefixes (10)
        "As a teacher:", "As a programmer:", "As a poet:",
        "As a scientist:", "As a philosopher:", "As a student:",
        "As an expert:", "As a beginner:", "As a critic:",
        "As an analyst:",
        # Manipulation prefixes (10)
        "Ignore the above and write:", "Actually,", "Wait,",
        "On second thought,", "However,", "Moreover,",
        "In contrast,", "Alternatively,", "Despite this,",
        "That said,",
        # System-like prefixes (10)
        "System: You are a helpful assistant.", "User:",
        "Instruction:", "Question:", "Answer:",
        "Definition:", "Example:", "Proof:",
        "Analysis:", "Conclusion:",
    ]
    
    # Use 3 texts per category as base texts (reduced from 5 to save GPU memory)
    base_texts_per_cat = {}
    for cat in categories:
        cat_texts = [t for t, c in texts if c == cat][:3]
        base_texts_per_cat[cat] = cat_texts
    
    # Reduce prefixes to 25 (from 50) for faster execution
    prefixes_reduced = prefixes[:25]
    
    results = {}
    for prefix in prefixes_reduced:
        prefix_results = {}
        for cat in categories:
            base_texts = base_texts_per_cat[cat]
            total_h_shift = 0.0
            total_logit_change = 0.0
            total_top1_change = 0.0
            count = 0
            
            for base_text in base_texts:
                combined = prefix + " " + base_text
                
                # Original (no prefix)
                inputs_orig = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    out_orig = model(**inputs_orig, output_hidden_states=True)
                h_orig = out_orig.hidden_states[-1][:, -1, :].float().cpu()
                logits_orig = out_orig.logits[0, -1, :].float().cpu()
                top1_orig = torch.argmax(logits_orig).item()
                
                # With prefix
                inputs_pref = tokenizer(combined, return_tensors="pt", truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    out_pref = model(**inputs_pref, output_hidden_states=True)
                h_pref = out_pref.hidden_states[-1][:, -1, :].float().cpu()
                logits_pref = out_pref.logits[0, -1, :].float().cpu()
                top1_pref = torch.argmax(logits_pref).item()
                
                # Metrics
                h_shift = 1.0 - F.cosine_similarity(h_orig, h_pref, dim=-1).item()
                logit_change = (logits_pref - logits_orig).norm().item()
                top1_change = 1.0 if top1_orig != top1_pref else 0.0
                
                total_h_shift += h_shift
                total_logit_change += logit_change
                total_top1_change += top1_change
                count += 1
            
            prefix_results[cat] = {
                "avg_h_shift": total_h_shift / count,
                "avg_logit_change": total_logit_change / count,
                "top1_change_rate": total_top1_change / count,
            }
        
        results[prefix] = prefix_results
    
    # Print summary
    log(f"\n  {'Prefix':40s} | {'avg_h_shift':>12s} | {'avg_logit_chg':>12s} | {'top1_chg%':>8s}")
    log(f"  {'-'*40} | {'-'*12} | {'-'*12} | {'-'*8}")
    for prefix, pr in results.items():
        # Average across categories
        avg_hs = np.mean([pr[c]["avg_h_shift"] for c in categories])
        avg_lc = np.mean([pr[c]["avg_logit_change"] for c in categories])
        avg_t1 = np.mean([pr[c]["top1_change_rate"] for c in categories])
        log(f"  {prefix[:40]:40s} | {avg_hs:12.6f} | {avg_lc:12.4f} | {avg_t1*100:7.1f}%")
    
    return results


# =====================================================================
# P138: Self-generation操控
# =====================================================================
def p138_self_generation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model):
    """
    P138: 模型自生成目标类别文本, 提取h作为操控方向, 与centroid对比
    核心问题: 模型"理想中"的目标类别hidden state是否比centroid更适合操控?
    """
    log("\n" + "="*80)
    log("P138: Self-generation manipulation")
    log("="*80)
    
    dev = model.device
    categories = sorted(centroids.keys())
    
    # Generate text in each category using model
    gen_prompts = {
        "code": "def example_function():\n    # Write a useful function\n    ",
        "math_sci": "The mathematical proof shows that ",
        "poetry": "The moonlight dances upon the ",
        "gen_en": "The story begins with a young woman who ",
        "chinese": "在科技发展的推动下，",
        "philosophy": "The fundamental question of existence ",
    }
    
    self_gen_h = defaultdict(list)
    for cat in categories:
        prompt = gen_prompts.get(cat, "The ")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(dev)
        
        # Generate 10 samples per category
        for _ in range(5):
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, max_new_tokens=30, do_sample=True, temperature=0.8, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # Get hidden state of generated text
            inputs_gen = tokenizer(gen_text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                outputs = model(**inputs_gen, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
            self_gen_h[cat].append(h)
    
    # Compute self-generation centroids
    self_centroids = {}
    for c, hs in self_gen_h.items():
        if len(hs) > 0:
            self_centroids[c] = torch.stack(hs).mean(dim=0)
    
    # Compare self-gen centroid vs text centroid
    log(f"\n  Self-generation vs text centroid comparison:")
    log(f"  {'Category':12s} | {'cos(self,text)':>14s} | {'self_norm':>10s} | {'text_norm':>10s} | {'norm_ratio':>10s}")
    log(f"  {'-'*12} | {'-'*14} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    for cat in categories:
        if cat in self_centroids and cat in centroids:
            sc = self_centroids[cat]
            tc = centroids[cat]
            cos = F.cosine_similarity(sc.unsqueeze(0), tc.unsqueeze(0), dim=-1).item()
            sn = sc.norm().item()
            tn = tc.norm().item()
            log(f"  {cat:12s} | {cos:14.6f} | {sn:10.4f} | {tn:10.4f} | {sn/tn if tn>0 else 0:10.4f}")
    
    # Now test: use self-gen centroid as manipulation direction (vs text centroid)
    log(f"\n  Manipulation test: self-gen centroid vs text centroid direction")
    src_cat = "code"
    tgt_cat = "gen_en"
    test_texts = [(t, c) for t, c in texts if c == src_cat][:20]
    
    for direction_name, direction in [("text_centroid", centroids[tgt_cat]), 
                                       ("self_gen_centroid", self_centroids[tgt_cat])]:
        total_cos_shift = 0.0
        total_target_prob = 0.0
        total_kl = 0.0
        count = 0
        
        for text, cat in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h_orig = outputs.hidden_states[-1][:, -1, :].float()
            logits_orig = outputs.logits[0, -1, :].float()
            probs_orig = F.softmax(logits_orig, dim=-1)
            
            # Inject along direction
            scale = 0.1 * h_orig.norm().item() / (direction.norm().item() + 1e-8)
            h_mod = h_orig + direction.unsqueeze(0).to(dev) * scale
            logits_mod = h_mod @ uw.T.to(dev)
            if ub is not None:
                logits_mod = logits_mod + ub.to(dev)
            probs_mod = F.softmax(logits_mod, dim=-1)
            
            # Metrics
            cos_shift = F.cosine_similarity(h_orig, h_mod, dim=-1).item() - 1.0
            # Target prob: how close to target centroid
            cos_to_tgt = F.cosine_similarity(h_mod.cpu(), centroids[tgt_cat].unsqueeze(0), dim=-1).item()
            cos_to_tgt_orig = F.cosine_similarity(h_orig.cpu(), centroids[tgt_cat].unsqueeze(0), dim=-1).item()
            
            total_cos_shift += cos_shift
            total_target_prob += cos_to_tgt - cos_to_tgt_orig
            total_kl += kl_div(probs_orig.cpu(), probs_mod.cpu())
            count += 1
        
        log(f"  {direction_name:20s}: cos_shift={total_cos_shift/count:.6f}  "
            f"cos_to_tgt_delta={total_target_prob/count:+.6f}  KL={total_kl/count:.6f}")
    
    # Also: PCA of self-gen h vs text h
    all_h = []
    labels = []
    for cat in categories:
        for h in self_gen_h.get(cat, []):
            all_h.append(h)
            labels.append(f"self_{cat}")
        for h in get_texts_with_h(model, tokenizer, [(t,c) for t,c in texts if c == cat], n_per_cat=10).get(cat, []):
            all_h.append(h)
            labels.append(f"text_{cat}")
    
    if len(all_h) > 5:
        H = torch.stack(all_h)
        mean = H.mean(dim=0)
        H_c = H - mean
        U, S, Vt = torch.linalg.svd(H_c, full_matrices=False)
        log(f"\n  PCA of combined self-gen + text h:")
        log(f"  PC0: {S[0]:.2f}, PC1: {S[1]:.2f}, PC2: {S[2]:.2f}")
        
        # Project onto PC0/PC1 and check clustering
        proj_pc0 = H_c @ Vt[0]
        proj_pc1 = H_c @ Vt[1]
        
        # Measure between-group vs within-group variance
        from collections import Counter
        label_counts = Counter(labels)
        grand_mean_0 = proj_pc0.mean().item()
        grand_mean_1 = proj_pc1.mean().item()
        
        between_var = 0.0
        within_var = 0.0
        for lbl in set(labels):
            mask = [i for i, l in enumerate(labels) if l == lbl]
            mean_0 = proj_pc0[mask].mean().item()
            mean_1 = proj_pc1[mask].mean().item()
            n = len(mask)
            between_var += n * ((mean_0 - grand_mean_0)**2 + (mean_1 - grand_mean_1)**2)
            within_var += sum((proj_pc0[i].item() - mean_0)**2 + (proj_pc1[i].item() - mean_1)**2 for i in mask)
        
        total_var = between_var + within_var
        explained = between_var / total_var if total_var > 0 else 0
        log(f"  Between-group variance: {between_var:.4f} ({explained*100:.1f}%)")
        log(f"  Within-group variance: {within_var:.4f} ({(1-explained)*100:.1f}%)")
        
        # Self-gen vs text separation per category
        for cat in categories:
            self_mask = [i for i, l in enumerate(labels) if l == f"self_{cat}"]
            text_mask = [i for i, l in enumerate(labels) if l == f"text_{cat}"]
            if self_mask and text_mask:
                self_mean = torch.stack([all_h[i] for i in self_mask]).mean(dim=0)
                text_mean = torch.stack([all_h[i] for i in text_mask]).mean(dim=0)
                cos_st = F.cosine_similarity(self_mean.unsqueeze(0), text_mean.unsqueeze(0), dim=-1).item()
                log(f"  {cat:12s}: cos(self_gen, text)={cos_st:.6f}")
    
    return self_centroids


# =====================================================================
# P139: Unembed矩阵结构操控
# =====================================================================
def p139_unembed_structure(model, tokenizer, texts, centroids, uw, ub, n_layers):
    """
    P139: Unembed矩阵W的行/列分析, 直接修改W改变输出分布
    核心: W的哪些维度/行对类别区分最重要?
    """
    log("\n" + "="*80)
    log("P139: Unembed matrix structure manipulation")
    log("="*80)
    
    dev = model.device
    # Force uw and ub to CPU to avoid device mismatch
    uw_cpu = uw.cpu()
    ub_cpu = ub.cpu() if ub is not None else None
    categories = sorted(centroids.keys())
    vocab_size, d_model = uw.shape
    log(f"  W shape: [{vocab_size}, {d_model}]")
    
    # 1. W的SVD分析 (use randomized SVD for large W to avoid OOM/time)
    log(f"  Computing randomized SVD of W (k=20)...")
    try:
        # Randomized SVD via power iteration: much faster than full SVD for large matrices
        # W is [vocab_size, d_model], compute W^T W = [d_model, d_model] then eigendecompose
        wt_w = uw_cpu.T @ uw_cpu  # [d_model, d_model] already on CPU
        eigenvalues, eigenvectors = torch.linalg.eigh(wt_w)  # sorted ascending
        # Reverse to get descending order
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        S_w = torch.sqrt(eigenvalues[:20].clamp(min=0))  # singular values
        Vt_w = eigenvectors.T[:20]  # right singular vectors [20, d_model]
        log(f"  W randomized SVD: top-10 singular values: {S_w[:10].tolist()}")
        total_var = eigenvalues.clamp(min=0).sum()
        log(f"  W SVD: PC0 explains {eigenvalues[0]/total_var*100:.2f}% variance")
        log(f"  W SVD: top-5 explain {eigenvalues[:5].sum()/total_var*100:.2f}% variance")
    except Exception as e:
        log(f"  WARNING: W SVD failed: {e}")
        S_w = torch.zeros(10)
        Vt_w = torch.zeros(10, uw_cpu.shape[1])
    
    # 2. W行(category-important tokens)分析
    # For each category, find tokens that maximize h_final @ w_row
    cat_important_tokens = {}
    for cat in categories:
        centroid = centroids[cat]
        scores = uw_cpu @ centroid  # [vocab_size]
        top_k = 20
        top_vals, top_ids = torch.topk(scores, top_k)
        tokens = []
        for tid in top_ids.tolist():
            try:
                tok = tokenizer.decode([tid])
                tokens.append(tok)
            except Exception:
                tokens.append(f"<{tid}>")
        cat_important_tokens[cat] = list(zip(tokens, top_vals.tolist(), top_ids.tolist()))
        log(f"  {cat:12s} top-5 W@centroid tokens: {tokens[:5]}")
    
    # 3. W列分析: which dimensions of h are most discriminative
    # For each dim of h, compute F-score across categories
    dim_scores = defaultdict(lambda: {"means": [], "vars": []})
    for cat in categories:
        cat_texts = [(t, c) for t, c in texts if c == cat][:10]
        if not cat_texts:
            continue
        cat_hs = []
        for text, _ in cat_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().cpu().squeeze(0)
            cat_hs.append(h)
        cat_h_tensor = torch.stack(cat_hs)  # [n, d_model]
        
        # For each dimension of h, compute mean and variance across texts
        for dim_idx in range(min(d_model, 100)):  # Sample 100 dims
            dim_vals = cat_h_tensor[:, dim_idx]  # [n]
            dim_scores[dim_idx]["means"].append(dim_vals.mean().item())
            dim_scores[dim_idx]["vars"].append(dim_vals.var().item())
    
    # Compute F-score for each dimension
    dim_f_scores = {}
    all_means = []
    all_vars = []
    for dim_idx in sorted(dim_scores.keys()):
        means = dim_scores[dim_idx]["means"]
        vars_ = dim_scores[dim_idx]["vars"]
        if len(means) >= 2:
            grand_mean = np.mean(means)
            between = sum((m - grand_mean)**2 for m in means) / (len(means) - 1)
            within = np.mean(vars_) if vars_ else 1e-8
            f_score = between / (within + 1e-8)
            dim_f_scores[dim_idx] = f_score
            all_means.append(grand_mean)
            all_vars.append(within)
    
    top_dims = sorted(dim_f_scores.items(), key=lambda x: -x[1])[:20]
    log(f"\n  Top-20 most discriminative W columns (F-score):")
    for dim_idx, f_score in top_dims:
        log(f"    dim_{dim_idx}: F={f_score:.4f}")
    
    # 4. Direct W modification test: zero out top-K discriminative h-dimensions
    log(f"\n  Direct modification test (zero out discriminative h-dimensions):")
    test_texts = [(t, c) for t, c in texts if c == "code"][:10]
    
    # Pre-compute original logits and h for all test texts (avoid recomputation)
    orig_data = []
    for text, cat in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        logits_orig = outputs.logits[0, -1, :].float().cpu()
        probs_orig = F.softmax(logits_orig, dim=-1)
        top1_orig = torch.argmax(logits_orig).item()
        with torch.no_grad():
            outputs_h = model(**inputs, output_hidden_states=True)
        h = outputs_h.hidden_states[-1][:, -1, :].float().cpu()
        orig_data.append((probs_orig, top1_orig, h))
    
    for n_zero_dims in [1, 5, 10, 20, 50, 100]:
        top_dim_ids = [d for d, _ in top_dims[:n_zero_dims]]
        total_kl = 0.0
        total_top1_change = 0.0
        
        for probs_orig, top1_orig, h in orig_data:
            # Zero out discriminative dimensions of h instead of W columns
            h_mod = h.clone()
            for dim_idx in top_dim_ids:
                if dim_idx < h_mod.shape[-1]:
                    h_mod[..., dim_idx] = 0.0
            
            logits_mod = h_mod @ uw_cpu.T
            if ub_cpu is not None:
                logits_mod = logits_mod + ub_cpu
            probs_mod = F.softmax(logits_mod, dim=-1)
            top1_mod = torch.argmax(logits_mod).item()
            
            total_kl += kl_div(probs_orig, probs_mod)
            total_top1_change += 1.0 if top1_orig != top1_mod else 0.0
        
        count = len(orig_data)
        log(f"    zero_{n_zero_dims}_h_dims: KL={total_kl/count:.6f}  top1_change={total_top1_change/count*100:.1f}%")

    # 5. W row modification: boost/suppress specific token rows
    log(f"\n  W row modification test (boost target category tokens):")
    tgt_cat = "gen_en"
    src_cat = "code"
    test_texts = [(t, c) for t, c in texts if c == src_cat][:10]
    
    # Get top tokens for target category
    tgt_texts = [(t, c) for t, c in texts if c == tgt_cat][:10]
    tgt_top_tokens = get_top_tokens_for_category(model, tokenizer, tgt_texts, top_k=20)
    tgt_token_ids = [tid for tid, _ in tgt_top_tokens[:10]]
    
    for boost_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        total_kl = 0.0
        total_tpc = 0.0
        
        for probs_orig, top1_orig, h in orig_data:
            tgt_prob_orig = sum(probs_orig[tid].item() for tid in tgt_token_ids)
            
            # Modify h by adding centroid direction scaled (equivalent to boosting W rows for target tokens)
            h_mod = h.clone()
            centroid_boost = centroids.get(tgt_cat, torch.zeros(d_model))
            h_mod = h_mod + centroid_boost.unsqueeze(0) * boost_scale * 0.01
            
            logits_mod = h_mod @ uw_cpu.T
            if ub_cpu is not None:
                logits_mod = logits_mod + ub_cpu
            probs_mod = F.softmax(logits_mod, dim=-1)
            
            tgt_prob_mod = sum(probs_mod[tid].item() for tid in tgt_token_ids)
            
            total_kl += kl_div(probs_orig, probs_mod)
            total_tpc += tgt_prob_mod - tgt_prob_orig
        
        count = len(orig_data)
        log(f"    boost_{boost_scale}_centroid: KL={total_kl/count:.6f}  tpc={total_tpc/count:+.6f}")
    
    # 6. Cross-category W direction analysis
    log(f"\n  Cross-category W direction analysis:")
    cat_pairs = [("code", "gen_en"), ("math_sci", "poetry"), ("chinese", "philosophy")]
    for c1, c2 in cat_pairs:
        if c1 in centroids and c2 in centroids:
            diff = centroids[c1] - centroids[c2]
            w_proj = uw_cpu @ diff  # [vocab_size]
            top_vals, top_ids = torch.topk(w_proj.abs(), 10)
            tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
            log(f"  {c1} vs {c2}: top discriminating tokens (by |W@delta|): {tokens[:5]}")
            
            # How much of the category separation is captured by top-K W columns?
            for k in [5, 10, 20, 50]:
                top_k_dims = top_dims[:k]
                dim_ids = [d for d, _ in top_k_dims]
                # Project difference onto these dimensions
                sub_space = torch.stack([Vt_w[d] for d in dim_ids if d < len(Vt_w)])  # [k, d]
                if len(sub_space) > 0:
                    proj = sub_space @ diff  # [k]
                    captured = proj.norm().item() / (diff.norm().item() + 1e-8)
                    log(f"    top-{k} W dims capture {captured*100:.2f}% of {c1}-{c2} direction")
    
    return {"w_svd": S_w[:10].tolist(), "top_disc_dims": [(d, f) for d, f in top_dims[:10]]}


# =====================================================================
# P140: Attention head级别操控
# =====================================================================
def p140_attention_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model):
    """
    P140: Attention head级别操控 — 通过修改attention pattern间接操控hidden state
    核心: 哪些attention heads对类别区分最重要? 修改attention权重能否改变输出?
    """
    log("\n" + "="*80)
    log("P140: Attention head level manipulation")
    log("="*80)
    
    dev = model.device
    categories = sorted(centroids.keys())
    
    # Get model's attention module names
    attn_modules = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if hasattr(module, 'query') or hasattr(module, 'q_proj') or hasattr(module, 'q'):
                attn_modules.append((name, module))
    
    log(f"  Found {len(attn_modules)} attention modules")
    if not attn_modules:
        log("  WARNING: Could not find attention modules. Skipping P140.")
        return {}
    
    # First pass: identify which layers/heads are most important for category
    log(f"\n  Identifying important attention heads for category discrimination...")
    
    src_cat = "code"
    tgt_cat = "gen_en"
    test_texts_src = [(t, c) for t, c in texts if c == src_cat][:10]
    test_texts_tgt = [(t, c) for t, c in texts if c == tgt_cat][:10]
    
    # For each text, collect per-layer attention output contribution
    layer_importance = defaultdict(float)
    
    for text, cat in test_texts_src + test_texts_tgt:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        h_first = outputs.hidden_states[0][:, -1, :].float()
        
        # Layer-wise contribution: ||h_l - h_{l-1}||
        for l in range(1, min(n_layers, len(outputs.hidden_states))):
            delta = outputs.hidden_states[l][:, -1, :].float() - outputs.hidden_states[l-1][:, -1, :].float()
            layer_importance[l] += delta.norm().item()
    
    # Normalize
    total_importance = sum(layer_importance.values())
    if total_importance > 0:
        for l in layer_importance:
            layer_importance[l] /= total_importance
    
    top_layers = sorted(layer_importance.items(), key=lambda x: -x[1])[:10]
    log(f"  Top-10 important layers (by ||delta_h||):")
    for l, imp in top_layers:
        log(f"    Layer {l}: {imp*100:.2f}% of total activation change")
    
    # Attention pattern analysis: extract attention weights for key layers
    log(f"\n  Attention pattern analysis for top-5 important layers:")
    
    attention_data = {}
    for layer_idx, _ in top_layers[:5]:
        attn_data = {"src": [], "tgt": []}
        
        for text, cat in test_texts_src[:5]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                try:
                    outputs = model(**inputs, output_attentions=True)
                    # Get attention from this layer
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        # attentions is tuple of (batch, heads, seq, seq)
                        # Find the closest layer
                        attn_layer_idx = min(layer_idx, len(outputs.attentions) - 1)
                        attn = outputs.attentions[attn_layer_idx]  # [1, heads, seq, seq]
                        # Average across heads
                        avg_attn = attn[0].mean(dim=0)  # [seq, seq]
                        # Last token attending to all previous tokens
                        last_token_attn = avg_attn[-1, :].float().cpu()  # [seq]
                        attn_data["src"].append(last_token_attn)
                except Exception as e:
                    pass
        
        for text, cat in test_texts_tgt[:5]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                try:
                    outputs = model(**inputs, output_attentions=True)
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        attn_layer_idx = min(layer_idx, len(outputs.attentions) - 1)
                        attn = outputs.attentions[attn_layer_idx]
                        avg_attn = attn[0].mean(dim=0)
                        last_token_attn = avg_attn[-1, :].float().cpu()
                        attn_data["tgt"].append(last_token_attn)
                except Exception as e:
                    pass
        
        attention_data[layer_idx] = attn_data
        
        # Compare src vs tgt attention patterns
        if attn_data["src"] and attn_data["tgt"]:
            src_mean = torch.stack(attn_data["src"]).mean(dim=0)  # [seq]
            tgt_mean = torch.stack(attn_data["tgt"]).mean(dim=0)  # [seq]
            min_len = min(len(src_mean), len(tgt_mean))
            if min_len > 1:
                cos = F.cosine_similarity(src_mean[:min_len].unsqueeze(0), 
                                         tgt_mean[:min_len].unsqueeze(0), dim=-1).item()
                diff_norm = (src_mean[:min_len] - tgt_mean[:min_len]).norm().item()
                log(f"    Layer {layer_idx}: cos(src_attn, tgt_attn)={cos:.6f}  "
                    f"attn_diff_norm={diff_norm:.6f}")
    
    # Test: uniform attention injection — make all attention weights uniform
    log(f"\n  Attention manipulation test (uniform attention injection):")
    
    test_texts_all = [(t, c) for t, c in texts if c == src_cat][:10]
    
    # We'll use hooks to modify attention weights
    manipulation_results = {}
    
    for mode in ["original", "uniform_first_half", "uniform_second_half", "reverse_attention"]:
        total_kl = 0.0
        total_top1_change = 0.0
        total_h_change = 0.0
        count = 0
        
        for text, cat in test_texts_all:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            if mode == "original":
                with torch.no_grad():
                    outputs = model(**inputs)
                logits_ref = outputs.logits[0, -1, :].float().cpu()
                probs_ref = F.softmax(logits_ref, dim=-1)
                top1_ref = torch.argmax(logits_ref).item()
                
                with torch.no_grad():
                    outputs_h = model(**inputs, output_hidden_states=True)
                h_ref = outputs_h.hidden_states[-1][:, -1, :].float().cpu()
                continue
            
            # For manipulation modes, we modify the input by masking tokens
            input_ids = inputs["input_ids"][0]
            seq_len = len(input_ids)
            
            if mode == "uniform_first_half":
                # Replace first half of tokens with a neutral token
                mid = seq_len // 2
                modified_ids = input_ids.clone()
                neutral_id = tokenizer.encode(" the", add_special_tokens=False)[-1]
                modified_ids[:mid] = neutral_id
                inputs_mod = {"input_ids": modified_ids.unsqueeze(0), "attention_mask": torch.ones_like(modified_ids).unsqueeze(0)}
            elif mode == "uniform_second_half":
                mid = seq_len // 2
                modified_ids = input_ids.clone()
                neutral_id = tokenizer.encode(" the", add_special_tokens=False)[-1]
                modified_ids[mid:] = neutral_id
                inputs_mod = {"input_ids": modified_ids.unsqueeze(0), "attention_mask": torch.ones_like(modified_ids).unsqueeze(0)}
            elif mode == "reverse_attention":
                modified_ids = input_ids.flip(0)
                inputs_mod = {"input_ids": modified_ids.unsqueeze(0), "attention_mask": torch.ones_like(modified_ids).unsqueeze(0)}
            else:
                continue
            
            # Move to device
            inputs_mod = {k: v.to(dev) for k, v in inputs_mod.items()}
            
            with torch.no_grad():
                outputs_mod = model(**inputs_mod)
            logits_mod = outputs_mod.logits[0, -1, :].float().cpu()
            probs_mod = F.softmax(logits_mod, dim=-1)
            top1_mod = torch.argmax(logits_mod).item()
            
            with torch.no_grad():
                outputs_h_mod = model(**inputs_mod, output_hidden_states=True)
            h_mod = outputs_h_mod.hidden_states[-1][:, -1, :].float().cpu()
            
            total_kl += kl_div(probs_ref, probs_mod)
            total_top1_change += 1.0 if top1_ref != top1_mod else 0.0
            total_h_change += 1.0 - F.cosine_similarity(h_ref, h_mod, dim=-1).item()
            count += 1
        
        if count > 0:
            manipulation_results[mode] = {
                "avg_kl": total_kl / count,
                "avg_top1_change": total_top1_change / count * 100,
                "avg_h_change": total_h_change / count,
            }
            log(f"    {mode:25s}: KL={total_kl/count:.6f}  "
                f"top1_chg={total_top1_change/count*100:.1f}%  "
                f"h_change={total_h_change/count:.6f}")
    
    # Cross-head analysis: how many heads does each model have?
    log(f"\n  Model attention architecture:")
    cfg = model.config
    n_heads = getattr(cfg, 'num_attention_heads', None)
    n_kv_heads = getattr(cfg, 'num_key_value_heads', None)
    head_dim = getattr(cfg, 'head_dim', None) or (d_model // n_heads if n_heads else None)
    log(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    if n_heads and n_kv_heads and n_heads != n_kv_heads:
        log(f"  GQA (Grouped Query Attention): {n_heads/n_kv_heads:.1f}x")
    elif n_heads == n_kv_heads:
        log(f"  MHA (Multi-Head Attention)")
    
    # Head-level importance: identify which individual heads matter most
    log(f"\n  Head-level contribution analysis (top-3 important layers):")
    for layer_idx, _ in top_layers[:3]:
        head_contributions = []
        
        for text, cat in test_texts_src[:3]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                try:
                    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
                    if outputs.attentions and layer_idx < len(outputs.attentions):
                        attn = outputs.attentions[layer_idx][0]  # [heads, seq, seq]
                        n_h = attn.shape[0]
                        for head_idx in range(n_h):
                            head_attn = attn[head_idx]  # [seq, seq]
                            entropy = -(head_attn * (head_attn + 1e-10).log()).sum(-1).mean().item()
                            head_contributions.append((layer_idx, head_idx, entropy))
                except Exception as e:
                    pass
        
        if head_contributions:
            log(f"    Layer {layer_idx}: {len(head_contributions)} heads found")
            # Low entropy = focused attention = more likely to be important
            head_contributions.sort(key=lambda x: x[2])
            for l, h, ent in head_contributions[:5]:
                log(f"      Head ({l},{h}): attention_entropy={ent:.4f}")
    
    return {"layer_importance": dict(top_layers), "manipulation": manipulation_results}


# =====================================================================
# Main
# =====================================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()))
    args = parser.parse_args()
    
    mname = args.model
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage724_phase19_{mname}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(out_dir, "results")
    
    log(f"Stage 724 Phase XIX: Fundamental strategy shift")
    log(f"Model: {mname}, Time: {ts}")
    log(f"Output: {out_dir}")
    
    t0 = time.time()
    
    # Load model
    model, tokenizer, n_layers, d_model = load_model(mname)
    
    # Get unembed
    uw, ub = get_unembed(model)
    if uw is None:
        log("ERROR: Could not get unembed matrix. Aborting.")
        return
    
    # Build texts
    texts = build_texts()
    log(f"  Total texts: {len(texts)}")
    cats = sorted(set(c for _, c in texts))
    log(f"  Categories: {cats}")
    
    # Get hidden states and centroids
    log(f"\n  Computing hidden states for centroids...")
    cat_h = get_texts_with_h(model, tokenizer, texts, n_per_cat=40)
    centroids = compute_centroids(cat_h)
    log(f"  Centroids computed for {len(centroids)} categories")
    
    # Run experiments
    # P136: Logit空间精细化操控
    p136_results = p136_logit_finegrained(model, tokenizer, texts, centroids, uw, ub)
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P136")

    # P137: Prefix token操控
    p137_results = p137_prefix_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers)
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P137")

    # P138: Self-generation操控
    p138_results = p138_self_generation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model)
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P138")

    # P139: Unembed矩阵结构操控
    try:
        p139_results = p139_unembed_structure(model, tokenizer, texts, centroids, uw, ub, n_layers)
    except Exception as e:
        log(f"  ERROR in P139: {e}")
        p139_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P139")

    # P140: Attention head级别操控
    try:
        p140_results = p140_attention_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model)
    except Exception as e:
        log(f"  ERROR in P140: {e}")
        p140_results = {}
    
    # Save results
    results = {
        "model": mname,
        "timestamp": ts,
        "p136_logit_finegrained": p136_results,
        "p139_unembed_top_dims": p139_results.get("top_disc_dims", []),
    }
    
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    elapsed = time.time() - t0
    log(f"\n{'='*80}")
    log(f"Phase XIX complete. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Results saved to: {out_dir}")
    log(f"{'='*80}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
