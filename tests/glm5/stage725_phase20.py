#!/usr/bin/env python3
"""
Stage 725: Phase XX — 多步验证+消融方法革新
================================================================================
Phase XIX确认boost_adaptive最优(tpc=0.77~0.94), 但仅测单步。
Phase XX目标: (1)验证多步效果 (2)探索全新消融方法

理论转变: 从"加法操控"到"减法消融"
==========================================================================
关键洞察:
- 之前Phase XV-XVIII所有操控都是"加法"(注入方向到h) → 全部失败
- 失败原因: 范数稀释(699x~3234x) + Hessian反转(倒坡效应)
- 但"减法"(移除维度/方向)不面临这些问题!
  - 移除不增加norm → 无稀释问题
  - 移除不是在loss landscape上移动 → 无Hessian问题
- 输入端干预(token移除/替换)是真正的do-calculus因果操作
- 观测式追踪(logit镜头)完全不需要干预

三种全新消融范式:
  1. 维度/方向移除消融: 从h中零化/移除特定分量
  2. 输入端token因果追踪: leave-one-out token替换
  3. 观测式logit镜头追踪: h_l@W^T逐层信息流观测
==========================================================================

P141: 多步logit操控 — boost_adaptive在自回归生成中的衰减曲线
P142: Token因果追踪 — leave-one-out token替换, 测量每token的因果贡献
P143: 维度移除消融 — 零化h的top-K维度(每层), 测量信息损失
P144: Logit镜头逐层演化 — h_l@W^T追踪每层"预测"的准确率演化
P145: 方向移除消融 — 从h中移除centroid方向/PC方向, 测量logit变化

用法: python stage725_phase20.py --model glm4
      python stage725_phase20.py --model qwen3
      python stage725_phase20.py --model deepseek7b
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
        "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u4e16\u754c\u3002","\u4e2d\u56fd\u7684\u7ecf\u6d4e\u53d1\u5c55\u53d6\u5f97\u4e86\u663e\u8457\u6210\u5c31\u3002",
        "\u6559\u80b2\u662f\u56fd\u4e4b\u5927\u8ba1\uff0c\u515a\u4e4b\u5927\u8ba1\u3002","\u79d1\u6280\u81ea\u7acb\u81ea\u5f3a\u662f\u56fd\u5bb6\u53d1\u5c55\u7684\u6218\u7565\u652f\u6491\u3002",
        "\u6587\u5316\u81ea\u4fe1\u662f\u4e00\u4e2a\u6c11\u65cf\u6700\u57fa\u672c\u7684\u529b\u91cf\u3002","\u7eff\u6c34\u9752\u5c71\u5c31\u662f\u91d1\u5c71\u94f6\u5c71\u3002",
        "\u4eba\u6c11\u5bf9\u7f8e\u597d\u751f\u6d3b\u7684\u5411\u5f80\u5c31\u662f\u6211\u4eec\u7684\u594b\u6597\u76ee\u6807\u3002","\u521b\u65b0\u662f\u5f15\u9886\u53d1\u5c55\u7684\u7b2c\u4e00\u52a8\u529b\u3002",
        "\u6570\u5b57\u7ecf\u6d4e\u6210\u4e3a\u65b0\u7684\u589e\u957f\u5f15\u64ce\u3002","\u4e61\u6751\u632f\u5174\u6218\u7565\u5168\u9762\u63a8\u8fdb\u3002",
        "\u9ad8\u8d28\u91cf\u53d1\u5c55\u662f\u65f6\u4ee3\u7684\u8981\u6c42\u3002","\u5bf9\u5916\u5f00\u653e\u7684\u57fa\u672c\u56fd\u7b56\u4e0d\u4f1a\u6539\u53d8\u3002",
        "\u4e2d\u56fd\u5f0f\u73b0\u4ee3\u5316\u9053\u8def\u8d8a\u8d70\u8d8a\u5bbd\u5e7f\u3002","\u78b3\u4e2d\u548c\u76ee\u6807\u63a8\u52a8\u80fd\u6e90\u8f6c\u578b\u3002",
        "\u4e2d\u533b\u836f\u4f20\u627f\u521b\u65b0\u53d1\u5c55\u3002","\u4f20\u7edf\u6587\u5316\u7684\u521b\u9020\u6027\u8f6c\u5316\u548c\u521b\u65b0\u6027\u53d1\u5c55\u3002",
        "\u56fd\u5bb6\u6cbb\u7406\u4f53\u7cfb\u548c\u6cbb\u7406\u80fd\u529b\u73b0\u4ee3\u5316\u3002","\u6784\u5efa\u4eba\u7c7b\u547d\u8fd0\u5171\u540c\u4f53\u3002",
        "\u4e00\u5e26\u4e00\u8def\u5021\u8bae\u4fc3\u8fdb\u5171\u540c\u53d1\u5c55\u3002","\u592a\u7a7a\u63a2\u7d22\u53d6\u5f97\u91cd\u5927\u7a81\u7834\u3002",
        "\u91cf\u5b50\u8ba1\u7b97\u7814\u7a76\u8fbe\u5230\u4e16\u754c\u5148\u8fdb\u6c34\u5e73\u3002","\u57fa\u56e0\u7f16\u8f91\u6280\u672f\u9020\u798f\u4eba\u7c7b\u5065\u5eb7\u3002",
        "\u65b0\u80fd\u6e90\u6c7d\u8f66\u4ea7\u4e1a\u84ec\u52c3\u53d1\u5c55\u3002","5G\u7f51\u7edc\u8986\u76d6\u4e0d\u65ad\u6269\u5c55\u3002",
        "\u82af\u7247\u81ea\u4e3b\u7814\u53d1\u6301\u7eed\u7a81\u7834\u3002","\u822a\u5929\u5458\u5b8c\u6210\u592a\u7a7a\u884c\u8d70\u4efb\u52a1\u3002",
        "\u6df1\u6d77\u63a2\u6d4b\u8fbe\u5230\u4e07\u7c73\u6df1\u5ea6\u3002","\u5927\u6570\u636e\u5206\u6790\u52a9\u529b\u79d1\u5b66\u51b3\u7b56\u3002",
        "\u667a\u6167\u57ce\u5e02\u5efa\u8bbe\u6539\u5584\u6c11\u751f\u3002","\u6559\u80b2\u5747\u8861\u53d1\u5c55\u7f29\u5c0f\u57ce\u4e61\u5dee\u8ddd\u3002",
        "\u533b\u7597\u4fdd\u969c\u4f53\u7cfb\u4e0d\u65ad\u5b8c\u5584\u3002","\u517b\u8001\u670d\u52a1\u4f53\u7cfb\u65e5\u76ca\u5065\u5168\u3002",
        "\u7cae\u98df\u5b89\u5168\u662f\u56fd\u5bb6\u5b89\u5168\u7684\u91cd\u8981\u57fa\u7840\u3002","\u751f\u6001\u73af\u5883\u6301\u7eed\u6539\u5584\u3002",
        "\u6587\u5316\u9057\u4ea7\u4fdd\u62a4\u4f20\u627f\u529b\u5ea6\u52a0\u5927\u3002","\u9752\u5e74\u662f\u56fd\u5bb6\u7684\u672a\u6765\u548c\u5e0c\u671b\u3002",
        "\u4e92\u8054\u7f51\u666e\u53ca\u7387\u6301\u7eed\u63d0\u9ad8\u3002","\u79fb\u52a8\u652f\u4ed8\u4fbf\u6377\u4eba\u4eec\u751f\u6d3b\u3002",
        "\u7535\u5b50\u5546\u52a1\u6539\u53d8\u6d88\u8d39\u65b9\u5f0f\u3002","\u5728\u7ebf\u6559\u80b2\u6253\u7834\u65f6\u7a7a\u9650\u5236\u3002",
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
    p = p + 1e-10
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum().item()

def get_top_tokens_for_category(model, tokenizer, category_texts, top_k=50):
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
    sorted_tokens = sorted(token_freq.items(), key=lambda x: -x[1])
    return [(tid, freq) for tid, freq in sorted_tokens[:top_k]]

def precompute_all_hidden(model, tokenizer, texts_subset, n_layers):
    """Pre-compute hidden states for all layers for a subset of texts."""
    dev = model.device
    all_data = []
    for idx, (text, cat) in enumerate(texts_subset):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(dev)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h_layers = {}
        for l in range(min(n_layers + 1, len(outputs.hidden_states))):
            h_layers[l] = outputs.hidden_states[l][:, -1, :].float().cpu().squeeze(0)
        logits_final = outputs.logits[0, -1, :].float().cpu()
        probs_final = F.softmax(logits_final, dim=-1)
        all_data.append({
            "text": text, "cat": cat, "h_layers": h_layers,
            "logits_final": logits_final, "probs_final": probs_final,
        })
    return all_data


# =====================================================================
# P141: Multi-step logit manipulation during autoregressive generation
# =====================================================================
def p141_multistep_logit_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model):
    """
    P141: Apply boost_adaptive at each autoregressive step for N steps.
    Measure tpc decay/amplification and top-1 change rate.
    
    Principle: Use LogitsProcessor to intercept and modify logits at each generation step.
    Compare boosted vs baseline generation to measure how manipulation effect evolves.
    """
    log("\n" + "="*80)
    log("P141: Multi-step logit manipulation (boost_adaptive in autoregressive generation)")
    log("="*80)
    
    dev = model.device
    from transformers import LogitsProcessor
    
    tgt_cat = "gen_en"
    src_cat = "code"
    tgt_texts = [(t, c) for t, c in texts if c == tgt_cat][:20]
    tgt_top_tokens = get_top_tokens_for_category(model, tokenizer, tgt_texts, top_k=100)
    tgt_token_ids = [tid for tid, freq in tgt_top_tokens[:20]]
    test_texts = [(t, c) for t, c in texts if c == src_cat][:20]
    
    log(f"  Target tokens (top-5 IDs): {tgt_token_ids[:5]}")
    log(f"  Test texts: {len(test_texts)} from '{src_cat}'")
    
    max_steps = 15
    
    class MetricsProcessor(LogitsProcessor):
        def __init__(self, boost_ids, do_boost=False):
            self.boost_ids = boost_ids
            self.do_boost = do_boost
            self.step_data = []
        def __call__(self, input_ids, scores):
            scores_f = scores.float()
            probs_orig = F.softmax(scores_f, dim=-1)
            tgt_p_orig = sum(probs_orig[0, t].item() for t in self.boost_ids[:20] if t < probs_orig.shape[-1])
            top1_orig = torch.argmax(probs_orig, dim=-1).item()
            if self.do_boost:
                for tid in self.boost_ids:
                    if tid < scores_f.shape[-1]:
                        p = probs_orig[0, tid].item()
                        scale = max(1.0, -math.log(p + 1e-10))
                        scores_f[0, tid] += scale
            probs_mod = F.softmax(scores_f, dim=-1)
            tgt_p_mod = sum(probs_mod[0, t].item() for t in self.boost_ids[:20] if t < probs_mod.shape[-1])
            top1_mod = torch.argmax(probs_mod, dim=-1).item()
            self.step_data.append({
                "step": len(self.step_data),
                "tgt_prob_orig": tgt_p_orig,
                "tgt_prob_mod": tgt_p_mod,
                "tpc": tgt_p_mod - tgt_p_orig if self.do_boost else 0.0,
                "top1_orig": top1_orig,
                "top1_mod": top1_mod,
                "top1_changed": int(top1_orig != top1_mod),
            })
            return scores_f
    
    step_tpc_boost = defaultdict(list)
    step_tpc_base = defaultdict(list)
    step_top1_change = defaultdict(list)
    step_base_prob = defaultdict(list)
    
    for idx, (text, cat) in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(dev)
        
        try:
            # Boosted generation
            bp = MetricsProcessor(tgt_token_ids, do_boost=True)
            with torch.no_grad():
                gen_boost = model.generate(
                    **inputs, max_new_tokens=max_steps, do_sample=False,
                    logits_processor=[bp], pad_token_id=tokenizer.pad_token_id
                )
            
            # Baseline generation
            blp = MetricsProcessor(tgt_token_ids, do_boost=False)
            with torch.no_grad():
                gen_base = model.generate(
                    **inputs, max_new_tokens=max_steps, do_sample=False,
                    logits_processor=[blp], pad_token_id=tokenizer.pad_token_id
                )
            
            for sd in bp.step_data:
                s = sd["step"]
                step_tpc_boost[s].append(sd["tpc"])
                step_top1_change[s].append(sd["top1_changed"])
            for sd in blp.step_data:
                s = sd["step"]
                step_base_prob[s].append(sd["tgt_prob_orig"])
                step_tpc_base[s].append(0.0)
        except Exception as e:
            log(f"  WARNING text {idx}: {e}")
    
    log(f"\n  {'Step':>5s} | {'avg_tpc':>10s} | {'tpc_std':>10s} | {'base_prob':>10s} | {'top1_chg%':>10s}")
    log(f"  {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    results = {}
    for step in range(max_steps):
        if step_tpc_boost[step]:
            avg_tpc = np.mean(step_tpc_boost[step])
            std_tpc = np.std(step_tpc_boost[step])
            avg_base = np.mean(step_base_prob.get(step, [0]))
            avg_t1c = np.mean(step_top1_change.get(step, [0]))
            log(f"  {step:5d} | {avg_tpc:10.6f} | {std_tpc:10.6f} | {avg_base:10.6f} | {avg_t1c*100:9.1f}%")
            results[step] = {"avg_tpc": float(avg_tpc), "std_tpc": float(std_tpc),
                           "base_prob": float(avg_base), "top1_chg": float(avg_t1c)}
    
    # Trend analysis
    if len(results) > 3:
        tpc_vals = [results[s]["avg_tpc"] for s in sorted(results.keys())]
        t1c_vals = [results[s]["top1_chg"] for s in sorted(results.keys())]
        # Linear trend
        x = np.arange(len(tpc_vals))
        slope_tpc = np.polyfit(x, tpc_vals, 1)[0]
        slope_t1c = np.polyfit(x, t1c_vals, 1)[0]
        log(f"\n  Trend: tpc_slope={slope_tpc:.6f}/step ({'decay' if slope_tpc < -0.001 else 'stable' if abs(slope_tpc) < 0.001 else 'growth'})")
        log(f"  Trend: top1_change_slope={slope_t1c:.4f}/step")
    
    return results


# =====================================================================
# P142: Token causal tracing (leave-one-out)
# =====================================================================
def p142_token_causal_tracing(model, tokenizer, texts, uw, ub):
    """
    P142: Remove each input token one at a time, measure logit change.
    
    Principle: This is a true do-calculus causal intervention on the INPUT side.
    By removing token i and measuring logit change, we get the CAUSAL contribution
    of each input token to the final output. This bypasses all hidden-state-level
    problems (norm dilution, Hessian) because the intervention is at the input.
    
    Three replacement strategies:
    1. Neutral token ("the")
    2. Random token
    3. Deletion (shift remaining)
    """
    log("\n" + "="*80)
    log("P142: Token causal tracing (leave-one-out replacement)")
    log("="*80)
    
    dev = model.device
    uw_cpu = uw.cpu()
    ub_cpu = ub.cpu() if ub is not None else None
    
    # Use diverse texts (5 per category)
    test_texts = []
    cats = sorted(set(c for _, c in texts))
    for cat in cats:
        cat_items = [(t, c) for t, c in texts if c == cat][:5]
        test_texts.extend(cat_items)
    
    log(f"  Test texts: {len(test_texts)} from {len(cats)} categories")
    
    neutral_id = tokenizer.encode(" the", add_special_tokens=False)[-1]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Position-aggregated results
    max_positions = 20
    total_tests = 0
    
    # Store per-text results
    text_results = []
    
    # Strategy results: {strategy: {pos: [kl_values]}}
    strategies = {
        "neutral": neutral_id,
        "pad": pad_id,
    }
    strat_kl = {s: defaultdict(list) for s in strategies}
    strat_top1 = {s: defaultdict(list) for s in strategies}
    strat_top1_prob = {s: defaultdict(list) for s in strategies}
    
    for idx, (text, cat) in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(dev)
        input_ids = inputs["input_ids"][0]
        seq_len = len(input_ids)
        
        with torch.no_grad():
            outputs = model(**inputs)
        logits_orig = outputs.logits[0, -1, :].float().cpu()
        probs_orig = F.softmax(logits_orig, dim=-1)
        top1_orig = torch.argmax(logits_orig).item()
        orig_top1_prob = probs_orig[top1_orig].item()
        
        n_pos = min(seq_len, max_positions)
        
        for sname, replace_id in strategies.items():
            for pos in range(n_pos):
                mod_ids = input_ids.clone()
                mod_ids[pos] = replace_id
                inputs_mod = {
                    "input_ids": mod_ids.unsqueeze(0).to(dev),
                    "attention_mask": torch.ones_like(mod_ids).unsqueeze(0).to(dev),
                }
                with torch.no_grad():
                    out_mod = model(**inputs_mod)
                logits_mod = out_mod.logits[0, -1, :].float().cpu()
                probs_mod = F.softmax(logits_mod, dim=-1)
                
                kl = kl_div(probs_orig, probs_mod)
                top1_mod = torch.argmax(probs_mod).item()
                t1c = 1.0 if top1_orig != top1_mod else 0.0
                t1p = probs_mod[top1_orig].item()
                
                strat_kl[sname][pos].append(kl)
                strat_top1[sname][pos].append(t1c)
                strat_top1_prob[sname][pos].append(t1p)
                total_tests += 1
        
        if (idx + 1) % 10 == 0:
            log(f"  Processed {idx+1}/{len(test_texts)} texts...")
    
    # Print results for each strategy
    for sname in strategies:
        log(f"\n  Strategy: {sname} (replace with '{sname}' token)")
        log(f"  {'Pos':>4s} | {'avg_KL':>10s} | {'top1_chg%':>10s} | {'orig_top1_p':>12s}")
        log(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*12}")
        for pos in range(max_positions):
            if strat_kl[sname][pos]:
                avg_kl = np.mean(strat_kl[sname][pos])
                avg_t1c = np.mean(strat_top1[sname][pos])
                avg_t1p = np.mean(strat_top1_prob[sname][pos])
                log(f"  {pos:4d} | {avg_kl:10.6f} | {avg_t1c*100:9.1f}% | {avg_t1p:12.6f}")
    
    # Aggregate pattern: first/mid/last position importance
    log(f"\n  Position importance pattern ({sname}):")
    for sname in strategies:
        first = np.mean(strat_kl[sname][0]) if strat_kl[sname][0] else 0
        last = np.mean(strat_kl[sname][max_positions-1]) if strat_kl[sname].get(max_positions-1) else 0
        mid = np.mean(strat_kl[sname][max_positions//2]) if strat_kl[sname].get(max_positions//2) else 0
        best_pos = max(strat_kl[sname].keys(), key=lambda p: np.mean(strat_kl[sname][p])) if strat_kl[sname] else 0
        best_kl = np.mean(strat_kl[sname][best_pos]) if strat_kl[sname].get(best_pos) else 0
        log(f"  {sname:10s}: first={first:.6f}, mid={mid:.6f}, last={last:.6f}, best_pos={best_pos}(KL={best_kl:.6f})")
    
    log(f"\n  Total token ablations: {total_tests}")
    return {s: {"avg_kl_per_pos": {p: float(np.mean(v)) for p, v in strat_kl[s].items() if v}}
            for s in strategies}


# =====================================================================
# P143: Dimension knockout at each layer
# =====================================================================
def p143_dimension_knockout(all_data, uw, ub, n_layers, d_model):
    """
    P143: At each key layer, zero out top-K dimensions of h_l and measure logit change.
    
    Principle: This is SUBTRACTION ablation (not addition manipulation).
    Instead of injecting a direction into h, we REMOVE dimensions from h.
    This doesn't face norm dilution (no norm increase) or Hessian problems (no movement on loss landscape).
    
    We compute h_l @ W^T + b for modified h_l to measure the DIRECT effect
    of removing dimensions at layer l on the output distribution.
    
    Top-K dimensions are selected by:
    1. Variance across texts (high variance = more information)
    2. F-score across categories (high F-score = more discriminative)
    """
    log("\n" + "="*80)
    log("P143: Dimension knockout at each layer (subtraction ablation)")
    log("="*80)
    
    uw_cpu = uw.cpu()
    ub_cpu = ub.cpu() if ub is not None else None
    
    # Key layers to test
    key_layers = sorted(set([
        0,
        max(1, n_layers // 4),
        n_layers // 2,
        min(n_layers - 2, 3 * n_layers // 4),
        n_layers - 1,
    ]))
    log(f"  Key layers: {key_layers}")
    log(f"  d_model: {d_model}")
    
    n_texts = len(all_data)
    log(f"  Texts: {n_texts}")
    
    results = {}
    
    for layer in key_layers:
        # Collect h vectors at this layer
        h_list = [d["h_layers"][layer] for d in all_data if layer in d["h_layers"]]
        if not h_list:
            continue
        H = torch.stack(h_list)  # [n, d]
        
        # Select dimensions by variance
        dim_var = H.var(dim=0)
        top_dims_var = torch.topk(dim_var, min(20, d_model)).indices.tolist()
        
        # Select dimensions by F-score (across categories)
        cat_h_per_dim = defaultdict(lambda: {"means": [], "vars": []})
        for d in all_data:
            if layer in d["h_layers"]:
                h = d["h_layers"][layer]
                cat = d["cat"]
                for dim_idx in range(min(d_model, 200)):
                    cat_h_per_dim[dim_idx]["means"].append(h[dim_idx].item())
        
        dim_f_scores = {}
        for dim_idx in cat_h_per_dim:
            means = cat_h_per_dim[dim_idx]["means"]
            if len(means) >= 6:
                cats_data = defaultdict(list)
                for d in all_data:
                    if layer in d["h_layers"]:
                        cats_data[d["cat"]].append(d["h_layers"][layer][dim_idx].item())
                cat_means = [np.mean(vals) for vals in cats_data.values()]
                grand_mean = np.mean(cat_means)
                between = sum((m - grand_mean)**2 for m in cat_means) / (len(cat_means) - 1)
                within = np.mean([np.var(vals) for vals in cats_data.values()])
                dim_f_scores[dim_idx] = between / (within + 1e-8)
        
        top_dims_f = sorted(dim_f_scores.items(), key=lambda x: -x[1])[:20]
        top_f_dims = [d for d, _ in top_dims_f]
        
        log(f"\n  Layer {layer}:")
        log(f"  {'K':>5s} | {'KL(var_topK)':>14s} | {'top1_chg%':>10s} | {'KL(f_topK)':>12s} | {'top1_chg%':>10s}")
        log(f"  {'-'*5} | {'-'*14} | {'-'*10} | {'-'*12} | {'-'*10}")
        
        layer_results = {}
        for k in [1, 5, 10, 20, 50, 100, 200, 500]:
            if k > d_model:
                continue
            
            # Test 1: Zero out top-K variance dimensions
            dims_var = top_dims_var[:k]
            total_kl_v = 0
            total_t1_v = 0
            # Test 2: Zero out top-K F-score dimensions
            dims_f = top_f_dims[:k]
            total_kl_f = 0
            total_t1_f = 0
            count_v = 0
            count_f = 0
            
            for d in all_data:
                if layer not in d["h_layers"]:
                    continue
                probs_orig = d["probs_final"]
                top1_orig = torch.argmax(probs_orig).item()
                
                # Variance knockout
                if dims_var:
                    h_mod = d["h_layers"][layer].clone()
                    for dim_idx in dims_var:
                        if dim_idx < h_mod.shape[-1]:
                            h_mod[dim_idx] = 0.0
                    logits_mod = h_mod @ uw_cpu.T
                    if ub_cpu is not None:
                        logits_mod = logits_mod + ub_cpu
                    probs_mod = F.softmax(logits_mod, dim=-1)
                    total_kl_v += kl_div(probs_orig, probs_mod)
                    total_t1_v += 1.0 if torch.argmax(probs_mod).item() != top1_orig else 0.0
                    count_v += 1
                
                # F-score knockout
                if dims_f:
                    h_mod2 = d["h_layers"][layer].clone()
                    for dim_idx in dims_f:
                        if dim_idx < h_mod2.shape[-1]:
                            h_mod2[dim_idx] = 0.0
                    logits_mod2 = h_mod2 @ uw_cpu.T
                    if ub_cpu is not None:
                        logits_mod2 = logits_mod2 + ub_cpu
                    probs_mod2 = F.softmax(logits_mod2, dim=-1)
                    total_kl_f += kl_div(probs_orig, probs_mod2)
                    total_t1_f += 1.0 if torch.argmax(probs_mod2).item() != top1_orig else 0.0
                    count_f += 1
            
            kl_v = total_kl_v / count_v if count_v > 0 else 0
            t1_v = total_t1_v / count_v * 100 if count_v > 0 else 0
            kl_f = total_kl_f / count_f if count_f > 0 else 0
            t1_f = total_t1_f / count_f * 100 if count_f > 0 else 0
            
            log(f"  {k:5d} | {kl_v:14.6f} | {t1_v:9.1f}% | {kl_f:12.6f} | {t1_f:9.1f}%")
            layer_results[k] = {"kl_var": float(kl_v), "top1_var": float(t1_v),
                               "kl_f": float(kl_f), "top1_f": float(t1_f)}
        
        results[layer] = layer_results
    
    return results


# =====================================================================
# P144: Logit lens per-layer evolution
# =====================================================================
def p144_logit_lens(all_data, uw, ub, n_layers, d_model):
    """
    P144: For each layer l, compute h_l @ W^T to get the "layer's prediction".
    
    Principle: This is OBSERVATION (not intervention). We simply READ what each layer
    predicts by computing h_l @ W^T + b, without modifying anything.
    
    This traces information flow: at which layer does the model "know" the answer?
    Key metrics:
    - Top-1 accuracy: how often does layer l predict the same token as the final layer?
    - Logit cosine: cos(logits_l, logits_final) — how close are the logit vectors?
    - Cross-entropy: H(final_top5 || layer_l_probs) — information loss at each layer
    """
    log("\n" + "="*80)
    log("P144: Logit lens per-layer evolution (observation)")
    log("="*80)
    
    uw_cpu = uw.cpu()
    ub_cpu = ub.cpu() if ub is not None else None
    
    n_texts = len(all_data)
    log(f"  Texts: {n_texts}")
    
    layer_metrics = defaultdict(lambda: {"top1_acc": [], "top5_acc": [], "logit_cos": [], 
                                          "kl_to_final": [], "entropy": []})
    
    for d in all_data:
        logits_final = d["logits_final"]
        top1_final = torch.argmax(logits_final).item()
        top5_final = set(torch.topk(logits_final, 5).indices.tolist())
        probs_final = d["probs_final"]
        
        for l, h in d["h_layers"].items():
            logits_l = h @ uw_cpu.T
            if ub_cpu is not None:
                logits_l = logits_l + ub_cpu
            probs_l = F.softmax(logits_l, dim=-1)
            
            top1_l = torch.argmax(logits_l).item()
            top5_l = set(torch.topk(logits_l, 5).indices.tolist())
            
            layer_metrics[l]["top1_acc"].append(int(top1_l == top1_final))
            layer_metrics[l]["top5_acc"].append(int(len(top5_l & top5_final) > 0))
            layer_metrics[l]["logit_cos"].append(
                F.cosine_similarity(logits_l.unsqueeze(0), logits_final.unsqueeze(0), dim=-1).item())
            layer_metrics[l]["kl_to_final"].append(kl_div(probs_l, probs_final))
            layer_metrics[l]["entropy"].append(
                -(probs_l * (probs_l + 1e-10).log()).sum().item())
    
    log(f"\n  {'Layer':>6s} | {'Top-1Acc':>9s} | {'Top-5Acc':>9s} | {'LogitCos':>10s} | {'KL(final)':>10s} | {'Entropy':>10s}")
    log(f"  {'-'*6} | {'-'*9} | {'-'*9} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    results = {}
    for l in sorted(layer_metrics.keys()):
        m = layer_metrics[l]
        acc1 = np.mean(m["top1_acc"])
        acc5 = np.mean(m["top5_acc"])
        lcos = np.mean(m["logit_cos"])
        kl_f = np.mean(m["kl_to_final"])
        ent = np.mean(m["entropy"])
        log(f"  {l:6d} | {acc1*100:8.1f}% | {acc5*100:8.1f}% | {lcos:10.6f} | {kl_f:10.4f} | {ent:10.4f}")
        results[l] = {"top1_acc": float(acc1), "top5_acc": float(acc5),
                      "logit_cos": float(lcos), "kl_to_final": float(kl_f), "entropy": float(ent)}
    
    # Key inflection points
    cos_vals = {l: m["logit_cos"] for l, m in layer_metrics.items() if m["logit_cos"]}
    if len(cos_vals) > 2:
        # Find layer where logit_cos first exceeds 0.95
        threshold_layers = [l for l in sorted(cos_vals.keys()) if np.mean(cos_vals[l]) > 0.95]
        if threshold_layers:
            log(f"\n  First layer with logit_cos > 0.95: L{threshold_layers[0]}")
        # Find layer where top1_acc first exceeds 0.5
        acc_layers = [l for l in sorted(layer_metrics.keys()) 
                     if np.mean(layer_metrics[l]["top1_acc"]) > 0.5]
        if acc_layers:
            log(f"  First layer with top1_acc > 50%: L{acc_layers[0]}")
    
    return results


# =====================================================================
# P145: Direction knockout at each layer
# =====================================================================
def p145_direction_knockout(all_data, centroids, uw, ub, n_layers, d_model):
    """
    P145: Remove specific DIRECTIONS from h at each layer, measure logit change.
    
    Principle: Subtraction ablation on DIRECTIONS (not individual dimensions).
    For each direction d: h_modified = h - (h . d_hat) * d_hat
    This removes the component of h along d while preserving all orthogonal components.
    
    Directions tested:
    1. Category centroid differences (code vs gen_en, etc.)
    2. Overall mean centroid direction
    3. First K principal components of all h vectors
    
    If removing a direction causes large KL change, that direction carries
    important information at that layer.
    """
    log("\n" + "="*80)
    log("P145: Direction knockout at each layer (subtraction ablation)")
    log("="*80)
    
    uw_cpu = uw.cpu()
    ub_cpu = ub.cpu() if ub is not None else None
    
    categories = sorted(centroids.keys())
    key_layers = sorted(set([
        0,
        max(1, n_layers // 4),
        n_layers // 2,
        min(n_layers - 2, 3 * n_layers // 4),
        n_layers - 1,
    ]))
    
    # Define directions
    directions = {}
    for c1, c2 in [("code", "gen_en"), ("math_sci", "poetry"), ("chinese", "philosophy")]:
        if c1 in centroids and c2 in centroids:
            d = centroids[c1] - centroids[c2]
            directions[f"cent_{c1}_vs_{c2}"] = d / (d.norm() + 1e-8)
    
    # Overall centroid
    all_c = torch.stack(list(centroids.values()))
    mean_c = all_c.mean(dim=0)
    directions["mean_centroid"] = mean_c / (mean_c.norm() + 1e-8)
    
    # PCA of final-layer h
    final_hs = [d["h_layers"].get(n_layers, d["h_layers"].get(max(d["h_layers"].keys()))) 
                for d in all_data]
    if final_hs:
        H = torch.stack(final_hs)
        H_c = H - H.mean(dim=0)
        _, S, Vt = torch.linalg.svd(H_c, full_matrices=False)
        for k in range(min(5, Vt.shape[0])):
            directions[f"PC{k}"] = Vt[k] / (Vt[k].norm() + 1e-8)
    
    log(f"  Directions: {list(directions.keys())}")
    log(f"  Key layers: {key_layers}")
    
    results = {}
    
    for dir_name, direction in directions.items():
        log(f"\n  Direction: {dir_name}")
        log(f"  {'Layer':>6s} | {'avg_KL':>10s} | {'top1_chg%':>10s} | {'cos(h,d)':>10s}")
        log(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
        
        dir_results = {}
        for layer in key_layers:
            total_kl = 0
            total_t1 = 0
            total_cos = 0
            count = 0
            
            for d in all_data:
                if layer not in d["h_layers"]:
                    continue
                h = d["h_layers"][layer]
                probs_orig = d["probs_final"]
                top1_orig = torch.argmax(probs_orig).item()
                
                # Remove projection onto direction
                proj = (h @ direction) * direction
                h_modified = h - proj
                
                # Measure cosine between original h and direction
                cos_hd = F.cosine_similarity(h.unsqueeze(0), direction.unsqueeze(0), dim=-1).item()
                
                logits_mod = h_modified @ uw_cpu.T
                if ub_cpu is not None:
                    logits_mod = logits_mod + ub_cpu
                probs_mod = F.softmax(logits_mod, dim=-1)
                
                total_kl += kl_div(probs_orig, probs_mod)
                total_t1 += 1.0 if torch.argmax(probs_mod).item() != top1_orig else 0.0
                total_cos += abs(cos_hd)
                count += 1
            
            if count > 0:
                dir_results[layer] = {
                    "avg_kl": float(total_kl / count),
                    "top1_change": float(total_t1 / count * 100),
                    "avg_cos_hd": float(total_cos / count),
                }
                log(f"  {layer:6d} | {total_kl/count:10.6f} | {total_t1/count*100:9.1f}% | {total_cos/count:10.6f}")
        
        results[dir_name] = dir_results
    
    return results


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
    out_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage725_phase20_{mname}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(out_dir, "results")
    
    log(f"Stage 725 Phase XX: Multi-step validation + Ablation paradigm revolution")
    log(f"Model: {mname}, Time: {ts}")
    log(f"Output: {out_dir}")
    
    t0 = time.time()
    
    model, tokenizer, n_layers, d_model = load_model(mname)
    uw, ub = get_unembed(model)
    if uw is None:
        log("ERROR: Could not get unembed matrix. Aborting.")
        return
    
    texts = build_texts()
    log(f"  Total texts: {len(texts)}")
    
    # Compute centroids
    log(f"\n  Computing centroids...")
    cat_h = get_texts_with_h(model, tokenizer, texts, n_per_cat=40)
    centroids = compute_centroids(cat_h)
    log(f"  Centroids: {len(centroids)} categories")
    gc.collect(); torch.cuda.empty_cache()
    
    # P141: Multi-step logit manipulation
    try:
        p141_results = p141_multistep_logit_manipulation(model, tokenizer, texts, centroids, uw, ub, n_layers, d_model)
    except Exception as e:
        log(f"  ERROR P141: {e}")
        p141_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P141")
    
    # P142: Token causal tracing
    try:
        p142_results = p142_token_causal_tracing(model, tokenizer, texts, uw, ub)
    except Exception as e:
        log(f"  ERROR P142: {e}")
        p142_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P142")
    
    # Pre-compute hidden states for P143-P145 (shared computation)
    log(f"\n  Pre-computing hidden states for P143-P145...")
    test_subset = []
    cats = sorted(set(c for _, c in texts))
    for cat in cats:
        cat_items = [(t, c) for t, c in texts if c == cat][:5]
        test_subset.extend(cat_items)
    log(f"  Subset: {len(test_subset)} texts, {len(cats)} categories")
    
    all_data = precompute_all_hidden(model, tokenizer, test_subset, n_layers)
    gc.collect(); torch.cuda.empty_cache()
    log(f"  Hidden states pre-computed. Layers per text: {len(all_data[0]['h_layers'])}")
    
    # P143: Dimension knockout
    try:
        p143_results = p143_dimension_knockout(all_data, uw, ub, n_layers, d_model)
    except Exception as e:
        log(f"  ERROR P143: {e}")
        p143_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P143")
    
    # P144: Logit lens
    try:
        p144_results = p144_logit_lens(all_data, uw, ub, n_layers, d_model)
    except Exception as e:
        log(f"  ERROR P144: {e}")
        p144_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P144")
    
    # P145: Direction knockout
    try:
        p145_results = p145_direction_knockout(all_data, centroids, uw, ub, n_layers, d_model)
    except Exception as e:
        log(f"  ERROR P145: {e}")
        p145_results = {}
    gc.collect(); torch.cuda.empty_cache()
    log("  [GC] After P145")
    
    # Save results
    results = {
        "model": mname,
        "timestamp": ts,
        "p141_multistep": p141_results,
        "p142_token_tracing": {s: {str(k): v for k, v in d.items()} for s, d in p142_results.items()} if p142_results else {},
        "p143_dim_knockout": {str(l): d for l, d in p143_results.items()} if p143_results else {},
        "p144_logit_lens": {str(l): d for l, d in p144_results.items()} if p144_results else {},
        "p145_dir_knockout": {dn: {str(l): d for l, d in dd.items()} for dn, dd in p145_results.items()} if p145_results else {},
    }
    
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    elapsed = time.time() - t0
    log(f"\n{'='*80}")
    log(f"Phase XX complete. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Results saved to: {out_dir}")
    log(f"{'='*80}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
