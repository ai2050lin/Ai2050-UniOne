"""
Phase CXXXII: 跨模型统一语义理论 (P571-P574)
==============================================
P571: 三模型语义方向对比 — 哪些W_U方向跨模型一致?哪些是模型特有的?
P572: 语义方向的功能分类与层级映射 — 方向5-50的精细分类
P573: 频谱→语义→预测完整因果链 — 中间变量是什么?
P574: 统一语义编码方程 — 跨模型一致的编码结构

大规模测试: 200+文本 × 3模型
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights

import torch

# ===================== 200+ 大规模测试文本 =====================
TEST_TEXTS = [
    # 科技/计算机 (1-20)
    "The development of artificial intelligence has transformed many aspects of modern life",
    "Quantum computing promises to revolutionize cryptography and data processing",
    "Artificial neural networks are inspired by the biological structure of the brain",
    "Advances in robotics are reshaping manufacturing and service industries worldwide",
    "Digital technology has fundamentally changed how we communicate and access information",
    "The spacecraft successfully completed its mission to study the distant planet",
    "Technological innovation drives economic growth and social transformation",
    "Technological literacy is increasingly important for participation in modern society",
    "Renewable energy technologies are becoming more efficient and cost-effective each year",
    "Understanding the brain remains one of the greatest challenges in modern science",
    "The engineering team designed a bridge that could withstand extreme weather conditions",
    "Advances in genetics have opened new possibilities for treating inherited diseases",
    "Medical imaging technology has greatly improved diagnostic accuracy in healthcare",
    "The laboratory conducted experiments to test the hypothesis under controlled conditions",
    "Space exploration expands our understanding of the universe and our place within it",
    "Cognitive science investigates the nature and mechanisms of mental processes",
    "The researcher published groundbreaking findings in the prestigious scientific journal",
    "Neuroscience research has revealed remarkable plasticity in the developing brain",
    "The semiconductor industry continues to push the boundaries of miniaturization",
    "Machine learning algorithms can identify patterns in vast datasets that humans cannot detect",
    # 自然/环境 (21-40)
    "In the early morning light, the birds sang their melodious songs across the valley",
    "Scientists discovered a new species of deep-sea fish near the volcanic vents",
    "Climate change poses significant challenges for coastal communities worldwide",
    "The river flowed peacefully through the countryside, reflecting the sunset",
    "The mountain trail wound through dense forests and alongside crystal streams",
    "The sunset painted the sky in shades of orange, pink, and purple",
    "Biodiversity is essential for maintaining healthy ecosystems and human well-being",
    "The rain fell gently on the roof, creating a soothing rhythm that lulled her to sleep",
    "Sustainable agriculture practices help preserve soil quality and protect water resources",
    "Environmental conservation requires balancing human needs with ecological preservation",
    "The documentary highlighted the plight of endangered species in tropical forests",
    "The waterfall cascaded down the rocky cliff into the crystal-clear pool below",
    "The glacier had been retreating for decades, a visible sign of global warming",
    "The forest ecosystem supports a diverse array of plant and animal species",
    "The river delta supports rich biodiversity and provides ecosystem services to millions",
    "Climate models predict significant changes in precipitation patterns over the coming decades",
    "The hiker reached the summit just as the sun broke through the clouds",
    "The ancient redwood trees have stood for thousands of years in the misty forest",
    "Volcanic eruptions can dramatically alter landscapes and affect global weather patterns",
    "Coral reefs are among the most diverse and fragile ecosystems on our planet",
    # 社会/人文 (41-60)
    "The ancient temple stood silently on the mountain, watching centuries pass by",
    "The economic crisis led to widespread unemployment and social unrest",
    "The orchestra performed Beethoven's ninth symphony with remarkable precision",
    "Education is the most powerful weapon which you can use to change the world",
    "The city skyline glittered with lights as night fell over the metropolis",
    "The detective carefully examined the evidence to solve the mysterious case",
    "Music has the power to transcend language barriers and unite people",
    "The professor delivered an insightful lecture on the history of philosophy",
    "Rapid urbanization has created both opportunities and challenges for modern societies",
    "International cooperation is essential for addressing global challenges effectively",
    "Understanding different cultures promotes tolerance and reduces prejudice in society",
    "The legislative process involves careful deliberation and compromise among stakeholders",
    "Social media has transformed the landscape of political discourse and civic engagement",
    "Economic inequality remains one of the most pressing challenges of the twenty-first century",
    "Traditional crafts represent an important cultural heritage that deserves preservation",
    "The university offered courses spanning the full range of human knowledge",
    "Urban planning must address issues of housing, transportation, and public spaces",
    "Volunteer organizations play a crucial role in supporting community welfare",
    "Human rights are universal and inalienable, belonging to every person by virtue of their humanity",
    "The citizens participated actively in the democratic process by voting in the election",
    # 日常/情感 (61-80)
    "She walked through the garden, picking roses and humming a gentle tune",
    "The chef carefully prepared the traditional recipe passed down through generations",
    "Children played happily in the park while their parents watched from the benches",
    "The artist captured the essence of human emotion in her latest painting",
    "The old man sat on the bench, watching children play and remembering his own youth",
    "The pianist performed a beautiful sonata that moved the audience to tears",
    "The community garden provided fresh vegetables and a gathering place for neighbors",
    "The dancer moved with grace and precision, expressing emotions without words",
    "The old library contained thousands of rare manuscripts from the medieval period",
    "The garden was filled with colorful flowers and the sweet scent of jasmine",
    "The night sky was filled with stars, each one a distant sun illuminating the cosmos",
    "The park provided a peaceful retreat from the noise and bustle of city life",
    "The symphony orchestra rehearsed diligently for their upcoming performance",
    "The painting depicted a scene of rural life in nineteenth-century Europe",
    "The musician composed a piece that blended classical and contemporary styles seamlessly",
    "The poem evoked images of autumn leaves falling silently on still water",
    "She found comfort in the familiar routine of morning coffee and quiet contemplation",
    "The warm embrace of family gathered around the holiday table brought tears of joy",
    "A gentle breeze carried the fragrance of blooming jasmine through the open window",
    "The children's laughter echoed through the playground on that sunny afternoon",
    # 学术/哲学 (81-100)
    "Mathematics provides the foundation for understanding the physical world",
    "Philosophers have debated the nature of consciousness for thousands of years",
    "The ancient philosopher taught that wisdom comes from understanding oneself",
    "Scientific research requires rigorous methodology and careful attention to detail",
    "The ancient text contained wisdom that remains relevant in contemporary society",
    "Philosophical inquiry seeks to understand the fundamental nature of reality and existence",
    "Literature provides insight into the human condition across cultures and throughout history",
    "The philosopher argued that ethical behavior is grounded in rational principles",
    "Understanding the brain remains one of the greatest challenges in modern science",
    "The historian analyzed primary sources to construct an accurate account of the event",
    "The novel explored themes of love, loss, and redemption in post-war society",
    "The seminar explored the intersection of technology, ethics, and public policy",
    "The ancient scrolls contained mathematical formulas that were centuries ahead of their time",
    "Social institutions shape individual behavior and collective outcomes in complex ways",
    "Language acquisition in children follows predictable developmental stages",
    "The constitution establishes the framework for governance and protects individual liberties",
    "The students debated the merits of different economic systems with passion and rigor",
    "Abstract art challenges our preconceptions about what constitutes beauty and meaning",
    "Epistemology examines the nature scope and limits of human knowledge and belief",
    "Metaphysics explores fundamental questions about existence objects and their properties",
    # 商业/经济 (101-120)
    "The market economy relies on supply and demand to allocate resources efficiently",
    "International trade creates economic interdependence that can promote peace and cooperation",
    "The bridge connected the two sides of the river, facilitating trade and communication",
    "The city implemented policies to reduce air pollution and improve public health",
    "Public health initiatives have dramatically reduced the incidence of preventable diseases",
    "The train journeyed through the countryside, offering passengers scenic views of rolling hills",
    "The coastal town relied on fishing and tourism as its primary economic activities",
    "Advances in medicine have significantly increased human life expectancy",
    "The discovery of antibiotics revolutionized medicine and saved countless lives",
    "Renewable energy sources are becoming increasingly important for sustainable development",
    "Innovation in fintech is democratizing access to financial services worldwide",
    "Supply chain disruptions have highlighted the vulnerability of global manufacturing networks",
    "Small businesses are the backbone of local economies and community identity",
    "The stock market reflected investor uncertainty amid geopolitical tensions",
    "Entrepreneurship requires resilience creativity and a willingness to embrace failure",
    "Central banks play a critical role in maintaining monetary stability and economic growth",
    "The gig economy has transformed traditional employment relationships and labor markets",
    "Infrastructure investment is essential for long-term economic competitiveness",
    "The tourism industry contributes significantly to employment and cultural exchange",
    "Property markets in major cities have become increasingly unaffordable for average families",
    # 历史/文化 (121-140)
    "The historical monument commemorates the sacrifices made during the struggle for independence",
    "The ancient civilization built remarkable structures that still impress engineers today",
    "Cultural exchange enriches societies by introducing new perspectives and practices",
    "The archaeologist carefully uncovered artifacts that had been buried for millennia",
    "The museum exhibited artifacts from civilizations spanning five thousand years of history",
    "The traditional ceremony celebrated the harvest and honored the ancestors",
    "The documentary filmmaker captured compelling stories of resilience and hope",
    "The ancient city was a thriving center of trade scholarship and artistic expression",
    "Renaissance artists revolutionized painting by mastering perspective and human anatomy",
    "The invention of the printing press transformed the spread of knowledge across Europe",
    "Medieval cathedrals stand as testaments to the architectural ingenuity of their builders",
    "The silk road connected diverse civilizations and facilitated cultural and commercial exchange",
    "Indigenous knowledge systems offer valuable insights into sustainable living practices",
    "The oral traditions of ancient cultures preserve wisdom that written records cannot capture",
    "Historical archives provide invaluable windows into the lives and thoughts of past generations",
    "The industrial revolution fundamentally altered social structures and economic relationships",
    "Ancient philosophers developed systems of thought that continue to influence modern ethics",
    "The preservation of cultural heritage sites requires international cooperation and funding",
    "Folk music traditions reflect the lived experiences and values of communities worldwide",
    "The translation of ancient texts opens new windows into forgotten civilizations",
    # 短句/简单句 (141-180)
    "The cat sat on the mat",
    "She reads books every day",
    "The sun rises in the east",
    "He walked to the store",
    "The dog chased the ball",
    "They ate dinner together",
    "The child smiled happily",
    "Rain fell from the sky",
    "Birds fly in formation",
    "The flowers bloomed beautifully",
    "Time passes quickly",
    "Water flows downhill",
    "Stars shine at night",
    "The wind blew gently",
    "Snow covered the mountains",
    "The fire burned brightly",
    "Leaves fell from the trees",
    "The ocean waves crashed",
    "The clock ticked steadily",
    "She sang a song",
    "The book was interesting",
    "He opened the door",
    "The car drove fast",
    "They laughed together",
    "The cake tasted sweet",
    "She wrote a letter",
    "The bell rang loudly",
    "The sky turned dark",
    "He ran very quickly",
    "The fish swam upstream",
    "Ice melted in the sun",
    "The child cried softly",
    "She painted a picture",
    "The music played softly",
    "He fixed the broken toy",
    "The clouds gathered overhead",
    "She planted seeds in spring",
    "The dog barked at strangers",
    "The train arrived on time",
    "He cooked a delicious meal",
    # 复杂/推理 (181-200)
    "Although the evidence was circumstantial, the jury found the defendant guilty beyond reasonable doubt",
    "The paradox of choice suggests that more options can lead to less satisfaction and decision paralysis",
    "Quantum entanglement allows particles to be correlated regardless of the distance separating them",
    "The placebo effect demonstrates the remarkable influence of psychological expectations on physical outcomes",
    "Emergent properties arise from the interactions of simpler components but cannot be reduced to them",
    "The butterfly effect illustrates how small changes in initial conditions can lead to vastly different outcomes",
    "Game theory provides mathematical frameworks for analyzing strategic interactions between rational agents",
    "The anthropic principle raises profound questions about the relationship between observers and the universe",
    "Chaos theory reveals that deterministic systems can produce behavior that appears random and unpredictable",
    "Cognitive biases systematically distort human judgment and decision making in predictable ways",
    "The tragedy of the commons occurs when individual incentives conflict with collective welfare",
    "Evolutionary algorithms use principles of natural selection to optimize complex engineering problems",
    "Information theory quantifies the fundamental limits of data compression and transmission",
    "Network effects create positive feedback loops that can lead to winner-take-all market dynamics",
    "Bayesian inference provides a principled framework for updating beliefs in light of new evidence",
    "The halting problem demonstrates that certain computational questions are fundamentally undecidable",
    "Thermodynamic entropy measures the irreversibility of physical processes and the arrow of time",
    "Emergent complexity in cellular automata shows how simple rules can generate rich behavior",
    "The Fermi paradox highlights the apparent contradiction between the probability of extraterrestrial life and the lack of evidence",
    "Consciousness remains one of the deepest unsolved mysteries in both philosophy and neuroscience",
    # 多语言混合 (201-210)
    "The café on the corner serves the best croissants in the entire neighborhood",
    "She practiced her español every day with a tutor from Barcelona",
    "The fiancé traveled from Paris to Tokyo for the important business meeting",
    "The kindergarten children learned about das Wetter in their German class",
    "The philosopher discussed the concept of nirvana in Eastern and Western traditions",
    "The sommelier recommended a vintage Bordeaux to complement the aged cheese",
    "The doppelgänger phenomenon has fascinated psychologists and literary scholars alike",
    "The entrepreneur launched a startup focusing on renewable energía solar technology",
    "The ballet dancer performed a pirouette with extraordinary grace and precision",
    "The zeitgeist of the era was captured perfectly in the novel about revolution",
]


def compute_wu_svd(model, k=300):
    """计算W_U的SVD, k=300保证足够的方向覆盖"""
    W_U = get_W_U(model)
    d_model, n_vocab = W_U.shape
    k = min(k, min(d_model, n_vocab) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.T, k=k)
    sort_idx = np.argsort(S_wu)[::-1]
    U_wu = U_wu[:, sort_idx]
    S_wu = S_wu[sort_idx]
    return U_wu, S_wu, W_U


def safe_decode(tokenizer, token_id):
    try:
        s = tokenizer.decode([token_id])
        return s.encode('ascii', 'replace').decode('ascii')
    except:
        return f"id={token_id}"


# ===================== P571: 三模型语义方向对比 =====================
def experiment_p571(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P571: 三模型语义方向对比
    对每个W_U方向(0-49), 分析:
    1. 方向的语义内容(Top-20正负token)
    2. 方向的能量权重(奇异值)
    3. 方向的功能分类(标点/功能词/名词/动词/形容词等)
    4. 200+文本中该方向的h投影统计
    
    结果将用于跨模型方向对齐分析
    """
    print(f"\n{'='*60}")
    print(f"P571: 语义方向跨模型对比 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # 定义更精细的词性/语义类别
    categories = {
        "punctuation": set([",", ".", "!", "?", ";", ":", "-", "(", ")", "\"", "'", "\u201c", "\u201d", "\u2018"]),
        "determiner": set(["the", "a", "an", "this", "that", "these", "those", "some", "any", "all", "every", "each"]),
        "pronoun": set(["he", "she", "it", "they", "we", "you", "i", "his", "her", "its", "their", "our", "my"]),
        "auxiliary": set(["is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                         "do", "does", "did", "will", "would", "can", "could", "shall", "should", "may", "might"]),
        "preposition": set(["of", "in", "for", "on", "with", "at", "by", "from", "to", "into", "through", "about"]),
        "conjunction": set(["and", "but", "or", "nor", "yet", "so", "because", "although", "while", "if", "unless"]),
        "animal": set(["cat", "dog", "bird", "fish", "horse", "elephant", "tiger", "lion", "bear", "whale",
                       "eagle", "snake", "rabbit", "deer", "wolf", "shark", "dolphin", "monkey", "penguin", "owl"]),
        "color": set(["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"]),
        "size": set(["big", "small", "large", "tiny", "huge", "little", "massive", "enormous", "vast", "minute"]),
        "emotion": set(["happy", "sad", "angry", "afraid", "love", "hate", "hope", "fear", "joy", "grief",
                        "anxiety", "pride", "shame", "guilt", "envy", "gratitude", "compassion", "jealousy"]),
        "action": set(["run", "walk", "eat", "drink", "sleep", "think", "speak", "write", "read", "build",
                       "create", "destroy", "transform", "discover", "explore", "develop", "achieve", "overcome"]),
        "abstract": set(["time", "space", "truth", "beauty", "justice", "freedom", "power", "knowledge",
                         "wisdom", "courage", "democracy", "equality", "consciousness", "existence"]),
        "technology": set(["computer", "internet", "software", "algorithm", "digital", "data", "network",
                          "system", "machine", "robot", "artificial", "quantum", "cyber"]),
        "nature": set(["mountain", "river", "ocean", "forest", "desert", "valley", "island", "volcano",
                      "glacier", "meadow", "sunrise", "sunset", "storm", "breeze"]),
        "science": set(["research", "experiment", "theory", "hypothesis", "analysis", "evidence",
                       "physics", "chemistry", "biology", "mathematics", "observation"]),
        "number": set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                       "first", "second", "third", "hundred", "thousand"]),
    }
    
    # ===== 1. 方向语义分析 =====
    direction_analysis = {}
    
    for d in range(50):
        direction = U_wu[:, d]
        projections = W_U @ direction
        
        # Top-20 正负tokens
        top_pos_ids = np.argsort(projections)[::-1][:20]
        top_neg_ids = np.argsort(projections)[:20]
        top_pos = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_pos_ids]
        top_neg = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_neg_ids]
        
        # 类别分布: Top-100中各类词的占比
        top100_ids = np.argsort(np.abs(projections))[::-1][:100]
        cat_counts = defaultdict(int)
        
        for tid in top100_ids:
            try:
                tok_str = tokenizer.decode([tid]).strip().lower()
                for cat_name, cat_set in categories.items():
                    if tok_str in cat_set:
                        cat_counts[cat_name] += 1
                        break
            except:
                pass
        
        s_weight = float(S_wu[d]) if d < len(S_wu) else 0
        
        direction_analysis[d] = {
            "top_pos": top_pos[:10],
            "top_neg": top_neg[:10],
            "category_counts": dict(cat_counts),
            "s_weight": s_weight,
        }
    
    # 打印关键方向
    print(f"\n--- W_U方向0-9语义分析 ---")
    for d in range(10):
        da = direction_analysis[d]
        pos_str = ", ".join([f"{t}({v:.2f})" for t, v in da["top_pos"][:5]])
        neg_str = ", ".join([f"{t}({v:.2f})" for t, v in da["top_neg"][:5]])
        cat_str = ", ".join([f"{k}:{v}" for k, v in sorted(da["category_counts"].items(), key=lambda x: -x[1]) if v > 0][:3])
        print(f"  Dir{d}(S={da['s_weight']:.1f}): +[{pos_str}] -[{neg_str}] cat={cat_str}")
    
    # ===== 2. 大规模文本的方向投影统计 =====
    print(f"\n--- 200+文本方向投影统计 ---")
    direction_proj_stats = {d: [] for d in range(50)}
    direction_layer_evolution = {d: {} for d in range(10)}  # 前10方向的层间演化
    
    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    n_processed = 0
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        # 末层最后位置
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        
        # 前50方向的投影
        for d in range(50):
            proj = np.dot(U_wu[:, d], h_last)
            direction_proj_stats[d].append(proj)
        
        # 前10方向的层间演化(仅前30文本)
        if n_processed < 30:
            for d in range(10):
                for li in sample_layers:
                    h_l = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
                    proj = np.dot(U_wu[:, d], h_l)
                    if li not in direction_layer_evolution[d]:
                        direction_layer_evolution[d][li] = []
                    direction_layer_evolution[d][li].append(proj)
        
        del outputs
        torch.cuda.empty_cache()
        n_processed += 1
    
    # 统计
    direction_stats = {}
    for d in range(50):
        projs = np.array(direction_proj_stats[d])
        direction_stats[d] = {
            "mean": float(np.mean(projs)),
            "std": float(np.std(projs)),
            "abs_mean": float(np.mean(np.abs(projs))),
            "skew": float(np.mean((projs - np.mean(projs))**3) / (np.std(projs)**3 + 1e-10)),
        }
    
    # 打印统计
    print(f"\n--- 方向投影统计(n={len(TEST_TEXTS)}) ---")
    for d in [0, 1, 2, 3, 4, 5, 10, 20, 30, 49]:
        ds = direction_stats[d]
        print(f"  Dir{d}: |mean|={ds['abs_mean']:.2f}, std={ds['std']:.2f}, skew={ds['skew']:.2f}")
    
    # 方向能量分布
    total_s = np.sum(S_wu[:50])
    cum_energy = np.cumsum(S_wu[:50]) / total_s
    print(f"\n--- 方向能量累积 ---")
    for d in [5, 10, 20, 30, 50]:
        if d <= len(cum_energy):
            print(f"  前{d}方向: {cum_energy[d-1]*100:.1f}% 能量")
    
    # 层间演化(前10方向)
    layer_evolution_summary = {}
    print(f"\n--- 前10方向层间投影演化(30文本均值) ---")
    for d in range(10):
        evol = []
        for li in sorted(sample_layers):
            if li in direction_layer_evolution[d]:
                mean_proj = np.mean(direction_layer_evolution[d][li])
                evol.append((li, mean_proj))
        evol_str = " → ".join([f"L{li}={v:.2f}" for li, v in evol])
        print(f"  Dir{d}: {evol_str}")
        layer_evolution_summary[d] = [(li, float(v)) for li, v in evol]
    
    result = {
        "model": model_name, "n_texts": len(TEST_TEXTS), "k_wu": k_wu,
        "direction_analysis": {
            str(d): direction_analysis[d] for d in range(50)
        },
        "direction_stats": direction_stats,
        "cum_energy_50": float(cum_energy[49]) if len(cum_energy) >= 50 else float(cum_energy[-1]),
        "layer_evolution_summary": {str(d): layer_evolution_summary[d] for d in range(10)},
        "s_weights": [float(S_wu[d]) for d in range(min(50, len(S_wu)))],
    }
    return result


# ===================== P572: 语义方向功能分类 =====================
def experiment_p572(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P572: 语义方向功能分类与层级映射
    对方向0-49做精细分类:
    1. 格式方向(标点/空格等)
    2. 功能方向(冠词/介词/连词等)
    3. 语义方向(名词/动词/形容词等)
    4. 专业化方向(特定领域词)
    5. 噪声方向(无明显语义)
    
    对200+文本, 测量每类方向的:
    - ratio(该类方向能量占比)
    - 与预测质量的关系
    - 层间演化特征
    """
    print(f"\n{'='*60}")
    print(f"P572: 语义方向功能分类与层级映射 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # ===== 1. 方向分类 =====
    # 基于P571的结果, 定义分类规则
    # 格式类词集合
    format_tokens = set()
    for t in [",", ".", "!", "?", ";", ":", "-", "(", ")", "\"", "'", "\n", "\t", " ", 
              "##", "#", "@", "$", "%", "^", "&", "*", "+", "=", "<", ">", "/", "\\",
              "|", "~", "`", "[", "]", "{", "}", "_", "\u201c", "\u201d", "\u2018", "\u2019",
              "\u00a0", "\u2026", "\u2013", "\u2014"]:
        format_tokens.add(t)
    
    function_tokens = set(["the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
                           "do", "does", "will", "would", "can", "could", "to", "of", "in", "for",
                           "on", "with", "at", "by", "and", "but", "or", "not", "that", "this",
                           "it", "he", "she", "they", "we", "you", "i", "his", "her", "its",
                           "their", "our", "my", "as", "if", "so", "no", "up", "out", "all"])
    
    # 对每个方向分类
    direction_class = {}
    for d in range(50):
        direction = U_wu[:, d]
        projections = W_U @ direction
        
        # Top-30正负token中各类词的比例
        top_ids = np.argsort(np.abs(projections))[::-1][:30]
        n_format = 0
        n_function = 0
        n_content = 0
        
        for tid in top_ids:
            try:
                tok_str = tokenizer.decode([tid]).strip()
                if tok_str in format_tokens:
                    n_format += 1
                elif tok_str.lower() in function_tokens:
                    n_function += 1
                else:
                    n_content += 1
            except:
                n_format += 1  # 解码失败的大概率是特殊token
        
        # 分类
        if n_format >= 10:  # 30个中>=10个是格式词
            direction_class[d] = "format"
        elif n_function >= 10:
            direction_class[d] = "function"
        elif n_content >= 10:
            direction_class[d] = "content"
        else:
            direction_class[d] = "mixed"
    
    # 统计分类
    class_counts = defaultdict(int)
    for d, cls in direction_class.items():
        class_counts[cls] += 1
    
    print(f"\n--- 方向分类统计 ---")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}个方向")
    print(f"  各方向分类: {dict(direction_class)}")
    
    # ===== 2. 大规模文本频谱分析 =====
    print(f"\n--- 200+文本频谱分析 ---")
    all_text_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        
        # 频谱系数
        h_coeffs = U_wu[:, :50].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # 各类方向的能量比
        format_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "format") / total_energy
        function_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "function") / total_energy
        content_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "content") / total_energy
        mixed_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "mixed") / total_energy
        
        # 预测质量
        with torch.no_grad():
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=0)
        top1_prob = probs.max().item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # Alpha
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        
        all_text_data.append({
            "format_energy": format_energy,
            "function_energy": function_energy,
            "content_energy": content_energy,
            "mixed_energy": mixed_energy,
            "ratio_50": float(np.sum(abs_coeffs[:50]**2) / total_energy),
            "top1_prob": top1_prob,
            "entropy": entropy,
            "alpha": alpha,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    # 统计分析
    format_energies = np.array([d["format_energy"] for d in all_text_data])
    function_energies = np.array([d["function_energy"] for d in all_text_data])
    content_energies = np.array([d["content_energy"] for d in all_text_data])
    mixed_energies = np.array([d["mixed_energy"] for d in all_text_data])
    top1_probs = np.array([d["top1_prob"] for d in all_text_data])
    entropies = np.array([d["entropy"] for d in all_text_data])
    alphas = np.array([d["alpha"] for d in all_text_data])
    
    print(f"\n--- 各类方向能量统计(n={len(all_text_data)}) ---")
    print(f"  格式方向: {np.mean(format_energies):.4f} ± {np.std(format_energies):.4f}")
    print(f"  功能方向: {np.mean(function_energies):.4f} ± {np.std(function_energies):.4f}")
    print(f"  内容方向: {np.mean(content_energies):.4f} ± {np.std(content_energies):.4f}")
    print(f"  混合方向: {np.mean(mixed_energies):.4f} ± {np.std(mixed_energies):.4f}")
    
    # 因果链: 各类能量→预测质量
    print(f"\n--- 各类能量→预测质量因果链 ---")
    for name, energies in [("format", format_energies), ("function", function_energies),
                            ("content", content_energies), ("mixed", mixed_energies)]:
        r_prob, p_prob = spearmanr(energies, top1_probs)
        r_ent, p_ent = spearmanr(energies, entropies)
        print(f"  {name}_energy → top1_prob: r={r_prob:.3f} (p={p_prob:.4f})")
        print(f"  {name}_energy → entropy:   r={r_ent:.3f} (p={p_ent:.4f})")
    
    # Alpha与各类能量的关系
    print(f"\n--- Alpha与各类能量关系 ---")
    for name, energies in [("format", format_energies), ("function", function_energies),
                            ("content", content_energies)]:
        r, p = spearmanr(alphas, energies)
        print(f"  alpha vs {name}_energy: r={r:.3f} (p={p:.4f})")
    
    # ===== 3. 关键方向的层级映射 =====
    print(f"\n--- 关键方向的层级映射(30文本) ---")
    sample_layers = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-1]
    
    # 选择5个关键方向
    key_dirs = []
    for cls in ["format", "function", "content"]:
        for d in range(50):
            if direction_class.get(d) == cls:
                key_dirs.append(d)
                break
    key_dirs = key_dirs[:5]  # 最多5个
    
    dir_layer_data = {d: {li: [] for li in sample_layers} for d in key_dirs}
    
    for text in TEST_TEXTS[:30]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for d in key_dirs:
            for li in sample_layers:
                h_l = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
                proj = np.dot(U_wu[:, d], h_l)
                dir_layer_data[d][li].append(proj)
        
        del outputs
        torch.cuda.empty_cache()
    
    for d in key_dirs:
        cls = direction_class.get(d, "unknown")
        evol = []
        for li in sample_layers:
            mean_proj = np.mean(dir_layer_data[d][li])
            evol.append((li, mean_proj))
        evol_str = " → ".join([f"L{li}={v:.3f}" for li, v in evol])
        print(f"  Dir{d}({cls}): {evol_str}")
    
    result = {
        "model": model_name, "n_texts": len(TEST_TEXTS),
        "direction_class": direction_class,
        "class_counts": dict(class_counts),
        "energy_stats": {
            "format_mean": float(np.mean(format_energies)),
            "function_mean": float(np.mean(function_energies)),
            "content_mean": float(np.mean(content_energies)),
            "mixed_mean": float(np.mean(mixed_energies)),
        },
        "causal_chain": {},
        "key_dir_layer_data": {str(d): {str(li): [float(v) for v in dir_layer_data[d][li]] for li in sample_layers} for d in key_dirs},
    }
    
    # 保存因果链数据
    for name, energies in [("format", format_energies), ("function", function_energies), ("content", content_energies)]:
        r_prob, p_prob = spearmanr(energies, top1_probs)
        r_ent, p_ent = spearmanr(energies, entropies)
        result["causal_chain"][name] = {
            "r_prob": float(r_prob), "p_prob": float(p_prob),
            "r_ent": float(r_ent), "p_ent": float(p_ent),
        }
    
    return result


# ===================== P573: 频谱→语义→预测完整因果链 =====================
def experiment_p573(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P573: 频谱→语义→预测完整因果链
    核心问题: 频谱参数(alpha, ratio(50), 各类能量)如何影响预测质量?
    中间变量是什么?
    
    测量200+文本的完整因果链:
    频谱参数 → 语义集中度 → logit分布 → 预测质量
    
    关键创新: 引入"语义集中度"作为中间变量
    语义集中度 = 各语义方向(非格式/功能)的贡献集中度
    """
    print(f"\n{'='*60}")
    print(f"P573: 频谱→语义→预测完整因果链 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # 大规模数据收集
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # ===== 频谱参数 =====
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # ratio(k) for k=10, 20, 50, 100
        ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
        ratio_20 = np.sum(abs_coeffs[:20]**2) / total_energy
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        ratio_100 = np.sum(abs_coeffs[:100]**2) / total_energy
        
        # Alpha
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        
        # Beta
        delta = h_last - h_prev
        delta_norm = np.linalg.norm(delta)
        h_prev_norm = np.linalg.norm(h_prev)
        beta = delta_norm / (h_prev_norm + 1e-10)
        
        # ===== 语义集中度(中间变量) =====
        # 1. 频谱集中度(Gini系数)
        sorted_energy = np.sort(abs_coeffs[:50]**2)[::-1]
        gini = 1 - 2 * np.sum(np.cumsum(sorted_energy) / np.sum(sorted_energy)) / len(sorted_energy)
        
        # 2. 频谱熵
        normalized_energy = abs_coeffs[:50]**2 / total_energy
        spectral_entropy = -np.sum(normalized_energy * np.log(normalized_energy + 1e-10))
        max_entropy = np.log(50)
        normalized_spectral_entropy = spectral_entropy / max_entropy
        
        # 3. Top-1/3/5方向的集中度
        top1_concentration = sorted_energy[0] / total_energy
        top3_concentration = np.sum(sorted_energy[:3]) / total_energy
        top5_concentration = np.sum(sorted_energy[:5]) / total_energy
        
        # 4. Delta频谱集中度
        delta_coeffs = U_wu[:, :50].T @ delta
        abs_delta = np.abs(delta_coeffs)
        delta_total = np.sum(abs_delta**2) + 1e-10
        delta_top5 = np.sum(np.sort(abs_delta**2)[-5:]) / delta_total
        delta_top10 = np.sum(np.sort(abs_delta**2)[-10:]) / delta_total
        
        # ===== Logit分布特征 =====
        with torch.no_grad():
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=0)
        
        top1_prob = probs.max().item()
        top5_probs = torch.topk(probs, 5).values.tolist()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # Logit方差/峰度
        logits_np = logits.detach().cpu().numpy()
        logit_var = float(np.var(logits_np))
        logit_kurtosis = float(np.mean((logits_np - np.mean(logits_np))**4) / (np.var(logits_np)**2 + 1e-10))
        
        # Top-1 logit gap
        sorted_logits = np.sort(logits_np)[::-1]
        logit_gap = float(sorted_logits[0] - sorted_logits[1])
        
        # ===== 频谱→logits的直接映射 =====
        # 用频谱系数重建logits
        full_logits = W_U @ h_last
        for K in [5, 10, 20, 50]:
            partial_logits = np.zeros_like(full_logits)
            for k in range(K):
                partial_logits += h_coeffs[k] * (W_U @ U_wu[:, k])
            cos_sim = np.dot(partial_logits, full_logits) / (np.linalg.norm(partial_logits) * np.linalg.norm(full_logits) + 1e-10)
            break  # 只测K=5作为代表
        logit_reconstruction_cos5 = float(cos_sim)
        
        all_data.append({
            "alpha": alpha, "beta": beta,
            "ratio_10": ratio_10, "ratio_20": ratio_20,
            "ratio_50": ratio_50, "ratio_100": ratio_100,
            "gini": gini,
            "spectral_entropy_norm": normalized_spectral_entropy,
            "top1_concentration": top1_concentration,
            "top3_concentration": top3_concentration,
            "top5_concentration": top5_concentration,
            "delta_top5": delta_top5,
            "delta_top10": delta_top10,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "logit_var": logit_var,
            "logit_kurtosis": logit_kurtosis,
            "logit_gap": logit_gap,
            "logit_reconstruction_cos5": logit_reconstruction_cos5,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    # ===== 统计分析 =====
    n = len(all_data)
    
    # 提取数组
    params = {key: np.array([d[key] for d in all_data]) for key in all_data[0].keys()}
    
    print(f"\n--- 参数统计(n={n}) ---")
    for key in ["alpha", "beta", "ratio_10", "ratio_50", "gini", "top1_concentration", "top5_concentration", "spectral_entropy_norm"]:
        print(f"  {key}: {np.mean(params[key]):.4f} ± {np.std(params[key]):.4f}")
    
    # ===== 完整因果链 =====
    print(f"\n=== 完整因果链分析 ===")
    
    # 第1环: 频谱参数 → 语义集中度
    print(f"\n--- 第1环: 频谱参数 → 语义集中度 ---")
    spectral_params = ["alpha", "ratio_10", "ratio_50", "beta"]
    concentration_targets = ["gini", "top1_concentration", "top5_concentration", "spectral_entropy_norm"]
    
    causal_chain_1 = {}
    for sp in spectral_params:
        for ct in concentration_targets:
            r, p = spearmanr(params[sp], params[ct])
            if abs(r) > 0.15 or p < 0.05:
                print(f"  {sp} → {ct}: r={r:.3f} (p={p:.4f})")
            causal_chain_1[f"{sp}->{ct}"] = {"r": float(r), "p": float(p)}
    
    # 第2环: 语义集中度 → logit分布
    print(f"\n--- 第2环: 语义集中度 → logit分布 ---")
    logit_targets = ["logit_var", "logit_kurtosis", "logit_gap"]
    
    causal_chain_2 = {}
    for ct in concentration_targets + ["delta_top5"]:
        for lt in logit_targets:
            r, p = spearmanr(params[ct], params[lt])
            if abs(r) > 0.15 or p < 0.05:
                print(f"  {ct} → {lt}: r={r:.3f} (p={p:.4f})")
            causal_chain_2[f"{ct}->{lt}"] = {"r": float(r), "p": float(p)}
    
    # 第3环: logit分布 → 预测质量
    print(f"\n--- 第3环: logit分布 → 预测质量 ---")
    quality_targets = ["top1_prob", "entropy"]
    
    causal_chain_3 = {}
    for lt in logit_targets + ["logit_reconstruction_cos5"]:
        for qt in quality_targets:
            r, p = spearmanr(params[lt], params[qt])
            if abs(r) > 0.15 or p < 0.05:
                print(f"  {lt} → {qt}: r={r:.3f} (p={p:.4f})")
            causal_chain_3[f"{lt}->{qt}"] = {"r": float(r), "p": float(p)}
    
    # 直接路径: 频谱参数 → 预测质量
    print(f"\n--- 直接路径: 频谱参数 → 预测质量 ---")
    direct_chain = {}
    for sp in spectral_params + ["gini", "top5_concentration", "delta_top5"]:
        for qt in quality_targets:
            r, p = spearmanr(params[sp], params[qt])
            print(f"  {sp} → {qt}: r={r:.3f} (p={p:.4f})")
            direct_chain[f"{sp}->{qt}"] = {"r": float(r), "p": float(p)}
    
    # ===== 多元回归: 频谱参数组合 → 预测质量 =====
    print(f"\n--- 多元回归: 频谱参数组合 → top1_prob ---")
    from scipy.stats import rankdata
    
    # 用rank做回归更鲁棒
    y = rankdata(params["top1_prob"])
    
    # 组合1: alpha + ratio_50
    x1 = rankdata(params["alpha"]) + rankdata(params["ratio_50"])
    r1, _ = spearmanr(x1, y)
    print(f"  alpha + ratio_50: r={r1:.3f}")
    
    # 组合2: gini + delta_top5
    x2 = rankdata(params["gini"]) + rankdata(params["delta_top5"])
    r2, _ = spearmanr(x2, y)
    print(f"  gini + delta_top5: r={r2:.3f}")
    
    # 组合3: top5_concentration + beta
    x3 = rankdata(params["top5_concentration"]) + rankdata(params["beta"])
    r3, _ = spearmanr(x3, y)
    print(f"  top5_concentration + beta: r={r3:.3f}")
    
    # 组合4: 全部
    x4 = (rankdata(params["alpha"]) + rankdata(params["ratio_50"]) + 
           rankdata(params["gini"]) + rankdata(params["beta"]) + 
           rankdata(params["delta_top5"]))
    r4, _ = spearmanr(x4, y)
    print(f"  全部组合: r={r4:.3f}")
    
    # ===== 中介效应检验 =====
    print(f"\n--- 中介效应检验 ---")
    # 如果"频谱→集中度→预测"路径比"频谱→预测"直接路径更强,
    # 则集中度是中介变量
    
    # 最佳频谱参数→top1_prob
    best_spectral_r = max(abs(direct_chain[f"{sp}->top1_prob"]["r"]) for sp in spectral_params)
    best_concentration_r = max(abs(direct_chain[f"{ct}->top1_prob"]["r"]) for ct in ["gini", "top5_concentration"])
    
    print(f"  最佳频谱参数→prob: |r|={best_spectral_r:.3f}")
    print(f"  最佳集中度→prob: |r|={best_concentration_r:.3f}")
    print(f"  中介效应: {'集中度是中介变量' if best_concentration_r > best_spectral_r else '直接路径更强'}")
    
    result = {
        "model": model_name, "n_texts": n,
        "param_stats": {key: {"mean": float(np.mean(params[key])), "std": float(np.std(params[key]))} for key in params},
        "causal_chain_1": causal_chain_1,
        "causal_chain_2": causal_chain_2,
        "causal_chain_3": causal_chain_3,
        "direct_chain": direct_chain,
        "multi_regression": {
            "alpha_ratio50": float(r1),
            "gini_delta_top5": float(r2),
            "top5_beta": float(r3),
            "all": float(r4),
        },
        "mediation": {
            "best_spectral_r": float(best_spectral_r),
            "best_concentration_r": float(best_concentration_r),
        },
    }
    return result


# ===================== P574: 统一语义编码方程 =====================
def experiment_p574(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P574: 统一语义编码方程
    目标: 找到跨模型一致的频谱编码结构
    
    核心方程: S(l+1) = alpha(l) * S(l) + beta(l) * f_mlp(l)
    但alpha/beta的含义和函数形式因模型而异
    
    本实验测量:
    1. 频谱传播方程的层间系数(alpha, beta, epsilon)的精确值
    2. 频谱形状的层级变化(幂律指数, 集中度)
    3. 频谱→logits→预测的完整数学路径
    4. 跨模型一致的编码不变量
    """
    print(f"\n{'='*60}")
    print(f"P574: 统一语义编码方程 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # ===== 1. 频谱传播方程层间系数 =====
    print(f"\n--- 频谱传播方程层间系数(30文本) ---")
    
    # 对30文本逐层测量频谱
    text = TEST_TEXTS[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # 逐层频谱系数
    layer_spectra = []
    for l in range(n_layers + 1):
        h_l = outputs.hidden_states[l][0, -1].detach().cpu().float().numpy()
        coeffs = U_wu[:, :50].T @ h_l
        abs_coeffs = np.abs(coeffs)
        energy = abs_coeffs**2
        total = np.sum(energy) + 1e-10
        spectrum = energy / total  # 归一化频谱
        layer_spectra.append(spectrum)
    
    del outputs
    torch.cuda.empty_cache()
    
    # 逐层alpha/beta
    layer_alphas = []
    layer_betas = []
    layer_epsilon_norms = []
    layer_spectral_corrs = []  # 频谱层间相关
    
    for l in range(1, n_layers + 1):
        # Alpha: S(l)在S(l-1)上的回归系数
        s_prev = layer_spectra[l-1]
        s_curr = layer_spectra[l]
        
        # 简单alpha = dot(S_l, S_{l-1}) / dot(S_{l-1}, S_{l-1})
        alpha = np.dot(s_curr, s_prev) / (np.dot(s_prev, s_prev) + 1e-10)
        
        # Epsilon
        epsilon = s_curr - alpha * s_prev
        epsilon_norm = np.linalg.norm(epsilon)
        
        # Beta (delta幅度)
        layer_alphas.append(alpha)
        layer_betas.append(epsilon_norm)
        layer_epsilon_norms.append(epsilon_norm)
        
        # 频谱相关
        corr, _ = pearsonr(s_curr, s_prev)
        layer_spectral_corrs.append(corr)
    
    # 分段统计
    n_third = n_layers // 3
    shallow_alpha = np.mean(layer_alphas[:n_third])
    mid_alpha = np.mean(layer_alphas[n_third:2*n_third])
    deep_alpha = np.mean(layer_alphas[2*n_third:])
    
    print(f"  浅层 alpha: {shallow_alpha:.4f}")
    print(f"  中层 alpha: {mid_alpha:.4f}")
    print(f"  深层 alpha: {deep_alpha:.4f}")
    print(f"  全层 alpha: {np.mean(layer_alphas):.4f}")
    print(f"  频谱层间相关(均值): {np.mean(layer_spectral_corrs):.4f}")
    
    # ===== 2. 频谱形状层级变化 =====
    print(f"\n--- 频谱形状层级变化 ---")
    
    # 对每个层测量幂律指数
    layer_power_exponents = []
    layer_top5_ratios = []
    
    for l in range(n_layers + 1):
        spectrum = layer_spectra[l]
        
        # Top-5集中度
        top5 = np.sum(np.sort(spectrum)[-5:])
        layer_top5_ratios.append(top5)
        
        # 幂律拟合(用对数空间的线性拟合)
        sorted_s = np.sort(spectrum)[::-1]
        sorted_s = sorted_s[sorted_s > 1e-10]  # 过滤零值
        if len(sorted_s) > 5:
            log_s = np.log(sorted_s)
            log_rank = np.log(np.arange(1, len(sorted_s) + 1))
            # 线性拟合
            slope, intercept = np.polyfit(log_rank, log_s, 1)
            layer_power_exponents.append(slope)
        else:
            layer_power_exponents.append(0)
    
    # 打印关键层
    print(f"  幂律指数趋势: 浅={np.mean(layer_power_exponents[:n_third]):.3f}, "
          f"中={np.mean(layer_power_exponents[n_third:2*n_third]):.3f}, "
          f"深={np.mean(layer_power_exponents[2*n_third:]):.3f}")
    print(f"  Top5集中度趋势: 浅={np.mean(layer_top5_ratios[:n_third]):.3f}, "
          f"中={np.mean(layer_top5_ratios[n_third:2*n_third]):.3f}, "
          f"深={np.mean(layer_top5_ratios[2*n_third:]):.3f}")
    
    # ===== 3. 频谱→logits→预测的数学路径 =====
    print(f"\n--- 频谱→logits→预测的数学路径(30文本) ---")
    
    path_data = []
    for text in TEST_TEXTS[:30]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_coeffs = U_wu[:, :50].T @ h_last
        
        # 完整logits
        full_logits = W_U @ h_last
        full_probs = np.exp(full_logits - np.max(full_logits))
        full_probs = full_probs / np.sum(full_probs)
        full_top1 = np.max(full_probs)
        
        # 逐步添加方向的logits重建
        for K in [1, 3, 5, 10, 20, 50]:
            partial_logits = np.zeros(full_logits.shape[0])
            for k in range(K):
                partial_logits += h_coeffs[k] * (W_U @ U_wu[:, k])
            
            # 重建质量
            cos_sim = np.dot(partial_logits, full_logits) / (np.linalg.norm(partial_logits) * np.linalg.norm(full_logits) + 1e-10)
            
            # Top-1预测是否一致
            partial_top1_id = np.argmax(partial_logits)
            full_top1_id = np.argmax(full_logits)
            top1_match = int(partial_top1_id == full_top1_id)
            
            # 重建概率
            partial_probs = np.exp(partial_logits - np.max(partial_logits))
            partial_probs = partial_probs / np.sum(partial_probs)
            partial_top1 = np.max(partial_probs)
            
            if K == 10:  # 只记录K=10
                path_data.append({
                    "K": K, "cos_sim": cos_sim, "top1_match": top1_match,
                    "partial_top1": partial_top1, "full_top1": full_top1,
                    "prob_ratio": partial_top1 / (full_top1 + 1e-10),
                })
        
        del outputs
        torch.cuda.empty_cache()
    
    # K=10的统计
    cos_sims = [d["cos_sim"] for d in path_data]
    top1_matches = [d["top1_match"] for d in path_data]
    prob_ratios = [d["prob_ratio"] for d in path_data]
    
    print(f"  K=10重建: cos_sim={np.mean(cos_sims):.4f}, top1匹配率={np.mean(top1_matches):.3f}, "
          f"概率比={np.mean(prob_ratios):.4f}")
    
    # 逐步重建曲线(用一条文本详细分析)
    print(f"\n--- 逐步重建曲线(首条文本) ---")
    text = TEST_TEXTS[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    h_coeffs = U_wu[:, :50].T @ h_last
    full_logits = W_U @ h_last
    
    reconstruction_curve = []
    for K in range(1, 51):
        partial_logits = np.zeros(full_logits.shape[0])
        for k in range(K):
            partial_logits += h_coeffs[k] * (W_U @ U_wu[:, k])
        cos_sim = np.dot(partial_logits, full_logits) / (np.linalg.norm(partial_logits) * np.linalg.norm(full_logits) + 1e-10)
        reconstruction_curve.append(float(cos_sim))
    
    for K in [1, 3, 5, 10, 20, 30, 50]:
        print(f"  K={K}: cos_sim={reconstruction_curve[K-1]:.4f}")
    
    del outputs
    torch.cuda.empty_cache()
    
    # ===== 4. 跨模型编码不变量 =====
    print(f"\n--- 编码不变量检测 ---")
    
    # 不变量1: 末层alpha的通用性
    # 不变量2: 频谱形状的幂律特征
    # 不变量3: 方向0-4的能量占比
    
    # 方向0-4总能量
    top5_energy_ratio = float(np.sum(S_wu[:5]**2) / np.sum(S_wu[:50]**2))
    
    # 幂律指数(全局)
    sorted_s = np.sort(S_wu[:50])[::-1]
    sorted_s = sorted_s[sorted_s > 1e-10]
    log_s = np.log(sorted_s)
    log_rank = np.log(np.arange(1, len(sorted_s) + 1))
    global_power_exponent, _ = np.polyfit(log_rank, log_s, 1)
    
    # 频谱Gini系数
    sorted_energy = np.sort(S_wu[:50]**2)[::-1]
    gini = 1 - 2 * np.sum(np.cumsum(sorted_energy) / np.sum(sorted_energy)) / len(sorted_energy)
    
    print(f"  方向0-4能量比: {top5_energy_ratio:.4f}")
    print(f"  全局幂律指数: {global_power_exponent:.3f}")
    print(f"  频谱Gini系数: {gini:.4f}")
    print(f"  末层alpha: {layer_alphas[-1]:.4f}")
    print(f"  末层频谱相关: {layer_spectral_corrs[-1]:.4f}")
    
    # ===== 5. 频谱编码方程验证 =====
    print(f"\n--- 频谱编码方程验证 ---")
    
    # 用5文本验证: S(l+1) ≈ alpha(l) * S(l) + epsilon(l)
    verification_errors = []
    
    for text in TEST_TEXTS[:5]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for l in range(1, n_layers + 1):
            h_l = outputs.hidden_states[l][0, -1].detach().cpu().float().numpy()
            h_l1 = outputs.hidden_states[l-1][0, -1].detach().cpu().float().numpy()
            
            s_l = np.abs(U_wu[:, :50].T @ h_l)**2
            s_l1 = np.abs(U_wu[:, :50].T @ h_l1)**2
            
            # 归一化
            s_l = s_l / (np.sum(s_l) + 1e-10)
            s_l1 = s_l1 / (np.sum(s_l1) + 1e-10)
            
            # 预测
            alpha_l = layer_alphas[min(l-1, len(layer_alphas)-1)]
            predicted = alpha_l * s_l1
            error = np.linalg.norm(s_l - predicted) / (np.linalg.norm(s_l) + 1e-10)
            verification_errors.append(error)
        
        del outputs
        torch.cuda.empty_cache()
    
    print(f"  预测误差(归一化L2): mean={np.mean(verification_errors):.4f}, std={np.std(verification_errors):.4f}")
    print(f"  浅层误差: {np.mean(verification_errors[:n_third*5]):.4f}")
    print(f"  中层误差: {np.mean(verification_errors[n_third*5:2*n_third*5]):.4f}")
    print(f"  深层误差: {np.mean(verification_errors[2*n_third*5:]):.4f}")
    
    result = {
        "model": model_name, "n_layers": n_layers,
        "spectral_propagation": {
            "shallow_alpha": float(shallow_alpha),
            "mid_alpha": float(mid_alpha),
            "deep_alpha": float(deep_alpha),
            "mean_alpha": float(np.mean(layer_alphas)),
            "mean_spectral_corr": float(np.mean(layer_spectral_corrs)),
            "layer_alphas": [float(a) for a in layer_alphas],
            "layer_spectral_corrs": [float(c) for c in layer_spectral_corrs],
        },
        "spectral_shape": {
            "shallow_power_exp": float(np.mean(layer_power_exponents[:n_third])),
            "mid_power_exp": float(np.mean(layer_power_exponents[n_third:2*n_third])),
            "deep_power_exp": float(np.mean(layer_power_exponents[2*n_third:])),
            "shallow_top5": float(np.mean(layer_top5_ratios[:n_third])),
            "mid_top5": float(np.mean(layer_top5_ratios[n_third:2*n_third])),
            "deep_top5": float(np.mean(layer_top5_ratios[2*n_third:])),
        },
        "logit_path": {
            "K10_cos_mean": float(np.mean(cos_sims)),
            "K10_top1_match_rate": float(np.mean(top1_matches)),
            "reconstruction_curve": reconstruction_curve,
        },
        "encoding_invariants": {
            "top5_energy_ratio": top5_energy_ratio,
            "global_power_exponent": float(global_power_exponent),
            "gini": float(gini),
            "final_layer_alpha": float(layer_alphas[-1]),
            "final_spectral_corr": float(layer_spectral_corrs[-1]),
        },
        "equation_verification": {
            "mean_error": float(np.mean(verification_errors)),
            "std_error": float(np.std(verification_errors)),
        },
    }
    return result


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p571", "p572", "p573", "p574"])
    args = parser.parse_args()
    
    start_time = time.time()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=300)
    
    if args.experiment == "p571":
        result = experiment_p571(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p572":
        result = experiment_p572(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p573":
        result = experiment_p573(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p574":
        result = experiment_p574(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = "results/phase_cxxxii"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.model}_{args.experiment}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    
    del model
    torch.cuda.empty_cache()
    print("GPU内存已释放")


if __name__ == "__main__":
    main()
