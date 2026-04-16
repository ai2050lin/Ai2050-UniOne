"""
Phase CXXXIII: 频谱→logit_gap的桥梁 (P575-P578)
================================================
P575: 频谱参数→logit_var→logit_gap路径分析 — 为什么频谱集中度→logit_var负相关?
P576: logit_gap的频谱分解 — logit_gap由哪些W_U方向贡献?
P577: W_U方向功能分类修正 — 用Top-100+语义类别池替代Top-30
P578: 频谱→logit_gap统一方程 — 跨模型一致的数学形式

大规模测试: 210文本 × 3模型
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
    """计算W_U的SVD"""
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


# ===================== P575: 频谱参数→logit_var→logit_gap路径分析 =====================
def experiment_p575(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P575: 频谱参数→logit_var→logit_gap路径分析
    核心问题: 为什么频谱集中度→logit_var是负相关?频谱→logit_gap的完整路径是什么?
    
    创新点:
    1. 逐方向分析: 每个W_U方向对logit_var和logit_gap的贡献
    2. 噪声方向vs信号方向: logit_var是来自信号方向(前K)还是噪声方向(K+)?
    3. 频谱参数的多步路径: alpha→ratio→concentration→logit_var→logit_gap→prob
    """
    print(f"\n{'='*60}")
    print(f"P575: 频谱参数→logit_var→logit_gap路径分析 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # ===== 频谱系数 =====
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # 频谱参数
        ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        ratio_100 = np.sum(abs_coeffs[:100]**2) / total_energy
        
        # Alpha/Beta
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        delta = h_last - h_prev
        beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        
        # ===== 逐方向logits贡献 =====
        # logit_i = W_U[i,:] @ h_last = sum_k h_coeffs[k] * W_U[i,:] @ U_wu[:,k]
        full_logits = W_U @ h_last  # [vocab_size]
        
        # Top-1和Top-2的logit
        sorted_ids = np.argsort(full_logits)[::-1]
        top1_id, top2_id = sorted_ids[0], sorted_ids[1]
        logit_gap = full_logits[top1_id] - full_logits[top2_id]
        
        # 每个方向对logit_gap的贡献
        # logit_gap = logit_top1 - logit_top2
        # 每个方向k对logit_gap的贡献 = h_coeffs[k] * (W_U[top1,:] @ U_wu[:,k] - W_U[top2,:] @ U_wu[:,k])
        dir_gap_contributions = np.zeros(50)
        for k in range(50):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            dir_gap_contributions[k] = h_coeffs[k] * (wu_top1_proj - wu_top2_proj)
        
        # 前10/20/50方向的logit_gap贡献比
        gap_contrib_10 = np.sum(np.abs(dir_gap_contributions[:10])) / (np.sum(np.abs(dir_gap_contributions)) + 1e-10)
        gap_contrib_20 = np.sum(np.abs(dir_gap_contributions[:20])) / (np.sum(np.abs(dir_gap_contributions)) + 1e-10)
        gap_contrib_50 = np.sum(np.abs(dir_gap_contributions[:50])) / (np.sum(np.abs(dir_gap_contributions)) + 1e-10)
        
        # ===== 逐方向logit方差贡献 =====
        # logit_var = var(W_U @ h_last) = sum_k h_coeffs[k]^2 * var(W_U @ U_wu[:,k])
        #          + 2 * sum_{k<j} h_coeffs[k]*h_coeffs[j]*cov(W_U @ U_wu[:,k], W_U @ U_wu[:,j])
        # 简化: 每个方向的独立贡献
        dir_logit_vectors = np.zeros((50, W_U.shape[0]))
        for k in range(50):
            dir_logit_vectors[k] = h_coeffs[k] * (W_U @ U_wu[:, k])
        
        # 每个方向的logit方差贡献
        logit_mean = np.mean(full_logits)
        dir_var_contrib = np.zeros(50)
        for k in range(50):
            dir_var_contrib[k] = np.sum((dir_logit_vectors[k] - np.mean(dir_logit_vectors[k]))**2)
        
        total_dir_var = np.sum(dir_var_contrib)
        var_ratio_10 = np.sum(dir_var_contrib[:10]) / (total_dir_var + 1e-10)
        var_ratio_50 = np.sum(dir_var_contrib[:50]) / (total_dir_var + 1e-10)
        
        # 噪声方向(K>50)的logit方差贡献
        residual_logits = full_logits - np.sum(dir_logit_vectors, axis=0)
        noise_var = np.var(residual_logits)
        signal_var = np.var(full_logits)
        noise_var_ratio = noise_var / (signal_var + 1e-10)
        
        # ===== Logit分布特征 =====
        with torch.no_grad():
            logits_t = outputs.logits[0, -1].float()
            probs = torch.softmax(logits_t, dim=0)
        top1_prob = probs.max().item()
        
        logit_var = float(np.var(full_logits))
        logit_kurtosis = float(np.mean((full_logits - np.mean(full_logits))**4) / (np.var(full_logits)**2 + 1e-10))
        
        # ===== 频谱集中度参数 =====
        sorted_energy = np.sort(abs_coeffs[:50]**2)[::-1]
        gini = 1 - 2 * np.sum(np.cumsum(sorted_energy) / np.sum(sorted_energy)) / len(sorted_energy)
        top1_concentration = sorted_energy[0] / total_energy
        top5_concentration = np.sum(sorted_energy[:5]) / total_energy
        
        # 频谱熵
        normalized_energy = abs_coeffs[:50]**2 / total_energy
        spectral_entropy = -np.sum(normalized_energy * np.log(normalized_energy + 1e-10))
        
        # ===== 路径分析: 频谱→集中度→logit_var→logit_gap→prob =====
        all_data.append({
            "alpha": alpha, "beta": beta,
            "ratio_10": ratio_10, "ratio_50": ratio_50, "ratio_100": ratio_100,
            "gini": gini, "top1_concentration": top1_concentration, "top5_concentration": top5_concentration,
            "spectral_entropy": spectral_entropy,
            "logit_var": logit_var, "logit_kurtosis": logit_kurtosis,
            "logit_gap": logit_gap,
            "top1_prob": top1_prob,
            "gap_contrib_10": gap_contrib_10, "gap_contrib_20": gap_contrib_20, "gap_contrib_50": gap_contrib_50,
            "var_ratio_10": var_ratio_10, "var_ratio_50": var_ratio_50,
            "noise_var_ratio": noise_var_ratio,
            "dir_gap_contributions": dir_gap_contributions.tolist(),
            "dir_var_contrib": dir_var_contrib.tolist(),
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    # ===== 统计分析 =====
    n = len(all_data)
    params = {key: np.array([d[key] for d in all_data]) for key in all_data[0].keys() if key not in ["dir_gap_contributions", "dir_var_contrib"]}
    
    print(f"\n--- 参数统计(n={n}) ---")
    for key in ["logit_gap", "logit_var", "gap_contrib_10", "var_ratio_10", "noise_var_ratio"]:
        print(f"  {key}: {np.mean(params[key]):.4f} ± {np.std(params[key]):.4f}")
    
    # ===== 完整路径分析 =====
    print(f"\n=== 完整路径分析: 频谱→集中度→logit_var→logit_gap→prob ===")
    
    # 路径1: 频谱参数 → logit_var
    print(f"\n--- 路径1: 频谱参数 → logit_var ---")
    for sp in ["ratio_10", "ratio_50", "gini", "top5_concentration", "spectral_entropy"]:
        r, p = spearmanr(params[sp], params["logit_var"])
        print(f"  {sp} → logit_var: r={r:.3f} (p={p:.4f})")
    
    # 路径2: 频谱参数 → logit_gap
    print(f"\n--- 路径2: 频谱参数 → logit_gap ---")
    for sp in ["ratio_10", "ratio_50", "gini", "top5_concentration", "spectral_entropy"]:
        r, p = spearmanr(params[sp], params["logit_gap"])
        print(f"  {sp} → logit_gap: r={r:.3f} (p={p:.4f})")
    
    # 路径3: logit_var → logit_gap → prob
    print(f"\n--- 路径3: logit_var → logit_gap → prob ---")
    for pair in [("logit_var", "logit_gap"), ("logit_var", "top1_prob"), ("logit_gap", "top1_prob")]:
        r, p = spearmanr(params[pair[0]], params[pair[1]])
        print(f"  {pair[0]} → {pair[1]}: r={r:.3f} (p={p:.4f})")
    
    # 路径4: 逐方向贡献 → logit_gap
    print(f"\n--- 路径4: 逐方向贡献与logit_gap的关系 ---")
    for key in ["gap_contrib_10", "gap_contrib_20", "gap_contrib_50"]:
        r, p = spearmanr(params[key], params["logit_gap"])
        print(f"  {key} → logit_gap: r={r:.3f} (p={p:.4f})")
    
    for key in ["gap_contrib_10", "var_ratio_10"]:
        r, p = spearmanr(params[key], params["top1_prob"])
        print(f"  {key} → top1_prob: r={r:.3f} (p={p:.4f})")
    
    # 路径5: 噪声方向的logit贡献
    print(f"\n--- 路径5: 噪声方向logit贡献 ---")
    r, p = spearmanr(params["noise_var_ratio"], params["logit_var"])
    print(f"  noise_var_ratio → logit_var: r={r:.3f} (p={p:.4f})")
    r, p = spearmanr(params["noise_var_ratio"], params["logit_gap"])
    print(f"  noise_var_ratio → logit_gap: r={r:.3f} (p={p:.4f})")
    r, p = spearmanr(params["noise_var_ratio"], params["top1_prob"])
    print(f"  noise_var_ratio → top1_prob: r={r:.3f} (p={p:.4f})")
    
    # ===== 中介效应: logit_var是否是频谱→logit_gap的中介? =====
    print(f"\n--- 中介效应检验 ---")
    # 直接路径: ratio_50 → logit_gap
    r_direct, _ = spearmanr(params["ratio_50"], params["logit_gap"])
    # 中介路径: ratio_50 → logit_var → logit_gap
    r1, _ = spearmanr(params["ratio_50"], params["logit_var"])
    r2, _ = spearmanr(params["logit_var"], params["logit_gap"])
    print(f"  直接: ratio_50 → logit_gap: r={r_direct:.3f}")
    print(f"  中介: ratio_50 → logit_var (r={r1:.3f}), logit_var → logit_gap (r={r2:.3f})")
    print(f"  中介乘积: r1*r2={r1*r2:.3f}")
    print(f"  中介效应: {'显著' if abs(r1*r2) > abs(r_direct) else '不显著'}")
    
    # ===== 逐方向logit_gap贡献的均值分析 =====
    print(f"\n--- 逐方向logit_gap贡献均值 ---")
    mean_gap_contrib = np.mean([np.array(d["dir_gap_contributions"]) for d in all_data], axis=0)
    mean_var_contrib = np.mean([np.array(d["dir_var_contrib"]) for d in all_data], axis=0)
    
    # 归一化
    gap_norm = np.sum(np.abs(mean_gap_contrib)) + 1e-10
    var_norm = np.sum(mean_var_contrib) + 1e-10
    
    for d in [0, 1, 2, 3, 4, 5, 10, 20, 30, 49]:
        gap_pct = np.abs(mean_gap_contrib[d]) / gap_norm * 100
        var_pct = mean_var_contrib[d] / var_norm * 100
        print(f"  Dir{d}: gap贡献={gap_pct:.2f}%, var贡献={var_pct:.2f}%")
    
    # Top-5贡献方向
    top5_gap_dirs = np.argsort(np.abs(mean_gap_contrib))[::-1][:5]
    top5_var_dirs = np.argsort(mean_var_contrib)[::-1][:5]
    print(f"\n  logit_gap Top-5贡献方向: {top5_gap_dirs.tolist()}")
    print(f"  logit_var Top-5贡献方向: {top5_var_dirs.tolist()}")
    
    result = {
        "model": model_name, "n_texts": n,
        "param_stats": {key: {"mean": float(np.mean(params[key])), "std": float(np.std(params[key]))} 
                       for key in params},
        "path_analysis": {
            "spectral_to_logit_var": {sp: {"r": float(spearmanr(params[sp], params["logit_var"])[0]),
                                           "p": float(spearmanr(params[sp], params["logit_var"])[1])}
                                      for sp in ["ratio_10", "ratio_50", "gini", "top5_concentration"]},
            "spectral_to_logit_gap": {sp: {"r": float(spearmanr(params[sp], params["logit_gap"])[0]),
                                           "p": float(spearmanr(params[sp], params["logit_gap"])[1])}
                                      for sp in ["ratio_10", "ratio_50", "gini", "top5_concentration"]},
            "logit_var_to_logit_gap": {"r": float(spearmanr(params["logit_var"], params["logit_gap"])[0]),
                                       "p": float(spearmanr(params["logit_var"], params["logit_gap"])[1])},
            "logit_gap_to_prob": {"r": float(spearmanr(params["logit_gap"], params["top1_prob"])[0]),
                                  "p": float(spearmanr(params["logit_gap"], params["top1_prob"])[1])},
        },
        "mediation": {
            "r_direct": float(r_direct),
            "r1_spectral_to_var": float(r1),
            "r2_var_to_gap": float(r2),
            "mediation_product": float(r1 * r2),
        },
        "top5_gap_dirs": top5_gap_dirs.tolist(),
        "top5_var_dirs": top5_var_dirs.tolist(),
        "mean_gap_contrib_pct": (np.abs(mean_gap_contrib) / gap_norm * 100).tolist(),
        "mean_var_contrib_pct": (mean_var_contrib / var_norm * 100).tolist(),
    }
    return result


# ===================== P576: logit_gap的频谱分解 =====================
def experiment_p576(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P576: logit_gap的频谱分解
    核心问题: logit_gap到底由哪些W_U方向贡献?这些方向的语义内容是什么?
    
    创新点:
    1. 逐方向分析logit_gap贡献的正负符号
    2. 不同文本类型(短句vs长句vs复杂句)的方向贡献差异
    3. logit_gap方向贡献的层级演化(哪些层的频谱决定了logit_gap)
    """
    print(f"\n{'='*60}")
    print(f"P576: logit_gap的频谱分解 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # ===== 1. 全文本logit_gap分解 =====
    all_gap_decompositions = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        
        # 完整logits
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        top1_id, top2_id = sorted_ids[0], sorted_ids[1]
        logit_gap = full_logits[top1_id] - full_logits[top2_id]
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        # 逐方向分解
        dir_contrib = np.zeros(50)
        for k in range(50):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            dir_contrib[k] = h_coeffs[k] * (wu_top1_proj - wu_top2_proj)
        
        # 正贡献方向 vs 负贡献方向
        pos_contrib = np.sum(dir_contrib[dir_contrib > 0])
        neg_contrib = np.sum(dir_contrib[dir_contrib < 0])
        net_contrib = pos_contrib + neg_contrib
        
        all_gap_decompositions.append({
            "text_type": "short" if len(text.split()) < 8 else ("medium" if len(text.split()) < 15 else "long"),
            "logit_gap": logit_gap, "top1_prob": top1_prob,
            "dir_contrib": dir_contrib.tolist(),
            "pos_contrib": pos_contrib, "neg_contrib": neg_contrib,
            "net_contrib": net_contrib,
            "top1_id": int(top1_id), "top2_id": int(top2_id),
            "top1_token": safe_decode(tokenizer, top1_id),
            "top2_token": safe_decode(tokenizer, top2_id),
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    # ===== 2. 全局统计 =====
    print(f"\n--- logit_gap分解全局统计(n={len(all_gap_decompositions)}) ---")
    mean_dir_contrib = np.mean([np.array(d["dir_contrib"]) for d in all_gap_decompositions], axis=0)
    
    # 归一化
    total_abs = np.sum(np.abs(mean_dir_contrib)) + 1e-10
    for d in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 49]:
        pct = np.abs(mean_dir_contrib[d]) / total_abs * 100
        sign = "+" if mean_dir_contrib[d] > 0 else "-"
        print(f"  Dir{d}: {sign}{pct:.2f}% (raw={mean_dir_contrib[d]:.4f})")
    
    # Top贡献方向
    sorted_by_abs = np.argsort(np.abs(mean_dir_contrib))[::-1]
    print(f"\n  Top-10 logit_gap贡献方向: {sorted_by_abs[:10].tolist()}")
    print(f"  Top-10贡献占比: {np.sum(np.abs(mean_dir_contrib[sorted_by_abs[:10]]))/total_abs*100:.1f}%")
    
    # ===== 3. 正/负贡献分析 =====
    print(f"\n--- 正/负贡献分析 ---")
    pos_contribs = [d["pos_contrib"] for d in all_gap_decompositions]
    neg_contribs = [d["neg_contrib"] for d in all_gap_decompositions]
    print(f"  正贡献均值: {np.mean(pos_contribs):.4f}")
    print(f"  负贡献均值: {np.mean(neg_contribs):.4f}")
    print(f"  净贡献均值: {np.mean([d["net_contrib"] for d in all_gap_decompositions]):.4f}")
    
    # ===== 4. 文本类型差异 =====
    print(f"\n--- 文本类型差异 ---")
    for text_type in ["short", "medium", "long"]:
        subset = [d for d in all_gap_decompositions if d["text_type"] == text_type]
        if len(subset) > 0:
            mean_gap = np.mean([d["logit_gap"] for d in subset])
            mean_prob = np.mean([d["top1_prob"] for d in subset])
            mean_dir = np.mean([np.array(d["dir_contrib"]) for d in subset], axis=0)
            top5_dirs = np.argsort(np.abs(mean_dir))[::-1][:5]
            print(f"  {text_type}(n={len(subset)}): gap={mean_gap:.3f}, prob={mean_prob:.3f}, top5_dirs={top5_dirs.tolist()}")
    
    # ===== 5. 层级演化: 哪些层决定了logit_gap =====
    print(f"\n--- logit_gap的层级演化(30文本) ---")
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    
    layer_gap_data = {li: [] for li in sample_layers}
    
    for text in TEST_TEXTS[:30]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for li in sample_layers:
            h_l = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
            logits_l = W_U @ h_l
            sorted_l = np.argsort(logits_l)[::-1]
            gap_l = logits_l[sorted_l[0]] - logits_l[sorted_l[1]]
            layer_gap_data[li].append(gap_l)
        
        del outputs
        torch.cuda.empty_cache()
    
    print(f"  各层logit_gap均值:")
    for li in sorted(sample_layers):
        mean_gap = np.mean(layer_gap_data[li])
        print(f"    L{li}: {mean_gap:.4f}")
    
    # 层间logit_gap与末层logit_gap的相关
    final_gaps = layer_gap_data[n_layers-1]
    print(f"\n  各层logit_gap与末层logit_gap的相关:")
    for li in sorted(sample_layers[:-1]):
        r, p = spearmanr(layer_gap_data[li], final_gaps)
        print(f"    L{li} → L{n_layers-1}: r={r:.3f} (p={p:.4f})")
    
    # ===== 6. 频谱方向语义映射 =====
    print(f"\n--- logit_gap Top-10贡献方向的语义内容 ---")
    top10_dirs = sorted_by_abs[:10]
    
    for d in top10_dirs:
        direction = U_wu[:, d]
        projections = W_U @ direction
        top_pos_ids = np.argsort(projections)[::-1][:5]
        top_neg_ids = np.argsort(projections)[:5]
        top_pos = [safe_decode(tokenizer, tid) for tid in top_pos_ids]
        top_neg = [safe_decode(tokenizer, tid) for tid in top_neg_ids]
        sign = "+" if mean_dir_contrib[d] > 0 else "-"
        print(f"  Dir{d}({sign}): +{top_pos} -{top_neg}")
    
    result = {
        "model": model_name, "n_texts": len(all_gap_decompositions),
        "mean_dir_contrib": mean_dir_contrib.tolist(),
        "mean_dir_contrib_pct": (np.abs(mean_dir_contrib) / total_abs * 100).tolist(),
        "top10_gap_dirs": top10_dirs.tolist(),
        "top10_gap_pct": float(np.sum(np.abs(mean_dir_contrib[top10_dirs])) / total_abs * 100),
        "pos_contrib_mean": float(np.mean(pos_contribs)),
        "neg_contrib_mean": float(np.mean(neg_contribs)),
        "layer_gap_stats": {str(li): {"mean": float(np.mean(layer_gap_data[li])),
                                       "std": float(np.std(layer_gap_data[li]))}
                            for li in sample_layers},
        "layer_gap_corr_to_final": {str(li): {"r": float(spearmanr(layer_gap_data[li], final_gaps)[0]),
                                               "p": float(spearmanr(layer_gap_data[li], final_gaps)[1])}
                                     for li in sample_layers[:-1]},
        "text_type_stats": {},
    }
    
    for text_type in ["short", "medium", "long"]:
        subset = [d for d in all_gap_decompositions if d["text_type"] == text_type]
        if len(subset) > 0:
            result["text_type_stats"][text_type] = {
                "n": len(subset),
                "mean_gap": float(np.mean([d["logit_gap"] for d in subset])),
                "mean_prob": float(np.mean([d["top1_prob"] for d in subset])),
            }
    
    return result


# ===================== P577: W_U方向功能分类修正 =====================
def experiment_p577(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P577: W_U方向功能分类修正
    问题: P572的Top-30分类阈值失败, 需要更精细的方法
    
    创新点:
    1. Top-100替代Top-30
    2. 扩大语义类别池: 30+类别
    3. 连续分类分数而非二分分类
    4. 方向间的聚类分析
    """
    print(f"\n{'='*60}")
    print(f"P577: W_U方向功能分类修正 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    
    # ===== 扩大的语义类别池 =====
    categories = {
        # 格式/结构
        "punctuation": set([",", ".", "!", "?", ";", ":", "-", "(", ")", "\"", "'", 
                           "\u201c", "\u201d", "\u2018", "\u2019", "\u2026", "\u2013", "\u2014"]),
        "whitespace": set(["\n", "\t", " ", "\u00a0"]),
        "bracket": set(["[", "]", "{", "}", "<", ">", "(", ")"]),
        "special_char": set(["#", "@", "$", "%", "^", "&", "*", "+", "=", "/", "\\", "|", "~", "`", "_"]),
        
        # 功能词
        "determiner": set(["the", "a", "an", "this", "that", "these", "those", "some", "any", "all", 
                          "every", "each", "no", "other", "another", "such", "many", "much", "few", "little"]),
        "pronoun": set(["he", "she", "it", "they", "we", "you", "i", "his", "her", "its", "their", 
                       "our", "my", "him", "them", "us", "me", "who", "whom", "which", "what", "that"]),
        "auxiliary": set(["is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                         "do", "does", "did", "will", "would", "can", "could", "shall", "should", 
                         "may", "might", "must", "need"]),
        "preposition": set(["of", "in", "for", "on", "with", "at", "by", "from", "to", "into", 
                           "through", "about", "between", "under", "over", "after", "before", "during",
                           "without", "within", "along", "across", "behind", "beyond", "upon"]),
        "conjunction": set(["and", "but", "or", "nor", "yet", "so", "because", "although", "while", 
                           "if", "unless", "since", "until", "whereas", "whether", "though"]),
        "adverb": set(["not", "very", "also", "just", "more", "most", "only", "even", "still", 
                      "already", "never", "always", "often", "sometimes", "usually", "quite", "rather",
                      "almost", "perhaps", "probably", "certainly", "indeed", "however"]),
        "particle": set(["up", "out", "off", "down", "away", "back", "on", "over", "around", "about"]),
        
        # 内容词 - 名词
        "person": set(["man", "woman", "child", "boy", "girl", "people", "person", "children",
                      "men", "women", "baby", "adult", "friend", "family", "mother", "father"]),
        "animal": set(["cat", "dog", "bird", "fish", "horse", "elephant", "tiger", "lion", "bear", 
                      "whale", "eagle", "snake", "rabbit", "deer", "wolf", "shark", "dolphin", "monkey"]),
        "place": set(["city", "country", "world", "house", "home", "school", "office", "street",
                     "road", "river", "mountain", "island", "village", "town", "park", "garden",
                     "forest", "desert", "ocean", "sea", "lake"]),
        "time": set(["year", "day", "time", "week", "month", "hour", "minute", "century", "decade",
                    "morning", "evening", "night", "summer", "winter", "spring", "autumn", "future", "past"]),
        "abstract_noun": set(["life", "death", "love", "war", "peace", "freedom", "truth", "beauty",
                             "justice", "power", "knowledge", "wisdom", "courage", "hope", "fear",
                             "nature", "science", "art", "history", "society", "culture", "education",
                             "technology", "democracy", "equality", "consciousness"]),
        "concrete_noun": set(["car", "book", "water", "food", "hand", "head", "eye", "door", "table",
                             "chair", "wall", "window", "tree", "flower", "stone", "fire", "air",
                             "earth", "sun", "moon", "star", "rain", "snow", "wind", "cloud"]),
        
        # 内容词 - 动词
        "motion_verb": set(["run", "walk", "fly", "swim", "climb", "jump", "fall", "rise", "move",
                           "turn", "drive", "ride", "travel", "cross", "enter", "leave", "return",
                           "approach", "pass", "reach", "follow", "lead"]),
        "cognitive_verb": set(["think", "know", "believe", "understand", "remember", "forget", "learn",
                              "discover", "imagine", "consider", "realize", "recognize", "wonder",
                              "doubt", "suppose", "assume", "expect", "predict", "analyze", "explain"]),
        "communication_verb": set(["say", "tell", "speak", "talk", "write", "read", "ask", "answer",
                                  "call", "explain", "describe", "discuss", "announce", "report",
                                  "claim", "argue", "suggest", "recommend", "declare", "state"]),
        "creation_verb": set(["make", "create", "build", "design", "develop", "produce", "generate",
                             "construct", "compose", "invent", "establish", "form", "shape", "craft"]),
        "change_verb": set(["become", "grow", "change", "transform", "convert", "evolve", "improve",
                           "increase", "decrease", "expand", "reduce", "shift", "adapt", "modify"]),
        
        # 内容词 - 形容词
        "size_adj": set(["big", "small", "large", "tiny", "huge", "little", "massive", "enormous",
                        "vast", "minute", "great", "minor", "major", "significant", "immense"]),
        "color_adj": set(["red", "blue", "green", "yellow", "black", "white", "orange", "purple",
                         "pink", "brown", "gray", "dark", "light", "bright"]),
        "emotion_adj": set(["happy", "sad", "angry", "afraid", "proud", "ashamed", "jealous",
                          "grateful", "anxious", "calm", "excited", "bored", "confused", "lonely"]),
        "quality_adj": set(["good", "bad", "beautiful", "ugly", "strong", "weak", "fast", "slow",
                          "hard", "soft", "hot", "cold", "new", "old", "rich", "poor", "clean", "dirty"]),
        
        # 特殊类别
        "number": set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "hundred", "thousand", "million", "first", "second", "third", "half", "quarter"]),
        "negation": set(["not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere",
                        "without", "cannot", "hardly", "scarcely", "barely", "rarely", "seldom"]),
        "technology": set(["computer", "internet", "software", "algorithm", "digital", "data", "network",
                         "system", "machine", "robot", "artificial", "quantum", "cyber", "electronic"]),
        "science": set(["research", "experiment", "theory", "hypothesis", "analysis", "evidence",
                       "physics", "chemistry", "biology", "mathematics", "observation", "formula"]),
        "non_english": set(),  # 将在运行时填充
    }
    
    # ===== 1. 方向分类(用Top-100+连续分数) =====
    direction_scores = {}  # 每个方向的类别分数
    
    for d in range(50):
        direction = U_wu[:, d]
        projections = W_U @ direction
        
        # Top-100正负token的类别分布
        top_pos_ids = np.argsort(projections)[::-1][:100]
        top_neg_ids = np.argsort(projections)[:100]
        top_ids = np.argsort(np.abs(projections))[::-1][:100]
        
        cat_scores = defaultdict(float)
        
        for tid in top_ids:
            try:
                tok_str = tokenizer.decode([tid]).strip().lower()
                weight = abs(projections[tid]) / (np.sum(np.abs(projections[top_ids])) + 1e-10)
                for cat_name, cat_set in categories.items():
                    if cat_name == "non_english":
                        continue
                    if tok_str in cat_set:
                        cat_scores[cat_name] += weight * 100
                        break
            except:
                cat_scores["special_char"] += 0.5
        
        direction_scores[d] = dict(cat_scores)
    
    # ===== 2. 基于分数的软分类 =====
    # 定义大类: format = punctuation + whitespace + bracket + special_char
    #           function = determiner + pronoun + auxiliary + preposition + conjunction + adverb + particle
    #           content = 所有其他
    
    format_cats = {"punctuation", "whitespace", "bracket", "special_char"}
    function_cats = {"determiner", "pronoun", "auxiliary", "preposition", "conjunction", "adverb", "particle"}
    
    direction_class = {}
    direction_class_scores = {}
    
    for d in range(50):
        scores = direction_scores[d]
        format_score = sum(scores.get(cat, 0) for cat in format_cats)
        function_score = sum(scores.get(cat, 0) for cat in function_cats)
        content_score = sum(v for cat, v in scores.items() if cat not in format_cats and cat not in function_cats)
        
        direction_class_scores[d] = {
            "format": format_score,
            "function": function_score,
            "content": content_score,
        }
        
        # 分类: 用加权分数
        total = format_score + function_score + content_score + 1e-10
        if format_score / total > 0.4:
            direction_class[d] = "format"
        elif function_score / total > 0.4:
            direction_class[d] = "function"
        elif content_score / total > 0.3:
            direction_class[d] = "content"
        else:
            direction_class[d] = "mixed"
    
    # 统计
    class_counts = defaultdict(int)
    for d, cls in direction_class.items():
        class_counts[cls] += 1
    
    print(f"\n--- 方向分类统计(Top-100方法) ---")
    for cls, cnt in sorted(class_counts.items()):
        dirs = [d for d, c in direction_class.items() if c == cls]
        print(f"  {cls}: {cnt}个方向 — dirs={dirs}")
    
    # ===== 3. 各方向的详细分类分数 =====
    print(f"\n--- 方向0-19分类分数 ---")
    for d in range(20):
        cs = direction_class_scores[d]
        cls = direction_class[d]
        s = direction_scores[d]
        top_cats = sorted(s.items(), key=lambda x: -x[1])[:3]
        cat_str = ", ".join([f"{k}:{v:.1f}" for k, v in top_cats if v > 0])
        print(f"  Dir{d}({cls}): format={cs['format']:.1f} func={cs['function']:.1f} content={cs['content']:.1f} | {cat_str}")
    
    # ===== 4. 大规模文本频谱分析(用新分类) =====
    print(f"\n--- 大规模文本频谱分析(新分类, n={len(TEST_TEXTS)}) ---")
    all_text_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_coeffs = U_wu[:, :50].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # 各类方向能量
        format_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "format") / total_energy
        function_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "function") / total_energy
        content_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "content") / total_energy
        mixed_energy = sum(abs_coeffs[d]**2 for d in range(50) if direction_class.get(d) == "mixed") / total_energy
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        # Logit gap
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        logit_gap = full_logits[sorted_ids[0]] - full_logits[sorted_ids[1]]
        
        all_text_data.append({
            "format_energy": format_energy,
            "function_energy": function_energy,
            "content_energy": content_energy,
            "mixed_energy": mixed_energy,
            "top1_prob": top1_prob,
            "logit_gap": logit_gap,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    # 统计
    format_energies = np.array([d["format_energy"] for d in all_text_data])
    function_energies = np.array([d["function_energy"] for d in all_text_data])
    content_energies = np.array([d["content_energy"] for d in all_text_data])
    top1_probs = np.array([d["top1_prob"] for d in all_text_data])
    logit_gaps = np.array([d["logit_gap"] for d in all_text_data])
    
    print(f"\n  格式方向能量: {np.mean(format_energies):.4f} ± {np.std(format_energies):.4f}")
    print(f"  功能方向能量: {np.mean(function_energies):.4f} ± {np.std(function_energies):.4f}")
    print(f"  内容方向能量: {np.mean(content_energies):.4f} ± {np.std(content_energies):.4f}")
    
    # 因果链
    print(f"\n--- 各类能量→预测质量因果链 ---")
    for name, energies in [("format", format_energies), ("function", function_energies), ("content", content_energies)]:
        r_prob, p_prob = spearmanr(energies, top1_probs)
        r_gap, p_gap = spearmanr(energies, logit_gaps)
        print(f"  {name}_energy → top1_prob: r={r_prob:.3f} (p={p_prob:.4f})")
        print(f"  {name}_energy → logit_gap: r={r_gap:.3f} (p={p_gap:.4f})")
    
    # ===== 5. 方向聚类分析 =====
    print(f"\n--- 方向聚类分析 ---")
    # 基于方向在词表空间的投影模式做聚类
    dir_projections = np.zeros((50, min(1000, W_U.shape[0])))
    for d in range(50):
        direction = U_wu[:, d]
        proj = W_U @ direction
        # 取Top-1000绝对值位置
        top_ids = np.argsort(np.abs(proj))[::-1][:1000]
        dir_projections[d, :len(top_ids)] = np.abs(proj[top_ids])
    
    # 简单聚类: 基于余弦相似度
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 归一化
    norms = np.linalg.norm(dir_projections, axis=1, keepdims=True) + 1e-10
    dir_projections_norm = dir_projections / norms
    
    # K=5聚类
    n_clusters = min(5, 50)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dir_projections_norm)
    
    for c in range(n_clusters):
        cluster_dirs = [d for d in range(50) if labels[d] == c]
        cluster_classes = [direction_class[d] for d in cluster_dirs]
        class_dist = defaultdict(int)
        for cls in cluster_classes:
            class_dist[cls] += 1
        print(f"  Cluster {c}: dirs={cluster_dirs}, classes={dict(class_dist)}")
    
    result = {
        "model": model_name, "n_texts": len(TEST_TEXTS),
        "direction_class": direction_class,
        "direction_class_scores": {str(d): direction_class_scores[d] for d in range(50)},
        "class_counts": dict(class_counts),
        "energy_stats": {
            "format_mean": float(np.mean(format_energies)),
            "function_mean": float(np.mean(function_energies)),
            "content_mean": float(np.mean(content_energies)),
        },
        "causal_chain": {},
        "cluster_labels": labels.tolist(),
    }
    
    for name, energies in [("format", format_energies), ("function", function_energies), ("content", content_energies)]:
        r_prob, p_prob = spearmanr(energies, top1_probs)
        r_gap, p_gap = spearmanr(energies, logit_gaps)
        result["causal_chain"][name] = {
            "r_prob": float(r_prob), "p_prob": float(p_prob),
            "r_gap": float(r_gap), "p_gap": float(p_gap),
        }
    
    return result


# ===================== P578: 频谱→logit_gap统一方程 =====================
def experiment_p578(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P578: 频谱→logit_gap统一方程
    核心问题: 能否找到跨模型一致的"频谱参数→logit_gap"数学方程?
    
    创新点:
    1. 用频谱系数的分布特征(偏度/峰度/分位数)预测logit_gap
    2. 逐层频谱→逐层logit_gap的因果链
    3. 多元回归: 频谱参数组合→logit_gap
    4. 频谱传播方程的修正: 加入非线性项
    """
    print(f"\n{'='*60}")
    print(f"P578: 频谱→logit_gap统一方程 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(300, U_wu.shape[1])
    
    # ===== 1. 大规模数据收集 =====
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # 频谱系数
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # 基本频谱参数
        ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        
        # 频谱分布的高阶统计量
        coeffs_sq = abs_coeffs[:50]**2 / total_energy  # 归一化能量分布
        spectral_skew = float(np.mean((coeffs_sq - np.mean(coeffs_sq))**3) / (np.std(coeffs_sq)**3 + 1e-10))
        spectral_kurtosis = float(np.mean((coeffs_sq - np.mean(coeffs_sq))**4) / (np.std(coeffs_sq)**4 + 1e-10))
        
        # 分位数比
        sorted_energy = np.sort(coeffs_sq)[::-1]
        q25_q75_ratio = sorted_energy[int(0.25*50)] / (sorted_energy[int(0.75*50)] + 1e-10)
        q10_q50_ratio = sorted_energy[4] / (sorted_energy[24] + 1e-10)
        
        # Alpha/Beta
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        delta = h_last - h_prev
        beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        
        # Delta频谱: MLP贡献的频谱特征
        delta_coeffs = U_wu[:, :50].T @ delta
        abs_delta = np.abs(delta_coeffs)
        delta_total = np.sum(abs_delta**2) + 1e-10
        delta_ratio_10 = np.sum(abs_delta[:10]**2) / delta_total
        delta_top1 = abs_delta[0]**2 / delta_total
        
        # Logit分布
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        logit_gap = full_logits[sorted_ids[0]] - full_logits[sorted_ids[1]]
        logit_var = float(np.var(full_logits))
        logit_kurtosis = float(np.mean((full_logits - np.mean(full_logits))**4) / (np.var(full_logits)**2 + 1e-10))
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        # 频谱熵
        spectral_entropy = -np.sum(coeffs_sq * np.log(coeffs_sq + 1e-10))
        
        all_data.append({
            "alpha": alpha, "beta": beta,
            "ratio_10": ratio_10, "ratio_50": ratio_50,
            "spectral_skew": spectral_skew, "spectral_kurtosis": spectral_kurtosis,
            "q25_q75_ratio": q25_q75_ratio, "q10_q50_ratio": q10_q50_ratio,
            "delta_ratio_10": delta_ratio_10, "delta_top1": delta_top1,
            "spectral_entropy": spectral_entropy,
            "logit_gap": logit_gap, "logit_var": logit_var, "logit_kurtosis": logit_kurtosis,
            "top1_prob": top1_prob,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    n = len(all_data)
    params = {key: np.array([d[key] for d in all_data]) for key in all_data[0]}
    
    # ===== 2. 单变量相关 =====
    print(f"\n--- 单变量与logit_gap的相关(n={n}) ---")
    spectral_features = ["ratio_10", "ratio_50", "spectral_skew", "spectral_kurtosis",
                        "q25_q75_ratio", "q10_q50_ratio", "delta_ratio_10", "delta_top1",
                        "spectral_entropy", "alpha", "beta"]
    
    gap_corrs = {}
    for feat in spectral_features:
        r, p = spearmanr(params[feat], params["logit_gap"])
        gap_corrs[feat] = {"r": r, "p": p}
        if abs(r) > 0.1 or p < 0.05:
            print(f"  {feat} → logit_gap: r={r:.3f} (p={p:.4f})")
    
    # ===== 3. 单变量与logit_var的相关 =====
    print(f"\n--- 单变量与logit_var的相关 ---")
    var_corrs = {}
    for feat in spectral_features:
        r, p = spearmanr(params[feat], params["logit_var"])
        var_corrs[feat] = {"r": r, "p": p}
        if abs(r) > 0.1 or p < 0.05:
            print(f"  {feat} → logit_var: r={r:.3f} (p={p:.4f})")
    
    # ===== 4. 多元回归: 频谱参数→logit_gap =====
    print(f"\n--- 多元回归: 频谱参数→logit_gap ---")
    from scipy.stats import rankdata
    
    y = rankdata(params["logit_gap"])
    
    # 组合1: ratio_50 + delta_ratio_10
    x1 = rankdata(params["ratio_50"]) + rankdata(params["delta_ratio_10"])
    r1, _ = spearmanr(x1, y)
    print(f"  ratio_50 + delta_ratio_10 → logit_gap: r={r1:.3f}")
    
    # 组合2: spectral_kurtosis + delta_top1
    x2 = rankdata(params["spectral_kurtosis"]) + rankdata(params["delta_top1"])
    r2, _ = spearmanr(x2, y)
    print(f"  spectral_kurtosis + delta_top1 → logit_gap: r={r2:.3f}")
    
    # 组合3: q10_q50_ratio + beta
    x3 = rankdata(params["q10_q50_ratio"]) + rankdata(params["beta"])
    r3, _ = spearmanr(x3, y)
    print(f"  q10_q50_ratio + beta → logit_gap: r={r3:.3f}")
    
    # 组合4: 全部
    x4 = (rankdata(params["ratio_50"]) + rankdata(params["delta_ratio_10"]) + 
           rankdata(params["spectral_kurtosis"]) + rankdata(params["beta"]) + 
           rankdata(params["delta_top1"]))
    r4, _ = spearmanr(x4, y)
    print(f"  全部组合 → logit_gap: r={r4:.3f}")
    
    # 组合5: logit_var作为中间变量
    r_var_gap, _ = spearmanr(params["logit_var"], params["logit_gap"])
    x5 = rankdata(params["ratio_50"]) + rankdata(params["logit_var"])
    r5, _ = spearmanr(x5, y)
    print(f"  ratio_50 + logit_var → logit_gap: r={r5:.3f}")
    
    # ===== 5. 完整路径: 频谱→logit_var→logit_gap→prob =====
    print(f"\n--- 完整路径系数 ---")
    
    # 最佳频谱→logit_var
    best_spectral_var = max(gap_corrs.keys(), key=lambda k: abs(var_corrs[k]["r"]))
    print(f"  最佳频谱→logit_var: {best_spectral_var} (r={var_corrs[best_spectral_var]['r']:.3f})")
    
    # logit_var→logit_gap
    r_var_gap, p_var_gap = spearmanr(params["logit_var"], params["logit_gap"])
    print(f"  logit_var → logit_gap: r={r_var_gap:.3f} (p={p_var_gap:.4f})")
    
    # logit_gap→prob
    r_gap_prob, p_gap_prob = spearmanr(params["logit_gap"], params["top1_prob"])
    print(f"  logit_gap → top1_prob: r={r_gap_prob:.3f} (p={p_gap_prob:.4f})")
    
    # 直接路径: 最佳频谱→prob
    best_spectral_prob = max(spectral_features, key=lambda k: abs(spearmanr(params[k], params["top1_prob"])[0]))
    r_sp_prob, _ = spearmanr(params[best_spectral_prob], params["top1_prob"])
    print(f"  直接: {best_spectral_prob} → top1_prob: r={r_sp_prob:.3f}")
    
    # 中介路径: 最佳频谱→logit_var→logit_gap→prob
    r_sp_var = var_corrs[best_spectral_prob]["r"]
    mediation_path = r_sp_var * r_var_gap * r_gap_prob
    print(f"  中介路径: {best_spectral_prob}→logit_var→logit_gap→prob: 乘积={mediation_path:.3f}")
    print(f"  中介vs直接: {'中介更强' if abs(mediation_path) > abs(r_sp_prob) else '直接更强'}")
    
    # ===== 6. 逐层频谱→logit_gap因果链 =====
    print(f"\n--- 逐层频谱→logit_gap因果链(30文本) ---")
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    
    layer_data = {li: {"ratio_50": [], "logit_gap": [], "logit_var": []} for li in sample_layers}
    
    for text in TEST_TEXTS[:30]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for li in sample_layers:
            h_l = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
            h_coeffs_l = U_wu[:, :50].T @ h_l
            abs_coeffs_l = np.abs(h_coeffs_l)
            total_l = np.sum(abs_coeffs_l**2) + 1e-10
            ratio_50_l = np.sum(abs_coeffs_l[:50]**2) / total_l
            
            logits_l = W_U @ h_l
            sorted_l = np.argsort(logits_l)[::-1]
            gap_l = logits_l[sorted_l[0]] - logits_l[sorted_l[1]]
            var_l = float(np.var(logits_l))
            
            layer_data[li]["ratio_50"].append(ratio_50_l)
            layer_data[li]["logit_gap"].append(gap_l)
            layer_data[li]["logit_var"].append(var_l)
        
        del outputs
        torch.cuda.empty_cache()
    
    for li in sorted(sample_layers):
        r, p = spearmanr(layer_data[li]["ratio_50"], layer_data[li]["logit_gap"])
        r_var, _ = spearmanr(layer_data[li]["logit_var"], layer_data[li]["logit_gap"])
        print(f"  L{li}: ratio_50→logit_gap r={r:.3f}, logit_var→logit_gap r={r_var:.3f}")
    
    # ===== 7. 频谱传播方程修正 =====
    print(f"\n--- 频谱传播方程修正(5文本) ---")
    # S(l+1) = alpha*S(l) + epsilon 可能过于简化
    # 考虑: S(l+1) = alpha*S(l) + gamma*f(S(l)) + epsilon
    # 其中 f(S(l)) 是频谱的非线性变换
    
    verification_errors_linear = []
    verification_errors_nonlinear = []
    
    for text in TEST_TEXTS[:5]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        for l in range(1, n_layers + 1):
            h_l = outputs.hidden_states[l][0, -1].detach().cpu().float().numpy()
            h_l1 = outputs.hidden_states[l-1][0, -1].detach().cpu().float().numpy()
            
            s_l = np.abs(U_wu[:, :50].T @ h_l)**2
            s_l1 = np.abs(U_wu[:, :50].T @ h_l1)**2
            s_l = s_l / (np.sum(s_l) + 1e-10)
            s_l1 = s_l1 / (np.sum(s_l1) + 1e-10)
            
            # 线性: S(l) = alpha * S(l-1) + epsilon
            alpha_l = np.dot(s_l, s_l1) / (np.dot(s_l1, s_l1) + 1e-10)
            pred_linear = alpha_l * s_l1
            err_linear = np.linalg.norm(s_l - pred_linear) / (np.linalg.norm(s_l) + 1e-10)
            verification_errors_linear.append(err_linear)
            
            # 非线性: S(l) = alpha * S(l-1) + gamma * S(l-1)^2 + epsilon
            # 用最小二乘拟合
            A = np.column_stack([s_l1, s_l1**2])
            coeffs, _, _, _ = np.linalg.lstsq(A, s_l, rcond=None)
            pred_nonlinear = A @ coeffs
            err_nonlinear = np.linalg.norm(s_l - pred_nonlinear) / (np.linalg.norm(s_l) + 1e-10)
            verification_errors_nonlinear.append(err_nonlinear)
        
        del outputs
        torch.cuda.empty_cache()
    
    print(f"  线性方程误差: {np.mean(verification_errors_linear):.4f}")
    print(f"  非线性方程误差: {np.mean(verification_errors_nonlinear):.4f}")
    print(f"  改善率: {(1-np.mean(verification_errors_nonlinear)/np.mean(verification_errors_linear))*100:.1f}%")
    
    result = {
        "model": model_name, "n_texts": n, "n_layers": n_layers,
        "gap_correlations": {k: {"r": float(v["r"]), "p": float(v["p"])} for k, v in gap_corrs.items()},
        "var_correlations": {k: {"r": float(v["r"]), "p": float(v["p"])} for k, v in var_corrs.items()},
        "multi_regression": {
            "ratio50_delta_ratio10": float(r1),
            "kurtosis_delta_top1": float(r2),
            "q10q50_beta": float(r3),
            "all": float(r4),
            "ratio50_logit_var": float(r5),
        },
        "path_analysis": {
            "best_spectral_var": best_spectral_var,
            "r_spectral_var": float(var_corrs[best_spectral_var]["r"]),
            "r_var_gap": float(r_var_gap),
            "r_gap_prob": float(r_gap_prob),
            "r_spectral_prob_direct": float(r_sp_prob),
            "mediation_product": float(mediation_path),
        },
        "layer_causal": {str(li): {"ratio50_to_gap": float(spearmanr(layer_data[li]["ratio_50"], layer_data[li]["logit_gap"])[0]),
                                    "var_to_gap": float(spearmanr(layer_data[li]["logit_var"], layer_data[li]["logit_gap"])[0])}
                         for li in sample_layers},
        "equation_verification": {
            "linear_error": float(np.mean(verification_errors_linear)),
            "nonlinear_error": float(np.mean(verification_errors_nonlinear)),
            "improvement_pct": float((1-np.mean(verification_errors_nonlinear)/np.mean(verification_errors_linear))*100),
        },
    }
    return result


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p575", "p576", "p577", "p578"])
    args = parser.parse_args()
    
    start_time = time.time()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=300)
    
    if args.experiment == "p575":
        result = experiment_p575(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p576":
        result = experiment_p576(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p577":
        result = experiment_p577(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p578":
        result = experiment_p578(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = "results/phase_cxxxiii"
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
