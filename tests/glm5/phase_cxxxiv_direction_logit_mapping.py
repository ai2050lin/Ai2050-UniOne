"""
Phase CXXXIV: 方向特异性与logit_gap的直接映射 (P579-P582)
========================================================
核心洞察: logit_gap = Sum_k c_k * Delta_k 是精确数学等式
  其中 c_k = h·U_k (频谱系数, 由频谱力学决定)
       Delta_k = W_U[top1,:]·U_k - W_U[top2,:]·U_k (方向k在top1/top2上的W_U投影差)

P579: 方向级logit_gap贡献的预测模型 — 用频谱系数和Delta_k直接预测方向对logit_gap的贡献
P580: logit_gap的方向分解方程 — 精确验证 Sum c_k * Delta_k = logit_gap, 分析每项的统计特性
P581: 层间logit_gap传播方程 — 逐层logit_gap的传播方程
P582: 频谱->logit_gap的直接路径修正 — 绕过logit_var, 直接从方向级频谱->logit_gap

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


# ===================== P579: 方向级logit_gap贡献的预测模型 =====================
def experiment_p579(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P579: 方向级logit_gap贡献的预测模型
    核心洞察: logit_gap = Sum_k c_k * Delta_k 是精确数学等式
    问题: 能否用频谱力学预测c_k, 然后与Delta_k组合预测logit_gap?
    
    创新点:
    1. 精确验证: Sum c_k * Delta_k == logit_gap (数值精度)
    2. c_k的可预测性: c_k由频谱力学决定, |c_k|^2~S_k(频谱能量)
    3. Delta_k的稳定性: Delta_k是否跨文本稳定?(如果是, 则只需要预测c_k)
    4. 频谱力学->|c_k|->logit_gap的直接路径
    """
    print(f"\n{'='*60}")
    print(f"P579: 方向级logit_gap贡献的预测模型 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    K = 50  # 分析前50个方向
    
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # 频谱系数 c_k = U_k^T @ h_last
        c_k = U_wu[:, :K].T @ h_last  # [K]
        
        # 完整logits
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        top1_id, top2_id = sorted_ids[0], sorted_ids[1]
        logit_gap = full_logits[top1_id] - full_logits[top2_id]
        
        # Delta_k = W_U[top1,:] · U_k - W_U[top2,:] · U_k
        # 预计算W_U投影: W_U @ U_k 对所有k
        Delta_k = np.zeros(K)
        for k in range(K):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            Delta_k[k] = wu_top1_proj - wu_top2_proj
        
        # 精确验证: Sum c_k * Delta_k == logit_gap
        reconstructed_gap = np.sum(c_k * Delta_k)
        # 注意: 只用前K个方向, 可能不完全等于logit_gap
        # 完整重建需要所有d_model个方向
        full_K = min(U_wu.shape[1], W_U.shape[1])
        c_k_full = U_wu[:, :full_K].T @ h_last
        Delta_k_full = np.zeros(full_K)
        for k in range(full_K):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            Delta_k_full[k] = wu_top1_proj - wu_top2_proj
        full_reconstructed = np.sum(c_k_full * Delta_k_full)
        
        # Alpha/Beta
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        delta = h_last - h_prev
        beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        # 频谱参数
        abs_coeffs = np.abs(c_k)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        
        # 方向级贡献
        dir_contributions = c_k * Delta_k  # [K] — 每个方向对logit_gap的精确贡献
        pos_contrib = np.sum(dir_contributions[dir_contributions > 0])
        neg_contrib = np.sum(dir_contributions[dir_contributions < 0])
        
        # c_k的统计特征
        c_k_abs = np.abs(c_k)
        c_k_rms = np.sqrt(np.mean(c_k**2))
        
        # Delta_k的统计特征
        Delta_k_abs = np.abs(Delta_k)
        Delta_k_rms = np.sqrt(np.mean(Delta_k**2))
        Delta_k_sign = np.sign(Delta_k)
        Delta_k_positive_ratio = np.mean(Delta_k > 0)
        
        # 方向贡献的集中度
        abs_contrib = np.abs(dir_contributions)
        total_contrib = np.sum(abs_contrib) + 1e-10
        top1_contrib_ratio = np.max(abs_contrib) / total_contrib
        top5_contrib_ratio = np.sum(np.sort(abs_contrib)[::-1][:5]) / total_contrib
        top10_contrib_ratio = np.sum(np.sort(abs_contrib)[::-1][:10]) / total_contrib
        
        all_data.append({
            "logit_gap": logit_gap,
            "reconstructed_gap_K50": reconstructed_gap,
            "full_reconstructed": full_reconstructed,
            "reconstruction_error_pct": abs(reconstructed_gap - logit_gap) / (abs(logit_gap) + 1e-10) * 100,
            "full_reconstruction_error_pct": abs(full_reconstructed - logit_gap) / (abs(logit_gap) + 1e-10) * 100,
            "alpha": alpha, "beta": beta,
            "ratio_10": ratio_10, "ratio_50": ratio_50,
            "c_k": c_k.tolist(),
            "Delta_k": Delta_k.tolist(),
            "dir_contributions": dir_contributions.tolist(),
            "pos_contrib": pos_contrib,
            "neg_contrib": neg_contrib,
            "c_k_rms": c_k_rms,
            "Delta_k_rms": Delta_k_rms,
            "Delta_k_positive_ratio": Delta_k_positive_ratio,
            "top1_contrib_ratio": top1_contrib_ratio,
            "top5_contrib_ratio": top5_contrib_ratio,
            "top10_contrib_ratio": top10_contrib_ratio,
            "top1_prob": top1_prob,
            "top1_id": int(top1_id),
            "top2_id": int(top2_id),
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    n = len(all_data)
    
    # ===== 1. 精确验证 =====
    print(f"\n=== 1. 精确验证: Sum c_k * Delta_k == logit_gap ===")
    recon_errors = [d["reconstruction_error_pct"] for d in all_data]
    full_recon_errors = [d["full_reconstruction_error_pct"] for d in all_data]
    print(f"  K=50重建误差: {np.mean(recon_errors):.4f}% (中位={np.median(recon_errors):.4f}%)")
    print(f"  全方向重建误差: {np.mean(full_recon_errors):.4f}% (中位={np.median(full_recon_errors):.4f}%)")
    print(f"  -> 前50方向解释logit_gap的: {100-np.mean(recon_errors):.1f}%")
    
    # ===== 2. c_k的可预测性 =====
    print(f"\n=== 2. c_k的统计特性 ===")
    c_k_all = np.array([d["c_k"] for d in all_data])  # [n, K]
    c_k_rms = np.array([d["c_k_rms"] for d in all_data])
    
    # c_k的跨文本变异性
    c_k_std = np.std(c_k_all, axis=0)  # [K] 每个方向的c_k跨文本标准差
    c_k_mean_abs = np.mean(np.abs(c_k_all), axis=0)  # [K]
    c_k_cv = c_k_std / (c_k_mean_abs + 1e-10)  # [K] 变异系数
    
    print(f"  c_k跨文本CV(变异系数)均值: {np.mean(c_k_cv):.3f}")
    print(f"  c_k跨文本CV(Dir0): {c_k_cv[0]:.3f}, Dir5: {c_k_cv[5]:.3f}, Dir10: {c_k_cv[10]:.3f}")
    
    # Alpha与c_k的关系
    alphas = np.array([d["alpha"] for d in all_data])
    print(f"\n  alpha与|c_k|的相关:")
    for k in [0, 1, 2, 5, 10, 20]:
        r, p = spearmanr(alphas, np.abs(c_k_all[:, k]))
        print(f"    alpha vs |c_{k}|: r={r:.3f} (p={p:.4f})")
    
    # ===== 3. Delta_k的跨文本稳定性 =====
    print(f"\n=== 3. Delta_k的跨文本稳定性 ===")
    Delta_k_all = np.array([d["Delta_k"] for d in all_data])  # [n, K]
    
    # Delta_k取决于top1和top2 token, 不同文本可能选择不同token
    # 所以Delta_k可能跨文本变化很大
    Delta_k_std = np.std(Delta_k_all, axis=0)
    Delta_k_mean_abs = np.mean(np.abs(Delta_k_all), axis=0)
    Delta_k_cv = Delta_k_std / (Delta_k_mean_abs + 1e-10)
    
    print(f"  Delta_k跨文本CV均值: {np.mean(Delta_k_cv):.3f}")
    print(f"  Delta_k跨文本CV(Dir0): {Delta_k_cv[0]:.3f}, Dir5: {Delta_k_cv[5]:.3f}, Dir10: {Delta_k_cv[10]:.3f}")
    
    # Delta_k的符号稳定性
    Delta_k_sign_consistency = np.zeros(K)
    for k in range(K):
        signs = np.sign(Delta_k_all[:, k])
        # 符号一致性 = max(正占比, 负占比)
        pos_ratio = np.mean(signs > 0)
        Delta_k_sign_consistency[k] = max(pos_ratio, 1 - pos_ratio)
    
    print(f"\n  Delta_k符号一致性(>0.5表示有偏):")
    for k in [0, 1, 2, 3, 5, 10, 20, 30, 49]:
        print(f"    Dir{k}: {Delta_k_sign_consistency[k]:.3f}")
    
    # ===== 4. c_k和Delta_k对logit_gap的独立预测力 =====
    print(f"\n=== 4. c_k和Delta_k对logit_gap的预测 ===")
    logit_gaps = np.array([d["logit_gap"] for d in all_data])
    top1_probs = np.array([d["top1_prob"] for d in all_data])
    
    # |c_k|^2的加权和 vs logit_gap
    # 理论: logit_gap ≈ Sum_k c_k * Delta_k
    # 如果Delta_k近似常数: logit_gap ≈ Sum_k c_k * const_k
    # 如果Delta_k跨文本变化: 需要同时知道c_k和Delta_k
    
    # 测试: Sum |c_k| * |Delta_k|_mean vs logit_gap
    Delta_k_mean = np.mean(np.abs(Delta_k_all), axis=0)  # [K] |Delta_k|的均值
    c_k_energy = np.sum(np.abs(c_k_all) * Delta_k_mean[np.newaxis, :], axis=1)  # [n]
    r_energy, p_energy = spearmanr(c_k_energy, logit_gaps)
    print(f"  Sum |c_k| * E[|Delta_k|] -> logit_gap: r={r_energy:.3f} (p={p_energy:.4f})")
    
    # 测试: Sum c_k^2 * Delta_k^2 vs logit_gap
    gap_energy = np.sum(c_k_all**2 * np.mean(Delta_k_all**2, axis=0)[np.newaxis, :], axis=1)
    r_gap_e, p_gap_e = spearmanr(gap_energy, logit_gaps)
    print(f"  Sum c_k^2 * E[Delta_k^2] -> logit_gap: r={r_gap_e:.3f} (p={p_gap_e:.4f})")
    
    # 对比: 直接用精确Sum c_k * Delta_k
    exact_gaps = np.array([np.sum(np.array(d["c_k"]) * np.array(d["Delta_k"])) for d in all_data])
    r_exact, p_exact = spearmanr(exact_gaps, logit_gaps)
    print(f"  精确 Sum c_k * Delta_k -> logit_gap: r={r_exact:.3f} (p={p_exact:.4f})")
    
    # ===== 5. 方向贡献集中度 =====
    print(f"\n=== 5. 方向贡献集中度 ===")
    top1_ratios = np.array([d["top1_contrib_ratio"] for d in all_data])
    top5_ratios = np.array([d["top5_contrib_ratio"] for d in all_data])
    top10_ratios = np.array([d["top10_contrib_ratio"] for d in all_data])
    print(f"  Top-1方向贡献比: {np.mean(top1_ratios):.4f}")
    print(f"  Top-5方向贡献比: {np.mean(top5_ratios):.4f}")
    print(f"  Top-10方向贡献比: {np.mean(top10_ratios):.4f}")
    
    # 集中度与预测质量的关系
    r_t1_prob, _ = spearmanr(top1_ratios, top1_probs)
    r_t5_prob, _ = spearmanr(top5_ratios, top1_probs)
    print(f"  Top-1贡献比 -> prob: r={r_t1_prob:.3f}")
    print(f"  Top-5贡献比 -> prob: r={r_t5_prob:.3f}")
    
    # ===== 6. 频谱参数->方向级贡献的直接路径 =====
    print(f"\n=== 6. 频谱参数->方向级贡献的直接路径 ===")
    # 路径1: alpha -> |c_k| -> dir_contribution -> logit_gap
    # 路径2: ratio_k -> c_k_rms -> logit_gap
    
    c_k_rms_arr = np.array([d["c_k_rms"] for d in all_data])
    r_rms_gap, _ = spearmanr(c_k_rms_arr, logit_gaps)
    print(f"  c_k_rms -> logit_gap: r={r_rms_gap:.3f}")
    
    # 方向贡献RMS
    contrib_rms = np.array([np.sqrt(np.mean(np.array(d["dir_contributions"])**2)) for d in all_data])
    r_contrib_rms_gap, _ = spearmanr(contrib_rms, logit_gaps)
    print(f"  dir_contribution_rms -> logit_gap: r={r_contrib_rms_gap:.3f}")
    
    # 正贡献 vs 负贡献
    pos_contribs = np.array([d["pos_contrib"] for d in all_data])
    neg_contribs = np.array([d["neg_contrib"] for d in all_data])
    net_contribs = pos_contribs + neg_contribs
    r_pos, _ = spearmanr(pos_contribs, logit_gaps)
    r_neg, _ = spearmanr(neg_contribs, logit_gaps)
    r_net, _ = spearmanr(net_contribs, logit_gaps)
    print(f"  正贡献 -> logit_gap: r={r_pos:.3f}")
    print(f"  负贡献 -> logit_gap: r={r_neg:.3f}")
    print(f"  净贡献 -> logit_gap: r={r_net:.3f}")
    
    # ===== 7. 逐方向分析: 哪些方向的c_k*Delta_k最预测logit_gap =====
    print(f"\n=== 7. 逐方向预测力 ===")
    dir_gap_corrs = np.zeros(K)
    for k in range(K):
        contribs_k = np.array([d["dir_contributions"][k] for d in all_data])
        r, _ = spearmanr(contribs_k, logit_gaps)
        dir_gap_corrs[k] = r
    
    top10_pred_dirs = np.argsort(np.abs(dir_gap_corrs))[::-1][:10]
    print(f"  Top-10预测方向(对logit_gap): {top10_pred_dirs.tolist()}")
    for d in top10_pred_dirs:
        print(f"    Dir{d}: c_k*Delta_k -> logit_gap r={dir_gap_corrs[d]:.3f}")
    
    result = {
        "model": model_name, "n_texts": n,
        "reconstruction": {
            "K50_error_pct": float(np.mean(recon_errors)),
            "K50_error_median": float(np.median(recon_errors)),
            "full_error_pct": float(np.mean(full_recon_errors)),
            "full_error_median": float(np.median(full_recon_errors)),
            "K50_explained_pct": float(100 - np.mean(recon_errors)),
        },
        "c_k_stats": {
            "cv_mean": float(np.mean(c_k_cv)),
            "cv_per_dir": c_k_cv.tolist(),
            "alpha_corr_per_dir": [float(spearmanr(alphas, np.abs(c_k_all[:, k]))[0]) for k in range(K)],
        },
        "Delta_k_stats": {
            "cv_mean": float(np.mean(Delta_k_cv)),
            "cv_per_dir": Delta_k_cv.tolist(),
            "sign_consistency": Delta_k_sign_consistency.tolist(),
        },
        "prediction": {
            "energy_method_r": float(r_energy),
            "gap_energy_r": float(r_gap_e),
            "exact_r": float(r_exact),
            "c_k_rms_r": float(r_rms_gap),
            "contrib_rms_r": float(r_contrib_rms_gap),
            "pos_contrib_r": float(r_pos),
            "neg_contrib_r": float(r_neg),
            "net_contrib_r": float(r_net),
        },
        "concentration": {
            "top1_mean": float(np.mean(top1_ratios)),
            "top5_mean": float(np.mean(top5_ratios)),
            "top10_mean": float(np.mean(top10_ratios)),
            "top1_to_prob_r": float(r_t1_prob),
            "top5_to_prob_r": float(r_t5_prob),
        },
        "dir_gap_corrs": dir_gap_corrs.tolist(),
        "top10_pred_dirs": top10_pred_dirs.tolist(),
    }
    return result


# ===================== P580: logit_gap的方向分解方程 =====================
def experiment_p580(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P580: logit_gap的方向分解方程
    核心等式: logit_gap = Sum_k c_k * Delta_k
    分析每项的统计特性和数学结构
    
    创新点:
    1. Delta_k的结构分析: Delta_k = W_U[top1,:]·U_k - W_U[top2,:]·U_k
       = (e_top1 - e_top2)^T W_U U_k 其中e_i是单位向量
    2. c_k*Delta_k的对称性: 正负贡献的统计分布
    3. 不同文本类型下分解的稳定性
    4. Top-1方向的Delta_k特征: 它的Delta_k是否系统性地大?
    """
    print(f"\n{'='*60}")
    print(f"P580: logit_gap的方向分解方程 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    K = 50
    
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        
        # 频谱系数
        c_k = U_wu[:, :K].T @ h_last
        
        # Logits和top token
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        top1_id, top2_id = sorted_ids[0], sorted_ids[1]
        logit_gap = full_logits[top1_id] - full_logits[top2_id]
        
        # Delta_k和贡献
        Delta_k = np.zeros(K)
        dir_contributions = np.zeros(K)
        for k in range(K):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            Delta_k[k] = wu_top1_proj - wu_top2_proj
            dir_contributions[k] = c_k[k] * Delta_k[k]
        
        # Delta_k的分解: = (W_U[top1,:] - W_U[top2,:]) @ U_k
        # 这个向量差在W_U行空间中的投影
        wu_diff = W_U[top1_id] - W_U[top2_id]  # [d_model]
        wu_diff_norm = np.linalg.norm(wu_diff)
        
        # wu_diff在U_k方向上的投影
        wu_diff_projs = U_wu[:, :K].T @ wu_diff  # [K]
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        all_data.append({
            "logit_gap": logit_gap,
            "top1_prob": top1_prob,
            "c_k": c_k.tolist(),
            "Delta_k": Delta_k.tolist(),
            "dir_contributions": dir_contributions.tolist(),
            "wu_diff_norm": wu_diff_norm,
            "wu_diff_projs": wu_diff_projs.tolist(),
            "top1_id": int(top1_id),
            "top2_id": int(top2_id),
            "top1_token": safe_decode(tokenizer, top1_id),
            "top2_token": safe_decode(tokenizer, top2_id),
            "text_type": "short" if len(text.split()) < 8 else ("medium" if len(text.split()) < 15 else "long"),
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    n = len(all_data)
    
    # ===== 1. 方程精确性验证 =====
    print(f"\n=== 1. 方程精确性验证 ===")
    exact_gaps = [np.sum(np.array(d["c_k"]) * np.array(d["Delta_k"])) for d in all_data]
    actual_gaps = [d["logit_gap"] for d in all_data]
    errors = [abs(e - a) / (abs(a) + 1e-10) * 100 for e, a in zip(exact_gaps, actual_gaps)]
    print(f"  K=50方向分解误差: {np.mean(errors):.4f}% (中位={np.median(errors):.4f}%)")
    
    # ===== 2. Delta_k = W_U[top1-top2] @ U_k 的结构分析 =====
    print(f"\n=== 2. Delta_k的结构: (W_U[top1]-W_U[top2])在U_k上的投影 ===")
    wu_diff_norms = np.array([d["wu_diff_norm"] for d in all_data])
    print(f"  W_U[top1]-W_U[top2]范数: {np.mean(wu_diff_norms):.4f} ± {np.std(wu_diff_norms):.4f}")
    
    # Delta_k与c_k的相关: 如果Delta_k和c_k独立, 则E[c_k*Delta_k] = E[c_k]*E[Delta_k]
    # 如果相关, 则可能有协同效应
    c_k_all = np.array([d["c_k"] for d in all_data])
    Delta_k_all = np.array([d["Delta_k"] for d in all_data])
    contrib_all = np.array([d["dir_contributions"] for d in all_data])
    
    print(f"\n  c_k与Delta_k的相关(逐方向):")
    for k in [0, 1, 2, 3, 5, 10, 20, 49]:
        r, p = spearmanr(c_k_all[:, k], Delta_k_all[:, k])
        print(f"    Dir{k}: c_k vs Delta_k r={r:.3f} (p={p:.4f})")
    
    # ===== 3. c_k*Delta_k的统计分布 =====
    print(f"\n=== 3. c_k*Delta_k的统计分布 ===")
    pos_contribs = np.sum(contrib_all * (contrib_all > 0), axis=1)
    neg_contribs = np.sum(contrib_all * (contrib_all < 0), axis=1)
    net_contribs = pos_contribs + neg_contribs
    
    print(f"  正贡献均值: {np.mean(pos_contribs):.4f}")
    print(f"  负贡献均值: {np.mean(neg_contribs):.4f}")
    print(f"  净贡献均值: {np.mean(net_contribs):.4f}")
    print(f"  正/负贡献比: {np.mean(np.abs(pos_contribs)/np.abs(neg_contribs+1e-10)):.2f}")
    
    # c_k*Delta_k的分布形状
    all_contribs_flat = contrib_all.flatten()
    contrib_skew = float(np.mean((all_contribs_flat - np.mean(all_contribs_flat))**3) / (np.std(all_contribs_flat)**3 + 1e-10))
    contrib_kurt = float(np.mean((all_contribs_flat - np.mean(all_contribs_flat))**4) / (np.std(all_contribs_flat)**4 + 1e-10))
    print(f"  c_k*Delta_k偏度: {contrib_skew:.3f}")
    print(f"  c_k*Delta_k峰度: {contrib_kurt:.3f}")
    
    # ===== 4. 逐方向贡献的统计特性 =====
    print(f"\n=== 4. 逐方向贡献的统计特性 ===")
    mean_contrib = np.mean(contrib_all, axis=0)  # [K]
    std_contrib = np.std(contrib_all, axis=0)
    mean_abs_contrib = np.mean(np.abs(contrib_all), axis=0)
    
    # 方向贡献的"可靠性": |mean|/std
    reliability = np.abs(mean_contrib) / (std_contrib + 1e-10)
    
    top10_reliable = np.argsort(reliability)[::-1][:10]
    print(f"  Top-10可靠方向(贡献最稳定): {top10_reliable.tolist()}")
    for d in top10_reliable:
        print(f"    Dir{d}: mean_contrib={mean_contrib[d]:.4f}, reliability={reliability[d]:.3f}")
    
    # ===== 5. Top-1贡献方向的Delta_k特征 =====
    print(f"\n=== 5. 每个文本的Top-1贡献方向的Delta_k特征 ===")
    # 对每个文本, 找到贡献最大的方向, 分析其Delta_k
    top_contrib_dirs = np.argmax(np.abs(contrib_all), axis=1)  # [n]
    top_dir_counts = np.bincount(top_contrib_dirs, minlength=K)
    
    top10_dirs = np.argsort(top_dir_counts)[::-1][:10]
    print(f"  最常成为Top-1贡献的方向: {top10_dirs.tolist()}")
    for d in top10_dirs:
        print(f"    Dir{d}: 出现{top_dir_counts[d]}次/{n}文本 ({top_dir_counts[d]/n*100:.1f}%)")
    
    # ===== 6. 文本类型差异 =====
    print(f"\n=== 6. 文本类型差异 ===")
    for text_type in ["short", "medium", "long"]:
        subset = [d for d in all_data if d["text_type"] == text_type]
        if len(subset) > 0:
            mean_gap = np.mean([d["logit_gap"] for d in subset])
            mean_prob = np.mean([d["top1_prob"] for d in subset])
            sub_contribs = np.array([d["dir_contributions"] for d in subset])
            top5_dirs = np.argsort(np.mean(np.abs(sub_contribs), axis=0))[::-1][:5]
            print(f"  {text_type}(n={len(subset)}): gap={mean_gap:.3f}, prob={mean_prob:.3f}, top5_dirs={top5_dirs.tolist()}")
    
    # ===== 7. W_U差向量的频谱分析 =====
    print(f"\n=== 7. W_U[top1]-W_U[top2]的频谱分析 ===")
    wu_diff_projs_all = np.array([d["wu_diff_projs"] for d in all_data])
    wu_diff_energy_per_dir = np.mean(wu_diff_projs_all**2, axis=0)  # [K]
    wu_diff_total = np.sum(wu_diff_energy_per_dir) + 1e-10
    wu_diff_ratio = np.cumsum(wu_diff_energy_per_dir) / wu_diff_total
    
    print(f"  W_U差在前10方向能量比: {wu_diff_ratio[9]:.4f}")
    print(f"  W_U差在前20方向能量比: {wu_diff_ratio[19]:.4f}")
    print(f"  W_U差在前50方向能量比: {wu_diff_ratio[49]:.4f}")
    
    # W_U差向量频谱与h频谱的相关
    h_energy_per_dir = np.mean(c_k_all**2, axis=0)
    r_spectra, _ = spearmanr(wu_diff_energy_per_dir, h_energy_per_dir)
    print(f"  W_U差频谱与h频谱相关: r={r_spectra:.3f}")
    
    result = {
        "model": model_name, "n_texts": n,
        "equation_precision": {
            "K50_error_pct": float(np.mean(errors)),
            "K50_error_median": float(np.median(errors)),
        },
        "delta_k_structure": {
            "wu_diff_norm_mean": float(np.mean(wu_diff_norms)),
            "wu_diff_norm_std": float(np.std(wu_diff_norms)),
            "c_delta_corr": [float(spearmanr(c_k_all[:, k], Delta_k_all[:, k])[0]) for k in range(K)],
        },
        "contribution_stats": {
            "pos_mean": float(np.mean(pos_contribs)),
            "neg_mean": float(np.mean(neg_contribs)),
            "net_mean": float(np.mean(net_contribs)),
            "skew": contrib_skew,
            "kurtosis": contrib_kurt,
            "mean_per_dir": mean_contrib.tolist(),
            "reliability_per_dir": reliability.tolist(),
        },
        "top_contrib_dirs": top10_dirs.tolist(),
        "top_dir_counts": top_dir_counts[:10].tolist(),
        "wu_diff_spectrum": {
            "ratio_10": float(wu_diff_ratio[9]),
            "ratio_20": float(wu_diff_ratio[19]),
            "ratio_50": float(wu_diff_ratio[49]),
            "h_spectrum_corr": float(r_spectra),
        },
    }
    return result


# ===================== P581: 层间logit_gap传播方程 =====================
def experiment_p581(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P581: 层间logit_gap传播方程
    核心发现: DS7B L26->L27 logit_gap r=0.899, logit_gap是层间连续传播的
    
    创新点:
    1. 逐层logit_gap的计算和传播分析
    2. logit_gap层间传播方程: gap(l+1) = f(gap(l), Dgap_MLP(l))
    3. MLP贡献对logit_gap的影响
    4. 频谱->logit_gap在各层的因果链强度变化
    """
    print(f"\n{'='*60}")
    print(f"P581: 层间logit_gap传播方程 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    K = 50
    
    # 采样层(每4层一个 + 首尾)
    sample_layers = list(range(0, n_layers, max(1, n_layers//15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    if n_layers - 2 not in sample_layers:
        sample_layers.append(n_layers - 2)
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    # 大规模数据收集(全部210文本)
    all_layer_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        text_layer_data = {}
        
        for li in sample_layers:
            h_l = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
            
            # Logits
            logits_l = W_U @ h_l
            sorted_l = np.argsort(logits_l)[::-1]
            gap_l = logits_l[sorted_l[0]] - logits_l[sorted_l[1]]
            var_l = float(np.var(logits_l))
            
            # 频谱系数
            c_k_l = U_wu[:, :K].T @ h_l
            abs_coeffs = np.abs(c_k_l)
            total_energy = np.sum(abs_coeffs**2) + 1e-10
            ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
            ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
            
            text_layer_data[li] = {
                "logit_gap": gap_l,
                "logit_var": var_l,
                "ratio_10": ratio_10,
                "ratio_50": ratio_50,
            }
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        all_layer_data.append({
            "layers": text_layer_data,
            "top1_prob": top1_prob,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    n = len(all_layer_data)
    
    # ===== 1. 逐层logit_gap的演化 =====
    print(f"\n=== 1. 逐层logit_gap的演化 ===")
    layer_gap_means = {}
    layer_gap_stds = {}
    for li in sample_layers:
        gaps = [d["layers"][li]["logit_gap"] for d in all_layer_data]
        layer_gap_means[li] = np.mean(gaps)
        layer_gap_stds[li] = np.std(gaps)
    
    for li in sorted(sample_layers):
        print(f"  L{li}: gap={layer_gap_means[li]:.4f} ± {layer_gap_stds[li]:.4f}")
    
    # ===== 2. 层间logit_gap的传播相关 =====
    print(f"\n=== 2. 层间logit_gap传播相关 ===")
    final_gaps = [d["layers"][n_layers-1]["logit_gap"] for d in all_layer_data]
    prev_gaps = [d["layers"][n_layers-2]["logit_gap"] for d in all_layer_data]
    
    # 末层前一层->末层
    r_prev_final, _ = spearmanr(prev_gaps, final_gaps)
    print(f"  L{n_layers-2} -> L{n_layers-1} logit_gap: r={r_prev_final:.3f}")
    
    # 所有采样层->末层
    print(f"\n  各层logit_gap与末层logit_gap的相关:")
    layer_to_final_r = {}
    for li in sorted(sample_layers):
        if li == n_layers - 1:
            continue
        gaps = [d["layers"][li]["logit_gap"] for d in all_layer_data]
        r, p = spearmanr(gaps, final_gaps)
        layer_to_final_r[li] = r
        print(f"    L{li} -> L{n_layers-1}: r={r:.3f} (p={p:.4f})")
    
    # ===== 3. 层间logit_gap传播方程 =====
    print(f"\n=== 3. 层间logit_gap传播方程 ===")
    # 线性模型: gap(l+1) ≈ a * gap(l) + b
    
    adjacent_pairs = [(sample_layers[i], sample_layers[i+1]) 
                      for i in range(len(sample_layers)-1)]
    
    print(f"  相邻层logit_gap线性关系:")
    for l1, l2 in adjacent_pairs:
        gaps1 = [d["layers"][l1]["logit_gap"] for d in all_layer_data]
        gaps2 = [d["layers"][l2]["logit_gap"] for d in all_layer_data]
        r, p = spearmanr(gaps1, gaps2)
        print(f"    L{l1} -> L{l2}: r={r:.3f}")
    
    # 整体传播效率
    # 定义: 传播效率 = 相邻层logit_gap相关的平均值
    adjacent_rs = []
    for l1, l2 in adjacent_pairs:
        gaps1 = [d["layers"][l1]["logit_gap"] for d in all_layer_data]
        gaps2 = [d["layers"][l2]["logit_gap"] for d in all_layer_data]
        r, _ = spearmanr(gaps1, gaps2)
        adjacent_rs.append(r)
    
    print(f"\n  平均层间传播相关: {np.mean(adjacent_rs):.3f}")
    print(f"  早期层(前1/3): {np.mean([r for r, (l1,_) in zip(adjacent_rs, adjacent_pairs) if l1 < n_layers//3]):.3f}")
    print(f"  中期层(中1/3): {np.mean([r for r, (l1,_) in zip(adjacent_rs, adjacent_pairs) if n_layers//3 <= l1 < 2*n_layers//3]):.3f}")
    print(f"  晚期层(后1/3): {np.mean([r for r, (l1,_) in zip(adjacent_rs, adjacent_pairs) if l1 >= 2*n_layers//3]):.3f}")
    
    # ===== 4. 各层频谱->logit_gap因果链 =====
    print(f"\n=== 4. 各层频谱->logit_gap因果链 ===")
    layer_spectral_gap_r = {}
    for li in sorted(sample_layers):
        ratio50 = [d["layers"][li]["ratio_50"] for d in all_layer_data]
        gaps = [d["layers"][li]["logit_gap"] for d in all_layer_data]
        vars_l = [d["layers"][li]["logit_var"] for d in all_layer_data]
        
        r_spec_gap, _ = spearmanr(ratio50, gaps)
        r_var_gap, _ = spearmanr(vars_l, gaps)
        layer_spectral_gap_r[li] = {"ratio50_gap": r_spec_gap, "var_gap": r_var_gap}
        
    for li in sorted(sample_layers):
        print(f"  L{li}: ratio50->gap r={layer_spectral_gap_r[li]['ratio50_gap']:.3f}, var->gap r={layer_spectral_gap_r[li]['var_gap']:.3f}")
    
    # ===== 5. logit_gap累积量分析 =====
    print(f"\n=== 5. logit_gap的层级累积 ===")
    # logit_gap从首层到末层是如何增长的?
    gap_growth = []
    for d in all_layer_data:
        first_gap = d["layers"][sample_layers[0]]["logit_gap"]
        final_gap = d["layers"][sample_layers[-1]]["logit_gap"]
        gap_growth.append({
            "first": first_gap,
            "final": final_gap,
            "growth": final_gap - first_gap,
            "growth_ratio": final_gap / (first_gap + 1e-10),
        })
    
    print(f"  首层logit_gap均值: {np.mean([g['first'] for g in gap_growth]):.4f}")
    print(f"  末层logit_gap均值: {np.mean([g['final'] for g in gap_growth]):.4f}")
    print(f"  增长量均值: {np.mean([g['growth'] for g in gap_growth]):.4f}")
    print(f"  增长倍数均值: {np.mean([g['growth_ratio'] for g in gap_growth]):.2f}")
    
    # 增长量与预测质量的关系
    growths = np.array([g['growth'] for g in gap_growth])
    top1_probs = np.array([d["top1_prob"] for d in all_layer_data])
    r_growth_prob, _ = spearmanr(growths, top1_probs)
    print(f"  logit_gap增长 -> prob: r={r_growth_prob:.3f}")
    
    # ===== 6. logit_gap传播的非线性特征 =====
    print(f"\n=== 6. logit_gap传播的非线性特征 ===")
    # gap(l+1) = f(gap(l)) 是否是线性的?
    # 用最后5层的logit_gap做非线性检验
    
    late_layers = [l for l in sample_layers if l >= 2*n_layers//3]
    if len(late_layers) >= 3:
        for i in range(len(late_layers)-2):
            l1, l2, l3 = late_layers[i], late_layers[i+1], late_layers[i+2]
            gaps1 = np.array([d["layers"][l1]["logit_gap"] for d in all_layer_data])
            gaps2 = np.array([d["layers"][l2]["logit_gap"] for d in all_layer_data])
            gaps3 = np.array([d["layers"][l3]["logit_gap"] for d in all_layer_data])
            
            # 线性: gap3 ≈ a*gap2 + b
            # 非线性: gap3 ≈ a*gap2 + c*gap2^2 + b
            # 检验: gap2^2是否提供额外信息?
            from scipy.stats import rankdata
            g2_rank = rankdata(gaps2)
            g2_sq_rank = rankdata(gaps2**2)
            g3_rank = rankdata(gaps3)
            
            r_linear, _ = spearmanr(g2_rank, g3_rank)
            r_nonlinear, _ = spearmanr(g2_rank + g2_sq_rank, g3_rank)
            
            print(f"  L{l2}->L{l3}: 线性r={r_linear:.3f}, 含非线性项r={r_nonlinear:.3f}, 改善={abs(r_nonlinear-r_linear)*100:.1f}%")
    
    result = {
        "model": model_name, "n_texts": n, "n_layers": n_layers,
        "sample_layers": sample_layers,
        "layer_gap_stats": {str(li): {"mean": float(layer_gap_means[li]), 
                                       "std": float(layer_gap_stds[li])}
                           for li in sample_layers},
        "propagation": {
            "prev_to_final_r": float(r_prev_final),
            "mean_adjacent_r": float(np.mean(adjacent_rs)),
            "layer_to_final_r": {str(li): float(r) for li, r in layer_to_final_r.items()},
        },
        "layer_causal": {str(li): layer_spectral_gap_r[li] for li in sample_layers},
        "gap_growth": {
            "first_mean": float(np.mean([g['first'] for g in gap_growth])),
            "final_mean": float(np.mean([g['final'] for g in gap_growth])),
            "growth_mean": float(np.mean([g['growth'] for g in gap_growth])),
            "growth_to_prob_r": float(r_growth_prob),
        },
    }
    return result


# ===================== P582: 频谱->logit_gap的直接路径修正 =====================
def experiment_p582(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P582: 频谱->logit_gap的直接路径修正
    核心洞察: 不是频谱整体参数->logit_gap, 而是方向级频谱->方向级贡献->logit_gap
    
    创新点:
    1. 绕过logit_var, 直接建模频谱->logit_gap
    2. 方向级模型: |c_k|^2 -> c_k*Delta_k -> logit_gap
    3. 两步预测: 第一步用频谱力学预测|c_k|, 第二步用W_U结构预测Delta_k的分布
    4. 跨模型验证: 这个路径是否在所有模型中一致?
    """
    print(f"\n{'='*60}")
    print(f"P582: 频谱->logit_gap的直接路径修正 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    K = 50
    
    all_data = []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # 频谱系数
        c_k = U_wu[:, :K].T @ h_last
        abs_coeffs = np.abs(c_k)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        # Alpha/Beta
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        delta = h_last - h_prev
        beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
        
        # Logits
        full_logits = W_U @ h_last
        sorted_ids = np.argsort(full_logits)[::-1]
        top1_id, top2_id = sorted_ids[0], sorted_ids[1]
        logit_gap = full_logits[top1_id] - full_logits[top2_id]
        
        # Delta_k和方向贡献
        Delta_k = np.zeros(K)
        dir_contributions = np.zeros(K)
        for k in range(K):
            wu_top1_proj = np.dot(W_U[top1_id], U_wu[:, k])
            wu_top2_proj = np.dot(W_U[top2_id], U_wu[:, k])
            Delta_k[k] = wu_top1_proj - wu_top2_proj
            dir_contributions[k] = c_k[k] * Delta_k[k]
        
        # 频谱参数
        ratio_10 = np.sum(abs_coeffs[:10]**2) / total_energy
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        
        # 预测质量
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        
        # Delta频谱
        delta_coeffs = U_wu[:, :K].T @ delta
        abs_delta = np.abs(delta_coeffs)
        delta_total = np.sum(abs_delta**2) + 1e-10
        delta_ratio_10 = np.sum(abs_delta[:10]**2) / delta_total
        
        # 方向级能量
        dir_energy = abs_coeffs**2  # [K]
        
        # 方向级|c_k|排名
        c_k_rank = np.argsort(abs_coeffs)[::-1]
        
        all_data.append({
            "logit_gap": logit_gap,
            "top1_prob": top1_prob,
            "alpha": alpha, "beta": beta,
            "ratio_10": ratio_10, "ratio_50": ratio_50,
            "c_k": c_k.tolist(),
            "Delta_k": Delta_k.tolist(),
            "dir_contributions": dir_contributions.tolist(),
            "dir_energy": dir_energy.tolist(),
            "delta_ratio_10": delta_ratio_10,
        })
        
        del outputs
        torch.cuda.empty_cache()
    
    n = len(all_data)
    logit_gaps = np.array([d["logit_gap"] for d in all_data])
    top1_probs = np.array([d["top1_prob"] for d in all_data])
    
    # ===== 1. 基线: 旧路径 vs 新路径 =====
    print(f"\n=== 1. 基线对比: 旧路径 vs 新路径 ===")
    
    alphas = np.array([d["alpha"] for d in all_data])
    betas = np.array([d["beta"] for d in all_data])
    ratio_50s = np.array([d["ratio_50"] for d in all_data])
    
    # 旧路径: 频谱整体参数 -> logit_var -> logit_gap
    r_ratio_gap, _ = spearmanr(ratio_50s, logit_gaps)
    r_alpha_gap, _ = spearmanr(alphas, logit_gaps)
    print(f"  旧路径: ratio_50->gap r={r_ratio_gap:.3f}")
    print(f"  旧路径: alpha->gap r={r_alpha_gap:.3f}")
    
    # 新路径1: 方向级精确分解
    exact_gaps = np.array([np.sum(np.array(d["c_k"]) * np.array(d["Delta_k"])) for d in all_data])
    r_exact_gap, _ = spearmanr(exact_gaps, logit_gaps)
    print(f"  新路径: Sum c_k*Delta_k -> gap r={r_exact_gap:.3f}")
    
    # 新路径2: 用|c_k|^2和E[Delta_k^2]近似
    Delta_k_all = np.array([d["Delta_k"] for d in all_data])
    Delta_k_sq_mean = np.mean(Delta_k_all**2, axis=0)  # [K]
    
    c_k_all = np.array([d["c_k"] for d in all_data])
    approx_gaps_v1 = np.sum(c_k_all**2 * Delta_k_sq_mean[np.newaxis, :], axis=1)  # Sum c_k^2 * E[Delta_k^2]
    r_approx_v1, _ = spearmanr(approx_gaps_v1, logit_gaps)
    print(f"  新路径: Sum c_k^2*E[Delta_k^2] -> gap r={r_approx_v1:.3f}")
    
    # 新路径3: 用|c_k|和E[|Delta_k|]近似
    Delta_k_abs_mean = np.mean(np.abs(Delta_k_all), axis=0)
    approx_gaps_v2 = np.sum(np.abs(c_k_all) * Delta_k_abs_mean[np.newaxis, :], axis=1)
    r_approx_v2, _ = spearmanr(approx_gaps_v2, logit_gaps)
    print(f"  新路径: Sum |c_k|*E[|Delta_k|] -> gap r={r_approx_v2:.3f}")
    
    # 新路径4: 只用方向能量|c_k|^2预测(不需要Delta_k)
    dir_energy_all = np.array([d["dir_energy"] for d in all_data])
    # 假设logit_gap ~ Sum sqrt(dir_energy_k) * const_k
    from scipy.stats import rankdata
    sqrt_energy_sum = np.sum(np.sqrt(dir_energy_all), axis=1)
    r_sqrt_energy, _ = spearmanr(sqrt_energy_sum, logit_gaps)
    print(f"  新路径: Sum sqrt(E_k) -> gap r={r_sqrt_energy:.3f}")
    
    # ===== 2. c_k的频谱力学预测 =====
    print(f"\n=== 2. c_k的频谱力学预测 ===")
    # 理论: |c_k|^2 ~ S_k (频谱能量)
    # S_k由频谱力学方程决定: S(l) = alpha*S(l-1) + beta*S_delta(l)
    
    # 检验: alpha和|c_k|的关系
    print(f"  alpha与各方向|c_k|的相关:")
    alpha_c_corrs = []
    for k in [0, 1, 2, 5, 10, 20]:
        r, p = spearmanr(alphas, np.abs(c_k_all[:, k]))
        alpha_c_corrs.append((k, r, p))
        print(f"    alpha vs |c_{k}|: r={r:.3f} (p={p:.4f})")
    
    # ===== 3. 方向能量->方向贡献的映射 =====
    print(f"\n=== 3. 方向能量->方向贡献的映射 ===")
    # 方向贡献 = c_k * Delta_k
    # 方向能量 = c_k^2
    # 两者之间的关系取决于Delta_k
    
    # 对每个方向k, |c_k| -> |c_k*Delta_k| 的相关
    dir_energy_to_contrib_r = np.zeros(K)
    for k in range(K):
        r, _ = spearmanr(dir_energy_all[:, k], np.abs(np.array([d["dir_contributions"][k] for d in all_data])))
        dir_energy_to_contrib_r[k] = r
    
    mean_r = np.mean(dir_energy_to_contrib_r)
    print(f"  |c_k|^2->|c_k*Delta_k| 平均相关: {mean_r:.3f}")
    print(f"  |c_k|^2->|c_k*Delta_k| 最强方向: Dir{np.argmax(dir_energy_to_contrib_r)} (r={np.max(dir_energy_to_contrib_r):.3f})")
    
    # ===== 4. 绕过logit_var的直接路径 =====
    print(f"\n=== 4. 绕过logit_var的直接路径 ===")
    # 旧: ratio_50 -> logit_var -> logit_gap (断裂在第二步)
    # 新: |c_k|^2 -> c_k*Delta_k -> logit_gap (精确等式, 不需要logit_var)
    
    # 验证: logit_var是否是必要的中间变量?
    logit_vars = np.array([float(np.var(W_U @ U_wu[:, :K] @ c_k_all[i])) for i in range(n)])
    
    # 方法1: 通过|c_k|预测logit_var
    dir_energy_sum = np.sum(dir_energy_all, axis=1)
    r_energy_var, _ = spearmanr(dir_energy_sum, logit_vars)
    print(f"  Sum|c_k|^2 -> logit_var: r={r_energy_var:.3f}")
    
    # 方法2: 直接从频谱力学到logit_gap
    # α->|c_k|, 但|c_k|->logit_gap需要知道Delta_k
    # 如果Delta_k近似常数: logit_gap ~ Sum |c_k| * Delta_k_mean
    # 如果Delta_k变化: 需要更多信息
    
    # 测试Delta_k的"可替代性": 用Delta_k的均值替代实际Delta_k
    Delta_k_mean = np.mean(Delta_k_all, axis=0)  # [K]
    Delta_k_mean_contrib = c_k_all * Delta_k_mean[np.newaxis, :]  # [n, K]
    approx_gap_mean_delta = np.sum(Delta_k_mean_contrib, axis=1)
    r_mean_delta, _ = spearmanr(approx_gap_mean_delta, logit_gaps)
    print(f"  Sum c_k * E[Delta_k] -> logit_gap: r={r_mean_delta:.3f}")
    
    # 用Delta_k中位数
    Delta_k_median = np.median(Delta_k_all, axis=0)
    approx_gap_median_delta = np.sum(c_k_all * Delta_k_median[np.newaxis, :], axis=1)
    r_median_delta, _ = spearmanr(approx_gap_median_delta, logit_gaps)
    print(f"  Sum c_k * median(Delta_k) -> logit_gap: r={r_median_delta:.3f}")
    
    # ===== 5. 方向选择效应: 哪些方向的贡献最关键 =====
    print(f"\n=== 5. 方向选择效应 ===")
    # 不是所有方向同等重要, 只有少数方向决定logit_gap
    
    # 逐步添加方向, 看logit_gap预测力如何增长
    dir_gap_corrs = np.zeros(K)
    for k in range(K):
        contribs_k = np.array([d["dir_contributions"][k] for d in all_data])
        r, _ = spearmanr(contribs_k, logit_gaps)
        dir_gap_corrs[k] = r
    
    # 按预测力排序方向
    sorted_dirs = np.argsort(np.abs(dir_gap_corrs))[::-1]
    
    # 逐步添加
    print(f"  逐步添加方向(按预测力排序):")
    cumulative_contrib = np.zeros(n)
    for n_dirs in [1, 3, 5, 10, 20, 50]:
        for d in sorted_dirs[:n_dirs]:
            cumulative_contrib += np.array([d_i["dir_contributions"][d] for d_i in all_data])
        r, _ = spearmanr(cumulative_contrib, logit_gaps)
        print(f"    Top-{n_dirs}方向: Sum c_k*Delta_k -> gap r={r:.3f}")
        cumulative_contrib = np.zeros(n)
    
    # ===== 6. 完整因果链汇总 =====
    print(f"\n=== 6. 完整因果链汇总 ===")
    
    # 路径A: 旧路径
    print(f"  路径A(旧): ratio_50->logit_var->logit_gap")
    print(f"    ratio_50->gap: r={r_ratio_gap:.3f}")
    
    # 路径B: 新路径(精确)
    print(f"  路径B(新-精确): |c_k|->c_k*Delta_k->logit_gap")
    print(f"    精确Sum c_k*Delta_k->gap: r={r_exact_gap:.3f}")
    
    # 路径C: 新路径(近似-仅频谱)
    print(f"  路径C(新-近似): |c_k|^2+Delta_k统计->logit_gap")
    print(f"    Sum c_k*E[Delta_k]->gap: r={r_mean_delta:.3f}")
    print(f"    Sum |c_k|*E[|Delta_k|]->gap: r={r_approx_v2:.3f}")
    
    result = {
        "model": model_name, "n_texts": n,
        "baseline": {
            "ratio50_to_gap_r": float(r_ratio_gap),
            "alpha_to_gap_r": float(r_alpha_gap),
        },
        "new_path": {
            "exact_r": float(r_exact_gap),
            "approx_v1_r": float(r_approx_v1),
            "approx_v2_r": float(r_approx_v2),
            "sqrt_energy_r": float(r_sqrt_energy),
            "mean_delta_r": float(r_mean_delta),
            "median_delta_r": float(r_median_delta),
        },
        "spectral_mechanics": {
            "alpha_c_corrs": {str(k): {"r": float(r), "p": float(p)} for k, r, p in alpha_c_corrs},
            "energy_to_contrib_mean_r": float(mean_r),
            "energy_to_contrib_per_dir": dir_energy_to_contrib_r.tolist(),
        },
        "dir_gap_corrs": dir_gap_corrs.tolist(),
        "sorted_dirs_by_pred": sorted_dirs.tolist(),
        "delta_k_stats": {
            "Delta_k_mean": Delta_k_mean.tolist(),
            "Delta_k_sq_mean": Delta_k_sq_mean.tolist(),
        },
    }
    return result


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p579", "p580", "p581", "p582"])
    args = parser.parse_args()
    
    start_time = time.time()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=300)
    
    if args.experiment == "p579":
        result = experiment_p579(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p580":
        result = experiment_p580(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p581":
        result = experiment_p581(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p582":
        result = experiment_p582(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = "results/phase_cxxxiv"
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
