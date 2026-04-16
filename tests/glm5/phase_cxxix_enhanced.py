"""
Phase CXXIX Enhanced: 大数据量语义编码深层机制验证
用100+文本和多层采样验证P563-P566的核心发现
"""
import argparse
import json
import os
import sys
import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr, ttest_ind
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights

import torch

# 大量测试文本
TEST_TEXTS = [
    "The development of artificial intelligence has transformed many aspects of modern life",
    "In the early morning light, the birds sang their melodious songs across the valley",
    "Scientists discovered a new species of deep-sea fish near the volcanic vents",
    "The ancient temple stood silently on the mountain, watching centuries pass by",
    "Quantum computing promises to revolutionize cryptography and data processing",
    "She walked through the garden, picking roses and humming a gentle tune",
    "The economic crisis led to widespread unemployment and social unrest",
    "Mathematics provides the foundation for understanding the physical world",
    "The chef carefully prepared the traditional recipe passed down through generations",
    "Climate change poses significant challenges for coastal communities worldwide",
    "The orchestra performed Beethoven's ninth symphony with remarkable precision",
    "Advances in medicine have significantly increased human life expectancy",
    "The old library contained thousands of rare manuscripts from the medieval period",
    "Children played happily in the park while their parents watched from the benches",
    "The spacecraft successfully completed its mission to study the distant planet",
    "Philosophers have debated the nature of consciousness for thousands of years",
    "The river flowed peacefully through the countryside, reflecting the sunset",
    "Technological innovation drives economic growth and social transformation",
    "The artist captured the essence of human emotion in her latest painting",
    "Education is the most powerful weapon which you can use to change the world",
    "The city skyline glittered with lights as night fell over the metropolis",
    "Renewable energy sources are becoming increasingly important for sustainable development",
    "The detective carefully examined the evidence to solve the mysterious case",
    "Music has the power to transcend language barriers and unite people",
    "The professor delivered an insightful lecture on the history of philosophy",
    "Rapid urbanization has created both opportunities and challenges for modern societies",
    "The mountain trail wound through dense forests and alongside crystal streams",
    "Artificial neural networks are inspired by the biological structure of the brain",
    "The novel explored themes of love, loss, and redemption in post-war society",
    "International cooperation is essential for addressing global challenges effectively",
    "The sunset painted the sky in shades of orange, pink, and purple",
    "Advances in genetics have opened new possibilities for treating inherited diseases",
    "The market economy relies on supply and demand to allocate resources efficiently",
    "The ancient philosopher taught that wisdom comes from understanding oneself",
    "Digital technology has fundamentally changed how we communicate and access information",
    "The garden was filled with colorful flowers and the sweet scent of jasmine",
    "Scientific research requires rigorous methodology and careful attention to detail",
    "The historical monument commemorates the sacrifices made during the struggle for independence",
    "Biodiversity is essential for maintaining healthy ecosystems and human well-being",
    "The pianist performed a beautiful sonata that moved the audience to tears",
    "Understanding different cultures promotes tolerance and reduces prejudice in society",
    "The spacecraft traveled through the asteroid belt on its journey to Jupiter",
    "Poetry captures the beauty and complexity of human experience in condensed form",
    "The legislative process involves careful deliberation and compromise among stakeholders",
    "The rain fell gently on the roof, creating a soothing rhythm that lulled her to sleep",
    "Sustainable agriculture practices help preserve soil quality and protect water resources",
    "The ancient civilization built remarkable structures that still impress engineers today",
    "Environmental conservation requires balancing human needs with ecological preservation",
    "The symphony orchestra rehearsed diligently for their upcoming performance",
    "Language acquisition in children follows predictable developmental stages",
    "The documentary highlighted the plight of endangered species in tropical forests",
    "Philosophical inquiry seeks to understand the fundamental nature of reality and existence",
    "The coastal town relied on fishing and tourism as its primary economic activities",
    "Medical imaging technology has greatly improved diagnostic accuracy in healthcare",
    "The ancient text contained wisdom that remains relevant in contemporary society",
    "Social media has transformed the landscape of political discourse and civic engagement",
    "The waterfall cascaded down the rocky cliff into the crystal-clear pool below",
    "Economic inequality remains one of the most pressing challenges of the twenty-first century",
    "The researcher published groundbreaking findings in the prestigious scientific journal",
    "Traditional crafts represent an important cultural heritage that deserves preservation",
    "The university offered courses spanning the full range of human knowledge",
    "Urban planning must address issues of housing, transportation, and public spaces",
    "The hiker reached the summit just as the sun broke through the clouds",
    "Cognitive science investigates the nature and mechanisms of mental processes",
    "The museum exhibited artifacts from civilizations spanning five thousand years of history",
    "Volunteer organizations play a crucial role in supporting community welfare",
    "The novelist created memorable characters who embodied the contradictions of their era",
    "Advances in robotics are reshaping manufacturing and service industries worldwide",
    "The traditional ceremony celebrated the harvest and honored the ancestors",
    "Human rights are universal and inalienable, belonging to every person by virtue of their humanity",
    "The glacier had been retreating for decades, a visible sign of global warming",
    "The teacher encouraged her students to think critically and question assumptions",
    "Renewable energy technologies are becoming more efficient and cost-effective each year",
    "The archaeological excavation revealed evidence of a previously unknown settlement",
    "Cultural exchange enriches societies by introducing new perspectives and practices",
    "The bridge connected the two sides of the river, facilitating trade and communication",
    "Neuroscience research has revealed remarkable plasticity in the developing brain",
    "The citizens participated actively in the democratic process by voting in the election",
    "The poem evoked images of autumn leaves falling silently on still water",
    "International trade creates economic interdependence that can promote peace and cooperation",
    "The laboratory conducted experiments to test the hypothesis under controlled conditions",
    "The forest ecosystem supports a diverse array of plant and animal species",
    "The philosopher argued that ethical behavior is grounded in rational principles",
    "The city implemented policies to reduce air pollution and improve public health",
    "The musician composed a piece that blended classical and contemporary styles seamlessly",
    "Space exploration expands our understanding of the universe and our place within it",
    "The community garden provided fresh vegetables and a gathering place for neighbors",
    "The historian analyzed primary sources to construct an accurate account of the event",
    "Public health initiatives have dramatically reduced the incidence of preventable diseases",
    "The dancer moved with grace and precision, expressing emotions without words",
    "Technological literacy is increasingly important for participation in modern society",
    "The river delta supports rich biodiversity and provides ecosystem services to millions",
    "The seminar explored the intersection of technology, ethics, and public policy",
    "The old man sat on the bench, watching children play and remembering his own youth",
    "Climate models predict significant changes in precipitation patterns over the coming decades",
    "The painting depicted a scene of rural life in nineteenth-century Europe",
    "Collaborative research across disciplines leads to innovative solutions to complex problems",
    "The train journeyed through the countryside, offering passengers scenic views of rolling hills",
    "The constitution establishes the framework for governance and protects individual liberties",
    "The chef combined exotic spices to create a dish with complex flavors and aromas",
    "Understanding the brain remains one of the greatest challenges in modern science",
    "The documentary filmmaker captured compelling stories of resilience and hope",
    "The ancient scrolls contained mathematical formulas that were centuries ahead of their time",
    "Social institutions shape individual behavior and collective outcomes in complex ways",
    "The night sky was filled with stars, each one a distant sun illuminating the cosmos",
    "The engineering team designed a bridge that could withstand extreme weather conditions",
    "Literature provides insight into the human condition across cultures and throughout history",
    "The discovery of antibiotics revolutionized medicine and saved countless lives",
    "The students debated the merits of different economic systems with passion and rigor",
    "The park provided a peaceful retreat from the noise and bustle of city life",
]


def compute_wu_svd(model, k=200):
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


def run_enhanced_p563(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """P563增强: 用更多文本和更大数据量验证W_U方向的语义内容"""
    print(f"\n{'='*60}")
    print(f"P563 Enhanced: W_U方向语义解码(大数据量) — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    k_wu = min(200, U_wu.shape[1])
    n_vocab = W_U.shape[0]
    
    # 1. 每个方向的Top-20 tokens (仅方向0-4)
    direction_data = {}
    for d in range(5):
        direction = U_wu[:, d]
        projections = W_U @ direction
        top_pos_ids = np.argsort(projections)[::-1][:20]
        top_neg_ids = np.argsort(projections)[:20]
        top_pos = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_pos_ids]
        top_neg = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_neg_ids]
        direction_data[d] = {"pos": top_pos, "neg": top_neg}
        pos_str = ", ".join([f"{t}({v:.2f})" for t, v in top_pos[:5]])
        neg_str = ", ".join([f"{t}({v:.2f})" for t, v in top_neg[:5]])
        print(f"  方向{d}: Top+=[{pos_str}]")
        print(f"         Top-=[{neg_str}]")
    
    # 2. 跨100+文本验证方向0-4的h投影统计
    print(f"\n--- 跨{len(TEST_TEXTS)}文本的方向投影统计 ---")
    direction_projections_all = {d: [] for d in range(5)}
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        for d in range(5):
            proj = np.dot(U_wu[:, d], h_last)
            direction_projections_all[d].append(proj)
        del outputs
        torch.cuda.empty_cache()
    
    for d in range(5):
        projs = np.array(direction_projections_all[d])
        print(f"  方向{d}: mean={np.mean(projs):.3f}, std={np.std(projs):.3f}, "
              f"|mean|/std={abs(np.mean(projs))/(np.std(projs)+1e-10):.3f}")
    
    # 3. 方向间的投影相关性
    print("\n--- 方向间投影相关性 ---")
    for i in range(5):
        for j in range(i+1, 5):
            r, _ = pearsonr(direction_projections_all[i], direction_projections_all[j])
            print(f"  方向{i} vs 方向{j}: r={r:.4f}")
    
    result = {
        "model": model_name,
        "n_texts": len(TEST_TEXTS),
        "direction_stats": {
            str(d): {
                "mean": float(np.mean(direction_projections_all[d])),
                "std": float(np.std(direction_projections_all[d])),
                "abs_mean_over_std": float(abs(np.mean(direction_projections_all[d])) / (np.std(direction_projections_all[d]) + 1e-10))
            }
            for d in range(5)
        }
    }
    return result


def run_enhanced_p564(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """P564增强: 用大量功能词/内容词验证双峰假设"""
    print(f"\n{'='*60}")
    print(f"P564 Enhanced: 双峰假设验证(大数据量) — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    k_wu = min(200, U_wu.shape[1])
    
    # 扩大的功能词和内容词集合
    func_words = [
        # 冠词
        "the", "a", "an",
        # be动词
        "is", "are", "was", "were", "be", "been", "being", "am",
        # 助动词
        "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "can", "could", "must",
        # 介词
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "between", "under", "over", "about", "against",
        # 连词
        "and", "but", "or", "nor", "not", "so", "yet",
        # 代词
        "that", "which", "who", "whom", "this", "these", "those",
        "it", "its", "he", "she", "they", "we", "you", "i", "me",
        "him", "her", "us", "them", "my", "your", "his", "our",
        # 限定词
        "some", "any", "all", "each", "every", "both", "either",
        "neither", "no", "other", "such",
    ]
    
    content_words = [
        # 动物
        "cat", "dog", "bird", "fish", "horse", "elephant", "tiger",
        "lion", "bear", "whale", "snake", "eagle", "dolphin",
        # 物品
        "house", "car", "tree", "water", "book", "table", "chair",
        "door", "window", "wall", "road", "bridge", "ship",
        # 动作
        "run", "eat", "walk", "think", "speak", "write", "read",
        "build", "create", "destroy", "discover", "explore",
        # 形容词
        "big", "small", "red", "blue", "happy", "sad", "fast",
        "slow", "hot", "cold", "dark", "bright", "strong",
        # 抽象概念
        "love", "time", "world", "power", "nature", "science",
        "music", "art", "life", "death", "truth", "beauty",
        "freedom", "justice", "knowledge", "wisdom", "courage",
        # 领域
        "computer", "planet", "ocean", "mountain", "forest",
        "desert", "river", "island", "city", "country",
    ]
    
    # 获取词在W_U空间中的频谱
    func_degrees = []
    content_degrees = []
    func_ratios = []
    content_ratios = []
    
    for word in func_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        tok_id = tok_ids[0]
        coord = W_U[tok_id]
        proj = U_wu[:, :k_wu].T @ coord
        abs_proj = np.abs(proj)
        total = np.sum(abs_proj**2) + 1e-10
        
        func_degree = np.sum(abs_proj[:5]**2) / total  # 前5方向能量比
        ratio_10 = np.sum(abs_proj[:10]**2) / total  # 前10方向
        ratio_50 = np.sum(abs_proj[:50]**2) / total  # 前50方向
        
        func_degrees.append(func_degree)
        func_ratios.append({"r5": func_degree, "r10": ratio_10, "r50": ratio_50})
    
    for word in content_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        tok_id = tok_ids[0]
        coord = W_U[tok_id]
        proj = U_wu[:, :k_wu].T @ coord
        abs_proj = np.abs(proj)
        total = np.sum(abs_proj**2) + 1e-10
        
        func_degree = np.sum(abs_proj[:5]**2) / total
        ratio_10 = np.sum(abs_proj[:10]**2) / total
        ratio_50 = np.sum(abs_proj[:50]**2) / total
        
        content_degrees.append(func_degree)
        content_ratios.append({"r5": func_degree, "r10": ratio_10, "r50": ratio_50})
    
    func_degrees = np.array(func_degrees)
    content_degrees = np.array(content_degrees)
    
    print(f"功能词数: {len(func_degrees)}, 内容词数: {len(content_degrees)}")
    
    # t-test
    t_stat, p_val = ttest_ind(func_degrees, content_degrees)
    print(f"\n--- 双峰检验 ---")
    print(f"  功能词功能度(前5方向能量比): {np.mean(func_degrees):.3f} ± {np.std(func_degrees):.3f}")
    print(f"  内容词功能度(前5方向能量比): {np.mean(content_degrees):.3f} ± {np.std(content_degrees):.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.6f} {'**显著**' if p_val < 0.01 else '不显著'}")
    
    # BIC比较
    all_degrees = np.concatenate([func_degrees, content_degrees])
    mu_single = np.mean(all_degrees)
    sigma_single = np.std(all_degrees) + 1e-10
    log_lik_single = np.sum(-0.5 * ((all_degrees - mu_single) / sigma_single)**2 - np.log(sigma_single))
    
    mu_func = np.mean(func_degrees)
    sigma_func = np.std(func_degrees) + 1e-10
    mu_content = np.mean(content_degrees)
    sigma_content = np.std(content_degrees) + 1e-10
    log_lik_dual = (np.sum(-0.5 * ((func_degrees - mu_func) / sigma_func)**2 - np.log(sigma_func)) +
                    np.sum(-0.5 * ((content_degrees - mu_content) / sigma_content)**2 - np.log(sigma_content)))
    
    bic_single = -2 * log_lik_single + 2 * np.log(len(all_degrees))
    bic_dual = -2 * log_lik_dual + 4 * np.log(len(all_degrees))
    
    print(f"  BIC: 单高斯={bic_single:.1f}, 双高斯={bic_dual:.1f}")
    print(f"  双高斯更优? {'是' if bic_dual < bic_single else '否'} (差距={bic_single - bic_dual:.1f})")
    
    # 各ratio的比较
    func_r5 = np.mean([r["r5"] for r in func_ratios])
    func_r10 = np.mean([r["r10"] for r in func_ratios])
    func_r50 = np.mean([r["r50"] for r in func_ratios])
    content_r5 = np.mean([r["r5"] for r in content_ratios])
    content_r10 = np.mean([r["r10"] for r in content_ratios])
    content_r50 = np.mean([r["r50"] for r in content_ratios])
    
    print(f"\n--- 频谱集中度对比 ---")
    print(f"  功能词: r5={func_r5:.3f}, r10={func_r10:.3f}, r50={func_r50:.3f}")
    print(f"  内容词: r5={content_r5:.3f}, r10={content_r10:.3f}, r50={content_r50:.3f}")
    print(f"  差距:   r5={func_r5-content_r5:+.3f}, r10={func_r10-content_r10:+.3f}, r50={func_r50-content_r50:+.3f}")
    
    result = {
        "model": model_name,
        "n_func": len(func_degrees),
        "n_content": len(content_degrees),
        "func_func_degree_mean": float(np.mean(func_degrees)),
        "content_func_degree_mean": float(np.mean(content_degrees)),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "bic_single": float(bic_single),
        "bic_dual": float(bic_dual),
        "dual_better": bool(bic_dual < bic_single),
        "func_r5": float(func_r5), "func_r10": float(func_r10), "func_r50": float(func_r50),
        "content_r5": float(content_r5), "content_r10": float(content_r10), "content_r50": float(content_r50),
    }
    return result


def run_enhanced_p565(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """P565增强: 用100+文本验证频谱到logits的路径"""
    print(f"\n{'='*60}")
    print(f"P565 Enhanced: 频谱→logits路径(大数据量) — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    k_wu = min(200, U_wu.shape[1])
    
    all_cos_k5, all_cos_k10, all_cos_k50 = [], [], []
    all_r5, all_r10, all_r50 = [], [], []
    all_top1_prob, all_entropy = [], []
    
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        r5 = np.sum(abs_coeffs[:5]**2) / total_energy
        r10 = np.sum(abs_coeffs[:10]**2) / total_energy
        r50 = np.sum(abs_coeffs[:50]**2) / total_energy
        
        full_logits = W_U @ h_last
        
        # K=5,10,50截断重建
        for K, cos_list in [(5, all_cos_k5), (10, all_cos_k10), (50, all_cos_k50)]:
            h_recon = U_wu[:, :K] @ h_coeffs[:K]
            logits_k = W_U @ h_recon
            cos = np.dot(full_logits, logits_k) / (np.linalg.norm(full_logits) * np.linalg.norm(logits_k) + 1e-10)
            cos_list.append(cos)
        
        # 预测概率
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        all_r5.append(r5)
        all_r10.append(r10)
        all_r50.append(r50)
        all_top1_prob.append(top1_prob)
        all_entropy.append(entropy)
        
        del outputs
        torch.cuda.empty_cache()
    
    # 统计
    print(f"\n--- 频谱集中度(跨{len(TEST_TEXTS)}文本) ---")
    print(f"  r(5):  mean={np.mean(all_r5):.3f}, std={np.std(all_r5):.3f}")
    print(f"  r(10): mean={np.mean(all_r10):.3f}, std={np.std(all_r10):.3f}")
    print(f"  r(50): mean={np.mean(all_r50):.3f}, std={np.std(all_r50):.3f}")
    
    print(f"\n--- logits重建余弦(跨{len(TEST_TEXTS)}文本) ---")
    print(f"  K=5:  mean={np.mean(all_cos_k5):.4f}, std={np.std(all_cos_k5):.4f}")
    print(f"  K=10: mean={np.mean(all_cos_k10):.4f}, std={np.std(all_cos_k10):.4f}")
    print(f"  K=50: mean={np.mean(all_cos_k50):.4f}, std={np.std(all_cos_k50):.4f}")
    
    print(f"\n--- 频谱→预测相关性 ---")
    corr_r10_prob, _ = spearmanr(all_r10, all_top1_prob)
    corr_r50_prob, _ = spearmanr(all_r50, all_top1_prob)
    corr_r10_ent, _ = spearmanr(all_r10, all_entropy)
    corr_r50_ent, _ = spearmanr(all_r50, all_entropy)
    corr_cos5_prob, _ = spearmanr(all_cos_k5, all_top1_prob)
    corr_cos10_prob, _ = spearmanr(all_cos_k10, all_top1_prob)
    
    print(f"  r(10) vs top1_prob: r={corr_r10_prob:.3f}")
    print(f"  r(50) vs top1_prob: r={corr_r50_prob:.3f}")
    print(f"  r(10) vs entropy: r={corr_r10_ent:.3f}")
    print(f"  r(50) vs entropy: r={corr_r50_ent:.3f}")
    print(f"  cos_k5 vs top1_prob: r={corr_cos5_prob:.3f}")
    print(f"  cos_k10 vs top1_prob: r={corr_cos10_prob:.3f}")
    
    result = {
        "model": model_name,
        "n_texts": len(TEST_TEXTS),
        "r5_mean": float(np.mean(all_r5)),
        "r10_mean": float(np.mean(all_r10)),
        "r50_mean": float(np.mean(all_r50)),
        "cos_k5_mean": float(np.mean(all_cos_k5)),
        "cos_k10_mean": float(np.mean(all_cos_k10)),
        "cos_k50_mean": float(np.mean(all_cos_k50)),
        "corr_r10_prob": float(corr_r10_prob),
        "corr_r50_prob": float(corr_r50_prob),
        "corr_r10_ent": float(corr_r10_ent),
        "corr_r50_ent": float(corr_r50_ent),
        "corr_cos5_prob": float(corr_cos5_prob),
        "corr_cos10_prob": float(corr_cos10_prob),
    }
    return result


def run_enhanced_p566(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """P566增强: 统一语言理论验证(100+文本)"""
    print(f"\n{'='*60}")
    print(f"P566 Enhanced: 统一语言理论验证(大数据量) — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    k_wu = min(200, U_wu.shape[1])
    n_layers = info.n_layers
    
    # 收集所有文本在各层的频谱数据
    all_alpha = []
    all_ratio_50 = []
    all_func_energy = []
    all_spectral_slope = []
    all_cos_k5 = []
    all_cos_k10 = []
    all_cos_k50 = []
    all_top1_prob = []
    all_entropy = []
    
    for text in TEST_TEXTS[:50]:  # 用50个文本做详细分析
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
        
        # alpha
        alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
        
        # 频谱
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        func_energy = np.sum(abs_coeffs[:5]**2) / total_energy
        
        # 幂律斜率
        sorted_spec = np.sort(abs_coeffs)[::-1]
        x = np.arange(1, len(sorted_spec)+1)
        log_x = np.log10(x[3:30])
        log_y = np.log10(sorted_spec[3:30] + 1e-10)
        valid = np.isfinite(log_y)
        if np.sum(valid) > 3:
            slope, _ = np.polyfit(log_x[valid], log_y[valid], 1)
        else:
            slope = 0
        
        # logits重建
        full_logits = W_U @ h_last
        for K, cos_list in [(5, all_cos_k5), (10, all_cos_k10), (50, all_cos_k50)]:
            h_recon = U_wu[:, :K] @ h_coeffs[:K]
            logits_k = W_U @ h_recon
            cos = np.dot(full_logits, logits_k) / (np.linalg.norm(full_logits) * np.linalg.norm(logits_k) + 1e-10)
            cos_list.append(cos)
        
        # 预测概率
        with torch.no_grad():
            probs = torch.softmax(outputs.logits[0, -1].float(), dim=0)
        top1_prob = probs.max().item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        all_alpha.append(alpha)
        all_ratio_50.append(ratio_50)
        all_func_energy.append(func_energy)
        all_spectral_slope.append(slope)
        all_top1_prob.append(top1_prob)
        all_entropy.append(entropy)
        
        del outputs
        torch.cuda.empty_cache()
    
    print(f"\n--- 频谱力学参数(跨50文本) ---")
    print(f"  alpha: mean={np.mean(all_alpha):.4f}, std={np.std(all_alpha):.4f}")
    print(f"  ratio(50): mean={np.mean(all_ratio_50):.4f}, std={np.std(all_ratio_50):.4f}")
    print(f"  func_energy(r5): mean={np.mean(all_func_energy):.4f}, std={np.std(all_func_energy):.4f}")
    print(f"  spectral_slope: mean={np.mean(all_spectral_slope):.3f}, std={np.std(all_spectral_slope):.3f}")
    
    print(f"\n--- logits重建(跨50文本) ---")
    print(f"  K=5:  mean={np.mean(all_cos_k5):.4f}")
    print(f"  K=10: mean={np.mean(all_cos_k10):.4f}")
    print(f"  K=50: mean={np.mean(all_cos_k50):.4f}")
    
    print(f"\n--- 因果链: 频谱参数→预测质量 ---")
    # alpha → 预测
    corr_alpha_prob, _ = spearmanr(all_alpha, all_top1_prob)
    corr_alpha_ent, _ = spearmanr(all_alpha, all_entropy)
    # ratio_50 → 预测
    corr_r50_prob, _ = spearmanr(all_ratio_50, all_top1_prob)
    corr_r50_ent, _ = spearmanr(all_ratio_50, all_entropy)
    # func_energy → 预测
    corr_fe_prob, _ = spearmanr(all_func_energy, all_top1_prob)
    corr_fe_ent, _ = spearmanr(all_func_energy, all_entropy)
    # slope → 预测
    corr_slope_prob, _ = spearmanr(all_spectral_slope, all_top1_prob)
    # cos_k5 → 预测
    corr_cos5_prob, _ = spearmanr(all_cos_k5, all_top1_prob)
    corr_cos10_prob, _ = spearmanr(all_cos_k10, all_top1_prob)
    
    print(f"  alpha → top1_prob: r={corr_alpha_prob:.3f}")
    print(f"  alpha → entropy: r={corr_alpha_ent:.3f}")
    print(f"  ratio(50) → top1_prob: r={corr_r50_prob:.3f}")
    print(f"  ratio(50) → entropy: r={corr_r50_ent:.3f}")
    print(f"  func_energy → top1_prob: r={corr_fe_prob:.3f}")
    print(f"  func_energy → entropy: r={corr_fe_ent:.3f}")
    print(f"  spectral_slope → top1_prob: r={corr_slope_prob:.3f}")
    print(f"  cos_k5 → top1_prob: r={corr_cos5_prob:.3f}")
    print(f"  cos_k10 → top1_prob: r={corr_cos10_prob:.3f}")
    
    result = {
        "model": model_name,
        "n_texts": 50,
        "alpha_mean": float(np.mean(all_alpha)),
        "ratio_50_mean": float(np.mean(all_ratio_50)),
        "func_energy_mean": float(np.mean(all_func_energy)),
        "spectral_slope_mean": float(np.mean(all_spectral_slope)),
        "cos_k5_mean": float(np.mean(all_cos_k5)),
        "cos_k10_mean": float(np.mean(all_cos_k10)),
        "cos_k50_mean": float(np.mean(all_cos_k50)),
        "corr_alpha_prob": float(corr_alpha_prob),
        "corr_r50_prob": float(corr_r50_prob),
        "corr_fe_prob": float(corr_fe_prob),
        "corr_slope_prob": float(corr_slope_prob),
        "corr_cos5_prob": float(corr_cos5_prob),
        "corr_cos10_prob": float(corr_cos10_prob),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p563", "p564", "p565", "p566"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    
    if args.experiment == "p563":
        result = run_enhanced_p563(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p564":
        result = run_enhanced_p564(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p565":
        result = run_enhanced_p565(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p566":
        result = run_enhanced_p566(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = "results/phase_cxxix_enhanced"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.model}_{args.experiment}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    del model
    torch.cuda.empty_cache()
    print("GPU内存已释放")


if __name__ == "__main__":
    main()
