"""
Phase CXXXI-CXXXIV: 频谱力学因果修正
P567: MLP贡献的因果分析 — 为什么GLM4的alpha→prob负相关?
P568: 训练策略→alpha的差异分析 — alpha的层间分布与模型差异
P569: 语义方向(方向5-20)解码 — 方向0-4是格式,更高方向编码什么?
P570: 方向贡献耦合分析 — 为什么DS7B方差比=2.53?
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights

import torch

# 110文本测试集
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


# ============== P567: MLP贡献的因果分析 ==============
def experiment_p567(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P567: MLP贡献的因果分析 — 为什么GLM4的alpha→prob负相关?
    核心假设: alpha高→残差保持多→MLP贡献少→预测差
    如果MLP是"微调"而非"保持", 那么alpha高意味着缺少微调
    """
    print(f"\n{'='*60}")
    print(f"P567: MLP贡献因果分析 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    layers = get_layers(model)
    
    # 收集大量文本的数据
    all_data = []
    
    for text in TEST_TEXTS[:60]:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        seq_len = outputs.hidden_states[0].shape[1]
        
        # 对最后位置的多个层分析
        for pos_idx in [-1, -3, -5]:
            pos = seq_len + pos_idx
            if pos < 1:
                continue
            
            # 末层和前一层
            h_L = outputs.hidden_states[-1][0, pos].detach().cpu().float().numpy()
            h_L1 = outputs.hidden_states[-2][0, pos].detach().cpu().float().numpy()
            h_L2 = outputs.hidden_states[-3][0, pos].detach().cpu().float().numpy() if n_layers > 2 else h_L1
            
            # Alpha: 残差保持系数
            alpha = np.dot(h_L, h_L1) / (np.dot(h_L1, h_L1) + 1e-10)
            
            # Delta: 总变化
            delta = h_L - h_L1
            delta_norm = np.linalg.norm(delta)
            h_L1_norm = np.linalg.norm(h_L1)
            
            # Beta: MLP/Attn贡献幅度
            beta = delta_norm / (h_L1_norm + 1e-10)
            
            # 频谱参数
            h_coeffs = U_wu[:, :k_wu].T @ h_L
            abs_coeffs = np.abs(h_coeffs)
            total_energy = np.sum(abs_coeffs**2) + 1e-10
            ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
            func_energy = np.sum(abs_coeffs[:5]**2) / total_energy
            
            # 频谱变化: delta在W_U空间中的能量
            delta_coeffs = U_wu[:, :k_wu].T @ delta
            abs_delta = np.abs(delta_coeffs)
            delta_wu_energy = np.sum(abs_delta[:50]**2) / (np.sum(abs_delta**2) + 1e-10)
            
            # 预测质量
            with torch.no_grad():
                logits = outputs.logits[0, pos].float()
                probs = torch.softmax(logits, dim=0)
            top1_prob = probs.max().item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # Top-5预测
            top5_ids = torch.topk(logits, 5).indices.tolist()
            
            all_data.append({
                "alpha": alpha, "beta": beta, "delta_norm": delta_norm,
                "ratio_50": ratio_50, "func_energy": func_energy,
                "delta_wu_energy": delta_wu_energy,
                "top1_prob": top1_prob, "entropy": entropy,
            })
        
        del outputs
        torch.cuda.empty_cache()
    
    # 统计分析
    alphas = np.array([d["alpha"] for d in all_data])
    betas = np.array([d["beta"] for d in all_data])
    ratio_50s = np.array([d["ratio_50"] for d in all_data])
    func_energies = np.array([d["func_energy"] for d in all_data])
    delta_wu_energies = np.array([d["delta_wu_energy"] for d in all_data])
    top1_probs = np.array([d["top1_prob"] for d in all_data])
    entropies = np.array([d["entropy"] for d in all_data])
    
    print(f"\n--- 参数统计(n={len(all_data)}) ---")
    print(f"  alpha: mean={np.mean(alphas):.4f}, std={np.std(alphas):.4f}")
    print(f"  beta:  mean={np.mean(betas):.4f}, std={np.std(betas):.4f}")
    print(f"  delta_wu_energy: mean={np.mean(delta_wu_energies):.4f}")
    
    print(f"\n--- 因果链分析 ---")
    # alpha → prob
    r_alpha_prob, p_ap = spearmanr(alphas, top1_probs)
    r_alpha_ent, p_ae = spearmanr(alphas, entropies)
    print(f"  alpha → top1_prob: r={r_alpha_prob:.3f} (p={p_ap:.4f})")
    print(f"  alpha → entropy:   r={r_alpha_ent:.3f} (p={p_ae:.4f})")
    
    # beta → prob
    r_beta_prob, p_bp = spearmanr(betas, top1_probs)
    r_beta_ent, p_be = spearmanr(betas, entropies)
    print(f"  beta  → top1_prob: r={r_beta_prob:.3f} (p={p_bp:.4f})")
    print(f"  beta  → entropy:   r={r_beta_ent:.3f} (p={p_be:.4f})")
    
    # alpha vs beta 关系
    r_alpha_beta, _ = spearmanr(alphas, betas)
    print(f"  alpha vs beta:     r={r_alpha_beta:.3f}")
    
    # 偏相关: 控制beta后alpha→prob
    # 简单方法: 在beta高/低两组中分别看alpha→prob
    median_beta = np.median(betas)
    high_beta_mask = betas >= median_beta
    low_beta_mask = betas < median_beta
    
    if np.sum(high_beta_mask) > 5 and np.sum(low_beta_mask) > 5:
        r_alpha_prob_high, _ = spearmanr(alphas[high_beta_mask], top1_probs[high_beta_mask])
        r_alpha_prob_low, _ = spearmanr(alphas[low_beta_mask], top1_probs[low_beta_mask])
        print(f"  alpha→prob (高beta组): r={r_alpha_prob_high:.3f}")
        print(f"  alpha→prob (低beta组): r={r_alpha_prob_low:.3f}")
    
    # delta_wu_energy → prob (MLP贡献在W_U空间的集中度)
    r_delta_wu_prob, _ = spearmanr(delta_wu_energies, top1_probs)
    r_delta_wu_ent, _ = spearmanr(delta_wu_energies, entropies)
    print(f"  delta_wu_energy → top1_prob: r={r_delta_wu_prob:.3f}")
    print(f"  delta_wu_energy → entropy:   r={r_delta_wu_ent:.3f}")
    
    # 组合模型: alpha + beta → prob
    from scipy.stats import rankdata
    rank_alpha = rankdata(alphas)
    rank_beta = rankdata(betas)
    rank_combined = rank_alpha + rank_beta
    r_combined_prob, _ = spearmanr(rank_combined, top1_probs)
    print(f"  alpha+beta组合 → top1_prob: r={r_combined_prob:.3f}")
    
    result = {
        "model": model_name, "n_samples": len(all_data),
        "alpha_mean": float(np.mean(alphas)), "beta_mean": float(np.mean(betas)),
        "r_alpha_prob": float(r_alpha_prob), "r_alpha_ent": float(r_alpha_ent),
        "r_beta_prob": float(r_beta_prob), "r_beta_ent": float(r_beta_ent),
        "r_alpha_beta": float(r_alpha_beta),
        "r_delta_wu_prob": float(r_delta_wu_prob), "r_delta_wu_ent": float(r_delta_wu_ent),
        "r_combined_prob": float(r_combined_prob),
    }
    if np.sum(high_beta_mask) > 5 and np.sum(low_beta_mask) > 5:
        result["r_alpha_prob_high_beta"] = float(r_alpha_prob_high)
        result["r_alpha_prob_low_beta"] = float(r_alpha_prob_low)
    
    return result


# ============== P568: 训练策略→alpha差异 ==============
def experiment_p568(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P568: 训练策略→alpha的差异分析
    对每一层测量alpha, 分析层间分布和模型差异
    """
    print(f"\n{'='*60}")
    print(f"P568: alpha层间分布与模型差异 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    
    # 对多个文本测量每层的alpha
    text = TEST_TEXTS[0]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # 逐层alpha
    layer_alphas = []
    layer_betas = []
    layer_ratio_50 = []
    layer_func_energy = []
    layer_delta_norms = []
    
    for l in range(1, n_layers + 1):
        h_l = outputs.hidden_states[l][0, -1].detach().cpu().float().numpy()
        h_l1 = outputs.hidden_states[l-1][0, -1].detach().cpu().float().numpy()
        
        # Alpha
        alpha = np.dot(h_l, h_l1) / (np.dot(h_l1, h_l1) + 1e-10)
        
        # Beta
        delta = h_l - h_l1
        delta_norm = np.linalg.norm(delta)
        h_l1_norm = np.linalg.norm(h_l1)
        beta = delta_norm / (h_l1_norm + 1e-10)
        
        # 频谱
        h_coeffs = U_wu[:, :k_wu].T @ h_l
        abs_coeffs = np.abs(h_coeffs)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        ratio_50 = np.sum(abs_coeffs[:50]**2) / total_energy
        func_energy = np.sum(abs_coeffs[:5]**2) / total_energy
        
        layer_alphas.append(alpha)
        layer_betas.append(beta)
        layer_ratio_50.append(ratio_50)
        layer_func_energy.append(func_energy)
        layer_delta_norms.append(delta_norm)
    
    del outputs
    torch.cuda.empty_cache()
    
    # 跨多个文本验证(用5个文本取平均)
    multi_text_alphas = []
    for text in TEST_TEXTS[:5]:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        text_alphas = []
        for l in range(1, n_layers + 1):
            h_l = outputs.hidden_states[l][0, -1].detach().cpu().float().numpy()
            h_l1 = outputs.hidden_states[l-1][0, -1].detach().cpu().float().numpy()
            alpha = np.dot(h_l, h_l1) / (np.dot(h_l1, h_l1) + 1e-10)
            text_alphas.append(alpha)
        multi_text_alphas.append(text_alphas)
        
        del outputs
        torch.cuda.empty_cache()
    
    multi_text_alphas = np.array(multi_text_alphas)  # [n_texts, n_layers]
    mean_alphas = np.mean(multi_text_alphas, axis=0)
    std_alphas = np.std(multi_text_alphas, axis=0)
    
    # 分段统计: 浅层/中层/深层
    n_third = n_layers // 3
    shallow_alpha = np.mean(mean_alphas[:n_third])
    mid_alpha = np.mean(mean_alphas[n_third:2*n_third])
    deep_alpha = np.mean(mean_alphas[2*n_third:])
    
    print(f"\n--- Alpha层间分布(5文本平均) ---")
    print(f"  浅层(L0-L{n_third-1}): alpha={shallow_alpha:.4f}")
    print(f"  中层(L{n_third}-L{2*n_third-1}): alpha={mid_alpha:.4f}")
    print(f"  深层(L{2*n_third}-L{n_layers-1}): alpha={deep_alpha:.4f}")
    print(f"  全层均值: alpha={np.mean(mean_alphas):.4f}, 标准差={np.mean(std_alphas):.4f}")
    
    # Alpha与beta的层间关系
    r_alpha_beta_layer, _ = spearmanr(layer_alphas, layer_betas)
    print(f"\n--- Alpha vs Beta层间相关 ---")
    print(f"  Spearman r={r_alpha_beta_layer:.3f}")
    
    # Alpha与频谱的层间关系
    r_alpha_ratio, _ = spearmanr(layer_alphas, layer_ratio_50)
    r_alpha_func, _ = spearmanr(layer_alphas, layer_func_energy)
    print(f"  alpha vs ratio(50): r={r_alpha_ratio:.3f}")
    print(f"  alpha vs func_energy: r={r_alpha_func:.3f}")
    
    # Alpha的变化趋势
    alpha_trend = np.polyfit(range(len(mean_alphas)), mean_alphas, 1)[0]
    print(f"\n--- Alpha趋势 ---")
    print(f"  线性趋势: {alpha_trend:.6f}/层 ({'递增' if alpha_trend > 0 else '递减'})")
    
    # 每层的alpha详细打印(选择性)
    print(f"\n--- 逐层Alpha(选择性) ---")
    sample_layers = [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    for l in sample_layers:
        if l < len(mean_alphas):
            print(f"  L{l}: alpha={mean_alphas[l]:.4f}±{std_alphas[l]:.4f}, beta={layer_betas[l]:.4f}, ratio50={layer_ratio_50[l]:.4f}")
    
    result = {
        "model": model_name, "n_layers": n_layers,
        "shallow_alpha": float(shallow_alpha),
        "mid_alpha": float(mid_alpha),
        "deep_alpha": float(deep_alpha),
        "mean_alpha": float(np.mean(mean_alphas)),
        "alpha_trend": float(alpha_trend),
        "r_alpha_beta_layer": float(r_alpha_beta_layer),
        "r_alpha_ratio_layer": float(r_alpha_ratio),
        "r_alpha_func_layer": float(r_alpha_func),
        "layer_alphas_mean": [float(x) for x in mean_alphas],
        "layer_alphas_std": [float(x) for x in std_alphas],
    }
    return result


# ============== P569: 语义方向(方向5-20)解码 ==============
def experiment_p569(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P569: 语义方向(方向5-20)解码
    方向0-4是标点/格式, 方向5-20编码什么?
    """
    print(f"\n{'='*60}")
    print(f"P569: 语义方向(5-20)解码 — {model_name}")
    print(f"{'='*60}")
    
    k_wu = min(200, U_wu.shape[1])
    n_vocab = W_U.shape[0]
    
    # 定义词性/语义类别
    categories = {
        "punctuation": set([",", ".", "!", "?", ";", ":", "-", "(", ")", "\"", "'"]),
        "function": set(["the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
                        "do", "does", "will", "would", "can", "could", "to", "of", "in", "for",
                        "on", "with", "at", "by", "and", "but", "or", "not", "that", "this"]),
        "animal": set(["cat", "dog", "bird", "fish", "horse", "elephant", "tiger", "lion", "bear", "whale"]),
        "color": set(["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"]),
        "size": set(["big", "small", "large", "tiny", "huge", "little", "massive", "enormous", "vast", "minute"]),
        "emotion": set(["happy", "sad", "angry", "afraid", "love", "hate", "hope", "fear", "joy", "grief"]),
        "action": set(["run", "walk", "eat", "drink", "sleep", "think", "speak", "write", "read", "build"]),
        "abstract": set(["time", "space", "truth", "beauty", "justice", "freedom", "power", "knowledge", "wisdom", "courage"]),
    }
    
    # 对每个方向(0-20)分析语义内容
    direction_analysis = {}
    
    for d in range(21):
        direction = U_wu[:, d]
        projections = W_U @ direction
        
        # Top-20 正负tokens
        top_pos_ids = np.argsort(projections)[::-1][:20]
        top_neg_ids = np.argsort(projections)[:20]
        top_pos = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_pos_ids]
        top_neg = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_neg_ids]
        
        # 类别分布: 在Top-100中各类词的占比
        top100_ids = np.argsort(np.abs(projections))[::-1][:100]
        cat_counts = defaultdict(int)
        total_categorized = 0
        
        for tid in top100_ids:
            try:
                tok_str = tokenizer.decode([tid]).strip().lower()
                for cat_name, cat_set in categories.items():
                    if tok_str in cat_set:
                        cat_counts[cat_name] += 1
                        total_categorized += 1
                        break
            except:
                pass
        
        # 奇异值权重
        s_weight = float(S_wu[d]) if d < len(S_wu) else 0
        
        direction_analysis[d] = {
            "top_pos": top_pos[:5],
            "top_neg": top_neg[:5],
            "category_counts": dict(cat_counts),
            "s_weight": s_weight,
        }
        
        # 打印
        pos_str = ", ".join([f"{t}({v:.2f})" for t, v in top_pos[:5]])
        neg_str = ", ".join([f"{t}({v:.2f})" for t, v in top_neg[:5]])
        cat_str = ", ".join([f"{k}:{v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]) if v > 0])
        print(f"  方向{d} (S={s_weight:.1f}): Top+=[{pos_str}]")
        print(f"         Top-=[{neg_str}]")
        if cat_str:
            print(f"         类别: {cat_str}")
    
    # 方向5-20的h投影随层演化(用一条文本)
    print(f"\n--- 方向5-20的h投影随层演化 ---")
    text = TEST_TEXTS[0]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    info = get_model_info(model, model_name)
    sample_layers = [0, info.n_layers//4, info.n_layers//2, 3*info.n_layers//4, info.n_layers-1]
    
    for d in [5, 8, 10, 15, 20]:
        if d >= U_wu.shape[1]:
            continue
        print(f"  方向{d}:", end="")
        for li in sample_layers:
            h = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
            proj = np.dot(U_wu[:, d], h)
            print(f" L{li}={proj:.2f}", end="")
        print()
    
    del outputs
    torch.cuda.empty_cache()
    
    # 方向0-20的能量占比
    print(f"\n--- 方向能量分布 ---")
    total_s = np.sum(S_wu[:21])
    for d in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
        if d < len(S_wu):
            print(f"  方向{d}: S={S_wu[d]:.1f}, 占比={S_wu[d]/total_s*100:.1f}%")
    
    result = {
        "model": model_name,
        "n_directions": 21,
        "direction_analysis": {
            str(d): {
                "top_pos": direction_analysis[d]["top_pos"],
                "top_neg": direction_analysis[d]["top_neg"],
                "category_counts": direction_analysis[d]["category_counts"],
                "s_weight": direction_analysis[d]["s_weight"],
            }
            for d in range(21)
        }
    }
    return result


# ============== P570: 方向贡献耦合分析 ==============
def experiment_p570(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """
    P570: 方向贡献耦合分析
    为什么DS7B方差比=2.53? 耦合的数学形式是什么?
    """
    print(f"\n{'='*60}")
    print(f"P570: 方向贡献耦合分析 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    k_wu = min(200, U_wu.shape[1])
    
    # 对大量文本测量方向贡献的耦合
    all_direction_contributions = []  # [n_samples, k_wu, n_vocab] 太大
    # 改为: 收集每个方向的top-5贡献logit值的汇总
    
    n_texts = 50
    direction_logit_contribs_all = []  # [n_samples, k_wu]
    
    for text in TEST_TEXTS[:n_texts]:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        
        # h的频谱系数
        h_coeffs = U_wu[:, :k_wu].T @ h_last
        
        # 每个方向对logits的贡献方向
        # logits_k = h_coeffs[k] * (W_U @ U_wu[:, k])
        # 我们关心的是贡献的相对大小和方向间的相关性
        
        # 计算前50个方向的logit贡献(简化: 用贡献范数)
        direction_contribution_norms = []
        for k in range(50):
            direction_in_logit_space = W_U @ U_wu[:, k]  # [n_vocab]
            contribution = h_coeffs[k] * direction_in_logit_space
            norm = np.linalg.norm(contribution)
            direction_contribution_norms.append(norm)
        
        direction_logit_contribs_all.append(direction_contribution_norms)
        
        del outputs
        torch.cuda.empty_cache()
    
    direction_logit_contribs_all = np.array(direction_logit_contribs_all)  # [n_texts, 50]
    
    # 1. 方向贡献的协方差矩阵
    print(f"\n--- 方向贡献协方差分析(n={n_texts}) ---")
    cov_matrix = np.cov(direction_logit_contribs_all.T)  # [50, 50]
    
    # 对角线 vs 非对角线
    diag_mean = np.mean(np.diag(cov_matrix))
    off_diag = cov_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_mean = np.mean(np.abs(off_diag))
    
    print(f"  对角线均值: {diag_mean:.4f}")
    print(f"  非对角线|均值|: {off_diag_mean:.4f}")
    print(f"  对角/非对角比: {diag_mean/(off_diag_mean+1e-10):.2f}")
    
    # 2. 耦合矩阵的特征值分析
    eigvals = np.linalg.eigvalsh(cov_matrix)
    eigvals = eigvals[::-1]  # 降序
    
    # 参与率
    total_var = np.sum(eigvals)
    PR = (np.sum(eigvals)**2) / (np.sum(eigvals**2) + 1e-10)
    print(f"\n--- 耦合矩阵参与率 ---")
    print(f"  PR = {PR:.3f} (50维中有效维度)")
    print(f"  Top-5特征值占比: {np.sum(eigvals[:5])/total_var:.3f}")
    print(f"  Top-10特征值占比: {np.sum(eigvals[:10])/total_var:.3f}")
    
    # 3. 耦合强度随方向距离的衰减
    print(f"\n--- 耦合随方向距离衰减 ---")
    for dist in [1, 2, 5, 10, 20]:
        # 距离为dist的协方差均值
        covs_at_dist = []
        for i in range(50 - dist):
            covs_at_dist.append(cov_matrix[i, i + dist])
        mean_cov = np.mean(np.abs(covs_at_dist))
        print(f"  距离{dist}: |协方差|均值={mean_cov:.4f}")
    
    # 4. 相关矩阵分析
    print(f"\n--- 相关矩阵 ---")
    # 用皮尔逊相关替代协方差
    corr_matrix = np.corrcoef(direction_logit_contribs_all.T)
    np.fill_diagonal(corr_matrix, 0)
    
    # 最大正相关和负相关
    max_pos_corr = np.max(corr_matrix)
    max_neg_corr = np.min(corr_matrix)
    mean_abs_corr = np.mean(np.abs(corr_matrix))
    
    print(f"  最大正相关: {max_pos_corr:.4f}")
    print(f"  最大负相关: {max_neg_corr:.4f}")
    print(f"  |相关|均值: {mean_abs_corr:.4f}")
    
    # 找出强耦合的方向对
    strong_couplings = []
    for i in range(50):
        for j in range(i+1, 50):
            if abs(corr_matrix[i, j]) > 0.3:
                strong_couplings.append((i, j, float(corr_matrix[i, j])))
    
    strong_couplings.sort(key=lambda x: -abs(x[2]))
    print(f"  强耦合方向对(|r|>0.3): {len(strong_couplings)}个")
    for i, j, r in strong_couplings[:5]:
        print(f"    方向{i} <-> 方向{j}: r={r:.3f}")
    
    # 5. 方差比验证
    print(f"\n--- 方差比验证 ---")
    # 对单文本详细分析
    text = TEST_TEXTS[0]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    h_coeffs = U_wu[:, :k_wu].T @ h_last
    full_logits = W_U @ h_last
    
    # 各方向贡献
    logit_var = np.var(full_logits)
    dir_vars = []
    for k in range(50):
        direction_in_logit_space = W_U @ U_wu[:, k]
        contribution = h_coeffs[k] * direction_in_logit_space
        dir_vars.append(np.var(contribution))
    
    sum_dir_vars = np.sum(dir_vars)
    var_ratio = sum_dir_vars / (logit_var + 1e-10)
    
    print(f"  logits方差: {logit_var:.2f}")
    print(f"  Σ方向贡献方差: {sum_dir_vars:.2f}")
    print(f"  方差比: {var_ratio:.4f} (>1说明方向贡献正相关)")
    
    # 理论预期: 如果独立, var_ratio ≈ 1
    # 实际: DS7B ≈ 2.53, 说明2×corr项 ≈ 1.53
    # corr贡献 = (var_ratio - 1) * logit_var
    corr_contribution = (var_ratio - 1) * logit_var
    print(f"  相关项贡献: {corr_contribution:.2f} (= {(var_ratio-1)*100:.1f}% 的logit方差)")
    
    del outputs
    torch.cuda.empty_cache()
    
    result = {
        "model": model_name, "n_texts": n_texts,
        "PR_coupling": float(PR),
        "diag_mean": float(diag_mean),
        "off_diag_mean": float(off_diag_mean),
        "diag_offdiag_ratio": float(diag_mean / (off_diag_mean + 1e-10)),
        "max_pos_corr": float(max_pos_corr),
        "max_neg_corr": float(max_neg_corr),
        "mean_abs_corr": float(mean_abs_corr),
        "n_strong_couplings": len(strong_couplings),
        "var_ratio": float(var_ratio),
        "corr_contribution_pct": float((var_ratio - 1) * 100),
        "top5_eigval_ratio": float(np.sum(eigvals[:5]) / total_var),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["p567", "p568", "p569", "p570"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    
    if args.experiment == "p567":
        result = experiment_p567(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p568":
        result = experiment_p568(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p569":
        result = experiment_p569(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p570":
        result = experiment_p570(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = "results/phase_cxxxi"
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
