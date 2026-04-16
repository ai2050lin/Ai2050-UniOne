"""
Phase CXXXVIII: 频谱→logit_gap的根本突破 (P601-P606)
=====================================================
核心洞察: gap和offset的线性耦合(r>-0.89), prob ≈ sigmoid(effective_gap)
  P601: gap-offset耦合的解析 — offset ≈ -a*gap + b, a和b由什么决定?
  P602: 有效gap方程 — effective_gap = (1+a)*gap, a从logit分布尾部估计
  P603: 频谱→effective_gap的因果链 — h频谱参数 + h L2范数 → effective_gap
  P604: h的L2范数→logit_max的因果链 — logit_max ≈ ||W_U||_F * ||h||_2 / sqrt(d_vocab)
  P605: 统一语言编码方程v4 — prob = sigmoid(effective_gap), effective_gap由h特征预测
  P606: 频谱力学的最终评估 — P1-P605所有频谱→语言行为路径的最佳路径

大规模测试: 210文本 x 3模型
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights

import torch

# ===================== 210 test texts =====================
TEST_TEXTS = [
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
    "The ancient temple stood silently on the mountain, watching centuries pass by",
    "The economic crisis led to widespread unemployment and social unrest",
    "The orchestra performed Beethoven's ninth symphony with remarkable precision",
    "Education is the most powerful weapon which you can use to change the world",
    "The city skyline glittered with lights as night fell over the metropolis",
    "Cultural heritage preservation requires careful planning and community engagement",
    "The philosopher argued that knowledge is acquired through experience and reflection",
    "The museum exhibition showcased artifacts from ancient civilizations around the world",
    "Historical records provide valuable insights into the lives of past generations",
    "The diplomatic negotiations resulted in a mutually beneficial agreement between nations",
    "Social media has transformed the way people communicate and share information globally",
    "The artist captured the essence of human emotion through abstract expressionism",
    "Philosophy encourages critical thinking and deep reflection on fundamental questions",
    "The architecture of the cathedral reflected the cultural values of medieval society",
    "Democratic institutions depend on active citizen participation and informed debate",
    "The novel explored themes of identity, belonging, and the search for meaning",
    "International cooperation is essential for addressing global challenges effectively",
    "The documentary examined the impact of globalization on local communities",
    "Music has the power to transcend cultural boundaries and unite people across the world",
    "The library contained thousands of rare manuscripts dating back to the medieval period",
    "She opened the door and stepped into the warm, inviting living room",
    "The aroma of freshly baked bread filled the entire house with warmth",
    "He carefully wrapped the fragile gift in colorful paper and tied it with a ribbon",
    "The children laughed and played in the park until the sun began to set",
    "After a long day at work, she enjoyed sitting by the fireplace with a good book",
    "The morning coffee was perfect, strong and aromatic, just the way he liked it",
    "They walked hand in hand along the beach, watching the waves roll in",
    "The old photograph brought back memories of summers spent at the grandmother's house",
    "The dinner table was set with fine china and crystal glasses for the special occasion",
    "She picked up her favorite novel and lost herself in the story for hours",
    "The garden was full of colorful flowers that attracted butterflies and bees",
    "He packed his suitcase carefully, making sure not to forget anything important",
    "The cat curled up on the soft cushion and purred contentedly in the sunlight",
    "They gathered around the campfire, roasting marshmallows and telling stories",
    "The morning jog through the park left her feeling energized and refreshed",
    "The recipe called for fresh herbs, garlic, and a splash of lemon juice",
    "She arranged the flowers in a beautiful vase and placed it on the dining table",
    "The rain stopped and a beautiful rainbow appeared across the sky",
    "He grabbed his umbrella before heading out into the drizzly afternoon",
    "The puppy chased its tail in circles, much to the amusement of everyone watching",
    "The mathematical proof required careful reasoning and attention to logical detail",
    "Statistical analysis revealed a significant correlation between the two variables",
    "The scientific method involves systematic observation, measurement, and experimentation",
    "Quantum mechanics describes the behavior of particles at the subatomic level",
    "The theory of evolution by natural selection explains the diversity of life on Earth",
    "Economic models attempt to predict market behavior based on historical data patterns",
    "The psychological experiment demonstrated the powerful effect of social conformity",
    "Anthropological research has documented remarkable cultural diversity across societies",
    "The chemistry experiment produced unexpected results that challenged existing theories",
    "Linguistic analysis reveals patterns in how languages evolve and change over time",
    "The astronomical observations confirmed the existence of the previously hypothetical planet",
    "Mathematical models of population dynamics help predict species extinction risks",
    "The archaeological excavation uncovered evidence of a previously unknown civilization",
    "Computational biology uses algorithms to analyze complex biological data sets",
    "The philosophical argument rested on the distinction between knowledge and belief",
    "Sociological research examines how social structures influence individual behavior",
    "The geological survey mapped the distribution of mineral resources in the region",
    "Ecological studies demonstrate the interconnectedness of all living organisms",
    "The engineering design process involves iterative testing and refinement of prototypes",
    "Historical analysis reveals recurring patterns in the rise and fall of civilizations",
    "The chef prepared an exquisite meal using locally sourced organic ingredients",
    "The basketball team celebrated their championship victory with a parade through the city",
    "She dialed the phone number and waited anxiously for someone to answer",
    "The train arrived at the station precisely on schedule despite the winter storm",
    "The detective carefully examined the evidence looking for any overlooked clues",
    "The concert was sold out weeks in advance due to the band's enormous popularity",
    "The weather forecast predicted heavy rain and strong winds for the weekend",
    "The entrepreneur launched a startup that quickly attracted significant investment",
    "The judge delivered a fair and balanced verdict after carefully considering all evidence",
    "The astronaut floated weightlessly in the space station observing the Earth below",
    "The baker decorated the wedding cake with intricate sugar flowers and delicate frosting",
    "The swimmer broke the world record by a fraction of a second in the final lap",
    "The teacher encouraged her students to think critically and ask thoughtful questions",
    "The airplane touched down smoothly on the runway despite the challenging crosswind",
    "The journalist interviewed several eyewitnesses to piece together an accurate report",
    "The mechanic diagnosed the engine problem and replaced the faulty component",
    "The volunteer organized a food drive to help families in need during the holidays",
    "The athlete trained rigorously for months in preparation for the Olympic competition",
    "The musician composed a beautiful melody that moved the audience to tears",
    "The photographer captured stunning images of the northern lights dancing across the sky",
    "The programmer debugged the software code to eliminate the critical security vulnerability",
    "The nurse provided compassionate care to patients throughout the long night shift",
    "The pilot navigated through the turbulent weather with skill and confidence",
    "The student completed the challenging assignment ahead of schedule and earned top marks",
    "The firefighter bravely entered the burning building to rescue the trapped residents",
    "The architect designed a sustainable building that minimized environmental impact",
    "The veterinarian treated the injured bird and nursed it back to health",
    "The translator converted the ancient text into modern language while preserving its meaning",
    "The electrician installed new wiring throughout the old house to meet safety standards",
    "The florist created an elegant arrangement using roses and lilies for the ceremony",
    "The librarian helped the young reader find books that matched his interests perfectly",
    "The painter transformed the blank canvas into a vibrant landscape of rolling hills",
    "The carpenter crafted a beautiful dining table from reclaimed oak wood",
    "The barber gave him a neat haircut that made him look professional and well-groomed",
    "The tailor altered the suit jacket to fit perfectly for the important interview",
    "The plumber fixed the leaking pipe and restored water pressure throughout the building",
    "The conductor led the orchestra through a powerful performance of the symphony",
    "The DJ mixed the tracks seamlessly keeping the dance floor packed all night",
    "The sculptor chiseled away at the marble block gradually revealing the hidden figure",
    "The potter shaped the clay on the wheel creating a graceful ceramic vase",
    "The author wrote a compelling mystery novel that kept readers guessing until the end",
    "The poet captured the fleeting beauty of autumn in a few carefully chosen words",
    "The comedian delivered a hilarious performance that had the audience roaring with laughter",
    "The dancer moved with grace and precision across the stage captivating the audience",
    "The actor delivered a powerful monologue that left the theater in complete silence",
    "The director filmed the scene multiple times to capture the perfect take",
    "The editor revised the manuscript extensively to improve clarity and flow",
    "The publisher released the book simultaneously in digital and print formats",
    "The critic reviewed the film offering a thoughtful analysis of its themes and technique",
    "The analyst examined the financial data to identify emerging market trends",
    "The consultant recommended strategic changes to improve organizational efficiency",
    "The manager delegated tasks effectively ensuring the project stayed on schedule",
    "The supervisor evaluated employee performance using objective criteria and feedback",
    "The coordinator organized the conference logistics down to the smallest detail",
    "The recruiter interviewed candidates seeking individuals with both skills and experience",
    "The trainer developed a customized fitness program tailored to individual goals",
    "The coach motivated the team with an inspiring speech before the championship game",
    "The referee made a controversial call that sparked debate among the spectators",
    "The umpire carefully reviewed the replay before making the final decision",
    "The commentator analyzed the match providing expert insight into the strategy",
    "The announcer introduced the next speaker with enthusiasm and energy",
    "The host welcomed guests warmly making everyone feel at home at the party",
    "The moderator facilitated a productive discussion among the panel of experts",
    "The negotiator found common ground between the two opposing parties",
    "The mediator helped the couple resolve their differences through constructive dialogue",
    "The advocate argued passionately for policy changes to protect the environment",
    "The activist organized a peaceful demonstration to raise awareness about the issue",
    "The campaigner traveled across the country building support for the reform movement",
    "The lobbyist presented compelling evidence to influence the legislative committee",
    "The diplomat represented her country with distinction at the international summit",
    "The ambassador fostered strong bilateral relations between the two nations",
    "The statesman delivered a visionary speech outlining a path toward lasting peace",
    "The senator proposed legislation to address the growing inequality in education funding",
    "The mayor implemented innovative policies to improve public transportation in the city",
    "The governor declared a state of emergency in response to the natural disaster",
    "The president addressed the nation outlining the comprehensive recovery plan",
    "The minister announced new funding initiatives to support small business development",
    "The official confirmed that the investigation would continue until all leads were exhausted",
    "The bureaucrat processed the application efficiently following established procedures",
    "The administrator managed the budget carefully ensuring resources were allocated fairly",
    "The clerk maintained accurate records of all transactions conducted during the fiscal year",
    "The auditor verified the financial statements confirming their accuracy and compliance",
    "The accountant prepared the tax returns ensuring all deductions were properly documented",
    "The cashier processed the transaction quickly and handed the customer a receipt",
    "The teller assisted the bank customer with opening a new savings account",
    "The banker advised the client on investment strategies for long-term growth",
    "The broker facilitated the real estate transaction between the buyer and seller",
    "The dealer offered a fair price for the antique furniture collection",
    "The merchant imported goods from overseas and distributed them to local retailers",
    "The retailer expanded the product line to meet changing consumer preferences",
    "The vendor set up the market stall early in the morning to attract customers",
    "The trader analyzed market conditions before making the significant investment decision",
    "The investor diversified the portfolio to minimize risk while maximizing returns",
    "The shareholder voted in favor of the merger at the annual general meeting",
    "The partner reviewed the contract terms carefully before signing the agreement",
    "The client requested additional modifications to the original project specification",
    "The customer provided positive feedback about the exceptional service received",
    "The patron supported the local theater by attending performances regularly",
    "The enthusiast joined the club to connect with others who shared the same passion",
    "The hobbyist spent weekends restoring the classic automobile to its original condition",
    "The amateur participated in the competition and surprisingly won first prize",
    "The professional delivered a polished performance that impressed the judges",
    "The expert testified before the committee sharing specialized knowledge on the subject",
    "The specialist diagnosed the rare condition based on subtle clinical symptoms",
    "The veteran shared stories of service and sacrifice with the younger generation",
    "The novice struggled at first but gradually improved with practice and determination",
    "The beginner followed the tutorial step by step to learn the basic technique",
    "The apprentice worked alongside the master craftsman to develop essential skills",
    "The intern gained valuable experience during the summer placement at the company",
    "The student studied diligently for the examination reviewing all course materials",
    "The scholar published a research paper that contributed new insights to the field",
    "The professor delivered a thought-provoking lecture that challenged conventional wisdom",
    "The researcher conducted a comprehensive literature review before designing the study",
    "The scientist formulated a novel hypothesis based on preliminary experimental results",
    "The engineer developed an innovative solution to the complex technical problem",
    "The inventor filed a patent application for the groundbreaking new device",
    "The designer created a user-friendly interface that simplified complex operations",
    "The planner organized the project timeline to ensure all milestones were met on schedule",
    "The strategist analyzed competitive intelligence to identify market opportunities",
    "The analyst reviewed the quarterly earnings report and adjusted the forecast accordingly",
    "The advisor recommended a balanced approach to managing the investment portfolio",
    "The counselor helped the student develop effective coping strategies for managing stress",
]


def compute_svd_W_U(W_U, k=100):
    W_U_T = W_U.T.astype(np.float32)
    k = min(k, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    k = max(k, 1)
    U_wut, s_wut, Vt_wut = svds(W_U_T, k=k)
    idx = np.argsort(s_wut)[::-1]
    return U_wut[:, idx], s_wut[idx], Vt_wut[idx, :]


def identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=100):
    W_U_proj = U_wut.T @ W_U.T
    W_U_energy = W_U_proj ** 2
    punct_tokens = set()
    for tid in range(W_U.shape[0]):
        try:
            token_str = tokenizer.decode([tid])
            stripped = token_str.strip()
            if stripped and not stripped[0].isalpha() and not stripped[0].isdigit():
                punct_tokens.add(tid)
        except:
            pass
    punct_fraction = np.zeros(k)
    total_energy = np.sum(W_U_energy, axis=1) + 1e-10
    for tid in punct_tokens:
        punct_fraction += W_U_energy[:, tid]
    punct_fraction /= total_energy
    return punct_fraction, punct_tokens


def get_hidden_and_logits(model, tokenizer, device, text):
    """获取最后一层隐藏状态和logits"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1].float().cpu().numpy()
        h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
    return h, logits


def compute_h_spectral_features(h, U_wut, s_wut, K=100):
    """计算h在W_U SVD基上的频谱特征"""
    c_k = U_wut[:, :K].T @ h  # [K]
    energy_k = c_k ** 2
    total_energy = np.sum(energy_k) + 1e-10

    # ratio50: 前50个方向能量占比
    ratio50 = float(np.sum(energy_k[:50]) / total_energy)
    # ratio10: 前10个方向能量占比
    ratio10 = float(np.sum(energy_k[:10]) / total_energy)
    # ratio90: 前90个方向能量占比
    ratio90 = float(np.sum(energy_k[:90]) / total_energy)

    # 幂律拟合: log(energy_k) ≈ alpha * log(s_k) + beta
    valid = energy_k > 1e-20
    if np.sum(valid) > 5:
        log_e = np.log(energy_k[valid] + 1e-30)
        log_s = np.log(s_wut[:K][valid] + 1e-30)
        alpha = float(np.polyfit(log_s, log_e, 1)[0])
    else:
        alpha = 0.0

    # h的L2范数
    h_norm = float(np.linalg.norm(h))

    # 加权alpha: 用s_k^2加权
    if np.sum(valid) > 5:
        weights = s_wut[:K][valid] ** 2
        weights /= weights.sum()
        weighted_alpha = float(np.average(
            np.log(energy_k[valid] + 1e-30) / (np.log(s_wut[:K][valid] + 1e-30) + 1e-30),
            weights=weights
        ))
    else:
        weighted_alpha = 0.0

    return {
        "ratio50": ratio50, "ratio10": ratio10, "ratio90": ratio90,
        "alpha": alpha, "h_norm": h_norm, "weighted_alpha": weighted_alpha,
        "total_energy": float(total_energy),
        "c_k": c_k, "energy_k": energy_k,
    }


def run_p601(model, tokenizer, device, model_info, texts, output_dir):
    """P601: gap-offset耦合的解析 — offset ≈ -a*gap + b, a和b由什么决定?"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P601: gap-offset coupling analysis -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p601", "n_texts": len(texts)}

    all_gaps = []
    all_offsets = []
    all_logit_maxs = []
    all_logit_stds = []
    all_logit_kurtosis = []
    all_logit_tail_sums = []
    all_probs = []
    all_h_norms = []
    all_ratio50s = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])

        # offset = log(sum_{i not top1} exp(logit_i - logit_1))
        logit_shifted = logits - logits[top1_idx]
        offset = float(np.log(np.sum(np.exp(logit_shifted)) - 1.0 + 1e-30))

        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))

        # logit分布统计量
        logit_max = float(logits[top1_idx])
        logit_std = float(np.std(logits))
        # 峰度(kurtosis)
        logit_mean = float(np.mean(logits))
        logit_kurt = float(np.mean(((logits - logit_mean) / (logit_std + 1e-10)) ** 4))
        # 尾部和: top10以外的logit和
        sorted_logits = np.sort(logits)[::-1]
        logit_tail_sum = float(np.sum(sorted_logits[10:]))

        # h的频谱特征
        spec = compute_h_spectral_features(h, U_wut, s_wut, K=K)

        all_gaps.append(logit_gap)
        all_offsets.append(offset)
        all_logit_maxs.append(logit_max)
        all_logit_stds.append(logit_std)
        all_logit_kurtosis.append(logit_kurt)
        all_logit_tail_sums.append(logit_tail_sum)
        all_probs.append(top1_prob)
        all_h_norms.append(spec["h_norm"])
        all_ratio50s.append(spec["ratio50"])

    all_gaps = np.array(all_gaps)
    all_offsets = np.array(all_offsets)
    all_logit_maxs = np.array(all_logit_maxs)
    all_logit_stds = np.array(all_logit_stds)
    all_logit_kurtosis = np.array(all_logit_kurtosis)
    all_logit_tail_sums = np.array(all_logit_tail_sums)
    all_probs = np.array(all_probs)
    all_h_norms = np.array(all_h_norms)
    all_ratio50s = np.array(all_ratio50s)

    # gap-offset线性回归: offset = -a * gap + b
    from numpy.linalg import lstsq
    X_ab = np.column_stack([all_gaps, np.ones(len(texts))])
    coeffs_ab, _, _, _ = lstsq(X_ab, all_offsets, rcond=None)
    a_coeff, b_coeff = coeffs_ab
    pred_offset = a_coeff * all_gaps + b_coeff
    r_gap_offset = float(np.corrcoef(all_gaps, all_offsets)[0, 1])

    # 什么决定a和b? — 分析logit分布统计量与offset的关系
    r_logit_max_offset, _ = pearsonr(all_logit_maxs, all_offsets)
    r_logit_std_offset, _ = pearsonr(all_logit_stds, all_offsets)
    r_logit_kurt_offset, _ = pearsonr(all_logit_kurtosis, all_offsets)
    r_logit_tail_offset, _ = pearsonr(all_logit_tail_sums, all_offsets)
    r_h_norm_offset, _ = pearsonr(all_h_norms, all_offsets)
    r_ratio50_offset, _ = pearsonr(all_ratio50s, all_offsets)

    # 多回归: offset = f(gap, logit_max, logit_std, logit_kurt, logit_tail, h_norm, ratio50)
    X_multi = np.column_stack([all_gaps, all_logit_maxs, all_logit_stds, all_logit_kurtosis,
                                all_logit_tail_sums, all_h_norms, all_ratio50s, np.ones(len(texts))])
    coeffs_multi, _, _, _ = lstsq(X_multi, all_offsets, rcond=None)
    pred_offset_multi = X_multi @ coeffs_multi
    r_offset_multi = float(np.corrcoef(pred_offset_multi, all_offsets)[0, 1])

    # gap-offset残差分析
    residual_offset = all_offsets - pred_offset
    r_residual_prob, _ = pearsonr(residual_offset, all_probs)

    # offset的组成部分: offset ≈ log(vocab_size) + log(mean(exp(logit_i - logit_max)))
    # 简化: offset ≈ -logit_max + log(sum_exp_logits)
    sum_exp_logits = np.array([float(np.sum(np.exp(logits - np.max(logits)))) for logits in
                                [get_hidden_and_logits(model, tokenizer, device, t)[1] for t in texts[:30]]])
    # 这太慢，用近似

    results["gap_offset_coupling"] = {
        "a_coefficient": float(a_coeff),
        "b_coefficient": float(b_coeff),
        "r_gap_offset": float(r_gap_offset),
    }
    results["offset_predictors"] = {
        "logit_max_to_offset_r": float(r_logit_max_offset),
        "logit_std_to_offset_r": float(r_logit_std_offset),
        "logit_kurtosis_to_offset_r": float(r_logit_kurt_offset),
        "logit_tail_sum_to_offset_r": float(r_logit_tail_offset),
        "h_norm_to_offset_r": float(r_h_norm_offset),
        "ratio50_to_offset_r": float(r_ratio50_offset),
        "multi_regression_r": float(r_offset_multi),
    }
    results["residual_analysis"] = {
        "offset_residual_to_prob_r": float(r_residual_prob),
    }
    results["statistics"] = {
        "gap_mean": float(np.mean(all_gaps)),
        "gap_std": float(np.std(all_gaps)),
        "offset_mean": float(np.mean(all_offsets)),
        "offset_std": float(np.std(all_offsets)),
        "logit_max_mean": float(np.mean(all_logit_maxs)),
        "logit_std_mean": float(np.mean(all_logit_stds)),
        "logit_kurtosis_mean": float(np.mean(all_logit_kurtosis)),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p601.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P601 Results for {model_name}:")
    print(f"  gap-offset coupling: offset = {a_coeff:.4f} * gap + {b_coeff:.4f}")
    print(f"  r(gap, offset) = {r_gap_offset:.4f}")
    print(f"  logit_max->offset r = {r_logit_max_offset:.4f}")
    print(f"  logit_std->offset r = {r_logit_std_offset:.4f}")
    print(f"  logit_kurtosis->offset r = {r_logit_kurt_offset:.4f}")
    print(f"  logit_tail_sum->offset r = {r_logit_tail_offset:.4f}")
    print(f"  h_norm->offset r = {r_h_norm_offset:.4f}")
    print(f"  ratio50->offset r = {r_ratio50_offset:.4f}")
    print(f"  Multi regression offset r = {r_offset_multi:.4f}")
    print(f"  Offset residual->prob r = {r_residual_prob:.4f}")
    return results


def run_p602(model, tokenizer, device, model_info, texts, output_dir):
    """P602: 有效gap方程 — effective_gap = (1+a)*gap - b, prob ≈ sigmoid(effective_gap)"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P602: Effective gap equation -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p602", "n_texts": len(texts)}

    all_gaps = []
    all_offsets = []
    all_probs = []
    all_logit_maxs = []
    all_h_norms = []
    all_ratio50s = []
    all_alphas = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        logit_shifted = logits - logits[top1_idx]
        offset = float(np.log(np.sum(np.exp(logit_shifted)) - 1.0 + 1e-30))
        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))
        logit_max = float(logits[top1_idx])

        spec = compute_h_spectral_features(h, U_wut, s_wut, K=K)

        all_gaps.append(logit_gap)
        all_offsets.append(offset)
        all_probs.append(top1_prob)
        all_logit_maxs.append(logit_max)
        all_h_norms.append(spec["h_norm"])
        all_ratio50s.append(spec["ratio50"])
        all_alphas.append(spec["alpha"])

    all_gaps = np.array(all_gaps)
    all_offsets = np.array(all_offsets)
    all_probs = np.array(all_probs)
    all_logit_maxs = np.array(all_logit_maxs)
    all_h_norms = np.array(all_h_norms)
    all_ratio50s = np.array(all_ratio50s)
    all_alphas = np.array(all_alphas)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # 1. 原始gap->prob
    r_gap_prob, _ = pearsonr(all_gaps, all_probs)

    # 2. effective_gap = gap - offset (直接计算)
    effective_gap_v1 = all_gaps - all_offsets
    r_eff_v1_prob, _ = pearsonr(effective_gap_v1, all_probs)
    pred_v1 = sigmoid(effective_gap_v1)
    r_sigmoid_v1 = float(np.corrcoef(pred_v1, all_probs)[0, 1])

    # 3. effective_gap = (1+a)*gap - b (用回归的a, b)
    from numpy.linalg import lstsq
    X_ab = np.column_stack([all_gaps, np.ones(len(texts))])
    coeffs_ab, _, _, _ = lstsq(X_ab, all_offsets, rcond=None)
    a_coeff, b_coeff = coeffs_ab

    # prob ≈ sigmoid(gap - offset) = sigmoid(gap - (a*gap + b)) = sigmoid((1-a)*gap - b)
    effective_gap_v2 = (1 - a_coeff) * all_gaps - b_coeff
    pred_v2 = sigmoid(effective_gap_v2)
    r_sigmoid_v2 = float(np.corrcoef(pred_v2, all_probs)[0, 1])

    # 4. Simple sigmoid(gap - mean_offset)
    mean_offset = float(np.mean(all_offsets))
    pred_simple = sigmoid(all_gaps - mean_offset)
    r_simple = float(np.corrcoef(pred_simple, all_probs)[0, 1])

    # 5. 优化effective_gap: prob = sigmoid(w1*gap + w2*offset + b)
    def loss_opt(params):
        w1, w2, b = params
        z = w1 * all_gaps + w2 * all_offsets + b
        pred = sigmoid(z)
        r = np.corrcoef(pred, all_probs)[0, 1]
        return -r

    res_opt = minimize(loss_opt, [1.0, -1.0, 0.0], method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8})
    w1_opt, w2_opt, b_opt = res_opt.x
    z_opt = w1_opt * all_gaps + w2_opt * all_offsets + b_opt
    pred_opt = sigmoid(z_opt)
    r_opt = float(np.corrcoef(pred_opt, all_probs)[0, 1])

    # 6. 从h特征预测effective_gap (不用offset)
    # effective_gap ≈ w1*h_norm + w2*ratio50 + w3*alpha + b
    X_h = np.column_stack([all_h_norms, all_ratio50s, all_alphas, np.ones(len(texts))])
    coeffs_h, _, _, _ = lstsq(X_h, effective_gap_v1, rcond=None)
    pred_eff_from_h = X_h @ coeffs_h
    r_h_to_eff = float(np.corrcoef(pred_eff_from_h, effective_gap_v1)[0, 1])
    pred_from_h = sigmoid(pred_eff_from_h)
    r_h_to_prob = float(np.corrcoef(pred_from_h, all_probs)[0, 1])

    # 7. h特征->logit_max路径
    r_h_norm_to_logit_max, _ = pearsonr(all_h_norms, all_logit_maxs)
    r_ratio50_to_logit_max, _ = pearsonr(all_ratio50s, all_logit_maxs)

    results["baseline"] = {
        "gap_to_prob_r": float(r_gap_prob),
        "simple_sigmoid_r": float(r_simple),
    }
    results["effective_gap_v1"] = {
        "effective_gap_to_prob_r": float(r_eff_v1_prob),
        "sigmoid_r": float(r_sigmoid_v1),
    }
    results["effective_gap_v2"] = {
        "a_coefficient": float(a_coeff),
        "b_coefficient": float(b_coeff),
        "sigmoid_r": float(r_sigmoid_v2),
    }
    results["optimized"] = {
        "w1": float(w1_opt), "w2": float(w2_opt), "b": float(b_opt),
        "sigmoid_r": float(r_opt),
    }
    results["h_features_to_effective_gap"] = {
        "h_to_eff_r": float(r_h_to_eff),
        "h_to_prob_r": float(r_h_to_prob),
    }
    results["h_features_to_logit_max"] = {
        "h_norm_to_logit_max_r": float(r_h_norm_to_logit_max),
        "ratio50_to_logit_max_r": float(r_ratio50_to_logit_max),
    }
    results["mean_offset"] = float(mean_offset)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p602.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P602 Results for {model_name}:")
    print(f"  Baseline: gap->prob r={r_gap_prob:.4f}, simple sigmoid r={r_simple:.4f}")
    print(f"  Effective gap v1 (gap-offset)->prob r={r_eff_v1_prob:.4f}, sigmoid r={r_sigmoid_v1:.4f}")
    print(f"  Effective gap v2 ((1-a)*gap-b)->sigmoid r={r_sigmoid_v2:.4f}")
    print(f"  Optimized sigmoid r={r_opt:.4f} (w1={w1_opt:.4f}, w2={w2_opt:.4f})")
    print(f"  h features->effective_gap r={r_h_to_eff:.4f}, ->prob r={r_h_to_prob:.4f}")
    print(f"  h_norm->logit_max r={r_h_norm_to_logit_max:.4f}")
    print(f"  ratio50->logit_max r={r_ratio50_to_logit_max:.4f}")
    return results


def run_p603(model, tokenizer, device, model_info, texts, output_dir):
    """P603: 频谱→effective_gap的因果链 — h频谱参数 + h L2范数 → effective_gap → prob"""
    model_name = model_info.name
    W_U = get_W_U(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"\n{'='*60}")
    print(f"P603: Spectral->effective_gap causal chain -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    # W_U的Frobenius范数
    W_U_frobenius = float(np.linalg.norm(W_U))
    vocab_size = W_U.shape[0]

    results = {"model": model_name, "experiment": "p603", "n_texts": len(texts),
               "W_U_frobenius": W_U_frobenius, "vocab_size": vocab_size, "d_model": d_model}

    all_gaps = []
    all_offsets = []
    all_eff_gaps = []
    all_probs = []
    all_h_norms = []
    all_ratio10s = []
    all_ratio50s = []
    all_ratio90s = []
    all_alphas = []
    all_weighted_alphas = []
    all_total_energies = []
    all_logit_maxs = []
    all_logit_stds = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        logit_shifted = logits - logits[top1_idx]
        offset = float(np.log(np.sum(np.exp(logit_shifted)) - 1.0 + 1e-30))
        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))
        eff_gap = logit_gap - offset

        spec = compute_h_spectral_features(h, U_wut, s_wut, K=K)

        all_gaps.append(logit_gap)
        all_offsets.append(offset)
        all_eff_gaps.append(eff_gap)
        all_probs.append(top1_prob)
        all_h_norms.append(spec["h_norm"])
        all_ratio10s.append(spec["ratio10"])
        all_ratio50s.append(spec["ratio50"])
        all_ratio90s.append(spec["ratio90"])
        all_alphas.append(spec["alpha"])
        all_weighted_alphas.append(spec["weighted_alpha"])
        all_total_energies.append(spec["total_energy"])
        all_logit_maxs.append(float(logits[top1_idx]))
        all_logit_stds.append(float(np.std(logits)))

    all_gaps = np.array(all_gaps)
    all_offsets = np.array(all_offsets)
    all_eff_gaps = np.array(all_eff_gaps)
    all_probs = np.array(all_probs)
    all_h_norms = np.array(all_h_norms)
    all_ratio10s = np.array(all_ratio10s)
    all_ratio50s = np.array(all_ratio50s)
    all_ratio90s = np.array(all_ratio90s)
    all_alphas = np.array(all_alphas)
    all_weighted_alphas = np.array(all_weighted_alphas)
    all_total_energies = np.array(all_total_energies)
    all_logit_maxs = np.array(all_logit_maxs)
    all_logit_stds = np.array(all_logit_stds)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # ===== 因果链: h特征 → effective_gap → prob =====
    # Step 1: h特征 → effective_gap
    from numpy.linalg import lstsq

    # 单变量
    r_h_norm_eff, _ = pearsonr(all_h_norms, all_eff_gaps)
    r_ratio10_eff, _ = pearsonr(all_ratio10s, all_eff_gaps)
    r_ratio50_eff, _ = pearsonr(all_ratio50s, all_eff_gaps)
    r_ratio90_eff, _ = pearsonr(all_ratio90s, all_eff_gaps)
    r_alpha_eff, _ = pearsonr(all_alphas, all_eff_gaps)
    r_walpha_eff, _ = pearsonr(all_weighted_alphas, all_eff_gaps)
    r_total_e_eff, _ = pearsonr(all_total_energies, all_eff_gaps)

    # 多变量: eff_gap = f(h_norm, ratio50, alpha, total_energy)
    X_eff = np.column_stack([all_h_norms, all_ratio50s, all_alphas,
                              all_total_energies, np.ones(len(texts))])
    coeffs_eff, _, _, _ = lstsq(X_eff, all_eff_gaps, rcond=None)
    pred_eff = X_eff @ coeffs_eff
    r_multi_eff = float(np.corrcoef(pred_eff, all_eff_gaps)[0, 1])

    # Step 2: h特征 → gap
    X_gap = np.column_stack([all_h_norms, all_ratio50s, all_alphas,
                              all_total_energies, np.ones(len(texts))])
    coeffs_gap, _, _, _ = lstsq(X_gap, all_gaps, rcond=None)
    pred_gap = X_gap @ coeffs_gap
    r_multi_gap = float(np.corrcoef(pred_gap, all_gaps)[0, 1])

    # Step 3: h特征 → offset
    X_off = np.column_stack([all_h_norms, all_ratio50s, all_alphas,
                              all_total_energies, np.ones(len(texts))])
    coeffs_off, _, _, _ = lstsq(X_off, all_offsets, rcond=None)
    pred_off = X_off @ coeffs_off
    r_multi_off = float(np.corrcoef(pred_off, all_offsets)[0, 1])

    # Step 4: h特征 → logit_max (验证近似: logit_max ≈ ||W_U||_F * ||h|| / sqrt(vocab))
    r_h_norm_logit_max, _ = pearsonr(all_h_norms, all_logit_maxs)
    # 理论预测
    theoretical_logit_max = W_U_frobenius * all_h_norms / np.sqrt(vocab_size)
    r_theory_logit_max = float(np.corrcoef(theoretical_logit_max, all_logit_maxs)[0, 1])

    # Step 5: h特征 → prob (完整因果链)
    pred_eff_prob = sigmoid(pred_eff)
    r_eff_chain = float(np.corrcoef(pred_eff_prob, all_probs)[0, 1])

    pred_gap_prob = sigmoid(pred_gap - float(np.mean(all_offsets)))
    r_gap_chain = float(np.corrcoef(pred_gap_prob, all_probs)[0, 1])

    # Step 6: 各h特征→prob的直接关系
    r_h_norm_prob, _ = pearsonr(all_h_norms, all_probs)
    r_ratio50_prob, _ = pearsonr(all_ratio50s, all_probs)
    r_total_e_prob, _ = pearsonr(all_total_energies, all_probs)

    # Step 7: 用logit_max作为中间变量
    # gap ≈ logit_max - logit_2nd, 但logit_2nd难以预测
    # 简化: prob = f(logit_max, logit_std)
    X_ls = np.column_stack([all_logit_maxs, all_logit_stds, np.ones(len(texts))])
    coeffs_ls, _, _, _ = lstsq(X_ls, all_probs, rcond=None)
    pred_ls = X_ls @ coeffs_ls
    r_logit_stats_prob = float(np.corrcoef(pred_ls, all_probs)[0, 1])

    results["h_features_to_effective_gap"] = {
        "h_norm_r": float(r_h_norm_eff),
        "ratio10_r": float(r_ratio10_eff),
        "ratio50_r": float(r_ratio50_eff),
        "ratio90_r": float(r_ratio90_eff),
        "alpha_r": float(r_alpha_eff),
        "weighted_alpha_r": float(r_walpha_eff),
        "total_energy_r": float(r_total_e_eff),
        "multi_regression_r": float(r_multi_eff),
    }
    results["h_features_to_gap"] = {
        "multi_regression_r": float(r_multi_gap),
    }
    results["h_features_to_offset"] = {
        "multi_regression_r": float(r_multi_off),
    }
    results["h_norm_to_logit_max"] = {
        "correlation_r": float(r_h_norm_logit_max),
        "theoretical_approximation_r": float(r_theory_logit_max),
        "W_U_frobenius": W_U_frobenius,
        "vocab_size": vocab_size,
    }
    results["full_causal_chain"] = {
        "h_features_to_eff_gap_r": float(r_multi_eff),
        "sigmoid_eff_gap_to_prob_r": float(r_eff_chain),
        "h_features_to_gap_r": float(r_multi_gap),
        "sigmoid_gap_to_prob_r": float(r_gap_chain),
    }
    results["h_features_to_prob_direct"] = {
        "h_norm_r": float(r_h_norm_prob),
        "ratio50_r": float(r_ratio50_prob),
        "total_energy_r": float(r_total_e_prob),
    }
    results["logit_stats_to_prob"] = {
        "multi_regression_r": float(r_logit_stats_prob),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p603.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P603 Results for {model_name}:")
    print(f"  h_norm->eff_gap r={r_h_norm_eff:.4f}, ratio50->eff_gap r={r_ratio50_eff:.4f}")
    print(f"  Multi h features->eff_gap r={r_multi_eff:.4f}")
    print(f"  Multi h features->gap r={r_multi_gap:.4f}")
    print(f"  h_norm->logit_max r={r_h_norm_logit_max:.4f}, theory r={r_theory_logit_max:.4f}")
    print(f"  Full chain: h->eff_gap->prob r={r_eff_chain:.4f}")
    print(f"  Full chain: h->gap->prob r={r_gap_chain:.4f}")
    print(f"  logit_stats->prob r={r_logit_stats_prob:.4f}")
    return results


def run_p604(model, tokenizer, device, model_info, texts, output_dir):
    """P604: h的L2范数→logit_max的因果链 — 验证近似: logit_max ≈ ||W_U||_F * ||h||_2 / sqrt(d_vocab)"""
    model_name = model_info.name
    W_U = get_W_U(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    vocab_size = W_U.shape[0]

    print(f"\n{'='*60}")
    print(f"P604: h L2 norm -> logit_max causal chain -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    W_U_frobenius = float(np.linalg.norm(W_U))

    results = {"model": model_name, "experiment": "p604", "n_texts": len(texts),
               "W_U_frobenius": W_U_frobenius, "vocab_size": vocab_size, "d_model": d_model}

    # h各层的L2范数
    sample_layers = get_sample_layers(n_layers, n_samples=10)

    all_h_norms = []
    all_logit_maxs = []
    all_logit_2nd = []
    all_logit_gaps = []
    all_probs = []
    all_c1_norms = []  # c_1 = U_wut[:,0].T @ h, 第1个SVD方向的投影
    all_c_k_norms = []  # ||c_k||_2

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_max = float(logits[top1_idx])
        logit_2nd = float(logits[top2_idx])
        logit_gap = logit_max - logit_2nd
        logit_shifted = logits - logits[top1_idx]
        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))

        h_norm = float(np.linalg.norm(h))
        c_k = U_wut[:, :K].T @ h
        c1_norm = float(np.abs(c_k[0]))
        c_k_norm = float(np.linalg.norm(c_k))

        all_h_norms.append(h_norm)
        all_logit_maxs.append(logit_max)
        all_logit_2nd.append(logit_2nd)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
        all_c1_norms.append(c1_norm)
        all_c_k_norms.append(c_k_norm)

    all_h_norms = np.array(all_h_norms)
    all_logit_maxs = np.array(all_logit_maxs)
    all_logit_2nd = np.array(all_logit_2nd)
    all_logit_gaps = np.array(all_logit_gaps)
    all_probs = np.array(all_probs)
    all_c1_norms = np.array(all_c1_norms)
    all_c_k_norms = np.array(all_c_k_norms)

    # 1. h_norm -> logit_max
    r_h_norm_logit_max, _ = pearsonr(all_h_norms, all_logit_maxs)

    # 2. 理论近似: logit_max ≈ ||W_U||_F * ||h|| / sqrt(vocab_size)
    theoretical = W_U_frobenius * all_h_norms / np.sqrt(vocab_size)
    r_theory, _ = pearsonr(theoretical, all_logit_maxs)

    # 更好的近似: logit_max ≈ max(W_U @ h)
    # 用c_k: logit_max = max(W_U @ U_wut @ c_k) ≈ max(S @ c_k) (仅前K个)
    # logit_max ≈ s_1 * c_1 (第1个SVD分量主导)
    r_c1_logit_max, _ = pearsonr(all_c1_norms, all_logit_maxs)

    # 3. c_k_norm -> logit_max
    r_ck_norm_logit_max, _ = pearsonr(all_c_k_norms, all_logit_maxs)

    # 4. h_norm -> logit_2nd
    r_h_norm_logit_2nd, _ = pearsonr(all_h_norms, all_logit_2nd)

    # 5. h_norm -> logit_gap
    r_h_norm_logit_gap, _ = pearsonr(all_h_norms, all_logit_gaps)

    # 6. 多回归: logit_max = f(h_norm, c1_norm, c_k_norm)
    from numpy.linalg import lstsq
    X = np.column_stack([all_h_norms, all_c1_norms, all_c_k_norms, np.ones(len(texts))])
    coeffs, _, _, _ = lstsq(X, all_logit_maxs, rcond=None)
    pred = X @ coeffs
    r_multi = float(np.corrcoef(pred, all_logit_maxs)[0, 1])

    # 7. logit_max -> prob
    r_logit_max_prob, _ = pearsonr(all_logit_maxs, all_probs)

    # 8. h_norm -> prob (直接)
    r_h_norm_prob, _ = pearsonr(all_h_norms, all_probs)

    # 9. 各层h_norm的变化
    layer_h_norms = {l: [] for l in sample_layers}
    for ti, text in enumerate(texts[:50]):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for l in sample_layers:
                h_l = outputs.hidden_states[l][0, -1].float().cpu().numpy()
                layer_h_norms[l].append(float(np.linalg.norm(h_l)))

    layer_h_norm_means = {str(l): float(np.mean(v)) for l, v in layer_h_norms.items()}

    results["h_norm_to_logit_max"] = {
        "correlation_r": float(r_h_norm_logit_max),
        "theoretical_r": float(r_theory),
        "c1_norm_r": float(r_c1_logit_max),
        "c_k_norm_r": float(r_ck_norm_logit_max),
        "multi_regression_r": float(r_multi),
    }
    results["h_norm_to_other"] = {
        "logit_2nd_r": float(r_h_norm_logit_2nd),
        "logit_gap_r": float(r_h_norm_logit_gap),
        "prob_r": float(r_h_norm_prob),
    }
    results["logit_max_to_prob"] = {
        "correlation_r": float(r_logit_max_prob),
    }
    results["layer_h_norms"] = layer_h_norm_means
    results["statistics"] = {
        "h_norm_mean": float(np.mean(all_h_norms)),
        "h_norm_std": float(np.std(all_h_norms)),
        "logit_max_mean": float(np.mean(all_logit_maxs)),
        "logit_max_std": float(np.std(all_logit_maxs)),
        "c1_norm_mean": float(np.mean(all_c1_norms)),
        "c_k_norm_mean": float(np.mean(all_c_k_norms)),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p604.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P604 Results for {model_name}:")
    print(f"  h_norm->logit_max r={r_h_norm_logit_max:.4f}")
    print(f"  Theory (||W_U||_F*||h||/sqrt(V)) r={r_theory:.4f}")
    print(f"  c1_norm->logit_max r={r_c1_logit_max:.4f}")
    print(f"  c_k_norm->logit_max r={r_ck_norm_logit_max:.4f}")
    print(f"  Multi regression r={r_multi:.4f}")
    print(f"  h_norm->logit_gap r={r_h_norm_logit_gap:.4f}")
    print(f"  h_norm->prob r={r_h_norm_prob:.4f}")
    print(f"  logit_max->prob r={r_logit_max_prob:.4f}")
    return results


def run_p605(model, tokenizer, device, model_info, texts, output_dir):
    """P605: 统一语言编码方程v4 — prob = sigmoid(effective_gap), effective_gap由h特征预测"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P605: Unified encoding equation v4 -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p605", "n_texts": len(texts)}

    all_gaps = []
    all_offsets = []
    all_probs = []
    all_h_norms = []
    all_ratio50s = []
    all_alphas = []
    all_total_energies = []
    all_logit_maxs = []
    all_logit_stds = []
    all_c1_norms = []
    all_c_k_norms = []
    all_weighted_alphas = []
    all_ratio10s = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        logit_shifted = logits - logits[top1_idx]
        offset = float(np.log(np.sum(np.exp(logit_shifted)) - 1.0 + 1e-30))
        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))

        spec = compute_h_spectral_features(h, U_wut, s_wut, K=K)
        c_k = spec["c_k"]

        all_gaps.append(logit_gap)
        all_offsets.append(offset)
        all_probs.append(top1_prob)
        all_h_norms.append(spec["h_norm"])
        all_ratio50s.append(spec["ratio50"])
        all_ratio10s.append(spec["ratio10"])
        all_alphas.append(spec["alpha"])
        all_weighted_alphas.append(spec["weighted_alpha"])
        all_total_energies.append(spec["total_energy"])
        all_logit_maxs.append(float(logits[top1_idx]))
        all_logit_stds.append(float(np.std(logits)))
        all_c1_norms.append(float(np.abs(c_k[0])))
        all_c_k_norms.append(float(np.linalg.norm(c_k)))

    all_gaps = np.array(all_gaps)
    all_offsets = np.array(all_offsets)
    all_probs = np.array(all_probs)
    all_h_norms = np.array(all_h_norms)
    all_ratio50s = np.array(all_ratio50s)
    all_ratio10s = np.array(all_ratio10s)
    all_alphas = np.array(all_alphas)
    all_weighted_alphas = np.array(all_weighted_alphas)
    all_total_energies = np.array(all_total_energies)
    all_logit_maxs = np.array(all_logit_maxs)
    all_logit_stds = np.array(all_logit_stds)
    all_c1_norms = np.array(all_c1_norms)
    all_c_k_norms = np.array(all_c_k_norms)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    from numpy.linalg import lstsq

    # ===== V4方程: prob = sigmoid(effective_gap) =====
    # effective_gap由h特征预测

    # Version A: effective_gap = h_norm * f(spectral)
    # 尝试: eff_gap = w1*h_norm + w2*ratio50 + w3*alpha + w4*total_energy + w5*c1_norm + w6*c_k_norm + b
    eff_gaps = all_gaps - all_offsets
    X_full = np.column_stack([all_h_norms, all_ratio50s, all_ratio10s, all_alphas,
                               all_weighted_alphas, all_total_energies, all_c1_norms,
                               all_c_k_norms, np.ones(len(texts))])
    coeffs_full, _, _, _ = lstsq(X_full, eff_gaps, rcond=None)
    pred_eff_full = X_full @ coeffs_full
    r_eff_full = float(np.corrcoef(pred_eff_full, eff_gaps)[0, 1])
    pred_v4_full = sigmoid(pred_eff_full)
    r_v4_full = float(np.corrcoef(pred_v4_full, all_probs)[0, 1])

    # Version B: 只用h_norm + ratio50 + total_energy (3个主要频谱特征)
    X_simple = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_simple, _, _, _ = lstsq(X_simple, eff_gaps, rcond=None)
    pred_eff_simple = X_simple @ coeffs_simple
    r_eff_simple = float(np.corrcoef(pred_eff_simple, eff_gaps)[0, 1])
    pred_v4_simple = sigmoid(pred_eff_simple)
    r_v4_simple = float(np.corrcoef(pred_v4_simple, all_probs)[0, 1])

    # Version C: gap的预测(不用offset)
    X_gap = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_gap, _, _, _ = lstsq(X_gap, all_gaps, rcond=None)
    pred_gap = X_gap @ coeffs_gap
    r_gap_pred = float(np.corrcoef(pred_gap, all_gaps)[0, 1])

    # Version D: offset的预测
    X_off = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_off, _, _, _ = lstsq(X_off, all_offsets, rcond=None)
    pred_off = X_off @ coeffs_off
    r_off_pred = float(np.corrcoef(pred_off, all_offsets)[0, 1])

    # Version E: 直接预测prob (不经过sigmoid)
    X_prob = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_prob, _, _, _ = lstsq(X_prob, all_probs, rcond=None)
    pred_prob = X_prob @ coeffs_prob
    r_prob_direct = float(np.corrcoef(pred_prob, all_probs)[0, 1])

    # Version F: prob = sigmoid(w1*pred_gap + w2*pred_offset + b) (两步预测)
    # 用预测的gap和offset计算prob
    pred_eff_v2 = pred_gap - pred_off
    pred_v4_v2 = sigmoid(pred_eff_v2)
    r_v4_v2 = float(np.corrcoef(pred_v4_v2, all_probs)[0, 1])

    # Version G: Oracle — 用实际gap和offset
    oracle_eff = all_gaps - all_offsets
    pred_oracle = sigmoid(oracle_eff)
    r_oracle = float(np.corrcoef(pred_oracle, all_probs)[0, 1])

    # Version H: 只用h_norm预测prob
    X_h_only = np.column_stack([all_h_norms, np.ones(len(texts))])
    coeffs_h, _, _, _ = lstsq(X_h_only, all_probs, rcond=None)
    pred_h = X_h_only @ coeffs_h
    r_h_only = float(np.corrcoef(pred_h, all_probs)[0, 1])

    # Version I: 非线性优化 — prob = sigmoid(w1*h_norm + w2*ratio50 + w3*total_energy + b)
    def loss_v4(params):
        w1, w2, w3, b = params
        z = w1 * all_h_norms + w2 * all_ratio50s + w3 * all_total_energies + b
        pred = sigmoid(z)
        r = np.corrcoef(pred, all_probs)[0, 1]
        return -r

    res_v4 = minimize(loss_v4, [1.0, 1.0, 1.0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 5000})
    w1_v4, w2_v4, w3_v4, b_v4 = res_v4.x
    z_v4 = w1_v4 * all_h_norms + w2_v4 * all_ratio50s + w3_v4 * all_total_energies + b_v4
    pred_v4_nl = sigmoid(z_v4)
    r_v4_nl = float(np.corrcoef(pred_v4_nl, all_probs)[0, 1])

    results["v4_full"] = {
        "eff_gap_prediction_r": float(r_eff_full),
        "v4_sigmoid_prob_r": float(r_v4_full),
    }
    results["v4_simple"] = {
        "eff_gap_prediction_r": float(r_eff_simple),
        "v4_sigmoid_prob_r": float(r_v4_simple),
    }
    results["gap_prediction"] = {"r": float(r_gap_pred)}
    results["offset_prediction"] = {"r": float(r_off_pred)}
    results["prob_direct"] = {"r": float(r_prob_direct)}
    results["v4_two_step"] = {"r": float(r_v4_v2)}
    results["oracle"] = {"r": float(r_oracle)}
    results["h_norm_only"] = {"r": float(r_h_only)}
    results["v4_nonlinear"] = {
        "r": float(r_v4_nl),
        "w1_h_norm": float(w1_v4),
        "w2_ratio50": float(w2_v4),
        "w3_total_energy": float(w3_v4),
        "bias": float(b_v4),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p605.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P605 Results for {model_name}:")
    print(f"  Oracle (actual gap-offset->sigmoid) r={r_oracle:.4f}")
    print(f"  V4 full (8 h features->eff_gap->sigmoid) r={r_v4_full:.4f}")
    print(f"  V4 simple (3 h features->eff_gap->sigmoid) r={r_v4_simple:.4f}")
    print(f"  V4 two-step (pred gap - pred offset->sigmoid) r={r_v4_v2:.4f}")
    print(f"  V4 nonlinear (optimize sigmoid) r={r_v4_nl:.4f}")
    print(f"  h features->gap r={r_gap_pred:.4f}")
    print(f"  h features->offset r={r_off_pred:.4f}")
    print(f"  Direct h features->prob r={r_prob_direct:.4f}")
    print(f"  h_norm only->prob r={r_h_only:.4f}")
    return results


def run_p606(model, tokenizer, device, model_info, texts, output_dir):
    """P606: 频谱力学的最终评估 — P1-P605所有频谱→语言行为路径的最佳路径"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P606: Final spectral mechanics evaluation -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p606", "n_texts": len(texts)}

    # 收集所有h特征
    all_gaps = []
    all_offsets = []
    all_probs = []
    all_h_norms = []
    all_ratio50s = []
    all_alphas = []
    all_total_energies = []
    all_logit_maxs = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        h, logits = get_hidden_and_logits(model, tokenizer, device, text)

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        logit_shifted = logits - logits[top1_idx]
        offset = float(np.log(np.sum(np.exp(logit_shifted)) - 1.0 + 1e-30))
        top1_prob = float(np.exp(logit_shifted[top1_idx]) / np.sum(np.exp(logit_shifted)))

        spec = compute_h_spectral_features(h, U_wut, s_wut, K=K)

        all_gaps.append(logit_gap)
        all_offsets.append(offset)
        all_probs.append(top1_prob)
        all_h_norms.append(spec["h_norm"])
        all_ratio50s.append(spec["ratio50"])
        all_alphas.append(spec["alpha"])
        all_total_energies.append(spec["total_energy"])
        all_logit_maxs.append(float(logits[top1_idx]))

    all_gaps = np.array(all_gaps)
    all_offsets = np.array(all_offsets)
    all_probs = np.array(all_probs)
    all_h_norms = np.array(all_h_norms)
    all_ratio50s = np.array(all_ratio50s)
    all_alphas = np.array(all_alphas)
    all_total_energies = np.array(all_total_energies)
    all_logit_maxs = np.array(all_logit_maxs)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    from numpy.linalg import lstsq

    # ===== 综合评估所有路径 =====
    paths = {}

    # Path 1: spectral_ratio50 -> prob (Phase I原始路径)
    r1, _ = pearsonr(all_ratio50s, all_probs)
    paths["P1_ratio50_to_prob"] = float(r1)

    # Path 2: spectral -> weighted_gap -> prob (Phase III-V路径)
    eff_gaps = all_gaps - all_offsets
    X_wg = np.column_stack([all_ratio50s, all_alphas, all_h_norms, np.ones(len(texts))])
    coeffs_wg, _, _, _ = lstsq(X_wg, all_gaps, rcond=None)
    pred_wg = X_wg @ coeffs_wg
    r2_pred = float(np.corrcoef(pred_wg, all_gaps)[0, 1])
    pred_wg_prob = sigmoid(pred_wg - float(np.mean(all_offsets)))
    r2_prob = float(np.corrcoef(pred_wg_prob, all_probs)[0, 1])
    paths["P2_spectral_to_gap_to_prob"] = {"gap_r": float(r2_pred), "prob_r": float(r2_prob)}

    # Path 3: logit_gap -> prob (Phase CXXXVI路径)
    r3, _ = pearsonr(all_gaps, all_probs)
    paths["P3_logit_gap_to_prob"] = float(r3)

    # Path 4: logit_gap -> logit(prob) (Phase CXXXVI路径)
    logit_probs = np.log(all_probs / (1 - all_probs + 1e-10) + 1e-10)
    r4, _ = pearsonr(all_gaps, logit_probs)
    paths["P4_logit_gap_to_logitprob"] = float(r4)

    # Path 5: logit_gap - offset -> prob (Phase CXXXVII路径)
    r5, _ = pearsonr(eff_gaps, all_probs)
    pred_p5 = sigmoid(eff_gaps)
    r5_sig = float(np.corrcoef(pred_p5, all_probs)[0, 1])
    paths["P5_effective_gap_to_prob"] = {"linear_r": float(r5), "sigmoid_r": float(r5_sig)}

    # Path 6: simple sigmoid(gap - mean_offset) -> prob (Phase CXXXVII路径)
    pred_p6 = sigmoid(all_gaps - float(np.mean(all_offsets)))
    r6 = float(np.corrcoef(pred_p6, all_probs)[0, 1])
    paths["P6_simple_sigmoid_to_prob"] = float(r6)

    # Path 7: V3 equation (Phase CXXXVII路径) — 已知r>0.994
    # 这里用简化版: prob = sigmoid(gap - offset + b)
    def loss_v3(params):
        b = params[0]
        z = all_gaps - all_offsets + b
        pred = sigmoid(z)
        return -float(np.corrcoef(pred, all_probs)[0, 1])
    res_v3 = minimize(loss_v3, [0.0], method='Nelder-Mead', options={'maxiter': 1000})
    pred_p7 = sigmoid(all_gaps - all_offsets + res_v3.x[0])
    r7 = float(np.corrcoef(pred_p7, all_probs)[0, 1])
    paths["P7_v3_equation_to_prob"] = float(r7)

    # Path 8: h features -> effective_gap -> sigmoid -> prob (Phase CXXXVIII路径)
    X_eff = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_eff, _, _, _ = lstsq(X_eff, eff_gaps, rcond=None)
    pred_eff = X_eff @ coeffs_eff
    r8_eff = float(np.corrcoef(pred_eff, eff_gaps)[0, 1])
    pred_p8 = sigmoid(pred_eff)
    r8 = float(np.corrcoef(pred_p8, all_probs)[0, 1])
    paths["P8_h_features_to_eff_gap_to_prob"] = {"eff_gap_r": float(r8_eff), "prob_r": float(r8)}

    # Path 9: h_norm -> logit_max -> prob
    r9a, _ = pearsonr(all_h_norms, all_logit_maxs)
    r9b, _ = pearsonr(all_logit_maxs, all_probs)
    paths["P9_h_norm_to_logit_max_to_prob"] = {"h_to_max_r": float(r9a), "max_to_prob_r": float(r9b)}

    # Path 10: direct h features -> prob (无中间变量)
    X_direct = np.column_stack([all_h_norms, all_ratio50s, all_total_energies, np.ones(len(texts))])
    coeffs_direct, _, _, _ = lstsq(X_direct, all_probs, rcond=None)
    pred_direct = X_direct @ coeffs_direct
    r10 = float(np.corrcoef(pred_direct, all_probs)[0, 1])
    paths["P10_direct_h_features_to_prob"] = float(r10)

    # 综合排名
    path_ranking = []
    path_ranking.append(("P7_v3_equation", r7, "Oracle: uses actual gap+offset"))
    path_ranking.append(("P5_effective_gap_sigmoid", r5_sig, "Oracle: uses actual gap-offset"))
    path_ranking.append(("P4_logit_gap_to_logitprob", r4, "Oracle: uses actual gap"))
    path_ranking.append(("P6_simple_sigmoid", r6, "Semi-oracle: uses actual gap+mean_offset"))
    path_ranking.append(("P3_logit_gap_to_prob", r3, "Oracle: uses actual gap"))
    path_ranking.append(("P10_direct_h_features", r10, "No oracle: h features only"))
    path_ranking.append(("P8_h_to_eff_gap_to_prob", r8, "No oracle: h features->eff_gap->sigmoid"))
    path_ranking.append(("P2_spectral_to_gap_to_prob", r2_prob, "No oracle: spectral->gap->sigmoid"))
    path_ranking.append(("P1_ratio50_to_prob", r1, "No oracle: single spectral feature"))
    path_ranking.sort(key=lambda x: x[1], reverse=True)

    results["all_paths"] = paths
    results["path_ranking"] = [
        {"rank": i+1, "path": p[0], "r": float(p[1]), "note": p[2]}
        for i, p in enumerate(path_ranking)
    ]

    # 关键问题: 非oracle路径的最佳r是多少?
    non_oracle_best = max(r10, r8, r2_prob, abs(r1))
    results["key_finding"] = {
        "best_non_oracle_r": float(non_oracle_best),
        "oracle_vs_non_oracle_gap": f"Oracle best r={r7:.4f} vs Non-oracle best r={non_oracle_best:.4f}",
        "gap_ratio": f"Non-oracle achieves {non_oracle_best/r7*100:.1f}% of oracle performance",
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p606.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P606 Results for {model_name}:")
    print(f"  === Path Ranking ===")
    for i, p in enumerate(path_ranking):
        print(f"  #{i+1}: {p[0]} r={p[1]:.4f} ({p[2]})")
    print(f"\n  Key finding: Non-oracle best r={non_oracle_best:.4f} vs Oracle best r={r7:.4f}")
    return results


# ===================== Main =====================
EXPERIMENTS = {
    "p601": run_p601,
    "p602": run_p602,
    "p603": run_p603,
    "p604": run_p604,
    "p605": run_p605,
    "p606": run_p606,
}


def main():
    parser = argparse.ArgumentParser(description="Phase CXXXVIII: Effective gap breakthrough")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()

    output_dir = "results/phase_cxxxviii"
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)

    n_texts = min(len(TEST_TEXTS), 210)
    texts = TEST_TEXTS[:n_texts]

    start_time = time.time()
    result = EXPERIMENTS[args.experiment](model, tokenizer, device, model_info, texts, output_dir)
    elapsed = time.time() - start_time

    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
