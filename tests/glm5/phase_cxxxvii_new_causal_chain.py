"""
Phase CXXXVII: 从频谱到语言行为的新因果链架构 (P595-P600)
================================================================
核心洞察: logit_gap->prob非线性变换r>0.93, 瓶颈完全在频谱->logit_gap
  P595: 格式-内容双通道模型 — prob = w_f * f(format_gap) + w_c * f(content_gap)
  P596: 内容方向的频谱力学 — content_ratio50->content_gap->prob
  P597: Delta_k条件期望模型 — 对高频token对构建Delta_k查找表
  P598: softmax偏移项(offset)的预测 — alpha->offset的因果链
  P599: 统一语言编码方程v3 — prob = sigmoid(Sum w_k * c_k * Delta_k - offset)
  P600: 跨模型不变量 — 哪些方程跨模型一致, 哪些不一致

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


def run_p595(model, tokenizer, device, model_info, texts, output_dir):
    """P595: Format-content dual channel model"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P595: Format-content dual channel model -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    punct_fraction, _ = identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=K)
    format_dirs = np.argsort(punct_fraction)[-5:]
    content_dirs = np.array([k for k in range(K) if k not in format_dirs])

    results = {
        "model": model_name, "experiment": "p595", "n_texts": len(texts),
        "format_directions": format_dirs.tolist(),
    }

    all_format_gaps = []
    all_content_gaps = []
    all_full_gaps = []
    all_probs = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))

        c_k = U_wut.T @ h
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k

        format_gap = float(np.sum(contrib_k[format_dirs]))
        content_gap = float(np.sum(contrib_k[content_dirs]))

        all_format_gaps.append(format_gap)
        all_content_gaps.append(content_gap)
        all_full_gaps.append(logit_gap)
        all_probs.append(top1_prob)

    all_format_gaps = np.array(all_format_gaps)
    all_content_gaps = np.array(all_content_gaps)
    all_full_gaps = np.array(all_full_gaps)
    all_probs = np.array(all_probs)

    # Single channel correlations
    try:
        r_format_prob, _ = pearsonr(all_format_gaps, all_probs)
    except:
        r_format_prob = 0.0
    try:
        r_content_prob, _ = pearsonr(all_content_gaps, all_probs)
    except:
        r_content_prob = 0.0
    try:
        r_full_prob, _ = pearsonr(all_full_gaps, all_probs)
    except:
        r_full_prob = 0.0

    # Dual channel linear regression: prob = w_f * format_gap + w_c * content_gap + b
    from numpy.linalg import lstsq
    X_dual = np.column_stack([all_format_gaps, all_content_gaps, np.ones(len(texts))])
    try:
        coeffs_dual, _, _, _ = lstsq(X_dual, all_probs, rcond=None)
        pred_dual = X_dual @ coeffs_dual
        r_dual = float(np.corrcoef(pred_dual, all_probs)[0, 1])
    except:
        coeffs_dual = [0, 0, 0]
        r_dual = 0.0

    # Dual channel with nonlinear: prob = sigmoid(w_f * format_gap + w_c * content_gap + b)
    # Optimize w_f, w_c, b via scipy
    from scipy.optimize import minimize

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def loss_sigmoid(params):
        w_f, w_c, b = params
        z = w_f * all_format_gaps + w_c * all_content_gaps + b
        pred = sigmoid(z)
        return -float(np.corrcoef(pred, all_probs)[0, 1])

    try:
        res = minimize(loss_sigmoid, [0.1, 0.1, 0.0], method='Nelder-Mead',
                       options={'maxiter': 1000})
        w_f_opt, w_c_opt, b_opt = res.x
        z_opt = w_f_opt * all_format_gaps + w_c_opt * all_content_gaps + b_opt
        pred_sigmoid = sigmoid(z_opt)
        r_sigmoid = float(np.corrcoef(pred_sigmoid, all_probs)[0, 1])
    except:
        w_f_opt, w_c_opt, b_opt = 0, 0, 0
        r_sigmoid = 0.0

    results["single_channel"] = {
        "format_gap_to_prob": float(r_format_prob),
        "content_gap_to_prob": float(r_content_prob),
        "full_gap_to_prob": float(r_full_prob),
    }
    results["dual_channel_linear"] = {
        "r": float(r_dual),
        "w_format": float(coeffs_dual[0]),
        "w_content": float(coeffs_dual[1]),
        "bias": float(coeffs_dual[2]),
    }
    results["dual_channel_sigmoid"] = {
        "r": float(r_sigmoid),
        "w_format": float(w_f_opt),
        "w_content": float(w_c_opt),
        "bias": float(b_opt),
    }

    # Individual direction contribution analysis
    dir_contrib_to_prob = {}
    for k in range(min(20, K)):
        c_k_all = []
        for ti, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
                logits_t = outputs.logits[0, -1].float().cpu().numpy()
            top1_t = np.argmax(logits_t)
            top2_t = np.argsort(logits_t)[-2]
            c_k_i = float((U_wut.T @ h)[k])
            delta_k_i = float((U_wut.T @ (W_U[top1_t] - W_U[top2_t]))[k])
            c_k_all.append(c_k_i * delta_k_i)
        try:
            r, _ = pearsonr(c_k_all, all_probs)
            dir_contrib_to_prob[f"dir{k}"] = float(r)
        except:
            dir_contrib_to_prob[f"dir{k}"] = 0.0

    results["direction_contrib_to_prob"] = dir_contrib_to_prob

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p595.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P595 Results for {model_name}:")
    print(f"  Format gap->prob r = {r_format_prob:.4f}")
    print(f"  Content gap->prob r = {r_content_prob:.4f}")
    print(f"  Full gap->prob r = {r_full_prob:.4f}")
    print(f"  Dual linear r = {r_dual:.4f} (w_f={coeffs_dual[0]:.4f}, w_c={coeffs_dual[1]:.4f})")
    print(f"  Dual sigmoid r = {r_sigmoid:.4f} (w_f={w_f_opt:.4f}, w_c={w_c_opt:.4f})")
    return results


def run_p596(model, tokenizer, device, model_info, texts, output_dir):
    """P596: Content direction spectral mechanics"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P596: Content direction spectral mechanics -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    punct_fraction, _ = identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=K)
    format_dirs = set(np.argsort(punct_fraction)[-5:])
    content_dirs = [k for k in range(K) if k not in format_dirs]

    results = {"model": model_name, "experiment": "p596", "n_texts": len(texts)}

    all_content_ratio50 = []
    all_format_ratio50 = []
    all_content_gaps = []
    all_probs = []
    all_alpha = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))

        c_k = U_wut.T @ h
        h_energy = c_k ** 2
        total_energy = np.sum(h_energy) + 1e-10

        # Content and format energy separately
        content_energy = np.sum(h_energy[content_dirs])
        format_energy = np.sum(h_energy[list(format_dirs)])
        all_content_ratio50.append(float(content_energy / total_energy))
        all_format_ratio50.append(float(format_energy / total_energy))

        # Content gap
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k
        content_gap = float(np.sum(contrib_k[content_dirs]))
        all_content_gaps.append(content_gap)
        all_probs.append(top1_prob)

        # Alpha
        log_s = np.log10(s_wut + 1e-10)
        log_rank = np.log10(np.arange(1, K+1))
        try:
            alpha_fit = np.polyfit(log_rank, log_s, 1)
            all_alpha.append(float(alpha_fit[0]))
        except:
            all_alpha.append(0.0)

    all_content_ratio50 = np.array(all_content_ratio50)
    all_format_ratio50 = np.array(all_format_ratio50)
    all_content_gaps = np.array(all_content_gaps)
    all_probs = np.array(all_probs)
    all_alpha = np.array(all_alpha)

    # Correlations
    corr = {}
    for name, arr in [("content_ratio50", all_content_ratio50), ("format_ratio50", all_format_ratio50),
                       ("alpha", all_alpha), ("content_gap", all_content_gaps)]:
        try:
            r, _ = pearsonr(arr, all_probs)
            corr[f"{name}_to_prob"] = float(r)
        except:
            corr[f"{name}_to_prob"] = 0.0

    # content_ratio50 -> content_gap -> prob chain
    try:
        r_cr50_cg, _ = pearsonr(all_content_ratio50, all_content_gaps)
    except:
        r_cr50_cg = 0.0

    # alpha -> content_gap
    try:
        r_alpha_cg, _ = pearsonr(all_alpha, all_content_gaps)
    except:
        r_alpha_cg = 0.0

    results["correlations"] = corr
    results["content_chain"] = {
        "content_ratio50_to_content_gap": float(r_cr50_cg),
        "alpha_to_content_gap": float(r_alpha_cg),
    }
    results["stats"] = {
        "mean_content_ratio50": float(np.mean(all_content_ratio50)),
        "mean_format_ratio50": float(np.mean(all_format_ratio50)),
        "mean_content_gap": float(np.mean(all_content_gaps)),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p596.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P596 Results for {model_name}:")
    print(f"  Content ratio50->prob r = {corr.get('content_ratio50_to_prob', 0):.4f}")
    print(f"  Content ratio50->content_gap r = {r_cr50_cg:.4f}")
    print(f"  Alpha->content_gap r = {r_alpha_cg:.4f}")
    print(f"  Mean content ratio50 = {np.mean(all_content_ratio50):.4f}")
    return results


def run_p597(model, tokenizer, device, model_info, texts, output_dir):
    """P597: Delta_k conditional expectation model"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P597: Delta_k conditional expectation model -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p597", "n_texts": len(texts)}

    # Collect all Delta_k, top1/top2 pairs
    all_data = []
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
        logit_gap = float(logits[top1_idx] - logits[top2_idx])

        c_k = U_wut.T @ h
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k

        all_data.append({
            "top1": int(top1_idx), "top2": int(top2_idx),
            "c_k": c_k, "Delta_k": Delta_k, "contrib_k": contrib_k,
            "logit_gap": logit_gap, "prob": top1_prob,
        })

    # 1. Top-1 token frequency distribution
    top1_counts = defaultdict(int)
    for d in all_data:
        top1_counts[d["top1"]] += 1
    top1_sorted = sorted(top1_counts.items(), key=lambda x: -x[1])
    results["top1_distribution"] = {
        "n_unique_top1": len(top1_counts),
        "top5_tokens": [(int(t), int(c)) for t, c in top1_sorted[:5]],
    }

    # 2. Conditional expectation: E[Delta_k | top1 = t]
    top1_groups = defaultdict(list)
    for d in all_data:
        top1_groups[d["top1"]].append(d)

    # Compute conditional mean Delta_k for frequent top1 tokens
    cond_mean_Delta = {}
    for tid, group in top1_groups.items():
        if len(group) >= 3:
            mean_delta = np.mean([d["Delta_k"] for d in group], axis=0)
            cond_mean_Delta[tid] = mean_delta

    # 3. Use conditional mean Delta_k to predict logit_gap
    # For each text, replace Delta_k with E[Delta_k | top1]
    predicted_gaps = {"cond_mean": [], "global_mean": [], "actual": []}
    global_mean_Delta = np.mean([d["Delta_k"] for d in all_data], axis=0)

    for d in all_data:
        actual_gap = d["logit_gap"]
        c_k = d["c_k"]
        predicted_gaps["actual"].append(actual_gap)

        # Conditional mean prediction
        if d["top1"] in cond_mean_Delta:
            pred_delta = cond_mean_Delta[d["top1"]]
        else:
            pred_delta = global_mean_Delta
        pred_gap_cond = float(np.sum(c_k * pred_delta))
        predicted_gaps["cond_mean"].append(pred_gap_cond)

        # Global mean prediction
        pred_gap_global = float(np.sum(c_k * global_mean_Delta))
        predicted_gaps["global_mean"].append(pred_gap_global)

    actual_gaps = np.array(predicted_gaps["actual"])
    cond_pred = np.array(predicted_gaps["cond_mean"])
    global_pred = np.array(predicted_gaps["global_mean"])
    probs = np.array([d["prob"] for d in all_data])

    try:
        r_cond, _ = pearsonr(cond_pred, actual_gaps)
    except:
        r_cond = 0.0
    try:
        r_global, _ = pearsonr(global_pred, actual_gaps)
    except:
        r_global = 0.0
    try:
        r_cond_prob, _ = pearsonr(cond_pred, probs)
    except:
        r_cond_prob = 0.0
    try:
        r_global_prob, _ = pearsonr(global_pred, probs)
    except:
        r_global_prob = 0.0

    results["prediction"] = {
        "cond_mean_to_gap_r": float(r_cond),
        "global_mean_to_gap_r": float(r_global),
        "cond_mean_to_prob_r": float(r_cond_prob),
        "global_mean_to_prob_r": float(r_global_prob),
        "n_groups_with_3plus": len(cond_mean_Delta),
    }

    # 4. Sign consistency conditioned on top1
    sign_consistency_cond = {}
    for tid, group in top1_groups.items():
        if len(group) >= 5:
            signs = np.array([np.sign(d["contrib_k"]) for d in group])
            consistency = np.mean(np.abs(np.mean(signs, axis=0)))
            sign_consistency_cond[tid] = float(consistency)

    if sign_consistency_cond:
        results["sign_consistency_conditional"] = {
            "mean": float(np.mean(list(sign_consistency_cond.values()))),
            "max": float(np.max(list(sign_consistency_cond.values()))),
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p597.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P597 Results for {model_name}:")
    print(f"  Cond mean Delta->gap r = {r_cond:.4f}")
    print(f"  Global mean Delta->gap r = {r_global:.4f}")
    print(f"  Cond mean->prob r = {r_cond_prob:.4f}")
    print(f"  N groups with 3+ = {len(cond_mean_Delta)}")
    return results


def run_p598(model, tokenizer, device, model_info, texts, output_dir):
    """P598: Softmax offset prediction"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P598: Softmax offset prediction -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p598", "n_texts": len(texts)}

    all_offsets = []
    all_gaps = []
    all_probs = []
    all_alpha = []
    all_ratio50 = []
    all_logit_max = []
    all_logit_var = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))

        # Compute offset = log(sum_{i>2} exp(logit_i - logit_1))
        logit_top1 = logits[top1_idx]
        logit_top2 = logits[top2_idx]
        other_sum = 0.0
        for j in range(len(logits)):
            if j != top1_idx and j != top2_idx:
                other_sum += np.exp(logits[j] - logit_top1)
        offset = float(np.log(other_sum + np.exp(logit_top2 - logit_top1)))

        all_offsets.append(offset)
        all_gaps.append(logit_gap)
        all_probs.append(top1_prob)
        all_logit_max.append(float(logit_top1))
        all_logit_var.append(float(np.var(logits)))

        # Spectral features
        c_k = U_wut.T @ h
        h_energy = c_k ** 2
        total_energy = np.sum(h_energy) + 1e-10
        all_ratio50.append(float(np.sum(h_energy[:50]) / total_energy))

        log_s = np.log10(s_wut + 1e-10)
        log_rank = np.log10(np.arange(1, K+1))
        try:
            alpha_fit = np.polyfit(log_rank, log_s, 1)
            all_alpha.append(float(alpha_fit[0]))
        except:
            all_alpha.append(0.0)

    all_offsets = np.array(all_offsets)
    all_gaps = np.array(all_gaps)
    all_probs = np.array(all_probs)
    all_alpha = np.array(all_alpha)
    all_ratio50 = np.array(all_ratio50)
    all_logit_max = np.array(all_logit_max)
    all_logit_var = np.array(all_logit_var)

    # 1. Offset statistics
    results["offset_stats"] = {
        "mean": float(np.mean(all_offsets)),
        "std": float(np.std(all_offsets)),
        "min": float(np.min(all_offsets)),
        "max": float(np.max(all_offsets)),
    }

    # 2. What predicts offset?
    corr_to_offset = {}
    for name, arr in [("alpha", all_alpha), ("ratio50", all_ratio50),
                       ("logit_max", all_logit_max), ("logit_var", all_logit_var),
                       ("logit_gap", all_gaps)]:
        try:
            r, _ = pearsonr(arr, all_offsets)
            corr_to_offset[f"{name}_to_offset"] = float(r)
        except:
            corr_to_offset[f"{name}_to_offset"] = 0.0

    results["offset_predictors"] = corr_to_offset

    # 3. Complete chain: spectral -> offset -> prob
    # prob = sigmoid(gap - offset)
    # If we can predict offset, we can predict prob from gap
    predicted_prob_simple = 1.0 / (1.0 + np.exp(-all_gaps + np.mean(all_offsets)))
    try:
        r_simple, _ = pearsonr(predicted_prob_simple, all_probs)
    except:
        r_simple = 0.0

    # Multi-regression: alpha + ratio50 + gap -> prob
    from numpy.linalg import lstsq
    X = np.column_stack([all_alpha, all_ratio50, all_gaps, np.ones(len(texts))])
    try:
        coeffs, _, _, _ = lstsq(X, all_probs, rcond=None)
        pred_prob = X @ coeffs
        r_multi = float(np.corrcoef(pred_prob, all_probs)[0, 1])
    except:
        coeffs = [0, 0, 0, 0]
        r_multi = 0.0

    results["complete_chain"] = {
        "simple_sigmoid_r": float(r_simple),
        "multi_regression_r": float(r_multi),
        "multi_coeffs": [float(c) for c in coeffs],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p598.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P598 Results for {model_name}:")
    print(f"  Offset: mean={np.mean(all_offsets):.4f}, std={np.std(all_offsets):.4f}")
    for k, v in corr_to_offset.items():
        print(f"  {k} = {v:.4f}")
    print(f"  Simple sigmoid r = {r_simple:.4f}")
    print(f"  Multi regression r = {r_multi:.4f}")
    return results


def run_p599(model, tokenizer, device, model_info, texts, output_dir):
    """P599: Unified language encoding equation v3"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P599: Unified encoding equation v3 -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    punct_fraction, _ = identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=K)
    format_dirs = set(np.argsort(punct_fraction)[-5:])
    content_dirs = [k for k in range(K) if k not in format_dirs]

    results = {"model": model_name, "experiment": "p599", "n_texts": len(texts)}

    all_format_gaps = []
    all_content_gaps = []
    all_offsets = []
    all_probs = []

    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
        logit_gap = float(logits[top1_idx] - logits[top2_idx])

        c_k = U_wut.T @ h
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k

        format_gap = float(np.sum(contrib_k[list(format_dirs)]))
        content_gap = float(np.sum(contrib_k[content_dirs]))

        # Offset
        logit_top1 = logits[top1_idx]
        logit_top2 = logits[top2_idx]
        other_sum = 0.0
        for j in range(len(logits)):
            if j != top1_idx and j != top2_idx:
                other_sum += np.exp(logits[j] - logit_top1)
        offset = float(np.log(other_sum + np.exp(logit_top2 - logit_top1)))

        all_format_gaps.append(format_gap)
        all_content_gaps.append(content_gap)
        all_offsets.append(offset)
        all_probs.append(top1_prob)

    all_format_gaps = np.array(all_format_gaps)
    all_content_gaps = np.array(all_content_gaps)
    all_offsets = np.array(all_offsets)
    all_probs = np.array(all_probs)

    # Unified equation: prob = sigmoid(w_f * format_gap + w_c * content_gap - offset + b)
    from scipy.optimize import minimize

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def loss_v3(params):
        w_f, w_c, b = params
        z = w_f * all_format_gaps + w_c * all_content_gaps - all_offsets + b
        pred = sigmoid(z)
        return -float(np.corrcoef(pred, all_probs)[0, 1]) if np.std(pred) > 1e-10 else 1.0

    try:
        res = minimize(loss_v3, [0.1, 0.1, 0.0], method='Nelder-Mead',
                       options={'maxiter': 2000})
        w_f_opt, w_c_opt, b_opt = res.x
        z_opt = w_f_opt * all_format_gaps + w_c_opt * all_content_gaps - all_offsets + b_opt
        pred_v3 = sigmoid(z_opt)
        r_v3 = float(np.corrcoef(pred_v3, all_probs)[0, 1])
    except:
        w_f_opt, w_c_opt, b_opt = 0, 0, 0
        r_v3 = 0.0

    # Simpler: prob = sigmoid(gap - offset)
    pred_simple = sigmoid(all_format_gaps + all_content_gaps - all_offsets)
    try:
        r_simple_offset, _ = pearsonr(pred_simple, all_probs)
    except:
        r_simple_offset = 0.0

    # Even simpler: prob = sigmoid(gap - mean_offset)
    pred_mean_offset = sigmoid(all_format_gaps + all_content_gaps - np.mean(all_offsets))
    try:
        r_mean_offset, _ = pearsonr(pred_mean_offset, all_probs)
    except:
        r_mean_offset = 0.0

    results["v3_equation"] = {
        "r_optimized": float(r_v3),
        "w_format": float(w_f_opt),
        "w_content": float(w_c_opt),
        "bias": float(b_opt),
    }
    results["simpler_models"] = {
        "sigmoid_gap_minus_offset_r": float(r_simple_offset),
        "sigmoid_gap_minus_mean_offset_r": float(r_mean_offset),
    }

    # How well does format_gap + content_gap = logit_gap?
    sum_gaps = all_format_gaps + all_content_gaps
    from scipy.stats import pearsonr as pr
    try:
        r_sum_full, _ = pr(sum_gaps, all_format_gaps + all_content_gaps)
    except:
        r_sum_full = 0.0

    results["decomposition_check"] = {
        "format_content_sum_to_prob_r": float(pr(all_format_gaps + all_content_gaps, all_probs)[0]) if True else 0.0,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p599.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P599 Results for {model_name}:")
    print(f"  V3 equation r = {r_v3:.4f} (w_f={w_f_opt:.4f}, w_c={w_c_opt:.4f})")
    print(f"  sigmoid(gap - offset) r = {r_simple_offset:.4f}")
    print(f"  sigmoid(gap - mean_offset) r = {r_mean_offset:.4f}")
    return results


def run_p600(model, tokenizer, device, model_info, texts, output_dir):
    """P600: Cross-model invariants"""
    model_name = model_info.name
    W_U = get_W_U(model)

    print(f"\n{'='*60}")
    print(f"P600: Cross-model invariants -- {model_name}")
    print(f"{'='*60}")

    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)

    results = {"model": model_name, "experiment": "p600", "n_texts": len(texts)}

    # Collect comprehensive data
    all_data = []
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()

        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))

        c_k = U_wut.T @ h
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k

        all_data.append({
            "c_k": c_k, "Delta_k": Delta_k, "contrib_k": contrib_k,
            "logit_gap": logit_gap, "prob": top1_prob,
            "top1": int(top1_idx), "top2": int(top2_idx),
        })

    # 1. Verify invariants
    # Invariant 1: Additive model (already verified in P589)
    # Invariant 2: |c_k|*|Delta_k| -> |contrib_k|
    r_abs_decomp = []
    for k in range(K):
        abs_c = np.abs([d["c_k"][k] for d in all_data])
        abs_d = np.abs([d["Delta_k"][k] for d in all_data])
        abs_contrib = np.abs([d["contrib_k"][k] for d in all_data])
        try:
            r, _ = pearsonr(abs_c * abs_d, abs_contrib)
            r_abs_decomp.append(r)
        except:
            pass

    # Invariant 3: logit_gap -> prob nonlinear transform
    gaps = np.array([d["logit_gap"] for d in all_data])
    probs = np.array([d["prob"] for d in all_data])
    prob_logit = np.log(np.clip(probs, 0.001, 0.999) / np.clip(1-probs, 0.001, 0.999))

    try:
        r_gap_prob, _ = pearsonr(gaps, probs)
    except:
        r_gap_prob = 0.0
    try:
        r_gap_logitprob, _ = pearsonr(gaps, prob_logit)
    except:
        r_gap_logitprob = 0.0

    # 2. Non-invariants: format/content direction roles
    punct_fraction, _ = identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=K)
    format_dirs = set(np.argsort(punct_fraction)[-5:])
    content_dirs = [k for k in range(K) if k not in format_dirs]

    format_gaps = np.array([float(np.sum(d["contrib_k"][list(format_dirs)])) for d in all_data])
    content_gaps = np.array([float(np.sum(d["contrib_k"][content_dirs])) for d in all_data])

    try:
        r_format_prob, _ = pearsonr(format_gaps, probs)
    except:
        r_format_prob = 0.0
    try:
        r_content_prob, _ = pearsonr(content_gaps, probs)
    except:
        r_content_prob = 0.0

    # 3. Spectral concentration (ratio50)
    c_k_matrix = np.array([d["c_k"] for d in all_data])
    h_energy = c_k_matrix ** 2
    total_energy = np.sum(h_energy, axis=1) + 1e-10
    ratio50 = np.sum(h_energy[:, :50], axis=1) / total_energy
    ratio10 = np.sum(h_energy[:, :10], axis=1) / total_energy

    try:
        r_ratio50_prob, _ = pearsonr(ratio50, probs)
    except:
        r_ratio50_prob = 0.0
    try:
        r_ratio10_prob, _ = pearsonr(ratio10, probs)
    except:
        r_ratio10_prob = 0.0

    # 4. Sign statistics
    sign_products = np.array([np.sign(d["c_k"]) * np.sign(d["Delta_k"]) for d in all_data])
    sign_agreement = np.mean(sign_products > 0, axis=0)

    results["invariants"] = {
        "abs_decomp_mean_r": float(np.mean(r_abs_decomp)) if r_abs_decomp else 0,
        "gap_to_prob_r": float(r_gap_prob),
        "gap_to_logitprob_r": float(r_gap_logitprob),
    }
    results["non_invariants"] = {
        "format_gap_to_prob_r": float(r_format_prob),
        "content_gap_to_prob_r": float(r_content_prob),
        "ratio50_to_prob_r": float(r_ratio50_prob),
        "ratio10_to_prob_r": float(r_ratio10_prob),
    }
    results["sign_stats"] = {
        "mean_agreement": float(np.mean(sign_agreement)),
        "max_agreement": float(np.max(sign_agreement)),
        "min_agreement": float(np.min(sign_agreement)),
    }
    results["spectral_stats"] = {
        "mean_ratio50": float(np.mean(ratio50)),
        "mean_ratio10": float(np.mean(ratio10)),
    }

    # 5. Top contributing directions
    mean_abs_contrib = np.mean(np.abs(np.array([d["contrib_k"] for d in all_data])), axis=0)
    top_dirs = np.argsort(mean_abs_contrib)[-10:][::-1]

    results["top_contributing_directions"] = {
        "top10_dirs": top_dirs.tolist(),
        "top10_contribution": mean_abs_contrib[top_dirs].tolist(),
        "is_format": [int(d in format_dirs) for d in top_dirs],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p600.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  P600 Results for {model_name}:")
    print(f"  Invariants: abs_decomp r={np.mean(r_abs_decomp):.4f}, gap->prob r={r_gap_prob:.4f}, gap->logitprob r={r_gap_logitprob:.4f}")
    print(f"  Non-invariants: format->prob r={r_format_prob:.4f}, content->prob r={r_content_prob:.4f}")
    print(f"  Sign agreement: mean={np.mean(sign_agreement):.4f}")
    print(f"  Ratio50->prob r={r_ratio50_prob:.4f}")
    return results


EXPERIMENTS = {
    "p595": run_p595,
    "p596": run_p596,
    "p597": run_p597,
    "p598": run_p598,
    "p599": run_p599,
    "p600": run_p600,
}


def main():
    parser = argparse.ArgumentParser(description="Phase CXXXVII: New causal chain architecture")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()

    output_dir = "results/phase_cxxxvii"

    print(f"\n{'#'*70}")
    print(f"# Phase CXXXVII: {args.experiment.upper()} -- {args.model}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)

    start_time = time.time()
    result = EXPERIMENTS[args.experiment](model, tokenizer, device, model_info, TEST_TEXTS, output_dir)
    elapsed = time.time() - start_time

    print(f"\n  Elapsed: {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

    print(f"\n  Done! Result saved to {output_dir}/{args.model}_{args.experiment}.json")


if __name__ == "__main__":
    main()
