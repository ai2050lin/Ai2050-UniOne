"""
Phase CXXXVI: 频谱→语言行为因果链的根本重构 (P589-P594)
=========================================================
核心洞察: 符号(sign)是关键噪声源, |c_k|*|Delta_k|精确预测|contrib_k|(r=0.999)
  P589: 加法传播方程的精确形式 — 用MLP增量预测delta(l)
  P590: 负相关之谜 — 去除格式方向后频谱→prob是否转正?
  P591: Delta_k的跨文本稳定性 — Delta_k条件期望模型
  P592: |c_k|*|Delta_k|分解的因果链 — 用频谱力学+ W_U结构预测|contrib_k|
  P593: 统一语言编码方程v2 — 符号预测是关键
  P594: 从logit_gap到prob的非线性映射 — softmax放大效应

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

# ===================== 210 large-scale test texts =====================
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
    "The therapist worked with the patient to address underlying emotional issues",
    "The psychologist conducted a cognitive assessment to evaluate memory and attention",
    "The psychiatrist prescribed medication to help manage the symptoms of the disorder",
    "The neurologist ordered an MRI scan to investigate the cause of the recurring headaches",
    "The surgeon performed the delicate operation with precision and extraordinary skill",
    "The physician reviewed the test results and recommended a course of treatment",
    "The pharmacist dispensed the medication and explained the proper dosage instructions",
    "The dentist examined the X-ray and identified a cavity that needed immediate attention",
    "The optometrist tested the patient's vision and prescribed corrective lenses",
    "The chiropractor adjusted the spine to alleviate the chronic back pain",
    "The physical therapist designed a rehabilitation program to restore mobility after surgery",
    "The nutritionist developed a meal plan that balanced taste and health requirements",
    "The dietitian calculated the daily caloric needs based on activity level and health goals",
    "The trainer supervised the workout session ensuring proper form and technique",
    "The coach analyzed the opponent's strategy and developed an effective counter-approach",
    "The referee enforced the rules consistently throughout the competitive match",
    "The judge evaluated each performance based on technical merit and artistic expression",
    "The umpire made the critical call that determined the outcome of the championship",
    "The official inspected the equipment to ensure compliance with safety regulations",
    "The inspector examined the facility and identified several areas requiring improvement",
    "The examiner administered the certification test under standardized conditions",
    "The surveyor measured the property boundaries with precision instruments",
    "The appraiser estimated the market value of the real estate based on comparable sales",
    "The evaluator assessed the program's effectiveness using both quantitative and qualitative metrics",
    "The reviewer provided constructive feedback to help improve the quality of the manuscript",
    "The critic offered a balanced assessment acknowledging both strengths and weaknesses",
    "The commentator provided real-time analysis of the unfolding events",
    "The reporter investigated the story from multiple angles to ensure accuracy",
    "The correspondent filed the report from the scene of the breaking news event",
    "The journalist interviewed key witnesses and verified facts before publishing the article",
]


# Punctuation/format tokens for filtering
FORMAT_TOKEN_IDS = {
    "qwen3": set(),
    "glm4": set(),
    "deepseek7b": set(),
}


def compute_svd_W_U(W_U, k=100):
    """Compute SVD of W_U^T to get W_U row space basis"""
    W_U_T = W_U.T.astype(np.float32)
    k = min(k, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    k = max(k, 1)
    U_wut, s_wut, Vt_wut = svds(W_U_T, k=k)
    idx = np.argsort(s_wut)[::-1]
    U_wut = U_wut[:, idx]
    s_wut = s_wut[idx]
    Vt_wut = Vt_wut[idx, :]
    return U_wut, s_wut, Vt_wut


def identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=100):
    """Identify which SVD directions correspond to punctuation/format tokens"""
    # Project each token's W_U row onto SVD directions
    W_U_proj = U_wut.T @ W_U.T  # [K, vocab_size]
    W_U_energy = W_U_proj ** 2  # [K, vocab_size]
    
    # Find punctuation tokens
    punct_tokens = set()
    for tid in range(W_U.shape[0]):
        try:
            token_str = tokenizer.decode([tid])
            # Check if it's primarily punctuation/whitespace
            stripped = token_str.strip()
            if stripped and not stripped[0].isalpha() and not stripped[0].isdigit():
                punct_tokens.add(tid)
        except:
            pass
    
    # For each direction, compute fraction of energy from punctuation tokens
    punct_fraction = np.zeros(k)
    total_energy = np.sum(W_U_energy, axis=1) + 1e-10
    for tid in punct_tokens:
        punct_fraction += W_U_energy[:, tid]
    punct_fraction /= total_energy
    
    return punct_fraction, punct_tokens


def run_p589(model, tokenizer, device, model_info, texts, output_dir):
    """P589: Additive propagation equation - MLP increment predicts delta(l)"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P589: Additive propagation -- MLP increment predicts delta(l) -- {model_name}")
    print(f"{'='*60}")
    
    results = {
        "model": model_name, "experiment": "p589", "n_texts": len(texts),
        "n_layers": n_layers, "additive_verification": {},
        "mlp_increment_analysis": {}
    }
    
    # Collect layer-wise logit_gaps and MLP increments
    all_layer_gaps = {l: [] for l in range(n_layers)}
    all_deltas = {l: [] for l in range(1, n_layers)}
    all_probs = []
    
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits_final = outputs.logits[0, -1].float().cpu().numpy()
        
        top1_idx = np.argmax(logits_final)
        top2_idx = np.argsort(logits_final)[-2]
        top1_prob = float(np.exp(logits_final[top1_idx]) / np.sum(np.exp(logits_final)))
        all_probs.append(top1_prob)
        
        # Compute logit_gap at each layer
        for l in range(n_layers):
            with torch.no_grad():
                h_l = outputs.hidden_states[l][0, -1].float().cpu().numpy()
            
            logits_l = W_U @ h_l
            gap_l = float(logits_l[top1_idx] - logits_l[top2_idx])
            all_layer_gaps[l].append(gap_l)
        
        # Compute deltas: delta(l) = gap(l) - gap(l-1)
        for l in range(1, n_layers):
            delta_l = all_layer_gaps[l][-1] - all_layer_gaps[l-1][-1]
            all_deltas[l].append(delta_l)
    
    # Verify additive model: gap_final = gap_0 + sum(delta_l)
    gaps_0 = np.array(all_layer_gaps[0])
    gaps_final = np.array(all_layer_gaps[n_layers-1])
    total_deltas = np.zeros(len(texts))
    for l in range(1, n_layers):
        total_deltas += np.array(all_deltas[l])
    
    predicted_gaps = gaps_0 + total_deltas
    try:
        r_additive, _ = pearsonr(predicted_gaps, gaps_final)
    except:
        r_additive = 0.0
    
    results["additive_verification"] = {
        "r_predicted_vs_actual": float(r_additive),
        "mean_gap_0": float(np.mean(gaps_0)),
        "mean_gap_final": float(np.mean(gaps_final)),
        "mean_total_delta": float(np.mean(total_deltas)),
    }
    
    # Analyze delta(l) properties
    delta_stats = {}
    for l in range(1, n_layers):
        d = np.array(all_deltas[l])
        delta_stats[f"L{l}"] = {
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
        }
    
    # Layer-wise correlation: delta(l) vs gap(l-1)
    delta_gap_corr = {}
    for l in range(1, n_layers):
        d = np.array(all_deltas[l])
        g_prev = np.array(all_layer_gaps[l-1])
        try:
            r, _ = pearsonr(d, g_prev)
        except:
            r = 0.0
        delta_gap_corr[f"L{l-1}_to_L{l}"] = float(r)
    
    # Cumulative delta analysis: which layers contribute most?
    abs_delta_sum = np.zeros(n_layers - 1)
    for l in range(1, n_layers):
        abs_delta_sum[l-1] = np.mean(np.abs(np.array(all_deltas[l])))
    
    # Layer groups
    early_layers = list(range(1, min(n_layers, n_layers//3)))
    mid_layers = list(range(n_layers//3, 2*n_layers//3))
    late_layers = list(range(2*n_layers//3, n_layers))
    
    results["mlp_increment_analysis"] = {
        "delta_stats": delta_stats,
        "delta_gap_correlation": delta_gap_corr,
        "mean_abs_delta_by_layer": {f"L{l+1}": float(abs_delta_sum[l]) for l in range(n_layers-1)},
        "early_layer_mean_abs_delta": float(np.mean(abs_delta_sum[:len(early_layers)])) if early_layers else 0,
        "mid_layer_mean_abs_delta": float(np.mean(abs_delta_sum[len(early_layers):len(early_layers)+len(mid_layers)])) if mid_layers else 0,
        "late_layer_mean_abs_delta": float(np.mean(abs_delta_sum[-len(late_layers):])) if late_layers else 0,
    }
    
    # Test: can we predict delta(l) from gap(l-1)?
    # delta(l) = a * gap(l-1) + b (linear model)
    delta_predict = {}
    for l in [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if l < 1 or l >= n_layers:
            continue
        d = np.array(all_deltas[l])
        g_prev = np.array(all_layer_gaps[l-1])
        try:
            r, _ = pearsonr(d, g_prev)
        except:
            r = 0.0
        delta_predict[f"L{l}"] = float(r)
    
    results["delta_from_gap_prediction"] = delta_predict
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p589.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P589 Results for {model_name}:")
    print(f"  Additive model r = {r_additive:.4f}")
    print(f"  Mean gap_0 = {np.mean(gaps_0):.4f}, Mean gap_final = {np.mean(gaps_final):.4f}")
    print(f"  Early delta mean = {results['mlp_increment_analysis']['early_layer_mean_abs_delta']:.4f}")
    print(f"  Mid delta mean = {results['mlp_increment_analysis']['mid_layer_mean_abs_delta']:.4f}")
    print(f"  Late delta mean = {results['mlp_increment_analysis']['late_layer_mean_abs_delta']:.4f}")
    
    return results


def run_p590(model, tokenizer, device, model_info, texts, output_dir):
    """P590: Negative correlation mystery -- removing format directions"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P590: Negative correlation mystery -- format direction removal -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    # Identify format directions
    punct_fraction, punct_tokens = identify_format_directions(W_U, U_wut, s_wut, tokenizer, k=K)
    
    # Define format directions (top-5 by punctuation fraction)
    format_dirs = np.argsort(punct_fraction)[-5:]
    content_dirs = np.array([k for k in range(K) if k not in format_dirs])
    
    print(f"  Format directions: {format_dirs.tolist()}")
    print(f"  Format direction punct fractions: {punct_fraction[format_dirs].tolist()}")
    
    results = {
        "model": model_name, "experiment": "p590", "n_texts": len(texts),
        "format_directions": format_dirs.tolist(),
        "format_punct_fraction": punct_fraction[format_dirs].tolist(),
    }
    
    all_gaps = {"full": [], "format_only": [], "content_only": []}
    all_probs = []
    all_spectral_features = {"alpha": [], "ratio50": [], "ratio10": []}
    
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
        
        all_gaps["full"].append(logit_gap)
        all_probs.append(top1_prob)
        
        # Compute c_k and Delta_k
        c_k = U_wut.T @ h  # [K]
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff  # [K]
        contrib_k = c_k * Delta_k  # [K]
        
        # Split contributions
        format_gap = float(np.sum(contrib_k[format_dirs]))
        content_gap = float(np.sum(contrib_k[content_dirs]))
        
        all_gaps["format_only"].append(format_gap)
        all_gaps["content_only"].append(content_gap)
        
        # Spectral features
        h_proj = U_wut.T @ h  # [K]
        h_energy = h_proj ** 2
        total_energy = np.sum(h_energy) + 1e-10
        
        # Alpha: power law fit
        log_s = np.log10(s_wut + 1e-10)
        log_rank = np.log10(np.arange(1, K+1))
        try:
            alpha_fit = np.polyfit(log_rank, log_s, 1)
            all_spectral_features["alpha"].append(float(alpha_fit[0]))
        except:
            all_spectral_features["alpha"].append(0.0)
        
        all_spectral_features["ratio50"].append(float(np.sum(h_energy[:50]) / total_energy))
        all_spectral_features["ratio10"].append(float(np.sum(h_energy[:10]) / total_energy))
    
    # Correlations
    gaps = {k: np.array(v) for k, v in all_gaps.items()}
    probs = np.array(all_probs)
    
    corr_results = {}
    for gap_type in ["full", "format_only", "content_only"]:
        try:
            r_gap_prob, _ = pearsonr(gaps[gap_type], probs)
        except:
            r_gap_prob = 0.0
        corr_results[f"{gap_type}_to_prob"] = float(r_gap_prob)
    
    for feat in ["alpha", "ratio50", "ratio10"]:
        feat_vals = np.array(all_spectral_features[feat])
        for gap_type in ["full", "format_only", "content_only"]:
            try:
                r, _ = pearsonr(feat_vals, gaps[gap_type])
            except:
                r = 0.0
            corr_results[f"{feat}_to_{gap_type}_gap"] = float(r)
        try:
            r, _ = pearsonr(feat_vals, probs)
        except:
            r = 0.0
        corr_results[f"{feat}_to_prob"] = float(r)
    
    results["correlations"] = corr_results
    results["gap_stats"] = {
        "full_mean": float(np.mean(gaps["full"])),
        "format_mean": float(np.mean(gaps["format_only"])),
        "content_mean": float(np.mean(gaps["content_only"])),
        "format_fraction": float(np.mean(np.abs(gaps["format_only"])) / (np.mean(np.abs(gaps["full"])) + 1e-10)),
    }
    
    # Content-only spectral->prob: does removing format fix negative correlation?
    content_gap = gaps["content_only"]
    content_ratio50 = np.array(all_spectral_features["ratio50"])
    
    # Compute content-only alpha (recompute without format directions)
    try:
        r_content_ratio50_prob, _ = pearsonr(content_ratio50, probs)
    except:
        r_content_ratio50_prob = 0.0
    
    results["content_only_test"] = {
        "content_gap_to_prob": corr_results.get("content_only_to_prob", 0.0),
        "format_gap_to_prob": corr_results.get("format_only_to_prob", 0.0),
        "full_gap_to_prob": corr_results.get("full_to_prob", 0.0),
        "ratio50_to_prob": corr_results.get("ratio50_to_prob", 0.0),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p590.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P590 Results for {model_name}:")
    print(f"  Format gap->prob r = {corr_results.get('format_only_to_prob', 0):.4f}")
    print(f"  Content gap->prob r = {corr_results.get('content_only_to_prob', 0):.4f}")
    print(f"  Full gap->prob r = {corr_results.get('full_to_prob', 0):.4f}")
    print(f"  Format fraction of gap = {results['gap_stats']['format_fraction']:.4f}")
    
    return results


def run_p591(model, tokenizer, device, model_info, texts, output_dir):
    """P591: Delta_k cross-text stability -- conditional expectation model"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P591: Delta_k cross-text stability -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {
        "model": model_name, "experiment": "p591", "n_texts": len(texts),
    }
    
    # Collect Delta_k across texts
    all_Delta_k = []
    all_top1_ids = []
    all_top2_ids = []
    all_c_k = []
    all_logit_gaps = []
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
        
        all_Delta_k.append(Delta_k)
        all_top1_ids.append(top1_idx)
        all_top2_ids.append(top2_idx)
        all_c_k.append(c_k)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
    
    all_Delta_k = np.array(all_Delta_k)  # [n_texts, K]
    all_c_k = np.array(all_c_k)
    
    # 1. Cross-text stability of Delta_k
    # Var(Delta_k) across texts
    delta_mean = np.mean(all_Delta_k, axis=0)  # [K]
    delta_std = np.std(all_Delta_k, axis=0)  # [K]
    delta_cv = delta_std / (np.abs(delta_mean) + 1e-10)  # coefficient of variation
    
    results["delta_k_stability"] = {
        "mean_abs_delta": float(np.mean(np.abs(delta_mean))),
        "mean_delta_std": float(np.mean(delta_std)),
        "mean_cv": float(np.mean(delta_cv)),
        "most_stable_dirs": np.argsort(delta_cv)[:10].tolist(),
        "least_stable_dirs": np.argsort(delta_cv)[-10:].tolist(),
    }
    
    # 2. Delta_k conditional on top1 token identity
    # Group by top1 token, compute Delta_k conditional expectation
    top1_groups = defaultdict(list)
    for i, tid in enumerate(all_top1_ids):
        top1_groups[tid].append(i)
    
    # Only use groups with >= 3 samples
    delta_cond_var = []
    delta_total_var = np.var(all_Delta_k, axis=0)  # [K]
    
    group_sizes = []
    for tid, indices in top1_groups.items():
        if len(indices) >= 3:
            group_Delta = all_Delta_k[indices]
            group_var = np.var(group_Delta, axis=0)
            delta_cond_var.append(group_var)
            group_sizes.append(len(indices))
    
    if delta_cond_var:
        avg_cond_var = np.mean(delta_cond_var, axis=0)
        variance_reduction = 1.0 - avg_cond_var / (delta_total_var + 1e-10)
        results["delta_k_conditional"] = {
            "n_groups_with_3plus": len(delta_cond_var),
            "mean_variance_reduction": float(np.mean(variance_reduction)),
            "variance_reduction_top5": variance_reduction[:5].tolist(),
        }
    
    # 3. Sign stability of Delta_k
    sign_consistency = np.zeros(K)
    for k in range(K):
        signs = np.sign(all_Delta_k[:, k])
        pos_frac = np.mean(signs > 0)
        sign_consistency[k] = max(pos_frac, 1 - pos_frac)
    
    results["sign_consistency"] = {
        "mean": float(np.mean(sign_consistency)),
        "top5_dirs": np.argsort(sign_consistency)[-5:].tolist(),
        "bottom5_dirs": np.argsort(sign_consistency)[:5].tolist(),
        "above_07_count": int(np.sum(sign_consistency > 0.7)),
    }
    
    # 4. Using mean Delta_k as predictor
    predicted_gap = np.sum(all_c_k * delta_mean[np.newaxis, :], axis=1)
    actual_gap = np.array(all_logit_gaps)
    try:
        r_mean_delta, _ = pearsonr(predicted_gap, actual_gap)
    except:
        r_mean_delta = 0.0
    
    # Using actual Delta_k (perfect predictor baseline)
    predicted_gap_actual = np.sum(all_c_k * all_Delta_k, axis=1)
    try:
        r_actual_delta, _ = pearsonr(predicted_gap_actual, actual_gap)
    except:
        r_actual_delta = 0.0
    
    results["delta_k_prediction"] = {
        "mean_delta_r": float(r_mean_delta),
        "actual_delta_r": float(r_actual_delta),
        "delta_mean_norm": float(np.linalg.norm(delta_mean)),
        "delta_std_mean_norm": float(np.mean(delta_std)),
    }
    
    # 5. Sign(c_k) * sign(mean_Delta_k) as predictor
    sign_contrib = np.sign(all_c_k) * np.sign(delta_mean[np.newaxis, :]) * np.abs(all_c_k) * np.abs(delta_mean[np.newaxis, :])
    predicted_gap_sign = np.sum(sign_contrib, axis=1)
    try:
        r_sign_pred, _ = pearsonr(predicted_gap_sign, actual_gap)
    except:
        r_sign_pred = 0.0
    
    results["sign_prediction"] = {
        "sign_mean_delta_r": float(r_sign_pred),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p591.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P591 Results for {model_name}:")
    print(f"  Mean CV of Delta_k = {np.mean(delta_cv):.4f}")
    print(f"  Sign consistency mean = {np.mean(sign_consistency):.4f}")
    print(f"  Mean Delta_k->gap r = {r_mean_delta:.4f}")
    print(f"  Actual Delta_k->gap r = {r_actual_delta:.4f}")
    print(f"  Sign prediction r = {r_sign_pred:.4f}")
    
    return results


def run_p592(model, tokenizer, device, model_info, texts, output_dir):
    """P592: |c_k|*|Delta_k| decomposition causal chain"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P592: |c_k|*|Delta_k| decomposition causal chain -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {
        "model": model_name, "experiment": "p592", "n_texts": len(texts),
    }
    
    all_c_k = []
    all_Delta_k = []
    all_logit_gaps = []
    all_probs = []
    all_spectral_features = {"alpha": [], "ratio50": []}
    
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
        
        all_c_k.append(c_k)
        all_Delta_k.append(Delta_k)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
        
        # Spectral features
        h_energy = c_k ** 2
        total_energy = np.sum(h_energy) + 1e-10
        all_spectral_features["ratio50"].append(float(np.sum(h_energy[:50]) / total_energy))
        
        log_s = np.log10(s_wut + 1e-10)
        log_rank = np.log10(np.arange(1, K+1))
        try:
            alpha_fit = np.polyfit(log_rank, log_s, 1)
            all_spectral_features["alpha"].append(float(alpha_fit[0]))
        except:
            all_spectral_features["alpha"].append(0.0)
    
    all_c_k = np.array(all_c_k)  # [n_texts, K]
    all_Delta_k = np.array(all_Delta_k)
    all_logit_gaps = np.array(all_logit_gaps)
    all_probs = np.array(all_probs)
    
    # 1. |c_k|*|Delta_k| -> |contrib_k| (verification)
    all_contrib = all_c_k * all_Delta_k  # [n_texts, K]
    abs_c_k = np.abs(all_c_k)
    abs_Delta_k = np.abs(all_Delta_k)
    abs_contrib = np.abs(all_contrib)
    
    # Per-direction correlation
    r_abs_decomp = []
    for k in range(K):
        try:
            r, _ = pearsonr(abs_c_k[:, k] * abs_Delta_k[:, k], abs_contrib[:, k])
            r_abs_decomp.append(r)
        except:
            r_abs_decomp.append(0.0)
    
    results["abs_decomposition"] = {
        "mean_r": float(np.mean(r_abs_decomp)),
        "min_r": float(np.min(r_abs_decomp)),
    }
    
    # 2. Can we predict |c_k| from spectral features?
    # Use ratio50 to predict sum|c_k|^2
    sum_ck2 = np.sum(all_c_k ** 2, axis=1)  # [n_texts]
    ratio50 = np.array(all_spectral_features["ratio50"])
    alpha_vals = np.array(all_spectral_features["alpha"])
    
    try:
        r_ratio50_ck2, _ = pearsonr(ratio50, sum_ck2)
    except:
        r_ratio50_ck2 = 0.0
    
    try:
        r_alpha_ck2, _ = pearsonr(alpha_vals, sum_ck2)
    except:
        r_alpha_ck2 = 0.0
    
    results["spectral_to_ck2"] = {
        "ratio50_to_sum_ck2": float(r_ratio50_ck2),
        "alpha_to_sum_ck2": float(r_alpha_ck2),
    }
    
    # 3. Can we predict |Delta_k| from W_U structure?
    # W_U_diff norm should predict Delta_k magnitude
    mean_abs_Delta = np.mean(abs_Delta_k, axis=0)  # [K]
    s_k = s_wut  # W_U singular values
    
    try:
        r_s_delta, _ = pearsonr(s_k, mean_abs_Delta)
    except:
        r_s_delta = 0.0
    
    results["wu_structure_to_delta"] = {
        "s_k_vs_mean_abs_Delta_k_r": float(r_s_delta),
    }
    
    # 4. Two-step prediction: |c_k|*|Delta_k| from spectral + W_U
    # Step 1: Predict |c_k| using spectral model
    # Simple model: |c_k| proportional to s_k^alpha (power law)
    # Step 2: Predict |Delta_k| using W_U structure
    # Simple model: |Delta_k| proportional to s_k
    
    # Use global mean |c_k| and |Delta_k| as predictors
    mean_abs_c = np.mean(abs_c_k, axis=0)  # [K]
    
    # Predict |contrib_k| = mean|c_k| * |Delta_k| (using actual Delta_k)
    pred_abs_contrib_mean_c = mean_abs_c[np.newaxis, :] * abs_Delta_k
    pred_gap_from_mean_c = np.sum(np.sign(all_contrib) * pred_abs_contrib_mean_c, axis=1)
    try:
        r_mean_c_pred, _ = pearsonr(pred_gap_from_mean_c, all_logit_gaps)
    except:
        r_mean_c_pred = 0.0
    
    # Predict |contrib_k| = |c_k| * mean|Delta_k| (using actual c_k)
    pred_abs_contrib_mean_d = abs_c_k * mean_abs_Delta[np.newaxis, :]
    pred_gap_from_mean_d = np.sum(np.sign(all_contrib) * pred_abs_contrib_mean_d, axis=1)
    try:
        r_mean_d_pred, _ = pearsonr(pred_gap_from_mean_d, all_logit_gaps)
    except:
        r_mean_d_pred = 0.0
    
    # Both mean
    pred_abs_contrib_both = mean_abs_c[np.newaxis, :] * mean_abs_Delta[np.newaxis, :]
    pred_gap_from_both = np.sum(np.sign(all_contrib) * pred_abs_contrib_both, axis=1)
    try:
        r_both_pred, _ = pearsonr(pred_gap_from_both, all_logit_gaps)
    except:
        r_both_pred = 0.0
    
    results["two_step_prediction"] = {
        "mean_c_actual_d_r": float(r_mean_c_pred),
        "actual_c_mean_d_r": float(r_mean_d_pred),
        "mean_c_mean_d_r": float(r_both_pred),
    }
    
    # 5. Sum |c_k|^2 * |Delta_k|^2 -> logit_var
    logit_var = np.var(all_logit_gaps)
    sum_ck2_dk2 = np.mean(np.sum(all_c_k**2 * all_Delta_k**2, axis=1))
    results["ck2_dk2_to_logitvar"] = {
        "sum_ck2_dk2_mean": float(sum_ck2_dk2),
        "logit_var": float(logit_var),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p592.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P592 Results for {model_name}:")
    print(f"  |c_k|*|Delta_k|->|contrib_k| mean r = {np.mean(r_abs_decomp):.4f}")
    print(f"  ratio50->sum|c_k|^2 r = {r_ratio50_ck2:.4f}")
    print(f"  s_k->mean|Delta_k| r = {r_s_delta:.4f}")
    print(f"  mean|c_k|+actual|Delta_k|->gap r = {r_mean_c_pred:.4f}")
    print(f"  actual|c_k|+mean|Delta_k|->gap r = {r_mean_d_pred:.4f}")
    print(f"  mean|c_k|+mean|Delta_k|->gap r = {r_both_pred:.4f}")
    
    return results


def run_p593(model, tokenizer, device, model_info, texts, output_dir):
    """P593: Unified language encoding equation v2 -- sign prediction"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P593: Unified encoding equation v2 -- sign prediction -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {
        "model": model_name, "experiment": "p593", "n_texts": len(texts),
    }
    
    all_c_k = []
    all_Delta_k = []
    all_logit_gaps = []
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
        
        all_c_k.append(c_k)
        all_Delta_k.append(Delta_k)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
    
    all_c_k = np.array(all_c_k)
    all_Delta_k = np.array(all_Delta_k)
    all_logit_gaps = np.array(all_logit_gaps)
    all_probs = np.array(all_probs)
    
    # 1. Sign decomposition analysis
    # logit_gap = Sum sign(c_k) * sign(Delta_k) * |c_k| * |Delta_k|
    sign_ck = np.sign(all_c_k)
    sign_dk = np.sign(all_Delta_k)
    abs_ck = np.abs(all_c_k)
    abs_dk = np.abs(all_Delta_k)
    
    # Exact decomposition
    sign_product = sign_ck * sign_dk  # [n_texts, K]
    
    # Sign agreement per direction
    sign_agreement = np.mean(sign_product > 0, axis=0)  # [K]
    
    results["sign_agreement"] = {
        "mean": float(np.mean(sign_agreement)),
        "most_agreeing": np.argsort(sign_agreement)[-5:].tolist(),
        "least_agreeing": np.argsort(sign_agreement)[:5].tolist(),
        "above_07_count": int(np.sum(sign_agreement > 0.7)),
        "below_03_count": int(np.sum(sign_agreement < 0.3)),
    }
    
    # 2. Sign prediction: use mean sign as predictor
    # For each direction, predict sign(c_k*Delta_k) = sign(mean_sign)
    mean_sign = np.mean(sign_product, axis=0)  # [K]
    predicted_sign = np.sign(mean_sign)  # +1 or -1
    
    # Predicted gap using predicted sign
    predicted_gap = np.sum(predicted_sign[np.newaxis, :] * abs_ck * abs_dk, axis=1)
    try:
        r_pred_sign, _ = pearsonr(predicted_gap, all_logit_gaps)
    except:
        r_pred_sign = 0.0
    
    # 3. Oracle sign: using actual sign (upper bound)
    oracle_gap = np.sum(sign_product * abs_ck * abs_dk, axis=1)
    # This should be exact = all_logit_gaps
    try:
        r_oracle, _ = pearsonr(oracle_gap, all_logit_gaps)
    except:
        r_oracle = 0.0
    
    # 4. Random sign baseline
    n_random = 100
    r_random = []
    for _ in range(n_random):
        random_sign = np.random.choice([-1, 1], size=K)
        random_gap = np.sum(random_sign[np.newaxis, :] * abs_ck * abs_dk, axis=1)
        try:
            r, _ = pearsonr(random_gap, all_logit_gaps)
            r_random.append(r)
        except:
            pass
    
    # 5. Sign(c_k) prediction from h
    # Can we predict sign(c_k) from other spectral features?
    # Use |c_k| to predict sign(c_k): larger |c_k| -> more stable sign
    sign_stability_vs_mag = []
    for k in range(K):
        # For texts with large |c_k|, is sign more stable?
        large_ck_mask = abs_ck[:, k] > np.median(abs_ck[:, k])
        if np.sum(large_ck_mask) > 5:
            stability = max(np.mean(sign_ck[large_ck_mask, k] > 0),
                          1 - np.mean(sign_ck[large_ck_mask, k] > 0))
        else:
            stability = 0.5
        sign_stability_vs_mag.append(stability)
    
    results["sign_analysis"] = {
        "predicted_sign_r": float(r_pred_sign),
        "oracle_sign_r": float(r_oracle),
        "random_sign_r_mean": float(np.mean(r_random)) if r_random else 0,
        "random_sign_r_std": float(np.std(r_random)) if r_random else 0,
        "sign_stability_vs_magnitude_mean": float(np.mean(sign_stability_vs_mag)),
    }
    
    # 6. Weighted sign prediction: weight by sign agreement
    # If a direction has high sign agreement, use its predicted sign; otherwise, set weight=0
    weights = np.abs(mean_sign)  # Higher agreement -> higher weight
    weighted_pred_gap = np.sum(
        predicted_sign[np.newaxis, :] * weights[np.newaxis, :] * abs_ck * abs_dk, axis=1
    )
    try:
        r_weighted_pred, _ = pearsonr(weighted_pred_gap, all_logit_gaps)
    except:
        r_weighted_pred = 0.0
    
    results["weighted_sign_prediction"] = {
        "r": float(r_weighted_pred),
    }
    
    # 7. Top-N sign-consistent directions only
    top_n_results = {}
    for n in [5, 10, 20, 30]:
        most_consistent = np.argsort(np.abs(mean_sign))[-n:]
        pred_gap_n = np.sum(
            predicted_sign[most_consistent][np.newaxis, :] * abs_ck[:, most_consistent] * abs_dk[:, most_consistent], axis=1
        )
        try:
            r, _ = pearsonr(pred_gap_n, all_logit_gaps)
        except:
            r = 0.0
        top_n_results[f"top{n}"] = float(r)
    
    results["top_n_consistent"] = top_n_results
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p593.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P593 Results for {model_name}:")
    print(f"  Sign agreement mean = {np.mean(sign_agreement):.4f}")
    print(f"  Predicted sign->gap r = {r_pred_sign:.4f}")
    print(f"  Oracle sign->gap r = {r_oracle:.4f}")
    print(f"  Random sign->gap r = {np.mean(r_random):.4f} +/- {np.std(r_random):.4f}")
    print(f"  Weighted sign->gap r = {r_weighted_pred:.4f}")
    print(f"  Top-N consistent: {top_n_results}")
    
    return results


def run_p594(model, tokenizer, device, model_info, texts, output_dir):
    """P594: From logit_gap to prob -- softmax amplification"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P594: logit_gap to prob -- softmax amplification -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {
        "model": model_name, "experiment": "p594", "n_texts": len(texts),
    }
    
    all_logit_gaps = []
    all_probs = []
    all_all_logits = []
    all_logit_vars = []
    all_logit_maxes = []
    
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
        
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
        
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
        all_all_logits.append(logits)
        all_logit_vars.append(float(np.var(logits)))
        all_logit_maxes.append(float(logits[top1_idx]))
    
    all_logit_gaps = np.array(all_logit_gaps)
    all_probs = np.array(all_probs)
    all_logit_vars = np.array(all_logit_vars)
    all_logit_maxes = np.array(all_logit_maxes)
    
    # 1. Direct logit_gap -> prob
    try:
        r_gap_prob, _ = pearsonr(all_logit_gaps, all_probs)
    except:
        r_gap_prob = 0.0
    
    # 2. logit_gap -> log(prob/(1-prob)) = logit of prob
    prob_clipped = np.clip(all_probs, 0.001, 0.999)
    prob_logit = np.log(prob_clipped / (1 - prob_clipped))
    try:
        r_gap_logit, _ = pearsonr(all_logit_gaps, prob_logit)
    except:
        r_gap_logit = 0.0
    
    # 3. Softmax amplification: prob = exp(logit1) / sum(exp(logit_i))
    # prob depends on: (a) logit_gap, (b) logit_max (temperature effect)
    # prob ≈ sigmoid(logit_gap - log(sum_{i>2} exp(logit_i - logit_1)))
    
    # 4. logit_var -> prob
    try:
        r_var_prob, _ = pearsonr(all_logit_vars, all_probs)
    except:
        r_var_prob = 0.0
    
    # 5. logit_max -> prob
    try:
        r_max_prob, _ = pearsonr(all_logit_maxes, all_probs)
    except:
        r_max_prob = 0.0
    
    # 6. logit_max + logit_gap -> prob (multiple regression)
    from numpy.linalg import lstsq
    X = np.column_stack([all_logit_gaps, all_logit_maxes, np.ones(len(texts))])
    try:
        coeffs, _, _, _ = lstsq(X, all_probs, rcond=None)
        pred_prob = X @ coeffs
        r_multi = float(np.corrcoef(pred_prob, all_probs)[0, 1])
    except:
        coeffs = [0, 0, 0]
        r_multi = 0.0
    
    # 7. Softmax simulation: how does prob change with gap for different temperatures?
    # T = logit_max (acts as temperature)
    gaps_sim = np.linspace(0.1, 20, 100)
    prob_by_temp = {}
    for temp_logit in [5, 10, 15, 20, 25]:
        probs_sim = []
        for g in gaps_sim:
            logit1 = temp_logit + g/2
            logit2 = temp_logit - g/2
            p = np.exp(logit1) / (np.exp(logit1) + np.exp(logit2) + np.exp(temp_logit) * 1000)
            probs_sim.append(p)
        prob_by_temp[f"temp_{temp_logit}"] = {
            "probs_at_gap1": float(probs_sim[4]),
            "probs_at_gap5": float(probs_sim[24]),
            "probs_at_gap10": float(probs_sim[49]),
        }
    
    results["gap_to_prob"] = {
        "direct_r": float(r_gap_prob),
        "gap_to_logit_prob_r": float(r_gap_logit),
        "var_to_prob_r": float(r_var_prob),
        "max_to_prob_r": float(r_max_prob),
        "multi_regression_r": float(r_multi),
        "multi_coeffs": [float(c) for c in coeffs],
    }
    
    results["softmax_simulation"] = prob_by_temp
    
    # 8. Nonlinear transformation: prob = 1 / (1 + exp(-gap + offset))
    # Fit offset = log(sum_{i>2} exp(logit_i - logit_1))
    offsets = []
    for i in range(len(texts)):
        logits = all_all_logits[i]
        top1 = logits[np.argmax(logits)]
        # Sum of exp(logits - top1) excluding top1 and top2
        top2 = np.sort(logits)[-2]
        other_sum = 0
        for j, l in enumerate(logits):
            if l != top1 and l != top2:
                other_sum += np.exp(l - top1)
        offset = np.log(other_sum + np.exp(top2 - top1))
        offsets.append(offset)
    
    offsets = np.array(offsets)
    predicted_prob = 1.0 / (1.0 + np.exp(-all_logit_gaps + offsets))
    try:
        r_nonlinear_pred, _ = pearsonr(predicted_prob, all_probs)
    except:
        r_nonlinear_pred = 0.0
    
    results["nonlinear_transform"] = {
        "r": float(r_nonlinear_pred),
        "mean_offset": float(np.mean(offsets)),
        "std_offset": float(np.std(offsets)),
    }
    
    # 9. Quantile analysis
    n_quantiles = 5
    gap_quantiles = np.percentile(all_logit_gaps, np.linspace(0, 100, n_quantiles+1))
    quantile_results = {}
    for q in range(n_quantiles):
        mask = (all_logit_gaps >= gap_quantiles[q]) & (all_logit_gaps < gap_quantiles[q+1])
        if np.sum(mask) > 0:
            quantile_results[f"q{q}"] = {
                "mean_gap": float(np.mean(all_logit_gaps[mask])),
                "mean_prob": float(np.mean(all_probs[mask])),
                "n": int(np.sum(mask)),
            }
    
    results["quantile_analysis"] = quantile_results
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p594.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  P594 Results for {model_name}:")
    print(f"  logit_gap->prob r = {r_gap_prob:.4f}")
    print(f"  logit_gap->logit(prob) r = {r_gap_logit:.4f}")
    print(f"  logit_var->prob r = {r_var_prob:.4f}")
    print(f"  logit_max->prob r = {r_max_prob:.4f}")
    print(f"  Multi-regression r = {r_multi:.4f}")
    print(f"  Nonlinear transform r = {r_nonlinear_pred:.4f}")
    
    return results


# ===================== Main =====================
EXPERIMENTS = {
    "p589": run_p589,
    "p590": run_p590,
    "p591": run_p591,
    "p592": run_p592,
    "p593": run_p593,
    "p594": run_p594,
}


def main():
    parser = argparse.ArgumentParser(description="Phase CXXXVI: Causal chain reconstruction")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    output_dir = "results/phase_cxxxvi"
    
    print(f"\n{'#'*70}")
    print(f"# Phase CXXXVI: {args.experiment.upper()} -- {args.model}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    
    start_time = time.time()
    result = EXPERIMENTS[args.experiment](model, tokenizer, device, model_info, TEST_TEXTS, output_dir)
    elapsed = time.time() - start_time
    
    print(f"\n  Elapsed: {elapsed:.1f}s")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    print(f"\n  Done! Result saved to {output_dir}/{args.model}_{args.experiment}.json")


if __name__ == "__main__":
    main()
