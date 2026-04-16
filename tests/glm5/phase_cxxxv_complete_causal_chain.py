"""
Phase CXXXV: 语义编码的数学基础 -- 完整因果链 (P583-P588)
=========================================================
核心洞察: 方向选择效应 -> 频谱力学到语言行为的完整路径
  P583: 方向选择加权完整因果链 -- weighted_gap = Sum w_k * c_k * Delta_k
  P584: 信噪比模型 -- SNR(k) = signal(k) / noise(k), 选择SNR>1的方向
  P585: 层间logit_gap传播统一方程 -- gap(l+1) = a(l)*gap(l) + b(l)*delta_gap(l)
  P586: W_U结构->Delta_k->logit_gap的完整数学
  P587: 训练策略->W_U结构->Delta_k的因果链
  P588: 统一语言编码方程

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

# ===================== 200+ large-scale test texts =====================
TEST_TEXTS = [
    # Tech/CS (1-20)
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
    # Nature/Environment (21-40)
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
    # Social/Humanities (41-60)
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
    # Daily/Personal (61-80)
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
    # Academic/Knowledge (81-100)
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
    # More diverse (101-210)
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
    "The editor revised the text for clarity, coherence, and grammatical correctness",
    "The publisher decided to release the book in both hardcover and electronic formats",
    "The author completed the manuscript after months of research and careful writing",
    "The writer crafted a compelling narrative that resonated with readers worldwide",
       "The poet expressed profound emotion through carefully chosen metaphors and imagery",
    "The novelist developed complex characters that evolved throughout the story",
    "The playwright wrote dialogue that captured the nuances of everyday conversation",
    "The screenwriter adapted the best-selling novel into a compelling film script",
    "The lyricist wrote words that perfectly complemented the melody of the song",
    "The composer created an orchestral piece that moved audiences to standing ovation",
    "The musician practiced scales and arpeggios daily to maintain technical proficiency",
    "The singer delivered a breathtaking performance that earned a thunderous applause",
    "The dancer executed the choreography with remarkable precision and artistic flair",
    "The actor portrayed the historical figure with remarkable authenticity and depth",
    "The director guided the cast and crew through the challenging production schedule",
    "The producer secured financing for the independent film through private investors",
    "The cinematographer captured stunning visuals that enhanced the storytelling",
    "The animator brought the characters to life with fluid motion and expressive detail",
    "The designer chose a color palette that evoked the mood of the historical period",
    "The curator organized the exhibition to guide visitors through a chronological journey",
    "The collector acquired a rare painting at auction for a record-breaking price",
    "The dealer authenticated the antique vase using scientific dating methods",
    "The restorer carefully cleaned the old painting revealing vibrant colors underneath",
    "The preservative treatment protected the wooden structure from insect damage and decay",
    "The conservator repaired the torn manuscript using archival-quality materials",
    "The archivist cataloged the historical documents and stored them in climate-controlled vaults",
    "The historian analyzed primary sources to construct an accurate account of the event",
    "The archaeologist excavated the site systematically recording the location of each artifact",
    "The anthropologist observed cultural practices and documented social customs",
    "The sociologist conducted surveys to gather data on attitudes toward social change",
    "The psychologist developed a new therapeutic approach based on cognitive restructuring",
    "The economist modeled the impact of trade policy on employment and economic growth",
    "The political scientist analyzed voting patterns to predict election outcomes",
    "The philosopher examined the ethical implications of emerging technologies",
    "The theologian explored the relationship between faith and reason in religious tradition",
    "The linguist documented the grammar and vocabulary of the endangered language",
    "The literary critic analyzed the symbolic significance of recurring motifs in the novel",
    "The art historian traced the influence of Renaissance techniques on modern painting",
    "The musicologist identified the harmonic structures that defined the composer's style",
    "The classicist translated the ancient Greek text making it accessible to modern readers",
    "The orientalist studied the cultural exchange between East and West along the Silk Road",
    "The sinologist analyzed the philosophical foundations of Confucian social ethics",
]


def compute_svd_W_U(W_U, k=100):
    """Compute SVD of W_U^T to get W_U row space basis"""
    W_U_T = W_U.T.astype(np.float32)
    k = min(k, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    k = max(k, 1)
    U_wut, s_wut, Vt_wut = svds(W_U_T, k=k)
    # Sort by descending singular values
    idx = np.argsort(s_wut)[::-1]
    U_wut = U_wut[:, idx]
    s_wut = s_wut[idx]
    Vt_wut = Vt_wut[idx, :]
    return U_wut, s_wut, Vt_wut


def run_p583(model, tokenizer, device, model_info, texts, output_dir):
    """P583: Direction selection weighted complete causal chain"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    print(f"\n{'='*60}")
    print(f"P583: Direction selection weighted causal chain -- {model_name}")
    print(f"{'='*60}")
    
    # Pre-compute W_U SVD (K=100 directions)
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {
        "model": model_name, "experiment": "p583", "n_texts": len(texts),
        "K": K, "direction_selection": {}
    }
    
    all_weighted_gaps = {f"top{n}": [] for n in [1, 3, 5, 10, 20, 50]}
    all_logit_gaps = []
    all_probs = []
    all_snr_gaps = {"high_snr": [], "low_snr": [], "snr_weighted": []}
    all_alpha = []
    
    # Pre-compute per-direction SNR from pilot run
    dir_snr = np.zeros(K)
    dir_signal = np.zeros(K)
    dir_noise = np.zeros(K)
    
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
        
        # Top-1 and top-2 tokens
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
        
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
        
        # Compute c_k and Delta_k
        c_k = U_wut.T @ h  # [K]
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]  # [d_model]
        Delta_k = U_wut.T @ W_U_diff  # [K]
        
        # Per-direction contribution
        contrib_k = c_k * Delta_k  # [K]
        
        # Direction selection: sort by |contrib_k| (prediction power)
        abs_contrib = np.abs(contrib_k)
        sorted_dirs = np.argsort(abs_contrib)[::-1]
        
        for n in [1, 3, 5, 10, 20, 50]:
            selected = sorted_dirs[:n]
            weighted_gap = float(np.sum(contrib_k[selected]))
            all_weighted_gaps[f"top{n}"].append(weighted_gap)
        
        # Accumulate for SNR computation
        dir_signal += np.abs(contrib_k)
        dir_noise += c_k**2  # noise proxy: c_k^2 when Delta_k random
    
    # Compute SNR per direction
    dir_noise_avg = dir_noise / len(texts) + 1e-10
    dir_signal_avg = dir_signal / len(texts)
    dir_snr = dir_signal_avg / dir_noise_avg
    
    # Now second pass: SNR-weighted gap
    print("\n  Computing SNR-weighted gap (second pass)...")
    high_snr_dirs = np.where(dir_snr > np.median(dir_snr))[0]
    low_snr_dirs = np.where(dir_snr <= np.median(dir_snr))[0]
    
    for ti, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
        
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        
        c_k = U_wut.T @ h
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k
        
        # High-SNR and low-SNR contributions
        high_gap = float(np.sum(contrib_k[high_snr_dirs]))
        low_gap = float(np.sum(contrib_k[low_snr_dirs]))
        
        # SNR-weighted: weight each direction by its SNR
        snr_weights = dir_snr / (np.sum(dir_snr) + 1e-10)
        snr_weighted = float(np.sum(snr_weights * contrib_k))
        
        all_snr_gaps["high_snr"].append(high_gap)
        all_snr_gaps["low_snr"].append(low_gap)
        all_snr_gaps["snr_weighted"].append(snr_weighted)
    
    # ===== Compute correlations =====
    logit_gaps = np.array(all_logit_gaps)
    probs = np.array(all_probs)
    
    print("\n=== Direction Selection Weighted Gap -> logit_gap ===")
    for n in [1, 3, 5, 10, 20, 50]:
        vals = np.array(all_weighted_gaps[f"top{n}"])
        r, p = pearsonr(vals, logit_gaps)
        r2, p2 = pearsonr(np.abs(vals), probs)
        print(f"  Top-{n} dirs Sum c_k*Delta_k -> gap: r={r:.3f} (p={p:.4f}), -> prob: r={r2:.3f}")
        results["direction_selection"][f"top{n}"] = {"r_gap": float(r), "p_gap": float(p), "r_prob": float(r2)}
    
    print("\n=== SNR-based Gap -> logit_gap ===")
    for key in ["high_snr", "low_snr", "snr_weighted"]:
        vals = np.array(all_snr_gaps[key])
        r, p = pearsonr(vals, logit_gaps)
        r2, p2 = pearsonr(np.abs(vals), probs)
        print(f"  {key} -> gap: r={r:.3f} (p={p:.4f}), -> prob: r={r2:.3f}")
        results[f"snr_{key}"] = {"r_gap": float(r), "p_gap": float(p), "r_prob": float(r2)}
    
    print("\n=== SNR Distribution ===")
    print(f"  SNR range: [{dir_snr.min():.3f}, {dir_snr.max():.3f}]")
    print(f"  SNR median: {np.median(dir_snr):.3f}")
    print(f"  High-SNR dirs (>{np.median(dir_snr):.3f}): {len(high_snr_dirs)} dirs")
    print(f"  Low-SNR dirs (<={np.median(dir_snr):.3f}): {len(low_snr_dirs)} dirs")
    top_snr_dirs = np.argsort(dir_snr)[::-1][:10]
    print(f"  Top-10 SNR dirs: {top_snr_dirs.tolist()}")
    print(f"  Their SNRs: {dir_snr[top_snr_dirs].tolist()}")
    results["snr_distribution"] = {
        "snr_range": [float(dir_snr.min()), float(dir_snr.max())],
        "snr_median": float(np.median(dir_snr)),
        "n_high_snr": int(len(high_snr_dirs)),
        "n_low_snr": int(len(low_snr_dirs)),
        "top10_snr_dirs": top_snr_dirs.tolist(),
        "top10_snr_values": dir_snr[top_snr_dirs].tolist(),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p583.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p583.json")
    return results


def run_p584(model, tokenizer, device, model_info, texts, output_dir):
    """P584: SNR model -- why Top-10 > Top-50? Noise cancellation analysis"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P584: SNR model and noise cancellation -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {"model": model_name, "experiment": "p584", "n_texts": len(texts), "K": K}
    
    # Collect per-direction contributions across texts
    all_contribs = []  # [n_texts, K]
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
        contrib_k = c_k * Delta_k
        
        all_contribs.append(contrib_k)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
    
    contribs = np.array(all_contribs)  # [n_texts, K]
    logit_gaps = np.array(all_logit_gaps)
    probs = np.array(all_probs)
    
    # === 1. Signal vs Noise decomposition ===
    print("\n=== 1. Signal vs Noise Decomposition ===")
    
    # Signal: |mean(contrib_k)| across texts
    # Noise: std(contrib_k) across texts
    signal_k = np.abs(np.mean(contribs, axis=0))  # [K]
    noise_k = np.std(contribs, axis=0) + 1e-10  # [K]
    snr_k = signal_k / noise_k  # [K]
    
    # Sort by SNR
    snr_sorted = np.argsort(snr_k)[::-1]
    
    print(f"  Signal range: [{signal_k.min():.4f}, {signal_k.max():.4f}]")
    print(f"  Noise range: [{noise_k.min():.4f}, {noise_k.max():.4f}]")
    print(f"  SNR range: [{snr_k.min():.4f}, {snr_k.max():.4f}]")
    print(f"  SNR > 1 dirs: {np.sum(snr_k > 1)}/{K}")
    print(f"  SNR > 0.5 dirs: {np.sum(snr_k > 0.5)}/{K}")
    
    results["snr_stats"] = {
        "n_snr_gt_1": int(np.sum(snr_k > 1)),
        "n_snr_gt_05": int(np.sum(snr_k > 0.5)),
        "snr_range": [float(snr_k.min()), float(snr_k.max())],
    }
    
    # === 2. Cumulative contribution with SNR-ordered directions ===
    print("\n=== 2. Cumulative Contribution (SNR-ordered) ===")
    
    for n in [1, 3, 5, 10, 20, 50]:
        # SNR-ordered: select top-n by SNR
        selected = snr_sorted[:n]
        cum_contrib = np.sum(contribs[:, selected], axis=1)  # [n_texts]
        r, p = pearsonr(cum_contrib, logit_gaps)
        r2, _ = pearsonr(np.abs(cum_contrib), probs)
        print(f"  SNR-Top-{n}: Sum -> gap r={r:.3f}, -> prob r={r2:.3f}")
        results[f"snr_top{n}"] = {"r_gap": float(r), "r_prob": float(r2)}
    
    # === 3. Noise cancellation analysis ===
    print("\n=== 3. Noise Cancellation Analysis ===")
    
    # Random direction subset: what if we randomly pick N directions?
    n_random_trials = 100
    for n in [5, 10, 20, 50]:
        random_rs = []
        for _ in range(n_random_trials):
            rand_dirs = np.random.choice(K, n, replace=False)
            cum_contrib = np.sum(contribs[:, rand_dirs], axis=1)
            r, _ = pearsonr(cum_contrib, logit_gaps)
            random_rs.append(r)
        print(f"  Random-{n} dirs (mean of {n_random_trials}): r={np.mean(random_rs):.3f} +/- {np.std(random_rs):.3f}")
        results[f"random_top{n}"] = {"mean_r": float(np.mean(random_rs)), "std_r": float(np.std(random_rs))}
    
    # === 4. Contribution sign consistency ===
    print("\n=== 4. Contribution Sign Consistency ===")
    
    sign_consistency = np.mean(contribs > 0, axis=0)  # [K]: fraction of positive contributions
    # Directions with consistent sign (consistency > 0.7 or < 0.3) have high SNR
    consistent_dirs = np.where((sign_consistency > 0.7) | (sign_consistency < 0.3))[0]
    inconsistent_dirs = np.where((sign_consistency >= 0.3) & (sign_consistency <= 0.7))[0]
    
    print(f"  Consistent dirs (sign>0.7 or <0.3): {len(consistent_dirs)}/{K}")
    print(f"  Inconsistent dirs (0.3<=sign<=0.7): {len(inconsistent_dirs)}/{K}")
    
    # Consistent vs inconsistent contribution
    if len(consistent_dirs) > 0:
        cons_contrib = np.sum(contribs[:, consistent_dirs], axis=1)
        r_cons, _ = pearsonr(cons_contrib, logit_gaps)
        print(f"  Consistent dirs Sum -> gap: r={r_cons:.3f}")
    else:
        r_cons = 0.0
    if len(inconsistent_dirs) > 0:
        incons_contrib = np.sum(contribs[:, inconsistent_dirs], axis=1)
        r_incons, _ = pearsonr(incons_contrib, logit_gaps)
        print(f"  Inconsistent dirs Sum -> gap: r={r_incons:.3f}")
    else:
        r_incons = 0.0
    
    results["sign_consistency"] = {
        "n_consistent": int(len(consistent_dirs)),
        "n_inconsistent": int(len(inconsistent_dirs)),
        "r_consistent": float(r_cons),
        "r_inconsistent": float(r_incons),
        "top10_consistent_dirs": consistent_dirs[:10].tolist() if len(consistent_dirs) >= 10 else consistent_dirs.tolist(),
    }
    
    # === 5. Optimal direction subset via cross-validation ===
    print("\n=== 5. Optimal Direction Subset ===")
    
    # Try all possible subset sizes with SNR ordering
    best_n = 1
    best_r = 0
    for n in range(1, K+1):
        selected = snr_sorted[:n]
        cum_contrib = np.sum(contribs[:, selected], axis=1)
        r, _ = pearsonr(cum_contrib, logit_gaps)
        if abs(r) > abs(best_r):
            best_r = r
            best_n = n
    
    print(f"  Optimal n (SNR-ordered): {best_n}, r={best_r:.3f}")
    
    # Also try by |signal_k| ordering
    signal_sorted = np.argsort(signal_k)[::-1]
    best_n_sig = 1
    best_r_sig = 0
    for n in range(1, K+1):
        selected = signal_sorted[:n]
        cum_contrib = np.sum(contribs[:, selected], axis=1)
        r, _ = pearsonr(cum_contrib, logit_gaps)
        if abs(r) > abs(best_r_sig):
            best_r_sig = r
            best_n_sig = n
    
    print(f"  Optimal n (signal-ordered): {best_n_sig}, r={best_r_sig:.3f}")
    
    results["optimal_subset"] = {
        "snr_ordered": {"best_n": int(best_n), "best_r": float(best_r)},
        "signal_ordered": {"best_n": int(best_n_sig), "best_r": float(best_r_sig)},
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p584.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p584.json")
    return results


def run_p585(model, tokenizer, device, model_info, texts, output_dir):
    """P585: Unified inter-layer logit_gap propagation equation"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P585: Unified inter-layer logit_gap propagation -- {model_name}")
    print(f"{'='*60}")
    
    K = 50
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    # Sample layers (every 2 layers + last)
    sample_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        "model": model_name, "experiment": "p585", "n_texts": len(texts),
        "sample_layers": sample_layers
    }
    
    # Collect per-layer logit_gap
    layer_gaps = defaultdict(list)  # {layer: [gaps]}
    layer_probs = defaultdict(list)
    
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for li in sample_layers:
            h = outputs.hidden_states[li][0, -1].float().cpu().numpy()
            logits_l = h @ W_U.T
            
            top1_idx = np.argmax(logits_l)
            top2_idx = np.argsort(logits_l)[-2]
            gap = float(logits_l[top1_idx] - logits_l[top2_idx])
            prob = float(np.exp(logits_l[top1_idx]) / np.sum(np.exp(logits_l)))
            
            layer_gaps[li].append(gap)
            layer_probs[li].append(prob)
    
    # === 1. Propagation equation: gap(l+1) = a(l)*gap(l) + b(l)*delta(l) ===
    print("\n=== 1. Inter-layer Propagation Equation ===")
    
    prop_coeffs = []
    for i in range(len(sample_layers) - 1):
        l1 = sample_layers[i]
        l2 = sample_layers[i + 1]
        gaps1 = np.array(layer_gaps[l1])
        gaps2 = np.array(layer_gaps[l2])
        
        # Linear regression: gap2 = a * gap1 + b
        a = np.cov(gaps1, gaps2)[0, 1] / (np.var(gaps1) + 1e-10)
        b = np.mean(gaps2) - a * np.mean(gaps1)
        r, p = pearsonr(gaps1, gaps2)
        
        print(f"  L{l1} -> L{l2}: a={a:.3f}, b={b:.4f}, r={r:.3f}")
        prop_coeffs.append({"l1": l1, "l2": l2, "a": float(a), "b": float(b), "r": float(r)})
    
    results["propagation_coeffs"] = prop_coeffs
    
    # === 2. Early/Mid/Late phase analysis ===
    print("\n=== 2. Phase Analysis ===")
    
    n_third = len(sample_layers) // 3
    early_layers = sample_layers[:n_third]
    mid_layers = sample_layers[n_third:2*n_third]
    late_layers = sample_layers[2*n_third:]
    
    for phase_name, phase_layers in [("Early", early_layers), ("Mid", mid_layers), ("Late", late_layers)]:
        if len(phase_layers) < 2:
            continue
        phase_rs = []
        for i in range(len(phase_layers) - 1):
            l1, l2 = phase_layers[i], phase_layers[i+1]
            r, _ = pearsonr(np.array(layer_gaps[l1]), np.array(layer_gaps[l2]))
            phase_rs.append(r)
        mean_r = np.mean(phase_rs)
        print(f"  {phase_name} layers: mean propagation r={mean_r:.3f}")
        results[f"{phase_name.lower()}_propagation"] = float(mean_r)
    
    # === 3. Final-layer prediction from early/mid/late layers ===
    print("\n=== 3. Final-layer logit_gap Prediction ===")
    
    final_gaps = np.array(layer_gaps[sample_layers[-1]])
    for li in sample_layers:
        gaps_l = np.array(layer_gaps[li])
        r, p = pearsonr(gaps_l, final_gaps)
        print(f"  L{li} -> L{sample_layers[-1]}: r={r:.3f} (p={p:.4f})")
        results[f"L{li}_to_final"] = float(r)
    
    # === 4. Multiplicative vs Additive growth model ===
    print("\n=== 4. Growth Model Comparison ===")
    
    final_gaps = np.array(layer_gaps[sample_layers[-1]])
    initial_gaps = np.array(layer_gaps[sample_layers[0]])
    
    # Additive: gap_final = gap_initial + total_delta
    # Multiplicative: gap_final = gap_initial * growth_factor
    # Log-linear: log(gap_final) = log(gap_initial) + cumulative_log_delta
    
    valid_mask = (initial_gaps > 0) & (final_gaps > 0)
    if np.sum(valid_mask) > 10:
        r_add, _ = pearsonr(final_gaps - initial_gaps, final_gaps)
        r_mul, _ = pearsonr(np.log(final_gaps[valid_mask]), np.log(initial_gaps[valid_mask]))
        print(f"  Additive model (delta -> final): r={r_add:.3f}")
        print(f"  Multiplicative model (log_init -> log_final): r={r_mul:.3f}")
        results["additive_r"] = float(r_add)
        results["multiplicative_r"] = float(r_mul)
    
    # === 5. Unified equation: gap(L) = gap(0) * Prod(a(l)) + Sigma(b(l)*delta(l)) ===
    print("\n=== 5. Unified Equation Validation ===")
    
    # Forward simulation: start with gap(L0), multiply by a(l), add b(l)
    pred_gaps = np.zeros(len(texts))
    pred_gaps[:] = np.array(layer_gaps[sample_layers[0]])
    
    for pc in prop_coeffs:
        l1, l2 = pc["l1"], pc["l2"]
        a, b = pc["a"], pc["b"]
        pred_gaps = a * pred_gaps + b
    
    r_pred, _ = pearsonr(pred_gaps, final_gaps)
    print(f"  Forward simulation r={r_pred:.3f}")
    print(f"  Predicted mean gap: {np.mean(pred_gaps):.3f}")
    print(f"  Actual mean gap: {np.mean(final_gaps):.3f}")
    results["forward_simulation_r"] = float(r_pred)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p585.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p585.json")
    return results


def run_p586(model, tokenizer, device, model_info, texts, output_dir):
    """P586: W_U structure -> Delta_k -> logit_gap complete math"""
    model_name = model_info.name
    d_model = model_info.d_model
    W_U = get_W_U(model)  # [vocab, d_model]
    
    print(f"\n{'='*60}")
    print(f"P586: W_U structure -> Delta_k -> logit_gap math -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {"model": model_name, "experiment": "p586", "n_texts": len(texts), "K": K}
    
    # === 1. W_U difference spectrum analysis ===
    print("\n=== 1. W_U Difference Spectrum Analysis ===")
    
    all_diff_spectra = []
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
        
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        diff_spectrum = U_wut.T @ W_U_diff  # [K]
        
        all_diff_spectra.append(diff_spectrum)
        all_logit_gaps.append(logit_gap)
        all_probs.append(top1_prob)
    
    diff_spectra = np.array(all_diff_spectra)  # [n_texts, K]
    logit_gaps = np.array(all_logit_gaps)
    probs = np.array(all_probs)
    
    # W_U difference spectrum statistics
    diff_norms = np.linalg.norm(diff_spectra, axis=1)
    diff_energy_k = np.mean(diff_spectra**2, axis=0)  # [K]: mean energy per direction
    
    print(f"  W_U diff norm: {np.mean(diff_norms):.4f} +/- {np.std(diff_norms):.4f}")
    print(f"  Top-10 diff energy dirs: {np.argsort(diff_energy_k)[::-1][:10].tolist()}")
    print(f"  Top-10 diff energy values: {np.sort(diff_energy_k)[::-1][:10].tolist()}")
    
    results["w_u_diff_stats"] = {
        "mean_norm": float(np.mean(diff_norms)),
        "std_norm": float(np.std(diff_norms)),
        "top10_energy_dirs": np.argsort(diff_energy_k)[::-1][:10].tolist(),
        "top10_energy_values": np.sort(diff_energy_k)[::-1][:10].tolist(),
    }
    
    # === 2. Delta_k = (W_U[top1]-W_U[top2]) . U_k: structural decomposition ===
    print("\n=== 2. Delta_k Structural Decomposition ===")
    
    # Delta_k = (W_U[top1]-W_U[top2]) . U_k
    # |Delta_k| depends on:
    #   (a) ||W_U[top1]-W_U[top2]|| - norm of token difference
    #   (b) cos(W_U_diff, U_k) - alignment of difference with direction k
    
    all_cos_align = []
    for ti in range(len(texts)):
        cos_align = diff_spectra[ti] / (np.linalg.norm(diff_spectra[ti]) + 1e-10)
        all_cos_align.append(cos_align)
    
    cos_aligns = np.array(all_cos_align)  # [n_texts, K]
    mean_cos = np.mean(np.abs(cos_aligns), axis=0)  # [K]
    
    print(f"  Mean |cos(Delta_k, U_k)| per direction:")
    for k in [0, 1, 2, 5, 10, 20, 50]:
        if k < K:
            print(f"    Dir{k}: {mean_cos[k]:.4f}")
    
    # Which directions have consistently high alignment?
    top_align_dirs = np.argsort(mean_cos)[::-1][:10]
    print(f"  Top-10 alignment dirs: {top_align_dirs.tolist()}")
    results["top_alignment_dirs"] = top_align_dirs.tolist()
    
    # === 3. W_U spectral structure -> Delta_k magnitude prediction ===
    print("\n=== 3. W_U Spectral Structure -> Delta_k Magnitude ===")
    
    # W_U singular values determine how much energy each direction can contribute
    # Delta_k magnitude should correlate with s_k (singular value of direction k)
    
    # Average |Delta_k| across texts
    mean_abs_Delta = np.mean(np.abs(diff_spectra), axis=0)  # [K]
    
    # Correlation: s_k vs mean |Delta_k|
    r_sv_delta, p_sv_delta = spearmanr(s_wut, mean_abs_Delta)
    print(f"  s_k vs mean|Delta_k|: r={r_sv_delta:.3f} (p={p_sv_delta:.4f})")
    
    # s_k^2 vs diff_energy_k
    r_sv2_energy, _ = spearmanr(s_wut**2, diff_energy_k)
    print(f"  s_k^2 vs diff_energy_k: r={r_sv2_energy:.3f}")
    
    results["sv_delta_correlation"] = {
        "r_sv_vs_delta": float(r_sv_delta),
        "r_sv2_vs_energy": float(r_sv2_energy),
    }
    
    # === 4. Complete math: logit_gap = Sum_k c_k * Delta_k ===
    print("\n=== 4. Complete Decomposition: logit_gap = Sum c_k * Delta_k ===")
    
    # For each text, compute per-direction contribution
    # Then analyze: which factors (c_k magnitude, Delta_k magnitude, sign alignment) matter most?
    
    c_k_all = []
    for ti, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
        c_k = U_wut.T @ h
        c_k_all.append(c_k)
    
    c_k_all = np.array(c_k_all)  # [n_texts, K]
    contribs = c_k_all * diff_spectra  # [n_texts, K]
    
    # Factor analysis
    # |c_k| -> |contrib_k|
    mean_abs_ck = np.mean(np.abs(c_k_all), axis=0)
    mean_abs_contrib = np.mean(np.abs(contribs), axis=0)
    r_ck_contrib, _ = spearmanr(mean_abs_ck, mean_abs_contrib)
    print(f"  |c_k| -> |contrib_k|: r={r_ck_contrib:.3f}")
    
    # |Delta_k| -> |contrib_k|
    r_delta_contrib, _ = spearmanr(mean_abs_Delta, mean_abs_contrib)
    print(f"  |Delta_k| -> |contrib_k|: r={r_delta_contrib:.3f}")
    
    # |c_k| * |Delta_k| -> |contrib_k|
    product = mean_abs_ck * mean_abs_Delta
    r_product_contrib, _ = spearmanr(product, mean_abs_contrib)
    print(f"  |c_k|*|Delta_k| -> |contrib_k|: r={r_product_contrib:.3f}")
    
    results["factor_analysis"] = {
        "r_ck_vs_contrib": float(r_ck_contrib),
        "r_delta_vs_contrib": float(r_delta_contrib),
        "r_product_vs_contrib": float(r_product_contrib),
    }
    
    # === 5. W_U structure -> logit_gap prediction ===
    print("\n=== 5. W_U Structure -> logit_gap Prediction ===")
    
    # Can we predict logit_gap from W_U structure alone (without h)?
    # Key insight: Delta_k = (W_U[top1]-W_U[top2]) . U_k depends on which tokens are top1/top2
    # And top1/top2 depend on h, so Delta_k is NOT independent of h
    
    # But: ||W_U[top1]-W_U[top2]|| might have systematic patterns
    r_diff_norm_gap, _ = pearsonr(diff_norms, logit_gaps)
    r_diff_norm_prob, _ = pearsonr(diff_norms, probs)
    print(f"  ||W_U_diff|| -> gap: r={r_diff_norm_gap:.3f}")
    print(f"  ||W_U_diff|| -> prob: r={r_diff_norm_prob:.3f}")
    
    results["w_u_structure_prediction"] = {
        "r_diff_norm_gap": float(r_diff_norm_gap),
        "r_diff_norm_prob": float(r_diff_norm_prob),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p586.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p586.json")
    return results


def run_p587(model, tokenizer, device, model_info, texts, output_dir):
    """P587: Training strategy -> W_U structure -> Delta_k causal chain"""
    model_name = model_info.name
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P587: Training strategy -> W_U structure -> Delta_k -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {"model": model_name, "experiment": "p587", "n_texts": len(texts)}
    
    # === 1. W_U spectral shape analysis ===
    print("\n=== 1. W_U Spectral Shape ===")
    
    # W_U singular value distribution
    print(f"  Top-10 singular values: {s_wut[:10].tolist()}")
    print(f"  s_1/s_10 = {s_wut[0]/s_wut[9]:.2f}")
    print(f"  s_1/s_50 = {s_wut[0]/s_wut[49]:.2f}")
    print(f"  s_1/s_100 = {s_wut[0]/s_wut[99]:.2f}")
    
    # Power law fit
    from scipy.optimize import curve_fit
    ranks = np.arange(1, K+1)
    try:
        def power_law(x, a, b):
            return a * x**(-b)
        popt, _ = curve_fit(power_law, ranks, s_wut, p0=[s_wut[0], 0.5])
        pl_pred = power_law(ranks, *popt)
        pl_r2 = 1 - np.sum((s_wut - pl_pred)**2) / np.sum((s_wut - np.mean(s_wut))**2)
        print(f"  Power law fit: s_k = {popt[0]:.2f} * k^(-{popt[1]:.3f}), R2={pl_r2:.3f}")
        results["power_law"] = {"a": float(popt[0]), "b": float(popt[1]), "R2": float(pl_r2)}
    except:
        print("  Power law fit failed")
        results["power_law"] = None
    
    # === 2. W_U vs W_embed comparison ===
    print("\n=== 2. W_U vs W_embed Structure ===")
    
    embed = model.get_input_embeddings()
    W_embed = embed.weight.detach().float().cpu().numpy()  # [vocab, d_model]
    
    # SVD of W_embed
    K_e = min(K, min(W_embed.shape) - 2)
    U_embed, s_embed, _ = svds(W_embed.T.astype(np.float32), k=K_e)
    idx = np.argsort(s_embed)[::-1]
    s_embed = s_embed[idx]
    
    # Compare spectral shapes
    r_spectra, _ = spearmanr(s_wut[:K_e], s_embed[:K_e])
    print(f"  W_U vs W_embed spectral correlation: r={r_spectra:.3f}")
    print(f"  W_U s_1/s_50: {s_wut[0]/s_wut[min(49, K-1)]:.2f}")
    print(f"  W_embed s_1/s_50: {s_embed[0]/s_embed[min(49, K_e-1)]:.2f}")
    
    results["wu_vs_wembed"] = {
        "spectral_correlation": float(r_spectra),
        "wu_ratio_1_50": float(s_wut[0]/s_wut[min(49, K-1)]),
        "wembed_ratio_1_50": float(s_embed[0]/s_embed[min(49, K_e-1)]),
    }
    
    # === 3. Functional token separation in W_U ===
    print("\n=== 3. Functional Token Separation ===")
    
    # Classify tokens: punctuation, articles, common words, content words
    punct_ids = set()
    article_ids = set()
    for tok, idx in tokenizer.get_vocab().items():
        tok_str = tok.strip().lower()
        if tok_str in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '"', "'"]:
            punct_ids.add(idx)
        if tok_str in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of']:
            article_ids.add(idx)
    
    punct_ids = list(punct_ids & set(range(W_U.shape[0])))
    article_ids = list(article_ids & set(range(W_U.shape[0])))
    content_ids = [i for i in range(min(5000, W_U.shape[0])) if i not in punct_ids and i not in article_ids]
    
    print(f"  Punct tokens: {len(punct_ids)}, Article tokens: {len(article_ids)}, Content tokens: {len(content_ids)}")
    
    # Spectral concentration of each category
    def spectral_concentration(token_ids, U_wut, K=50):
        if len(token_ids) < 5:
            return 0.0
        W_cat = W_U[token_ids]  # [n, d_model]
        coeffs = W_cat @ U_wut[:, :K]  # [n, K]
        total_energy = np.sum(coeffs**2)
        top10_energy = np.sum(np.sort(np.sum(coeffs**2, axis=0))[::-1][:10])
        return float(top10_energy / (total_energy + 1e-10))
    
    r_punct = spectral_concentration(punct_ids, U_wut)
    r_article = spectral_concentration(article_ids, U_wut)
    r_content = spectral_concentration(content_ids[:500], U_wut)
    
    print(f"  Top-10 concentration: Punct={r_punct:.3f}, Article={r_article:.3f}, Content={r_content:.3f}")
    
    results["functional_separation"] = {
        "punct_concentration": r_punct,
        "article_concentration": r_article,
        "content_concentration": r_content,
    }
    
    # === 4. W_U structure determines Delta_k distribution ===
    print("\n=== 4. W_U Structure -> Delta_k Distribution ===")
    
    # Compute Delta_k for sample texts
    delta_k_stats = np.zeros(K)  # mean |Delta_k|
    
    for ti, text in enumerate(texts[:50]):  # subsample for speed
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
        
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        delta_k_stats += np.abs(Delta_k)
    
    delta_k_stats /= min(50, len(texts))
    
    # Correlation: singular value vs mean |Delta_k|
    r_sv_delta, _ = spearmanr(s_wut, delta_k_stats)
    print(f"  s_k vs mean|Delta_k|: r={r_sv_delta:.3f}")
    
    # Top-10 Delta_k directions
    top_delta_dirs = np.argsort(delta_k_stats)[::-1][:10]
    print(f"  Top-10 Delta_k dirs: {top_delta_dirs.tolist()}")
    print(f"  Their s_k: {s_wut[top_delta_dirs].tolist()}")
    
    results["wu_to_delta"] = {
        "r_sv_vs_delta": float(r_sv_delta),
        "top10_delta_dirs": top_delta_dirs.tolist(),
        "top10_delta_svs": s_wut[top_delta_dirs].tolist(),
    }
    
    # === 5. Training signature: W_U spectral anisotropy ===
    print("\n=== 5. W_U Spectral Anisotropy (Training Signature) ===")
    
    # Anisotropy = ratio of top-k energy to total energy
    total_sv_energy = np.sum(s_wut**2)
    for k in [1, 5, 10, 50]:
        anisotropy_k = np.sum(s_wut[:k]**2) / total_sv_energy
        print(f"  Anisotropy(top-{k}): {anisotropy_k:.4f}")
        results[f"anisotropy_{k}"] = float(anisotropy_k)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p587.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p587.json")
    return results


def run_p588(model, tokenizer, device, model_info, texts, output_dir):
    """P588: Unified language encoding equation"""
    model_name = model_info.name
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model)
    
    print(f"\n{'='*60}")
    print(f"P588: Unified language encoding equation -- {model_name}")
    print(f"{'='*60}")
    
    K = 100
    U_wut, s_wut, _ = compute_svd_W_U(W_U, k=K)
    
    results = {"model": model_name, "experiment": "p588", "n_texts": len(texts), "K": K}
    
    # Collect comprehensive data
    all_data = {
        "logit_gaps": [], "probs": [], "alpha": [],
        "ratio_50": [], "c_k_rms": [], "c_k_top10_ratio": [],
        "diff_norms": [], "snr_weighted_gap": [], "best_subset_gap": [],
    }
    
    # First pass: compute per-direction SNR
    print("  Computing per-direction SNR...")
    dir_signals = np.zeros(K)
    dir_noises = np.zeros(K)
    contribs_all = []
    logit_gaps_all = []
    probs_all = []
    
    for ti, text in enumerate(texts):
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
        
        dir_signals += np.abs(contrib_k)
        dir_noises += c_k**2
        contribs_all.append(contrib_k)
        logit_gaps_all.append(logit_gap)
        probs_all.append(top1_prob)
    
    dir_snr = (dir_signals / len(texts)) / (dir_noises / len(texts) + 1e-10)
    snr_sorted = np.argsort(dir_snr)[::-1]
    
    # Find optimal subset size
    contribs_all = np.array(contribs_all)
    best_n = 1
    best_r = 0
    for n in range(1, K+1):
        selected = snr_sorted[:n]
        cum = np.sum(contribs_all[:, selected], axis=1)
        r, _ = pearsonr(cum, np.array(logit_gaps_all))
        if abs(r) > abs(best_r):
            best_r = r
            best_n = n
    
    print(f"  Optimal SNR subset: n={best_n}, r={best_r:.3f}")
    
    # Second pass: collect all features
    print("  Collecting comprehensive features...")
    all_weighted_gaps = {f"top{n}": [] for n in [5, 10, 20, best_n]}
    all_snr_weighted = []
    
    for ti, text in enumerate(texts):
        if ti % 50 == 0:
            print(f"  Processing text {ti+1}/{len(texts)}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1].float().cpu().numpy()
            h_L = outputs.hidden_states[-1][0, -1].float().cpu().numpy()
            h_Lm1 = outputs.hidden_states[-2][0, -1].float().cpu().numpy()
        
        top1_idx = np.argmax(logits)
        top2_idx = np.argsort(logits)[-2]
        logit_gap = float(logits[top1_idx] - logits[top2_idx])
        top1_prob = float(np.exp(logits[top1_idx]) / np.sum(np.exp(logits)))
        
        # Spectral features
        c_k = U_wut.T @ h_L
        c_k_rms = float(np.sqrt(np.mean(c_k**2)))
        c_k_top10_ratio = float(np.sum(np.sort(c_k**2)[::-1][:10]) / (np.sum(c_k**2) + 1e-10))
        
        # Alpha
        c_km1 = U_wut.T @ h_Lm1
        alpha = float(np.sum(c_k * c_km1) / (np.sum(c_km1**2) + 1e-10))
        
        # Ratio_50
        ratio_50 = float(np.sum(c_k[:50]**2) / (np.sum(c_k**2) + 1e-10))
        
        # W_U difference
        W_U_diff = W_U[top1_idx] - W_U[top2_idx]
        Delta_k = U_wut.T @ W_U_diff
        contrib_k = c_k * Delta_k
        
        diff_norm = float(np.linalg.norm(W_U_diff))
        
        # Weighted gaps
        for n in [5, 10, 20, best_n]:
            selected = snr_sorted[:n]
            weighted_gap = float(np.sum(contrib_k[selected]))
            all_weighted_gaps[f"top{n}"].append(weighted_gap)
        
        # SNR-weighted
        snr_weights = dir_snr / (np.sum(dir_snr) + 1e-10)
        snr_weighted = float(np.sum(snr_weights * contrib_k))
        all_snr_weighted.append(snr_weighted)
        
        all_data["logit_gaps"].append(logit_gap)
        all_data["probs"].append(top1_prob)
        all_data["alpha"].append(alpha)
        all_data["ratio_50"].append(ratio_50)
        all_data["c_k_rms"].append(c_k_rms)
        all_data["c_k_top10_ratio"].append(c_k_top10_ratio)
        all_data["diff_norms"].append(diff_norm)
        all_data["snr_weighted_gap"].append(snr_weighted)
    
    logit_gaps = np.array(all_data["logit_gaps"])
    probs = np.array(all_data["probs"])
    
    # === 1. Complete causal chain evaluation ===
    print("\n=== 1. Complete Causal Chain ===")
    
    chain_results = {}
    
    # Path A: spectral -> logit_gap (old)
    for feat_name in ["alpha", "ratio_50", "c_k_rms", "c_k_top10_ratio"]:
        vals = np.array(all_data[feat_name])
        r, p = pearsonr(vals, logit_gaps)
        r2, _ = pearsonr(vals, probs)
        print(f"  {feat_name} -> gap: r={r:.3f}, -> prob: r={r2:.3f}")
        chain_results[f"{feat_name}_gap"] = float(r)
        chain_results[f"{feat_name}_prob"] = float(r2)
    
    # Path B: direction selection weighted -> gap
    for n in [5, 10, 20, best_n]:
        vals = np.array(all_weighted_gaps[f"top{n}"])
        r, p = pearsonr(vals, logit_gaps)
        r2, _ = pearsonr(np.abs(vals), probs)
        print(f"  SNR-Top-{n} Sum c_k*Delta_k -> gap: r={r:.3f}, -> prob: r={r2:.3f}")
        chain_results[f"snr_top{n}_gap"] = float(r)
        chain_results[f"snr_top{n}_prob"] = float(r2)
    
    # Path C: SNR-weighted
    vals = np.array(all_snr_weighted)
    r, p = pearsonr(vals, logit_gaps)
    r2, _ = pearsonr(np.abs(vals), probs)
    print(f"  SNR-weighted -> gap: r={r:.3f}, -> prob: r={r2:.3f}")
    chain_results["snr_weighted_gap"] = float(r)
    chain_results["snr_weighted_prob"] = float(r2)
    
    # Path D: W_U diff norm
    vals = np.array(all_data["diff_norms"])
    r, p = pearsonr(vals, logit_gaps)
    r2, _ = pearsonr(vals, probs)
    print(f"  ||W_U_diff|| -> gap: r={r:.3f}, -> prob: r={r2:.3f}")
    chain_results["diff_norm_gap"] = float(r)
    chain_results["diff_norm_prob"] = float(r2)
    
    # === 2. Unified equation: best path from spectrum to prob ===
    print("\n=== 2. Best Path: spectrum -> prob ===")
    
    # Compare all paths
    all_paths = {}
    for key, val in chain_results.items():
        if key.endswith("_prob"):
            path_name = key.replace("_prob", "")
            all_paths[path_name] = val
    
    best_path = max(all_paths, key=lambda k: abs(all_paths[k]))
    print(f"  Best path: {best_path} -> prob r={all_paths[best_path]:.3f}")
    results["best_path"] = best_path
    results["best_path_r"] = float(all_paths[best_path])
    
    # === 3. Two-stage model: spectral -> weighted_gap -> prob ===
    print("\n=== 3. Two-stage Model ===")
    
    # Stage 1: spectral features -> weighted_gap
    best_n_key = f"top{best_n}"
    weighted_gaps = np.array(all_weighted_gaps[best_n_key])
    
    for feat_name in ["alpha", "ratio_50", "c_k_rms"]:
        vals = np.array(all_data[feat_name])
        r, _ = pearsonr(vals, weighted_gaps)
        print(f"  {feat_name} -> weighted_gap(SNR-{best_n}): r={r:.3f}")
        chain_results[f"{feat_name}_weighted_gap"] = float(r)
    
    # Stage 2: weighted_gap -> prob
    r_wg_prob, _ = pearsonr(np.abs(weighted_gaps), probs)
    print(f"  |weighted_gap(SNR-{best_n})| -> prob: r={r_wg_prob:.3f}")
    chain_results["weighted_gap_prob"] = float(r_wg_prob)
    
    # === 4. Summary: Unified Language Encoding Equation ===
    print("\n=== 4. Unified Language Encoding Equation ===")
    print(f"  logit_gap = Sum_k c_k * Delta_k (exact, K={K})")
    print(f"  c_k = h . U_k (spectral coefficient, determined by spectral mechanics)")
    print(f"  Delta_k = (W_U[top1] - W_U[top2]) . U_k (W_U structure)")
    print(f"  SNR-Top-{best_n} Sum c_k*Delta_k -> gap: r={chain_results.get(f'snr_top{best_n}_gap', 0):.3f}")
    print(f"  |SNR-Top-{best_n}| -> prob: r={chain_results.get(f'snr_top{best_n}_prob', 0):.3f}")
    print(f"  alpha -> c_k_rms: r={chain_results.get('alpha_weighted_gap', 0):.3f}")
    
    results["chain_results"] = chain_results
    results["optimal_subset_n"] = int(best_n)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_p588.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}/{model_name}_p588.json")
    return results


EXPERIMENTS = {
    "p583": run_p583,
    "p584": run_p584,
    "p585": run_p585,
    "p586": run_p586,
    "p587": run_p587,
    "p588": run_p588,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    
    t0 = time.time()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    
    output_dir = os.path.join("results", "phase_cxxxv")
    
    print(f"\nModel: {args.model}, Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    
    EXPERIMENTS[args.experiment](
        model=model, tokenizer=tokenizer, device=device,
        model_info=model_info, texts=TEST_TEXTS, output_dir=output_dir
    )
    
    from model_utils import release_model
    release_model(model)
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
