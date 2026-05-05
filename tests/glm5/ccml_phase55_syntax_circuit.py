"""
Phase 55-v2: 语法Circuit验证 — 解决5个硬伤
============================================

核心教训: mean-head attention信号太弱, 必须用per-head分析!

硬伤1+5: 结构扰动+POS打乱 → 区分"语法" vs "位置模板" vs "POS pattern"
硬伤2: Top-k head vs naive decoding → 证明"信息存在但未被直接读取"
硬伤3: Multi-head Aggregation → 解决被动句
硬伤4: 渐变shift验证 → 用距离分桶替代人为分段
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse, torch, numpy as np, gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_model, get_model_info, release_model, safe_decode


# ============================================================
# ★ 精确token对齐的句子数据
# ============================================================

# ---- Exp1: 结构扰动 ----
STRUCTURE_PERTURB = [
    {
        "name": "S1: The cat chased the mouse (SVO)",
        "sentence": "The cat chased the mouse",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
    },
    {
        "name": "S2: The mouse chased the cat (OSV-swap)",
        "sentence": "The mouse chased the cat",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
    },
    {
        "name": "S3: The cat that the dog chased ran (center-embed)",
        "sentence": "The cat that the dog chased ran",
        "deps": [(7,2,"nsubj"), (6,5,"nsubj"), (6,2,"dobj")],
    },
    {
        "name": "S4: The dog chased the cat that ran (right-branch)",
        "sentence": "The dog chased the cat that ran",
        "deps": [(3,2,"nsubj"), (3,5,"dobj"), (7,5,"nsubj")],
    },
]

# ---- Exp2: POS打乱 ----
POS_SCRAMBLE = [
    {
        "name": "Normal: DET-NOUN-VERB-DET-NOUN",
        "sentence": "The cat chased the mouse",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
    },
    {
        "name": "POS-scrambled: DET-ADV-VERB-DET-ADJ",
        "sentence": "The quickly chased the blue",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
    },
    {
        "name": "POS-scrambled: DET-NOUN-VERB-DET-VERB",
        "sentence": "The cat chased the running",
        "deps": [(3,2,"nsubj"), (3,5,"dobj")],
    },
    {
        "name": "Passive-Normal: DET-NOUN-AUX-VERB-PREP-DET-NOUN",
        "sentence": "The mouse was chased by the cat",
        "deps": [(4,2,"nsubj_pass")],
    },
    {
        "name": "Passive-Scrambled: DET-ADJ-AUX-VERB-PREP-DET-ADV",
        "sentence": "The blue was chased by the quickly",
        "deps": [(4,2,"nsubj_pass")],
    },
]

# ---- 综合 ----
ALL_SENTENCES = {
    "simple_svo": [
        {"sentence": "The cat chased the mouse", "deps": [(3,2,"nsubj"),(3,5,"dobj")]},
        {"sentence": "The dog bit the man", "deps": [(3,2,"nsubj"),(3,5,"dobj")]},
        {"sentence": "The boy ate the apple", "deps": [(3,2,"nsubj"),(3,5,"dobj")]},
        {"sentence": "The woman drove the car", "deps": [(3,2,"nsubj"),(3,5,"dobj")]},
    ],
    "center_embed": [
        {"sentence": "The cat that the dog chased ran", "deps": [(7,2,"nsubj"),(6,5,"nsubj"),(6,2,"dobj")]},
        {"sentence": "The man who the woman saw left", "deps": [(7,2,"nsubj"),(6,5,"nsubj"),(6,2,"dobj")]},
    ],
    "passive": [
        {"sentence": "The mouse was chased by the cat", "deps": [(4,2,"nsubj_pass")]},
        {"sentence": "The car was repaired by the mechanic", "deps": [(4,2,"nsubj_pass")]},
        {"sentence": "The book was written by the author", "deps": [(4,2,"nsubj_pass")]},
        {"sentence": "The city was destroyed by the army", "deps": [(4,2,"nsubj_pass")]},
    ],
    "rel_clause": [
        {"sentence": "The woman who saw the man left", "deps": [(7,2,"nsubj"),(4,3,"nsubj"),(4,6,"dobj")]},
        {"sentence": "The dog saw the cat that ran", "deps": [(3,2,"nsubj"),(3,5,"dobj"),(7,5,"nsubj")]},
    ],
}


def get_attention_weights(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = outputs.attentions
        except:
            return None, None, None
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]
    tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
    return attn_dict, inputs, tokens


def find_head_in_row(A, li, src_pos, tgt_pos, n_top=5):
    """在指定层中, 找src_pos对tgt_pos的attention最高的heads"""
    if li not in A:
        return []
    attn = A[li]  # [n_heads, seq, seq]
    n_heads = attn.shape[0]
    
    # src_pos attend to previous positions
    if tgt_pos < src_pos:
        rows = attn[:, src_pos, 1:src_pos]  # [n_heads, src_pos-1]
        target = tgt_pos - 1
    elif tgt_pos > src_pos:
        rows = attn[:, tgt_pos, 1:tgt_pos]
        target = src_pos - 1
    else:
        return []
    
    if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
        return []
    
    results = []
    for h in range(n_heads):
        score = float(rows[h, target])
        sorted_idx = np.argsort(rows[h])[::-1]
        rank = int(np.where(sorted_idx == target)[0][0]) + 1 if target in sorted_idx else -1
        results.append({"head": h, "score": score, "rank": rank, "k1": rank == 1, "k3": rank <= 3})
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n_top]


def evaluate_per_head(attn_dict, head_pos, dep_pos, seq_len, layers_to_check):
    """对指定层做per-head评估, 返回最佳head的结果"""
    best_overall = {"rank": 999, "k1": False, "layer": -1, "head": -1}
    
    for li in layers_to_check:
        if li not in attn_dict:
            continue
        results = find_head_in_row(attn_dict, li, dep_pos if dep_pos > head_pos else head_pos, 
                                   head_pos if dep_pos > head_pos else dep_pos)
        for r in results:
            if r["rank"] < best_overall["rank"]:
                best_overall = {"rank": r["rank"], "k1": r["k1"], "k3": r.get("k3", False),
                               "layer": li, "head": r["head"], "score": r["score"]}
    
    return best_overall


def nearest_noun_baseline(head_pos, dep_pos, tokens, rel_type, seq_len):
    def is_verb(t):
        t = t.lower().strip()
        return t.endswith("ed") or t.endswith("s") or t.endswith("ing") or t in ["was","is","are","were","had","has"]
    def is_noun(t):
        t = t.lower().strip()
        if t in ["the","a","an"]: return False
        if is_verb(t): return False
        if t in ["that","who","which","by","and","or"]: return False
        return True
    
    if rel_type in ["nsubj", "nsubj_pass"]:
        for j in range(dep_pos + 1, seq_len):
            if is_verb(tokens[j]):
                return j
        return dep_pos + 1
    elif rel_type == "dobj":
        for j in range(dep_pos - 1, 0, -1):
            if is_verb(tokens[j]):
                return j
        return dep_pos - 1
    return -1


# ============================================================
# ★ Experiment 1: 结构扰动 + Per-Head分析
# ============================================================
def exp1_structure_perturb(model, tokenizer, device, info):
    print("\n" + "="*70)
    print("★ Experiment 1: Structure Perturbation + Per-Head Analysis")
    print("="*70)
    print("Logic: Same words, different order → does attention track position or role?")
    
    layers_to_check = [0, 1, 2, 3, 5, 6, 10, 15, 20]
    layers_to_check = [l for l in layers_to_check if l < info.n_layers]
    
    # Part A: Per-head UAS for each sentence
    print("\n--- Part A: Per-head best rank for each dependency ---")
    
    for item in STRUCTURE_PERTURB:
        sentence = item["sentence"]
        deps = item["deps"]
        name = item["name"]
        
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        seq_len = len(tokens)
        print(f"\n  [{name}]")
        print(f"  Tokens: {[(i,t) for i,t in enumerate(tokens)]}")
        
        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue
            
            dep_token = tokens[dep_pos]
            head_token = tokens[head_pos]
            distance = abs(head_pos - dep_pos)
            
            # 找最佳per-head结果
            best = evaluate_per_head(attn_dict, head_pos, dep_pos, seq_len, layers_to_check)
            
            # 找每个关键层的最佳head
            layer_best = {}
            for li in layers_to_check:
                if li not in attn_dict:
                    continue
                results = find_head_in_row(attn_dict, li, 
                                          dep_pos if dep_pos > head_pos else head_pos,
                                          head_pos if dep_pos > head_pos else dep_pos)
                if results:
                    layer_best[li] = results[0]
            
            print(f"  {dep_token}({dep_pos})→{head_token}({head_pos}) [{rel_type}] dist={distance}")
            print(f"    Best overall: L{best['layer']} H{best['head']} rank={best['rank']} {'✓' if best['k1'] else '✗'}")
            
            # 打印关键层的结果
            for li in [0, 3, 10, 15]:
                if li in layer_best:
                    r = layer_best[li]
                    print(f"    L{li} best: H{r['head']} rank={r['rank']} score={r['score']:.4f} {'✓' if r['k1'] else ''}")
        
        del attn_dict, inputs
        gc.collect()
    
    # Part B: ★★★ 关键对比 — Swap Analysis
    print("\n\n★★★ Part B: Swap Analysis — Position vs Role Tracking ★★★")
    print("S1: 'The CAT chased the MOUSE' → cat=nsubj, mouse=dobj")
    print("S2: 'The MOUSE chased the CAT' → mouse=nsubj, cat=dobj")
    print("If head tracks POSITION: nsubj_pref stays same across swaps")
    print("If head tracks CONTENT: nsubj_pref follows the word")
    
    # 收集两句的attention
    swap_data = {}
    for sent_idx in [0, 1]:
        item = STRUCTURE_PERTURB[sent_idx]
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, item["sentence"], device)
        if attn_dict is not None:
            swap_data[sent_idx] = {"attn": attn_dict, "tokens": tokens}
    
    if 0 in swap_data and 1 in swap_data:
        A1 = swap_data[0]["attn"]
        A2 = swap_data[1]["attn"]
        
        for li in [0, 1, 3, 6, 10, 15]:
            if li not in A1 or li not in A2:
                continue
            
            n_heads = A1[li].shape[0]
            
            # 统计: 有多少head的nsubj偏好(位置2 vs 位置5)在swap后保持一致
            position_heads = 0  # nsubj_pref在两句中一致(都偏好位置2)
            content_heads = 0  # nsubj_pref跟随词swap
            neutral_heads = 0
            
            print(f"\n  Layer {li}:")
            
            head_details = []
            for h in range(n_heads):
                # verb(3) attend to pos2 and pos5
                # 注意: 这里看的是verb(3)的attention row, 即哪些token attend to verb
                # 实际上应该看: verb(3)作为head时, nsubj(2)和dobj(5)对verb的attention
                # 即 A[h, 2, 3] (nsubj→verb) 和 A[h, 5, 3] (dobj→verb)
                # 或者: verb attend to nsubj: A[h, 3, 2] 和 verb attend to dobj: A[h, 3, 5]
                
                # 方式1: verb对位置2和5的attention
                attn_s1_v2n = float(A1[li][h, 3, 2])  # S1: verb→pos2(cat=nsubj)
                attn_s1_v2d = float(A1[li][h, 3, 5])  # S1: verb→pos5(mouse=dobj)
                attn_s2_v2n = float(A2[li][h, 3, 2])  # S2: verb→pos2(mouse=nsubj)
                attn_s2_v2d = float(A2[li][h, 3, 5])  # S2: verb→pos5(cat=dobj)
                
                total1 = attn_s1_v2n + attn_s1_v2d
                total2 = attn_s2_v2n + attn_s2_v2d
                
                if total1 < 0.001 or total2 < 0.001:
                    continue
                
                # nsubj偏好 = verb对nsubj位置(pos2)的attention比例
                nsubj_pref1 = attn_s1_v2n / total1
                nsubj_pref2 = attn_s2_v2n / total2
                
                pref_delta = nsubj_pref1 - nsubj_pref2
                
                head_details.append({
                    "h": h,
                    "pref1": nsubj_pref1, "pref2": nsubj_pref2,
                    "delta": pref_delta,
                    "a1_2n": attn_s1_v2n, "a1_5d": attn_s1_v2d,
                    "a2_2n": attn_s2_v2n, "a2_5d": attn_s2_v2d,
                })
                
                # 分类
                if abs(pref_delta) < 0.1 and nsubj_pref1 > 0.5:
                    position_heads += 1  # 两句都偏好位置2 → 位置pattern
                elif pref_delta < -0.2:
                    content_heads += 1  # swap后偏好反转 → 跟踪词内容
                else:
                    neutral_heads += 1
            
            total_analyzed = position_heads + content_heads + neutral_heads
            if total_analyzed > 0:
                print(f"    Position-tracking: {position_heads}/{total_analyzed} "
                      f"Content-tracking: {content_heads}/{total_analyzed} "
                      f"Neutral: {neutral_heads}/{total_analyzed}")
            
            # 打印最显著的heads
            head_details.sort(key=lambda x: abs(x["delta"]), reverse=True)
            for hd in head_details[:3]:
                t = "CONTENT" if hd["delta"] < -0.1 else "POSITION" if abs(hd["delta"]) < 0.1 and hd["pref1"] > 0.5 else "MIXED"
                print(f"    H{hd['h']}: nsubj_pref S1={hd['pref1']:.3f} S2={hd['pref2']:.3f} Δ={hd['delta']:+.3f} → {t}")
    
    for k in swap_data:
        del swap_data[k]["attn"]
    gc.collect()


# ============================================================
# ★ Experiment 2: POS打乱
# ============================================================
def exp2_pos_scramble(model, tokenizer, device, info):
    print("\n" + "="*70)
    print("★ Experiment 2: POS Scramble (Position vs POS Sensitivity)")
    print("="*70)
    
    layers_to_check = [l for l in [0, 1, 3, 6, 10, 15] if l < info.n_layers]
    
    for item in POS_SCRAMBLE:
        sentence = item["sentence"]
        deps = item["deps"]
        name = item["name"]
        
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        seq_len = len(tokens)
        print(f"\n  [{name}] {sentence}")
        
        for head_pos, dep_pos, rel_type in deps:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue
            
            dep_token = tokens[dep_pos]
            head_token = tokens[head_pos]
            
            best = evaluate_per_head(attn_dict, head_pos, dep_pos, seq_len, layers_to_check)
            
            print(f"    {dep_token}({dep_pos})→{head_token}({head_pos}) [{rel_type}]: "
                  f"Best L{best['layer']} H{best['head']} rank={best['rank']} {'✓' if best['k1'] else '✗'}")
        
        del attn_dict, inputs
        gc.collect()
    
    # ★★★ 关键对比: Normal vs POS-scrambled per-head pattern
    print("\n\n★★★ KEY: Normal vs POS-Scrambled — Per-head verb attention ★★★")
    
    pairs = [
        ("Normal NOUN", POS_SCRAMBLE[0]),
        ("Scrambled ADV", POS_SCRAMBLE[1]),
        ("Scrambled VERB", POS_SCRAMBLE[2]),
    ]
    
    for label, item in pairs:
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, item["sentence"], device)
        if attn_dict is None:
            continue
        
        seq_len = len(tokens)
        
        for li in [0, 3, 10]:
            if li not in attn_dict:
                continue
            A = attn_dict[li]
            n_heads = A.shape[0]
            
            # 统计: 有多少head的verb(3)最高attention指向位置2(nsubj) vs 位置5(dobj)
            nsubj_wins = 0
            dobj_wins = 0
            for h in range(n_heads):
                # verb attend to pos2 vs pos5
                attn_2 = float(A[h, 3, 2])
                attn_5 = float(A[h, 3, 5])
                if attn_2 > attn_5:
                    nsubj_wins += 1
                else:
                    dobj_wins += 1
            
            print(f"  {label} L{li}: nsubj_wins={nsubj_wins} dobj_wins={dobj_wins} "
                  f"ratio={nsubj_wins/(nsubj_wins+dobj_wins):.2f}")
        
        del attn_dict, inputs
        gc.collect()


# ============================================================
# ★ Experiment 3: Multi-Head Aggregation (被动句)
# ============================================================
def exp3_multihead_aggregation(model, tokenizer, device, info):
    print("\n" + "="*70)
    print("★ Experiment 3: Multi-Head Aggregation (Passive Voice Recovery)")
    print("="*70)
    
    n_layers = info.n_layers
    
    for group_name in ["simple_svo", "center_embed", "passive", "rel_clause"]:
        sentences = ALL_SENTENCES[group_name]
        print(f"\n--- {group_name} ---")
        
        for sent_data in sentences:
            sentence = sent_data["sentence"]
            dep_list = sent_data["deps"]
            
            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue
            
            seq_len = len(tokens)
            print(f"\n  {sentence}")
            
            for head_pos, dep_pos, rel_type in dep_list:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                dep_token = tokens[dep_pos]
                head_token = tokens[head_pos]
                distance = abs(head_pos - dep_pos)
                
                # ★ 核心对比: 单head vs 多head聚合
                for li in [3, 10, 15]:
                    if li >= n_layers or li not in attn_dict:
                        continue
                    A = attn_dict[li]
                    n_heads = A.shape[0]
                    
                    # 构造rows: 每个head的attention row
                    if head_pos < dep_pos:
                        rows = A[:, dep_pos, 1:dep_pos]  # [n_heads, dep_pos-1]
                        target = head_pos - 1
                    else:
                        rows = A[:, head_pos, 1:head_pos]
                        target = dep_pos - 1
                    
                    if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
                        continue
                    
                    # 方法1: Best single head
                    best_h = np.argmax([float(rows[h, target]) for h in range(n_heads)])
                    single_row = rows[best_h]
                    single_rank = int(np.where(np.argsort(single_row)[::-1] == target)[0][0]) + 1
                    
                    # 方法2: Top-4 heads equal sum
                    head_target_scores = [(h, float(rows[h, target])) for h in range(n_heads)]
                    head_target_scores.sort(key=lambda x: x[1], reverse=True)
                    top4 = [h for h, s in head_target_scores[:4]]
                    agg4 = sum(rows[h] for h in top4)
                    agg4_rank = int(np.where(np.argsort(agg4)[::-1] == target)[0][0]) + 1
                    
                    # 方法3: Top-8 heads
                    top8 = [h for h, s in head_target_scores[:8]]
                    agg8 = sum(rows[h] for h in top8)
                    agg8_rank = int(np.where(np.argsort(agg8)[::-1] == target)[0][0]) + 1
                    
                    # 方法4: All heads with score-weighted sum
                    agg_weighted = np.zeros(rows.shape[1])
                    for h, s in head_target_scores:
                        if s > 0:
                            agg_weighted += s * rows[h]
                    if agg_weighted.sum() > 0:
                        agg_w_rank = int(np.where(np.argsort(agg_weighted)[::-1] == target)[0][0]) + 1
                    else:
                        agg_w_rank = -1
                    
                    # 方法5: Voting — 每个head投一票给自己的top-1
                    votes = np.zeros(rows.shape[1])
                    for h in range(n_heads):
                        votes[np.argmax(rows[h])] += 1
                    vote_rank = int(np.where(np.argsort(votes)[::-1] == target)[0][0]) + 1
                    
                    marker = " ★★" if rel_type == "nsubj_pass" else ""
                    print(f"    {dep_token}→{head_token} [{rel_type}] dist={distance} L{li}: "
                          f"1h={single_rank} top4={agg4_rank} top8={agg8_rank} "
                          f"weighted={agg_w_rank} vote={vote_rank}{marker}")
            
            del attn_dict, inputs
            gc.collect()
    
    # ★★★ 被动句全层扫描
    print("\n\n★★★ PASSIVE FOCUS: All-layer Multi-head Aggregation ★★★")
    
    for sent_data in ALL_SENTENCES["passive"]:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        seq_len = len(tokens)
        print(f"\n  {sentence}")
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len:
                continue
            
            dep_token = tokens[dep_pos]
            head_token = tokens[head_pos]
            
            print(f"  Target: {dep_token}({dep_pos})→{head_token}({head_pos}) [{rel_type}]")
            
            # 全层扫描
            for li in range(min(n_layers, 28)):
                if li not in attn_dict:
                    continue
                A = attn_dict[li]
                n_heads = A.shape[0]
                
                if head_pos < dep_pos:
                    rows = A[:, dep_pos, 1:dep_pos]
                    target = head_pos - 1
                else:
                    rows = A[:, head_pos, 1:head_pos]
                    target = dep_pos - 1
                
                if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
                    continue
                
                # Top-8 aggregation
                head_scores = [(h, float(rows[h, target])) for h in range(n_heads)]
                head_scores.sort(key=lambda x: x[1], reverse=True)
                top8 = [h for h, s in head_scores[:8]]
                
                agg = sum(rows[h] for h in top8)
                rank = int(np.where(np.argsort(agg)[::-1] == target)[0][0]) + 1
                
                # Voting
                votes = np.zeros(rows.shape[1])
                for h in range(n_heads):
                    votes[np.argmax(rows[h])] += 1
                vote_rank = int(np.where(np.argsort(votes)[::-1] == target)[0][0]) + 1
                
                if rank <= 3 or vote_rank <= 3:
                    print(f"    L{li}: top8_agg rank={rank} vote_rank={vote_rank} "
                          f"{'✓✓' if rank==1 or vote_rank==1 else '✓'}")
            
            # All-layers combined
            all_agg = None
            for li in attn_dict:
                A = attn_dict[li]
                n_heads = A.shape[0]
                if head_pos < dep_pos:
                    rows = A[:, dep_pos, 1:dep_pos]
                    target = head_pos - 1
                else:
                    rows = A[:, head_pos, 1:head_pos]
                    target = dep_pos - 1
                
                if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
                    continue
                
                head_scores = [(h, float(rows[h, target])) for h in range(n_heads)]
                head_scores.sort(key=lambda x: x[1], reverse=True)
                
                if all_agg is None:
                    all_agg = np.zeros(rows.shape[1])
                for h, s in head_scores[:2]:
                    all_agg += rows[h]
            
            if all_agg is not None and target in all_agg:
                rank = int(np.where(np.argsort(all_agg)[::-1] == target)[0][0]) + 1
                print(f"    ALL-LAYERS: rank={rank} {'✓✓✓' if rank==1 else ''}")
        
        del attn_dict, inputs
        gc.collect()


# ============================================================
# ★ Experiment 4: 距离分桶对比 + 渐变shift
# ============================================================
def exp5_distance_buckets(model, tokenizer, device, info):
    print("\n" + "="*70)
    print("★ Experiment 4: Distance-Bucketed Comparison + Gradual Shift")
    print("="*70)
    
    n_layers = info.n_layers
    
    # 收集数据
    bucket_data = defaultdict(lambda: defaultdict(list))
    # bucket_data[bucket][method] = [correct/incorrect]
    
    # 同时收集per-layer-per-distance数据
    layer_dist_data = defaultdict(lambda: defaultdict(list))  # layer -> distance -> [k1 results]
    
    for group_name, sentences in ALL_SENTENCES.items():
        for sent_data in sentences:
            sentence = sent_data["sentence"]
            dep_list = sent_data["deps"]
            
            attn_dict, inputs, tokens = get_attention_weights(model, tokenizer, sentence, device)
            if attn_dict is None:
                continue
            
            seq_len = len(tokens)
            
            for head_pos, dep_pos, rel_type in dep_list:
                if head_pos >= seq_len or dep_pos >= seq_len:
                    continue
                
                distance = abs(head_pos - dep_pos)
                
                # Per-layer per-head k=1
                for li in range(min(n_layers, 28)):
                    if li not in attn_dict:
                        continue
                    A = attn_dict[li]
                    n_heads = A.shape[0]
                    
                    if head_pos < dep_pos:
                        rows = A[:, dep_pos, 1:dep_pos]
                        target = head_pos - 1
                    else:
                        rows = A[:, head_pos, 1:head_pos]
                        target = dep_pos - 1
                    
                    if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
                        continue
                    
                    # Best single head k=1
                    any_k1 = any(int(np.argsort(rows[h])[::-1][0]) == target for h in range(n_heads))
                    
                    layer_dist_data[li][distance].append(float(any_k1))
                
                # 分桶
                if distance <= 2:
                    bucket = "short(1-2)"
                elif distance <= 4:
                    bucket = "medium(3-4)"
                else:
                    bucket = "long(5+)"
                
                # Attention: any layer any head k=1
                any_k1 = False
                for li in attn_dict:
                    A = attn_dict[li]
                    if head_pos < dep_pos:
                        rows = A[:, dep_pos, 1:dep_pos]
                        target = head_pos - 1
                    else:
                        rows = A[:, head_pos, 1:head_pos]
                        target = dep_pos - 1
                    
                    if len(rows) == 0 or target < 0 or target >= rows.shape[1]:
                        continue
                    
                    for h in range(A.shape[0]):
                        if int(np.argsort(rows[h])[::-1][0]) == target:
                            any_k1 = True
                            break
                    if any_k1:
                        break
                
                bucket_data[bucket]["attn_any_k1"].append(float(any_k1))
                
                # Baseline: nearest_noun
                pred_head = nearest_noun_baseline(head_pos, dep_pos, tokens, rel_type, seq_len)
                bucket_data[bucket]["nearest_noun"].append(float(pred_head == head_pos))
            
            del attn_dict, inputs
            gc.collect()
    
    # 打印分桶结果
    print(f"\n{'Bucket':<15} {'n':<5} {'attn_k1':<10} {'near_noun':<10} {'delta':<10}")
    print("-" * 50)
    
    for bucket in ["short(1-2)", "medium(3-4)", "long(5+)"]:
        d = bucket_data[bucket]
        n = len(d["attn_any_k1"])
        if n == 0:
            continue
        attn_rate = np.mean(d["attn_any_k1"])
        nn_rate = np.mean(d["nearest_noun"])
        delta = attn_rate - nn_rate
        marker = "★" if delta > 0.1 else "✗" if delta < -0.1 else "="
        print(f"  {bucket:<13} {n:<5} {attn_rate:<10.3f} {nn_rate:<10.3f} {delta:+.3f} {marker}")
    
    # ★★★ 渐变shift: Per-layer UAS by distance
    print(f"\n\n★★★ Gradual Shift: Per-layer k=1 UAS by distance ★★★")
    
    key_layers = [0, 1, 2, 3, 5, 10, 15, 20, 27]
    key_layers = [l for l in key_layers if l < n_layers]
    
    print(f"{'Layer':<7}", end="")
    for dist in [1, 2, 3, 4, 5]:
        print(f" {'d='+str(dist):>6}", end="")
    print()
    print("-" * 40)
    
    for li in key_layers:
        print(f"  L{li:<4}", end="")
        for dist in [1, 2, 3, 4, 5]:
            vals = layer_dist_data[li].get(dist, [])
            rate = np.mean(vals) if vals else float('nan')
            print(f" {rate:>6.2f}", end="")
        print()
    
    # 渐变趋势: 短距离vs长距离的最佳层
    print(f"\n\n★★★ Best layer by distance ★★★")
    for dist in [1, 2, 3, 4, 5]:
        best_layer = -1
        best_rate = 0
        for li in key_layers:
            vals = layer_dist_data[li].get(dist, [])
            rate = np.mean(vals) if vals else 0
            if rate > best_rate:
                best_rate = rate
                best_layer = li
        n_samples = sum(len(layer_dist_data[li].get(dist, [])) for li in key_layers)
        print(f"  Distance {dist}: Best layer = L{best_layer} (rate={best_rate:.3f}, n={n_samples})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek7b", choices=["deepseek7b","glm4","qwen3"])
    parser.add_argument("--exp", type=int, default=0, 
                       help="0=all, 1=structure_perturb, 2=pos_scramble, 3=multihead, 4=distance")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 55-v2: Syntax Circuit Verification")
    print(f"Model={args.model}, Exp={args.exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Layers: {info.n_layers}, d_model: {info.d_model}")
    
    if args.exp in [0, 1]:
        exp1_structure_perturb(model, tokenizer, device, info)
    
    if args.exp in [0, 2]:
        exp2_pos_scramble(model, tokenizer, device, info)
    
    if args.exp in [0, 3]:
        exp3_multihead_aggregation(model, tokenizer, device, info)
    
    if args.exp in [0, 4]:
        exp5_distance_buckets(model, tokenizer, device, info)
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPhase 55-v2 Complete!")


if __name__ == "__main__":
    main()
