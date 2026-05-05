"""
Phase 53: Attention Head Functional Localization — 论文级验证
============================================================

Phase 52硬伤修复:
  1. bidir attention作弊 → 只用causal direction
  2. 位置先验未排除 → 增加baseline消融
  3. det=6600可能伪影 → 跨距离/跨结构验证
  4. "信息流=语法"推理跳跃 → 严格区分

Phase 53 四个实验:
  53A: Baseline消融 — attention vs 位置/词频/启发式baseline
       ★ 关键: 证明attention > trivial pattern
  53B: Head功能定位 — Score(h,r) + 跨句泛化
       ★ 关键: 找到语法特异head的最小集合
  53C: Causal direction only — 只用因果方向恢复依赖
       ★ 关键: 在不违反因果结构的前提下恢复语法
  53D: Head因果干预 — zero-out语法head，观察输出崩溃
       ★ 关键: 证明head是"因果语法模块"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import torch
import numpy as np
import gc
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, get_W_U)


# ============================================================
# 控制变量数据集 — 关键创新
# ============================================================
# 同词不同角色 / 长距离 / 嵌套结构 / 非典型结构

CONTROLLED_SENTENCES = [
    # --- Group 1: 同词不同角色 (主语 vs 宾语) ---
    {
        "id": "role_swap_1",
        "sentence": "The cat chased the mouse",
        "deps": [
            (2, 1, "det"),    # cat ← The
            (3, 2, "nsubj"),  # chased ← cat
            (5, 4, "det"),    # mouse ← the
            (3, 5, "dobj"),   # chased ← mouse
        ],
        "note": "cat=nsubj, mouse=dobj"
    },
    {
        "id": "role_swap_2",
        "sentence": "The mouse chased the cat",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # chased ← mouse (mouse now nsubj!)
            (5, 4, "det"),
            (3, 5, "dobj"),   # chased ← cat (cat now dobj!)
        ],
        "note": "mouse=nsubj, cat=dobj (角色互换)"
    },
    
    # --- Group 2: 限定词+距离变化 ---
    {
        "id": "det_short",
        "sentence": "The cat sat",
        "deps": [
            (2, 1, "det"),  # cat ← The (distance=1)
            (3, 2, "nsubj"),
        ],
        "note": "DET距离=1"
    },
    {
        "id": "det_long",
        "sentence": "The big fluffy angry cat sat",
        "deps": [
            (5, 1, "det"),  # cat ← The (distance=4!)
            (5, 2, "amod"), # cat ← big
            (5, 3, "amod"), # cat ← fluffy
            (5, 4, "amod"), # cat ← angry
            (6, 5, "nsubj"),
        ],
        "note": "DET距离=4 (有修饰语插入)"
    },
    
    # --- Group 3: 长距离依赖 ---
    {
        "id": "long_nsubj",
        "sentence": "The cat that the dog chased ran away",
        "deps": [
            (2, 1, "det"),    # cat ← The
            (7, 2, "nsubj"),  # ran ← cat (距离5!)
            (5, 4, "det"),    # dog ← the
            (6, 5, "nsubj"),  # chased ← dog
            (6, 2, "dobj"),   # chased ← cat
            (7, 6, "acl"),    # ran ← chased? (关系从句)
        ],
        "note": "长距离nsubj (cat→ran, 距离5)"
    },
    
    # --- Group 4: 嵌套结构 ---
    {
        "id": "nested_1",
        "sentence": "The dog saw the cat that chased the mouse",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # saw ← dog
            (6, 4, "det"),    # cat ← the
            (3, 6, "dobj"),   # saw ← cat
            (6, 7, "acl"),    # cat ← that
            (7, 6, "nsubj"),  # chased ← cat (cat有多角色!)
            (9, 8, "det"),    # mouse ← the
            (7, 9, "dobj"),   # chased ← mouse
        ],
        "note": "cat同时是saw的dobj和chased的nsubj"
    },
    
    # --- Group 5: 多种关系类型 ---
    {
        "id": "multi_rel",
        "sentence": "The red car hit the old tree",
        "deps": [
            (3, 1, "det"),    # car ← The
            (3, 2, "amod"),   # car ← red
            (4, 3, "nsubj"),  # hit ← car
            (7, 5, "det"),    # tree ← the
            (7, 6, "amod"),   # tree ← old
            (4, 7, "dobj"),   # hit ← tree
        ],
        "note": "包含amod关系"
    },
    {
        "id": "prep_phrase",
        "sentence": "The bird flew over the house",
        "deps": [
            (2, 1, "det"),
            (3, 2, "nsubj"),  # flew ← bird
            (3, 4, "prep"),   # flew ← over
            (6, 5, "det"),    # house ← the
            (4, 6, "pobj"),   # over ← house
        ],
        "note": "包含介词关系"
    },
]

# 扩展测试句子 (更多样本)
EXTENDED_SENTENCES = [
    {"sentence": "The boy ate the apple", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The girl read the book", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "The man pushed the door", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(3,5,"dobj")]},
    {"sentence": "A small dog barked", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj")]},
    {"sentence": "The old man walked slowly", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj"),(4,5,"advmod")]},
    {"sentence": "She found the key quickly", "deps": [(2,1,"nsubj"),(2,4,"dobj"),(4,3,"det"),(2,5,"advmod")]},
    {"sentence": "The cat sat on the mat", "deps": [(2,1,"det"),(3,2,"nsubj"),(3,4,"prep"),(6,5,"det"),(4,6,"pobj")]},
    {"sentence": "My friend bought a new car", "deps": [(3,1,"nmod"),(4,3,"nsubj"),(6,5,"det"),(4,6,"dobj"),(6,7,"amod")]},
    {"sentence": "The teacher gave the student a book", "deps": [(2,1,"det"),(3,2,"nsubj"),(3,5,"iobj"),(5,4,"det"),(3,7,"dobj"),(7,6,"det")]},
    {"sentence": "The fish swam in the deep water", "deps": [(2,1,"det"),(3,2,"nsubj"),(3,4,"prep"),(7,5,"det"),(7,6,"amod"),(4,7,"pobj")]},
    {"sentence": "The tall building fell down", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj"),(4,5,"advmod")]},
    {"sentence": "The young girl sang a song", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj"),(6,5,"det"),(4,6,"dobj")]},
    {"sentence": "The dog and the cat played", "deps": [(2,1,"det"),(4,2,"conj"),(4,3,"cc"),(5,4,"nsubj"),(5,6,"nsubj")]},
    {"sentence": "Heavy rain damaged the small garden", "deps": [(3,1,"amod"),(3,2,"compound"),(4,3,"nsubj"),(6,5,"det"),(6,7,"amod"),(4,6,"dobj")]},
    {"sentence": "The scientist discovered a rare mineral", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(5,6,"amod"),(3,5,"dobj")]},
    {"sentence": "The artist painted the old wall", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(5,6,"amod"),(3,5,"dobj")]},
    {"sentence": "The black cat caught a small mouse", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj"),(7,5,"det"),(7,6,"amod"),(4,7,"dobj")]},
    {"sentence": "The wind blew the dry leaves away", "deps": [(2,1,"det"),(3,2,"nsubj"),(3,5,"dobj"),(5,4,"amod"),(3,6,"advmod")]},
    {"sentence": "The woman drove the red car", "deps": [(2,1,"det"),(3,2,"nsubj"),(5,4,"det"),(5,6,"amod"),(3,5,"dobj")]},
    {"sentence": "A bright star shone in the sky", "deps": [(3,1,"det"),(3,2,"amod"),(4,3,"nsubj"),(4,5,"prep"),(7,6,"det"),(5,7,"pobj")]},
]


# ============================================================
# 辅助函数
# ============================================================
def tokenize_and_resolve(sentence, tokenizer, dep_list):
    """Tokenize句子并解析依赖位置，处理多token词"""
    inputs = tokenizer(sentence, return_tensors="pt")
    token_ids = inputs["input_ids"][0].cpu().numpy()
    tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
    
    # 简单的词→token位置映射
    words = sentence.split()
    word_positions = {}
    current_pos = 1  # 跳过BOS
    for w_idx, word in enumerate(words):
        # 找到对应的token起始位置
        word_token = tokenizer.encode(" " + word, add_special_tokens=False)
        word_positions[w_idx] = list(range(current_pos, current_pos + len(word_token)))
        current_pos += len(word_token)
    
    # 解析依赖
    resolved = []
    for head_word_idx, dep_word_idx, rel_type in dep_list:
        # dep_list中的索引是词索引(1-based for head, 1-based for dep)
        # 但我们的标注用的是token位置(1-based, 0=BOS)
        # 这里dep_list中的数字已经是token位置了
        head_pos = head_word_idx
        dep_pos = dep_word_idx
        if head_pos < len(tokens) and dep_pos < len(tokens):
            resolved.append((head_pos, dep_pos, rel_type))
    
    return tokens, token_ids, resolved


def get_attention_weights(model, tokenizer, sentence, device):
    """获取所有层的attention权重"""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
            all_attn = outputs.attentions
        except:
            return None, None
    
    attn_dict = {}
    for li, attn in enumerate(all_attn):
        if attn is not None:
            attn_dict[li] = attn.detach().float().cpu().numpy()[0]  # [n_heads, seq, seq]
    
    tokens = [safe_decode(tokenizer, tid) for tid in inputs["input_ids"][0].cpu().numpy()]
    
    return attn_dict, tokens


# ============================================================
# 53A: Baseline消融 — attention vs 位置/词频/启发式
# ============================================================
def exp_53a_baseline_ablation(model, tokenizer, info, model_name):
    """
    ★★★ 最关键实验: 证明attention > trivial pattern ★★★
    
    Baseline列表:
    1. left_attach: 总是连接到i-1
    2. distance_decay: 按距离打分 P(i→j) ∝ exp(-|i-j|/τ)
    3. pos_heuristic_det: DET → 最近名词
    4. pos_heuristic_noun: NOUN → 最近动词
    5. random: 随机
    6. attention (causal only): 只用因果方向
    """
    print("\n" + "="*70)
    print("53A: Baseline Ablation — 证明attention > trivial pattern")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # 合并所有句子
    all_sents = []
    for s in CONTROLLED_SENTENCES:
        all_sents.append(s)
    for s in EXTENDED_SENTENCES:
        all_sents.append(s)
    
    print(f"\n  Total test sentences: {len(all_sents)}")
    
    # ---- 方法定义 ----
    # 对每条依赖边 (head_pos, dep_pos, rel_type):
    #   "恢复" = 预测的head位置是否正确 (UAS)
    #   在因果模型中, token i 只能 attend 到 j < i
    #   所以: 如果 head > dep, 用 A[head, :] 找 dep
    #        如果 head < dep, 用 A[dep, :] 找 head
    
    results = defaultdict(lambda: defaultdict(list))  # method -> rel_type -> [hit/miss]
    
    for sent_idx, sent_data in enumerate(all_sents):
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        # 简单依赖位置解析
        # dep_list中的数字已经是token位置 (1-based, 0=BOS)
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        # 获取attention
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        
        if attn_dict is None:
            continue
        
        # 对每条依赖边评估
        for head_pos, dep_pos, rel_type in resolved_deps:
            # === Baseline 1: left_attach (always i-1) ===
            # dependent的head预测为dep_pos-1
            pred_head = max(1, dep_pos - 1)
            results["left_attach"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 2: distance_decay ===
            # P(head=j | dep=i) ∝ exp(-|i-j|/τ), j != i
            tau = 2.0
            scores = np.array([np.exp(-abs(dep_pos - j) / tau) if j != dep_pos else -np.inf 
                              for j in range(1, seq_len)])
            pred_head = np.argmax(scores) + 1
            results["distance_decay"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 3: pos_heuristic ===
            # DET → 最近名词 (heuristic: DET的head是它右边的第一个非DET词)
            # NOUN → 最近动词 (heuristic: 名词的head是它左边或右边最近的动词)
            # 简化: 对所有关系, 用"最近的内容词"作为head
            # 更简化: 对每个dep, 预测head = 最近的非自身token (偏向左)
            if rel_type == "det":
                # DET的head = 右边第一个token (通常是名词)
                pred_head = min(dep_pos + 1, seq_len - 1)
            elif rel_type in ["nsubj", "dobj"]:
                # nsubj/dobj的head = 左边或右边最近的动词
                # 简化: 找最近的不同位置
                pred_head = max(1, dep_pos - 1) if dep_pos > 1 else dep_pos + 1
            elif rel_type == "amod":
                pred_head = min(dep_pos + 1, seq_len - 1)
            else:
                pred_head = max(1, dep_pos - 1)
            results["pos_heuristic"][rel_type].append(float(pred_head == head_pos))
            
            # === Baseline 4: random ===
            choices = [j for j in range(1, seq_len) if j != dep_pos]
            if choices:
                pred_head = np.random.choice(choices)
                results["random"][rel_type].append(float(pred_head == head_pos))
            
            # === Attention (causal direction) ===
            # ★ 只用因果方向, 不作弊
            # 对每条依赖 (head_pos, dep_pos):
            #   如果 head_pos < dep_pos: dep可以attend到head → 检查A[dep, head]
            #   如果 head_pos > dep_pos: head可以attend到dep → 检查A[head, dep]
            #   评估: 对dep(或head)的attention row, head(或dep)是否在top-k
            
            for k_val in [1, 3, 5]:
                method_name = f"attn_causal_k{k_val}"
                hit = False
                
                if head_pos < dep_pos:
                    # dep可以attend到head → 检查A[dep, 1:dep]中head是否在top-k
                    for li in sorted(attn_dict.keys()):
                        A = attn_dict[li].mean(axis=0)  # mean over heads → [seq, seq]
                        row = A[dep_pos, 1:dep_pos]  # causal: only j < dep_pos, skip BOS
                        if len(row) > 0:
                            k_actual = min(k_val, len(row))
                            top_k = np.argsort(row)[-k_actual:] + 1
                            if head_pos in top_k:
                                hit = True
                                break  # 只要有一层hit就行
                        break  # 只检查第一层(后面会按层分析)
                    
                elif head_pos > dep_pos:
                    # head可以attend到dep → 检查A[head, 1:head]中dep是否在top-k
                    for li in sorted(attn_dict.keys()):
                        A = attn_dict[li].mean(axis=0)
                        row = A[head_pos, 1:head_pos]
                        if len(row) > 0:
                            k_actual = min(k_val, len(row))
                            top_k = np.argsort(row)[-k_actual:] + 1
                            if dep_pos in top_k:
                                hit = True
                                break
                        break
                
                results[method_name][rel_type].append(float(hit))
        
        # 释放attention内存
        del attn_dict
        gc.collect()
        
        if (sent_idx + 1) % 5 == 0:
            print(f"  Processed {sent_idx+1}/{len(all_sents)} sentences...")
    
    # ---- 按层分析attention (更详细) ----
    print("\n--- Per-layer Attention UAS ---")
    
    # 对所有句子, 收集每层的attention UAS
    layer_uas = defaultdict(lambda: defaultdict(list))  # layer -> method -> [acc]
    
    for sent_data in all_sents[:10]:  # 用前10句做按层分析
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for li in sorted(attn_dict.keys()):
            A = attn_dict[li].mean(axis=0)  # [seq, seq]
            
            correct = 0
            total = 0
            for head_pos, dep_pos, rel_type in resolved_deps:
                total += 1
                hit = False
                
                if head_pos < dep_pos:
                    row = A[dep_pos, 1:dep_pos]
                    if len(row) > 0:
                        top3 = np.argsort(row)[-3:] + 1
                        if head_pos in top3:
                            hit = True
                elif head_pos > dep_pos:
                    row = A[head_pos, 1:head_pos]
                    if len(row) > 0:
                        top3 = np.argsort(row)[-3:] + 1
                        if dep_pos in top3:
                            hit = True
                
                if hit:
                    correct += 1
            
            if total > 0:
                layer_uas[li]["attn_causal_k3"].append(correct / total)
        
        del attn_dict
        gc.collect()
    
    # ---- 打印结果 ----
    print("\n" + "="*70)
    print("BASELINE ABLATION RESULTS")
    print("="*70)
    
    # 总体UAS
    print("\n--- Overall UAS (all relation types) ---")
    overall = {}
    for method in ["attn_causal_k1", "attn_causal_k3", "attn_causal_k5", 
                    "left_attach", "distance_decay", "pos_heuristic", "random"]:
        all_vals = []
        for rel_type in results[method]:
            all_vals.extend(results[method][rel_type])
        if all_vals:
            overall[method] = np.mean(all_vals)
    
    for method, score in sorted(overall.items(), key=lambda x: -x[1]):
        print(f"  {method:25s}: {score:.3f}")
    
    # 按关系类型
    print("\n--- UAS by Relation Type ---")
    for rel_type in ["nsubj", "dobj", "det", "amod", "prep", "advmod"]:
        print(f"\n  [{rel_type}]")
        for method in ["attn_causal_k3", "left_attach", "distance_decay", "pos_heuristic", "random"]:
            vals = results[method].get(rel_type, [])
            if vals:
                print(f"    {method:25s}: {np.mean(vals):.3f} (N={len(vals)})")
    
    # 按层
    print("\n--- Attention UAS by Layer (causal, k=3) ---")
    if layer_uas:
        key_layers = sorted(layer_uas.keys())[::max(1, len(layer_uas)//8)]
        for li in key_layers:
            vals = layer_uas[li].get("attn_causal_k3", [])
            if vals:
                print(f"  Layer {li:3d}: {np.mean(vals):.3f}")
        
        best_layer = max(layer_uas.keys(), 
                        key=lambda li: np.mean(layer_uas[li].get("attn_causal_k3", [0])))
        best_score = np.mean(layer_uas[best_layer]["attn_causal_k3"])
        print(f"  Best: Layer {best_layer} = {best_score:.3f}")
    
    print("\n" + "="*70)
    print("53A SUMMARY")
    print("="*70)
    print(f"  attn_causal_k3 overall: {overall.get('attn_causal_k3', 0):.3f}")
    print(f"  left_attach overall:    {overall.get('left_attach', 0):.3f}")
    print(f"  distance_decay overall: {overall.get('distance_decay', 0):.3f}")
    print(f"  pos_heuristic overall:  {overall.get('pos_heuristic', 0):.3f}")
    print(f"  random overall:         {overall.get('random', 0):.3f}")
    
    attn_score = overall.get('attn_causal_k3', 0)
    best_baseline = max(overall.get('left_attach', 0), overall.get('distance_decay', 0),
                        overall.get('pos_heuristic', 0))
    margin = attn_score - best_baseline
    print(f"\n  ★ Attention优势: {margin:.3f} over best baseline ({best_baseline:.3f})")
    if margin > 0.1:
        print("  ✅ Attention显著优于所有baseline → 语法信息确实编码在attention中")
    elif margin > 0.05:
        print("  ⚠️ Attention略优于baseline → 需要更多证据")
    else:
        print("  ❌ Attention不优于baseline → 之前结果可能是位置伪影")


# ============================================================
# 53B: Head功能定位 — Score(h,r) + 跨句泛化
# ============================================================
def exp_53b_head_functional_mapping(model, tokenizer, info, model_name):
    """
    对每个head定义语法功能:
    Score(h, r) = E[A_h(i,j) | (i,j)∈r] - E[A_h(i,j) | (i,j)∉r]
    Specificity(h, r) = Score(h,r) / std(A_h)
    
    然后验证: 同一head在不同句子中功能是否稳定
    """
    print("\n" + "="*70)
    print("53B: Head Functional Mapping — Head语法功能定位")
    print("="*70)
    
    device = next(model.parameters()).device
    n_layers = info.n_layers
    
    # Step 1: 收集所有句子的head attention
    print("\n--- Step 1: Compute Score(h, r) for all heads ---")
    
    head_rel_attn = defaultdict(lambda: defaultdict(list))  # (layer, head) -> rel_type -> [attn_vals]
    head_all_attn = defaultdict(list)  # (layer, head) -> [all_attn_vals]
    
    # ★ 简化: 只用5个关键层
    sample_layers = sorted(set([3, 6, 10, 15, 21] + [n_layers-1]))
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    for sent_idx, sent_data in enumerate(CONTROLLED_SENTENCES + EXTENDED_SENTENCES[:8]):
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        resolved_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                resolved_deps.append((head_pos, dep_pos, rel_type))
        
        if len(resolved_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for li in sample_layers:
            if li not in attn_dict:
                continue
            A = attn_dict[li]  # [n_heads, seq, seq]
            n_heads = A.shape[0]
            
            for h_idx in range(n_heads):
                # 简化: 直接用矩阵的均值和标准差作为background
                # 不再逐值收集, 大幅节省内存
                A_h = A[h_idx]  # [seq, seq]
                # 只看因果区域 (j < i)
                causal_vals = []
                for i in range(1, min(seq_len, A_h.shape[0])):
                    row = A_h[i, 1:i]
                    causal_vals.extend(row.tolist())
                
                if len(causal_vals) < 5:
                    continue
                
                overall_mean = np.mean(causal_vals)
                overall_std = np.std(causal_vals) + 1e-8
                head_all_attn[(li, h_idx)].append((overall_mean, overall_std, len(causal_vals)))
                
                # 收集关系特定的attention
                for head_pos, dep_pos, rel_type in resolved_deps:
                    if dep_pos > head_pos and dep_pos < A_h.shape[0] and head_pos < A_h.shape[1]:
                        val = float(A_h[dep_pos, head_pos])
                    elif head_pos > dep_pos and head_pos < A_h.shape[0] and dep_pos < A_h.shape[1]:
                        val = float(A_h[head_pos, dep_pos])
                    else:
                        continue
                    
                    head_rel_attn[(li, h_idx)][rel_type].append(val)
        
        del attn_dict
        gc.collect()
        
        if (sent_idx + 1) % 4 == 0:
            print(f"  Processed {sent_idx+1} sentences...", flush=True)
    
    # Step 2: 计算Score和Specificity
    print("\n--- Step 2: Score(h, r) and Specificity ---")
    
    head_scores = {}  # (layer, head) -> {rel_type: score}
    head_specificities = {}  # (layer, head) -> {rel_type: specificity}
    
    for (li, h_idx), rel_dict in head_rel_attn.items():
        if (li, h_idx) not in head_all_attn:
            continue
        
        # 从多个句子的统计信息计算加权平均
        stats_list = head_all_attn[(li, h_idx)]
        total_n = sum(s[2] for s in stats_list)
        if total_n < 10:
            continue
        
        overall_mean = sum(s[0] * s[2] for s in stats_list) / total_n
        overall_var = sum(s[1]**2 * s[2] for s in stats_list) / total_n
        overall_std = np.sqrt(overall_var) + 1e-8
        
        scores = {}
        specificities = {}
        
        for rel_type, vals in rel_dict.items():
            if len(vals) < 3:
                continue
            
            rel_mean = np.mean(vals)
            score = rel_mean - overall_mean
            specificity = score / overall_std
            
            scores[rel_type] = score
            specificities[rel_type] = specificity
        
        head_scores[(li, h_idx)] = scores
        head_specificities[(li, h_idx)] = specificities
    
    # Step 3: 找每个关系的最佳head
    print("\n--- Step 3: Best Head per Relation Type ---")
    
    for rel_type in ["nsubj", "dobj", "det", "amod"]:
        best_heads = []
        for (li, h_idx), spec in head_specificities.items():
            if rel_type in spec:
                best_heads.append((li, h_idx, spec[rel_type], head_scores[(li, h_idx)][rel_type]))
        
        best_heads.sort(key=lambda x: -x[2])
        
        print(f"\n  [{rel_type}] Top-5 heads:")
        for li, h_idx, spec, score in best_heads[:5]:
            print(f"    L{li:2d} H{h_idx:2d}: specificity={spec:.2f}, score={score:.4f}")
    
    # Step 4: 跨句泛化验证
    print("\n--- Step 4: Cross-Sentence Generalization ---")
    
    # 对每个关系的top-1 head, 验证在Controlled Sentences中的稳定性
    top_heads = {}
    for rel_type in ["nsubj", "dobj", "det"]:
        best_heads = []
        for (li, h_idx), spec in head_specificities.items():
            if rel_type in spec:
                best_heads.append((li, h_idx, spec[rel_type]))
        if best_heads:
            best_heads.sort(key=lambda x: -x[2])
            top_heads[rel_type] = best_heads[0][:2]  # (layer, head)
    
    print(f"\n  Top heads: {top_heads}")
    
    # 对role_swap句子验证: "The cat chased the mouse" vs "The mouse chased the cat"
    # nsubj head应该: cat→chased attention高(第一句) / mouse→chased attention高(第二句)
    for rel_type, (best_li, best_h) in top_heads.items():
        print(f"\n  [{rel_type}] Cross-sentence validation for L{best_li} H{best_h}:")
        
        for sent_data in CONTROLLED_SENTENCES[:2]:  # role_swap_1 and role_swap_2
            sentence = sent_data["sentence"]
            attn_dict, tokens = get_attention_weights(model, tokenizer, sentence, device)
            
            if attn_dict is None or best_li not in attn_dict:
                continue
            
            A = attn_dict[best_li][best_h]  # [seq, seq]
            
            # 找到动词位置(通常是"chased"的位置)
            verb_positions = [i for i, t in enumerate(tokens) if 'chase' in t.lower() or 'chas' in t.lower()]
            if not verb_positions:
                continue
            verb_pos = verb_positions[0]
            
            # 打印动词对各token的attention
            if verb_pos > 0:
                attn_from_verb = A[verb_pos, 1:verb_pos]
                top_tokens = np.argsort(attn_from_verb)[-3:][::-1]
                top_strs = [f"{tokens[t+1]}({attn_from_verb[t]:.3f})" for t in top_tokens]
                print(f"    '{sentence}' → verb attends to: {', '.join(top_strs)}")
            
            del attn_dict
            gc.collect()
    
    # Step 5: Head→Function分类
    print("\n--- Step 5: Head → Function Classification ---")
    
    head_functions = {}
    for (li, h_idx), spec in head_specificities.items():
        if not spec:
            continue
        best_rel = max(spec, key=spec.get)
        best_spec = spec[best_rel]
        
        if best_spec > 1.0:  # 至少1个标准差
            head_functions[(li, h_idx)] = (best_rel, best_spec)
    
    # 按层统计
    layer_func_count = defaultdict(lambda: defaultdict(int))
    for (li, h_idx), (func, spec) in head_functions.items():
        layer_func_count[li][func] += 1
    
    print("\n  Head function distribution by layer:")
    for li in sorted(layer_func_count.keys()):
        funcs = layer_func_count[li]
        func_str = ", ".join([f"{r}:{c}" for r, c in sorted(funcs.items(), key=lambda x: -x[1])])
        print(f"    Layer {li:2d}: {func_str}")
    
    print("\n" + "="*70)
    print("53B SUMMARY")
    print("="*70)
    print(f"  Total heads with syntax function: {len(head_functions)}")
    print(f"  Top heads: {top_heads}")


# ============================================================
# 53C: Causal Direction Analysis — 只用因果方向
# ============================================================
def exp_53c_causal_direction(model, tokenizer, info, model_name):
    """
    ★★★ 严格因果分析: 不用bidir作弊 ★★★
    
    核心问题: 在严格的因果约束下，attention能恢复多少语法结构？
    
    方法:
    1. 对每条依赖 (head, dep):
       - head < dep: dep attend to head (自然方向)
       - head > dep: 不可能 (因果mask) → 标记为"不可恢复"
    2. 区分"可恢复"和"不可恢复"的依赖
    3. 只评估可恢复的依赖
    """
    print("\n" + "="*70)
    print("53C: Causal Direction Analysis — 严格因果方向")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # 分类依赖
    recoverable_count = 0
    unrecoverable_count = 0
    rel_recoverable = defaultdict(lambda: [0, 0])  # rel_type -> [recoverable, total]
    
    for sent_data in CONTROLLED_SENTENCES + EXTENDED_SENTENCES:
        dep_list = sent_data["deps"]
        sentence = sent_data["sentence"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        seq_len = len(token_ids)
        
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos >= seq_len or dep_pos >= seq_len or head_pos == dep_pos:
                continue
            rel_recoverable[rel_type][1] += 1
            if head_pos < dep_pos:
                # head在dep左边 → dep可以attend到head → 可恢复
                recoverable_count += 1
                rel_recoverable[rel_type][0] += 1
            else:
                # head在dep右边 → dep不能attend到head → 不可恢复
                unrecoverable_count += 1
    
    total = recoverable_count + unrecoverable_count
    print(f"\n  依赖可恢复性分析 (因果约束下):")
    print(f"    可恢复 (head < dep):  {recoverable_count}/{total} = {recoverable_count/total:.1%}")
    print(f"    不可恢复 (head > dep): {unrecoverable_count}/{total} = {unrecoverable_count/total:.1%}")
    
    print(f"\n  按关系类型:")
    for rel_type, (rec, tot) in sorted(rel_recoverable.items()):
        if tot > 0:
            print(f"    {rel_type:10s}: {rec}/{tot} = {rec/tot:.1%} 可恢复")
    
    # 只评估可恢复的依赖
    print("\n--- UAS for Recoverable Dependencies Only (causal, k=3) ---")
    
    uas_by_layer = defaultdict(list)
    
    for sent_data in CONTROLLED_SENTENCES + EXTENDED_SENTENCES[:10]:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
        seq_len = len(tokens)
        
        # 只保留可恢复的依赖
        recoverable_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                if head_pos < dep_pos:  # ★ 只保留head在dep左边的
                    recoverable_deps.append((head_pos, dep_pos, rel_type))
        
        if len(recoverable_deps) == 0:
            continue
        
        attn_dict, _ = get_attention_weights(model, tokenizer, sentence, device)
        if attn_dict is None:
            continue
        
        for li in sorted(attn_dict.keys()):
            A = attn_dict[li].mean(axis=0)  # [seq, seq]
            
            correct = 0
            total = 0
            for head_pos, dep_pos, rel_type in recoverable_deps:
                total += 1
                # dep attend to head: A[dep_pos, 1:dep_pos]中head_pos是否在top-3
                row = A[dep_pos, 1:dep_pos]
                if len(row) > 0:
                    k = min(3, len(row))
                    top_k = np.argsort(row)[-k:] + 1
                    if head_pos in top_k:
                        correct += 1
            
            if total > 0:
                uas_by_layer[li].append(correct / total)
        
        del attn_dict
        gc.collect()
    
    # 打印结果
    if uas_by_layer:
        print("\n  Layer-wise UAS (recoverable only):")
        key_layers = sorted(uas_by_layer.keys())[::max(1, len(uas_by_layer)//8)]
        for li in key_layers:
            vals = uas_by_layer[li]
            if vals:
                print(f"    Layer {li:3d}: {np.mean(vals):.3f}")
        
        best_layer = max(uas_by_layer.keys(), key=lambda li: np.mean(uas_by_layer[li]))
        print(f"\n    Best: Layer {best_layer} = {np.mean(uas_by_layer[best_layer]):.3f}")
    
    # 对比: baseline在可恢复依赖上的表现
    print("\n--- Baseline Comparison (recoverable only) ---")
    
    baseline_uas = defaultdict(list)
    
    for sent_data in CONTROLLED_SENTENCES + EXTENDED_SENTENCES:
        sentence = sent_data["sentence"]
        dep_list = sent_data["deps"]
        
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        seq_len = len(token_ids)
        
        recoverable_deps = []
        for head_pos, dep_pos, rel_type in dep_list:
            if head_pos < seq_len and dep_pos < seq_len and head_pos != dep_pos:
                if head_pos < dep_pos:
                    recoverable_deps.append((head_pos, dep_pos, rel_type))
        
        for head_pos, dep_pos, rel_type in recoverable_deps:
            # left_attach
            pred = max(1, dep_pos - 1)
            baseline_uas["left_attach"].append(float(pred == head_pos))
            
            # distance_decay
            tau = 2.0
            scores = np.array([np.exp(-abs(dep_pos - j) / tau) if j != dep_pos else -np.inf 
                              for j in range(1, seq_len)])
            pred = np.argmax(scores) + 1
            baseline_uas["distance_decay"].append(float(pred == head_pos))
            
            # random
            choices = [j for j in range(1, seq_len) if j != dep_pos]
            if choices:
                pred = np.random.choice(choices)
                baseline_uas["random"].append(float(pred == head_pos))
    
    for method in ["left_attach", "distance_decay", "random"]:
        vals = baseline_uas.get(method, [])
        if vals:
            print(f"    {method:20s}: {np.mean(vals):.3f}")
    
    print("\n" + "="*70)
    print("53C SUMMARY")
    print("="*70)
    print(f"  可恢复依赖比例: {recoverable_count}/{total} = {recoverable_count/total:.1%}")
    if uas_by_layer:
        best_layer = max(uas_by_layer.keys(), key=lambda li: np.mean(uas_by_layer[li]))
        print(f"  Attention UAS (recoverable): {np.mean(uas_by_layer[best_layer]):.3f}")
        print(f"  Best baseline: {max(np.mean(baseline_uas['left_attach']), np.mean(baseline_uas['distance_decay'])):.3f}")
    print("  ★ 关键: 只在因果约束下评估, 不作弊")


# ============================================================
# 53D: Head因果干预 — zero-out语法head
# ============================================================
def exp_53d_head_intervention(model, tokenizer, info, model_name):
    """
    ★★★ 论文杀手级实验: 证明head的因果性 ★★★
    
    方法:
    1. 找到语法特异head (从53B)
    2. Zero-out这些head的attention
    3. 观察模型输出是否产生语法错误
    
    预期:
    - Zero-out语法head → 语法错误增加
    - Zero-out随机head → 语法错误不增加
    """
    print("\n" + "="*70)
    print("53D: Head Causal Intervention — Head因果干预")
    print("="*70)
    
    device = next(model.parameters()).device
    n_layers = info.n_layers
    
    # 测试句子 — 需要完整的语法结构
    test_sentences = [
        "The cat chased the mouse",
        "The dog saw the bird",
        "The girl read the book",
        "A tall man walked slowly",
        "The red car hit the tree",
    ]
    
    # Step 1: 先找出哪些层可以hook
    print("\n--- Step 1: Identify hookable layers ---")
    
    layers = get_layers(model)
    print(f"  Total layers: {len(layers)}")
    
    # Step 2: 基线生成 (无干预)
    print("\n--- Step 2: Baseline generation (no intervention) ---")
    
    baseline_outputs = {}
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        baseline_outputs[sent] = generated
        print(f"  '{sent}' → '{generated}'")
    
    # Step 3: Zero-out specific heads
    print("\n--- Step 3: Head ablation experiments ---")
    
    # 从53B的结果选择语法head (如果还没运行53B, 用经验值)
    # 经验: 中层(6-15)通常是语法层
    syntax_heads = [
        (8, 8),   # nsubj head (from Phase 52D)
        (10, 24), # dobj head
        (15, 18), # det head
    ]
    
    # 限制到模型实际有的层
    syntax_heads = [(l, h) for l, h in syntax_heads if l < n_layers]
    
    # 随机head作为对照
    np.random.seed(42)
    random_heads = [(np.random.randint(5, n_layers-5), np.random.randint(0, 32)) 
                    for _ in range(3)]
    random_heads = [(l, h) for l, h in random_heads if l < n_layers]
    
    print(f"  Syntax heads to ablate: {syntax_heads}")
    print(f"  Random heads to ablate: {random_heads}")
    
    # 定义hook函数
    ablation_results = defaultdict(dict)
    
    def make_zero_head_hook(head_indices):
        """创建zero-out指定head的hook"""
        hooks = []
        def hook_fn(module, input, output):
            # output通常是 (attn_output,) 或 (attn_output, attn_weights)
            if isinstance(output, tuple):
                attn_output = output[0]  # [batch, seq, d_model]
                # Zero-out specific heads
                d_head = attn_output.shape[-1] // 32  # 假设32 heads
                for h_idx in head_indices:
                    start = h_idx * d_head
                    end = (h_idx + 1) * d_head
                    attn_output[:, :, start:end] = 0
                return (attn_output,) + output[1:]
            return output
        return hook_fn
    
    # 对每组head进行ablation
    for group_name, head_list in [("syntax", syntax_heads), ("random", random_heads)]:
        print(f"\n  --- Ablating {group_name} heads: {head_list} ---")
        
        # 按层分组
        heads_by_layer = defaultdict(list)
        for li, h_idx in head_list:
            heads_by_layer[li].append(h_idx)
        
        # 注册hooks
        hooks = []
        for li, h_indices in heads_by_layer.items():
            if li < len(layers):
                try:
                    # 尝试hook attention output
                    hook = layers[li].self_attn.register_forward_hook(
                        make_zero_head_hook(h_indices)
                    )
                    hooks.append(hook)
                except Exception as e:
                    print(f"    Warning: Could not hook layer {li}: {e}")
        
        if not hooks:
            print(f"    No hooks registered, skipping {group_name}")
            continue
        
        # 生成
        for sent in test_sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=1.0,
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            ablation_results[group_name][sent] = generated
            print(f"    '{sent}' → '{generated}'")
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
    
    # Step 4: 分析差异
    print("\n--- Step 4: Ablation Impact Analysis ---")
    
    for group_name in ["syntax", "random"]:
        print(f"\n  [{group_name} ablation]")
        for sent in test_sentences:
            baseline = baseline_outputs[sent]
            ablated = ablation_results[group_name].get(sent, "N/A")
            changed = baseline != ablated
            print(f"    '{sent}'")
            print(f"      Baseline: '{baseline}'")
            print(f"      Ablated:  '{ablated}'")
            print(f"      Changed:  {'✓' if changed else '✗'}")
    
    print("\n" + "="*70)
    print("53D SUMMARY")
    print("="*70)
    
    syntax_changes = sum(1 for s in test_sentences 
                        if ablation_results["syntax"].get(s) != baseline_outputs[s])
    random_changes = sum(1 for s in test_sentences 
                        if ablation_results.get("random", {}).get(s) != baseline_outputs[s])
    
    print(f"  Syntax head ablation changed output: {syntax_changes}/{len(test_sentences)}")
    print(f"  Random head ablation changed output: {random_changes}/{len(test_sentences)}")
    
    if syntax_changes > random_changes:
        print("  ✅ Syntax head ablation has MORE impact → causally relevant")
    else:
        print("  ⚠️ No significant difference → need more investigation")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 53: Attention Head Functional Localization")
    parser.add_argument("--model", type=str, default="deepseek7b", 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment: 1=53A, 2=53B, 3=53C, 4=53D, 0=all")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 53: Attention Head Functional Localization")
    print(f"Model: {args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Layers: {info.n_layers}, d_model: {info.d_model}, vocab: {info.vocab_size}")
    
    if args.exp in [0, 1]:
        exp_53a_baseline_ablation(model, tokenizer, info, args.model)
    
    if args.exp in [0, 2]:
        exp_53b_head_functional_mapping(model, tokenizer, info, args.model)
    
    if args.exp in [0, 3]:
        exp_53c_causal_direction(model, tokenizer, info, args.model)
    
    if args.exp in [0, 4]:
        exp_53d_head_intervention(model, tokenizer, info, args.model)
    
    release_model(model)
    print(f"\n{'='*70}")
    print("Phase 53 Complete!")
    print(f"{'='*70}")
